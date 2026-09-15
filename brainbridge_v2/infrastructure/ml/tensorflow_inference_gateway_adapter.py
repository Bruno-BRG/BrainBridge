"""
Inference gateway backed by TensorFlow models.
"""

from typing import Callable, Optional, Sequence, Tuple
from threading import RLock

import numpy as np

from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult
from brainbridge_v2.application.runtime_config import DEFAULT_RUNTIME_CONFIG
from brainbridge_v2.infrastructure.ml.tensorflow_adapter import TensorFlowMLAdapter
from .eeg_pipeline import (
    CANONICAL_CHANNELS,
    CANONICAL_SAMPLE_RATE,
    CANONICAL_WINDOW_SAMPLES,
    EASessionAligner,
    adapt_channel_count,
    manifest_uses_ea,
    preprocess_ea_window,
    preprocess_window,
    preprocess_window_adaptive,
    read_pipeline_manifest,
    resample_window,
)


class TensorFlowInferenceGatewayAdapter:
    """
    Loads TensorFlow models and executes inference on normalized EEG windows.
    """

    def __init__(
        self,
        adapter_factory: Optional[Callable[[], TensorFlowMLAdapter]] = None,
        *,
        warmup_enabled: bool = DEFAULT_RUNTIME_CONFIG.tensorflow_warmup_enabled,
        input_sample_rate: float = CANONICAL_SAMPLE_RATE,
        ea_calibration_seconds: float = 30.0,
    ):
        self._adapter_factory = adapter_factory or (
            lambda: TensorFlowMLAdapter(config={})
        )
        self._warmup_enabled = bool(warmup_enabled)
        self._input_sample_rate = float(input_sample_rate)
        self._ea_calibration_samples = max(
            250, int(float(ea_calibration_seconds) * float(input_sample_rate)))
        self._ea_enabled = False
        self._ea_aligner = EASessionAligner(
            channels=CANONICAL_CHANNELS,
            required_samples=self._ea_calibration_samples)
        self._adapter: Optional[TensorFlowMLAdapter] = None
        self._loaded_model: Optional[ModelMetadata] = None
        self._lock = RLock()

    def load_model(self, model_path: str) -> ModelMetadata:
        with self._lock:
            return self._load_model(model_path)

    def _load_model(self, model_path: str) -> ModelMetadata:
        # Runtime supplies RAW. The old IQR/crop/pad path is not safe for it.
        self._adapter = None
        self._loaded_model = None
        read_pipeline_manifest(model_path)
        try:
            self._ea_enabled = manifest_uses_ea(model_path)
        except ValueError:
            self._ea_enabled = False
        self._ea_aligner.reset()
        adapter = self._adapter_factory()
        model = adapter.load_model(model_path)
        shape = self._extract_input_shape(model)
        if not self._is_compatible_shape(shape):
            raise ValueError(
                "Modelo incompativel: esperado (None, T, 16) com 2 saidas T1/T2; "
                f"recebido {shape}. Retreine com 16 canais."
            )
        if tuple(getattr(model, "output_shape", ())) != (None, 2):
            raise ValueError("Modelo incompativel com classes T1/T2; retreine.")
        self._adapter = adapter
        try:
            self._loaded_model = self._build_model_metadata(model_path, model)
            self._loaded_model.validate()
            if self._warmup_enabled:
                self._warmup_model()
        except BaseException:
            self._adapter = None
            self._loaded_model = None
            raise
        return self._loaded_model

    def get_loaded_model(self) -> Optional[ModelMetadata]:
        # Metadata snapshot only: GUI reads must not wait for inference.
        return self._loaded_model

    def predict(self, eeg_window: Sequence[Sequence[float]], *,
                  input_fs: Optional[float] = None) -> PredictionResult:
        with self._lock:
            return self._predict(eeg_window, input_fs=input_fs)

    def set_input_sample_rate(self, sample_rate: float) -> None:
        with self._lock:
            fs = float(sample_rate)
            if not (0 < fs <= 2000):
                raise ValueError("input sample rate invalida.")
            self._input_sample_rate = fs

    # -- Euclidean Alignment (calibracao online, nao supervisionada) ---------
    def ea_enabled(self) -> bool:
        with self._lock:
            return bool(self._ea_enabled and self._loaded_model is not None)

    def ea_is_calibrated(self) -> bool:
        with self._lock:
            return (not self._ea_enabled) or self._ea_aligner.is_ready

    def ea_progress(self) -> tuple[int, int]:
        with self._lock:
            return self._ea_aligner.progress()

    def ea_observe_sample(self, sample) -> bool:
        """Alimenta 1 amostra RAW ao calibrador. Retorna True se calibrado."""
        with self._lock:
            if not self._ea_enabled:
                return True
            return bool(self._ea_aligner.observe(sample))

    def ea_reset(self) -> None:
        with self._lock:
            self._ea_aligner.reset()

    # -- RL online (feedback humano): clone-fit-swap -----------------------
    def rl_snapshot_weights(self) -> bool:
        """Guarda copia dos pesos atuais (checkpoint pre-RL)."""
        with self._lock:
            if self._adapter is None or self._adapter.model is None:
                return False
            try:
                import copy
                self._rl_snapshot = [np.array(w, copy=True)
                                     for w in self._adapter.model.get_weights()]
                self._rl_updates_applied = 0
                return True
            except Exception:
                return False

    def rl_has_snapshot(self) -> bool:
        with self._lock:
            return bool(getattr(self, "_rl_snapshot", None))

    def rl_updates_count(self) -> int:
        with self._lock:
            return int(getattr(self, "_rl_updates_applied", 0) or 0)

    def rl_restore_snapshot(self) -> bool:
        """Restaura pesos pre-RL (trava de seguranca contra drift)."""
        with self._lock:
            snapshot = getattr(self, "_rl_snapshot", None)
            if not snapshot or self._adapter is None or self._adapter.model is None:
                return False
            try:
                self._adapter.model.set_weights([np.array(w, copy=True) for w in snapshot])
                self._rl_snapshot = None
                self._rl_updates_applied = 0
                return True
            except Exception:
                return False

    def rl_online_update(self, windows, labels, *, sample_weights=None,
                         epochs: int = 3, lr: float = 5e-5,
                         freeze_backbone: bool = True) -> dict:
        """Aplica 1 passo de aprendizado com janelas rotuladas pelo feedback.

        Clona o modelo, treina o clone (inferencia segue no original) e troca
        sob lock. Erros devem chegar com peso maior via sample_weights.
        Retorna {"n": int, "loss": float|None}.
        """
        import importlib as _il

        adapter = self._adapter
        if adapter is None or getattr(adapter, "model", None) is None:
            raise RuntimeError("Nenhum modelo foi carregado para RL.")
        X = np.asarray(
            [np.asarray(w.tolist() if hasattr(w, "tolist") else w, dtype=np.float32)
             for w in windows], dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32).reshape(-1)
        if X.ndim != 3 or len(X) != len(y) or len(X) == 0:
            raise ValueError("RL requer janelas (N,T,C) e labels nao vazios.")
        if X.shape[-1] != CANONICAL_CHANNELS:
            raise ValueError("RL requer 16 canais.")
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("RL requer labels 0/1.")

        tf_models = _il.import_module("tensorflow.keras.models")
        optimizers = _il.import_module("tensorflow.keras.optimizers")
        with self._lock:
            if self._adapter is None or getattr(self._adapter, "model", None) is None:
                raise RuntimeError("Modelo descarregado durante o RL.")
            base = self._adapter.model
            if not getattr(self, "_rl_snapshot", None):
                try:
                    self._rl_snapshot = [np.array(w, copy=True) for w in base.get_weights()]
                except Exception:
                    self._rl_snapshot = None
            clone = tf_models.clone_model(base)
            clone.set_weights(base.get_weights())
        dense_layers = [layer for layer in clone.layers
                        if layer.__class__.__name__ == "Dense"]
        if freeze_backbone and dense_layers:
            head = dense_layers[-1]
            for layer in clone.layers:
                layer.trainable = layer is head
        clone.compile(optimizer=optimizers.Adam(learning_rate=float(lr)),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        fit_kwargs = dict(epochs=int(epochs), batch_size=min(8, len(X)), verbose=0)
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        history = clone.fit(X, y, **fit_kwargs)
        with self._lock:
            if self._adapter is not adapter or getattr(self._adapter, "model", None) is not base:
                raise RuntimeError("Modelo trocado durante o RL; update descartado.")
            self._adapter.model = clone
            self._rl_updates_applied = int(getattr(self, "_rl_updates_applied", 0) or 0) + 1
        try:
            loss = float(history.history.get("loss", [None])[-1])
        except Exception:
            loss = None
        return {"n": int(len(X)), "loss": loss}

    def _predict(self, eeg_window: Sequence[Sequence[float]],
                 *, input_fs: Optional[float] = None) -> PredictionResult:
        if self._adapter is None or self._adapter.model is None or self._loaded_model is None:
            raise RuntimeError("Nenhum modelo foi carregado para inferencia.")

        fs = float(input_fs) if input_fs is not None else self._input_sample_rate
        window = np.asarray(
            eeg_window.tolist() if hasattr(eeg_window, "tolist") else eeg_window,
            dtype=np.float64,
        )
        if window.ndim != 2 or not np.isfinite(window).all():
            raise ValueError("Janela EEG invalida para inferencia.")
        expected_t = self._loaded_model.expected_time_steps
        expected_c = self._loaded_model.expected_channels or CANONICAL_CHANNELS
        # Qualquer combinacao de canais -> 16; qualquer fs -> canonico.
        if window.shape[1] != expected_c:
            window = adapt_channel_count(window, target_channels=int(expected_c))
        if self._ea_enabled:
            # Mesmo placement do treino: bandpass -> EA -> z-score.
            # (O widget so prediz apos a calibracao; sem referencia, erro.)
            adapted_window = preprocess_ea_window(
                window, self._ea_aligner.whitening_matrix, input_fs=fs)
        elif window.shape == (CANONICAL_WINDOW_SAMPLES, CANONICAL_CHANNELS) and abs(fs - CANONICAL_SAMPLE_RATE) < 1e-9:
            adapted_window = preprocess_window(window)
        else:
            try:
                adapted_window = preprocess_window_adaptive(window, input_fs=fs)
            except ValueError:
                # Janela canonica com fs rotulado diferente: tenta o caminho estrito.
                adapted_window = preprocess_window(
                    self._adapt_window(window, expected_t, expected_c))
        # Modelo adaptativo (None,16) aceita T variavel; modelo fixo exige T exato.
        if expected_t is not None and adapted_window.shape[0] != expected_t:
            adapted_window = self._adapt_window(adapted_window, expected_t, expected_c)
        batch = adapted_window.reshape(1, adapted_window.shape[0], adapted_window.shape[1])
        raw_output = np.asarray(self._adapter.predict(batch), dtype="float32")
        if raw_output.shape not in ((1, 2), (2,)) or not np.isfinite(raw_output).all():
            raise ValueError("Saida de inferencia deve conter duas probabilidades finitas.")

        if raw_output.ndim == 2:
            probabilities = raw_output[0]
        elif raw_output.ndim == 1:
            probabilities = raw_output
        else:
            raise ValueError("Saida de inferencia inesperada.")

        predicted_index = int(np.argmax(probabilities))
        result = PredictionResult(
            predicted_index=predicted_index,
            confidence=float(probabilities[predicted_index]),
            probabilities=tuple(float(value) for value in probabilities.tolist()),
        )
        result.validate()
        return result

    def _build_model_metadata(self, model_path: str, model: object) -> ModelMetadata:
        input_shape = self._extract_input_shape(model)
        expected_time_steps, expected_channels = self._extract_expected_dimensions(
            input_shape
        )

        return ModelMetadata(
            path=str(model_path),
            name=str(model_path).split("\\")[-1].split("/")[-1],
            input_shape=input_shape,
            expected_time_steps=expected_time_steps,
            expected_channels=expected_channels,
        )

    def _warmup_model(self) -> None:
        if self._adapter is None or self._loaded_model is None:
            return
        time_steps = self._loaded_model.expected_time_steps or CANONICAL_WINDOW_SAMPLES
        channels = self._loaded_model.expected_channels or CANONICAL_CHANNELS
        if time_steps is None or channels is None:
            return
        dummy_batch = np.zeros((1, int(time_steps), int(channels)), dtype="float32")
        self._adapter.predict(dummy_batch)

    @staticmethod
    def _is_compatible_shape(shape) -> bool:
        # Aceita (None,250,16) legado e (None,None,16) adaptativo.
        # Canais !=16 sao rejeitados (mapeamento acontece no preprocessing).
        try:
            dims = list(shape)
        except TypeError:
            return False
        if len(dims) != 3 or dims[0] not in (None, -1):
            return False
        t_ok = dims[1] in (None, -1) or (isinstance(dims[1], int) and dims[1] > 0)
        c_ok = dims[2] == CANONICAL_CHANNELS
        return bool(t_ok and c_ok)

    @staticmethod
    def _extract_input_shape(model: object) -> Optional[Tuple[Optional[int], ...]]:
        try:
            if hasattr(model, "input_shape") and model.input_shape is not None:
                return tuple(model.input_shape)
            if hasattr(model, "inputs") and getattr(model, "inputs"):
                return tuple(model.inputs[0].shape.as_list())
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_expected_dimensions(
        input_shape: Optional[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], Optional[int]]:
        if input_shape is None:
            return None, None

        dims = list(input_shape)
        if len(dims) == 3 and dims[0] in (None, -1):
            return (
                TensorFlowInferenceGatewayAdapter._safe_int(dims[1]),
                TensorFlowInferenceGatewayAdapter._safe_int(dims[2]),
            )
        if len(dims) == 2:
            return (
                TensorFlowInferenceGatewayAdapter._safe_int(dims[0]),
                TensorFlowInferenceGatewayAdapter._safe_int(dims[1]),
            )
        return None, None

    @staticmethod
    def _safe_int(value: object) -> Optional[int]:
        if value in (None, -1):
            return None
        return int(value)

    @staticmethod
    def _normalize_window(window: np.ndarray) -> np.ndarray:
        """Legacy IQR helper only; deliberately unreachable from RAW inference."""
        normalized = window.copy()
        for channel_index in range(normalized.shape[1]):
            channel_data = normalized[:, channel_index]
            q75, q25 = np.percentile(channel_data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            channel_mean = float(np.mean(channel_data))
            normalized[:, channel_index] = (channel_data - channel_mean) / iqr
        return normalized

    @staticmethod
    def _adapt_window(
        window: np.ndarray,
        expected_time_steps: Optional[int],
        expected_channels: Optional[int],
    ) -> np.ndarray:
        adapted = window

        if expected_time_steps is not None and expected_time_steps != adapted.shape[0]:
            if expected_time_steps < adapted.shape[0]:
                start = (adapted.shape[0] - expected_time_steps) // 2
                adapted = adapted[start : start + expected_time_steps, :]
            else:
                padding_rows = expected_time_steps - adapted.shape[0]
                padding = np.zeros((padding_rows, adapted.shape[1]), dtype=adapted.dtype)
                adapted = np.vstack([adapted, padding])

        if expected_channels is not None and expected_channels != adapted.shape[1]:
            if expected_channels < adapted.shape[1]:
                adapted = adapted[:, :expected_channels]
            else:
                padding_columns = expected_channels - adapted.shape[1]
                padding = np.zeros((adapted.shape[0], padding_columns), dtype=adapted.dtype)
                adapted = np.hstack([adapted, padding])

        return adapted
