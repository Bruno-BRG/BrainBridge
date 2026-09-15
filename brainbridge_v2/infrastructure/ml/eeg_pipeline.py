"""Versioned, closed-window preprocessing for RAW full16 EEG.

Canonical profile (OpenBCI Daisy 16ch):
- 16 channels (EXG0..EXG15), 125 Hz, 250 samples (2 s), band 8-30 Hz.
- Live OpenBCI Daisy hardware runs at 125 Hz; some GUIs round it to 128 Hz
  in labels, so 128 Hz inputs are accepted and resampled.

Adaptive inputs (new):
- `preprocess_window_adaptive` accepts any sample rate (125/128/250/...)
  and any channel count, resampling/remapping to the canonical
  (250, 16) @ 125 Hz before the strict filter + z-score stage.
- This keeps already-trained (250, 16) checkpoints valid while letting
  the network consume 250 Hz or other-rate / any-channel data.
"""

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

PIPELINE_VERSION = "raw-full16-sos6-zscore-v1"

CANONICAL_SAMPLE_RATE = 125.0
CANONICAL_CHANNELS = 16
CANONICAL_WINDOW_SAMPLES = 250
CANONICAL_BAND = (8.0, 30.0)

# Sample rates commonly seen in the wild (OpenBCI labels, public datasets).
# Anything in (0, 2000] Hz is resampled; anything else is rejected.
ADAPTIVE_MIN_FS = 1.0
ADAPTIVE_MAX_FS = 2000.0
ADAPTIVE_ACCEPTED_RATES = (125.0, 128.0, 250.0, 256.0, 500.0, 512.0, 1000.0)


def bandpass_window(window, *, sample_rate=125.0, band=(8.0, 30.0)) -> np.ndarray:
    """Butterworth SOS ordem 6, 8-30 Hz, sosfiltfilt na janela fechada."""
    from scipy.signal import butter as _butter, sosfiltfilt as _sos
    data = np.asarray(window, dtype=np.float64)
    sos = _butter(6, tuple(band), btype="bandpass", fs=float(sample_rate), output="sos")
    return np.asarray(_sos(sos, data, axis=0), dtype=np.float64)


def zscore_window(window) -> np.ndarray:
    """Z-score por canal + cast float32, rejeitando nao-finitos."""
    data = np.asarray(window, dtype=np.float64)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-6)
    result = data.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite preprocessing output.")
    return result


def preprocess_window(window, *, sample_rate=125.0, band=(8.0, 30.0), apply_filter=True) -> np.ndarray:
    data = np.asarray(window, dtype=np.float64)
    if data.shape != (250, 16) or not np.isfinite(data).all():
        raise ValueError("EEG requires finite shape (250, 16); no padding or cropping.")
    if sample_rate != 125.0 or tuple(band) != (8.0, 30.0):
        raise ValueError("Pipeline requires 125 Hz and band 8-30 Hz.")
    if apply_filter:
        data = bandpass_window(data, sample_rate=sample_rate, band=tuple(band))
    return zscore_window(data)


def preprocess_ea_window(window, whitening, *, input_fs: float = CANONICAL_SAMPLE_RATE,
                         band=CANONICAL_BAND) -> np.ndarray:
    """Pipeline com EA: reamostra -> EA -> bandpass -> z-score -> (250, 16).

    Ordem conforme He & Wu / revisita 2025: o alinhamento usa janelas
    filtradas (sem DC), e o branqueamento precede a normalizacao final.
    """
    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Adaptive EEG requires finite shape (T, C).")
    data = adapt_channel_count(data)
    if abs(float(input_fs) - CANONICAL_SAMPLE_RATE) > 1e-9:
        want = window_samples_for_duration(canonical_window_duration_s(), float(input_fs))
        seg = data
        if seg.shape[0] != want:
            if seg.shape[0] > want:
                s = (seg.shape[0] - want) // 2
                seg = seg[s:s + want, :]
            else:
                seg = np.vstack([seg, np.zeros((want - seg.shape[0], seg.shape[1]))])
        data = resample_window(seg, input_fs=float(input_fs), target_fs=CANONICAL_SAMPLE_RATE)
        if data.shape[0] != CANONICAL_WINDOW_SAMPLES:
            if data.shape[0] > CANONICAL_WINDOW_SAMPLES:
                s = (data.shape[0] - CANONICAL_WINDOW_SAMPLES) // 2
                data = data[s:s + CANONICAL_WINDOW_SAMPLES, :]
            else:
                data = np.vstack([data, np.zeros((CANONICAL_WINDOW_SAMPLES - data.shape[0], data.shape[1]))])
    data = apply_ea(data, whitening)
    data = bandpass_window(data, sample_rate=CANONICAL_SAMPLE_RATE, band=tuple(band))
    if data.shape != (CANONICAL_WINDOW_SAMPLES, CANONICAL_CHANNELS):
        raise ValueError("EA pipeline requires canonical (250, 16) after resampling.")
    return zscore_window(data)


def resample_window(window: np.ndarray, *, input_fs: float,
                    target_fs: float = CANONICAL_SAMPLE_RATE) -> np.ndarray:
    """Resample time axis from input_fs to target_fs (Fourier method)."""
    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 8:
        raise ValueError("Adaptive resample requires shape (T, C) with T>=8.")
    try:
        input_fs = float(input_fs)
        target_fs = float(target_fs)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid sample rate for resampling.") from exc
    if not (ADAPTIVE_MIN_FS <= input_fs <= ADAPTIVE_MAX_FS):
        raise ValueError(f"Unsupported input sample rate: {input_fs} Hz.")
    if not (ADAPTIVE_MIN_FS <= target_fs <= ADAPTIVE_MAX_FS):
        raise ValueError(f"Unsupported target sample rate: {target_fs} Hz.")
    if input_fs == target_fs:
        return data
    n_target = max(8, int(round(data.shape[0] * target_fs / input_fs)))
    out = resample(data, n_target, axis=0)
    out = np.asarray(out, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite resampling output.")
    return out


def adapt_channel_count(window: np.ndarray, *,
                        target_channels: int = CANONICAL_CHANNELS,
                        channel_names: Optional[Sequence[str]] = None,
                        target_names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Map any channel count to target_channels.

    - If names are provided for both sides, align by name (case-insensitive).
    - Otherwise truncate extras or zero-pad missing channels.
    - Never fabricates brain signal: missing channels are zeros and the
      caller should prefer a montage-matched converter for training data.
    """
    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Adaptive channel mapping requires shape (T, C).")
    if target_channels <= 0:
        raise ValueError("target_channels must be positive.")
    n_time, n_ch = data.shape
    if n_ch == target_channels and channel_names is None:
        return data
    if channel_names is not None and target_names is not None:
        lut = {str(n).strip().lower(): i for i, n in enumerate(channel_names)}
        mapped = np.zeros((n_time, len(list(target_names))), dtype=np.float64)
        for j, want in enumerate(target_names):
            idx = lut.get(str(want).strip().lower())
            if idx is not None:
                mapped[:, j] = data[:, idx]
        return mapped
    if n_ch >= target_channels:
        return data[:, :target_channels]
    pad = np.zeros((n_time, target_channels - n_ch), dtype=np.float64)
    return np.hstack([data, pad])


def canonical_window_duration_s() -> float:
    return CANONICAL_WINDOW_SAMPLES / CANONICAL_SAMPLE_RATE


def window_samples_for_duration(duration_s: float, sample_rate: float) -> int:
    return max(8, int(round(float(duration_s) * float(sample_rate))))


def preprocess_window_adaptive(window, *, input_fs: float,
                               input_channels: Optional[int] = None,
                               channel_names: Optional[Sequence[str]] = None,
                               target_names: Optional[Sequence[str]] = None,
                               band: Tuple[float, float] = CANONICAL_BAND,
                               apply_filter: bool = True) -> np.ndarray:
    """Adaptive entry point: any (T, C) @ input_fs -> canonical (250, 16).

    Steps: validate finite -> channel-map to 16 -> resample time axis so the
    window covers the canonical 2 s duration -> strict canonical filter+zscore.
    Accepts 125/128/250 Hz and any channel count.
    """
    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Adaptive EEG requires finite shape (T, C).")
    if data.shape[0] < 8 or data.shape[1] < 1:
        raise ValueError("Adaptive EEG window too small.")
    if input_channels is not None and int(input_channels) != data.shape[1]:
        raise ValueError("input_channels does not match window shape.")
    mapped = adapt_channel_count(data, channel_names=channel_names,
                                 target_names=target_names)
    # Resample so the window spans the canonical duration (2 s).
    want_samples = window_samples_for_duration(canonical_window_duration_s(),
                                               float(input_fs))
    seg = mapped
    if seg.shape[0] != want_samples:
        # Center-crop or zero-pad time to the expected input length first,
        # so resampling always maps the same duration.
        if seg.shape[0] > want_samples:
            start = (seg.shape[0] - want_samples) // 2
            seg = seg[start:start + want_samples, :]
        else:
            pad_rows = want_samples - seg.shape[0]
            seg = np.vstack([seg, np.zeros((pad_rows, seg.shape[1]))])
    resampled = resample_window(seg, input_fs=float(input_fs),
                                target_fs=CANONICAL_SAMPLE_RATE)
    # Guard against off-by-one from rounding: center-crop/pad to 250.
    if resampled.shape[0] != CANONICAL_WINDOW_SAMPLES:
        if resampled.shape[0] > CANONICAL_WINDOW_SAMPLES:
            start = (resampled.shape[0] - CANONICAL_WINDOW_SAMPLES) // 2
            resampled = resampled[start:start + CANONICAL_WINDOW_SAMPLES, :]
        else:
            pad_rows = CANONICAL_WINDOW_SAMPLES - resampled.shape[0]
            resampled = np.vstack(
                [resampled, np.zeros((pad_rows, resampled.shape[1]))])
    return preprocess_window(resampled, sample_rate=CANONICAL_SAMPLE_RATE,
                             band=tuple(band), apply_filter=apply_filter)


def parse_sample_rate_header(header: str) -> Optional[float]:
    """Parse '%Sample Rate = 125 Hz' style headers (125/128/250/...)."""
    import re as _re
    if not header:
        return None
    m = _re.search(r"[:=]\s*([\d.]+)", str(header))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Euclidean Alignment (He & Wu, TBME 2020) para transferencia cross-subject.
#
# Cada dominio (sujeito/sessao) tem sua matriz de referencia R-barra =
# media aritmetica das covariancias espaciais das janelas, e cada janela
# e branqueada: X_til = X @ R-barra^{-1/2}  (janelas no formato (T, C)).
# Nao supervisionado (sem labels), closed-form, barato. Apos o alinhamento,
# a covariancia media de cada dominio vira a identidade, removendo o shift
# de segunda ordem entre sujeitos. Na literatura (revisita JNE 2025, MOABB)
# o placement recomendado e: filtro temporal -> EA -> resto do pipeline.
# Aqui: janela RAW -> EA -> bandpass 8-30 + z-score (preprocess_window).
# ---------------------------------------------------------------------------

def ea_reference_matrix(windows, *, shrinkage: float = 1e-2) -> np.ndarray:
    """Media das covariancias espaciais de um dominio (sujeito/sessao).

    windows: (N, T, C). Retorna R-barra (C, C) regularizada.
    """
    data = np.asarray(windows, dtype=np.float64)
    if data.ndim != 3 or data.shape[0] < 1 or data.shape[2] < 1:
        raise ValueError("EA requer janelas com shape (N, T, C).")
    if not np.isfinite(data).all():
        raise ValueError("EA requer janelas finitas.")
    n, t, c = data.shape
    covs = np.einsum("nti,ntj->nij", data, data) / max(1, t)
    ref = covs.mean(axis=0)
    ref = 0.5 * (ref + ref.T)
    mu = float(np.trace(ref)) / max(1, c)
    if not np.isfinite(mu) or mu <= 0:
        mu = 1.0
    alpha = float(shrinkage)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("shrinkage deve estar em [0, 1).")
    return (1.0 - alpha) * ref + alpha * mu * np.eye(c)


def ea_whitening_matrix(reference: np.ndarray) -> np.ndarray:
    """R-barra^{-1/2} via decomposicao espectral."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError("Referencia EA deve ser (C, C).")
    ref = 0.5 * (ref + ref.T)
    eigvals, eigvecs = np.linalg.eigh(ref)
    floor = max(float(eigvals.max()) * 1e-6, 1e-12)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, floor))) @ eigvecs.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)


def apply_ea(window, whitening: np.ndarray) -> np.ndarray:
    """Aplica X @ R-barra^{-1/2} numa janela (T, C) ou lote (N, T, C)."""
    data = np.asarray(window, dtype=np.float64)
    mat = np.asarray(whitening, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Matriz EA deve ser (C, C).")
    if data.ndim == 2:
        if data.shape[1] != mat.shape[0]:
            raise ValueError("Canais da janela incompativeis com a matriz EA.")
        return data @ mat
    if data.ndim == 3:
        if data.shape[2] != mat.shape[0]:
            raise ValueError("Canais do lote incompativeis com a matriz EA.")
        return data @ mat
    raise ValueError("EA requer janela (T, C) ou lote (N, T, C).")


class EASessionAligner:
    """Estima R-barra^{-1/2} online a partir das primeiras amostras (nao supervisionado).

    Uso ao vivo: alimente amostras RAW ate `required_samples` (default 30 s
    a 125 Hz); apos isso, `whitening` congela e `apply()` branqueia janelas.
    """

    def __init__(self, *, channels: int = CANONICAL_CHANNELS,
                 required_samples: int = 3750, shrinkage: float = 1e-2):
        if channels <= 0:
            raise ValueError("channels deve ser positivo.")
        if required_samples <= 0:
            raise ValueError("required_samples deve ser positivo.")
        self.channels = int(channels)
        self.required_samples = int(required_samples)
        self.shrinkage = float(shrinkage)
        self._outer_sum: Optional[np.ndarray] = None
        self._count = 0
        self._whitening: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._outer_sum = None
        self._count = 0
        self._whitening = None

    @property
    def samples_collected(self) -> int:
        return self._count

    @property
    def is_ready(self) -> bool:
        return self._whitening is not None

    @property
    def whitening_matrix(self) -> np.ndarray:
        if self._whitening is None:
            raise RuntimeError("EA ainda calibrando; aguarde o buffer inicial.")
        return self._whitening

    def progress(self) -> tuple[int, int]:
        return (min(self._count, self.required_samples), self.required_samples)

    def observe(self, sample) -> bool:
        """Acumula uma amostra RAW (C,). Retorna True quando calibra."""
        if self._whitening is not None:
            return True
        vec = np.asarray(sample, dtype=np.float64).reshape(-1)
        if vec.shape[0] < self.channels:
            vec = np.concatenate([vec, np.zeros(self.channels - vec.shape[0])])
        else:
            vec = vec[: self.channels]
        if not np.isfinite(vec).all():
            return False
        outer = np.outer(vec, vec)
        self._outer_sum = outer if self._outer_sum is None else self._outer_sum + outer
        self._count += 1
        if self._count >= self.required_samples:
            ref = self._outer_sum / float(self._count)
            c = ref.shape[0]
            mu = float(np.trace(ref)) / max(1, c)
            if not np.isfinite(mu) or mu <= 0:
                mu = 1.0
            ref = (1.0 - self.shrinkage) * ref + self.shrinkage * mu * np.eye(c)
            self._whitening = ea_whitening_matrix(ref)
            return True
        return False

    def apply(self, window) -> np.ndarray:
        if self._whitening is None:
            raise RuntimeError("EA ainda calibrando; aguarde o buffer inicial.")
        return apply_ea(window, self._whitening)


def pipeline_path(model_path):
    return Path(model_path).with_suffix(".pipeline.json")


def pipeline_manifest():
    return {
        "pipeline_version": PIPELINE_VERSION,
        "sample_rate": 125.0,
        "channels": [f"EXG{i}" for i in range(16)],
        "window_samples": 250,
        "classes": {"0": "T1", "1": "T2"},
        "source_expectation": "raw",
        "band": [8.0, 30.0],
        "apply_filter": True,
    }


def write_pipeline_manifest(model_path, *, training_source_stages,
                            euclidean_alignment: bool = False):
    manifest = pipeline_manifest()
    manifest["training_source_stages"] = sorted(set(training_source_stages)) or ["unknown"]
    manifest["euclidean_alignment"] = bool(euclidean_alignment)
    pipeline_path(model_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def manifest_uses_ea(model_path) -> bool:
    """True se o checkpoint foi treinado com Euclidean Alignment."""
    try:
        return bool(read_pipeline_manifest(model_path).get("euclidean_alignment", False))
    except ValueError:
        raise


def read_pipeline_manifest(model_path):
    try:
        manifest = json.loads(pipeline_path(model_path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value for key, value in pipeline_manifest().items()
        ):
            raise ValueError("Incompatible pipeline")
        stages = manifest.get("training_source_stages")
        if not isinstance(stages, list) or not stages or any(stage not in ("raw", "unknown") for stage in stages):
            raise ValueError("Missing training provenance")
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Modelo legado/manifesto ausente ou incompativel com entrada RAW. "
            "Retreine do zero (retrain); fine-tuning nao migra preprocessing legado."
        ) from exc
    return manifest
