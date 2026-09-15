"""
Pipeline de treinamento de modelos para BrainBridge v2

Fluxo (alinhado ao HardThinking):
 - Lê CSVs no formato OpenBCI (com coluna Annotations: T1/T2/T0)
 - Extrai segmentos entre marcadores: T1..T0 (classe 0), T2..T0 (classe 1)
 - Janela deslizante (window=250, overlap=125) sobre cada segmento
 - Pré-processamento por janela: Butterworth 8–30 Hz e normalização z-score por canal
 - Treina um modelo CNN 1D Keras com callbacks (EarlyStopping/ReduceLROnPlateau)
 - Split estratificado explícito para validação e métricas
 - Salva o modelo em data/models
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import csv
import time
import re
import hashlib
import warnings

from .models import build_cnn_1d, load_keras_model
from .physionet_eegmmidb_protocol import is_left_right_training_file
from ..config.settings import MODELS_DIR
from .eeg_pipeline import (
    ADAPTIVE_MAX_FS,
    ADAPTIVE_MIN_FS,
    CANONICAL_SAMPLE_RATE,
    preprocess_window,
    read_pipeline_manifest,
    resample_window,
    write_pipeline_manifest,
)

@dataclass
class TrainResult:
    model_path: str
    final_accuracy: Optional[float]
    final_loss: Optional[float]
    history: Dict[str, List[float]]
    training_time: float
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None


@dataclass
class SubjectWindowSummary:
    group_id: str
    csv_files: List[str]
    windows: int
    class_counts: Dict[int, int]


@dataclass
class GeneralizedTrainResult(TrainResult):
    heldout_groups: List[str] = None
    train_groups: List[str] = None
    group_summaries: List[SubjectWindowSummary] = None
    group_metrics: Dict[str, Dict[str, float]] = None


def _load_openbci_csv(csv_path: Path, *, source_stages=None) -> Tuple[np.ndarray, List[str]]:
    """Carrega CSV OpenBCI e retorna (data, markers).

    data: np.ndarray shape (n_samples, 16) @ 125 Hz canonico.
    markers: lista de strings (ex: '', 'T1', 'T2', 'T0').

    Aceita Sample Rate 125/128/250/... (perfil OpenBCI Daisy = 125 Hz;
    rotulos 128 Hz sao reamostrados para 125 Hz). Conversores em
    tools/datasets sempre gravam 16 canais @ 125 Hz.
    """
    rows = []
    markers = []
    stage = "unknown"
    sample_rate_known = False
    file_fs = CANONICAL_SAMPLE_RATE
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        # pular headers do OpenBCI começando com '%'
        for row in reader:
            if not row:
                continue
            if row[0].startswith('%'):
                header = ','.join(row).strip()
                if header.lower().startswith('%sample rate'):
                    match = re.search(r'[:=]\s*([\d.]+)', header)
                    if not match:
                        raise ValueError(f"Sample Rate ilegivel: {csv_path}")
                    try:
                        file_fs = float(match.group(1))
                    except ValueError as exc:
                        raise ValueError(f"Sample Rate ilegivel: {csv_path}") from exc
                    if not (ADAPTIVE_MIN_FS <= file_fs <= ADAPTIVE_MAX_FS):
                        raise ValueError(f"Sample Rate nao suportada ({file_fs} Hz): {csv_path}")
                    sample_rate_known = True
                if header.lower().startswith('%signal stage'):
                    stage = re.split(r'[:=]', header, maxsplit=1)[-1].strip().lower()
                    if stage not in ("raw", "unknown"):
                        raise ValueError(f"Signal Stage must be raw: {csv_path}")
                continue
            # Header real tem "Sample Index"; vamos detectar e pular a linha de header
            if row[0] == 'Sample Index':
                continue
            rows.append(row)

    # Cada linha: [Sample Index, EXG0..EXG15, Accel0..3, Other..., Analog..., Timestamp..., Annotations]
    # Precisamos extrair EXG0..EXG15 (colunas 1..16) e a última coluna (Annotations)
    data = []
    for r in rows:
        if len(r) < 18:
            raise ValueError(f"Incomplete EEG row: {csv_path}")
        try:
            channels = [float(x) for x in r[1:17]]
            data.append(channels)
            markers.append(r[-1].strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EEG row: {csv_path}") from exc

    array = np.array(data, dtype=np.float32).reshape(-1, 16)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite EEG: {csv_path}")
    # Reamostra arquivos 128/250/... Hz para o canonico 125 Hz, remapeando
    # os indices dos marcadores por vizinho mais proximo.
    if sample_rate_known and abs(file_fs - CANONICAL_SAMPLE_RATE) > 1e-6:
        n_target = max(1, int(round(len(array) * CANONICAL_SAMPLE_RATE / file_fs)))
        resampled = np.asarray(
            resample_window(np.asarray(array, dtype=np.float64),
                            input_fs=file_fs, target_fs=CANONICAL_SAMPLE_RATE),
            dtype=np.float32,
        )
        # Ajusta para o n_target exato (resample_window arredonda por janela).
        if len(resampled) != n_target:
            if len(resampled) > n_target:
                s = (len(resampled) - n_target) // 2
                resampled = resampled[s:s + n_target]
            else:
                pad = np.zeros((n_target - len(resampled), 16), dtype=np.float32)
                resampled = np.vstack([resampled, pad])
        new_markers = [""] * n_target
        scale = n_target / max(1, len(markers))
        for old_idx, mk in enumerate(markers):
            if mk:
                new_idx = min(n_target - 1, int(round(old_idx * scale)))
                # Preserva o primeiro marcador em caso de colisao.
                if not new_markers[new_idx]:
                    new_markers[new_idx] = mk
        array, markers = resampled.reshape(-1, 16), new_markers
    if stage == "unknown":
        warnings.warn(f"Legacy CSV {csv_path}: Signal Stage unknown; RAW provenance unverified.", RuntimeWarning)
    if not sample_rate_known:
        warnings.warn(f"Legacy CSV {csv_path}: Sample Rate missing; assuming 125 Hz.", RuntimeWarning)
    if source_stages is not None:
        source_stages.append(stage)
    return array, markers


def count_labeled_trials(csv_path: str | Path) -> Dict[str, int]:
    """Conta trials T1/T2 (limitados por qualquer marcador seguinte) num CSV."""
    _, markers = _load_openbci_csv(Path(csv_path))
    boundaries = [i for i, marker in enumerate(markers) if marker.strip()]
    counts = {"T1": 0, "T2": 0}
    for start, end in zip(boundaries, boundaries[1:]):
        if markers[start] in counts and end - start >= 250:
            counts[markers[start]] += 1
    return counts


def count_labeled_trials_total(csv_path: str | Path) -> int:
    counts = count_labeled_trials(csv_path)
    return int(counts.get("T1", 0) + counts.get("T2", 0))


def _extract_segments_between_markers(data: np.ndarray,
                                      start_indices: List[int],
                                      end_indices: List[int]) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    for s in start_indices:
        e = next((e for e in end_indices if e > s), None)
        if e is None:
            continue
        seg = data[s:e]
        if len(seg) > 0:
            segments.append(seg)
    return segments


def _create_windows_ht(data: np.ndarray,
                       markers: List[str],
                       window_size: int = 250,
                       step: int = 125,
                       fs: float = 125.0,
                       apply_filter: bool = True,
                       band: Tuple[float, float] = (8.0, 30.0),
                       *, return_groups: bool = False, source_id: Optional[str] = None,
                       raw_windows: bool = False):
    """Extrai janelas rotuladas no estilo HardThinking.

    - Constrói segmentos T1..T0 (label 0) e T2..T0 (label 1)
    - Desliza janela com 'window_size' e 'step' dentro de cada segmento
    - Aplica filtro 8–30 Hz e normalização z-score por canal por janela
    """
    if window_size != 250 or step <= 0 or data.ndim != 2 or data.shape[1] != 16:
        raise ValueError("Requires full16, window_size=250 and positive step.")
    if len(markers) != len(data) or not np.isfinite(data).all():
        raise ValueError("Invalid samples/markers.")
    source_id = source_id or hashlib.sha256(data.tobytes() + repr(markers).encode()).hexdigest()
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups = []
    boundaries = [i for i, marker in enumerate(markers) if marker.strip()]
    for start, end in zip(boundaries, boundaries[1:]):
        if markers[start] not in ("T1", "T2"):
            continue
        for offset in range(start, end - window_size + 1, step):
            if raw_windows:
                X_list.append(np.asarray(data[offset:offset + window_size],
                                         dtype=np.float64))
            else:
                X_list.append(preprocess_window(data[offset:offset + window_size],
                                               sample_rate=fs, band=band, apply_filter=apply_filter))
            y_list.append(0 if markers[start] == "T1" else 1)
            groups.append(f"{source_id}:{start}")
    X = np.stack(X_list) if X_list else np.zeros((0, 250, 16), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    if return_groups:
        return X, y, np.array(groups, dtype=object)
    return X, y


def _split_trials(y, groups):
    """Stratify unique trials, never their overlapping windows."""
    rng = np.random.default_rng(42)
    heldout = []
    for label in (0, 1):
        trials = np.unique(groups[y == label])
        if len(trials) < 2:
            raise ValueError("Train/validation require at least two distinct trials per class T1/T2.")
        rng.shuffle(trials)
        heldout.extend(trials[:max(1, int(np.ceil(len(trials) * .2)))])
    val = np.isin(groups, heldout)
    return np.flatnonzero(~val), np.flatnonzero(val)


def _infer_group_id_from_path(csv_path: str | Path) -> str:
    path = Path(csv_path)
    candidates = [path.parent.name, path.stem]
    for candidate in candidates:
        match = re.search(r"([PS]\d{2,4})", candidate, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if path.parent.name:
        return path.parent.name
    return path.stem


def _build_training_callbacks():
    callbacks = []
    try:
        import importlib  # lazy import to avoid hard dependency at import time
        tf_cb = importlib.import_module('tensorflow.keras.callbacks')
        EarlyStopping = getattr(tf_cb, 'EarlyStopping')
        ReduceLROnPlateau = getattr(tf_cb, 'ReduceLROnPlateau')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-4)
        ]
    except Exception:
        callbacks = []
    return callbacks


def _collect_windowed_dataset(
    csv_files: List[str],
    *,
    window_size: int = 250,
    step: int = 125,
    fs: float = 125.0,
    apply_filter: bool = True,
    band: Tuple[float, float] = (8.0, 30.0),
    group_resolver: Optional[Callable[[str], str]] = None,
    left_right_only: bool = False,
    source_stages=None,
    raw_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SubjectWindowSummary]]:
    all_X = []
    all_y = []
    all_groups = []
    summaries_by_group: Dict[str, SubjectWindowSummary] = {}
    resolver = group_resolver or _infer_group_id_from_path
    seen_sources = {}

    for path in csv_files:
        if left_right_only and not is_left_right_training_file(path):
            continue
        group_id = str(resolver(path))
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest in seen_sources:
            if seen_sources[digest] != group_id:
                raise ValueError("Duplicate CSV assigned to different subjects; leakage risk.")
            continue
        seen_sources[digest] = group_id
        data, markers = _load_openbci_csv(Path(path), source_stages=source_stages)
        X, y = _create_windows_ht(
            data,
            markers,
            window_size=window_size,
            step=step,
            fs=fs,
            apply_filter=apply_filter,
            band=band,
            raw_windows=raw_windows,
        )
        if len(X) == 0:
            continue

        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.array([group_id] * len(y), dtype=object))

        summary = summaries_by_group.get(group_id)
        if summary is None:
            summary = SubjectWindowSummary(
                group_id=group_id,
                csv_files=[],
                windows=0,
                class_counts={0: 0, 1: 0},
            )
            summaries_by_group[group_id] = summary
        summary.csv_files.append(str(path))
        summary.windows += int(len(y))
        labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            summary.class_counts[int(label)] = summary.class_counts.get(int(label), 0) + int(count)

    if not all_X:
        channels = 16
        return (
            np.zeros((0, window_size, channels), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=object),
            [],
        )

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_groups, axis=0),
        list(summaries_by_group.values()),
    )


def load_generalized_windowed_dataset(
    csv_files: List[str],
    *,
    window_size: int = 250,
    step: int = 125,
    group_resolver: Optional[Callable[[str], str]] = None,
    left_right_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SubjectWindowSummary]]:
    """Carrega janelas rotuladas preservando o grupo/paciente de cada CSV.

    API de desenvolvimento para validar datasets generalizados sem treinar.
    """
    return _collect_windowed_dataset(
        csv_files,
        window_size=window_size,
        step=step,
        group_resolver=group_resolver,
        left_right_only=left_right_only,
    )


def train_from_csvs(csv_files: List[str],
                    window_size: int = 250,
                    step: int = 125,
                    epochs: int = 30,
                    batch_size: int = 32,
                    model_name: Optional[str] = None,
                    base_model_path: Optional[str] = None,
                    model_builder: Optional[Callable[[Tuple[int, int], int], Any]] = None,
                    model_loader: Optional[Callable[[str], Any]] = None,
                    fine_tune_lr: Optional[float] = None,
                    freeze_backbone: bool = False,
                    sample_weights: Optional[np.ndarray] = None) -> TrainResult:
    """Treina a partir de uma lista de CSVs OpenBCI.

    Salva o modelo em data/models e retorna métricas básicas.

    Fine-tuning ponderado (calibração): com `base_model_path`,
    `freeze_backbone=True` congela tudo exceto o Dense final (adaptação
    rápida com poucas trials) e `fine_tune_lr` recompila com LR próprio.
    `sample_weights` (N_train,) dá peso maior a trials novos.
    """
    import tensorflow as tf  # lança ImportError cedo se faltar

    # Carregar e empilhar dados (segmentação T1/T2 -> T0)
    all_X = []
    all_y = []
    all_groups = []
    source_stages = []
    for path in csv_files:
        data, markers = _load_openbci_csv(Path(path), source_stages=source_stages)
        X, y, groups = _create_windows_ht(
            data, markers,
            window_size=window_size, step=step,
            fs=125.0, apply_filter=True, band=(8.0, 30.0),
            return_groups=True, source_id=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        )
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_groups.append(groups)

    if not all_X:
        raise ValueError("Nenhuma janela válida encontrada nos CSVs fornecidos.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups)
    train_idx, val_idx = _split_trials(y, groups)
    X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    # Construir modelo novo ou continuar fine-tuning a partir de um checkpoint.
    expected_input_shape = (window_size, X.shape[-1])
    if base_model_path:
        base_path = Path(base_model_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Modelo base nao encontrado: {base_model_path}")
        base_manifest = read_pipeline_manifest(base_path)
        source_stages.extend(base_manifest["training_source_stages"])
        loader = model_loader or load_keras_model
        model = loader(str(base_path))
        model_input_shape = getattr(model, "input_shape", None)
        try:
            dims = list(model_input_shape)
        except TypeError:
            dims = None
        # Aceita (None,250,16) legado e (None,None,16) adaptativo; canais !=16 rejeitados.
        if (dims is None or len(dims) != 3 or dims[0] not in (None, -1)
                or dims[2] != 16
                or not (dims[1] in (None, -1) or (isinstance(dims[1], int) and dims[1] > 0))):
            raise ValueError("Modelo base incompativel: retreine com 16 canais (250 fixo ou tempo variavel).")
        if tuple(getattr(model, "output_shape", ())) != (None, 2):
            raise ValueError("Modelo base incompativel: retreine com duas classes T1/T2.")
        if freeze_backbone:
            # Congela tudo exceto o classificador final (Dense softmax).
            dense_layers = [layer for layer in model.layers
                            if layer.__class__.__name__ == "Dense"]
            head = dense_layers[-1] if dense_layers else model.layers[-1]
            for layer in model.layers:
                layer.trainable = layer is head
            import importlib as _il
            optimizers = _il.import_module("tensorflow.keras.optimizers")
            lr = float(fine_tune_lr) if fine_tune_lr else 1e-3
            model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
        elif fine_tune_lr:
            import importlib as _il
            optimizers = _il.import_module("tensorflow.keras.optimizers")
            model.compile(optimizer=optimizers.Adam(learning_rate=float(fine_tune_lr)),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
    else:
        builder = model_builder or (lambda input_shape, num_classes: build_cnn_1d(input_shape, num_classes))
        model = builder(expected_input_shape, 2)

    # Callbacks estilo HardThinking
    callbacks = _build_training_callbacks()

    # Treinar
    t0 = time.time()
    fit_kwargs = dict(validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(X_train):
            raise ValueError("sample_weights deve ter uma entrada por janela de treino.")
        fit_kwargs["sample_weight"] = weights
    history = model.fit(X_train, y_train, **fit_kwargs)
    try:
        eval_loss, eval_acc = model.evaluate(X_val, y_val, verbose=0)
        val_loss, val_acc = float(eval_loss), float(eval_acc)
    except Exception:
        val_loss, val_acc = None, None
    t1 = time.time()

    # Salvar
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = model_name or f"cnn1d_{int(t0)}"
    out_path = MODELS_DIR / f"{model_name}.keras"
    if base_model_path and out_path.resolve() == Path(base_model_path).resolve():
        raise ValueError("Use a new model_name; fine-tuning must preserve the base checkpoint.")
    model.save(out_path)
    write_pipeline_manifest(out_path, training_source_stages=source_stages)

    # Resultados
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    final_acc = float(history.history.get('accuracy', [None])[-1]) if 'accuracy' in history.history else None
    final_loss = float(history.history.get('loss', [None])[-1]) if 'loss' in history.history else None

    return TrainResult(
        model_path=str(out_path),
        final_accuracy=final_acc,
        final_loss=final_loss,
        history=hist,
        training_time=(t1 - t0),
        val_accuracy=val_acc,
        val_loss=val_loss
    )


def _balance_and_augment_train(X_train, y_train, groups_train, *,
                               cap_per_group: int = 400,
                               augment: bool = True,
                               seed: int = 42):
    """Balanceia grupos no treino (cap estratificado) e aumenta minoritarios.

    - Grupos acima do cap: subsample estratificado por classe (sem reposicao).
    - Grupos abaixo do cap (com augment=True): replicas com ruido gaussiano
      (sigma 0.08 pos-zscore), deslocamento temporal +-12 amostras e
      channel-dropout (p=0.5 zera 1 canal) — robustez a canais ausentes
      (ex: BNCI 2b tem 13/16 zerados).
    - Validação nunca é tocada (métrica honesta).
    """
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    groups_train = np.asarray(groups_train)
    keep_parts = []
    extra_X, extra_y, extra_g = [], [], []
    for group in np.unique(groups_train):
        idx = np.flatnonzero(groups_train == group)
        idx0 = idx[y_train[idx] == 0]
        idx1 = idx[y_train[idx] == 1]
        if len(idx) > cap_per_group:
            if len(idx0) > 0 and len(idx1) > 0:
                n0 = max(1, int(round(cap_per_group * len(idx0) / len(idx))))
                n1 = cap_per_group - n0
                sel = np.concatenate([
                    rng.choice(idx0, min(n0, len(idx0)), replace=False),
                    rng.choice(idx1, min(n1, len(idx1)), replace=False),
                ])
            else:
                sel = rng.choice(idx, cap_per_group, replace=False)
            rng.shuffle(sel)
            keep_parts.append(sel)
            base = sel
        else:
            keep_parts.append(idx)
            base = idx
        if augment and len(base) < cap_per_group and len(base) > 0:
            need = int(cap_per_group - len(base))
            src = rng.choice(base, need, replace=True)
            aug = np.asarray(X_train[src], dtype=np.float64).copy()
            aug += rng.normal(0.0, 0.08, size=aug.shape)
            n_time = aug.shape[1]
            for i in range(len(aug)):
                shift = int(rng.integers(-12, 13))
                if shift:
                    aug[i] = np.roll(aug[i], shift, axis=0)
                if rng.random() < 0.5:
                    aug[i][:, int(rng.integers(0, aug.shape[2]))] = 0.0
            extra_X.append(aug.astype(np.float32))
            extra_y.append(np.asarray(y_train[src]))
            extra_g.append(np.asarray(groups_train[src]))
    keep = np.concatenate(keep_parts)
    if extra_X:
        Xb = np.concatenate([np.asarray(X_train[keep])] + extra_X, axis=0)
        yb = np.concatenate([np.asarray(y_train[keep])] + extra_y, axis=0)
        gb = np.concatenate([np.asarray(groups_train[keep])] + extra_g, axis=0)
    else:
        Xb, yb, gb = (np.asarray(X_train[keep]),
                      np.asarray(y_train[keep]),
                      np.asarray(groups_train[keep]))
    perm = rng.permutation(len(Xb))
    return Xb[perm], yb[perm], gb[perm]


def train_generalized_from_csvs(
    csv_files: List[str],
    window_size: int = 250,
    step: int = 125,
    epochs: int = 30,
    batch_size: int = 32,
    model_name: Optional[str] = None,
    group_resolver: Optional[Callable[[str], str]] = None,
    model_builder: Optional[Callable[[Tuple[int, int], int], Any]] = None,
    validation_size: float = 0.2,
    left_right_only: bool = True,
    max_windows_per_group: int = 400,
    augment: bool = True,
    euclidean_alignment: bool = False,
) -> GeneralizedTrainResult:
    """Treina modelo dev/generalizado com validação por grupo/paciente.

    Diferente de `train_from_csvs`, este fluxo nunca mistura janelas do mesmo
    grupo entre treino e validação. O grupo padrão é inferido do caminho do CSV
    (ex: pasta/arquivo contendo P001, P002, S001, etc.).

    Balanceamento: o treino limita cada grupo a `max_windows_per_group`
    janelas (estratificado por classe) e completa grupos minoritários com
    aumento (ruído + shift temporal + channel-dropout). A validação é
    intocada. `max_windows_per_group=None` desliga o balanceamento.

    Euclidean Alignment (He & Wu 2020): com `euclidean_alignment=True`,
    cada grupo tem suas janelas RAW branqueadas pela propria referencia
    (R-barra^{-1/2}) antes do filtro+z-score — nao supervisionado, sem
    leakage de labels. No uso ao vivo, a referencia do sujeito e estimada
    nos primeiros 30 s de EEG (EASessionAligner no gateway).
    """
    import tensorflow as tf  # noqa: F401  # lança ImportError cedo se faltar

    if not csv_files:
        raise ValueError("Lista de CSVs nao pode ser vazia.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size deve ficar entre 0 e 1.")

    source_stages = []
    X, y, groups, summaries = _collect_windowed_dataset(
        csv_files,
        window_size=window_size,
        step=step,
        group_resolver=group_resolver,
        left_right_only=left_right_only,
        source_stages=source_stages,
        raw_windows=bool(euclidean_alignment),
    )
    if len(X) == 0:
        raise ValueError("Nenhuma janela válida encontrada nos CSVs fornecidos.")

    if euclidean_alignment:
        from .eeg_pipeline import (apply_ea, bandpass_window, ea_reference_matrix,
                                   ea_whitening_matrix, zscore_window)
        # Placement correto (He & Wu / revisita 2025): filtro temporal
        # primeiro, EA sobre janelas filtradas, z-score por ultimo.
        filt = np.stack([bandpass_window(w) for w in np.asarray(X, dtype=np.float64)])
        aligned = np.zeros_like(filt)
        for group in np.unique(groups):
            mask = groups == group
            ref = ea_reference_matrix(filt[mask])
            aligned[mask] = apply_ea(filt[mask], ea_whitening_matrix(ref))
        # Pipeline fechado apos o alinhamento: z-score por canal.
        X = np.stack([zscore_window(w) for w in aligned]).astype(np.float32)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            "Treino generalizado exige ao menos dois grupos/pacientes distintos."
        )

    if len(np.unique(y)) < 2:
        raise ValueError("Treino generalizado exige janelas das classes T1 e T2.")

    try:
        from sklearn.model_selection import GroupShuffleSplit
    except Exception as e:
        raise ImportError("scikit-learn nao esta instalado. 'pip install scikit-learn'") from e

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    if set(y_train) != {0, 1} or set(y_val) != {0, 1}:
        raise ValueError("Both train and validation must contain T1 and T2; revise subject groups.")
    train_groups = sorted(str(group) for group in np.unique(groups[train_idx]))
    heldout_groups = sorted(str(group) for group in np.unique(groups[val_idx]))

    if max_windows_per_group is not None and int(max_windows_per_group) > 0:
        X_train, y_train, _ = _balance_and_augment_train(
            X_train, y_train, groups[train_idx],
            cap_per_group=int(max_windows_per_group), augment=augment)
    n0 = int(np.sum(y_train == 0))
    n1 = int(np.sum(y_train == 1))
    class_weight = None
    if n0 > 0 and n1 > 0:
        total = float(n0 + n1)
        class_weight = {0: total / (2.0 * n0), 1: total / (2.0 * n1)}

    builder = model_builder or (lambda input_shape, num_classes: build_cnn_1d(input_shape, num_classes))
    model = builder((window_size, X.shape[-1]), 2)
    callbacks = _build_training_callbacks()

    t0 = time.time()
    fit_kwargs = dict(validation_data=(X_val, y_val), epochs=epochs,
                      batch_size=batch_size, callbacks=callbacks, verbose=0)
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight
    history = model.fit(X_train, y_train, **fit_kwargs)
    try:
        eval_loss, eval_acc = model.evaluate(X_val, y_val, verbose=0)
        val_loss = float(eval_loss)
        val_acc = float(eval_acc)
    except Exception:
        val_loss, val_acc = None, None

    group_metrics: Dict[str, Dict[str, float]] = {}
    try:
        probabilities = model.predict(X_val, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        for group_id in heldout_groups:
            mask = groups[val_idx] == group_id
            if np.any(mask):
                group_metrics[group_id] = {
                    "accuracy": float(np.mean(y_pred[mask] == y_val[mask])),
                    "windows": float(np.sum(mask)),
                }
    except Exception:
        group_metrics = {}

    t1 = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = model_name or f"generalized_cnn1d_{int(t0)}"
    out_path = MODELS_DIR / f"{model_name}.keras"
    model.save(out_path)
    write_pipeline_manifest(out_path, training_source_stages=source_stages,
                            euclidean_alignment=bool(euclidean_alignment))

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    final_acc = float(history.history.get('accuracy', [None])[-1]) if 'accuracy' in history.history else None
    final_loss = float(history.history.get('loss', [None])[-1]) if 'loss' in history.history else None

    return GeneralizedTrainResult(
        model_path=str(out_path),
        final_accuracy=final_acc,
        final_loss=final_loss,
        history=hist,
        training_time=(t1 - t0),
        val_accuracy=val_acc,
        val_loss=val_loss,
        heldout_groups=heldout_groups,
        train_groups=train_groups,
        group_summaries=summaries,
        group_metrics=group_metrics,
    )


class ModelTrainer:
    """API simples de treinamento no estilo HardThinking, encapsulada em uma classe.

    Exemplos de uso:
        trainer = ModelTrainer()
        result = trainer.train_from_csvs(["S001/session1.csv"])  
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_from_csvs(self,
                        csv_files: List[str],
                        window_size: int = 250,
                        step: int = 125,
                        epochs: int = 30,
                        batch_size: int = 32,
                        model_name: Optional[str] = None,
                        base_model_path: Optional[str] = None,
                        fine_tune_lr: Optional[float] = None,
                        freeze_backbone: bool = False,
                        sample_weights: Optional[np.ndarray] = None) -> TrainResult:
        """Encapsula a função de treinamento de lista de CSVs. (opções de fine-tuning ponderado inclusas)"""
        # Delegamos para a função já implementada acima para reuso
        return train_from_csvs(
            csv_files=csv_files,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            base_model_path=base_model_path,
            fine_tune_lr=fine_tune_lr,
            freeze_backbone=freeze_backbone,
            sample_weights=sample_weights,
        )

    def train_from_directory(self,
                             directory: str,
                             pattern: str = "*.csv",
                             window_size: int = 250,
                             step: int = 125,
                             epochs: int = 30,
                             batch_size: int = 32,
                             model_name: Optional[str] = None,
                             base_model_path: Optional[str] = None) -> TrainResult:
        """Treina buscando todos os CSVs em um diretório (não recursivo).

        Útil para treinar rapidamente a partir de uma pasta de sujeito.
        """
        dir_path = Path(directory)
        csvs = [str(p) for p in dir_path.glob(pattern) if p.is_file()]
        if not csvs:
            raise ValueError(f"Nenhum CSV encontrado em {directory} com padrão {pattern}")
        return self.train_from_csvs(
            csv_files=csvs,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            base_model_path=base_model_path,
        )

    def train_generalized_from_csvs(self,
                                    csv_files: List[str],
                                    window_size: int = 250,
                                    step: int = 125,
                                    epochs: int = 30,
                                    batch_size: int = 32,
                                    model_name: Optional[str] = None,
                                    group_resolver: Optional[Callable[[str], str]] = None,
                                    validation_size: float = 0.2,
                                    left_right_only: bool = True,
                                    max_windows_per_group: int = 400,
                                    augment: bool = True,
                                    euclidean_alignment: bool = False) -> GeneralizedTrainResult:
        """Treino dev/generalizado com validação por grupo/paciente."""
        return train_generalized_from_csvs(
            csv_files=csv_files,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            group_resolver=group_resolver,
            validation_size=validation_size,
            left_right_only=left_right_only,
            max_windows_per_group=max_windows_per_group,
            augment=augment,
            euclidean_alignment=euclidean_alignment,
        )

    def train_generalized_from_directory(self,
                                         directory: str,
                                         pattern: str = "*.csv",
                                         window_size: int = 250,
                                         step: int = 125,
                                         epochs: int = 30,
                                         batch_size: int = 32,
                                         model_name: Optional[str] = None,
                                         validation_size: float = 0.2,
                                         left_right_only: bool = True) -> GeneralizedTrainResult:
        """Treino dev/generalizado buscando CSVs recursivamente por diretório."""
        dir_path = Path(directory)
        csvs = [str(p) for p in dir_path.rglob(pattern) if p.is_file()]
        if not csvs:
            raise ValueError(f"Nenhum CSV encontrado em {directory} com padrão {pattern}")
        return self.train_generalized_from_csvs(
            csv_files=csvs,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            validation_size=validation_size,
            left_right_only=left_right_only,
        )
