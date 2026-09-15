"""Common OpenBCI standardization for public MI datasets.

Canonical profile (OpenBCI Daisy 16ch):
- 16 EXG channels, 125 Hz, RAW uV, OpenBCI CSV with Annotations T1/T2(/T0).
- Live Daisy hardware is 125 Hz (some GUIs label it 128 Hz); the adaptive
  pipeline (eeg_pipeline.preprocess_window_adaptive) also accepts 128/250 Hz.

Target montage (matches brainbridge_v2/infrastructure/config/constants.py):
  Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4
"""

from pathlib import Path

OPENBCI_16 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4",
]

CANONICAL_FS = 125.0

# Fallback motor-priority order when exact 10-20 names are missing
# (e.g. BCI IV 2a has FC/CP but no Fp/T).
MOTOR_FALLBACK = [
    "C3", "C4", "Cz", "C1", "C2", "C5", "C6",
    "Fc3", "Fc4", "Fc1", "Fc2", "Fcz", "Fc5", "Fc6",
    "Cp3", "Cp4", "Cp1", "Cp2", "Cpz", "Cp5", "Cp6",
    "F3", "F4", "Fz", "P3", "P4", "Pz",
    "T7", "T8", "F7", "F8", "Fp1", "Fp2",
    "P7", "P8", "P1", "P2", "Poz",
]


def _norm_ch(name: str) -> str:
    """Normaliza nome de eletrodo: minusculas, sem pontos/espacos extras.

    PhysioNet/MNE expoe 'Fc5.', 'C3..'; BNCI expoe 'C3'. Sem isso o
    mapeamento exato falha e o fallback embaralha a montagem.
    """
    import re
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def map_channels_to_openbci16(available_names):
    """Return list of 16 source names (or None for zero-pad) for OPENBCI_16."""
    lut = {_norm_ch(n): str(n) for n in available_names}
    used = set()
    mapping = []
    for want in OPENBCI_16:
        key = _norm_ch(want)
        if key in lut and lut[key] not in used:
            mapping.append(lut[key])
            used.add(lut[key])
            continue
        # fallback: first unused motor channel
        pick = None
        for fb in MOTOR_FALLBACK:
            src = lut.get(_norm_ch(fb))
            if src is not None and src not in used:
                pick = src
                used.add(src)
                break
        if pick is None:
            # any unused channel
            for src in available_names:
                if src not in used:
                    pick = src
                    used.add(src)
                    break
        mapping.append(pick)  # None -> zero-pad
    return mapping


def write_openbci_csv(path, data_uv_16xN_or_Nx16, *, fs_in, annotations,
                      source_dataset, source_subject, source_run="",
                      channel_mapping=None, signal_stage="raw"):
    """Write canonical OpenBCI CSV (16ch @ 125 Hz).

    data: np.ndarray shape (n_samples, 16) in microvolts @ 125 Hz.
    annotations: iterable of (sample_index_0based, "T1"/"T2"/"T0").
    """
    import numpy as np
    from datetime import datetime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data_uv_16xN_or_Nx16, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 16:
        raise ValueError("data must be (n_samples, 16).")
    n = len(data)
    marks = [""] * n
    for idx, code in annotations:
        i = int(idx)
        if 0 <= i < n and code in ("T1", "T2", "T0"):
            if not marks[i]:
                marks[i] = code
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["%OpenBCI Raw EXG Data"])
        w.writerow(["%Number of channels = 16"])
        w.writerow([f"%Sample Rate = {CANONICAL_FS:.1f} Hz"])
        w.writerow([f"%Signal Stage = {signal_stage}"])
        w.writerow(["%Timestamp Source = dataset conversion; resampled to 125 Hz"])
        w.writerow([f"%Source Dataset = {source_dataset}"])
        w.writerow([f"%Source Subject = {source_subject}"])
        if source_run:
            w.writerow([f"%Source Run = {source_run}"])
        w.writerow([f"%Source Fs = {fs_in} Hz"])
        w.writerow([f"%Montage = openbci16 target {','.join(OPENBCI_16)}"])
        if channel_mapping:
            w.writerow([f"%Channel Mapping = {channel_mapping}"])
        w.writerow(["%Board = OpenBCI_GUI$BoardCytonSerialDaisy"])
        w.writerow(
            ["Sample Index"]
            + [f"EXG Channel {i}" for i in range(16)]
            + ["Accel Channel 0", "Accel Channel 1", "Accel Channel 2"]
            + ["Other", "Other.1", "Other.2", "Other.3", "Other.4", "Other.5", "Other.6"]
            + ["Analog Channel 0", "Analog Channel 1", "Analog Channel 2"]
            + ["Timestamp", "Other.7", "Timestamp (Formatted)", "Annotations"]
        )
        t0 = datetime.now().isoformat(timespec="milliseconds")
        for i in range(n):
            row = [i]
            row.extend([f"{v:.5g}" for v in data[i].tolist()])
            row.extend([0, 0, 0])  # accel
            row.extend([0, 0, 0, 0, 0, 0, 0])  # other
            row.extend([0, 0, 0])  # analog
            row.extend([0, 0, t0, marks[i]])
            w.writerow(row)
    return str(path)


def resample_to_125(data_TxC, fs_in):
    """Resample (n, C) from fs_in to 125 Hz (no-op if already 125)."""
    import numpy as np
    from scipy.signal import resample as _resample

    data = np.asarray(data_TxC, dtype=np.float64)
    if abs(float(fs_in) - CANONICAL_FS) < 1e-9:
        return data
    n_target = max(1, int(round(len(data) * CANONICAL_FS / float(fs_in))))
    out = np.asarray(_resample(data, n_target, axis=0), dtype=np.float64)
    return out


def group_id_for(dataset, subject):
    return f"{dataset}:S{subject}"
