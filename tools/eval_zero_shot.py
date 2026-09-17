"""Zero-shot dos candidatos a modelo-base em sujeitos nao vistos no FT.
Uso: python tools/eval_zero_shot.py --out metrics_zeroshot.json
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainbridge_v2.infrastructure.ml import trainer as T

DATA_ROOT = Path(__file__).resolve().parent / "downloader" / "data"
CANDIDATES = {
    "generalized_left_right": "brainbridge_v2/infrastructure/data/models/generalized_left_right_eegmmidb_20260518_231607.keras",
    "metrics_generalized_cnn": "brainbridge_v2/infrastructure/data/models/metrics_generalized_cnn.keras",
    "modelo_full": "brainbridge_v2/data/models/modelo_full.keras",
}
SUBJECT_RUNS = {"S011": [3, 4, 7, 12], "S012": [3, 7, 8, 11, 12], "S016": [3, 4, 7, 8]}


def _real_csvs(subject, runs):
    out = []
    for run in runs:
        cands = sorted(DATA_ROOT.rglob(f"{subject}R{run:02d}_csv_openbci.csv"))
        cands = [c for c in cands if c.stat().st_size > 10000]
        if not cands:
            raise FileNotFoundError(f"missing real CSV {subject} R{run:02d}")
        out.append(str(cands[0]))
    return out


def _windows(files):
    xs, ys = [], []
    for f in files:
        data, markers = T._load_openbci_csv(Path(f))
        sid = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        x, y, _g = T._create_windows_ht(data, markers, return_groups=True, source_id=sid)
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))
    return np.concatenate(xs), np.concatenate(ys).astype(int)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics_zeroshot.json")
    a = ap.parse_args()
    import tensorflow as tf
    report = {}
    for name, path in CANDIDATES.items():
        model = tf.keras.models.load_model(path)
        per_subject, ns = {}, {}
        for subj, runs in SUBJECT_RUNS.items():
            try:
                files = _real_csvs(subj, runs)
            except FileNotFoundError as exc:
                print(f"skip {subj} p/ {name}: {exc}")
                continue
            x, y = _windows(files)
            p = np.asarray(model.predict(np.asarray(x, dtype=np.float32),
                                         batch_size=64, verbose=0)).argmax(axis=-1)
            acc = float((p.reshape(-1) == y.reshape(-1)).mean())
            per_subject[subj] = round(acc, 4)
            ns[subj] = int(len(y))
            print(f"{name} {subj}: {acc:.4f} (n={len(y)})", flush=True)
        vals = list(per_subject.values())
        report[name] = {"per_subject": per_subject, "n": ns,
                         "mean": round(sum(vals) / len(vals), 4) if vals else None}
        del model
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
