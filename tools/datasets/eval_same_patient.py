"""Same-patient eval harness: base -> fine-tune -> RL-stream -> test.

Protocol (default S012, subject unseen by base):
  FT   = R03 + R07 (executed, 28 trials)
  RL   = R08 (imagined, live order, 14 trials)
  TEST = R11 + R12 (exec + imag, 28 trials, 112 windows)

Mirrors the app: trainer.train_from_csvs for calibration, gateway
rl_online_update (freeze_backbone=False, mistake weight 3x) for RL.
Usage:
  python tools/datasets/eval_same_patient.py --base <base.keras>
  python tools/datasets/eval_same_patient.py --base <b> --epochs 10 --lr 5e-05
"""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainbridge_v2.infrastructure.ml import trainer as T
from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
    TensorFlowInferenceGatewayAdapter,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "downloader" / "data"


def _real_csvs(subject, runs):
    out = []
    for run in runs:
        cands = sorted(DATA_ROOT.rglob(f"{subject}R{run:02d}_csv_openbci.csv"))
        cands = [c for c in cands if c.stat().st_size > 10000]
        if not cands:
            raise FileNotFoundError(f"missing real CSV {subject} R{run:02d}")
        out.append(str(cands[0]))
    return out


def _windows(files, raw=False):
    xs, ys = [], []
    for f in files:
        data, markers = T._load_openbci_csv(Path(f))
        sid = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        x, y, _g = T._create_windows_ht(
            data, markers, return_groups=True, source_id=sid, raw_windows=raw)
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))
    return np.concatenate(xs), np.concatenate(ys).astype(int)


def _acc(model, x, y):
    p = np.asarray(model.predict(np.asarray(x, dtype=np.float32),
                                 batch_size=64, verbose=0)).argmax(axis=-1)
    return float((p.reshape(-1) == np.asarray(y).reshape(-1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subject", default="S012")
    ap.add_argument("--ft-runs", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--rl-run", type=int, default=8)
    ap.add_argument("--test-runs", nargs="+", type=int, default=[11, 12])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rl-k", type=int, default=5)
    ap.add_argument("--rl-epochs", type=int, default=3)
    ap.add_argument("--rl-lr", type=float, default=5e-5)
    ap.add_argument("--rl-augment", action="store_true")
    ap.add_argument("--ft-augment", action="store_true")
    ap.add_argument("--ft-cw", action="store_true")
    ap.add_argument("--out", default="metrics_eval.json")
    a = ap.parse_args()

    ft_files = _real_csvs(a.subject, a.ft_runs)
    rl_files = _real_csvs(a.subject, [a.rl_run])
    test_files = _real_csvs(a.subject, a.test_runs)
    x_test, y_test = _windows(test_files)
    print(f"TEST windows={len(y_test)}", flush=True)

    import tensorflow as tf
    base = tf.keras.models.load_model(a.base)
    acc_base = _acc(base, x_test, y_test)
    print(f"base TEST acc={acc_base:.4f}", flush=True)
    del base

    import time
    tag = f"eval_{a.subject}_{int(time.time()) % 1000000}"
    res = T.train_from_csvs(ft_files, model_name=tag, base_model_path=a.base,
                             epochs=a.epochs, fine_tune_lr=a.lr,
                             finetune_augment=bool(a.ft_augment),
                             finetune_class_weight=bool(a.ft_cw))
    print(f"FT val={res.val_accuracy:.4f} train={res.final_accuracy:.4f}", flush=True)
    ft_model = tf.keras.models.load_model(res.model_path)
    acc_ft = _acc(ft_model, x_test, y_test)
    print(f"FT TEST acc={acc_ft:.4f}", flush=True)
    del ft_model

    gw = TensorFlowInferenceGatewayAdapter()
    gw.load_model(res.model_path)
    x_raw, y_raw = _windows(rl_files, raw=True)
    print(f"RL stream windows={len(y_raw)}", flush=True)
    traj = [round(acc_ft, 4)]
    buf_w, buf_y, buf_sw, n_up = [], [], [], 0
    preq_correct, preq_n = 0, 0
    for w, y in zip(x_raw, y_raw):
        try:
            pr = gw.predict(np.asarray(w, dtype=np.float64))
            correct = (int(pr.predicted_index) == int(y))
        except Exception:
            correct = False
        preq_n += 1
        preq_correct += int(correct)
        buf_w.append(np.asarray(w, dtype=np.float64))
        buf_y.append(int(y))
        buf_sw.append(1.0 if correct else 3.0)
        if len(buf_w) >= a.rl_k:
            gw.rl_online_update(buf_w, buf_y, sample_weights=np.array(buf_sw),
                                epochs=a.rl_epochs, lr=a.rl_lr,
                                freeze_backbone=False, augment=bool(a.rl_augment))
            n_up += 1
            m = gw._adapter.model
            traj.append(round(_acc(m, x_test, y_test), 4))
            print(f"RL update {n_up}: TEST acc={traj[-1]:.4f}", flush=True)
            buf_w, buf_y, buf_sw = [], [], []
    out = {
        "subject": a.subject,
        "base": a.base,
        "test_windows": int(len(y_test)),
        "epochs": a.epochs,
        "ft_augment": bool(a.ft_augment),
        "ft_cw": bool(a.ft_cw),
        "lr": a.lr,
        "acc_base": acc_base,
        "ft_val": res.val_accuracy,
        "ft_train": res.final_accuracy,
        "acc_ft": acc_ft,
        "rl_run": a.rl_run,
        "rl_stream_windows": int(len(y_raw)),
        "rl_updates": n_up,
        "rl_trajectory": traj,
        "rl_prequential": round(preq_correct / max(1, preq_n), 4),
    }
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    try:
        Path(res.model_path).unlink()
        Path(res.model_path).with_suffix(".pipeline.json").unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
