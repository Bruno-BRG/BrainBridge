"""Fast FT ablation: base -> fine-tune -> TEST (no RL). Multi-seed."""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brainbridge_v2.infrastructure.ml import trainer as T
DATA_ROOT = Path(__file__).resolve().parent.parent / "downloader" / "data"
def _real(subject, runs):
    out = []
    for run in runs:
        c = sorted(p for p in DATA_ROOT.rglob(f"{subject}R{run:02d}_csv_openbci.csv") if p.stat().st_size > 10000)
        out.append(str(c[0]))
    return out
def _win(files):
    xs, ys = [], []
    for f in files:
        d, m = T._load_openbci_csv(Path(f))
        x, y, _g = T._create_windows_ht(d, m, return_groups=True,
            source_id=hashlib.sha256(Path(f).read_bytes()).hexdigest())
        xs.append(np.asarray(x)); ys.append(np.asarray(y))
    return np.concatenate(xs), np.concatenate(ys).astype(int)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subject", default="S012")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ft-augment", action="store_true")
    ap.add_argument("--ft-cw", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--ft-runs", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--test-runs", nargs="+", type=int, default=[11, 12])
    ap.add_argument("--out", default="metrics_ft_abl.json")
    a = ap.parse_args()
    ft = _real(a.subject, a.ft_runs); te = _real(a.subject, a.test_runs)
    xt, yt = _win(te)
    import tensorflow as tf
    accs, vals = [], []
    for sd in a.seeds:
        tf.keras.utils.set_random_seed(sd); np.random.seed(sd)
        import time; tag = f"evalft_{a.subject}_{sd}_{int(time.time()) % 1000000}"
        r = T.train_from_csvs(ft, model_name=tag, base_model_path=a.base,
            epochs=a.epochs, fine_tune_lr=a.lr,
            finetune_augment=bool(a.ft_augment), finetune_class_weight=bool(a.ft_cw))
        m = tf.keras.models.load_model(r.model_path)
        p = np.asarray(m.predict(np.asarray(xt, dtype=np.float32), batch_size=64, verbose=0)).argmax(-1)
        acc = float((p.reshape(-1) == yt.reshape(-1)).mean())
        accs.append(round(acc, 4)); vals.append(round(float(r.val_accuracy), 4))
        print(f"seed={sd} val={r.val_accuracy:.4f} TEST={acc:.4f}", flush=True)
        try:
            Path(r.model_path).unlink(); Path(r.model_path).with_suffix(".pipeline.json").unlink()
        except OSError:
            pass
    out = {"subject": a.subject, "ft_runs": a.ft_runs, "test_runs": a.test_runs,
        "epochs": a.epochs, "lr": a.lr,
        "aug": bool(a.ft_augment), "cw": bool(a.ft_cw),
        "test_mean": round(float(np.mean(accs)), 4), "test_acc_runs": accs, "val_runs": vals}
    Path(a.out).write_text(json.dumps(out, indent=2)); print(json.dumps(out))
if __name__ == "__main__":
    main()
