"""Head-only vs full-network FT on S012 imagined (R08 -> R12)."""
import hashlib, json, sys, warnings, time
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
import tensorflow as tf
base = "brainbridge_v2/infrastructure/data/models/metrics_generalized_cnn.keras"
xt, yt = _win(_real("S012", [12]))
ft = _real("S012", [8])
for freeze, nm in [(True, "head-only"), (False, "full")]:
    for sd in (7, 21):
        tf.keras.utils.set_random_seed(sd); np.random.seed(sd)
        tag = f"evalhd_{sd}_{int(time.time()) % 1000000}"
        r = T.train_from_csvs(ft, model_name=tag, base_model_path=base,
            epochs=10, fine_tune_lr=5e-5, freeze_backbone=freeze,
            finetune_augment=True, finetune_class_weight=True)
        m = tf.keras.models.load_model(r.model_path)
        p = np.asarray(m.predict(np.asarray(xt, dtype=np.float32), batch_size=64, verbose=0)).argmax(-1)
        print(f"{nm} seed={sd} val={r.val_accuracy:.4f} TEST={float((p.reshape(-1)==yt.reshape(-1)).mean()):.4f}", flush=True)
        try:
            Path(r.model_path).unlink(); Path(r.model_path).with_suffix(".pipeline.json").unlink()
        except OSError:
            pass
