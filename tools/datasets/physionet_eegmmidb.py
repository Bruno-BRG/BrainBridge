"""PhysioNet EEGMMIDB -> OpenBCI 16ch @ 125 Hz (left/right hand MI).

Uses only unilateral fist runs (protocol map in
brainbridge_v2/infrastructure/ml/physionet_eegmmidb_protocol.py):
  executed R03/R07/R11 + imagined R04/R08/R12  (T1=left, T2=right).
Baseline (R01/R02) and both-fists/feet (R05/R06/R09/R10/R13/R14) are skipped.

Usage:
  python tools/datasets/physionet_eegmmidb.py --subjects 1 2 --runs 4 8 12 --out tools/datasets/data
  python tools/datasets/physionet_eegmmidb.py --subjects 1-20 --runs imagined --out tools/datasets/data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMAGINED_RUNS = [4, 8, 12]
EXECUTED_RUNS = [3, 7, 11]
LEFT_RIGHT_RUNS = [3, 4, 7, 8, 11, 12]


def parse_subjects(spec):
    out = []
    for part in spec:
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["1", "2"],
                    help="ex: 1 2 3 ou 1-20")
    ap.add_argument("--runs", nargs="+", default=["imagined"],
                    help="ex: 4 8 12 ou 'imagined' 'executed' 'all-lr'")
    ap.add_argument("--out", default="tools/datasets/data/physionet")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    runs = []
    for r in args.runs:
        rl = r.lower()
        if rl == "imagined":
            runs.extend(IMAGINED_RUNS)
        elif rl == "executed":
            runs.extend(EXECUTED_RUNS)
        elif rl in ("all-lr", "all"):
            runs.extend(LEFT_RIGHT_RUNS)
        else:
            runs.append(int(r))
    runs = sorted(set(r for r in runs if r in LEFT_RIGHT_RUNS))
    if not runs:
        raise SystemExit("Nenhum run left/right valido (3,4,7,8,11,12).")

    import mne
    from mne.datasets import eegbci
    from common import (OPENBCI_16, map_channels_to_openbci16,
                        resample_to_125, write_openbci_csv)
    import numpy as np

    try:
        from brainbridge_v2.infrastructure.ml.physionet_eegmmidb_protocol import (
            describe_run)
    except Exception:
        def describe_run(r):
            return f"R{r:02d}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = out_dir / "_mne_cache"
    dl_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for subj in subjects:
        for run in runs:
            fname = out_dir / f"physionet_S{subj:03d}_R{run:02d}_openbci.csv"
            if fname.exists() and not args.overwrite:
                print(f"skip {fname.name} (existe)")
                total += 1
                continue
            print(f"[{subj:03d}/R{run:02d}] {describe_run(run)} ...", flush=True)
            paths = eegbci.load_data([subj], runs=[run], path=str(dl_dir),
                                     update_path=True, verbose=False)
            raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
            fs_in = float(raw.info["sfreq"])
            mapping = map_channels_to_openbci16(raw.ch_names)
            # Monta (n,16): pega canais existentes, zeros nos ausentes.
            picks = [c for c in mapping if c is not None]
            raw_pick = raw.copy().pick(picks)
            data_volts, _ = raw_pick[:, :]  # (Cpick, T)
            data_volts = data_volts.T  # (T, Cpick)
            # Reinsere na ordem OPENBCI_16.
            full = np.zeros((data_volts.shape[0], 16))
            j = 0
            for k, src in enumerate(mapping):
                if src is not None:
                    col = list(raw_pick.ch_names).index(src)
                    full[:, k] = data_volts[:, col]
            data_uv = full * 1e6
            data_125 = resample_to_125(data_uv, fs_in)
            # Anotacoes: T0/T1/T2 ja no protocolo correto para estes runs.
            scale = len(data_125) / max(1, len(data_uv))
            annots = []
            for ann in raw.annotations:
                desc = str(ann["description"]).strip().upper()
                code = {"T0": "T0", "T1": "T1", "T2": "T2"}.get(desc)
                if code is None:
                    continue
                onset = float(ann["onset"]) - float(raw.first_time)
                old_idx = int(round(onset * fs_in))
                new_idx = int(round(old_idx * scale))
                annots.append((new_idx, code))
            write_openbci_csv(
                fname, data_125, fs_in=fs_in, annotations=annots,
                source_dataset="physionet-eegmmidb", source_subject=f"{subj:03d}",
                source_run=f"R{run:02d}",
                channel_mapping=";".join(f"{w}={s}" for w, s in zip(OPENBCI_16, mapping)))
            print(f"  -> {fname.name} ({len(data_125)} amostras, {len(annots)} marcas)")
            total += 1
    print(f"OK: {total} arquivos em {out_dir}")


if __name__ == "__main__":
    main()
