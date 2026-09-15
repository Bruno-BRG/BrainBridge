"""Treino generalizado multi-dataset (PhysioNet + BNCI IV 2a/2b + locais).

Group = dataset:sujeito (evita leakage entre treino/validacao).
Exige >=2 grupos e classes T1+T2 em ambas as particoes.

Usage:
  python tools/datasets/train_generalized_all.py --data tools/datasets/data --epochs 30 --model-name generalized_mi_v1
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def group_resolver(path: str) -> str:
    p = Path(path)
    name = p.stem
    # bnci2a_S01... -> bnci2a:S01 (sujeito; sessoes/runs do mesmo sujeito
    # ficam no mesmo grupo para Leave-Subject-Out de verdade).
    m = re.search(r"bnci(2a|2b)[._-]*S(\d{2})", name, re.I)
    if m:
        return f"bnci{m.group(1).lower()}:S{m.group(2)}"
    m = re.search(r"physionet[._-]*S(\d{3})", name, re.I)
    if m:
        return f"physionet:S{m.group(1)}"
    # physionet_S001_R04_openbci / bnci2a_S01T_openbci / P001_... (locais)
    m = re.search(r"(physionet|bnci2?a|bnci2?b)[._-]*S(\d+[A-Z]?)", name, re.I)
    if m:
        return f"{m.group(1).lower()}:S{m.group(2).upper()}"
    m = re.search(r"S(\d{3})R(\d{2})", name, re.I)
    if m:
        return f"physionet:S{m.group(1)}"
    m = re.search(r"([PS]\d{2,4})", name, re.I)
    if m:
        return m.group(1).upper()
    return p.parent.name or p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["tools/datasets/data"],
                    help="pastas com *_openbci.csv (busca recursiva)")
    ap.add_argument("--pattern", default="*_openbci.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--validation-size", type=float, default=0.2)
    ap.add_argument("--arch", choices=["cnn", "eegnet", "shallow"], default="cnn")
    ap.add_argument("--cap", type=int, default=400,
                    help="max de janelas por grupo no treino (0 desliga)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--l2", type=float, default=None)
    ap.add_argument("--ea", dest="ea", action="store_true", default=True,
                    help="Euclidean Alignment por grupo (default: sim)")
    ap.add_argument("--no-ea", dest="ea", action="store_false",
                    help="desliga o Euclidean Alignment")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    csvs = []
    for d in args.data:
        csvs.extend(str(p) for p in Path(d).rglob(args.pattern) if p.is_file())
    csvs = sorted(set(csvs))
    print(f"CSVs encontrados: {len(csvs)}")
    groups = {}
    for c in csvs:
        g = group_resolver(c)
        groups.setdefault(g, []).append(c)
    for g in sorted(groups):
        print(f"  {g}: {len(groups[g])} arquivos")
    if args.list_only or not csvs:
        return
    if len(groups) < 2:
        raise SystemExit("Treino generalizado exige >=2 grupos/sujeitos.")

    from brainbridge_v2.infrastructure.ml.trainer import train_generalized_from_csvs
    if args.arch == "eegnet":
        from brainbridge_v2.infrastructure.ml.models import build_eegnet_adaptive
        kw = {}
        if args.dropout is not None:
            kw["dropout"] = float(args.dropout)
        if args.l2 is not None:
            kw["l2"] = float(args.l2)
        builder = lambda shape, n: build_eegnet_adaptive(num_classes=n, channels=shape[1], **kw)
    elif args.arch == "shallow":
        from brainbridge_v2.infrastructure.ml.models import build_shallow_convnet_adaptive
        kw = {}
        if args.dropout is not None:
            kw["dropout"] = float(args.dropout)
        if args.l2 is not None:
            kw["l2"] = float(args.l2)
        builder = lambda shape, n: build_shallow_convnet_adaptive(num_classes=n, channels=shape[1], **kw)
    else:
        builder = None
    result = train_generalized_from_csvs(
        csvs, epochs=args.epochs, batch_size=args.batch_size,
        model_name=args.model_name or f"generalized_mi_multidataset_{args.arch}",
        group_resolver=group_resolver, validation_size=args.validation_size,
        left_right_only=True, model_builder=builder,
        max_windows_per_group=(None if args.cap == 0 else args.cap),
        augment=(not args.no_augment),
        euclidean_alignment=bool(args.ea))
    print(f"Modelo: {result.model_path}")
    print(f"val_acc={result.val_accuracy} val_loss={result.val_loss}")
    print(f"train_groups={result.train_groups}")
    print(f"heldout_groups={result.heldout_groups}")
    for gid, m in (result.group_metrics or {}).items():
        print(f"  heldout {gid}: acc={m.get('accuracy')} n={m.get('windows')}")


if __name__ == "__main__":
    main()
