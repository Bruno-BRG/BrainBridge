"""BCI Competition IV 2a/2b -> OpenBCI 16ch @ 125 Hz (left/right hand MI).

Fontes:
- 2a (9 subj, 22ch @ 250 Hz, 4 classes): usa so mao esquerda (769->T1)
  e mao direita (770->T2); pe/tongue ignorados. Trial imagery = 4 s.
- 2b (9 subj, 3ch C3/Cz/C4 @ 250 Hz, 2 classes): 769->T1, 770->T2.

Via preferencial: MOABB (espelhos atualizados, sem URL quebrada):
  python tools/datasets/bnci_iv_2a_2b.py --via moabb --dataset 2a --subjects 1 --out tools/datasets/data/bnci
Via direta (legado BNCI Horizon, pode estar fora do ar):
  python tools/datasets/bnci_iv_2a_2b.py --via direct --dataset 2a --subjects 1 --sessions T E

Usage padrao: --via moabb.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

URLS = {
    "2a": "http://bnci-horizon-2020.eu/database/data-sets/001-2014",
    "2b": "http://bnci-horizon-2020.eu/database/data-sets/002-2014",
}
PREFIX = {"2a": "A", "2b": "B"}
N_SUBJECTS = 9

LEFT_CODE = 769
RIGHT_CODE = 770
IMAGERY_S = 4.0


def download(url, dest: Path):
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"baixando {url} ...", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def convert_gdf(gdf_path: Path, out_csv: Path, *, dataset, subject, session):
    import mne
    import numpy as np
    from common import (OPENBCI_16, map_channels_to_openbci16,
                        resample_to_125, write_openbci_csv)

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    # Remove EOG/EOG-* e stimul channels: mantem so EEG.
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False,
                               exclude="bads")
    raw_eeg = raw.copy().pick(eeg_picks)
    fs_in = float(raw_eeg.info["sfreq"])
    mapping = map_channels_to_openbci16(raw_eeg.ch_names)
    data_v, _ = raw_eeg[:, :]
    data_v = data_v.T
    full = np.zeros((data_v.shape[0], 16))
    name_to_col = {n: i for i, n in enumerate(raw_eeg.ch_names)}
    for k, src in enumerate(mapping):
        if src is not None and src in name_to_col:
            full[:, k] = data_v[:, name_to_col[src]]
    data_uv = full * 1e6
    data_125 = resample_to_125(data_uv, fs_in)
    scale = len(data_125) / max(1, len(data_uv))

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # event_id mapeia desc->code; inverte para code->desc quando preciso.
    annots = []
    sf = fs_in
    first = float(raw.first_time)
    for ann in raw.annotations:
        desc = str(ann["description"]).strip()
        try:
            code = int(desc)
        except ValueError:
            # MNE pode expor "Comment/769" etc.
            code = None
            for token in desc.replace("/", " ").split():
                if token.isdigit():
                    code = int(token)
                    break
            if code is None:
                continue
        if code not in (LEFT_CODE, RIGHT_CODE):
            continue
        cue = "T1" if code == LEFT_CODE else "T2"
        onset = float(ann["onset"]) - first
        idx = int(round(onset * sf * scale))
        annots.append((idx, cue))
        # T0 ao fim da imaginacao (4 s depois do cue).
        end_idx = int(round((onset + IMAGERY_S) * 125.0))
        annots.append((end_idx, "T0"))
    annots.sort()
    write_openbci_csv(out_csv, data_125, fs_in=fs_in, annotations=annots,
                      source_dataset=f"bnci-iv-{dataset}",
                      source_subject=f"{subject:02d}{session}",
                      source_run=f"{PREFIX[dataset]}{subject:02d}{session}",
                      channel_mapping=";".join(f"{w}={s}" for w, s in zip(OPENBCI_16, mapping)))
    n_t1 = sum(1 for _, c in annots if c == "T1")
    n_t2 = sum(1 for _, c in annots if c == "T2")
    print(f"  -> {out_csv.name} ({len(data_125)} amostras, T1={n_t1} T2={n_t2})")


def raw_to_openbci(raw, out_csv: Path, *, dataset, subject, session):
    """Converte um MNE Raw (qualquer fs/canais) para OpenBCI 16ch @125Hz."""
    import numpy as np
    import mne
    from common import (OPENBCI_16, map_channels_to_openbci16,
                        resample_to_125, write_openbci_csv)

    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False,
                               exclude="bads")
    raw_eeg = raw.copy().pick(eeg_picks)
    fs_in = float(raw_eeg.info["sfreq"])
    mapping = map_channels_to_openbci16(raw_eeg.ch_names)
    data_v, _ = raw_eeg[:, :]
    data_v = data_v.T
    full = np.zeros((data_v.shape[0], 16))
    name_to_col = {n: i for i, n in enumerate(raw_eeg.ch_names)}
    for k, src in enumerate(mapping):
        if src is not None and src in name_to_col:
            full[:, k] = data_v[:, name_to_col[src]]
    data_uv = full * 1e6
    data_125 = resample_to_125(data_uv, fs_in)
    scale = len(data_125) / max(1, len(data_uv))
    annots = []
    first = float(raw.first_time)
    for ann in raw.annotations:
        desc = str(ann["description"]).strip()
        low = desc.lower()
        code = None
        try:
            code = int(desc)
        except ValueError:
            for token in desc.replace("/", " ").replace(",", " ").split():
                if token.isdigit():
                    code = int(token)
                    break
        if code in (LEFT_CODE,):
            cue = "T1"
        elif code in (RIGHT_CODE,):
            cue = "T2"
        elif low in ("left_hand", "left", "left-hand", "lefthand", "t1"):
            cue = "T1"
        elif low in ("right_hand", "right", "right-hand", "righthand", "t2"):
            cue = "T2"
        else:
            continue
        try:
            dur = float(ann["duration"])
        except Exception:
            dur = IMAGERY_S
        if not (1.0 <= dur <= 10.0):
            dur = IMAGERY_S
        onset = float(ann["onset"]) - first
        idx = int(round(onset * fs_in * scale))
        annots.append((idx, cue))
        end_idx = int(round((onset + dur) * 125.0))
        annots.append((end_idx, "T0"))
    annots.sort()
    write_openbci_csv(out_csv, data_125, fs_in=fs_in, annotations=annots,
                      source_dataset=f"bnci-iv-{dataset}",
                      source_subject=f"{subject:02d}{session}",
                      source_run=f"{PREFIX[dataset]}{subject:02d}{session}",
                      channel_mapping=";".join(f"{w}={s}" for w, s in zip(OPENBCI_16, mapping)))
    n_t1 = sum(1 for _, c in annots if c == "T1")
    n_t2 = sum(1 for _, c in annots if c == "T2")
    print(f"  -> {out_csv.name} ({len(data_125)} amostras, T1={n_t1} T2={n_t2})")


def convert_gdf(gdf_path: Path, out_csv: Path, *, dataset, subject, session):
    import mne

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    raw_to_openbci(raw, out_csv, dataset=dataset, subject=subject, session=session)


def convert_via_moabb(*, dataset, subjects, out_dir: Path, overwrite=False):
    """Baixa via MOABB (espelhos atuais) e converte cada sessao."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "2a":
        from moabb.datasets import BNCI2014_001 as DS
    elif dataset == "2b":
        from moabb.datasets import BNCI2014_004 as DS
    else:
        raise ValueError(dataset)
    ds = DS()

    def iter_raws(obj, prefix=""):
        # MOABB: subject -> session -> run -> Raw (niveis podem variar).
        if hasattr(obj, "info") and hasattr(obj, "annotations"):
            yield prefix, obj
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from iter_raws(v, f"{prefix}{k}_" if prefix else f"{k}_")
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from iter_raws(v, f"{prefix}{i}_")
            return

    total = 0
    for subj in subjects:
        data = ds.get_data(subjects=[subj])
        subj_data = data.get(subj, data)
        raws = list(iter_raws(subj_data))
        if not raws:
            print(f"[moabb] sujeito {subj}: nenhum Raw encontrado, pulando")
            continue
        for sess_key, raw in raws:
            tag = "".join(ch for ch in str(sess_key) if ch.isalnum())[:12] or "S"
            out_csv = out_dir / f"bnci{dataset}_S{subj:02d}{tag}_openbci.csv"
            if out_csv.exists() and not overwrite:
                print(f"skip {out_csv.name} (existe)")
                total += 1
                continue
            raw_to_openbci(raw, out_csv, dataset=dataset, subject=subj,
                           session=tag)
            total += 1
    print(f"OK: {total} arquivos em {out_dir}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2a", "2b"], default="2a")
    ap.add_argument("--via", choices=["moabb", "direct"], default="moabb")
    ap.add_argument("--subjects", nargs="+", default=["1"],
                    help="ex: 1 2 3 ou 1-9")
    ap.add_argument("--sessions", nargs="+", default=["T", "E"],
                    help="2a/2b: T (treino) e E (avaliacao) (via direct)")
    ap.add_argument("--out", default="tools/datasets/data/bnci")
    ap.add_argument("--cache", default="tools/datasets/data/_gdf_cache")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subs = []
    for s in args.subjects:
        if "-" in s:
            a, b = s.split("-", 1)
            subs.extend(range(int(a), int(b) + 1))
        else:
            subs.append(int(s))
    subs = sorted(set(s for s in subs if 1 <= s <= N_SUBJECTS))
    out_dir = Path(args.out)
    if args.via == "moabb":
        convert_via_moabb(dataset=args.dataset, subjects=subs, out_dir=out_dir,
                          overwrite=args.overwrite)
        return
    cache = Path(args.cache)
    base = URLS[args.dataset]
    pre = PREFIX[args.dataset]
    total = 0
    for subj in subs:
        for sess in args.sessions:
            gdf_name = f"{pre}{subj:02d}{sess}.gdf"
            gdf_path = cache / args.dataset / gdf_name
            out_csv = out_dir / f"bnci{args.dataset}_S{subj:02d}{sess}_openbci.csv"
            if out_csv.exists() and not args.overwrite:
                print(f"skip {out_csv.name} (existe)")
                total += 1
                continue
            download(f"{base}/{gdf_name}", gdf_path)
            convert_gdf(gdf_path, out_csv, dataset=args.dataset,
                        subject=subj, session=sess)
            total += 1
    print(f"OK: {total} arquivos em {out_dir}")


if __name__ == "__main__":
    main()
