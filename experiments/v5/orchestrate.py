"""Turn the layer selection into run lists for IIA, erasure and the applications, and submit them.

    python -m experiments.v5.orchestrate iia --split val   [--submit]
    python -m experiments.v5.orchestrate erasure --split val [--submit]
    python -m experiments.v5.orchestrate apps               (writes records/app_selection.json)

Validation (sanity, brief section 8 step 4): seed 0, k = 1, at most 200 pairs per (target, question), plus the full-patch
site bound at every selected layer and DAS (k = 1) at every layer (its layer = the brief's rule: validation direction
accuracy of its learned direction). Test: every seed at k = 1; k = 2, 5 at seed 0; at most 500 pairs per (target,
question); DAS at k = 1, 2, 5 at its selected layer.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import pandas as pd

from experiments.v5.common import LAYERS, MODEL_TAG, RECORDS, REPO, ROWS, SEEDS, read_json, v5_root, write_json

ERASE_ROWS = [(d, m) for d, m, _, _ in ROWS if not d.startswith(("sycophancy", "refusal"))]
# IIA rows: rows with at least one factor whose localization bound is >= 0.5 (CEBaB has none: #21)
IIA_ROWS = [(d, m) for d, m in ERASE_ROWS if not d.startswith("cebab")]
DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}
CAP = {"val": 200, "test": 500}
DAS_FACTORS = {"cebab-pairs": ["food", "service", "ambiance", "noise"], "bias-in-bios-strat28": ["gender"],
               "bias-in-bios-strat2": ["gender"], "bias-in-bios-full": ["gender"],
               "truthful-qa-judge-v2": ["truth"], "truthful-qa-judge": ["truth"], "truthful-qa-judge-noconst": ["truth"]}


def row_methods(dataset: str, model: str) -> list[dict]:
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    s = sel[(sel.dataset == dataset) & (sel.model == model)]
    # v5 method id (v4 fits are located through registry.V4_ID); GemmaScope keeps its variant id
    return [{"method": r.run_id if r.method == "gemmascope" else r.method, "family": r.method, "layer": int(r.layer)}
            for r in s.itertuples()]


def selection_for(dataset: str, model: str, split: str, only_diag: bool = False) -> list[dict]:
    out = []
    for m in ([] if only_diag else row_methods(dataset, model)):
        if split == "val":
            seeds, ks_by_seed = [0], {0: [1]}
        else:
            seeds = [0] if m["family"].split("@")[0] in DETERMINISTIC else list(SEEDS)
            ks_by_seed = {s: ([1, 2, 5] if s == 0 else [1]) for s in seeds}
        for s in seeds:
            out.append({"method": m["method"], "layer": m["layer"], "seeds": [s], "ks": ks_by_seed[s]})
    layers = sorted({m["layer"] for m in row_methods(dataset, model)}) if split != "val" else list(LAYERS)
    diag = [{"id": f"identity|L{L}|read", "method": "identity", "kind": "identity", "layer": L, "position": "read",
             "k": 0, "seed": 0} for L in (LAYERS if split == "val" else layers)]
    das = []
    if split == "val":
        for L in LAYERS:
            for f in DAS_FACTORS[dataset]:
                das.append({"id": f"DAS|{f}|k1|L{L}", "method": "DAS", "kind": "das", "target": f, "k": 1, "layer": L,
                            "seed": 0, "train_cap": 500, "epochs": 2})
    else:
        ch = read_json(RECORDS / "das_layers.json").get(f"{dataset}|{model}", {})
        for f, L in ch.items():
            for k in (1, 2, 5):
                das.append({"id": f"DAS|{f}|k{k}|L{L}", "method": "DAS", "kind": "das", "target": f, "k": k,
                            "layer": int(L), "seed": 0, "train_cap": 500, "epochs": 2})
    if out:
        out[0] = {**out[0], "das": diag + das}
    else:
        out = [{"method": None, "layer": None, "seeds": [], "ks": [], "das": diag + das}]
    return out


def submit(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def head_sha() -> str:
    dirty = [l for l in subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                       capture_output=True, text=True).stdout.splitlines() if not l[3:].startswith("results/")]
    if dirty:
        raise SystemExit(f"refusing to submit from a dirty tree: {dirty}")
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def iia(split: str, do_submit: bool, shards: int, only_diag: bool = False) -> None:
    sha = head_sha() if do_submit else None
    ids = []
    for ds, model in IIA_ROWS:
        sel = selection_for(ds, model, split, only_diag)
        path = v5_root() / "specs" / f"iia_{split}__{ds}__{MODEL_TAG[model]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sel))
        if not do_submit:
            print(ds, MODEL_TAG[model], len(sel), "entries")
            continue
        n = shards if split == "test" else 1
        prep = submit(["sbatch", "--parsable", "--partition=long", "--gres=gpu:1", "--constraint=lovelace|40gb|80gb",
                       "--cpus-per-task=8", "--mem=64G", "--time=16:00:00", f"--job-name=v5-iia-{split}-{ds[:12]}",
                       f"--output={v5_root() / 'logs' / f'iia-{split}-{ds}-{MODEL_TAG[model]}-0-%j.out'}",
                       str(REPO / "experiments/v5/jobs/iia.sh"), sha, ds, model, split, str(path), str(CAP[split]),
                       f"0/{n}"])
        ids.append(prep)
        for i in range(1, n):   # further shards start after the export + filter of shard 0
            ids.append(submit(["sbatch", "--parsable", "--partition=long", "--gres=gpu:1",
                               "--constraint=lovelace|40gb|80gb", "--cpus-per-task=6", "--mem=48G", "--time=16:00:00",
                               f"--dependency=after:{prep}+15", f"--job-name=v5-iia-{split}-{ds[:12]}",
                               f"--output={v5_root() / 'logs' / f'iia-{split}-{ds}-{MODEL_TAG[model]}-{i}-%j.out'}",
                               str(REPO / "experiments/v5/jobs/iia.sh"), sha, ds, model, split, str(path),
                               str(CAP[split]), f"{i}/{n}", "--skip-prep"]))
    if ids:
        print("JOBS " + ",".join(ids))


def erasure(split: str, do_submit: bool) -> None:
    sha = head_sha() if do_submit else None
    ids = []
    for ds, model in ERASE_ROWS:
        sel = []
        for m in row_methods(ds, model):
            seeds = [0] if (split == "val" or m["family"].split("@")[0] in DETERMINISTIC) else list(SEEDS)
            sel.append({"method": m["method"], "layer": m["layer"], "seeds": seeds})
        path = v5_root() / "specs" / f"erasure_{split}__{ds}__{MODEL_TAG[model]}.json"
        path.write_text(json.dumps(sel))
        if do_submit:
            ids.append(submit(["sbatch", "--parsable", "--partition=long", "--gres=gpu:1",
                               "--constraint=lovelace|40gb|80gb", "--cpus-per-task=8", "--mem=64G", "--time=12:00:00",
                               f"--job-name=v5-era-{split}-{ds[:12]}",
                               f"--output={v5_root() / 'logs' / f'erasure-{split}-{ds}-{MODEL_TAG[model]}-%j.out'}",
                               str(REPO / "experiments/v5/jobs/run.sh"), sha, "experiments.v5.erasure", "--dataset", ds,
                               "--model", model, "--split", split, "--selection", str(path)]))
    if ids:
        print("JOBS " + ",".join(ids))


def apps() -> None:
    from experiments.v5.applications import STEER_ROWS
    out = {ds: [{"method": m["method"], "layer": m["layer"]} for m in row_methods(ds, "google/gemma-2-2b-it")]
           for ds in STEER_ROWS}
    write_json(RECORDS / "app_selection.json", out)
    print({k: len(v) for k, v in out.items()})


def das_layers() -> dict:
    """DAS layer per (row, factor): validation direction accuracy of its learned k = 1 direction (brief section 4 rule):
    |cos| with the validation reference; a 28-valued profession: norm of the projection on the reference subspace."""
    import numpy as np

    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split

    out, detail = {}, {}
    for ds, model in IIA_ROWS:
        job = v5_root() / "iia" / "val" / f"{ds}__{MODEL_TAG[model]}" / "results"
        key = f"{ds}|{model}"
        for f in DAS_FACTORS[ds]:
            best, scores = None, {}
            for L in LAYERS:
                path = job / f"DAS|{f}|k1|L{L}__rotation.npy"
                if not path.is_file():
                    continue
                v = np.load(path)[:, 0].astype(np.float64)
                v /= np.linalg.norm(v)
                va = load_split(ds, "val", L, model)
                if f in va.factors:
                    refs, _, _ = M.reference_directions(va.X, va.Y, C=va.C)
                    acc = float(abs(v @ refs[:, va.factors.index(f)]))
                elif ds in SUBSPACE:
                    ref, _ = S.profession_reference(va.X, va.pairs)
                    acc = float(np.linalg.norm(ref.T @ v))
                else:
                    continue
                scores[L] = acc
                if best is None or acc > scores[best]:
                    best = L
            if best is not None:
                out.setdefault(key, {})[f] = best
                detail.setdefault(key, {})[f] = scores
    write_json(RECORDS / "das_layers.json", out)
    write_json(RECORDS / "das_layers_detail.json", detail)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["iia", "erasure", "apps", "das"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--only-diag", action="store_true", help="site bounds + DAS only (no fitted method needed)")
    args = ap.parse_args()
    if args.what == "iia":
        iia(args.split, args.submit, args.shards, args.only_diag)
    elif args.what == "erasure":
        erasure(args.split, args.submit)
    elif args.what == "das":
        print(das_layers())
    else:
        apps()


if __name__ == "__main__":
    main()
