"""MIB localization before any swap test (user decision #15 (1)): on validation, the full-vector-patch Cause of every
factor at every candidate read position and layer; choose the read position per dataset x model.

    python -m experiments.v5.localize prepare --dataset bias-in-bios-strat2 --model google/gemma-2-2b-it
    python -m experiments.v5.localize submit
    python -m experiments.v5.localize aggregate

Candidates: CEBaB = the "Overall rating" suffix ("rating"; v5/v4 read name "suffix") and the last review token
("last"); Bias-in-Bios = occupation suffix ("occupation"; v4 read name "suffix"), pronoun suffix ("pronoun") and the
last bio token ("last"). Bound of a factor = Cause of the identity featurizer (all dimensions patched) at the read
position: validation pairs where the factor differs, question = the factor, filter as in the IIA (at most 600 eligible
pairs per question enter the filter; at most 200 per question are patched). Rule: the position with the most factors
whose best-layer bound is >= 0.5; ties by the mean best-layer bound, then by mean validation R2_concept.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from experiments.v5.common import LAYERS, MODEL_TAG, RECORDS, REPO, ROWS, TABLES, V4_RESULTS, read_json, v5_root, write_json

MIN_N = 20      # minimum Cause examples behind a bound (DEVIATIONS #19)
CANDIDATES = {"cebab-pairs": ["rating", "last"], "bias-in-bios-strat28": ["occupation", "pronoun", "last"],
              "bias-in-bios-strat2": ["occupation", "pronoun", "last"], "bias-in-bios-full": ["occupation", "pronoun", "last"]}
LOC_ROWS = [(d, m) for d, m, _, _ in ROWS if d in CANDIDATES]
DATA_READ = {"rating": "suffix"}
ORIGINAL = {"cebab-pairs": "rating", "bias-in-bios-strat28": "occupation", "bias-in-bios-strat2": "occupation",
            "bias-in-bios-full": "occupation"}          # read positions before the localization step      # candidate name -> stored read name (v4 Bias-in-Bios: occupation = "suffix")


def data_read(dataset: str, model: str, cand: str) -> str:
    from experiments.v5.common import V4_NAME
    if cand == "occupation" and V4_NAME[(dataset, model)] is not None:
        return "suffix"
    return DATA_READ.get(cand, cand)


def prepare(dataset: str, model: str) -> None:
    from experiments.v5.iia_prep import build_task, job_dir

    for cand in CANDIDATES[dataset]:
        info = build_task(dataset, model, "val", read=cand, read_tag=cand)
        d = job_dir(dataset, model, "val", cand)
        cfgs = [{"id": f"identity|L{L}|read", "method": "identity", "kind": "identity", "layer": L, "position": "read",
                 "k": 0, "seed": 0} for L in LAYERS]
        (d / "configs.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cfgs))
        print(json.dumps(info), flush=True)


SCRIPT = """#!/bin/bash
set -euo pipefail
SHA="$1"; DS="$2"; MODEL="$3"
REPO_DIR="${SSAE_V5_REPO:-$HOME/causalrepl_space/ssae-dev}"
source "$REPO_DIR/experiments/v5/jobs/env.sh"
SNAP="$SSAE_V5_ROOT/code/$SHA"
if [ ! -f "$SNAP/.complete" ]; then
    TMP="$SNAP.tmp.${SLURM_JOB_ID:-$$}.$RANDOM"; mkdir -p "$TMP"
    git -C "$REPO_DIR" archive "$SHA" -- . ':(exclude)results' | tar -x -C "$TMP"; touch "$TMP/.complete"
    mv -T "$TMP" "$SNAP" 2>/dev/null || rm -rf "$TMP"
    for _ in $(seq 60); do [ -f "$SNAP/.complete" ] && break; sleep 5; done
fi
export SSAE_V5_RESULTS="$REPO_DIR/results/v5" SSAE_V4_RESULTS="$REPO_DIR/results/v4" SSAE_V5_REPO="$REPO_DIR"
export SSAE_V4_GIT_COMMIT="$SHA" HF_HUB_OFFLINE=1
cd "$SNAP"
PYTHONPATH="$SNAP" "$PY" -u -m experiments.v5.localize prepare --dataset "$DS" --model "$MODEL"
TAG=$(basename "$MODEL")
FCAP=600; case "$DS" in cebab-*) FCAP=0;; esac      # CEBaB: every eligible pair (few survive the filter; #19)
for J in "$SSAE_V5_ROOT"/iia/val/"${DS}__${TAG}"__read-*; do
    env -u PYTHONPATH "$PY" -u "$SNAP/experiments/v5/iia_lib/run_iia.py" "$J" --split val --cap 200 --filter-cap $FCAP
done
"""


def submit() -> None:
    dirty = [l for l in subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                       capture_output=True, text=True).stdout.splitlines() if not l[3:].startswith("results/")]
    if dirty:
        raise SystemExit(f"dirty tree: {dirty}")
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    script = v5_root() / "localize.sh"
    script.write_text(SCRIPT)
    script.chmod(0o755)
    ids = []
    for ds, model in LOC_ROWS:
        ids.append(subprocess.run(["sbatch", "--parsable", "--partition=long", "--gres=gpu:1",
                                   "--constraint=lovelace|40gb|80gb", "--cpus-per-task=6", "--mem=48G", "--time=6:00:00",
                                   f"--job-name=v5-loc-{ds[:14]}",
                                   f"--output={v5_root() / 'logs' / f'loc-{ds}-{MODEL_TAG[model]}-%j.out'}",
                                   str(script), sha, ds, model], capture_output=True, text=True, check=True).stdout.strip())
    print("JOBS " + ",".join(ids))


def r2_by_read(dataset: str, model: str) -> dict:
    from experiments.v5.common import V4_NAME
    v4 = V4_NAME[(dataset, model)]
    if v4 is not None:
        j = read_json(V4_RESULTS / "read_position.json").get(f"{v4}|{model}", {})
        r = j.get("mean_R2_concept_val_by_read", {})
        return {"rating": r.get("suffix"), "occupation": r.get("suffix"), "last": r.get("last")}
    j = read_json(RECORDS.parent / "read_position.json").get(f"{dataset}|{model}", {})
    r = j.get("mean_R2_concept_val_by_read", {})
    return {"rating": r.get("suffix"), "occupation": r.get("occupation"), "pronoun": r.get("pronoun"), "last": r.get("last")}


def aggregate() -> dict:
    from experiments.v5.iia_prep import IIA_FACTORS, job_dir

    rows, choice = [], {}
    for ds, model in LOC_ROWS:
        r2 = r2_by_read(ds, model)
        summ = {}
        for cand in CANDIDATES[ds]:
            d = job_dir(ds, model, "val", cand)
            filt = json.loads((d / "filter_val.json").read_text()) if (d / "filter_val.json").is_file() else {}
            per = {f: {} for f in IIA_FACTORS[ds]}
            nper = {f: {} for f in IIA_FACTORS[ds]}
            for L in LAYERS:
                p = d / "results" / f"identity|L{L}|read.json"
                if not p.is_file():
                    continue
                res = json.loads(p.read_text())
                for f in IIA_FACTORS[ds]:
                    sc = res["scores"].get(f, {}).get(f, {}).get("scores", [])
                    per[f][L] = float(np.mean(sc)) if len(sc) else np.nan
                    nper[f][L] = int(len(sc))
            # a bound counts only if it rests on >= MIN_N Cause examples (DEVIATIONS #19)
            usable = {f: {L: v for L, v in per[f].items() if nper[f].get(L, 0) >= MIN_N and np.isfinite(v)} for f in per}
            best = {f: (max(v, key=v.get) if v else None) for f, v in usable.items()}
            bval = {f: (per[f][best[f]] if best[f] is not None else np.nan) for f in per}
            n_ok = int(sum(1 for v in bval.values() if np.isfinite(v) and v >= 0.5))
            summ[cand] = {"bounds": per, "n_cause": nper, "best_layer": best, "best_bound": bval, "n_factors_ge_0.5": n_ok,
                          "mean_best_bound": float(np.nanmean(list(bval.values()))) if bval else np.nan,
                          "R2_concept": r2.get(cand), "filter": filt.get("stats", {})}
            for f in per:
                for L, v in per[f].items():
                    rows.append({"dataset": ds, "model": MODEL_TAG[model], "position": cand, "factor": f, "layer": L,
                                 "bound": v})
        key = lambda c: (summ[c]["n_factors_ge_0.5"], np.nan_to_num(summ[c]["mean_best_bound"], nan=-1),
                         -1 if summ[c]["R2_concept"] is None else summ[c]["R2_concept"])
        done = [c for c in summ if all(len(summ[c]["bounds"][f]) == len(LAYERS) for f in summ[c]["bounds"])]
        if len(done) == len(CANDIDATES[ds]):
            win = max(done, key=key)
            if max(summ[c]["n_factors_ge_0.5"] for c in done) == 0:     # user rule (#24): keep the original read
                win = ORIGINAL[ds]
            choice[f"{ds}|{model}"] = {"position": win, "data_read": data_read(ds, model, win), "summary": summ,
                                       "applicable": {f: bool(np.isfinite(v) and v >= 0.5)
                                                      for f, v in summ[win]["best_bound"].items()},
                                       "insufficient": [f for f, b in summ[win]["best_layer"].items() if b is None]}
        else:
            choice[f"{ds}|{model}"] = {"position": None, "summary": summ, "missing": sorted(set(summ) - set(done))}
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "localization.csv", index=False)
    write_json(RECORDS / "localization.json", choice)
    write_md(choice)
    return choice


def write_md(choice: dict) -> None:
    L = ["# Localization (full-vector patch Cause; validation)", "",
         "User decision #15 (1), MIB's localization step. For each dataset x model, each candidate read position and each "
         "layer: the Cause of patching the whole residual vector (identity featurizer) at the read position, on validation "
         "pairs where the factor differs, asked about that factor, after the IIA filter (<= 600 eligible pairs per "
         "question filtered, <= 200 patched). This is the upper bound of any featurizer's Cause at that site. Chosen "
         "position: most factors with best-layer bound >= 0.5, ties by mean best-layer bound, then R2_concept. A factor "
         "with bound < 0.5 at the chosen position gets IIA \"not applicable (bound < 0.5)\". A bound resting on fewer than "
         f"{MIN_N} Cause examples is not used (marked `(nN)`; DEVIATIONS #19). Bold = best usable layer.", ""]
    for key, c in choice.items():
        ds, model = key.split("|")
        L += [f"## {ds} · {MODEL_TAG[model]}", ""]
        s = c["summary"]
        if not s:
            continue
        factors = list(next(iter(s.values()))["bounds"])
        L += ["| position | " + " | ".join(f"{f}: bound by layer (6..24)" for f in factors) +
              " | factors >= 0.5 | mean best bound | R2_concept |",
              "|---|" + "---|" * len(factors) + "---|---|---|"]
        for cand, v in s.items():
            cells = []
            for f in factors:
                row = [(v["bounds"][f].get(Lr), v["n_cause"][f].get(Lr, 0)) for Lr in LAYERS]
                bl = v["best_layer"][f]
                cells.append(" ".join("—" if x is None or not np.isfinite(x) else
                                      (f"**{x:.2f}**" if Lr == bl else f"{x:.2f}") + ("" if n >= MIN_N else f"(n{n})")
                                      for Lr, (x, n) in zip(LAYERS, row)))
            L.append(f"| {cand} | " + " | ".join(cells) + f" | {v['n_factors_ge_0.5']} | "
                     f"{v['mean_best_bound']:.3f} | " + ("—" if v["R2_concept"] is None else f"{v['R2_concept']:.3f}") + " |")
        pos = c.get("position")
        L += ["", f"Chosen position: **{pos or 'pending'}**" + ("" if not pos else
              f" (data read `{c['data_read']}`); IIA applicable: " + ", ".join(
                  f"{f} {'yes' if a else 'no (bound < 0.5)'}" for f, a in c["applicable"].items())), ""]
    (TABLES / "localization.md").write_text("\n".join(L) + "\n")


def finalize() -> dict:
    """Write ``results/v5/read_position_final.json`` (data read name per localized row) once every row is decided."""
    choice = read_json(RECORDS / "localization.json")
    missing = [k for k, v in choice.items() if not v.get("position")]
    if missing:
        raise SystemExit(f"localization pending for {missing}")
    out = {k: v["data_read"] for k, v in choice.items()}
    write_json(RECORDS.parent / "read_position_final.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prepare", "submit", "aggregate", "finalize"])
    ap.add_argument("--dataset")
    ap.add_argument("--model")
    args = ap.parse_args()
    if args.what == "prepare":
        prepare(args.dataset, args.model)
    elif args.what == "submit":
        submit()
    elif args.what == "finalize":
        print(finalize())
    else:
        c = aggregate()
        print({k: (v.get("position"), v.get("applicable")) for k, v in c.items()})


if __name__ == "__main__":
    main()
