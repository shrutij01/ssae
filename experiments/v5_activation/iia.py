"""Interchange jobs of this study through the v5 pipeline (MIB CausalAbstraction, f9ed677). Two analyses:

- ``own``: the activation baselines at their own selected layers (the 9 IIA rows);
- ``bound``: every method whose bound-aware layer differs from its selected layer, at the bound-aware layer (DAS
  included). A method whose bound-aware layer equals its selected layer keeps its identical result (v5's test / validation
  interchange for the v5 methods, the ``own`` analysis for the new ones).

Task and filter files are v5's (same texts, pairs, model and read position), copied into this study's job directories.

    python -m experiments.v5_activation.iia plan --split val          (login node: per-job selections)
    python -m experiments.v5_activation.iia prep --split val          (CPU job: featurizer exports + configs)
    python -m experiments.v5_activation.iia agg --split val
Stage 2 (GPU): experiments/v5_activation/jobs/iia_run.sh runs iia_lib/run_iia.py on the job directories.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5_activation.common import (IIA_ROWS, LAYERS, MODEL_TAG, NEW_DETERMINISTIC, RECORDS, SEEDS, TABLES,
                                              act_root, assert_split_allowed, install, read_json, write_json)

ANALYSES = ("own", "bound")
V5_DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}
CAP = {"val": 200, "test": 500}


def job_dir(analysis: str, ds: str, model: str, split: str) -> Path:
    return act_root() / "iia" / split / analysis / f"{ds}__{MODEL_TAG[model]}"


def v5_job(ds: str, model: str, split: str) -> Path:
    from experiments.v5.common import v5_root
    return v5_root() / "iia" / split / f"{ds}__{MODEL_TAG[model]}"


def seeds_ks(method: str, split: str) -> dict:
    if split == "val":
        return {0: [1]}
    det = method in NEW_DETERMINISTIC or method.split("__")[0] in V5_DETERMINISTIC
    return {s: ([1, 2, 5] if s == 0 else [1]) for s in ([0] if det else list(SEEDS))}


def spec_path(split: str, analysis: str, ds: str, model: str) -> Path:
    return act_root() / "specs" / f"iia_{split}__{analysis}__{ds}__{MODEL_TAG[model]}.json"


def plan(split: str) -> list[tuple[str, str, str]]:
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    ba = pd.DataFrame(read_json(RECORDS / "bound_aware.json")["choices"])
    items = []
    for ds, model in IIA_ROWS:
        for analysis in ANALYSES:
            if analysis == "own":
                entries = [(r.run_id, int(r.layer)) for r in sel[(sel.dataset == ds) & (sel.model == model)].itertuples()]
            else:
                x = ba[(ba.dataset == ds) & (ba.model == model) & ba.changed & (ba.method != "DAS")]
                entries = [(r.method_id_bound_aware, int(r.layer_bound_aware)) for r in x.itertuples()]
            out, layers = [], set()
            for meth, L in entries:
                for s, ks in seeds_ks(meth, split).items():
                    out.append({"method": meth, "layer": L, "seeds": [s], "ks": ks})
                layers.add(L)
            das = []
            if analysis == "bound" and split == "test":      # validation DAS exists at every layer (v5)
                for r in ba[(ba.dataset == ds) & (ba.model == model) & ba.changed & (ba.method == "DAS")].itertuples():
                    f, L = r.run_id.split("|")[1], int(r.layer_bound_aware)
                    layers.add(L)
                    das += [{"id": f"DAS|{f}|k{k}|L{L}", "method": "DAS", "kind": "das", "target": f, "k": k,
                             "layer": L, "seed": 0, "train_cap": 500, "epochs": 2} for k in (1, 2, 5)]
            diag = [{"id": f"identity|L{L}|read", "method": "identity", "kind": "identity", "layer": L,
                     "position": "read", "k": 0, "seed": 0} for L in sorted(layers)]
            if not out and not das:
                continue
            if out:
                out[0] = {**out[0], "das": diag + das}
            else:
                out = [{"method": None, "layer": None, "seeds": [], "ks": [], "das": diag + das}]
            p = spec_path(split, analysis, ds, model)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out))
            items.append((analysis, ds, model))
            print(f"{split} {analysis} {ds} {MODEL_TAG[model]}: {len(entries)} fitted methods, {len(das)} DAS, "
                  f"layers {sorted(layers)}", flush=True)
    write_json(act_root() / "specs" / f"iia_{split}__items.json", items)
    return items


def prep(split: str, only: list[str] | None = None) -> None:
    """Featurizer exports + configs for every planned job directory (v5 exporter, this study's fits and directories)."""
    assert_split_allowed(split)
    install()
    import experiments.v5.iia_prep as IP

    for analysis, ds, model in read_json(act_root() / "specs" / f"iia_{split}__items.json"):
        if only and f"{analysis}|{ds}|{MODEL_TAG[model]}" not in only:
            continue
        d = job_dir(analysis, ds, model, split)
        (d / "feat").mkdir(parents=True, exist_ok=True)
        IP.job_dir = lambda dataset, m, sp, read_tag=None, _a=analysis: job_dir(_a, dataset, m, sp)
        src = v5_job(ds, model, split)
        for name in ("task.json", f"filter_{split}.json", "filter_train.json"):
            if (src / name).is_file() and not (d / name).is_file():
                shutil.copy2(src / name, d / name)
        if not (d / "task.json").is_file():
            IP.build_task(ds, model, split)
        rows = []
        for s in read_json(spec_path(split, analysis, ds, model)):
            for seed in s["seeds"]:
                try:
                    rows += IP.export_fit(ds, model, split, s["method"], s["layer"], seed, tuple(s["ks"]))
                except Exception as err:
                    print(f"export failed {analysis} {ds} {s['method']} L{s['layer']} s{seed}: {err!r}", flush=True)
            rows += s.get("das", [])
        (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[prep {split}] {analysis} {ds} {MODEL_TAG[model]}: {len(rows)} configs", flush=True)


def per_question(split: str) -> pd.DataFrame:
    rows = []
    for analysis in ANALYSES:
        for job in sorted(glob.glob(str(act_root() / "iia" / split / analysis / "*"))):
            job = Path(job)
            if not (job / "task.json").is_file():
                continue
            task = json.loads((job / "task.json").read_text())
            for f in sorted(glob.glob(str(job / "results" / "*.json"))):
                if f.endswith(".error.json"):
                    continue
                res = json.loads(Path(f).read_text())
                cfg = res["config"]
                for target, by_q in res["scores"].items():
                    for q, v in by_q.items():
                        sc = np.asarray(v["scores"], dtype=float)
                        rows.append({"analysis": analysis, "dataset": task["dataset"], "model": task["model_id"],
                                     "method": cfg["method"] if cfg["kind"] != "das" else "DAS",
                                     "seed": cfg.get("seed", 0), "k": cfg["k"], "layer": cfg["layer"], "target": target,
                                     "question": q, "score": float(sc.mean()) if len(sc) else np.nan, "n": int(len(sc)),
                                     "sum": float(sc.sum()), "config_id": cfg["id"]})
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"iia_{split}_per_question.csv", index=False)
    return df


def summary_val(pq: pd.DataFrame) -> pd.DataFrame:
    """Validation sanity: k = 1 Cause (seed 0) per analysis, row, method and layer, with v5's validation bound there."""
    from experiments.v5_activation.common import V5
    v5 = pd.read_csv(V5 / "tables" / "iia_val_per_question.csv")
    idt = v5[(v5.method == "identity") & (v5.target == v5.question)]
    b = idt.groupby(["dataset", "model", "layer"])[["sum", "n"]].sum()
    bound = (b["sum"] / b["n"]).rename("bound").reset_index()
    x = pq[(pq.k == 1) & (pq.target == pq.question) & (pq.method != "identity")]
    g = x.groupby(["analysis", "dataset", "model", "method", "layer"])[["sum", "n"]].sum().reset_index()
    g["cause"] = g["sum"] / g["n"]
    g = g.merge(bound, on=["dataset", "model", "layer"], how="left")
    g["model"] = g.model.map(MODEL_TAG)
    g = g[["analysis", "dataset", "model", "method", "layer", "cause", "bound", "n"]]
    g.to_csv(TABLES / "iia_val_summary.csv", index=False)
    return g


def status(split: str) -> dict:
    out = {}
    for analysis, ds, model in read_json(act_root() / "specs" / f"iia_{split}__items.json"):
        d = job_dir(analysis, ds, model, split)
        ids = [json.loads(x)["id"] for x in (d / "configs.jsonl").read_text().splitlines() if x.strip()] \
            if (d / "configs.jsonl").is_file() else []
        done = {Path(f).stem for f in glob.glob(str(d / "results" / "*.json")) if not f.endswith(".error.json")}
        errs = glob.glob(str(d / "results" / "*.error.json"))
        out[f"{analysis}|{ds}|{MODEL_TAG[model]}"] = {"configs": len(ids), "done": len(set(ids) & done),
                                                      "errors": len(errs)}
    return out


def submit(split: str, n_jobs: int, sha: str, per_shard: int = 40, dependency: str | None = None,
           partitions: list[str] | None = None, only: list[str] | None = None,
           skip: list[str] | None = None) -> list[str]:
    """Stage 2 on at most ``n_jobs`` GPUs: every job directory's configs cut into shards of <= ``per_shard`` configs,
    spread over the jobs by config count (greedy)."""
    import subprocess
    from experiments.v5.common import REPO
    work = []
    for key, st in status(split).items():
        if (only and key not in only) or (skip and key in skip):
            continue
        analysis, ds, tag = key.split("|")
        todo = st["configs"] - st["done"]
        if todo <= 0:
            continue
        n = max(1, int(np.ceil(st["configs"] / per_shard)))
        for i in range(n):
            work.append((f"{analysis}/{ds}__{tag}:{i}/{n}", todo / n))
    buckets = [[[], 0.0] for _ in range(n_jobs)]
    for item, w in sorted(work, key=lambda x: -x[1]):
        b = min(buckets, key=lambda x: x[1])
        b[0].append(item)
        b[1] += w
    ids = []
    for j, (items, w) in enumerate(buckets):
        if not items:
            continue
        part = partitions[j % len(partitions)] if partitions else "long"
        cmd = ["sbatch", "--parsable", f"--partition={part}", "--gres=gpu:1", "--constraint=lovelace|40gb|80gb",
               "--cpus-per-task=8", f"--mem={'48G' if part == 'main' else '64G'}", "--time=24:00:00",
               f"--job-name=act-iia-{split}-{j}",
               f"--output={act_root() / 'logs' / f'act-iia-{split}-{j}-%j.out'}"]
        if dependency:
            cmd.append(f"--dependency={dependency}")
        cmd += [str(REPO / "experiments/v5_activation/jobs/iia_run.sh"), sha, split, str(CAP[split])] + items
        ids.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
        print(f"job {ids[-1]}: {len(items)} shards, ~{w:.0f} configs", flush=True)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plan", "prep", "agg", "status", "submit"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sha", default=None)
    ap.add_argument("--dependency", default=None)
    ap.add_argument("--partitions", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    if args.what == "plan":
        plan(args.split)
    elif args.what == "prep":
        prep(args.split, args.only)
    elif args.what == "agg":
        pq = per_question(args.split)
        print(pq.groupby(["analysis", "dataset", "model"]).size().to_string())
        if args.split == "val":
            summary_val(pq)
    elif args.what == "submit":
        print(submit(args.split, args.jobs, args.sha, dependency=args.dependency, partitions=args.partitions,
                     only=args.only, skip=args.skip))
    else:
        print(json.dumps(status(args.split), indent=1))


if __name__ == "__main__":
    main()
