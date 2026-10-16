"""Without-LayerNorm originals (v5's L0-noLN and L1-noLN: the frozen v3 settings without LayerNorm, unit input norm,
per-pair budget 0.005 x width = 11.52, width = hidden size 2304) on the 13 rows at each row's current read position
(user, 2026-10-02). Validation only: a layer sweep (seed 0, 7 layers), then seeds 0-4 at the validation-chosen layer
(v5's rule: the layer with the highest validation ``dir_acc``). v5's existing ladder fits at the row's read (same
settings, same pairs) count as sweep entries; the missing fits are written under ``results/v6_rows/records/fit/...``
(states under ``$SSAE_V6_ROOT/runs/...``) in the v5 runner's format.

    python -m experiments.v6_rows.noln pool --procs 5     # inside a SLURM job
    python -m experiments.v6_rows.noln status
    python -m experiments.v6_rows.noln budget              # curves/noln_budget.{csv,md,png}
    python -m experiments.v6_rows.noln choice              # VARIANT_CHOICE_complete.md (rule unchanged)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V5, V6, v6_root

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
ROWS13 = [("bias-in-bios-full", BASE), ("bias-in-bios-full", IT), ("bias-in-bios-strat28", BASE),
          ("bias-in-bios-strat28", IT), ("cebab-pairs", BASE), ("cebab-pairs", IT), ("refusal-caa", IT),
          ("refusal-swap", IT), ("sycophancy-caa", IT), ("sycophancy-swap", IT), ("truthful-qa-judge-noconst", IT),
          ("truthful-qa-judge-v2", IT), ("truthful-qa-judge", IT)]
VARIANTS = ("L0-noLN", "L1-noLN")
LAYERS = (6, 9, 12, 15, 18, 21, 24)
SEEDS = (0, 1, 2, 3, 4)
BUDGET = 0.005 * 2304


def read_of(ds: str, model: str) -> str:
    from experiments.v5.data import chosen_read
    return chosen_read(ds, model)


def metrics_path(ds: str, model: str, layer: int, seed: int, v: str) -> Path | None:
    """This study's record first, else v5's ladder record at the row's current read."""
    from experiments.v6_rows.common import record_dir
    p6 = record_dir(ds, model, layer, seed, v) / "metrics.json"
    if p6.is_file():
        return p6
    p5 = V5 / "records" / "fit" / f"{ds}__{MODEL_TAG[model]}__{read_of(ds, model)}" / f"L{layer}" / f"s{seed}" / v / "metrics.json"
    return p5 if p5.is_file() else None


def error_path(ds: str, model: str, layer: int, seed: int, v: str) -> Path:
    from experiments.v6_rows.common import record_dir
    return record_dir(ds, model, layer, seed, v) / "error.json"


def chosen_layer(ds: str, model: str, v: str) -> int | None:
    vals = {}
    for L in LAYERS:
        p = metrics_path(ds, model, L, 0, v)
        if p is None:
            if error_path(ds, model, L, 0, v).is_file():
                continue
            return None
        vals[L] = json.loads(p.read_text())["dir_acc"]
    return max(vals, key=vals.get) if vals else None


def tasks() -> list[tuple]:
    """Sweep tasks (seed 0, every layer) first, then seeds 1-4 at each completed sweep's chosen layer."""
    out = []
    for ds, m in ROWS13:
        for v in VARIANTS:
            out += [(ds, m, v, L, 0) for L in LAYERS
                    if metrics_path(ds, m, L, 0, v) is None and not error_path(ds, m, L, 0, v).is_file()]
    for ds, m in ROWS13:
        for v in VARIANTS:
            L = chosen_layer(ds, m, v)
            if L is not None:
                out += [(ds, m, v, L, s) for s in SEEDS[1:]
                        if metrics_path(ds, m, L, s, v) is None and not error_path(ds, m, L, s, v).is_file()]
    return out


def key(t: tuple) -> str:
    ds, m, v, L, s = t
    return f"noln__{ds}__{MODEL_TAG[m]}__{v}__L{L}__s{s}"


def run_task(t: tuple, dev: str = "cuda") -> None:
    from experiments.v4 import metrics as M
    from experiments.v5 import models as V5M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v5.runner import validation_metrics
    from experiments.v6_rows.common import save_fit, write_json
    ds, m, v, L, sd = t
    try:
        read = read_of(ds, m)
        train = load_split(ds, "train", L, m, read)
        val = load_split(ds, "val", L, m, read)
        refs, kinds, counts = M.reference_directions(val.X, val.Y, C=val.C)
        prof_ref = S.profession_reference(val.X, val.pairs)[0] if ds in SUBSPACE else None
        fit = V5M.fit_ladder(v, train, sd, dev, width=2304)
        met = validation_metrics(fit, train, val, refs, prof_ref, sd)
        met.update({"method": v, "fit_seconds": fit.seconds, "info": fit.info, "factors": train.factors,
                    "eval_split": "val", "counted_active_val": V5M.counted_active(v, fit.state, val.X),
                    "budget_per_pair": BUDGET})
        cfg = {"phase": "noln", "dataset": ds, "model": m, "read": read, "layer": L, "seed": sd, "method": v,
               "eval_split": "val", "fit_kind": str(fit.state.get("kind", v)),
               "settings": "frozen v3 without LayerNorm (unit input norm, per-pair budget 11.52, 105,000 updates), width 2304"}
        for attempt in range(30):            # $HOME's quota can be full for a moment: retry the small record writes
            try:
                save_fit(ds, m, L, sd, v, fit, met, cfg)
                break
            except OSError as e:
                if e.errno != 122 or attempt == 29:
                    raise
                time.sleep(60)
        print(f"[noln] {key(t)}: val dir_acc {met['dir_acc']:.3f}, active/pair {fit.info.get('counted_active_train'):.2f} "
              f"vs {BUDGET:.2f} ({fit.seconds:.0f}s)", flush=True)
    except Exception as err:
        if "CUDA" in repr(err) or "cuda" in repr(err):
            raise
        write_json(error_path(ds, m, L, sd, v), {"error": repr(err), "traceback": traceback.format_exc()})
        print(f"[noln] {key(t)}: ERROR {err!r}", flush=True)


def worker(dev: str = "cuda") -> None:
    from experiments.v6_rows.dial_fit import _claims, claim, release
    while True:
        took = False
        for t in tasks():
            k = key(t)
            if not claim(k):
                continue
            took = True
            try:
                if metrics_path(t[0], t[1], t[3], t[4], t[2]) is None:
                    run_task(t, dev)
            except Exception as err:
                print(f"[noln] {k}: CUDA ERROR {err!r}; releasing and exiting", flush=True)
                release(k)
                os._exit(3)
            release(k)
            break
        if not took:
            # seed tasks appear when a sweep completes: wait while other workers still hold sweep claims
            if any(p.name.startswith("noln__") for p in _claims().glob("*.claim")):
                time.sleep(60)
                continue
            return


def pool(procs: int) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [sys.executable, "-u", "-m", "experiments.v6_rows.noln", "worker"]
    live = {i: subprocess.Popen(cmd, env=env) for i in range(procs)}
    restarts = 0
    while live:
        time.sleep(30)
        for i, p in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            del live[i]
            if rc == 3 and restarts < 20:
                restarts += 1
                live[i] = subprocess.Popen(cmd, env=env)


def all_fits() -> pd.DataFrame:
    rows = []
    for ds, m in ROWS13:
        for v in VARIANTS:
            for L in LAYERS:
                for sd in SEEDS:
                    p = metrics_path(ds, m, L, sd, v)
                    if p is None:
                        continue
                    j = json.loads(p.read_text())
                    i = j.get("info", {})
                    rows.append({"dataset": ds, "model": MODEL_TAG[m], "variant": v, "layer": L, "seed": sd,
                                 "source": "v6" if "v6_rows" in str(p) else "v5 (ladder study)",
                                 "val_dir_acc": j.get("dir_acc"),
                                 "val_value": j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j.get("dir_acc"),
                                 "active_per_pair_train": i.get("counted_active_train"),
                                 "active_per_pair_val": j.get("counted_active_val"), "budget_per_pair": BUDGET,
                                 "final_violation": i.get("final_violation"), "dual_multiplier": i.get("dual_multiplier"),
                                 "updates": i.get("updates")})
    return pd.DataFrame(rows)


def budget() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.v6_rows.common import status_line, write_md_table
    t = all_fits()
    t["meets_budget"] = t["active_per_pair_train"] <= BUDGET
    out = V6 / "curves"
    t.to_csv(out / "noln_budget.csv", index=False)
    write_md_table(t, out / "noln_budget.md", "Without-LayerNorm originals: active coordinates per pair (the variant's own "
                   f"count, |h_i| > 0.05 ||shift||) against the budget ({BUDGET:.2f} per pair), every fit (validation phase).")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False, sharey=True)
    for k, v in enumerate(VARIANTS):
        ax = axes[0, k]
        e = t[t["variant"] == v]
        ax.scatter(e["layer"] + 0.6 * (e["seed"] - 2) / 4, e["active_per_pair_train"], s=10, color="#0072B2")
        ax.axhline(BUDGET, color="#D55E00", lw=1.2, ls="--", label=f"budget {BUDGET:.2f}")
        ax.set_title(f"{v}: active coordinates per pair (train), all rows", fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("active coordinates per pair")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "noln_budget.png", dpi=140)
    plt.close(fig)
    status_line("without-LayerNorm originals: active coordinates per pair vs budget", "done",
                ["curves/noln_budget.csv", "curves/noln_budget.md", "curves/noln_budget.png"])


def complete_rows() -> pd.DataFrame:
    """Per row and variant: the seed-0 value at the validation-chosen layer (the rule's definition)."""
    rows = []
    for ds, m in ROWS13:
        for v in VARIANTS:
            L = chosen_layer(ds, m, v)
            if L is None:
                continue
            j = json.loads(metrics_path(ds, m, L, 0, v).read_text())
            rows.append({"dataset": ds, "model": m, "variant": {"L0-noLN": "original SSAE, step-L0 (no LayerNorm)",
                                                               "L1-noLN": "original SSAE, L1 (no LayerNorm)"}[v],
                         "val": j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"],
                         "layer": L, "n_layers": len([x for x in LAYERS if metrics_path(ds, m, x, 0, v) is not None])})
    return pd.DataFrame(rows)


def status() -> pd.DataFrame:
    t = all_fits()
    todo = tasks()
    print(f"fits on record: {len(t)} (v6 {int((t['source'] == 'v6').sum()) if len(t) else 0}); open tasks: {len(todo)}")
    return t


def wait(hours: float = 40.0) -> None:
    """Block until no task is open and no claim is held (fits written or errors recorded)."""
    from experiments.v6_rows.dial_fit import _claims
    t0 = time.time()
    while time.time() - t0 < hours * 3600:
        if not tasks() and not any(p.name.startswith("noln__") for p in _claims().glob("*.claim")):
            return
        time.sleep(60)
    raise TimeoutError("without-LayerNorm tasks still open")


def committed(path: str) -> bool:
    r = subprocess.run(["git", "-C", str(V6.parent.parent), "status", "--porcelain", "--", path], capture_output=True, text=True)
    t = subprocess.run(["git", "-C", str(V6.parent.parent), "ls-files", "--error-unmatch", path], capture_output=True, text=True)
    return t.returncode == 0 and not r.stdout.strip()


def test(row: int | None = None) -> None:
    """Section 10's test phase (direction accuracy and erasure; interchange runs in ixc.py), after
    VARIANT_CHOICE_complete.md. Per row (``row`` = index into ROWS13, or every row): scores are saved per row
    (``records/noln/scores_test/<row>.json``) and finished parts are skipped, so a killed run resumes."""
    from experiments.v4 import metrics as M
    from experiments.v5 import erasure as ER
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import boot_counts, boot_references, gender_only, score_fit
    from experiments.v6_rows.common import assert_split_allowed, install, write_json
    assert_split_allowed("test")
    if not committed("results/v6_rows/VARIANT_CHOICE_complete.md"):
        raise RuntimeError("VARIANT_CHOICE_complete.md is not committed yet")
    install()
    ER.RECORDS = RECORDS / "noln"              # this ablation's erasure records stay apart from the main study's
    for i, (ds, m) in enumerate(ROWS13):
        if row is not None and i != row:
            continue
        tag = f"{ds}__{MODEL_TAG[m]}"
        layers = {v: chosen_layer(ds, m, v) for v in VARIANTS}
        out = RECORDS / "noln" / "scores_test" / f"{tag}.json"
        if not out.is_file():
            rows = []
            for L in sorted(set(layers.values())):
                ev = gender_only(load_split(ds, "test", L, m, endpoints=True))
                tr = gender_only(load_split(ds, "train", L, m, endpoints=True))
                refs, kinds, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
                rb = boot_references(ev, boot_counts(len(ev.X)))
                for v in [x for x in VARIANTS if layers[x] == L]:
                    for sd in SEEDS:
                        try:
                            fit = R.load(ds, m, v, L, sd)
                            res = score_fit(fit, tr, ev, refs, None, sd, rb, None)
                            res.update({"dataset": ds, "model": m, "method": v, "layer": L, "seed": sd})
                        except Exception as err:
                            res = {"dataset": ds, "model": m, "method": v, "layer": L, "seed": sd, "error": repr(err)}
                        rows.append(res)
                        print(f"[noln test] {tag} {v} L{L} s{sd}: {res.get('dir_acc_mean', res.get('error'))}", flush=True)
            write_json(out, {"rows": rows})
        if not (RECORDS / "noln" / "erasure" / "test" / f"{tag}.json").is_file():
            ER.run_row(ds, m, "test", [{"method": v, "layer": layers[v], "seeds": list(SEEDS)} for v in VARIANTS])


def summary() -> None:
    """Test tables next to the LayerNorm counterparts (v5's test records of L0-v3 / L1-v3)."""
    from experiments.v6_rows.common import write_md_table
    sc = [r for f in sorted((RECORDS / "noln" / "scores_test").glob("*.json")) for r in json.loads(f.read_text())["rows"]]
    t = pd.DataFrame([r for r in sc if "error" not in r])
    out = []
    for (ds, m, v), g in t.groupby(["dataset", "model", "method"]):
        cnt = {"L0-noLN": "L0-v3", "L1-noLN": "L1-v3"}[v]
        f5 = V5 / "records" / "scores" / "test" / f"{ds}__{MODEL_TAG[m]}.json"
        ln = [r for r in json.loads(f5.read_text())["rows"] if r.get("method") == cnt and "dir_acc_mean" in r] if f5.is_file() else []
        out.append({"dataset": ds, "model": MODEL_TAG[m], "variant": v, "layer": int(g["layer"].iloc[0]), "seeds": len(g),
                    "dir_acc": g["dir_acc_mean"].mean(), "dir_acc_sd": g["dir_acc_mean"].std(ddof=1),
                    "layernorm_counterpart": cnt, "counterpart_layer": ln[0]["layer"] if ln else None,
                    "counterpart_dir_acc": float(np.mean([r["dir_acc_mean"] for r in ln])) if ln else None,
                    "counterpart_seeds": len(ln)})
    d = pd.DataFrame(out)
    d["difference_noLN_minus_LN"] = d["dir_acc"] - d["counterpart_dir_acc"]
    d.to_csv(V6 / "curves" / "noln_test_direction.csv", index=False)
    write_md_table(d, V6 / "curves" / "noln_test_direction.md", "LayerNorm ablation, test direction accuracy: "
                   "without-LayerNorm originals (5 seeds at the validation-chosen layer) vs their LayerNorm counterparts "
                   "(v5's test numbers at v5's selected layer).")
    er = []
    for ds, m in ROWS13:
        for src, path in (("without LayerNorm", RECORDS / "noln" / "erasure" / "test" / f"{ds}__{MODEL_TAG[m]}.json"),
                          ("LayerNorm (v5)", V5 / "records" / "erasure" / "test" / f"{ds}__{MODEL_TAG[m]}.json")):
            if not path.is_file():
                continue
            for r in json.loads(path.read_text())["rows"]:
                if r.get("method") not in ("L0-noLN", "L1-noLN", "L0-v3", "L1-v3", "none"):
                    continue
                f = r.get("factor")
                pr = r.get("probes", {})
                tgt = pr.get(f, {})
                other = [k for k in pr if k != f]
                er.append({"dataset": ds, "model": MODEL_TAG[m], "source": src, "method": r["method"], "factor": f,
                           "layer": r.get("layer"), "seed": r.get("seed"), "target_acc": tgt.get("acc"),
                           "target_majority": tgt.get("majority"),
                           "other_acc": pr[other[0]].get("acc") if other else None,
                           "rms_tpr_gap": pr[other[0]].get("rms_tpr_gap") if other else None})
    e = pd.DataFrame(er)
    if len(e):
        g = e.groupby(["dataset", "model", "source", "method", "factor", "layer"], dropna=False).agg(
            target_acc=("target_acc", "mean"), target_majority=("target_majority", "mean"), other_acc=("other_acc", "mean"),
            rms_tpr_gap=("rms_tpr_gap", "mean"), n=("target_acc", "size")).reset_index()
        g.to_csv(V6 / "curves" / "noln_test_erasure.csv", index=False)
        write_md_table(g, V6 / "curves" / "noln_test_erasure.md", "LayerNorm ablation, erasure on test (v5's probes): "
                       "target-probe accuracy after projecting out the factor's k = 1 direction (majority = chance), the "
                       "other probe and the RMS TPR gap (Bias-in-Bios); method 'none' = no erasure.")
    from experiments.v6_rows.common import status_line
    status_line("without-LayerNorm originals on test (LayerNorm ablation: direction accuracy, erasure)", "done",
                ["curves/noln_test_direction.csv", "curves/noln_test_direction.md", "curves/noln_test_erasure.csv",
                 "curves/noln_test_erasure.md"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pool", "worker", "status", "budget", "choice", "wait", "test", "summary"])
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--row", type=int, default=None)
    a = ap.parse_args()
    if a.what == "pool":
        pool(a.procs)
    elif a.what == "worker":
        worker()
    elif a.what == "status":
        status()
    elif a.what == "budget":
        budget()
    elif a.what == "wait":
        wait()
    elif a.what == "test":
        test(a.row)
    elif a.what == "summary":
        summary()
    elif a.what == "choice":
        from experiments.v6_rows.variant_choice import main as vc
        vc(complete=True)
