"""Queue for fast results (DEVIATIONS #35, preregistration Addendum 8).

    python -m experiments.v6_mib.plan_fast

Priorities (lower first):
 0.10  original SSAE with LayerNorm (step-L0, L1): fits at the layer-subset units, then validation DBM and public-test
       scoring per (variable, layer), MIB standard (dbm_feat_) and direction-based (dbm_dir_) rows;
 0.15  the same without LayerNorm;
 0.30  direction-based rows (and any missing standard rows) at the subset for the capped SSAE (seed 0), the converged
       budget-free SSAE and SSAE-A (v3);
 0.40  original-SSAE fits on the RAVEL natural pair sets at layer 14 (absorption study);
 0.50  remaining layers: original-SSAE fits at every other unit, then their standard and direction-based rows, and the
       direction-based rows of the existing fits;
 9     the SSAE-B polish (last).
Held (queue_held/): the remaining RAVEL unit items (benchmarking moved to the RAVEL agent) and the original-SSAE fits
on the other pair sets. Items needing new code run the current snapshot through jobs/bootstrap.py.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from experiments.v6_mib.qitems import boot_mib, boot_module, snapshot

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(__file__).resolve().parents[2] / "results" / "v6_mib"
Q, HELD = ROOT / "queue_v2", ROOT / "queue_held"
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
LN, NOLN = ["L0ln", "L1ln"], ["L0noln", "L1noln"]
EXISTING = {"ssae_s0": "dbm_ssae", "ssaefull_s0_ad": "dbm_feat_ssaefull_s0_ad", "ssaeA2_s0": "dbm_feat_ssaeA2_s0"}
RAVEL_SPECS = ["s2_unit_L14_dev", "t2_unit_L14_final", "s2_unit_L14_dev_add1", "t2_unit_L14_final_add1",
               "f4_units_dev", "t4_units_final", "f6_indep_newfits_dev", "f6_indep_newfits_final", "v2_units_dev",
               "v2_units_final", "o_units_dev", "o_units_final"]


def h(it: dict) -> str:
    return hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]


def runs(t, m, split, method, var, layers, seed=0) -> list[str]:
    tag = method + (f"_s{seed}" if method == "dbm_ssae" else "")
    return [str(ROOT / "mib" / f"{t}__{m}" / "runs" / split / tag / var / f"L{L}.json") for L in layers]


CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}


def chunks(t: str, layers: list, per_layer: bool) -> list[list]:
    """One layer per item at the subset (fast first results); consecutive runs of up to CHUNK layers elsewhere."""
    if per_layer:
        return [[L] for L in layers]
    out, cur = [], []
    for L in layers:
        if cur and (L != cur[-1] + 1 or len(cur) >= CHUNK[t]):
            out.append(cur)
            cur = []
        cur.append(L)
    return out + ([cur] if cur else [])


def score_items(snap, t, m, meta, tag, std_method, layers, prio, direction=True, per_layer=True) -> list[dict]:
    """Validation DBM + public-test items per (variable, layer chunk): the standard row and the direction-based row."""
    out = []
    td = ROOT / "mib" / f"{t}__{m}"
    for var in VARIABLES[t]:
        for lay in chunks(t, layers, per_layer):
            L = lay
            feat = [str(td / "feat" / tag / f"L{l}_{p}.npz") for l in lay for p in meta["positions"]]
            base = {"task": t, "model": m, "variable": var, "seed": 0, "layers": f"{lay[0]}-{lay[-1]}"}
            if std_method == "dbm_ssae":
                out.append({"cmd": "eval_saved", **base, "split": "test", "method": "dbm_ssae", "prio": prio})
            else:
                out.append({"cmd": "train_eval", **base, "split": "val", "method": std_method, "prio": prio, "needs": feat})
                out.append({"cmd": "eval_saved", **base, "split": "test", "method": std_method, "prio": prio})
            if direction:
                meth = f"dbm_dir_{tag}"
                ls = f"{lay[0]}-{lay[-1]}"
                it = boot_mib(snap, "train_eval", task=t, model=m, variable=var, seed=0, layers=ls, split="val",
                              method=meth)
                out.append({**it, "prio": prio, "needs": feat, "done": runs(t, m, "val", meth, var, lay)})
                it = boot_mib(snap, "eval_saved", task=t, model=m, variable=var, seed=0, layers=ls, split="test",
                              method=meth)
                out.append({**it, "prio": prio, "needs": runs(t, m, "val", meth, var, lay),
                            "done": runs(t, m, "test", meth, var, lay)})
    return out


def reprio(spec: str, fn) -> tuple[int, int]:
    """Rewrite the priority of unclaimed items of a spec: fn(item) -> new prio, or None to hold the item."""
    f = Q / f"{spec}.jsonl"
    if not f.exists():
        return 0, 0
    keep, held, n = [], [], 0
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if (ROOT / "claims" / spec / h(it)).exists():
            keep.append(it)
            continue
        p = fn(it)
        if p is None:
            held.append(it)
        else:
            n += p != it.get("prio")
            it["prio"] = p
            keep.append(it)
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(i) + "\n" for i in keep))
    os.replace(tmp, f)
    if held:
        HELD.mkdir(exist_ok=True)
        with open(HELD / f"{spec}.jsonl", "a") as g:
            g.write("".join(json.dumps(i) + "\n" for i in held))
    return n, len(held)


def main() -> None:
    snap = snapshot()
    sub = json.loads((RESULTS / "records" / "layer_subset.json").read_text())
    items = []
    for key, s in sub.items():
        t, m = key.split("/")
        td = ROOT / "mib" / f"{t}__{m}"
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        S = s["subset"]
        rest = [L for L in meta["layers"] if L not in S]
        for variants, prio in ((LN, 0.10), (NOLN, 0.15)):
            for L in S:
                for p in meta["positions"]:
                    uk = f"L{L}_{p}"
                    items.append({**boot_module(snap, "experiments.v6_mib.fit_orig",
                                                ["units", "--task", t, "--model", m, "--units", uk, "--parallel",
                                                 "--variants", ",".join(variants)]), "prio": prio,
                                  "done": [str(td / "feat" / f"orig_{v}_s0" / f"{uk}.npz") for v in variants]})
            for v in variants:
                items += score_items(snap, t, m, meta, f"orig_{v}_s0", f"dbm_feat_orig_{v}_s0", S, prio)
        for tag, std in EXISTING.items():
            items += score_items(snap, t, m, meta, tag, std, S, 0.30)
        for v in LN + NOLN:                                              # remaining layers (if time allows)
            items += score_items(snap, t, m, meta, f"orig_{v}_s0", f"dbm_feat_orig_{v}_s0", rest, 0.55,
                                 per_layer=False)
        for tag, std in EXISTING.items():
            items += score_items(snap, t, m, meta, tag, std, rest, 0.60, per_layer=False)
    (Q / "a_fast.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print("a_fast", len(items))
    # original-SSAE fits: other units after the subset (0.50); pair sets: natural only (0.40), the rest held
    print("o_fits_units", reprio("o_fits_units", lambda it: 0.50))
    print("o_fits_sets", reprio("o_fits_sets", lambda it: 0.40 if re.search(r"--sets', 'natural_s\d", str(it["args"]))
                                or any(f"natural_s{i}" in a for a in it["args"] for i in range(5)) else None))
    print("b3_polish_units", reprio("b3_polish_units", lambda it: 9))
    print("b3_polish_sets", reprio("b3_polish_sets", lambda it: 9))
    # o_sweep / o_test: the original-SSAE entries are superseded by a_fast (per-layer items); the polished SSAE-B
    # entries (and its projection items) wait for the polish
    for spec in ("o_sweep", "o_test"):
        print(spec, reprio(spec, lambda it: None if "orig_" in it["method"] else 9.5))
    for spec in ("p2_bp_val", "p2_bp_test"):
        print(spec, reprio(spec, lambda it: 9.5))
    for spec in RAVEL_SPECS:                                             # RAVEL benchmarking moved to the RAVEL agent
        print(spec, reprio(spec, lambda it: None))


if __name__ == "__main__":
    main()


def dir_items(snap, t, m, meta, method, layers, prio, per_layer=True, needs_tag=None) -> list[dict]:
    """Direction-version items only (validation DBM, then public test) for a direction method name."""
    out = []
    td = ROOT / "mib" / f"{t}__{m}"
    for var in VARIABLES[t]:
        for lay in chunks(t, layers, per_layer):
            ls = f"{lay[0]}-{lay[-1]}"
            it = boot_mib(snap, "train_eval", task=t, model=m, variable=var, seed=0, layers=ls, split="val", method=method)
            it.update(prio=prio, done=runs(t, m, "val", method, var, lay))
            if needs_tag:
                it["needs"] = [str(td / "feat" / needs_tag / f"L{l}_{p}.npz") for l in lay for p in meta["positions"]]
            out.append(it)
            it = boot_mib(snap, "eval_saved", task=t, model=m, variable=var, seed=0, layers=ls, split="test", method=method)
            it.update(prio=prio, needs=runs(t, m, "val", method, var, lay), done=runs(t, m, "test", method, var, lay))
            out.append(it)
    return out


def plan_fairness() -> None:
    """DEVIATIONS #38-39: direction versions of SSAE-B (pre-polish) and GemmaScope, and the rotation null (3 seeds,
    layer subset only)."""
    snap = snapshot()
    sub = json.loads((RESULTS / "records" / "layer_subset.json").read_text())
    items = []
    for key, s in sub.items():
        t, m = key.split("/")
        meta = json.loads((ROOT / "mib" / f"{t}__{m}" / "shifts" / "train" / "meta.json").read_text())
        S = s["subset"]
        rest = [L for L in meta["layers"] if L not in S]
        items += dir_items(snap, t, m, meta, "dbm_dir_ssaeB2_s0", S, 0.30, needs_tag="ssaeB2_s0")
        items += dir_items(snap, t, m, meta, "dbm_dir_ssaeB2_s0", rest, 0.60, per_layer=False, needs_tag="ssaeB2_s0")
        if m == "gemma":
            items += dir_items(snap, t, m, meta, "dbm_dirsae_gemmascope", S, 0.30)
            items += dir_items(snap, t, m, meta, "dbm_dirsae_gemmascope", rest, 0.60, per_layer=False)
        for k in range(3):
            items += dir_items(snap, t, m, meta, f"dbm_dir_rot_s{k}", S, 0.30)
    (Q / "a_fast2.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print("a_fast2", len(items))
