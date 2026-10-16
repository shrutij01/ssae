"""Queue items for Part 1, "MIB done MIB's way" (DEVIATIONS #45-#46, preregistration Addendum 12). Append-only.

    python -m experiments.v6_mib.plan_p1 [--dry]      -> queue_v2/p1_fits.jsonl, queue_v2/p1_dbm_<task>_<model>.jsonl

Fits: per unit, experiments.v6_mib.fit_p1 --parallel (L0-v3 and the two without-LayerNorm variants on {cf, rp} pairs, and
PCA_delta on rp) and a second item for L1-v3 (#48), once the task's activation pool exists. DBM: one item per (row, variable, DBM seed, layer chunk), running mib_run.py
train_test (validation, then the public test) for the chunk's layers. Rows: decoder inverse for each fit family, SAE
treatment for the without-LayerNorm families only (#48: never a LayerNorm encoder on single activations); the
orthonormalized basis for a family once its fits show cond(W_d) > 1e4 at some unit; neurons, PCA (MIB),
GemmaScope (Gemma), DAS, PCA_delta on cf and on rp pairs, the random basis (3 draws), full vector. DBM seeds 0-2 (seed 0
first). ARC chunks carry GPU-memory tags (A100L, or an L40S with enough free memory; layers 0-5: A100L only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
TASKS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
VARIANTS = ["L0ln", "L1ln", "L0noln", "L1noln"]     # L0-v3, L1-v3 (the paper's two columns, #48), LayerNorm ablation
LN = ("L0ln", "L1ln")                                 # never a LayerNorm encoder on single activations (#48)
PAIRSETS = ["cf", "rp"]
OFFSET = {"ravel/gemma": 0.0, "mcqa/gemma": 0.0, "arc/gemma": 0.02, "mcqa/qwen": 0.02}
CHUNKS = {"gemma": [(0, 6), (7, 12), (13, 18), (19, 25)], "qwen": [(0, 5), (6, 11), (12, 17), (18, 23)]}
ARC_CHUNKS = [(0, 5), (6, 11), (12, 17), (18, 25)]
COND_MAX = 1e4


def td(t: str, m: str) -> Path:
    return ROOT / "mib" / f"{t}__{m}"


def units(t: str, m: str) -> list[str]:
    meta = json.loads((td(t, m) / "shifts" / "train" / "meta.json").read_text())
    return [f"L{L}_{p}" for L in meta["layers"] for p in meta["positions"]]


def arc_tags(L0: int, extra_gb: float = 0.0) -> dict:
    """ARC DBM memory (#44): ~28.3 GiB measured at layer 18, ~1 GiB more per earlier layer; A100L, or an L40S with
    enough free memory when that fits in 43 GiB."""
    need = max(33.0, math.ceil(22 + 1.04 * (26 - L0))) + extra_gb
    return {"gpu_min_gb": 60, **({"free_gb_ok": need} if need <= 43 else {})}


def rows(t: str, m: str) -> list[tuple[str, list[int], list[str]]]:
    """(method, DBM seeds, feature tags it needs)."""
    R = []
    for ps in PAIRSETS:
        for v in VARIANTS:
            tag = f"p1{ps}_{v}"
            R.append((f"dbm_pinv_{tag}", [0, 1, 2], [tag]))
            if v not in LN:                          # SAE treatment: linear encoders only
                R.append((f"dbm_sae_{tag}", [0, 1, 2], [tag]))
            if ortho_needed(t, m, tag):
                R.append((f"dbm_dir_{tag}", [0, 1, 2], [tag]))
    R += [("dbm_neurons", [0, 1, 2], []), ("dbm_mibpca", [0, 1, 2], []), ("das", [0, 1, 2], []),
          ("dbm_feat_pcafull", [0, 1, 2], ["pcafull"]), ("dbm_feat_p1rp_pcafull", [0, 1, 2], ["p1rp_pcafull"])]
    if m == "gemma":
        R.append(("dbm_gemmascope", [0, 1, 2], []))
    R += [(f"dbm_dir_rot_s{k}", [0], []) for k in range(3)] + [("full_vector", [0], [])]
    return R


def ortho_needed(t: str, m: str, tag: str) -> bool:
    """True once any fit of the family has cond(W_d) > 1e4 (from feat/fit_p1_info)."""
    ps, v = tag[2:].split("_", 1)
    for f in (td(t, m) / "feat" / "fit_p1_info").glob(f"*_{ps}_{v}.json"):
        if json.loads(f.read_text()).get("cond", 0) > COND_MAX:
            return True
    return False


def plan(dry: bool = False, only: str = "") -> None:
    from experiments.v6_mib.qitems import boot_mib, boot_module, snapshot
    snap = snapshot()
    specs: dict[str, list] = {}
    for t, m in TASKS:
        key = f"{t}/{m}"
        pool = td(t, m) / "pool" / "train" / "meta.json"
        fit_items = specs.setdefault("p1_fits", [])
        for uk in units(t, m):
            v3 = ["L0ln", "L0noln", "L1noln"]                # the first fit items (before #48)
            done = [str(td(t, m) / "feat" / f"p1{ps}_{v}" / f"{uk}.npz") for ps in PAIRSETS for v in v3] + \
                   [str(td(t, m) / "feat" / "p1rp_pcafull" / f"{uk}.npz")]
            it = boot_module(snap, "experiments.v6_mib.fit_p1", ["--task", t, "--model", m, "--unit", uk, "--parallel"])
            it.update(prio=round(0.17 + OFFSET[key], 3), needs=[str(pool)], done=done)
            fit_items.append(it)
            it = boot_module(snap, "experiments.v6_mib.fit_p1", ["--task", t, "--model", m, "--unit", uk, "--parallel",
                                                                 "--variants", "L1ln"])          # L1-v3 (#48)
            it.update(prio=round(0.17 + OFFSET[key], 3), needs=[str(pool)],
                      done=[str(td(t, m) / "feat" / f"p1{ps}_L1ln" / f"{uk}.npz") for ps in PAIRSETS])
            fit_items.append(it)
        L_all = json.loads((td(t, m) / "shifts" / "train" / "meta.json").read_text())["layers"]
        positions = json.loads((td(t, m) / "shifts" / "train" / "meta.json").read_text())["positions"]
        chunks = ARC_CHUNKS if t == "arc" else CHUNKS[m]
        assert chunks[-1][1] == L_all[-1], (key, chunks, L_all[-1])
        items = specs.setdefault(f"p1_dbm_{t}_{m}", [])
        for method, seeds, ftags in rows(t, m):
            for d in seeds:
                tag = method + (f"_d{d}" if d else "")
                for var in NV[t]:
                    for a, b in chunks:
                        it = boot_mib(snap, "train_test", task=t, model=m, variable=var, seed=0, layers=f"{a}-{b}",
                                      method=method, dbm_seed=d)
                        it["prio"] = round(0.20 + OFFSET[key] + (0.0 if d == 0 else 0.03), 3)
                        it["done"] = [str(td(t, m) / "runs" / "test" / tag / var / f"L{L}.json") for L in range(a, b + 1)]
                        needs = [str(td(t, m) / "feat" / ft / f"L{L}_{p}.npz") for ft in ftags
                                 for L in range(a, b + 1) for p in positions]
                        if needs:
                            it["needs"] = needs
                        if t == "arc":
                            it.update(arc_tags(a, extra_gb=3.0 if method in ("dbm_gemmascope", "das") else 0.0))
                        items.append(it)
    for name, items in specs.items():
        if only and not name.startswith(only):
            continue
        f = ROOT / "queue_v2" / f"{name}.jsonl"
        have = set()
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    have.add(json.loads(line)["done"][0])
        new = [it for it in items if it["done"][0] not in have and not all(Path(p).exists() for p in it["done"])]
        print(f"[plan_p1] {name}: {len(new)} new items (of {len(items)})", flush=True)
        if new and not dry:
            old = f.read_text() if f.exists() else ""
            tmp = f.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(old + "".join(json.dumps(it) + "\n" for it in new))
            os.replace(tmp, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="", help="spec name prefix, e.g. p1_fits")
    a = ap.parse_args()
    plan(a.dry, a.only)


if __name__ == "__main__":
    main()
