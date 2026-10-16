"""Queue items for Part 1 as revised by Addendum 15 (DEVIATIONS #53-#54; Addendum 14's collection, fits, conditioning
and secondary DBM rows unchanged). Append-only; re-run by the controls cycle.

    python -m experiments.v6_mib.plan_p1v15 [--dry] [--only PREFIX]

Addendum 15 rows: span2_<tag> ("SSAE directions, encoder-selected"; PCA_delta selected the same way): validation for
k in {1, 2, 5, 10} (span2_val), then the public test at the variable's k (span2_test, once every layer's validation is
in); span2rand_k<k> (random bases, directions selected the same way, 20 draws) and das_k<k> (DAS, every layer, seeds
0-2) at each k chosen by the fixed-column rows (L0-v3, L1-v3, each pair set); the submission folders of the fixed-column
rows (span_submission.py --span2), once their validation is complete.

1. collect_pairs: MIB's training pool per task (RAVEL: one per variable; MCQA, ARC: one), base and counterfactual
   activations at every site (vpairs/<pool>/).
2. Random-pair fits of the four variants and PCA_delta per (pool, unit) (experiments.v6_mib.fit_p1 --pool).
3. Span rows (no DBM): span_<fit tag> for every fit (the reused orig_<variant>_s0 fits on MIB's counterfactual pairs;
   the random-pair fits p1rp<pool>_<variant>), spanpca_<PCA_delta tag>, spanrand_orig_L0ln_s0; per (row, variable,
   layer chunk); validation for every k, public test at the chosen k.
4. Secondary DBM rows (train_test, DBM seeds 0-2): the inverse decoder and the SAE treatment for the without-LayerNorm
   fits, the orthonormalized basis for fits with cond(W_d) > 1e4 at some site, PCA_delta on the random pairs; the DBM
   baselines stay in their existing p1_dbm_* specs.
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
VARIANTS = ["L0ln", "L1ln", "L0noln", "L1noln"]
LN = ("L0ln", "L1ln")
OFF = {"ravel/gemma": 0.0, "mcqa/gemma": 0.0, "arc/gemma": 0.02, "mcqa/qwen": 0.02}
SPAN_CHUNKS = {"gemma": [(a, min(a + 3, 25)) for a in range(0, 26, 4)], "qwen": [(a, min(a + 3, 23)) for a in range(0, 24, 4)]}
DBM_CHUNKS = {"gemma": [(0, 6), (7, 12), (13, 18), (19, 25)], "qwen": [(0, 5), (6, 11), (12, 17), (18, 23)]}
ARC_DBM_CHUNKS = [(0, 5), (6, 11), (12, 17), (18, 25)]
COND_MAX = 1e4


def td(t, m) -> Path:
    return ROOT / "mib" / f"{t}__{m}"


def pools(t: str) -> list[str]:
    return list(NV[t]) if t == "ravel" else ["all"]


def pool_of(t: str, var: str) -> str:
    return var if t == "ravel" else "all"


def positions(t, m) -> list[str]:
    return json.loads((td(t, m) / "shifts" / "train" / "meta.json").read_text())["positions"]


def n_layers(m) -> int:
    return 26 if m == "gemma" else 24


def arc_tags(L0: int, extra_gb: float = 0.0) -> dict:
    need = max(33.0, math.ceil(22 + 1.04 * (26 - L0))) + extra_gb
    return {"gpu_min_gb": 60, **({"free_gb_ok": need} if need <= 43 else {})}


def cond_over(t, m, tag: str) -> bool:
    """True when the fit family has cond(W_d) > 1e4 at some site (Part 1 fit info, or the cond table's records)."""
    if tag.startswith("orig_"):
        rec = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
        p = rec / "records" / f"p1_cond_orig_{t}_{m}.json"
        if p.exists():
            d = json.loads(p.read_text())
            return any(v > COND_MAX for k, v in d.items() if k.startswith(tag + "|"))
        return tag.split("_")[1] in LN                     # the LayerNorm fits were all > 1e4 where checked
    for f in (td(t, m) / "feat" / "fit_p1_info").glob(f"*_{tag}.json"):
        if json.loads(f.read_text()).get("cond", 0) > COND_MAX:
            return True
    return False


SPAN2_K = (1, 2, 5, 10)


def span2_choice(t: str, m: str, method: str, var: str) -> int | None:
    """The variable's k (Addendum 15) from every layer's validation record; None until all are in (same rule as
    mib_run.span2_choice)."""
    ext = {"ravel": (20, 50, 100)}.get(t, ())            # RAVEL's extended grid (DEVIATIONS #69, Addendum 19)
    tag = method[len("span2_"):]
    recs = []
    for L in range(n_layers(m)):
        f = td(t, m) / "runs" / "val" / method / var / f"L{L}.json"
        if not f.exists():
            return None
        pk = dict(json.loads(f.read_text())["per_k"])
        if ext:
            e = None
            for src in (f"span2x_{tag}", f"span_{tag}"):
                g = td(t, m) / "runs" / "val" / src / var / f"L{L}.json"
                if g.exists():
                    q = json.loads(g.read_text()).get("per_k", {})
                    if all(str(k) in q for k in ext):
                        e = {str(k): q[str(k)] for k in ext}
                        break
            if e is None:
                return None
            pk.update(e)
        recs.append(pk)
    metric = lambda k: float(sum(max(r[str(k)].values()) for r in recs) / len(recs))   # noqa: E731
    return min(tuple(SPAN2_K) + tuple(ext), key=lambda k: (-round(metric(k), 12), k))


def plan(dry: bool = False, only: str = "") -> None:
    from experiments.v6_mib.qitems import boot_mib, boot_module, snapshot
    snap = snapshot()
    specs: dict[str, list] = {}
    for t, m in TASKS:
        key = f"{t}/{m}"
        o = OFF[key]
        P = positions(t, m)
        Ls = list(range(n_layers(m)))
        # 1. MIB's training pools
        for pool in pools(t):
            it = boot_mib(snap, "collect_pairs", task=t, model=m, pool=pool, batch_size=16 if t == "arc" else 64)
            it.update(prio=0.12, done=[str(td(t, m) / "vpairs" / pool / "meta.json")])
            if t == "arc":
                it.update(gpu_min_gb=60, free_gb_ok=30)
            specs.setdefault("p2v14_collect", []).append(it)
        # 2. random-pair fits
        for pool in pools(t):
            for L in Ls:
                for p in P:
                    uk = f"L{L}_{p}"
                    it = boot_module(snap, "experiments.v6_mib.fit_p1", ["--task", t, "--model", m, "--unit", uk,
                                                                         "--pool", pool, "--parallel"])
                    it.update(prio=round(0.17 + o, 3), needs=[str(td(t, m) / "vpairs" / pool / "meta.json")],
                              done=[str(td(t, m) / "feat" / f"p1rp{pool}_{v}" / f"{uk}.npz") for v in VARIANTS]
                              + [str(td(t, m) / "feat" / f"p1rp{pool}_pcafull" / f"{uk}.npz")])
                    specs.setdefault("p2v14_fits", []).append(it)
        # 3. Addendum 15 span2 rows: validation for every k of the grid, then the test at the variable's k
        for var in NV[t]:
            pool = pool_of(t, var)
            vp = str(td(t, m) / "vpairs" / pool / "meta.json")
            rows = [(f"orig_{v}_s0", round(0.14 + o, 3)) for v in VARIANTS] + [("pcafull", round(0.14 + o, 3))] + \
                   [(f"p1rp{pool}_{v}", round(0.18 + o, 3)) for v in VARIANTS] + [(f"p1rp{pool}_pcafull", round(0.18 + o, 3))]
            for ftag, prio in rows:
                method = f"span2_{ftag}"
                vdir = td(t, m) / "runs" / "val" / method / var
                all_val = [str(vdir / f"L{L}.json") for L in Ls]
                for a_, b_ in SPAN_CHUNKS[m]:
                    feat_needs = [str(td(t, m) / "feat" / ftag / f"L{L}_{p}.npz") for L in range(a_, b_ + 1) for p in P]
                    it = boot_mib(snap, "span2_val", task=t, model=m, variable=var, layers=f"{a_}-{b_}", method=method)
                    it.update(prio=prio, needs=[vp] + feat_needs,
                              done=[str(vdir / f"L{L}.json") for L in range(a_, b_ + 1)])
                    specs.setdefault(f"p2v15_span2val_{t}_{m}", []).append(it)
                    it = boot_mib(snap, "span2_test", task=t, model=m, variable=var, layers=f"{a_}-{b_}", method=method)
                    it.update(prio=round(prio + 0.002, 3), needs=all_val + feat_needs,
                              done=[str(td(t, m) / "runs" / "test" / method / var / f"L{L}.json") for L in range(a_, b_ + 1)])
                    specs.setdefault(f"p2v15_span2test_{t}_{m}", []).append(it)
        # 4. secondary DBM rows (DBM seeds 0-2)
        chunks = ARC_DBM_CHUNKS if t == "arc" else DBM_CHUNKS[m]
        for var in NV[t]:
            pool = pool_of(t, var)
            rows = []
            for v in VARIANTS:
                for ftag in (f"orig_{v}_s0", f"p1rp{pool}_{v}"):
                    if v not in LN:
                        rows.append((f"dbm_pinv_{ftag}", ftag))                      # inverse decoder (#50 H)
                        rows.append(((f"dbm_feat_{ftag}" if ftag.startswith("orig_") else f"dbm_sae_{ftag}"), ftag))
                    if cond_over(t, m, ftag):
                        rows.append((f"dbm_dir_{ftag}", ftag))                       # orthonormalized basis
            rows.append((f"dbm_feat_p1rp{pool}_pcafull", f"p1rp{pool}_pcafull"))
            for method, ftag in rows:
                for d in (0, 1, 2):
                    tag = method + (f"_d{d}" if d else "")
                    for a, b in chunks:
                        it = boot_mib(snap, "train_test", task=t, model=m, variable=var, seed=0, layers=f"{a}-{b}",
                                      method=method, dbm_seed=d)
                        it["prio"] = round(0.24 + o + (0.0 if d == 0 else 0.04), 3)
                        it["done"] = [str(td(t, m) / "runs" / "test" / tag / var / f"L{L}.json") for L in range(a, b + 1)]
                        it["needs"] = [str(td(t, m) / "feat" / ftag / f"L{L}_{p}.npz") for L in range(a, b + 1) for p in P]
                        if t == "arc":
                            it.update(arc_tags(a))
                        specs.setdefault(f"p2v14_dbm_{t}_{m}", []).append(it)
    # 5. at each k chosen by the fixed-column rows: random bases selected the same way (20 draws) and DAS (seeds 0-2)
    chosen = {}
    for t, m in TASKS:
        key = f"{t}/{m}"
        o = OFF[key]
        P = positions(t, m)
        for var in NV[t]:
            pool = pool_of(t, var)
            ks = set()
            for ftag in (f"orig_L0ln_s0", f"orig_L1ln_s0", f"p1rp{pool}_L0ln", f"p1rp{pool}_L1ln"):
                k = span2_choice(t, m, f"span2_{ftag}", var)
                if k is not None:
                    ks.add(k)
                    chosen[f"{key}/{var}/{ftag}"] = k
            vp = str(td(t, m) / "vpairs" / pool / "meta.json")
            for k in sorted(ks):
                for a_, b_ in SPAN_CHUNKS[m]:
                    it = boot_mib(snap, "span2rand_test", task=t, model=m, variable=var, layers=f"{a_}-{b_}", ks=k)
                    it.update(prio=round(0.20 + o, 3), needs=[vp],
                              done=[str(td(t, m) / "runs" / "test" / f"span2rand_k{k}" / var / f"L{L}.json")
                                    for L in range(a_, b_ + 1)])
                    specs.setdefault(f"p2v15_span2rand_{t}_{m}", []).append(it)
                for d in (0, 1, 2):
                    tag = f"das_k{k}" + (f"_d{d}" if d else "")
                    for a_, b_ in (ARC_DBM_CHUNKS if t == "arc" else DBM_CHUNKS[m]):
                        it = boot_mib(snap, "train_test", task=t, model=m, variable=var, seed=0, layers=f"{a_}-{b_}",
                                      method=f"das_k{k}", dbm_seed=d)
                        it.update(prio=0.055 if d == 0 else 0.058,      # DAS at the same k comes next (#55, #60)
                                  done=[str(td(t, m) / "runs" / "test" / tag / var / f"L{L}.json") for L in range(a_, b_ + 1)])
                        if t == "arc":
                            it.update(arc_tags(a_, extra_gb=3.0))
                        specs.setdefault(f"p2v15_dask_{t}_{m}", []).append(it)
    rec = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
    if chosen and not dry:
        (rec / "records" / "span2_chosen_k.json").write_text(json.dumps(chosen, indent=1, sort_keys=True))
    # 6. submission folders of the fixed-column rows (prepared, not submitted), once their validation is complete
    for ftag in ("orig_L0ln_s0", "orig_L1ln_s0", "p1rp{pool}_L0ln", "p1rp{pool}_L1ln"):
        needs = []
        for t, m in TASKS:
            for var in NV[t]:
                tg = ftag.replace("{pool}", pool_of(t, var))
                needs += [str(td(t, m) / "runs" / "val" / f"span2_{tg}" / var / f"L{L}.json") for L in range(n_layers(m))]
        it = {"cmd": "module", "module": "cProfile",
              "args": [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "mib", "span_submission.py",
                       "--tag", ftag, "--span2"],
              "prio": 0.35, "needs": needs,
              "done": [str(rec / "records" / f"span2_submission_{ftag.replace('{pool}', 'POOL')}.json")]}
        specs.setdefault("p2v15_submission", []).append(it)
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
        print(f"[plan_p1v15] {name}: {len(new)} new items (of {len(items)})", flush=True)
        if new and not dry:
            old = f.read_text() if f.exists() else ""
            tmp = f.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(old + "".join(json.dumps(it) + "\n" for it in new))
            os.replace(tmp, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    plan(a.dry, a.only)


if __name__ == "__main__":
    main()
