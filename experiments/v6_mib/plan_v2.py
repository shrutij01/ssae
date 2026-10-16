"""Queue items for the RAVEL-test completion and the SSAE-A/B v2 refits (DEVIATIONS #17-18).

    python -m experiments.v6_mib.plan_v2

f6_indep_newfits_{dev,final}: Cause / Isolate of the Addendum-2 fits (tuned-width ICA, all-component PCA, random basis)
    on the independent-regime sets (planned only for the natural sets before), so the regimes table is complete; prio 0.
b2_bf_sets: converged-Adam budget-free fits (``--solver adamconv --tag _ad``, 4 restarts) on the 72 pair sets SSAE-A/B
    use (language dial, natural and independent regimes, tie dial, OOD sets): the v2 warm starts; prio 1.
b1_bf_units (the #14 items z_budgetfree_adamconv, renamed so they sort first): the same at every MIB unit; raised
    from 12 to 1 (v2 warm starts).
v2_fits_{units,sets}: ``fit_v2`` per unit / set and version (run from the current code snapshot via jobs/bootstrap.py,
    so any worker can take them); ready once the warm start exists and the version passed the synthetic check (marker
    constrained_v2/PASS_<version>); prio 1.
v2_sweep / v2_test: MIB DBM validation sweep, public-test scoring and k = 1, 2, 5 of dbm_feat_ssae{A,B}2_s0 (prio 1 /
    2), and the DBM sweep and test of the converged budget-free fits dbm_feat_ssaefull_s0_ad (prio 2 / 3; they replace
    the hybrid fits' sweep f1_sweep_budgetfree, which is taken off the queue).
v2_units_{dev,final}: RAVEL Cause / Isolate (DBM) of SSAE-A/B v2 and of the converged budget-free fits (ssaefull_ad)
    on the pair sets of the tests (same sets and variables as the Addendum-2 items, plus the independent regime);
    prio 1 (A/B) and 2 (budget-free).
"""
import hashlib
import json
import os
import re
from pathlib import Path

from experiments.v6_mib.qitems import boot_module, snapshot

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
Q = ROOT / "queue_v2"
FEAT = ROOT / "ravel" / "L14_entity_last_token" / "feat"
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
SETS = re.compile(r"^(dialL_g\d+_s\d|natural_s\d|indep_s\d|tie_b\d+_s\d|tmplA_s0|noK\w+_s0)$")


def write(name: str, items: list[dict]) -> None:
    (Q / f"{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print(name, len(items))


def clone(it: dict, old: str, new: str, prio, extra_needs=()) -> dict:
    s = json.dumps(it).replace(f"/{old}.npz", f"/{new}.npz").replace(f"__{old}__", f"__{new}__")
    c = json.loads(s)
    c["prio"] = prio
    c["needs"] = [c["feat"]["path"], *extra_needs]
    return c


def main() -> None:
    snap = snapshot()
    sets = sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json") if SETS.match(p.stem))
    dev = [json.loads(l) for l in (Q / "f4_units_dev.jsonl").read_text().splitlines() if l.strip()]
    fin = [json.loads(l) for l in (Q / "t4_units_final.jsonl").read_text().splitlines() if l.strip()]
    icw = next(re.search(r"__(icaw\d+|icawfull)__", i["id"]).group(1) for i in dev if "__icaw" in i["id"])

    # 1. independent-regime items for the Addendum-2 fits (cloned from the natural-set items)
    ind_dev, ind_fin = [], []
    for src, dst in ((dev, ind_dev), (fin, ind_fin)):
        for it in src:
            g = it["id"].split("__")[0]
            if re.match(r"^natural_s\d$", g) and not re.search(r"__ssaefull", it["id"]):
                c = json.loads(json.dumps(it).replace("natural_s", "indep_s"))
                c["prio"] = 0
                dst.append(c)
    write("f6_indep_newfits_dev", ind_dev)
    write("f6_indep_newfits_final", ind_fin)

    # 2. converged budget-free fits on the A/B sets (warm starts)
    write("b2_bf_sets", [{"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 1,
                          "args": ["sets", "--layer", "14", "--sets", s, "--which", "ssaefull", "--solver", "adamconv",
                                   "--tag", "_ad", "--restarts", "4"], "done": [str(FEAT / s / "ssaefull_ad.npz")]}
                         for s in sets])

    # 3. raise the MIB-unit converged budget-free fits (unclaimed ones only: the claim is keyed by the item's hash)
    zf = Q / "b1_bf_units.jsonl"          # the #14 items (z_budgetfree_adamconv), renamed to run first
    out, raised = [], 0
    for l in zf.read_text().splitlines():
        it = json.loads(l)
        h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
        if not (ROOT / "claims" / zf.stem / h).exists() and it["prio"] != 1:
            it["prio"] = 1
            raised += 1
        out.append(it)
    tmp = zf.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(i) + "\n" for i in out))
    os.replace(tmp, zf)
    print("b1_bf_units raised", raised)

    # 4. v2 fits
    fits_u, sweep, test = [], [], []
    for tm in ["ravel__gemma", "arc__gemma", "mcqa__gemma", "mcqa__qwen"]:
        t, m = tm.split("__")
        td = ROOT / "mib" / tm
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        for L in meta["layers"]:
            for p in meta["positions"]:
                uk = f"L{L}_{p}"
                for v in "AB":
                    fits_u.append({**boot_module(snap, "experiments.v6_mib.fit_v2",
                                                 ["units", "--task", t, "--model", m, "--units", uk, "--seed", "0",
                                                  "--which", v]), "prio": 1,
                                   "done": [str(td / "feat" / f"ssae{v}2_s0" / f"{uk}.npz")],
                                   "needs": [str(td / "feat" / "ssaefull_s0_ad" / f"{uk}.npz"),
                                             str(ROOT / "constrained_v2" / f"PASS_{v}")]})
        Ls, c = meta["layers"], CHUNK[t]
        for var in VARIABLES[t]:
            for a0 in range(0, len(Ls), c):
                lay = Ls[a0:a0 + c]
                ls = f"{lay[0]}-{lay[-1]}"
                for tag, pr in (("ssaeA2_s0", 1), ("ssaeB2_s0", 1), ("ssaefull_s0_ad", 2)):
                    needs = [str(td / "feat" / tag / f"L{L}_{p}.npz") for L in lay for p in meta["positions"]]
                    base = {"task": t, "model": m, "variable": var, "seed": 0, "layers": ls}
                    sweep.append({"cmd": "train_eval", **base, "split": "val", "method": f"dbm_feat_{tag}", "prio": pr,
                                  "needs": needs})
                    test.append({"cmd": "eval_saved", **base, "split": "test", "method": f"dbm_feat_{tag}", "prio": pr + 1})
                    if tag != "ssaefull_s0_ad":
                        test.append({"cmd": "eval_saved", **base, "split": "test", "needs": needs, "prio": pr + 1,
                                     "method": f"k1_feat_{tag},k2_feat_{tag},k5_feat_{tag}"})
    write("v2_fits_units", fits_u)
    write("v2_fits_sets", [{**boot_module(snap, "experiments.v6_mib.fit_v2",
                                          ["sets", "--layer", "14", "--sets", s, "--which", v]), "prio": 1,
                            "done": [str(FEAT / s / f"ssae{v}2.npz")],
                            "needs": [str(FEAT / s / "ssaefull_ad.npz"), str(ROOT / "constrained_v2" / f"PASS_{v}")]}
                           for s in sets for v in "AB"])
    fb = Q / "f1_sweep_budgetfree.jsonl"
    if fb.exists():
        (ROOT / "queue_held").mkdir(exist_ok=True)
        os.replace(fb, ROOT / "queue_held" / fb.name)
        print("f1_sweep_budgetfree moved to queue_held/")
    write("v2_sweep", sweep)
    write("v2_test", test)

    # 5. RAVEL Cause / Isolate items for the v2 fits and the converged budget-free fits
    tmpl_dev = [i for i in dev + ind_dev if f"__{icw}__" in i["id"]]
    tmpl_fin = [i for i in fin + ind_fin if f"__{icw}__" in i["id"]]
    udev, ufin = [], []
    for fam, prio in (("ssaeA2", 1), ("ssaeB2", 1), ("ssaefull_ad", 2)):
        udev += [clone(i, icw, fam, prio) for i in tmpl_dev]
        ufin += [clone(i, icw, fam, prio) for i in tmpl_fin]
    for i in ufin:
        i["select"]["from"] = i["id"][:-len("__final")]
    write("v2_units_dev", udev)
    write("v2_units_final", ufin)


if __name__ == "__main__":
    main()
