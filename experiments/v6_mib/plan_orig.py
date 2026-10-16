"""Queue items for the polished SSAE-B and the four original-SSAE rows (DEVIATIONS #32, preregistration Addendum 7).

    python -m experiments.v6_mib.plan_orig

o_fits_{units,sets}: ``fit_orig`` (four variants in parallel on one GPU) per MIB unit and RAVEL pair set; prio 1.2.
b3_polish_{units,sets}: ``fit_v2 polish`` per SSAE-B fit (ready once the v3 fit exists); prio 1.1.
o_sweep / o_test: MIB DBM validation sweep and public-test scoring of every new featurizer (orig_*_s0 and ssaeB2p_s0;
  the k = 1, 2, 5 matched-atom variants on test for ssaeB2p); prio 1.3 / 1.4.
o_units_{dev,final}: RAVEL Cause / Isolate (DBM) of the new featurizers on the pair sets of the unit tests (cloned from
  the Addendum-2 items, as plan_v2); prio 1.3.
Pending (unclaimed) items for the pre-polish SSAE-B downstream (dbm_feat_ssaeB2_s0 sweep / test / k-atoms, the RAVEL
items of ssaeB2, the projection items of ssaeB2_s0) are retargeted to ssaeB2p; finished ones stay as the record.
All fit items run the current code snapshot through jobs/bootstrap.py.
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
ORIG = ["L0ln", "L1ln", "L0noln", "L1noln"]


def write(name: str, items: list[dict]) -> None:
    (Q / f"{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print(name, len(items))


def h(it: dict) -> str:
    return hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]


def retarget(spec: str) -> int:
    """Unclaimed items of ``spec`` naming the pre-polish SSAE-B are rewritten for the polished fits."""
    f = Q / f"{spec}.jsonl"
    if not f.exists():
        return 0
    out, n = [], 0
    for line in f.read_text().splitlines():
        it = json.loads(line)
        s = json.dumps(it)
        if "ssaeB2" in s and "ssaeB2p" not in s and not (ROOT / "claims" / spec / h(it)).exists():
            it = json.loads(s.replace("ssaeB2_s0", "ssaeB2p_s0").replace("/ssaeB2.npz", "/ssaeB2p.npz")
                            .replace("__ssaeB2__", "__ssaeB2p__"))
            n += 1
        out.append(it)
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(i) + "\n" for i in out))
    os.replace(tmp, f)
    return n


def main() -> None:
    snap = snapshot()
    sets = sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json") if SETS.match(p.stem))
    fits_u, pol_u, sweep, test = [], [], [], []
    for tm in ["ravel__gemma", "arc__gemma", "mcqa__gemma", "mcqa__qwen"]:
        t, m = tm.split("__")
        td = ROOT / "mib" / tm
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        for L in meta["layers"]:
            for p in meta["positions"]:
                uk = f"L{L}_{p}"
                fits_u.append({**boot_module(snap, "experiments.v6_mib.fit_orig",
                                             ["units", "--task", t, "--model", m, "--units", uk, "--parallel"]),
                               "prio": 1.2, "done": [str(td / "feat" / f"orig_{v}_s0" / f"{uk}.npz") for v in ORIG]})
                pol_u.append({**boot_module(snap, "experiments.v6_mib.fit_v2",
                                            ["polish", "--task", t, "--model", m, "--units", uk, "--seed", "0"]),
                              "prio": 1.1, "done": [str(td / "feat" / "ssaeB2p_s0" / f"{uk}.npz")],
                              "needs": [str(td / "feat" / "ssaeB2_s0" / f"{uk}.npz"),
                                        str(td / "feat" / "fit_v2_info" / f"{uk}_s0_B.json")]})
        Ls, c = meta["layers"], CHUNK[t]
        for var in VARIABLES[t]:
            for a0 in range(0, len(Ls), c):
                lay = Ls[a0:a0 + c]
                base = {"task": t, "model": m, "variable": var, "seed": 0, "layers": f"{lay[0]}-{lay[-1]}"}
                for tag in [f"orig_{v}_s0" for v in ORIG] + ["ssaeB2p_s0"]:
                    needs = [str(td / "feat" / tag / f"L{L}_{p}.npz") for L in lay for p in meta["positions"]]
                    sweep.append({"cmd": "train_eval", **base, "split": "val", "method": f"dbm_feat_{tag}",
                                  "prio": 1.3, "needs": needs})
                    test.append({"cmd": "eval_saved", **base, "split": "test", "method": f"dbm_feat_{tag}",
                                 "prio": 1.4})
                    test.append({"cmd": "eval_saved", **base, "split": "test", "needs": needs, "prio": 1.4,
                                 "method": f"k1_feat_{tag},k2_feat_{tag},k5_feat_{tag}"})
    write("o_fits_units", fits_u)
    write("o_fits_sets", [{**boot_module(snap, "experiments.v6_mib.fit_orig",
                                         ["sets", "--layer", "14", "--sets", s, "--parallel"]),
                           "prio": 1.2, "done": [str(FEAT / s / f"orig_{v}.npz") for v in ORIG]} for s in sets])
    write("b3_polish_units", pol_u)
    write("b3_polish_sets", [{**boot_module(snap, "experiments.v6_mib.fit_v2",
                                            ["polish", "--layer", "14", "--sets", s]), "prio": 1.1,
                              "done": [str(FEAT / s / "ssaeB2p.npz")],
                              "needs": [str(FEAT / s / "ssaeB2.npz"), str(FEAT / s / "fit_v2_info_B.json")]}
                             for s in sets])
    write("o_sweep", sweep)
    write("o_test", test)
    # RAVEL unit items: clone the Addendum-2 items of the tuned-width ICA (all the test sets) for every new family
    dev = [json.loads(l) for l in (Q / "f4_units_dev.jsonl").read_text().splitlines() if l.strip()]
    dev += [json.loads(l) for l in (Q / "f6_indep_newfits_dev.jsonl").read_text().splitlines() if l.strip()]
    fin = [json.loads(l) for l in (Q / "t4_units_final.jsonl").read_text().splitlines() if l.strip()]
    fin += [json.loads(l) for l in (Q / "f6_indep_newfits_final.jsonl").read_text().splitlines() if l.strip()]
    icw = next(re.search(r"__(icaw\d+|icawfull)__", i["id"]).group(1) for i in dev if "__icaw" in i["id"])
    udev, ufin = [], []
    for fam in [f"orig_{v}" for v in ORIG] + ["ssaeB2p"]:
        for src, dst in ((dev, udev), (fin, ufin)):
            for it in src:
                if f"__{icw}__" not in it["id"]:
                    continue
                s = json.dumps(it).replace(f"/{icw}.npz", f"/{fam}.npz").replace(f"__{icw}__", f"__{fam}__")
                c = json.loads(s)
                c["prio"] = 1.3
                c["needs"] = [c["feat"]["path"]]
                if c["id"].endswith("__final"):
                    c["select"]["from"] = c["id"][:-len("__final")]
                dst.append(c)
    write("o_units_dev", udev)
    write("o_units_final", ufin)
    for spec in ("v2_sweep", "v2_test", "v2_units_dev", "v2_units_final", "p_val", "p_test"):
        print("retargeted to ssaeB2p:", spec, retarget(spec))


if __name__ == "__main__":
    main()
