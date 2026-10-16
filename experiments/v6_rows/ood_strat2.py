"""OOD without refitting, held-out professions (user checklist of 2026-10-02, item 4a; DEVIATIONS #21).

    python -m experiments.v6_rows.ood_strat2 score     # direction accuracy, ID (strat2 test) vs held-out (CPU)
    python -m experiments.v6_rows.ood_strat2 iia       # decoder-direction Cause, k = 1, ID vs held-out (GPU)
    python -m experiments.v6_rows.ood_strat2 results

v5's bias-in-bios-strat2 fits saw only nurse (13) and surgeon (25) bios. Their gender direction -- the direction
(decoder column) v5 matched to gender on strat2 train, at the method's selected strat2 layer -- is scored, unchanged, on
(i) strat2 test (in-distribution; v5's recorded value) and (ii) bias-in-bios-strat28 test pairs whose two bios are
both from the other 26 professions (held out): direction accuracy against those pairs' supervised gender direction
(v5's reference), and the k = 1 interchange along the direction (v5's MIB pipeline and pronoun readout, the projection
featurizer E = a^T, D = a; cap 500).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V5, checked_savez, read_json, v6_root, write_json

SRC, TGT = "bias-in-bios-strat2", "bias-in-bios-strat28"
HELD_OUT_EXCLUDE = (13, 25)                     # nurse, surgeon
MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")
OUT = RECORDS / "ood_strat2"


def fits() -> list[dict]:
    """v5's scored strat2 test fits: (model, method, layer, seed, matched gender direction index, ID value)."""
    out = []
    for model in MODELS:
        f = V5 / "records" / "scores" / "test" / f"{SRC}__{MODEL_TAG[model]}.json"
        if not f.is_file():
            continue
        for r in read_json(f)["rows"]:
            if "error" in r or "assignment" not in r:
                continue
            out.append({"model": model, "method": r["method"], "layer": int(r["layer"]), "seed": int(r["seed"]),
                        "j": int(r["assignment"][0]), "dir_acc_id": float(r["dir_acc_per_factor"][0])})
    return out


def held_out_mask(model: str) -> np.ndarray:
    from experiments.v5.data import pairs as load_pairs
    p = load_pairs(TGT, model, "test").reset_index(drop=True)
    return (~p.profession_from.isin(HELD_OUT_EXCLUDE) & ~p.profession_to.isin(HELD_OUT_EXCLUDE)).to_numpy()


def score() -> pd.DataFrame:
    from experiments.v4 import metrics as M
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    rows, refs = [], {}
    for it in fits():
        model, L = it["model"], it["layer"]
        if (model, L) not in refs:
            ev = gender_only(load_split(TGT, "test", L, model))
            m = held_out_mask(model)
            r = M.reference_directions(ev.X[m], ev.Y[m], C=ev.C[m] if ev.C is not None else None)[0][:, 0]
            refs[(model, L)] = (r / np.linalg.norm(r), int(m.sum()))
        ref, n = refs[(model, L)]
        try:
            a = R.load(SRC, model, it["method"], L, it["seed"]).atoms[:, it["j"]].astype(np.float64)
        except Exception as err:
            print(f"[ood strat2] {model} {it['method']} L{L} s{it['seed']}: {err!r}", flush=True)
            continue
        a /= np.linalg.norm(a)
        rows.append({**{k: it[k] for k in ("method", "layer", "seed", "j", "dir_acc_id")}, "model": MODEL_TAG[model],
                     "dir_acc_heldout": float(abs(a @ ref)), "n_heldout_pairs": n})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "direction_accuracy_test.csv", index=False)
    return df


def iia_dir(model: str, kind: str) -> Path:
    return v6_root() / "iia" / "test" / "ood_strat2" / f"{MODEL_TAG[model]}__{kind}"


def iia_prep() -> None:
    """ID: v5's strat2 test task and filter; held-out: v5's strat28 test task with its filter restricted to pairs of
    the 26 other professions. Featurizer: the projection onto the matched gender direction (k = 1)."""
    from experiments.v5 import registry as R
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    fl = fits()
    for model in MODELS:
        for kind, src_row in (("ID", SRC), ("heldout", TGT)):
            d = iia_dir(model, kind)
            (d / "feat").mkdir(parents=True, exist_ok=True)
            src = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / f"{src_row}__{MODEL_TAG[model]}"
            shutil.copy2(src / "task.json", d / "task.json")
            filt = json.loads((src / "filter_test.json").read_text())
            if kind == "heldout":
                m = held_out_mask(model)
                filt = {"kept": {"gender": [int(i) for i in filt["kept"]["gender"] if m[int(i)]]},
                        "stats": {"gender": {"note": "v5's filter restricted to the held-out professions"}}}
            (d / "filter_test.json").write_text(json.dumps({"kept": {"gender": filt["kept"]["gender"]},
                                                            "stats": filt.get("stats", {})}))
            rows = []
            for it in [x for x in fl if x["model"] == model]:
                try:
                    a = R.load(SRC, model, it["method"], it["layer"], it["seed"]).atoms[:, it["j"]].astype(np.float64)
                except Exception as err:
                    print(f"[ood strat2 iia] {model} {it['method']}: {err!r}", flush=True)
                    continue
                a /= np.linalg.norm(a)
                fname = f"feat/{it['method']}__L{it['layer']}__s{it['seed']}__dir.npz"
                checked_savez(d / fname, E=a[None, :].astype(np.float32), D=a[:, None].astype(np.float32),
                         latents=np.array([it["j"]]), readout_dot=np.ones(1, dtype=np.float32))
                rows.append({"id": f"{it['method']}|L{it['layer']}|s{it['seed']}|k1|dir", "method": it["method"],
                             "seed": it["seed"], "k": 1, "layer": it["layer"], "kind": "linear", "file": fname,
                             "features": {"gender": [0]}, "matched_latents": {"gender": [it["j"]]},
                             "variant": "decoder direction"})
            (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[ood strat2 iia] {MODEL_TAG[model]} {kind}: {len(filt['kept']['gender'])} kept pairs, "
                  f"{len(rows)} configs", flush=True)


def iia_run() -> None:
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for model in MODELS:
        for kind in ("ID", "heldout"):
            subprocess.run([sys.executable, "-u", str(runner), str(iia_dir(model, kind)), "--split", "test", "--cap",
                            "500", "--batch", "12"], env=env, check=False)


def results() -> pd.DataFrame:
    d = pd.read_csv(OUT / "direction_accuracy_test.csv")
    g = d.groupby(["model", "method"]).agg(layer=("layer", "first"), seeds=("seed", "nunique"),
                                           dir_acc_id=("dir_acc_id", "mean"),
                                           dir_acc_heldout=("dir_acc_heldout", "mean")).reset_index()
    cause = {}
    for model in MODELS:
        for kind in ("ID", "heldout"):
            for f in glob.glob(str(iia_dir(model, kind) / "results" / "*.json")):
                if f.endswith(".error.json"):
                    continue
                r = json.loads(Path(f).read_text())
                v = (r.get("scores") or {}).get("gender", {}).get("gender")
                if v:
                    cause.setdefault((MODEL_TAG[model], r["config"]["method"], kind), []).append(float(np.mean(v["scores"])))
    for kind in ("ID", "heldout"):
        g[f"cause_dir_{kind}"] = [np.mean(cause[(m, x, kind)]) if (m, x, kind) in cause else np.nan
                                  for m, x in zip(g.model, g.method)]
    g["drop_dir_acc"] = g.dir_acc_id - g.dir_acc_heldout
    g["drop_cause_dir"] = g.cause_dir_ID - g.cause_dir_heldout
    g.to_csv(OUT / "results_test.csv", index=False)
    return g


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "score":
        print(score().shape)
    elif what == "iia":
        iia_prep()
        iia_run()
    elif what in ("results", "all"):
        if what == "all":
            score()
            iia_prep()
            iia_run()
        print(results().round(3).to_string())
