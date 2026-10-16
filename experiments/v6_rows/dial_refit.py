"""Refits of the dial's and the composition's two paper columns with their full encoder and decoder (user, 2026-10-03;
PREREGISTRATION.md section 13).

    python -m experiments.v6_rows.dial_refit pool --procs 4     # claim-based worker pool (GPU)
    python -m experiments.v6_rows.dial_refit status

The original step-L0 and L1 (LayerNorm; dial methods ``orig_L0`` / ``orig_L1``) are refitted exactly as the dial fitted
them (``dial_fit.run_ladder``: ``experiments.v5.models.fit_ladder`` on the unit's fit pairs, the unit's seed), for every
level and seed of the Bias-in-Bios dial (item 1), the composition (item 3) and the CEBaB dial (item 4), both models. The
state (W_enc, rms, LayerNorm gain, raw-space decoder) goes to ``$SSAE_V6_ROOT/dial/fits_full/``. A refit is used only if
its matched directions (the dial's matching on the unit's match pairs) equal the stored ones: |cos| > 0.9999 per concept.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, RECORDS, checked_savez, write_json

VARIANTS = {"orig_L0": "L0-v3", "orig_L1": "L1-v3"}
TOL = 0.9999
OUT = RECORDS / "dial_refit"


def full_root() -> Path:
    from experiments.v6_rows.dial_fit import root
    return root() / "fits_full"


def tasks() -> list[tuple]:
    from experiments.v6_rows.dial_fit import units
    return [(u, name) for u in units() for name in VARIANTS]


def key(u: tuple, name: str) -> str:
    from experiments.v6_rows.dial_fit import level_tag
    item, level, sd, model = u
    return f"refit__{item}__{MODEL_TAG[model]}__{level_tag(item, level)}__s{sd}__{name}"


def paths(u: tuple, name: str) -> tuple[Path, Path, Path]:
    from experiments.v6_rows.dial_fit import level_tag, unit_dir
    item, level, sd, model = u
    rel = Path(item) / MODEL_TAG[model] / level_tag(item, level) / f"s{sd}"
    return full_root() / rel / f"{name}.npz", OUT / rel / f"{name}.json", unit_dir(*u) / f"{name}.npz"


def run(u: tuple, name: str, dev: str) -> None:
    from experiments.v5 import models as V5
    from experiments.v6_rows.dial_fit import Unit, fit_info, match
    st_path, rec_path, stored = paths(u, name)
    t0 = time.time()
    U = Unit(*u)
    fit = V5.fit_ladder(VARIANTS[name], U.split("fit", False), U.seed, dev)
    dirs, info = match(fit, U.split("match", False), U.concepts)
    old = np.load(stored)
    cos = {c: float(abs(np.asarray(dirs[c], np.float64) @ old[c].astype(np.float64) /
                        (np.linalg.norm(dirs[c]) * np.linalg.norm(old[c]) + 1e-30))) for c in U.concepts}
    st = fit.state
    st_path.parent.mkdir(parents=True, exist_ok=True)
    checked_savez(st_path, kind=np.array(str(st["kind"])), W_enc=np.asarray(st["W_enc"], np.float32), rms=np.asarray(st["rms"]),
             ln_weight=np.asarray(st["ln_weight"], np.float32), D_raw=np.asarray(st["D_raw"], np.float32))
    write_json(rec_path, {"item": u[0], "level": u[1], "seed": u[2], "model": u[3], "method": name,
                          "variant": VARIANTS[name], "layer": U.layer, "match": info, "cos_to_stored": cos,
                          "reproduces": bool(all(v > TOL for v in cos.values())), "tolerance": TOL,
                          "state": str(st_path), "fit": fit_info(fit), "seconds": time.time() - t0})
    print(f"[dial_refit] {key(u, name)}: min |cos| {min(cos.values()):.6f} ({time.time() - t0:.0f}s)", flush=True)


def worker(dev: str = "cuda") -> None:
    from experiments.v6_rows.dial_fit import claim, release
    while True:
        took = False
        for u, name in tasks():
            if paths(u, name)[1].is_file():
                continue
            k = key(u, name)
            if not claim(k):
                continue
            took = True
            try:
                run(u, name, dev)
            except Exception as err:
                if "CUDA" in repr(err) or "cuda" in repr(err):
                    release(k)
                    raise
                write_json(paths(u, name)[1].with_suffix(".error.json"), {"error": repr(err),
                                                                         "traceback": traceback.format_exc()[-3000:]})
                print(f"[dial_refit] {k}: ERROR {err!r}", flush=True)
            finally:
                release(k)
            break
        if not took:
            return


def pool(procs: int) -> None:
    import subprocess
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ps = [subprocess.Popen([sys.executable, "-u", "-m", "experiments.v6_rows.dial_refit", "worker"], env=env)
          for _ in range(procs)]
    for p in ps:
        p.wait()


def status() -> None:
    done = sum(paths(u, n)[1].is_file() for u, n in tasks())
    err = sum(paths(u, n)[1].with_suffix(".error.json").is_file() for u, n in tasks())
    bad = []
    for u, n in tasks():
        f = paths(u, n)[1]
        if f.is_file() and not json.loads(f.read_text())["reproduces"]:
            bad.append(key(u, n))
    print(f"refits: {done}/{len(tasks())} done, {err} errors, {len(bad)} not reproducing: {bad[:10]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "worker":
        import torch                              # no precision override: the dial's own workers set none
        worker("cuda" if torch.cuda.is_available() else "cpu")
    elif cmd == "pool":
        pool(int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4)
    elif cmd == "status":
        status()
    else:
        raise SystemExit(f"unknown command {cmd}")
