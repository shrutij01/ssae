"""D.2 — run SteeringSafety's own pipeline (patched copy) for one primary perspective: the benchmark's baseline, DIM
and PCA, then each of our label-free directions with the benchmark's DIM recipe (``extdim``: ablation at every layer)
and PCA recipe (``extadd``: activation addition at the source layer), each with the benchmark's layer / coefficient
grid, KL check and validation selection; then its metrics (effectiveness, entanglement) against the baseline.

    python -m experiments.v6_steer.ss_run --concept refusal_base [--groups theirs ssae spca ...] [--debug]

Runs in this repo's context and calls the benchmark in its own (``cwd`` = patched copy, ``PYTHONPATH`` = it).
Output: ``$SCRATCH/ssae_v6_steer/ss/runs/<concept>/<group>/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from steer.common import scratch

SS = scratch() / "steeringsafety"
RUNS = scratch() / "ss" / "runs"
SKIP = ["GSM8K", "EnronEmail", "PreciseWiki", "LongBenchV2", "DarkBenchAnthro", "DarkBenchBrandBias",
        "DarkBenchSynchopancy", "DarkBenchSneaking", "DarkBenchRetention"]
OURS = ("ssae", "spca", "sica", "apca", "aica")


def job_copy(tag: str):
    """A private copy of the patched repo per job: the benchmark writes configs/generated/<experiment>.yaml under its
    own tree, which races when runs of different perspectives / groups share one copy (DEVIATIONS #16)."""
    import shutil
    d = scratch() / "ss" / "repos" / tag
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(SS, d, ignore=shutil.ignore_patterns("generated", "experiments", "__pycache__"))
    return d


def env(ext: str | None = None) -> dict:
    e = dict(os.environ)
    e.update({"PYTHONPATH": str(SS), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
              "HF_DATASETS_CACHE": str(scratch() / "hf_cache" / "datasets"), "TOKENIZERS_PARALLELISM": "false",
              "PYTHONUNBUFFERED": "1", "SS_MAX_EVAL": os.environ.get("SS_MAX_EVAL", "300")})
    if ext:
        e["SS_EXT_DIRS"] = ext
    return e


def pipeline(concept: str, methods, exp_dir, ext=None, debug=False, extra=()) -> int:
    cmd = [sys.executable, "scripts/run/run_full_pipeline.py", "-m", "gemma2-2b", "-c", concept, "-M", *methods,
           "--experiments-dir", str(exp_dir), "--skip-datasets", *SKIP, *extra]
    if debug:
        cmd.append("--debug")
    t0 = time.time()
    print("RUN", " ".join(cmd), flush=True)
    repo = job_copy(f"{concept}__{os.environ.get('SLURM_JOB_ID', 'nojob')}_{os.getpid()}")   # unique per process (packed jobs)
    e = env(ext)
    e["PYTHONPATH"] = str(repo)
    r = subprocess.run(cmd, cwd=repo, env=e)
    print(f"exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
    return r.returncode


def baseline_dir(concept: str) -> str | None:
    d = RUNS / concept / "theirs" / "baseline_gemma2-2b"
    ts = sorted(p for p in d.glob("*") if p.is_dir()) if d.is_dir() else []
    return str(ts[-1]) if ts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--groups", nargs="*", default=["theirs", *OURS])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    for g in args.groups:
        rule = None
        if ":" in g:                                  # E1 sensitivity: "<group>:cos" -> dirs_v2/<group>__<c>__cos.pt
            g, rule = g.split(":")
        seed = g.endswith("@seed")                    # fit-seed refits (#64): "<method>_s<k>@seed" -> dirs_seeds/
        g = g[:-5] if seed else g
        new = g.endswith("@new")                      # new fits (DEVIATIONS #22): "<method>@new" -> dirs_new/
        g = g[:-4] if new else g
        if g in ("ica_tuned", "act_ica_tuned"):       # width chosen on the validation split (DEVIATIONS #23)
            import json
            from steer.common import RECORDS
            w = json.load(open(RECORDS / "ss_ica_width.json"))[g.replace("_tuned", "")]["width"]
            g = f"{g.replace('_tuned', '')}_w{w}_s0"
        tag = (f"seed_{g}" if seed else f"new_{g}" if new else g) + ("" if rule is None else f"_{rule}")
        exp = RUNS / concept_dir(args.concept, args.debug) / tag
        if g != "baseline":
            exp.mkdir(parents=True, exist_ok=True)
        if g == "baseline":
            pipeline(args.concept, ["baseline"], RUNS / concept_dir(args.concept, args.debug) / "theirs", debug=args.debug)
        elif g == "theirs":
            pipeline(args.concept, ["baseline", "dim_nokl", "pca"], exp, debug=args.debug)
        else:
            if seed:
                ext = scratch() / "ss" / "dirs_seeds" / f"{g}__{args.concept}.pt"
            elif new:
                ext = scratch() / "ss" / "dirs_new" / (f"{g}__{args.concept}.pt" if rule is None else
                                                      f"{g}__{args.concept}__{rule}.pt")
            else:
                ext = (scratch() / "ss" / "dirs" / f"{g}__{args.concept}.pt" if rule is None else
                       scratch() / "ss" / "dirs_v2" / f"{g}__{args.concept}__{rule}.pt")
            base = baseline_dir(concept_dir(args.concept, args.debug))
            extra = ("--baseline-dir", base) if base else ()
            pipeline(args.concept, ["extdimnokl", "extadd"], exp, ext=str(ext), debug=args.debug, extra=extra)


def concept_dir(concept: str, debug: bool) -> str:
    return concept + ("_debug" if debug else "")


if __name__ == "__main__":
    main()
