"""Shared paths, the test guard and the redirects of the v6 rows study (``results/v6_rows``;
docs/v6/v6_constrained_brief.md step 2).

Inputs read in place and never written: results/v4, results/v5, results/v5_activation, results/v5_fullwidth (and their
scratch fits). New records go to ``results/v6_rows``, heavy files to ``$SCRATCH/ssae_v6_rows``. The v5 code is reused
unchanged: ``install()`` points its fit lookup, IIA job directories, record directories and test guard at this study.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.v5.common import (IDENT_ROWS as V5_IDENT_ROWS, LAYERS, MODEL_TAG, REPO,  # noqa: F401
                                   ROWS as V5_ROWS, SEEDS, read_json,
                                   run_record_v5, write_json)

V6 = Path(os.environ.get("SSAE_V6_RESULTS", REPO / "results" / "v6_rows"))
FW = V6
ACT = V6
RECORDS = V6 / "records"
TABLES = V6 / "tables"
FIGURES = V6 / "figures"
V5 = REPO / "results" / "v5" if "SSAE_V5_RESULTS" not in os.environ else Path(os.environ["SSAE_V5_RESULTS"])

# Methods of this study: SSAE-A (budget-constrained) and SSAE-B (fidelity-constrained) at default settings and the
# sensitivity rows (brief section 2). Comparison rows: the original SSAE (NeurIPS, LayerNorm) = v5's L0-v3 fits; the
# budget-free SSAE = results/v5_fullwidth's ssae_full fits (read only).
VERSIONS = {"ssae_A": [1.0, 0.5, 2.0], "ssae_B": [1.0, 0.5, 2.0]}     # beta (A) / tau (B) multipliers
FAMILIES = ("ssae_A", "ssae_B")
DETERMINISTIC: set = set()
NEW = FAMILIES
NEW_DETERMINISTIC = DETERMINISTIC
HIDDEN = 2304


def method_id(version: str, mult: float) -> str:
    return version if mult == 1.0 else f"{version}__x{mult:g}"


def is_new(method: str) -> bool:
    return method.startswith(("ssae_A", "ssae_B"))


def family_of(method: str) -> str:
    return method.split("__")[0]


# scope (user decision before the preregistration commit, DEVIATIONS #1): bias-in-bios-strat2 is dropped
EXCLUDED = {"bias-in-bios-strat2"}
ROWS = [r for r in V5_ROWS if r[0] not in EXCLUDED]
IDENT_ROWS = [r for r in V5_IDENT_ROWS if r[0] not in EXCLUDED]
IIA_ROWS = [(d, m) for d, m, _, _ in ROWS if d.startswith(("bias-in-bios", "truthful"))]
BOUND_MIN = 0.5


def v6_root() -> Path:
    root = os.environ.get("SSAE_V6_ROOT")
    return Path(root) if root else Path(os.environ["SCRATCH"]) / "ssae_v6_rows"


def live_repo() -> Path:
    return Path(os.environ.get("SSAE_V5_REPO", REPO))


def preregistration_committed() -> bool:
    """True only if HEAD contains results/v6_rows/PREREGISTRATION.md quoting the SHA-256 of the committed
    results/v6_rows/preregistered_configs.json."""
    import hashlib
    import re
    import subprocess

    def show(path: str) -> bytes | None:
        out = subprocess.run(["git", "-C", str(live_repo()), "show", f"HEAD:{path}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg = show("results/v6_rows/PREREGISTRATION.md")
    configs = show("results/v6_rows/preregistered_configs.json")
    if not prereg or not configs:
        return False
    m = re.search(rb"PREREGISTERED-CONFIGS-SHA256: ([0-9a-f]{64})", prereg)
    return bool(m) and m.group(1).decode() == hashlib.sha256(configs).hexdigest()


def only_methods() -> set | None:
    """Test-phase pass filter: $V6_METHODS = comma-separated method ids (unset: every selected method)."""
    v = os.environ.get("V6_METHODS", "").strip()
    return set(v.split(",")) if v else None


def assert_split_allowed(split: str) -> None:
    if split == "test" and not preregistration_committed():
        raise RuntimeError("refusing to compute a TEST metric before results/v6_rows/PREREGISTRATION.md is committed")


# ------------------------------------------------------------------ fits of this study
def unit_rel(dataset: str, model: str, layer: int, seed: int) -> str:
    from experiments.v5.data import chosen_read
    return f"fit/{dataset}__{MODEL_TAG[model]}__{chosen_read(dataset, model)}/L{layer}/s{seed}"


def record_dir(dataset: str, model: str, layer: int, seed: int, method: str) -> Path:
    return RECORDS / unit_rel(dataset, model, layer, seed) / method


def state_path(dataset: str, model: str, layer: int, seed: int, method: str) -> Path:
    return v6_root() / "runs" / unit_rel(dataset, model, layer, seed) / f"{method}.npz"


AB_STOPPED = ("SSAE-A / SSAE-B work stopped by the user on 2026-10-03 (DEVIATIONS #46): no refits, polish or "
              "convergence checks; the recorded fits stay as appendix ablations")


def assert_ab_allowed() -> None:
    raise SystemExit(AB_STOPPED)


OWN_FITS = {"L0-noLN", "L1-noLN"}      # without-LayerNorm originals completed in v6 (user, 2026-10-02)
STEP_RULE = "v2_choice"                  # the refit with constrained_choice_v2.json; earlier fits are not current


def is_current(metrics: dict) -> bool:
    return (metrics.get("info") or {}).get("step_rule") == STEP_RULE


def locate_v6(dataset: str, model: str, method: str, layer: int, seed: int):
    st, rec = state_path(dataset, model, layer, seed, method), record_dir(dataset, model, layer, seed, method)
    # "v5" = saved in the v5 runner's format, so the v5 registry rebuilds it unchanged
    if not (st.is_file() and (rec / "metrics.json").is_file()):
        return None
    if method in OWN_FITS:                 # this study's own fits of v5 methods (no step rule)
        return ("v5", st, rec)
    try:
        ok = is_current(json.loads((rec / "metrics.json").read_text()))
    except (ValueError, OSError):
        ok = False
    return ("v5", st, rec) if ok else None


def _check_arrays(where, arrays: dict) -> None:
    """Finiteness and precision guard (DEVIATIONS #56): no float16 storage, no non-finite entries."""
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.dtype.kind == "f":
            if a.dtype == np.float16:
                raise ValueError(f"{where}: {k} is float16 (store float32)")
            if not np.isfinite(a).all():
                raise ValueError(f"{where}: {k} has {int((~np.isfinite(a)).sum())} non-finite entries")


def checked_savez(file, *args, **arrays) -> None:
    _check_arrays(file, arrays)
    np.savez(file, *args, **arrays)


def checked_savez_c(file, *args, **arrays) -> None:
    _check_arrays(file, arrays)
    np.savez_compressed(file, *args, **arrays)


def save_fit(dataset: str, model: str, layer: int, seed: int, method: str, fit, metrics: dict, config: dict) -> None:
    """Same files as the v5 runner (record: config.json + metrics.json; state: the fit's arrays + atoms)."""
    run_record_v5(record_dir(dataset, model, layer, seed, method), config, metrics)
    arrays = {k: v for k, v in fit.state.items() if isinstance(v, np.ndarray)}
    arrays["atoms"] = fit.atoms.astype(np.float32)
    sp = state_path(dataset, model, layer, seed, method)
    sp.parent.mkdir(parents=True, exist_ok=True)
    checked_savez_c(sp, **{k: np.asarray(v) for k, v in arrays.items()})


def locate_fw_study(dataset: str, model: str, method: str, layer: int, seed: int):
    """The budget-free full-width SSAE fits of the v5_fullwidth study (read only)."""
    rel = unit_rel(dataset, model, layer, seed)
    st = Path(os.environ["SCRATCH"]) / "ssae_v5_fw" / "runs" / rel / f"{method}.npz"
    rec = live_repo() / "results" / "v5_fullwidth" / "records" / rel / method
    return ("v5", st, rec) if st.is_file() and (rec / "metrics.json").is_file() else None


def locate_act_study(dataset: str, model: str, method: str, layer: int, seed: int):
    """Comparison rows' seed fits at bound-aware layers written by the v5_activation study (read only)."""
    rel = unit_rel(dataset, model, layer, seed)
    st = Path(os.environ["SCRATCH"]) / "ssae_v5_act" / "runs" / rel / f"{method}.npz"
    rec = live_repo() / "results" / "v5_activation" / "records" / rel / method
    return ("v5", st, rec) if st.is_file() and (rec / "metrics.json").is_file() else None


def v6_job_dir(dataset: str, model: str, split: str, read_tag: str | None = None) -> Path:
    name = f"{dataset}__{MODEL_TAG[model]}" + (f"__read-{read_tag}" if read_tag else "")
    return v6_root() / "iia" / split / name


def locate_v4refit(dataset: str, model: str, method: str, layer: int, seed: int):
    """Full-state refits of the v4 LayerNorm fits (DEVIATIONS #51): used in place of the v4 state once the refit
    reproduces the recorded fit's validation direction accuracy exactly."""
    if method not in ("L0-v3", "L1-v3"):
        return None
    from experiments.v5.data import chosen_read
    rel = Path(f"{dataset}__{MODEL_TAG[model]}__{chosen_read(dataset, model)}") / f"L{layer}" / f"s{seed}"
    st, rec = v6_root() / "ln_diag" / "v4refit" / rel / f"{method}.npz", RECORDS / "v4refit" / rel / method
    if st.is_file() and (rec / "metrics.json").is_file() and read_json(rec / "metrics.json").get("reproduces") is True:
        return "v5", st, rec
    return None


_INSTALLED = False


def locate_unconv(dataset: str, model: str, method: str, layer: int, seed: int):
    """The unconverged record: SSAE-A / SSAE-B fitted with the v1 settings and the 10,000-step cap (method id
    "<version>@unconv"), in its archive location or, before archiving, at its original place."""
    base = method.split("@")[0]
    rel = unit_rel(dataset, model, layer, seed)
    cands = [(v6_root() / "runs_unconverged" / rel / f"{base}.npz", RECORDS / rel.replace("fit/", "fit_unconverged/", 1) / base),
             (state_path(dataset, model, layer, seed, base), record_dir(dataset, model, layer, seed, base))]
    for st, rec in cands:
        if st.is_file() and (rec / "metrics.json").is_file():
            try:
                if not is_current(json.loads((rec / "metrics.json").read_text())):
                    return ("v5", st, rec)
            except (ValueError, OSError):
                pass
    return None


def install() -> None:
    """Point the v5 modules at this study (fit lookup: this study's fits first, then v5 / v4; IIA job directories,
    erasure / application records and every test guard: this study's)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import experiments.v5.common as C5
    import experiments.v5.registry as R

    orig = R.locate

    def locate(dataset, model, method, layer, seed):
        refit = locate_v4refit(dataset, model, method, layer, seed)
        if refit is not None:
            return refit
        if method.endswith("@unconv"):
            return locate_unconv(dataset, model, method, layer, seed)
        found = locate_v6(dataset, model, method, layer, seed)
        if found is None and method == "ssae_full":
            found = locate_fw_study(dataset, model, method, layer, seed)
        if found is None and (method in ("L0-v3", "L1-v3", "ssae_gated_count", "ssae_subspace")
                              or method.startswith(("gemmascope", "act_"))):
            found = locate_act_study(dataset, model, method, layer, seed)   # v5_activation's fits first (bound-aware)
        return found if found is not None else orig(dataset, model, method, layer, seed)

    R.locate = locate
    orig_load = R.load

    def load(dataset, model, method, layer, seed):
        fit = orig_load(dataset, model, method, layer, seed)
        st = fit.state if isinstance(fit.state, dict) else {}
        _check_arrays(f'load {dataset} {method} L{layer} s{seed}',
                      {'atoms': fit.atoms, **{k: v for k, v in st.items() if isinstance(v, np.ndarray) and v.dtype.kind == 'f'
                                              and v.dtype != np.float16}})
        return fit
    R.load = load
    C5.assert_split_allowed = assert_split_allowed
    import importlib
    for name in ("experiments.v5.iia_prep", "experiments.v5.erasure", "experiments.v5.applications",
                 "experiments.v5.scoring"):
        mod = importlib.import_module(name)
        if hasattr(mod, "assert_split_allowed"):
            mod.assert_split_allowed = assert_split_allowed
        if hasattr(mod, "RECORDS"):
            mod.RECORDS = RECORDS
        if hasattr(mod, "job_dir"):
            mod.job_dir = v6_job_dir
        if name == "experiments.v5.applications":      # refusal generations: this study's scratch, not v5's
            mod.v5_root = v6_root
    _INSTALLED = True


def write_md_table(df, path, title: str = "", floatfmt: str = ".3f") -> None:
    """A readable markdown version of a results table (no tabulate dependency)."""
    import numpy as _np
    cols = list(df.columns)
    fmt = lambda v: ("" if v is None or (isinstance(v, float) and _np.isnan(v)) else
                     (format(v, floatfmt) if isinstance(v, (float, _np.floating)) else str(v)))
    lines = ([f"# {Path(path).stem}", "", title, ""] if title else [f"# {Path(path).stem}", ""])
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    Path(path).write_text("\n".join(lines) + "\n")


def status_line(item: str, outcome: str, files: list[str]) -> None:
    """One line per finished item in results/v6_rows/STATUS.md (user, 2026-10-02)."""
    from datetime import datetime, timezone
    p = V6 / "STATUS.md"
    if not p.is_file():
        p.write_text("# Status of the correlation-dial program (one line per item)\n\n| when (UTC) | item | outcome | files |\n|---|---|---|---|\n")
    with p.open("a") as fh:
        fh.write(f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | {item} | {outcome} | {', '.join(files)} |\n")
