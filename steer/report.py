"""Statistics and tables (brief section 1.6).

* ``boot_ci``: mean with a 95% percentile bootstrap CI over prompts (1000 resamples; each prompt's value is its mean
  over samples).
* ``paired_tests``: paired Wilcoxon signed-rank tests over prompts, SSAE vs each other method, Holm-corrected within
  a family; effect sizes = median paired difference and matched-pairs rank-biserial correlation.
* Label-free, supervised and reference rows are always reported separately (``steer.common.CATEGORY``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

N_BOOT = 1000


def boot_ci(values, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_diff_ci(a, b, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Paired (same prompts) mean difference a - b with a bootstrap CI over prompts."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return boot_ci(d, n, seed)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * q[i]))
        adj[i] = running
    out[ok] = adj
    return out


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_tests(per_prompt: pd.DataFrame, value: str, ref: str, others, key: str = "prompt_id",
                 method_col: str = "method") -> pd.DataFrame:
    """``per_prompt``: one row per (method, prompt) with column ``value``. Returns one row per comparison
    (ref vs other): n, mean difference, median difference, rank-biserial r, Wilcoxon p, Holm p."""
    wide = per_prompt.pivot_table(index=key, columns=method_col, values=value, aggfunc="mean")
    rows = []
    for o in others:
        if o not in wide.columns or ref not in wide.columns:
            rows.append({"ref": ref, "other": o, "n": 0, "p": np.nan})
            continue
        pair = wide[[ref, o]].dropna()
        d = (pair[ref] - pair[o]).to_numpy()
        if len(d) == 0 or np.all(d == 0):
            p = 1.0 if len(d) else np.nan
        else:
            p = float(wilcoxon(pair[ref], pair[o], zero_method="wilcox", alternative="two-sided").pvalue)
        rows.append({"ref": ref, "other": o, "n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else np.nan,
                     "median_diff": float(np.median(d)) if len(d) else np.nan,
                     "rank_biserial": rank_biserial(d) if len(d) else np.nan, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out


def fmt_ci(m, lo, hi, nd: int = 2) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("n/a" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _selftest() -> None:
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"prompt_id": np.repeat(np.arange(40), 2), "method": ["a", "b"] * 40,
                       "v": np.r_[[rng.normal(1, .1), rng.normal(0, .1)] * 20, [rng.normal(1, .1), rng.normal(0, .1)] * 20]})
    t = paired_tests(df, "v", "a", ["b"])
    assert t.loc[0, "p"] < 1e-5 and t.loc[0, "rank_biserial"] == 1.0
    m, lo, hi = boot_ci(np.ones(10))
    assert m == lo == hi == 1.0


if __name__ == "__main__":
    _selftest()
    print("report self-test ok")
