"""SSAE-B with its reconstruction constraint computed from the second-moment matrix (exact reformulation; DEVIATIONS).

mean_s ||x_s - D E x_s||^2 = tr((I - DE) M (I - DE)^T), M = X^T X / n (unit-norm train shifts; M formed once per run
in float64, then float32). Values and gradients are those of the verbatim step-1 code (``constrained.Problem``) up to
float32 rounding; the per-step cost drops from five n x d x m products to two (codes and their encoder gradient) plus
O(d^2 m). Everything else (objective, eps schedule, optimizer, restarts, dual restarts, logging, convergence rule,
final per-sample statistics) is the verbatim ``constrained.train``, which is called unchanged with this ``Problem``.

    python -m experiments.v6_rows.fastB check     # agreement with the verbatim code (CPU, subsample of a real row)
"""

from __future__ import annotations

import contextlib

import cooper
import torch

from experiments.v6_rows import constrained as K


class ProblemFastB(K.Problem):
    def compute_cmp_state(self, E, D, X):
        if self.version != "B":
            return super().compute_cmp_state(E, D, X)
        if getattr(self, "_M", None) is None:
            Xd = X.double()
            self._M = (Xd.T @ Xd / X.shape[0]).to(X.dtype)
            self._I = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
        h = X @ E.T
        loss = 0.5 * torch.log(h * h + self.eps ** 2).sum(1).mean()
        R = self._I - D @ E
        rec = ((R @ self._M) * R).sum()
        viol = strict = rec - self.level
        self.last = {"loss": float(loss), "violation": float(viol), "strict": float(strict)}
        st = cooper.ConstraintState(violation=viol.reshape(1), strict_violation=strict.reshape(1))
        return cooper.CMPState(loss=loss, observed_constraints={self.constraint: st})


@contextlib.contextmanager
def fast_b():
    orig = K.Problem
    K.Problem = ProblemFastB
    try:
        yield
    finally:
        K.Problem = orig


def train(X, version, cand, init, seed, level, **kw):
    """``constrained.train`` with the reformulated constraint for B (A unchanged)."""
    from experiments.v6_rows.common import assert_ab_allowed
    assert_ab_allowed()
    if version != "B":
        return K.train(X, version, cand, init, seed, level, **kw)
    with fast_b():
        return K.train(X, version, cand, init, seed, level, **kw)


def check(steps: int = 300, n: int = 3000) -> dict:
    import json
    import time

    import numpy as np

    from experiments.v5.data import load_split
    from experiments.v6_rows.common import RECORDS, read_json
    torch.set_num_threads(8)
    ds, model, layer = "cebab-pairs", "google/gemma-2-2b", 24
    X = load_split(ds, "train", layer, model).X.astype(np.float64)
    X = X[np.random.default_rng(0).choice(len(X), n, replace=False)]
    tau = next(r["tau"] for r in read_json(RECORDS / "levels.json")["rows"]
               if r["dataset"] == ds and r["model"] == model and r["layer"] == layer)
    out = {}
    for init in ("random", "pca"):
        t0 = time.time()
        E0, D0, a = K.train(X, "B", 2, init, 0, tau, steps=steps, device="cpu")
        t1 = time.time()
        E1, D1, b = train(X, "B", 2, init, 0, tau, steps=steps, device="cpu")
        t2 = time.time()
        rel = {k: float(np.max(np.abs(np.array(a["history"][k]) - np.array(b["history"][k])) /
                            (np.abs(np.array(a["history"][k])) + 1e-6))) for k in ("loss", "strict", "multiplier")}
        out[init] = {"max_rel_diff": rel, "max_abs_diff_E": float(np.abs(E0 - E1).max()),
                     "max_abs_diff_D": float(np.abs(D0 - D1).max()), "final_verbatim": a["final"],
                     "final_fast": b["final"], "seconds_verbatim": t1 - t0, "seconds_fast": t2 - t1}
        print(init, json.dumps(out[init], indent=1), flush=True)
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["check"]:
        import json
        from experiments.v6_rows.common import RECORDS, write_json
        write_json(RECORDS / "fastB_check.json", check())


# ------------------------------------------------------------------ v2 trainer (constrained_choice_v2.json)
def problem_v2_fastb():
    """The v2 ``Problem`` (scaled violations; per-step values kept on the GPU) with B's reconstruction constraint in
    its exact second-moment form; A unchanged."""
    from experiments.v6_rows import constrained_v2 as C2

    class ProblemV2FastB(C2.Problem):
        def compute_cmp_state(self, E, D, X):
            if self.version != "B":
                return super().compute_cmp_state(E, D, X)
            if getattr(self, "_M", None) is None:
                Xd = X.double()
                self._M = (Xd.T @ Xd / X.shape[0]).to(X.dtype)
                self._I = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            h = X @ E.T
            s = self.level if self.scaled else 1.0
            loss = 0.5 * torch.log(h * h + self.eps ** 2).sum(1).mean()
            R = self._I - D @ E
            rec = ((R @ self._M) * R).sum()                       # = mean ||x - D E x||^2 (unit-norm shifts)
            viol = strict = (rec - self.level) / s
            self.calls.append(torch.stack([loss.detach(), viol.detach(), strict.detach(), rec.detach()]))
            if self.sync:
                self.last = {"loss": float(loss), "violation": float(viol), "strict": float(strict)}
            st = cooper.ConstraintState(violation=viol.reshape(1), strict_violation=strict.reshape(1))
            return cooper.CMPState(loss=loss, observed_constraints={self.constraint: st})

    return ProblemV2FastB


def train_v2(X, version, cand, level, E0, D0, **kw):
    """``constrained_v2.train_v2`` unchanged, with the exact second-moment form of B's constraint."""
    from experiments.v6_rows.common import assert_ab_allowed
    assert_ab_allowed()
    from experiments.v6_rows import constrained_v2 as C2
    orig = C2.Problem
    C2.Problem = problem_v2_fastb()
    try:
        return C2.train_v2(X, version, cand, level, E0, D0, **kw)
    finally:
        C2.Problem = orig
