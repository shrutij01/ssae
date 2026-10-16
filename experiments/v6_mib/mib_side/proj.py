"""SSAE atoms (projection) (DEVIATIONS #20): MIB Subspace featurizers on the orthonormalized span of an SSAE's k matched
atom directions, the featurizer type MIB's leaderboard evaluation accepts.

Methods: ``p<k>_ssae`` (the capped, preregistered SSAE of seed ``a.seed``; atoms from its matching.json, as the k-atom
variants ``k<k>_ssae``) and ``p<k>_feat_<tag>`` (a fitted family feat/<tag>, e.g. ssaefull_s0_hy, ssaeA2_s0; atoms
ranked by ``match_order`` on the train-shift codes, as ``k<k>_feat_<tag>``). Q = an orthonormal basis of the k decoder
columns (QR, float64); the featurizer is f = Q^T h, inverse Q f + (h - Q Q^T h), all k features intervened (no DBM):
the interchange replaces Q Q^T h_base by Q Q^T h_cf.
"""

from __future__ import annotations

import re

import numpy as np
import torch

import mibcore as mc


def parse(method: str) -> tuple[int, str, str | None]:
    """-> (k, family, tag): ('ssae', None) for p<k>_ssae; ('feat', tag) for p<k>_feat_<tag>."""
    m = re.match(r"p(\d+)_(feat_(.+)|ssae)$", method)
    if not m:
        raise ValueError(method)
    return int(m.group(1)), ("feat" if m.group(3) else "ssae"), m.group(3)


def atoms(a, method: str, L: int, pid: str, R) -> tuple[np.ndarray, list[int]]:
    """Decoder columns (d x k) of the k matched atoms and their indices. ``R`` = the mib_run module (matching)."""
    k, fam, tag = parse(method)
    uk = mc.unit_key(L, pid)
    if fam == "ssae":
        z = mc.load_npz(R.feat_file(a, "ssae", L, pid))
        idx = R.matched_indices(a, "ssae", L, pid, k)
    else:
        z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{uk}.npz")
        order = R.match_order(a, R.train_shifts(a, L, pid) @ z["E"].astype(np.float32).T, a.variable)
        idx = order[:k]
    return z["D"].astype(np.float64)[:, idx], [int(i) for i in idx]


def featurizers(a, method: str, L: int, tps, R) -> tuple[dict, dict]:
    feats, info = {}, {}
    for tp in tps:
        Dk, idx = atoms(a, method, L, tp.id, R)
        Q, _ = np.linalg.qr(Dk)
        feats[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32),
                                                  trainable=False, id="SSAEproj")
        info[mc.unit_key(L, tp.id)] = {"atoms": idx, "k": len(idx)}
    return feats, info
