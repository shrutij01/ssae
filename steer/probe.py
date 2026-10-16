"""Logistic-regression probes (torch, full-batch L-BFGS; the sklearn objective ``C * sum(loss) + ||W||^2 / 2``).

* ``fit_logistic`` — binary probe; its unit weight vector is the supervised "probe" steering baseline next to
  DiffMean (user addendum #4). Features are centred and divided by their global RMS (a scalar, so the direction is
  in raw residual coordinates).
* ``fit_multinomial`` — the 28-way profession probe (per-feature standardised), used to score "occupation kept" on
  generated text.
"""

from __future__ import annotations

import numpy as np
import torch


def _lbfgs(params, closure, iters: int = 300):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")
    opt.step(closure)


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, device=None) -> dict:
    """Binary L2 logistic regression. Returns unit direction ``v`` (toward y = 1) and diagnostics."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    rms = float(np.sqrt(((X - mu) ** 2).mean()))
    Xt = torch.tensor((X - mu) / rms, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)

    def closure():
        for p in (w, b):
            if p.grad is not None:
                p.grad.zero_()
        z = Xt @ w + b
        loss = C * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="sum") + 0.5 * (w @ w)
        loss.backward()
        return loss

    _lbfgs([w, b], closure)
    with torch.no_grad():
        acc = float(((Xt @ w + b > 0).float() == yt).float().mean())
    wv = w.detach().double().cpu().numpy()
    return {"v": wv / np.linalg.norm(wv), "train_acc": acc, "C": C, "rms": rms, "n": int(len(y))}


def fit_multinomial(X: np.ndarray, y: np.ndarray, n_classes: int, C: float = 1.0, device=None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    W = torch.zeros(n_classes, X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(n_classes, device=dev, requires_grad=True)

    def closure():
        for p in (W, b):
            if p.grad is not None:
                p.grad.zero_()
        loss = C * torch.nn.functional.cross_entropy(Xt @ W.T + b, yt, reduction="sum") + 0.5 * (W * W).sum()
        loss.backward()
        return loss

    _lbfgs([W, b], closure, iters=500)
    return {"W": W.detach().double().cpu().numpy(), "b": b.detach().double().cpu().numpy(), "mu": mu, "sd": sd}


def multinomial_predict(model: dict, X: np.ndarray) -> np.ndarray:
    Z = (np.asarray(X, dtype=np.float64) - model["mu"]) / model["sd"]
    return (Z @ model["W"].T + model["b"]).argmax(1)
