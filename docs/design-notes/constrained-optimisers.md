---
title: Constrained Optimisers
description: Why the primary constrained SSAE runs use extrapolation and separate primal and dual ExtraAdam updates.
---

<p class="kicker">Design notes · optimisation</p>

# Choosing the optimiser for constrained SSAEs

<p class="lede">Constrained SSAE training is a saddle problem: model parameters minimise the objective while a non-negative multiplier responds when the sparsity ceiling is violated.</p>

The frozen primary comparison uses:

| Role | Optimiser | Learning rate |
|---|---|---|
| Primal model parameters | ExtraAdam, minimising | \(5\times10^{-4}\) |
| Dual sparsity multiplier | ExtraAdam, maximising | \(2.5\times10^{-4}\) |
| Primal–dual wrapper | Extrapolation | Two-stage lookahead/update |

The dual rate is the primal rate divided by `dual_lr_div=2`. Cooper projects
the inequality multiplier back to the non-negative domain after its update.

Why extrapolation? A simultaneous descent–ascent step can orbit in a coupled
min–max problem: each side reacts to where the other side was. The
extra-gradient step first evaluates a provisional move and then updates from
that lookahead point. This costs roughly twice as much per update, but is the
stability-oriented choice frozen in the reported runs.

The code also supports alternating updates and a cheaper single-pass
simultaneous update. Those are solver alternatives, not additional SSAE model
families, and the current result catalog does not contain an optimiser
ablation. Decoder-gradient projection and per-update column renormalisation are
separate geometric maintenance steps.

See the [Cooper optimiser documentation](https://cooper.readthedocs.io/en/stable/optim.html)
for the update wrappers used by the implementation.
