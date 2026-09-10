---
title: Augmented Lagrangian
description: How an augmented-Lagrangian formulation changes the constrained SSAE objective, and its current status in the experiments.
---

<p class="kicker">Design notes · formulation</p>

# Framing SSAE training as an augmented Lagrangian

<p class="lede">An augmented Lagrangian makes constraint violation increasingly costly while retaining an adaptive multiplier. It is implemented as an option, but it is not the formulation used in the primary frozen comparison.</p>

Write the aggregate sparsity ceiling as

\[
g(\theta)=S(h_\theta)-\tau\leq 0,
\]

where \(S\) is mean absolute code magnitude, hard active fraction, or smooth
occupancy. The standard Lagrangian used in the frozen runs is

\[
\mathcal L(\theta,\lambda)=R(\theta)+\lambda g(\theta),
\qquad \lambda\geq 0.
\]

An augmented-Lagrangian formulation adds a quadratic response to violation.
Schematically, its extra term is \(\frac{\rho}{2}[g(\theta)]_+^2\). Cooper uses
the inequality-specific projected form, so negative slack is not naively
squared and penalised. The multiplier still adapts the reconstruction–sparsity
trade-off; \(\rho\) changes the local cost of remaining infeasible.

This can be useful when a cheap simultaneous update oscillates or responds too
slowly to violation. It does not change the SSAE observation model or its
identifiability assumptions—it changes only how the constrained problem is
solved.

The implementation exposes:

```bash
python -m ssae.training embeddings.h5 data.yaml \
  --constrained-optim simultaneous \
  --formulation augmented-lagrangian \
  --penalty-coeff 1.0
```

This flag selects Cooper's augmented-Lagrangian **formulation** inside a
gradient-based primal–dual loop. It should not automatically be called the full
classical augmented-Lagrangian method, which additionally specifies an
alternating update schedule, inner primal minimisation, and a penalty schedule.

Most importantly, the frozen primary comparison uses plain `lagrangian` with
`extrapolation`, not the augmented option. Any augmented-Lagrangian advantage
therefore remains a future ablation. A fair comparison should fix seeds,
updates, schedule, target, and model, then report reconstruction, achieved
constraint value, gap, multiplier, and run failures—not only the configured
target.

See [Cooper's formulation reference](https://cooper.readthedocs.io/en/stable/formulations.html)
for the inequality-specific definition and the distinction between a
formulation and an optimisation method.
