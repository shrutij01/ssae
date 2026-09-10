---
title: Design Notes
description: Short explanations of the implementation choices behind the current SSAE protocol.
---

# Design Notes

These short notes explain consequential choices in the current SSAE implementation, including what the frozen protocol does and why.

## Notes

1. **[Antipodal latents](antipodal-latents.md)** - How a non-negative gate represents both orientations of a shift.
2. **[Normalisation](normalisation.md)** - How input RMS, encoder-output scale, and decoder geometry differ.
3. **[Constrained optimisers](constrained-optimisers.md)** - How model weights and the sparsity multiplier are updated.
4. **[Augmented Lagrangian](augmented-lagrangian.md)** - What changes when constraint violation receives a quadratic response.

The primary comparison is defined in <code>experiments/ssae_rms_protocol.py</code>. Notes label optional alternatives separately from the reported configuration.
