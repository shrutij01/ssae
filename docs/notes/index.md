---
title: Theory & Notes
description: The assumptions, propositions, and open questions behind Sparse Shift Autoencoders.
---

# Theory & Notes

These notes explain why sparse changes can identify concept directions and how the theoretical conditions connect to the empirical results.

## Overview

1. **[Observation model](../theory/shift-observation-model.md)** - Why learn a shift? Pairing removes shared variation when the pair construction isolates a meaningful change.
2. **[Equivalence](../theory/identifiability.md)** - What does identifiable mean? Recovery is stated up to permutation and component-wise scaling.
3. **[Sufficient variability](../theory/sufficient-variability.md)** - When are shifts informative enough? Changing supports must separate the latent coordinates.
4. **[Constrained sparsity](../theory/constrained-optimisation.md)** - Why constrain sparsity? A dual variable turns the desired sparse regime into an explicit optimization target.
5. **[Limits and open questions](../theory/limits-open-questions.md)** - Which assumptions define the scope? Pair construction, local geometry, dictionary width, and selection all matter.

## How Theory and Evidence Connect

The theory states what follows under the observation and variability assumptions. The Evidence section tests the corresponding empirical outcomes: concept recovery, steering, transfer, and cross-seed stability. Reconstruction establishes that the representation retains the target signal; the remaining studies test whether SSAE coordinates make that signal useful.
