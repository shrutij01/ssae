---
title: Sufficient Variability
description: Why sparse shift supports must vary enough to distinguish latent coordinates.
---

<p class="note-kicker">Theory III · support information</p>

# Sparse is not enough; supports must separate factors

If two latent factors always change together in the same proportion, their
individual decoder directions are not identifiable from their sum. Sparsity
reduces the candidate explanations, but **variability across changed supports**
is what distinguishes the coordinates.

Let (S_i=\operatorname{supp}(\Delta c_i)) be the changed-factor support for
pair (i). Informally, a sufficiently variable collection
(\{S_i\}_{i=1}^n) must expose enough different co-change patterns that no two
latent coordinates have indistinguishable support signatures. The exact rank
or combinatorial condition depends on the theorem's observation assumptions.

## Three failure cases

1. **Permanent co-change.** Factors (j) and (k) occur together in every
   pair. Only their combined direction is observable.
2. **A never-changing factor.** A decoder column for that factor is not
   identified by the shift data.
3. **Dense interventions.** Nearly every factor changes in every pair, erasing
   the support contrast that sparse identification uses.

## Testable consequence

<div class="hypothesis-sheet">
<p><b>Null \(H_0\).</b> Concept recovery is invariant to the diversity and rank
of the changed-support design once sample size and marginal change frequency
are held fixed.</p>
<p><b>Alternative \(H_1\).</b> Recovery improves as support signatures become
more separable and collapses for deliberately aliased co-change patterns.</p>
<p><b>Clean test.</b> Generate mechanisms with identical factor marginals but
different support incidence rank; pre-specify MCC after permutation/scale
matching as the primary outcome.</p>
</div>

A code-backed synthetic identification result is not yet published. The
required falsification experiment should map recovery as support signatures
become aliased, rather than report only favorable sufficiently variable
mechanisms.
