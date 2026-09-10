---
title: Incoherence fails for correlated concepts
description: Sparse coding guarantees require approximately uncorrelated dictionary columns. On two anti-correlated factors every SAE baseline falls to MCC 0.50 while SSAE reaches 0.90.
---

<p class="kicker">Notes · 2026-07-15 · theory</p>

# Incoherence fails for correlated concepts

<p class="lede">Sparse autoencoders inherit their justification from sparse
coding, where recovery is a theorem. That theorem assumes the dictionary atoms
are close to orthogonal. Concepts in language models are correlated, the
assumption fails, and the failure is visible: on a dataset with two
anti-correlated factors, every SAE baseline recovers at chance.</p>

## What classical sparse coding actually proves

Recover $\mathbf{u}$ from $\mathbf{x} = \mathbf{D}\mathbf{u}$ with
$\mathbf{u}$ sparse. The recovery guarantees — exact support recovery, stable
recovery under noise, equivalence of the $L_0$ and $L_1$ solutions — rest on one
of two conditions on the dictionary.

**Incoherence.** The mutual coherence
$\mu(\mathbf{D}) = \max_{i \neq j} \lvert\langle \mathbf{d}_i, \mathbf{d}_j\rangle\rvert$
must be small. Recovery of a $k$-sparse signal is guaranteed when roughly
$k < \tfrac12(1 + 1/\mu)$.

**Restricted isometry.** Every submatrix of $\mathbf{D}$ formed from $k$ columns
must act approximately as an isometry.

Both say the same thing in different currencies: the atoms must be nearly
orthogonal. Two dictionary columns that are strongly correlated cannot be
distinguished from each other by a sparse code, because a signal explained by one
is nearly as sparsely explained by the other.

## Why that fails for language model concepts

The atoms here are concept directions, and the concepts of interest are
correlated in the data and, if the linear representation hypothesis holds, plausibly
correlated in geometry. Formality co-varies with register, gender with
occupation — the premise of the Bias-in-Bios dataset — and sentiment with topic.
Better training does not reduce these correlations, because they are properties
of the language rather than of the estimator.

The hypotheses therefore fail for reasons intrinsic to the domain, and the
conclusion fails with them. The failure mode is specific and predictable: for two
correlated concepts, a code that merges them into one atom is *sparser* than one
that keeps them apart, because the merged atom fires on the union of occasions
while each separate atom would fire on its own. Sparsity actively selects the
mixed solution.

This is testable, and the paper tests it. On `CORR`, a semi-synthetic dataset
with two anti-correlated translation directions, every SAE baseline falls to
MCC $\approx 0.50$ — chance for a two-factor problem — while SSAE reaches
$0.90$. GemmaScope, tuned ReLU, TopK, and JumpReLU all collapse together, which
is the signature of a shared structural cause rather than a tuning failure.

The comparison that makes the point sharpest is against a *supervised* logistic
probe, given the concept labels outright. It reaches $0.77$, below the
unsupervised SSAE. Per-concept logistic regressions have no reason to return
mutually orthogonal directions when the concepts are correlated; the sparsity
constraint enforces separation directly. Label access buys information; the
sparsity constraint buys a geometry that labels alone do not supply.

## What shift sparsity asks for instead

The condition behind Proposition 2 is
[Assumption 4](../theory/conditions.md), support variability:

$$\bigcup_{S \in \mathcal{S}\,:\,k \notin S} S \;=\; V \setminus \{k\} \qquad \text{for every } k \in V .$$

For every concept $k$, the pairs in which $k$ does *not* change must together
exercise every other varying concept.

The condition is **combinatorial**. It constrains which subsets of concepts
co-vary, and says nothing about the geometry of $\mathbf{A}_V$ beyond
injectivity, nor about the correlation between concept magnitudes. Two
concepts may be perfectly anti-correlated in value and satisfy it, provided they
sometimes change apart. Two concepts that always change together violate it, and
correctly so — nothing distinguishes them from their sum.

The escape route is that the assumption is about *supports* while incoherence is
about *directions*, and correlation in magnitude does not constrain supports.
That is the formal reason SSAE survives the regime that breaks sparse coding,
and it is also why the advantage is largest exactly where interpretability is
hardest.

## Two further differences

**The overcomplete dictionary problem disappears.** Sparsity on $\mathbf{c}$
requires a dictionary wide enough for every concept the model might represent,
which is why SAE dictionaries keep growing and why the true width is unknown.
Sparsity on $\Delta\mathbf{c}$ requires only that few concepts change per pair.
The total number of concepts $d$ never enters Proposition 2; only $\lvert V\rvert$
does.

**Amortisation is a separate gap.** Classical sparse coding solves
$\mathbf{u}^\star(\mathbf{x}) = \arg\min \tfrac12\lVert\mathbf{x} - \mathbf{D}\mathbf{u}\rVert_2^2 + \lambda\lVert\mathbf{u}\rVert_1$
per input. An SAE replaces that optimisation with a single amortised encoder
pass, and the gap between the two is provable and non-zero. So even where the
geometric conditions hold, an SAE does not inherit the sparse coding guarantee —
it inherits the guarantee for a solver it is not running.

## The claim, stated carefully

None of this shows SSAEs dominate SAEs, and the paper explicitly declines that
claim. Three narrower statements hold: the standard justification for sparse
dictionary learning does not transfer to a setting with correlated concepts, the
empirical failure appears where that analysis predicts, and an observation model
based on shifts rests on a condition that correlation leaves intact.

Where the frozen catalog in this repository disagrees, it is reported.
[Experiment 02](../results/identification.md) does not reproduce the `CORR`
separation on Gemma at a much wider code width — SAE TopK edges SSAE by 0.006
MCC-P there — which is consistent with the theorem's premises failing at
$m = d_z$, and is stated on the page rather than reconciled away.

```bibtex
@article{donoho2003optimally,
  title   = {Optimally Sparse Representation in General (Nonorthogonal)
             Dictionaries via $\ell^1$ Minimization},
  author  = {Donoho, David L. and Elad, Michael},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {100},
  pages   = {2197--2202},
  year    = {2003}
}

@article{candes2005decoding,
  title   = {Decoding by Linear Programming},
  author  = {Cand{\`e}s, Emmanuel J. and Tao, Terence},
  journal = {IEEE Transactions on Information Theory},
  volume  = {51},
  pages   = {4203--4215},
  year    = {2005}
}

@article{baraniuk2008simple,
  title   = {A Simple Proof of the Restricted Isometry Property for Random
             Matrices},
  author  = {Baraniuk, Richard and Davenport, Mark and DeVore, Ronald and
             Wakin, Michael},
  journal = {Constructive Approximation},
  volume  = {28},
  pages   = {253--263},
  year    = {2008}
}

@article{oneill2025amortisation,
  title   = {Compute Optimal Inference and Provable Amortisation Gap in
             Sparse Autoencoders},
  author  = {O'Neill, Charles and Gumran, Alim and Klindt, David},
  journal = {arXiv preprint arXiv:2411.13117},
  year    = {2025}
}

@article{de2019bias,
  title   = {Bias in Bios: A Case Study of Semantic Representation Bias in a
             High-Stakes Setting},
  author  = {De-Arteaga, Maria and Romanov, Alexey and Wallach, Hanna and
             Chayes, Jennifer and Borgs, Christian and Chouldechova, Alexandra
             and Geyik, Sahin and Kenthapadi, Krishnaram and Kalai, Adam Tauman},
  journal = {ACM Conference on Fairness, Accountability, and Transparency},
  year    = {2019}
}
```
