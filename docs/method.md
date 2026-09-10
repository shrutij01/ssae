---
title: Method
description: The paired observation model, the constrained sparse-shift objective, and what a learned decoder column represents.
---

<p class="kicker">Method</p>

# The model and its estimator

<p class="lede">A Sparse Shift Autoencoder reconstructs the difference between two
related activations under a constraint on the expected number of active code
coordinates. This page fixes the notation, states the observation model, writes
down the objective the theory analyses, and records how the implemented
estimator departs from it.</p>

## Notation

| Symbol | Object | Space |
|---|---|---|
| $\mathbf{x}$ | text observation | $\mathcal{X} \subseteq \mathbb{R}^{d_x}$ |
| $\mathbf{c}$ | concept representation, unobserved | $\mathcal{C} \subseteq \mathbb{R}^{d}$ |
| $\mathbf{z}$ | language model activation | $\mathcal{Z} \subseteq \mathbb{R}^{d_z}$ |
| $g$ | generative map, concepts to text | $\mathcal{C} \to \mathcal{X}$ |
| $f$ | the language model up to a fixed layer | $\mathcal{X} \to \mathcal{Z}$ |
| $\mathbf{A}$ | mixing matrix, $\mathbf{z} = \mathbf{A}\mathbf{c}$ | $\mathbb{R}^{d_z \times d}$ |
| $S$ | concepts that change within one pair | $S \subseteq [d]$ |
| $V$ | concepts that change in *some* pair | $V \subseteq [d]$ |
| $r$ | encoder | $\mathbb{R}^{d_z} \to \mathbb{R}^{m}$ |
| $q$ | decoder | $\mathbb{R}^{m} \to \mathbb{R}^{d_z}$ |
| $m$ | width of the learned code | $m = \lvert V \rvert$ in the theory |

Random vectors are bold and lowercase; their coordinates are $c_k$, $z_j$. The
ground-truth concept dimension is $d$ and the learned code dimension is $m$. The
identification result is stated at $m = \lvert V \rvert$; the experiments also
report the overcomplete case $m = d_z$.

## Why a difference

Consider the two phrases the paper uses. The first is *pretty pink round matte
frame*; the second is *ugly grey round metallic frame*. Each is described by
roughly five attributes, so neither concept vector is sparse. They differ in
three: description, colour, texture. Shape and object are common to both.

This asymmetry is general. Language model activations mix language, sentiment,
tense, topic, and register at once, so $\mathbf{c}$ is dense and a sparsity
prior on it is misspecified. A difference between two related activations
retains only what changed, and *that* is sparse. SSAE takes the difference as
its observation.

## The observation model

<div class="defn" markdown="1">
<span class="label">Definition 1 (Paired data-generating process).</span>
Draw a change set $S \sim p(S)$ and then a pair of concept representations
$(\mathbf{c}, \tilde{\mathbf{c}}) \sim p(\mathbf{c}, \tilde{\mathbf{c}} \mid S)$
such that $c_k = \tilde c_k$ almost surely for every $k \notin S$. The observed
texts are $\mathbf{x} = g(\mathbf{c})$ and $\tilde{\mathbf{x}} = g(\tilde{\mathbf{c}})$.
</div>

The learner sees $(\mathbf{x}, \tilde{\mathbf{x}})$ and never $S$, $\mathbf{c}$,
or $\tilde{\mathbf{c}}$. Writing $\mathcal{S} = \{S \subseteq [d] : p(S) > 0\}$
for the supported change sets, the concepts that can move anywhere in the
distribution are

$$V \;=\; \bigcup_{S \in \mathcal{S}} S ,$$

taken without loss of generality to be $V = \{1, \dots, \lvert V \rvert\}$.
Concepts outside $V$ may differ between unrelated examples, but they are frozen
within a pair.

<div class="defn" markdown="1">
<span class="label">Assumption 1 (Linear representation hypothesis).</span>
The composition $f \circ g$ is linear: there is a matrix
$\mathbf{A} \in \mathbb{R}^{d_z \times d}$ with
$\mathbf{z} = f(g(\mathbf{c})) = \mathbf{A}\mathbf{c}$.
</div>

Extracting both members of the pair at the same layer and token position and
subtracting gives the quantity SSAE models,

$$\Delta\mathbf{z} \;=\; f(\tilde{\mathbf{x}}) - f(\mathbf{x}) \;=\; \mathbf{A}\Delta\mathbf{c} \;=\; \mathbf{A}_V \Delta\mathbf{c}_V ,$$

where the last equality holds because $\Delta\mathbf{c}_{\bar V} = \mathbf{0}$ by
construction. The reduction to $\mathbf{A}_V$ matters: the full matrix
$\mathbf{A}$ need not be injective, and $d$ may far exceed $d_z$, but the
submatrix indexed by the varying concepts can still have full column rank
whenever $\lvert V \rvert \le d_z$.

Pair construction is therefore part of the model, not preprocessing.
Subtraction cancels a component only when that component is equal in both
members. Two texts that differ in an unrecorded property — length, syntax,
register — leave that difference inside $\Delta\mathbf{z}$, where the estimator
will treat it as a concept.

## The estimator

The encoder and decoder are affine, with the decoder bias reused as a
pre-encoder offset:

$$\widehat{\Delta\mathbf{c}}_V = r(\Delta\mathbf{z}) = \mathbf{W}_e(\Delta\mathbf{z} - \mathbf{b}_d) + \mathbf{b}_e, \qquad
\widehat{\Delta\mathbf{z}} = q(\widehat{\Delta\mathbf{c}}_V) = \mathbf{W}_d \widehat{\Delta\mathbf{c}}_V + \mathbf{b}_d .$$

The population problem the identification result analyses is reconstruction
subject to a hard budget on expected code occupancy,

$$
(\hat r, \hat q) \in \arg\min_{r, q} \;
\mathbb{E}\big[\lVert \Delta\mathbf{z} - q(r(\Delta\mathbf{z})) \rVert_2^2\big]
\quad \text{subject to} \quad
\mathbb{E}\big[\lVert r(\Delta\mathbf{z}) \rVert_0\big] \le \beta ,
$$

at the correctly specified budget $\beta = \mathbb{E}\lVert\Delta\mathbf{c}_V\rVert_0$.
The constraint is what separates SSAE from an ordinary affine autoencoder on
differences: without it, the solution is determined only up to an arbitrary
invertible mixing, as [the theory page](theory/identification.md) shows.

## What is actually optimised

Three departures separate the implemented estimator from the population problem,
and each is a place the guarantee can fail rather than a detail of engineering.

**The norm is relaxed.** Training replaces $L_0$ by $L_1$, since the former has
no useful gradient. The empirical objective normalises the reconstruction term
per sample and the constraint by the code width,

$$
\min \; \frac{1}{N}\sum_{i=1}^{N} \frac{\lVert \Delta\mathbf{z}_i - q(r(\Delta\mathbf{z}_i))\rVert_2^2}{\lVert \Delta\mathbf{z}_i \rVert_2^2}
\quad \text{subject to} \quad
\frac{1}{mN}\sum_{i=1}^{N} \lVert r(\Delta\mathbf{z}_i) \rVert_1 \le \beta .
$$

An $L_1$ target of $0.005$ bounds a mean absolute code value. It does not mean
that half a percent of coordinates are active, and the two quantities need
separate calibration before any target is transferred between estimators.

**The constraint is enforced, not penalised.** The Lagrangian saddle point is
found with ExtraAdam through the `cooper` library, using the same optimiser for
primal and dual variables. A penalty $\lambda\lVert\cdot\rVert_1$ would make the
achieved sparsity an outcome of tuning rather than a specification; a constraint
makes $\beta$ directly interpretable and avoids the feature-suppression that
penalties induce.

**The geometry is normalised.** The encoder output is layer-normalised, decoder
columns are renormalised every step, and gradient components parallel to those
columns are projected out after renormalisation so that the two operations do
not fight.

Model selection uses UDR, ranking runs by the median pairwise MCC over ten pairs
drawn from five seeds, which requires no labels. [Training options in full →](reproduce/train.md)

## What a decoder column is

Column $k$ of $\mathbf{W}_d$ is the activation-space direction the model
associates with shift coordinate $k$. Under the identification result it equals
$\mathbf{A}_V \mathbf{e}_{\pi(k)}$ up to a non-zero scale, so adding it to an
activation moves one concept:

$$\mathbf{z} + \hat q(\mathbf{e}_k) = \mathbf{A}\mathbf{c} + \lambda \mathbf{A}\mathbf{e}_{\pi(k)} = \mathbf{A}\big(\mathbf{c} + \lambda\mathbf{e}_{\pi(k)}\big) = f\big(g(\tilde{\mathbf{c}}_{\pi(k), \lambda})\big) .$$

Three things remain unknown and no amount of training supplies them: which
concept $\pi(k)$ names, the sign of the direction, and its unit $\lambda$.
Naming requires held-out labels; sign and scale require calibration. The
guarantee is that the column does not *mix*, not that it announces itself.

## SSAE against a sparse autoencoder

The two models differ in their reconstruction target, and every other axis is
independent of that choice.

| | SSAE | SAE |
|---|---|---|
| Reconstructs | paired difference $\Delta\mathbf{z}$ | single activation $\mathbf{z}$ |
| Sparsity assumed on | concept *shifts* $\Delta\mathbf{c}$ | concept *levels* $\mathbf{c}$ |
| Needs $d$ known | no — only $V$ enters | yes, implicitly, via dictionary width |
| Correlated concepts | permitted | violate incoherence |
| Identifiability | permutation and scaling | none in general |

Activation function (linear, ReLU, TopK, JumpReLU) and optimisation style
(penalised, constrained) are separate choices that either family can make. In
this repository a model trained on differences is an SSAE regardless of what its
activation is or what its historical run identifier says.

One consequence is easy to miss. Sparsity on
$\mathbf{c}$ requires committing to a dictionary large enough to hold every
concept the model might represent, which is the overcomplete-dictionary problem
that motivates ever-wider SAEs. Sparsity on $\Delta\mathbf{c}$ requires only
that few concepts change per pair. The number of concepts the model represents
in total never enters the argument.

## Where this can fail

The linear representation hypothesis is an assumption about a particular layer
of a particular model, not a property of language. If the representation is
nonlinear, $\mathbf{z} = h(\mathbf{c})$, then a small change satisfies only the
first-order relation $\Delta\mathbf{z} \approx \mathbf{J}_h(\mathbf{c})\Delta\mathbf{c}$;
when the Jacobian varies strongly with context, one decoder column does not
denote the same concept everywhere. The theory's remaining conditions concern
the pairs rather than the model, and are collected on
[the conditions page](theory/conditions.md).

## Sources

The model and both propositions are from Joshi et al.; the sparsity argument
that drives the proof is due to Lachapelle et al.; the paired-observation
setting generalises Locatello et al. and Ahuja et al.

```bibtex
@misc{joshi2026sparseshiftautoencodersidentifying,
      title={Sparse Shift Autoencoders for Identifying Concepts from Large Language Model Activations},
      author={Shruti Joshi and Andrea Dittadi and Sébastien Lachapelle and Dhanya Sridhar},
      year={2026},
      eprint={2502.12179},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12179},
}

@inproceedings{lachapelle2023synergies,
  title     = {Synergies between Disentanglement and Sparsity: Generalization
               and Identifiability in Multi-Task Learning},
  author    = {Lachapelle, S{\'e}bastien and Deleu, Tristan and Mahajan, Divyat
               and Mitliagkas, Ioannis and Bengio, Yoshua and
               Lacoste-Julien, Simon and Bertrand, Quentin},
  booktitle = {International Conference on Machine Learning},
  year      = {2023}
}

@inproceedings{locatello2020weakly,
  title     = {Weakly-Supervised Disentanglement Without Compromises},
  author    = {Locatello, Francesco and Poole, Ben and R{\"a}tsch, Gunnar and
               Sch{\"o}lkopf, Bernhard and Bachem, Olivier and
               Tschannen, Michael},
  booktitle = {International Conference on Machine Learning},
  year      = {2020}
}

@inproceedings{ahuja2022weakly,
  title     = {Weakly Supervised Representation Learning with Sparse
               Perturbations},
  author    = {Ahuja, Kartik and Hartford, Jason and Bengio, Yoshua},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}

@inproceedings{gallegoPosada2022cooper,
  title  = {Cooper: A Toolkit for Lagrangian-Based Constrained Optimization},
  author = {Gallego-Posada, Jose and Ramirez, Juan},
  year   = {2022}
}
```
