---
title: The Identification Result
description: Linear identifiability from reconstruction alone, why it is insufficient for steering, and the sparsity argument that reduces the ambiguity to permutation and scaling.
---

<p class="kicker">Theory 01 · result</p>

# From a subspace to its coordinates

<p class="lede">Two propositions do the work. The first says that reconstruction
recovers the shift subspace in an arbitrary coordinate system. The second says
that adding a correctly specified sparsity budget collapses that arbitrariness to
a permutation and a rescaling. The gap between them is the entire contribution.</p>

## Step one: reconstruction identifies a subspace

Three assumptions set up the linear result. The first is the linear
representation hypothesis of [the method page](../method.md); the remaining two
concern the varying concepts.

<div class="defn" markdown="1">
<span class="label">Assumption 2 (Injectivity).</span>
The matrix $\mathbf{A}_V \in \mathbb{R}^{d_z \times \lvert V \rvert}$ is
injective, which requires $d_z \ge \lvert V \rvert$.
</div>

<div class="defn" markdown="1">
<span class="label">Assumption 3 (Sufficient variability).</span>
Let $\Delta^c_V$ denote the support of the random vector $\Delta\mathbf{c}_V$.
Then $\operatorname{span}(\Delta^c_V) = \mathbb{R}^{\lvert V \rvert}$.
</div>

Assumption 2 says distinct shifts among the varying concepts produce distinct
activation shifts, so nothing is lost in the mapping. Assumption 3 says the
observed shifts explore every direction of the varying-concept space, so nothing
is left unconstrained for want of data.

<div class="defn" markdown="1">
<span class="label">Proposition 1 (Linear identifiability).</span>
Let $(\hat r, \hat q)$ solve the unconstrained reconstruction problem. Under
Assumptions 1–3 there exists an invertible
$\mathbf{L} \in \mathbb{R}^{\lvert V\rvert \times \lvert V\rvert}$ with
$$\hat q = \mathbf{A}_V \mathbf{L}, \qquad \hat r(\mathbf{z}) = \mathbf{L}^{-1}\mathbf{A}_V^{+}\mathbf{z}$$
for every $\mathbf{z} \in \operatorname{Im}(\mathbf{A}_V)$.
</div>

Outside the image of $\mathbf{A}_V$ the encoder is unconstrained, because the
objective never evaluates it there. The proposition therefore describes the
model on the support of $\Delta\mathbf{z}$ and says nothing beyond it.

## Why that is not enough

Proposition 1 certifies that the learned representation has the right
information and the right dimension. It does not certify that its axes are
concept axes. Expanding the decoder direction for coordinate $k$,

$$\hat q(\mathbf{e}_k) \;=\; \mathbf{A}_V\mathbf{L}\mathbf{e}_k \;=\; \sum_{j} L_{jk}\,\mathbf{A}_V\mathbf{e}_j ,$$

and adding it to an activation yields

$$\mathbf{z} + \hat q(\mathbf{e}_k) \;=\; \mathbf{A}\Big(\mathbf{c} + \sum_j L_{jk}\mathbf{e}_j\Big) \;=\; f\Big(g\big(\mathbf{c} + \textstyle\sum_j L_{jk}\mathbf{e}_j\big)\Big) .$$

The intervention is well defined and lands on a real point of concept space. It
simply moves every concept for which $L_{jk} \neq 0$. A method that stops at
Proposition 1 delivers steering directions that are coherent and reproducible
and that change the wrong number of things.

## Step two: sparsity picks the coordinate system

The extra assumption governs how change patterns vary across pairs. Recall
$\mathcal{S} = \{S \subseteq [d] : p(S) > 0\}$.

<div class="defn" markdown="1">
<span class="label">Assumption 4 (Sufficient diversity of multi-concept shifts).</span>
Both conditions hold.

1. *Support variability.* For every varying concept $k \in V$,
   $$\bigcup_{S \in \mathcal{S}\,:\,k \notin S} S \;=\; V \setminus \{k\} .$$
2. *Continuity.* For every $S \in \mathcal{S}$, the conditional distribution
   $\mathbb{P}_{\Delta\mathbf{c}_S \mid S}$ admits a density with respect to
   Lebesgue measure on $\mathbb{R}^{\lvert S\rvert}$.
</div>

Condition 1 is the substantive one, and it is best read as a requirement that no
concept be inseparable from the rest. For each concept $k$, the pairs in which
$k$ does *not* change must together exercise every other varying concept. If two
concepts always change together, no observation distinguishes them from their
sum, and no estimator can. Condition 2 rules out degenerate distributions
concentrated on a lower-dimensional surface, where a mixed coordinate could be
sparse by coincidence.

<div class="defn" markdown="1">
<span class="label">Proposition 2 (Identifiability up to permutation).</span>
Let $(\hat r, \hat q)$ solve the constrained problem at the correctly specified
budget $\beta = \mathbb{E}\lVert\Delta\mathbf{c}_V\rVert_0$. Under
Assumptions 1–4 there exist an invertible diagonal $\mathbf{D}$ and a
permutation $\mathbf{P}$ such that
$$\hat q = \mathbf{A}_V \mathbf{D}\mathbf{P}, \qquad \hat r(\mathbf{z}) = \mathbf{P}^{\top}\mathbf{D}^{-1}\mathbf{A}_V^{+}\mathbf{z}$$
for every $\mathbf{z} \in \operatorname{Im}(\mathbf{A}_V)$.
</div>

## The proof in three moves

**The constraint is not binding at the truth.** The true solution reconstructs
exactly and meets the budget with equality, so the constrained optimum also
attains zero reconstruction error. It is therefore an unconstrained optimum too,
and Proposition 1 applies: $\hat q = \mathbf{A}_V\mathbf{L}$ for some invertible
$\mathbf{L}$.

**The constraint transfers to the code.** From Proposition 1 the encoder
recovers $\hat r(\Delta\mathbf{z}) = \mathbf{L}^{-1}\Delta\mathbf{c}_V$, so
feasibility reads

$$\mathbb{E}\big\lVert \mathbf{L}^{-1}\Delta\mathbf{c}_V \big\rVert_0 \;\le\; \beta \;=\; \mathbb{E}\big\lVert \Delta\mathbf{c}_V \big\rVert_0 .$$

Mixing is required to be no worse than the truth, in expected support size.

**Only permutation-scalings achieve that.** The following lemma, due to
Lachapelle et al., closes the argument.

<div class="defn" markdown="1">
<span class="label">Lemma (Lachapelle et al., 2023).</span>
Let $\mathbf{L} \in \mathbb{R}^{m \times m}$ be invertible and let $\mathbf{u}$
be a random vector with support pattern $S = \{j : u_j \neq 0\}$ and
$\mathcal{S}$ the support of $p(S)$. Assume (i) for all $j$,
$\bigcup_{S \in \mathcal{S}: j \notin S} S = [m]\setminus\{j\}$, and (ii) for all
$S \in \mathcal{S}$ and all non-zero $\mathbf{a}$,
$\mathbb{P}_{\mathbf{u}_S \mid S}\{\mathbf{a}^{\top}\mathbf{u}_S = 0\} = 0$. If
$\mathbb{E}\lVert\mathbf{L}\mathbf{u}\rVert_0 \le \mathbb{E}\lVert\mathbf{u}\rVert_0$,
then $\mathbf{L} = \mathbf{D}\mathbf{P}$ for a diagonal $\mathbf{D}$ and a
permutation $\mathbf{P}$.
</div>

Applying it to $\mathbf{L}^{-1}$ and $\Delta\mathbf{c}_V$, with Assumption 4
supplying exactly hypotheses (i) and (ii), gives
$\mathbf{L}^{-1} = \mathbf{D}\mathbf{P}$ and hence Proposition 2.

The lemma has a one-line intuition. Mixing two coordinates
that are separately active creates a coordinate active in the union of their
change sets, which is strictly larger unless one of the two never changes
independently. Support variability is the assumption that forbids that escape
route, coordinate by coordinate.

## What follows for intervention

With $\hat q = \mathbf{A}_V\mathbf{D}\mathbf{P}$, write
$\lambda = D_{\pi(k),\pi(k)}$. Then

$$\mathbf{z} + \hat q(\mathbf{e}_k) \;=\; \mathbf{A}\mathbf{c} + \lambda\mathbf{A}\mathbf{e}_{\pi(k)} \;=\; \mathbf{A}\big(\mathbf{c} + \lambda\mathbf{e}_{\pi(k)}\big) \;=\; f\big(g(\tilde{\mathbf{c}}_{\pi(k),\lambda})\big) ,$$

which is the same expression as in the unidentified case with one difference:
the sum has collapsed to a single term. Adding the column moves concept
$\pi(k)$ and nothing else.

<div class="finding" markdown="1">
<span class="label">Why this is a weaker requirement than an SAE's</span>
Proposition 2 constrains the sparsity of *shifts* between concept
representations, not of the representations themselves. A standard sparse
autoencoder assumes few concepts are active per input; SSAE assumes few change
per pair. Because only the varying set $V$ enters, the guarantee never requires
knowing the total number of concepts $d$ — which is the overcomplete-dictionary
problem that SAEs must solve by making the dictionary ever wider.
</div>

## What the propositions do not say

They are statements about a population optimum of an $L_0$-constrained problem
at width $m = \lvert V\rvert$ with exact reconstruction. They do not say that a
finite-sample $L_1$-relaxed run at width $m = d_z$ inherits the conclusion, and
[the estimation page](estimation.md) is about that gap. They also say nothing
about which concept a recovered coordinate is, nor whether the assumptions hold
for a given model and dataset, which no internal diagnostic can establish.

## Sources

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

@inproceedings{lachapelle2022disentanglement,
  title     = {Disentanglement via Mechanism Sparsity Regularization: A New
               Principle for Nonlinear ICA},
  author    = {Lachapelle, S{\'e}bastien and Rodriguez, Pau and Sharma, Yash
               and Everett, Katie and Le Priol, R{\'e}mi and Lacoste, Alexandre
               and Lacoste-Julien, Simon},
  booktitle = {Conference on Causal Learning and Reasoning},
  year      = {2022}
}

@article{roeder2021linear,
  title   = {On Linear Identifiability of Learned Representations},
  author  = {Roeder, Geoffrey and Metz, Luke and Kingma, Diederik P.},
  journal = {International Conference on Machine Learning},
  year    = {2021}
}

@article{khemakhem2020vae,
  title   = {Variational Autoencoders and Nonlinear ICA: A Unifying Framework},
  author  = {Khemakhem, Ilyes and Kingma, Diederik P. and Monti, Ricardo P.
             and Hyv{\"a}rinen, Aapo},
  journal = {International Conference on Artificial Intelligence and Statistics},
  year    = {2020}
}
```
