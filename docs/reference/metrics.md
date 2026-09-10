---
title: Metrics
description: What each reported metric measures, which transformations it cannot see, and the claim it is allowed to support.
---

<p class="kicker">Reference · metrics</p>

# Choosing a metric from the claim

<p class="lede">A metric earns its place when it is blind to the ambiguities the
theory permits and sensitive to the ones it forbids. Reconstruction error is
invariant to every invertible mixing of the code; matched correlation is
invariant only to a permutation and a coordinate-wise rescaling. That gap is why
both are reported. This page defines every quantity the catalog carries, states
what it cannot see, and names the experiment that uses it.</p>

## Pick by question

<div class="scroll" markdown="1">

| Question | Metric | Better | Cannot see | Reported in |
|---|---|---:|---|---|
| Does one coordinate track one known factor? | MCC-P | higher | unmatched coordinates when $m > d$ | [02](../results/identification.md), [06](../results/synthetic.md) |
| Does it track monotonically rather than linearly? | MCC-S | higher | the curvature it just absorbed | [02](../results/identification.md) |
| Is each factor carried by few coordinates? | DCI-C | higher | one coordinate serving two factors | [02](../results/identification.md), [06](../results/synthetic.md) |
| Is the factor recoverable from the code at all? | DCI-I, held-out $R^2$ | higher | any invertible mixing of the code | [02](../results/identification.md) |
| Does a single feature detect the concept's presence? | feature ROC-AUC | higher | duplicated features; signed factors | [02](../results/identification.md) |
| Is the set of active coordinates the right set? | support precision, recall, F1 | higher | the threshold that produced it | [02](../results/identification.md), [06](../results/synthetic.md) |
| Does the dictionary geometry recover the factor? | decoder-projection MCC-P | higher | whether the encoder fires on the right examples | [02](../results/identification.md) |
| Is the named reconstruction target preserved? | NMSE | lower | which coordinate system holds the information | [01](../results/reconstruction.md) |
| Is pair-level direction preserved? | reconstruction cosine | higher | magnitude | [01](../results/reconstruction.md) |
| Does a direction move an unseen pair? | endpoint cosine improvement | higher | whether the movement is along the concept | [03](../results/steering.md), [04](../results/transfer.md) |
| How occupied is the code? | mean $L_0(\tau)$ | lower | the ordering at another threshold | [05](../results/stability.md) |
| Do independent seeds find the same atom? | response correlation with matched decoder cosine | higher | agreement with ground truth | [05](../results/stability.md) |

</div>

## Concept recovery

Fix an evaluation set of $n$ paired examples. The known factor changes occupy
$\mathbf{C} \in \mathbb{R}^{n \times d}$, one column per ground-truth concept,
and the learned codes occupy $\widehat{\mathbf{C}} \in \mathbb{R}^{n \times m}$,
one column per coordinate of the encoder $r$. Every metric here compares columns
of the two matrices, so every one needs a planted factor.

<div class="defn" markdown="1">
<span class="label">Definition 1 (Mean correlation coefficient).</span>
For a correlation coefficient $\rho$ and an injective assignment $\pi$ from the
$d$ factors to the $m \ge d$ code coordinates,

$$\mathrm{MCC} \;=\; \max_{\pi} \frac{1}{d} \sum_{k=1}^{d} \bigl\lvert \rho\bigl(\mathbf{C}_{\cdot k},\, \widehat{\mathbf{C}}_{\cdot \pi(k)}\bigr) \bigr\rvert ,$$

with the maximising assignment found by the Hungarian algorithm on the matrix of
absolute correlations.
</div>

**MCC-P** takes $\rho$ to be Pearson correlation, **MCC-S** Spearman rank
correlation. Both are invariant to the class Proposition 2 leaves free:
permuting the code leaves the optimal assignment unchanged, and replacing a
coordinate $\hat c_j$ by $a\hat c_j + b$ with $a \neq 0$ leaves
$\lvert\rho\rvert$ unchanged, the sign absorbed by the absolute value. Mixing
two coordinates does change the score, which is what makes MCC the metric
[the identification result](../theory/identification.md) is scored against.
MCC-S also credits strictly monotone coordinate-wise transformations, so a gap
between the two measures curvature rather than recovery.

Matching stops one coordinate from claiming several factors and says nothing
about the $m - d$ coordinates left unmatched: an overcomplete code carrying a
duplicate of every factor scores what a code of width $d$ scores.

The predictability metrics form a chain of increasing permissiveness. Writing
$\rho_k^{\text{match}}$ for the matched correlation of factor $k$,
$\rho_k^{\max} = \max_j \lvert\rho_{kj}\rvert$ for its best single coordinate,
and $R_k$ for its multiple correlation against the whole code,

$$\rho_k^{\text{match}} \;\le\; \rho_k^{\max} \;\le\; R_k, \qquad R_k^2 = \text{the reported } R^2 ,$$

where each inequality buys its increase by dropping a constraint the theory
imposes. High $R^2$ with low MCC is the signature of a distributed code.

<div class="defn" markdown="1">
<span class="label">Definition 2 (DCI completeness).</span>
Let $\iota_{jk}$ be the importance a probe assigns to code coordinate $j$ when
predicting factor $k$, normalised to $p_{jk} = \iota_{jk} / \sum_{j'} \iota_{j'k}$.
Completeness of factor $k$ is one minus the entropy of that distribution in base
$m$,

$$\mathrm{DCI\text{-}C}_k \;=\; 1 + \sum_{j=1}^{m} p_{jk} \log_m p_{jk} .$$
</div>

Completeness reaches one when a single coordinate carries all of a factor's
predictive importance. It is computed per factor and never compares factors, so
one coordinate can be the sole predictor of two factors and score one for both —
the failure MCC's injective assignment forbids and DCI-C permits. **DCI-I** is
the held-out predictive performance of the same probe, a gradient-boosted tree
ensemble on a seeded 80/20 split. Both values describe the probe family and its
regularisation as much as the code.

**Held-out $R^2$** is linear predictability of a factor from the full code. For
any invertible $\mathbf{L}$ the codes $\widehat{\mathbf{C}}$ and
$\widehat{\mathbf{C}}\mathbf{L}$ span the same column space and give identical
fitted values, so $R^2$ is invariant to the ambiguity Proposition 1 leaves
unresolved on [the theory page](../theory/index.md) and cannot separate a
recovered code from a mixed one.

**Feature ROC-AUC** scores one coordinate against a binary presence label as the
probability that a positive example outranks a negative one. The implementation
takes $\max(\mathrm{AUC}, 1 - \mathrm{AUC})$ per feature before Hungarian
matching, making it invariant to sign and to any strictly increasing
transformation of the activation, so the scaling freedom $\mathbf{D}$ costs it
nothing. That folding also lifts the null above one half and matching over many
candidates lifts it further: both arms of
[experiment 06](../results/synthetic.md) sit at $\approx 0.51$, chance for this
estimator, because continuous signed factors carry no presence-absence structure
for one feature to detect. The metric is inapplicable there and is reported
rather than dropped.

**Decoder-projection MCC-P** replaces the code by the projection of each
observed shift onto the dictionary, $\mathbf{W}_d^{\top}\Delta\mathbf{z}$, and
applies Definition 1 to that. It asks whether the geometry recovers the factor
with the encoder's gating removed, and disagreement between the two versions
localises the failure: a strong projection score against a weak code score puts
the problem in the threshold rather than the dictionary.

**Support precision, recall, and F1** compare the active set of the matched
coordinate, $\{i : \lvert \hat c_{i\pi(k)} \rvert > \tau\}$, with the examples
where factor $k$ changed, at an activation threshold fixed at $\tau = 0.1$ in
the synthetic runner. These are the only recovery metrics here that are *not*
invariant to coordinate-wise scaling, since rescaling a coordinate by ten
changes its support at fixed $\tau$. Both one-sided failures appear in the
catalog: a code that activates almost nothing wins precision cheaply, which is
how SAE ReLU takes support precision in
[experiment 02](../results/identification.md) on 3.9 selected latents per
thousand, and a code with 97% of coordinates active wins recall cheaply, which
is the affine baseline of [experiment 06](../results/synthetic.md).

<div class="finding" markdown="1">
<span class="label">What these metrics found</span>
SSAE variants take <b>131 of 160</b> metric cells in
[experiment 02](../results/identification.md), or 82%. The margins are widest on
the detection metrics: feature ROC-AUC is <b>1.000</b> for SSAE ReLU against
0.859 for SAE ReLU on Gemma <code>masc-fem-eng</code>, and support F1 is
<b>0.832</b> against 0.049 on Pythia <code>eng-french</code>, where the SAE code
activates a support almost unrelated to the concept that changed.
</div>

## Reconstruction and geometry

For targets $y_i$ and reconstructions $\hat y_i$, a raw-activation SAE takes
$y_i = \mathbf{z}_i$ and a difference-input SSAE takes
$y_i = \Delta\mathbf{z}_i$.

| Metric | Definition | Better | Sensitive to |
|---|---|---:|---|
| MSE | $\frac{1}{n}\sum_i \lVert y_i - \hat y_i \rVert_2^2$ | lower | target scale and dimension |
| NMSE | $\sum_i \lVert y_i - \hat y_i \rVert_2^2 \big/ \sum_i \lVert y_i \rVert_2^2$ | lower | small target energy in the denominator |
| cosine | mean angle between $y_i$ and $\hat y_i$ | higher | direction only |

Dividing by target energy makes NMSE invariant to the scale of the activations,
which is what puts cells from different layers and models on one axis; raw MSE
stays in the run table so the normalisation can be audited. A value above one
means the model reconstructs worse than the mean shift. NMSE and cosine fail in
different directions, since a reconstruction can have the right direction at the
wrong length, or acceptable energy with the pair-level directions scrambled.

Every fidelity metric shares one blindness, which is why
[experiment 01](../results/reconstruction.md) settles nothing on its own: under
the mixing $\mathbf{L}$ of [the theory page](../theory/index.md), a model that
blends all concepts reconstructs as well as one that separates them.
[Experiment 06](../results/synthetic.md) shows the pair side by side, with the
unconstrained model reconstructing slightly *better*.

<div class="finding neutral" markdown="1">
<span class="label">Targets are never pooled</span>
A raw-activation SAE reconstructs $\mathbf{z}$ and a difference-input SSAE
reconstructs $\Delta\mathbf{z}$. No model in this catalog is fit to both, so
every paired-difference cell holds only SSAE variants and every raw-activation
cell only SAE variants. Within the difference target SSAE variants take the
lowest NMSE in <b>13 of 13</b> cells; within the raw target SAE JumpReLU takes
it in <b>13 of 13</b>. The two counts describe disjoint contests, and adding
them measures which target is easier to fit.
</div>

The steering claims rest on two further angles. **Endpoint cosine improvement**
subtracts the unsteered endpoint cosine from the steered one on the same
held-out pair. **Direction cosine** compares the steered direction with the
target domain's own difference in means. They disagree in
[experiment 04](../results/transfer.md), where a vector improves an endpoint
without pointing along the concept, so both are reported.

## Sparsity and stability

<div class="defn" markdown="1">
<span class="label">Definition 3 (Thresholded occupancy).</span>
For codes $\hat c_{ij}$ and an activation threshold $\tau$, the mean number of
active coordinates per example is

$$L_0(\tau) \;=\; \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} \mathbf{1}\{\lvert \hat c_{ij} \rvert > \tau\} ,$$

and the active fraction is $L_0(\tau)/m$.
</div>

Occupancy is a count; mean $L_1$ mass, $\frac{1}{nm}\sum_{i,j}\lvert \hat c_{ij}\rvert$,
is a magnitude. A TopK gate fixes $L_0$ by construction while an
$L_1$-constrained model can hold many small non-zero values, so a method
ordering at one $\tau$ can reverse at another. An $L_1$ ceiling of $0.005$ and a
Step-$L_0$ ceiling of $0.005$ constrain different quantities, as
[the estimation page](../theory/estimation.md) sets out, and a configured target
is no evidence the optimiser reached it — constrained runs report the achieved
value, the signed gap, and the dual multiplier beside the ceiling.

Dead-feature fraction counts coordinates that never exceed $\tau$ on a named
evaluation set. For signed ungated codes that is a reporting convention rather
than an absorbing state, since a coordinate below threshold everywhere may still
be receiving gradient.

<div class="defn" markdown="1">
<span class="label">Definition 4 (Cross-seed atom identity).</span>
Two latents from independently seeded runs match when they clear a response
criterion, $\lvert \rho_{\text{resp}} \rvert \ge 0.50$ between their activation
vectors on a shared evaluation set, together with a geometric criterion,
$\lvert \cos_{\text{dec}} \rvert \ge 0.80$ between their decoder columns. A
stable atom is a strict-identity component spanning at least two of three seeds.
</div>

Population standard deviation across seeds describes run-to-run variation and
says nothing about correctness. Atom-level stability needs the matching in
Definition 4, with both criteria.

<div class="finding neutral" markdown="1">
<span class="label">Why both criteria</span>
In [experiment 05](../results/stability.md), JumpReLU features recur in response
across seeds — 33.7% for SSAE, 24.2% for SAE — while <b>zero</b>
concept-conditioned pairs clear the response and decoder-geometry thresholds
together. A feature can fire on the same inputs in two runs with its decoder
column pointing elsewhere. Reporting response agreement alone would have called
these features stable.
</div>

## Reporting rules

1. Compare reconstruction only within one target. Raw SAEs reconstruct
   $\mathbf{z}$ and difference-trained SSAEs reconstruct $\Delta\mathbf{z}$, and
   the cells are populated by disjoint model families.
2. State every invariance the matching used: permutation, sign, coordinate-wise
   scale, monotone transformation.
3. Name the split used for feature selection, for calibration, and for scoring,
   and keep them distinct.
4. Give the threshold with every thresholded quantity. Occupancy, support
   scores, and dead-feature counts are undefined without $\tau$.
5. Show the individual seed values and say what varies across them. Three seeds
   do not define a sampling distribution.
6. Leave unavailable cells empty. Zero is a measurement.
7. Fix the metric before seeing which one favours the conclusion, and report
   metrics that turn out to be inapplicable — the chance-level feature ROC-AUC
   in [experiment 06](../results/synthetic.md) stays in the table.

```bibtex
@inproceedings{hyvarinen2016,
  title     = {Unsupervised Feature Extraction by Time-Contrastive Learning
               and Nonlinear ICA},
  author    = {Hyv{\"a}rinen, Aapo and Morioka, Hiroshi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2016}
}

@inproceedings{eastwood2018,
  title     = {A Framework for the Quantitative Evaluation of Disentangled
               Representations},
  author    = {Eastwood, Cian and Williams, Christopher K. I.},
  booktitle = {International Conference on Learning Representations},
  year      = {2018}
}

@inproceedings{locatello2019challenging,
  title     = {Challenging Common Assumptions in the Unsupervised Learning of
               Disentangled Representations},
  author    = {Locatello, Francesco and Bauer, Stefan and Lucic, Mario and
               R{\"a}tsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf,
               Bernhard and Bachem, Olivier},
  booktitle = {International Conference on Machine Learning},
  year      = {2019}
}

@article{kuhn1955,
  title   = {The Hungarian Method for the Assignment Problem},
  author  = {Kuhn, Harold W.},
  journal = {Naval Research Logistics Quarterly},
  volume  = {2},
  pages   = {83--97},
  year    = {1955}
}
```
