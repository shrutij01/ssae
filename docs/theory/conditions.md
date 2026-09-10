---
title: When It Holds
description: The data conditions behind the identification result, what violates them, and how pair design controls whether they are satisfied.
---

<p class="kicker">Theory 02 · data</p>

# The conditions are about the pairs

<p class="lede">Proposition 2 depends on four assumptions. One concerns the
language model, three concern the distribution of pairs. The three that concern
the pairs are under the experimenter's control, which makes pair design the part
of an SSAE study where identification is won or lost.</p>

## The four conditions and what each rules out

| Assumption | Statement | Fails when |
|---|---|---|
| 1 · Linear representation | $f \circ g$ is linear, $\mathbf{z} = \mathbf{A}\mathbf{c}$ | the chosen layer encodes concepts nonlinearly or context-dependently |
| 2 · Injectivity | $\mathbf{A}_V$ has full column rank | more concepts vary than the activation has dimensions, or two concepts share a direction |
| 3 · Sufficient variability | $\operatorname{span}(\Delta^c_V) = \mathbb{R}^{\lvert V\rvert}$ | some direction of concept-shift space is never observed |
| 4 · Sufficient diversity | supports vary, and shift magnitudes are continuous | two concepts always change together, or shifts take finitely many values |

Assumptions 3 and 4 are frequently conflated and they are different
requirements. Assumption 3 is about *directions*: the shifts must span the space,
which an unconstrained affine model needs in order to see all of it. Assumption 4
is about *patterns of co-occurrence*: the shifts must arrive in enough different
subsets that no coordinate can hide inside another. A distribution can span
$\mathbb{R}^{\lvert V\rvert}$ while always changing both concepts at once, in
which case Proposition 1 holds and Proposition 2 does not.

## Reading the support-variability condition

The condition is

$$\bigcup_{S \in \mathcal{S}\,:\,k \notin S} S \;=\; V \setminus \{k\} \qquad \text{for every } k \in V,$$

and the quickest way to understand it is to build the smallest design that
satisfies it and the smallest that does not.

**A design that fails.** Two concepts, language and formality, that always change
together: $\mathcal{S} = \{\{1,2\}\}$. Fix $k = 1$. There is no supported set
excluding concept 1, so the union on the left is empty while the right-hand side
is $\{2\}$. The condition fails, and it should: nothing in the data distinguishes
"language" from "language and formality moving in lockstep". A single decoder
column carrying both is exactly as sparse as the truth.

**A design that succeeds.** Three concepts with
$\mathcal{S} = \{\{1\},\{2\},\{3\},\{1,2\},\{2,3\}\}$. For $k=1$, the sets
avoiding it are $\{2\},\{3\},\{2,3\}$, whose union is $\{2,3\}$, as required. The
same check passes for $k=2$ and $k=3$. Note that singletons are not necessary —
the condition is satisfied by overlapping multi-concept shifts, which is the
point of the paper's generalisation. Earlier weakly supervised results assumed
pairs differing in a single factor.

<div class="finding" markdown="1">
<span class="label">Design consequence</span>
A dataset in which every pair changes the same set of concepts identifies
nothing beyond a subspace, no matter how many pairs it contains. Variety in
*which* concepts change is a separate resource from the number of observations,
and only the first buys identification.
</div>

## Correlated concepts are permitted

The condition constrains supports, not values. Two concepts may be strongly
correlated in magnitude — anti-correlated, even — and the condition still holds
as long as they sometimes change apart. This is the formal reason SSAE survives
the regime that breaks sparse coding.

Classical dictionary-learning identifiability rests on geometric conditions on
the dictionary: restricted isometry, or incoherence, both of which demand
approximately uncorrelated columns. Language model concepts are correlated by
construction, so those conditions fail and the guarantees resting on them do
not apply. Assumption 4 constrains a combinatorial property instead — which
concepts co-vary — and correlation in magnitude leaves it intact.

The paper's `CORR` dataset is built to make the distinction visible: two
anti-correlated translation directions, where SAE baselines fall to MCC
$\approx 0.50$ and SSAE holds at $0.90$. The empirical separation is reported on
[the identification experiment page](../results/identification.md).

## What pair construction has to specify

Subtraction cancels a component only when it is identical in both members of the
pair. Everything else survives into $\Delta\mathbf{z}$ and is treated by the
estimator as a concept that changed. A pair design is therefore a statistical
claim, and reporting it means stating

- how the two members were produced, and which concepts were intended to change;
- what else co-varies with the intended change — length, syntax, register,
  vocabulary domain;
- the model, layer, token position, and pooling rule that define $f$;
- whether pair orientation is fixed, since $\Delta\mathbf{z}$ and
  $-\Delta\mathbf{z}$ carry the same information but not the same sign
  convention.

The fourth item is not pedantry. [Experiment 04](../results/transfer.md) contains
a regime, `corr-binary → eng-french`, whose source and target vocabularies
overlap: an audit found 100 of 100 source rows matching a target row at cosine
$1.0000$. It is reported as a diagnostic rather than a transfer result for that
reason, alongside a clean lexically disjoint regime built to replace it.

## Checking what can be checked

None of these assumptions is testable from $\Delta\mathbf{z}$ alone — that is
what makes them assumptions. Three things can nonetheless be monitored, and the
repository tracks each.

**Effective rank of the observed shifts** bears on Assumption 3. If the empirical
shift covariance is nearly singular, the span condition is in doubt and the
excess directions are unconstrained. `ssae.tracking.DiversityTracker` reports it
during training.

**Achieved sparsity against the configured budget** bears on whether the
constraint is active at all. A run that never approaches its budget is
effectively unconstrained, and Proposition 1 is the only result that applies to
it.

**Recovery on a planted mechanism** bears on the whole chain, by construction.
[Experiment 06](../results/synthetic.md) generates data satisfying every
assumption exactly and asks whether the estimator recovers the planted factors;
it reaches MCC $0.995$–$1.000$ against $0.58$–$0.83$ for the unconstrained
affine model. Success there does not certify the assumptions on real
activations. Failure there would refute the estimator regardless of them, which
is why the experiment is run.

## Sources

```bibtex
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

@article{brehmer2022weakly,
  title   = {Weakly Supervised Causal Representation Learning},
  author  = {Brehmer, Johann and de Haan, Pim and Lippe, Phillip and
             Cohen, Taco},
  journal = {arXiv preprint arXiv:2203.16437},
  year    = {2022}
}

@article{donoho2003optimally,
  title   = {Optimally Sparse Representation in General (Nonorthogonal)
             Dictionaries via $\ell^1$ Minimization},
  author  = {Donoho, David L. and Elad, Michael},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {100},
  pages   = {2197--2202},
  year    = {2003}
}
```
