---
title: Theorem to Estimator
description: The population L0 problem is not the objective anyone trains. What the relaxation preserves, what it forfeits, and how to tell which regime a run is in.
---

<p class="kicker">Theory 03 · practice</p>

# Four departures from the theorem

<p class="lede">Proposition 2 describes a population optimum of an
$L_0$-constrained problem at the correct width with exact reconstruction. Every
one of those four qualifiers is violated by a training run. This page states each
departure, what is known about it, and which measurement detects the failure it
causes.</p>

## Four departures

| Theorem | Implementation | Consequence |
|---|---|---|
| population expectation | $N$ finite samples | budget is met on average in-sample, not in the population |
| $L_0$ occupancy | $L_1$ magnitude | the constrained quantity is not the constrained quantity |
| width $m = \lvert V\rvert$ | often $m = d_z$ | the code is overcomplete; uniqueness argument does not apply |
| exact reconstruction | NMSE $> 0$ | the optimum analysed is not the optimum reached |

None of these is fatal on its own and none is repaired by training longer. They
are the reason the empirical claims on this site are stated as measured recovery
rather than as consequences of the theorem.

## The relaxation of $L_0$ to $L_1$

The identification argument uses $L_0$ because the lemma counts non-zeros. An
$L_0$ constraint has zero gradient almost everywhere, so training relaxes it to

$$\frac{1}{mN}\sum_{i=1}^{N} \big\lVert r(\Delta\mathbf{z}_i) \big\rVert_1 \;\le\; \beta .$$

The two constraints coincide only under conditions of the kind classical sparse
recovery invokes, and those conditions are the ones this setting was chosen to
avoid. What the relaxation preserves is the *incentive*: mixing coordinates
raises $L_1$ as well as $L_0$ under most distributions, so the pressure toward an
unmixed basis remains. What it forfeits is the exact equivalence that makes the
lemma bite.

The practical trap is that $\beta$ now carries different units. A target of
$0.005$ under $L_1$ bounds a mean absolute code value; under a thresholded
occupancy statistic it bounds an active fraction. Transferring a target between
the two without recalibrating produces a run that is either unconstrained or
collapsed, and the repository supports three sparsity statistics — `l1`,
`step_l0`, `sigmoid_l0` — that are not interchangeable for this reason.

## Constrained rather than penalised

A penalty $\lambda\lVert r(\Delta\mathbf{z})\rVert_1$ added to the reconstruction
loss is the more common construction, and SSAE avoids it deliberately.

Under a penalty, the achieved sparsity is an outcome of $\lambda$, the data
scale, and the optimisation trajectory jointly. Two runs on different datasets
at the same $\lambda$ sit at different points on the sparsity–fidelity curve,
which makes them incomparable, and the correspondence with the theorem's $\beta$
is lost entirely. Penalties also shrink the coefficients they retain — the
feature-suppression effect — so the recovered magnitudes are biased even when the
support is right.

The constrained form solves the saddle-point problem for the Lagrangian with
ExtraAdam on both primal and dual variables, through the `cooper` library. The
dual variable finds the multiplier that meets the specified budget, so $\beta$
means the same thing across datasets and the achieved value can be audited
against it directly.

<div class="finding" markdown="1">
<span class="label">A measurement this makes possible</span>
Because $\beta$ is specified rather than induced, a run can be checked for
whether its constraint was ever active. A model whose achieved sparsity sits far
below its budget is an unconstrained affine autoencoder in all but name, and only
Proposition 1 applies to it. [Experiment 06](../results/synthetic.md) uses exactly
this contrast as its baseline: the same architecture with the constraint removed
recovers MCC $0.58$–$0.83$ where the constrained model reaches $0.99$–$1.00$.
</div>

## The overcomplete case

Proposition 2 is stated at $m = \lvert V \rvert$. In practice $\lvert V\rvert$ is
unknown for real activations, and the experiments that predict single-concept
shifts set $m = d_z$ — $2304$ for Gemma-2-2B. At that width the theorem's
premises are not met: the code has more coordinates than there are varying
concepts, and the surplus can be filled arbitrarily.

What remains is an inductive bias rather than a guarantee. Shift sparsity still
pushes the model toward an unmixed basis, and the paper reports that the
resulting directions still predict held-out single-concept shifts better than
SAE directions do, with the largest margins on `BINARY` and `CORR`. This should
be read as an empirical finding about a heuristic, not as the theorem operating
at scale.

Choosing $m$ is therefore a modelling decision with a stated cost. At
$m = \lvert V\rvert$ the theory applies and $\lvert V \rvert$ must be assumed. At
$m = d_z$ nothing must be assumed and the theory does not apply.

## Model selection without labels

Selecting hyperparameters by downstream MCC would use the labels the method
claims not to need. The repository instead uses UDR, which ranks a configuration
by the median pairwise MCC among runs at different seeds: a configuration whose
seeds agree with each other is preferred, with no ground truth consulted.

Its assumption is that mixed solutions are less reproducible than unmixed ones,
which is plausible and unproven. Seed agreement is necessary for recovery but not
sufficient, since runs sharing an inductive bias can share a mixture — the point
made on [the overview page](index.md) and measured directly by
[experiment 05](../results/stability.md), where response recurrence across seeds
and decoder-geometry agreement come apart sharply for JumpReLU variants.

## What to measure, given all of this

The gap between theorem and estimator means the guarantee cannot be inherited and
must be checked. Four measurements are needed, and they are separate questions:

1. **Fidelity** — does the model reconstruct held-out shifts of its own target?
   Necessary, and uninformative about identification on its own.
2. **Recovery** — after one-to-one matching, do coordinates track known factors?
   This is the direct empirical analogue of Proposition 2.
3. **Prediction** — does a direction chosen on training data predict an unseen
   shift, and does it survive a change of domain?
4. **Intervention** — does adding the direction move the intended behaviour, in
   both orientations, without off-target damage?

[The results overview](../results/index.md) maps each of the six frozen
experiments onto one of these questions and reports what it found.

## Sources

```bibtex
@article{gidel2020variational,
  title   = {A Variational Inequality Perspective on Generative Adversarial
             Networks},
  author  = {Gidel, Gauthier and Berard, Hugo and Vignoud, Ga{\"e}tan and
             Vincent, Pascal and Lacoste-Julien, Simon},
  journal = {International Conference on Learning Representations},
  year    = {2019}
}

@inproceedings{gallegoPosada2022cooper,
  title  = {Cooper: A Toolkit for Lagrangian-Based Constrained Optimization},
  author = {Gallego-Posada, Jose and Ramirez, Juan},
  year   = {2022}
}

@article{duan2019unsupervised,
  title   = {Unsupervised Model Selection for Variational Disentangled
             Representation Learning},
  author  = {Duan, Sunny and Matthey, Loic and Saraiva, Andre and
             Watters, Nicholas and Burgess, Christopher P. and
             Lerchner, Alexander and Higgins, Irina},
  journal = {arXiv preprint arXiv:1905.12614},
  year    = {2019}
}

@article{gao2024scaling,
  title   = {Scaling and Evaluating Sparse Autoencoders},
  author  = {Gao, Leo and Dupr{\'e} la Tour, Tom and Tillman, Henk and
             Goh, Gabriel and Troll, Rajan and Radford, Alec and
             Sutskever, Ilya and Leike, Jan and Wu, Jeffrey},
  journal = {arXiv preprint arXiv:2406.04093},
  year    = {2024}
}

@article{rajamanoharan2024gated,
  title   = {Improving Dictionary Learning with Gated Sparse Autoencoders},
  author  = {Rajamanoharan, Senthooran and Conmy, Arthur and Smith, Lewis and
             Lieberum, Tom and Varma, Vikrant and Kram{\'a}r, J{\'a}nos and
             Shah, Rohin and Nanda, Neel},
  journal = {arXiv preprint arXiv:2404.16014},
  year    = {2024}
}
```
