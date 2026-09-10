---
title: Where It Fails
description: The assumptions that cannot be tested, the regimes that break them, and the claims the identification result never licenses.
---

<p class="kicker">Theory 04 · scope</p>

# Where the result fails

<p class="lede">A guarantee is only as informative as the conditions attached to
it. This page collects the ways SSAE can be wrong, including the cases where the
frozen experiments on this site show it losing, because a method whose failure
modes are unstated is not a method one can rely on.</p>

## The assumption that carries the most weight

Linearity is the load-bearing assumption and the least verifiable one. It asserts
that at a fixed layer, token position, and pooling rule, the composition
$f \circ g$ is linear in concepts. Evidence for the linear representation
hypothesis is substantial and it is evidence about *particular* concepts in
*particular* models, not a theorem about language models. There are theoretical
arguments for why linear structure should emerge in next-token predictors, and
they too are conditional.

Where it breaks, it breaks quietly. If the true representation is
$\mathbf{z} = h(\mathbf{c})$ with $h$ nonlinear, then
$\Delta\mathbf{z} \approx \mathbf{J}_h(\mathbf{c})\Delta\mathbf{c}$ holds only to
first order and only locally. A decoder column then denotes one concept *in the
neighbourhood where it was fit*, and a Jacobian that varies with context makes
that neighbourhood small. Nothing in the reconstruction error announces this. The
symptom is a direction that works on the data it was selected on and degrades
off-distribution, which is what [experiment 04](../results/transfer.md) is built
to detect.

## Failures the experiments actually exhibit

Three appear in the frozen catalog.

**Sparsity does not guarantee stability.** JumpReLU variants — SSAE and SAE
alike — produce features whose responses recur across seeds ($33.7\%$ and
$24.2\%$ respectively) while having *zero* concept-conditioned pairs that meet
both the response threshold $\lvert r\rvert \ge 0.5$ and the decoder-geometry
threshold $\lvert\cos\rvert \ge 0.8$. Zero stable two-of-three atom identities
follow. A feature can respond the same way across seeds while its decoder column
points elsewhere, and the two properties separate cleanly at these thresholds.
[Experiment 05 →](../results/stability.md)

**Calibration can defeat a correct direction.** In held-out steering, SSAE
constrained-$L_1$ produces the two largest gains in the catalog ($+0.144$,
$+0.118$) and SSAE ReLU produces the two worst ($-0.086$, $-0.151$). The failing
runs carry train-calibrated step sizes two orders of magnitude larger — $\alpha
\approx 210$–$241$ against $8$–$15$. Proposition 2 identifies a direction up to
scale and says nothing about how to pick the scale; that remains an empirical
calibration problem, and it can dominate the result.
[Experiment 03 →](../results/steering.md)

**Small models leave little to recover.** On Pythia-70M the entire OOD transfer
effect is near zero for every method, SSAE included, with the best improvement
$+0.0074$ against a baseline cosine of $0.504$. The paper likewise reports SSAE
losing to the SAE baselines on Pythia Sycophancy ($0.43$ against $0.49$). A
$512$-dimensional layer-5 residual stream may simply not represent the concepts
being probed, in which case identifiability of the shift model is beside the
point. [Experiment 04 →](../results/transfer.md)

## What the guarantee never supplies

**A name.** Proposition 2 fixes the equivalence class up to permutation, and the
permutation is exactly the map from coordinate index to concept. Labelling
requires held-out annotated data. A recovered coordinate is anonymous by
construction.

**A sign or a unit.** The diagonal $\mathbf{D}$ is unrestricted apart from
invertibility, so orientation and magnitude must be calibrated. This is the
mechanism behind the steering failure above.

**A test of its own premises.** No statistic computed from $\Delta\mathbf{z}$
verifies linearity, injectivity, or support variability. Diagnostics can raise
alarms — a near-singular shift covariance impugns Assumption 3 — but they cannot
confirm.

**Anything outside the observed support.** Proposition 1 explicitly restricts to
$\operatorname{Im}(\mathbf{A}_V)$ because the encoder is unconstrained elsewhere.
Concepts that never vary in the training pairs are not in $V$, are not
recoverable, and are not detectable as missing.

**A claim of universal superiority.** The paper states this directly: it does not
claim SSAEs outperform SAEs universally. The contribution is a practically
relevant observation model under which existing identifiability results apply,
together with evidence that the guarantee has measurable consequences. The
frozen catalog on this site includes the cells where SSAE loses for the same
reason.

## Open questions

**Does the $L_1$ relaxation preserve the argument?** The lemma requires $L_0$.
Conditions under which the relaxation inherits the conclusion in this setting are
not established, and the honest position is that $L_1$ supplies an inductive bias
whose fixed points have not been characterised.

**What is $\lvert V\rvert$ for real activations?** The theorem needs it and no
estimator provides it. The overcomplete runs at $m = d_z$ sidestep the question
at the cost of the guarantee.

**Which layer, and why?** A sweep over Llama-3.1-8B finds cosine similarities
peaking at the last layer, falling, then rising again around layers 13–15. No
account of this structure is available, and layer choice is currently empirical.

**When does subtraction remove a nuisance rather than a concept?** Pairing is
assumed to isolate an intended change. In unpaired settings — the Bias-in-Bios
study pairs two biographies at random — the number of varying concepts per pair
is uncontrolled and possibly large. MCC there is $0.82$, below the naturally
paired benchmarks, and how the guarantee degrades as $\lvert S\rvert$ grows is
not characterised.

## Sources

```bibtex
@article{park2023linear,
  title   = {The Linear Representation Hypothesis and the Geometry of Large
             Language Models},
  author  = {Park, Kiho and Choe, Yo Joong and Veitch, Victor},
  journal = {arXiv preprint arXiv:2311.03658},
  year    = {2023}
}

@article{jiang2024origins,
  title   = {On the Origins of Linear Representations in Large Language Models},
  author  = {Jiang, Yibo and Rajendran, Goutham and Ravikumar, Pradeep and
             Aragam, Bryon and Veitch, Victor},
  journal = {arXiv preprint arXiv:2403.03867},
  year    = {2024}
}

@article{marconato2024all,
  title   = {All or None: Identifiable Linear Properties of Next-Token
             Predictors in Language Modeling},
  author  = {Marconato, Emanuele and Lachapelle, S{\'e}bastien and
             Weichwald, Sebastian and Gresele, Luigi},
  journal = {arXiv preprint arXiv:2410.23501},
  year    = {2024}
}

@article{wu2025axbench,
  title   = {AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse
             Autoencoders},
  author  = {Wu, Zhengxuan and Arora, Aryaman and Geiger, Atticus and
             Wang, Zheng and Huang, Jing and Jurafsky, Dan and
             Manning, Christopher D. and Potts, Christopher},
  journal = {arXiv preprint arXiv:2501.17148},
  year    = {2025}
}

@article{templeton2024scaling,
  title   = {Scaling Monosemanticity: Extracting Interpretable Features from
             Claude 3 Sonnet},
  author  = {Templeton, Adly and Conerly, Tom and Marcus, Jonathan and
             Lindsey, Jack and Bricken, Trenton and Chen, Brian and
             Pearce, Adam and Citro, Craig and Ameisen, Emmanuel and
             Jones, Andy and Cunningham, Hoagy and Turner, Nicholas L. and
             McDougall, Callum and MacDiarmid, Monte and Tamkin, Alex and
             Durmus, Esin and Hume, Tristan and Mosconi, Francesco and
             Freeman, C. Daniel and Sumers, Theodore R. and Rees, Edward and
             Batson, Joshua and Jermyn, Adam and Carter, Shan and Olah, Chris
             and Henighan, Tom},
  journal = {Transformer Circuits Thread},
  year    = {2024}
}
```
