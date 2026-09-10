---
title: Why Identifiability
description: Reconstruction fixes the shift subspace and leaves its basis free. Sparsity on shifts fixes the basis, up to permutation and scaling.
---

<p class="kicker">Theory · the question</p>

# Reconstruction fixes a subspace, sparsity fixes its basis

<p class="lede">An interpretability method is useful when one learned coordinate
corresponds to one concept. Reconstruction error leaves that correspondence
undetermined, and so does agreement across seeds. Identifiability is the property
that determines it, and it constrains the model and the data distribution rather
than the quality of the fit.</p>

<div class="defn" markdown="1">
<span class="label">Definition (Identifiability up to a class $\mathcal{T}$).</span>
A model is identifiable up to a set of transformations $\mathcal{T}$ when any two
parameter settings that induce the same observed distribution differ by an
element of $\mathcal{T}$. The set $\mathcal{T}$ records the ambiguities that no
amount of data can resolve.
</div>

For SSAE the target class is permutations composed with coordinate-wise
scalings. A permutation changes which index carries a concept; a scaling changes
its sign and unit. Neither blends two concepts together, which is exactly the
ambiguity that would ruin an intervention.

## The ambiguity reconstruction leaves

Suppose the activation shifts follow the linear model
$\Delta\mathbf{z} = \mathbf{A}_V\Delta\mathbf{c}_V$. For any invertible
$\mathbf{L} \in \mathbb{R}^{\lvert V\rvert \times \lvert V\rvert}$,

$$\Delta\mathbf{z} \;=\; \mathbf{A}_V \Delta\mathbf{c}_V \;=\; (\mathbf{A}_V \mathbf{L})\,(\mathbf{L}^{-1}\Delta\mathbf{c}_V) ,$$

and the two factorisations are indistinguishable from the data: they produce the
same shifts, the same reconstruction error, the same likelihood. A learner that
minimises reconstruction has no basis on which to prefer one.

Write out what the second factorisation does to a single decoder column. The
direction the model associates with coordinate $k$ is

$$\hat q(\mathbf{e}_k) \;=\; \mathbf{A}_V \mathbf{L}\mathbf{e}_k \;=\; \sum_{j=1}^{\lvert V\rvert} L_{jk}\, \mathbf{A}_V \mathbf{e}_j ,$$

a weighted sum of true concept directions. Unless column $k$ of $\mathbf{L}$ has
a single non-zero entry, adding $\hat q(\mathbf{e}_k)$ to an activation changes
several concepts in fixed proportion. The reconstruction error remains zero
throughout. A feature can be perfectly reconstructive, perfectly reproducible,
and still be the wrong object to intervene on.

## Two diagnostics that do not settle the question

**Low reconstruction error.** By the display above, it is invariant to
$\mathbf{L}$. It measures whether the representation retains the information in
$\Delta\mathbf{z}$, which every identifiable solution does and so does every
mixed one.

**Agreement across seeds.** Two runs that converge to the same mixed solution
agree perfectly. Stability is a necessary condition — a coordinate that moves
between seeds cannot be recovering a fixed concept — but a shared inductive bias
is enough to produce it without recovery. This is why
[experiment 05](../results/stability.md) treats cross-seed identity as a separate
criterion from concept selection, and why the two disagree in practice.

## What sparsity buys

Sparsity breaks the tie because it is not invariant to $\mathbf{L}$. Mixing
coordinates generically increases the number of non-zeros: if two concepts
change in different pairs, a coordinate that combines them is active whenever
*either* changes. Under conditions on how change patterns vary across pairs, the
only invertible transformations that leave expected sparsity unincreased are
permutation-scalings.

The resulting equivalence class is
$\hat q = \mathbf{A}_V\mathbf{D}\mathbf{P}$, with $\mathbf{P}$ a permutation and
$\mathbf{D}$ diagonal and invertible. Each decoder column then tracks one column
of $\mathbf{A}_V$.

<div class="finding" markdown="1">
<span class="label">What identifiability adds</span>
Reconstruction asks whether a representation preserves the data. Identifiability
asks whether its coordinates recover the factors we intended to interpret. Only
the second licenses reading a decoder column as a concept, and only the second
predicts that the column keeps its meaning on data the model was not fit to.
</div>

## Why the sparsity has to be on shifts

A standard sparse autoencoder also imposes sparsity, on the code for
$\mathbf{z}$ itself, and the same argument would apply if the premise held. It
does not. Sparsity on $\mathbf{c}$ asserts that few concepts are *present* per
input; sparsity on $\Delta\mathbf{c}$ asserts that few concepts *change* per
pair. The second is strictly weaker, and it is the one language model
activations satisfy.

The gap between them is widest exactly where interpretability is hardest. When
two concepts are correlated, an activation-level sparse code has an incentive to
merge them into one atom, because the merged atom is active less often than
either would be separately. Classical dictionary-learning guarantees exclude
this case by assumption, through incoherence or restricted isometry conditions
that require approximately uncorrelated dictionary columns. Shift sparsity does
not need those conditions: correlated concepts still change in distinguishable
combinations, which is all the argument uses.

## The argument in four steps

1. **[What is identified](identification.md)** - Linear identifiability from reconstruction, followed by permutation-scaling from sparsity.
2. **[When it holds](conditions.md)** - Injectivity, spanning shifts, and the diversity of change patterns required by the proof.
3. **[From theorem to estimator](estimation.md)** - How the trainable relaxation relates to the population \(L_0\) problem.
4. **[Scope](limits.md)** - The assumptions and regimes that define where the result applies.

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

@inproceedings{xu2024sparsity,
  title     = {A Sparsity Principle for Partially Observable Causal
               Representation Learning},
  author    = {Xu, Danru and Yao, Dingling and Lachapelle, S{\'e}bastien and
               Taslakian, Perouz and von K{\"u}gelgen, Julius and
               Locatello, Francesco and Magliacane, Sara},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}

@article{locatello2019challenging,
  title   = {Challenging Common Assumptions in the Unsupervised Learning of
             Disentangled Representations},
  author  = {Locatello, Francesco and Bauer, Stefan and Lucic, Mario and
             R{\"a}tsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard
             and Bachem, Olivier},
  journal = {arXiv preprint arXiv:1811.12359},
  year    = {2019}
}

@article{oneill2025amortisation,
  title   = {Compute Optimal Inference and Provable Amortisation Gap in
             Sparse Autoencoders},
  author  = {O'Neill, Charles and Gumran, Alim and Klindt, David},
  journal = {arXiv preprint arXiv:2411.13117},
  year    = {2025}
}
```
