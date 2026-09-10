---
title: Reconstruction
description: What the reconstruction study shows about SSAE shift models and their retained signal.
---

# Reconstruction

The reconstruction study asks whether SSAEs retain enough information about paired activation changes to reproduce their training target.

## Outcome

The frozen bundle contains 516 source runs grouped into 39 separately scaled target cells.

- Among the behavioral and semi-synthetic paired-difference cells, SSAE JumpReLU has the lowest NMSE in 17 of 18 cells.
- In the corresponding raw-activation cells, SAE JumpReLU has the lowest NMSE in all 18 cells.
- In the three synthetic difference cells, the unconstrained affine control reconstructs most accurately, while the constrained SSAE achieves substantially better factor recovery.

## Implication

SSAEs can preserve the activation-change signal while learning a sparse shift representation. The synthetic result also separates reconstruction from identification: nearly perfect difference reconstruction is possible without recovering the planted axes, whereas the constrained SSAE accepts a small reconstruction cost and recovers those axes much more accurately.

The useful SSAE property is therefore not reconstruction alone. It is reconstruction combined with a sparse coordinate system for changes.

## Scope

SSAEs reconstruct paired differences and SAEs reconstruct raw activations. Their NMSE values answer different questions and are not ranked against each other. Comparisons are made only within a fixed dataset, language model, input regime, and reconstruction target.

## Supporting Material

- [Reconstruction figure](../results/assets/exp01/figures/semisynthetic_reconstruction.svg)
- [Summary table](../results/assets/exp01/tables/reconstruction_summary.csv)
- [Run-level table](../results/assets/exp01/tables/reconstruction_runs.csv)
- [Artifact manifest](../results/assets/exp01/artifact_manifest.json)

## Reproduce

~~~bash
python3 -m experiments.exp01.report
~~~
