# Sparse Shift Autoencoders Documentation

We developed Sparse Shift Autoencoders (SSAEs) to learn sparse concept directions from controlled changes in paired language-model activations.

## Overview

An SSAE learns from the difference between two related activations. Shared properties cancel in the pair, while the factors that changed receive a direct sparse training signal. The method is built around three core components:

1. **[Paired representation changes](guide/method.md)** - Learn from controlled differences instead of requiring every concept in one activation to be sparse.
2. **[Constrained shift sparsity](theory/constrained-optimisation.md)** - Set an explicit sparsity ceiling and adapt its multiplier during training.
3. **[Shared decoder directions](tutorials/steering.md)** - Reconstruct both members of a pair in one representation space and use the learned directions for concept recovery and steering.

## Why SSAEs?

Compared with a sparse autoencoder trained on raw activations, an SSAE:

- focuses capacity on factors that vary within the paired data;
- uses controlled differences as supervision without requiring feature labels;
- makes the desired shift budget explicit and comparable across runs; and
- retains decoder directions in the language model's activation space.

## Current Results

The repository currently reports seven experiment families:

1. **[Reconstruction](evidence/reconstruction.md)** - 516 source runs across 39 separately scaled target cells. Each loss is reported against its actual reconstruction target rather than ranked across incompatible scales.
2. **[Concept identification](notes/evaluating-identifiability.md)** - The best SSAE family leads the best SAE family in 101 of 132 reported Gemma and Pythia cells; SAEs lead in 27 and four are tied.
3. **[Held-out steering](evidence/heldout-steering.md)** - A complete 270-checkpoint evaluation reports all 90 dataset × LLM × method cells. Adding and subtracting each train-selected direction from the same source tests the positive > baseline > negative ordering, with Gemma and Pythia kept separate.
4. **[Out-of-distribution transfer](evidence/ood-transfer.md)** - SSAEs lead in five of six source-regime × language-model cells. On the clean lexically disjoint comparison, the relative margins are +0.0071 for Gemma and +0.0003 for Pythia.
5. **[Feature stability](evidence/feature-stability.md)** - The atlas contains 2,468 stable components, including 2,387 for SSAE ReLU. JumpReLU response recurrence is 33.7% for SSAEs and 24.2% for SAEs; neither family produces strict two-of-three recurring identities.
6. **[Synthetic identification](evidence/synthetic-identification.md)** - Constrained SSAEs lead the affine control on all three dataset means and on 14 of 15 paired seeds, with a mean paired gain of +0.256.
7. **[Target-side steering overshoot](evidence/target-overshoot.md)** - At $2\alpha$, all 18 model × LLM cells retain at least their baseline French classification, but all 18 preserve within-French geometry less well than their matched-random mean. Stronger language evidence is therefore not equivalent to target preservation.

Counts refer to the frozen result bundles linked from the result pages. Separately scaled metrics are not pooled.

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Complete Results](results/index.md)
- [Repository Structure](STRUCTURE.md)
- [Design Notes](design-notes/index.md)
- [Reproduction Guide](reproduce/index.md)
- [API Reference](api/model.md)
- [GitHub Repository](https://github.com/shrutij01/ssae)

## Quick Example

Train a constrained SSAE from a file of paired activations:

~~~bash
python -m ssae.training \
  embeddings.h5 \
  config.yaml \
  --oc 512 \
  --target 0.1 \
  --lr 5e-4 \
  --constrained-optim extrapolation
~~~

See the [Quick Start](getting-started/quickstart.md) for activation extraction, configuration, and evaluation.

## Citation

If you use SSAEs in your research, please cite:

~~~bibtex
@misc{joshi2026sparseshiftautoencodersidentifying,
      title={Sparse Shift Autoencoders for Identifying Concepts from Large Language Model Activations},
      author={Shruti Joshi and Andrea Dittadi and Sébastien Lachapelle and Dhanya Sridhar},
      year={2026},
      eprint={2502.12179},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12179},
}
~~~

See the [paper](https://arxiv.org/abs/2502.12179) for the full method and analysis, or [download the BibTeX file](assets/ssae.bib).
