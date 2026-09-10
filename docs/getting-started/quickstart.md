# Quick Start

This guide walks through the three steps to train and evaluate an SSAE: extract embeddings, train the model, and evaluate identifiability.

## 1. Extract Embeddings

Extract residual-stream activations from a language model at a chosen layer:

```bash
python ssae/store_embeddings.py \
    --model "EleutherAI/pythia-70m" \
    --dataset 2-binary \
    --layer 5 \
    --output ./embeddings/pythia70m_layer5.h5
```

This produces an HDF5 file containing paired embeddings in shape `(N, 2, rep_dim)`.

## 2. Create a Config

Create a YAML file describing the LLM:

```yaml
# configs/pythia70m.yaml
model: "EleutherAI/pythia-70m"
rep_dim: 512
layer: 5
num_concepts: 2
```

## 3. Train an SSAE

```bash
python -m ssae.training \
    ./embeddings/pythia70m_layer5.h5 \
    ./configs/pythia70m.yaml \
    --oc 512 \
    --target 0.1 \
    --epochs 15000 \
    --lr 5e-4 \
    --constrained-optim extrapolation \
    --seed 0
```

Key flags:

- `--target 0.1` — exactly 10% of hidden units active (hard constraint)
- `--constrained-optim extrapolation` — most stable primal-dual scheme
- `--oc 512` — hidden dimension matches embedding dimension

The trained model is saved to `./embeddings/run_out/`.

## 4. Evaluate Identifiability

```bash
python -m experiments.legacy.semi_synth \
    --embeddings ./embeddings/pythia70m_layer5.h5 \
    --model ./embeddings/run_out/ssae_*_seed0 \
    --dataset 2-binary \
    --linear-probe \
    --auc
```

This reports:

- **SAE MCC** — how well individual SAE features align with ground-truth concepts
- **Probe MCC** — how well linear probes on raw LLM embeddings recover concepts (baseline)
- **SAE AUC / Probe AUC** — ROC-AUC variants of the above
- **Decoder MCC** — projection onto decoder columns

## Next Steps

- [Training SSAEs](../guide/training.md) — full configuration reference
- [Constrained Optimisation](../theory/constrained-optimisation.md) — understand the core algorithm
- [Evidence](../eval/overview.md) — one empirical test and inference boundary per page
- [Model API](../api/model.md) — configure SSAE and SAE comparators
