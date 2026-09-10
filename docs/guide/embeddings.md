# Extracting Embeddings

SSAEs are trained on LLM residual-stream activations stored in HDF5 files. This page describes how to extract and manage these embeddings.

## Command

```bash
python ssae/store_embeddings.py \
    --model "google/gemma-2-2b-it" \
    --dataset 2-binary \
    --layer 20 \
    --batch-size 32 \
    --output ./embeddings/gemma2b_layer20.h5
```

## Supported Datasets

Datasets are registered in `data/registry.py`. The registry tracks metadata (number of concepts, loader type, vocabulary keys) so that embedding extraction and evaluation scripts can look up properties without scattered if/elif chains.

### Counterfactual Datasets

These generate sentence pairs by swapping concept-bearing tokens:

| Dataset | Concepts | Description |
|---------|----------|-------------|
| `eng-french` | 1 | English vs French single-word translations |
| `eng-german` | 1 | English vs German single-word translations |
| `masc-fem-eng` | 1 | English masculine vs feminine names |
| `masc-fem-mixed` | 1 | English + French masculine vs feminine |
| `2-binary` | 2 | Language (eng/fr) + gender (masc/fem) |
| `corr-binary` | 2 | Correlated: eng-french + eng-german |

### External Datasets

| Dataset | Concepts | Description |
|---------|----------|-------------|
| `sycophancy` | 1 | Sycophantic vs non-sycophantic responses |
| `refusal` | 1 | Refusal vs compliance |
| `safearena` | 1 | Safe vs harmful intents |
| `truthful-qa` | 1 | TruthfulQA correct vs wrong answers |
| `bias-in-bios` | 29 | Uniform arbitrary biography pairs; gender and 28 profession deltas are retained only as evaluation factors |

## HDF5 File Format

```
embeddings.h5
├── cfc_train          # shape: (N, 2, rep_dim) or (N, rep_dim)
├── cfc_test           # shape: (N, 2, rep_dim) or (N, rep_dim)
└── cfc_test_labels    # shape: (N,) — per-pair concept labels (optional)
```

### 3-D Format: `(N, 2, rep_dim)`

Contrastive pairs. Each row contains two embeddings `[z_base, z_counterfactual]`. The SSAE trains on their difference `z_cf - z_base`.

For Bias-in-Bios, endpoint A and endpoint B are sampled uniformly from distinct
biographies in the official source split without conditioning on gender or
profession. Training pairs come from the official train split and held-out
pairs from the official test split. Each HDF5 row stores the endpoint IDs,
text hashes, source split, and the endpoint-B-minus-endpoint-A deltas for gender
and all 28 professions. The labels are never consulted by the pair sampler.

### 2-D Format: `(N, rep_dim)`

Sequential embeddings. The SSAE trains on consecutive differences `z[i+1] - z[i]`.

## Pooling Methods

| Method | Description |
|--------|-------------|
| `last_token` | Use the last non-padding token's hidden state (default) |
| `sum` | Sum all token hidden states |

## Activation Extraction Methods

| Method | Padding | Truncation | Use case |
|--------|---------|------------|----------|
| `ours` | Left-padded | Right-truncated | Default — last token is always the final real token |
| `reference` | Right-padded | Left-truncated | Compatibility with some baselines |
