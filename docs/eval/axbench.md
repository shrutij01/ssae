# AxBench

<p class="result-lede">
AxBench is an <strong>external concept-detection protocol</strong>. No local
AxBench result is presented on this page; it defines the test that a future
comparison must pass.
</p>

## Testable hypotheses

<div class="hypothesis-sheet">
<p><b>Unit.</b> Held-out concept, with texts nested within concept.</p>
<p><b>Null \(H_0\).</b> Mean held-out AUC for SSAE is no greater than that of a
pre-specified SAE comparator under the same layer, token pooling, and feature
selection protocol.</p>
<p><b>Alternative \(H_1\).</b> SSAE has higher mean held-out AUC.</p>
<p><b>Decision rule.</b> Fix the comparator and feature-selection split before
evaluation; bootstrap or hierarchically model concepts rather than treating
every text as an independent model replicate. Report hard-negative accuracy
and imbalanced AUC as separate estimands.</p>
</div>

[AxBench](https://github.com/stanfordnlp/axbench) is an external benchmark for evaluating SAE concept detection at scale. This page describes how SSAEs are evaluated on AxBench and how the evaluation protocol works.

## Overview

AxBench provides 10 or 500 human-annotated concepts with positive, negative, and hard-negative examples. The evaluation protocol tests whether individual SAE features can **zero-shot detect** these concepts — without any concept-specific training.

Datasets available on HuggingFace:

- `pyvene/axbench-concept10` — 10 concepts (quick test)
- `pyvene/axbench-concept500` — 500 concepts (main benchmark)

## Evaluation Protocol

### 1. Load Concept Data

Each concept comes with:

- **Positives** — texts where the concept is present
- **Negatives** — texts where the concept is absent
- **Hard negatives** — texts that are semantically similar to positives but lack the concept

### 2. Extract Embeddings

For each text, extract per-token residual-stream embeddings at the SAE's layer:

```python
# Per-token embeddings: list of (seq_i, D) tensors
pos_tok_embs = extract_text_embeddings(
    texts, llm_model, tokenizer, layer, pooling="all_tokens"
)
```

### 3. SAE Activation with Max Pooling

Run each token's embedding through the SAE, then **max-pool** across tokens per feature:

$$a_k = \max_{t \in \text{tokens}} h_k^{(t)}$$

This gives one activation score per feature per text. Max pooling means a feature fires if the concept is present *anywhere* in the text.

### 4. Zero-Shot Feature Mapping

For each concept, find the single SAE feature that best separates positives from negatives:

```python
# Score each feature's discriminative power
scores = _score_auc(all_acts, labels)  # (K,) — AUC per feature
best_feature = scores.argmax()
```

Available scoring methods:

| Method | Description |
|--------|-------------|
| `auc` | ROC-AUC (default, sign-invariant) |
| `pearson` | Absolute Pearson correlation |
| `mean_diff` | Absolute mean activation difference |

### 5. Metrics

**AUC-ROC** — how well the best feature's activation separates positives from negatives.

**Hard-Negative Accuracy** — at the optimal threshold (Youden's J), how well can the feature distinguish positives from hard negatives? This tests fine-grained concept understanding.

**Imbalanced AUC-ROC** — AUC-ROC with 100:1 negative-to-positive ratio, simulating real-world sparsity where most inputs don't contain the concept.

## Usage

### Quick Test (10 concepts)

```bash
python -m evals.external.axbench \
    --model path/to/ssae_seed0 \
    --hf-dataset pyvene/axbench-concept10 \
    --llm google/gemma-2-2b-it \
    --layer 20
```

### Full Evaluation (500 concepts)

```bash
python -m evals.external.axbench \
    --model path/to/ssae_seed0 \
    --hf-dataset pyvene/axbench-concept500 \
    --llm google/gemma-2-2b-it \
    --layer 20 \
    --output axbench_results.jsonl
```

### With Pre-Extracted Embeddings

Skip the LLM embedding extraction step (faster for repeated evaluations):

```bash
python -m evals.external.axbench \
    --model path/to/ssae_seed0 \
    --embeddings-dir path/to/axbench_embeddings/ \
    --output axbench_results.jsonl
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to trained SSAE directory |
| `--hf-dataset` | `pyvene/axbench-concept10` | HuggingFace dataset ID |
| `--max-concepts` | all | Limit number of concepts |
| `--llm` | — | HuggingFace model ID for embedding extraction |
| `--layer` | `20` | Layer for embedding extraction |
| `--embeddings-dir` | — | Pre-extracted embeddings (skip LLM) |
| `--mapping-method` | `auc` | Scoring: `auc`, `pearson`, `mean_diff` |
| `--batch-size` | `32` | Batch size for embedding extraction |
| `--output` | — | Output `.jsonl` file |

## Differences from Our Semi-Synthetic Evaluation

| Aspect | AxBench | Semi-Synthetic |
|--------|---------|----------------|
| **Ground truth** | Human-annotated concepts | Controlled counterfactual structure |
| **Concept count** | 10 or 500 | 1-5 (known exactly) |
| **Matching** | Greedy (best feature per concept) | Hungarian (one-to-one) |
| **Training data** | Independent of SAE training | Same distribution as SAE training |
| **Metric** | AUC-ROC | MCC (Pearson + Hungarian) |
| **Token handling** | Max-pool over tokens | Last-token pooling |

AxBench tests generalisation to unseen concepts at scale, while semi-synthetic evaluation provides controlled identifiability analysis. Both are complementary.

## Takeaway

This protocol can test whether shift-trained features detect many externally
annotated concepts. It cannot by itself test coordinate identifiability or
causal specificity, and the absence of results here must not be read as a
benchmark claim.
