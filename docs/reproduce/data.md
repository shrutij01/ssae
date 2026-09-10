---
title: Paired Activations
description: Extracting activation pairs, the HDF5 schema, the dataset registry, and why pair construction is part of the statistical model.
---

<p class="kicker">Reproduce · data</p>

# Building the pairs

<p class="lede">Extraction fixes the model, the layer, the token-position rule,
the pair orientation, and the train/test split. Each of those is a modelling
choice that the identification argument depends on, so extraction settings belong
in the experiment specification rather than in a preprocessing appendix.</p>

## Extract

```bash
export SSAE_DATA_ROOT=/absolute/path/to/ssae-data

python3 -m ssae.store_embeddings \
  --dataset 2-binary \
  --model_id google/gemma-2-2b-it \
  --layer 25 \
  --num-samples 100 \
  --split 0.9 \
  --pooling-method last_token \
  --batch-size 32
```

`--model_id` takes an underscore; every other multi-word flag takes hyphens.
`--help` is authoritative. Accepted models are `EleutherAI/pythia-70m-deduped`,
`google/gemma-2-2b-it`, `google/gemma-3-1b-pt`, `meta-llama/Llama-3.1-8B`, and
its instruct variant, subject to Hugging Face access.

There is no output-path flag. The extractor writes two files under the data root
and moves any existing pair to timestamped backups first:

```text
$SSAE_DATA_ROOT/<dataset>/<dataset>_<model-tag>_<layer>_<pooling>.h5
$SSAE_DATA_ROOT/<dataset>/<dataset>_<model-tag>_<layer>_<pooling>.yaml
```

The YAML sidecar records dataset, model, `rep_dim`, `llm_layer`,
`num_concepts`, and pooling rule. The trainer merges it with CLI flags and stores
the resolved configuration in the run directory, which is why the representation
dimension should never be entered by hand.

## Schema

```text
cfc_train          (N_train, 2, rep_dim)   paired train activations
cfc_test           (N_test,  2, rep_dim)   paired test activations
cfc_train_labels   serialized pair-factor labels, where available
cfc_test_labels    serialized pair-factor labels, where available
```

For a stored pair $[\mathbf{z}, \tilde{\mathbf{z}}]$, training uses the oriented
difference $\Delta\mathbf{z} = \tilde{\mathbf{z}} - \mathbf{z}$. Reversing the
stored order negates the target, so orientation must agree across extraction,
evaluation, and steering — a sign error here inverts every steering result
without changing any reconstruction number.

Counterfactual datasets also retain the paired text that produced the
activations, which is what makes the [feature atlas](../atlas/index.md) able to
show examples. Some non-counterfactual loaders store different shapes and may
carry no labels; inspect the file rather than assuming.

## Pooling

| Flag | Choice | Behaviour |
|---|---|---|
| `--pooling-method` | `last_token` | final non-padding token (default) |
| | `sum` | sum of hidden states across tokens |
| `--activation-method` | `ours` | left padding, final stored position (default) |
| | `reference` | right padding, last actual token |

With `last_token` and `--max-seq-len`, truncation is from the left so the final
token survives. Every method in a comparison must share the layer and pooling
rule; they define $f$, and a comparison across different $f$ is not a comparison.

## Dataset registry

`data/registry.py` is the source of truth for names, concept counts, loaders, and
counterfactual sampling plans.

```bash
python3 - <<'PY'
from data.registry import DATASETS
for name, info in sorted(DATASETS.items()):
    print(f"{name:28} concepts={info.num_concepts} loader={info.loader}")
PY
```

The controlled semi-synthetic sets are `eng-french`, `eng-german`,
`masc-fem-eng`, `masc-fem-mixed`, `2-binary`, `corr-binary`, and
`corr-binary-occupation`. Behavioural, categorical, and OOD-probe datasets are
registered alongside them.

<div class="finding neutral" markdown="1">
<span class="label">Why <code>corr-binary-occupation</code> exists</span>
The original <code>corr-binary</code> shares its household-object vocabulary with
<code>eng-french</code>. An audit at cosine threshold 0.9999 found 100 of 100
source rows matching a target row exactly, so transfer measured on that pair is
not transfer. The occupation variant was built with a disjoint vocabulary, and
[experiment 04](../results/transfer.md) reports both — the contaminated one
labelled as a diagnostic rather than deleted.
</div>

## Designing a pair set

The identification result needs the change patterns to be diverse, in the precise
sense of [Assumption 4](../theory/conditions.md): for every varying concept $k$,
the pairs that leave $k$ fixed must together exercise every other varying
concept. Two practical consequences follow.

**Concepts that always move together are not separable.** No estimator can
distinguish them, and no amount of data helps. If a design has this property, the
recovered "concept" will be their combination and will behave like one under
steering.

**Single-concept pairs are not required.** The paper's contribution over earlier
weakly supervised results is precisely that multi-concept shifts suffice, as long
as the supports vary. A design with $\mathcal{S} = \{\{1,2\},\{2,3\},\{1,3\}\}$
satisfies the condition without ever changing one concept alone.

Before publishing a pair set, record what changed and what else changed with it:

- how each member was produced, and the concepts intended to differ;
- co-varying nuisances — length, syntax, register, vocabulary domain;
- the model, layer, token position, and pooling rule defining $f$;
- pair orientation and whether it is fixed;
- train/test pair counts and label coverage;
- a hash of the HDF5 source.

Subtraction cancels a component only when it is identical in both members. A
label stored beside a pair records what the designer intended, not what the
activation difference contains.
