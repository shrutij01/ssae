# Semi-Synthetic Datasets

## Motivation

Identifiability needs known ground truth, but ordinary LLM activations do not
tell us which dimensions encode language, gender, or another concept.
Semi-synthetic datasets close part of that gap: a real LLM supplies the
activation geometry, while controlled lexical pairs specify which factors
changed.

This gives the evaluation three useful properties:

1. factor changes are known for every pair;
2. activations retain the geometry of the pretrained language model; and
3. train and held-out endpoints can be separated before feature selection,
   direction orientation, or steering calibration.

The construction is semi-synthetic rather than fully synthetic because only
the input contrasts and labels are controlled. The representations themselves
come from Gemma or Pythia.

## Counterfactual pairs

The simplest example is an aligned translation pair:

| Source | Target | Factor changed |
|---|---|---|
| `Chair` | `Chaise` | language |

After extracting activations $z_0$ and $z_1$, the observed shift is
$\Delta z=z_1-z_0$. Because the lexical entries are aligned by underlying
meaning, this is a controlled language contrast rather than a claim that an
arbitrary pair of words differs only by language.

Multi-factor datasets combine aligned language and gender lists. Each pair has
an explicit label such as `{language: true, gender: false}` or
`{language: true, gender: true}`. Those labels are persisted alongside the
activation pairs in HDF5.

## Dataset registry

[`data/registry.py`](https://github.com/shrutij01/ssae/blob/main/data/registry.py)
is the source of truth for dataset
metadata, pair plans, and factor labels:

```python
from data.registry import get_dataset_info

info = get_dataset_info("2-binary")
print(info.num_concepts)       # 2
print(info.is_counterfactual)  # True
print(info.pair_factor_labels)
```

The main semi-synthetic datasets used in the published experiments are:

| Dataset | Controlled factor(s) | Structure |
|---|---|---|
| `eng-french` | language | aligned English/French objects |
| `eng-german` | language | aligned English/German objects |
| `masc-fem-eng` | gender | aligned English masculine/feminine terms |
| `masc-fem-mixed` | gender | mixed English/French pool |
| `2-binary` | language + gender | single- and joint-factor edges |
| `corr-binary` | two language edges | shared English anchor |
| `categorical-contrastive` | color + shape + object | generated prompt contrasts |

The registry also contains explicit semantic-domain and cross-language transfer
variants. Inspect it directly for the current complete list rather than copying
dataset metadata into a second configuration file.

## Pair plans and held-out data

Each counterfactual dataset specifies separate training and test plans. For
example, the `2-binary` training plan includes language-only, gender-only, and
joint edges:

```python
train_plan_template = {
    ("eng_masc", "french_masc"): 1,
    ("eng_fem", "french_fem"): 1,
    ("eng_masc", "eng_fem"): 1,
    ("french_masc", "french_fem"): 1,
    ("eng_masc", "french_fem"): 4,
}
```

The numeric values are sample-count multipliers. Pair generation is
deterministic for its data seed. Test entries are drawn from the held-out part
of each aligned vocabulary; a literal endpoint that also appears in training is
removed even if it occurred at a different list index.

This split boundary is important for the steering studies. Feature selection,
sign orientation, and step-size fitting use training pairs only. The stored
held-out pairs are used only to evaluate the fitted intervention.

## Why train on differences?

Training an SSAE on $\Delta z$ suppresses information shared by both endpoints
and makes the supervised change the reconstruction target. It therefore aligns
the learning problem with the intervention structure assumed by the
identifiability argument.

Raw-activation SAE baselines answer a different reconstruction question. The
result reports retain that distinction and do not put raw-$z$ and
$\Delta z$ reconstruction errors on a shared scale.

## Source-data layout

```text
data/
├── base.py                    # aligned language and gender lists
├── registry.py                # metadata, plans, and factor labels
├── categorical.py             # categorical contrast generator
├── categorical/               # frozen generated splits
├── vocab/corr_occupations.json
├── behavioral/                # licensed or user-supplied behavioral data
└── oodprobe/                  # user-supplied external probe data
```

Most controlled vocabularies currently live in `data/base.py` for checkpoint
compatibility. The occupation transfer vocabulary is a structured JSON file.
There is no public automatic-vocabulary generator in this repository: new
vocabularies must be curated, aligned, reviewed for leakage, and registered
explicitly.

## From pairs to evaluation

Keep generated activations and checkpoints outside the Git checkout:

```bash
export SSAE_DATA_ROOT=/path/to/ssae-data

python -m ssae.store_embeddings \
  --dataset eng-french \
  --model_id google/gemma-2-2b-it \
  --layer 25 \
  --pooling-method last_token \
  --num-samples 100

python -m ssae.training \
  /path/to/embeddings.h5 \
  /path/to/config.yaml \
  --seed 0

python -m experiments.legacy.semi_synth \
  --embeddings /path/to/embeddings.h5 \
  --model /path/to/checkpoint \
  --aggregate-seeds --linear-probe --auc
```

Run `python -m ssae.store_embeddings --help` for the exact extractor options.
Compact frozen inputs and report artifacts belong under `results/`; large HDF5
files, model weights, and run caches do not.

## Limitations

Controlled lexical pairs do not make an activation change perfectly causal or
semantically pure. Tokenization, frequency, morphology, and model-specific
associations can remain as confounds. These datasets support a sharper test of
factor recovery than uncontrolled text, but they do not replace behavioral
generation evaluations.

See the [data source and license inventory](https://github.com/shrutij01/ssae/blob/main/data/README.md)
before adding or redistributing any external dataset.
