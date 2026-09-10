# Semi-synthetic data

## Purpose

The semi-synthetic datasets pair controlled text changes with ground-truth
factor labels. For example, `("chair", "chaise")` changes language while
preserving the underlying object. SSAEs are trained on activation differences
between paired texts and evaluated on whether individual coordinates recover
the known factors.

## Repository layout

```text
data/
├── base.py                    # controlled language and gender vocabularies
├── registry.py                # dataset names, factor labels, and pair plans
├── generator.py               # behavioral/external dataset loaders
├── categorical.py             # categorical contrast generator
├── categorical/               # checked-in train/validation/test contrasts
├── vocab/
│   └── corr_occupations.json  # aligned English/French/German occupations
├── behavioral/
│   ├── README.md              # download and attribution notes
│   └── sycophancy.json        # bundled CC BY 4.0 dataset
├── oodprobe/
│   └── README.md              # placement notes; external CSVs are ignored
└── README.md                  # authoritative source and license inventory
```

The vocabularies used by `eng-french`, `eng-german`, `masc-fem-eng`,
`masc-fem-mixed`, `2-binary`, and related transfer datasets currently live in
`data/base.py`. `data/registry.py` is the source of truth for the exact training
and held-out pair plans.

## Counterfactual construction

For aligned vocabulary lists, item `i` denotes the same underlying entity in
each condition. This permits an observed change such as

```text
English object[i] -> French object[i]       factor: language
masculine role[i] -> feminine role[i]       factor: gender
```

Multi-factor datasets combine these edges and persist explicit factor-change
labels. Train/test splitting occurs before model selection and steering
calibration; the held-out endpoints are not used to orient or scale directions.

The main design constraints are:

1. aligned entries preserve the underlying concept;
2. a single-factor edge changes only its declared factor;
3. lists are deduplicated within each condition; and
4. mixed-factor edges carry explicit multi-hot labels.

## Categorical contrasts

`data/categorical.py` constructs prompts whose color, shape, and object
attributes can change independently. The checked-in files under
`data/categorical/` are the frozen project-generated train, validation, and test
splits; `stats.json` records their codebook and sizes.

## Producing embeddings

Set a data root outside the Git checkout, then invoke the embedding CLI. The
exact available flags are shown by `--help`.

```bash
export SSAE_DATA_ROOT=/path/to/ssae-data

python -m ssae.store_embeddings \
  --dataset eng-french \
  --model_id google/gemma-2-2b-it \
  --layer 25 \
  --pooling-method last_token \
  --num-samples 200
```

Generated activations, checkpoints, and run caches belong under
`SSAE_DATA_ROOT`; they are not repository source data and must not be committed.
Published, compact result bundles belong under `results/`.

## External datasets

The repository MIT License applies only to original project data and code. It
does not relicense external corpora or model weights. Consult
[`data/README.md`](README.md) before downloading, using, or redistributing any
external dataset. In particular, refusal and OOD-probe source files are not
bundled because their upstream repositories do not currently declare terms that
authorize redistribution here.
