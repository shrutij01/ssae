---
title: SAEBench Protocol
description: Run SSAEs and matched SAE baselines through the external SAEBench evaluation suite.
---

<p class="note-kicker">Evidence · external protocol</p>

# SAEBench: broad behavior, fixed adapter

<p class="result-lede">
The repository includes an SSAE-to-SAEBench adapter and runner. This page
defines a <strong>prospective benchmark test</strong>; it does not claim a
SAEBench win without exported result files.
</p>

## Testable hypotheses

<div class="hypothesis-sheet">
<p><b>Unit.</b> Benchmark task item, nested within evaluation family and model
checkpoint.</p>
<p><b>Primary null \(H_0\).</b> Under matched model, layer, dictionary width,
and activation budget, SSAE has no improvement over the pre-specified SAE on
the named SAEBench primary score.</p>
<p><b>Alternative \(H_1\).</b> SSAE improves that score.</p>
<p><b>Multiplicity rule.</b> `core`, `sparse_probing`, `absorption`, `scr`, and
`tpp` are different tests. Select a primary family in advance or correct the
family-wise analysis; do not choose a headline after inspecting all outputs.</p>
</div>

## What the adapter preserves

`SSAEForSAEBench` exposes `encode`, `decode`, `forward`, `W_enc`, `W_dec`,
`b_enc`, `b_dec`, and the configuration fields expected by SAEBench. Decoder
rows are normalized while the encoder weights and biases are rescaled to
preserve the encode–decode relationship.

The adapter can load either an SSAE checkpoint or an additional trained SAE
baseline. This keeps the external runner identical across the compared models.

## Run the default test

```bash
python -m experiments.legacy.benchmarks saebench \
  --model path/to/ssae_seed0 \
  --trained-sae path/to/matched_sae_seed0 \
  --model-name gemma-2-2b \
  --hook-layer 20 \
  --eval-types core sparse_probing \
  --output-dir results/findings/saebench/runs
```

Available runner families include `core`, `sparse_probing`, `absorption`,
`scr`, and `tpp`. Autointerpretability evaluations additionally require the
explicit API key option.

## Fair-comparison checklist

- Match transformer model, hook layer, context size, and evaluation corpus.
- State dictionary width and training-token budget.
- Match or stratify by achieved sparsity; architecture labels alone do not
  control activation budget.
- Keep raw-input and difference-input checkpoints distinct in any
  reconstruction analysis.
- Run the same seed set and expose individual seed results.
- Preserve every raw SAEBench output file in the artifact record.

## Takeaway

SAEBench broadens the evaluation beyond the project-specific metrics, but it
does not replace the controlled counterfactual tests. Until result artifacts
are exported, this page documents an executable null hypothesis—not an
observed effect.
