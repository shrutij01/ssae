---
title: Reconstruction & Geometry Metrics
description: How to interpret MSE, NMSE, R², cosine, and dictionary geometry for raw and difference inputs.
---

<p class="note-kicker">Metric handbook · fidelity and geometry</p>

# Did the model preserve the activation change?

For target vectors (y_i) and reconstructions (hat y_i), use (y_i=z_i)
for a raw-input SAE and (y_i=\Delta z_i) for an SSAE. Activation and
optimization do not change that family assignment. Those regimes must not
share a reconstruction axis.

## Reconstruction scores

| Metric | Definition | Better | Main sensitivity |
|---|---|---:|---|
| MSE | (n^{-1}\sum_i\lVert y_i-\hat y_i\rVert_2^2) | lower | target scale and dimension |
| NMSE | MSE divided by target energy | lower | unstable if target energy is tiny |
| (R^2) | (1-\mathrm{SSE}/\mathrm{SST}) | higher | reference mean and dataset composition |
| cosine | mean angular alignment of (y_i,\hat y_i) | higher | ignores magnitude |

NMSE and cosine are complementary. A reconstruction can point in the right
direction while having the wrong length, or match average energy while missing
pair-level directions.

## Direction and dictionary scores

- **Held-out shift cosine** compares a decoder column with a shift created by
  the assigned concept on new pairs.
- **Decoder-difference cosine** compares directions induced by paired
  dictionaries.
- **Dictionary best cosine** matches each learned decoder direction with its
  closest reference direction.
- **Angular error** reports the same alignment in degrees.
- **Norm ratio** checks scale after a matching has been defined.

High dictionary cosine is evidence about geometry, not a guarantee that the
encoder activates the direction on the right examples.

## Plotting convention

Use a logarithmic x-axis for NMSE when values span orders of magnitude. If
plotting (R^2) near one, plot (1-R^2) on a log scale and label the direction
of improvement explicitly.
