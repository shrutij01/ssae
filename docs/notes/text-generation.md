---
title: Text Generation
description: Protocol for validating behavioral interventions with recovered SSAE directions.
---

<p class="note-kicker">Planned evidence · intervention</p>

# From a vector to generated text

Embedding-space alignment is useful only if the direction survives the rest of
the network. The intervention adds a selected decoder direction to the hidden
state during generation:

\[
h_{\ell,t}^{\,\prime}=h_{\ell,t}+\alpha d_k.
\]

The confirmatory unit should be a prompt under a fixed decoding seed and
pre-specified strength. A paired analysis should compare methods on the same
prompts and test both target success and off-target semantic or fluency change.

## Publication gate

The previous website figure reproduced aggregate counts embedded directly in a
plotting script. The prompt-level source needed to validate those counts is not
committed, so neither the plot nor the numerical claims are published here.
Before this page displays a result, the repository needs:

- prompt-level outcomes with method, direction, strength, decoding seed, and
  classifier result;
- the corresponding generation or an immutable reference to it;
- a validator for the paired design and denominators;
- an `experiments/expNN/report.py` builder using the shared result style;
- generated figures, tables, a conclusion, and a hashed artifact manifest.

This preserves the distinction between an experiment protocol and an audited
result. Once the source is available, rebuilding the experiment directory will
update this page without a second hand-maintained data copy.

*Return to the [research notebook →](index.md)*
