# Steering with an SSAE direction

This tutorial describes the intervention used in the generation study. A
decoder column is added to one hidden state during autoregressive generation.

## 1. Select a direction on held-out data

Do not choose an atom from the prompts used to report generation outcomes.
Use a held-out annotated subset to assign a concept to a decoder column:

```python
from evals.concept_mapping import prepare_from_embeddings, map_concepts

concepts = prepare_from_embeddings(
    sae,
    positive_embeddings=[female_embeddings],
    negative_embeddings=[male_embeddings],
    names=["gender_indicator"],
)
mapping = map_concepts(concepts, method="pearson", matching="hungarian")
feature_index = mapping.assignments[0]["feature_idx"]
direction = sae.decoder.weight[:, feature_index]
```

The labels select and name the direction after unsupervised training. They are
not used to train the SSAE.

## 2. Match the intervention site

Use the same model family, residual-stream convention, token position, and
layer from which the training embeddings were extracted. The Bias-in-Bios study
uses Gemma-2B layer 25 and perturbs the last token.

\[
h_{\ell,t}^{\,\prime}=h_{\ell,t}+\alpha d_k.
\]

Here `alpha` is the intervention strength. Its numeric value is meaningful only
relative to the norm of `direction` and the scale of hidden states at that
layer.

## 3. Sweep strength in both directions

Include zero and several positive and negative strengths. Reuse prompts and
decoding seeds across methods.

```python
strengths = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

for alpha in strengths:
    steered_direction = alpha * direction
    # Register the residual-stream hook, generate, then remove the hook.
```

A dose–response curve can reveal onset, saturation, sign errors, and
non-monotonic behavior that a single strength conceals.

## 4. Measure target and off-target effects

At minimum, report:

- target-concept success;
- fluency or likelihood degradation;
- semantic similarity to the unsteered completion;
- changes to named off-target attributes;
- per-prompt paired uncertainty.

The Bias-in-Bios demonstration counts female, male, and neutral pronoun usage.
That is deliberately narrow. It shows a behavioral response but does not
establish that the intervention captures gender completely or changes nothing
else.

## 5. Keep the comparison controlled

When comparing SSAE with an SAE dictionary:

- select both atoms on the same held-out examples;
- intervene at the same layer and token position;
- compare both equal raw coefficient and equal direction norm;
- use identical prompts, seeds, and decoding parameters;
- retain all generations, including malformed or neutral outputs.

This is an implementation tutorial, not a published behavioral result. A
generation claim should be added only after prompt-level outcomes, decoding
settings, and a reproducible result builder are committed.
