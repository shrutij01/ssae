---
title: Identifiability leaves the scale free
description: Proposition 2 identifies a concept axis up to signed scale; experiments 03 and 07 therefore test sign specificity and target-side overshoot separately.
---

<p class="kicker">Notes · 2026-07-29 · theory</p>

# Identifiability leaves the scale free

<p class="lede">Proposition 2 concludes that
$\hat q = \mathbf{A}_V\mathbf{D}\mathbf{P}$. The permutation
$\mathbf{P}$ means a recovered coordinate still needs a label. The diagonal
$\mathbf{D}$ means its sign and magnitude still need to be estimated before it
can be used as an intervention.</p>

## What $\mathbf{D}$ leaves undetermined

$\mathbf{D}$ is invertible and diagonal, but otherwise unrestricted. Each
decoder column is therefore a true concept direction multiplied by an arbitrary
non-zero real number. The theorem identifies a one-dimensional subspace -- a
line through the origin -- while an intervention needs an oriented and scaled
vector on that line.

This ambiguity is intrinsic to the observation model. Rescaling concept $k$ by
$c$ and its decoder column by $1/c$ leaves every observed
$\Delta\mathbf{z}$ unchanged. A steering procedure must import sign and scale
from additional data.

In this repository both are fitted on training pairs only. The sign is oriented
so that the selected direction agrees with the mean source-to-target shift. A
least-squares coefficient $\alpha$ then sets the step length. Held-out examples
are never used for either choice.

## The signed endpoint test

[Experiment 03](../results/steering.md) keeps one held-out source $z_0$, one
paired target $z_1$, and one train-fitted vector $\alpha d$ fixed. It compares

$$
b = \cos(z_0,z_1),\qquad
s_+ = \cos(z_0 + \alpha d,z_1),\qquad
s_- = \cos(z_0 - \alpha d,z_1).
$$

The expected ordering is $s_+ > b > s_-$. The positive and negative points are
two rays from the same baseline vertex. They are not two task directions and
they are not averaged together: addition tests whether the oriented vector
moves toward the target, while subtraction is a sign-specific control that
should move away from that same target.

This corrects the older analysis that averaged forward and reverse traversals.
That average answered a different question and could make both signs appear to
improve because each traversal started at a different endpoint. The current
design has one origin and one scoring target, so the signs have an unambiguous
interpretation.

## What the signed test does not establish

The ordering establishes sign specificity, but not that the fitted magnitude is
optimal. A very small step can have the right ordering while producing a
negligible effect. A very large step can initially move toward the paired target
and then pass it. Endpoint cosine can also hide drift in directions unrelated to
the supervised factor.

For that reason the report retains the raw baseline and both endpoint cosines,
the two signed changes, the ordered-pair fraction, the fitted vector norm, and
matched-norm random and train-mean controls. Seed standard deviations remain in
the tables rather than being placed as ambiguous error bars on the ray plot.
Gemma and Pythia are reported on separate pages, with SSAEs and SAEs in separate
panels.

## The overshoot test

[Experiment 07](../results/overshoot.md) asks the complementary scale question.
It starts from an activation that is already the French target and adds the
English-to-French vector again:

$$
z_{\mathrm{FR}}(m) = z_{\mathrm{FR}} + m(\alpha d),
\qquad m \in \{0, 0.25, 0.5, 1, 2\}.
$$

This is deliberately different from negative steering in experiment 03. The
negative sign control starts from English and subtracts the vector. The
overshoot test starts from French and continues in the positive direction. It
measures cosine to the original French activation, fixed-probe French retention,
paired retrieval, within-French geometry, off-axis drift, and matched-norm random
controls across the whole strength sweep.

The distinction matters: a language classifier can become more confident that
an activation is French even as the activation loses the identity and geometry
of the particular French example. "More French" is therefore not the same as
"better preservation of the target."

## The honest position

Identifiability up to $\mathbf{D}\mathbf{P}$ is the strongest conclusion the
observation model supports, and it distinguishes a usable axis from a mixed
one. It does not by itself deliver a steering intervention. Naming, orienting,
and scaling the coordinate remain separate estimation problems, so the public
results report direction recovery, signed endpoint behavior, and overshoot as
separate claims rather than collapsing them into one gain statistic.
