---
title: Antipodal Latents
description: Why the frozen nonlinear SSAEs use exact positive and negative feature pairs.
---

<p class="kicker">Design notes · representation</p>

# Why the frozen nonlinear SSAEs use antipodal latents

<p class="lede">Difference training gives every pair an orientation. Antipodal feature pairs let a non-negative gate represent either direction along one learned axis.</p>

An ordered pair gives an oriented shift:

\[
\Delta z(a,b)=z_b-z_a, \qquad \Delta z(b,a)=-\Delta z(a,b).
\]

A signed linear code can represent both orders with one coefficient: reversing
the pair simply negates that coefficient. ReLU, JumpReLU, and TopK codes are
non-negative, so one gated coordinate cannot carry both signs of the same
axis. The frozen nonlinear SSAEs therefore store each independent axis as two
physical latents:

\[
w_{k+m}=-w_k, \qquad d_{k+m}=-d_k, \qquad
\alpha_k=h_k-h_{k+m}.
\]

Here \(w\) is an encoder row, \(d\) is a decoder column, and \(\alpha_k\) is the
single signed coefficient used for evaluation. The two poles are one learned
axis, not two concepts.

The implementation keeps this relationship exact by projecting parameters and
gradients onto opposite encoder and decoder pairs. Encoder offsets and
JumpReLU thresholds are shared across a pair, the offsets are non-positive so
zero does not activate a pole, and the decoder bias is fixed at zero. Training
also presents both endpoint orders.

This pairing halves physical-column capacity, so the frozen comparison gives a
nonlinear SSAE \(2d\) columns to retain \(d\) independent axes. Signed linear
SSAEs need only \(d\) columns. Raw-activation SAEs are not given antipodal pairs.

Antipodality is therefore a parameterisation chosen for the current nonlinear
difference-input protocol. It is not a claim that difference training alone
forces antipodality, or that the underlying concepts were empirically
discovered to be antipodal.
