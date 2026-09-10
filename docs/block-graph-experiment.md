# Post-Hoc Block Discovery from SSAE Difference Codes

## Experimental question

Given a trained Sparse Shift Autoencoder (SSAE), can a partition of its
dictionary atoms into variable-sized concept blocks be recovered after
training?

The experiment distinguishes three objects:

- the activation difference,
  \[
  \Delta\mathbf{x}_n
  =
  \mathbf{x}^{\mathrm{cf}}_n-\mathbf{x}^{\mathrm{base}}_n;
  \]
- the signed SSAE code of that difference,
  \[
  \mathbf{h}_n=r(\Delta\mathbf{x}_n);
  \]
- the magnitude used to estimate statistical affinity,
  \[
  \mathbf{a}_n=|\mathbf{h}_n|.
  \]

The graph is therefore constructed from
\[
|r(\mathbf{x}^{\mathrm{cf}}-\mathbf{x}^{\mathrm{base}})|,
\]
not from raw activations \(r(\mathbf{x})\), and not directly from
\(|\mathbf{x}^{\mathrm{cf}}-\mathbf{x}^{\mathrm{base}}|\). This choice matches
the input distribution on which the SSAE was trained and removes the arbitrary
orientation of a signed concept direction.

In general,
\[
|r(\mathbf{x}^{\mathrm{cf}}-\mathbf{x}^{\mathrm{base}})|
\neq
|r(\mathbf{x}^{\mathrm{cf}})-r(\mathbf{x}^{\mathrm{base}})|,
\]
because encoder biases, normalization, and nonlinear gates need not commute
with differencing. The experiment uses the expression on the left.

## Estimand

The trained model supplies a decoder dictionary
\[
\mathbf{D}
=
[\mathbf{d}_1,\ldots,\mathbf{d}_m]
\in\mathbb{R}^{d\times m}
\]
and codes \(\mathbf{h}_n\in\mathbb{R}^m\) satisfying
\[
\Delta\mathbf{x}_n
\approx
\mathbf{b}
+
\sum_{j=1}^{m}h_{nj}\mathbf{d}_j.
\]

The target estimand is a partition
\[
\mathcal{P}
=
\{B_1,\ldots,B_G\}
\]
of the \(m\) dictionary atoms. Each block has a membership size
\[
s_g=|B_g|
\]
and a separately estimated intrinsic dimension \(r_g\leq s_g\). The experiment
does not require the membership sizes or intrinsic dimensions to be equal
across blocks.

## Data separation

The requested HDF5 split must exist and be non-empty. The experiment deliberately
does not use the repository's train-split fallback.

The evaluation differences are randomly divided into:

1. a discovery subset, used to estimate graph affinities and propose partitions;
2. a validation subset, used to select a partition by reconstruction and
   description length.

Concept labels are not used in either step. On controlled datasets, labels may
subsequently assess whether discovered blocks correspond to known factors.

## Stage 1: Select graph nodes

For feature \(j\), define its held-out root-mean-square activity as
\[
\rho_j
=
\sqrt{\frac{1}{N}\sum_{n=1}^{N}h_{nj}^{2}}.
\]

Features with \(\rho_j\) below a numerical floor are excluded from graph
estimation. When the dictionary is too wide for a dense pilot affinity matrix,
the experiment retains the features with the largest \(\rho_j\). Excluded
features remain singleton blocks; they are not silently discarded from
reconstruction.

## Stage 2: Estimate graph affinity

### Magnitude dependence

The primary affinity is the absolute Pearson dependence between magnitude
profiles:
\[
A^{\mathrm{mag}}_{ij}
=
\left|
\operatorname{corr}
\left(
|h_{\cdot i}|,
|h_{\cdot j}|
\right)
\right|.
\]

Taking the outer absolute value admits both positive and negative dependence.
This matters when different atoms tile mutually exclusive regions of the same
concept manifold.

### Support dependence

Each feature receives a scale-adaptive threshold
\[
\tau_j
=
\operatorname{Quantile}_{q}
\left(
|h_{\cdot j}|
\right),
\]
and binary support
\[
s_{nj}
=
\mathbb{1}\{|h_{nj}|>\tau_j\}.
\]

The support affinity is the absolute phi correlation
\[
A^{\mathrm{sup}}_{ij}
=
\left|
\operatorname{corr}
\left(
s_{\cdot i},
s_{\cdot j}
\right)
\right|.
\]

This threshold is used to compare activation patterns, not to redefine the
SSAE's training constraint.

### Optional decoder evidence

An optional secondary term is
\[
A^{\mathrm{dec}}_{ij}
=
\frac{|\mathbf{d}_i^\top\mathbf{d}_j|}
{\|\mathbf{d}_i\|_2\|\mathbf{d}_j\|_2}.
\]

Its default weight is zero. Orthonormal directions can belong to the same
Grassmannian block, so decoder cosine cannot be the primary signal.

### Combined graph

The pilot graph uses
\[
A_{ij}
=
\frac{
w_{\mathrm{mag}}A^{\mathrm{mag}}_{ij}
+
w_{\mathrm{sup}}A^{\mathrm{sup}}_{ij}
+
w_{\mathrm{dec}}A^{\mathrm{dec}}_{ij}
}{
w_{\mathrm{mag}}+w_{\mathrm{sup}}+w_{\mathrm{dec}}
}.
\]

Only the union of each node's \(k\) strongest edges is retained. The default is
\[
(w_{\mathrm{mag}},w_{\mathrm{sup}},w_{\mathrm{dec}})
=
(0.75,0.25,0).
\]

These coefficients define a pilot, not a settled estimator. Ablations over the
three terms are required before substantive conclusions.

## Stage 3: Propose variable-sized partitions

Edges are processed from strongest to weakest at several affinity quantiles.
The endpoints of an edge are merged when:

1. the edge exceeds the candidate threshold; and
2. the merged component does not exceed a conservative maximum size.

This produces variable-sized connected blocks plus singleton features. The
maximum size prevents a percolating graph from collapsing into one component;
it is a computational regularizer rather than an asserted concept dimension.

The singleton partition is always included. Every graph partition is paired
with random partitions having exactly the same block-size multiset.

The next implementation stage should replace this thresholded pilot with
sparse subspace clustering or an MDL-scored agglomerative merge procedure. The
current pilot makes the graph and evaluation protocol testable before adding
that computational cost.

## Stage 4: Evaluate a proposed partition

For block \(B_g\), its realized contribution is
\[
\mathbf{m}_{ng}
=
\sum_{j\in B_g}h_{nj}\mathbf{d}_j
=
\mathbf{D}_{B_g}\mathbf{h}_{n,B_g}.
\]

Its presence energy is
\[
e_{ng}
=
\|\mathbf{m}_{ng}\|_2
=
\sqrt{
\mathbf{h}_{n,B_g}^{\top}
\mathbf{D}_{B_g}^{\top}\mathbf{D}_{B_g}
\mathbf{h}_{n,B_g}
}.
\]

This definition, rather than \(\|\mathbf{h}_{n,B_g}\|_2\), accounts for
non-orthogonal decoder atoms and cancellations.

For a block budget \(\ell\), retain the \(\ell\) largest energies and reconstruct
\[
\widehat{\Delta\mathbf{x}}_n^{(\ell)}
=
\mathbf{b}
+
\sum_{g\in\operatorname{Top}_{\ell}(e_n)}
\mathbf{m}_{ng}.
\]

The held-out variance explained is
\[
R^2
=
1-
\frac{
\sum_n
\|\Delta\mathbf{x}_n-\widehat{\Delta\mathbf{x}}_n^{(\ell)}\|_2^2
}{
\sum_n
\|\Delta\mathbf{x}_n-\overline{\Delta\mathbf{x}}\|_2^2
}.
\]

## Stage 5: Select by description length

At distortion variance \(\delta\), the pilot score is
\[
L_\delta(\mathcal{P})
=
L(\text{support})
+
L_\delta(\text{block contributions})
+
L_\delta(\text{residual})
+
\frac{1}{N}L(\mathcal{P}).
\]

For exactly \(\ell\) active blocks,
\[
L(\text{support})
=
\log_2\binom{G}{\ell}.
\]

The code and residual terms apply the smooth Gaussian rate
\[
R_\delta(\lambda)
=
\frac{1}{2}\log_2
\left(
1+\frac{\lambda}{\delta}
\right)
\]
to their covariance eigenvalues. The partition metadata term uses the number
of unlabeled partitions with the observed block sizes:
\[
\frac{1}{N}L(\mathcal{P})
\approx
\frac{1}{N}
\log_2
\left[
\frac{m!}
{\left(\prod_{g=1}^{G}|B_g|!\right)
\left(\prod_s n_s!\right)}
\right].
\]

Here, \(n_s\) is the number of discovered blocks having membership size \(s\).
The second product removes permutations among blocks with the same size.

Among non-random candidates reaching a prescribed held-out \(R^2\), the
experiment selects the candidate with the smallest total bits per sample.

## Stage 6: Estimate block dimension

For each selected block, compute a thin singular value decomposition
\[
\mathbf{D}_{B_g}
=
\mathbf{U}_g\mathbf{\Sigma}_g\mathbf{V}_g^\top.
\]

The contribution coordinates in the orthonormal basis \(\mathbf{U}_g\) are
\[
\mathbf{c}_{ng}
=
\mathbf{\Sigma}_g\mathbf{V}_g^\top\mathbf{h}_{n,B_g}.
\]

Let \(\lambda_{g1},\ldots,\lambda_{gr}\) denote the eigenvalues of the
conditional covariance of \(\mathbf{c}_{ng}\) when block \(g\) is selected.
The report contains:

\[
\operatorname{srank}(g)
=
\frac{\sum_a\lambda_{ga}}{\max_a\lambda_{ga}},
\]

\[
\operatorname{PR}(g)
=
\frac{\left(\sum_a\lambda_{ga}\right)^2}
{\sum_a\lambda_{ga}^2},
\]

\[
\operatorname{erank}(g)
=
\exp\left(
-\sum_a p_{ga}\log p_{ga}
\right),
\qquad
p_{ga}
=
\frac{\lambda_{ga}}{\sum_b\lambda_{gb}},
\]

and the smallest PCA dimension explaining \(95\%\) of the conditional
contribution variance.

## Pilot command

```bash
python -m experiments.legacy.block_graph \
    /path/to/ssae_seed0 \
    /path/to/embeddings.h5 \
    --output-dir results/findings/block_graph/runs/pilot \
    --split cfc_test \
    --device cpu
```

For an initial controlled experiment, use a semi-synthetic dataset with a
genuine non-empty test split. Keep labels hidden until the partition has been
selected.

## Outputs

The command writes:

- `affinity_edges.npz`: sparse graph edges and their component affinities;
- `feature_statistics.npz`: feature energy, support thresholds, and the
  discovery/validation indices;
- `report.json`: candidate partitions, random controls, reconstruction/MDL
  sweeps, the selected partition, and per-block dimension estimates.

Every report records the defining convention:

```text
difference:        delta_x = x_counterfactual - x_base
code affinity:     abs(h(delta_x))
not code affinity: raw x or abs(delta_x)
```

## Required ablations

The pilot should not be interpreted without the following comparisons:

1. magnitude affinity only;
2. support affinity only;
3. magnitude plus support affinity;
4. decoder cosine only;
5. all affinity terms;
6. shuffled rows within each code coordinate;
7. random partitions matched on block sizes;
8. singleton blocks;
9. sensitivity to the graph \(k\), support quantile, and discovery split;
10. stability across SSAE training seeds, measured by principal angles between
    recovered block subspaces.

Success requires more than a lower in-sample reconstruction error. A useful
partition should:

- reduce held-out description length relative to singleton and random controls;
- preserve the target reconstruction fidelity with fewer active blocks;
- recur across data resamples and SSAE seeds;
- align with held-out concepts on controlled data;
- support specific within-block interventions with limited collateral effects.

## Current limitation and next estimator

Pairwise magnitude and support dependence cannot identify every block. In
particular, uniformly distributed orthogonal coordinates of a block can be
uncorrelated. The next estimator should add dictionary self-expression:

\[
\mathbf{c}_i^\star
=
\arg\min_{\mathbf{c}}
\frac{1}{2}
\left\|
\mathbf{d}_i-\mathbf{D}_{-i}\mathbf{c}
\right\|_2^2
+
\lambda\|\mathbf{c}\|_1,
\]

followed by the symmetric affinity
\[
A^{\mathrm{ssc}}_{ij}
=
|c_{ij}^\star|+|c_{ji}^\star|.
\]

This term is informative only when a concept subspace contains more learned
atoms than its linear dimension. No method can identify an arbitrary partition
from one nonredundant basis per block without activation-distribution
information or additional assumptions.

## Reference

```bibtex
@article{fel2026structuring,
  title   = {Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds},
  author  = {Fel, Thomas and Kowal, Matthew and Jacobs, Mozes and Hazra, Dron and Bhalla, Usha and others},
  journal = {arXiv preprint arXiv:2606.25234},
  year    = {2026}
}
```
