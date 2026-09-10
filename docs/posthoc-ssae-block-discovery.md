Yes—we can discover candidate blocks after SSAE training, but not reliably from decoder vectors alone. The best approach is to combine decoder geometry with the learned code statistics and choose the grouping by held-out compression/reconstruction.

## What the paper actually hand-designs

The paper fixes:

- a common maximum block dimension \(b\);
- a partition of the latent coordinates into consecutive blocks;
- the number of active blocks \(\ell\).

It then sweeps \(b\in\{1,2,3,4,6,8,12,16,32\}\) and uses minimum description length to select a global range around \(2\)–\(4\). Stable rank is measured only after those blocks have already been trained; it estimates how many dimensions an assigned block uses, not which directions belong together. [Paper, Appendix C](https://arxiv.org/html/2606.25234v1)

So the paper does not solve automatic block discovery.

## Our post-training problem

Your trained SSAE gives

\[
x_n \approx \sum_{j=1}^{K} h_{nj}d_j,
\]

where \(d_j\) is decoder column \(j\), and \(h_{nj}\) is its signed code.

We want to infer a partition

\[
\mathcal P=\{B_1,\ldots,B_G\},
\]

such that only a few groups are present in each example, while coordinates within an active group jointly describe one concept’s internal variation.

There are actually two quantities to discover:

1. \( |B_g| \): how many SSAE atoms belong to the group;
2. \( r_g \): the intrinsic dimension of their joint contribution.

These need not be equal. A flat SSAE might use eight atoms to tile a concept whose actual subspace is three-dimensional.

## Why cosine clustering is insufficient

In a Grassmannian block, the decoder vectors inside a block are orthogonal:

\[
d_i^\top d_j=0.
\]

Therefore, two directions belonging to the same block may have zero cosine similarity. Clustering \(D^\top D\) would often do exactly the wrong thing.

More fundamentally, suppose a block contains exactly \(b\) basis vectors. From those vectors alone, any grouping of \(b\) independent vectors defines a \(b\)-dimensional subspace. There is no statistical evidence telling us which grouping is correct.

Dictionary-only recovery becomes possible only if the SSAE is sufficiently overcomplete within each concept: for example, six or eight atoms lying in a common three-dimensional container. That redundancy produces a detectable low-rank structure.

## Recommended discovery procedure

### 1. Collect held-out SSAE codes

Run a large held-out collection of the same activation differences through the frozen SSAE:

\[
H=[h_{nj}]\in\mathbb R^{N\times K}.
\]

Because your SSAE uses signed features and regularizes \(|h|\), all presence statistics should use absolute activation or energy, not raw signed correlation. This matches the implementation in [`ssae/constraints.py`](https://github.com/shrutij01/ssae/blob/main/ssae/constraints.py#L283).

Use a per-feature activity threshold rather than one universal threshold, because different features can have different scales or firing rates.

### 2. Build a sparse atom-affinity graph

Use several complementary signals.

Decoder subspace evidence:

- Normalize the decoder atoms.
- For every atom, try to reconstruct it as a sparse combination of other atoms:

\[
c_i^\star
=
\arg\min_c
\|d_i-D_{-i}c\|_2^2+\lambda\|c\|_1.
\]

- Connect \(i\) and \(j\) when they reconstruct one another.

This is sparse subspace clustering applied to dictionary atoms. If several atoms lie in one low-dimensional concept container, they should preferentially represent one another.

Code evidence:

- overlap or NPMI of thresholded supports;
- dependence between \(|h_i|\) and \(|h_j|\);
- nonlinear dependence such as distance correlation or HSIC;
- overlap between the features’ top-activating examples.

Do not rely only on Pearson correlation:

- coordinates such as \(\cos\theta\) and \(\sin\theta\) can belong to the same manifold while having zero correlation;
- two separate concepts can correlate merely because they often co-occur.

The repository already computes a preliminary binary coactivation matrix in [`ssae/tracking.py`](https://github.com/shrutij01/ssae/blob/main/ssae/tracking.py#L340). That is useful as one graph signal, but its current feature subsampling and pairwise correlation are not enough for final block discovery.

### 3. Produce candidate partitions

Run hierarchical clustering, Leiden community detection, or spectral clustering on the combined sparse graph.

I would start with singleton blocks and use agglomerative merging. A proposed merge \(A\cup B\) should be accepted only if it improves a held-out block-model score. This avoids committing to a block count or common block size.

### 4. Select blocks with a post-hoc group-sparse reconstruction test

For a candidate block \(B_g\), define its actual reconstructed contribution:

\[
m_{ng}=D_{B_g}h_{n,B_g}.
\]

Use

\[
e_{ng}=\|m_{ng}\|_2
\]

as its presence score. This is better than \(\|h_{n,B_g}\|_2\) because the SSAE decoder atoms within a candidate block are not orthonormal.

Then test whether reconstruction remains good when only a few block contributions are retained. An even stronger test freezes \(D\) and re-infers the codes using group lasso:

\[
h_n^\star
=
\arg\min_h
\|x_n-Dh\|_2^2
+
\lambda\sum_g\|h_{B_g}\|_2.
\]

A genuine partition should achieve lower held-out distortion at a given number of active groups than:

- singleton blocks;
- random partitions with the same block sizes;
- partitions based only on cosine similarity.

### 5. Choose the partition by MDL

Use the paper’s central idea, but optimize it over partitions:

\[
\operatorname{Score}(\mathcal P)
=
L(\text{block supports})
+
L(\text{within-block codes})
+
L(\text{residual})
+
L(\text{partition}).
\]

At a fixed held-out distortion, accept merges that reduce total description length. This naturally balances:

- one huge block, which has cheap support but an expensive dense code;
- all singleton blocks, which have cheap coordinates but expensive support;
- intermediate, variable-sized blocks.

This is the most principled way to avoid manually choosing \(b\).

## Estimating each discovered block’s dimension

Once membership \(B_g\) is inferred:

1. Compute an orthonormal basis \(Q_g\) for \(\operatorname{span}(D_{B_g})\).
2. Express contributions in that basis:

\[
c_{ng}=Q_g^\top m_{ng}.
\]

3. Compute the covariance of \(c_{ng}\) conditional on block \(g\) being active.
4. Inspect its eigenvalues.

Useful dimension estimates are:

\[
r_{\mathrm{stable}}
=
\frac{\operatorname{tr}(C_g)}
{\lambda_{\max}(C_g)},
\]

participation ratio, effective rank, and a cross-validated PCA reconstruction threshold.

This is the correct analogue of the paper’s stable-rank analysis. Importantly, compute rank from realized contributions, not merely from \(D_{B_g}\): the decoder may span six directions while the data only uses three.

## SSAE-specific recommendation

For your models I would use:

\[
\boxed{
\text{decoder subspace graph}
\;+\;
\text{absolute-code dependence}
\;+\;
\text{held-out MDL merge selection}
}
\]

The practical pipeline would be:

1. load the frozen SSAE;
2. collect \(H\) on held-out counterfactual differences;
3. build sparse candidate edges;
4. agglomeratively merge blocks when validation MDL improves;
5. estimate a separate intrinsic rank \(r_g\) for every block;
6. validate blocks using counterfactual labels only after unsupervised discovery;
7. compare discovered subspaces across seeds using principal angles.

For discovery geometry, use the standardized-space decoder or consistently whiten the raw-space atoms. For causal steering validation, use the folded raw-space decoder loaded by [`loaders/modelloader.py`](https://github.com/shrutij01/ssae/blob/main/loaders/modelloader.py#L59).

The major limitation is that an independently sparse SSAE was not trained to preserve blocks. The paper itself reports that a conventional TopK SAE can shatter a manifold across atoms. Therefore, post-hoc recovery gives candidate blocks, not the same guarantee as training with block sparsity. The strongest final test would be to initialize a BSF from the discovered groups, orthonormalize each group, briefly fine-tune with block sparsity, and verify that the proposed groups survive. [Paper](https://arxiv.org/html/2606.25234v1)
