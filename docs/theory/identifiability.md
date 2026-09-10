# Identifiability

<p class="result-lede">
For SSAE, “identified” means recoverable <strong>up to permutation and
component-wise scaling</strong> under the stated pairing, variability,
reconstruction, and sparsity assumptions—not uniquely named semantic atoms in
every LLM context.
</p>

## Testable consequence

<div class="hypothesis-sheet">
<p><b>Null \(H_0\).</b> On data generated inside the assumed observation model,
learned coordinates are no more aligned with the true sources than an
unidentified affine reconstruction.</p>
<p><b>Alternative \(H_1\).</b> After the permitted permutation and component
scaling, SSAE has higher source MCC and converges toward the same equivalence
class across independent seeds.</p>
<p><b>Boundary.</b> Seed agreement alone cannot reject the null: multiple seeds
can reproducibly converge to the same wrong representation.</p>
</div>

Identifiability asks: **can we uniquely recover the true underlying factors of variation from learned representations?** This page covers the theoretical motivation and the design choices that make SSAEs identifiable.

## The Problem

Standard autoencoders are **non-identifiable**: for any learned representation $h$, there exist infinitely many rotations $Rh$ that achieve identical reconstruction loss. This means:

- Features may not correspond to meaningful concepts
- Different random seeds produce incomparable representations
- Interpretability claims are unfalsifiable

## Why Identifiability is Hard at Scale

| Challenge | Description |
|-----------|-------------|
| **Unknown ground truth** | We don't know how many "true concepts" exist in LLM representations |
| **Superposition** | LLMs encode more concepts than dimensions, violating standard assumptions |
| **Distributional shift** | Training and deployment distributions differ |
| **Scale** | Hidden dimensions of 65,536+ make exhaustive verification infeasible |

## Design Choices for Identifiability

### 1. Difference-Based Training

Instead of training on raw activations $z$, we train on differences $\Delta z = z_2 - z_1$ between counterfactual pairs.

**Why this helps:**

- Removes shared/static information, isolating concept-specific variation
- Creates a natural "intervention" structure for causal identification
- Under the nonlinear ICA framework, observing how representations change under known interventions is sufficient for identification

### 2. Unit-Norm Decoder Columns

Decoder columns are maintained at unit $L_2$ norm throughout training via two mechanisms:

- **Gradient projection**: After each step, gradients are projected to the tangent space of the unit-sphere constraint: $g \leftarrow g - w(w^\top g)$
- **Periodic renormalisation**: Columns are hard-projected back to unit norm every $N$ epochs

**Why this helps:**

- Removes scale ambiguity — features cannot be arbitrarily rescaled
- Each decoder column becomes an interpretable "concept direction" in activation space
- Enables meaningful comparison of feature importance via coefficient magnitude

### 3. Constrained Sparsity

Hard sparsity constraints (rather than L1 regularisation) ensure a consistent, controllable sparsity level.

**Why this helps:**

- Sparsity is a necessary condition for identifiability under ICA-like assumptions
- Exact control means the same sparsity level is maintained across runs, datasets, and models — enabling fair comparison
- The constrained formulation avoids the trade-off opacity of regularisation

### 4. ZCA Whitening (Optional)

Pre-processing with ZCA whitening decorrelates input features:

$$W_{\text{ZCA}} = U D^{-1/2} U^\top$$

where $U, D$ come from the eigendecomposition of the covariance matrix.

**Why this helps:**

- Under ICA assumptions, whitening + sparsity is sufficient for identification
- Equalises variance across dimensions, preventing dominant features from dominating learning
- Improves optimisation landscape by reducing the condition number

## Measuring Identifiability

### Mean Correlation Coefficient (MCC)

The primary metric. Given $C$ ground-truth concepts and $K$ learned features:

1. Compute the absolute Pearson correlation $|r_{c,k}|$ for all $(c, k)$ pairs
2. Build the $(C, K)$ correlation matrix
3. Apply Hungarian matching to find the optimal one-to-one assignment
4. Report the mean matched correlation

MCC = 1.0 means perfect recovery of ground-truth concepts. MCC = 0.0 means no alignment.

### Pairwise MCC (Representation Stability)

If representations are truly identifiable, different random seeds should converge to the same solution (up to permutation). Pairwise MCC compares decoder weight matrices across seeds:

1. For each pair of seeds $(i, j)$, compute the correlation between their decoder columns
2. Hungarian-match to account for permutation ambiguity
3. High pairwise MCC indicates stable, reproducible representations

### Probe MCC vs SAE MCC

The gap between probe MCC (linear probes on raw LLM embeddings) and SAE MCC reveals how much the SAE captures relative to what is linearly accessible:

- **Probe MCC >> SAE MCC**: The SAE is leaving information on the table
- **Probe MCC $\approx$ SAE MCC**: The SAE is extracting most of the linearly available concept information
- **SAE MCC > Probe MCC**: Should not happen under correct evaluation (both are on the same data)
