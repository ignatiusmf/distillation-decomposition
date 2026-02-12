# CKA Cross-Layer Heatmaps — `cka_cross_layer.png`

## Graph Layout

- **Panels:** 4 side-by-side heatmaps
  - Panel 1: "Teacher (self)" — teacher layers vs. teacher layers
  - Panel 2: "Teacher vs No KD"
  - Panel 3: "Teacher vs Logit KD"
  - Panel 4: "Teacher vs Factor Transfer"
- **Each heatmap:** 3×3 grid (rows = 3 layers, columns = 3 layers)
- **Row axis (Y):**
  - Panel 1: "Layer" (L1, L2, L3 of the teacher)
  - Panels 2-4: "Teacher layer" (L1, L2, L3)
- **Column axis (X):**
  - Panel 1: "Layer" (L1, L2, L3 of the teacher)
  - Panels 2-4: "Student layer" (L1, L2, L3)
- **Cell values:** Linear CKA score (0 to 1), printed in each cell
- **Color scale:** Viridis colormap — dark purple (0) to bright yellow (1)
- **Colorbar:** Shared, labeled "Linear CKA"
- **Cell text color:** White on dark cells (CKA < 0.5), black on bright cells (CKA ≥ 0.5)
- **Averaging:** Each cell is the mean CKA across 3 seeds

---

## The Representation: What Goes In

Each cell computes CKA between two matrices, both of shape **(10000, C)** — GAP-pooled representations:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

Crucially, the two matrices in a cross-model comparison **can have different numbers of columns** because different layers have different channel counts. For example, comparing teacher layer 1 (10000, 16) against student layer 3 (10000, 64). CKA handles this naturally — it compares *pairwise image similarities*, not individual features, so dimensionality mismatch is not a problem.

### CKA under the three representation regimes

| Regime | Kernel matrix size | What CKA compares |
|--------|-------------------|-------------------|
| **GAP (B, C)** (ours) | 10000 × 10000 | Do images relate the same way in channel-summary space? |
| **Flatten (B, C×H×W)** | 10000 × 10000 | Do images relate the same way in full spatial-detail space? |
| **Channel-centric (C, B×H×W)** | C × C (e.g., 64 × 64) | Do channels relate to each other the same way? |

**Flatten** — CKA would compare pairwise *image* similarity structures that include spatial information. Two models detecting the same features at slightly different positions would show *lower* CKA under flattening (spatial misalignment reduces pairwise agreement), but *higher* CKA under GAP (spatial differences erased). Cross-layer comparisons become harder under flattening because Layer 1 (32×32 spatial) and Layer 3 (8×8 spatial) have incompatible dimensions — the heatmap cells would compare (10000, 16384) against (10000, 4096), conflating spatial resolution changes with feature changes.

**Channel-centric** — This is a fundamentally different use of CKA. Instead of comparing how *images* relate (N=10000 kernel), it compares how *channels* relate (N=64 kernel at layer 3, N=16 at layer 1). The kernel matrix entry K[i,j] = dot product of channel i's activation pattern with channel j's, across all images and spatial positions. CKA would then ask: does the teacher organize its channels the same way as the student? Do channels that co-activate in the teacher also co-activate in the student?

The concern with channel-centric CKA is sample size: with N=64 (layer 3) or N=16 (layer 1), the kernel matrices are small and CKA estimates become noisier. N=16 at layer 1 is genuinely underpowered. N=64 at layer 3 is usable but should be interpreted with caution — error bars across seeds would reveal how stable the estimates are.

GAP normalizes all layers to `(10000, C)`, removing spatial resolution as a confound and giving well-powered CKA estimates. It isolates the question: **do these layers encode similar channel-level relationships between images?**

---

## What CKA Is — The Full Explanation

CKA (Centered Kernel Alignment) measures whether two representations encode the same **relationships between data points**. It doesn't compare feature values directly — it compares the *similarity structure* over images.

### The problem CKA solves

You have two representations of the same 10,000 images — say, the teacher's layer 2 and the student's layer 2. Both are (10000, 32) matrices. But teacher channel 5 has no reason to correspond to student channel 5 — the channels aren't aligned. You can't just do element-wise comparison.

What you want to know is: **do these two representations tell the same story about which images are similar to which?** If image A and image B are close together in the teacher's representation, are they also close in the student's? CKA measures this.

### Step 1: From representations to kernel matrices

Instead of thinking about individual features, think about **pairs of images**. Given representation X (N × D), compute the kernel matrix:

    K = X X^T    (shape: N × N)

Entry K[i, j] is the dot product of image i's representation with image j's representation. This is a measure of similarity: high dot product means similar representations, low means dissimilar.

This N × N matrix encodes the full relational structure: which images the model "thinks" are similar or different. It abstracts away the specific features and keeps only the relationships.

### Step 2: Center the kernel matrices

Before comparing, we center both kernel matrices. Centering removes the overall "average similarity level" and keeps only the *structure* — which pairs are more or less similar than average.

In the code, centering happens implicitly:

```python
# from analyze.py, lines 123-124
X = X - X.mean(0)
Y = Y - Y.mean(0)
```

Mean-centering X and Y before computing X X^T and Y Y^T is mathematically equivalent to applying kernel centering (HKH where H = I - 1/n · 11^T) to the uncentered kernel matrices. The effect is the same: the resulting kernel matrices have zero row and column means.

This centering is analogous to subtracting the mean before computing a correlation coefficient — it ensures you're measuring *pattern* similarity, not just overall magnitude.

### Step 3: Measure alignment via HSIC

The **Hilbert-Schmidt Independence Criterion (HSIC)** measures the alignment between two kernel matrices. Conceptually, HSIC asks: "when K says images i and j are similar, does L also say they're similar?"

For linear kernels, HSIC reduces to:

    HSIC(X, Y) = ||Y^T X||_F^2

where ||·||_F is the Frobenius norm (the square root of the sum of all squared entries of the matrix).

**What Y^T X is:** This is a (D_Y × D_X) matrix where entry (a, b) is the dot product of student feature a with teacher feature b across all images. Large values mean those specific features are correlated across the dataset. The Frobenius norm aggregates all of these cross-feature correlations into a single number.

**Why squaring the Frobenius norm:** The squared Frobenius norm ||Y^T X||_F^2 equals the sum of squared entries of Y^T X, which equals Tr((Y^T X)^T (Y^T X)) = Tr(X^T Y Y^T X). This is the same as Tr(K L) where K = X X^T and L = Y Y^T — the inner product of the two kernel matrices. Positive where both kernels agree (both say similar or both say dissimilar), negative where they disagree.

### Step 4: Normalize to get CKA

Raw HSIC depends on the scale of the representations. If you multiply all activations by 2, HSIC quadruples. To make the metric invariant to scaling, CKA normalizes:

```python
# from analyze.py, lines 125-128
num = np.linalg.norm(Y.T @ X, 'fro') ** 2
den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
return num / den
```

This is:

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) · HSIC(Y, Y))

This normalization is exactly like how **Pearson correlation** normalizes covariance:

    correlation = covariance(x, y) / (std(x) · std(y))

CKA does the same thing at the level of kernel matrices: it divides the cross-alignment by the geometric mean of the self-alignments. The result is always between 0 and 1.

### What CKA values mean

- **CKA = 1.0:** The two representations induce *identical* pairwise similarity structures, up to scaling and rotation. Every pair of images has the same relative similarity in both representations. The two networks "see" data identically at those layers.

- **CKA = 0.0:** No linear relationship between the similarity structures. How one representation groups images has nothing to do with how the other groups them. The representations are *orthogonal* in their relational structure.

- **CKA = 0.7:** Substantial but imperfect alignment. Most image-pair relationships are preserved, but there are meaningful differences. Some distinctions captured by one representation are missed by the other.

### What CKA is invariant to

- **Isotropic scaling:** Multiplying all features by a constant doesn't change CKA. This is good — if one model just uses larger activation magnitudes, that shouldn't count as a structural difference.
- **Orthogonal transformations:** Rotating the feature space (applying an orthogonal matrix) doesn't change CKA. This is crucial — two models might store the same information in different "rotations" of their channel axes, and CKA correctly says they're equivalent.

### What CKA is NOT invariant to

- **Arbitrary invertible transformations:** If two representations encode the same information but through nonlinear or non-orthogonal entanglement, CKA won't recognize them as equivalent. CKA specifically measures *linear* similarity structure.

---

## How to Read the Heatmaps

### Panel 1: Teacher self-comparison

- **Diagonal (always 1.0):** Every layer is perfectly similar to itself.
- **Off-diagonal entries:** How much representational structure is shared between the teacher's own layers.
  - High off-diagonal values (e.g., 0.8): layers are somewhat redundant — they encode similar information despite being at different depths.
  - Low off-diagonal values (e.g., 0.2): the network transforms representations substantially as data passes through layers. Early and late representations are very different.
- This provides a **reference**: you can't reasonably expect a student to be more aligned with the teacher than the teacher's own layers are with each other.

### Panels 2-4: Teacher vs. student

Cell at row i, column j = CKA(teacher layer i, student layer j).

- **Strong diagonal** (high values on the diagonal, lower off-diagonal): the student's layers correspond to the teacher's in order. Student layer 1 is most similar to teacher layer 1, student layer 2 to teacher layer 2, etc. This is a "well-aligned" student — it develops representations at the same rate as the teacher.

- **High values in a single row:** One teacher layer is similar to *all* student layers. The student doesn't differentiate as much across depth — its layers are more uniform.

- **High values in a single column:** One student layer captures information present across all teacher layers — that student layer has learned a "universal" representation.

- **Off-diagonal peaks** (e.g., teacher L2 most similar to student L3): The student develops representations at a different rate than the teacher. It might need more depth to reach the same representational stage.

### The averaging

```python
# from analyze.py, lines 283-300
for s_key in students:
    mats = []
    for seed in seeds:
        m = np.zeros((3, 3))
        for i, li in enumerate(LAYERS):
            for j, lj in enumerate(LAYERS):
                m[i, j] = linear_cka(reps[teacher][seed][li], reps[s_key][seed][lj])
        mats.append(m)
    mean_matrices.append(np.mean(mats, axis=0))
```

For each seed, the full 3×3 CKA matrix is computed. Then the three matrices are averaged element-wise. This gives a stable estimate of the cross-layer CKA structure.

---

## Why This Matters for Your Thesis

The cross-layer heatmaps directly visualize whether distillation causes the student to develop representations that *structurally mirror* the teacher's, layer by layer. If the distilled student shows stronger diagonal alignment than the independently-trained student, that's evidence that distillation transfers structural organization — not just output behavior. The teacher self-comparison provides essential context for interpreting the cross-model comparisons.
