# Principal Angles — `principal_angles.png`

## Graph Layout

- **Panels:** 3 side-by-side (one per layer: Layer 1, Layer 2, Layer 3)
- **X-axis:** "Component index" — 1, 2, 3, ... up to k (the subspace dimension; see below)
- **Y-axis:** "Principal angle (degrees)" — 0° to 90°
- **Lines:** One per student model (comparing that student to the teacher), color-coded:
  - **Red** — No KD vs. Teacher
  - **Orange** — Logit KD vs. Teacher
  - **Green** — Factor Transfer vs. Teacher
- **Markers:** Small circles on each data point
- **Shaded bands:** ±1 standard deviation across 3 seeds
- **Y-axis limits:** -2 to 92 (slightly padded beyond the 0°–90° theoretical range)

---

## The Representation: What Goes In

Same GAP-pooled representations of shape **(10000, C)**:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

Principal angles are computed between the **PCA subspaces** of the teacher and student at the same layer. Both representations have the same number of channels C, so the subspaces live in the same ambient space (ℝ^C) and can be directly compared.

### Principal angles under the three representation regimes

| Regime | Ambient space (layer 3) | Subspaces live in | What angles measure |
|--------|------------------------|-------------------|---------------------|
| **GAP (B, C)** (ours) | ℝ^64 | ℝ^64 | Channel-combination alignment |
| **Flatten (B, C×H×W)** | ℝ^4096 | ℝ^4096 | Channel-at-position alignment |
| **Channel-centric (C, B×H×W)** | ℝ^64 | ℝ^64 | Spatial co-activation pattern alignment |

**Flatten** — The ambient space becomes ℝ^4096 (layer 3). PCA subspaces are still low-dimensional (effective dim << 4096), so principal angles are well-defined and computationally cheap. But the subspaces now include spatial directions — a principal component could point along "channel 5 at position (3,4)" rather than just "channel 5." Two models detecting the same features at slightly different positions would have misaligned subspaces under flattening, even if their channel-level structure is identical.

**Channel-centric** — PCA is computed on (C, B×H×W), and the covariance matrix is (C × C) — same ambient space as GAP (ℝ^64 at layer 3). But the PCA directions mean something different. Under GAP, a PCA direction is "this combination of channels captures the most image-to-image variance." Under channel-centric, a PCA direction is "this combination of channels has the most correlated spatial co-activation." The principal angles between teacher and student channel-centric subspaces ask: **do the dominant spatial co-firing patterns of the teacher align with the student's?** This is well-powered (covariance estimated from 640K observations) and the same dimensionality as GAP, so no computational concerns.

The key difference across regimes:
- **GAP:** Do the models emphasize the same *channel combinations*?
- **Flatten:** Do the models emphasize the same *features at the same positions*?
- **Channel-centric:** Do the models have the same *spatial co-activation geometry*?

---

## What "Component Index" Means on the X-Axis

This is the question you asked about, so let me be very precise.

Principal angles decompose the alignment between two subspaces into a **sequence** of angles, one per dimension. The sequence is sorted from smallest (best alignment) to largest (worst alignment).

- **Component index 1** = the 1st principal angle = the angle between the **most-aligned pair** of directions from the two subspaces. This is the best-case alignment.
- **Component index 2** = the 2nd principal angle = the angle between the **next most-aligned pair**, after removing the first pair from consideration. By definition, θ₂ ≥ θ₁.
- **Component index k** = the k-th principal angle = the angle between the k-th best pair. This is the worst-case alignment among the k pairs considered.

The "component" here is a **paired direction** — one direction from the teacher's subspace matched with one from the student's subspace. Component index 1 is the first such pair (best match), component index 2 is the second pair, and so on.

The curve always goes **upward or stays flat** because the angles are sorted. Early components show the best alignment; later components show progressively worse alignment.

### How many components are shown

The number of points on the x-axis (the length of the curve) is determined by k, the subspace dimensionality used for comparison:

```python
# from analyze.py, lines 387-391
_, vecs_t, cum_t = pca(reps[teacher][seed][layer])
_, vecs_s, cum_s = pca(reps[s_key][seed][layer])
k = max(effective_dim(cum_t, 0.95), effective_dim(cum_s, 0.95))
k = min(k, vecs_t.shape[1], vecs_s.shape[1])
angles = principal_angles_deg(vecs_t[:, :k], vecs_s[:, :k])
```

k = max(teacher's 95% effective dim, student's 95% effective dim), but capped at the available dimensionality. This ensures we compare the full "informative" subspace of whichever model uses more dimensions.

Because k can differ slightly between seeds (effective dimensionality varies), the code truncates to the minimum across seeds before averaging:

```python
# from analyze.py, lines 395-396
min_len = min(len(a) for a in angle_sets)
angle_arr = np.array([a[:min_len] for a in angle_sets])
```

---

## What "Subspace" Means

At layer 3, a model produces 64-dimensional representation vectors. PCA tells you that most of the variance lives in far fewer than 64 dimensions — say, 15 dimensions capture 95%. Those 15 most important PCA directions span a 15-dimensional **subspace** within ℝ^64.

Think of a subspace as a "hyperplane" — a flat slice through the high-dimensional space where the data actually lives. In 3D, a 2D subspace is a plane through the origin. In 64D, a 15D subspace is a 15-dimensional "plane" through the origin.

The teacher has its own subspace (defined by its PCA eigenvectors). The student has its own subspace (defined by its PCA eigenvectors). Both are embedded in the same ℝ^64 (because both layer 3 representations have 64 channels). Principal angles describe the geometric relationship between these two subspaces.

---

## What Principal Angles Are — The Geometric Intuition

### The 2D analogy

Imagine two lines through the origin in a 2D plane. The angle between them (0° to 90°) completely describes their relationship: 0° means they're the same line, 90° means they're perpendicular.

Now imagine two planes (2D subspaces) in 3D space. A single angle isn't enough — the two planes might share one direction exactly but diverge in the other. You need **two angles** to describe their relationship. These are the two principal angles.

### Generalizing to k dimensions

For k-dimensional subspaces, you need k principal angles: θ₁ ≤ θ₂ ≤ ... ≤ θ_k. They're defined by a **greedy** pairing process:

1. **θ₁:** Find one direction in the teacher's subspace and one direction in the student's subspace that are as aligned as possible (smallest angle between them). These two directions form the first "matched pair."

2. **θ₂:** Now restrict to directions **perpendicular to the first pair**. Find the most-aligned directions in the remaining subspaces. Their angle is θ₂. By construction, θ₂ ≥ θ₁.

3. **θ₃, θ₄, ..., θ_k:** Continue this process, each time removing the previously matched directions and finding the next-best pair.

### What angle values mean

- **θ = 0°:** Those two directions are identical. Both subspaces contain that exact direction. Along that axis, the teacher and student are perfectly aligned.
- **θ = 45°:** Partial alignment. The two directions overlap somewhat but aren't the same.
- **θ = 90°:** Complete misalignment. One subspace extends in a direction that the other subspace doesn't touch at all. That dimension of variation in one model has no counterpart in the other.

---

## The Computation — Step by Step

```python
# from analyze.py, lines 131-137
def principal_angles_deg(U, V):
    """Principal angles (degrees) between subspaces spanned by columns of U, V."""
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    svals = np.linalg.svd(U.T @ V, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    return np.degrees(np.arccos(svals))
```

### Step 1: QR factorization

```python
U, _ = np.linalg.qr(U)
V, _ = np.linalg.qr(V)
```

U is a (D × k) matrix whose columns are the teacher's top-k PCA eigenvectors. V is the same for the student. QR decomposition orthogonalizes the columns — the PCA eigenvectors are already orthogonal (they're eigenvectors of a symmetric matrix), but QR ensures numerical stability. After QR, U and V have orthonormal columns (unit length, mutually perpendicular).

### Step 2: Cross-product matrix

```python
svals = np.linalg.svd(U.T @ V, compute_uv=False)
```

`U.T @ V` is a (k × k) matrix. Entry (i, j) is the **dot product** of teacher PCA direction i with student PCA direction j. A dot product between unit vectors equals the cosine of the angle between them. So this matrix contains the cosines of all pairwise angles between teacher and student directions.

But we don't want all pairwise angles — we want the **optimally matched** angles. That's what the SVD does.

### Step 3: SVD extracts the principal angles

The **singular values** of U^T @ V are the cosines of the principal angles. The SVD finds the optimal matching: it identifies which teacher direction should pair with which student direction to produce the sequence θ₁ ≤ θ₂ ≤ ... ≤ θ_k.

This is because the SVD decomposes U^T V = P Σ Q^T, where P and Q are rotations within each subspace and Σ is diagonal with the singular values. Each singular value σ_i = cos(θ_i) gives the cosine of the i-th principal angle.

The key mathematical fact: the SVD of the cross-product matrix U^T V solves exactly the same optimization problem as the greedy pairing procedure described above, but it does it all at once in one computation instead of iteratively.

### Step 4: Convert to degrees

```python
svals = np.clip(svals, 0, 1)
return np.degrees(np.arccos(svals))
```

The singular values are cosines, so `arccos` converts them to angles (in radians), and `np.degrees` converts to degrees. `np.clip(svals, 0, 1)` handles floating-point edge cases — singular values slightly above 1.0 or below 0.0 due to numerical noise would make `arccos` fail.

---

## How to Read the Curves

- **A curve that stays near 0° across many components:** The two subspaces are nearly identical — they share many directions. The teacher's important variance directions are also the student's. Strong geometric alignment.

- **A curve that rises quickly:** Only the first few directions are shared; after that, the subspaces diverge. The two models agree on the dominant structure (the "big picture") but disagree on finer-grained directions (the "details").

- **A curve that starts high (near 90° from the first component):** Even the most-aligned directions are substantially different. The subspaces point in genuinely different directions — the two models organize information along fundamentally different axes.

- **A curve that reaches 90° and plateaus:** You've exhausted all shared directions. Beyond that point, every remaining direction in one subspace is orthogonal to the other subspace.

- **Comparing curves between student models:** If Logit KD's curve stays lower (closer to 0°) than No KD's curve, the logit-distilled student's subspace is more aligned with the teacher's — distillation produced more geometrically similar representations.

- **Comparing across panels (layers):** If alignment is better at early layers than late layers, the models share low-level feature structure but diverge in high-level features.

---

## Relationship to CKA

CKA and principal angles measure related but distinct things:

- **CKA** compares *similarity structures over data points*: do images cluster the same way?
- **Principal angles** compare *subspace geometry*: do the feature directions point the same way?

Two representations could have high CKA but moderate principal angles if they encode similar image relationships using somewhat rotated coordinate systems — the "meaning" is preserved even though the "axes" differ. Conversely, well-aligned subspaces (small principal angles) generally imply high CKA, because if the representations live in similar regions of space, their pairwise similarity patterns will also agree.

Principal angles give a *per-direction* decomposition of alignment, while CKA gives a single aggregate number. The curve shape reveals structure that CKA hides: are the first 3 directions perfectly aligned while the rest diverge? Or is there moderate alignment across all directions? CKA would give the same number in both cases, but the principal angle curves would look very different.

---

## Why This Matters for Your Thesis

Principal angles decompose the question "are these representations aligned?" into a per-direction answer. You can see whether distillation aligns the dominant directions (early components) while leaving finer directions free to diverge, or whether it produces comprehensive alignment across the full subspace. This directly probes the geometry of what's being transferred: does distillation transfer the "big picture" structure, the fine details, or both?
