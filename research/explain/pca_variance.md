# PCA Cumulative Variance — `pca_variance.png`

## Graph Layout

- **Panels:** 3 side-by-side (one per network layer: Layer 1, Layer 2, Layer 3)
- **X-axis:** Number of principal components kept (1, 2, 3, ... up to D, the channel count of that layer)
- **Y-axis:** Cumulative variance explained (%), ranging from 0 to ~105
- **Lines:** One per model, color-coded:
  - **Blue** — Teacher (ResNet-112)
  - **Red** — Student - No KD (ResNet-56, trained independently)
  - **Orange** — Student - Logit KD (ResNet-56, distilled via soft labels)
  - **Green** — Student - Factor Transfer (ResNet-56, distilled via feature factors)
- **Shaded bands:** ±1 standard deviation across 3 seeds
- **Dashed horizontal lines:** Reference thresholds at 90% and 95%
- **Legend:** Top-left or top-right of each panel

---

## The Representation: What Goes In

Before any analysis, it's crucial to understand what data this graph operates on. Each model produces, for every test image, a feature map tensor at each layer:

```
model.forward(x) returns [out1, out2, out3, logits]
# out1: (B, 16, 32, 32)  — 16 channels, 32×32 spatial
# out2: (B, 32, 16, 16)  — 32 channels, 16×16 spatial
# out3: (B, 64,  8,  8)  — 64 channels,  8×8  spatial
```

During extraction, we apply **Global Average Pooling (GAP)** over the spatial dimensions:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

This collapses each spatial feature map into its average value per channel. For 10,000 test images, we get a matrix of shape **(10000, C)** where C is 16, 32, or 64 depending on the layer.

Each row is one image's representation: a C-dimensional vector where each entry is the average activation of one channel across all spatial positions.

### The three representation regimes

The raw tensor `(B, C, H, W)` can be collapsed to a 2D matrix in three different ways, each asking a fundamentally different question. Our analysis uses regime 1 (GAP):

| Regime | Reshape | Samples | Features | Layer 3 shape |
|--------|---------|---------|----------|---------------|
| **1. GAP** (ours) | `mean(dim=[2,3])` → (B, C) | Images (10,000) | Channels (64) | (10000, 64) |
| **2. Flatten** | `reshape` → (B, C×H×W) | Images (10,000) | Channel-at-position (4,096) | (10000, 4096) |
| **3. Channel-centric** | `reshape` → (C, B×H×W) | Channels (64) | Activation-at-image-and-position (640,000) | (64, 640000) |

#### Regime 1 — GAP (B, C): "What features exist?"

GAP throws away all **spatial information** — where in the image a feature fires. It only keeps **how much** each channel fires on average. PCA under GAP asks: across the channel dimensions alone, how is variance distributed? Which channel-level features carry the most image-to-image variation?

#### Regime 2 — Flatten (B, C×H×W): "What features exist, and where?"

Flattening preserves spatial detail. PCA would capture variance from both *which channels activate* and *where in the image they activate*. The variance curves would behave differently:

- **More components needed** to reach 90%/95% — spatial variation adds many directions of spread
- **Effective dimensionality much higher** — "vertical edge at position (3,4)" and "vertical edge at position (6,7)" are separate variance directions
- **Cross-model comparison captures spatial alignment** — two models detecting the same features at slightly different positions would show different PCA structure, which GAP would mask

#### Regime 3 — Channel-centric (C, B×H×W): "How do channels co-activate spatially?"

This flips the perspective entirely. Instead of asking how images vary, it asks **how channels relate to each other**. The covariance matrix is still (C × C) — same size as GAP — but the entries measure something different:

- **GAP covariance(i, j):** "When channel i has high *average* activation on an image, does channel j?" — co-occurrence across images, spatial detail lost
- **Channel-centric covariance(i, j):** "When channel i fires at a specific position in a specific image, does channel j fire *at that same position*?" — **spatial co-activation**

Two channels might have correlated averages (similar under GAP) but fire at completely different spatial locations within each image (uncorrelated under channel-centric). Or vice versa — channels with uncorrelated averages might always co-fire at object boundaries.

PCA under the channel-centric regime finds the dominant **spatial co-firing patterns** — which combinations of channels tend to activate together at the same spatial locations. The covariance matrix is estimated from B×H×W = 640,000 observations (at layer 3), making it very well-powered statistically. The variance curves would reveal: is spatial co-activation concentrated in a few channel combinations, or distributed broadly?

#### Why GAP was chosen for the current analysis

GAP isolates the question: **what features does the model detect?** — stripped of spatial detail. This is the cleanest starting point for comparing teacher and student representations. The other two regimes answer related but distinct questions and would make valuable complementary analyses.

---

## What PCA Actually Does

PCA decomposes the variance in the representation into ranked directions. Here is each step, tied to the code.

### Step 1: Center the data

```python
# from analyze.py, line 107
X_c = X - X.mean(axis=0)
```

Subtract the mean of each column (each channel) so the cloud of 10,000 points is centered at the origin. Before centering, if channel 5 has a mean activation of 2.3, every image "starts" at 2.3 for that channel. After centering, we only see *how much each image deviates from the average*. This removes the "DC offset" and leaves only the signal — the variation.

Without centering, the first principal component would just point toward the mean (the "brightness" direction), which tells you nothing about how images differ.

### Step 2: Compute the covariance matrix

```python
# from analyze.py, line 108
cov = np.cov(X_c, rowvar=False)
```

This produces a **(D × D)** matrix. For layer 3, that's (64 × 64). Entry `cov[i, j]` tells you: **when channel i is above its mean, does channel j tend to also be above its mean?**

- **Diagonal entries** `cov[i, i]` are each channel's variance — how much that channel's activation varies across images.
- **Off-diagonal entries** `cov[i, j]` are covariances — do channels i and j co-activate? Positive covariance means they tend to activate together (e.g., both respond to animal textures). Negative covariance means one is high when the other is low. Zero means they vary independently.

The covariance matrix fully describes the second-order statistical structure of the representation — how all channels relate to each other in terms of linear co-variation.

### Step 3: Eigendecompose

```python
# from analyze.py, lines 109-112
vals, vecs = np.linalg.eigh(cov)
idx = vals.argsort()[::-1]
vals, vecs = vals[idx], vecs[:, idx]
vals = np.maximum(vals, 0)
```

`np.linalg.eigh` decomposes the symmetric covariance matrix into **eigenvalues** (`vals`) and **eigenvectors** (`vecs`). The eigenvalues come out in ascending order, so `argsort()[::-1]` reverses them to descending. `np.maximum(vals, 0)` clips any tiny negative eigenvalues that arise from floating-point error (a covariance matrix is positive semi-definite, so true eigenvalues are ≥ 0).

#### What an eigenvalue *is*

Imagine the 10,000 representation vectors as a cloud of dots in D-dimensional space. That cloud has a shape — it's elongated in some directions and flat in others, like an ellipsoid. Each eigenvalue is the **variance of the data along one of the ellipsoid's natural axes**. A large eigenvalue means the cloud stretches far in that direction — there's a lot of variation. A tiny eigenvalue means the cloud is nearly flat there — images barely differ along that direction.

Concretely: if eigenvalue 1 is 5.2 and eigenvalue 2 is 1.1, then the data spreads √5.2 ≈ 2.3 times further along direction 1 than along direction 2. The first direction captures 5.2/(5.2+1.1+...) of the total variance.

#### What an eigenvector *is*

Each eigenvector is a **D-dimensional unit vector** — a recipe for combining channels. For layer 3, eigenvector 1 might be `[0.12, -0.03, 0.41, ..., -0.22]` (64 entries). This says: "to compute the projection onto principal component 1, take 0.12 of channel 0, subtract 0.03 of channel 1, add 0.41 of channel 2, ..." and so on.

The eigenvectors are always **orthogonal** to each other — they point in perpendicular directions. Together, they form a rotated coordinate system where the axes are aligned with the ellipsoid's natural axes rather than the original channel axes. In this rotated system, the data's variance is completely disentangled: variance along axis 1 is entirely captured by eigenvalue 1, variance along axis 2 by eigenvalue 2, etc., with no cross-axis correlation.

### Step 4: Cumulative variance ratio

```python
# from analyze.py, line 113
cum_var = np.cumsum(vals) / vals.sum()
```

`vals.sum()` is the total variance across all directions. `np.cumsum(vals)` gives a running total: eigenvalue 1, then eigenvalue 1+2, then 1+2+3, etc. Dividing by the total gives the fraction of variance explained by the first k components.

If `cum_var[4] = 0.90`, it means the first 5 components (indices 0-4) together explain 90% of all the variance in the representation. The remaining D-5 components share only 10%.

### Step 5: Plotting across seeds

```python
# from analyze.py, lines 193-204
for key in MODEL_KEYS:
    seeds = available_seeds(reps, key)
    curves = []
    for seed in seeds:
        _, _, cum = pca(reps[key][seed][layer])
        curves.append(cum * 100)
    curves = np.array(curves)
    mean_c = curves.mean(axis=0)
    std_c = curves.std(axis=0)
    x = np.arange(1, len(mean_c) + 1)
    ax.plot(x, mean_c, label=SHORT[key], color=COLORS[key], linewidth=2)
    ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                    color=COLORS[key], alpha=0.15)
```

For each model, PCA is run independently for each of the 3 seeds, producing 3 cumulative variance curves. These are stacked into an array and averaged (`mean`) with standard deviation (`std`). The mean curve is drawn as a solid line; `fill_between` creates the shaded band at ±1 std.

---

## How to Read the Curve

- **A curve that shoots up quickly and flattens early** means most variance is concentrated in a few directions. The representation is "low-rank" — a small subspace carries almost all the information. The remaining channels are redundant or carry noise. This typically indicates an *efficient* representation where the network has compressed its information.

- **A curve that climbs slowly and linearly** means variance is spread evenly across many directions. Every channel contributes roughly equally. The representation genuinely uses its full dimensionality. Whether this is "better" or "worse" depends on context — it could mean richer encoding or it could mean the network hasn't learned to compress.

- **Where the curve crosses 90% or 95%** tells you the **effective dimensionality** (plotted separately in `effective_dim.png`). If 5 out of 64 components reach 95%, then 59 channels contribute only 5% of the variance.

- **The shaded band width** reflects consistency across training runs. A narrow band means the variance structure is a stable property of that architecture+method. A wide band means it depends on the particular random initialization — the representation geometry is less deterministic.

- **Comparing curves between models at the same layer** reveals whether distillation changes the variance structure. If a distilled student's curve matches the teacher's more closely than the independently-trained student does, that's evidence the distillation signal shapes *how the student distributes information across its channels*, not just what accuracy it achieves.

---

## Why This Matters for Your Thesis

This figure directly addresses: **how compactly does each model encode information at each layer?** If distillation compresses the student's representation (steeper curve, fewer components needed), it's discarding nuance the teacher retained. If it makes the student's variance profile more teacher-like, that's evidence of structural transfer — the student inherits the teacher's information geometry, not just its output behavior. This is one of the most basic structural fingerprints of a representation, and it sets the stage for the deeper analyses (CKA, principal angles, ICA) that follow.
