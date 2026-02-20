# Effective Dimensionality — `effective_dim.png`

## Graph Layout

- **Panels:** 2 side-by-side (left: 90% variance threshold, right: 95% variance threshold)
- **X-axis:** Three groups, one per layer (Layer 1 — 16 ch, Layer 2 — 32 ch, Layer 3 — 64 ch)
- **Y-axis:** Number of principal components needed to reach the threshold
- **Bars:** 4 per group, one per model, color-coded:
  - **Blue** — Teacher (ResNet-112)
  - **Red** — Student - No KD
  - **Orange** — Student - Logit KD
  - **Green** — Student - Factor Transfer
- **Error bars:** ±1 standard deviation across 3 seeds (black caps)
- **Text labels:** Mean value printed above each bar

---

## The Representation: What Goes In

Same as all other figures — GAP-pooled representations of shape **(10000, C)**:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

| Layer | Channels (C) | Maximum possible effective dim |
|-------|-------------|-------------------------------|
| Layer 1 | 16 | 16 |
| Layer 2 | 32 | 32 |
| Layer 3 | 64 | 64 |

The maximum effective dimensionality at any layer is bounded by C — you can never need more components than you have channels.

### Effective dimensionality under the three representation regimes

| Regime | Max possible dim (layer 3) | What effective dim measures |
|--------|---------------------------|---------------------------|
| **GAP (B, C)** (ours) | 64 | How many independent channel-level features carry variance |
| **Flatten (B, C×H×W)** | 4,096 | How many channel-at-position features carry variance |
| **Channel-centric (C, B×H×W)** | 64 | How many independent spatial co-activation patterns exist |

**Flatten** would show dramatically higher effective dimensionality because spatial variation adds thousands of variance directions. A single "vertical edge" channel contributes differently at every spatial location — each becomes a separate variance direction. Under GAP, that entire spatial pattern collapses to one number (the average edge strength), contributing one variance direction.

**Channel-centric** would show the same maximum dimensionality as GAP (both have C × C covariance matrices), but the effective dim values would likely differ. Under GAP, the covariance is estimated from B=10,000 observations (images). Under channel-centric, it's estimated from B×H×W=640,000 observations (image-position pairs). The channel-centric estimate is statistically more stable, and it captures spatial co-activation structure that GAP averages away. A pair of channels that always co-fire at object boundaries would contribute to the effective dimensionality differently under channel-centric (where that spatial co-firing is visible) vs. GAP (where it's averaged out).

---

## What Effective Dimensionality Measures

This figure distills the PCA variance curves (see `pca_variance.md`) into a single number per model-per-layer: **how many principal components do you need to capture X% of the total variance?**

### The computation

```python
# from analyze.py, lines 117-119
def effective_dim(cum_var, threshold):
    """Minimum components to reach threshold fraction of variance."""
    return int(np.searchsorted(cum_var, threshold) + 1)
```

`np.searchsorted(cum_var, threshold)` finds the first index where the cumulative variance curve reaches or exceeds the threshold. Add 1 because components are 1-indexed (the "first" component is component 1, but it's at array index 0).

For example, if the cumulative variance array is `[0.45, 0.72, 0.88, 0.94, 0.97, ...]`:
- `searchsorted(cum_var, 0.90)` returns index 3 (the first entry ≥ 0.90 is `0.94` at index 3)
- Effective dim at 90% = 3 + 1 = **4 components**
- `searchsorted(cum_var, 0.95)` returns index 4 (`0.97` at index 4)
- Effective dim at 95% = 4 + 1 = **5 components**

### The multi-seed averaging

```python
# from analyze.py, lines 237-253
for i, key in enumerate(MODEL_KEYS):
    seeds = available_seeds(reps, key)
    dims_per_seed = []
    for seed in seeds:
        dims = []
        for layer in LAYERS:
            _, _, cum = pca(reps[key][seed][layer])
            dims.append(effective_dim(cum, thresh))
        dims_per_seed.append(dims)
    arr = np.array(dims_per_seed, dtype=float)
    mean_d = arr.mean(axis=0)
    std_d = arr.std(axis=0)

    ax.bar(x + i * width, mean_d, width, yerr=std_d,
           label=SHORT[key], color=COLORS[key], capsize=3)
```

For each model, effective dimensionality is computed separately for each of the 3 seeds, then averaged. `arr` has shape `(3, 3)` — 3 seeds × 3 layers. `mean(axis=0)` averages across seeds, giving one mean per layer. Error bars show ±1 std.

---

## Why Two Thresholds

### 90% — the core dimensionality

This captures the "dominant subspace." It answers: **what's the bare minimum number of directions needed to represent the bulk of the information?** It's a loose cut that captures the large eigenvalues — the major axes of the ellipsoid.

### 95% — the extended dimensionality

This is stricter, capturing more of the "tail." It includes smaller eigenvalues that represent finer-grained directions of variation. The gap between the 90% and 95% numbers is informative:

- **Small gap** (e.g., 90% needs 5, 95% needs 7): variance drops off sharply after the dominant components. The representation has a clean low-rank structure with a short tail.
- **Large gap** (e.g., 90% needs 5, 95% needs 15): there's a long tail of many small-but-nonzero variance directions. The representation has a compact core but also substantial distributed structure beyond it.

---

## How to Read the Bars

- **A short bar** means the representation is compact — most information lives in a small subspace. Few directions dominate. This is the fingerprint of a low-rank representation.

- **A tall bar** (approaching the full channel count) means the representation uses its dimensions evenly. No small set of directions dominates. The representation is closer to full-rank.

- **Comparing bars within a group** (same layer, different models): which model is more compact vs. more distributed? If the teacher's bar is taller than the student's, the student has compressed information into fewer dimensions — it learned a lower-rank structure.

- **Comparing bars across groups** (same model, different layers): how dimensionality evolves with depth. Early layers (16 channels) are naturally constrained. Deeper layers (64 channels) have room to spread, so the bar height relative to the maximum is more informative than the absolute number.

- **Error bars:** small = consistent across seeds, large = representation structure depends on initialization.

---

## Relationship to pca_variance.png

These two figures show the same underlying data sliced differently:

- `pca_variance.png` shows the full cumulative curve — the complete picture of how variance accumulates
- `effective_dim.png` reads off two specific points from that curve (where it crosses 90% and 95%) and plots them as bars for easy comparison

You could eyeball the effective dimensionality from the variance curves by finding where they cross the dashed lines — but the bar chart makes cross-model comparison much cleaner.

---

## Why This Matters for Your Thesis

Effective dimensionality is one of the most direct measures of *representation geometry*. If distillation changes the effective dimensionality of the student compared to training independently, that's concrete evidence that distillation shapes *how the student organizes information* — not just what accuracy it achieves. A student matching the teacher's effective dimensionality more closely than the No KD baseline has inherited structural properties from the teacher: it represents data in a similarly-shaped subspace.
