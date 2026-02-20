# ICA Correlation Heatmap — `ica_correlation.png`

## Graph Layout

- **Grid:** 3 rows × 3 columns
  - **Rows:** One per student model (No KD, Logit KD, Factor Transfer)
  - **Columns:** One per layer (Layer 1, Layer 2, Layer 3)
- **Each cell:** A square heatmap of size (C × C) where C is the channel count at that layer
  - Layer 1: 16×16
  - Layer 2: 32×32
  - Layer 3: 64×64
- **Y-axis:** "Teacher IC" — teacher independent component index (0 to C-1)
- **X-axis:** "Student IC (matched)" — student independent component index, **reordered** by the Hungarian matching
- **Color scale:** YlOrRd (yellow → orange → red), range 0 to 1
- **Colorbar:** Shared, labeled "|Correlation|"
- **Title:** Indicates this uses a single representative seed (seed 0)

---

## The Representation: What Goes In

Same GAP-pooled representations of shape **(10000, C)**:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

ICA is run on these (10000, C) matrices — one ICA decomposition for the teacher's representation and one for the student's.

### ICA under the three representation regimes

ICA is the metric most affected by the choice of regime, because it scales poorly with dimensionality.

| Regime | ICA input shape (layer 3) | # components | Feasibility |
|--------|--------------------------|--------------|-------------|
| **GAP (B, C)** (ours) | (10000, 64) | 64 | Easy — seconds |
| **Flatten (B, C×H×W)** | (10000, 4096) | up to 4096 | Feasible but harder (see below) |
| **Channel-centric (C, B×H×W)** | (640000, 64) | 64 | Easy — well-powered, same component count as GAP |

**Flatten** — FastICA on (10000, 4096) is borderline: 10,000 samples for 4,096 components. It's technically feasible (especially with cluster compute), but:
- Convergence is slower and less stable
- Layer 1 at (10000, 16384) is **underdetermined** — more features than samples. You'd need to PCA-reduce first (e.g., keep 500 components, then ICA those)
- The Hungarian matching on a 4096×4096 matrix is O(n³) ≈ 70 billion operations — still doable but no longer instant
- The independent components would mix spatial and channel information: "channel 5 at position (2,3)" is a separate dimension from "channel 5 at position (6,7)." This is a valid decomposition but answers a different question: "what are the independent spatially-localized features?"

**Channel-centric** — This is actually the most natural fit for ICA. You have (640000, 64): 640,000 observations of 64 variables. ICA finds 64 independent components — the same count as GAP. But the observations are (image, position) pairs instead of just images, so ICA is finding channel combinations that are **spatially independent** — combinations of channels whose spatial activation patterns are statistically independent of each other. With 640K observations for only 64 components, the estimates would be very reliable. The Hungarian matching is the same 64×64 as under GAP. Computationally trivial.

The key difference: GAP-ICA finds channels that fire independently *across images*. Channel-centric-ICA finds channels that fire independently *across spatial positions*. These are different notions of independence.

Under GAP, ICA answers: **what are the statistically independent channel-level features, and do teacher and student share them?**

---

## What ICA Is — The Full Explanation

### Why ICA, when we already have PCA

PCA finds directions of maximum *variance*. But maximum-variance directions aren't necessarily the most *meaningful*. PCA directions are merely uncorrelated — knowing the value of PC 1 tells you nothing *linear* about PC 2. But they can still be statistically dependent through higher-order relationships.

**ICA (Independent Component Analysis)** finds directions that are *statistically independent* — knowing the value of one component tells you *absolutely nothing* about any other component, not even through nonlinear relationships. Independence is a much stronger condition than uncorrelatedness.

### The cocktail party analogy

Two microphones in a room record a mixture of two speakers. Each microphone gets a different linear combination of the two voices. PCA on the microphone signals would find the "loudest direction" and the "second loudest direction," but these are combinations of both speakers. ICA would recover the two individual speakers — the independent *sources* that combine to create the observed signals.

In a neural network, the "observed signals" are the channel activations. The "sources" are underlying independent factors — perhaps corresponding to visual attributes, object parts, or abstract features. ICA tries to unmix the channels to find these sources.

### The mathematical model

ICA assumes each image's representation is a linear mixture of independent sources:

    x = A s

where:
- x is the observed C-dimensional representation vector (one image)
- s is a C-dimensional vector of independent source signals
- A is a (C × C) mixing matrix

ICA finds the *unmixing matrix* W such that s = W x, where the resulting source signals are as independent as possible.

### How FastICA works

FastICA finds independence by maximizing **non-Gaussianity**. The key insight comes from the Central Limit Theorem: a mixture of independent signals is *more Gaussian* than the individual signals. Therefore, the most non-Gaussian projections of the data are the ones that best isolate individual sources.

FastICA iteratively finds projection directions that maximize a measure of non-Gaussianity (approximated via negentropy — the difference between a signal's entropy and a Gaussian with the same variance). Each iteration finds one independent component; the algorithm deflates the data and repeats.

```python
# from analyze.py, lines 160-163
ica_t = FastICA(n_components=n_comp, random_state=42, max_iter=2000, tol=1e-3)
ica_s = FastICA(n_components=n_comp, random_state=42, max_iter=2000, tol=1e-3)
S_t = ica_t.fit_transform(X_t_norm)
S_s = ica_s.fit_transform(X_s_norm)
```

`n_components=n_comp` extracts C independent components (one per channel — full decomposition). `random_state=42` ensures reproducibility within a seed. `max_iter=2000` and `tol=1e-3` control convergence.

### What an independent component *is* in a neural network

Each IC is a direction in channel space where the projected data has maximum non-Gaussianity. Concretely, IC 1 might be a weighted combination of channels that responds to a specific visual pattern (say, "horizontal texture at medium frequency") in a way that's statistically independent of all other ICs. Different images activate this IC to different degrees, and that activation pattern is independent of every other IC's activation pattern.

### Why independence matters for representation analysis

If the teacher's ICA components match the student's, it means both networks have decomposed the visual world into the same set of independent factors. This is a much stronger claim than "they have similar variance structure" (PCA) or "they have similar pairwise image relationships" (CKA). It says: the two networks have discovered the same fundamental independent features.

---

## The Matching Problem and the Hungarian Algorithm

### Why matching is necessary

ICA is run **separately** on the teacher and student representations. The resulting components are ordered arbitrarily — teacher IC 1 has no inherent relationship to student IC 1. Teacher IC 1 might correspond to student IC 37, or to no student IC at all.

### The cross-correlation matrix

```python
# from analyze.py, lines 165-174
A = S_t - S_t.mean(0)
B = S_s - S_s.mean(0)
std_a, std_b = A.std(0), B.std(0)
ok_a, ok_b = std_a > 1e-10, std_b > 1e-10
A[:, ok_a] /= std_a[ok_a]
B[:, ok_b] /= std_b[ok_b]
cross_corr = (A.T @ B) / len(A)
```

This computes the **Pearson correlation** between every teacher IC and every student IC. `S_t` and `S_s` are both (10000, C) — each column is one IC's activations across all test images.

1. **Center:** `A = S_t - S_t.mean(0)` — subtract column means
2. **Normalize:** `A[:, ok_a] /= std_a[ok_a]` — divide by standard deviation (the `ok_a` mask avoids division by zero for degenerate components)
3. **Cross-correlate:** `(A.T @ B) / len(A)` — dot products between standardized columns, divided by N, giving Pearson correlations. The result is a (C × C) matrix.

Entry `cross_corr[i, j]` is: the correlation between teacher IC i and student IC j across 10,000 images. High absolute value means they co-vary strongly — they respond to the same patterns in the data. Near zero means they're unrelated.

### The Hungarian algorithm

```python
# from analyze.py, lines 176-178
row_ind, col_ind = linear_sum_assignment(-np.abs(cross_corr))
matched_corr = np.abs(cross_corr[row_ind, col_ind])
corr_display = np.abs(cross_corr[:, col_ind])
```

`linear_sum_assignment` is the **Hungarian algorithm** (also called Kuhn-Munkres). It solves the assignment problem: pair each teacher IC with exactly one student IC, one-to-one, to **maximize** the total absolute correlation.

We pass `-np.abs(cross_corr)` because `linear_sum_assignment` *minimizes* cost, and we want to *maximize* correlation, so we negate.

The result:
- `row_ind, col_ind`: the optimal pairing (teacher IC `row_ind[i]` matches student IC `col_ind[i]`)
- `matched_corr`: the absolute correlation for each matched pair
- `corr_display`: the full cross-correlation matrix with student columns **reordered** according to the matching, so the diagonal shows matched pairs

### Why the Hungarian algorithm is necessary

For a 64×64 matrix, there are 64! ≈ 10^89 possible one-to-one matchings. Brute force is impossible. The Hungarian algorithm finds the optimal matching in O(n³) time — roughly 64³ = 262,144 operations, which is instantaneous.

The matching uses absolute correlation (not signed) because ICA components have arbitrary sign — negating a component doesn't change its independence. A correlation of -0.9 is just as good a match as +0.9.

---

## What the Heatmap Shows

After the Hungarian matching reorders the student columns, the heatmap displays `np.abs(cross_corr[:, col_ind])` — the absolute cross-correlation between all teacher ICs (rows) and the reordered student ICs (columns).

```python
# from analyze.py, lines 449-451
im = ax.imshow(corr_display, cmap='YlOrRd', vmin=0, vmax=1,
               aspect='auto', interpolation='nearest')
```

### What to look for

- **Bright diagonal, dark off-diagonal:** Each teacher IC has a clear, unique counterpart in the student. The matching is clean — components correspond one-to-one without cross-talk. The two models have discovered the same independent factors.

- **Bright diagonal with some bright off-diagonal entries:** The matched pairs are good, but some teacher ICs also correlate with non-partner student ICs. The factors partially overlap — some information is shared across multiple components.

- **Dim or patchy diagonal:** Some teacher ICs have no good match in the student (or vice versa). Those independent factors exist in one model but not the other.

- **Overall dimness:** The teacher and student have discovered largely *different* independent factors. Their representations are organized around different axes of variation.

### Reading a specific cell

Cell (row 5, column 12) shows: the absolute correlation between teacher IC 5 and the student IC that was assigned to position 12 in the matched ordering. If the diagonal at column 12 is bright but this off-diagonal cell at row 5 is also bright, it means teacher IC 5 shares information with both its own match and with the student IC assigned to position 12.

---

## Why Only One Seed

The heatmap is shown for seed 0 only because **ICA component ordering is arbitrary between runs**. Seed 0's IC 3 has no relationship to seed 1's IC 3. Averaging the raw (C × C) heatmaps across seeds would produce meaningless noise. The summary statistics (mean matched correlation, count of strong matches) — which *are* averaged across seeds — are shown separately in `ica_summary.png`.

---

## Why This Matters for Your Thesis

ICA provides the finest-grained comparison available in this analysis. While PCA asks about variance structure and CKA about global similarity, ICA asks: **has the student discovered the same underlying independent factors as the teacher?** If distillation produces more one-to-one component correspondence than independent training, it's evidence that the distillation signal transfers specific *feature identities*, not just aggregate statistical properties.
