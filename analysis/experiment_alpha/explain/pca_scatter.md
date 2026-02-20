# PCA Scatter Plot — `pca_scatter.png`

## Graph Layout

- **Panels:** 4 side-by-side (one per model)
  - Panel 1: Teacher (ResNet-112)
  - Panel 2: Student - No KD
  - Panel 3: Student - Logit KD
  - Panel 4: Student - Factor Transfer
- **X-axis:** "PC 1" — projection onto the teacher's first principal component
- **Y-axis:** "PC 2" — projection onto the teacher's second principal component
- **Points:** Each dot = one test image (10,000 total per panel, though overlap hides many)
- **Colors:** One color per class, using the `tab10` colormap (10 distinct colors)
  - CIFAR-10: all 10 classes shown
  - CIFAR-100: 10 evenly-spaced classes (0, 10, 20, ..., 90) — see below for why
- **Point size:** 5 (small, to show density)
- **Alpha:** 0.4 (semi-transparent, so overlapping regions appear darker)
- **Legend:** Shared, to the right of all panels
- **Suptitle:** "PCA Projection (Layer 3 (64 ch), seed 0)"
- **Layer:** Layer 3 only
- **Seed:** Seed 0 only (representative)

---

## The Representation: What Goes In

Same GAP-pooled representations of shape **(10000, C)**, specifically at **layer 3 (C=64)**:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

This figure also uses the **class labels** for coloring each point.

### PCA scatter under the three regimes

| Regime | Each dot is... | PC axes capture... | Applicable? |
|--------|---------------|-------------------|-------------|
| **GAP (B, C)** (ours) | An image (10,000 dots) | Dominant channel-level variation | Yes |
| **Flatten (B, C×H×W)** | An image (10,000 dots) | Dominant channel+spatial variation | Yes |
| **Channel-centric (C, B×H×W)** | A channel (64 dots) | Dominant spatial co-activation variation | Technically yes, but only 64 points — not very informative as a scatter |

**Flatten** — Each image's representation is 4096-dimensional. PCA finds principal components that mix channel and spatial information. The scatter plot would still work mechanically (project onto top-2 PCs, plot), but:
- **Spatial structure may dominate the top PCs** — positional patterns (objects centered, backgrounds varying) rather than purely semantic patterns
- **Class clusters might appear tighter** (spatial structure helps distinguish classes) **or noisier** (within-class spatial jitter dominates)
- **Cross-model comparison becomes harder** because the teacher's PCA basis includes spatial directions that the student might handle differently, even if both encode the same semantic content

**Channel-centric** — Each dot would be a *channel* (64 of them at layer 3), projected onto the two dominant spatial co-activation axes. With only 64 points and no class labels to color them by, this scatter plot would be far less informative — you could color by channel index, but there's no natural grouping. The scatter format is not well-suited to this regime; the other metrics (PCA variance, principal angles, ICA) are more informative for channel-centric analysis.

Under GAP, the scatter plot shows a purely channel-level view: how does each model's 64-channel summary organize images? This is cleanest for visual comparison because spatial variation has been removed and each dot is a class-labeled image.

---

## The Critical Design Choice: Shared Coordinate System

### All four panels use the teacher's PCA basis

This is the most important design decision in this figure and is easy to miss.

```python
# from analyze.py, lines 596-599
X_teacher = reps[MODEL_KEYS[0]][ref_seed][layer]
teacher_mean = X_teacher.mean(axis=0)
_, vecs, _ = pca(X_teacher)
proj = vecs[:, :2]  # top 2 eigenvectors
```

1. PCA is computed **only on the teacher's** layer 3 representation
2. The top 2 eigenvectors (`proj`, shape 64×2) define the 2D projection "screen"
3. The teacher's mean is used as the centering point

Then **every model** — teacher and all students — is projected using this same basis:

```python
# from analyze.py, lines 606-608
X = reps[key][ref_seed][layer]
labels = reps[key][ref_seed]['labels']
X_proj = (X - teacher_mean) @ proj  # project onto teacher's PCA basis
```

The subtraction of `teacher_mean` centers the data at the teacher's origin. The matrix multiplication `@ proj` projects the 64-dimensional representation onto the teacher's two principal directions.

### Why this matters

If each model were projected onto its **own** PCA basis, the four panels would be incomparable. Each model's PC 1 might point in a completely different 64-dimensional direction. Cluster positions would be arbitrary — "upper left" in one panel has no relationship to "upper left" in another.

By using a shared basis (the teacher's), all panels share the same coordinate axes. If class 5 (red dots) forms a cluster at coordinates (3, -2) in the teacher's panel, you can meaningfully ask: does class 5 also land near (3, -2) in the student's panel?

### What this biases

This projection is **inherently biased toward the teacher**. The two directions shown are the teacher's most important directions, not necessarily the student's. A student that organizes information along completely different axes might look "noisy" or "unstructured" in the teacher's basis, even if its own representation is perfectly organized — just along different directions.

This bias is deliberate: the question being asked is "does the student look like the teacher?" not "is the student well-organized in general?" The other metrics (Fisher, CKA) answer the latter question without projection bias.

---

## What PC 1 and PC 2 Represent

PC 1 is the single direction of maximum variance in the teacher's 64-dimensional layer 3 representation. It captures whatever distinction the teacher considers most important — likely the coarsest class separation (e.g., animals vs. vehicles, or broad category boundaries).

PC 2 is the direction of maximum variance *perpendicular* to PC 1. It captures the second most important distinction.

Together, PC 1 and PC 2 show the 2D "best summary" of the teacher's representation. They typically capture 30-60% of the total variance (check `pca_variance.png` at layer 3, x=2 for the exact number). The remaining 40-70% of structure is invisible in this 2D projection but is captured by the other metrics.

The actual semantics of PC 1 and PC 2 (what visual distinction they encode) are emergent from training — they're not predefined. You can sometimes guess by looking at which classes separate along each axis.

---

## How to Read the Scatter Plots

### Class clusters

- **Tight, well-separated colored clusters:** The representation clearly distinguishes those classes along the teacher's top-2 directions. Each class occupies a distinct region.

- **Overlapping or diffuse clusters:** Classes aren't well-separated in this 2D view. This doesn't necessarily mean the model is bad — separation might exist along PC 3, 4, ..., 64, which are invisible here. But if the Fisher criterion (which considers all dimensions) is also low, then the model genuinely struggles to separate classes.

- **Same cluster positions across panels:** The models organize these classes the same way. If blue dots (class 0) are always in the upper-right across all four panels, the models agree on the structure of the two dominant directions. This is visual confirmation of what CKA measures numerically.

- **Different cluster positions across panels:** The student organizes classes differently from the teacher along these two directions. Even if the student separates classes well in its own basis, it's doing so via different feature combinations.

### Point density and spread

- **Dense, tight cluster:** All images of that class have very similar representations (at least in this 2D projection). Low within-class variance along PC 1 and PC 2.

- **Diffuse, spread-out cluster:** More variability within the class. Some "cat" images might end up near "dog" clusters even if the average cat representation is far from the average dog.

- **Elongated clusters:** The representation captures within-class variation along a specific axis. For example, an elongated "automobile" cluster stretched along PC 1 might encode viewpoint changes (front view vs. side view) within the automobile class.

### Semi-transparency and density

Points have alpha=0.4, meaning they're semi-transparent. Where many points overlap, the color appears **darker/more saturated**. This reveals the density structure:
- **Dark core with faded edges:** Most images of that class map to a tight region, with outliers spreading further.
- **Uniformly faded:** The class is diffusely spread across the projection.

---

## Why Only Layer 3

Layer 3 is the deepest representation layer — the most processed, closest to the classification output. It's where you most expect to see class-specific structure. Earlier layers encode more generic features (edges, textures) that don't separate classes as clearly.

The model's actual classifier operates on the GAP-pooled layer 3 output, so this is the representation that matters most for classification performance.

---

## Why Only One Seed

Scatter plots can't be meaningfully averaged across seeds. Each seed produces a different model with slightly different PCA directions, so the teacher's PCA basis differs between seeds. Averaging point positions across different coordinate systems would be meaningless.

Seed 0 is used as a representative example. The quantitative comparisons (CKA, Fisher, etc.) handle multi-seed averaging properly.

---

## Why Only 10 Classes for CIFAR-100

```python
# from analyze.py, lines 587-590
if dataset == 'Cifar10':
    selected_classes = list(range(10))
else:
    selected_classes = list(range(0, 100, 10))  # classes 0, 10, 20, ..., 90
```

CIFAR-100 has 100 classes. Plotting all 100 with 10 visually distinct colors (tab10 colormap only has 10) would create an unreadable mess — 10 classes would share each color. Instead, 10 classes are sampled (every 10th: 0, 10, 20, ..., 90) to give a readable visualization with distinct colors while still spanning the full range of the dataset.

This is a visualization choice, not an analysis choice — the other metrics (CKA, Fisher, etc.) use all classes.

---

## Relationship to Other Figures

This is the qualitative, visual companion to the quantitative metrics:

| Metric | Numerical result | This figure shows... |
|--------|-----------------|---------------------|
| CKA = 0.85 | "High similarity" | *Why:* same clusters in same places |
| Fisher = 2.3 | "Good separability" | *Whether:* clusters are visually distinct |
| Principal angles near 0° | "Aligned subspaces" | *How:* the first 2 directions in action |
| Effective dim = 8 | "Compact representation" | *Whether:* a 2D slice captures most of the story |

It's deliberately the simplest figure in the set: raw data, minimal processing, maximum visual intuition. It grounds the abstract numbers in something directly observable.

---

## Why This Matters for Your Thesis

This figure provides **visual evidence** that either supports or contradicts the numerical metrics. A thesis reader who doesn't deeply understand CKA or principal angles can look at this figure and immediately see whether the student's representation "looks like" the teacher's. If the numerical metrics say "high alignment" and the scatter plots show the same class arrangement across panels, that's powerful and intuitive confirmation. If they disagree, that reveals something interesting about what the metrics do and don't capture.
