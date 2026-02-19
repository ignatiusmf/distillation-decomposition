# Class Separability (Fisher Criterion) — `class_separability.png`

## Graph Layout

- **Panels:** 1 (single chart)
- **X-axis:** Three groups, one per layer (Layer 1 — 16 ch, Layer 2 — 32 ch, Layer 3 — 64 ch)
- **Y-axis:** "Between-class / Within-class variance" (the Fisher criterion ratio, no upper bound)
- **Bars:** 4 per group, one per model (including the teacher this time):
  - **Blue** — Teacher (ResNet-112)
  - **Red** — Student - No KD
  - **Orange** — Student - Logit KD
  - **Green** — Student - Factor Transfer
- **Error bars:** ±1 standard deviation across 3 seeds
- **Text labels:** Mean Fisher value printed above each bar
- **Grid:** Horizontal gridlines on y-axis

---

## The Representation: What Goes In

Same GAP-pooled representations of shape **(10000, C)**:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

Crucially, this figure also uses the **class labels** — it's the only metric (alongside PCA scatter) that looks at what the images *are*, not just how they're represented.

### Fisher criterion under the three regimes

| Regime | Fisher measures | Applicable? |
|--------|----------------|-------------|
| **GAP (B, C)** (ours) | Class separability in channel-summary space | Yes — each image has a label |
| **Flatten (B, C×H×W)** | Class separability in spatially-detailed space | Yes — each image has a label |
| **Channel-centric (C, B×H×W)** | — | **No** — channels don't have class labels |

**Flatten** — Fisher would measure class separability in the full spatially-detailed space. This would typically yield **higher** Fisher values because spatial structure provides additional discriminative information (objects at different positions, class-specific spatial patterns). However, the within-class scatter also increases (different images of the same class have spatial jitter), and comparisons between models become less clean — spatial alignment differences between teacher and student would appear as separability differences even if both detect the same features.

**Channel-centric** — Fisher is **not applicable**. The Fisher criterion requires class labels on the samples. Under channel-centric, the samples are *channels* (64 of them), and channels don't belong to image classes. You could ask a different question — "do certain channels specialize for certain classes?" — but that requires a different analysis (e.g., computing per-channel, per-class mean activations), not the Fisher ratio.

Under GAP, Fisher measures separability purely in channel space: **how well do channel-level averages separate the classes?** This isolates whether the model has learned features that distinguish classes, independent of spatial organization.

---

## What the Fisher Criterion Measures

All previous metrics (PCA, CKA, principal angles, ICA) are **unsupervised** — they examine representation structure without knowing what the images contain. The Fisher criterion is **supervised** — it asks: **given that we know the class labels, how well does this representation separate the classes?**

### The intuition

Imagine plotting all 10,000 representations in C-dimensional space, colored by class. A "good" representation would show tight, well-separated clusters — all cats in one region, all dogs in another, with clear gaps between them. A "bad" representation would show classes overlapping — cats and dogs intermixed.

The Fisher criterion quantifies this with a ratio:

    Fisher = (how spread out are the class centers?) / (how spread out are points within each class?)

High ratio = well-separated, compact clusters. Low ratio = overlapping, diffuse clusters.

### The computation — step by step

```python
# from analyze.py, lines 140-150
def fisher_criterion(X, labels):
    """Trace(S_between) / Trace(S_within)."""
    classes = np.unique(labels)
    mu = X.mean(axis=0)
    Sb, Sw = 0.0, 0.0
    for c in classes:
        Xc = X[labels == c]
        mc = Xc.mean(axis=0)
        Sb += len(Xc) * np.sum((mc - mu) ** 2)
        Sw += np.sum((Xc - mc) ** 2)
    return Sb / Sw if Sw > 0 else 0.0
```

#### Step 1: Global mean

```python
mu = X.mean(axis=0)
```

Average all 10,000 representation vectors to get a single C-dimensional vector. This is the center of the entire data cloud — the "average image representation."

#### Step 2: Class means

```python
Xc = X[labels == c]
mc = Xc.mean(axis=0)
```

For each class c, select only the images belonging to that class and compute their mean representation. For CIFAR-10, this gives 10 class means; for CIFAR-100, 100 class means. Each class mean is a C-dimensional vector representing the "prototype" for that class.

#### Step 3: Between-class scatter (S_b)

```python
Sb += len(Xc) * np.sum((mc - mu) ** 2)
```

For each class, compute the squared Euclidean distance between its mean `mc` and the global mean `mu`:

    ||mc - mu||² = Σ_d (mc[d] - mu[d])²

This measures how far this class's center is from the overall center, summed across all channels. Multiply by `len(Xc)` (the number of images in class c) to weight the contribution by class size.

Sum across all classes to get the total between-class scatter:

    S_b = Σ_c  n_c · ||μ_c - μ||²

**What S_b captures:** How dispersed the class prototypes are. If every class mean is at the same location (all prototypes clumped together), S_b = 0 and the representation doesn't distinguish classes at all. If prototypes are far apart, S_b is large and each class occupies a distinct region.

The weighting by n_c means that a class with more samples contributes more to S_b. For CIFAR-10 (1000 images per class in the test set), all classes are equally weighted. For CIFAR-100 (100 images per class), likewise.

#### Step 4: Within-class scatter (S_w)

```python
Sw += np.sum((Xc - mc) ** 2)
```

For each class c, compute the sum of squared distances from each image's representation to its class mean:

    S_w(c) = Σ_{x in class c} ||x - μ_c||²

This measures how spread out the individual representations are around their class center. Sum across all classes:

    S_w = Σ_c Σ_{x in c} ||x - μ_c||²

**What S_w captures:** The total "noise" within classes. Even if class centers are far apart, classification is difficult if individual images are so spread out that they overlap with neighboring classes. Low S_w means compact, tight clusters; high S_w means diffuse, sprawling clusters.

#### Step 5: The ratio

```python
return Sb / Sw if Sw > 0 else 0.0
```

    Fisher = S_b / S_w

Neither S_b nor S_w alone is informative:
- Large S_b (far apart centers) + even larger S_w (huge within-class spread) = overlapping clusters despite separated centers
- Small S_b (close centers) + tiny S_w (very tight clusters) = well-separated despite close centers

The ratio captures the fundamental trade-off: **classes need to be far apart relative to their internal spread.**

### A concrete example

Two classes in 1D representation space:
- Class A: centered at 5.0, std = 1.0 (100 samples)
- Class B: centered at 10.0, std = 1.0 (100 samples)

Global mean = 7.5. S_b = 100 · (5 - 7.5)² + 100 · (10 - 7.5)² = 100 · 6.25 + 100 · 6.25 = 1250. S_w ≈ 100 · 1² + 100 · 1² = 200 (each sample deviates ~1 from its class mean). Fisher ≈ 1250/200 = 6.25. Well-separated.

Now make both classes have std = 4.0: S_w ≈ 100 · 16 + 100 · 16 = 3200. Fisher ≈ 1250/3200 = 0.39. Heavily overlapping. Same class centers, but too much within-class spread.

### Note: this uses Trace, not the full matrix formulation

The standard Fisher discriminant uses full scatter *matrices* (S_b and S_w are D × D), and the criterion is often Tr(S_w^{-1} S_b). Our simplified version uses only the **traces** — the sum of diagonal elements — which reduces to scalar between/within variance sums. This is computationally simpler and avoids numerical issues when S_w is singular (which happens when D > N_class), but it ignores channel *correlations*. It measures separability in a "per-channel, summed" sense rather than accounting for how channels jointly contribute to separation.

---

## How to Read the Bars

- **A tall bar** means that representation separates classes well at that layer. Class clusters are compact and far apart. Higher values generally correlate with better classification accuracy, though the relationship isn't perfectly linear.

- **A short bar** means classes overlap substantially at that layer. The representation hasn't (yet) organized itself to distinguish classes. At early layers, this is expected — early features are generic.

- **Bars increasing from Layer 1 to Layer 3** (the typical pattern): the network progressively develops class-specific structure. Early layers encode generic features; deeper layers organize for classification. This is the expected hierarchical behavior.

- **Comparing bars within a group** (same layer, different models): which model best separates classes at that depth? The teacher might have higher values due to its greater capacity. Whether the distilled student improves over the independently-trained student tells you whether distillation transfers class-separating structure.

- **Error bars:** Small = class separation is consistent across training runs. Large = depends on initialization.

### The multi-seed computation

```python
# from analyze.py, lines 549-557
for i, key in enumerate(MODEL_KEYS):
    vals_per_seed = []
    for seed in seeds:
        labels = reps[key][seed]['labels']
        vals = [fisher_criterion(reps[key][seed][l], labels) for l in LAYERS]
        vals_per_seed.append(vals)
    arr = np.array(vals_per_seed)
    mean_v = arr.mean(axis=0)
    std_v = arr.std(axis=0)
```

Fisher criterion is computed separately for each seed, then averaged. Note that the labels come from `reps[key][seed]['labels']` — the test set labels for that seed's data loader (which should be identical across seeds since the test set doesn't change, but the representations differ).

---

## Relationship to Accuracy

The Fisher criterion and test accuracy are related but not identical:

- **Fisher** measures geometric separation in the *intermediate* representation (layers 1-3, after GAP pooling)
- **Accuracy** measures performance of the *full pipeline* (all layers + final linear classifier on top of GAP-pooled layer 3)

A model could have moderate Fisher at layer 3 but still achieve high accuracy because:
1. The final linear layer can draw hyperplane decision boundaries that separate classes even when Fisher isn't high
2. Fisher uses the simplified trace ratio, ignoring beneficial channel correlations the classifier exploits
3. Fisher is computed on GAP-pooled features, and the model's classifier is also applied after GAP — but they measure different things (geometric scatter ratio vs. classification boundary)

However, higher Fisher values generally make the classifier's job easier — well-separated clusters are easier to classify than overlapping ones, regardless of the classifier.

---

## Why This Matters for Your Thesis

The previous metrics ask "do teacher and student representations have similar *structure*?" This metric asks "do they have similar *functional utility*?" If distillation increases the Fisher criterion compared to independent training, the distillation signal helps the student organize its representations in a more classification-friendly way. This connects structural alignment (CKA, principal angles) to functional benefit (class separability), completing the picture: distillation doesn't just make representations *look* similar, it makes them *work* similarly.
