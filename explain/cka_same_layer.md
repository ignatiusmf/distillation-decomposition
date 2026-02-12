# CKA Same-Layer Bar Chart — `cka_same_layer.png`

## Graph Layout

- **Panels:** 1 (single chart)
- **X-axis:** Three groups, one per layer (Layer 1 — 16 ch, Layer 2 — 32 ch, Layer 3 — 64 ch)
- **Y-axis:** Linear CKA value (0 to ~1.15)
- **Bars:** 3 per group, one per *student* model (teacher is excluded — CKA of teacher with itself is trivially 1.0):
  - **Red** — Student - No KD
  - **Orange** — Student - Logit KD
  - **Green** — Student - Factor Transfer
- **Error bars:** ±1 standard deviation across 3 seeds (black caps)
- **Text labels:** Mean CKA value printed above each bar
- **Grid:** Horizontal gridlines on y-axis for easier reading

---

## The Representation: What Goes In

Same as `cka_cross_layer.md` — GAP-pooled representations of shape **(10000, C)**. But unlike the cross-layer heatmaps, this figure only compares **matching layers**: teacher layer 1 vs. student layer 1, teacher layer 2 vs. student layer 2, teacher layer 3 vs. student layer 3.

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

At each layer, both the teacher and student representations have the same number of channels (16, 32, or 64), because both architectures (ResNet-112 and ResNet-56) use the same channel widths — they differ only in depth (number of blocks per layer group).

### Same-layer CKA under the three regimes

For same-layer comparisons, teacher and student always have the same channel count and spatial resolution, so all three regimes are straightforward:

| Regime | Shape at layer 3 | What CKA measures |
|--------|------------------|-------------------|
| **GAP (B, C)** (ours) | (10000, 64) vs (10000, 64) | Do images relate the same way in channel-summary space? |
| **Flatten (B, C×H×W)** | (10000, 4096) vs (10000, 4096) | Do images relate the same way in spatially-detailed space? |
| **Channel-centric (C, B×H×W)** | (64, 640000) vs (64, 640000) | Do channels co-activate in the same spatial patterns? |

**Flatten** — CKA values would differ from GAP because spatial alignment is now part of the similarity structure. Two models detecting "vertical edges" at slightly different positions show lower CKA under flattening (spatial misalignment reduces agreement) but identical CKA under GAP (spatial differences averaged away). Comparing GAP-CKA vs. flatten-CKA directly tells you how much alignment is at the "what" level vs. the "what+where" level.

**Channel-centric** — CKA with N=C samples (64 at layer 3). This compares how channels relate to each other across spatial positions, rather than how images relate to each other. A high value means teacher and student channels have the same co-activation structure — channels that fire together spatially in the teacher also fire together in the student. The small N is a concern for reliability (see `cka_cross_layer.md` for details).

---

## What This Figure Measures

This is a simplified extraction from the cross-layer CKA heatmaps (see `cka_cross_layer.md`). It takes only the **diagonal** entries — where the teacher and student layer indices match — and displays them as bars for clean comparison.

Each bar answers: **at this specific layer, how similarly does this student represent the relationships between images compared to the teacher?**

### The computation per bar

```python
# from analyze.py, lines 339-346
for i, s_key in enumerate(students):
    vals_per_seed = []
    for seed in seeds:
        vals = [linear_cka(reps[teacher][seed][l], reps[s_key][seed][l])
                for l in LAYERS]
        vals_per_seed.append(vals)
    arr = np.array(vals_per_seed)
    mean_v = arr.mean(axis=0)
    std_v = arr.std(axis=0)
```

For each student model and each layer, `linear_cka()` is called with the teacher's and student's representation at that layer and that seed. This produces one CKA value per seed per layer. The three seed values are averaged (`mean`) and their standard deviation computed (`std`).

The CKA function itself (see `cka_cross_layer.md` for full explanation):

```python
# from analyze.py, lines 122-128
def linear_cka(X, Y):
    X = X - X.mean(0)    # center
    Y = Y - Y.mean(0)    # center
    num = np.linalg.norm(Y.T @ X, 'fro') ** 2    # cross-alignment
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')  # self-alignments
    return num / den
```

---

## How to Read the Bars

- **A bar near 1.0:** The student's representation at that layer is structurally nearly identical to the teacher's. The same images that are similar/dissimilar in the teacher's space are similarly similar/dissimilar in the student's space.

- **A bar near 0.5:** Moderate alignment. There's shared structure, but substantial differences. The student organizes some image relationships the same way as the teacher and others differently.

- **A bar near 0.0:** The representations are essentially unrelated in their pairwise similarity structure.

- **Comparing bars within a group** (same layer, different students): directly answers which training regime produces representations most similar to the teacher's. If Logit KD is higher than No KD at layer 3, logit distillation causes the student to better mirror the teacher's late-stage representation structure.

- **Comparing bars across groups** (same student, different layers): tells you at which depth the alignment is strongest or weakest. Often different patterns emerge: early layers may show high alignment (because early features are more universal — edges, textures) while later layers show more divergence (because later features are more task-specific and architecture-dependent).

---

## Why This Is Separate from the Heatmaps

The cross-layer heatmap shows all 9 layer-pair combinations, which reveals depth-alignment patterns (does student layer 2 look more like teacher layer 1 or teacher layer 3?). But for the specific question "does distillation make the student's corresponding layers match the teacher's?", the diagonal comparison is cleaner. This bar chart isolates exactly that question and makes cross-method comparison easy at a glance.

---

## Relationship to Other Metrics

CKA, principal angles, and ICA all measure representation similarity but at different levels:

| Metric | Question | Level |
|--------|----------|-------|
| **CKA** (this) | Do images relate to each other the same way? | Global similarity structure |
| **Principal angles** | Do the subspaces point in the same directions? | Geometric subspace alignment |
| **ICA** | Can you match individual components one-to-one? | Individual feature correspondence |

CKA is the broadest — it considers all 10,000 images simultaneously and asks whether the full relational pattern matches. Principal angles drill into the geometry of the feature directions. ICA goes finest, looking for specific component-level matches. They complement each other: high CKA with low ICA matching means "similar overall pattern, achieved through different underlying features."

---

## Why This Matters for Your Thesis

This is the most direct single-number answer to "does this student's layer look like the teacher's layer?" Comparing the three student bars tells you whether distillation (and which kind) makes the student's representation more teacher-like. It's the simplest, most interpretable representation similarity metric, and it provides the headline result that the other metrics (principal angles, ICA) then decompose into finer detail.
