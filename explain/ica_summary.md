# ICA Summary — `ica_summary.png`

## Graph Layout

- **Panels:** 2 side-by-side
  - **Left:** "Mean Matched |Correlation| (avg over seeds)"
  - **Right:** "Strongly Matched Components (|corr| > 0.5, avg)"
- **X-axis (both):** Three groups, one per layer (Layer 1 — 16 ch, Layer 2 — 32 ch, Layer 3 — 64 ch)
- **Y-axis:**
  - Left: "|Correlation|" (0 to ~1.15)
  - Right: "Count" (number of strongly matched components)
- **Bars:** 3 per group, one per student model:
  - **Red** — No KD
  - **Orange** — Logit KD
  - **Green** — Factor Transfer
- **Error bars:** ±1 standard deviation across 3 seeds
- **Text labels:**
  - Left: Mean correlation value above each bar
  - Right: "mean_count/total" above each bar (e.g., "12.0/64")

---

## The Representation: What Goes In

Same as `ica_correlation.md` — GAP-pooled representations of shape **(10000, C)**, with ICA run separately on teacher and student:

```python
# from extract.py, line 60-61
layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
# (B, C, H, W) -> (B, C)
```

### ICA summary under the three regimes

See `ica_correlation.md` for the full regime comparison. The summary statistics are affected differently:

| Regime | # matched pairs (layer 3) | Summary behavior |
|--------|--------------------------|------------------|
| **GAP (B, C)** (ours) | 64 | Clean, interpretable — "X out of 64 channel-level factors match" |
| **Flatten (B, C×H×W)** | up to 4096 | Sparser matching — most pairs would be weak; "X out of 4096" fractions would be small even with good alignment. The 0.5 threshold may need recalibration |
| **Channel-centric (C, B×H×W)** | 64 | Same count as GAP, but measures spatial-co-activation independence rather than image-level independence. Very well-powered (640K observations). Would directly compare whether teacher and student have the same spatially-independent channel combinations |

Channel-centric is particularly interesting for the summary statistics because it has the same 64-component granularity as GAP (making the "X/64" numbers directly comparable) but captures different information. If GAP-ICA shows 30/64 strongly matched components and channel-centric-ICA shows 45/64, it would mean the spatial co-activation structure is more preserved by distillation than the image-level structure — or vice versa.

---

## What This Figure Summarizes

This figure distills the ICA correlation heatmaps (see `ica_correlation.md`) into two scalar metrics per model-per-layer, **averaged across all three seeds**. It solves the problem that raw heatmaps can only be shown for one seed.

### How the seed-level metrics are computed

```python
# from analyze.py, lines 467-483
for s_key in students:
    for layer in LAYERS:
        seed_metrics = []
        for seed in seeds:
            X_t = reps[teacher][seed][layer]
            X_s = reps[s_key][seed][layer]
            n_comp = X_t.shape[1]
            try:
                matched_corr, _ = ica_cross_correlation(X_t, X_s, n_comp)
            except Exception:
                continue
            seed_metrics.append({
                'mean_corr': matched_corr.mean(),
                'n_strong': int((matched_corr > 0.5).sum()),
                'n_total': n_comp,
            })
        summary_all[(s_key, layer)] = seed_metrics
```

For each (student model, layer, seed) combination:
1. Run `ica_cross_correlation()` — which performs ICA on both representations, builds the cross-correlation matrix, and runs the Hungarian matching (see `ica_correlation.md` for full details)
2. From the matched correlations, compute:
   - `mean_corr`: the average absolute correlation across all matched pairs
   - `n_strong`: how many matched pairs have |correlation| > 0.5
   - `n_total`: total number of components (= C)
3. Store these per-seed metrics, then average across seeds for plotting

---

## Left Panel: Mean Matched Correlation

### What it computes

After the Hungarian algorithm matches each teacher IC to a student IC (see `ica_correlation.md`), you get C matched pairs, each with a correlation value. The "mean matched correlation" is the average of the absolute correlations across all C pairs.

```python
# from analyze.py, lines 492-496
vals = [np.mean([m['mean_corr'] for m in summary_all[(s_key, l)]])
        for l in LAYERS]
stds = [np.std([m['mean_corr'] for m in summary_all[(s_key, l)]])
        for l in LAYERS]
```

### What it means

This measures **average component-level agreement**. For the typical matched teacher-student IC pair, how correlated are their activations?

- **Near 1.0:** On average, every teacher IC has a strong counterpart in the student. The two models have decomposed the representation into very similar independent factors.

- **Near 0.5:** Moderate correspondence. Some components match well, others poorly.

- **Near 0.0 (or below ~0.2):** The matching is essentially random — no meaningful one-to-one correspondences. The models have discovered different independent factors.

### Subtlety: the mean can hide bimodality

Imagine 10 matched pairs where 5 have correlation 0.9 and 5 have correlation 0.1. The mean is 0.5. But the reality is bimodal — half match perfectly, half don't match at all. This is why the right panel exists.

---

## Right Panel: Strongly Matched Components

### What it computes

Count the matched pairs with |correlation| > 0.5. This threshold means the two ICs share at least 25% of their variance (r² = 0.25), which is substantial enough to say they're "detecting similar patterns."

```python
# from analyze.py, lines 512-517
vals = [np.mean([m['n_strong'] for m in summary_all[(s_key, l)]])
        for l in LAYERS]
stds = [np.std([m['n_strong'] for m in summary_all[(s_key, l)]])
        for l in LAYERS]
totals = [summary_all[(s_key, l)][0]['n_total'] for l in LAYERS]
```

The text label shows "mean_count/total" — e.g., "12.0/64" means, on average across seeds, 12 out of 64 ICA components have a strong match.

### What it means

This measures **how many teacher ICs have a genuine counterpart in the student**.

- **"60/64":** Nearly all components match. The student has discovered almost exactly the same set of independent factors as the teacher.

- **"12/64":** Only 12 of 64 factors are shared. The student has "inherited" or "rediscovered" those 12 factors, but the remaining 52 are different between teacher and student.

- **"2/64":** Almost nothing matches. The two models operate on fundamentally different independent factors.

### The fraction is what matters

A raw count of 30 strongly matched components means different things depending on the total:
- 30/64 at layer 3 = 47% match
- 12/16 at layer 1 = 75% match

Even though 30 > 12, the layer 1 result is proportionally stronger. The "count/total" label provides the context needed for correct interpretation.

### Why 0.5 as a threshold

The choice of 0.5 is conventional and has a concrete interpretation:

| |Correlation| | Shared variance (r²) | Interpretation |
|---|----|---|
| 0.0 – 0.2 | 0 – 4% | Essentially unrelated |
| 0.2 – 0.5 | 4 – 25% | Weak correspondence |
| 0.5 – 0.7 | 25 – 49% | Moderate match |
| 0.7 – 0.9 | 49 – 81% | Strong match |
| 0.9 – 1.0 | 81 – 100% | Near-identical |

The 0.5 threshold separates "there's a meaningful relationship" from "the relationship is too weak to be informative." Any pair above 0.5 shares enough variance to say they respond to overlapping patterns.

---

## Why This Is Averaged Across Seeds (But the Heatmaps Aren't)

ICA component *ordering* changes between seeds — IC 5 in seed 0 has no relationship to IC 5 in seed 1 because the FastICA algorithm can discover the same components in any order. Averaging the raw (C × C) heatmaps would produce noise.

But scalar summary statistics like "mean matched correlation" and "count of strong matches" *are* meaningful to average. The mean correlation will be similar across seeds if the underlying independent structure is consistent. Three seeds give a mean and standard deviation, which is what the bars and error bars show.

---

## Relationship to Other Metrics

| Metric | What it asks | Granularity |
|--------|-------------|-------------|
| PCA variance / effective dim | How is variance distributed? | Per-direction magnitude |
| CKA | Do images relate to each other the same way? | Global aggregate |
| Principal angles | Do the variance subspaces point the same way? | Per-direction geometry |
| **ICA summary (this)** | Do the independent factors match one-to-one? | Per-component correspondence |

ICA is the finest-grained metric. High CKA with low ICA matching means "similar overall relational pattern, achieved through different component organization." High ICA matching with moderate CKA would be unusual but would mean "same independent factors, but they combine differently to create the pairwise similarity structure."

---

## Why This Matters for Your Thesis

If distillation increases the number of strongly matched ICA components (compared to independent training), that's the strongest available evidence that distillation transfers *specific feature detectors*, not just statistical summaries. This directly addresses the "what is transferred" question at the most granular level: not just "similar structure" but "the same independent factors."
