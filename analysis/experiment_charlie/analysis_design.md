# Experiment Charlie — Analysis Suite Design

**Task [12] from Plan.md**

Charlie is the definitive experiment. The analysis pipeline must be fully specified before
training starts so all necessary outputs are captured.

*Reference implementation:* `analysis/experiment_alpha/extract.py` + `analyze.py`.
These cover 9 metrics across 3 layers for a small set of models (6 per dataset). Charlie
has ~312 models — the tooling needs to scale.

---

## Scale

- **312 total experiments:** 6 methods x 4 alphas x 4 datasets x 3 seeds = 288 KD + 24 pure baselines
- **Alphas:** 0.25, 0.5, 0.75, 0.95
- **Methods:** logit, factor_transfer, attention_transfer, fitnets, rkd, nst
- **Datasets:** CIFAR-10, CIFAR-100, SVHN, TinyImageNet
- **Seeds:** 0, 1, 2

Alpha's `representations/` held ~24 .npz files (2 datasets x 6 models x [ignored seeds]).
Charlie will have ~312 x 3 layers worth of data. Each (N_test, D) array is small
(e.g. CIFAR-10: 10000 x 64 float32 = 2.5MB), so total is manageable (~2GB for all
datasets combined).

---

## extract.py Design

The extraction step is straightforward:

1. Walk `experiments/` directory tree, find all completed experiments (`status.json` says completed, `best.pth` exists)
2. Load model, run test set, GAP-pool intermediate representations
3. Save as `.npz` with consistent naming

**Directory structure:**
```
experiments/{dataset}/{method}/alpha_{alpha}/{teacher}_to_{model}/{seed}/
experiments/{dataset}/pure/{model}/{seed}/
```

**Output naming:**
```
representations/{dataset}/{method}_a{alpha}_s{seed}.npz   (KD models)
representations/{dataset}/teacher_s{seed}.npz              (teacher baselines)
representations/{dataset}/pure_s{seed}.npz                 (student baselines)
```

**Key change from alpha:** Replace hardcoded `MODEL_CONFIGS` with dynamic directory traversal. Auto-discover all completed experiments via `status.json`.

---

## analyze.py Design

### What alpha taught us about the metrics

The alpha analysis had 4 hardcoded model configs. Charlie has 20+ configs per dataset (6 methods x 4 alphas + 2 baselines). The code needs a **dynamic model registry** — scan the representations directory, parse filenames, build configs automatically.

### Existing 9 Metrics (carry forward, all still answer the same representational questions)

1. **PCA variance curves** — Representation compactness. In charlie: does higher alpha force representations into lower-dimensional subspaces?
2. **Effective dimensionality (90%/95%)** — Scalar from PCA. Most likely to show clean alpha trends. Perfect for alpha sweep plots.
3. **CKA same-layer** — The headline alignment metric. Directly measures teacher-student similarity. Centerpiece of the alpha sweep.
4. **CKA cross-layer heatmaps (3x3)** — Powerful but don't scale to 24+ models. For charlie: per-method figures (subplotted across alphas) or summarise as off-diagonal CKA.
5. **Principal angles** — Geometrically detailed, hard to summarise. Consider reducing to a scalar (mean angle) for sweep analysis.
6. **ICA correlation heatmaps** — Per-seed, don't average — inherently qualitative. Keep summary stats (mean matched correlation, count strong), drop per-model heatmaps. Show one representative per method at alpha=0.5.
7. **ICA summary** — Mean matched correlation + count |r|>0.5. Quantitative.
8. **Class separability (Fisher criterion)** — Supervised metric. Key question: does forcing alignment (high alpha) help or hurt class separation?
9. **PCA scatter (2D)** — Purely visual, single-seed, layer 3 only. Qualitative supplement. For 10-class datasets only (CIFAR-100/TinyImageNet use UMAP instead).

### New Analysis Functions (the alpha sweep)

**10. Alpha sweep line plots** — For each (metric, layer, dataset), plot metric value (y) vs alpha (x). One line per method, shaded +/-1sigma across seeds. Directly answers: "does more KD pressure increase alignment?"

Metrics to sweep: CKA same-layer, effective dim, Fisher criterion, mean ICA matched correlation, mean principal angle. That's 5 metrics x 3 layers x 4 datasets = 60 plots if exhaustive. For the thesis: show layer 3 for all metrics on CIFAR-100 (the hardest task), appendix for rest.

**11. Method x alpha heatmap** — 6x4 grid (method rows, alpha cols). Cell color = metric value. One figure per (dataset, layer, metric). Compact and information-dense.

**12. Accuracy vs alignment scatter** — X = alignment metric (CKA layer 3), Y = test accuracy. Each point = one (method, alpha, seed) combination. Color by method. Tests the thesis question: does better alignment predict better accuracy?

If the scatter shows clear positive correlation: strong evidence for the representation-alignment explanation of KD. If noisy or method-dependent: equally interesting — suggests the linear-alignment lens has limits (material for section 7.3).

**13. Experiment status overview figure** — Visualise the current state of experiment charlie: completion rates, per-method/per-dataset breakdown, accuracy distributions. For supervisor meetings and self-monitoring.

**14. Training dynamics (nice to have)** — CKA computed at checkpoints (e.g. every 30 epochs) to see when alignment develops. Requires train.py to save intermediate checkpoints. Significantly increases storage.

---

## Nonlinear Methods — The Unexplored Half

Every metric in the current pipeline is fundamentally linear. PCA, effective dimensionality (PCA eigenvalues), linear CKA (dot-product kernel), principal angles (SVD), ICA, Fisher criterion — all assume representations live in a flat Euclidean subspace. This isn't wrong, it's a design choice inherited from the thesis framing. But it is a blind spot worth naming explicitly.

KD methods do not operate on linear structure. RKD trains on pairwise distances. NST trains on Gram matrices. FitNets and AT use L2 and attention norms that are sensitive to nonlinear geometry. The linear analysis may be *missing the mechanism* — a student could learn the same semantic manifold as the teacher but rotated or warped, appearing unaligned by CKA while actually being geometrically equivalent.

### 1. Kernel CKA (RBF) vs Linear CKA — the cleanest extension

The current CKA uses a linear kernel: `K(X) = XX^T`. Kornblith et al. (2019) also define an RBF variant: `K(X)_ij = exp(-||xi-xj||^2 / 2sigma^2)`. Switching kernels requires changing ~3 lines of code.

The comparison is the important part:
- **Both high**: representations are linearly aligned (strong case for the thesis claim)
- **Linear low, RBF high**: representations are topologically similar but in different linear coordinates — the student learned the same thing via a nonlinear transformation
- **Both low**: no alignment at any level
- **Linear high, RBF low**: pathological case, probably doesn't occur in practice

This creates a genuine taxonomy of alignment modes. Add a column to the method x alpha heatmap: "delta CKA" = RBF CKA - linear CKA. Methods with positive delta are inducing nonlinear alignment that the linear probe misses.

### 2. UMAP latent space visualization

UMAP (McInnes et al. 2018) preserves local manifold topology. Key application: **teacher-space projection** — fit UMAP on the teacher's layer 3 representations, then transform each student through the *same fitted embedding*. Students aligned with the teacher land near correct teacher clusters.

Dataset-specific strategy:
- **CIFAR-10/SVHN (10 classes)**: PCA scatter is still fine, UMAP optional
- **CIFAR-100 (100 classes)**: Color by superclass (20 superclasses x 5 subclasses = 20 readable colors). Tests whether KD preserves coarse semantic structure.
- **TinyImageNet (200 classes)**: Topology itself is the signal — cluster compactness, separation, whether teacher and student manifolds have the same shape.

Bonus use: UMAP of the *model space* — run UMAP on vectors of per-layer metric values (CKA, effective dim, Fisher) across all 312 models. Might cluster models by method in ways invisible to individual metric comparisons.

**Practical notes**: `umap-learn` package. Our representations are already GAP-pooled to (N, 64) — UMAP on 10k x 64 runs in seconds.

### 3. Intrinsic Dimensionality: TwoNN Estimator

The TwoNN estimator (Facco et al. 2017) estimates intrinsic dimension from the ratio of first and second nearest-neighbour distances. ~10 lines of numpy. The gap between PCA effective dim and TwoNN intrinsic dim measures *manifold curvature*.

KD question: does distillation change the intrinsic dimensionality separately from the linear effective dimensionality? Logit KD (outputs only) may increase linear alignment without changing intrinsic dimension, while FitNets (intermediate activations) may change actual manifold shape.

### 4. RSA with Spearman Correlation

Compute N x N pairwise distance matrix for each model's representations (RSM). Compare teacher RSM to student RSM using Spearman rank correlation. Fully nonlinear (rank-based), captures ordinal geometric structure.

Direct connection to RKD: RKD explicitly trains the student to match the teacher's pairwise distances. Spearman RSM correlation is the evaluation metric that directly tests whether RKD achieves its stated objective. If RKD at alpha=0.95 has high Spearman but lower linear CKA, it means RKD aligns relational structure without replicating the exact linear subspace.

### 5. k-NN Probing as Nonlinear Fisher Complement

Fisher criterion measures linear class separability. k-NN classifier measures nonlinear separability. Running both and comparing:
- **Fisher high, k-NN high**: linearly well-structured
- **Fisher low, k-NN high**: nonlinearly separable but linear probe misses it
- **Fisher high, k-NN low**: unusual
- **Fisher low, k-NN low**: representations do not separate classes

The second case is particularly interesting for KD: a student might sacrifice linear separability to match the teacher's geometry while retaining discriminability in a nonlinear sense.

Implementation: scikit-learn `KNeighborsClassifier(n_neighbors=5)`. Runs in seconds.

### What to Skip

- **t-SNE**: UMAP is strictly better — more faithful to global structure, reproducible, faster.
- **Persistent homology / TDA**: Requires expertise to interpret, hard to calibrate, no clear hypothesis.
- **MINE / mutual information**: Neural estimators noisy, k-NN estimators add hyperparameters.
- **Deep CCA / kernel CCA**: Overkill given linear CKA + RBF CKA covers the essential comparison.

### The Thesis Framing

If nonlinear analysis is included, the natural framing is side-by-side: "we find that [method X] shows low linear CKA but high kernel CKA, suggesting alignment at a nonlinear geometric level." This reframes a potential negative result into a nuanced positive finding.

Alternatively, if all methods show consistent linear and nonlinear alignment (or neither), that validates the linear analysis as sufficient.

**These methods should be included in initial exploration but flagged as optional for the final thesis** — include in analyze.py as separate functions, run on all data, decide based on results.

---

## Figure Organisation

```
analysis/experiment_charlie/
├── extract.py
├── analyze.py
├── representations/{dataset}/{method}_a{alpha}_s{seed}.npz
└── figures/{dataset}/
    ├── per_method/{method}/
    │   ├── pca_variance.png
    │   ├── cka_cross_layer.png
    │   └── ...
    ├── sweeps/
    │   ├── alpha_sweep_cka_same_layer.png
    │   ├── alpha_sweep_effective_dim.png
    │   ├── alpha_sweep_fisher.png
    │   ├── method_alpha_heatmap_cka.png
    │   └── accuracy_vs_alignment.png
    └── summary/
        ├── pca_scatter.png
        ├── umap_teacher_space.png
        └── accuracy_table.txt
```

---

## What Maps to the Thesis

For Chapter 7 (~10 figures budget):
- **7.1**: 2-3 alpha sweep line plots (CKA, effective dim, Fisher — layer 3, CIFAR-100), 1 method x alpha heatmap
- **7.2**: 1 accuracy vs alignment scatter, 1 CKA same-layer bar chart (best alpha per method vs baselines)
- **7.3**: 1-2 figures showing where alignment doesn't predict accuracy (specific methods/layers where the lens fails)
- **Appendix**: Full per-method deep dives, all datasets, all layers

---

## Implementation Priority

1. **extract.py** — Must work before anything else. Auto-discovery, robust to incomplete experiments.
2. **Alpha sweep plots** (Tier 2) — Highest-value new analysis. Directly thesis-relevant.
3. **Accuracy vs alignment scatter** — The capstone figure.
4. **Method x alpha heatmaps** — Compact comparison.
5. **Adapted per-method metrics** (Tier 1) — Important but lower priority; the alpha sweep captures most of the signal in scalar form.
6. **Nonlinear methods** — Run after linear pipeline is working. Decide on inclusion based on results.
7. **Summary table / print output** — Quick sanity check.

---

## Open Questions

- Should we compute CKA between *all pairs of students* (not just teacher-student)? This would show whether different KD methods converge to similar representations regardless of mechanism. Expensive but interesting.
- Should the accuracy metric come from stored logits (like alpha) or from `status.json` max_acc? Using logits is more principled (same test set, computed during extraction), but requires extraction to run first.
- ICA is slow and has convergence issues. For 312 models x 3 layers, that's ~936 ICA fits. May need to be selective (e.g. only seed 0, or only CIFAR-100).
- Does the linear vs nonlinear CKA gap (kernel CKA - linear CKA) correlate with KD method type? Prediction: relational methods (RKD) and attention-based methods (AT, NST) show larger gaps than output-matching methods (logit KD).

---

## Status

**Not yet implemented.** This is the design-phase document that gates charlie analysis. Training is in progress (20/312 completed as of 2026-02-21). Implementation starts when sufficient experiments complete.
