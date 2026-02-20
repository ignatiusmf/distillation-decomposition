# Experiment Alpha

Initial limited analysis run. Archived 2026-02-20.

## Scope

This was the first pass — limited to 2 datasets and 3 training conditions to verify the analysis pipeline works before scaling up.

- **Datasets:** CIFAR-10, CIFAR-100
- **Methods:** pure (no KD), logit KD, factor transfer KD
- **Seeds:** 0, 1, 2 (3 seeds per condition)
- **Teacher:** ResNet-112, **Student:** ResNet-56

Total: 2 datasets × (1 teacher + 3 student conditions) × 3 seeds = **24 representation extractions** (12 per dataset).

## Training Configuration

Same as experiment beta (see `analysis/experiment_beta/README.md` for full details). In brief:
- 150 epochs, SGD (LR=0.1, cosine annealing), batch=128, weight_decay=5e-4, label_smoothing=0.1
- KD alpha=0.5, temperature=4.0

## Representations Extracted

Located in `representations/`. Each `.npz` (~4.8MB for CIFAR-10, ~8.6MB for CIFAR-100) contains GAP-pooled layer outputs from the trained models evaluated on the **test set**.

**Contents of each .npz:**
- `layer1` — shape [N_test, 16] — GAP of first layer group output
- `layer2` — shape [N_test, 32] — GAP of second layer group output
- `layer3` — shape [N_test, 64] — GAP of third layer group output
- `logits` — shape [N_test, C] — classifier output (C=10 or C=100)
- `labels` — shape [N_test] — ground truth labels

Extracted using `extract.py` (included in this directory). The script loads `best.pth` from each experiment, runs a forward pass on the test set, applies GAP to each layer group's spatial feature maps, and saves the result.

**CIFAR-10 files (12):**
```
representations/Cifar10/
├── teacher_ResNet112_seed{0,1,2}.npz
├── student_ResNet56_pure_seed{0,1,2}.npz
├── student_ResNet56_logit_seed{0,1,2}.npz
└── student_ResNet56_factor_seed{0,1,2}.npz
```

**CIFAR-100 files (12):**
```
representations/Cifar100/
├── teacher_ResNet112_seed{0,1,2}.npz
├── student_ResNet56_pure_seed{0,1,2}.npz
├── student_ResNet56_logit_seed{0,1,2}.npz
└── student_ResNet56_factor_seed{0,1,2}.npz
```

## Analysis Performed

Run using `analyze.py` (included in this directory). For each dataset, the script loads all representations and computes the following metrics, producing one figure per metric:

Located in `figures/{Cifar10,Cifar100}/`:

| Figure | Metric | What It Shows |
|--------|--------|---------------|
| `pca_variance.png` | Cumulative PCA explained variance | How many components capture 90%/95%/99% of variance per layer per model |
| `effective_dim.png` | Participation ratio (effective dimensionality) | How many dimensions are "active" — higher = more distributed representation |
| `cka_same_layer.png` | Linear CKA (same layer) | Representational similarity between model pairs at matching layers |
| `cka_cross_layer.png` | Linear CKA (cross-layer heatmaps) | Full layer×layer similarity matrices between model pairs |
| `principal_angles.png` | Principal angles between subspaces | Angular alignment of top-k PCA subspaces between models |
| `ica_correlation.png` | ICA component correlation matrices | How well independent components match between models |
| `ica_summary.png` | ICA matching summary | Aggregate ICA alignment scores |
| `class_separability.png` | Fisher criterion | How well-separated class clusters are in each layer's representation |
| `pca_scatter.png` | 2D PCA projections | Visual scatter of representations in first 2 PCs, coloured by class |

## Metric Explanations

Detailed write-ups for each analysis metric are in `explain/`. Each file explains the mathematical definition, intuition, how to read the corresponding figure, and what patterns to look for:

| File | Metric |
|------|--------|
| `explain/pca_variance.md` | Cumulative PCA explained variance |
| `explain/effective_dim.md` | Participation ratio / effective dimensionality |
| `explain/cka_same_layer.md` | Linear CKA at matching layers |
| `explain/cka_cross_layer.md` | Linear CKA cross-layer heatmaps |
| `explain/principal_angles.md` | Principal angles between subspaces |
| `explain/ica_correlation.md` | ICA component correlation matrices |
| `explain/ica_summary.md` | ICA matching summary |
| `explain/class_separability.md` | Fisher criterion for class separation |
| `explain/pca_scatter.md` | 2D PCA scatter projections |

## Analysis Scripts

### `extract.py`
- Loads a trained model from `experiments/{dataset}/{method}/{model}/{seed}/best.pth`
- Runs forward pass on test set
- Hooks into each layer group to capture pre-GAP feature maps
- Applies GAP: `mean(feature_map, dim=(H,W))` → C-dimensional vector per sample
- Saves as `.npz` to `representations/`

### `analyze.py`
- Loads all `.npz` files for a dataset
- Computes all 9 metrics listed above
- Generates matplotlib figures to `figures/`
- `print_summary()` function (line ~643) also reports test accuracies computed from the stored logits

## What Alpha Did NOT Cover

- SVHN and TinyImageNet datasets
- attention_transfer, fitnets, rkd, nst KD methods
- Any hyperparameter variations
- These are all covered in experiment beta.
