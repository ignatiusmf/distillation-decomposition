# Experiment Alpha

First round of training experiments. Archived 2026-02-20.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Batch size | 128 |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Learning rate | 0.1 |
| LR schedule | CosineAnnealingLR |
| Label smoothing | 0.1 |
| KD alpha | 0.5 (loss = 0.5×CE + 0.5×distill) |
| KD temperature | 4.0 |
| Teacher | ResNet-112 (18 blocks/group) |
| Student | ResNet-56 (9 blocks/group) |

## Accuracy Results

### CIFAR-10

| Model / Method | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|----------------|--------|--------|--------|------------|
| Teacher (ResNet-112) | 92.54% | 94.21% | 93.89% | 93.55 ± 0.76% |
| Student pure (ResNet-56) | 93.11% | 93.29% | 93.19% | 93.20 ± 0.08% |
| Logit KD | 92.35% | 93.97% | 93.80% | 93.37 ± 0.73% |
| Factor Transfer | 93.74% | 93.57% | 93.89% | 93.73 ± 0.13% |
| Attention Transfer | 94.12% | — | — | — |
| FitNets | 93.84% | — | — | — |
| NST | 93.62% | — | — | — |
| **RKD** | **18.13%** | — | — | **FAILED** |

### CIFAR-100

| Model / Method | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|----------------|--------|--------|--------|------------|
| Teacher (ResNet-112) | 71.59% | 72.88% | 72.43% | 72.30 ± 0.65% |
| Student pure (ResNet-56) | 70.43% | 71.54% | 71.54% | 71.17 ± 0.59% |
| Logit KD | 69.39% | 72.96% | 73.16% | 71.84 ± 1.73% |
| Factor Transfer | 71.20% | 70.64% | 72.13% | 71.32 ± 0.63% |
| Attention Transfer | 71.56% | — | — | — |
| FitNets | 70.23% | — | — | — |
| NST | 71.46% | — | — | — |
| **RKD** | **1.00%** | — | — | **FAILED** |

### TinyImageNet (partial)

| Model / Method | Seed 0 | Status |
|----------------|--------|--------|
| Teacher (ResNet-112) | 53.31% | in-progress (epoch 127/150) |
| Student pure (ResNet-56) | 55.65% | completed |
| Logit KD | 36.27% | in-progress (epoch 58/150) |

### SVHN

All 8 experiments failed at initialization: `ModuleNotFoundError: No module named 'scipy'`.

## Known Issues

1. **RKD catastrophic failure** — 18% on CIFAR-10 and 1% on CIFAR-100. Likely implementation or hyperparameter issue.
2. **SVHN never ran** — scipy dependency missing from environment.
3. **Incomplete seeds** — AT, FitNets, RKD, NST only have seed 0 (seeds 1-2 missing).
4. **TinyImageNet incomplete** — only pure ResNet-56 finished; teacher and logit still in-progress.
5. **Small accuracy gaps** — KD methods typically within ~0.5-1% of pure student, making analysis differences subtle.

## Extracted Representations

Located in `representations/`. Only CIFAR-10 and CIFAR-100 extracted (pure, logit, factor_transfer × 3 seeds each).

| Dataset | Files | Models Covered |
|---------|-------|----------------|
| Cifar10 | 12 .npz files (~57MB) | teacher×3, pure×3, logit×3, factor×3 |
| Cifar100 | 12 .npz files (~57MB) | teacher×3, pure×3, logit×3, factor×3 |

Each .npz contains: `layer1`, `layer2`, `layer3`, `logits`, `labels` (GAP-pooled representations).

## Analysis Figures

Located in `figures/`. 9 figures per dataset (CIFAR-10, CIFAR-100):

1. `pca_variance.png` — cumulative explained variance
2. `effective_dim.png` — participation ratio
3. `cka_same_layer.png` — linear CKA between models at same layer
4. `cka_cross_layer.png` — linear CKA heatmaps across layers
5. `principal_angles.png` — subspace alignment angles
6. `ica_correlation.png` — ICA component correlation matrices
7. `ica_summary.png` — ICA matching summary
8. `class_separability.png` — Fisher criterion
9. `pca_scatter.png` — 2D PCA projections

## Experiment Paths

All trained models stored in `experiments/` at the project root:
```
experiments/{Dataset}/{method}/{model_or_pair}/{seed}/
```
Each contains: `best.pth`, `checkpoint.pth`, `metrics.json`, `status.json`, `Accuracy.png`, `Loss.png`, `logs`, `errors`.
