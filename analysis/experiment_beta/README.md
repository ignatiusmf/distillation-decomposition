# Experiment Beta

Second round of training experiments. Started 2026-02-20.
**Status: IN PROGRESS** — waiting for new training runs to complete.

## Motivation

Experiment alpha produced small accuracy gaps between KD and pure students (~0.5–1% on CIFAR-10, ~2% on CIFAR-100). Several methods also failed outright (RKD) or never ran (SVHN). This round aims to produce more differentiated results. See `research/accuracy_considerations.md` for detailed reasoning.

## What Changed From Alpha

| Parameter | Alpha | Beta | Rationale |
|-----------|-------|------|-----------|
| _TBD — fill in when new configs are decided_ | | | |

**Known issues from alpha that need fixing:**
- RKD completely broken (18% CIFAR-10, 1% CIFAR-100) — implementation or hyperparameter bug
- SVHN failed: `ModuleNotFoundError: No module named 'scipy'` on CHPC cluster (need `pip install scipy` in `myenv`)
- Only seed 0 for AT, FitNets, RKD, NST — `runner.py` had `runs = 1`
- TinyImageNet incomplete (teacher still training at epoch 127/150, logit at 58/150)

## Training Configuration (baseline — same as alpha until changed)

| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Batch size | 128 |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Learning rate | 0.1 |
| LR schedule | CosineAnnealingLR (T_max=150) |
| Label smoothing | 0.1 |
| Loss formula | `(1 - alpha) * CE + alpha * distill_loss` |
| KD alpha | 0.5 |
| KD temperature | 4.0 (logit method only) |
| Teacher | ResNet-112 (18 blocks/group, 3 layer groups: 16→32→64ch) |
| Student | ResNet-56 (9 blocks/group, same channel structure) |
| Seeding | `torch.manual_seed`, `cuda.manual_seed`, `np.random.seed`, `random.seed`, `cudnn.deterministic=True`, `cudnn.benchmark=False` |

## Experiment Matrix

**Datasets:** Cifar10, Cifar100, SVHN, TinyImageNet
**Methods:** pure (no KD), logit, factor_transfer, attention_transfer, fitnets, rkd, nst
**Seeds:** 0 (runner.py `runs = 1`; increase to 3 for beta)
**Models:** Teacher=ResNet112, Student=ResNet56

Full matrix = 4 datasets × (2 pure + 6 KD) × 3 seeds = **96 experiments**

## Experiment Alpha Results (for comparison)

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

Key observations:
- Pure student (93.20%) is already very close to teacher (93.55%). CIFAR-10 is too easy for this architecture pair.
- Best KD method (AT: 94.12%) actually *exceeds* the teacher, which is interesting but the margin is tiny.
- Logit seed 0 (92.35%) is an outlier — worse than pure. High seed variance (0.73%).

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

Key observations:
- Teacher-student gap is ~1.1% — still small.
- Logit KD has huge seed variance (1.73%) — seed 0 is *below* pure student, seeds 1-2 are *above* teacher.
- FitNets (70.23%) actually hurt compared to pure (71.17%).

### TinyImageNet (alpha — incomplete)

| Model / Method | Seed 0 | Status |
|----------------|--------|--------|
| Teacher (ResNet-112) | 53.31% | in-progress (epoch 127/150) |
| Student pure (ResNet-56) | 55.65% | completed |
| Logit KD | 36.27% | in-progress (epoch 58/150) |

Note: Student *outperforming* teacher is suspicious — may indicate the larger model is harder to train on TinyImageNet with these hyperparameters.

### SVHN

All 8 experiments (pure×2 + 6 KD methods, seed 0 each) failed at initialization:
```
File ".../toolbox/data_loader.py", line 59, in get_loaders
  trainset = ds(root=data_root, split='train', download=True, transform=transform_train)
File ".../torchvision/datasets/svhn.py", line 75, in __init__
  import scipy.io as sio
ModuleNotFoundError: No module named 'scipy'
```
Fix: `pip install scipy` in CHPC environment (`/home/iferreira/myenv/`).

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Training script (150 epochs, SGD, cosine LR) |
| `runner.py` | PBS job generation and queuing (CHPC cluster) |
| `toolbox/models.py` | ResNet architectures (ResNet112/56/20/Baby) |
| `toolbox/data_loader.py` | Dataset loaders (CIFAR-10/100, SVHN, TinyImageNet) |
| `toolbox/distillation.py` | KD method implementations |
| `run.job` | PBS job template |

## Experiment Beta Results

_To be filled in as experiments complete._

### CIFAR-10

| Model / Method | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|----------------|--------|--------|--------|------------|
| Teacher (ResNet-112) | | | | |
| Student pure (ResNet-56) | | | | |
| Logit KD | | | | |
| Factor Transfer | | | | |
| Attention Transfer | | | | |
| FitNets | | | | |
| NST | | | | |
| RKD | | | | |

### CIFAR-100

| Model / Method | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|----------------|--------|--------|--------|------------|
| Teacher (ResNet-112) | | | | |
| Student pure (ResNet-56) | | | | |
| Logit KD | | | | |
| Factor Transfer | | | | |
| Attention Transfer | | | | |
| FitNets | | | | |
| NST | | | | |
| RKD | | | | |

### SVHN

| Model / Method | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|----------------|--------|--------|--------|------------|
| Teacher (ResNet-112) | | | | |
| Student pure (ResNet-56) | | | | |
| Logit KD | | | | |
| Factor Transfer | | | | |
| Attention Transfer | | | | |
| FitNets | | | | |
| NST | | | | |
| RKD | | | | |

### TinyImageNet

| Model / Method | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|----------------|--------|--------|--------|------------|
| Teacher (ResNet-112) | | | | |
| Student pure (ResNet-56) | | | | |
| Logit KD | | | | |
| Factor Transfer | | | | |
| Attention Transfer | | | | |
| FitNets | | | | |
| NST | | | | |
| RKD | | | | |

## Archival Checklist (when all runs are done)

- [ ] Fill in all accuracy tables above
- [ ] Move `experiments/` into `analysis/experiment_beta/experiments/`
- [ ] Create new blank `experiments/` directory
- [ ] Run `extract.py` to get representations
- [ ] Run `analyze.py` to generate figures
- [ ] Move analysis scripts and outputs into this directory
