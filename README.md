# Distillation Decomposition

Research project investigating the structural properties of neural network representations that enable knowledge distillation (KD). Decomposes KD mechanisms using linear latent analysis (PCA, ICA, CKA, principal angles, Fisher criterion) to understand what is actually transferred between teacher and student models.

**Key Research Question:** What structural properties of neural network representations enable effective knowledge distillation?

## Current Status: Experiment Charlie (Preparation)

Experiment charlie is the definitive experiment round with a full analysis pipeline. Bug fixes and design improvements from experiment beta have been applied. Charlie introduces alpha variation (KD strength) as a first-class experimental variable.

### Experiment Charlie Scope

- **Datasets:** CIFAR-10, CIFAR-100, SVHN, TinyImageNet
- **KD Methods:** logit, factor_transfer, attention_transfer, fitnets, rkd, nst
- **Alpha levels:** 0.25, 0.5, 0.75
- **Seeds:** 3 per condition (0, 1, 2)
- **Total:** ~240 experiments (216 KD + 24 pure baselines)

### Architecture

| Model | Role | Blocks/Group | Channels |
|-------|------|-------------|----------|
| ResNet-112 | Teacher | 18 | 16 -> 32 -> 64 |
| ResNet-56 | Student | 9 | 16 -> 32 -> 64 |

Both use GAP -> linear classifier. Forward pass returns `[layer1, layer2, layer3, logits]`.

### Directory Structure

```
experiments/{dataset}/{method}/alpha_{alpha}/{teacher}_to_{student}/{seed}/
experiments/{dataset}/pure/{model}/{seed}/
```

## Project Layout

```
train.py              - Training script (150 epochs, SGD, CosineAnnealing, AMP)
runner.py             - PBS job queue manager with teacher-readiness checks
toolbox/
  models.py           - ResNet architectures (112, 56, 20, Baby)
  data_loader.py      - CIFAR-10/100, SVHN, TinyImageNet loaders
  distillation.py     - 6 KD methods + NoDistillation base
  utils.py            - Evaluation utilities
analysis/
  experiment_alpha/   - First experiment round (2 methods, 2 datasets)
  experiment_beta/    - Second round (6 methods, 4 datasets, halted)
  experiment_charlie/ - Definitive round (to be created)
msc-cs/thesis/        - MSc thesis LaTeX source
research/             - Literature notes, structure, explanations
```

## KD Methods

| Method | Paper | What It Matches |
|--------|-------|----------------|
| Logit | Hinton et al. 2015 | Softened output distributions (KL divergence) |
| Factor Transfer | Kim et al. 2018 | Factorised intermediate features (two-stage: pretrain paraphrasers, then train translators) |
| Attention Transfer | Zagoruyko & Komodakis 2017 | Spatial attention maps (L2 on sum-of-squares) |
| FitNets | Romero et al. 2015 | Intermediate features via guided/hint layer pairing (student layer1 -> teacher layer2) |
| RKD | Park et al. 2019 | Inter-sample distance and cosine similarity relations |
| NST | Huang & Wang 2017 | Neuron selectivity patterns (MMD on Gram matrices) |

## Previous Experiments

- **Experiment Alpha** (`analysis/experiment_alpha/`): Initial exploration with logit and factor_transfer on CIFAR-10/100. Established the 9-metric analysis pipeline.
- **Experiment Beta** (`analysis/experiment_beta/`): Full-scale run with all 6 methods and 4 datasets. Revealed RKD loss scale bug, label smoothing interaction, FitNets layer pairing issue. Halted to apply fixes before charlie.
