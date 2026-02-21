# Training & Distillation

Tasks related to training speed, plotting, and distillation method bug fixes.

---

## [4] Training speed improvements

**Status:** DONE

**Description:** Add mixed precision (AMP) and persistent workers. Keep `cudnn.deterministic=True` and `benchmark=False` for reproducibility.

### Exploration
- `train.py:184-210` — No AMP, everything FP32.
- `data_loader.py:77-85` — `num_workers=8`, `pin_memory=True`, but no `persistent_workers`. Workers respawned each epoch.

### Resolution
**A) AMP in train.py:** Added `torch.amp.autocast` + `GradScaler` wrapping forward/loss computation. Scaler state included in checkpoint save/load.

**B) persistent_workers in data_loader.py:** Added `persistent_workers=num_workers > 0` to both train and test DataLoaders.

**Files changed:** `toolbox/train.py` (~15 lines), `toolbox/data_loader.py` (~2 lines)

---

## [5] Move plotting out of training loop

**Status:** DONE

**Description:** Remove `plot_the_things(...)` from the per-epoch training loop. Metrics are already saved to `checkpoint.pth` and `metrics.json`. Write a standalone `plot_experiments.py` for cron/manual use.

### Exploration
- `train.py:249` — `plot_the_things()` called every epoch, writing 2 PNGs to Lustre.
- `toolbox/utils.py:22-53` — `plot_the_things()` creates 2 matplotlib figures.
- `train.py:256-263` — `metrics.json` saved at end. `checkpoint.pth` (every epoch) contains all metric arrays.

### Resolution
1. Deleted `plot_the_things()` call from `train.py`
2. Removed `plot_the_things` from import and deleted function from `utils.py`
3. Wrote standalone `toolbox/plot_experiments.py` that reads `metrics.json` (or `checkpoint.pth`) and generates plots

**Files changed:** `toolbox/train.py` (2 lines removed), `toolbox/utils.py` (function deleted), new `toolbox/plot_experiments.py`

---

## [11] Fix training and distillation bugs

**Status:** DONE (all sub-items)

---

### [11a] RKD loss scale — CRITICAL

**Problem:** `distillation.py:310` — `distance_weight=25.0, angle_weight=50.0`. At alpha=0.5 the gradient split is ~94% RKD / 6% CE. Student never learns discriminative features → random accuracy.

**Resolution:** Changed defaults to `distance_weight=1.0, angle_weight=2.0` in class and factory. The original Park et al. 2019 values assumed RKD was the sole loss; our setup adds it to CE via alpha-weighting.

**Files changed:** `toolbox/distillation.py`

---

### [11b] RKD _angle() wrong formulation

**Problem:** `distillation.py:323-327` computes NxN pairwise cosine similarity, not Park et al.'s triplet angle formulation (`cosine(e_i - e_j, e_k - e_j)`).

**Resolution:** Kept simplified version (O(N^2) vs O(N^3)), updated docstring to describe "pairwise cosine similarity" honestly. Added comment explaining deviation from Park et al. The loss scale fix [11a] is what actually fixes RKD.

**Files changed:** `toolbox/distillation.py`

---

### [11c] No torch.no_grad() in evaluate_model — SIGNIFICANT

**Problem:** `utils.py:6-19` runs forward pass without `torch.no_grad()`, building computation graphs during eval. Wastes GPU memory every epoch.

**Resolution:** Wrapped the eval loop in `with torch.no_grad():`.

**Files changed:** `toolbox/utils.py`

---

### [11d] Label smoothing + logit KD double-softens — SIGNIFICANT

**Problem:** `label_smoothing=0.1` applied for ALL methods including logit KD. Hinton 2015 uses hard-label CE for the ground truth term. Double-softening likely explains anomalous seed variance in CIFAR-100 logit results.

**Resolution:** `smoothing = 0.0 if args.distillation != 'none' else 0.1`. Pure students get regularisation, KD students get clean CE signal.

**Files changed:** `toolbox/train.py`

---

### [11e] FactorTransfer joint training deviates from paper — SIGNIFICANT

**Problem:** Kim et al. 2018 prescribes two stages: (1) pre-train paraphraser on frozen teacher features, (2) train translator + student jointly. Current code trains both simultaneously from scratch.

**Resolution:** Added two-stage logic to `train.py`:
1. Check for `paraphraser.pth` in experiment directory
2. If not found: run paraphraser pre-training (teacher frozen, autoencoder-style loss with decoder modules)
3. Save `paraphraser.pth`, then proceed with normal joint training (only translator + student trainable)

Added decoder modules in `distillation.py` for the pre-training phase.

**Files changed:** `toolbox/train.py`, `toolbox/distillation.py`

---

### [11f] evaluate_model hardcodes outputs[3] — MINOR

**Problem:** `utils.py:12,14` use `outputs[3]` while `train.py` uses `outputs[-1]`. Same result for 4-output model but inconsistent.

**Resolution:** Changed `outputs[3]` to `outputs[-1]`.

**Files changed:** `toolbox/utils.py`

---

### [11g] FitNets hint_layer=0 (Option B) — MINOR

**Problem:** `hint_layer=1` makes a 32→32 connector with no capacity expansion incentive.

**Resolution:** Split into `guided_layer=0` (student layer1, 16ch) and `hint_layer=1` (teacher layer2, 32ch). Connector becomes `Conv2d(16, 32, 1)` — matching the classic FitNets setup. Updated class and factory.

**Files changed:** `toolbox/distillation.py`
