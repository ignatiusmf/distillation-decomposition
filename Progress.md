# Progress Tracker — Experiment Charlie

Reference: `Plan.md` (do not modify)

---

## [1] Fix teacher-readiness bug in runner.py

**Status:** DONE

### Problem
`runner.py` queues KD students as soon as their slot is free, but does not verify
that the teacher model has finished training. The student loads `best.pth` from
an incomplete teacher (e.g. epoch 127/150). This happened in beta: TinyImageNet
logit KD started at epoch 58 while the teacher was still at epoch 127.

### Exploration
- `runner.py:88-90` — `get_teacher_weights_path()` returns the path
  `experiments/{dataset}/pure/{teacher_model}/{seed}/best.pth` but never checks
  whether training is done.
- `runner.py:126-137` — The KD loop calls `check_path_and_skip()` which only
  checks the *student's* status via `is_training_complete()`. The teacher's
  completion status is never consulted.
- `runner.py:38-50` — `is_training_complete()` reads `status.json` and checks
  `status == 'completed'`. This exact function can be reused for the teacher.
- `train.py:252` — Status is set to `'completed'` only after the final epoch
  finishes and the final `save_status()` call runs.

### Resolution
In the KD loop (`runner.py:124-137`), before `check_path_and_skip()`, add a
teacher-readiness check:

```python
# Before queuing any KD student, verify teacher is done
if not is_training_complete(dataset, 'pure', teacher_model, run):
    print(f"Teacher {teacher_model} seed {run} not complete for {dataset}, skipping KD")
    continue
```

This goes at `runner.py:126`, inside the method/seed loop, before the existing
`check_path_and_skip` call. Uses the existing `is_training_complete()` function
with `method='pure'` — no new code needed beyond the conditional.

**Files to change:** `runner.py` (1 file, ~3 lines added)

---

## [2] Fix duplicate job queuing + orphaned in_progress in runner.py

**Status:** DONE

### Problem
`runner.py:38-50` — `is_training_complete()` only returns `True` when
`status == 'completed'`. An `in_progress` experiment is treated as "not done"
and gets re-queued, leading to two PBS jobs writing to the same `checkpoint.pth`
simultaneously (corruption). But simply skipping `in_progress` creates orphans:
a walltime-killed job leaves status as `in_progress` forever.

### Exploration
- `runner.py:29-31` — `generate_pbs_script()` calls `qsub` and captures
  `result.stdout.strip()` (the job ID), but discards it. The ID is only printed.
- `train.py:177` — `save_status()` writes `{'status': 'in_progress', ...}` at
  the start of training. There is no `job_id` field.
- `run.job` — The PBS job name is set to `{experiment_name}` (e.g.
  `Cifar10/pure/ResNet112/0`). This is readable via `qstat -f`.
- `train.py:152-172` — Checkpoint resume works: if `checkpoint.pth` exists,
  training continues from `start_epoch`. So re-queuing a dead job is safe.

### Resolution
Two changes:

**A) Store job ID in status.json on submission:**
In `generate_pbs_script()`, after `qsub` succeeds, write the job ID into the
experiment's `status.json`:

```python
# In generate_pbs_script(), after successful qsub:
job_id = result.stdout.strip()
status_path = Path(f'experiments/{experiment_name}/status.json')
status_path.parent.mkdir(parents=True, exist_ok=True)
import json
status = {}
if status_path.exists():
    with open(status_path) as f:
        status = json.load(f)
status['pbs_job_id'] = job_id
with open(status_path, 'w') as f:
    json.dump(status, f, indent=2)
```

**B) Check qstat before re-queuing in_progress:**
Modify `is_training_complete()` (or add a new function) to handle three states:

```python
def should_skip(dataset, method, model_name, seed, teacher_model=None):
    """Returns True if experiment should be skipped (completed or actively running)."""
    import json, subprocess
    # ... build status_path same as is_training_complete ...
    if not status_path.exists():
        return False  # Never started → run it
    with open(status_path) as f:
        status = json.load(f)
    if status.get('status') == 'completed':
        return True   # Done → skip
    if status.get('status') == 'in_progress':
        job_id = status.get('pbs_job_id')
        if job_id:
            result = subprocess.run(['qstat', job_id], capture_output=True, text=True)
            if result.returncode == 0:
                return True   # Still running → skip
        return False  # Dead job → re-queue
    return False
```

Replace calls to `check_path_and_skip()` with this new `should_skip()`.

**Files to change:** `runner.py` (1 file, ~30 lines modified/added)

---

## [3] Robust checkpointing (not clean exits)

**Status:** DONE (no code changes needed — by design)

### Problem
Jobs can be killed mid-epoch by PBS walltime. Need graceful recovery.

### Exploration
- `train.py:227-244` — Checkpoint is saved **every epoch** after eval. Contains
  full training state: model, optimizer, scheduler, distillation modules, all
  metric arrays, max_acc, and epoch number.
- `train.py:152-172` — Resume logic: loads checkpoint, sets `start_epoch = epoch + 1`,
  restores all state. This is already correct.
- The only risk: a kill during `torch.save()` itself could corrupt the file. But
  `torch.save()` writes atomically on most filesystems (writes to temp, renames).
  On Lustre this should be fine.

### Resolution
**No code changes needed.** The existing per-epoch checkpoint + [2]'s PBS job ID
approach handles this fully:
1. Job dies dirty at walltime → status stays `in_progress`
2. `runner.py` (with [2] fix) detects dead job via `qstat` → re-queues
3. `train.py` finds `checkpoint.pth` → resumes from last completed epoch

The Plan.md note about "clean-exit chunking" was explicitly rejected. This
approach is simpler and relies on existing infrastructure.

**Files to change:** None

---

## [4] Training speed improvements

**Status:** DONE

### Problem
Training is slow; each 150-epoch run takes ~1.5-2 hours on a V100. With ~240
charlie experiments, total GPU-hours matter.

### Exploration
- `train.py:184-210` — Training loop has no AMP (mixed precision). Every operation
  runs in FP32.
- `data_loader.py:77-85` — DataLoader has `num_workers=8`, `pin_memory=True`, but
  no `persistent_workers`. Workers are spawned fresh each epoch.
- `train.py:106-107` — `cudnn.deterministic=True`, `benchmark=False`. These are
  intentional for reproducibility (Plan.md explicitly says to keep them).

### Resolution
Two changes:

**A) Add AMP (mixed precision) to train.py:**

```python
# After optimizer/scheduler creation (~line 141):
scaler = torch.cuda.amp.GradScaler()

# In training loop, wrap forward + loss:
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    ce_loss = F.cross_entropy(outputs[-1], targets, label_smoothing=0.1)
    if args.distillation != 'none':
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        distill_loss = distillation.extra_loss(outputs, teacher_outputs, targets)
        loss = (1 - distillation.alpha) * ce_loss + distillation.alpha * distill_loss
    else:
        loss = ce_loss

# Replace loss.backward() + optimizer.step() with:
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()  # Move zero_grad after step
```

Also add scaler state to checkpoint save/load.

**B) Add persistent_workers to DataLoader:**

In `data_loader.py:77-85`, add `persistent_workers=True` to both train and test
DataLoaders (only when `num_workers > 0`):

```python
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True,
    persistent_workers=num_workers > 0,
    worker_init_fn=seed_worker, generator=g
)
```

**Files to change:** `train.py` (~15 lines), `data_loader.py` (~2 lines)

---

## [5] Move plotting out of training loop

**Status:** DONE

### Problem
`plot_the_things()` is called every epoch (`train.py:249`), writing 2 PNGs to
Lustre per epoch. This is slow I/O in the training hot path and unnecessary
since metrics are already saved to `checkpoint.pth` and `metrics.json`.

### Exploration
- `train.py:249` — `plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_dir)`
  called inside the epoch loop.
- `train.py:3` — `from toolbox.utils import plot_the_things, evaluate_model`
- `toolbox/utils.py:22-53` — `plot_the_things()` creates 2 matplotlib figures
  and saves them as Loss.png and Accuracy.png.
- `train.py:256-263` — `metrics.json` is saved at the END of training only.
  But `checkpoint.pth` (saved every epoch at line 232-244) contains all metric
  arrays. So metrics are recoverable from checkpoints at any point.

### Resolution
Three changes:

**A) Remove plot_the_things from train.py:**
- Delete `train.py:249` (the `plot_the_things(...)` call)
- Remove `plot_the_things` from the import at `train.py:3`

**B) Remove plot_the_things from utils.py:**
- Delete the entire `plot_the_things()` function from `toolbox/utils.py:22-53`
- Remove the `matplotlib` and `numpy` imports if they become unused (numpy is
  still used by evaluate_model — no, it's not. matplotlib is the only one to
  remove; actually numpy isn't used by evaluate_model either, only torch and F).
  Wait — `evaluate_model` doesn't use numpy or matplotlib. But utils.py might
  be imported elsewhere. Check: only `train.py` imports from utils.py. So remove
  matplotlib import. Keep numpy import removal safe by checking.

Actually, simplest: just delete `plot_the_things` and its matplotlib import.
Keep numpy import since it's cheap and may be used later.

**C) Write a standalone plot_experiments.py:**
New script that reads `metrics.json` (or `checkpoint.pth`) from experiment dirs
and generates plots. This can be run locally via cron (see [10]) or manually.
This is a separate task from removing the in-loop plotting — the core change is
just deleting the call from train.py.

**Files to change:** `train.py` (2 lines removed), `toolbox/utils.py` (function deleted),
new `plot_experiments.py` (to be written separately)

---

## [11] Fix training and distillation bugs

**Status:** DONE (all sub-items)

### Sub-items explored individually below.

### [11a] RKD loss scale — CRITICAL

**Problem:** `distillation.py:310` — `distance_weight=25.0, angle_weight=50.0`.
At alpha=0.5: `loss = (1-0.5)*CE + 0.5*(25*d + 50*a)`. CE is ~2-4 for CIFAR.
RKD loss ≈ 25*d + 50*a ≈ 37.5 (rough). So effective gradient: ~6% CE, ~94% RKD.
Student optimises for relational structure, never learns to classify → 18% CIFAR-10
(near random), 1% CIFAR-100 (literally random).

**Exploration:**
- `distillation.py:310` — Default weights `distance_weight=25.0, angle_weight=50.0`
- `distillation.py:342-348` — `return self.distance_weight * loss_d + self.angle_weight * loss_a`
- `runner.py:135` — All methods get `alpha=0.5`, no per-method weight override.
- Park et al. 2019 used these large weights assuming RKD was the **sole** loss.
  Our setup adds it to CE via alpha-weighting, so the weights must be reduced.

**Resolution:** Change defaults in `distillation.py:310`:
```python
def __init__(self, alpha: float = 0.5, distance_weight: float = 1.0, angle_weight: float = 2.0):
```
And update the factory at `distillation.py:473-477`:
```python
distance_weight=kwargs.get('distance_weight', 1.0),
angle_weight=kwargs.get('angle_weight', 2.0),
```

### [11b] RKD _angle() wrong formulation

**Problem:** `distillation.py:323-327` computes NxN pairwise cosine similarity
(`e_norm @ e_norm.t()`), not Park et al.'s triplet angle formulation which computes
`cosine(e_i - e_j, e_k - e_j)` for ordered triplets.

**Exploration:**
- Park et al. 2019 defines angle-wise distillation as: for each triplet (i,j,k),
  compute the angle at vertex j: `cos(angle) = <(e_i-e_j), (e_k-e_j)>` after
  normalization. This is O(N^3) in batch size.
- Current code: O(N^2) pairwise cosine similarity. This captures a *related*
  but different relational structure.
- The pairwise cosine similarity still transfers useful relational information —
  it just doesn't match the paper's specific formulation.

**Resolution:** Two options:
1. **Keep simplified version** — rename/docstring to "pairwise cosine similarity"
   instead of "triplet angles". Honest about what it computes. Simpler, cheaper.
2. **Implement correct triplet version** — significantly more expensive for large
   batches (128^3 = 2M triplets), needs sampling or approximation.

Recommendation: **Option 1** — keep simplified, fix docstring. The loss scale fix
([11a]) is what actually fixes RKD. The angle formulation difference is secondary.
Add a comment explaining the deviation from Park et al.

### [11c] No torch.no_grad() in evaluate_model — SIGNIFICANT

**Problem:** `toolbox/utils.py:6-19` — `evaluate_model()` runs the full forward
pass without `torch.no_grad()`, building computation graphs during eval. Wastes
GPU memory and compute every epoch.

**Exploration:**
- `utils.py:7` — `model.eval()` is called (disables dropout/batchnorm training
  mode) but this does NOT disable gradient computation.
- `train.py:215` — Called every epoch: `tel, tea = evaluate_model(model, testloader)`
- For 150 epochs × ~78 batches = 11,700 unnecessary graph constructions.

**Resolution:** Wrap the loop in `torch.no_grad()`:
```python
def evaluate_model(model, loader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # ... existing code ...
    # ... print and return ...
```

### [11d] Label smoothing + logit KD double-softens — SIGNIFICANT

**Problem:** `train.py:192` — `label_smoothing=0.1` is applied for ALL methods
including logit KD. Hinton 2015 uses hard-label CE for the ground truth term.
With logit KD: the student gets softened teacher logits (via temperature) AND
smoothed one-hot targets (via label smoothing). These interact non-obviously and
likely explain the anomalous seed variance in CIFAR-100 logit results (69.39%
to 73.16%).

**Exploration:**
- `train.py:192` — `ce_loss = F.cross_entropy(outputs[-1], targets, label_smoothing=0.1)`
- This applies to ALL distillation methods, not just logit.
- For feature-based methods (AT, FitNets, FT, NST, RKD), label smoothing on CE
  is defensible — it regularises the classification head independently of the
  feature matching loss.
- For logit KD specifically: the teacher logits already provide a soft target
  distribution. Adding label smoothing to the CE term means the student sees
  *two* different softenings: smoothed one-hot (CE) + tempered teacher (KL).

**Resolution:** Disable label smoothing when any distillation is active:
```python
smoothing = 0.0 if args.distillation != 'none' else 0.1
ce_loss = F.cross_entropy(outputs[-1], targets, label_smoothing=smoothing)
```

This is the cleanest approach: pure students get the regularisation benefit,
KD students get a clean CE signal that doesn't interact with distillation losses.

### [11e] FactorTransfer joint training deviates from paper — SIGNIFICANT

**Problem:** Kim et al. 2018 prescribes two stages: (1) pre-train paraphraser on
frozen teacher features, (2) train translator + student jointly. Current code
trains paraphrasers and translators simultaneously from scratch.

**Exploration:**
- `distillation.py:136-159` — Paraphrasers and translators are created together.
- `distillation.py:192-194` — `get_trainable_modules()` returns both, so both
  get optimizer gradients from epoch 0.
- `train.py:136-138` — All trainable modules are collected into one optimizer.
- Kim et al.'s reasoning: paraphrasers must first learn to extract meaningful
  "factors" from the teacher's (frozen, good) features. Only then can the
  translator have a stable target to match.

**Resolution:** Add two-stage logic to `train.py`:
1. When `args.distillation == 'factor_transfer'`, check for `paraphraser.pth`
   in the experiment directory.
2. If not found: run a paraphraser pre-training phase:
   - Freeze teacher and student. Only train paraphrasers.
   - Loss: reconstruction loss on teacher features (paraphraser output should
     preserve information from teacher — use L2 or factor reconstruction).
   - Run for ~30-50 epochs (configurable).
   - Save `paraphraser.pth`.
3. Load paraphraser weights, freeze paraphrasers, proceed with normal training
   (only translator + student are trainable).

Implementation detail: The paraphraser pre-training loss needs thought. Kim et al.
use an autoencoder-style loss: paraphraser encodes, then a decoder reconstructs
the original teacher feature. We'd need a small decoder module for pre-training
only. Alternative: train paraphrasers as part of an autoencoder on teacher
features (add a decoder mirror of each paraphraser).

This is the most complex fix in the plan. Estimate: ~50-80 lines in train.py
plus a decoder module in distillation.py.

**Files to change:** `train.py`, `distillation.py`

### [11f] evaluate_model hardcodes outputs[3] — MINOR

**Problem:** `utils.py:12,14` use `outputs[3]` while `train.py:192,207` use
`outputs[-1]`. Both work for the current 4-output model but are inconsistent.

**Resolution:** Change `outputs[3]` to `outputs[-1]` in `utils.py:12,14`.

### [11g] FitNets hint_layer=0 (Option B) — MINOR

**Problem:** `distillation.py:264` defaults to `hint_layer=1`, making a 32→32
connector with no capacity expansion incentive.

**Exploration:**
- `distillation.py:270-274` — Connector: `Conv2d(student_channels[hint_layer], teacher_channels[hint_layer], 1)`
- With `hint_layer=1`: `Conv2d(32, 32, 1)` — trivial identity-like mapping.
- With `hint_layer=0`: `Conv2d(16, 32, 1)` — student layer1 (16ch) must expand
  to match teacher layer2 (32ch). Wait — that's not right either.

  FitNets pairs: `student_outputs[hint_layer]` vs `teacher_outputs[hint_layer]`.
  So `hint_layer=0` means: student layer1 (16ch) vs teacher layer1 (16ch).
  That's a 16→16 connector — even worse!

  Re-reading Plan.md: "pair teacher layer2 (32ch) as hint with student layer1
  (16ch) as guided layer." This means:
  - guided_layer (student) = 0 → student_outputs[0] = 16ch
  - hint_layer (teacher) = 1 → teacher_outputs[1] = 32ch
  - Connector: Conv2d(16, 32, 1) ← this gives the capacity expansion

  But current FitNets code uses a single `hint_layer` index for BOTH student and
  teacher: `s_feat = student_outputs[self.hint_layer]` and
  `t_feat = teacher_outputs[self.hint_layer]`.

  **The fix requires two separate indices**, not just changing hint_layer.

**Revised resolution:** Modify FitNets to take `guided_layer` (student) and
`hint_layer` (teacher) separately:
```python
class FitNets(DistillationMethod):
    def __init__(self, student_channels, teacher_channels, alpha=0.5,
                 guided_layer=0, hint_layer=1):
        super().__init__(alpha)
        self.guided_layer = guided_layer
        self.hint_layer = hint_layer
        self.connector = nn.Conv2d(
            student_channels[guided_layer],
            teacher_channels[hint_layer],
            kernel_size=1, bias=False
        )

    def extra_loss(self, student_outputs, teacher_outputs, targets):
        s_feat = student_outputs[self.guided_layer]   # 16ch
        t_feat = teacher_outputs[self.hint_layer]      # 32ch
        s_proj = self.connector(s_feat)                # 16→32
        if s_proj.shape[2:] != t_feat.shape[2:]:
            s_proj = F.adaptive_avg_pool2d(s_proj, t_feat.shape[2:])
        return F.mse_loss(s_proj, t_feat.detach())
```

Update factory at `distillation.py:461-470` to pass both parameters.

**Files to change:** `distillation.py` (FitNets class + factory)

---

## [6] Archive experiment beta

**Status:** DONE

### Problem
Experiment beta is officially over. Need to move `experiments/` into
`analysis/experiment_beta/experiments/`, write up final results, create a new
blank `experiments/`, and generate a new top-level README for charlie.

### Exploration
- `experiments/` currently contains 4 dataset dirs: Cifar10, Cifar100, SVHN,
  TinyImageNet.
- Beta status from README.md: CIFAR-10 and CIFAR-100 fully done, SVHN all failed
  (scipy missing), TinyImageNet partially done (teacher incomplete).
- `analysis/experiment_beta/README.md` — Already has comprehensive results tables
  and known issues documented.
- `analysis/experiment_alpha/` — Shows the archival pattern: README.md + explain/
  + extract.py + analyze.py + representations/ + figures/.
- The existing top-level `README.md` is from experiment alpha era (refers to
  "3 distillation methods" and only logit/factor_transfer).

### Resolution
Steps:
1. `mv experiments/ analysis/experiment_beta/experiments/`
2. `mkdir experiments/`
3. Verify `analysis/experiment_beta/README.md` is complete (it largely is —
   may need minor updates for final TinyImageNet/SVHN status)
4. Delete old top-level `README.md` (outdated alpha-era content)
5. Write new top-level `README.md` oriented toward charlie:
   - Current project state, 6 KD methods, 4 datasets
   - Updated toolbox description (all methods listed)
   - Links to analysis/experiment_alpha/ and analysis/experiment_beta/
   - Placeholder for charlie

**Files to change:** Filesystem moves + `README.md` (rewrite)

---

## [7] Add alpha variation (gamma levels) to experiment charlie

**Status:** DONE

### Problem
Charlie needs 3 alpha levels (0.25, 0.5, 0.75) per method, requiring a new
directory structure and runner.py changes.

### Exploration
- `runner.py:93-98` — `get_experiment_name()` builds path as
  `{dataset}/{method}/{teacher}_to_{model}/{seed}`. No alpha encoding.
- `runner.py:71-85` — `generate_python_cmd()` already passes `--alpha` to train.py.
  Alpha value is hardcoded at `runner.py:135` as `0.5`.
- `train.py:72-73` — Experiment dir path is built from args but doesn't include
  alpha in the path.
- Total experiments: 3 alphas × 6 methods × 4 datasets × 3 seeds = 216 KD +
  4 datasets × 2 models × 3 seeds = 24 pure = **240 total**.

### Resolution
**Directory structure change:**
```
experiments/{dataset}/{method}/alpha_{alpha}/{teacher}_to_{model}/{seed}/
experiments/{dataset}/pure/{model}/{seed}/              (unchanged for baselines)
```

**runner.py changes:**
```python
alphas = [0.25, 0.5, 0.75]
runs = 3  # seeds 0, 1, 2

# Update get_experiment_name to include alpha:
def get_experiment_name(dataset, model, seed, distillation='none', teacher_model=None, alpha=None):
    if distillation == 'none':
        return f'{dataset}/pure/{model}/{seed}'
    else:
        return f'{dataset}/{distillation}/alpha_{alpha}/{teacher_model}_to_{model}/{seed}'

# Update is_training_complete similarly
# Add alpha loop in KD section:
for alpha in alphas:
    for method in distillation_methods:
        for run in range(runs):
            # ... queue with this alpha ...
```

**train.py changes:**
```python
# Update experiment_dir construction (~line 72-73):
if args.distillation == 'none':
    experiment_dir = Path(f'experiments/{DATASET}/pure/{MODEL}/{seed}')
else:
    experiment_dir = Path(f'experiments/{DATASET}/{args.distillation}/alpha_{args.alpha}/{args.teacher_model}_to_{MODEL}/{seed}')
```

**Files to change:** `runner.py` (~20 lines), `train.py` (~2 lines)

---

## [10] Consolidate cron automation into toolbox/chpc_train.sh

**Status:** DONE

### Problem
Currently there are separate scripts for rsync (`toolbox/rsync.sh`) and job
queuing. Need a single `chpc_train.sh` that handles everything.

### Exploration
- `toolbox/rsync.sh` — Push code up, pull experiments down. Two rsync commands.
- `toolbox/util.sh` — Cleans stale experiment directories on CHPC.
- `toolbox/setup.sh` — Creates Python venv on CHPC.
- No existing `chpc_train.sh`.
- No existing `plot_experiments.py` (needed by [5]).

### Resolution
Create `toolbox/chpc_train.sh`:

```bash
#!/bin/bash
# Usage: chpc_train.sh on|off|cron

case "$1" in
  on)
    SCRIPT=$(realpath "$0")
    (crontab -l 2>/dev/null; echo "*/30 * * * * $SCRIPT cron >> toolbox/logs/cron.log 2>&1") | crontab -
    echo "Cron installed (every 30 min)"
    ;;
  off)
    crontab -l | grep -v "chpc_train.sh" | crontab -
    echo "Cron removed"
    ;;
  cron)
    echo "=== $(date) ==="
    # 1. Pull results from CHPC
    rsync -avz --delete --include='experiments/***' --exclude='*' \
      iferreira@lengau.chpc.ac.za:/home/iferreira/lustre/distillation-decomposition/ \
      /home/ignatius/Lab/studies/repos/distillation-decomposition/
    # 2. Regenerate plots locally (if plot_experiments.py exists)
    if [ -f plot_experiments.py ]; then
      python plot_experiments.py
    fi
    # 3. Push code + queue new jobs on CHPC
    rsync -avz --delete --exclude='analysis' --exclude='.git' --exclude='.venv' --exclude='experiments/' \
      /home/ignatius/Lab/studies/repos/distillation-decomposition/ \
      iferreira@lengau.chpc.ac.za:/home/iferreira/lustre/distillation-decomposition/
    ssh iferreira@lengau.chpc.ac.za "cd /home/iferreira/lustre/distillation-decomposition && python runner.py"
    ;;
  *)
    echo "Usage: $0 on|off|cron"
    ;;
esac
```

**Files to change:** New `toolbox/chpc_train.sh`, `mkdir toolbox/logs/`

---

## [12] Design charlie analysis suite

**Status:** Not started (pre-planning done in Plan.md, needs deeper work)

### Problem
Charlie has ~240 models with a new alpha dimension. The existing analysis pipeline
(`analysis/experiment_alpha/extract.py` + `analyze.py`) handles only 4 model
configs × 2 datasets with hardcoded paths. Needs to scale and add alpha-sweep
analysis.

### Exploration

**extract.py current state:**
- `extract.py:22-46` — `MODEL_CONFIGS` is a hardcoded dict of 4 entries with
  path templates like `experiments/{dataset}/pure/ResNet112/{seed}/best.pth`.
- `extract.py:49-68` — `extract()` function is generic (takes any model + loader),
  no changes needed to core extraction logic.
- `extract.py:86-87` — Output dir: `analysis/representations/{dataset}/` with
  naming `{model_key}_seed{seed}.npz`.
- Needs: dynamic discovery of experiment directories instead of hardcoded configs.

**analyze.py current state:**
- `analyze.py:31-48` — `MODEL_KEYS`, `DISPLAY`, `SHORT`, `COLORS` all hardcoded
  for 4 models. Charlie needs 6 methods × 3 alphas + 2 baselines = 20 configs.
- `analyze.py:77-92` — `load_all_representations()` iterates `MODEL_KEYS` and
  loads npz files by name pattern.
- 7 analysis functions (186-628), each taking `reps` dict and `fig_dir`.
- All functions assume `MODEL_KEYS[0]` is teacher, `MODEL_KEYS[1:]` are students.

**What needs to change for charlie:**

1. **extract.py** — Replace hardcoded `MODEL_CONFIGS` with directory traversal:
   - Walk `experiments/{dataset}/{method}/alpha_{alpha}/{pair}/{seed}/`
   - Auto-discover all completed experiments via `status.json`
   - Naming: `{method}_a{alpha}_s{seed}.npz` + `teacher_s{seed}.npz` + `pure_s{seed}.npz`

2. **analyze.py existing 9 metrics** — Must work with 20+ model configs:
   - Parameterise `MODEL_KEYS` from discovered representations
   - Color/display maps generated programmatically
   - Existing plots will be very crowded with 20 models — may need per-method
     or per-alpha subplots instead of all-in-one

3. **New analysis #10: Alpha sweep plots** — For each (method, dataset, layer, metric):
   plot metric value (y) vs alpha (x). One line per method, shaded ±1σ across seeds.
   Scalar metrics to sweep: CKA same-layer, effective dim, Fisher criterion,
   ICA mean matched correlation.

4. **New analysis #11: Method × alpha heatmap** — 6×3 grid, one per (dataset, layer).
   Cell value = scalar metric (CKA, effective dim, etc.).

5. **New analysis #12: Accuracy vs alignment scatter** — X = representation metric
   (CKA layer3, effective dim), Y = test accuracy. One point per
   (method, alpha, seed). Faceted by dataset.

### Resolution
Will create `analysis/experiment_charlie/` with:
- `extract.py` — Auto-discovery version
- `analyze.py` — Scaled version with all 12 analyses
- `representations/` — Auto-populated
- `figures/` — Auto-populated
- `explain/` — Copied from alpha, updated for new analyses

The detailed implementation spec will be written when [6] (archive beta) and
the bug fixes are complete. This is the design-phase item that gates charlie
launch.

**Files to create:** `analysis/experiment_charlie/extract.py`, `analysis/experiment_charlie/analyze.py`

---

## [8] Add CHPC cluster details to methodology chapter (Ch6)

**Status:** Not started (thesis writing, can do anytime)

### Notes
- Cluster: CHPC Lengau, 30x V100 GPUs across 9 nodes
- PBS job system, single GPU per job (multi-GPU needs special access)
- walltime=2:00:00, mem=16gb, ncpus=8
- Deterministic training: all seeds set, cudnn.deterministic=True
- Check wiki links in Plan.md for exact specs before writing

**No code exploration needed — this is pure thesis writing.**

---

## [9] Generate weekly progress summary from git history

**Status:** Not started (utility task, can do anytime)

### Notes
A script or one-liner that runs `git log --since="1 week ago" --oneline` and
`git diff --stat HEAD~N` to produce a supervisor-friendly narrative.

**No code exploration needed — simple git automation.**

---

## Sequencing

```
[1]  Fix teacher-readiness bug   ─┐
[2]  Fix duplicate queue bug     ─┤── Before ANY new runs
[3]  Robust checkpointing        ─┤   (3 is a no-op)
[4]  AMP + persistent_workers    ─┤
[5]  Move plotting to cron       ─┤
[11] Fix all distillation bugs   ─┘
[6]  Archive beta                ─── Filesystem moves + README
[12] Design charlie analysis     ─── After [6], before launching charlie
[7]  Launch charlie (w/ gamma)   ─── After [1-6], [11], [12] done
[8]  Methodology chapter         ─── Anytime
[9]  Weekly summary              ─── Anytime
[10] Consolidate cron scripts    ─── Anytime
```

---

## Work Log

| Date | Task | Action | Outcome |
|------|------|--------|---------|
| 2026-02-21 | All | Initial exploration of all source files, defined resolutions for all Plan.md items | Progress.md created |
| 2026-02-21 | [1] | Added teacher-readiness check in runner.py KD loop | Done |
| 2026-02-21 | [2] | Added PBS job ID storage in status.json, qstat-based orphan detection, should_skip() | Done |
| 2026-02-21 | [3] | Confirmed no code changes needed — existing checkpoint + [2] handles this | Done (by design) |
| 2026-02-21 | [4] | Added AMP (torch.amp.autocast + GradScaler) to train.py, persistent_workers to data_loader.py | Done |
| 2026-02-21 | [5] | Removed plot_the_things from train.py and utils.py, wrote standalone plot_experiments.py | Done |
| 2026-02-21 | [11a] | Changed RKD distance_weight 25→1.0, angle_weight 50→2.0 in distillation.py + factory | Done |
| 2026-02-21 | [11b] | Updated _angle() docstring to describe pairwise cosine similarity, not triplet angles | Done |
| 2026-02-21 | [11c] | Wrapped evaluate_model loop in torch.no_grad() | Done |
| 2026-02-21 | [11d] | Label smoothing=0.0 when distillation != 'none', 0.1 for pure training | Done |
| 2026-02-21 | [11e] | Added FactorTransfer two-stage: decoders for pretrain, pretrain_loss(), freeze_paraphrasers(), train.py stage 1 logic | Done |
| 2026-02-21 | [11f] | Changed outputs[3] to outputs[-1] in evaluate_model | Done |
| 2026-02-21 | [11g] | Split FitNets into guided_layer=0 (student) + hint_layer=1 (teacher), connector 16→32 | Done |
| 2026-02-21 | [6] | Moved experiments/ to analysis/experiment_beta/experiments/, created fresh experiments/, wrote new README.md | Done |
| 2026-02-21 | [7] | Added alphas=[0.25,0.5,0.75], runs=3, alpha in dir paths for runner.py + train.py | Done |
| 2026-02-21 | [10] | Created toolbox/chpc_train.sh (on/off/cron), mkdir toolbox/logs/ | Done |
| 2026-02-21 | Verify | All Python files compile, factory tests pass, path consistency verified, archive confirmed | All checks pass |

---

## Outstanding Tasks

The following Plan.md items are **not yet implemented** and require future work:

- **[8] Add CHPC cluster details to methodology chapter (Ch6)** — Thesis writing task. Document cluster specs, PBS config, training times, reproducibility settings. Can do anytime.
- **[9] Generate weekly progress summary from git history** — Utility script for supervisor updates. Can do anytime.
- **[12] Design charlie analysis suite** — Create `analysis/experiment_charlie/extract.py` and `analyze.py` with auto-discovery of experiment dirs, all 9 existing metrics scaled to ~240 models, plus 3 new alpha-sweep analyses (alpha sweep per metric, method x alpha heatmap, accuracy vs alignment scatter). Gates charlie launch — should be done before training starts producing data.
