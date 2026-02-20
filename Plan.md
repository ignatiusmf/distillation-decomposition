# Action Plan — Building Experiment Charlie

Everything in this plan is in service of designing and launching experiment charlie:
the definitive experiment round with a full analysis pipeline.

## Immediate / Bug Fixes
Must be done before ANY new training runs are launched.

**[1] Fix teacher-readiness bug in runner.py**
KD students can be queued before their teacher finishes training. The student then loads
best.pth from an incomplete teacher run (e.g. epoch 127/150). Fix: check that the teacher's
status.json says "completed" before queuing any dependent student.

**[2] Fix duplicate job queuing + orphaned in_progress in runner.py**
runner.py skips `completed` but not `in_progress`. Two jobs writing to the same checkpoint.pth
simultaneously will corrupt each other. But naively skipping all `in_progress` experiments
creates an orphan problem: a walltime-killed job leaves status as `in_progress` forever and
the experiment never gets re-queued.

Solution — store PBS job ID, check qstat:
- When runner.py submits a job via qsub, capture the returned job ID and store it in
  status.json alongside the `in_progress` status.
- When runner.py encounters an `in_progress` experiment, call `qstat <job_id>`. If the job
  no longer exists in the queue, it died (walltime kill or crash) → safe to re-queue.
  If qstat shows it still running → skip as intended.
- This is authoritative: no timing guesses, no heartbeat drift.

Note: train.py already checkpoints every epoch, so a re-queued job will resume from the
last completed epoch automatically.

**[3] Robust checkpointing (not clean exits)**
Do NOT add clean-exit chunking. Each distillation method takes different time, making
per-method walltime calibration impractical. Instead, rely on the existing per-epoch
checkpoint in train.py (lines 152-174) + the PBS job ID approach in [2] to handle
walltime-killed jobs gracefully. The job dies dirty, status stays `in_progress`, but
runner.py detects the dead job via qstat and re-queues it to continue from the checkpoint.

**[4] Training speed improvements**
- Add mixed precision (AMP): wrap forward/backward in `torch.cuda.amp.autocast` + `GradScaler`.
  Expected ~1.5-2x speedup on GPU, no reproducibility impact.
- Add `persistent_workers=True` to DataLoader in data_loader.py. Avoids respawning 8 workers
  each epoch. Small free win.
- Keep `cudnn.deterministic=True` and `benchmark=False` (reproducibility is intentional).

**[5] Move plotting out of training loop**
Remove `plot_the_things(...)` from the per-epoch training loop. Keep metrics saving to
metrics.json / status.json every epoch. Write a separate `plot_experiments.py` script that
reads all experiment directories and regenerates plots. Run this on a cron job or manually.
This removes per-epoch Lustre writes from the hot training path. Make changes in my train.py and other python files that does the plot_the_things and remove it, we won't need it anymore.

**[11] Fix training and distillation bugs**

*Critical (break training):*

- **RKD loss scale** — `extra_loss` returns `25*loss_d + 50*loss_a` ≈ 37.5, so at alpha=0.5
  the gradient split is ~94% RKD / 6% CE. Student never learns discriminative features →
  random accuracy. Fix: reduce distance_weight/angle_weight from 25/50 → 1.0/2.0.
  These values assume standalone training without a CE component.
- **RKD `_angle()` wrong formulation** — computes NxN pairwise cosine similarity
  (`e_norm @ e_norm.t()`), not the triplet angle relations from Park et al. 2019
  which compute `cosine(e_i - e_j, e_k - e_j)` for each ordered triplet. Decide whether
  to implement the correct triplet formulation or keep simplified version (fix the docstring
  either way).

*Significant (affect result validity):*

- **No `torch.no_grad()` in `evaluate_model`** (utils.py) — every eval pass builds a full
  computation graph, wasting memory and compute every epoch. Fix: wrap loop in
  `with torch.no_grad():`.
- **Label smoothing + logit KD double-softens targets** (train.py:192) —
  `label_smoothing=0.1` is applied for ALL methods including logit KD. For logit KD this
  combines softened one-hot targets with softened teacher logits, interacting in a
  non-obvious way. Hinton 2015 uses hard-label CE for the ground truth term. Likely explains
  the anomalous seed variance in CIFAR-100 logit results. Fix: remove label_smoothing when
  distillation != 'none', or apply it only for pure training.
- **FactorTransfer joint training deviates from paper** — Kim et al. 2018 prescribes two
  stages: pre-train the paraphraser on frozen teacher features first, then train translator +
  student jointly. Current impl trains both from scratch simultaneously, so paraphrasers don't
  learn meaningful teacher factors. May explain FT's inconsistent results.
  Fix: train.py detects it is running a factor_transfer experiment and checks for a saved
  paraphraser checkpoint in the experiment dir. If none exists, it runs the paraphraser
  pre-training phase first (teacher frozen, only paraphrasers trained), saves
  `paraphraser.pth`, then proceeds to the normal joint training phase automatically.
  No CLI flags needed — queueing the experiment is sufficient.

*Minor / design:*

- **`evaluate_model` hardcodes `outputs[3]`** vs `outputs[-1]` in train.py — same result
  for the current 4-output model but inconsistent and fragile.
- **FitNets 32→32 connector** — teacher and student have identical channel widths, so
  `hint_layer=1` is a 32→32 1×1 conv with no structural incentive to learn anything
  meaningful. Fix: use Option B — pair teacher layer2 (32ch) as hint with student layer1
  (16ch) as guided layer. Connector becomes 16→32 (channel expansion), matching the
  classic FitNets setup where a thin early student layer is forced to expand to match
  a wider intermediate teacher layer. Change: `hint_layer=0` in distillation.py FitNets.

- **AttentionTransfer implicit spatial weighting** — INVESTIGATED, NO ACTION NEEDED.
  The concern (larger spatial maps dominating) is neutralised by the L2 normalization inside
  `_attention_map`: for any two unit vectors, ||a-b||² = 2*(1-cos θ) ∈ [0,4] regardless of
  spatial size. Layer1 (32×32) and layer3 (8×8) contribute equally bounded values.
  Minor note: paper sums across layers; code averages (/3) — a constant factor absorbed by alpha.

---

## Before Next Experiment Round

**[6] Archive experiment beta**
When all current beta runs finish: fill in analysis/experiment_beta/README.md with final
results, move experiments/ into analysis/experiment_beta/experiments/, create a fresh blank
experiments/. (This mirrors what was done for alpha — see CLAUDE.md reminder.). Also look at the contents of the top level readme.md to fill in the details for experiment beta. Then delete the current top level readme and genreate a new one which I will later again be able to move in charlie/
Note from user: Experiment beta training has been halted and it can be moved. Experiment beta is officially over.

---

## Experiment Gamma

**[7] Add alpha variation (gamma levels) to experiment charlie**
Charlie runs each KD method at three alpha levels: 0.25, 0.5, 0.75.
- 3 alphas × 6 KD methods × 4 datasets × 3 seeds = 216 KD experiments
- Plus pure baselines (teacher + student, 3 seeds × 4 datasets = 24 runs)
- Total: ~240 jobs
- Use 3 seeds (not 6).
- Depends on: [1], [2], [3], [4], [5], [6], [11] all done first.

Code changes required: runner.py experiment dir structure assumes a flat method name (e.g.
`logit/ResNet112_to_ResNet56/0/`). With three alpha levels, the path needs to encode alpha
(e.g. `logit_a0.25/...` or `logit/alpha_0.25/ResNet112_to_ResNet56/0/`). Update runner.py,
train.py, and is_training_complete() accordingly before launching.

**[12] Design charlie analysis suite**
Using learnings from alpha/, beta/, and research/, design the analysis pipeline that will be
run on charlie's results. Charlie is the definitive experiment round — the analysis suite
needs to be fully specified before training starts so that all necessary outputs are captured.
Read through:
- research/structure.md and research/kd_explore.md for thesis framing
- analysis/experiment_alpha/ for the reference analysis implementation
- analysis/experiment_beta/accuracy_considerations.md for known issues and insights
Training in charlie produces three gamma levels per method, so the analysis must account for
alpha as an independent variable (e.g. how does representation geometry shift with alpha?).

---

## PBS / Cluster Config (already done)
- ncpus=8, mpiprocs=8 (matches num_workers=8 in DataLoader)
- mem=16gb (down from 32gb; peak usage was ~10.4 GB)
- walltime=2:00:00 — revisit once chunked checkpointing is in place (can drop to ~30min per chunk)

---

## Tooling / Automation

**[10] Consolidate cron automation into toolbox/chpc_train.sh**
Single script that manages itself as a cron job and does all experiment automation.
Replaces ~/queue_chpc.sh and the inline rsync cron entry.

Flags:
  `chpc_train.sh on`   — installs the crontab entry. The cron will call `chpc_train.sh cron`.
  `chpc_train.sh off`  — removes the crontab entry.
  `chpc_train.sh cron` — the actual work, run by cron every 30 min:
      1. Pull experiments/ from CHPC (rsync HPC→local)
      2. Run plot generator script (plot_experiments.py — see [5]) to regenerate all plots
      3. SSH to CHPC and run runner.py to queue new/resumed jobs

One script, one crontab entry, one log file in toolbox/logs/.

Sub-item (done): Added concise comments to existing toolbox sh files (rsync.sh, util.sh, setup.sh).

---

## Thesis Writing (can do anytime)

**[8] Add CHPC cluster details to methodology chapter (Ch6)**
Document: CHPC cluster, GPU type/count, PBS job system, walltime/resource settings, training
times per dataset, checkpointing strategy, reproducibility settings (seeds, deterministic cuDNN).

Note: Check https://wiki.chpc.ac.za/chpc:gpu and https://wiki.chpc.ac.za/guide:gpu before
writing this section — the wiki has exact specs (GPU model, node counts, queue details).
Cluster has 30x V100 GPUs across 9 nodes. Multi-GPU (ngpus>1) is available but requires
special access beyond the standard allocation — not viable for this project. The methodology should maybe just (briefly) include some non functional facts about the context in which I train, for historically dating for future readers. 

**[9] Generate weekly progress summary from git history**
Run git log + git diff to produce a narrative of work done this week. Focus on story and flow
for supervisor updates, not raw diffs.

---

## Sequencing

```
[1]  Fix teacher-readiness bug   ─┐
[2]  Fix duplicate queue bug     ─┤── Before ANY new runs
[3]  Robust checkpointing        ─┤
[4]  AMP + persistent_workers    ─┤
[5]  Move plotting to cron       ─┤
[11] Fix RKD                     ─┘
[6]  Archive beta                ─── When current runs finish
[12] Design charlie analysis     ─── After [6], before launching charlie
[7]  Launch charlie (w/ gamma)   ─── After [1-6], [11], [12] done
[8]  Methodology chapter         ─── Anytime
[9]  Weekly summary              ─── Anytime
[10] Consolidate cron scripts    ─── Anytime
```
