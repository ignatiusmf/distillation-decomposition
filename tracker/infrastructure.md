# Infrastructure & Automation

Tasks related to PBS job management, cron automation, and experiment infrastructure.

---

## [1] Fix teacher-readiness bug in runner.py

**Status:** DONE

**Description:** KD students can be queued before their teacher finishes training. The student then loads `best.pth` from an incomplete teacher run (e.g. epoch 127/150). Fix: check that the teacher's `status.json` says "completed" before queuing any dependent student.

### Exploration
- `runner.py:88-90` — `get_teacher_weights_path()` returns the path but never checks whether training is done.
- `runner.py:126-137` — The KD loop calls `check_path_and_skip()` which only checks the *student's* status. The teacher's completion is never consulted.
- `runner.py:38-50` — `is_training_complete()` reads `status.json` and checks `status == 'completed'`. Can be reused for the teacher.
- `train.py:252` — Status is set to `'completed'` only after the final epoch.

### Resolution
In the KD loop (`runner.py:126`), before `check_path_and_skip()`, added a teacher-readiness check:
```python
if not is_training_complete(dataset, 'pure', teacher_model, run):
    print(f"Teacher {teacher_model} seed {run} not complete for {dataset}, skipping KD")
    continue
```
Uses existing `is_training_complete()` with `method='pure'` — no new code needed beyond the conditional.

**Files changed:** `toolbox/runner.py` (1 file, ~3 lines added)

---

## [2] Fix duplicate job queuing + orphaned in_progress in runner.py

**Status:** DONE

**Description:** `runner.py` skips `completed` but not `in_progress`. Two jobs writing to the same `checkpoint.pth` simultaneously corrupt each other. But naively skipping all `in_progress` creates orphans: a walltime-killed job leaves status as `in_progress` forever. Solution: store PBS job ID, check qstat.

### Exploration
- `runner.py:29-31` — `generate_pbs_script()` calls `qsub` and captures job ID but discards it.
- `train.py:177` — `save_status()` writes `{'status': 'in_progress', ...}` with no `job_id` field.
- `train.py:152-172` — Checkpoint resume works, so re-queuing a dead job is safe.

### Resolution
**A) Store job ID in status.json on submission:** After `qsub` succeeds, write the job ID into the experiment's `status.json`.

**B) Check qstat before re-queuing in_progress:** Added `get_experiment_status()` returning 'completed'/'running'/'pending'. If `in_progress` and `qstat <job_id>` shows it still exists → skip. If job no longer exists → re-queue.

**Files changed:** `toolbox/runner.py` (1 file, ~30 lines modified/added)

---

## [3] Robust checkpointing (not clean exits)

**Status:** DONE (no code changes needed — by design)

**Description:** Jobs can be killed mid-epoch by PBS walltime. Need graceful recovery. Do NOT add clean-exit chunking — rely on existing per-epoch checkpoint + PBS job ID approach from [2].

### Exploration
- `train.py:227-244` — Checkpoint saved **every epoch** after eval. Contains full training state.
- `train.py:152-172` — Resume logic loads checkpoint, sets `start_epoch = epoch + 1`. Already correct.
- Only risk: kill during `torch.save()` — but it writes atomically on most filesystems.

### Resolution
**No code changes needed.** The existing per-epoch checkpoint + [2]'s PBS job ID approach handles this:
1. Job dies at walltime → status stays `in_progress`
2. `runner.py` (with [2] fix) detects dead job via `qstat` → re-queues
3. `train.py` finds `checkpoint.pth` → resumes from last completed epoch

**Files changed:** None

---

## [10] Consolidate cron automation into toolbox/chpc_train.sh

**Status:** DONE

**Description:** Single script that manages itself as a cron job and does all experiment automation. Replaces `~/queue_chpc.sh` and the inline rsync cron entry. Usage: `chpc_train.sh on|off|cron`.

### Resolution
Created `toolbox/chpc_train.sh` with three modes:
- `on` — Install crontab entry (every 2 minutes)
- `off` — Remove crontab entry
- `cron` — Pull experiments (rsync), regenerate plots, push code (rsync), queue jobs (SSH + runner.py)

One script, one crontab entry, one log file in `toolbox/logs/`.

**Files created:** `toolbox/chpc_train.sh`, `toolbox/logs/`

---

## Post-deployment Fixes

These fixes were made iteratively after the initial implementation was deployed to CHPC and issues were discovered during live training.

| Date | Commit | Fix | Details |
|------|--------|-----|---------|
| 2026-02-21 | `4191110` | Loop order + walltime + rsync pull | **runner.py:** Reordered loops from `dataset→model→run` to `run→dataset→model` so all seed-0 baselines queue before seed-1, ensuring teachers finish before KD jobs. **run.job:** Reduced walltime 2:00:00→1:00:00 (AMP makes runs faster). **rsync.sh:** Uncommented pull command. |
| 2026-02-21 | `4a9830a` | File moves + path fixes + pbs_job_id preservation | **Moved** `plot_experiments.py` → `toolbox/`, `tools.py` → `toolbox/reorganize_tinyimagenet.py`. **chpc_train.sh:** Fixed plot script path. **plot_experiments.py:** Fixed `default_dir` to resolve relative to script location. **train.py:** `save_status()` now preserves `pbs_job_id` via `setdefault()`. |
| 2026-02-21 | `1332e87` | Memory, venv, SSH, rsync, counters | **run.job:** Increased memory 16gb→24gb (SVHN OOM). **chpc_train.sh:** Added `VENV_PYTHON`, `SSH_OPTS`, `--exclude='*.png'` on pull, cron interval 30min→10min, section banners. **runner.py:** Replaced verbose skip messages with counters + summary line. |
| 2026-02-21 | `a0470b6` | ExperimentTracker dashboard | **runner.py:** Added `ExperimentTracker` class with categorised lists (completed, running, queued, pending, teacher_pending). Prints completion %, breakdown, lists running/queued experiments. |
| 2026-02-21 | `e14c882` | Alpha 0.95 added | **runner.py:** Changed `alphas = [0.25, 0.5, 0.75]` to `[0.25, 0.5, 0.75, 0.95]`. Total experiments: 312. |
| 2026-02-21 | — | plot_experiments.py crash fix | Added `try/except` around `torch.load` so corrupted checkpoints are skipped with warning. Added `plotted/total` summary. |
| 2026-02-21 | — | plot_experiments.py skipped names | `plot_experiments.py`: Now lists which experiments were skipped (with relative paths) instead of just a count. Fixed WARN message to show full path instead of just leaf dir name. |
| 2026-02-21 | — | Improve runner.py cron log output | `runner.py`: Reworked `ExperimentTracker.summary()` to show: finished count, not-started count (split into pending vs waiting-on-teacher), experiments already on PBS queue (with names), experiments queued this run (with names). |
| 2026-02-21 | — | Fix PBS queue detection bug | **runner.py:** `get_experiment_status()` only checked `_is_job_alive(job_id)` when `status == 'in_progress'`, but submitted jobs have `pbs_job_id` without `status: in_progress` (that's set by train.py when it actually starts). Jobs sitting in PBS queue (status Q) were invisible — runner.py thought they needed re-queuing. Fix: check `pbs_job_id` + `_is_job_alive()` regardless of status field. Also removed `teacher_pending` as a separate category — folded into `pending`. |
| 2026-02-21 | — | Show PBS job state and elapsed time | **runner.py:** Replaced `_is_job_alive()` with `_get_job_info()` that parses `qstat -f` output for `job_state` (Q/R/etc) and `resources_used.walltime` (or `stime` fallback). `get_experiment_status()` now returns `(status, job_info)` tuple. Tracker stores job info for running experiments and displays as `[R 0:42:15] Cifar100/logit/...` in the summary. |
