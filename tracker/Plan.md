# Experiment Charlie — Task Index

This is the index for all experiment charlie tasks. Each task's full description, exploration notes, and progress are in the linked category files.

---

## Status Overview

| # | Task | Status | File |
|---|------|--------|------|
| [1] | Fix teacher-readiness bug | DONE | [infrastructure.md](infrastructure.md) |
| [2] | Fix duplicate job queuing + orphaned in_progress | DONE | [infrastructure.md](infrastructure.md) |
| [3] | Robust checkpointing | DONE (by design) | [infrastructure.md](infrastructure.md) |
| [4] | Training speed improvements (AMP + persistent workers) | DONE | [training.md](training.md) |
| [5] | Move plotting out of training loop | DONE | [training.md](training.md) |
| [6] | Archive experiment beta | DONE | [experiment.md](experiment.md) |
| [7] | Add alpha variation to experiment charlie | DONE | [experiment.md](experiment.md) |
| [8] | Add CHPC cluster details to methodology (Ch6) | Not started | [thesis.md](thesis.md) |
| [9] | Generate weekly progress summary | DONE | [thesis.md](thesis.md) |
| [10] | Consolidate cron automation | DONE | [infrastructure.md](infrastructure.md) |
| [11] | Fix training and distillation bugs (7 sub-items) | DONE | [training.md](training.md) |
| [12] | Design charlie analysis suite | Design done | [experiment.md](experiment.md) / [analysis_design.md](../analysis/experiment_charlie/analysis_design.md) |
| [13] | Plot skip reasons in plot_experiments.py | Not started | [infrastructure.md](infrastructure.md) |
| [14] | Merge experiment_summary.py and plot_experiments.py | Not started | [infrastructure.md](infrastructure.md) |
| [15] | RKD broken — ~1% accuracy after full training | Not started | [training.md](training.md) |

---

## Category Files

- **[infrastructure.md](infrastructure.md)** — PBS job management, cron automation, post-deployment fixes ([1], [2], [3], [10], [13], [14])
- **[training.md](training.md)** — Training speed, plotting, distillation bug fixes ([4], [5], [11], [15])
- **[experiment.md](experiment.md)** — Archiving experiments, launching new rounds, analysis design ([6], [7], [12])
- **[thesis.md](thesis.md)** — Thesis writing tasks ([8], [9])
- **[worklog.md](worklog.md)** — Chronological log of all actions, sequencing diagram, outstanding tasks

---

## Experiment Charlie Status

**312 total experiments:** 6 methods x 4 alphas x 4 datasets x 3 seeds = 288 KD + 24 pure baselines

As of 2026-02-21: **20/312 completed (6.4%)**

Training is managed by `toolbox/chpc_train.sh` (cron every 2 minutes) which syncs results and queues jobs on CHPC.
