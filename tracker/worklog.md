# Work Log

Chronological record of all actions taken.

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

## Outstanding Tasks

- **[8] Add CHPC cluster details to methodology chapter (Ch6)** — See [thesis.md](thesis.md)
- **[9] Generate weekly progress summary from git history** — See [thesis.md](thesis.md)
- **[12] Design charlie analysis suite** — Design done ([analysis_design.md](../analysis/experiment_charlie/analysis_design.md)), implementation pending
- **[13] Plot skip reasons in plot_experiments.py** — Show why each experiment is skipped (broken/in-progress/not-started) instead of just the path
- **[14] Merge experiment_summary.py and plot_experiments.py** — Both scan experiments and generate figures; consolidate into one tool
- **[15] RKD still broken** — 2 completed CIFAR-100 runs at alpha=0.75/0.95 show ~1.25% accuracy (random baseline). [11a] fix may be insufficient or experiments ran before fix. Needs investigation + possible force-restart.

---

## Log

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
| 2026-02-21 | Deploy | Post-deployment fixes: loop order, walltime, rsync, memory, venv, SSH, counters, tracker dashboard, alpha 0.95 | See [infrastructure.md](infrastructure.md) |
| 2026-02-21 | [12] | Analysis suite design written | Design complete, impl pending |
| 2026-02-21 | Reorg | Moved train.py/runner.py/run.job to toolbox/, created tracker/ directory, split Progress.md into category files | Done |
