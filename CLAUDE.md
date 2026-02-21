# Project Rules

## Thesis

- After making changes to thesis .tex or .bib files, always build and verify compilation succeeds:
  ```bash
  cd ~/Lab/studies/repos/distillation-decomposition/msc-cs/thesis && latexmk -pdf -outdir=build thesis.tex
  ```
- If the build fails, fix the errors before moving on.
- The tracker file at `msc-cs/tracker.md` has build commands, chapter status, and important file references.
- All analysis code and outputs live in `analysis/`. When writing chapters that reference experimental results, use what is in `analysis/` and leave TODO comments for experiments still needed.
- Placeholder figures use `\framebox` with `[PLACEHOLDER FIGURE]` text and descriptive captions.

## Experiments

- Current experiment: **charlie** (312 experiments, training in progress on CHPC)
- Archived: `analysis/experiment_alpha/`, `analysis/experiment_beta/`
- Analysis design: `analysis/experiment_charlie/analysis_design.md`

## Active Focus: tracker/

- `tracker/Plan.md` is the **task index** — read it first each session to understand current state.
- Task details and progress are in category files:
  - `tracker/infrastructure.md` — PBS, cron, deployment ([1], [2], [3], [10])
  - `tracker/training.md` — AMP, plotting, distillation fixes ([4], [5], [11])
  - `tracker/experiment.md` — Archiving, launching, analysis design ([6], [7], [12])
  - `tracker/thesis.md` — Thesis writing ([8], [9])
  - `tracker/worklog.md` — Chronological log, sequencing, outstanding tasks
- When working on a task, update the relevant category file with progress.
- Update `tracker/worklog.md` with each action taken.


## Documentation
- Keep the project README.md up to date with any relevant info you come across while working. It should serve as high level documentation for experiment_charlie. Once we move on to the next experiment, it will simply be moved to analysis/experiment_charlie with tracker/ and they will serve as archival documentation
