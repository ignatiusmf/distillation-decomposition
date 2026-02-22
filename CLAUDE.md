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
- **After every fix or change**, add an entry to the appropriate tracker category file documenting the problem and the fix. Every change should be traceable in tracker/.


## Three Components

The project has three distinct intellectual components:

1. **Data** — Training results (312 ResNet56 students with varying KD methods, alphas, datasets, seeds). Lives in `experiments/` during training, archived to `analysis/experiment_*/experiments/` after.
2. **Analysis methods** — The dimensionality reduction and representation comparison techniques themselves (PCA, ICA, CKA, UMAP, etc.). Deep elaboration and exploration of each method lives in `research/methods/`. This is method knowledge independent of our specific experiments.
3. **Application** — Applying the analysis methods to our data. The experiment-specific analysis design, extract.py, analyze.py, and figures. Lives in `analysis/experiment_charlie/`.

Key coupling files:
- `toolbox/experiment_summary.py` — generates graphical progress figures to `analysis/experiment_charlie/`
- `tracker/thesis.md` — tracks thesis writing tasks
- `analysis/experiment_charlie/analysis_design.md` — specifies how methods (2) are applied to data (1)

## Documentation
- Keep the project README.md up to date with any relevant info you come across while working. It should serve as high level documentation for experiment_charlie. Once we move on to the next experiment, it will simply be moved to analysis/experiment_charlie with tracker/ and they will serve as archival documentation
