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

- **REMINDER:** When ALL experiment_beta training runs are finished, remind the user to:
  1. Move `experiments/` into `analysis/experiment_beta/experiments/`
  2. Create a new blank `experiments/` directory
  This mirrors what was done for experiment_alpha — archive everything before starting fresh.
- Current experiment structure: `analysis/experiment_alpha/` (archived), `analysis/experiment_beta/` (active).
