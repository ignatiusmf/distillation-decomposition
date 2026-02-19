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
