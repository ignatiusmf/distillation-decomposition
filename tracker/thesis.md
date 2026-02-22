# Thesis Writing

Tasks related to writing the MSc thesis. Can be done anytime.

---

## [8] Add CHPC cluster details to methodology chapter (Ch6)

**Status:** Not started

**Description:** Document: CHPC cluster, GPU type/count, PBS job system, walltime/resource settings, training times per dataset, checkpointing strategy, reproducibility settings (seeds, deterministic cuDNN). Briefly include non-functional context for historically dating the work for future readers.

### Notes
- Cluster: CHPC Lengau, 30x V100 GPUs across 9 nodes
- PBS job system, single GPU per job (multi-GPU needs special access beyond standard allocation)
- walltime=1:00:00 (reduced from 2:00:00 after AMP), mem=24gb, ncpus=8
- Deterministic training: all seeds set, `cudnn.deterministic=True`, `benchmark=False`
- Check wiki links before writing:
  - https://wiki.chpc.ac.za/chpc:gpu
  - https://wiki.chpc.ac.za/guide:gpu

**Files to change:** `msc-cs/thesis/chapters/methodology.tex`

---

## [9] Experiment summary and progress narrative

**Status:** DONE

**Description:** Two deliverables:

### A) Graphical experiment summary
`toolbox/experiment_summary.py` — Scans `experiments/`, reads all status.json and metrics.json, generates figures to `analysis/experiment_charlie/`:
- `progress_grid.png` — Completion grid (method x dataset, cells show alpha x seed status)
- `accuracy_overview.png` — Bar chart of max accuracy for completed experiments by dataset

```
python toolbox/experiment_summary.py
```

### B) Git history narrative
`tracker/worklog_narrative_2026-01-30_to_2026-02-21.md` — A ~1500-word narrative covering all work from Jan 30 to the current day, organized:

**Files created:** `toolbox/experiment_summary.py`, `tracker/worklog_narrative_2026-01-30_to_2026-02-21.md`
