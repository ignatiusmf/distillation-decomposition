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

## [9] Generate weekly progress summary from git history

**Status:** Not started

**Description:** A script or one-liner that runs `git log --since="1 week ago" --oneline` and `git diff --stat HEAD~N` to produce a supervisor-friendly narrative. Focus on story and flow, not raw diffs.

**Files to create:** Could be a script in `toolbox/` or just a documented one-liner.
