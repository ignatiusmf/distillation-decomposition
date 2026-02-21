# Experiment Management

Tasks related to archiving experiments and launching new rounds.

---

## [6] Archive experiment beta

**Status:** DONE

**Description:** Experiment beta is officially over. Move `experiments/` into `analysis/experiment_beta/experiments/`, write up final results, create a new blank `experiments/`, and generate a new top-level README for charlie.

### Exploration
- `experiments/` contained 4 dataset dirs: Cifar10, Cifar100, SVHN, TinyImageNet.
- Beta status: CIFAR-10 and CIFAR-100 fully done, SVHN all failed (scipy missing), TinyImageNet partially done (teacher incomplete).
- `analysis/experiment_beta/README.md` already had comprehensive results tables and known issues.
- `analysis/experiment_alpha/` showed the archival pattern: README.md + explain/ + extract.py + analyze.py + representations/ + figures/.

### Resolution
1. `mv experiments/ analysis/experiment_beta/experiments/`
2. `mkdir experiments/`
3. Verified `analysis/experiment_beta/README.md` is complete
4. Deleted old top-level `README.md` (outdated alpha-era content)
5. Wrote new top-level `README.md` oriented toward charlie: current project state, 6 KD methods, 4 datasets, links to analysis archives

**Files changed:** Filesystem moves + `README.md` (rewrite)

---

## [7] Add alpha variation (gamma levels) to experiment charlie

**Status:** DONE

**Description:** Charlie runs each KD method at multiple alpha levels: 0.25, 0.5, 0.75, 0.95. Requires new directory structure encoding alpha in the path.

### Exploration
- `runner.py:93-98` — `get_experiment_name()` builds path as `{dataset}/{method}/{teacher}_to_{model}/{seed}`. No alpha encoding.
- `runner.py:71-85` — `generate_python_cmd()` already passes `--alpha` to train.py. Alpha was hardcoded at 0.5.
- Total experiments: 4 alphas x 6 methods x 4 datasets x 3 seeds = 288 KD + 24 pure = **312 total**.

### Resolution
**Directory structure:**
```
experiments/{dataset}/{method}/alpha_{alpha}/{teacher}_to_{model}/{seed}/
experiments/{dataset}/pure/{model}/{seed}/    (unchanged for baselines)
```

**runner.py:** Added `alphas = [0.25, 0.5, 0.75, 0.95]`, `runs = 3`, updated `get_experiment_name()` and loop structure to include alpha in paths.

**train.py:** Updated `experiment_dir` construction to include `alpha_{args.alpha}` for KD experiments.

**Files changed:** `toolbox/runner.py` (~20 lines), `toolbox/train.py` (~2 lines)

---

## [12] Design charlie analysis suite

**Status:** Design complete, implementation pending

**Description:** Charlie is the definitive experiment. The analysis pipeline must be fully specified before training starts so all necessary outputs are captured.

Full design document: [analysis/experiment_charlie/analysis_design.md](../analysis/experiment_charlie/analysis_design.md)

Summary:
- 9 existing linear metrics carried forward (PCA, CKA, principal angles, ICA, Fisher, etc.)
- 3 new alpha-sweep analyses: sweep line plots, method x alpha heatmaps, accuracy vs alignment scatter
- 5 nonlinear methods for exploration: kernel CKA, UMAP, TwoNN intrinsic dim, RSA Spearman, k-NN probing
- Experiment status overview figure for monitoring
- Dynamic model registry replaces hardcoded configs

Implementation starts when sufficient experiments complete. Training is in progress (20/312 completed as of 2026-02-21).
