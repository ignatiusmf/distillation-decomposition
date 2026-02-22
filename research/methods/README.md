# Analysis Methods — Deep Exploration

This directory contains detailed elaboration and exploration of each representation analysis method used in the thesis. The content here is **method knowledge** — independent of any specific experiment round.

The goal is to build deep understanding of each technique's assumptions, failure modes, and interpretive nuances before applying them to experiment charlie data.

## Planned Contents

### Linear Methods
- `pca.md` — PCA variance curves, effective dimensionality, scree plots, limitations of linear subspace assumption
- `cka.md` — Linear CKA derivation, invariance properties, relationship to other similarity measures
- `principal_angles.md` — Subspace geometry, connection to CKA, scalar summaries
- `ica.md` — Independence assumption, convergence issues at scale, matched correlation interpretation
- `fisher.md` — Fisher criterion for class separability, relationship to LDA, multi-class extensions

### Nonlinear Methods
- `kernel_cka.md` — RBF kernel CKA, sigma selection, linear vs nonlinear CKA gap interpretation
- `umap.md` — Topology preservation, teacher-space projection strategy, comparison with t-SNE
- `twonn.md` — TwoNN intrinsic dimensionality estimator, relationship to PCA effective dim
- `rsa.md` — Representational Similarity Analysis, Spearman correlation, connection to RKD objective
- `knn_probing.md` — k-NN as nonlinear separability probe, complement to Fisher criterion

## Relationship to Other Directories

- `analysis/experiment_charlie/analysis_design.md` — How these methods are *applied* to our data
- `research/explain/` — Shorter metric explanations from experiment alpha (predecessor)
- `msc-cs/thesis/chapters/` — Where the polished versions end up
