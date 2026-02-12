"""
Analyze extracted representations: PCA, ICA, CKA, principal angles, class separability.

Produces thesis-quality figures in analysis/figures/{dataset}/.
All metrics are computed per seed and then averaged (+/- std).

Usage:
    python analysis/analyze.py [--dataset all|Cifar10|Cifar100] [--figures-dir analysis/figures]

Prerequisites:
    python analysis/extract.py --dataset all     (run first)
    uv add scikit-learn scipy                    (or pip install)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
import argparse
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2]

MODEL_KEYS = [
    'teacher_ResNet112',
    'student_ResNet56_pure',
    'student_ResNet56_logit',
    'student_ResNet56_factor',
]
DISPLAY = {
    'teacher_ResNet112':       'Teacher (ResNet-112)',
    'student_ResNet56_pure':   'Student - No KD',
    'student_ResNet56_logit':  'Student - Logit KD',
    'student_ResNet56_factor': 'Student - Factor Transfer',
}
SHORT = {
    'teacher_ResNet112':       'Teacher',
    'student_ResNet56_pure':   'No KD',
    'student_ResNet56_logit':  'Logit KD',
    'student_ResNet56_factor': 'Factor Transfer',
}
COLORS = {
    'teacher_ResNet112':       '#1f77b4',
    'student_ResNet56_pure':   '#d62728',
    'student_ResNet56_logit':  '#ff7f0e',
    'student_ResNet56_factor': '#2ca02c',
}
LAYERS = ['layer1', 'layer2', 'layer3']
LAYER_CHANNELS = {'layer1': 16, 'layer2': 32, 'layer3': 64}
LAYER_LABELS = {
    'layer1': 'Layer 1 (16 ch)',
    'layer2': 'Layer 2 (32 ch)',
    'layer3': 'Layer 3 (64 ch)',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_representations(dataset, seeds=SEEDS, reps_dir='analysis/representations'):
    """Load representations for all models and seeds.

    Returns: {model_key: {seed: {layer1: array, layer2: ..., logits: ..., labels: ...}}}
    """
    reps = {}
    for key in MODEL_KEYS:
        reps[key] = {}
        for seed in seeds:
            path = Path(reps_dir) / dataset / f'{key}_seed{seed}.npz'
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping")
                continue
            data = np.load(path)
            reps[key][seed] = {k: data[k] for k in data.files}
    return reps


def available_seeds(reps, key):
    """Return sorted list of seeds loaded for a given model key."""
    return sorted(reps[key].keys())


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def pca(X):
    """PCA via covariance eigendecomposition.
    Returns (eigenvalues, eigenvectors, cumulative_variance_ratio)."""
    X_c = X - X.mean(axis=0)
    cov = np.cov(X_c, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    vals = np.maximum(vals, 0)
    cum_var = np.cumsum(vals) / vals.sum()
    return vals, vecs, cum_var


def effective_dim(cum_var, threshold):
    """Minimum components to reach threshold fraction of variance."""
    return int(np.searchsorted(cum_var, threshold) + 1)


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al., 2019)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X, 'fro') ** 2
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return num / den


def principal_angles_deg(U, V):
    """Principal angles (degrees) between subspaces spanned by columns of U, V."""
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    svals = np.linalg.svd(U.T @ V, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    return np.degrees(np.arccos(svals))


def fisher_criterion(X, labels):
    """Trace(S_between) / Trace(S_within)."""
    classes = np.unique(labels)
    mu = X.mean(axis=0)
    Sb, Sw = 0.0, 0.0
    for c in classes:
        Xc = X[labels == c]
        mc = Xc.mean(axis=0)
        Sb += len(Xc) * np.sum((mc - mu) ** 2)
        Sw += np.sum((Xc - mc) ** 2)
    return Sb / Sw if Sw > 0 else 0.0


def ica_cross_correlation(X_t, X_s, n_comp):
    """Run ICA on teacher and student, return matched correlations.

    Returns (matched_abs_corr, abs_cross_corr_reordered) or (None, None) on failure.
    """
    X_t_norm = (X_t - X_t.mean(0)) / (X_t.std(0) + 1e-10)
    X_s_norm = (X_s - X_s.mean(0)) / (X_s.std(0) + 1e-10)
    ica_t = FastICA(n_components=n_comp, random_state=42, max_iter=2000, tol=1e-3)
    ica_s = FastICA(n_components=n_comp, random_state=42, max_iter=2000, tol=1e-3)
    S_t = ica_t.fit_transform(X_t_norm)
    S_s = ica_s.fit_transform(X_s_norm)

    A = S_t - S_t.mean(0)
    B = S_s - S_s.mean(0)
    std_a, std_b = A.std(0), B.std(0)
    ok_a, ok_b = std_a > 1e-10, std_b > 1e-10
    A[:, ok_a] /= std_a[ok_a]
    B[:, ok_b] /= std_b[ok_b]
    cross_corr = (A.T @ B) / len(A)
    cross_corr[~ok_a, :] = 0
    cross_corr[:, ~ok_b] = 0
    cross_corr = np.nan_to_num(cross_corr)

    row_ind, col_ind = linear_sum_assignment(-np.abs(cross_corr))
    matched_corr = np.abs(cross_corr[row_ind, col_ind])
    corr_display = np.abs(cross_corr[:, col_ind])
    return matched_corr, corr_display


# ---------------------------------------------------------------------------
# Analysis 1: PCA cumulative variance (mean +/- std shading)
# ---------------------------------------------------------------------------

def analyze_pca_variance(reps, fig_dir):
    print("\n[1/7] PCA variance curves...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col, layer in enumerate(LAYERS):
        ax = axes[col]
        for key in MODEL_KEYS:
            seeds = available_seeds(reps, key)
            curves = []
            for seed in seeds:
                _, _, cum = pca(reps[key][seed][layer])
                curves.append(cum * 100)
            curves = np.array(curves)
            mean_c = curves.mean(axis=0)
            std_c = curves.std(axis=0)
            x = np.arange(1, len(mean_c) + 1)
            ax.plot(x, mean_c, label=SHORT[key], color=COLORS[key], linewidth=2)
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color=COLORS[key], alpha=0.15)

        ax.set_title(LAYER_LABELS[layer])
        ax.set_xlabel('# Principal Components')
        ax.set_ylabel('Cumulative Variance (%)')
        ax.set_ylim(0, 105)
        ax.axhline(90, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax.axhline(95, color='gray', ls=':', alpha=0.5, lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'pca_variance.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> pca_variance.png")


# ---------------------------------------------------------------------------
# Analysis 2: Effective dimensionality (mean +/- std error bars)
# ---------------------------------------------------------------------------

def analyze_effective_dim(reps, fig_dir):
    print("\n[2/7] Effective dimensionality...")
    thresholds = [0.90, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_models = len(MODEL_KEYS)
    width = 0.8 / n_models

    for t_idx, thresh in enumerate(thresholds):
        ax = axes[t_idx]
        x = np.arange(len(LAYERS))

        for i, key in enumerate(MODEL_KEYS):
            seeds = available_seeds(reps, key)
            dims_per_seed = []
            for seed in seeds:
                dims = []
                for layer in LAYERS:
                    _, _, cum = pca(reps[key][seed][layer])
                    dims.append(effective_dim(cum, thresh))
                dims_per_seed.append(dims)
            arr = np.array(dims_per_seed, dtype=float)
            mean_d = arr.mean(axis=0)
            std_d = arr.std(axis=0)

            ax.bar(x + i * width, mean_d, width, yerr=std_d,
                   label=SHORT[key], color=COLORS[key], capsize=3)
            for j, (m, s) in enumerate(zip(mean_d, std_d)):
                ax.text(x[j] + i * width, m + s + 0.3, f'{m:.1f}',
                        ha='center', fontsize=8)

        ax.set_title(f'Effective Dimensionality ({int(thresh*100)}% variance)')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
        ax.set_ylabel('# Components')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'effective_dim.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> effective_dim.png")


# ---------------------------------------------------------------------------
# Analysis 3: CKA (averaged across seeds)
# ---------------------------------------------------------------------------

def analyze_cka(reps, fig_dir):
    print("\n[3/7] CKA analysis...")
    teacher = MODEL_KEYS[0]
    students = MODEL_KEYS[1:]
    seeds = available_seeds(reps, teacher)

    # --- Cross-layer CKA heatmaps (averaged) ---
    titles = ['Teacher (self)']
    mean_matrices = []

    # Teacher self
    mats = []
    for seed in seeds:
        m = np.zeros((3, 3))
        for i, li in enumerate(LAYERS):
            for j, lj in enumerate(LAYERS):
                m[i, j] = linear_cka(reps[teacher][seed][li], reps[teacher][seed][lj])
        mats.append(m)
    mean_matrices.append(np.mean(mats, axis=0))

    for s_key in students:
        mats = []
        for seed in seeds:
            m = np.zeros((3, 3))
            for i, li in enumerate(LAYERS):
                for j, lj in enumerate(LAYERS):
                    m[i, j] = linear_cka(reps[teacher][seed][li], reps[s_key][seed][lj])
            mats.append(m)
        mean_matrices.append(np.mean(mats, axis=0))
        titles.append(f'Teacher vs {SHORT[s_key]}')

    n_panels = len(mean_matrices)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for idx, (ax, mat, title) in enumerate(zip(axes, mean_matrices, titles)):
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['L1', 'L2', 'L3'])
        ax.set_yticks(range(3))
        ax.set_yticklabels(['L1', 'L2', 'L3'])
        if idx == 0:
            ax.set_ylabel('Layer')
            ax.set_xlabel('Layer')
        else:
            ax.set_ylabel('Teacher layer')
            ax.set_xlabel('Student layer')
        for i in range(3):
            for j in range(3):
                color = 'white' if mat[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=10, color=color)

    fig.colorbar(im, ax=list(axes), shrink=0.8, label='Linear CKA')
    fig.tight_layout()
    fig.savefig(fig_dir / 'cka_cross_layer.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> cka_cross_layer.png")

    # --- Same-layer CKA bar chart (with error bars) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LAYERS))
    n_s = len(students)
    width = 0.8 / n_s

    for i, s_key in enumerate(students):
        vals_per_seed = []
        for seed in seeds:
            vals = [linear_cka(reps[teacher][seed][l], reps[s_key][seed][l])
                    for l in LAYERS]
            vals_per_seed.append(vals)
        arr = np.array(vals_per_seed)
        mean_v = arr.mean(axis=0)
        std_v = arr.std(axis=0)

        ax.bar(x + i * width, mean_v, width, yerr=std_v,
               label=SHORT[s_key], color=COLORS[s_key], capsize=3)
        for j, (m, s) in enumerate(zip(mean_v, std_v)):
            ax.text(x[j] + i * width, m + s + 0.01, f'{m:.3f}',
                    ha='center', fontsize=8)

    ax.set_title('Same-Layer CKA: Teacher vs Students')
    ax.set_xticks(x + width * (n_s - 1) / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('Linear CKA')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'cka_same_layer.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> cka_same_layer.png")


# ---------------------------------------------------------------------------
# Analysis 4: Principal angles (mean +/- std shading)
# ---------------------------------------------------------------------------

def analyze_principal_angles(reps, fig_dir):
    print("\n[4/7] Principal angles...")
    teacher = MODEL_KEYS[0]
    students = MODEL_KEYS[1:]
    seeds = available_seeds(reps, teacher)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col, layer in enumerate(LAYERS):
        ax = axes[col]

        for s_key in students:
            angle_sets = []
            for seed in seeds:
                _, vecs_t, cum_t = pca(reps[teacher][seed][layer])
                _, vecs_s, cum_s = pca(reps[s_key][seed][layer])
                k = max(effective_dim(cum_t, 0.95), effective_dim(cum_s, 0.95))
                k = min(k, vecs_t.shape[1], vecs_s.shape[1])
                angles = principal_angles_deg(vecs_t[:, :k], vecs_s[:, :k])
                angle_sets.append(angles)

            # Truncate to minimum length across seeds for consistent averaging
            min_len = min(len(a) for a in angle_sets)
            angle_arr = np.array([a[:min_len] for a in angle_sets])
            mean_a = angle_arr.mean(axis=0)
            std_a = angle_arr.std(axis=0)
            x = np.arange(1, min_len + 1)

            ax.plot(x, mean_a, label=SHORT[s_key], color=COLORS[s_key],
                    linewidth=2, marker='o', markersize=3)
            ax.fill_between(x, mean_a - std_a, mean_a + std_a,
                            color=COLORS[s_key], alpha=0.15)

        ax.set_title(LAYER_LABELS[layer])
        ax.set_xlabel('Component index')
        ax.set_ylabel('Principal angle (degrees)')
        ax.set_ylim(-2, 92)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'principal_angles.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> principal_angles.png")


# ---------------------------------------------------------------------------
# Analysis 5: ICA component matching (averaged summary)
# ---------------------------------------------------------------------------

def analyze_ica(reps, fig_dir):
    print("\n[5/7] ICA component analysis...")
    teacher = MODEL_KEYS[0]
    students = MODEL_KEYS[1:]
    seeds = available_seeds(reps, teacher)
    ref_seed = seeds[0]

    # --- Heatmaps for reference seed (representative) ---
    n_students = len(students)
    fig, axes = plt.subplots(n_students, 3, figsize=(15, 4.5 * n_students))
    if n_students == 1:
        axes = axes[np.newaxis, :]

    for row, s_key in enumerate(students):
        for col, layer in enumerate(LAYERS):
            ax = axes[row, col]
            X_t = reps[teacher][ref_seed][layer]
            X_s = reps[s_key][ref_seed][layer]
            n_comp = X_t.shape[1]

            try:
                _, corr_display = ica_cross_correlation(X_t, X_s, n_comp)
            except Exception as e:
                print(f"  ICA failed for {SHORT[s_key]} {layer} seed {ref_seed}: {e}")
                ax.text(0.5, 0.5, 'ICA failed', transform=ax.transAxes, ha='center')
                continue

            im = ax.imshow(corr_display, cmap='YlOrRd', vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest')
            ax.set_title(LAYER_LABELS[layer])
            if col == 0:
                ax.set_ylabel(f'{SHORT[s_key]}\nTeacher IC')
            ax.set_xlabel('Student IC (matched)')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='|Correlation|')
    fig.suptitle(f'ICA Component Matching (seed {ref_seed})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / 'ica_correlation.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> ica_correlation.png")

    # --- Compute summary metrics across all seeds ---
    summary_all = {}  # (s_key, layer) -> list of {mean_corr, n_strong, n_total}

    for s_key in students:
        for layer in LAYERS:
            seed_metrics = []
            for seed in seeds:
                X_t = reps[teacher][seed][layer]
                X_s = reps[s_key][seed][layer]
                n_comp = X_t.shape[1]
                try:
                    matched_corr, _ = ica_cross_correlation(X_t, X_s, n_comp)
                except Exception:
                    continue
                seed_metrics.append({
                    'mean_corr': matched_corr.mean(),
                    'n_strong': int((matched_corr > 0.5).sum()),
                    'n_total': n_comp,
                })
            summary_all[(s_key, layer)] = seed_metrics

    # --- Summary bar charts (averaged over seeds) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(LAYERS))
    width = 0.8 / n_students

    # Mean matched correlation
    ax = axes[0]
    for i, s_key in enumerate(students):
        vals = [np.mean([m['mean_corr'] for m in summary_all[(s_key, l)]])
                for l in LAYERS]
        stds = [np.std([m['mean_corr'] for m in summary_all[(s_key, l)]])
                for l in LAYERS]
        ax.bar(x + i * width, vals, width, yerr=stds,
               label=SHORT[s_key], color=COLORS[s_key], capsize=3)
        for j, (v, s) in enumerate(zip(vals, stds)):
            ax.text(x[j] + i * width, v + s + 0.01, f'{v:.2f}',
                    ha='center', fontsize=8)
    ax.set_title('Mean Matched |Correlation| (avg over seeds)')
    ax.set_xticks(x + width * (n_students - 1) / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('|Correlation|')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Count of strongly matched components
    ax = axes[1]
    for i, s_key in enumerate(students):
        vals = [np.mean([m['n_strong'] for m in summary_all[(s_key, l)]])
                for l in LAYERS]
        stds = [np.std([m['n_strong'] for m in summary_all[(s_key, l)]])
                for l in LAYERS]
        totals = [summary_all[(s_key, l)][0]['n_total'] for l in LAYERS]
        ax.bar(x + i * width, vals, width, yerr=stds,
               label=SHORT[s_key], color=COLORS[s_key], capsize=3)
        for j, (v, s, t) in enumerate(zip(vals, stds, totals)):
            ax.text(x[j] + i * width, v + s + 0.3, f'{v:.1f}/{t}',
                    ha='center', fontsize=8)
    ax.set_title('Strongly Matched Components (|corr| > 0.5, avg)')
    ax.set_xticks(x + width * (n_students - 1) / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'ica_summary.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> ica_summary.png")


# ---------------------------------------------------------------------------
# Analysis 6: Class separability (Fisher criterion, mean +/- std)
# ---------------------------------------------------------------------------

def analyze_class_separability(reps, fig_dir):
    print("\n[6/7] Class separability...")
    seeds = available_seeds(reps, MODEL_KEYS[0])
    n_models = len(MODEL_KEYS)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LAYERS))

    for i, key in enumerate(MODEL_KEYS):
        vals_per_seed = []
        for seed in seeds:
            labels = reps[key][seed]['labels']
            vals = [fisher_criterion(reps[key][seed][l], labels) for l in LAYERS]
            vals_per_seed.append(vals)
        arr = np.array(vals_per_seed)
        mean_v = arr.mean(axis=0)
        std_v = arr.std(axis=0)

        ax.bar(x + i * width, mean_v, width, yerr=std_v,
               label=SHORT[key], color=COLORS[key], capsize=3)
        for j, (m, s) in enumerate(zip(mean_v, std_v)):
            ax.text(x[j] + i * width, m + s + m * 0.02, f'{m:.2f}',
                    ha='center', fontsize=8)

    ax.set_title('Class Separability (Fisher Criterion)')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('Between-class / Within-class variance')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'class_separability.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> class_separability.png")


# ---------------------------------------------------------------------------
# Analysis 7: PCA scatter (layer 3, teacher PC basis, representative seed)
# ---------------------------------------------------------------------------

def analyze_pca_scatter(reps, fig_dir, dataset):
    print("\n[7/7] PCA scatter plots...")
    seeds = available_seeds(reps, MODEL_KEYS[0])
    ref_seed = seeds[0]

    if dataset == 'Cifar10':
        selected_classes = list(range(10))
    else:
        selected_classes = list(range(0, 100, 10))

    cmap = plt.cm.tab10
    layer = 'layer3'

    # Use teacher's PCA basis so all models share a coordinate system
    X_teacher = reps[MODEL_KEYS[0]][ref_seed][layer]
    teacher_mean = X_teacher.mean(axis=0)
    _, vecs, _ = pca(X_teacher)
    proj = vecs[:, :2]

    n_models = len(MODEL_KEYS)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

    for i, key in enumerate(MODEL_KEYS):
        ax = axes[i]
        X = reps[key][ref_seed][layer]
        labels = reps[key][ref_seed]['labels']
        X_proj = (X - teacher_mean) @ proj

        for ci, cls in enumerate(selected_classes):
            mask = labels == cls
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                       c=[cmap(ci % 10)], s=5, alpha=0.4, label=f'Class {cls}')

        ax.set_title(DISPLAY[key])
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True, alpha=0.2)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='center right', fontsize=8,
               markerscale=3, bbox_to_anchor=(1.06, 0.5))
    fig.suptitle(f'PCA Projection ({LAYER_LABELS[layer]}, seed {ref_seed})',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(fig_dir / 'pca_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> pca_scatter.png")


# ---------------------------------------------------------------------------
# Summary table (printed to console, averaged over seeds)
# ---------------------------------------------------------------------------

def print_summary(reps, dataset):
    teacher = MODEL_KEYS[0]
    seeds = available_seeds(reps, teacher)

    print(f"\n{'='*80}")
    print(f"SUMMARY — {dataset} (averaged over seeds {seeds})")
    print(f"{'='*80}")

    # Test accuracies
    print("\nTest accuracies:")
    for key in MODEL_KEYS:
        ks = available_seeds(reps, key)
        accs = []
        for seed in ks:
            preds = np.argmax(reps[key][seed]['logits'], axis=1)
            acc = (preds == reps[key][seed]['labels']).mean() * 100
            accs.append(acc)
        m, s = np.mean(accs), np.std(accs)
        per_seed = ', '.join(f'{a:.2f}' for a in accs)
        print(f"  {DISPLAY[key]:35s} {m:.2f}% +/- {s:.2f}%  ({per_seed})")

    for layer in LAYERS:
        print(f"\n--- {LAYER_LABELS[layer]} ---")

        # Effective dimensionality
        for key in MODEL_KEYS:
            d90s, d95s = [], []
            for seed in seeds:
                _, _, cum = pca(reps[key][seed][layer])
                d90s.append(effective_dim(cum, 0.90))
                d95s.append(effective_dim(cum, 0.95))
            print(f"  {SHORT[key]:20s}  eff_dim(90%)={np.mean(d90s):5.1f}+/-{np.std(d90s):.1f}"
                  f"   eff_dim(95%)={np.mean(d95s):5.1f}+/-{np.std(d95s):.1f}")

        # CKA
        for s_key in MODEL_KEYS[1:]:
            ckas = [linear_cka(reps[teacher][s][layer], reps[s_key][s][layer])
                    for s in seeds]
            print(f"  CKA(Teacher, {SHORT[s_key]:20s}) = {np.mean(ckas):.4f} +/- {np.std(ckas):.4f}")

        # Fisher
        for key in MODEL_KEYS:
            fcs = [fisher_criterion(reps[key][s][layer], reps[key][s]['labels'])
                   for s in seeds]
            print(f"  Fisher({SHORT[key]:20s}) = {np.mean(fcs):.4f} +/- {np.std(fcs):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['Cifar10', 'Cifar100', 'all'])
    parser.add_argument('--figures-dir', default='analysis/figures')
    parser.add_argument('--reps-dir', default='analysis/representations')
    args = parser.parse_args()

    datasets = ['Cifar100', 'Cifar10'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*60}")

        fig_dir = Path(args.figures_dir) / dataset
        fig_dir.mkdir(parents=True, exist_ok=True)

        reps = load_all_representations(dataset, reps_dir=args.reps_dir)

        analyze_pca_variance(reps, fig_dir)
        analyze_effective_dim(reps, fig_dir)
        analyze_cka(reps, fig_dir)
        analyze_principal_angles(reps, fig_dir)
        analyze_ica(reps, fig_dir)
        analyze_class_separability(reps, fig_dir)
        analyze_pca_scatter(reps, fig_dir, dataset)

        print_summary(reps, dataset)
        print(f"\nAll figures saved to {fig_dir}/")


if __name__ == '__main__':
    main()
