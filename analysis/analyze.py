"""
Analyze extracted representations: PCA, ICA, CKA, principal angles, class separability.

Produces thesis-quality figures in analysis/figures/.

Usage:
    python analysis/analyze.py [--seed 0] [--figures-dir analysis/figures]

Prerequisites:
    python analysis/extract.py       (run first)
    uv add scikit-learn scipy        (or pip install)
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
MODEL_KEYS = [
    'teacher_ResNet112',
    'student_ResNet56_logit',
    'student_ResNet56_factor',
]
DISPLAY = {
    'teacher_ResNet112':       'Teacher (ResNet-112)',
    'student_ResNet56_logit':  'Student - Logit KD',
    'student_ResNet56_factor': 'Student - Factor Transfer',
}
SHORT = {
    'teacher_ResNet112':       'Teacher',
    'student_ResNet56_logit':  'Logit KD',
    'student_ResNet56_factor': 'Factor Transfer',
}
COLORS = {
    'teacher_ResNet112':       '#1f77b4',
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
# Utility functions
# ---------------------------------------------------------------------------

def load_representations(seed, reps_dir='analysis/representations'):
    reps = {}
    for key in MODEL_KEYS:
        path = Path(reps_dir) / f'{key}_seed{seed}.npz'
        data = np.load(path)
        reps[key] = {k: data[k] for k in data.files}
    # Print test accuracies from logits
    print("Test accuracies (from saved logits):")
    for key in MODEL_KEYS:
        preds = np.argmax(reps[key]['logits'], axis=1)
        acc = (preds == reps[key]['labels']).mean() * 100
        print(f"  {DISPLAY[key]:35s} {acc:.2f}%")
    return reps


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


# ---------------------------------------------------------------------------
# Analysis 1: PCA cumulative variance
# ---------------------------------------------------------------------------

def analyze_pca_variance(reps, fig_dir):
    print("\n[1/7] PCA variance curves...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col, layer in enumerate(LAYERS):
        ax = axes[col]
        for key in MODEL_KEYS:
            _, _, cum = pca(reps[key][layer])
            ax.plot(range(1, len(cum) + 1), cum * 100,
                    label=SHORT[key], color=COLORS[key], linewidth=2)
        ax.set_title(LAYER_LABELS[layer])
        ax.set_xlabel('# Principal Components')
        ax.set_ylabel('Cumulative Variance Explained (%)')
        ax.set_ylim(0, 105)
        ax.axhline(90, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax.axhline(95, color='gray', ls=':', alpha=0.5, lw=0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'pca_variance.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> pca_variance.png")


# ---------------------------------------------------------------------------
# Analysis 2: Effective dimensionality
# ---------------------------------------------------------------------------

def analyze_effective_dim(reps, fig_dir):
    print("\n[2/7] Effective dimensionality...")
    thresholds = [0.90, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for t_idx, thresh in enumerate(thresholds):
        ax = axes[t_idx]
        x = np.arange(len(LAYERS))
        width = 0.25

        for i, key in enumerate(MODEL_KEYS):
            dims = []
            for layer in LAYERS:
                _, _, cum = pca(reps[key][layer])
                dims.append(effective_dim(cum, thresh))
            bars = ax.bar(x + i * width, dims, width,
                          label=SHORT[key], color=COLORS[key])
            for j, d in enumerate(dims):
                ax.text(x[j] + i * width, d + 0.3, str(d),
                        ha='center', fontsize=9)

        ax.set_title(f'Effective Dimensionality ({int(thresh*100)}% variance)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
        ax.set_ylabel('# Components')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'effective_dim.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> effective_dim.png")


# ---------------------------------------------------------------------------
# Analysis 3: CKA
# ---------------------------------------------------------------------------

def analyze_cka(reps, fig_dir):
    print("\n[3/7] CKA analysis...")
    teacher = MODEL_KEYS[0]
    students = MODEL_KEYS[1:]

    # --- Cross-layer CKA heatmaps ---
    titles = ['Teacher (self)']
    matrices = []

    # Teacher self
    m = np.zeros((3, 3))
    for i, li in enumerate(LAYERS):
        for j, lj in enumerate(LAYERS):
            m[i, j] = linear_cka(reps[teacher][li], reps[teacher][lj])
    matrices.append(m)

    for s_key in students:
        m = np.zeros((3, 3))
        for i, li in enumerate(LAYERS):
            for j, lj in enumerate(LAYERS):
                m[i, j] = linear_cka(reps[teacher][li], reps[s_key][lj])
        matrices.append(m)
        titles.append(f'Teacher vs {SHORT[s_key]}')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, (ax, mat, title) in enumerate(zip(axes, matrices, titles)):
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title)
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

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Linear CKA')
    fig.tight_layout()
    fig.savefig(fig_dir / 'cka_cross_layer.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> cka_cross_layer.png")

    # --- Same-layer CKA bar chart ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LAYERS))
    width = 0.3

    for i, s_key in enumerate(students):
        vals = [linear_cka(reps[teacher][l], reps[s_key][l]) for l in LAYERS]
        ax.bar(x + i * width, vals, width,
               label=SHORT[s_key], color=COLORS[s_key])
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f'{v:.3f}',
                    ha='center', fontsize=9)

    ax.set_title('Same-Layer CKA: Teacher vs Students')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('Linear CKA')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'cka_same_layer.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> cka_same_layer.png")


# ---------------------------------------------------------------------------
# Analysis 4: Principal angles
# ---------------------------------------------------------------------------

def analyze_principal_angles(reps, fig_dir):
    print("\n[4/7] Principal angles...")
    teacher = MODEL_KEYS[0]
    students = MODEL_KEYS[1:]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col, layer in enumerate(LAYERS):
        ax = axes[col]
        _, vecs_t, cum_t = pca(reps[teacher][layer])

        for s_key in students:
            _, vecs_s, cum_s = pca(reps[s_key][layer])

            # Subspace dimension: cover 95% variance for the more demanding model
            k = max(effective_dim(cum_t, 0.95), effective_dim(cum_s, 0.95))
            k = min(k, vecs_t.shape[1], vecs_s.shape[1])

            angles = principal_angles_deg(vecs_t[:, :k], vecs_s[:, :k])
            ax.plot(range(1, len(angles) + 1), angles,
                    label=SHORT[s_key], color=COLORS[s_key],
                    linewidth=2, marker='o', markersize=3)

        ax.set_title(LAYER_LABELS[layer])
        ax.set_xlabel('Component index')
        ax.set_ylabel('Principal angle (degrees)')
        ax.set_ylim(-2, 92)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'principal_angles.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> principal_angles.png")


# ---------------------------------------------------------------------------
# Analysis 5: ICA component matching
# ---------------------------------------------------------------------------

def analyze_ica(reps, fig_dir):
    print("\n[5/7] ICA component analysis...")
    teacher = MODEL_KEYS[0]
    students = MODEL_KEYS[1:]

    summary = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, s_key in enumerate(students):
        for col, layer in enumerate(LAYERS):
            ax = axes[row, col]
            X_t = reps[teacher][layer]
            X_s = reps[s_key][layer]
            n_comp = X_t.shape[1]

            try:
                # Standardize inputs to avoid overflow in ICA
                X_t_norm = (X_t - X_t.mean(0)) / (X_t.std(0) + 1e-10)
                X_s_norm = (X_s - X_s.mean(0)) / (X_s.std(0) + 1e-10)
                ica_t = FastICA(n_components=n_comp, random_state=42,
                                max_iter=2000, tol=1e-3)
                ica_s = FastICA(n_components=n_comp, random_state=42,
                                max_iter=2000, tol=1e-3)
                S_t = ica_t.fit_transform(X_t_norm)
                S_s = ica_s.fit_transform(X_s_norm)
            except Exception as e:
                print(f"  ICA failed for {SHORT[s_key]} {layer}: {e}")
                ax.text(0.5, 0.5, 'ICA failed', transform=ax.transAxes,
                        ha='center')
                continue

            # Cross-correlation (vectorized, safe for zero-variance components)
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

            # Optimal matching via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-np.abs(cross_corr))
            matched_corr = np.abs(cross_corr[row_ind, col_ind])

            # Reorder columns so matched pairs sit on the diagonal
            corr_display = np.abs(cross_corr[:, col_ind])

            im = ax.imshow(corr_display, cmap='YlOrRd', vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest')
            ax.set_title(LAYER_LABELS[layer])
            if col == 0:
                ax.set_ylabel(f'{SHORT[s_key]}\nTeacher IC')
            ax.set_xlabel('Student IC (matched)')

            summary[(s_key, layer)] = {
                'mean_corr': matched_corr.mean(),
                'n_strong': int((matched_corr > 0.5).sum()),
                'n_total': n_comp,
                'matched_corr': matched_corr,
            }

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='|Correlation|')
    fig.tight_layout()
    fig.savefig(fig_dir / 'ica_correlation.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> ica_correlation.png")

    # --- Summary bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(LAYERS))
    width = 0.3

    # Mean matched correlation
    ax = axes[0]
    for i, s_key in enumerate(students):
        vals = [summary[(s_key, l)]['mean_corr'] for l in LAYERS]
        ax.bar(x + i * width, vals, width,
               label=SHORT[s_key], color=COLORS[s_key])
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f'{v:.2f}',
                    ha='center', fontsize=9)
    ax.set_title('Mean Matched |Correlation|')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('|Correlation|')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Count of strongly matched components
    ax = axes[1]
    for i, s_key in enumerate(students):
        vals = [summary[(s_key, l)]['n_strong'] for l in LAYERS]
        totals = [summary[(s_key, l)]['n_total'] for l in LAYERS]
        ax.bar(x + i * width, vals, width,
               label=SHORT[s_key], color=COLORS[s_key])
        for j, (v, t) in enumerate(zip(vals, totals)):
            ax.text(x[j] + i * width, v + 0.3, f'{v}/{t}',
                    ha='center', fontsize=9)
    ax.set_title('Strongly Matched Components (|corr| > 0.5)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'ica_summary.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> ica_summary.png")


# ---------------------------------------------------------------------------
# Analysis 6: Class separability (Fisher criterion)
# ---------------------------------------------------------------------------

def analyze_class_separability(reps, fig_dir):
    print("\n[6/7] Class separability...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LAYERS))
    width = 0.25

    for i, key in enumerate(MODEL_KEYS):
        labels = reps[key]['labels']
        vals = [fisher_criterion(reps[key][l], labels) for l in LAYERS]
        ax.bar(x + i * width, vals, width,
               label=SHORT[key], color=COLORS[key])
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v * 1.02, f'{v:.2f}',
                    ha='center', fontsize=8)

    ax.set_title('Class Separability (Fisher Criterion)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax.set_ylabel('Between-class / Within-class variance')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'class_separability.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> class_separability.png")


# ---------------------------------------------------------------------------
# Analysis 7: PCA scatter (layer 3, teacher PC basis, 10 classes)
# ---------------------------------------------------------------------------

def analyze_pca_scatter(reps, fig_dir):
    print("\n[7/7] PCA scatter plots...")
    selected_classes = list(range(0, 100, 10))  # 10 evenly-spaced classes
    cmap = plt.cm.tab10
    layer = 'layer3'

    # Use teacher's PCA basis so all three models are in the same coordinate system
    X_teacher = reps[MODEL_KEYS[0]][layer]
    teacher_mean = X_teacher.mean(axis=0)
    _, vecs, _ = pca(X_teacher)
    proj = vecs[:, :2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, key in enumerate(MODEL_KEYS):
        ax = axes[i]
        X = reps[key][layer]
        labels = reps[key]['labels']

        X_proj = (X - teacher_mean) @ proj

        for ci, cls in enumerate(selected_classes):
            mask = labels == cls
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                       c=[cmap(ci)], s=5, alpha=0.4, label=f'Class {cls}')

        ax.set_title(DISPLAY[key])
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True, alpha=0.2)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='center right', fontsize=8,
               markerscale=3, bbox_to_anchor=(1.06, 0.5))

    fig.suptitle(f'PCA Projection ({LAYER_LABELS[layer]}) — Teacher PC basis',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(fig_dir / 'pca_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print("  -> pca_scatter.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(reps):
    teacher = MODEL_KEYS[0]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for layer in LAYERS:
        print(f"\n--- {LAYER_LABELS[layer]} ---")

        # Effective dimensionality
        for key in MODEL_KEYS:
            _, _, cum = pca(reps[key][layer])
            d90, d95 = effective_dim(cum, 0.90), effective_dim(cum, 0.95)
            print(f"  {SHORT[key]:20s}  eff_dim(90%)={d90:2d}   eff_dim(95%)={d95:2d}")

        # CKA
        for s_key in MODEL_KEYS[1:]:
            cka = linear_cka(reps[teacher][layer], reps[s_key][layer])
            print(f"  CKA(Teacher, {SHORT[s_key]:20s}) = {cka:.4f}")

        # Fisher
        for key in MODEL_KEYS:
            fc = fisher_criterion(reps[key][layer], reps[key]['labels'])
            print(f"  Fisher({SHORT[key]:20s}) = {fc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--figures-dir', default='analysis/figures')
    args = parser.parse_args()

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    reps = load_representations(args.seed)

    analyze_pca_variance(reps, fig_dir)
    analyze_effective_dim(reps, fig_dir)
    analyze_cka(reps, fig_dir)
    analyze_principal_angles(reps, fig_dir)
    analyze_ica(reps, fig_dir)
    analyze_class_separability(reps, fig_dir)
    analyze_pca_scatter(reps, fig_dir)

    print_summary(reps)
    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == '__main__':
    main()
