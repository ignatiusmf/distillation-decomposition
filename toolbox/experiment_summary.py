"""
Experiment Charlie — graphical summary generator.

Scans experiments/ directory, reads status.json and metrics.json from each
experiment, and generates overview figures to analysis/experiment_charlie/.

Figures:
  1. progress_grid.png     — Completion grid (method x dataset x alpha)
  2. accuracy_overview.png  — Bar chart of max accuracy for completed experiments

Usage: python toolbox/experiment_summary.py [--experiments_dir PATH] [--output_dir PATH]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================================
# DATA COLLECTION
# ============================================================================

METHODS = ['logit', 'factor_transfer', 'attention_transfer', 'fitnets', 'rkd', 'nst']
ALPHAS = [0.25, 0.5, 0.75, 0.95]
DATASETS = ['Cifar10', 'Cifar100', 'SVHN', 'TinyImageNet']
SEEDS = [0, 1, 2]
METHOD_SHORT = {
    'logit': 'Logit', 'factor_transfer': 'FT', 'attention_transfer': 'AT',
    'fitnets': 'FitNets', 'rkd': 'RKD', 'nst': 'NST',
    'pure_teacher': 'Teacher', 'pure_student': 'Student',
}


def scan_experiments(root: Path):
    """Scan all experiment directories and return structured data."""
    experiments = []

    for status_file in root.rglob('status.json'):
        exp_dir = status_file.parent
        rel = exp_dir.relative_to(root)
        parts = list(rel.parts)

        with open(status_file) as f:
            status = json.load(f)

        info = {
            'path': str(rel),
            'dir': exp_dir,
            'status': status.get('status', 'queued'),
            'epoch': status.get('epoch', 0),
            'max_acc': status.get('max_acc', 0),
            'config': status.get('config', {}),
        }

        # Parse path to extract dataset/method/alpha/seed
        cfg = info['config']
        info['dataset'] = cfg.get('dataset', parts[0] if parts else '?')
        info['seed'] = cfg.get('seed', int(parts[-1]) if parts else 0)
        info['distillation'] = cfg.get('distillation', 'none')
        info['alpha'] = cfg.get('alpha', None)
        info['model'] = cfg.get('model', '?')

        # Load accuracy history if metrics.json exists
        metrics_path = exp_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            info['test_acc'] = metrics.get('test_acc', [])
        else:
            info['test_acc'] = []

        experiments.append(info)

    return experiments


# ============================================================================
# FIGURE 1: PROGRESS GRID
# ============================================================================

def plot_progress_grid(experiments, output_dir: Path):
    """
    Grid showing completion status.
    Rows: methods (+ pure baselines)
    Columns: datasets
    Each cell: 4 alpha columns x 3 seed rows (or 1x3 for pure)
    Color: green=completed, yellow=in_progress, red=failed/queued, grey=not started
    """
    # Build lookup: (dataset, method, alpha, seed) -> status
    lookup = {}
    for exp in experiments:
        ds = exp['dataset']
        method = exp['distillation']
        alpha = exp['alpha']
        seed = exp['seed']
        model = exp['model']

        if method == 'none':
            key = (ds, f'pure_{model.lower()}', None, seed)
        else:
            key = (ds, method, alpha, seed)
        lookup[key] = exp['status']

    row_labels = ['pure_teacher', 'pure_student'] + METHODS
    n_rows = len(row_labels)
    n_cols = len(DATASETS)

    # Each cell is a small grid: alphas wide x seeds tall
    # Pure cells: 1 wide x 3 tall; KD cells: 4 wide x 3 tall
    cell_w = len(ALPHAS)
    cell_h = len(SEEDS)

    fig_w = 2 + n_cols * (cell_w * 0.35 + 0.3)
    fig_h = 1.5 + n_rows * (cell_h * 0.25 + 0.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    status_colors = {
        'completed': '#4CAF50',
        'in_progress': '#FFC107',
        'queued': '#FF5722',
        None: '#E0E0E0',  # not started
    }

    for row_idx, method in enumerate(row_labels):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(0, cell_w)
            ax.set_ylim(0, cell_h)
            ax.set_aspect('equal')

            is_pure = method.startswith('pure_')

            if is_pure:
                # Only 1 column (no alpha dimension)
                for s_idx, seed in enumerate(SEEDS):
                    model_name = 'ResNet112' if 'teacher' in method else 'ResNet56'
                    key = (dataset, method.replace('resnet', 'ResNet'), None, seed)
                    # Try lookup
                    status = None
                    for exp in experiments:
                        if (exp['dataset'] == dataset and
                            exp['distillation'] == 'none' and
                            exp['model'] == model_name and
                            exp['seed'] == seed):
                            status = exp['status']
                            break
                    color = status_colors.get(status, status_colors[None])
                    # Center the single column
                    rect = plt.Rectangle((1.5, cell_h - 1 - s_idx), 1, 1,
                                         facecolor=color, edgecolor='white', linewidth=0.5)
                    ax.add_patch(rect)
            else:
                for a_idx, alpha in enumerate(ALPHAS):
                    for s_idx, seed in enumerate(SEEDS):
                        key = (dataset, method, alpha, seed)
                        status = lookup.get(key)
                        color = status_colors.get(status, status_colors[None])
                        rect = plt.Rectangle((a_idx, cell_h - 1 - s_idx), 1, 1,
                                             facecolor=color, edgecolor='white', linewidth=0.5)
                        ax.add_patch(rect)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax.set_title(dataset, fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(METHOD_SHORT.get(method, method),
                             fontsize=8, rotation=0, labelpad=50, va='center')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#4CAF50', label='Completed'),
        mpatches.Patch(color='#FFC107', label='In progress'),
        mpatches.Patch(color='#FF5722', label='Queued/failed'),
        mpatches.Patch(color='#E0E0E0', label='Not started'),
    ]

    # Count stats
    completed = sum(1 for e in experiments if e['status'] == 'completed')
    total = len(experiments)
    in_progress = sum(1 for e in experiments if e['status'] == 'in_progress')
    expected = len(METHODS) * len(ALPHAS) * len(DATASETS) * len(SEEDS) + 2 * len(DATASETS) * len(SEEDS)

    fig.suptitle(
        f'Experiment Charlie — {completed}/{expected} completed ({completed/expected*100:.1f}%)\n'
        f'{in_progress} in progress, {expected - total} not yet started\n'
        f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        fontsize=11, fontweight='bold', y=0.98
    )
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0.12, 0.04, 1.0, 0.90])
    fig.savefig(output_dir / 'progress_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved progress_grid.png')


# ============================================================================
# FIGURE 2: ACCURACY OVERVIEW
# ============================================================================

def plot_accuracy_overview(experiments, output_dir: Path):
    """Bar chart of max accuracy for completed experiments, grouped by dataset."""
    completed = [e for e in experiments if e['status'] == 'completed' and e['max_acc'] > 0]
    if not completed:
        print('  No completed experiments for accuracy overview')
        return

    # Group by dataset
    by_dataset = defaultdict(list)
    for exp in completed:
        label = exp['distillation']
        if label == 'none':
            label = exp['model']
        else:
            label = f"{METHOD_SHORT.get(label, label)} a={exp['alpha']}"
        by_dataset[exp['dataset']].append({
            'label': label,
            'acc': exp['max_acc'],
            'seed': exp['seed'],
            'method': exp['distillation'],
        })

    n_datasets = len(by_dataset)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 6), squeeze=False)

    method_colors = {
        'none': '#607D8B', 'logit': '#2196F3', 'factor_transfer': '#FF9800',
        'attention_transfer': '#4CAF50', 'fitnets': '#9C27B0',
        'rkd': '#F44336', 'nst': '#00BCD4',
    }

    for idx, (dataset, exps) in enumerate(sorted(by_dataset.items())):
        ax = axes[0, idx]

        # Average across seeds
        acc_by_label = defaultdict(list)
        method_by_label = {}
        for e in exps:
            acc_by_label[e['label']].append(e['acc'])
            method_by_label[e['label']] = e['method']

        labels = sorted(acc_by_label.keys(),
                        key=lambda l: np.mean(acc_by_label[l]), reverse=True)
        means = [np.mean(acc_by_label[l]) for l in labels]
        stds = [np.std(acc_by_label[l]) for l in labels]
        colors = [method_colors.get(method_by_label[l], '#999') for l in labels]

        bars = ax.barh(range(len(labels)), means, xerr=stds, color=colors,
                       edgecolor='white', linewidth=0.5, capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Test Accuracy (%)', fontsize=8)
        ax.set_title(dataset, fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

        # Annotate bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 1, i, f'{m:.1f}', va='center', fontsize=7)

    fig.suptitle('Experiment Charlie — Accuracy Overview (completed experiments)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / 'accuracy_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved accuracy_overview.png')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate experiment charlie summary figures')
    default_exp = Path(__file__).resolve().parent.parent / 'experiments'
    default_out = Path(__file__).resolve().parent.parent / 'analysis' / 'experiment_charlie'
    parser.add_argument('--experiments_dir', type=str, default=str(default_exp))
    parser.add_argument('--output_dir', type=str, default=str(default_out))
    args = parser.parse_args()

    root = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f'Experiments directory not found: {root}')
        return

    print(f'Scanning {root}...')
    experiments = scan_experiments(root)
    print(f'Found {len(experiments)} experiments')

    print('Generating figures...')
    plot_progress_grid(experiments, output_dir)
    plot_accuracy_overview(experiments, output_dir)

    # Print text summary
    completed = sum(1 for e in experiments if e['status'] == 'completed')
    in_progress = sum(1 for e in experiments if e['status'] == 'in_progress')
    expected = len(METHODS) * len(ALPHAS) * len(DATASETS) * len(SEEDS) + 2 * len(DATASETS) * len(SEEDS)
    print(f'\nExperiment Charlie: {completed}/{expected} completed ({completed/expected*100:.1f}%)')
    print(f'  In progress: {in_progress}')
    print(f'  Not started: {expected - completed - in_progress}')
    print(f'\nFigures saved to {output_dir}/')


if __name__ == '__main__':
    main()
