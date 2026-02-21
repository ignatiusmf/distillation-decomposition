"""
Standalone plot generator for experiment directories.

Reads metrics.json (or checkpoint.pth) from each experiment and generates
Loss.png and Accuracy.png. Designed to run locally via cron or manually,
replacing the removed per-epoch plot_the_things() call from train.py.

Usage: python plot_experiments.py [--experiments_dir PATH]
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def find_experiment_dirs(root: Path):
    """Walk experiment tree and yield dirs containing status.json."""
    for status_file in root.rglob('status.json'):
        yield status_file.parent


def load_metrics(exp_dir: Path):
    """Load training curves from metrics.json or checkpoint.pth."""
    metrics_path = exp_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    # Fall back to checkpoint (may be corrupted if rsync pulled mid-write)
    ckpt_path = exp_dir / 'checkpoint.pth'
    if ckpt_path.exists():
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except (EOFError, RuntimeError):
            print(f"  WARN: corrupted checkpoint, skipping: {exp_dir.name}")
            return None
        return {
            'train_loss': ckpt.get('train_loss', []),
            'train_acc': ckpt.get('train_acc', []),
            'test_loss': ckpt.get('test_loss', []),
            'test_acc': ckpt.get('test_acc', []),
        }

    return None


def plot_experiment(exp_dir: Path, metrics: dict):
    """Generate Loss.png and Accuracy.png for a single experiment."""
    train_loss = metrics.get('train_loss', [])
    test_loss = metrics.get('test_loss', [])
    train_acc = metrics.get('train_acc', [])
    test_acc = metrics.get('test_acc', [])

    if not train_loss:
        return False

    # Loss plot
    fig, ax = plt.subplots()
    ax.plot(train_loss, linestyle='dotted', color='b', label='Train Loss')
    ax.plot(test_loss, linestyle='solid', color='b', label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_xlim(0, len(test_loss))
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(exp_dir / 'Loss.png', dpi=100)
    plt.close(fig)

    # Accuracy plot
    fig, ax = plt.subplots()
    ax.plot(train_acc, linestyle='dotted', color='r', label='Train Accuracy')
    ax.plot(test_acc, linestyle='solid', color='r', label='Test Accuracy')
    max_acc = max(test_acc) if test_acc else 0
    ax.axhline(y=max_acc, color='black', linestyle='-', linewidth=0.5)
    ax.text(0, max_acc + 1, f"Max Acc = {max_acc:.2f}", color='black', fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_xlim(0, len(test_acc))
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 105, 5))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.savefig(exp_dir / 'Accuracy.png', dpi=100)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate plots for all experiments')
    default_dir = Path(__file__).resolve().parent.parent / 'experiments'
    parser.add_argument('--experiments_dir', type=str, default=str(default_dir),
                        help='Root experiments directory')
    args = parser.parse_args()

    root = Path(args.experiments_dir)
    if not root.exists():
        print(f"Experiments directory not found: {root}")
        return

    dirs = list(find_experiment_dirs(root))
    plotted, skipped = 0, 0
    for exp_dir in dirs:
        metrics = load_metrics(exp_dir)
        if metrics:
            if plot_experiment(exp_dir, metrics):
                plotted += 1
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"Plotted {plotted}/{len(dirs)} experiments ({skipped} skipped)")


if __name__ == '__main__':
    main()
