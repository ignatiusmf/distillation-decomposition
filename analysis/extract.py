"""
Extract GAP-pooled representations from trained models.

Run this once to save representations to .npz files.
The analysis script only needs those files (no GPU required).

Usage:
    python analysis/extract.py [--device cpu|cuda] [--dataset all|Cifar10|Cifar100]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import argparse

from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar10, Cifar100

DATASETS = {
    'Cifar10':  {'loader': Cifar10,  'class_num': 10},
    'Cifar100': {'loader': Cifar100, 'class_num': 100},
}

SEEDS = [0, 1, 2]

MODEL_CONFIGS = {
    'teacher_ResNet112': {
        'arch': ResNet112,
        'path_template': 'experiments/{dataset}/pure/ResNet112/{seed}/best.pth',
    },
    'student_ResNet56_pure': {
        'arch': ResNet56,
        'path_template': 'experiments/{dataset}/pure/ResNet56/{seed}/best.pth',
    },
    'student_ResNet56_logit': {
        'arch': ResNet56,
        'path_template': 'experiments/{dataset}/logit/ResNet112_to_ResNet56/{seed}/best.pth',
    },
    'student_ResNet56_factor': {
        'arch': ResNet56,
        'path_template': 'experiments/{dataset}/factor_transfer/ResNet112_to_ResNet56/{seed}/best.pth',
    },
}


def extract(model, dataloader, device):
    """Run model on dataset, return GAP-pooled features at each layer + logits + labels."""
    model.eval()
    layers = {f'layer{i}': [] for i in range(1, 4)}
    logits_all, labels_all = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            out = model(inputs)  # [layer1, layer2, layer3, logits]
            for i in range(3):
                # GAP-pool: (B, C, H, W) -> (B, C)
                layers[f'layer{i+1}'].append(out[i].mean(dim=[2, 3]).cpu().numpy())
            logits_all.append(out[3].cpu().numpy())
            labels_all.append(targets.numpy())

    result = {k: np.concatenate(v) for k, v in layers.items()}
    result['logits'] = np.concatenate(logits_all)
    result['labels'] = np.concatenate(labels_all)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', default='all', choices=['Cifar10', 'Cifar100', 'all'])
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]

    for dataset_name in datasets:
        ds_info = DATASETS[dataset_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        out_dir = Path(f'analysis/representations/{dataset_name}')
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            data = ds_info['loader'](args.batch_size, seed=seed)

            for name, cfg in MODEL_CONFIGS.items():
                weights_path = cfg['path_template'].format(dataset=dataset_name, seed=seed)
                if not Path(weights_path).exists():
                    print(f"  [{name}] Weights not found: {weights_path}")
                    continue

                model = cfg['arch'](data.class_num).to(args.device)
                ckpt = torch.load(weights_path, map_location=args.device, weights_only=True)
                model.load_state_dict(ckpt['weights'])

                reps = extract(model, data.testloader, args.device)
                save_path = out_dir / f'{name}_seed{seed}.npz'
                np.savez(save_path, **reps)

                acc = (np.argmax(reps['logits'], axis=1) == reps['labels']).mean() * 100
                print(f"  [{name}] acc={acc:.2f}%  saved -> {save_path}")

                del model
                if args.device == 'cuda':
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
