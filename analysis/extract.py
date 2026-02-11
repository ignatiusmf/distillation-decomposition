"""
Extract GAP-pooled representations from trained models.

Run this once to save representations to .npz files.
The analysis script only needs those files (no GPU required).

Usage:
    python analysis/extract.py [--device cpu|cuda] [--seed 0]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import argparse

from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar100


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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    dataset = 'Cifar100'
    seed = args.seed
    device = args.device

    configs = {
        'teacher_ResNet112': (
            ResNet112,
            f'experiments/{dataset}/pure/ResNet112/{seed}/best.pth',
        ),
        'student_ResNet56_logit': (
            ResNet56,
            f'experiments/{dataset}/logit/ResNet112_to_ResNet56/{seed}/best.pth',
        ),
        'student_ResNet56_factor': (
            ResNet56,
            f'experiments/{dataset}/factor_transfer/ResNet112_to_ResNet56/{seed}/best.pth',
        ),
    }

    print(f"Device: {device} | Seed: {seed}")
    data = Cifar100(args.batch_size, seed=seed)

    out_dir = Path('analysis/representations')
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (arch, weights_path) in configs.items():
        print(f"\n--- {name} ---")
        if not Path(weights_path).exists():
            print(f"  Weights not found: {weights_path}")
            continue

        model = arch(data.class_num).to(device)
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['weights'])
        print(f"  Loaded {weights_path}")

        reps = extract(model, data.testloader, device)
        save_path = out_dir / f'{name}_seed{seed}.npz'
        np.savez(save_path, **reps)
        print(f"  Saved {save_path}")
        for k, v in reps.items():
            print(f"    {k}: {v.shape}")

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
