This summary is outdated. Run the updated analysis pipeline to regenerate figures:

    python analysis/extract.py --dataset all
    python analysis/analyze.py --dataset all

Figures are now saved per-dataset under analysis/figures/Cifar10/ and analysis/figures/Cifar100/.
All metrics are averaged over 3 seeds (0, 1, 2) with std error bars.
The pure ResNet-56 (no distillation) baseline is now included alongside Logit KD and Factor Transfer.
