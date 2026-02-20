#!/bin/bash

# Push local code changes up to CHPC. Excludes experiments/, analysis/, .git, .venv
# so only source files are synced — not results or large data.
rsync -avz --delete --exclude='analysis' --exclude='.git' --exclude='.venv' --exclude='experiments/' /home/ignatius/Lab/studies/repos/distillation-decomposition/ iferreira@lengau.chpc.ac.za:/home/iferreira/lustre/distillation-decomposition/

# Pull experiment results back from CHPC. Only syncs experiments/ — nothing else comes down.
rsync -avz --delete --include='experiments/***' --exclude='*' iferreira@lengau.chpc.ac.za:/home/iferreira/lustre/distillation-decomposition/ /home/ignatius/Lab/studies/repos/distillation-decomposition/

