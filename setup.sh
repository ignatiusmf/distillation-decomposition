#!/bin/bash
# This can be run on the CHPC to setup a venv
# Deactivate and remove the venv
deactivate
rm -rf ~/myenv

# Recreate fresh venv
python -m venv ~/myenv
source ~/myenv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install with binary-only flag
pip install --only-binary=:all: torch torchvision matplotlib numpy

# Test
python -c "import torch; print(torch.__version__)"
