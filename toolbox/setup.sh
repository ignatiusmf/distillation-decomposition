#!/bin/bash
# Run on CHPC to (re)create the Python venv from scratch.
# Use this when setting up a new environment or after dependency changes.

# Tear down any existing venv
deactivate
rm -rf ~/myenv

# Fresh venv
python -m venv ~/myenv
source ~/myenv/bin/activate

pip install --upgrade pip

# Binary-only install avoids compiling from source on the cluster
pip install --only-binary=:all: torch torchvision matplotlib numpy

python -c "import torch; print(torch.__version__)"
