#!/bin/bash
# Setup script for Georgia Tech PACE cluster
# Loads required modules and activates conda environment

set -e

echo "Setting up cluster environment..."

# TODO: Customize for your specific cluster setup

# Load required modules
# module load anaconda3/2023.03
# module load cuda/12.1

# Activate conda environment
# conda activate probe-evasion

# Verify GPU availability
# python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

echo "Cluster setup not yet implemented"
echo "Edit this script with your cluster-specific configuration"
exit 1
