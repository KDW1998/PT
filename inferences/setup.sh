#!/bin/bash

# Crack Detection Inference Setup Script
# This script sets up the environment for crack detection inference

set -e  # Exit on any error

echo "=========================================="
echo "Crack Detection Inference Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "hard"; then
    echo "Environment 'hard' already exists. Removing it first..."
    conda env remove -n hard -y
fi

# Create conda environment from environment.yml
echo "Creating conda environment 'hard'..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hard

# Verify installation
echo "Verifying installation..."
python test_setup.py

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo "To use the inference script:"
echo "1. Activate the environment: conda activate hard"
echo "2. Run inference: ./run_inference.sh <config_path> <checkpoint_path> <input_dir> <output_dir>"
echo "   or: python inference.py --crack_config <config_path> --crack_checkpoint <checkpoint_path> --srx_dir <input_dir> --rst_dir <output_dir>"
echo "=========================================="
