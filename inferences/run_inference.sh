#!/bin/bash

# Crack Detection Inference Runner Script
# Usage: ./run_inference.sh <config_path> <checkpoint_path> <input_dir> <output_dir>

if [ $# -ne 4 ]; then
    echo "Usage: $0 <config_path> <checkpoint_path> <input_dir> <output_dir>"
    echo "Example: $0 /path/to/config.py /path/to/checkpoint.pth /path/to/input/images /path/to/output/results"
    exit 1
fi

CONFIG_PATH=$1
CHECKPOINT_PATH=$2
INPUT_DIR=$3
OUTPUT_DIR=$4

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hard

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "hard" ]]; then
    echo "Error: Failed to activate 'hard' environment. Please run setup.sh first."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run inference
echo "Running crack detection inference..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python inference.py \
    --crack_config "$CONFIG_PATH" \
    --crack_checkpoint "$CHECKPOINT_PATH" \
    --srx_dir "$INPUT_DIR" \
    --rst_dir "$OUTPUT_DIR"

echo "Inference completed! Results saved to: $OUTPUT_DIR"

