#!/bin/bash
# Training script with logging - run this in tmux

# Set up environment
cd /data/mehmet/projects/mixture-of-LoRA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mixture-lora

# Set timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "=========================================="
echo "Starting LoRA Training for text_to_sql"
echo "Log file: ${LOG_FILE}"
echo "Started at: $(date)"
echo "=========================================="

# Run training with evaluation after
# Using 7000 samples to match your previous training
python main.py train \
    --domain text_to_sql \
    --max-samples 7000 \
    --eval \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="
