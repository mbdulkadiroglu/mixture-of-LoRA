#!/bin/bash
# Monitor BIRD v3 training progress and save to JSON in real time
LOGFILE="results/bird_v3_training.log"
JSONFILE="results/bird_v3_training_progress.json"

while true; do
    if [ -f "$LOGFILE" ]; then
        # Extract the latest training progress line
        PROGRESS=$(grep -oP '\d+/1239' "$LOGFILE" | tail -1)
        LOSS=$(grep -oP "'loss': [\d.]+" "$LOGFILE" | tail -1 | grep -oP '[\d.]+$')
        LR=$(grep -oP "'learning_rate': [\de.-]+" "$LOGFILE" | tail -1 | grep -oP '[\de.-]+$')
        EPOCH=$(grep -oP "'epoch': [\d.]+" "$LOGFILE" | tail -1 | grep -oP '[\d.]+$')
        GPU_UTIL=$(nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        GPU_MEM=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        GPU_TEMP=$(nvidia-smi -i 3 --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
        TIMESTAMP=$(date -Iseconds)

        cat > "$JSONFILE" << EOF
{
  "task": "BIRD v3 training",
  "timestamp": "$TIMESTAMP",
  "progress": "$PROGRESS",
  "loss": "$LOSS",
  "learning_rate": "$LR",
  "epoch": "$EPOCH",
  "total_steps": 1239,
  "gpu": {
    "index": 3,
    "utilization_pct": $GPU_UTIL,
    "memory_used_mib": $GPU_MEM,
    "temperature_c": $GPU_TEMP
  },
  "config": {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "dataset": "BIRD (6601 samples)",
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,
    "effective_batch_size": 16,
    "lr": "2e-4",
    "lora_r": 32,
    "lora_alpha": 32,
    "modifications": [
      "Rich inline prompt (matching LoRA_SGD)",
      "Schema backticks for special chars",
      "1-message format (user only + raw SQL + EOS)",
      "repetition_penalty=1.2",
      "Semicolon-based SQL extraction"
    ]
  }
}
EOF
    fi

    # Check if training is done
    if grep -q "Training complete!" "$LOGFILE" 2>/dev/null; then
        echo "Training complete! Final status saved to $JSONFILE"
        # Update JSON with completion
        FINAL_LOSS=$(grep -oP "'train_loss': [\d.]+" "$LOGFILE" | tail -1 | grep -oP '[\d.]+$')
        FINAL_STEPS=$(grep -oP "'train_steps': \d+" "$LOGFILE" | tail -1 | grep -oP '\d+$')
        cat > "$JSONFILE" << EOF
{
  "task": "BIRD v3 training",
  "status": "completed",
  "timestamp": "$(date -Iseconds)",
  "final_loss": "$FINAL_LOSS",
  "final_steps": "$FINAL_STEPS",
  "total_steps": 1239,
  "config": {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "dataset": "BIRD (6601 samples)",
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,
    "effective_batch_size": 16,
    "lr": "2e-4",
    "lora_r": 32,
    "lora_alpha": 32,
    "modifications": [
      "Rich inline prompt (matching LoRA_SGD)",
      "Schema backticks for special chars",
      "1-message format (user only + raw SQL + EOS)",
      "repetition_penalty=1.2",
      "Semicolon-based SQL extraction"
    ]
  }
}
EOF
        break
    fi

    sleep 60
done
