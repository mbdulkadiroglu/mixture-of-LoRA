#!/bin/bash
# Wait for BIRD v3 training to finish, then auto-launch evaluation on the same GPU.
#
# Usage:
#   tmux new-session -d -s eval-v3-wait 'bash scripts/eval_after_training.sh'

LOGFILE="results/bird_v3_training.log"
GPU=3
EVAL_OUTPUT="results/bird_v3_lora_eval.json"
PRED_DIR="results/predictions_bird_v3"
RESULT_JSON="results/bird_v3_eval_progress.json"

echo "=== Waiting for BIRD v3 training to complete ==="
echo "  Watching: $LOGFILE"
echo "  GPU for eval: $GPU"
echo ""

# Poll until training completes or the tmux session dies
while true; do
    # Check if training is done (trainer logs "Training complete!" or the process ended)
    if grep -q "Training complete!" "$LOGFILE" 2>/dev/null; then
        echo "[$(date)] Training complete detected in log!"
        break
    fi

    # Also check if the training tmux session is gone (crashed or finished)
    if ! tmux has-session -t bird-v3-train 2>/dev/null; then
        echo "[$(date)] Training tmux session ended."
        break
    fi

    sleep 30
done

# Extract final training info
FINAL_LOSS=$(grep -oP "'train_loss': [\d.]+" "$LOGFILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$')
echo ""
echo "=== Training finished ==="
echo "  Final loss: ${FINAL_LOSS:-unknown}"
echo ""

# Save status
cat > "$RESULT_JSON" << EOF
{
  "status": "training_complete_starting_eval",
  "timestamp": "$(date -Iseconds)",
  "final_train_loss": "${FINAL_LOSS:-unknown}",
  "gpu": $GPU
}
EOF

# Wait a moment for GPU memory to fully free
echo "Waiting 30s for GPU memory to clear..."
sleep 30

# Find the latest adapter path from the registry
ADAPTER_PATH=$(python3 -c "
import json
with open('data/lora_adapters/registry.json') as f:
    reg = json.load(f)
bird_adapters = [a for a in reg.get('adapters', []) if a['domain'] == 'text_to_sql_bird']
latest = max(bird_adapters, key=lambda a: a['version'])
print(latest['path'])
" 2>/dev/null)

if [ -z "$ADAPTER_PATH" ]; then
    echo "ERROR: Could not find BIRD adapter in registry!"
    exit 1
fi

echo "=== Starting BIRD v3 LoRA evaluation ==="
echo "  Adapter: $ADAPTER_PATH"
echo "  GPU: $GPU"
echo "  Output: $EVAL_OUTPUT"
echo ""

# Update status
cat > "$RESULT_JSON" << EOF
{
  "status": "evaluating",
  "timestamp": "$(date -Iseconds)",
  "adapter": "$ADAPTER_PATH",
  "gpu": $GPU
}
EOF

# Run evaluation
source ~/miniconda3/bin/activate mixture-lora
CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_lora_bird.py \
    --gpu $GPU \
    --adapter-path "$ADAPTER_PATH" \
    --output "$EVAL_OUTPUT" \
    --predictions-dir "$PRED_DIR" \
    2>&1 | tee results/bird_v3_eval.log

# Update final status
cat > "$RESULT_JSON" << EOF
{
  "status": "eval_complete",
  "timestamp": "$(date -Iseconds)",
  "adapter": "$ADAPTER_PATH",
  "results": "$EVAL_OUTPUT"
}
EOF

echo ""
echo "=== Evaluation complete! ==="
echo "  Results: $EVAL_OUTPUT"
echo "  Predictions: $PRED_DIR"
