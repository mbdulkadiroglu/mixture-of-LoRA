#!/bin/bash
# Watch for training completion and auto-start BIRD LoRA eval on GPU 3
# Usage: bash scripts/watch_and_eval.sh

TRAINING_OUTPUT="/tmp/claude-1005/-data-mehmet-projects-mixture-of-LoRA/tasks/b4ea2f1.output"
RESULTS_FILE="results/llama_bird_lora_eval.json"

echo "Watching for training completion..."
echo "  Training log: $TRAINING_OUTPUT"
echo "  Will run: python scripts/eval_lora_bird.py --gpu 3 --output $RESULTS_FILE"

while true; do
    if grep -q "Training complete!" "$TRAINING_OUTPUT" 2>/dev/null; then
        echo ""
        echo "=========================================="
        echo "  Training complete! Starting LoRA eval..."
        echo "=========================================="
        echo ""

        # Activate env and run eval
        source ~/miniconda3/bin/activate mixture-lora
        python scripts/eval_lora_bird.py --gpu 3 --output "$RESULTS_FILE"

        echo "LoRA eval finished. Results: $RESULTS_FILE"
        break
    fi

    if grep -q "Error\|Traceback\|FAILED" "$TRAINING_OUTPUT" 2>/dev/null; then
        echo "WARNING: Training may have errored. Check log."
        tail -20 "$TRAINING_OUTPUT"
        break
    fi

    sleep 60
done
