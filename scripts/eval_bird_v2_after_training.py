#!/usr/bin/env python3
"""
Wait for BIRD v2 training to finish, then evaluate the newly trained adapter.
Saves results to a JSON file in real time.
"""

import json
import os
import sys
import time
from pathlib import Path

# Wait for training process to finish
TRAINING_PID = 310223

print(f"Waiting for training process (PID {TRAINING_PID}) to finish...")
while True:
    try:
        os.kill(TRAINING_PID, 0)  # Check if process exists
        time.sleep(30)
    except OSError:
        print("Training process finished!")
        break

# --- Now run evaluation ---

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE PyTorch initializes
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Use GPU 2 (same as training)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Import unsloth BEFORE other ML libraries
import unsloth  # noqa: F401

from src.config import load_config
from src.framework import AdaptiveSLMFramework

print("=" * 60)
print("Starting BIRD v2 evaluation")
print("=" * 60)

config = load_config("configs/config.yaml")
framework = AdaptiveSLMFramework(config)
framework.initialize(load_student=True, load_teacher=False)

# Find the bird v2 adapter path from the registry
bird_adapters = framework.adapter_manager.list_adapters("text_to_sql_bird")
bird_adapters.sort(key=lambda a: a.version)
print(f"\nRegistered BIRD adapters: {[a.name for a in bird_adapters]}")

# Get the latest (v2) adapter
latest = bird_adapters[-1]
print(f"Evaluating: {latest.name} at {latest.path}")

# Run evaluation with explicit adapter_path (uses our bug fix)
eval_result = framework.evaluate_domain(
    "text_to_sql_bird",
    adapter_path=latest.path,
)

print(f"\n{'=' * 60}")
print(f"BIRD v2 Evaluation Results")
print(f"{'=' * 60}")
print(f"  Adapter: {latest.name}")
print(f"  Score: {eval_result.score:.2%}")
print(f"  Correct: {eval_result.correct}/{eval_result.num_samples}")

# Save results to JSON
results_path = Path("data/eval_results/bird_v2_eval.json")
results_path.parent.mkdir(parents=True, exist_ok=True)

results = {
    "adapter_name": latest.name,
    "adapter_path": latest.path,
    "adapter_version": latest.version,
    "domain": "text_to_sql_bird",
    "score": eval_result.score,
    "correct": eval_result.correct,
    "total": eval_result.num_samples,
    "gpu": "NVIDIA RTX 6000 Ada (GPU 2)",
    "timestamp": __import__("datetime").datetime.now().isoformat(),
    "details": {
        "metric": eval_result.metric if hasattr(eval_result, "metric") else "execution_accuracy",
    },
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_path}")
print("Done!")
