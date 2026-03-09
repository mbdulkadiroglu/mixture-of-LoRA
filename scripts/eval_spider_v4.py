#!/usr/bin/env python3
"""
Evaluate Spider LoRA v4 adapter via framework.evaluate_domain() which also
updates the registry score (testing the adapter selection bug fix).

Usage:
    python scripts/eval_spider_v4.py --gpu 3
"""

import json
import os
import sys
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE PyTorch initializes
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Override GPU AFTER .env load
import argparse
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

# Import unsloth BEFORE other ML libraries
import unsloth  # noqa: F401

import datetime

from src.config import load_config
from src.framework import AdaptiveSLMFramework


def main():
    parser = argparse.ArgumentParser(description="Evaluate Spider LoRA v4 and update registry")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate (default: full test set)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spider LoRA v4 Evaluation (with registry update)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    framework = AdaptiveSLMFramework(config)
    framework.initialize(load_student=True, load_teacher=False)

    # Find v4 adapter explicitly
    adapters = framework.adapter_manager.list_adapters("text_to_sql")
    adapters.sort(key=lambda a: a.version)
    print(f"\nRegistered Spider adapters: {[(a.name, f'score={a.eval_score}') for a in adapters]}")

    v4 = next((a for a in adapters if a.version == 4), None)
    if v4 is None:
        print("ERROR: text_to_sql v4 not found in registry!")
        return
    print(f"Evaluating: {v4.name} at {v4.path}")

    # Results file — write partial results immediately
    results_path = Path("data/eval_results/spider_v4_eval.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    partial = {
        "adapter_name": v4.name,
        "adapter_path": v4.path,
        "adapter_version": v4.version,
        "domain": "text_to_sql",
        "status": "running",
        "gpu": f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
        "started_at": datetime.datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(partial, f, indent=2)
    print(f"Partial results: {results_path}")

    # Run evaluation — this updates the registry via our bug fix
    eval_result = framework.evaluate_domain(
        "text_to_sql",
        max_samples=args.max_samples,
        adapter_path=v4.path,
    )

    print(f"\n{'=' * 60}")
    print(f"  Spider v4 Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Adapter: {v4.name}")
    print(f"  Score: {eval_result.score:.2%}")
    print(f"  Correct: {eval_result.correct}/{eval_result.num_samples}")
    print(f"  Base model: 72.34% (748/1034)")
    improvement = eval_result.score - 0.7234
    if improvement > 0:
        print(f"  Improvement: +{improvement:.2%}")
    elif improvement < 0:
        print(f"  Regression: {improvement:.2%}")

    # Save final results
    results = {
        "adapter_name": v4.name,
        "adapter_path": v4.path,
        "adapter_version": v4.version,
        "domain": "text_to_sql",
        "score": eval_result.score,
        "correct": eval_result.correct,
        "total": eval_result.num_samples,
        "gpu": f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
        "status": "completed",
        "started_at": partial["started_at"],
        "finished_at": datetime.datetime.now().isoformat(),
        "details": {
            "metric": eval_result.metric_name if hasattr(eval_result, "metric_name") else "execution_accuracy",
            "base_model_score": 0.7234,
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Verify registry was updated
    registry_path = Path("data/lora_adapters/registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    for adapter in registry.get("text_to_sql", []):
        if adapter["version"] == 4:
            print(f"Registry check: {adapter['name']} eval_score = {adapter['eval_score']}")
            break

    print("Done!")


if __name__ == "__main__":
    main()
