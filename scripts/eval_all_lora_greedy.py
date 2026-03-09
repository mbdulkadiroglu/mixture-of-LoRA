#!/usr/bin/env python3
"""
Re-evaluate all LoRA adapters with greedy decoding (temperature=0, do_sample=False).
Updates registry scores via framework.evaluate_domain().

Usage:
    python scripts/eval_all_lora_greedy.py --gpu 3
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


def evaluate_adapter(framework, domain, adapter, results_dir):
    """Evaluate a single adapter and save results."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating {adapter.name} on {domain}")
    print(f"  Path: {adapter.path}")
    print(f"{'=' * 60}")

    started = datetime.datetime.now()

    # Write partial result
    results_path = results_dir / f"{adapter.name}_greedy_eval.json"
    partial = {
        "adapter_name": adapter.name,
        "domain": domain,
        "status": "running",
        "started_at": started.isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(partial, f, indent=2)

    eval_result = framework.evaluate_domain(
        domain,
        adapter_path=adapter.path,
    )

    finished = datetime.datetime.now()

    print(f"  Score: {eval_result.score:.2%} ({eval_result.correct}/{eval_result.num_samples})")

    results = {
        "adapter_name": adapter.name,
        "adapter_path": adapter.path,
        "adapter_version": adapter.version,
        "domain": domain,
        "decoding": "greedy (temperature=0.0, do_sample=False)",
        "score": eval_result.score,
        "correct": eval_result.correct,
        "total": eval_result.num_samples,
        "gpu": f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate all LoRA adapters with greedy decoding")
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  LoRA Adapter Re-evaluation (Greedy Decoding)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    framework = AdaptiveSLMFramework(config)
    framework.initialize(load_student=True, load_teacher=False)

    results_dir = Path("data/eval_results/greedy")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 1. Spider LoRA v4
    spider_adapters = framework.adapter_manager.list_adapters("text_to_sql")
    v4 = next((a for a in spider_adapters if a.version == 4), None)
    if v4:
        r = evaluate_adapter(framework, "text_to_sql", v4, results_dir)
        all_results.append(r)
    else:
        print("WARNING: text_to_sql v4 not found!")

    # 2. BIRD LoRA v1
    bird_adapters = framework.adapter_manager.list_adapters("text_to_sql_bird")
    v1 = next((a for a in bird_adapters if a.version == 1), None)
    if v1:
        r = evaluate_adapter(framework, "text_to_sql_bird", v1, results_dir)
        all_results.append(r)
    else:
        print("WARNING: text_to_sql_bird v1 not found!")

    # Summary
    print(f"\n{'=' * 60}")
    print("  GREEDY DECODING RE-EVAL SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        print(f"  {r['adapter_name']:>25}: {r['score']:.2%} ({r['correct']}/{r['total']})")

    # Save combined summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
