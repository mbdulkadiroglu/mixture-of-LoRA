#!/usr/bin/env python3
"""
Compare base model vs LoRA adapter on text_to_sql domain.

Usage:
    python scripts/compare_base_vs_lora.py --samples 100
    python scripts/compare_base_vs_lora.py --adapter-version 3 --samples 100
    python scripts/compare_base_vs_lora.py --mode base --samples 100
    python scripts/compare_base_vs_lora.py --mode lora --samples 100
"""

import sys
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE PyTorch initializes
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Import unsloth BEFORE other ML libraries for optimizations
import unsloth  # noqa: F401

import argparse
import json

from src.config import load_config
from src.datasets import DatasetLoader
from src.framework import AdaptiveSLMFramework


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base model vs LoRA adapter")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--adapter-version", type=int, default=None,
                        help="Specific adapter version (default: latest)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Direct path to adapter (overrides version)")
    parser.add_argument("--mode", type=str, choices=["base", "lora", "both"], default="both",
                        help="What to evaluate: base, lora, or both")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Base Model vs LoRA Adapter Comparison")
    print("=" * 70)

    # Load config and initialize framework
    config = load_config(args.config)
    framework = AdaptiveSLMFramework(config)
    framework.initialize(load_student=True, load_teacher=False)

    # Load test data
    print("\nLoading Spider test data...")
    loader = DatasetLoader()
    dataset = loader.load_spider(max_samples=args.samples * 2)
    test_data = dataset.test

    # Determine adapter path
    adapter_path = None
    adapter_name = None
    if args.mode in ["lora", "both"]:
        if args.adapter_path:
            adapter_path = args.adapter_path
            adapter_name = Path(adapter_path).name
        elif args.adapter_version:
            adapters = framework.adapter_manager.list_adapters("text_to_sql")
            adapter = next((a for a in adapters if a.version == args.adapter_version), None)
            if adapter is None:
                print(f"Error: Adapter version {args.adapter_version} not found")
                return
            adapter_path = adapter.path
            adapter_name = adapter.name
        else:
            # Use latest version
            adapters = framework.adapter_manager.list_adapters("text_to_sql")
            if adapters:
                latest = max(adapters, key=lambda a: a.version)
                adapter_path = latest.path
                adapter_name = latest.name
                print(f"Using latest adapter: {latest.name} (v{latest.version})")
            else:
                print("No adapters found!")
                return

        print(f"Adapter path: {adapter_path}")

    print(f"Evaluating on {args.samples} samples...")

    results = {}

    # Evaluate based on mode
    if args.mode == "lora":
        # Only evaluate LoRA
        print("\n" + "-" * 50)
        print("Evaluating WITH LoRA ADAPTER...")
        print("-" * 50)

        framework.student.load_adapter(adapter_path)
        lora_result = framework.evaluator.evaluate_dataset(
            framework.student,
            test_data,
            "text_to_sql",
            max_samples=args.samples,
        )
        print(f"\nLoRA Adapter Score: {lora_result.score:.2%} ({lora_result.correct}/{lora_result.num_samples})")
        results["lora"] = {
            "score": lora_result.score,
            "correct": lora_result.correct,
            "total": lora_result.num_samples,
            "adapter": adapter_name,
        }

    elif args.mode == "base":
        # Only evaluate base
        print("\n" + "-" * 50)
        print("Evaluating BASE MODEL (no adapter)...")
        print("-" * 50)

        # Don't load any adapter
        base_result = framework.evaluator.evaluate_dataset(
            framework.student,
            test_data,
            "text_to_sql",
            max_samples=args.samples,
        )
        print(f"\nBase Model Score: {base_result.score:.2%} ({base_result.correct}/{base_result.num_samples})")
        results["base"] = {
            "score": base_result.score,
            "correct": base_result.correct,
            "total": base_result.num_samples,
        }

    else:
        # Both - evaluate LoRA first (since model loads without adapter by default,
        # loading adapter is safer than unloading)
        print("\n" + "-" * 50)
        print("Evaluating WITH LoRA ADAPTER...")
        print("-" * 50)

        framework.student.load_adapter(adapter_path)
        lora_result = framework.evaluator.evaluate_dataset(
            framework.student,
            test_data,
            "text_to_sql",
            max_samples=args.samples,
        )
        print(f"\nLoRA Adapter Score: {lora_result.score:.2%} ({lora_result.correct}/{lora_result.num_samples})")
        results["lora"] = {
            "score": lora_result.score,
            "correct": lora_result.correct,
            "total": lora_result.num_samples,
            "adapter": adapter_name,
        }

        # Try to evaluate base model
        print("\n" + "-" * 50)
        print("Evaluating BASE MODEL (no adapter)...")
        print("-" * 50)

        try:
            framework.student.unload_adapter()
            base_result = framework.evaluator.evaluate_dataset(
                framework.student,
                test_data,
                "text_to_sql",
                max_samples=args.samples,
            )
            print(f"\nBase Model Score: {base_result.score:.2%} ({base_result.correct}/{base_result.num_samples})")
            results["base"] = {
                "score": base_result.score,
                "correct": base_result.correct,
                "total": base_result.num_samples,
            }
        except Exception as e:
            print(f"\nError evaluating base model: {e}")
            print("Run separately with --mode base to get base model results")

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    if "base" in results:
        print(f"\n  Base Model:    {results['base']['score']:.2%} ({results['base']['correct']}/{results['base']['total']})")
    if "lora" in results:
        print(f"  LoRA Adapter:  {results['lora']['score']:.2%} ({results['lora']['correct']}/{results['lora']['total']})")

    if "base" in results and "lora" in results:
        improvement = results["lora"]["score"] - results["base"]["score"]
        if improvement > 0:
            print(f"\n  Improvement:   +{improvement:.2%}")
        elif improvement < 0:
            print(f"\n  Regression:    {improvement:.2%}")
        else:
            print(f"\n  No change")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
