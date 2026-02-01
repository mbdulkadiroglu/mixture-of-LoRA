#!/usr/bin/env python3
"""
Diagnose evaluation differences between base model and LoRA adapter.

Usage:
    python scripts/diagnose_eval.py --mode lora --samples 10
    python scripts/diagnose_eval.py --mode base --samples 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import unsloth  # noqa: F401

from src.config import load_config
from src.datasets import DatasetLoader
from src.framework import AdaptiveSLMFramework
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "lora"], required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"  Diagnostic: {args.mode.upper()} model outputs")
    print("=" * 70)

    # Load config and initialize
    config = load_config("configs/config.yaml")
    framework = AdaptiveSLMFramework(config)
    framework.initialize(load_student=True, load_teacher=False)

    # Load test data
    loader = DatasetLoader()
    dataset = loader.load_spider(max_samples=args.samples * 2)
    test_data = dataset.test

    # Load adapter if needed
    if args.mode == "lora":
        adapters = framework.adapter_manager.list_adapters("text_to_sql")
        latest = max(adapters, key=lambda a: a.version)
        print(f"\nLoading adapter: {latest.name}")
        framework.student.load_adapter(latest.path)
    else:
        print("\nUsing base model (no adapter)")

    # System prompt (same as evaluation)
    system_prompt = """You are an expert SQL assistant. Convert natural language queries to SQL.
Think step by step, then provide the final SQL query in a code block."""

    evaluator = Evaluator()

    results = []

    print(f"\nGenerating outputs for {args.samples} samples...")
    print("=" * 70)

    for i in range(min(args.samples, len(test_data))):
        example = test_data[i]
        query = example["prompt"]
        reference = example["query"]
        db_id = example.get("db_id", "unknown")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        output = framework.student.generate_chat(messages, temperature=0.1)
        extracted_sql = evaluator._extract_sql(output)

        # Check execution
        exec_correct = False
        if framework.evaluator._sql_executor:
            exec_correct, _ = framework.evaluator._sql_executor.evaluate_single(
                extracted_sql, reference, db_id
            )

        result = {
            "index": i,
            "db_id": db_id,
            "question": example.get("question", "")[:100],
            "reference_sql": reference,
            "output_length": len(output),
            "extracted_sql": extracted_sql,
            "execution_correct": exec_correct,
            "full_output": output,
        }
        results.append(result)

        status = "✓" if exec_correct else "✗"
        print(f"  [{i+1:2d}] {status} db={db_id[:15]:15s} out_len={len(output):4d} sql_len={len(extracted_sql):4d}")

    # Summary
    correct = sum(1 for r in results if r["execution_correct"])
    print(f"\n{'=' * 70}")
    print(f"  Summary: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")
    print(f"  Avg output length: {sum(r['output_length'] for r in results)/len(results):.0f} chars")
    print(f"  Avg SQL length: {sum(len(r['extracted_sql']) for r in results)/len(results):.0f} chars")
    print(f"{'=' * 70}")

    # Show some examples
    print("\n--- Sample Outputs ---")
    for r in results[:3]:
        print(f"\n[{r['index']}] Question: {r['question']}...")
        print(f"Reference: {r['reference_sql']}")
        print(f"Extracted: {r['extracted_sql']}")
        print(f"Correct: {r['execution_correct']}")
        print(f"Full output preview: {r['full_output'][:300]}...")

    # Save results
    output_file = args.output or f"logs/diagnose_{args.mode}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
