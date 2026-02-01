#!/usr/bin/env python3
"""
Evaluate the Adaptive SLM Framework across all domains.

Compares:
1. Student model (with LoRA adapters)
2. Student model (base, no adapters)
3. Teacher model (GPT-5 mini high)

Usage:
    python scripts/evaluate_all.py --samples 100
    python scripts/evaluate_all.py --domains text_to_sql math_reasoning
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.datasets import DatasetLoader
from src.framework import AdaptiveSLMFramework
from src.evaluation import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all domains")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["text_to_sql", "math_reasoning", "code_generation"],
        help="Domains to evaluate",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per domain",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--include-teacher",
        action="store_true",
        help="Also evaluate teacher model",
    )

    return parser.parse_args()


def evaluate_student_with_adapter(framework, domain, test_data, num_samples):
    """Evaluate student model with domain-specific adapter."""
    adapter_path = framework.adapter_manager.get_adapter_path(domain)

    if adapter_path is None:
        return None, "No adapter available"

    framework.student.load_adapter(adapter_path)

    result = framework.evaluator.evaluate_dataset(
        framework.student,
        test_data,
        domain,
        max_samples=num_samples,
    )

    return result, None


def evaluate_student_base(framework, domain, test_data, num_samples):
    """Evaluate student model without adapter."""
    framework.student.unload_adapter()

    result = framework.evaluator.evaluate_dataset(
        framework.student,
        test_data,
        domain,
        max_samples=num_samples,
    )

    return result, None


def evaluate_teacher(framework, domain, test_data, num_samples, query_key, ref_key):
    """Evaluate teacher model on test data."""
    from tqdm import tqdm

    predictions = []
    references = []

    test_subset = test_data.select(range(min(num_samples, len(test_data))))

    for example in tqdm(test_subset, desc=f"Teacher eval ({domain})"):
        query = example[query_key]

        # Get reference based on domain
        if domain == "text_to_sql":
            reference = example.get("query", "")
        elif domain == "math_reasoning":
            reference = example.get("answer", "")
        elif domain == "code_generation":
            reference = example.get("code", "")
        else:
            reference = example.get(ref_key, "")

        # Generate from teacher
        response = framework.teacher.generate_training_response(query, domain)

        predictions.append(response)
        references.append(reference)

    result = framework.evaluator.evaluate(predictions, references, domain)

    return result, None


def main():
    args = parse_args()

    print("=" * 70)
    print("  Adaptive SLM Framework - Comprehensive Evaluation")
    print("=" * 70)
    print()

    # Load configuration
    config = load_config(args.config)

    # Initialize framework
    print("Initializing framework...")
    framework = AdaptiveSLMFramework(config)
    framework.initialize(
        load_student=True,
        load_teacher=args.include_teacher,
    )

    # Load datasets
    print("\nLoading datasets...")
    loader = DatasetLoader()

    datasets = {}
    if "text_to_sql" in args.domains:
        datasets["text_to_sql"] = loader.load_spider(max_samples=args.samples * 2)
    if "math_reasoning" in args.domains:
        datasets["math_reasoning"] = loader.load_gsm8k(max_samples=args.samples * 2)
    if "code_generation" in args.domains:
        datasets["code_generation"] = loader.load_mbpp(max_samples=args.samples * 2)

    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "samples_per_domain": args.samples,
            "domains": args.domains,
        },
        "results": {},
    }

    # Evaluate each domain
    for domain in args.domains:
        print(f"\n{'=' * 50}")
        print(f"Evaluating domain: {domain}")
        print("=" * 50)

        if domain not in datasets:
            print(f"  Skipping - no dataset loaded")
            continue

        test_data = datasets[domain].test
        domain_results = {}

        # Get query/reference keys
        # For text_to_sql, use 'prompt' which includes schema context
        if domain == "text_to_sql":
            query_key, ref_key = "prompt", "query"
        elif domain == "math_reasoning":
            query_key, ref_key = "question", "answer"
        elif domain == "code_generation":
            query_key, ref_key = "text", "code"
        else:
            query_key, ref_key = "question", "answer"

        # 1. Student with adapter
        print("\n  Student (with LoRA adapter):")
        result, error = evaluate_student_with_adapter(
            framework, domain, test_data, args.samples
        )
        if result:
            print(f"    Score: {result.score:.2%} ({result.correct}/{result.num_samples})")
            domain_results["student_adapter"] = {
                "score": result.score,
                "correct": result.correct,
                "total": result.num_samples,
            }
        else:
            print(f"    Error: {error}")
            domain_results["student_adapter"] = {"error": error}

        # 2. Student base (no adapter)
        print("\n  Student (base, no adapter):")
        result, error = evaluate_student_base(
            framework, domain, test_data, args.samples
        )
        if result:
            print(f"    Score: {result.score:.2%} ({result.correct}/{result.num_samples})")
            domain_results["student_base"] = {
                "score": result.score,
                "correct": result.correct,
                "total": result.num_samples,
            }
        else:
            print(f"    Error: {error}")
            domain_results["student_base"] = {"error": error}

        # 3. Teacher (if requested)
        if args.include_teacher:
            print("\n  Teacher (GPT-5 mini high):")
            result, error = evaluate_teacher(
                framework, domain, test_data, args.samples, query_key, ref_key
            )
            if result:
                print(f"    Score: {result.score:.2%} ({result.correct}/{result.num_samples})")
                domain_results["teacher"] = {
                    "score": result.score,
                    "correct": result.correct,
                    "total": result.num_samples,
                }
            else:
                print(f"    Error: {error}")
                domain_results["teacher"] = {"error": error}

        results["results"][domain] = domain_results

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for domain, domain_results in results["results"].items():
        print(f"\n{domain}:")
        for model, scores in domain_results.items():
            if "error" in scores:
                print(f"  {model}: {scores['error']}")
            else:
                print(f"  {model}: {scores['score']:.2%}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
