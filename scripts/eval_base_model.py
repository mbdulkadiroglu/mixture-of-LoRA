#!/usr/bin/env python3
"""
Evaluate base Qwen 2.5 14B (no LoRA adapter) on Spider and BIRD test sets.

Saves all predictions alongside scores for debugging.
Predictions are saved to disk before scoring, so scoring can be re-run
without re-doing inference via --score-only.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_base_model.py
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_base_model.py --output results/base_model_eval.json

    # Re-score saved predictions without inference:
    python scripts/eval_base_model.py --score-only --predictions-dir results/predictions
"""

import sys
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE PyTorch initializes
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import argparse
import json
import time

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base model on Spider + BIRD")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="results/base_model_eval.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--predictions-dir", type=str, default="results/predictions",
                        help="Directory to save/load intermediate predictions")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip inference, re-score from saved predictions")
    return parser.parse_args()


def generate_predictions(model, dataset, domain, evaluator):
    """Run inference on dataset, returning predictions, references, and db_ids."""
    system_prompt = evaluator.SYSTEM_PROMPTS.get(domain, evaluator.SYSTEM_PROMPTS["general"])

    predictions = []
    references = []
    db_ids = []
    questions = []

    for example in tqdm(dataset, desc=f"Evaluating {domain}"):
        # Build query (prefer prompt field with schema context for SQL)
        if domain in ("text_to_sql", "text_to_sql_bird") and "prompt" in example:
            query = example["prompt"]
        else:
            query = example.get("question", "")

        # Get the natural language question (without schema) for readability
        question = example.get("question", "")

        # Get reference
        if domain == "text_to_sql":
            reference = example.get("query", "")
        elif domain == "text_to_sql_bird":
            reference = example.get("SQL", example.get("query", ""))
        else:
            reference = example.get("answer", "")

        db_id = example.get("db_id", None)
        db_ids.append(db_id)

        # BIRD uses user-only messages (rich prompt has all instructions inline)
        if domain == "text_to_sql_bird":
            messages = [{"role": "user", "content": query}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        prediction = model.generate_chat(messages, temperature=0.0, do_sample=False)

        predictions.append(prediction)
        references.append(reference)
        questions.append(question)

    return predictions, references, db_ids, questions


def save_predictions(predictions, references, db_ids, questions, pred_dir, dataset_name):
    """Save predictions to disk so scoring can be re-run without inference."""
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for i, (pred, ref, db_id, q) in enumerate(zip(predictions, references, db_ids, questions)):
        data.append({
            "index": i,
            "db_id": db_id,
            "question": q,
            "prediction": pred,
            "reference": ref,
        })

    path = pred_dir / f"{dataset_name}_predictions.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Predictions saved to: {path} ({len(data)} samples)")
    return data


def load_predictions(pred_dir, dataset_name):
    """Load saved predictions from disk."""
    path = Path(pred_dir) / f"{dataset_name}_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    predictions = [d["prediction"] for d in data]
    references = [d["reference"] for d in data]
    db_ids = [d["db_id"] for d in data]
    questions = [d["question"] for d in data]

    print(f"  Loaded {len(data)} predictions from: {path}")
    return predictions, references, db_ids, questions


def score_predictions(evaluator, predictions, references, db_ids, domain):
    """Score predictions using evaluator's execution-based methods."""
    if domain == "text_to_sql" and evaluator._sql_executor and db_ids:
        return evaluator.evaluate_sql_execution(predictions, references, db_ids)
    elif domain == "text_to_sql_bird" and evaluator._bird_executor and db_ids:
        return evaluator.evaluate_bird_sql_execution(predictions, references, db_ids)
    else:
        return evaluator.evaluate(predictions, references, domain)


def evaluate_on_dataset(framework, dataset, domain, dataset_name, pred_dir, score_only=False):
    """Run inference + evaluation, return result dict with all predictions."""
    print(f"\n{'=' * 60}")
    print(f"  {'Scoring' if score_only else 'Evaluating BASE MODEL on'} {dataset_name}")
    print(f"{'=' * 60}")

    start = time.time()

    if score_only:
        predictions, references, db_ids, questions = load_predictions(pred_dir, dataset_name)
    else:
        # Generate all predictions
        predictions, references, db_ids, questions = generate_predictions(
            framework.student, dataset, domain, framework.evaluator
        )
        # Save to disk BEFORE scoring (so we don't lose them if scoring fails)
        save_predictions(predictions, references, db_ids, questions, pred_dir, dataset_name)

    # Score
    print(f"  Scoring {len(predictions)} predictions...")
    score_start = time.time()
    result = score_predictions(framework.evaluator, predictions, references, db_ids, domain)
    score_elapsed = time.time() - score_start
    elapsed = time.time() - start

    print(f"\n  {dataset_name} Score: {result.score:.2%} ({result.correct}/{result.num_samples})")
    print(f"  Metric: {result.metric_name}")
    print(f"  Scoring time: {score_elapsed:.0f}s")
    if not score_only:
        print(f"  Total time: {elapsed:.0f}s ({elapsed / result.num_samples:.1f}s/sample)")

    # Build per-sample records for debugging
    samples = []
    for i, (pred, ref, question) in enumerate(zip(predictions, references, questions)):
        samples.append({
            "index": i,
            "db_id": db_ids[i],
            "question": question,
            "prediction": pred,
            "reference": ref,
        })

    return {
        "score": result.score,
        "correct": result.correct,
        "total": result.num_samples,
        "metric": result.metric_name,
        "elapsed_seconds": round(elapsed, 1),
        "scoring_seconds": round(score_elapsed, 1),
        "details": result.details,
        "samples": samples,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("  Base Model Evaluation — Spider + BIRD")
    print("=" * 60)

    if args.score_only:
        print(f"\n  MODE: Score-only (loading predictions from {args.predictions_dir})")
        # Import unsloth only when needed (score-only still needs framework for evaluator)
        import unsloth  # noqa: F401

    else:
        # Import unsloth BEFORE other ML libraries for optimizations
        import unsloth  # noqa: F401

    from src.config import load_config
    from src.datasets import DatasetLoader
    from src.framework import AdaptiveSLMFramework

    # Load config and initialize framework
    config = load_config(args.config)
    framework = AdaptiveSLMFramework(config)

    if args.score_only:
        # Only need evaluator, not the full model
        framework.initialize(load_student=True, load_teacher=False)
    else:
        framework.initialize(load_student=True, load_teacher=False)

    results = {}

    if args.score_only:
        # Score-only mode: load predictions from disk
        results["spider"] = evaluate_on_dataset(
            framework, None, "text_to_sql", "spider", args.predictions_dir, score_only=True
        )
        results["bird"] = evaluate_on_dataset(
            framework, None, "text_to_sql_bird", "bird", args.predictions_dir, score_only=True
        )
    else:
        loader = DatasetLoader()

        # 1. Spider evaluation
        print("\nLoading Spider test data...")
        spider_data = loader.load_spider()
        spider_test = spider_data.test
        print(f"Spider test set: {len(spider_test)} samples")
        results["spider"] = evaluate_on_dataset(
            framework, spider_test, "text_to_sql", "spider", args.predictions_dir
        )

        # 2. BIRD evaluation
        print("\nLoading BIRD test data...")
        bird_data = loader.load_bird()
        bird_test = bird_data.test
        print(f"BIRD test set: {len(bird_test)} samples")
        results["bird"] = evaluate_on_dataset(
            framework, bird_test, "text_to_sql_bird", "bird", args.predictions_dir
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY (Base Model, no LoRA)")
    print(f"{'=' * 60}")
    for name, r in results.items():
        print(f"  {name.upper():>8}: {r['score']:.2%}  ({r['correct']}/{r['total']})  [{r['metric']}]")
    print()

    # Save results (includes all predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    print(f"  (includes {sum(len(r['samples']) for r in results.values())} per-sample predictions for debugging)")
    print("Done!")


if __name__ == "__main__":
    main()
