#!/usr/bin/env python3
"""
Evaluate BIRD LoRA adapter on the Spider test set (cross-domain generalization).

Usage:
    python scripts/eval_bird_lora_on_spider.py --gpu 2
    python scripts/eval_bird_lora_on_spider.py --gpu 2 --output results/bird_lora_on_spider_eval.json

    # Re-score saved predictions without inference:
    python scripts/eval_bird_lora_on_spider.py --score-only --predictions-dir results/predictions
"""

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

# Import unsloth BEFORE other ML libraries for optimizations
import unsloth  # noqa: F401

import json
import time

from tqdm import tqdm

from src.config import load_config
from src.datasets import DatasetLoader
from src.framework import AdaptiveSLMFramework


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BIRD LoRA adapter on Spider test set (cross-domain)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="results/bird_lora_on_spider_eval.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--predictions-dir", type=str, default="results/predictions",
                        help="Directory to save/load intermediate predictions")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Direct path to BIRD adapter (default: latest from registry)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip inference, re-score from saved predictions")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU index to use (overrides .env CUDA_VISIBLE_DEVICES)")
    return parser.parse_args()


def generate_predictions(model, dataset, domain, evaluator):
    """Run inference on dataset, returning predictions, references, and db_ids."""
    system_prompt = evaluator.SYSTEM_PROMPTS.get(domain, evaluator.SYSTEM_PROMPTS["general"])

    predictions = []
    references = []
    db_ids = []
    questions = []

    for example in tqdm(dataset, desc=f"Evaluating {domain}"):
        if domain in ("text_to_sql", "text_to_sql_bird") and "prompt" in example:
            query = example["prompt"]
        else:
            query = example.get("question", "")

        question = example.get("question", "")
        reference = example.get("query", "")

        db_id = example.get("db_id", None)
        db_ids.append(db_id)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        prediction = model.generate_chat(messages, temperature=0.0, do_sample=False)

        predictions.append(prediction)
        references.append(reference)
        questions.append(question)

    return predictions, references, db_ids, questions


def save_predictions(predictions, references, db_ids, questions, pred_dir, name):
    """Save predictions to disk before scoring."""
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

    path = pred_dir / f"{name}_predictions.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Predictions saved to: {path} ({len(data)} samples)")
    return data


def load_predictions(pred_dir, name):
    """Load saved predictions from disk."""
    path = Path(pred_dir) / f"{name}_predictions.json"
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


def main():
    args = parse_args()

    print("=" * 60)
    print("  BIRD LoRA on Spider Test (Cross-Domain Generalization)")
    print("=" * 60)

    config = load_config(args.config)
    framework = AdaptiveSLMFramework(config)
    framework.initialize(load_student=True, load_teacher=False)

    # Find BIRD adapter
    adapter_path = args.adapter_path
    if adapter_path is None:
        adapters = framework.adapter_manager.list_adapters("text_to_sql_bird")
        if not adapters:
            print("ERROR: No BIRD adapters found in registry!")
            return
        latest = max(adapters, key=lambda a: a.version)
        adapter_path = latest.path
        print(f"  Using BIRD adapter: {latest.name} (v{latest.version})")
    print(f"  Adapter path: {adapter_path}")

    pred_name = "bird_lora_on_spider"

    if args.score_only:
        print(f"\n  MODE: Score-only (loading from {args.predictions_dir})")
        predictions, references, db_ids, questions = load_predictions(
            args.predictions_dir, pred_name
        )
    else:
        # Load BIRD adapter
        print("\n  Loading BIRD LoRA adapter...")
        framework.student.load_adapter(adapter_path)

        # Load Spider test data
        print("  Loading Spider test data...")
        loader = DatasetLoader()
        spider_data = loader.load_spider()
        spider_test = spider_data.test
        print(f"  Spider test set: {len(spider_test)} samples")

        # Inference — use text_to_sql domain (Spider prompts/schema format)
        print(f"\n{'=' * 60}")
        print(f"  Evaluating BIRD LoRA on Spider test ({len(spider_test)} samples)")
        print(f"{'=' * 60}")

        start_inf = time.time()
        predictions, references, db_ids, questions = generate_predictions(
            framework.student, spider_test, "text_to_sql", framework.evaluator
        )
        inf_elapsed = time.time() - start_inf
        print(f"\n  Inference time: {inf_elapsed:.0f}s ({inf_elapsed / len(predictions):.1f}s/sample)")

        # Save predictions BEFORE scoring
        save_predictions(predictions, references, db_ids, questions, args.predictions_dir, pred_name)

    # Score using Spider executor
    print(f"\n  Scoring {len(predictions)} predictions...")
    score_start = time.time()
    evaluator = framework.evaluator
    if evaluator._sql_executor and db_ids:
        result = evaluator.evaluate_sql_execution(predictions, references, db_ids)
    else:
        result = evaluator.evaluate(predictions, references, "text_to_sql")
    score_elapsed = time.time() - score_start

    print(f"\n  BIRD LoRA on Spider Score: {result.score:.2%} ({result.correct}/{result.num_samples})")
    print(f"  Metric: {result.metric_name}")
    print(f"  Scoring time: {score_elapsed:.0f}s")

    # Build per-sample records
    samples = []
    for i, (pred, ref, question) in enumerate(zip(predictions, references, questions)):
        samples.append({
            "index": i,
            "db_id": db_ids[i],
            "question": question,
            "prediction": pred,
            "reference": ref,
        })

    results = {
        "bird_lora_on_spider": {
            "score": result.score,
            "correct": result.correct,
            "total": result.num_samples,
            "metric": result.metric_name,
            "adapter": str(adapter_path),
            "scoring_seconds": round(score_elapsed, 1),
            "details": result.details,
            "samples": samples,
        }
    }

    # Summary with comparisons
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY (Cross-Domain Generalization)")
    print(f"{'=' * 60}")
    print(f"  BIRD LoRA on Spider:   {result.score:.2%}  ({result.correct}/{result.num_samples})")
    print(f"  Spider Base (ref):     72.34%  (748/1034)  [from previous run]")
    print(f"  Spider LoRA (ref):     (pending)")
    diff_vs_base = result.score - 0.7234
    if diff_vs_base > 0:
        print(f"  vs Base: +{diff_vs_base:.2%}")
    else:
        print(f"  vs Base: {diff_vs_base:.2%}")
    print()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    print(f"  (includes {len(samples)} per-sample predictions for debugging)")
    print("Done!")


if __name__ == "__main__":
    main()
