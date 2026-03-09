#!/usr/bin/env python3
"""
Evaluate GPT-5 mini (teacher) on Spider text-to-SQL using execution accuracy.

No GPU required — pure API calls + SQLite execution.
Uses the same system prompt as the student for fair comparison.

Usage:
    python scripts/eval_teacher_spider.py --samples 200
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables for OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import load_config
from src.datasets.loader import DatasetLoader
from src.evaluation.sql_executor import get_spider_executor
from src.models.teacher import TeacherModel


# Same system prompt used by student/evaluator for fair comparison
SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, output only the SQL query. Do not include any explanation, "
    "formatting, or markdown. Output only valid SQLite SQL."
)

RESULTS_DIR = Path("data/eval_results")
RESULTS_PATH = RESULTS_DIR / "teacher_spider_eval.json"
SAVE_EVERY = 20  # Save partial results every N samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher (GPT-5 mini) on Spider")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of samples to evaluate (default: 200)")
    args = parser.parse_args()

    num_samples = args.samples

    print("=" * 60)
    print("  Teacher (GPT-5 mini) Spider Evaluation")
    print(f"  Samples: {num_samples}")
    print("=" * 60)

    # Initialize config and teacher
    config = load_config("configs/config.yaml")
    teacher = TeacherModel(config.teacher)
    print(f"Teacher model: {teacher.model_name}")

    # Load Spider dataset
    loader = DatasetLoader()
    spider_data = loader.load_spider()
    test_set = spider_data.test
    print(f"Spider test set size: {len(test_set)}")

    # Limit samples
    if num_samples and num_samples < len(test_set):
        test_set = test_set.select(range(num_samples))
    actual_samples = len(test_set)
    print(f"Evaluating {actual_samples} samples")

    # Initialize SQL executor
    executor = get_spider_executor()
    if executor is None:
        print("ERROR: Spider database directory not found!")
        sys.exit(1)
    print(f"SQL executor ready: {executor.database_dir}")

    # Prepare results structure
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.datetime.now().isoformat()
    results = {
        "model": teacher.model_name,
        "dataset": "spider",
        "num_samples": actual_samples,
        "execution_accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "errors": 0,
        "api_errors": 0,
        "status": "running",
        "started_at": started_at,
        "per_sample": [],
    }

    # Save initial state
    _save_results(results)

    correct = 0
    total = 0
    api_errors = 0
    exec_errors = 0

    for i in range(actual_samples):
        sample = test_set[i]
        prompt = sample["prompt"]
        gold_sql = sample["query"]
        db_id = sample["db_id"]

        sample_result = {
            "idx": i,
            "db_id": db_id,
            "question": sample.get("question", ""),
            "gold_sql": gold_sql,
            "predicted_sql": None,
            "correct": False,
            "error": None,
        }

        # Call teacher API
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            predicted_sql = teacher.generate(messages)
            sample_result["predicted_sql"] = predicted_sql
        except Exception as e:
            api_errors += 1
            sample_result["error"] = f"API error: {e}"
            results["per_sample"].append(sample_result)
            total += 1
            _print_progress(i, actual_samples, correct, total, api_errors, "API_ERR")
            continue

        # Evaluate via SQL execution
        try:
            is_correct, details = executor.evaluate_single(predicted_sql, gold_sql, db_id)
            sample_result["correct"] = is_correct
            if details.get("pred_error"):
                sample_result["error"] = details["pred_error"]
                exec_errors += 1
            if is_correct:
                correct += 1
        except Exception as e:
            sample_result["error"] = f"Execution error: {e}"
            exec_errors += 1

        total += 1
        results["per_sample"].append(sample_result)

        status = "OK" if sample_result["correct"] else "WRONG"
        if sample_result["error"] and not sample_result["correct"]:
            status = "ERR"
        _print_progress(i, actual_samples, correct, total, api_errors, status)

        # Save partial results periodically
        if (i + 1) % SAVE_EVERY == 0:
            results["correct"] = correct
            results["total"] = total
            results["errors"] = exec_errors
            results["api_errors"] = api_errors
            results["execution_accuracy"] = correct / total if total > 0 else 0.0
            _save_results(results)

    # Final results
    results["correct"] = correct
    results["total"] = total
    results["errors"] = exec_errors
    results["api_errors"] = api_errors
    results["execution_accuracy"] = correct / total if total > 0 else 0.0
    results["status"] = "completed"
    results["finished_at"] = datetime.datetime.now().isoformat()
    _save_results(results)

    print(f"\n{'=' * 60}")
    print(f"  Teacher Spider Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Model: {teacher.model_name}")
    print(f"  Execution accuracy: {results['execution_accuracy']:.2%}")
    print(f"  Correct: {correct}/{total}")
    print(f"  SQL execution errors: {exec_errors}")
    print(f"  API errors: {api_errors}")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"{'=' * 60}")

    # Decision gate
    acc = results["execution_accuracy"]
    if acc >= 0.90:
        print("\n  DECISION: Teacher >90% — safe to train on teacher responses with minimal filtering")
    elif acc >= 0.60:
        print(f"\n  DECISION: Teacher {acc:.0%} — must verify via SQL execution before using as training data")
    else:
        print(f"\n  DECISION: Teacher {acc:.0%} — teacher prompts need improvement before proceeding")


def _print_progress(i: int, total: int, correct: int, evaluated: int, api_errors: int, status: str):
    acc = correct / evaluated if evaluated > 0 else 0.0
    print(f"  [{i+1:4d}/{total}] {status:8s} | acc: {acc:.2%} ({correct}/{evaluated}) | api_err: {api_errors}")


def _save_results(results: dict):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
