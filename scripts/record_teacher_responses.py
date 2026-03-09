#!/usr/bin/env python3
"""
Record GPT-5 mini responses on Spider train samples with execution accuracy.

Produces a JSON dataset of teacher responses that can be reused for training
without redundant API calls. Supports incremental recording — run with --samples 20
first, then --resume --samples 3000 to continue, then --resume --samples 7000 later.

No GPU required — pure API calls + SQLite execution.

Usage:
    # First run: record 20 samples
    python scripts/record_teacher_responses.py --samples 20

    # Continue to 3000 (skips already-recorded samples)
    python scripts/record_teacher_responses.py --samples 3000 --resume

    # Continue to full dataset
    python scripts/record_teacher_responses.py --samples 7000 --resume
"""

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables for OPENAI_API_KEY
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from cascade.prompts import build_query_messages
from src.config import load_config
from src.datasets.loader import DatasetLoader
from src.evaluation.sql_cleaning import extract_sql_from_text
from src.evaluation.sql_executor import BIRDExecutor, get_spider_executor
from src.models.teacher import TeacherModel

DEFAULT_OUTPUT = "results/cascade/teacher_responses_spider_train.json"
SAVE_EVERY = 20


def load_existing(path: Path) -> dict | None:
    """Load existing results file if it exists."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_results(results: dict, path: Path) -> None:
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def build_sample_order(train_data, seed: int, num_samples: int) -> list[int]:
    """
    Build a deterministic sample ordering from the train set.

    The ordering is stable: requesting 3000 gives the same first 20 as
    requesting 20. This ensures incremental runs are consistent.
    """
    indices = list(range(len(train_data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:num_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Record GPT-5 mini responses on Spider train set"
    )
    parser.add_argument(
        "--dataset", type=str, default="spider", choices=["spider", "bird"],
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="Total number of samples to record (default: 3000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already-recorded samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto from dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample ordering (default: 42, matches cascade experiments)",
    )
    args = parser.parse_args()

    # Dataset/executor/output switching
    if args.dataset == "bird":
        dataset_label, split_label = "bird", "dev"
    else:
        dataset_label, split_label = "spider", "train"

    if args.output is None:
        args.output = f"results/cascade/teacher_responses_{dataset_label}_{split_label}.json"
    output_path = Path(args.output)

    print("=" * 60)
    print(f"  Record Teacher (GPT-5 mini) Responses — {dataset_label.upper()} {split_label}")
    print(f"  Target samples: {args.samples}")
    print(f"  Output: {output_path}")
    print(f"  Resume: {args.resume}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Load existing results for resume
    existing = None
    completed_indices: set[int] = set()
    if args.resume:
        existing = load_existing(output_path)
        if existing:
            for entry in existing.get("per_sample", []):
                completed_indices.add(entry["idx"])
            print(f"  Resuming: {len(completed_indices)} samples already recorded")
        else:
            print("  No existing file found, starting fresh")

    # Initialize teacher
    config = load_config("configs/config.yaml")
    teacher = TeacherModel(config.teacher)
    print(f"  Teacher model: {teacher.model_name}")

    # Load dataset
    loader = DatasetLoader()
    if args.dataset == "bird":
        data = loader.load_bird()
        eval_data = data.test
        executor = BIRDExecutor("bird_data")
        print(f"  BIRD dev set size: {len(eval_data)}")
    else:
        data = loader.load_spider()
        eval_data = data.train
        executor = get_spider_executor()
        if executor is None:
            print("ERROR: Spider database directory not found!")
            sys.exit(1)
        print(f"  Spider train set size: {len(eval_data)}")

    # Clamp samples to available data
    num_samples = min(args.samples, len(eval_data))
    if num_samples < args.samples:
        print(f"  Clamped to {num_samples} (dataset has {len(eval_data)} samples)")

    # Build deterministic sample order
    sample_indices = build_sample_order(eval_data, args.seed, num_samples)
    to_process = [idx for idx in sample_indices if idx not in completed_indices]
    print(f"  Samples to process this run: {len(to_process)}")
    exec_dir = getattr(executor, "database_dir", None) or getattr(executor, "bird_data_dir", None)
    print(f"  SQL executor ready: {exec_dir}")

    # Initialize or reuse results structure
    if existing and args.resume:
        results = existing
        results["num_requested"] = num_samples
        results["status"] = "running"
    else:
        results = {
            "model": teacher.model_name,
            "dataset": dataset_label,
            "split": split_label,
            "seed": args.seed,
            "num_requested": num_samples,
            "num_completed": 0,
            "correct": 0,
            "execution_accuracy": 0.0,
            "api_errors": 0,
            "exec_errors": 0,
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(),
            "finished_at": None,
            "per_sample": [],
        }

    save_results(results, output_path)

    # Process samples
    new_correct = 0
    new_total = 0
    new_api_errors = 0
    new_exec_errors = 0

    for progress_i, dataset_idx in enumerate(to_process):
        sample = eval_data[dataset_idx]
        prompt = sample["prompt"]
        gold_sql = sample["query"]
        db_id = sample["db_id"]

        sample_result = {
            "idx": dataset_idx,
            "db_id": db_id,
            "question": sample.get("question", ""),
            "gold_sql": gold_sql,
            "prompt": prompt,
            "teacher_response_raw": None,
            "teacher_sql": None,
            "correct": False,
            "exec_details": None,
            "error": None,
            "elapsed_seconds": None,
        }
        if args.dataset == "bird":
            sample_result["evidence"] = sample.get("evidence", "")
            sample_result["difficulty"] = sample.get("difficulty", "unknown")

        # Build messages (same format as cascade experiments)
        messages = build_query_messages(args.dataset, prompt)

        # Call teacher API
        t0 = time.time()
        try:
            raw_response = teacher.generate(messages, temperature=0.0)
            sample_result["teacher_response_raw"] = raw_response
            sample_result["teacher_sql"] = extract_sql_from_text(raw_response)
        except Exception as e:
            new_api_errors += 1
            sample_result["error"] = f"API error: {e}"
            sample_result["elapsed_seconds"] = time.time() - t0
            results["per_sample"].append(sample_result)
            new_total += 1
            _print_progress(progress_i, len(to_process), new_correct, new_total, "API_ERR")
            continue

        # Evaluate via SQL execution
        try:
            is_correct, details = executor.evaluate_single(
                sample_result["teacher_sql"], gold_sql, db_id
            )
            sample_result["correct"] = is_correct
            sample_result["exec_details"] = details
            if details.get("pred_error"):
                new_exec_errors += 1
            if is_correct:
                new_correct += 1
        except Exception as e:
            sample_result["error"] = f"Execution error: {e}"
            new_exec_errors += 1

        sample_result["elapsed_seconds"] = time.time() - t0
        new_total += 1
        results["per_sample"].append(sample_result)

        status = "OK" if sample_result["correct"] else "WRONG"
        if sample_result["error"] and not sample_result["correct"]:
            status = "ERR"
        _print_progress(progress_i, len(to_process), new_correct, new_total, status)

        # Save periodically
        if (progress_i + 1) % SAVE_EVERY == 0:
            _update_totals(results)
            save_results(results, output_path)

    # Final save
    results["status"] = "completed"
    results["finished_at"] = datetime.datetime.now().isoformat()
    _update_totals(results)
    save_results(results, output_path)

    total_correct = results["correct"]
    total_completed = results["num_completed"]
    acc = results["execution_accuracy"]

    print(f"\n{'=' * 60}")
    print(f"  Recording Complete")
    print(f"{'=' * 60}")
    print(f"  Model: {teacher.model_name}")
    print(f"  Total recorded: {total_completed}/{num_samples}")
    print(f"  Execution accuracy: {acc:.2%} ({total_correct}/{total_completed})")
    print(f"  API errors: {results['api_errors']}")
    print(f"  Exec errors: {results['exec_errors']}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")


def _update_totals(results: dict) -> None:
    """Recompute top-level totals from per_sample entries."""
    per_sample = results["per_sample"]
    results["num_completed"] = len(per_sample)
    results["correct"] = sum(1 for s in per_sample if s.get("correct"))
    results["api_errors"] = sum(
        1 for s in per_sample if (s.get("error") or "").startswith("API error")
    )
    results["exec_errors"] = sum(
        1 for s in per_sample
        if (s.get("error") or "").startswith("Execution error")
        or (s.get("exec_details") or {}).get("pred_error")
    )
    n = results["num_completed"]
    results["execution_accuracy"] = results["correct"] / n if n > 0 else 0.0


def _print_progress(i: int, total: int, correct: int, evaluated: int, status: str):
    acc = correct / evaluated if evaluated > 0 else 0.0
    print(
        f"  [{i + 1:4d}/{total}] {status:8s} | "
        f"acc: {acc:.2%} ({correct}/{evaluated})"
    )


if __name__ == "__main__":
    main()
