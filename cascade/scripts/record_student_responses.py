#!/usr/bin/env python3
"""
Record base Qwen 2.5 14B (no LoRA) responses on Spider train samples
with execution accuracy — for comparison against teacher responses.

Uses the same 3000 samples (seed=42 ordering) as record_teacher_responses.py.

Supports incremental recording via --resume.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/record_student_responses.py --samples 20
    CUDA_VISIBLE_DEVICES=0 python scripts/record_student_responses.py --samples 3000 --resume
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE PyTorch initializes (controls CUDA_VISIBLE_DEVICES)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Import unsloth BEFORE PyTorch/transformers (enables kernel optimizations)
import unsloth  # noqa: F401

import argparse
import datetime
import json
import random
import time

from cascade.prompts import build_query_messages
from src.config import load_config, LoRAConfig
from src.datasets.loader import DatasetLoader
from src.evaluation.sql_cleaning import extract_sql_from_text, extract_sql_from_json
from src.evaluation.sql_executor import BIRDExecutor, get_spider_executor
from src.models.student import StudentModel

DEFAULT_OUTPUT = "cascade/results/student_base_responses_spider_train.json"
SAVE_EVERY = 20


def load_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def build_sample_order(train_data, seed: int, num_samples: int) -> list[int]:
    """Same deterministic ordering as record_teacher_responses.py."""
    indices = list(range(len(train_data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:num_samples]


def update_totals(results: dict) -> None:
    per_sample = results["per_sample"]
    results["num_completed"] = len(per_sample)
    results["correct"] = sum(1 for s in per_sample if s.get("correct"))
    results["errors"] = sum(
        1 for s in per_sample
        if (s.get("error") or "").startswith("Execution error")
        or (s.get("exec_details") or {}).get("pred_error")
    )
    n = results["num_completed"]
    results["execution_accuracy"] = results["correct"] / n if n > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Record base Qwen 2.5 14B responses on Spider train set"
    )
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird"])
    parser.add_argument("--json", action="store_true",
                        help="Enable JSON structured output with reasoning (BIRD only)")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine prompt variant
    if args.dataset == "bird" and args.json:
        args.prompt_dataset = "bird_json"
    else:
        args.prompt_dataset = args.dataset

    # Dataset/executor/output switching
    if args.dataset == "bird":
        dataset_label, split_label = "bird", "dev"
    else:
        dataset_label, split_label = "spider", "train"

    variant_tag = "_json" if args.json else ""
    if args.output is None:
        args.output = f"cascade/results/student_base_responses_{dataset_label}_{split_label}{variant_tag}.json"
    output_path = Path(args.output)

    print("=" * 60)
    variant_label = " (JSON)" if args.json else ""
    print(f"  Record Base Student (Qwen 2.5 14B){variant_label} — {dataset_label.upper()} {split_label}")
    print(f"  Target samples: {args.samples}")
    print(f"  Output: {output_path}")
    print(f"  Resume: {args.resume}")
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

    # Load student model (base, no LoRA)
    config = load_config("configs/config.yaml")
    student = StudentModel(config.student, LoRAConfig())
    student.load_model()
    print(f"  Student model: {config.student.name}")

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

    num_samples = min(args.samples, len(eval_data))
    sample_indices = build_sample_order(eval_data, args.seed, num_samples)
    to_process = [idx for idx in sample_indices if idx not in completed_indices]
    print(f"  Samples to process this run: {len(to_process)}")
    exec_dir = getattr(executor, "database_dir", None) or getattr(executor, "bird_data_dir", None)
    print(f"  SQL executor ready: {exec_dir}")

    # Initialize or reuse results
    if existing and args.resume:
        results = existing
        results["num_requested"] = num_samples
        results["status"] = "running"
    else:
        results = {
            "model": config.student.name,
            "dataset": dataset_label,
            "split": split_label,
            "seed": args.seed,
            "num_requested": num_samples,
            "num_completed": 0,
            "correct": 0,
            "execution_accuracy": 0.0,
            "errors": 0,
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(),
            "finished_at": None,
            "per_sample": [],
        }

    save_results(results, output_path)

    new_correct = 0
    new_total = 0

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
            "student_response_raw": None,
            "student_sql": None,
            "correct": False,
            "exec_details": None,
            "error": None,
            "elapsed_seconds": None,
        }
        if args.dataset == "bird":
            sample_result["evidence"] = sample.get("evidence", "")
            sample_result["difficulty"] = sample.get("difficulty", "unknown")

        messages = build_query_messages(args.prompt_dataset, prompt)

        t0 = time.time()
        try:
            raw_response = student.generate_chat(
                messages, temperature=0.0, do_sample=False,
                max_new_tokens=2048 if args.json else 512,
            )
            sample_result["student_response_raw"] = raw_response
            if args.json:
                sql, reasoning = extract_sql_from_json(raw_response)
                sample_result["student_sql"] = sql
                sample_result["student_reasoning"] = reasoning
            else:
                sample_result["student_sql"] = extract_sql_from_text(raw_response)
        except Exception as e:
            sample_result["error"] = f"Generation error: {e}"
            sample_result["elapsed_seconds"] = time.time() - t0
            results["per_sample"].append(sample_result)
            new_total += 1
            _print_progress(progress_i, len(to_process), new_correct, new_total, "GEN_ERR")
            continue

        try:
            is_correct, details = executor.evaluate_single(
                sample_result["student_sql"], gold_sql, db_id
            )
            sample_result["correct"] = is_correct
            sample_result["exec_details"] = details
            if is_correct:
                new_correct += 1
        except Exception as e:
            sample_result["error"] = f"Execution error: {e}"

        sample_result["elapsed_seconds"] = time.time() - t0
        new_total += 1
        results["per_sample"].append(sample_result)

        status = "OK" if sample_result["correct"] else "WRONG"
        if sample_result["error"] and not sample_result["correct"]:
            status = "ERR"
        _print_progress(progress_i, len(to_process), new_correct, new_total, status)

        if (progress_i + 1) % SAVE_EVERY == 0:
            update_totals(results)
            save_results(results, output_path)

    # Final save
    results["status"] = "completed"
    results["finished_at"] = datetime.datetime.now().isoformat()
    update_totals(results)
    save_results(results, output_path)

    print(f"\n{'=' * 60}")
    print(f"  Recording Complete")
    print(f"{'=' * 60}")
    print(f"  Model: {config.student.name}")
    print(f"  Total recorded: {results['num_completed']}/{num_samples}")
    print(f"  Execution accuracy: {results['execution_accuracy']:.2%} ({results['correct']}/{results['num_completed']})")
    print(f"  Errors: {results['errors']}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")


def _print_progress(i: int, total: int, correct: int, evaluated: int, status: str):
    acc = correct / evaluated if evaluated > 0 else 0.0
    print(
        f"  [{i + 1:4d}/{total}] {status:8s} | "
        f"acc: {acc:.2%} ({correct}/{evaluated})"
    )


if __name__ == "__main__":
    main()
