#!/usr/bin/env python3
"""
Record responses from an Ollama model on Spider train samples
with execution accuracy.

Uses the same 3000 samples (seed=42 ordering) as the teacher/student scripts.
Calls Ollama's OpenAI-compatible API at localhost:11434.

Usage:
    python scripts/record_ollama_responses.py --model gpt-oss:120b --samples 3000
    python scripts/record_ollama_responses.py --model gpt-oss:120b --samples 3000 --resume
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import argparse
import datetime
import json
import random
import time

from openai import OpenAI

from cascade.prompts import build_query_messages
from src.datasets.loader import DatasetLoader
from src.evaluation.sql_cleaning import extract_sql_from_text, extract_sql_from_json
from src.evaluation.sql_executor import BIRDExecutor, get_spider_executor

SAVE_EVERY = 10


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
    indices = list(range(len(train_data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:num_samples]


def update_totals(results: dict) -> None:
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


def generate_sql(client: OpenAI, model: str, messages: list[dict], timeout: float = 120.0) -> str:
    """Call Ollama via OpenAI-compatible API.

    Handles thinking models (qwen3, etc.) that put reasoning in a separate
    field and may return empty content. Falls back to Ollama's native API
    to capture the full response including <think> blocks.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=8192,
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""
    if content:
        return content

    # Fallback for thinking models: use Ollama's native /api/chat endpoint
    # which returns the full text including <think>...</think> blocks
    import re
    import requests

    # Derive base Ollama URL from the OpenAI-compat URL
    base_url = client.base_url
    ollama_url = str(base_url).replace("/v1/", "/").replace("/v1", "/")
    resp = requests.post(
        f"{ollama_url}api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 8192},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    full_text = resp.json().get("message", {}).get("content", "")

    # Strip <think>...</think> blocks to get the actual answer
    cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
    return cleaned if cleaned else full_text


def main():
    parser = argparse.ArgumentParser(
        description="Record Ollama model responses on Spider train set"
    )
    parser.add_argument("--model", type=str, default="gpt-oss:120b")
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird"])
    parser.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought reasoning (BIRD only)")
    parser.add_argument("--json", action="store_true",
                        help="Enable JSON structured output with reasoning (BIRD only)")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto from model name)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/v1")
    args = parser.parse_args()

    # Determine the prompt variant for build_query_messages
    if args.dataset == "bird" and args.json:
        args.prompt_dataset = "bird_json"
    elif args.dataset == "bird" and args.cot:
        args.prompt_dataset = "bird_cot"
    else:
        args.prompt_dataset = args.dataset

    # Dataset/executor/output switching
    model_slug = args.model.replace(":", "_").replace("/", "_")
    if args.dataset == "bird":
        dataset_label, split_label = "bird", "dev"
    else:
        dataset_label, split_label = "spider", "train"

    variant_tag = "_json" if args.json else ("_cot" if args.cot else "")
    if args.output is None:
        args.output = f"cascade/results/{model_slug}_responses_{dataset_label}_{split_label}{variant_tag}.json"
    output_path = Path(args.output)

    variant_label = " (JSON)" if args.json else (" (CoT)" if args.cot else "")
    print("=" * 60)
    print(f"  Record Ollama ({args.model}){variant_label} — {dataset_label.upper()} {split_label}")
    print(f"  Target samples: {args.samples}")
    print(f"  Output: {output_path}")
    print(f"  Resume: {args.resume}")
    print(f"  Ollama URL: {args.ollama_url}")
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

    # Initialize Ollama client
    client = OpenAI(base_url=args.ollama_url, api_key="ollama")

    # Quick connectivity check
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if args.model not in available:
            print(f"WARNING: {args.model} not in available models: {available}")
        else:
            print(f"  Model {args.model} available")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama at {args.ollama_url}: {e}")
        sys.exit(1)

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
            "model": args.model,
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
            "prompt": prompt,
            "model_response_raw": None,
            "model_sql": None,
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
            raw_response = generate_sql(client, args.model, messages)
            sample_result["model_response_raw"] = raw_response
            if args.json:
                sql, reasoning = extract_sql_from_json(raw_response)
                sample_result["model_sql"] = sql
                sample_result["model_reasoning"] = reasoning
            else:
                sample_result["model_sql"] = extract_sql_from_text(raw_response)
        except Exception as e:
            sample_result["error"] = f"API error: {e}"
            sample_result["elapsed_seconds"] = time.time() - t0
            results["per_sample"].append(sample_result)
            new_total += 1
            _print_progress(progress_i, len(to_process), new_correct, new_total, "API_ERR")
            # Save after errors
            if (progress_i + 1) % SAVE_EVERY == 0:
                update_totals(results)
                save_results(results, output_path)
            continue

        try:
            is_correct, details = executor.evaluate_single(
                sample_result["model_sql"], gold_sql, db_id
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
    print(f"  Model: {args.model}")
    print(f"  Total recorded: {results['num_completed']}/{num_samples}")
    print(f"  Execution accuracy: {results['execution_accuracy']:.2%} ({results['correct']}/{results['num_completed']})")
    print(f"  API errors: {results['api_errors']}")
    print(f"  Exec errors: {results['exec_errors']}")
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
