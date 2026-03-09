#!/usr/bin/env python3
"""
Self-consistency evaluation: generate N SQL candidates per sample,
execute all, take the execution result that appears most often.

Reuses the existing temp=0 results as the first vote, then runs
additional passes at temperature > 0.

Usage:
    python scripts/record_ollama_self_consistency.py \
        --dataset bird --model gpt-oss:120b --votes 5 --temperature 0.7
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
from collections import Counter

from openai import OpenAI

from cascade.prompts import build_query_messages
from src.datasets.loader import DatasetLoader
from src.evaluation.sql_cleaning import extract_sql_from_text
from src.evaluation.sql_executor import BIRDExecutor, get_spider_executor

SAVE_EVERY = 5


def load_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def build_sample_order(data, seed: int, num_samples: int) -> list[int]:
    indices = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:num_samples]


def generate_sql(client: OpenAI, model: str, messages: list[dict],
                 temperature: float = 0.7, timeout: float = 120.0) -> str:
    """Call Ollama via OpenAI-compatible API."""
    import re
    import requests

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""
    if content:
        return content

    # Fallback for thinking models
    base_url = client.base_url
    ollama_url = str(base_url).replace("/v1/", "/").replace("/v1", "/")
    resp = requests.post(
        f"{ollama_url}api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 4096},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    full_text = resp.json().get("message", {}).get("content", "")
    cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
    return cleaned if cleaned else full_text


def execute_sql_safe(executor, sql: str, db_id: str) -> tuple[bool, str]:
    """Execute SQL and return (success, result_key).

    result_key is a hashable representation of the result set for voting.
    """
    if not sql:
        return False, "EMPTY_SQL"
    try:
        success, result = executor.execute_sql(sql, db_id)
        if not success:
            return False, f"ERROR:{str(result)[:100]}"
        if result is None:
            return False, "NONE_RESULT"
        # Convert to a hashable string for comparison
        return True, str(sorted(str(r) for r in result[:50]))
    except Exception as e:
        return False, f"ERROR:{str(e)[:100]}"


def majority_vote(candidates: list[dict], executor, gold_sql: str,
                  db_id: str) -> dict:
    """Take majority vote across candidates by execution result."""
    result_groups: dict[str, list[int]] = {}

    for i, cand in enumerate(candidates):
        key = cand["result_key"]
        if key not in result_groups:
            result_groups[key] = []
        result_groups[key].append(i)

    # Find the most common result (excluding errors)
    non_error_groups = {k: v for k, v in result_groups.items()
                        if not k.startswith("ERROR:") and k != "EMPTY_SQL" and k != "NONE_RESULT"}

    if non_error_groups:
        best_key = max(non_error_groups, key=lambda k: len(non_error_groups[k]))
        best_idx = non_error_groups[best_key][0]
        best_sql = candidates[best_idx]["sql"]
        vote_count = len(non_error_groups[best_key])
    else:
        # All errored — pick the first non-empty one
        best_sql = next((c["sql"] for c in candidates if c["sql"]), candidates[0]["sql"])
        best_key = "ALL_ERRORS"
        vote_count = 0

    # Evaluate the winning SQL against gold
    try:
        is_correct, details = executor.evaluate_single(best_sql, gold_sql, db_id)
    except Exception as e:
        is_correct = False
        details = {"error": str(e)}

    return {
        "winning_sql": best_sql,
        "correct": is_correct,
        "vote_count": vote_count,
        "total_votes": len(candidates),
        "unique_results": len(result_groups),
        "exec_details": details,
    }


def update_totals(results: dict) -> None:
    per_sample = results["per_sample"]
    results["num_completed"] = len(per_sample)
    results["correct"] = sum(1 for s in per_sample if s.get("correct"))
    n = results["num_completed"]
    results["execution_accuracy"] = results["correct"] / n if n > 0 else 0.0
    # Also track greedy (vote 0) accuracy
    results["greedy_correct"] = sum(
        1 for s in per_sample
        if any(c.get("correct_individual") for c in s.get("candidates", [])[:1])
    )
    results["greedy_accuracy"] = results["greedy_correct"] / n if n > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Self-consistency evaluation with majority vote"
    )
    parser.add_argument("--model", type=str, default="gpt-oss:120b")
    parser.add_argument("--dataset", type=str, default="bird", choices=["spider", "bird"])
    parser.add_argument("--samples", type=int, default=1534)
    parser.add_argument("--votes", type=int, default=5,
                        help="Number of SQL candidates per sample (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for candidate generation (default: 0.7)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/v1")
    args = parser.parse_args()

    model_slug = args.model.replace(":", "_").replace("/", "_")
    if args.dataset == "bird":
        dataset_label, split_label = "bird", "dev"
    else:
        dataset_label, split_label = "spider", "train"

    if args.output is None:
        args.output = (f"results/cascade/{model_slug}_self_consistency_"
                       f"{dataset_label}_{split_label}_n{args.votes}.json")
    output_path = Path(args.output)

    print("=" * 60)
    print(f"  Self-Consistency: {args.model} — {dataset_label.upper()} {split_label}")
    print(f"  Votes per sample: {args.votes}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Target samples: {args.samples}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Resume support
    existing = None
    completed_indices: set[int] = set()
    if args.resume:
        existing = load_existing(output_path)
        if existing:
            for entry in existing.get("per_sample", []):
                completed_indices.add(entry["idx"])
            print(f"  Resuming: {len(completed_indices)} samples already recorded")

    # Initialize Ollama client
    client = OpenAI(base_url=args.ollama_url, api_key="ollama")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if args.model not in available:
            print(f"WARNING: {args.model} not in available models: {available}")
        else:
            print(f"  Model {args.model} available")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
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
            print("ERROR: Database directory not found!")
            sys.exit(1)
        print(f"  Spider train set size: {len(eval_data)}")

    num_samples = min(args.samples, len(eval_data))
    sample_indices = build_sample_order(eval_data, args.seed, num_samples)
    to_process = [idx for idx in sample_indices if idx not in completed_indices]
    print(f"  Samples to process: {len(to_process)}")

    # Initialize results
    if existing and args.resume:
        results = existing
        results["num_requested"] = num_samples
        results["status"] = "running"
    else:
        results = {
            "model": args.model,
            "dataset": dataset_label,
            "split": split_label,
            "method": "self_consistency",
            "votes_per_sample": args.votes,
            "temperature": args.temperature,
            "seed": args.seed,
            "num_requested": num_samples,
            "num_completed": 0,
            "correct": 0,
            "execution_accuracy": 0.0,
            "greedy_correct": 0,
            "greedy_accuracy": 0.0,
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(),
            "finished_at": None,
            "per_sample": [],
        }

    save_results(results, output_path)

    sc_correct = 0
    greedy_correct = 0
    total = 0

    for progress_i, dataset_idx in enumerate(to_process):
        sample = eval_data[dataset_idx]
        prompt = sample["prompt"]
        gold_sql = sample["query"]
        db_id = sample["db_id"]

        messages = build_query_messages(args.dataset, prompt)

        # Generate N candidates
        candidates = []
        t0 = time.time()

        for vote_i in range(args.votes):
            temp = 0.0 if vote_i == 0 else args.temperature
            try:
                raw = generate_sql(client, args.model, messages, temperature=temp)
                sql = extract_sql_from_text(raw)
                success, result_key = execute_sql_safe(executor, sql, db_id)

                # Check individual correctness
                try:
                    indiv_correct, _ = executor.evaluate_single(sql, gold_sql, db_id)
                except Exception:
                    indiv_correct = False

                candidates.append({
                    "vote": vote_i,
                    "temperature": temp,
                    "sql": sql,
                    "result_key": result_key,
                    "exec_success": success,
                    "correct_individual": indiv_correct,
                })
            except Exception as e:
                candidates.append({
                    "vote": vote_i,
                    "temperature": temp,
                    "sql": None,
                    "result_key": f"ERROR:{e}",
                    "exec_success": False,
                    "correct_individual": False,
                })

        # Majority vote
        vote_result = majority_vote(candidates, executor, gold_sql, db_id)
        elapsed = time.time() - t0

        sample_result = {
            "idx": dataset_idx,
            "db_id": db_id,
            "question": sample.get("question", ""),
            "gold_sql": gold_sql,
            "correct": vote_result["correct"],
            "winning_sql": vote_result["winning_sql"],
            "vote_count": vote_result["vote_count"],
            "unique_results": vote_result["unique_results"],
            "greedy_correct": candidates[0]["correct_individual"] if candidates else False,
            "elapsed_seconds": elapsed,
            "candidates": [
                {"vote": c["vote"], "sql": c["sql"],
                 "correct_individual": c["correct_individual"],
                 "exec_success": c["exec_success"]}
                for c in candidates
            ],
        }
        if args.dataset == "bird":
            sample_result["evidence"] = sample.get("evidence", "")
            sample_result["difficulty"] = sample.get("difficulty", "unknown")

        results["per_sample"].append(sample_result)
        total += 1
        if vote_result["correct"]:
            sc_correct += 1
        if candidates and candidates[0]["correct_individual"]:
            greedy_correct += 1

        sc_acc = sc_correct / total if total > 0 else 0.0
        g_acc = greedy_correct / total if total > 0 else 0.0
        status = "SC_OK" if vote_result["correct"] else "SC_WRONG"
        g_tag = "g:OK" if candidates[0]["correct_individual"] else "g:X"
        print(
            f"  [{progress_i + 1:4d}/{len(to_process)}] {status:8s} {g_tag:4s} | "
            f"SC:{sc_acc:.2%} greedy:{g_acc:.2%} | "
            f"votes:{vote_result['vote_count']}/{args.votes} "
            f"uniq:{vote_result['unique_results']} "
            f"({elapsed:.1f}s)"
        )

        if (progress_i + 1) % SAVE_EVERY == 0:
            update_totals(results)
            save_results(results, output_path)

    # Final save
    results["status"] = "completed"
    results["finished_at"] = datetime.datetime.now().isoformat()
    update_totals(results)
    save_results(results, output_path)

    print(f"\n{'=' * 60}")
    print(f"  Self-Consistency Complete")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Votes per sample: {args.votes}")
    print(f"  Total: {results['num_completed']}/{num_samples}")
    print(f"  SC accuracy:     {results['execution_accuracy']:.2%} ({results['correct']}/{results['num_completed']})")
    print(f"  Greedy accuracy: {results['greedy_accuracy']:.2%} ({results['greedy_correct']}/{results['num_completed']})")
    print(f"  Lift: +{(results['execution_accuracy'] - results['greedy_accuracy']) * 100:.2f}%")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
