"""
Pre-cache teacher (gpt-oss:120b) responses for BIRD train queries.

Replicates the runner's deterministic sampling logic (seed=42, shuffled pool,
per-round Random(seed+round_idx) sampling without replacement) to identify the
exact queries used in rounds 0–N. Then generates teacher responses via Ollama.

Saves to cascade/results/bird_train_teacher_cache.json in the same format as
existing response files (per_sample list with idx, db_id, question, gold_sql,
model_sql, correct, ...).

Supports --resume for incremental progress (saves every 10 samples).
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def get_round_queries(
    training_pool: list[dict],
    num_rounds: int,
    queries_per_round: int,
    seed: int,
) -> list[tuple[int, dict]]:
    """Replicate runner's _sample_queries() for rounds 0..num_rounds-1.

    Returns list of (pool_index, query) tuples in round order.
    """
    used_indices: set[int] = set()
    all_sampled: list[tuple[int, dict]] = []

    for round_idx in range(num_rounds):
        rng = random.Random(seed + round_idx)
        available = [
            (i, q) for i, q in enumerate(training_pool)
            if i not in used_indices
        ]
        n = min(queries_per_round, len(available))
        sampled = rng.sample(available, n)
        for idx, _ in sampled:
            used_indices.add(idx)
        all_sampled.extend(sampled)

    return all_sampled


def load_training_pool(pool_size: int, seed: int) -> list[dict]:
    """Load BIRD train data and shuffle to match runner's _load_data()."""
    from src.datasets.loader import DatasetLoader

    loader = DatasetLoader()
    domain_ds = loader.load_bird()
    train_raw = list(domain_ds.train)

    rng = random.Random(seed)
    # Runner does stratified eval sampling first (consuming RNG state),
    # then shuffles train_raw. We need to replicate that exact RNG state.
    test_raw = list(domain_ds.test)
    # Replicate _stratified_sample RNG consumption for eval set
    _replicate_stratified_sample(test_raw, 350, rng)

    rng.shuffle(train_raw)
    return train_raw[:pool_size]


def _replicate_stratified_sample(
    data: list[dict], n: int, rng: random.Random
) -> None:
    """Consume the same RNG state as CascadeRunner._stratified_sample().

    We don't need the result — just need to advance the RNG identically.
    """
    if n >= len(data):
        return

    groups: dict[str, list[dict]] = {}
    for item in data:
        key = item.get("difficulty", "unknown")
        groups.setdefault(key, []).append(item)

    if len(groups) <= 1:
        shuffled = list(data)
        rng.shuffle(shuffled)
        return

    total = len(data)
    sampled = []
    remainder = []
    for key, items in sorted(groups.items()):
        rng.shuffle(items)
        k = round(len(items) / total * n)
        sampled.extend(items[:k])
        remainder.extend(items[k:])

    if len(sampled) < n:
        rng.shuffle(remainder)
    elif len(sampled) > n:
        rng.shuffle(sampled)

    rng.shuffle(sampled if len(sampled) <= n else sampled[:n])


def generate_teacher_response(
    client, model: str, messages: list[dict], temperature: float = 0.0
) -> str:
    """Call Ollama teacher model and return response text."""
    import re
    import requests

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=8192,
        timeout=120.0,
    )
    content = resp.choices[0].message.content or ""
    if content:
        return content

    # Fallback for thinking models
    base_url = str(client.base_url)
    ollama_url = base_url.replace("/v1/", "/").replace("/v1", "/")
    native_resp = requests.post(
        f"{ollama_url}api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 8192},
        },
        timeout=120.0,
    )
    native_resp.raise_for_status()
    full_text = native_resp.json().get("message", {}).get("content", "")
    cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
    return cleaned if cleaned else full_text


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Pre-cache teacher responses for BIRD train queries")
    parser.add_argument("--gpu", default="2", help="GPU device(s)")
    parser.add_argument("--num-rounds", type=int, default=4, help="Number of rounds to pre-cache")
    parser.add_argument("--queries-per-round", type=int, default=400, help="Queries per round")
    parser.add_argument("--pool-size", type=int, default=6601, help="Training pool size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", default="gpt-oss:120b", help="Teacher model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1", help="Ollama URL")
    parser.add_argument("--output", default="cascade/results/bird_train_teacher_cache.json", help="Output path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--eval-size", type=int, default=350, help="Eval set size (for RNG alignment)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    existing_results: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        existing_results = data.get("per_sample", [])
        for entry in existing_results:
            completed_keys.add((entry["db_id"], entry["question"]))
        logger.info(f"Resuming: {len(existing_results)} already completed")

    # Load dataset and replicate sampling
    logger.info("Loading BIRD training pool...")
    training_pool = load_training_pool(args.pool_size, args.seed)
    logger.info(f"Training pool: {len(training_pool)} queries")

    sampled = get_round_queries(
        training_pool, args.num_rounds, args.queries_per_round, args.seed
    )
    logger.info(f"Total queries to cache: {len(sampled)}")

    # Filter out already completed
    to_process = [
        (pool_idx, q) for pool_idx, q in sampled
        if (q["db_id"], q.get("question", "")) not in completed_keys
    ]
    logger.info(f"Remaining after resume filter: {len(to_process)}")

    if not to_process:
        logger.info("All queries already cached!")
        return

    # Initialize Ollama client
    from openai import OpenAI
    client = OpenAI(base_url=args.ollama_url, api_key="ollama")

    # Import helpers
    from cascade.prompts import build_query_messages
    from src.evaluation.sql_cleaning import extract_sql_from_json
    from src.evaluation.sql_executor import BIRDExecutor

    executor = BIRDExecutor()

    results = list(existing_results)
    correct_count = sum(1 for r in results if r.get("correct"))
    error_count = sum(1 for r in results if r.get("error"))
    start_time = time.time()

    for i, (pool_idx, query) in enumerate(to_process):
        db_id = query["db_id"]
        question = query.get("question", "")
        gold_sql = query.get("query") or query.get("SQL", "")
        prompt = query.get("prompt", question)

        entry = {
            "idx": pool_idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "prompt": prompt[:2000],
            "model_response_raw": "",
            "model_sql": "",
            "correct": False,
            "exec_details": "",
            "error": None,
            "elapsed_seconds": 0.0,
            "evidence": query.get("evidence", ""),
            "difficulty": query.get("difficulty", ""),
            "model_reasoning": "",
        }

        t0 = time.time()
        try:
            messages = build_query_messages("bird_json", prompt)
            response = generate_teacher_response(client, args.model, messages)
            entry["model_response_raw"] = response

            sql, reasoning = extract_sql_from_json(response)
            entry["model_sql"] = sql
            entry["model_reasoning"] = reasoning

            is_correct, details = executor.evaluate_single(sql, gold_sql, db_id)
            entry["correct"] = is_correct
            entry["exec_details"] = str(details) if details else ""
            if is_correct:
                correct_count += 1

        except Exception as e:
            entry["error"] = str(e)
            error_count += 1
            logger.warning(f"Error on query {pool_idx} ({db_id}): {e}")

        entry["elapsed_seconds"] = time.time() - t0
        results.append(entry)

        # Progress logging
        total_done = len(results)
        total_target = len(sampled)
        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (len(to_process) - i - 1) / rate / 60 if rate > 0 else 0
            accuracy = correct_count / total_done if total_done > 0 else 0
            logger.info(
                f"Progress: {total_done}/{total_target} "
                f"({accuracy:.1%} correct, {error_count} errors, "
                f"{rate:.2f} q/s, ETA {eta_min:.0f}min)"
            )

            # Save checkpoint
            _save_results(output_path, results, args, correct_count, error_count)

    # Final save
    _save_results(output_path, results, args, correct_count, error_count)
    executor.close()

    logger.info(
        f"Done! {len(results)} queries cached to {output_path} "
        f"({correct_count}/{len(results)} correct = "
        f"{correct_count / len(results):.1%})"
    )


def _save_results(
    path: Path,
    results: list[dict],
    args: argparse.Namespace,
    correct_count: int,
    error_count: int,
) -> None:
    """Save results to JSON file."""
    output = {
        "model": args.model,
        "dataset": "bird",
        "split": "train",
        "seed": args.seed,
        "num_requested": args.num_rounds * args.queries_per_round,
        "num_completed": len(results),
        "correct": correct_count,
        "execution_accuracy": correct_count / len(results) if results else 0,
        "api_errors": error_count,
        "exec_errors": 0,
        "status": "in_progress" if len(results) < args.num_rounds * args.queries_per_round else "completed",
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "per_sample": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
