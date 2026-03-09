"""
Threshold calibration for cascade routing.

Runs the student on the eval set, records (confidence, correct) pairs,
and sweeps thresholds to find one yielding ~85% student accuracy
on the confident partition.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger

if __package__ in (None, ""):
    # Support direct script execution: `python cascade/calibrate.py ...`
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from cascade.config import CascadeConfig
from cascade.evaluator import CascadeEvaluator
from cascade.student import CascadeStudent


def calibrate_threshold(
    student: CascadeStudent,
    evaluator: CascadeEvaluator,
    eval_queries: list[dict],
    metric: str = "mean_logprob",
    target_accuracy: float = 0.85,
) -> dict:
    """
    Sweep thresholds and find one that yields ~target_accuracy on the
    "confident" partition (queries above threshold).

    Returns:
        {threshold, student_accuracy_above, cascade_rate, distribution_stats}
    """
    if not eval_queries:
        raise ValueError("eval_queries is empty; cannot calibrate threshold")

    # Collect (confidence_value, correct) pairs
    pairs = []

    for i, query in enumerate(eval_queries):
        messages = evaluator._build_messages(query)
        gen_result = student.generate_with_logprobs(messages)

        gold_sql = query.get("query") or query.get("SQL", "")
        db_id = query["db_id"]
        correct = evaluator.check_single(gen_result.text, gold_sql, db_id)

        if metric == "mean_logprob":
            value = gen_result.mean_log_prob
        elif metric == "min_logprob":
            value = gen_result.min_log_prob
        elif metric == "mean_entropy":
            value = -gen_result.mean_entropy
        else:
            value = gen_result.mean_log_prob

        pairs.append((value, correct))

        if (i + 1) % 20 == 0:
            logger.info(f"Calibration: {i + 1}/{len(eval_queries)} queries processed")

    values = [p[0] for p in pairs]
    corrects = [p[1] for p in pairs]

    # Distribution stats
    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "overall_accuracy": sum(corrects) / len(corrects),
    }

    # Sweep thresholds
    best_threshold = None
    best_diff = float("inf")
    sweep_results = []

    thresholds = np.linspace(np.min(values), np.max(values), 50)
    for t in thresholds:
        above = [(v, c) for v, c in pairs if v >= t]
        below = [(v, c) for v, c in pairs if v < t]

        if not above:
            continue

        acc_above = sum(c for _, c in above) / len(above)
        cascade_rate = len(below) / len(pairs)

        sweep_results.append({
            "threshold": float(t),
            "accuracy_above": float(acc_above),
            "cascade_rate": float(cascade_rate),
            "n_above": len(above),
            "n_below": len(below),
        })

        diff = abs(acc_above - target_accuracy)
        if diff < best_diff:
            best_diff = diff
            best_threshold = float(t)

    result = {
        "threshold": best_threshold,
        "metric": metric,
        "target_accuracy": target_accuracy,
        "student_accuracy_above": None,
        "cascade_rate": None,
        "distribution_stats": stats,
        "sweep_results": sweep_results,
    }

    # Fill in best threshold stats
    for sr in sweep_results:
        if sr["threshold"] == best_threshold:
            result["student_accuracy_above"] = sr["accuracy_above"]
            result["cascade_rate"] = sr["cascade_rate"]
            break

    logger.info(
        f"Calibrated threshold: {best_threshold:.4f} "
        f"(acc_above={result['student_accuracy_above']:.3f}, "
        f"cascade_rate={result['cascade_rate']:.3f})"
    )

    return result


def main():
    """CLI entry point for threshold calibration."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Calibrate cascade router threshold")
    parser.add_argument("--dataset", default="spider", choices=["spider", "bird"])
    parser.add_argument("--prompt-variant", default="",
                        help="Prompt variant for student eval (e.g. 'bird_json'). Empty = use dataset.")
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--metric", default="mean_logprob")
    parser.add_argument("--target-accuracy", type=float, default=0.85)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--output", default="cascade/results/calibration.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = CascadeConfig(
        experiment_name="calibration",
        dataset=args.dataset,
        prompt_variant=args.prompt_variant,
        eval_set_size=args.eval_size,
        gpu_devices=args.gpu,
    )

    # Load student
    student = CascadeStudent(config)
    student.load()

    # Load evaluator
    evaluator = CascadeEvaluator(config)
    evaluator.load()

    # Load eval data
    from src.datasets.loader import DatasetLoader
    loader = DatasetLoader()
    if args.dataset == "spider":
        domain_ds = loader.load_spider()
    else:
        domain_ds = loader.load_bird()

    rng = random.Random(config.seed)
    eval_raw = list(domain_ds.test)
    rng.shuffle(eval_raw)
    eval_queries = eval_raw[:args.eval_size]

    # Calibrate
    result = calibrate_threshold(
        student, evaluator, eval_queries,
        metric=args.metric,
        target_accuracy=args.target_accuracy,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Calibration results saved to {output_path}")
    evaluator.close()


if __name__ == "__main__":
    main()
