"""
Run all Phase 1 experiments sequentially.

Each experiment gets its own CascadeRunner instance. The student model
is reloaded from scratch for each experiment to ensure independence.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def _build_phase1_experiments(threshold: float, gpu: str | None = None):
    from cascade.phase1.exp_1_1_baseline import get_config as get_1_1
    from cascade.phase1.exp_1_2_static import get_config as get_1_2
    from cascade.phase1.exp_1_3_ground_truth import get_config as get_1_3

    experiments = [
        ("1.1 Baseline Trajectory", get_1_1(threshold)),
        ("1.2 Static Control", get_1_2(threshold)),
        ("1.3 Ground Truth", get_1_3(threshold)),
    ]

    if gpu is not None:
        for _, config in experiments:
            config.gpu_devices = gpu

    return experiments


def run_phase1(threshold: float | None = None, gpu: str | None = None):
    load_dotenv(override=True)

    from cascade.runner import CascadeRunner

    # Load calibrated threshold if available
    if threshold is None:
        cal_path = Path("cascade/results/calibration.json")
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            threshold = cal.get("threshold", -4.0)
            logger.info(f"Using calibrated threshold: {threshold}")
        else:
            threshold = -4.0
            logger.info(f"No calibration found, using default threshold: {threshold}")

    experiments = _build_phase1_experiments(threshold, gpu=gpu)

    results = {}

    for name, config in experiments:
        logger.info(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        start = time.time()

        try:
            runner = CascadeRunner(config)
            db_path = runner.run()
            elapsed = time.time() - start
            results[config.experiment_name] = {
                "db_path": db_path,
                "elapsed_seconds": elapsed,
                "status": "success",
            }
            logger.info(f"{name} completed in {elapsed:.0f}s: {db_path}")
        except Exception as e:
            elapsed = time.time() - start
            results[config.experiment_name] = {
                "error": str(e),
                "elapsed_seconds": elapsed,
                "status": "failed",
            }
            logger.error(f"{name} FAILED after {elapsed:.0f}s: {e}")

        # Force cleanup between experiments
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary_path = Path("cascade/results/phase1_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nPhase 1 complete. Summary: {summary_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 1 experiments")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_phase1(args.threshold, gpu=args.gpu)


if __name__ == "__main__":
    main()
