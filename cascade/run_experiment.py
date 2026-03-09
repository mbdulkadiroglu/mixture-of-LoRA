"""
CLI entry point for running cascade experiments.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Run a cascade distillation experiment")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--name", type=str, help="Experiment name (overrides config)")
    parser.add_argument("--rounds", type=int, help="Number of rounds (overrides config)")
    parser.add_argument("--queries-per-round", type=int, help="Queries per round (overrides config)")
    parser.add_argument("--gpu", type=str, help="GPU devices (overrides config)")
    parser.add_argument("--no-train", action="store_true", help="Disable training")
    parser.add_argument("--training-source", choices=["teacher", "gold"], help="Training source")
    parser.add_argument("--threshold", type=float, help="Router threshold")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    try:
        from .config import CascadeConfig
        from .runner import CascadeRunner
    except ImportError:
        # Support direct script execution: `python cascade/run_experiment.py ...`
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from cascade.config import CascadeConfig
        from cascade.runner import CascadeRunner

    if args.config:
        config = CascadeConfig.load(args.config)
    else:
        config = CascadeConfig()

    # Apply CLI overrides
    if args.name:
        config.experiment_name = args.name
    if args.rounds is not None:
        config.num_rounds = args.rounds
    if args.queries_per_round is not None:
        config.queries_per_round = args.queries_per_round
    if args.gpu:
        config.gpu_devices = args.gpu
    if args.no_train:
        config.train_after_round = False
    if args.training_source:
        config.training_source = args.training_source
    if args.threshold is not None:
        config.router_threshold = args.threshold
    if args.output_dir:
        config.output_dir = args.output_dir

    # Set GPU before any torch imports in runner
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices

    runner = CascadeRunner(config)
    db_path = runner.run()

    logger.info(f"Experiment complete. Results: {db_path}")
    return db_path


if __name__ == "__main__":
    main()
