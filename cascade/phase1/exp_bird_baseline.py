"""
Experiment: BIRD Baseline — Ollama Teacher (gpt-oss:120b) Cascade Distillation

Teacher uses JSON-structured prompts (bird_json) for better SQL quality.
Student trains on extracted SQL with regular BIRD few-shot prompts.
20 rounds x 330 queries = 6600 training queries from the BIRD train split.
350 stratified eval samples (preserves difficulty distribution).

LR warmup: 2e-5 → 1e-4 over first 5 rounds, then constant.
1 epoch per round to limit memorization of noisy teacher data.
Eval every 2 rounds for speed.
"""

from cascade.config import CascadeConfig


def get_config(threshold: float = -4.0) -> CascadeConfig:
    return CascadeConfig(
        experiment_name="bird_baseline",
        dataset="bird",
        prompt_variant="bird_json",      # JSON for teacher
        teacher_backend="ollama",
        teacher_model="gpt-oss:120b",
        ollama_url="http://localhost:11434/v1",
        num_rounds=20,
        queries_per_round=330,
        training_pool_size=6601,         # Full BIRD train
        eval_set_size=350,               # Stratified sample from BIRD dev (preserves difficulty distribution)
        eval_every_n_rounds=2,           # Eval every 2 rounds for speed
        train_after_round=True,
        training_source="teacher",
        train_student_responses=False,
        training_epochs=1,               # 1 epoch — less memorization of noisy data
        training_lr=1e-4,                # Target LR
        training_lr_start=2e-5,          # Start low
        training_lr_warmup_rounds=5,     # Ramp over first 5 rounds
        router_threshold=threshold,
        cascade_rate_start=0.7,          # 70% to teacher initially
        cascade_rate_end=0.3,            # 30% to teacher by end
        seed=42,
    )


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Run BIRD baseline cascade experiment")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load calibrated threshold if available
    import json
    from pathlib import Path

    from loguru import logger

    threshold = args.threshold
    if threshold is None:
        cal_path = Path("cascade/results/calibration_bird.json")
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            threshold = cal.get("threshold", -4.0)
            logger.info(f"Using calibrated threshold: {threshold}")
        else:
            threshold = -4.0
            logger.info(f"No BIRD calibration found, using default threshold: {threshold}")

    config = get_config(threshold)
    config.gpu_devices = args.gpu

    from cascade.runner import CascadeRunner

    runner = CascadeRunner(config)
    db_path = runner.run()
    logger.info(f"Experiment complete. DB: {db_path}")


if __name__ == "__main__":
    main()
