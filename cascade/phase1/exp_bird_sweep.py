"""
Hyperparameter sweep: 6 configs testing LR, replay ratio, and training schedule
to prevent the eval accuracy dip observed in the BIRD baseline experiment.

All configs: BIRD, bird_json, 400 queries/round, 350 eval, seed=42,
cascade 0.7→0.3, LoRA r=32, 1 epoch, teacher cache, 4 rounds, eval every round.

Baseline reference (old run): LR warmup 2e-5→1e-4/5r, replay=0.2
  → R0:38.9%, R2:28.6% (−10pp dip)

Usage:
    python -m cascade.phase1.exp_bird_sweep --config low_lr --gpu 2
    python -m cascade.phase1.exp_bird_sweep --config all --gpu 2  # run all sequentially
"""

from cascade.config import CascadeConfig

TEACHER_CACHE_PATH = "results/cascade/bird_train_teacher_cache.json"

SWEEP_CONFIGS = {
    "low_lr": {
        "hypothesis": "High LR causes catastrophic forgetting",
        "overrides": {
            "training_lr": 5e-6,
            "training_lr_start": 0.0,  # No warmup, constant LR
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "ultra_low_lr": {
        "hypothesis": "Even more conservative updates",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "high_replay": {
        "hypothesis": "More regularization from diverse replay",
        "overrides": {
            "training_lr": 1e-4,
            "training_lr_start": 2e-5,
            "training_lr_warmup_rounds": 5,
            "replay_ratio": 0.5,
        },
    },
    "very_high_replay": {
        "hypothesis": "Maximum regularization",
        "overrides": {
            "training_lr": 1e-4,
            "training_lr_start": 2e-5,
            "training_lr_warmup_rounds": 5,
            "replay_ratio": 0.8,
        },
    },
    "low_lr_high_replay": {
        "hypothesis": "Combined conservative approach",
        "overrides": {
            "training_lr": 2e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.5,
        },
    },
    "delayed_train": {
        "hypothesis": "Accumulate 2 rounds of data before first training",
        "overrides": {
            "training_lr": 1e-4,
            "training_lr_start": 2e-5,
            "training_lr_warmup_rounds": 5,
            "replay_ratio": 0.2,
            "train_start_round": 2,
        },
    },
    # --- Phase 2: LR binary search (informed by phase 1 results) ---
    # low_lr (5e-6) prevents dip but stays flat. Baseline (1e-4) dips badly.
    # Goal: find highest LR that avoids the dip while enabling improvement.
    "lr_1e5": {
        "hypothesis": "1e-5 is just above the safe 5e-6 — does it still avoid the dip?",
        "overrides": {
            "training_lr": 1e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "lr_2e5": {
        "hypothesis": "2e-5 constant — midpoint between safe and dangerous LR",
        "overrides": {
            "training_lr": 2e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "lr_5e5": {
        "hypothesis": "5e-5 constant — approaching baseline territory, does dip return?",
        "overrides": {
            "training_lr": 5e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "low_lr_3epoch": {
        "hypothesis": "5e-6 with 3 epochs — more gradient steps at safe LR to enable learning",
        "overrides": {
            "training_lr": 5e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_epochs": 3,
        },
    },
    "lr_1e5_3epoch": {
        "hypothesis": "1e-5 with 3 epochs — more aggressive safe training",
        "overrides": {
            "training_lr": 1e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_epochs": 3,
        },
    },
    "lr_2e5_3epoch": {
        "hypothesis": "2e-5 with 3 epochs — is this the sweet spot?",
        "overrides": {
            "training_lr": 2e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_epochs": 3,
        },
    },
    # --- Phase 3: Batch size experiments ---
    # Effective BS=16 is current default (bs=4, accum=4).
    # Larger batch = smoother gradients ≈ lower effective LR noise.
    # Can we use higher LR safely with larger batch?
    "lr_2e5_bs32": {
        "hypothesis": "2e-5 with effective BS=32 — smoother updates at moderate LR",
        "overrides": {
            "training_lr": 2e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_grad_accum": 8,
        },
    },
    "lr_5e5_bs32": {
        "hypothesis": "5e-5 with effective BS=32 — does larger batch tame the dip?",
        "overrides": {
            "training_lr": 5e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_grad_accum": 8,
        },
    },
    "lr_1e4_bs64": {
        "hypothesis": "Original 1e-4 but effective BS=64 — does bigger batch prevent dip at baseline LR?",
        "overrides": {
            "training_lr": 1e-4,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_grad_accum": 16,
        },
    },
    # --- Phase 4: Deep dive into ultra-low LR range ---
    # ultra_low_lr (1e-6) is improving at R2 (39.7%). Explore this range.
    "lr_5e7": {
        "hypothesis": "Even lower than 1e-6 — does 5e-7 still learn?",
        "overrides": {
            "training_lr": 5e-7,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "lr_2e6": {
        "hypothesis": "2e-6 — between ultra_low_lr (1e-6) and low_lr (5e-6)",
        "overrides": {
            "training_lr": 2e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "lr_3e6": {
        "hypothesis": "3e-6 — slightly above 2e-6, still in safe zone?",
        "overrides": {
            "training_lr": 3e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "lr_1e6_3epoch": {
        "hypothesis": "1e-6 with 3 epochs — more gradient steps at the best LR so far",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_epochs": 3,
        },
    },
    "lr_2e6_3epoch": {
        "hypothesis": "2e-6 with 3 epochs — sweet spot of LR × steps?",
        "overrides": {
            "training_lr": 2e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_epochs": 3,
        },
    },
    "lr_1e6_5epoch": {
        "hypothesis": "1e-6 with 5 epochs — push the best LR harder",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_epochs": 5,
        },
    },
    "lr_1e6_bs32": {
        "hypothesis": "1e-6 with BS=32 — smoother gradients at ultra-low LR",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_grad_accum": 8,
        },
    },
    "lr_2e6_bs32": {
        "hypothesis": "2e-6 with BS=32 — smoother gradients, slightly higher LR",
        "overrides": {
            "training_lr": 2e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_grad_accum": 8,
        },
    },
    # --- Phase 5: Sub-micro LR ---
    "lr_1e7": {
        "hypothesis": "1e-7 — barely moving weights, does it still learn?",
        "overrides": {
            "training_lr": 1e-7,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    "lr_1e8": {
        "hypothesis": "1e-8 — near-zero updates, control for noise",
        "overrides": {
            "training_lr": 1e-8,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
        },
    },
    # --- Phase 6: Ground truth training (is teacher quality the bottleneck?) ---
    "gold_1e6": {
        "hypothesis": "Ground truth SQL at best LR — is teacher quality the bottleneck?",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_source": "gold",
        },
    },
    "gold_5e6": {
        "hypothesis": "Ground truth at 5e-6 — does perfect signal tolerate higher LR?",
        "overrides": {
            "training_lr": 5e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_source": "gold",
        },
    },
    "gold_1e5": {
        "hypothesis": "Ground truth at 1e-5 — higher LR safe with clean signal?",
        "overrides": {
            "training_lr": 1e-5,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_source": "gold",
        },
    },
    "gold_1e6_3epoch": {
        "hypothesis": "Ground truth at 1e-6, 3 epochs — more steps with perfect signal",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_source": "gold",
            "training_epochs": 3,
        },
    },
    # --- Full runs: 20 rounds × 330 queries = 6,600 training queries ---
    "full_ultra_low_lr": {
        "hypothesis": "Full run: 1e-6 constant, most stable config from sweep",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "num_rounds": 20,
            "queries_per_round": 330,
            "eval_every_n_rounds": 1,
        },
    },
    "full_lr_5e7": {
        "hypothesis": "Full run: 5e-7 constant, best improving teacher config from sweep",
        "overrides": {
            "training_lr": 5e-7,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "num_rounds": 20,
            "queries_per_round": 330,
            "eval_every_n_rounds": 1,
        },
    },
    "full_lr_1e6_bs32": {
        "hypothesis": "Full run: 1e-6 BS=32, best R3 teacher config from sweep",
        "overrides": {
            "training_lr": 1e-6,
            "training_lr_start": 0.0,
            "training_lr_warmup_rounds": 0,
            "replay_ratio": 0.2,
            "training_grad_accum": 8,
            "num_rounds": 20,
            "queries_per_round": 330,
            "eval_every_n_rounds": 1,
        },
    },
}


def get_config(name: str, gpu: str = "2") -> CascadeConfig:
    """Build a sweep config by name."""
    if name not in SWEEP_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(SWEEP_CONFIGS.keys())}")

    spec = SWEEP_CONFIGS[name]

    # Shared baseline
    config = CascadeConfig(
        experiment_name=f"sweep_{name}",
        dataset="bird",
        prompt_variant="bird_json",
        teacher_backend="ollama",
        teacher_model="gpt-oss:120b",
        ollama_url="http://localhost:11434/v1",
        num_rounds=4,
        queries_per_round=400,
        training_pool_size=6601,
        eval_set_size=350,
        eval_every_n_rounds=1,
        train_after_round=True,
        training_source="teacher",
        train_student_responses=False,
        training_epochs=1,
        lora_r=32,
        lora_alpha=32,
        cascade_rate_start=0.7,
        cascade_rate_end=0.3,
        seed=42,
        teacher_cache_path=TEACHER_CACHE_PATH,
        gpu_devices=gpu,
    )

    # Apply overrides
    for key, value in spec["overrides"].items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config field: {key}")
        setattr(config, key, value)

    return config


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Run BIRD hyperparameter sweep experiments")
    parser.add_argument(
        "--config",
        default="all",
        help=f"Config name or 'all'. Available: {', '.join(SWEEP_CONFIGS.keys())}",
    )
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from loguru import logger

    configs_to_run = list(SWEEP_CONFIGS.keys()) if args.config == "all" else [args.config]

    for config_name in configs_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting sweep config: {config_name}")
        logger.info(f"Hypothesis: {SWEEP_CONFIGS[config_name]['hypothesis']}")
        logger.info(f"{'='*60}")

        config = get_config(config_name, gpu=args.gpu)

        from cascade.runner import CascadeRunner

        runner = CascadeRunner(config)
        db_path = runner.run()
        logger.info(f"Config {config_name} complete. DB: {db_path}")

        # Clean up GPU memory between runs
        import gc

        import torch

        del runner
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
