"""
Experiment 1.1 — Baseline Trajectory

Teacher distillation, 10 rounds. The core experiment: student learns from
teacher responses, routing changes over time. Does accuracy improve? Does
the training signal degrade as fewer queries reach the teacher?
"""

from cascade.config import CascadeConfig


def get_config(threshold: float = -4.0) -> CascadeConfig:
    return CascadeConfig(
        experiment_name="1.1_baseline_trajectory",
        dataset="spider",
        num_rounds=10,
        queries_per_round=200,
        training_pool_size=2000,
        eval_set_size=200,
        train_after_round=True,
        training_source="teacher",
        initial_adapter_path=None,  # Start from base Qwen (no adapter)
        router_threshold=threshold,
        seed=42,
    )
