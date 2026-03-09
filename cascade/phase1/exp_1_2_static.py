"""
Experiment 1.2 — Static Control

No training, 10 rounds. The control: same routing decisions, but student
never improves. Cascade rate and accuracy should remain constant.
"""

from cascade.config import CascadeConfig


def get_config(threshold: float = -4.0) -> CascadeConfig:
    return CascadeConfig(
        experiment_name="1.2_static_control",
        dataset="spider",
        num_rounds=10,
        queries_per_round=200,
        training_pool_size=2000,
        eval_set_size=200,
        train_after_round=False,  # No training
        training_source="teacher",
        initial_adapter_path=None,
        router_threshold=threshold,
        seed=42,
    )
