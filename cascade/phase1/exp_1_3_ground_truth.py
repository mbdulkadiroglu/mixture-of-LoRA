"""
Experiment 1.3 — Ground Truth Upper Bound

Train on gold SQL instead of teacher responses, 10 rounds. Establishes
the upper bound: what happens when the training signal is perfectly clean?
If 1.1 underperforms 1.3, the gap is attributable to teacher noise.
"""

from cascade.config import CascadeConfig


def get_config(threshold: float = -4.0) -> CascadeConfig:
    return CascadeConfig(
        experiment_name="1.3_ground_truth_training",
        dataset="spider",
        num_rounds=10,
        queries_per_round=200,
        training_pool_size=2000,
        eval_set_size=200,
        train_after_round=True,
        training_source="gold",  # Train on gold SQL
        initial_adapter_path=None,
        router_threshold=threshold,
        seed=42,
    )
