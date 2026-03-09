"""
Multi-experiment comparison: overlay trajectories on the same axes.

Used after all Phase 1 experiments complete to compare baseline vs
static vs ground truth in a single figure.
"""

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger


def _load_round_data(db_path: str | Path) -> dict:
    """Load per-round data from an experiment DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            round,
            CAST(SUM(student_correct) AS REAL) / COUNT(*) as student_acc,
            CAST(SUM(final_correct) AS REAL) / COUNT(*) as final_acc,
            CAST(SUM(CASE WHEN routed_to='teacher' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as cascade_rate,
            AVG(student_mean_logprob) as avg_logprob
        FROM interactions GROUP BY round ORDER BY round
    """).fetchall()

    adapter_rows = conn.execute(
        "SELECT round, eval_accuracy, training_loss FROM adapter_versions ORDER BY round"
    ).fetchall()
    conn.close()

    eval_map = {r["round"]: r["eval_accuracy"] for r in adapter_rows if r["eval_accuracy"] is not None}

    return {
        "rounds": [r["round"] for r in rows],
        "student_acc": [r["student_acc"] for r in rows],
        "final_acc": [r["final_acc"] for r in rows],
        "cascade_rate": [r["cascade_rate"] for r in rows],
        "avg_logprob": [r["avg_logprob"] for r in rows],
        "eval_acc": [eval_map.get(r["round"]) for r in rows],
    }


def compare_experiments(
    db_paths: list[str | Path],
    labels: list[str] | None = None,
    output_dir: str | Path = "cascade/results/comparison",
) -> None:
    """Generate multi-experiment comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [Path(p).stem for p in db_paths]

    experiments = {}
    for db_path, label in zip(db_paths, labels):
        experiments[label] = _load_round_data(db_path)

    # Combined comparison figure (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Eval accuracy
    ax = axes[0, 0]
    for label, data in experiments.items():
        eval_rounds = [r for r, a in zip(data["rounds"], data["eval_acc"]) if a is not None]
        eval_accs = [a for a in data["eval_acc"] if a is not None]
        if eval_rounds:
            ax.plot(eval_rounds, eval_accs, marker="o", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Eval Accuracy")
    ax.set_title("Eval Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cascade rate
    ax = axes[0, 1]
    for label, data in experiments.items():
        ax.plot(data["rounds"], data["cascade_rate"], marker="s", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cascade Rate")
    ax.set_title("Cascade Rate (fraction to teacher)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Student accuracy (on training queries)
    ax = axes[1, 0]
    for label, data in experiments.items():
        ax.plot(data["rounds"], data["student_acc"], marker="^", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Student Accuracy")
    ax.set_title("Student Accuracy (on training queries)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Average log-prob (confidence shift)
    ax = axes[1, 1]
    for label, data in experiments.items():
        ax.plot(data["rounds"], data["avg_logprob"], marker="D", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg Mean Log-Prob")
    ax.set_title("Confidence Level (avg log-prob)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 1 Experiment Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(output_dir / "phase1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save data
    with open(output_dir / "phase1_comparison.json", "w") as f:
        json.dump(experiments, f, indent=2, default=str)

    logger.info(f"Comparison plots saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple cascade experiments")
    parser.add_argument("db_paths", nargs="+", help="Paths to experiment .db files")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", default="cascade/results/comparison")
    args = parser.parse_args()

    compare_experiments(args.db_paths, args.labels, args.output)


if __name__ == "__main__":
    main()
