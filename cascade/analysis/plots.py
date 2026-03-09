"""
Phase 1 analysis plots.

5 key plots:
1. Accuracy trajectory (round vs eval accuracy)
2. Cascade rate trajectory (round vs fraction routed to teacher)
3. Teacher accuracy on deferred queries
4. Training signal quality (fraction of correct teacher examples)
5. Confidence distribution shift (overlaid histograms per round)

All read from SQLite via CascadeLogger.query(). Save as PNG + source data JSON.
"""

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def _load_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _get_experiment_name(db_path: str | Path) -> str:
    return Path(db_path).stem


def _query(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def plot_accuracy_trajectory(db_paths: list[str | Path], output_dir: str | Path) -> None:
    """Plot 1: round vs eval accuracy for each experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    all_data = {}

    for db_path in db_paths:
        conn = _load_db(db_path)
        name = _get_experiment_name(db_path)
        rows = _query(conn, "SELECT round, eval_accuracy FROM adapter_versions ORDER BY round")
        conn.close()

        if not rows:
            # Fall back to GPU JSON
            gpu_json = Path(db_path).with_suffix("").parent / f"{name}_gpu_results.json"
            if gpu_json.exists():
                with open(gpu_json) as f:
                    gpu_data = json.load(f)
                rows = [{"round": e["round"], "eval_accuracy": e["metrics"].get("eval_accuracy")}
                        for e in gpu_data if e["metrics"].get("eval_accuracy") is not None]

        rounds = [r["round"] for r in rows if r["eval_accuracy"] is not None]
        accs = [r["eval_accuracy"] for r in rows if r["eval_accuracy"] is not None]

        if rounds:
            ax.plot(rounds, accs, marker="o", label=name)
            all_data[name] = {"rounds": rounds, "accuracies": accs}

    ax.set_xlabel("Round")
    ax.set_ylabel("Eval Accuracy")
    ax.set_title("Accuracy Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "accuracy_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(output_dir / "accuracy_trajectory.json", "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info(f"Plot saved: {output_dir / 'accuracy_trajectory.png'}")


def plot_cascade_rate(db_paths: list[str | Path], output_dir: str | Path) -> None:
    """Plot 2: round vs fraction routed to teacher."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    all_data = {}

    for db_path in db_paths:
        conn = _load_db(db_path)
        name = _get_experiment_name(db_path)
        rows = _query(conn, """
            SELECT round,
                   CAST(SUM(CASE WHEN routed_to='teacher' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as cascade_rate
            FROM interactions
            GROUP BY round ORDER BY round
        """)
        conn.close()

        rounds = [r["round"] for r in rows]
        rates = [r["cascade_rate"] for r in rows]

        if rounds:
            ax.plot(rounds, rates, marker="s", label=name)
            all_data[name] = {"rounds": rounds, "cascade_rates": rates}

    ax.set_xlabel("Round")
    ax.set_ylabel("Cascade Rate (fraction to teacher)")
    ax.set_title("Cascade Rate Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "cascade_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(output_dir / "cascade_rate.json", "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info(f"Plot saved: {output_dir / 'cascade_rate.png'}")


def plot_teacher_accuracy(db_paths: list[str | Path], output_dir: str | Path) -> None:
    """Plot 3: teacher exec accuracy on cascaded queries only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    all_data = {}

    for db_path in db_paths:
        conn = _load_db(db_path)
        name = _get_experiment_name(db_path)
        rows = _query(conn, """
            SELECT round,
                   CAST(SUM(teacher_correct) AS REAL) / COUNT(*) as teacher_acc
            FROM interactions
            WHERE routed_to = 'teacher'
            GROUP BY round ORDER BY round
        """)
        conn.close()

        rounds = [r["round"] for r in rows]
        accs = [r["teacher_acc"] for r in rows]

        if rounds:
            ax.plot(rounds, accs, marker="^", label=name)
            all_data[name] = {"rounds": rounds, "teacher_accuracies": accs}

    ax.set_xlabel("Round")
    ax.set_ylabel("Teacher Accuracy (on cascaded queries)")
    ax.set_title("Teacher Accuracy on Deferred Queries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "teacher_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(output_dir / "teacher_accuracy.json", "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info(f"Plot saved: {output_dir / 'teacher_accuracy.png'}")


def plot_training_signal_quality(db_paths: list[str | Path], output_dir: str | Path) -> None:
    """Plot 4: fraction of teacher examples that are correct per round."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    all_data = {}

    for db_path in db_paths:
        conn = _load_db(db_path)
        name = _get_experiment_name(db_path)
        rows = _query(conn, """
            SELECT round,
                   CAST(SUM(was_correct) AS REAL) / COUNT(*) as correct_fraction
            FROM training_examples
            WHERE is_replay = 0
            GROUP BY round ORDER BY round
        """)
        conn.close()

        rounds = [r["round"] for r in rows]
        fracs = [r["correct_fraction"] for r in rows]

        if rounds:
            ax.plot(rounds, fracs, marker="D", label=name)
            all_data[name] = {"rounds": rounds, "correct_fractions": fracs}

    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction Correct")
    ax.set_title("Training Signal Quality (new examples only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "training_signal_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(output_dir / "training_signal_quality.json", "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info(f"Plot saved: {output_dir / 'training_signal_quality.png'}")


def plot_confidence_distribution(db_paths: list[str | Path], output_dir: str | Path) -> None:
    """Plot 5: overlaid histograms of mean_log_prob per round (first experiment only)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_paths:
        return

    # Use first experiment for distribution shift visualization
    db_path = db_paths[0]
    conn = _load_db(db_path)
    name = _get_experiment_name(db_path)

    rounds = _query(conn, "SELECT DISTINCT round FROM interactions ORDER BY round")
    round_nums = [r["round"] for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 6))
    all_data = {}

    # Use a colormap to show progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(round_nums)))

    for i, rnd in enumerate(round_nums):
        rows = _query(conn, "SELECT student_mean_logprob FROM interactions WHERE round = ?", (rnd,))
        values = [r["student_mean_logprob"] for r in rows if r["student_mean_logprob"] is not None]

        if values:
            ax.hist(values, bins=30, alpha=0.4, color=colors[i], label=f"Round {rnd}")
            all_data[f"round_{rnd}"] = values

    conn.close()

    ax.set_xlabel("Mean Token Log-Prob")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution Shift ({name})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(output_dir / "confidence_distribution.json", "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info(f"Plot saved: {output_dir / 'confidence_distribution.png'}")


def generate_all_plots(db_paths: list[str | Path], output_dir: str | Path) -> None:
    """Generate all 5 Phase 1 plots."""
    plot_accuracy_trajectory(db_paths, output_dir)
    plot_cascade_rate(db_paths, output_dir)
    plot_teacher_accuracy(db_paths, output_dir)
    plot_training_signal_quality(db_paths, output_dir)
    plot_confidence_distribution(db_paths, output_dir)
    logger.info(f"All plots generated in {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Phase 1 analysis plots")
    parser.add_argument("--experiments", nargs="+", help="Paths to experiment directories or DB files")
    parser.add_argument("--output", default="results/cascade/plots")
    args = parser.parse_args()

    if not args.experiments:
        parser.error("--experiments is required")

    # Find .db files
    db_paths = []
    for p in args.experiments:
        p = Path(p)
        if p.suffix == ".db":
            db_paths.append(p)
        elif p.is_dir():
            db_paths.extend(sorted(p.glob("*.db")))
        else:
            # Try glob
            db_paths.extend(sorted(Path(".").glob(str(p) + "/**/*.db")))

    if not db_paths:
        logger.error("No experiment databases found")
        return

    logger.info(f"Found {len(db_paths)} experiment DBs: {[p.name for p in db_paths]}")
    generate_all_plots(db_paths, args.output)


if __name__ == "__main__":
    main()
