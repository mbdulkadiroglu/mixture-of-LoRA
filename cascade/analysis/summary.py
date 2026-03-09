"""
Per-round summary statistics from experiment SQLite databases.

Extracts key metrics per round and writes to JSON for easy consumption.
"""

import json
import sqlite3
from pathlib import Path

from loguru import logger


def summarize_experiment(db_path: str | Path) -> dict:
    """
    Extract per-round summary statistics from an experiment DB.

    Returns a dict with experiment metadata and per-round metrics.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    name = db_path.stem

    # Get all rounds
    rounds = [r[0] for r in conn.execute("SELECT DISTINCT round FROM interactions ORDER BY round").fetchall()]

    round_summaries = []
    for rnd in rounds:
        # Interaction stats
        stats = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(student_correct) as student_correct,
                SUM(final_correct) as final_correct,
                SUM(CASE WHEN routed_to='teacher' THEN 1 ELSE 0 END) as teacher_routed,
                AVG(student_mean_logprob) as avg_logprob,
                AVG(student_mean_entropy) as avg_entropy
            FROM interactions WHERE round = ?
        """, (rnd,)).fetchone()

        total = stats["total"]
        teacher_routed = stats["teacher_routed"]

        # Teacher accuracy on cascaded
        teacher_acc = None
        if teacher_routed > 0:
            ta = conn.execute("""
                SELECT CAST(SUM(teacher_correct) AS REAL) / COUNT(*) as acc
                FROM interactions WHERE round = ? AND routed_to = 'teacher'
            """, (rnd,)).fetchone()
            teacher_acc = ta["acc"]

        # Training stats
        train_stats = conn.execute("""
            SELECT
                COUNT(*) as total_examples,
                SUM(CASE WHEN is_replay = 0 THEN 1 ELSE 0 END) as new_examples,
                SUM(CASE WHEN is_replay = 1 THEN 1 ELSE 0 END) as replay_examples,
                AVG(CASE WHEN is_replay = 0 THEN was_correct END) as new_correct_rate
            FROM training_examples WHERE round = ?
        """, (rnd,)).fetchone()

        # Adapter stats
        adapter = conn.execute(
            "SELECT eval_accuracy, training_loss FROM adapter_versions WHERE round = ?",
            (rnd,),
        ).fetchone()

        summary = {
            "round": rnd,
            "total_queries": total,
            "student_accuracy": stats["student_correct"] / total if total else 0,
            "final_accuracy": stats["final_correct"] / total if total else 0,
            "cascade_rate": teacher_routed / total if total else 0,
            "avg_logprob": stats["avg_logprob"],
            "avg_entropy": stats["avg_entropy"],
            "teacher_accuracy_on_cascaded": teacher_acc,
            "new_training_examples": train_stats["new_examples"],
            "replay_training_examples": train_stats["replay_examples"],
            "training_signal_quality": train_stats["new_correct_rate"],
            "eval_accuracy": adapter["eval_accuracy"] if adapter else None,
            "training_loss": adapter["training_loss"] if adapter else None,
        }
        round_summaries.append(summary)

    conn.close()

    return {
        "experiment_name": name,
        "db_path": str(db_path),
        "num_rounds": len(rounds),
        "rounds": round_summaries,
    }


def summarize_to_json(db_path: str | Path, output_path: str | Path | None = None) -> str:
    """Summarize an experiment and write to JSON."""
    summary = summarize_experiment(db_path)

    if output_path is None:
        output_path = Path(db_path).with_suffix(".summary.json")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to {output_path}")
    return str(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment summaries")
    parser.add_argument("db_paths", nargs="+", help="Paths to experiment .db files")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    for db_path in args.db_paths:
        out = None
        if args.output_dir:
            out = Path(args.output_dir) / f"{Path(db_path).stem}.summary.json"
        summarize_to_json(db_path, out)


if __name__ == "__main__":
    main()
