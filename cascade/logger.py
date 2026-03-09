"""
Structured logging for cascade experiments.

One SQLite database per experiment with three tables:
- interactions: per-query per-round data
- training_examples: per-training-example per-round data
- adapter_versions: per-trained-adapter metadata

Also writes GPU results to JSON after each round (per CLAUDE.md requirement).
"""

import json
import sqlite3
import time
from pathlib import Path

import torch
from loguru import logger


class CascadeLogger:
    def __init__(self, experiment_name: str, output_dir: str | Path):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.output_dir / f"{experiment_name}.db"
        self.gpu_json_path = self.output_dir / f"{experiment_name}_gpu_results.json"

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Accumulate GPU results across rounds
        self._gpu_results: list[dict] = []
        if self.gpu_json_path.exists():
            with open(self.gpu_json_path) as f:
                self._gpu_results = json.load(f)

        logger.info(f"CascadeLogger: {self.db_path}")

    def _create_tables(self) -> None:
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round INTEGER NOT NULL,
                query_idx INTEGER NOT NULL,
                prompt TEXT,
                db_id TEXT,
                gold_sql TEXT,
                student_sql TEXT,
                student_mean_logprob REAL,
                student_min_logprob REAL,
                student_mean_entropy REAL,
                student_num_tokens INTEGER,
                routed_to TEXT,
                router_threshold REAL,
                teacher_sql TEXT,
                student_correct INTEGER,
                teacher_correct INTEGER,
                final_correct INTEGER,
                timestamp REAL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round INTEGER NOT NULL,
                source_round INTEGER,
                is_replay INTEGER DEFAULT 0,
                prompt TEXT,
                target_sql TEXT,
                db_id TEXT,
                source TEXT,
                was_correct INTEGER,
                quality_weight REAL DEFAULT 1.0,
                timestamp REAL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS adapter_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round INTEGER NOT NULL,
                adapter_path TEXT,
                parent_version INTEGER,
                new_examples_count INTEGER,
                replay_examples_count INTEGER,
                teacher_accuracy_on_new REAL,
                eval_accuracy REAL,
                eval_samples INTEGER,
                training_loss REAL,
                timestamp REAL
            )
        """)

        self._conn.commit()

    def log_interaction(
        self,
        round_idx: int,
        query_idx: int,
        prompt: str,
        db_id: str,
        gold_sql: str,
        student_sql: str,
        student_mean_logprob: float,
        student_min_logprob: float,
        student_mean_entropy: float,
        student_num_tokens: int,
        routed_to: str,
        router_threshold: float,
        teacher_sql: str | None,
        student_correct: bool,
        teacher_correct: bool | None,
        final_correct: bool,
    ) -> None:
        self._conn.execute(
            """INSERT INTO interactions (
                round, query_idx, prompt, db_id, gold_sql,
                student_sql, student_mean_logprob, student_min_logprob,
                student_mean_entropy, student_num_tokens,
                routed_to, router_threshold, teacher_sql,
                student_correct, teacher_correct, final_correct, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                round_idx, query_idx, prompt, db_id, gold_sql,
                student_sql, student_mean_logprob, student_min_logprob,
                student_mean_entropy, student_num_tokens,
                routed_to, router_threshold, teacher_sql,
                int(student_correct), int(teacher_correct) if teacher_correct is not None else None,
                int(final_correct), time.time(),
            ),
        )
        self._conn.commit()

    def log_training_example(
        self,
        round_idx: int,
        source_round: int,
        is_replay: bool,
        prompt: str,
        target_sql: str,
        db_id: str,
        source: str,
        was_correct: bool,
        quality_weight: float = 1.0,
    ) -> None:
        self._conn.execute(
            """INSERT INTO training_examples (
                round, source_round, is_replay, prompt, target_sql,
                db_id, source, was_correct, quality_weight, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                round_idx, source_round, int(is_replay), prompt, target_sql,
                db_id, source, int(was_correct), quality_weight, time.time(),
            ),
        )
        self._conn.commit()

    def log_adapter_version(
        self,
        round_idx: int,
        adapter_path: str,
        parent_version: int | None,
        new_examples_count: int,
        replay_examples_count: int,
        teacher_accuracy_on_new: float | None,
        eval_accuracy: float | None,
        eval_samples: int | None,
        training_loss: float | None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO adapter_versions (
                round, adapter_path, parent_version,
                new_examples_count, replay_examples_count,
                teacher_accuracy_on_new, eval_accuracy, eval_samples,
                training_loss, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                round_idx, adapter_path, parent_version,
                new_examples_count, replay_examples_count,
                teacher_accuracy_on_new, eval_accuracy, eval_samples,
                training_loss, time.time(),
            ),
        )
        self._conn.commit()

    def log_gpu_results(self, round_idx: int, metrics: dict) -> None:
        """Log GPU memory and round metrics to JSON file."""
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info[f"gpu_{i}"] = {
                    "name": props.name,
                    "total_gb": round(props.total_memory / 1e9, 2),
                    "allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 2),
                }

        entry = {
            "round": round_idx,
            "timestamp": time.time(),
            "gpu": gpu_info,
            "metrics": metrics,
        }
        self._gpu_results.append(entry)

        with open(self.gpu_json_path, "w") as f:
            json.dump(self._gpu_results, f, indent=2, default=str)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a query against the experiment DB and return rows as dicts."""
        cur = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_round_summary(self, round_idx: int) -> dict:
        """Get summary statistics for a round."""
        rows = self.query(
            """SELECT
                COUNT(*) as total,
                SUM(student_correct) as student_correct,
                SUM(final_correct) as final_correct,
                SUM(CASE WHEN routed_to='teacher' THEN 1 ELSE 0 END) as teacher_count,
                AVG(student_mean_logprob) as avg_logprob,
                SUM(CASE WHEN routed_to='teacher' AND teacher_correct=1 THEN 1 ELSE 0 END) as teacher_correct_count
            FROM interactions WHERE round = ?""",
            (round_idx,),
        )
        if not rows:
            return {}
        r = rows[0]
        total = r["total"] or 0
        teacher_count = r["teacher_count"] or 0
        return {
            "round": round_idx,
            "total_queries": total,
            "student_accuracy": (r["student_correct"] or 0) / total if total else 0,
            "final_accuracy": (r["final_correct"] or 0) / total if total else 0,
            "cascade_rate": teacher_count / total if total else 0,
            "avg_logprob": r["avg_logprob"],
            "teacher_accuracy_on_cascaded": (
                (r["teacher_correct_count"] or 0) / teacher_count if teacher_count else None
            ),
        }

    def close(self) -> None:
        self._conn.close()
