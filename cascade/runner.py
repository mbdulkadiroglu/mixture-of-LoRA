"""
Cascade experiment runner — the N-round experiment loop.

Per-round flow:
1. Sample queries_per_round queries (without replacement across rounds)
2. For each query: student generates (with logprobs), router decides, optionally teacher generates
3. If training enabled: train on new examples + replay buffer
4. Evaluate on frozen eval set
5. Log everything to SQLite + GPU results JSON
"""

import json as _json
import os
import random
import time
from pathlib import Path

import torch
from loguru import logger

from src.evaluation.sql_cleaning import extract_sql_from_json, extract_sql_from_text

from .config import CascadeConfig
from .evaluator import CascadeEvaluator
from .logger import CascadeLogger
from .prompts import build_query_messages
from .replay_buffer import CascadeReplayBuffer
from .router import CascadeRouter
from .student import CascadeStudent
from .teacher import CascadeTeacher
from .trainer import CascadeTrainer


class CascadeRunner:
    def __init__(self, config: CascadeConfig):
        self.config = config

        self.student: CascadeStudent | None = None
        self.teacher: CascadeTeacher | None = None
        self.router: CascadeRouter | None = None
        self.evaluator: CascadeEvaluator | None = None
        self.trainer: CascadeTrainer | None = None
        self.cascade_logger: CascadeLogger | None = None
        self.replay_buffer: CascadeReplayBuffer | None = None

        # Data
        self._training_pool: list[dict] = []
        self._eval_set: list[dict] = []
        self._used_indices: set[int] = set()

    def _set_random_seeds(self) -> None:
        """Seed python/torch (and numpy if available) for reproducibility."""
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        try:
            import numpy as np

            np.random.seed(self.config.seed)
        except Exception:
            pass

    def run(self) -> str:
        """
        Run the full experiment. Returns path to the SQLite DB.
        """
        self._setup()

        logger.info(
            f"Starting experiment '{self.config.experiment_name}': "
            f"{self.config.num_rounds} rounds, {self.config.queries_per_round} queries/round"
        )

        for round_idx in range(self.config.num_rounds):
            round_start = time.time()
            logger.info(f"\n{'='*60}\nRound {round_idx}\n{'='*60}")

            # Sample queries for this round
            queries = self._sample_queries(round_idx)
            if not queries:
                logger.warning(f"Round {round_idx}: no queries available, stopping")
                break

            # Run the round
            round_metrics = self._run_round(round_idx, queries)

            # Train (if enabled and there are training examples)
            train_metrics = None
            all_examples = round_metrics.get("teacher_examples", []) + round_metrics.get("student_examples", [])
            if self.config.train_after_round and all_examples:
                if round_idx >= self.config.train_start_round:
                    train_metrics = self._train_round(round_idx, all_examples)
                else:
                    # Buffer examples without training
                    for ex in all_examples:
                        self.replay_buffer.add(ex)
                    logger.info(
                        f"Round {round_idx}: buffering {len(all_examples)} examples "
                        f"(training starts at round {self.config.train_start_round})"
                    )

            # Evaluate
            eval_accuracy = None
            if round_idx % self.config.eval_every_n_rounds == 0:
                eval_accuracy = self._run_eval(round_idx)

            # Log adapter version
            if train_metrics:
                self.cascade_logger.log_adapter_version(
                    round_idx=round_idx,
                    adapter_path=train_metrics.get("adapter_path", ""),
                    parent_version=round_idx - 1 if round_idx > 0 else None,
                    new_examples_count=train_metrics.get("new_examples", 0),
                    replay_examples_count=train_metrics.get("replay_examples", 0),
                    teacher_accuracy_on_new=round_metrics.get("teacher_accuracy_on_new"),
                    eval_accuracy=eval_accuracy,
                    eval_samples=self.config.eval_set_size,
                    training_loss=train_metrics.get("train_loss"),
                )

            # GPU results JSON
            round_elapsed = time.time() - round_start
            summary = self.cascade_logger.get_round_summary(round_idx)
            summary["eval_accuracy"] = eval_accuracy
            summary["training_loss"] = train_metrics.get("train_loss") if train_metrics else None
            summary["round_time_seconds"] = round_elapsed
            self.cascade_logger.log_gpu_results(round_idx, summary)

            logger.info(
                f"Round {round_idx} complete: "
                f"final_acc={summary.get('final_accuracy', 0):.3f}, "
                f"cascade_rate={summary.get('cascade_rate', 0):.3f}, "
                f"eval_acc={eval_accuracy}, "
                f"time={round_elapsed:.1f}s"
            )

        # Save replay buffer
        buf_path = Path(self.config.output_dir) / f"exp_{self.config.experiment_name}" / "replay_buffer.json"
        self.replay_buffer.save(buf_path)

        self.cascade_logger.close()
        self.evaluator.close()

        logger.info(f"Experiment complete. DB: {self.cascade_logger.db_path}")
        return str(self.cascade_logger.db_path)

    def _setup(self) -> None:
        """Initialize all modules, load data, create DB."""
        # Set GPU devices
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_devices
        self._set_random_seeds()

        exp_dir = Path(self.config.output_dir) / f"exp_{self.config.experiment_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(exp_dir / "config.yaml")

        # Initialize logger
        self.cascade_logger = CascadeLogger(self.config.experiment_name, exp_dir)

        # Load student
        self.student = CascadeStudent(self.config)
        self.student.load()

        # Load initial adapter if specified
        if self.config.initial_adapter_path:
            self.student.load_adapter(self.config.initial_adapter_path)

        # Ensure consistent model path from round 0 onward when training is enabled.
        if self.config.train_after_round and self.student.inner.peft_model is None:
            self.student.setup_for_training()

        # Load teacher
        self.teacher = CascadeTeacher(self.config)
        self.teacher.load()

        # Load teacher cache if configured
        self._teacher_cache: dict[tuple[str, str], str] = {}
        if self.config.teacher_cache_path:
            self._load_teacher_cache(self.config.teacher_cache_path)

        # Initialize router
        self.router = CascadeRouter(self.config)

        # Initialize evaluator
        self.evaluator = CascadeEvaluator(self.config)
        self.evaluator.load()

        # Initialize trainer
        self.trainer = CascadeTrainer(self.student, self.config)

        # Initialize replay buffer
        self.replay_buffer = CascadeReplayBuffer(self.config.replay_buffer_size)

        # Load dataset
        self._load_data()

        logger.info(
            f"Setup complete: {len(self._training_pool)} training queries, "
            f"{len(self._eval_set)} eval queries"
        )

    def _load_data(self) -> None:
        """Load and split dataset into training pool and eval set."""
        from src.datasets.loader import DatasetLoader

        loader = DatasetLoader()

        if self.config.dataset == "spider":
            domain_ds = loader.load_spider()
        elif self.config.dataset == "bird":
            domain_ds = loader.load_bird()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        # Use train split as training pool, test (dev) split as eval set base
        train_raw = list(domain_ds.train)
        test_raw = list(domain_ds.test)

        # Deterministic shuffle for reproducibility
        rng = random.Random(self.config.seed)

        # Eval set: stratified sample by difficulty to preserve distribution
        self._eval_set = self._stratified_sample(
            test_raw, self.config.eval_set_size, rng
        )

        # Training pool: use train split, cap at training_pool_size
        rng.shuffle(train_raw)
        self._training_pool = train_raw[:self.config.training_pool_size]

    @staticmethod
    def _stratified_sample(
        data: list[dict], n: int, rng: random.Random
    ) -> list[dict]:
        """Sample n items preserving the difficulty distribution.

        Falls back to plain shuffle if no difficulty field is present or
        if n >= len(data).
        """
        if n >= len(data):
            return data

        # Group by difficulty
        groups: dict[str, list[dict]] = {}
        for item in data:
            key = item.get("difficulty", "unknown")
            groups.setdefault(key, []).append(item)

        if len(groups) <= 1:
            # No meaningful stratification — plain shuffle
            shuffled = list(data)
            rng.shuffle(shuffled)
            return shuffled[:n]

        # Proportional allocation per stratum
        total = len(data)
        sampled = []
        remainder = []
        for key, items in sorted(groups.items()):
            rng.shuffle(items)
            k = round(len(items) / total * n)
            sampled.extend(items[:k])
            remainder.extend(items[k:])

        # Adjust for rounding: add or trim to hit exactly n
        if len(sampled) < n:
            rng.shuffle(remainder)
            sampled.extend(remainder[: n - len(sampled)])
        elif len(sampled) > n:
            rng.shuffle(sampled)
            sampled = sampled[:n]

        rng.shuffle(sampled)
        return sampled

    def _sample_queries(self, round_idx: int) -> list[dict]:
        """Sample queries for a round without replacement across rounds."""
        rng = random.Random(self.config.seed + round_idx)

        available = [
            (i, q) for i, q in enumerate(self._training_pool)
            if i not in self._used_indices
        ]

        if not available:
            logger.warning("Training pool exhausted")
            return []

        n = min(self.config.queries_per_round, len(available))
        sampled = rng.sample(available, n)

        for idx, _ in sampled:
            self._used_indices.add(idx)

        return [q for _, q in sampled]

    def _build_messages(self, query: dict) -> list[dict]:
        """Build chat messages for a query (student prompt)."""
        prompt = query.get("prompt", query.get("question", ""))
        return build_query_messages(self.config.dataset, prompt)

    def _build_teacher_messages(self, query: dict) -> list[dict]:
        """Build chat messages for the teacher.

        Uses prompt_variant if set (e.g. "bird_json" for JSON-structured output),
        otherwise falls back to the dataset default.
        """
        prompt = query.get("prompt", query.get("question", ""))
        variant = self.config.prompt_variant or self.config.dataset
        return build_query_messages(variant, prompt)

    def _load_teacher_cache(self, path: str) -> None:
        """Load pre-cached teacher responses from JSON file."""
        cache_path = Path(path)
        if not cache_path.exists():
            logger.warning(f"Teacher cache not found: {path}")
            return

        with open(cache_path) as f:
            data = _json.load(f)

        entries = data.get("per_sample", [])
        for entry in entries:
            db_id = entry.get("db_id", "")
            question = entry.get("question", "")
            model_sql = entry.get("model_sql", "")
            if db_id and question and model_sql:
                self._teacher_cache[(db_id, question)] = model_sql

        logger.info(f"Loaded teacher cache: {len(self._teacher_cache)} entries from {path}")

    def _run_round(self, round_idx: int, queries: list[dict]) -> dict:
        """
        Run inference for one round (two-pass when adaptive cascade rate is active).

        Pass 1: Student generates all responses + logprobs.
        Batch routing: decide all at once using target cascade rate percentile.
        Pass 2: Check correctness, call teacher if routed, build training examples, log.
        """
        # --- Pass 1: Student generates all responses ---
        gen_results = []
        messages_list = []
        for q_idx, query in enumerate(queries):
            messages = self._build_messages(query)
            messages_list.append(messages)
            gen_result = self.student.generate_with_logprobs(messages)
            gen_results.append(gen_result)

            if (q_idx + 1) % 50 == 0:
                logger.info(f"  Round {round_idx}: generated {q_idx + 1}/{len(queries)} student responses")

        # --- Batch routing decision ---
        if self.config.cascade_rate_start > 0:
            decisions = self.router.decide_batch(gen_results, round_idx)
            n_teacher = sum(1 for d in decisions if d.target == "teacher")
            logger.info(
                f"  Round {round_idx}: adaptive routing — "
                f"target_rate={self.config.get_cascade_rate(round_idx):.2f}, "
                f"actual={n_teacher}/{len(queries)} ({n_teacher/len(queries):.1%}) to teacher"
            )
        else:
            decisions = [self.router.decide(gr) for gr in gen_results]

        # --- Pass 2: Evaluate, teacher calls, logging ---
        teacher_examples = []
        student_examples = []
        teacher_correct_count = 0
        teacher_total_count = 0
        cache_hits = 0
        cache_misses = 0

        for q_idx, (query, gen_result, decision) in enumerate(zip(queries, gen_results, decisions)):
            gold_sql = query.get("query") or query.get("SQL", "")
            db_id = query["db_id"]

            # Check student correctness
            student_correct = self.evaluator.check_single(gen_result.text, gold_sql, db_id)

            # Teacher generates if routed
            teacher_sql = None
            teacher_correct = None
            final_correct = student_correct

            if decision.target == "teacher":
                # Try teacher cache first
                cache_key = (db_id, query.get("question", ""))
                cached_sql = self._teacher_cache.get(cache_key)
                if cached_sql is not None:
                    teacher_sql = cached_sql
                    cache_hits += 1
                else:
                    # Live teacher call
                    teacher_messages = self._build_teacher_messages(query)
                    teacher_response = self.teacher.generate(teacher_messages)

                    # Extract SQL from teacher response (may be JSON-structured)
                    teacher_variant = self.config.prompt_variant or self.config.dataset
                    if teacher_variant == "bird_json":
                        teacher_sql, _ = extract_sql_from_json(teacher_response)
                    else:
                        teacher_sql = extract_sql_from_text(teacher_response)
                    cache_misses += 1

                teacher_correct = self.evaluator.check_single(teacher_sql, gold_sql, db_id)
                final_correct = teacher_correct
                teacher_total_count += 1
                if teacher_correct:
                    teacher_correct_count += 1

                # Determine training target based on config
                if self.config.training_source == "gold":
                    training_target = gold_sql
                    training_correct = True  # Gold is always "correct"
                else:
                    # Student trains on raw SQL, not JSON
                    training_target = teacher_sql
                    training_correct = teacher_correct

                # Build training example
                teacher_examples.append({
                    "prompt": query.get("prompt", query.get("question", "")),
                    "response": training_target,
                    "db_id": db_id,
                    "source_round": round_idx,
                    "was_correct": training_correct,
                })

                # Log training example
                self.cascade_logger.log_training_example(
                    round_idx=round_idx,
                    source_round=round_idx,
                    is_replay=False,
                    prompt=query.get("prompt", "")[:2000],
                    target_sql=training_target,
                    db_id=db_id,
                    source=self.config.training_source,
                    was_correct=training_correct,
                )

            elif self.config.train_student_responses:
                # Self-reinforcement: train on student's own non-cascaded responses
                student_examples.append({
                    "prompt": query.get("prompt", query.get("question", "")),
                    "response": gen_result.text,
                    "db_id": db_id,
                    "source_round": round_idx,
                    "was_correct": student_correct,
                })

                self.cascade_logger.log_training_example(
                    round_idx=round_idx,
                    source_round=round_idx,
                    is_replay=False,
                    prompt=query.get("prompt", "")[:2000],
                    target_sql=gen_result.text,
                    db_id=db_id,
                    source="student",
                    was_correct=student_correct,
                )

            # Log interaction
            self.cascade_logger.log_interaction(
                round_idx=round_idx,
                query_idx=q_idx,
                prompt=query.get("prompt", "")[:2000],
                db_id=db_id,
                gold_sql=gold_sql,
                student_sql=gen_result.text,
                student_mean_logprob=gen_result.mean_log_prob,
                student_min_logprob=gen_result.min_log_prob,
                student_mean_entropy=gen_result.mean_entropy,
                student_num_tokens=gen_result.num_tokens,
                routed_to=decision.target,
                router_threshold=decision.threshold,
                teacher_sql=teacher_sql,
                student_correct=student_correct,
                teacher_correct=teacher_correct,
                final_correct=final_correct,
            )

            if (q_idx + 1) % 50 == 0:
                logger.info(f"  Round {round_idx}: processed {q_idx + 1}/{len(queries)} queries (pass 2)")

        teacher_acc_on_new = (
            teacher_correct_count / teacher_total_count if teacher_total_count > 0 else None
        )

        if student_examples:
            logger.info(
                f"  Round {round_idx}: training examples — "
                f"{len(teacher_examples)} teacher + {len(student_examples)} student self-reinforcement"
            )

        if cache_hits > 0 or cache_misses > 0:
            logger.info(
                f"  Round {round_idx}: teacher cache — "
                f"{cache_hits} hits, {cache_misses} misses"
            )

        return {
            "teacher_examples": teacher_examples,
            "student_examples": student_examples,
            "teacher_accuracy_on_new": teacher_acc_on_new,
        }

    def _train_round(self, round_idx: int, new_examples: list[dict]) -> dict | None:
        """Train on new examples + replay buffer samples."""
        if not new_examples:
            return None

        # Sample replay from prior rounds only (before inserting current round).
        num_replay = int(len(new_examples) * self.config.replay_ratio)
        replay_examples = self.replay_buffer.sample(num_replay) if num_replay > 0 else []

        # Add new examples after replay sampling.
        for ex in new_examples:
            self.replay_buffer.add(ex)

        # Log replay examples
        for ex in replay_examples:
            self.cascade_logger.log_training_example(
                round_idx=round_idx,
                source_round=ex.get("source_round", 0),
                is_replay=True,
                prompt=ex.get("prompt", "")[:2000],
                target_sql=ex.get("response", ""),
                db_id=ex.get("db_id", ""),
                source=self.config.training_source,
                was_correct=ex.get("was_correct", False),
            )

        # Train
        metrics = self.trainer.train_round(new_examples, replay_examples, round_idx)

        # Keep model in eval mode after training; do not call
        # FastLanguageModel.for_inference() here since we intentionally bypass
        # Unsloth fast-generate in cascade/student.py.
        model = self.student.inner.peft_model or self.student.inner.model
        model.eval()

        torch.cuda.empty_cache()

        return metrics

    def _run_eval(self, round_idx: int) -> float:
        """Evaluate student on the frozen eval set."""
        logger.info(f"Round {round_idx}: evaluating on {len(self._eval_set)} eval queries")
        return self.evaluator.evaluate_set(self.student, self._eval_set)
