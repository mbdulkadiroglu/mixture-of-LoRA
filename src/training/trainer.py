"""
LoRA trainer for fine-tuning the student model.
"""

from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from trl import SFTConfig, SFTTrainer

from ..config import TrainingConfig, LoRAConfig
from ..models.student import StudentModel
from ..utils import ExperienceReplayBuffer, get_timestamp


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning of the student model.

    Supports both batch training and online/continual learning.
    """

    def __init__(
        self,
        student: StudentModel,
        training_config: TrainingConfig,
        lora_config: LoRAConfig,
        output_dir: str | Path,
        replay_buffer: ExperienceReplayBuffer | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            student: Student model to train.
            training_config: Training configuration.
            lora_config: LoRA configuration.
            output_dir: Directory for outputs and checkpoints.
            replay_buffer: Optional replay buffer for continual learning.
        """
        self.student = student
        self.training_config = training_config
        self.lora_config = lora_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.replay_buffer = replay_buffer or ExperienceReplayBuffer(
            training_config.replay_buffer_size
        )

        self._trainer = None
        self._training_step = 0

        logger.info(f"LoRATrainer initialized. Output dir: {self.output_dir}")

    def prepare_for_training(self, domain: str) -> None:
        """
        Prepare the model for training a specific domain.

        Args:
            domain: Domain to train for.
        """
        if not self.student.is_loaded:
            self.student.load_model()

        # Setup LoRA if not already done
        if self.student.peft_model is None:
            self.student.setup_lora()

        logger.info(f"Model prepared for training domain: {domain}")

    def _tokenize_no_special(self, text: str) -> list[int]:
        """Tokenize text without injecting tokenizer-added special tokens."""
        try:
            tokenized = self.student.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )
        except TypeError:
            tokenized = self.student.tokenizer(text, truncation=False)

        input_ids = tokenized.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return input_ids

    def _ensure_tokenized_completion_dataset(self, dataset: Dataset) -> Dataset:
        """
        Ensure dataset contains explicit completion masks for completion-only loss.

        Preferred raw format is prompt/completion; this method converts it into
        pretokenized rows with ``input_ids`` + ``completion_mask`` so the trainer
        only backpropagates on assistant/completion tokens.
        """
        column_names = set(dataset.column_names)

        if "prompt" in column_names and "completion" in column_names:
            rows = []
            skipped = 0
            max_seq_len = self.student.config.max_seq_length

            for ex in dataset:
                prompt = ex.get("prompt", "")
                completion = ex.get("completion", "")

                if not completion:
                    skipped += 1
                    continue

                prompt_ids = self._tokenize_no_special(prompt)
                completion_ids = self._tokenize_no_special(completion)
                if len(completion_ids) == 0:
                    skipped += 1
                    continue

                input_ids = prompt_ids + completion_ids
                if len(input_ids) > max_seq_len:
                    skipped += 1
                    continue

                completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

                row = dict(ex)
                row["input_ids"] = input_ids
                row["attention_mask"] = [1] * len(input_ids)
                row["completion_mask"] = completion_mask
                rows.append(row)

            if not rows:
                raise ValueError(
                    "No valid prompt/completion rows available after tokenization."
                )

            if skipped > 0:
                logger.warning(
                    f"Skipped {skipped} examples while building completion-only dataset."
                )

            return Dataset.from_list(rows)

        if {"input_ids", "completion_mask"}.issubset(column_names):
            return dataset

        raise ValueError(
            "Training dataset must contain either prompt/completion columns "
            "or pretokenized input_ids/completion_mask columns."
        )

    def train(
        self,
        dataset: Dataset,
        domain: str,
        num_epochs: int | None = None,
        max_steps: int | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> dict:
        """
        Train the model on a dataset.

        Args:
            dataset: Training dataset with prompt/completion or tokenized fields.
            domain: Domain being trained.
            num_epochs: Number of epochs (overrides config).
            max_steps: Maximum steps (overrides config).
            resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            Training metrics dictionary.
        """
        self.prepare_for_training(domain)

        # Mix with replay buffer for continual learning
        if len(self.replay_buffer) > 0 and self.training_config.replay_ratio > 0:
            dataset = self._mix_with_replay(dataset, domain)

        # Build pretokenized completion masks so loss is computed only on completion tokens.
        dataset = self._ensure_tokenized_completion_dataset(dataset)

        # Create output directory for this training run
        run_dir = self.output_dir / f"{domain}_{get_timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Configure training arguments
        training_args = SFTConfig(
            output_dir=str(run_dir),
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            num_train_epochs=num_epochs or self.training_config.num_epochs,
            max_steps=max_steps if max_steps else self.training_config.max_steps,
            optim=self.training_config.optimizer,
            weight_decay=self.training_config.weight_decay,
            max_grad_norm=self.training_config.max_grad_norm,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            max_seq_length=self.student.config.max_seq_length,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            seed=42,
            report_to="none",  # Disable wandb by default
        )

        # Get trainable model
        model = self.student.get_trainable_model()

        # Create trainer
        self._trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=self.student.tokenizer,
            args=training_args,
        )

        logger.info(
            f"Starting training: {len(dataset)} samples, "
            f"{training_args.num_train_epochs} epochs"
        )

        # Train
        train_result = self._trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        # Get metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0
            ),
            "train_steps": train_result.global_step,
            "domain": domain,
        }

        self._training_step += train_result.global_step

        logger.info(f"Training completed. Loss: {metrics['train_loss']:.4f}")

        return metrics

    def train_online(
        self,
        examples: list[dict],
        domain: str,
        min_batch_size: int = 4,
    ) -> dict | None:
        """
        Online training with accumulated examples.

        Trains when enough examples are accumulated.

        Args:
            examples: List of training examples with 'text' key.
            domain: Domain being trained.
            min_batch_size: Minimum examples before training.

        Returns:
            Training metrics if training occurred, None otherwise.
        """
        # Add to replay buffer
        for ex in examples:
            self.replay_buffer.add({
                "prompt": ex.get("prompt", ""),
                "completion": ex.get("completion", ""),
                "domain": domain,
            })

        # Check if we have enough examples
        domain_examples = [
            ex for ex in self.replay_buffer.buffer if ex.get("domain") == domain
        ]

        if len(domain_examples) < min_batch_size:
            logger.debug(
                f"Accumulated {len(domain_examples)}/{min_batch_size} examples for {domain}"
            )
            return None

        # Create dataset from accumulated examples
        dataset = Dataset.from_list(domain_examples)

        # Train
        metrics = self.train(
            dataset,
            domain,
            num_epochs=1,  # Single epoch for online learning
        )

        return metrics

    def _mix_with_replay(self, dataset: Dataset, domain: str) -> Dataset:
        """
        Mix new data with replay buffer samples.

        Args:
            dataset: New training dataset.
            domain: Current domain.

        Returns:
            Mixed dataset.
        """
        # Calculate number of replay samples
        num_new = len(dataset)
        num_replay = int(num_new * self.training_config.replay_ratio)

        if num_replay == 0:
            return dataset

        # Sample from replay buffer (prioritize other domains for diversity)
        replay_samples = self.replay_buffer.sample(num_replay)

        if not replay_samples:
            return dataset

        # Combine datasets
        combined_data = list(dataset) + replay_samples
        logger.info(
            f"Mixed {num_new} new samples with {len(replay_samples)} replay samples"
        )

        return Dataset.from_list(combined_data)

    def save_adapter(self, path: str | Path) -> None:
        """
        Save the trained adapter.

        Args:
            path: Path to save the adapter.
        """
        self.student.save_adapter(path)
        logger.info(f"Adapter saved to: {path}")

    def get_training_state(self) -> dict:
        """
        Get current training state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "training_step": self._training_step,
            "replay_buffer_size": len(self.replay_buffer),
        }

    def save_replay_buffer(self, path: str | Path) -> None:
        """Save replay buffer to disk."""
        self.replay_buffer.save(path)
        logger.info(f"Replay buffer saved to: {path}")

    def load_replay_buffer(self, path: str | Path) -> None:
        """Load replay buffer from disk."""
        self.replay_buffer.load(path)
        logger.info(f"Replay buffer loaded: {len(self.replay_buffer)} examples")


class OnlineTrainingManager:
    """
    Manages online training across multiple domains.

    Accumulates training examples and triggers training when appropriate.
    """

    def __init__(
        self,
        trainer: LoRATrainer,
        domains: list[str],
        batch_threshold: int = 16,
        time_threshold_seconds: int = 3600,
    ):
        """
        Initialize the online training manager.

        Args:
            trainer: LoRA trainer instance.
            domains: List of domains to manage.
            batch_threshold: Number of examples before training.
            time_threshold_seconds: Time before forced training.
        """
        self.trainer = trainer
        self.domains = domains
        self.batch_threshold = batch_threshold
        self.time_threshold = time_threshold_seconds

        # Pending examples per domain
        self._pending: dict[str, list[dict]] = {d: [] for d in domains}

        # Last training time per domain
        self._last_train: dict[str, float] = {}

        logger.info(
            f"OnlineTrainingManager initialized for domains: {domains}"
        )

    def add_example(self, example: dict, domain: str) -> dict | None:
        """
        Add a training example and potentially trigger training.

        Args:
            example: Training example with 'text' key.
            domain: Domain of the example.

        Returns:
            Training metrics if training was triggered.
        """
        import time

        if domain not in self._pending:
            self._pending[domain] = []

        self._pending[domain].append(example)

        # Check if training should be triggered
        should_train = False
        reason = ""

        if len(self._pending[domain]) >= self.batch_threshold:
            should_train = True
            reason = f"Batch threshold reached ({len(self._pending[domain])})"

        elif domain in self._last_train:
            elapsed = time.time() - self._last_train[domain]
            if elapsed >= self.time_threshold and len(self._pending[domain]) > 0:
                should_train = True
                reason = f"Time threshold reached ({elapsed:.0f}s)"

        if should_train:
            logger.info(f"Triggering training for {domain}: {reason}")

            metrics = self.trainer.train_online(
                self._pending[domain],
                domain,
                min_batch_size=1,  # Train what we have
            )

            self._pending[domain] = []
            self._last_train[domain] = time.time()

            return metrics

        return None

    def force_train(self, domain: str | None = None) -> dict:
        """
        Force training on pending examples.

        Args:
            domain: Specific domain or None for all.

        Returns:
            Dictionary of metrics per domain.
        """
        results = {}
        domains = [domain] if domain else self.domains

        for d in domains:
            if self._pending.get(d):
                metrics = self.trainer.train_online(
                    self._pending[d], d, min_batch_size=1
                )
                if metrics:
                    results[d] = metrics
                self._pending[d] = []

        return results

    def pending_counts(self) -> dict[str, int]:
        """Get count of pending examples per domain."""
        return {d: len(examples) for d, examples in self._pending.items()}
