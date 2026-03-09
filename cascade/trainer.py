"""
Cascade trainer — wraps LoRATrainer + DataProcessor for per-round training.

Converts raw examples into TrainingExample objects, formats via chat template,
and delegates to LoRATrainer for SFT. Saves adapter per round for versioning.
"""

from pathlib import Path

import torch
from loguru import logger

from .config import CascadeConfig
from .student import CascadeStudent


class CascadeTrainer:
    def __init__(self, student: CascadeStudent, config: CascadeConfig):
        self.student = student
        self.config = config
        self._lora_setup_done = False

    def _ensure_lora_setup(self, inner) -> None:
        """Initialize LoRA once, skipping if a PEFT model is already attached."""
        if self._lora_setup_done:
            return

        if inner.peft_model is None:
            inner.setup_lora()

        self._lora_setup_done = True

    def train_round(
        self,
        new_examples: list[dict],
        replay_examples: list[dict],
        round_num: int,
    ) -> dict:
        """
        Train the student on new + replay examples for one round.

        Each example dict has: {prompt, response, db_id, source_round, was_correct}

        Returns training metrics dict.
        """
        from src.training.data_processor import DataProcessor, TrainingExample
        from src.training.trainer import LoRATrainer

        inner = self.student.inner

        # Setup LoRA on first training round (unless pre-initialized at setup).
        self._ensure_lora_setup(inner)

        # Convert to TrainingExample objects
        domain = "text_to_sql" if self.config.dataset == "spider" else "text_to_sql_bird"
        training_examples = []
        for ex in new_examples + replay_examples:
            training_examples.append(
                TrainingExample(
                    query=ex["prompt"],
                    response=ex["response"],
                    domain=domain,
                )
            )

        if not training_examples:
            logger.warning(f"Round {round_num}: no training examples, skipping")
            return {"train_loss": 0.0, "train_samples": 0}

        # Prepare dataset via DataProcessor
        processor = DataProcessor(inner.tokenizer, inner.config.max_seq_length)
        dataset = processor.prepare_dataset(training_examples)

        if len(dataset) == 0:
            logger.warning(f"Round {round_num}: all examples filtered out, skipping")
            return {"train_loss": 0.0, "train_samples": 0}

        # Create output dir for this round
        round_dir = Path(self.config.output_dir) / f"exp_{self.config.experiment_name}" / "adapters" / f"round_{round_num}"

        # Build LoRATrainer with per-round LR
        training_config = self.config.to_training_config()
        round_lr = self.config.get_lr_for_round(round_num)
        training_config.learning_rate = round_lr
        lora_config = self.config.to_lora_config()

        trainer = LoRATrainer(
            student=inner,
            training_config=training_config,
            lora_config=lora_config,
            output_dir=round_dir / "checkpoints",
        )

        # Train
        logger.info(
            f"Round {round_num}: training on {len(new_examples)} new + "
            f"{len(replay_examples)} replay = {len(dataset)} examples "
            f"(lr={round_lr:.2e})"
        )
        metrics = trainer.train(
            dataset=dataset,
            domain=domain,
            num_epochs=self.config.training_epochs,
        )

        # Save adapter for versioning
        adapter_path = round_dir / "adapter"
        inner.save_adapter(adapter_path)

        # Clean up trainer to free memory
        del trainer
        torch.cuda.empty_cache()

        metrics["adapter_path"] = str(adapter_path)
        metrics["new_examples"] = len(new_examples)
        metrics["replay_examples"] = len(replay_examples)
        metrics["round"] = round_num

        return metrics
