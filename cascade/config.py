"""
Configuration for cascade distillation experiments.

Every experiment is fully specified by a single CascadeConfig instance.
Self-contained — does not import existing FrameworkConfig. Constructs
existing config dataclasses (StudentModelConfig, TrainingConfig, etc.)
on demand when wrapper modules need them.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class CascadeConfig:
    # Identity & reproducibility
    experiment_name: str = "default"
    seed: int = 42

    # Dataset
    dataset: str = "spider"  # "spider" or "bird"
    eval_set_size: int = 200  # Frozen eval set, never used for training
    training_pool_size: int = 2000  # Queries available across all rounds

    # Student
    student_model: str = "unsloth/Qwen2.5-14B-Instruct"
    initial_adapter_path: str | None = None  # None = base model (start from scratch)

    # Teacher
    teacher_model: str = "gpt-5-mini"
    teacher_backend: str = "openai"  # "openai" or "ollama"
    ollama_url: str = "http://localhost:11434/v1"
    teacher_temperature: float = 0.0
    teacher_max_retries: int = 3
    teacher_retry_backoff_seconds: float = 1.0
    teacher_corruption_rate: float = 0.0  # Phase 2
    teacher_corruption_strategy: str = "none"  # Phase 2

    # Prompts
    prompt_variant: str = ""  # Override prompt variant for teacher (e.g. "bird_json"). Empty = use dataset.

    # Router
    router_threshold: float = -4.0  # Mean token log-prob
    router_metric: str = "mean_logprob"  # or "min_logprob", "mean_entropy"
    adaptive_threshold: bool = False  # Phase 3
    cascade_rate_start: float = 0.0  # Target cascade rate at round 0. 0 = use fixed router_threshold.
    cascade_rate_end: float = 0.0    # Target cascade rate at final round.

    # Training
    train_after_round: bool = True
    training_source: str = "teacher"  # "teacher" or "gold" (exp 1.3)
    train_student_responses: bool = False  # Include non-cascaded student responses in training
    train_start_round: int = 0  # Skip training before this round (still buffer examples)
    teacher_cache_path: str | None = None  # Path to pre-cached teacher responses JSON
    num_rounds: int = 10
    queries_per_round: int = 200
    training_epochs: int = 3
    training_lr: float = 1e-4
    training_lr_start: float = 0.0  # Cross-round LR warmup start. 0 = no warmup (use training_lr).
    training_lr_warmup_rounds: int = 0  # Rounds to linearly ramp from lr_start to lr. 0 = no warmup.
    training_batch_size: int = 4
    training_grad_accum: int = 4
    lora_r: int = 32
    lora_alpha: int = 32
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.2

    # Filtering (Phase 3)
    filter_teacher_consistency: bool = False
    consistency_samples: int = 3
    confidence_weighting: bool = False
    min_confidence_weight: float = 0.1

    # Audit (Phase 3)
    audit_rate: float = 0.0

    # Evaluation
    eval_every_n_rounds: int = 1

    # Output
    output_dir: str = "cascade/results"
    gpu_devices: str = "2"  # Single GPU — Unsloth fast inference crashes on multi-GPU

    def save(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "CascadeConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_student_model_config(self):
        """Construct a StudentModelConfig for the wrapped StudentModel."""
        from src.config import StudentModelConfig
        return StudentModelConfig(name=self.student_model)

    def to_lora_config(self):
        """Construct a LoRAConfig for LoRA setup."""
        from src.config import LoRAConfig
        return LoRAConfig(r=self.lora_r, lora_alpha=self.lora_alpha)

    def to_training_config(self):
        """Construct a TrainingConfig for LoRATrainer."""
        from src.config import TrainingConfig
        return TrainingConfig(
            batch_size=self.training_batch_size,
            gradient_accumulation_steps=self.training_grad_accum,
            learning_rate=self.training_lr,
            num_epochs=self.training_epochs,
            replay_buffer_size=self.replay_buffer_size,
            replay_ratio=0.0,  # We handle replay mixing ourselves
        )

    def to_teacher_model_config(self):
        """Construct a TeacherModelConfig for the wrapped TeacherModel."""
        import os
        from src.config import TeacherModelConfig
        return TeacherModelConfig(
            name=self.teacher_model,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=self.teacher_temperature,
        )

    def get_cascade_rate(self, round_idx: int) -> float:
        """Get target cascade rate for a given round (linear interpolation)."""
        if self.num_rounds <= 1:
            return self.cascade_rate_start
        frac = min(round_idx / (self.num_rounds - 1), 1.0)
        return self.cascade_rate_start + frac * (self.cascade_rate_end - self.cascade_rate_start)

    def get_lr_for_round(self, round_idx: int) -> float:
        """Get learning rate for a given round, applying cross-round warmup."""
        if self.training_lr_warmup_rounds <= 0 or self.training_lr_start <= 0:
            return self.training_lr

        if round_idx >= self.training_lr_warmup_rounds:
            return self.training_lr

        # Linear interpolation from lr_start to lr
        frac = round_idx / self.training_lr_warmup_rounds
        return self.training_lr_start + frac * (self.training_lr - self.training_lr_start)

    @property
    def experiment_dir(self) -> Path:
        """Get the experiment output directory."""
        return Path(self.output_dir) / f"exp_{self.experiment_name}"
