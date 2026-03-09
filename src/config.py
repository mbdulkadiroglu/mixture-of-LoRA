"""
Configuration management for the Adaptive SLM Framework.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class TeacherModelConfig:
    """Configuration for the teacher model (GPT-5 mini)."""

    name: str = "gpt-5-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class StudentModelConfig:
    """Configuration for the student model."""

    name: str = "unsloth/Qwen2.5-14B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    online_learning: bool = True
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    num_epochs: int = 3
    max_steps: int = -1
    optimizer: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.2


@dataclass
class RouterConfig:
    """Configuration for the routing mechanism."""

    confidence_threshold: float = 0.7
    routing_method: str = "self_eval"
    max_retries: int = 2
    self_eval_prompt: str = ""


@dataclass
class AdapterManagerConfig:
    """Configuration for adapter management."""

    base_path: str = "data/lora_adapters"
    merge_strategy: str = "none"
    max_adapters_per_domain: int = 5
    selection_strategy: str = "best"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    batch_size: int = 8
    quick_eval_samples: int = 100
    full_eval_frequency: int = 500
    metrics: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "exact_match",
            "bleu",
            "execution_accuracy",
        ]
    )
    spider_db_dir: str | None = None  # Path to Spider database for SQL execution eval
    bird_db_dir: str | None = None  # Path to BIRD data directory for SQL execution eval


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    name: str
    description: str
    dataset: str
    evaluation_metric: str


@dataclass
class FrameworkConfig:
    """Main configuration container."""

    teacher: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    student: StudentModelConfig = field(default_factory=StudentModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    adapter_manager: AdapterManagerConfig = field(default_factory=AdapterManagerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    domains: dict[str, DomainConfig] = field(default_factory=dict)
    seed: int = 42
    log_level: str = "INFO"
    device_map: str = "auto"
    num_workers: int = 4


def load_config(config_path: str | Path | None = None) -> FrameworkConfig:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        FrameworkConfig object with all settings.
    """
    # Load environment variables from project root (override existing)
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path, override=True)

    # Initialize with defaults
    config = FrameworkConfig()

    # Load from YAML if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
                config = _merge_yaml_config(config, yaml_config)

    # Override with environment variables
    config.teacher.api_key = os.getenv("OPENAI_API_KEY", "")
    config.teacher.api_base = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )

    if os.getenv("TEACHER_MODEL"):
        config.teacher.name = os.getenv("TEACHER_MODEL")

    if os.getenv("STUDENT_MODEL"):
        config.student.name = os.getenv("STUDENT_MODEL")

    if os.getenv("LORA_ADAPTERS_PATH"):
        config.adapter_manager.base_path = os.getenv("LORA_ADAPTERS_PATH")

    if os.getenv("ROUTER_CONFIDENCE_THRESHOLD"):
        config.router.confidence_threshold = float(
            os.getenv("ROUTER_CONFIDENCE_THRESHOLD")
        )

    return config


def _merge_yaml_config(config: FrameworkConfig, yaml_config: dict[str, Any]) -> FrameworkConfig:
    """Merge YAML configuration into the config object."""
    if "models" in yaml_config:
        if "teacher" in yaml_config["models"]:
            for key, value in yaml_config["models"]["teacher"].items():
                if hasattr(config.teacher, key):
                    setattr(config.teacher, key, value)

        if "student" in yaml_config["models"]:
            for key, value in yaml_config["models"]["student"].items():
                if hasattr(config.student, key):
                    setattr(config.student, key, value)

    if "lora" in yaml_config:
        for key, value in yaml_config["lora"].items():
            if hasattr(config.lora, key):
                setattr(config.lora, key, value)

    if "training" in yaml_config:
        for key, value in yaml_config["training"].items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

    if "router" in yaml_config:
        for key, value in yaml_config["router"].items():
            if hasattr(config.router, key):
                setattr(config.router, key, value)

    if "adapter_manager" in yaml_config:
        for key, value in yaml_config["adapter_manager"].items():
            if hasattr(config.adapter_manager, key):
                setattr(config.adapter_manager, key, value)

    if "evaluation" in yaml_config:
        for key, value in yaml_config["evaluation"].items():
            if hasattr(config.evaluation, key):
                setattr(config.evaluation, key, value)

    if "domains" in yaml_config:
        for domain_key, domain_data in yaml_config["domains"].items():
            config.domains[domain_key] = DomainConfig(
                name=domain_data.get("name", domain_key),
                description=domain_data.get("description", ""),
                dataset=domain_data.get("dataset", ""),
                evaluation_metric=domain_data.get("evaluation_metric", "accuracy"),
            )

    if "system" in yaml_config:
        if "seed" in yaml_config["system"]:
            config.seed = yaml_config["system"]["seed"]
        if "log_level" in yaml_config["system"]:
            config.log_level = yaml_config["system"]["log_level"]
        if "device_map" in yaml_config["system"]:
            config.device_map = yaml_config["system"]["device_map"]
        if "num_workers" in yaml_config["system"]:
            config.num_workers = yaml_config["system"]["num_workers"]

    return config
