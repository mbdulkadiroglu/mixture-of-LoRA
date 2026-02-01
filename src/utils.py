"""
Utility functions for the Adaptive SLM Framework.
"""

import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging with loguru."""
    logger.remove()

    # Console logging
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File logging if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="100 MB",
            retention="7 days",
        )


def compute_hash(text: str) -> str:
    """Compute a hash for a text string."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict | list:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def classify_query_domain(query: str) -> str:
    """
    Simple rule-based domain classification.
    In production, this could be replaced with a classifier model.
    """
    query_lower = query.lower()

    # Text-to-SQL patterns
    sql_keywords = [
        "sql",
        "query",
        "database",
        "table",
        "select",
        "from",
        "where",
        "join",
        "group by",
        "order by",
        "schema",
    ]
    if any(kw in query_lower for kw in sql_keywords):
        return "text_to_sql"

    # Math reasoning patterns
    math_patterns = [
        r"\d+\s*[\+\-\*\/]\s*\d+",
        r"how many",
        r"calculate",
        r"compute",
        r"solve",
        r"equation",
        r"percentage",
        r"fraction",
        r"total",
        r"average",
        r"sum of",
    ]
    if any(re.search(pattern, query_lower) for pattern in math_patterns):
        return "math_reasoning"

    # Code generation patterns
    code_keywords = [
        "python",
        "function",
        "code",
        "implement",
        "write a program",
        "algorithm",
        "class",
        "def ",
        "import",
        "loop",
        "array",
        "list",
    ]
    if any(kw in query_lower for kw in code_keywords):
        return "code_generation"

    # Default to general
    return "general"


def format_chat_prompt(
    query: str,
    system_prompt: str | None = None,
    examples: list[dict] | None = None,
) -> list[dict]:
    """
    Format a query into chat message format.

    Args:
        query: The user's query.
        system_prompt: Optional system prompt.
        examples: Optional few-shot examples.

    Returns:
        List of message dictionaries.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if examples:
        for example in examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})

    messages.append({"role": "user", "content": query})

    return messages


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (rough estimate).
    For accurate counts, use the actual tokenizer.
    """
    # Rough approximation: ~4 characters per token
    return len(text) // 4


def gpu_memory_status() -> dict:
    """Get GPU memory status for all available GPUs."""
    if not torch.cuda.is_available():
        return {"available": False}

    status = {"available": True, "gpus": []}

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = props.total_memory

        status["gpus"].append(
            {
                "id": i,
                "name": props.name,
                "total_gb": total / 1e9,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "free_gb": (total - reserved) / 1e9,
            }
        )

    return status


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ExperienceReplayBuffer:
    """
    Experience replay buffer for continual learning.
    Stores training examples for later replay to prevent forgetting.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: list[dict] = []
        self.domain_counts: dict[str, int] = {}

    def add(self, example: dict) -> None:
        """Add an example to the buffer."""
        if len(self.buffer) >= self.max_size:
            # Remove oldest example from the most represented domain
            if self.domain_counts:
                max_domain = max(self.domain_counts, key=self.domain_counts.get)
                for i, item in enumerate(self.buffer):
                    if item.get("domain") == max_domain:
                        self.buffer.pop(i)
                        self.domain_counts[max_domain] -= 1
                        break

        self.buffer.append(example)
        domain = example.get("domain", "unknown")
        self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1

    def sample(self, n: int, domain: str | None = None) -> list[dict]:
        """Sample examples from the buffer."""
        if domain:
            domain_examples = [ex for ex in self.buffer if ex.get("domain") == domain]
            if not domain_examples:
                return []
            return random.sample(domain_examples, min(n, len(domain_examples)))

        if not self.buffer:
            return []
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def save(self, path: str | Path) -> None:
        """Save buffer to disk."""
        save_json(
            {"buffer": self.buffer, "domain_counts": self.domain_counts},
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load buffer from disk."""
        path = Path(path)
        if path.exists():
            data = load_json(path)
            self.buffer = data.get("buffer", [])
            self.domain_counts = data.get("domain_counts", {})

    def __len__(self) -> int:
        return len(self.buffer)
