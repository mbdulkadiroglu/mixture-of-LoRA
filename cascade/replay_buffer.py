"""
Simple FIFO replay buffer for cascade experiments.

Single-domain, no domain balancing — just a capped deque with random sampling.
"""

import json
import random
from collections import deque
from pathlib import Path


class CascadeReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque[dict] = deque(maxlen=max_size)

    def add(self, item: dict) -> None:
        """Add an item. Oldest items are evicted when full (FIFO)."""
        self._buffer.append(item)

    def sample(self, n: int) -> list[dict]:
        """Sample n items uniformly at random (without replacement)."""
        if not self._buffer:
            return []
        return random.sample(list(self._buffer), min(n, len(self._buffer)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(list(self._buffer), f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._buffer = deque(data, maxlen=self.max_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)
