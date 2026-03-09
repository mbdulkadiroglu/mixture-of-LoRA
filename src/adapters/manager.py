"""
LoRA adapter manager for storing, loading, and managing domain-specific adapters.
"""

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..config import AdapterManagerConfig


@dataclass
class AdapterInfo:
    """Metadata for a LoRA adapter."""

    name: str
    domain: str
    version: int
    created_at: str
    training_samples: int
    eval_score: float
    path: str
    lora_r: int
    lora_alpha: int
    base_model: str
    is_active: bool = True


class AdapterManager:
    """
    Manages LoRA adapters for different domains.

    Handles adapter storage, versioning, loading, and lifecycle management.
    """

    def __init__(self, config: AdapterManagerConfig):
        """
        Initialize the adapter manager.

        Args:
            config: Adapter manager configuration.
        """
        self.config = config
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # In-memory adapter registry
        self._adapters: dict[str, list[AdapterInfo]] = {}

        # Load existing adapters from disk
        self._load_registry()

        logger.info(f"AdapterManager initialized at: {self.base_path}")

    def _load_registry(self) -> None:
        """Load adapter registry from disk."""
        registry_path = self.base_path / "registry.json"

        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)

            for domain, adapters in data.items():
                self._adapters[domain] = [
                    AdapterInfo(**adapter) for adapter in adapters
                ]

            logger.info(f"Loaded {sum(len(a) for a in self._adapters.values())} adapters from registry")

    def _save_registry(self) -> None:
        """Save adapter registry to disk."""
        registry_path = self.base_path / "registry.json"

        data = {
            domain: [asdict(adapter) for adapter in adapters]
            for domain, adapters in self._adapters.items()
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_adapter(
        self,
        adapter_path: str | Path,
        domain: str,
        training_samples: int,
        eval_score: float,
        lora_r: int,
        lora_alpha: int,
        base_model: str,
    ) -> AdapterInfo:
        """
        Register a new adapter.

        Args:
            adapter_path: Path to the adapter files.
            domain: Domain this adapter serves.
            training_samples: Number of training samples used.
            eval_score: Evaluation score.
            lora_r: LoRA rank used.
            lora_alpha: LoRA alpha used.
            base_model: Base model name.

        Returns:
            AdapterInfo for the registered adapter.
        """
        adapter_path = Path(adapter_path)

        # Get next version number
        existing = self._adapters.get(domain, [])
        version = max([a.version for a in existing], default=0) + 1

        # Create adapter name
        name = f"{domain}_v{version}"

        # Target path in managed storage
        target_path = self.base_path / domain / name

        # Copy adapter files if not already in managed storage
        if adapter_path != target_path:
            target_path.mkdir(parents=True, exist_ok=True)
            if adapter_path.is_dir():
                for item in adapter_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, target_path / item.name)
            else:
                shutil.copy2(adapter_path, target_path)

        # Create adapter info
        info = AdapterInfo(
            name=name,
            domain=domain,
            version=version,
            created_at=datetime.now().isoformat(),
            training_samples=training_samples,
            eval_score=eval_score,
            path=str(target_path),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            base_model=base_model,
            is_active=True,
        )

        # Add to registry
        if domain not in self._adapters:
            self._adapters[domain] = []
        self._adapters[domain].append(info)

        # Enforce max adapters per domain
        self._enforce_limits(domain)

        # Save registry
        self._save_registry()

        logger.info(f"Registered adapter: {name} (score: {eval_score:.3f})")

        return info

    def _enforce_limits(self, domain: str) -> None:
        """Enforce maximum adapters per domain limit."""
        adapters = self._adapters.get(domain, [])

        if len(adapters) <= self.config.max_adapters_per_domain:
            return

        # Sort by eval score (keep best) and version (prefer newer for ties)
        adapters.sort(key=lambda a: (a.eval_score, a.version), reverse=True)

        # Remove excess adapters
        to_remove = adapters[self.config.max_adapters_per_domain:]
        self._adapters[domain] = adapters[: self.config.max_adapters_per_domain]

        for adapter in to_remove:
            self._delete_adapter_files(adapter)
            logger.info(f"Removed old adapter: {adapter.name}")

    def _delete_adapter_files(self, adapter: AdapterInfo) -> None:
        """Delete adapter files from disk."""
        adapter_path = Path(adapter.path)
        if adapter_path.exists():
            if adapter_path.is_dir():
                shutil.rmtree(adapter_path)
            else:
                adapter_path.unlink()

    def get_adapter(
        self,
        domain: str,
        version: int | None = None,
    ) -> AdapterInfo | None:
        """
        Get adapter for a domain.

        Args:
            domain: Domain to get adapter for.
            version: Specific version (None for best/latest based on strategy).

        Returns:
            AdapterInfo or None if not found.
        """
        adapters = self._adapters.get(domain, [])

        if not adapters:
            return None

        if version is not None:
            for adapter in adapters:
                if adapter.version == version:
                    return adapter
            return None

        # Select based on strategy
        if self.config.selection_strategy == "best":
            return max(adapters, key=lambda a: a.eval_score)
        elif self.config.selection_strategy == "latest":
            return max(adapters, key=lambda a: a.version)
        else:
            # Default to best
            return max(adapters, key=lambda a: a.eval_score)

    def get_adapter_path(self, domain: str) -> str | None:
        """
        Get the path to the best adapter for a domain.

        Args:
            domain: Domain to get adapter for.

        Returns:
            Path string or None.
        """
        adapter = self.get_adapter(domain)
        return adapter.path if adapter else None

    def get_adapter_by_path(self, domain: str, path: str) -> AdapterInfo | None:
        """Get adapter by its file path."""
        for adapter in self._adapters.get(domain, []):
            if adapter.path == str(path):
                return adapter
        return None

    def list_adapters(self, domain: str | None = None) -> list[AdapterInfo]:
        """
        List all adapters, optionally filtered by domain.

        Args:
            domain: Optional domain filter.

        Returns:
            List of AdapterInfo objects.
        """
        if domain:
            return self._adapters.get(domain, []).copy()

        all_adapters = []
        for adapters in self._adapters.values():
            all_adapters.extend(adapters)
        return all_adapters

    def list_domains(self) -> list[str]:
        """
        List all domains with registered adapters.

        Returns:
            List of domain names.
        """
        return list(self._adapters.keys())

    def delete_adapter(self, domain: str, version: int) -> bool:
        """
        Delete a specific adapter.

        Args:
            domain: Domain of the adapter.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        adapters = self._adapters.get(domain, [])

        for i, adapter in enumerate(adapters):
            if adapter.version == version:
                self._delete_adapter_files(adapter)
                adapters.pop(i)
                self._save_registry()
                logger.info(f"Deleted adapter: {adapter.name}")
                return True

        return False

    def update_adapter_score(
        self,
        domain: str,
        version: int,
        new_score: float,
    ) -> bool:
        """
        Update the evaluation score for an adapter.

        Args:
            domain: Domain of the adapter.
            version: Version to update.
            new_score: New evaluation score.

        Returns:
            True if updated, False if not found.
        """
        adapters = self._adapters.get(domain, [])

        for adapter in adapters:
            if adapter.version == version:
                adapter.eval_score = new_score
                self._save_registry()
                logger.info(f"Updated score for {adapter.name}: {new_score:.3f}")
                return True

        return False

    def get_domain_stats(self, domain: str) -> dict:
        """
        Get statistics for a domain's adapters.

        Args:
            domain: Domain to get stats for.

        Returns:
            Statistics dictionary.
        """
        adapters = self._adapters.get(domain, [])

        if not adapters:
            return {
                "domain": domain,
                "num_adapters": 0,
                "best_score": 0.0,
                "avg_score": 0.0,
                "total_training_samples": 0,
            }

        scores = [a.eval_score for a in adapters]
        samples = [a.training_samples for a in adapters]

        return {
            "domain": domain,
            "num_adapters": len(adapters),
            "best_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "total_training_samples": sum(samples),
            "latest_version": max(a.version for a in adapters),
        }

    def should_create_new_adapter(self, domain: str, new_score: float) -> bool:
        """
        Determine if a new adapter should be created based on score improvement.

        Args:
            domain: Domain to check.
            new_score: Score of potential new adapter.

        Returns:
            True if new adapter should be created.
        """
        current = self.get_adapter(domain)

        if current is None:
            return True

        # Create new if score improved by at least 1%
        improvement_threshold = 0.01
        return new_score > current.eval_score + improvement_threshold
