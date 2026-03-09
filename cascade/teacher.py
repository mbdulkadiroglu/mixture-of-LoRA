"""
Cascade teacher model — wraps TeacherModel with corruption support (Phase 2).

Supports two backends:
- "openai": OpenAI Responses API via TeacherModel (GPT-5 mini, etc.)
- "ollama": Ollama via OpenAI-compatible Chat Completions API (local models)

Phase 0/1: pure pass-through.
Phase 2: injects controlled errors into teacher SQL responses.
"""

import random
import re
import time

from loguru import logger

from .config import CascadeConfig


class CascadeTeacher:
    def __init__(self, config: CascadeConfig):
        self.config = config
        self._teacher = None  # OpenAI backend (TeacherModel)
        self._client = None   # Ollama backend (OpenAI client)

    def load(self) -> None:
        if self.config.teacher_backend == "ollama":
            self._load_ollama()
        else:
            self._load_openai()

    def _load_openai(self) -> None:
        from src.models.teacher import TeacherModel
        teacher_config = self.config.to_teacher_model_config()
        self._teacher = TeacherModel(teacher_config)
        logger.info(f"CascadeTeacher loaded (openai): {self.config.teacher_model}")

    def _load_ollama(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=self.config.ollama_url, api_key="ollama")

        # Connectivity check
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            if self.config.teacher_model not in available:
                logger.warning(
                    f"Model {self.config.teacher_model} not in available Ollama models: {available}"
                )
            else:
                logger.info(f"Ollama model {self.config.teacher_model} available")
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")

        logger.info(
            f"CascadeTeacher loaded (ollama): {self.config.teacher_model} "
            f"at {self.config.ollama_url}"
        )

    def generate(self, messages: list[dict]) -> str:
        """Generate a response. Applies corruption if configured (Phase 2)."""
        max_retries = max(0, self.config.teacher_max_retries)
        response = None

        for attempt in range(max_retries + 1):
            try:
                if self.config.teacher_backend == "ollama":
                    response = self._generate_ollama(messages)
                else:
                    response = self._teacher.generate(messages)
                break
            except Exception as e:
                if attempt >= max_retries:
                    logger.error(f"Teacher generation failed after {attempt + 1} attempts: {e}")
                    raise

                backoff = self.config.teacher_retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    f"Teacher generation attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {backoff:.2f}s..."
                )
                time.sleep(backoff)

        if (
            self.config.teacher_corruption_rate > 0
            and self.config.teacher_corruption_strategy != "none"
            and random.random() < self.config.teacher_corruption_rate
        ):
            response = self._corrupt_sql(response, self.config.teacher_corruption_strategy)

        return response

    def _generate_ollama(self, messages: list[dict]) -> str:
        """Generate via Ollama's OpenAI-compatible Chat Completions API.

        Handles thinking models (qwen3, etc.) that put reasoning in a separate
        field and may return empty content. Falls back to Ollama's native API.
        """
        resp = self._client.chat.completions.create(
            model=self.config.teacher_model,
            messages=messages,
            temperature=self.config.teacher_temperature,
            max_tokens=8192,
            timeout=120.0,
        )
        content = resp.choices[0].message.content or ""
        if content:
            return content

        # Fallback for thinking models: native Ollama /api/chat endpoint
        import requests

        base_url = str(self._client.base_url)
        ollama_url = base_url.replace("/v1/", "/").replace("/v1", "/")
        native_resp = requests.post(
            f"{ollama_url}api/chat",
            json={
                "model": self.config.teacher_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.teacher_temperature,
                    "num_predict": 8192,
                },
            },
            timeout=120.0,
        )
        native_resp.raise_for_status()
        full_text = native_resp.json().get("message", {}).get("content", "")

        # Strip <think>...</think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        return cleaned if cleaned else full_text

    def _corrupt_sql(self, sql: str, strategy: str) -> str:
        """Apply a corruption strategy to SQL. Phase 2 — stub implementations."""
        if strategy == "swap_columns":
            return self._swap_columns(sql)
        elif strategy == "change_join":
            return self._change_join(sql)
        elif strategy == "alter_where":
            return self._alter_where(sql)
        elif strategy == "modify_aggregate":
            return self._modify_aggregate(sql)
        else:
            logger.warning(f"Unknown corruption strategy: {strategy}")
            return sql

    # --- Phase 2 corruption implementations (deferred) ---

    @staticmethod
    def _swap_columns(sql: str) -> str:
        """Replace a column name with a placeholder corruption."""
        # Find column-like identifiers after SELECT
        match = re.search(r"SELECT\s+(.+?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
        if match:
            cols_str = match.group(1)
            cols = [c.strip() for c in cols_str.split(",")]
            if len(cols) >= 2:
                i, j = random.sample(range(len(cols)), 2)
                cols[i], cols[j] = cols[j], cols[i]
                new_cols = ", ".join(cols)
                sql = sql[:match.start(1)] + new_cols + sql[match.end(1):]
        return sql

    @staticmethod
    def _change_join(sql: str) -> str:
        """Swap INNER JOIN <-> LEFT JOIN."""
        if "INNER JOIN" in sql.upper():
            return re.sub(r"INNER\s+JOIN", "LEFT JOIN", sql, count=1, flags=re.IGNORECASE)
        elif "LEFT JOIN" in sql.upper():
            return re.sub(r"LEFT\s+JOIN", "INNER JOIN", sql, count=1, flags=re.IGNORECASE)
        return sql

    @staticmethod
    def _alter_where(sql: str) -> str:
        """Change a comparison operator in WHERE clause."""
        swaps = {"=": "!=", ">": "<", "<": ">", ">=": "<=", "<=": ">=", "!=": "="}
        for old, new in swaps.items():
            pattern = rf"(WHERE\s+\S+\s*){re.escape(old)}"
            if re.search(pattern, sql, re.IGNORECASE):
                return re.sub(pattern, rf"\g<1>{new}", sql, count=1, flags=re.IGNORECASE)
        return sql

    @staticmethod
    def _modify_aggregate(sql: str) -> str:
        """Swap aggregate functions."""
        swaps = {"SUM": "COUNT", "COUNT": "SUM", "AVG": "MAX", "MAX": "AVG", "MIN": "MAX"}
        for old, new in swaps.items():
            pattern = rf"\b{old}\s*\("
            if re.search(pattern, sql, re.IGNORECASE):
                return re.sub(pattern, f"{new}(", sql, count=1, flags=re.IGNORECASE)
        return sql
