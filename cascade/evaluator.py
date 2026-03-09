"""
Cascade evaluator — wraps SQLExecutor / BIRDExecutor for execution-based evaluation.
"""

from loguru import logger

from .config import CascadeConfig
from .prompts import build_query_messages
from .student import CascadeStudent


class CascadeEvaluator:
    def __init__(self, config: CascadeConfig):
        self.config = config
        self._executor = None

    def load(self) -> None:
        """Initialize the SQL executor for the configured dataset."""
        if self.config.dataset == "spider":
            from src.evaluation.sql_executor import get_spider_executor
            self._executor = get_spider_executor()
            if self._executor is None:
                raise RuntimeError("Spider database directory not found")
        elif self.config.dataset == "bird":
            from src.evaluation.sql_executor import BIRDExecutor
            self._executor = BIRDExecutor()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        logger.info(f"CascadeEvaluator loaded for {self.config.dataset}")

    def check_single(self, predicted_sql: str, gold_sql: str, db_id: str) -> bool:
        """Check if a single prediction matches the gold SQL via execution."""
        is_correct, _ = self._executor.evaluate_single(predicted_sql, gold_sql, db_id)
        return is_correct

    def evaluate_set(
        self,
        student: CascadeStudent,
        eval_queries: list[dict],
    ) -> float:
        """
        Evaluate the student on a frozen eval set.

        Generates predictions for each query, checks via SQL execution,
        and returns accuracy.
        """
        if not eval_queries:
            return 0.0

        correct = 0
        total = 0

        for query in eval_queries:
            messages = self._build_messages(query)
            gen_result = student.generate_with_logprobs(messages)

            gold_sql = query.get("query") or query.get("SQL", "")
            db_id = query["db_id"]

            is_correct = self.check_single(gen_result.text, gold_sql, db_id)
            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Eval accuracy: {correct}/{total} = {accuracy:.4f}")
        return accuracy

    def _build_messages(self, query: dict) -> list[dict]:
        """Build chat messages for a query (mirrors runner logic)."""
        prompt = query.get("prompt", query.get("question", ""))
        return build_query_messages(self.config.dataset, prompt)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.close()
