"""
Evaluation system for measuring model performance across domains.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from .sql_executor import SQLExecutor, get_spider_executor


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""

    domain: str
    metric_name: str
    score: float
    num_samples: int
    correct: int
    details: dict = field(default_factory=dict)


class Evaluator:
    """
    Evaluates model performance on various domains.

    Supports multiple evaluation metrics:
    - Exact match (for SQL, math answers)
    - Execution accuracy (for SQL)
    - Pass@k (for code generation)
    - BLEU score (for general text)
    """

    def __init__(self, spider_db_dir: str | Path | None = None):
        """
        Initialize the evaluator.

        Args:
            spider_db_dir: Path to Spider database directory for execution-based eval.
        """
        self._metrics: dict[str, Callable] = {
            "exact_match": self._exact_match,
            "sql_exact_match": self._sql_exact_match,
            "math_answer_match": self._math_answer_match,
            "code_execution": self._code_execution,
        }

        # Initialize SQL executor if available
        self._sql_executor: SQLExecutor | None = None
        if spider_db_dir:
            try:
                self._sql_executor = SQLExecutor(spider_db_dir)
            except Exception as e:
                logger.warning(f"Failed to initialize SQL executor: {e}")
        else:
            # Try to auto-detect
            self._sql_executor = get_spider_executor()

        if self._sql_executor:
            logger.info("Evaluator initialized with SQL execution support")
        else:
            logger.info("Evaluator initialized (SQL execution not available, using exact match)")

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        domain: str,
        metric: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate predictions against references.

        Args:
            predictions: List of model predictions.
            references: List of ground truth references.
            domain: Domain being evaluated.
            metric: Specific metric to use (auto-selected if None).

        Returns:
            EvaluationResult with scores and details.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length"
            )

        # Auto-select metric based on domain
        if metric is None:
            metric = self._get_default_metric(domain)

        logger.info(f"Evaluating {len(predictions)} samples for domain '{domain}' with metric '{metric}'")

        metric_fn = self._metrics.get(metric, self._exact_match)

        correct = 0
        details = {"incorrect_samples": []}

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            is_correct = metric_fn(pred, ref)
            if is_correct:
                correct += 1
            else:
                if len(details["incorrect_samples"]) < 10:  # Keep first 10 errors
                    details["incorrect_samples"].append({
                        "index": i,
                        "prediction": pred[:200],
                        "reference": ref[:200],
                    })

        score = correct / len(predictions) if predictions else 0.0

        logger.info(f"Evaluation complete: {score:.2%} ({correct}/{len(predictions)})")

        return EvaluationResult(
            domain=domain,
            metric_name=metric,
            score=score,
            num_samples=len(predictions),
            correct=correct,
            details=details,
        )

    # Domain-specific system prompts for evaluation
    SYSTEM_PROMPTS = {
        "text_to_sql": """You are an expert SQL assistant. Convert natural language queries to SQL.
Think step by step, then provide the final SQL query in a code block.""",
        "math_reasoning": """You are a mathematics tutor. Solve problems step by step.
Show your work clearly and provide the final numerical answer.""",
        "code_generation": """You are an expert Python programmer.
Write clean, efficient, well-documented code with type hints.""",
        "general": """You are a helpful, accurate assistant.
Provide clear and well-structured responses.""",
    }

    def evaluate_dataset(
        self,
        model,
        dataset: Dataset,
        domain: str,
        query_key: str = "question",
        reference_key: str = "answer",
        max_samples: int | None = None,
        batch_size: int = 1,
        use_execution: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a model on a dataset.

        Args:
            model: Model with generate_chat method.
            dataset: Dataset to evaluate on.
            domain: Domain being evaluated.
            query_key: Key for input queries in dataset.
            reference_key: Key for reference answers in dataset.
            max_samples: Maximum samples to evaluate.
            batch_size: Batch size for generation.
            use_execution: Use execution-based eval for SQL (if available).

        Returns:
            EvaluationResult with scores.
        """
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        predictions = []
        references = []
        db_ids = []

        logger.info(f"Evaluating model on {len(dataset)} samples...")

        # Get system prompt for this domain
        system_prompt = self.SYSTEM_PROMPTS.get(domain, self.SYSTEM_PROMPTS["general"])

        for example in tqdm(dataset, desc="Evaluating"):
            # For text_to_sql, prefer 'prompt' field which includes schema context
            if domain == "text_to_sql" and "prompt" in example:
                query = example["prompt"]
            else:
                query = example[query_key]

            # Get reference based on domain
            if domain == "text_to_sql":
                reference = example.get("query", example.get(reference_key, ""))
                db_id = example.get("db_id", None)
                db_ids.append(db_id)
            elif domain == "math_reasoning":
                reference = example.get("answer", example.get(reference_key, ""))
            elif domain == "code_generation":
                reference = example.get("code", example.get(reference_key, ""))
            else:
                reference = example.get(reference_key, "")

            # Generate prediction with system prompt for context
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            prediction = model.generate_chat(messages, temperature=0.1)

            predictions.append(prediction)
            references.append(reference)

        # Use execution-based evaluation for SQL if available
        if domain == "text_to_sql" and use_execution and self._sql_executor and db_ids:
            return self.evaluate_sql_execution(predictions, references, db_ids)

        return self.evaluate(predictions, references, domain)

    def evaluate_sql_execution(
        self,
        predictions: list[str],
        references: list[str],
        db_ids: list[str],
    ) -> EvaluationResult:
        """
        Evaluate SQL predictions using execution-based comparison.

        Args:
            predictions: List of predicted SQL queries.
            references: List of reference SQL queries.
            db_ids: List of database identifiers.

        Returns:
            EvaluationResult with execution accuracy.
        """
        if self._sql_executor is None:
            logger.warning("SQL executor not available, falling back to exact match")
            return self.evaluate(predictions, references, "text_to_sql")

        logger.info(f"Evaluating {len(predictions)} SQL queries with execution...")

        results = self._sql_executor.evaluate_batch(predictions, references, db_ids)

        logger.info(
            f"Execution evaluation: {results['execution_accuracy']:.2%} "
            f"({results['correct']}/{results['total']}), "
            f"execution errors: {results['execution_errors']}"
        )

        return EvaluationResult(
            domain="text_to_sql",
            metric_name="execution_accuracy",
            score=results["execution_accuracy"],
            num_samples=results["total"],
            correct=results["correct"],
            details={
                "execution_errors": results["execution_errors"],
                "execution_error_rate": results["execution_error_rate"],
                "sample_details": results["details"],
            },
        )

    def _get_default_metric(self, domain: str) -> str:
        """Get default metric for a domain."""
        domain_metrics = {
            "text_to_sql": "sql_exact_match",
            "math_reasoning": "math_answer_match",
            "code_generation": "code_execution",
            "general": "exact_match",
        }
        return domain_metrics.get(domain, "exact_match")

    def _exact_match(self, prediction: str, reference: str) -> bool:
        """Simple exact match after normalization."""
        pred_norm = self._normalize_text(prediction)
        ref_norm = self._normalize_text(reference)
        return pred_norm == ref_norm

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, remove extra whitespace
        text = text.lower().strip()
        text = " ".join(text.split())
        return text

    def _sql_exact_match(self, prediction: str, reference: str) -> bool:
        """
        SQL-specific exact match.

        Extracts SQL from code blocks and normalizes for comparison.
        """
        pred_sql = self._extract_sql(prediction)
        ref_sql = self._extract_sql(reference)

        pred_norm = self._normalize_sql(pred_sql)
        ref_norm = self._normalize_sql(ref_sql)

        return pred_norm == ref_norm

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from text, handling code blocks."""
        # Try to extract from code block
        sql_block_pattern = r"```(?:sql)?\s*(.*?)```"
        matches = re.findall(sql_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # Try to find SQL-like content
        # Look for SELECT, INSERT, UPDATE, DELETE statements
        sql_pattern = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+.*"
        match = re.search(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()

        return text.strip()

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        # Remove comments
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

        # Lowercase
        sql = sql.lower()

        # Normalize whitespace
        sql = " ".join(sql.split())

        # Remove trailing semicolon
        sql = sql.rstrip(";")

        # Normalize quotes
        sql = sql.replace('"', "'")

        return sql

    def _math_answer_match(self, prediction: str, reference: str) -> bool:
        """
        Match mathematical answers.

        Extracts final numerical answer and compares.
        """
        pred_answer = self._extract_math_answer(prediction)
        ref_answer = self._extract_math_answer(reference)

        if pred_answer is None or ref_answer is None:
            return False

        # Allow small floating point tolerance
        try:
            return abs(float(pred_answer) - float(ref_answer)) < 0.01
        except ValueError:
            return pred_answer == ref_answer

    def _extract_math_answer(self, text: str) -> str | None:
        """Extract final numerical answer from text."""
        # GSM8K format: #### <answer>
        match = re.search(r"####\s*([\d,.-]+)", text)
        if match:
            return match.group(1).replace(",", "")

        # Look for "answer is X" pattern
        match = re.search(r"(?:answer|result|total)\s*(?:is|=|:)\s*([\d,.-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

        # Last number in the text
        numbers = re.findall(r"[\d,]+\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")

        return None

    def _code_execution(self, prediction: str, reference: str) -> bool:
        """
        Evaluate code by execution.

        Note: This is a simplified version. Full implementation would
        run the code in a sandbox and check test cases.
        """
        # Extract code from prediction
        pred_code = self._extract_code(prediction)
        ref_code = self._extract_code(reference)

        # Simple structural comparison for safety
        # In production, use sandboxed execution
        pred_norm = self._normalize_code(pred_code)
        ref_norm = self._normalize_code(ref_code)

        # Check if key function names and structure match
        return self._code_structure_match(pred_norm, ref_norm)

    def _extract_code(self, text: str) -> str:
        """Extract code from text, handling code blocks."""
        # Try to extract from code block
        code_block_pattern = r"```(?:python)?\s*(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Look for function definitions
        if "def " in text:
            # Extract from first 'def' to end of function
            lines = text.split("\n")
            code_lines = []
            in_function = False
            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)

            return "\n".join(code_lines)

        return text

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

        # Remove docstrings
        code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)

        # Normalize whitespace
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        code = "\n".join(lines)

        return code

    def _code_structure_match(self, pred: str, ref: str) -> bool:
        """Check if code structures are similar."""
        # Extract function signature
        pred_funcs = re.findall(r"def\s+(\w+)\s*\(", pred)
        ref_funcs = re.findall(r"def\s+(\w+)\s*\(", ref)

        # Check if main function name matches
        if pred_funcs and ref_funcs:
            if pred_funcs[0] != ref_funcs[0]:
                return False

        # Check for key operations (very basic)
        key_ops = ["return", "for", "while", "if"]
        for op in key_ops:
            pred_has = op in pred
            ref_has = op in ref
            if pred_has != ref_has:
                return False

        return True


class DomainEvaluator:
    """
    Domain-specific evaluation utilities.
    """

    @staticmethod
    def evaluate_sql_execution(
        prediction: str,
        reference: str,
        db_path: str,
    ) -> bool:
        """
        Evaluate SQL by executing against a database.

        Args:
            prediction: Predicted SQL query.
            reference: Reference SQL query.
            db_path: Path to SQLite database.

        Returns:
            True if results match.
        """
        import sqlite3

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Execute reference query
            cursor.execute(reference)
            ref_result = set(cursor.fetchall())

            # Execute prediction
            cursor.execute(prediction)
            pred_result = set(cursor.fetchall())

            conn.close()

            return pred_result == ref_result

        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return False

    @staticmethod
    def evaluate_code_execution(
        code: str,
        test_cases: list[str],
        timeout: float = 5.0,
    ) -> tuple[bool, list[str]]:
        """
        Evaluate code by running test cases.

        Args:
            code: Python code to test.
            test_cases: List of assert statements.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple of (all_passed, list of failed tests).
        """
        import multiprocessing

        def run_tests(code: str, tests: list[str], result_queue):
            try:
                # Create execution namespace
                namespace = {}
                exec(code, namespace)

                failed = []
                for test in tests:
                    try:
                        exec(test, namespace)
                    except AssertionError:
                        failed.append(test)
                    except Exception as e:
                        failed.append(f"{test} (Error: {e})")

                result_queue.put((len(failed) == 0, failed))
            except Exception as e:
                result_queue.put((False, [f"Execution error: {e}"]))

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_tests,
            args=(code, test_cases, result_queue),
        )
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            return False, ["Timeout"]

        if not result_queue.empty():
            return result_queue.get()

        return False, ["Unknown error"]
