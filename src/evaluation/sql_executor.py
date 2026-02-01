"""
SQL execution-based evaluation for text-to-SQL tasks.

Uses actual database execution to compare predicted and reference SQL queries.
"""

import re
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger


class SQLExecutor:
    """
    Executes SQL queries against Spider databases and compares results.

    This provides a more accurate evaluation than exact string matching,
    as semantically equivalent SQL queries may have different syntax.
    """

    def __init__(self, database_dir: str | Path):
        """
        Initialize the SQL executor.

        Args:
            database_dir: Path to the Spider database directory.
        """
        self.database_dir = Path(database_dir)

        if not self.database_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {self.database_dir}")

        self._db_cache: dict[str, sqlite3.Connection] = {}
        logger.info(f"SQLExecutor initialized with database dir: {self.database_dir}")

    def get_db_path(self, db_id: str) -> Path | None:
        """
        Get the path to a database file.

        Args:
            db_id: Database identifier.

        Returns:
            Path to the database file or None if not found.
        """
        # Try common patterns
        patterns = [
            self.database_dir / db_id / f"{db_id}.sqlite",
            self.database_dir / db_id / f"{db_id}.db",
            self.database_dir / f"{db_id}.sqlite",
            self.database_dir / f"{db_id}.db",
        ]

        for path in patterns:
            if path.exists():
                return path

        return None

    def get_connection(self, db_id: str) -> sqlite3.Connection | None:
        """
        Get a database connection, using cache if available.

        Args:
            db_id: Database identifier.

        Returns:
            SQLite connection or None if database not found.
        """
        if db_id in self._db_cache:
            return self._db_cache[db_id]

        db_path = self.get_db_path(db_id)
        if db_path is None:
            logger.warning(f"Database not found: {db_id}")
            return None

        try:
            conn = sqlite3.connect(str(db_path))
            conn.text_factory = str
            self._db_cache[db_id] = conn
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database {db_id}: {e}")
            return None

    def execute_sql(
        self,
        sql: str,
        db_id: str,
        timeout: float = 30.0
    ) -> tuple[bool, Any]:
        """
        Execute a SQL query and return results.

        Args:
            sql: SQL query to execute.
            db_id: Database identifier.
            timeout: Query timeout in seconds.

        Returns:
            Tuple of (success, results or error message).
        """
        conn = self.get_connection(db_id)
        if conn is None:
            return False, f"Database not found: {db_id}"

        try:
            # Clean the SQL
            sql = self._clean_sql(sql)

            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()

            return True, results
        except sqlite3.Error as e:
            return False, f"SQL error: {e}"
        except Exception as e:
            return False, f"Execution error: {e}"

    def _clean_sql(self, sql: str) -> str:
        """
        Clean and normalize SQL for execution.

        Args:
            sql: Raw SQL string.

        Returns:
            Cleaned SQL string.
        """
        # Extract SQL from code blocks
        sql = self._extract_sql_from_text(sql)

        # Remove leading/trailing whitespace
        sql = sql.strip()

        # Remove trailing semicolon (SQLite doesn't need it)
        sql = sql.rstrip(';')

        return sql

    def _extract_sql_from_text(self, text: str) -> str:
        """
        Extract SQL query from text that may contain markdown or explanations.

        Args:
            text: Text containing SQL.

        Returns:
            Extracted SQL query.
        """
        # Try to extract from code block
        sql_block_pattern = r"```(?:sql)?\s*(.*?)```"
        matches = re.findall(sql_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # Try to find SQL statement
        sql_pattern = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+.+"
        match = re.search(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()

        return text

    def compare_results(
        self,
        results1: list[tuple],
        results2: list[tuple],
        order_matters: bool = False
    ) -> bool:
        """
        Compare two sets of SQL results.

        Args:
            results1: First result set.
            results2: Second result set.
            order_matters: Whether row order should be considered.

        Returns:
            True if results are equivalent.
        """
        if order_matters:
            return results1 == results2

        # Convert to sets of tuples for order-independent comparison
        # Handle potential unhashable types by converting to strings
        try:
            set1 = set(tuple(str(v) for v in row) for row in results1)
            set2 = set(tuple(str(v) for v in row) for row in results2)
            return set1 == set2
        except Exception:
            # Fallback to sorted comparison
            sorted1 = sorted([tuple(str(v) for v in row) for row in results1])
            sorted2 = sorted([tuple(str(v) for v in row) for row in results2])
            return sorted1 == sorted2

    def evaluate_single(
        self,
        prediction: str,
        reference: str,
        db_id: str,
        order_matters: bool = False
    ) -> tuple[bool, dict]:
        """
        Evaluate a single prediction against reference.

        Args:
            prediction: Predicted SQL query.
            reference: Reference SQL query.
            db_id: Database identifier.
            order_matters: Whether result order matters.

        Returns:
            Tuple of (is_correct, details dict).
        """
        details = {
            "db_id": db_id,
            "prediction": prediction[:500],
            "reference": reference[:500],
            "pred_error": None,
            "ref_error": None,
        }

        # Execute reference query
        ref_success, ref_results = self.execute_sql(reference, db_id)
        if not ref_success:
            details["ref_error"] = str(ref_results)
            # If reference fails, we can't evaluate
            return False, details

        # Execute prediction query
        pred_success, pred_results = self.execute_sql(prediction, db_id)
        if not pred_success:
            details["pred_error"] = str(pred_results)
            return False, details

        # Compare results
        is_correct = self.compare_results(pred_results, ref_results, order_matters)

        details["pred_rows"] = len(pred_results)
        details["ref_rows"] = len(ref_results)
        details["match"] = is_correct

        return is_correct, details

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str],
        db_ids: list[str],
        order_matters: bool = False
    ) -> dict:
        """
        Evaluate a batch of predictions.

        Args:
            predictions: List of predicted SQL queries.
            references: List of reference SQL queries.
            db_ids: List of database identifiers.
            order_matters: Whether result order matters.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not (len(predictions) == len(references) == len(db_ids)):
            raise ValueError("All input lists must have the same length")

        correct = 0
        execution_errors = 0
        details_list = []

        for pred, ref, db_id in zip(predictions, references, db_ids):
            is_correct, details = self.evaluate_single(pred, ref, db_id, order_matters)

            if is_correct:
                correct += 1

            if details.get("pred_error"):
                execution_errors += 1

            details_list.append(details)

        total = len(predictions)

        return {
            "execution_accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "execution_errors": execution_errors,
            "execution_error_rate": execution_errors / total if total > 0 else 0.0,
            "details": details_list[:10],  # Keep first 10 for inspection
        }

    def close(self):
        """Close all database connections."""
        for conn in self._db_cache.values():
            try:
                conn.close()
            except Exception:
                pass
        self._db_cache.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def get_spider_executor(spider_data_dir: str | Path | None = None) -> SQLExecutor | None:
    """
    Get a SQLExecutor configured for Spider dataset.

    Args:
        spider_data_dir: Path to spider_data directory. Auto-detects if None.

    Returns:
        SQLExecutor instance or None if not found.
    """
    if spider_data_dir is None:
        # Try common locations
        possible_paths = [
            Path("spider_data/database"),
            Path("data/spider_data/database"),
            Path("/data/mehmet/projects/mixture-of-LoRA/spider_data/database"),
        ]

        for path in possible_paths:
            if path.exists():
                spider_data_dir = path
                break

    if spider_data_dir is None:
        logger.warning("Spider database directory not found")
        return None

    return SQLExecutor(spider_data_dir)


def get_bird_executor(bird_data_dir: str | Path | None = None, split: str = "dev") -> SQLExecutor | None:
    """
    Get a SQLExecutor configured for BIRD dataset.

    Args:
        bird_data_dir: Path to bird_data directory. Auto-detects if None.
        split: Data split ("dev" or "train").

    Returns:
        SQLExecutor instance or None if not found.
    """
    if bird_data_dir is None:
        # Try common locations
        possible_paths = [
            Path("bird_data"),
            Path("data/bird_data"),
            Path("/data/mehmet/projects/mixture-of-LoRA/bird_data"),
        ]

        for path in possible_paths:
            if path.exists():
                bird_data_dir = path
                break

    if bird_data_dir is None:
        logger.warning("BIRD data directory not found")
        return None

    bird_data_dir = Path(bird_data_dir)

    # Find the databases directory for the given split
    patterns = [
        f"**/{split}_databases",
        f"**/dev_databases",  # fallback
    ]

    for pattern in patterns:
        matches = list(bird_data_dir.glob(pattern))
        if matches:
            return SQLExecutor(matches[0])

    logger.warning(f"BIRD {split}_databases directory not found in {bird_data_dir}")
    return None


class BIRDExecutor:
    """
    SQL executor that handles both BIRD train and dev databases.

    Automatically routes queries to the correct database directory based on db_id.
    """

    def __init__(self, bird_data_dir: str | Path | None = None):
        """
        Initialize BIRD executor with both train and dev database directories.

        Args:
            bird_data_dir: Path to bird_data directory. Auto-detects if None.
        """
        if bird_data_dir is None:
            possible_paths = [
                Path("bird_data"),
                Path("data/bird_data"),
                Path("/data/mehmet/projects/mixture-of-LoRA/bird_data"),
            ]
            for path in possible_paths:
                if path.exists():
                    bird_data_dir = path
                    break

        if bird_data_dir is None:
            raise FileNotFoundError("BIRD data directory not found")

        self.bird_data_dir = Path(bird_data_dir)

        # Find all database directories
        self._db_dirs: list[Path] = []
        for pattern in ["**/train_databases", "**/dev_databases"]:
            self._db_dirs.extend(self.bird_data_dir.glob(pattern))

        # Build db_id -> path mapping
        self._db_paths: dict[str, Path] = {}
        for db_dir in self._db_dirs:
            for db_path in db_dir.iterdir():
                if db_path.is_dir():
                    db_id = db_path.name
                    sqlite_file = db_path / f"{db_id}.sqlite"
                    if sqlite_file.exists():
                        self._db_paths[db_id] = sqlite_file

        self._db_cache: dict[str, sqlite3.Connection] = {}
        logger.info(f"BIRDExecutor initialized with {len(self._db_paths)} databases")

    def get_connection(self, db_id: str) -> sqlite3.Connection | None:
        """Get database connection for a db_id."""
        if db_id in self._db_cache:
            return self._db_cache[db_id]

        if db_id not in self._db_paths:
            logger.warning(f"Database not found: {db_id}")
            return None

        try:
            conn = sqlite3.connect(str(self._db_paths[db_id]))
            conn.text_factory = str
            self._db_cache[db_id] = conn
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database {db_id}: {e}")
            return None

    def execute_sql(self, sql: str, db_id: str, timeout: float = 30.0) -> tuple[bool, Any]:
        """Execute SQL query and return results."""
        conn = self.get_connection(db_id)
        if conn is None:
            return False, f"Database not found: {db_id}"

        try:
            # Clean the SQL
            sql = self._clean_sql(sql)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return True, results
        except sqlite3.Error as e:
            return False, f"SQL error: {e}"
        except Exception as e:
            return False, f"Execution error: {e}"

    def _clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL for execution."""
        # Extract SQL from code blocks
        sql_block_pattern = r"```(?:sql)?\s*(.*?)```"
        matches = re.findall(sql_block_pattern, sql, re.DOTALL | re.IGNORECASE)
        if matches:
            sql = matches[-1].strip()
        else:
            # Try to find SQL statement
            sql_pattern = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\s+.+"
            match = re.search(sql_pattern, sql, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(0).strip()

        sql = sql.strip().rstrip(';')
        return sql

    def compare_results(self, results1: list[tuple], results2: list[tuple]) -> bool:
        """Compare two sets of SQL results (order-independent)."""
        try:
            set1 = set(tuple(str(v) for v in row) for row in results1)
            set2 = set(tuple(str(v) for v in row) for row in results2)
            return set1 == set2
        except Exception:
            sorted1 = sorted([tuple(str(v) for v in row) for row in results1])
            sorted2 = sorted([tuple(str(v) for v in row) for row in results2])
            return sorted1 == sorted2

    def evaluate_single(self, prediction: str, reference: str, db_id: str) -> tuple[bool, dict]:
        """Evaluate a single prediction against reference."""
        details = {
            "db_id": db_id,
            "prediction": prediction[:500],
            "reference": reference[:500],
            "pred_error": None,
            "ref_error": None,
        }

        # Execute reference query
        ref_success, ref_results = self.execute_sql(reference, db_id)
        if not ref_success:
            details["ref_error"] = str(ref_results)
            return False, details

        # Execute prediction query
        pred_success, pred_results = self.execute_sql(prediction, db_id)
        if not pred_success:
            details["pred_error"] = str(pred_results)
            return False, details

        # Compare results
        is_correct = self.compare_results(pred_results, ref_results)
        details["pred_rows"] = len(pred_results)
        details["ref_rows"] = len(ref_results)
        details["match"] = is_correct

        return is_correct, details

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str],
        db_ids: list[str],
    ) -> dict:
        """Evaluate a batch of predictions."""
        correct = 0
        execution_errors = 0
        details_list = []

        for pred, ref, db_id in zip(predictions, references, db_ids):
            is_correct, details = self.evaluate_single(pred, ref, db_id)
            if is_correct:
                correct += 1
            if details.get("pred_error"):
                execution_errors += 1
            details_list.append(details)

        total = len(predictions)
        return {
            "execution_accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "execution_errors": execution_errors,
            "execution_error_rate": execution_errors / total if total > 0 else 0.0,
            "details": details_list[:10],
        }

    def close(self):
        """Close all database connections."""
        for conn in self._db_cache.values():
            try:
                conn.close()
            except Exception:
                pass
        self._db_cache.clear()
