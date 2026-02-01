"""
Dataset loader for various benchmark datasets.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict
from loguru import logger


@dataclass
class DomainDataset:
    """Container for a domain's train/test datasets."""

    domain: str
    train: Dataset
    test: Dataset
    description: str = ""


class DatasetLoader:
    """
    Loads and prepares datasets for different domains.

    Supported datasets:
    - Spider (Text-to-SQL) - local or HuggingFace
    - GSM8K (Mathematical Reasoning)
    - MBPP (Python Code Generation)
    - BIRD (Text-to-SQL, more challenging) - local or HuggingFace
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        spider_data_dir: str | Path | None = None,
        bird_data_dir: str | Path | None = None,
    ):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Directory for caching downloaded datasets.
            spider_data_dir: Path to local spider_data directory (if available).
            bird_data_dir: Path to local BIRD data directory (if available).
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.spider_data_dir = Path(spider_data_dir) if spider_data_dir else None
        self.bird_data_dir = Path(bird_data_dir) if bird_data_dir else None

        # Try to auto-detect spider_data if not provided
        if self.spider_data_dir is None:
            possible_paths = [
                Path("spider_data"),
                Path("data/spider_data"),
                Path("/data/mehmet/projects/mixture-of-LoRA/spider_data"),
            ]
            for path in possible_paths:
                if path.exists() and (path / "train_spider.json").exists():
                    self.spider_data_dir = path
                    break

        # Try to auto-detect bird_data if not provided
        if self.bird_data_dir is None:
            possible_paths = [
                Path("bird_data"),
                Path("data/bird_data"),
                Path("/data/mehmet/projects/mixture-of-LoRA/bird_data"),
            ]
            for path in possible_paths:
                if path.exists():
                    # Check for dev or train subdirectories
                    for subdir in path.iterdir():
                        if subdir.is_dir() and (subdir / "dev.json").exists():
                            self.bird_data_dir = path
                            break
                    if self.bird_data_dir:
                        break

        # Load schema information if available
        self._schemas: dict[str, dict] = {}
        self._bird_schemas: dict[str, dict] = {}
        if self.spider_data_dir and (self.spider_data_dir / "tables.json").exists():
            self._load_schemas()
        if self.bird_data_dir:
            self._load_bird_schemas()

        logger.info(f"DatasetLoader initialized (spider_data: {self.spider_data_dir}, bird_data: {self.bird_data_dir})")

    def _load_schemas(self) -> None:
        """Load database schemas from tables.json."""
        tables_path = self.spider_data_dir / "tables.json"
        try:
            with open(tables_path) as f:
                tables_data = json.load(f)

            for db in tables_data:
                db_id = db["db_id"]
                self._schemas[db_id] = {
                    "tables": db["table_names_original"],
                    "columns": db["column_names_original"],
                    "column_types": db["column_types"],
                    "primary_keys": db["primary_keys"],
                    "foreign_keys": db["foreign_keys"],
                }

            logger.info(f"Loaded schemas for {len(self._schemas)} databases")
        except Exception as e:
            logger.warning(f"Failed to load schemas: {e}")

    def _load_bird_schemas(self) -> None:
        """Load BIRD database schemas from dev_tables.json or train_tables.json."""
        # Find tables file in BIRD data directory
        tables_files = list(self.bird_data_dir.glob("**/dev_tables.json")) + \
                       list(self.bird_data_dir.glob("**/train_tables.json"))

        for tables_path in tables_files:
            try:
                with open(tables_path) as f:
                    tables_data = json.load(f)

                for db in tables_data:
                    db_id = db["db_id"]
                    self._bird_schemas[db_id] = {
                        "tables": db["table_names_original"],
                        "columns": db["column_names_original"],
                        "column_descriptions": db.get("column_names", db["column_names_original"]),
                        "column_types": db["column_types"],
                        "primary_keys": db.get("primary_keys", []),
                        "foreign_keys": db.get("foreign_keys", []),
                    }

                logger.info(f"Loaded BIRD schemas from {tables_path}: {len(tables_data)} databases")
            except Exception as e:
                logger.warning(f"Failed to load BIRD schemas from {tables_path}: {e}")

    def get_bird_schema_string(self, db_id: str) -> str:
        """
        Get a formatted schema string for a BIRD database in CREATE TABLE format.

        Args:
            db_id: Database identifier.

        Returns:
            Formatted schema string for prompts (SQLite CREATE TABLE syntax).
        """
        if db_id not in self._bird_schemas:
            return ""

        schema = self._bird_schemas[db_id]
        tables = schema["tables"]
        columns = schema["columns"]
        column_descriptions = schema.get("column_descriptions", columns)
        column_types = schema["column_types"]
        primary_keys = schema.get("primary_keys", [])
        foreign_keys = schema.get("foreign_keys", [])

        # Build column index to table mapping for FK resolution
        col_to_table = {}
        for col_idx, (table_idx, col_name) in enumerate(columns):
            if table_idx >= 0:
                col_to_table[col_idx] = (tables[table_idx], col_name)

        # Flatten primary keys (can be int or list)
        pk_set = set()
        for pk in primary_keys:
            if isinstance(pk, list):
                pk_set.update(pk)
            else:
                pk_set.add(pk)

        # Build FK mapping: column_idx -> (ref_table, ref_column)
        fk_map = {}
        for fk in foreign_keys:
            if len(fk) == 2:
                from_col, to_col = fk
                if to_col in col_to_table:
                    fk_map[from_col] = col_to_table[to_col]

        lines = []
        for table_idx, table_name in enumerate(tables):
            col_defs = []
            for col_idx, (col_table_idx, col_name) in enumerate(columns):
                if col_table_idx == table_idx:
                    col_type = column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                    col_type = col_type.upper()

                    # Get description if different from original name
                    desc = ""
                    if col_idx < len(column_descriptions):
                        _, desc_name = column_descriptions[col_idx]
                        if desc_name != col_name and desc_name != "*":
                            desc = f"  -- {desc_name}"

                    # Check if primary key
                    pk_suffix = " PRIMARY KEY" if col_idx in pk_set else ""

                    # Check if foreign key
                    fk_suffix = ""
                    if col_idx in fk_map:
                        ref_table, ref_col = fk_map[col_idx]
                        fk_suffix = f" REFERENCES {ref_table}({ref_col})"

                    col_defs.append(f"  {col_name} {col_type}{pk_suffix}{fk_suffix}{desc}")

            if col_defs:
                lines.append(f"CREATE TABLE {table_name} (")
                lines.append(",\n".join(col_defs))
                lines.append(");")
                lines.append("")

        return "\n".join(lines).strip()

    def get_schema_string(self, db_id: str) -> str:
        """
        Get a formatted schema string for a database.

        Args:
            db_id: Database identifier.

        Returns:
            Formatted schema string for prompts.
        """
        if db_id not in self._schemas:
            return ""

        schema = self._schemas[db_id]
        tables = schema["tables"]
        columns = schema["columns"]
        column_types = schema["column_types"]

        lines = []
        for table_idx, table_name in enumerate(tables):
            if isinstance(table_name, list):
                table_name = table_name[1] if len(table_name) > 1 else table_name[0]

            # Get columns for this table
            table_columns = []
            for col_idx, (col_table_idx, col_name) in enumerate(columns):
                if col_table_idx == table_idx:
                    col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
                    table_columns.append(f"{col_name} ({col_type})")

            if table_columns:
                lines.append(f"Table: {table_name}")
                lines.append(f"  Columns: {', '.join(table_columns)}")

        return "\n".join(lines)

    def load_spider(
        self,
        split_ratio: float = 0.9,
        max_samples: int | None = None,
        include_schema: bool = True,
        use_local: bool = True,
    ) -> DomainDataset:
        """
        Load the Spider text-to-SQL dataset.

        Args:
            split_ratio: Train/test split ratio.
            max_samples: Maximum samples to load (for testing).
            include_schema: Whether to include schema context in prompts.
            use_local: Whether to use local spider_data files (if available).

        Returns:
            DomainDataset with train and test splits.
        """
        logger.info("Loading Spider dataset...")

        # Try local loading first if available and requested
        if use_local and self.spider_data_dir:
            try:
                return self._load_spider_local(max_samples, include_schema)
            except Exception as e:
                logger.warning(f"Failed to load local Spider: {e}, falling back to HF")

        try:
            # Load Spider dataset from HuggingFace
            dataset = load_dataset(
                "xlangai/spider",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            train_data = dataset["train"]
            # Spider has a validation set we can use as test
            test_data = dataset.get("validation", dataset["train"])

            if max_samples:
                train_data = train_data.select(range(min(max_samples, len(train_data))))
                test_data = test_data.select(range(min(max_samples // 5, len(test_data))))

            logger.info(
                f"Spider loaded: {len(train_data)} train, {len(test_data)} test samples"
            )

            return DomainDataset(
                domain="text_to_sql",
                train=train_data,
                test=test_data,
                description="Spider text-to-SQL benchmark",
            )

        except Exception as e:
            logger.warning(f"Failed to load Spider from HF: {e}")
            return self._create_synthetic_sql_dataset(max_samples or 1000)

    def _load_spider_local(
        self,
        max_samples: int | None = None,
        include_schema: bool = True,
    ) -> DomainDataset:
        """
        Load Spider dataset from local files.

        Args:
            max_samples: Maximum samples to load.
            include_schema: Whether to include schema context.

        Returns:
            DomainDataset with train and test splits.
        """
        train_path = self.spider_data_dir / "train_spider.json"
        dev_path = self.spider_data_dir / "dev.json"

        logger.info(f"Loading Spider from local files: {self.spider_data_dir}")

        with open(train_path) as f:
            train_raw = json.load(f)

        with open(dev_path) as f:
            dev_raw = json.load(f)

        # Process examples with schema context
        train_examples = self._process_spider_examples(train_raw, include_schema)
        dev_examples = self._process_spider_examples(dev_raw, include_schema)

        if max_samples:
            train_examples = train_examples[:max_samples]
            # Use same max_samples for test set (not // 5 which was a bug)
            dev_examples = dev_examples[:max_samples]

        train_data = Dataset.from_list(train_examples)
        test_data = Dataset.from_list(dev_examples)

        logger.info(
            f"Spider loaded from local: {len(train_data)} train, {len(test_data)} test samples"
        )

        return DomainDataset(
            domain="text_to_sql",
            train=train_data,
            test=test_data,
            description="Spider text-to-SQL benchmark (local)",
        )

    def _process_spider_examples(
        self,
        examples: list[dict],
        include_schema: bool = True,
    ) -> list[dict]:
        """
        Process Spider examples, optionally adding schema context.

        Args:
            examples: Raw Spider examples.
            include_schema: Whether to include schema.

        Returns:
            Processed examples with schema-augmented prompts.
        """
        processed = []

        for ex in examples:
            db_id = ex["db_id"]
            question = ex["question"]
            query = ex["query"]

            # Build the prompt with optional schema
            if include_schema and db_id in self._schemas:
                schema_str = self.get_schema_string(db_id)
                prompt = f"""Given the following database schema:

{schema_str}

Convert this question to SQL: {question}"""
            else:
                prompt = f"Convert this question to SQL: {question}"

            # Create training text in chat format
            text = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{query}<|im_end|>"""

            processed.append({
                "question": question,
                "query": query,
                "db_id": db_id,
                "text": text,
                "prompt": prompt,
            })

        return processed

    def load_gsm8k(
        self,
        max_samples: int | None = None,
    ) -> DomainDataset:
        """
        Load the GSM8K mathematical reasoning dataset.

        Args:
            max_samples: Maximum samples to load.

        Returns:
            DomainDataset with train and test splits.
        """
        logger.info("Loading GSM8K dataset...")

        try:
            dataset = load_dataset(
                "openai/gsm8k",
                "main",
                cache_dir=self.cache_dir,
            )

            train_data = dataset["train"]
            test_data = dataset["test"]

            if max_samples:
                train_data = train_data.select(range(min(max_samples, len(train_data))))
                test_data = test_data.select(range(min(max_samples // 5, len(test_data))))

            logger.info(
                f"GSM8K loaded: {len(train_data)} train, {len(test_data)} test samples"
            )

            return DomainDataset(
                domain="math_reasoning",
                train=train_data,
                test=test_data,
                description="GSM8K grade school math problems",
            )

        except Exception as e:
            logger.warning(f"Failed to load GSM8K: {e}")
            return self._create_synthetic_math_dataset(max_samples or 1000)

    def load_mbpp(
        self,
        max_samples: int | None = None,
    ) -> DomainDataset:
        """
        Load the MBPP Python code generation dataset.

        Args:
            max_samples: Maximum samples to load.

        Returns:
            DomainDataset with train and test splits.
        """
        logger.info("Loading MBPP dataset...")

        try:
            dataset = load_dataset(
                "mbpp",
                "full",
                cache_dir=self.cache_dir,
            )

            train_data = dataset["train"]
            test_data = dataset["test"]

            if max_samples:
                train_data = train_data.select(range(min(max_samples, len(train_data))))
                test_data = test_data.select(range(min(max_samples // 5, len(test_data))))

            logger.info(
                f"MBPP loaded: {len(train_data)} train, {len(test_data)} test samples"
            )

            return DomainDataset(
                domain="code_generation",
                train=train_data,
                test=test_data,
                description="MBPP Python programming problems",
            )

        except Exception as e:
            logger.warning(f"Failed to load MBPP: {e}")
            return self._create_synthetic_code_dataset(max_samples or 1000)

    def load_bird(
        self,
        max_samples: int | None = None,
        include_schema: bool = True,
        include_evidence: bool = True,
        use_local: bool = False,
        use_cleaned_hf: bool = True,
    ) -> DomainDataset:
        """
        Load the BIRD text-to-SQL dataset (more challenging than Spider).

        Args:
            max_samples: Maximum samples to load.
            include_schema: Whether to include schema context in prompts.
            include_evidence: Whether to include evidence hints in prompts.
            use_local: Whether to use local BIRD data files (if available).
            use_cleaned_hf: Whether to use cleaned HuggingFace datasets (recommended).

        Returns:
            DomainDataset with train and test splits.
        """
        logger.info("Loading BIRD dataset...")

        # Use cleaned HuggingFace datasets (recommended)
        if use_cleaned_hf:
            try:
                return self._load_bird_hf_cleaned(max_samples, include_schema, include_evidence)
            except Exception as e:
                logger.warning(f"Failed to load cleaned BIRD from HF: {e}, trying local")

        # Try local loading if available and requested
        if use_local and self.bird_data_dir:
            try:
                return self._load_bird_local(max_samples, include_schema, include_evidence)
            except Exception as e:
                logger.warning(f"Failed to load local BIRD: {e}, falling back to HF")

        try:
            # BIRD dataset from HuggingFace (original)
            dataset = load_dataset(
                "DAMO-NLP-SG/bird",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            train_data = dataset["train"]
            test_data = dataset.get("dev", dataset["train"])

            if max_samples:
                train_data = train_data.select(range(min(max_samples, len(train_data))))
                test_data = test_data.select(range(min(max_samples, len(test_data))))

            logger.info(
                f"BIRD loaded: {len(train_data)} train, {len(test_data)} test samples"
            )

            return DomainDataset(
                domain="text_to_sql",
                train=train_data,
                test=test_data,
                description="BIRD text-to-SQL benchmark",
            )

        except Exception as e:
            logger.warning(f"Failed to load BIRD: {e}")
            return self._create_synthetic_sql_dataset(max_samples or 1000)

    def _load_bird_hf_cleaned(
        self,
        max_samples: int | None = None,
        include_schema: bool = True,
        include_evidence: bool = True,
    ) -> DomainDataset:
        """
        Load cleaned BIRD datasets from HuggingFace.

        Uses:
        - birdsql/bird23-train-filtered (6601 train samples)
        - birdsql/bird_sql_dev_20251106 (1534 dev samples)

        Args:
            max_samples: Maximum samples to load.
            include_schema: Whether to include schema context.
            include_evidence: Whether to include evidence hints.

        Returns:
            DomainDataset with train and test splits.
        """
        logger.info("Loading cleaned BIRD datasets from HuggingFace...")

        # Load train dataset
        train_ds = load_dataset("birdsql/bird23-train-filtered", cache_dir=self.cache_dir)
        train_raw = list(train_ds["train"])

        # Load dev dataset
        dev_ds = load_dataset("birdsql/bird_sql_dev_20251106", cache_dir=self.cache_dir)
        dev_raw = list(dev_ds["dev_20251106"])

        # Process examples
        train_examples = self._process_bird_examples(train_raw, include_schema, include_evidence)
        dev_examples = self._process_bird_examples(dev_raw, include_schema, include_evidence)

        if max_samples:
            train_examples = train_examples[:max_samples]
            dev_examples = dev_examples[:max_samples]

        train_data = Dataset.from_list(train_examples)
        test_data = Dataset.from_list(dev_examples)

        logger.info(
            f"BIRD loaded from HF (cleaned): {len(train_data)} train, {len(test_data)} test samples"
        )

        return DomainDataset(
            domain="text_to_sql",
            train=train_data,
            test=test_data,
            description="BIRD text-to-SQL benchmark (cleaned HF)",
        )

    def _load_bird_local(
        self,
        max_samples: int | None = None,
        include_schema: bool = True,
        include_evidence: bool = True,
    ) -> DomainDataset:
        """
        Load BIRD dataset from local files.

        Args:
            max_samples: Maximum samples to load.
            include_schema: Whether to include schema context.
            include_evidence: Whether to include evidence hints.

        Returns:
            DomainDataset with train and test splits.
        """
        # Find the data directories
        train_json = None
        dev_json = None

        for subdir in self.bird_data_dir.iterdir():
            if not subdir.is_dir():
                continue
            if (subdir / "train.json").exists():
                train_json = subdir / "train.json"
            if (subdir / "dev.json").exists():
                dev_json = subdir / "dev.json"

        if dev_json is None:
            raise FileNotFoundError("Could not find dev.json in BIRD data directory")

        logger.info(f"Loading BIRD from local files: train={train_json}, dev={dev_json}")

        # Load dev data (always available)
        with open(dev_json) as f:
            dev_raw = json.load(f)
        dev_examples = self._process_bird_examples(dev_raw, include_schema, include_evidence)

        # Load train data if available
        if train_json and train_json.exists():
            with open(train_json) as f:
                train_raw = json.load(f)
            train_examples = self._process_bird_examples(train_raw, include_schema, include_evidence)
        else:
            # Use part of dev for training if train not available
            logger.warning("BIRD train.json not found, using dev data for both train and test")
            train_examples = dev_examples

        if max_samples:
            train_examples = train_examples[:max_samples]
            dev_examples = dev_examples[:max_samples]

        train_data = Dataset.from_list(train_examples)
        test_data = Dataset.from_list(dev_examples)

        logger.info(
            f"BIRD loaded from local: {len(train_data)} train, {len(test_data)} test samples"
        )

        return DomainDataset(
            domain="text_to_sql",
            train=train_data,
            test=test_data,
            description="BIRD text-to-SQL benchmark (local)",
        )

    def _process_bird_examples(
        self,
        examples: list[dict],
        include_schema: bool = True,
        include_evidence: bool = True,
    ) -> list[dict]:
        """
        Process BIRD examples, optionally adding schema context and evidence.

        Args:
            examples: Raw BIRD examples.
            include_schema: Whether to include schema.
            include_evidence: Whether to include evidence hints.

        Returns:
            Processed examples with schema-augmented prompts.
        """
        processed = []

        for ex in examples:
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex["SQL"]
            evidence = ex.get("evidence", "")
            difficulty = ex.get("difficulty", "unknown")

            # Build the prompt with optional schema and evidence
            prompt_parts = []

            if include_schema and db_id in self._bird_schemas:
                schema_str = self.get_bird_schema_string(db_id)
                prompt_parts.append(f"Database schema:\n{schema_str}")

            if include_evidence and evidence:
                prompt_parts.append(f"Hint: {evidence}")

            prompt_parts.append(f"Question: {question}")
            prompt = "\n\n".join(prompt_parts)

            # Create training text in chat format
            text = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{sql}<|im_end|>"""

            processed.append({
                "question": question,
                "query": sql,
                "SQL": sql,
                "db_id": db_id,
                "evidence": evidence,
                "difficulty": difficulty,
                "text": text,
                "prompt": prompt,
            })

        return processed

    def get_bird_db_path(self, db_id: str, split: str = "dev") -> Path | None:
        """
        Get the path to a BIRD database file.

        Args:
            db_id: Database identifier.
            split: Data split ("dev" or "train").

        Returns:
            Path to the SQLite database file, or None if not found.
        """
        if self.bird_data_dir is None:
            return None

        # Search for the database file
        patterns = [
            f"**/dev_databases/{db_id}/{db_id}.sqlite",
            f"**/train_databases/{db_id}/{db_id}.sqlite",
            f"**/{db_id}/{db_id}.sqlite",
        ]

        for pattern in patterns:
            matches = list(self.bird_data_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def load_all(
        self,
        max_samples_per_domain: int | None = None,
    ) -> dict[str, DomainDataset]:
        """
        Load all supported datasets.

        Args:
            max_samples_per_domain: Maximum samples per domain.

        Returns:
            Dictionary mapping domain names to DomainDataset objects.
        """
        datasets = {}

        # Load each dataset
        datasets["text_to_sql"] = self.load_spider(max_samples=max_samples_per_domain)
        datasets["math_reasoning"] = self.load_gsm8k(max_samples=max_samples_per_domain)
        datasets["code_generation"] = self.load_mbpp(max_samples=max_samples_per_domain)

        logger.info(f"Loaded {len(datasets)} domain datasets")

        return datasets

    def _create_synthetic_sql_dataset(self, num_samples: int) -> DomainDataset:
        """Create synthetic SQL dataset for testing."""
        logger.info(f"Creating synthetic SQL dataset with {num_samples} samples")

        examples = []
        templates = [
            ("Find all {entity} where {column} is greater than {value}",
             "SELECT * FROM {table} WHERE {column} > {value}"),
            ("Count the number of {entity} in the database",
             "SELECT COUNT(*) FROM {table}"),
            ("Get the average {column} for each {group}",
             "SELECT {group}, AVG({column}) FROM {table} GROUP BY {group}"),
            ("List all {entity} ordered by {column}",
             "SELECT * FROM {table} ORDER BY {column}"),
            ("Find {entity} with {column} equal to '{value}'",
             "SELECT * FROM {table} WHERE {column} = '{value}'"),
        ]

        entities = ["customers", "products", "orders", "employees", "sales"]
        columns = ["name", "price", "quantity", "date", "status", "category"]
        values = ["100", "active", "2024", "premium"]

        import random

        for i in range(num_samples):
            template = random.choice(templates)
            entity = random.choice(entities)
            column = random.choice(columns)
            value = random.choice(values)
            group = random.choice(columns)

            question = template[0].format(
                entity=entity, column=column, value=value,
                table=entity, group=group
            )
            query = template[1].format(
                entity=entity, column=column, value=value,
                table=entity, group=group
            )

            examples.append({
                "question": question,
                "query": query,
                "db_id": f"db_{i % 10}",
            })

        # Split train/test
        split_idx = int(len(examples) * 0.9)
        train_data = Dataset.from_list(examples[:split_idx])
        test_data = Dataset.from_list(examples[split_idx:])

        return DomainDataset(
            domain="text_to_sql",
            train=train_data,
            test=test_data,
            description="Synthetic SQL dataset for testing",
        )

    def _create_synthetic_math_dataset(self, num_samples: int) -> DomainDataset:
        """Create synthetic math dataset for testing."""
        logger.info(f"Creating synthetic math dataset with {num_samples} samples")

        import random

        examples = []

        templates = [
            ("If John has {a} apples and Mary gives him {b} more, how many apples does John have?",
             "John starts with {a} apples.\nMary gives him {b} more apples.\n{a} + {b} = {result}\n#### {result}"),
            ("A store has {a} items. If {b} items are sold, how many remain?",
             "The store starts with {a} items.\n{b} items are sold.\n{a} - {b} = {result}\n#### {result}"),
            ("Each box contains {a} items. How many items are in {b} boxes?",
             "Each box has {a} items.\nThere are {b} boxes.\n{a} × {b} = {result}\n#### {result}"),
        ]

        for i in range(num_samples):
            template = random.choice(templates)
            a = random.randint(5, 100)
            b = random.randint(1, 50)

            if "-" in template[1]:
                result = a - b
            elif "×" in template[1]:
                result = a * b
            else:
                result = a + b

            question = template[0].format(a=a, b=b)
            answer = template[1].format(a=a, b=b, result=result)

            examples.append({
                "question": question,
                "answer": answer,
            })

        split_idx = int(len(examples) * 0.9)
        train_data = Dataset.from_list(examples[:split_idx])
        test_data = Dataset.from_list(examples[split_idx:])

        return DomainDataset(
            domain="math_reasoning",
            train=train_data,
            test=test_data,
            description="Synthetic math dataset for testing",
        )

    def _create_synthetic_code_dataset(self, num_samples: int) -> DomainDataset:
        """Create synthetic code dataset for testing."""
        logger.info(f"Creating synthetic code dataset with {num_samples} samples")

        examples = []

        code_templates = [
            {
                "text": "Write a function to calculate the factorial of a number.",
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "test_list": ["assert factorial(5) == 120", "assert factorial(0) == 1"],
            },
            {
                "text": "Write a function to check if a string is a palindrome.",
                "code": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                "test_list": ["assert is_palindrome('radar') == True", "assert is_palindrome('hello') == False"],
            },
            {
                "text": "Write a function to find the maximum element in a list.",
                "code": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)",
                "test_list": ["assert find_max([1, 5, 3]) == 5", "assert find_max([]) == None"],
            },
            {
                "text": "Write a function to reverse a string.",
                "code": "def reverse_string(s):\n    return s[::-1]",
                "test_list": ["assert reverse_string('hello') == 'olleh'"],
            },
            {
                "text": "Write a function to calculate the sum of a list of numbers.",
                "code": "def list_sum(numbers):\n    return sum(numbers)",
                "test_list": ["assert list_sum([1, 2, 3]) == 6", "assert list_sum([]) == 0"],
            },
        ]

        import random

        for i in range(num_samples):
            template = random.choice(code_templates)
            examples.append(template.copy())

        split_idx = int(len(examples) * 0.9)
        train_data = Dataset.from_list(examples[:split_idx])
        test_data = Dataset.from_list(examples[split_idx:])

        return DomainDataset(
            domain="code_generation",
            train=train_data,
            test=test_data,
            description="Synthetic code dataset for testing",
        )


def get_dataset_info() -> dict:
    """
    Get information about available datasets.

    Returns:
        Dictionary with dataset information.
    """
    return {
        "spider": {
            "name": "Spider",
            "domain": "text_to_sql",
            "description": "Large-scale complex and cross-domain text-to-SQL benchmark",
            "size": "~10K examples",
            "source": "xlangai/spider",
        },
        "gsm8k": {
            "name": "GSM8K",
            "domain": "math_reasoning",
            "description": "Grade school math word problems requiring multi-step reasoning",
            "size": "~8.5K train, 1.3K test",
            "source": "openai/gsm8k",
        },
        "mbpp": {
            "name": "MBPP",
            "domain": "code_generation",
            "description": "Mostly Basic Python Problems for code generation",
            "size": "~1K examples",
            "source": "mbpp",
        },
        "bird": {
            "name": "BIRD",
            "domain": "text_to_sql",
            "description": "Big Bench for Large-scale Database Grounded Text-to-SQL",
            "size": "~12K examples",
            "source": "DAMO-NLP-SG/bird",
        },
    }
