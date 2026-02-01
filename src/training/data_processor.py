"""
Data processing for training.
"""

from dataclasses import dataclass
from typing import Iterator

from datasets import Dataset
from loguru import logger


@dataclass
class TrainingExample:
    """A single training example."""

    query: str
    response: str
    domain: str
    system_prompt: str | None = None
    metadata: dict | None = None


class DataProcessor:
    """
    Processes data for LoRA training.

    Handles formatting training examples into the appropriate format
    for the student model's chat template.
    """

    def __init__(self, tokenizer, max_seq_length: int = 4096):
        """
        Initialize the data processor.

        Args:
            tokenizer: The tokenizer for the student model.
            max_seq_length: Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Domain-specific system prompts
        self.system_prompts = {
            "text_to_sql": """You are an expert SQL assistant. Convert natural language queries to SQL.
Think step by step, then provide the final SQL query in a code block.""",
            "math_reasoning": """You are a mathematics tutor. Solve problems step by step.
Show your work clearly and provide the final numerical answer.""",
            "code_generation": """You are an expert Python programmer.
Write clean, efficient, well-documented code with type hints.""",
            "general": """You are a helpful, accurate assistant.
Provide clear and well-structured responses.""",
        }

    def format_example(self, example: TrainingExample) -> str:
        """
        Format a training example using the chat template.

        Args:
            example: Training example to format.

        Returns:
            Formatted string for training.
        """
        system_prompt = example.system_prompt or self.system_prompts.get(
            example.domain, self.system_prompts["general"]
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.query},
            {"role": "assistant", "content": example.response},
        ]

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return formatted

    def prepare_dataset(
        self,
        examples: list[TrainingExample],
        domain: str | None = None,
    ) -> Dataset:
        """
        Prepare a HuggingFace Dataset from training examples.

        Args:
            examples: List of training examples.
            domain: Optional domain filter.

        Returns:
            HuggingFace Dataset ready for training.
        """
        if domain:
            examples = [ex for ex in examples if ex.domain == domain]

        # Format examples
        formatted_data = []
        for example in examples:
            formatted = self.format_example(example)

            # Check length
            tokens = self.tokenizer(formatted, truncation=False)
            if len(tokens["input_ids"]) > self.max_seq_length:
                logger.warning(
                    f"Example exceeds max length ({len(tokens['input_ids'])} > {self.max_seq_length}), truncating"
                )
                continue

            formatted_data.append({
                "text": formatted,
                "domain": example.domain,
            })

        logger.info(f"Prepared {len(formatted_data)} examples for training")

        return Dataset.from_list(formatted_data)

    def prepare_dataset_streaming(
        self,
        examples: Iterator[TrainingExample],
        batch_size: int = 100,
    ) -> Iterator[Dataset]:
        """
        Prepare datasets in streaming batches.

        Args:
            examples: Iterator of training examples.
            batch_size: Size of each batch.

        Yields:
            HuggingFace Datasets in batches.
        """
        batch = []

        for example in examples:
            formatted = self.format_example(example)

            tokens = self.tokenizer(formatted, truncation=False)
            if len(tokens["input_ids"]) <= self.max_seq_length:
                batch.append({
                    "text": formatted,
                    "domain": example.domain,
                })

            if len(batch) >= batch_size:
                yield Dataset.from_list(batch)
                batch = []

        if batch:
            yield Dataset.from_list(batch)

    @staticmethod
    def from_spider(
        data: list[dict],
        include_schema: bool = True,
        schemas: dict[str, dict] | None = None,
    ) -> list[TrainingExample]:
        """
        Convert Spider dataset entries to training examples.

        Args:
            data: List of Spider dataset entries.
            include_schema: Whether to include schema in the query.
            schemas: Optional dictionary of database schemas (db_id -> schema dict).
                     If not provided, will use 'prompt' field if available,
                     otherwise falls back to db_id only.

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        for entry in data:
            question = entry.get("question", "")
            query = entry.get("query", "")
            db_id = entry.get("db_id", "")

            # Determine the full query with context
            if include_schema:
                # Priority 1: Use pre-processed 'prompt' field if available (includes schema)
                if "prompt" in entry:
                    full_query = entry["prompt"]
                # Priority 2: Build schema from provided schemas dict
                elif schemas and db_id in schemas:
                    schema_str = DataProcessor._format_schema(schemas[db_id])
                    full_query = f"Given the following database schema:\n\n{schema_str}\n\nConvert this question to SQL: {question}"
                # Priority 3: Fallback to just db_id
                else:
                    full_query = f"Database: {db_id}\n\nQuestion: {question}"
            else:
                full_query = question

            # Format the SQL response
            response = f"Based on the question, here's the SQL query:\n\n```sql\n{query}\n```"

            examples.append(
                TrainingExample(
                    query=full_query,
                    response=response,
                    domain="text_to_sql",
                    metadata={"db_id": db_id},
                )
            )

        return examples

    @staticmethod
    def _format_schema(schema: dict) -> str:
        """Format a schema dictionary into a readable string."""
        tables = schema.get("tables", [])
        columns = schema.get("columns", [])
        column_types = schema.get("column_types", [])

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

    @staticmethod
    def from_gsm8k(data: list[dict]) -> list[TrainingExample]:
        """
        Convert GSM8K dataset entries to training examples.

        Args:
            data: List of GSM8K dataset entries.

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        for entry in data:
            question = entry.get("question", "")
            answer = entry.get("answer", "")

            # GSM8K answers include reasoning, format appropriately
            examples.append(
                TrainingExample(
                    query=question,
                    response=answer,
                    domain="math_reasoning",
                )
            )

        return examples

    @staticmethod
    def from_mbpp(data: list[dict]) -> list[TrainingExample]:
        """
        Convert MBPP dataset entries to training examples.

        Args:
            data: List of MBPP dataset entries.

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        for entry in data:
            prompt = entry.get("text", entry.get("prompt", ""))
            code = entry.get("code", "")
            test_list = entry.get("test_list", [])

            # Include test cases in prompt for clarity
            full_prompt = prompt
            if test_list:
                full_prompt += "\n\nTest cases:\n" + "\n".join(test_list[:3])

            response = f"Here's the Python solution:\n\n```python\n{code}\n```"

            examples.append(
                TrainingExample(
                    query=full_prompt,
                    response=response,
                    domain="code_generation",
                )
            )

        return examples

    @staticmethod
    def from_teacher_response(
        query: str,
        response: str,
        domain: str,
    ) -> TrainingExample:
        """
        Create a training example from a teacher model response.

        Args:
            query: The original query.
            response: The teacher's response.
            domain: The domain of the query.

        Returns:
            TrainingExample object.
        """
        return TrainingExample(
            query=query,
            response=response,
            domain=domain,
            metadata={"source": "teacher"},
        )
