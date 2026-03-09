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

    def __init__(self, tokenizer, max_seq_length: int = 8192):
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
            "text_to_sql": """You are an expert SQL assistant. Given a database schema and a natural language question, output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL.""",
            "text_to_sql_bird": """You are an expert SQL assistant. Given a database schema and a natural language question, output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL.""",
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

        For BIRD (text_to_sql_bird), uses 1-message format matching LoRA_SGD:
        [USER_TOKEN] rich_prompt [ASSISTANT_TOKEN] raw_SQL [EOS]
        The rich prompt already contains all instructions inline.

        For other domains, uses 3-message format: system + user + assistant.

        Args:
            example: Training example to format.

        Returns:
            Formatted string for training.
        """
        if example.domain == "text_to_sql_bird":
            # 1-message format: user only, then raw SQL appended after generation prompt
            messages = [{"role": "user", "content": example.query}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted_prompt + example.response + self.tokenizer.eos_token

        # Default 3-message format for other domains
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

    def format_example_split(self, example: TrainingExample) -> dict:
        """
        Format a training example into prompt/completion split.

        SFTTrainer computes loss only on the completion tokens when data
        is provided in {"prompt": ..., "completion": ...} format.

        Args:
            example: Training example to format.

        Returns:
            Dict with "prompt" and "completion" keys.
        """
        if example.domain == "text_to_sql_bird":
            messages = [{"role": "user", "content": example.query}]
        else:
            system_prompt = example.system_prompt or self.system_prompts.get(
                example.domain, self.system_prompts["general"]
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example.query},
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        completion = example.response + self.tokenizer.eos_token

        return {"prompt": prompt, "completion": completion}

    def _tokenize_no_special(self, text: str) -> list[int]:
        """
        Tokenize text without adding special tokens.

        Some tokenizer stubs used in tests don't accept ``add_special_tokens``.
        We gracefully fall back in that case.
        """
        try:
            tokenized = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )
        except TypeError:
            tokenized = self.tokenizer(text, truncation=False)

        input_ids = tokenized.get("input_ids", [])
        # HF tokenizers can return nested lists for batched inputs.
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return input_ids

    def _build_tokenized_fields(self, prompt: str, completion: str) -> dict | None:
        """
        Build tokenized fields with explicit completion masking.

        Returns ``None`` when example is invalid or exceeds ``max_seq_length``.
        """
        prompt_ids = self._tokenize_no_special(prompt)
        completion_ids = self._tokenize_no_special(completion)

        if len(completion_ids) == 0:
            return None

        input_ids = prompt_ids + completion_ids
        if len(input_ids) > self.max_seq_length:
            return None

        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "completion_mask": completion_mask,
        }

    def prepare_dataset(
        self,
        examples: list[TrainingExample],
        domain: str | None = None,
    ) -> Dataset:
        """
        Prepare a HuggingFace Dataset from training examples.

        Uses prompt/completion split so SFTTrainer computes loss only on
        the completion (assistant response) tokens.

        Args:
            examples: List of training examples.
            domain: Optional domain filter.

        Returns:
            HuggingFace Dataset ready for training.
        """
        if domain:
            examples = [ex for ex in examples if ex.domain == domain]

        # Format examples as prompt/completion splits with explicit completion masks
        formatted_data = []
        for example in examples:
            split = self.format_example_split(example)
            combined = split["prompt"] + split["completion"]
            tokenized = self._build_tokenized_fields(
                split["prompt"],
                split["completion"],
            )
            if tokenized is None:
                logger.warning(
                    "Example is invalid for completion-only SFT or exceeds max length, skipping"
                )
                continue

            formatted_data.append({
                # Unsloth SFTTrainer still expects dataset_text_field to exist
                # during dataset preparation. Keep text for compatibility while
                # also providing prompt/completion for completion-only loss.
                "text": combined,
                "prompt": split["prompt"],
                "completion": split["completion"],
                "domain": example.domain,
                **tokenized,
            })

        logger.info(
            f"Prepared {len(formatted_data)} examples for training "
            "(completion-only loss with explicit completion masks)"
        )

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
            split = self.format_example_split(example)
            combined = split["prompt"] + split["completion"]
            tokenized = self._build_tokenized_fields(
                split["prompt"],
                split["completion"],
            )
            if tokenized is not None:
                batch.append({
                    "text": combined,
                    "prompt": split["prompt"],
                    "completion": split["completion"],
                    "domain": example.domain,
                    **tokenized,
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

            # Response is just raw SQL (consistent with system prompt and BIRD format)
            response = query

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
        """Format a schema dictionary into CREATE TABLE syntax with PK/FK."""
        tables = schema.get("tables", [])
        columns = schema.get("columns", [])
        column_types = schema.get("column_types", [])
        primary_keys = schema.get("primary_keys", [])
        foreign_keys = schema.get("foreign_keys", [])

        # Build column index to table mapping for FK resolution
        col_to_table = {}
        for col_idx, (table_idx, col_name) in enumerate(columns):
            if table_idx >= 0:
                t_name = tables[table_idx]
                if isinstance(t_name, list):
                    t_name = t_name[1] if len(t_name) > 1 else t_name[0]
                col_to_table[col_idx] = (t_name, col_name)

        # Flatten primary keys
        pk_set = set()
        for pk in primary_keys:
            if isinstance(pk, list):
                pk_set.update(pk)
            else:
                pk_set.add(pk)

        # Build FK mapping
        fk_map = {}
        for fk in foreign_keys:
            if len(fk) == 2:
                from_col, to_col = fk
                if to_col in col_to_table:
                    fk_map[from_col] = col_to_table[to_col]

        lines = []
        for table_idx, table_name in enumerate(tables):
            if isinstance(table_name, list):
                table_name = table_name[1] if len(table_name) > 1 else table_name[0]

            col_defs = []
            for col_idx, (col_table_idx, col_name) in enumerate(columns):
                if col_table_idx == table_idx:
                    col_type = column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                    col_type = col_type.upper()

                    pk_suffix = " PRIMARY KEY" if col_idx in pk_set else ""

                    fk_suffix = ""
                    if col_idx in fk_map:
                        ref_table, ref_col = fk_map[col_idx]
                        fk_suffix = f" REFERENCES {ref_table}({ref_col})"

                    col_defs.append(f"  {col_name} {col_type}{pk_suffix}{fk_suffix}")

            if col_defs:
                lines.append(f"CREATE TABLE {table_name} (")
                lines.append(",\n".join(col_defs))
                lines.append(");")
                lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def from_bird(
        data: list[dict],
        include_schema: bool = True,
    ) -> list[TrainingExample]:
        """
        Convert BIRD dataset entries to training examples.

        BIRD examples have 'prompt' field with schema and evidence already included.

        Args:
            data: List of BIRD dataset entries.
            include_schema: Whether to use the prompt with schema (vs raw question).

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        for entry in data:
            question = entry.get("question", "")
            sql = entry.get("SQL", entry.get("query", ""))
            db_id = entry.get("db_id", "")

            # Use pre-processed prompt (includes schema and evidence) if available
            if include_schema and "prompt" in entry:
                full_query = entry["prompt"]
            else:
                full_query = f"Database: {db_id}\n\nConvert this question to SQL: {question}"

            # Response is just raw SQL, no explanation or formatting
            response = sql

            examples.append(
                TrainingExample(
                    query=full_query,
                    response=response,
                    domain="text_to_sql_bird",
                    metadata={"db_id": db_id, "source": "bird"},
                )
            )

        return examples

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
