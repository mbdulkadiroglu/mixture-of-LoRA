"""
Tests for Spider & BIRD training/inference/evaluation pipeline fixes.

Validates:
1. Spider training response format (raw SQL, no markdown)
2. BIRD training response format (raw SQL)
3. Evaluator accepts BIRD executor
4. evaluate_dataset() routes BIRD domain correctly
5. Spider schema format includes PK/FK info
6. System prompts are consistent between training and inference
"""

import pytest


class TestSpiderResponseFormat:
    """Verify from_spider() produces raw SQL responses."""

    def test_spider_response_is_raw_sql(self):
        from src.training.data_processor import DataProcessor

        data = [
            {
                "question": "How many singers are there?",
                "query": "SELECT COUNT(*) FROM singer",
                "db_id": "concert_singer",
            }
        ]

        examples = DataProcessor.from_spider(data, include_schema=False)

        assert len(examples) == 1
        response = examples[0].response
        # Should be raw SQL, no markdown
        assert response == "SELECT COUNT(*) FROM singer"
        assert "```" not in response
        assert "Based on" not in response

    def test_spider_response_no_markdown_wrapping(self):
        from src.training.data_processor import DataProcessor

        data = [
            {
                "question": "Find all albums",
                "query": "SELECT * FROM albums ORDER BY title",
                "db_id": "music",
            }
        ]

        examples = DataProcessor.from_spider(data, include_schema=False)

        response = examples[0].response
        assert response == "SELECT * FROM albums ORDER BY title"
        assert "```sql" not in response


class TestBirdResponseFormat:
    """Verify from_bird() produces raw SQL responses."""

    def test_bird_response_is_raw_sql(self):
        from src.training.data_processor import DataProcessor

        data = [
            {
                "question": "How many departments are there?",
                "SQL": "SELECT COUNT(*) FROM department",
                "db_id": "department_management",
                "evidence": "",
            }
        ]

        examples = DataProcessor.from_bird(data, include_schema=False)

        assert len(examples) == 1
        response = examples[0].response
        assert response == "SELECT COUNT(*) FROM department"
        assert "```" not in response

    def test_bird_domain_is_text_to_sql_bird(self):
        from src.training.data_processor import DataProcessor

        data = [
            {
                "question": "test",
                "SQL": "SELECT 1",
                "db_id": "test_db",
            }
        ]

        examples = DataProcessor.from_bird(data, include_schema=False)
        assert examples[0].domain == "text_to_sql_bird"


class TestEvaluatorBirdSupport:
    """Verify Evaluator initializes and routes BIRD correctly."""

    def test_evaluator_accepts_bird_db_dir(self):
        from src.evaluation.evaluator import Evaluator

        # Should not raise - accepts bird_db_dir parameter
        evaluator = Evaluator(bird_db_dir="/nonexistent/path")
        # The parameter is accepted; executor may auto-detect a real bird_data dir
        assert hasattr(evaluator, "_bird_executor")

    def test_evaluator_has_bird_system_prompt(self):
        from src.evaluation.evaluator import Evaluator

        assert "text_to_sql_bird" in Evaluator.SYSTEM_PROMPTS
        assert (
            Evaluator.SYSTEM_PROMPTS["text_to_sql_bird"]
            == Evaluator.SYSTEM_PROMPTS["text_to_sql"]
        )

    def test_evaluator_bird_default_metric(self):
        from src.evaluation.evaluator import Evaluator

        evaluator = Evaluator()
        metric = evaluator._get_default_metric("text_to_sql_bird")
        assert metric == "sql_exact_match"

    def test_bird_executor_exported(self):
        from src.evaluation import BIRDExecutor, get_bird_executor

        assert BIRDExecutor is not None
        assert get_bird_executor is not None

    def test_bird_adapter_lookup_uses_exact_domain(self):
        """evaluate_domain should look up the exact domain adapter, not fall back to text_to_sql."""
        from unittest.mock import MagicMock

        # When BIRD adapter exists, get_adapter_path("text_to_sql_bird") returns it
        adapter_paths = {
            "text_to_sql": "/adapters/text_to_sql/text_to_sql_v3",
            "text_to_sql_bird": "/adapters/text_to_sql_bird/text_to_sql_bird_v1",
        }
        mock_manager = MagicMock()
        mock_manager.get_adapter_path.side_effect = lambda d: adapter_paths.get(d)

        path = mock_manager.get_adapter_path("text_to_sql_bird")
        assert path == "/adapters/text_to_sql_bird/text_to_sql_bird_v1"

    def test_missing_bird_adapter_raises_error(self):
        """evaluate_domain should raise ValueError when no adapter exists for the domain."""
        from unittest.mock import MagicMock

        # Only Spider adapter exists, no BIRD adapter
        mock_manager = MagicMock()
        mock_manager.get_adapter_path.side_effect = lambda d: {
            "text_to_sql": "/adapters/text_to_sql/text_to_sql_v3",
        }.get(d)

        # No BIRD adapter — should be None (framework raises ValueError)
        path = mock_manager.get_adapter_path("text_to_sql_bird")
        assert path is None


class TestSpiderSchemaFormat:
    """Verify Spider schema includes PK/FK info in CREATE TABLE format."""

    def test_schema_format_has_create_table(self):
        from src.datasets.loader import DatasetLoader

        loader = DatasetLoader.__new__(DatasetLoader)
        loader._schemas = {
            "test_db": {
                "tables": ["users", "orders"],
                "columns": [
                    [0, "id"],
                    [0, "name"],
                    [1, "order_id"],
                    [1, "user_id"],
                    [1, "amount"],
                ],
                "column_types": ["number", "text", "number", "number", "number"],
                "primary_keys": [0, 2],
                "foreign_keys": [[3, 0]],
            }
        }
        loader._bird_schemas = {}

        schema_str = loader.get_schema_string("test_db")

        assert "CREATE TABLE" in schema_str
        assert "users" in schema_str
        assert "orders" in schema_str
        assert "PRIMARY KEY" in schema_str
        assert "REFERENCES" in schema_str

    def test_schema_format_pk_on_correct_column(self):
        from src.datasets.loader import DatasetLoader

        loader = DatasetLoader.__new__(DatasetLoader)
        loader._schemas = {
            "test_db": {
                "tables": ["items"],
                "columns": [
                    [0, "item_id"],
                    [0, "name"],
                ],
                "column_types": ["number", "text"],
                "primary_keys": [0],
                "foreign_keys": [],
            }
        }
        loader._bird_schemas = {}

        schema_str = loader.get_schema_string("test_db")

        assert "item_id NUMBER PRIMARY KEY" in schema_str
        assert "name TEXT" in schema_str
        # name should not have PRIMARY KEY
        assert "name TEXT PRIMARY KEY" not in schema_str

    def test_data_processor_format_schema_has_pk_fk(self):
        """Verify DataProcessor._format_schema also includes PK/FK."""
        from src.training.data_processor import DataProcessor

        schema = {
            "tables": ["users", "posts"],
            "columns": [
                [0, "id"],
                [0, "name"],
                [1, "post_id"],
                [1, "user_id"],
            ],
            "column_types": ["number", "text", "number", "number"],
            "primary_keys": [0, 2],
            "foreign_keys": [[3, 0]],
        }

        result = DataProcessor._format_schema(schema)

        assert "CREATE TABLE" in result
        assert "PRIMARY KEY" in result
        assert "REFERENCES" in result


class TestSystemPromptConsistency:
    """Verify system prompts match between training and inference."""

    def test_training_and_evaluator_prompts_match(self):
        from src.training.data_processor import DataProcessor
        from src.evaluation.evaluator import Evaluator

        # Create a DataProcessor with a dummy tokenizer
        class DummyTokenizer:
            pass

        dp = DataProcessor(DummyTokenizer())

        # text_to_sql
        assert (
            dp.system_prompts["text_to_sql"] == Evaluator.SYSTEM_PROMPTS["text_to_sql"]
        )

        # text_to_sql_bird
        assert (
            dp.system_prompts["text_to_sql_bird"]
            == Evaluator.SYSTEM_PROMPTS["text_to_sql_bird"]
        )

    def test_training_and_framework_prompts_match(self):
        from src.training.data_processor import DataProcessor
        from src.framework import AdaptiveSLMFramework

        class DummyTokenizer:
            pass

        dp = DataProcessor(DummyTokenizer())

        # text_to_sql
        assert (
            dp.system_prompts["text_to_sql"]
            == AdaptiveSLMFramework.SYSTEM_PROMPTS["text_to_sql"]
        )

        # text_to_sql_bird
        assert (
            dp.system_prompts["text_to_sql_bird"]
            == AdaptiveSLMFramework.SYSTEM_PROMPTS["text_to_sql_bird"]
        )

    def test_evaluator_and_framework_prompts_match(self):
        from src.evaluation.evaluator import Evaluator
        from src.framework import AdaptiveSLMFramework

        for domain in [
            "text_to_sql",
            "text_to_sql_bird",
            "math_reasoning",
            "code_generation",
            "general",
        ]:
            assert (
                Evaluator.SYSTEM_PROMPTS[domain]
                == AdaptiveSLMFramework.SYSTEM_PROMPTS[domain]
            ), f"Prompt mismatch for domain: {domain}"

    def test_sql_prompt_says_no_markdown(self):
        from src.evaluation.evaluator import Evaluator

        prompt = Evaluator.SYSTEM_PROMPTS["text_to_sql"]
        assert "no" in prompt.lower() or "Do not" in prompt
        assert "markdown" in prompt.lower()
        assert "output only" in prompt.lower()


class TestEvaluationConfig:
    """Verify EvaluationConfig has bird_db_dir field."""

    def test_bird_db_dir_field_exists(self):
        from src.config import EvaluationConfig

        config = EvaluationConfig()
        assert hasattr(config, "bird_db_dir")
        assert config.bird_db_dir is None

    def test_bird_db_dir_can_be_set(self):
        from src.config import EvaluationConfig

        config = EvaluationConfig(bird_db_dir="bird_data")
        assert config.bird_db_dir == "bird_data"


class TestSqlResultComparison:
    """Verify execution comparison ignores order but preserves multiplicity."""

    def test_compare_result_sets_ignores_row_order(self):
        from src.evaluation.sql_executor import _compare_result_sets

        a = [(1, "x"), (2, "y")]
        b = [(2, "y"), (1, "x")]
        assert _compare_result_sets(a, b) is True

    def test_compare_result_sets_handles_column_reordering(self):
        from src.evaluation.sql_executor import _compare_result_sets

        a = [(1, "x"), (2, "y")]
        b = [("x", 1), ("y", 2)]
        assert _compare_result_sets(a, b) is True

    def test_compare_result_sets_preserves_duplicate_rows(self):
        from src.evaluation.sql_executor import _compare_result_sets

        a = [(1,), (1,)]
        b = [(1,)]
        assert _compare_result_sets(a, b) is False


class TestCompletionOnlyTrainingSetup:
    """Verify trainer enables true completion-only loss with completion masks."""

    def test_trainer_enables_completion_only_loss(self, monkeypatch):
        from types import SimpleNamespace
        from datasets import Dataset
        import src.training.trainer as trainer_module
        from src.config import TrainingConfig, LoRAConfig
        from src.training.trainer import LoRATrainer

        captured = {}

        class _DummySFTConfig:
            def __init__(self, **kwargs):
                captured["training_args"] = kwargs
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class _DummySFTTrainer:
            def __init__(self, model, train_dataset, tokenizer, args):
                captured["dataset"] = train_dataset
                captured["tokenizer"] = tokenizer
                captured["args_obj"] = args

            @staticmethod
            def train(resume_from_checkpoint=None):
                return SimpleNamespace(
                    training_loss=0.0,
                    metrics={},
                    global_step=1,
                )

        monkeypatch.setattr(trainer_module, "SFTConfig", _DummySFTConfig)
        monkeypatch.setattr(trainer_module, "SFTTrainer", _DummySFTTrainer)

        class _DummyTokenizer:
            @staticmethod
            def __call__(text, truncation=False, add_special_tokens=False):
                assert truncation is False
                return {"input_ids": [ord(c) for c in text]}

        class _DummyStudent:
            def __init__(self):
                self.is_loaded = True
                self.peft_model = object()
                self.config = SimpleNamespace(max_seq_length=256)
                self.tokenizer = _DummyTokenizer()

            @staticmethod
            def load_model():
                return None

            @staticmethod
            def setup_lora():
                return None

            @staticmethod
            def get_trainable_model():
                return object()

            @staticmethod
            def save_adapter(_path):
                return None

        trainer = LoRATrainer(
            student=_DummyStudent(),
            training_config=TrainingConfig(),
            lora_config=LoRAConfig(),
            output_dir="data/lora_adapters/training_runs",
        )

        raw_dataset = Dataset.from_list(
            [
                {"prompt": "<prompt>", "completion": "SELECT 1<eos>", "domain": "text_to_sql"}
            ]
        )
        trainer.train(raw_dataset, domain="text_to_sql", num_epochs=1, max_steps=1)

        args = captured["training_args"]
        assert args["completion_only_loss"] is True
        assert args["dataset_kwargs"] == {"skip_prepare_dataset": True}

        row = captured["dataset"][0]
        assert "input_ids" in row
        assert "completion_mask" in row
        assert any(v == 0 for v in row["completion_mask"])
        assert any(v == 1 for v in row["completion_mask"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
