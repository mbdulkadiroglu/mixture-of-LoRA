"""
Regression tests for cascade hidden issues.
"""

import random
from types import SimpleNamespace

import torch

from cascade.config import CascadeConfig
from cascade.evaluator import CascadeEvaluator
from cascade.phase1.run_phase1 import _build_phase1_experiments
from cascade.prompts import SPIDER_SQL_SYSTEM_PROMPT
from cascade.runner import CascadeRunner
from cascade.teacher import CascadeTeacher
from cascade.student import CascadeStudent, GenerationResult
from cascade.trainer import CascadeTrainer
from src.models.teacher import TeacherModel
from src.training.data_processor import DataProcessor, TrainingExample
from src.evaluation.sql_cleaning import extract_sql_from_text


def test_extract_sql_preserves_semicolon_inside_string_literal() -> None:
    raw = "SELECT name FROM users WHERE note = 'a;b';\nExplanation: done."
    cleaned = extract_sql_from_text(raw)
    assert cleaned == "SELECT name FROM users WHERE note = 'a;b'"


def test_extract_sql_from_markdown_block() -> None:
    raw = "```sql\nSELECT COUNT(*) FROM singer;\n```\nExtra commentary"
    cleaned = extract_sql_from_text(raw)
    assert cleaned == "SELECT COUNT(*) FROM singer"


def test_runner_cleans_teacher_sql_before_training() -> None:
    class _Student:
        @staticmethod
        def generate_with_logprobs(_messages):
            return GenerationResult(
                text="SELECT 1",
                token_log_probs=[-0.1],
                mean_log_prob=-0.1,
                min_log_prob=-0.1,
                mean_entropy=0.1,
                num_tokens=1,
            )

    class _Router:
        @staticmethod
        def decide(_gen_result):
            return SimpleNamespace(target="teacher", threshold=-4.0)

    class _Evaluator:
        @staticmethod
        def check_single(_pred, _gold, _db):
            return True

    class _Teacher:
        @staticmethod
        def generate(_messages):
            return "```sql\nSELECT ';' AS note;\n```\nExplanation: extra text"

    class _Logger:
        @staticmethod
        def log_training_example(**_kwargs):
            return None

        @staticmethod
        def log_interaction(**_kwargs):
            return None

    runner = CascadeRunner(CascadeConfig(dataset="spider", training_source="teacher"))
    runner.student = _Student()
    runner.router = _Router()
    runner.evaluator = _Evaluator()
    runner.teacher = _Teacher()
    runner.cascade_logger = _Logger()

    out = runner._run_round(
        round_idx=0,
        queries=[{"prompt": "Q", "query": "SELECT 1", "db_id": "db"}],
    )
    assert out["teacher_examples"][0]["response"] == "SELECT ';' AS note"


def test_generate_with_logprobs_is_greedy_by_default() -> None:
    class _Batch(dict):
        def to(self, _device):
            return self

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def apply_chat_template(_messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return "prompt"

        @staticmethod
        def __call__(prompt, return_tensors, truncation, max_length):
            assert prompt == "prompt"
            assert return_tensors == "pt"
            assert truncation is True
            assert max_length > 0
            return _Batch({"input_ids": torch.tensor([[10, 11, 12]])})

        @staticmethod
        def decode(_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return "SELECT 1"

    class _Model:
        device = torch.device("cpu")

        def __init__(self):
            self.kwargs = {}

        @staticmethod
        def modules():
            return []

        @staticmethod
        def eval():
            return None

        def _old_generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[10, 11, 12, 13]])

        @staticmethod
        def __call__(input_ids, use_cache=False):
            assert use_cache is False
            vocab = 32
            logits = torch.zeros((1, input_ids.shape[1], vocab))
            return SimpleNamespace(logits=logits)

    student = CascadeStudent.__new__(CascadeStudent)
    student.config = CascadeConfig()
    student._student = SimpleNamespace(
        peft_model=None,
        model=_Model(),
        tokenizer=_Tokenizer(),
        config=SimpleNamespace(max_seq_length=1024),
    )

    student.generate_with_logprobs([{"role": "user", "content": "Q"}])
    kwargs = student._student.model.kwargs

    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_train_round_samples_replay_before_adding_new_examples() -> None:
    old_example = {"prompt": "old", "response": "SELECT old", "source_round": -1}
    new_examples = [
        {"prompt": "new1", "response": "SELECT 1", "source_round": 0},
        {"prompt": "new2", "response": "SELECT 2", "source_round": 0},
    ]

    class _ReplayBuffer:
        def __init__(self):
            self.buffer = [old_example]

        def sample(self, n):
            return self.buffer[:n]

        def add(self, item):
            self.buffer.append(item)

    captured = {}

    class _Trainer:
        @staticmethod
        def train_round(new_examples, replay_examples, round_num):
            captured["new_examples"] = list(new_examples)
            captured["replay_examples"] = list(replay_examples)
            captured["round_num"] = round_num
            return {"train_loss": 0.0}

    class _Logger:
        @staticmethod
        def log_training_example(**_kwargs):
            return None

    class _Model:
        @staticmethod
        def eval():
            return None

    runner = CascadeRunner(CascadeConfig(replay_ratio=0.5))
    runner.replay_buffer = _ReplayBuffer()
    runner.trainer = _Trainer()
    runner.cascade_logger = _Logger()
    runner.student = SimpleNamespace(inner=SimpleNamespace(peft_model=None, model=_Model()))

    runner._train_round(round_idx=0, new_examples=new_examples)

    assert captured["round_num"] == 0
    assert captured["new_examples"] == new_examples
    assert captured["replay_examples"] == [old_example]
    assert new_examples[0] in runner.replay_buffer.buffer
    assert new_examples[1] in runner.replay_buffer.buffer


def test_teacher_model_sends_structured_system_instruction() -> None:
    class _Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output_text="SELECT 1")

    responses = _Responses()
    teacher = TeacherModel.__new__(TeacherModel)
    teacher.model_name = "gpt-5-mini"
    teacher.client = SimpleNamespace(responses=responses)

    result = teacher.generate(
        [
            {"role": "system", "content": "Return only SQL."},
            {"role": "user", "content": "Question text"},
        ]
    )

    assert result == "SELECT 1"
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["instructions"] == "Return only SQL."
    assert call["input"] == [{"role": "user", "content": "Question text"}]
    assert "System:" not in str(call["input"])


def test_cascade_teacher_retries_transient_errors(monkeypatch) -> None:
    class _FlakyTeacher:
        def __init__(self):
            self.calls = 0

        def generate(self, _messages):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary upstream failure")
            return "SELECT 1"

    sleeps = []
    monkeypatch.setattr("cascade.teacher.time.sleep", lambda seconds: sleeps.append(seconds))

    teacher = CascadeTeacher(
        CascadeConfig(
            teacher_max_retries=3,
            teacher_retry_backoff_seconds=0.01,
        )
    )
    teacher._teacher = _FlakyTeacher()

    out = teacher.generate([{"role": "user", "content": "Q"}])

    assert out == "SELECT 1"
    assert teacher._teacher.calls == 3
    assert sleeps == [0.01, 0.02]


def test_runner_set_random_seeds_is_reproducible() -> None:
    runner = CascadeRunner(CascadeConfig(seed=123))

    runner._set_random_seeds()
    first_random = random.random()
    first_torch = torch.rand(3)

    runner._set_random_seeds()
    second_random = random.random()
    second_torch = torch.rand(3)

    assert first_random == second_random
    assert torch.equal(first_torch, second_torch)


def test_trainer_ensure_lora_setup_skips_when_peft_exists() -> None:
    trainer = CascadeTrainer(
        student=SimpleNamespace(inner=None),
        config=CascadeConfig(),
    )
    calls = {"setup_lora": 0}

    class _Inner:
        peft_model = object()

        @staticmethod
        def setup_lora():
            calls["setup_lora"] += 1

    trainer._ensure_lora_setup(_Inner())

    assert calls["setup_lora"] == 0
    assert trainer._lora_setup_done is True


def test_runner_setup_prepares_lora_before_first_round(monkeypatch) -> None:
    calls = {"setup_for_training": 0}

    class _Student:
        def __init__(self, _config):
            self.inner = SimpleNamespace(peft_model=None)

        @staticmethod
        def load():
            return None

        @staticmethod
        def load_adapter(_path):
            return None

        @staticmethod
        def setup_for_training():
            calls["setup_for_training"] += 1

    class _Teacher:
        def __init__(self, _config):
            return None

        @staticmethod
        def load():
            return None

    class _Router:
        def __init__(self, _config):
            return None

    class _Evaluator:
        def __init__(self, _config):
            return None

        @staticmethod
        def load():
            return None

    class _Trainer:
        def __init__(self, _student, _config):
            return None

    class _Logger:
        def __init__(self, _name, _exp_dir):
            return None

    class _ReplayBuffer:
        def __init__(self, _max_size):
            return None

    monkeypatch.setattr("cascade.runner.CascadeStudent", _Student)
    monkeypatch.setattr("cascade.runner.CascadeTeacher", _Teacher)
    monkeypatch.setattr("cascade.runner.CascadeRouter", _Router)
    monkeypatch.setattr("cascade.runner.CascadeEvaluator", _Evaluator)
    monkeypatch.setattr("cascade.runner.CascadeTrainer", _Trainer)
    monkeypatch.setattr("cascade.runner.CascadeLogger", _Logger)
    monkeypatch.setattr("cascade.runner.CascadeReplayBuffer", _ReplayBuffer)
    monkeypatch.setattr(
        CascadeRunner,
        "_load_data",
        lambda self: (
            setattr(self, "_training_pool", [{"id": 1}]),
            setattr(self, "_eval_set", [{"id": 2}]),
        ),
    )

    runner = CascadeRunner(CascadeConfig(train_after_round=True))
    runner._setup()

    assert calls["setup_for_training"] == 1


def test_phase1_gpu_override_propagates_to_experiment_configs() -> None:
    experiments = _build_phase1_experiments(threshold=-4.0, gpu="7")
    assert len(experiments) == 3
    for _, config in experiments:
        assert config.gpu_devices == "7"


def test_prepare_dataset_streaming_uses_prompt_completion_format() -> None:
    class _DummyTokenizer:
        eos_token = "<eos>"

        @staticmethod
        def apply_chat_template(_messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return "<prompt>"

        @staticmethod
        def __call__(text, truncation=False):
            assert truncation is False
            return {"input_ids": list(range(len(text)))}

    processor = DataProcessor(_DummyTokenizer(), max_seq_length=4096)
    examples = iter(
        [
            TrainingExample(
                query="How many rows?",
                response="SELECT COUNT(*) FROM t",
                domain="text_to_sql",
            )
        ]
    )

    batches = list(processor.prepare_dataset_streaming(examples, batch_size=8))
    assert len(batches) == 1
    dataset = batches[0]
    assert set(dataset.column_names) == {
        "text",
        "prompt",
        "completion",
        "domain",
        "input_ids",
        "attention_mask",
        "completion_mask",
    }
    row = dataset[0]
    assert row["text"] == "<prompt>SELECT COUNT(*) FROM t<eos>"
    assert row["prompt"] == "<prompt>"
    assert row["completion"] == "SELECT COUNT(*) FROM t<eos>"
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["completion_mask"])
    assert all(v == 0 for v in row["completion_mask"][: len("<prompt>")])
    assert all(v == 1 for v in row["completion_mask"][len("<prompt>"):])


def test_prepare_dataset_includes_compat_text_field() -> None:
    class _DummyTokenizer:
        eos_token = "<eos>"

        @staticmethod
        def apply_chat_template(_messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return "<prompt>"

        @staticmethod
        def __call__(text, truncation=False):
            assert truncation is False
            return {"input_ids": list(range(len(text)))}

    processor = DataProcessor(_DummyTokenizer(), max_seq_length=4096)
    dataset = processor.prepare_dataset(
        [
            TrainingExample(
                query="Q",
                response="SELECT 1",
                domain="text_to_sql",
            )
        ]
    )
    row = dataset[0]
    assert row["text"] == "<prompt>SELECT 1<eos>"
    assert row["prompt"] == "<prompt>"
    assert row["completion"] == "SELECT 1<eos>"
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["completion_mask"])
    assert all(v == 0 for v in row["completion_mask"][: len("<prompt>")])
    assert all(v == 1 for v in row["completion_mask"][len("<prompt>"):])


def test_runner_and_evaluator_share_query_message_builder() -> None:
    spider_query = {"prompt": "Convert this to SQL"}
    spider_config = CascadeConfig(dataset="spider")

    runner_spider = CascadeRunner(spider_config)
    evaluator_spider = CascadeEvaluator(spider_config)
    spider_runner_messages = runner_spider._build_messages(spider_query)
    spider_eval_messages = evaluator_spider._build_messages(spider_query)

    assert spider_runner_messages == spider_eval_messages
    assert spider_runner_messages[0]["role"] == "system"
    assert spider_runner_messages[0]["content"] == SPIDER_SQL_SYSTEM_PROMPT

    bird_query = {"prompt": "BIRD prompt text"}
    bird_config = CascadeConfig(dataset="bird")

    runner_bird = CascadeRunner(bird_config)
    evaluator_bird = CascadeEvaluator(bird_config)
    bird_runner_messages = runner_bird._build_messages(bird_query)
    bird_eval_messages = evaluator_bird._build_messages(bird_query)

    assert bird_runner_messages == bird_eval_messages
    assert bird_runner_messages == [{"role": "user", "content": "BIRD prompt text"}]
