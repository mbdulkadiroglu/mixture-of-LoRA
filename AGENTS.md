# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the CLI entry point (`train`, `evaluate`, `demo`, `serve`, `info`).
- Core logic lives in `src/`: `framework.py` (orchestration), `router/`, `models/`, `training/`, `datasets/`, `evaluation/`, and `adapters/`.
- Utility workflows live in `scripts/` (for example `train_domain.py`, `evaluate_all.py`, `compare_base_vs_lora.py`).
- Config is in `configs/`; runtime outputs are in `data/` (`lora_adapters/`, `checkpoints/`, `training_data/`, `eval_results/`).
- Tests are in `tests/` (currently regression-focused in `tests/test_pipeline_fixes.py`).
- Planning/notes are in `tasks/` (`todo.md`, `lessons.md`, `project_vision.md`).

## Build, Test, and Development Commands
- `source ~/miniconda3/bin/activate mixture-lora`: activate the standard dev environment.
- `pip install -e ".[dev]"`: install editable package plus dev tools.
- `python main.py train --domain text_to_sql --max-samples 1000 --eval`: train a domain adapter.
- `python main.py evaluate --domains text_to_sql math_reasoning`: run evaluation.
- `python scripts/compare_base_vs_lora.py --mode lora --samples 100`: compare adapter vs base behavior.
- `python main.py demo`: run interactive routing demo.
- `pytest -q tests/test_pipeline_fixes.py`: run regression tests.
- `ruff check src/ scripts/ main.py && black src/ scripts/ main.py`: lint and format.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, and explicit type hints for new/changed code.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, canonical domain IDs (`text_to_sql`, `text_to_sql_bird`, `math_reasoning`, `code_generation`).
- Prefer `pathlib.Path`, dataclasses for structured results, and `loguru` over stdlib logging.
- Critical startup order: load `.env` before ML initialization; import `unsloth` before PyTorch/Transformers.

## Testing & Evaluation Guidelines
- Use `pytest`; follow `test_*.py` and `test_*` naming.
- Add targeted regression tests for SQL format, schema handling, routing behavior, and prompt consistency.
- For evaluations, default to full test splits. Use sample-limited runs only when explicitly requested.
- For text-to-SQL, include execution-oriented checks (not only string matching) when possible.

## Architecture Focus & Workflow
- Core thesis: teacher-student cascade with feedback loops. Teacher fallbacks should produce training examples and update routing statistics.
- Current priorities: automate cascade -> collect -> retrain loop, add cost tracking (tokens/latency), and prototype adaptive routing thresholds.
- For non-trivial work, track a checklist in `tasks/todo.md`; after user corrections, record lessons in `tasks/lessons.md`.

## Commit, PR, and Security Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
- PRs should include scope, affected domains/datasets, config/env changes, commands run, and key metrics.
- Keep secrets in `.env` (`OPENAI_API_KEY`, `HF_TOKEN`); never commit credentials or unnecessary large generated artifacts.
- Config precedence is: dataclass defaults -> `configs/config.yaml` -> `.env` overrides.
