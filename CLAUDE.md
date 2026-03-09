# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An adaptive teacher-student framework where a large model (GPT-5 mini) improves a small model (Qwen 2.5 14B + LoRA) through online distillation. The core thesis: **when the router sends a query to the big model, use that response to train the small model.** Over time, the student improves, the router sends fewer queries to the teacher, and inference cost drops.

The system is **domain-agnostic by design**. Text-to-SQL (Spider, BIRD) is the current test domain because it has convenient evaluation infrastructure, but the framework applies to any task where a larger model outperforms a smaller one.

- A **Router** that decides whether the student or teacher handles each query
- A **Student Model** (Qwen 2.5 14B + LoRA) for domain-specific tasks
- A **Teacher Model** (GPT-5 mini) for complex queries and training data generation
- **Online Learning** to continuously improve the student from teacher responses

Test domains: `text_to_sql` (Spider + BIRD), `math_reasoning` (GSM8K), `code_generation` (MBPP).

**Hardware**: Requires 4x NVIDIA RTX 6000 Ada GPUs (~196GB VRAM), CUDA 12.0+, Python 3.10+.

## Common Commands

```bash
# Activate environment
source ~/miniconda3/bin/activate mixture-lora

# Install (with dev tools)
pip install -e ".[dev]"

# Train a domain adapter
python main.py train --domain text_to_sql --max-samples 1000 --eval
python main.py train --domain math_reasoning --use-teacher --epochs 3

# Evaluate models
python main.py evaluate --samples 100
python main.py evaluate --include-teacher --output results.json

# Compare base model vs LoRA adapter
python scripts/compare_base_vs_lora.py --mode lora --samples 100
python scripts/compare_base_vs_lora.py --mode base --samples 100

# Diagnose model outputs
python scripts/diagnose_eval.py --mode lora --samples 10

# Interactive demo
python main.py demo

# Show framework info
python main.py info

# Lint and format
ruff check src/ scripts/ main.py
black src/ scripts/ main.py
```

## Architecture

### Data Flow

1. Query enters `AdaptiveSLMFramework.process_query()` (src/framework.py)
2. Domain is classified (keyword/regex-based, not ML) or passed explicitly
3. Router checks adapter availability and applies routing strategy (perplexity/self_eval/stats)
4. If confidence >= threshold (0.7): student loads domain LoRA adapter, generates response
5. If confidence < threshold or student fails: falls back to teacher (GPT-5 mini API)
6. **Teacher responses are collected as training data** — this is the core learning signal. The student learns from the teacher, not from ground truth labels. (Ground truth training was only used to validate that LoRA fine-tuning works at all.)
7. `train_domain()` fine-tunes LoRA adapter via SFTTrainer, mixing new data with replay buffer (20% ratio)

### Key Modules

- **src/framework.py** - `AdaptiveSLMFramework`: Main orchestrator. Entry point for all operations (query processing, training, evaluation). Coordinates all other components.
- **src/config.py** - Typed dataclasses for all configuration. `load_config()` loads YAML then overrides with env vars (env vars take precedence).
- **src/models/student.py** - `StudentModel`: Qwen 2.5 14B via Unsloth `FastLanguageModel` (4-bit quantization, bfloat16). Handles dynamic LoRA adapter loading/unloading.
- **src/models/teacher.py** - `TeacherModel`: Uses OpenAI **Responses API** (`client.responses.create()`), not the Chat Completions API.
- **src/training/data_processor.py** - `DataProcessor`: Converts raw dataset samples (Spider/BIRD/GSM8K/MBPP) into chat-templated training text. Domain-specific system prompts are defined here.
- **src/training/trainer.py** - `LoRATrainer`: SFT training with experience replay buffer to prevent catastrophic forgetting.
- **src/datasets/loader.py** - `DatasetLoader`: Auto-detects local data dirs (`spider_data/`, `bird_data/`), falls back to HuggingFace. Loads database schemas for text-to-sql.
- **src/evaluation/sql_executor.py** - Executes SQL against actual SQLite databases (Spider/BIRD) for execution accuracy evaluation. Note: this is a domain-specific evaluator for the text-to-SQL test domain, not a core part of the framework.
- **src/adapters/manager.py** - `AdapterManager`: Versioned adapter storage with registry (`data/lora_adapters/registry.json`), symlinked `latest/` directories.

### Adapter Storage Layout

```
data/lora_adapters/
├── {domain}/{domain}_v{N}/    # Versioned adapters (PEFT weights + tokenizer)
├── {domain}/latest/           # Symlink to best version
├── training_runs/             # Timestamped checkpoint directories
└── registry.json              # Metadata: scores, versions, paths
```

### Configuration Precedence

1. Dataclass defaults in `src/config.py`
2. `configs/config.yaml` overrides
3. `.env` environment variables override (for `OPENAI_API_KEY`, `HF_TOKEN`, `TEACHER_MODEL`, `STUDENT_MODEL`, `LORA_ADAPTERS_PATH`, `ROUTER_CONFIDENCE_THRESHOLD`, `CUDA_VISIBLE_DEVICES`)

## GPU Allocation

There are 4 available GPUs (0, 1, 2, 3). You may use any of them if they are available. Note that `.env` values can override command-line environment variables if loaded with `override=True`; use each script's `--gpu` flag (or set `CUDA_VISIBLE_DEVICES`) after `.env` loads when you need to select a specific GPU.

## Code Conventions

- **Critical import order**: `unsloth` must be imported before PyTorch/transformers (enables kernel optimizations). `.env` must be loaded before PyTorch initializes (controls `CUDA_VISIBLE_DEVICES` for GPU allocation). See `main.py` for the correct pattern.
- Use `loguru` for logging, not stdlib logging
- Type hints with Python 3.10+ union syntax (`str | None`)
- Dataclasses for structured return types (`QueryResult`, `EvaluationResult`, `TrainingExample`, `AdapterInfo`, `RoutingDecision`)
- `pathlib.Path` for file paths

## Current Status

LoRA v3 trained on Spider text_to_sql achieves 74.18% accuracy (+5.13% over base model). Adapters stored at `data/lora_adapters/text_to_sql/text_to_sql_v3/`.

Active work on `feature/bird-training` branch: BIRD dataset infrastructure is complete, training blocked on corrupted `train.zip` (can use dev set with 1534 samples). See `tasks/todo.md`.

## Workflow Guidelines

### Planning & Execution
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Write plan to `tasks/todo.md` with checkable items, verify before implementing
- Track progress by marking items complete, document results

### Subagent Strategy
- Offload research, exploration, and parallel analysis to subagents to keep main context clean
- One task per subagent for focused execution
- For complex problems, use multiple subagents

### Verification
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Diff behavior between main and changes when relevant

### Self-Improvement
- After ANY correction from user: update `tasks/lessons.md` with the pattern
- Write rules that prevent the same mistake
- Review lessons at session start

### Evaluation
- **Always use the full test set** for evaluations unless the user explicitly requests a smaller sample size. Never default to `max_samples=100` or any other subset — pass `None` (or omit) so the entire test split is used.

### Long-Running Processes
- **Always use tmux** for any process that isn't instant (training, evaluation, experiments, etc.). Never run long processes as background shell tasks — they die if the session ends.
- Use `tmux new-session -d -s <name> "<command>"` to start, `tmux capture-pane -t <name> -p` to check output.

### Core Principles
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Only touch what's necessary. Avoid introducing bugs.
- **Demand Elegance**: For non-trivial changes, ask "is there a more elegant way?" (but don't over-engineer simple fixes)
