# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An adaptive teacher-student framework for fine-tuning a small language model (Qwen 2.5 14B) using domain-specific LoRA adapters, with GPT-5 mini as the teacher model. The system includes:
- A **Router** that decides whether the student or teacher handles each query
- A **Student Model** (Qwen 2.5 14B + LoRA) for domain-specific tasks
- A **Teacher Model** (GPT-5 mini) for complex queries and training data generation
- **Online Learning** to continuously improve the student from teacher responses

Supported domains: `text_to_sql` (Spider dataset), `math_reasoning` (GSM8K), `code_generation` (MBPP).

## Common Commands

```bash
# Activate environment
source ~/miniconda3/bin/activate mixture-lora

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
```

## Architecture

### Core Components (src/)

- **framework.py**: `AdaptiveSLMFramework` - Main orchestrator that coordinates routing, generation, training, and evaluation. Entry point for programmatic usage.

- **models/student.py**: `StudentModel` - Wraps Qwen 2.5 14B using Unsloth for 4-bit quantization and efficient inference. Handles LoRA adapter loading/saving and generation.

- **models/teacher.py**: `TeacherModel` - OpenAI API client for GPT-5 mini. Used for generating training data and evaluating student responses.

- **router/router.py**: `QueryRouter` - Routes queries between student/teacher based on confidence. Supports three strategies:
  - `perplexity`: Uses model perplexity on the query
  - `self_eval`: Asks teacher to evaluate query difficulty
  - `stats`: Uses historical success rates per domain

- **training/trainer.py**: `LoRATrainer` - SFT training using TRL's SFTTrainer. Supports experience replay for continual learning.

- **adapters/manager.py**: `AdapterManager` - Manages LoRA adapter versioning, storage, and registry at `data/lora_adapters/`.

- **evaluation/evaluator.py**: `Evaluator` - Domain-specific evaluation with execution-based SQL accuracy (against Spider databases) and math answer extraction.

### Data Flow

1. Query enters `AdaptiveSLMFramework.process_query()`
2. Router checks adapter availability and confidence
3. If routed to student: load domain LoRA adapter, generate response
4. If routed to teacher: call GPT-5 mini API
5. Teacher responses are collected for future training
6. `train_domain()` fine-tunes LoRA adapter on collected examples

### Key Configuration

Edit `configs/config.yaml` for:
- LoRA parameters: r=32, alpha=32, dropout=0.0
- Training: batch_size=4, grad_accum=4, lr=2e-4, epochs=3
- Router threshold: 0.7 confidence for student routing
- Adapter storage: `data/lora_adapters/`

Environment variables in `.env`:
- `OPENAI_API_KEY`: For GPT-5 mini teacher
- `HF_TOKEN`: For Hugging Face model access

## Code Conventions

- Import `unsloth` before PyTorch/transformers (enables optimizations)
- Load `.env` before PyTorch initializes (GPU allocation)
- Use `loguru` for logging, not stdlib logging
- Type hints with Python 3.10+ union syntax (`str | None`)
- Dataclasses for structured return types (`QueryResult`, `EvaluationResult`, etc.)

## Current Status

LoRA v3 trained on Spider text_to_sql achieves 74.18% accuracy (+5.13% over base model). Adapters stored at `data/lora_adapters/text_to_sql/text_to_sql_v3/`.

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

### Core Principles
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Only touch what's necessary. Avoid introducing bugs.
- **Demand Elegance**: For non-trivial changes, ask "is there a more elegant way?" (but don't over-engineer simple fixes)
