# CLAUDE.md

## Research Goal

**Research question**: Under what conditions does cascade-based online distillation converge, and when does distribution shift from improved routing degrade the training signal enough to stall or reverse learning?

**Core idea**: When a router decides the small model (student) isn't confident enough and defers a query to the big model (teacher), use that teacher response to train the student. Over time the student improves, fewer queries go to the teacher, and inference cost drops.

This is **domain-agnostic online distillation** — the student learns from the teacher's responses, not from ground truth labels. Text-to-SQL (Spider, BIRD) is the test domain because it has convenient execution-based evaluation, but the approach applies to any task where a larger model outperforms a smaller one.

**The scientific challenge** is a coupled feedback loop:
1. The router decides which queries go to the teacher
2. That determines what training data the student receives
3. Training shifts the student's confidence distribution
4. That changes future routing decisions and training data

Routing and learning co-evolve. This creates failure modes like distribution shift under fixed thresholds and catastrophic forgetting from noisy teacher signal. Understanding when this loop converges vs degrades is the core contribution.

**Ideally**, an adaptive router learns to route queries optimally as the student improves. For now, the experiments use a temporary stand-in: the student generates responses for an entire batch, all queries are sorted by confidence (mean token log-prob), and the bottom N% are sent to the teacher. The cascade rate interpolates linearly from `cascade_rate_start` to `cascade_rate_end` across rounds (e.g. 70%→30%). This percentile-based approach is a simplification that isolates the distillation dynamics — it controls how much teacher signal the student receives without needing a learned routing policy. Building a proper adaptive router is a later phase (see `tasks/project_vision.md`).

See `tasks/project_vision.md` for the full phased roadmap and open research questions.

## Project Structure

```
cascade/                     # All experiment code and data
├── runner.py                # CascadeRunner: the N-round experiment loop
├── config.py                # CascadeConfig: single dataclass per experiment
├── student.py               # CascadeStudent: wraps StudentModel for inference + logprobs
├── teacher.py               # CascadeTeacher: wraps Ollama/OpenAI teacher calls
├── trainer.py               # CascadeTrainer: per-round LoRA SFT
├── router.py                # CascadeRouter: log-prob confidence → cascade decision
├── evaluator.py             # CascadeEvaluator: SQL execution accuracy on frozen eval set
├── prompts.py               # Prompt templates (bird_json, spider, etc.)
├── replay_buffer.py         # Experience replay to mitigate catastrophic forgetting
├── logger.py                # SQLite logging for per-query interaction data
├── calibrate.py             # Router threshold calibration
├── analysis/                # Comparison and plotting tools
├── phase0/                  # Verification experiments
├── phase1/                  # Spider baseline + BIRD sweep experiments
│   ├── exp_1_1_baseline.py  # Teacher distillation (core hypothesis)
│   ├── exp_1_2_static.py    # Static control (no training)
│   ├── exp_1_3_ground_truth.py  # Ground truth training control
│   ├── exp_bird_baseline.py # BIRD cascade baseline
│   ├── exp_bird_sweep.py    # Hyperparameter sweep (32 configs)
│   └── precache_bird_teacher.py  # Pre-generate teacher responses
├── phase2/                  # (planned) Teacher noise tolerance
├── phase3/                  # (planned) Adaptive routing
└── results/                 # All experiment outputs
    ├── bird_train_teacher_cache.json  # Pre-cached teacher responses for sweeps
    ├── experiment_archive_summary.json  # Archived per-experiment summaries after raw exp dirs are removed
    ├── sweep_results.json   # Aggregated sweep experiment results
    ├── sweep_report.md      # Hyperparameter sweep findings report
    └── phase1_report.md     # Phase 1 experiment report

src/                         # Shared library (cascade depends on these)
├── models/student.py        # StudentModel: Qwen 2.5 14B via Unsloth (4-bit quantized)
├── models/teacher.py        # TeacherModel: OpenAI Responses API
├── training/trainer.py      # LoRATrainer: SFT with replay buffer
├── training/data_processor.py  # DataProcessor: chat-template formatting
├── datasets/loader.py       # DatasetLoader: Spider/BIRD data loading + schema
├── evaluation/sql_executor.py  # SQL execution against SQLite databases
├── evaluation/sql_cleaning.py  # SQL extraction from model responses
└── config.py                # LoRAConfig, TrainingConfig, etc.

bird_data/                   # BIRD benchmark databases (needed for SQL execution)
spider_data/                 # Spider benchmark databases
tasks/
├── project_vision.md        # Full research vision, phased roadmap, open questions
└── lessons.md               # Patterns learned from past mistakes
```

## Common Commands

```bash
# Activate environment
source ~/miniconda3/bin/activate mixture-lora

# Run a cascade experiment
python -m cascade.phase1.exp_bird_sweep --config lr_1e6_bs32 --gpu 2

# Run all sweep configs sequentially
python -m cascade.phase1.exp_bird_sweep --config all --gpu 2

# Pre-cache teacher responses (eliminates per-round inference cost)
python -m cascade.phase1.precache_bird_teacher --gpu 2 --resume

# Lint and format
ruff check src/ cascade/
black src/ cascade/
```

## Experiment Flow

Each experiment is defined by a `CascadeConfig` and run by `CascadeRunner`:

1. Load dataset, split into frozen eval set (stratified by difficulty) + training pool (shuffled deterministically by seed)
2. Per round:
   - Sample `queries_per_round` queries without replacement from the training pool
   - **Pass 1**: Student generates SQL responses for all queries, collecting log-probabilities
   - **Batch routing**: Sort all queries by confidence (mean log-prob). Route the bottom `cascade_rate`% to teacher (e.g. 70%→30% linearly over rounds). This percentile-based approach is immune to distribution shift from training.
   - **Pass 2**: For teacher-routed queries, look up cached teacher response (or call teacher live). Build training examples from teacher responses. Check correctness of all queries against gold SQL via execution.
3. After each round: train student LoRA on new teacher examples + replay buffer samples (SFT, completion-only loss)
4. Evaluate on frozen eval set (execution accuracy against actual SQLite databases)
5. Log all per-query interaction data to SQLite (3 tables: interactions, training_examples, adapter_versions), save GPU metrics to JSON

### Database Schema

Each experiment produces a SQLite DB with:
- **interactions**: per-query per-round — prompt, student SQL, student log-probs (mean/min/entropy), routing decision, teacher SQL, correctness (student/teacher/final)
- **training_examples**: per-training-example — prompt, target SQL, source (teacher/gold/student), was_correct, is_replay, quality_weight
- **adapter_versions**: per-round — eval_accuracy, training_loss, example counts, adapter path

## Hardware

4x NVIDIA RTX 6000 Ada GPUs (~196GB VRAM total), CUDA 12.0+, Python 3.10+.

## GPU Allocation

GPUs 0, 1, 2, 3 are all available. Use each experiment's `--gpu` flag or set `CUDA_VISIBLE_DEVICES`. Note: `.env` values can override environment variables if loaded with `override=True`.

## Code Conventions

- **Critical import order**: `unsloth` must be imported before PyTorch/transformers (enables kernel optimizations). `.env` must be loaded before PyTorch initializes (controls `CUDA_VISIBLE_DEVICES`).
- Use `loguru` for logging, not stdlib logging
- Type hints with Python 3.10+ union syntax (`str | None`)
- Dataclasses for configuration and structured return types
- `pathlib.Path` for file paths

## Workflow Guidelines

### Planning & Execution
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Keep a short checkable plan in the current conversation or a temporary working note, and verify before implementing
- Track progress by marking items complete, document results

### Verification
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness

### Self-Improvement
- After ANY correction from user: update `tasks/lessons.md` with the pattern
- Write rules that prevent the same mistake

### Evaluation
- **Always use the full test set** unless the user explicitly requests a smaller sample size

### Long-Running Processes
- **Always use tmux** for any process that isn't instant (training, evaluation, experiments)
- Use `tmux new-session -d -s <name> "<command>"` to start, `tmux capture-pane -t <name> -p` to check output

### Commits & Code Review
- For non-trivial changes, make exactly one commit for the task with message `claude: <task-name>`
- Use a dedicated branch only for large, risky, or easily separable tasks; otherwise stay on the current branch
- If the user wants GitHub review, push the branch and open or update a PR against `main`
- If Codex automatic review is enabled for the repo, expect review on each push; otherwise use `@codex review`
- If review is expected, stop after committing or pushing and report:
  - branch name
  - commit hash
  - files changed
  - review commands:
    - `git show --stat --patch <commit>`
    - `git diff --check <commit>^..<commit>`
- When asked to address PR feedback, inspect PR comments with `gh` if available, fix the issues, and push a new commit to the same branch
