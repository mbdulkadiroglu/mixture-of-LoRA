# Adaptive Small Language Model Framework

A framework for adapting a small language model (Qwen 2.5 14B) to specific domains using LoRA adapters and teacher-student learning with GPT-5 mini (high).

## Overview

This framework implements an adaptive system where:

1. **Router** decides if the student model (with domain-specific LoRA) can handle a query
2. **Student Model** (Qwen 2.5 14B + LoRA) handles queries when confident
3. **Teacher Model** (GPT-5 mini high) handles complex queries and provides training data
4. **Online Learning** continuously improves the student from teacher responses

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────────────┐
│   Query     │────▶│  Router  │────▶│  Student (Qwen 2.5) │
└─────────────┘     └──────────┘     │  + LoRA Adapter     │
                          │          └─────────────────────┘
                          │                    │
                    (if low confidence)        │
                          │                    │
                          ▼                    ▼
                   ┌─────────────┐      ┌──────────┐
                   │   Teacher   │─────▶│ Response │
                   │ (GPT-5 mini)│      └──────────┘
                   └─────────────┘
                          │
                          │ (collect for training)
                          ▼
                   ┌─────────────┐
                   │   Training  │
                   │   Pipeline  │
                   └─────────────┘
```

## Supported Domains

- **Text-to-SQL**: Convert natural language to SQL (Spider, BIRD datasets)
- **Math Reasoning**: Solve mathematical word problems (GSM8K dataset)
- **Code Generation**: Generate Python code (MBPP dataset)

## Requirements

- Python 3.10+
- 4x NVIDIA RTX 6000 Ada GPUs (or equivalent ~196GB total VRAM)
- CUDA 12.0+

## Installation

```bash
# Clone the repository
cd /data/mehmet/projects/mixture-of-LoRA

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` with your credentials:

```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Edit `configs/config.yaml` for training/model parameters.

## Quick Start

### 1. Train a Domain Adapter

```bash
# Train on Spider (Text-to-SQL)
python main.py train --domain text_to_sql --max-samples 1000 --eval

# Train with teacher-generated data
python main.py train --domain math_reasoning --use-teacher --epochs 3
```

### 2. Evaluate Performance

```bash
# Evaluate all domains
python main.py evaluate --samples 100

# Include teacher comparison
python main.py evaluate --include-teacher --output results.json
```

### 3. Interactive Demo

```bash
# Run interactive demo
python main.py demo

# Teacher-only mode for comparison
python main.py demo --teacher-only
```

### 4. Get Framework Info

```bash
python main.py info
```

## Project Structure

```
mixture-of-LoRA/
├── configs/
│   └── config.yaml          # Main configuration
├── data/
│   ├── lora_adapters/       # Stored LoRA adapters
│   ├── training_data/       # Training data cache
│   └── checkpoints/         # Model checkpoints
├── scripts/
│   ├── train_domain.py      # Domain training script
│   ├── demo_interactive.py  # Interactive demo
│   └── evaluate_all.py      # Evaluation script
├── src/
│   ├── adapters/            # LoRA adapter management
│   ├── datasets/            # Dataset loading
│   ├── evaluation/          # Evaluation metrics
│   ├── models/              # Teacher/Student models
│   ├── router/              # Query routing
│   ├── training/            # Training pipeline
│   ├── config.py            # Configuration
│   ├── framework.py         # Main orchestrator
│   └── utils.py             # Utilities
├── .env                     # Environment variables
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
└── README.md
```

## Key Components

### Student Model (Qwen 2.5 14B)
- Uses Unsloth for 2x faster training and inference
- 4-bit quantization for memory efficiency
- Domain-specific LoRA adapters (rank 32)

### Teacher Model (GPT-5 mini high)
- OpenAI API for high-quality responses
- Used for complex queries and training data generation
- Evaluates student response quality

### Router
- Multiple routing strategies: perplexity, self-eval, statistics
- Confidence threshold-based routing
- Tracks domain performance statistics

### Training Pipeline
- Online/continual learning support
- Experience replay buffer to prevent forgetting
- Automatic adapter versioning and management

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA r | 32 | LoRA rank |
| LoRA alpha | 32 | LoRA scaling |
| Learning rate | 2e-4 | Training LR |
| Batch size | 4 | Per-device batch |
| Grad accum | 4 | Gradient accumulation |
| Epochs | 3 | Training epochs |
| Router threshold | 0.7 | Confidence threshold |

## API Reference

### Framework Initialization

```python
from src.framework import AdaptiveSLMFramework
from src.config import load_config

config = load_config("configs/config.yaml")
framework = AdaptiveSLMFramework(config)
framework.initialize()
```

### Process Query

```python
result = framework.process_query(
    query="Find all customers who spent more than $100",
    domain="text_to_sql",  # Optional, auto-detected
)
print(result.response)
print(f"Used teacher: {result.used_teacher}")
```

### Train Domain

```python
from src.training import TrainingExample

examples = [
    TrainingExample(query="...", response="...", domain="text_to_sql")
    for ...
]
metrics = framework.train_domain("text_to_sql", examples)
```

### Evaluate

```python
result = framework.evaluate_domain("text_to_sql", max_samples=100)
print(f"Score: {result.score:.2%}")
```

## Development Progress

### Checkpoint: 2026-01-31

#### Completed
- [x] Framework architecture implemented (router, student, teacher, training pipeline)
- [x] Text-to-SQL adapter trained (v3) on Spider dataset (7000 samples, 3 epochs)
- [x] Full evaluation on Spider test set (1034 samples)

#### Results

| Model | Accuracy | Samples |
|-------|----------|---------|
| **LoRA v3 (text_to_sql)** | **74.18%** | 767/1034 |
| Base Model (Qwen 2.5 14B) | 69.05% | 714/1034 |

**Improvement: +5.13%** over base model

#### Adapter Locations
- `data/lora_adapters/text_to_sql/text_to_sql_v3/` - Latest trained adapter
- `data/lora_adapters/registry.json` - Adapter registry

#### Next Steps
1. [ ] Train math_reasoning adapter on GSM8K dataset
   ```bash
   python scripts/train_domain.py --domain math_reasoning --eval-after
   ```
2. [ ] Train code_generation adapter on MBPP dataset
   ```bash
   python scripts/train_domain.py --domain code_generation --eval-after
   ```
3. [ ] Test router functionality across domains
   ```bash
   python main.py demo
   ```
4. [ ] Benchmark full adaptive system (student+router vs teacher-only)
5. [ ] Test online learning from teacher responses

#### Useful Commands
```bash
# Activate environment
source ~/miniconda3/bin/activate mixture-lora

# Compare base vs LoRA on N samples
python scripts/compare_base_vs_lora.py --mode lora --samples 100
python scripts/compare_base_vs_lora.py --mode base --samples 100

# Diagnose model outputs
python scripts/diagnose_eval.py --mode lora --samples 10
python scripts/diagnose_eval.py --mode base --samples 10

# Check adapter registry
cat data/lora_adapters/registry.json
```

---

## Sources

- [Unsloth Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth LoRA Hyperparameters](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [OpenAI GPT-5 mini](https://platform.openai.com/docs/models/gpt-5-mini)
- [Spider Dataset](https://yale-lily.github.io/spider)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [MBPP Dataset](https://huggingface.co/datasets/mbpp)
- [BIRD Benchmark](https://bird-bench.github.io/)

## License

MIT License
