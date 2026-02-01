#!/usr/bin/env python3
"""
Script to train a LoRA adapter for a specific domain.

Usage:
    python scripts/train_domain.py --domain text_to_sql --max-samples 1000
    python scripts/train_domain.py --domain math_reasoning --epochs 3
    python scripts/train_domain.py --domain code_generation --use-teacher-data
"""

import sys
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE PyTorch initializes
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Import unsloth BEFORE other ML libraries for optimizations
import unsloth  # noqa: F401

import argparse

from src.config import load_config
from src.datasets import DatasetLoader
from src.framework import AdaptiveSLMFramework
from src.training import DataProcessor, TrainingExample


def parse_args():
    parser = argparse.ArgumentParser(description="Train a domain-specific LoRA adapter")

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["text_to_sql", "text_to_sql_bird", "math_reasoning", "code_generation"],
        help="Domain to train (text_to_sql_bird uses BIRD dataset)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--use-teacher-data",
        action="store_true",
        help="Generate training data from teacher model",
    )
    parser.add_argument(
        "--teacher-samples",
        type=int,
        default=100,
        help="Number of samples to generate from teacher",
    )
    parser.add_argument(
        "--eval-after",
        action="store_true",
        help="Run evaluation after training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training LoRA adapter for domain: {args.domain}")
    print("=" * 60)

    # Load configuration
    config = load_config(args.config)

    # Initialize framework
    framework = AdaptiveSLMFramework(config)
    framework.initialize(
        load_student=True,
        load_teacher=args.use_teacher_data,
    )

    # Load dataset
    print(f"\nLoading {args.domain} dataset...")
    loader = DatasetLoader()

    if args.domain == "text_to_sql":
        domain_data = loader.load_spider(max_samples=args.max_samples)
        # Use 'prompt' which includes schema context, not raw 'question'
        query_key = "prompt"
    elif args.domain == "text_to_sql_bird":
        domain_data = loader.load_bird(max_samples=args.max_samples, use_local=True)
        query_key = "prompt"
    elif args.domain == "math_reasoning":
        domain_data = loader.load_gsm8k(max_samples=args.max_samples)
        query_key = "question"
    elif args.domain == "code_generation":
        domain_data = loader.load_mbpp(max_samples=args.max_samples)
        query_key = "text"
    else:
        raise ValueError(f"Unknown domain: {args.domain}")

    print(f"Loaded {len(domain_data.train)} training samples")

    # Prepare training examples
    examples = []

    if args.use_teacher_data:
        print(f"\nGenerating {args.teacher_samples} samples from teacher model...")

        for i, sample in enumerate(domain_data.train):
            if i >= args.teacher_samples:
                break

            query = sample[query_key]
            print(f"  Processing sample {i + 1}/{args.teacher_samples}...", end="\r")

            # Get teacher response
            response = framework.teacher.generate_training_response(
                query, args.domain
            )

            examples.append(
                TrainingExample(
                    query=query,
                    response=response,
                    domain=args.domain,
                )
            )

        print(f"\nGenerated {len(examples)} training examples from teacher")
    else:
        # Use dataset directly
        if args.domain == "text_to_sql":
            examples = DataProcessor.from_spider(list(domain_data.train))
        elif args.domain == "text_to_sql_bird":
            examples = DataProcessor.from_bird(list(domain_data.train))
        elif args.domain == "math_reasoning":
            examples = DataProcessor.from_gsm8k(list(domain_data.train))
        elif args.domain == "code_generation":
            examples = DataProcessor.from_mbpp(list(domain_data.train))

    print(f"\nPrepared {len(examples)} training examples")

    # Train
    print("\nStarting training...")
    print("-" * 40)

    result = framework.train_domain(
        args.domain,
        examples=examples,
        num_epochs=args.epochs,
    )

    print("\nTraining complete!")
    print(f"  Loss: {result['training_metrics']['train_loss']:.4f}")
    print(f"  Steps: {result['training_metrics']['train_steps']}")
    print(f"  Adapter saved: {result['adapter_info'].path}")

    # Evaluate if requested
    if args.eval_after:
        print("\nRunning evaluation...")
        eval_result = framework.evaluate_domain(
            args.domain,
            test_dataset=domain_data.test,
            max_samples=100,
        )
        print(f"  Score: {eval_result.score:.2%}")
        print(f"  Correct: {eval_result.correct}/{eval_result.num_samples}")

    print("\nDone!")


if __name__ == "__main__":
    main()
