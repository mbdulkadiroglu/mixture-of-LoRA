#!/usr/bin/env python3
"""
Quick start example demonstrating the core workflow.

This script shows how to:
1. Initialize the framework
2. Process queries with automatic routing
3. Collect and train on teacher responses
4. Evaluate performance

Run this script to verify your installation works.
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


def main():
    print("=" * 70)
    print("  Adaptive SLM Framework - Quick Start Example")
    print("=" * 70)
    print()

    # Step 1: Import and configure
    print("Step 1: Importing framework...")
    from src.config import load_config
    from src.framework import AdaptiveSLMFramework
    from src.datasets import DatasetLoader, get_dataset_info

    # Show available datasets
    print("\nAvailable datasets:")
    for name, info in get_dataset_info().items():
        print(f"  - {info['name']}: {info['description']}")

    # Step 2: Initialize
    print("\nStep 2: Initializing framework...")
    print("  (This will load Qwen 2.5 14B - may take a few minutes)")

    config = load_config("configs/config.yaml")

    # For quick testing, you can disable student model loading
    # by setting load_student=False
    framework = AdaptiveSLMFramework(config)

    # Initialize with teacher only for API testing
    # Change to load_student=True for full functionality
    framework.initialize(
        load_student=False,  # Set to True to load Qwen 2.5 14B
        load_teacher=True,
    )

    print("  Framework initialized!")

    # Step 3: Test teacher model
    print("\nStep 3: Testing teacher model (GPT-5 mini high)...")

    test_queries = [
        ("Find all employees earning more than $50000", "text_to_sql"),
        ("If John has 5 apples and buys 3 more, how many does he have?", "math_reasoning"),
        ("Write a function to check if a number is prime", "code_generation"),
    ]

    for query, expected_domain in test_queries:
        print(f"\n  Query: {query}")
        print(f"  Expected domain: {expected_domain}")

        result = framework.process_query(
            query,
            force_teacher=True,  # Force teacher since student not loaded
        )

        print(f"  Detected domain: {result.domain}")
        print(f"  Response preview: {result.response[:100]}...")

    # Step 4: Show statistics
    print("\n" + "=" * 70)
    print("Step 4: Framework Statistics")
    print("=" * 70)

    stats = framework.get_statistics()
    print(f"  Registered domains: {stats['domains']}")
    print(f"  Pending training examples: {stats['pending_training_examples']}")
    print(f"  Replay buffer size: {stats['replay_buffer_size']}")

    # Step 5: Next steps
    print("\n" + "=" * 70)
    print("Quick Start Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Set load_student=True in this script to load the full model")
    print("  2. Run 'python main.py train --domain text_to_sql' to train an adapter")
    print("  3. Run 'python main.py demo' for interactive testing")
    print("  4. Run 'python main.py evaluate' to benchmark performance")
    print()
    print("Make sure to configure your .env file with:")
    print("  OPENAI_API_KEY=your_key_here")
    print("  HF_TOKEN=your_token_here")


if __name__ == "__main__":
    main()
