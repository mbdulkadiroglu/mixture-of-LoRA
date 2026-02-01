#!/usr/bin/env python3
"""
Interactive demo of the Adaptive SLM Framework.

This script demonstrates the core functionality:
1. Query routing between student and teacher models
2. Online learning from teacher responses
3. Domain-specific LoRA adapter switching

Usage:
    python scripts/demo_interactive.py
    python scripts/demo_interactive.py --teacher-only  # For comparison
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.framework import AdaptiveSLMFramework


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Adaptive SLM Demo")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="Use teacher model only (for comparison)",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Disable online training",
    )

    return parser.parse_args()


def print_header():
    print("=" * 70)
    print("  Adaptive Small Language Model Framework - Interactive Demo")
    print("=" * 70)
    print()
    print("This demo showcases the teacher-student learning system.")
    print("The router will decide whether to use the student model (with LoRA)")
    print("or forward to the teacher model (GPT-5 mini high).")
    print()
    print("Commands:")
    print("  /stats    - Show framework statistics")
    print("  /train    - Train on collected examples")
    print("  /eval     - Run quick evaluation")
    print("  /quit     - Exit the demo")
    print()
    print("-" * 70)


def print_result(result):
    print()
    print(f"Domain: {result.domain}")
    print(f"Model: {'Teacher (GPT-5 mini)' if result.used_teacher else 'Student (Qwen 2.5 14B)'}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Routing reason: {result.routing_decision.reason}")
    print()
    print("Response:")
    print("-" * 40)
    print(result.response)
    print("-" * 40)


def main():
    args = parse_args()

    print_header()

    # Load configuration
    config = load_config(args.config)

    # Initialize framework
    print("Initializing framework...")
    framework = AdaptiveSLMFramework(config)
    framework.initialize(
        load_student=not args.teacher_only,
        load_teacher=True,
    )
    print("Ready!\n")

    # Interactive loop
    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            # Handle commands
            if query.startswith("/"):
                if query == "/quit" or query == "/exit":
                    print("Goodbye!")
                    break
                elif query == "/stats":
                    stats = framework.get_statistics()
                    print("\nFramework Statistics:")
                    print(f"  Domains: {stats['domains']}")
                    print(f"  Pending training examples: {stats['pending_training_examples']}")
                    print(f"  Replay buffer size: {stats['replay_buffer_size']}")
                    for domain, domain_stats in stats['adapters'].items():
                        print(f"\n  {domain}:")
                        for key, value in domain_stats.items():
                            print(f"    {key}: {value}")
                    continue
                elif query == "/train":
                    print("\nTraining on collected examples...")
                    for domain in ["text_to_sql", "math_reasoning", "code_generation"]:
                        try:
                            result = framework.train_domain(domain)
                            print(f"  {domain}: trained on {result['adapter_info'].training_samples} samples")
                        except ValueError as e:
                            print(f"  {domain}: {e}")
                    continue
                elif query == "/eval":
                    print("\nRunning quick evaluation...")
                    for domain in framework.adapter_manager.list_domains():
                        try:
                            result = framework.evaluate_domain(domain, max_samples=50)
                            print(f"  {domain}: {result.score:.2%}")
                        except Exception as e:
                            print(f"  {domain}: Error - {e}")
                    continue
                else:
                    print(f"Unknown command: {query}")
                    continue

            # Process query
            result = framework.process_query(
                query,
                force_teacher=args.teacher_only,
                collect_training_data=not args.no_training,
            )

            print_result(result)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
