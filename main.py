#!/usr/bin/env python3
"""
Main entry point for the Adaptive SLM Framework.

This script provides a CLI interface to the framework's main functionalities.

Usage:
    python main.py train --domain text_to_sql
    python main.py evaluate --domain math_reasoning
    python main.py serve --port 8000
    python main.py demo
"""

# Load environment variables FIRST (before PyTorch initializes)
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

# Import unsloth BEFORE other ML libraries for optimizations
import unsloth  # noqa: F401

import argparse
import sys

from src.config import load_config
from src.framework import AdaptiveSLMFramework
from src.datasets import DatasetLoader, get_dataset_info
from src.utils import setup_logging


def cmd_train(args):
    """Train a domain-specific LoRA adapter."""
    from scripts.train_domain import main as train_main

    sys.argv = [
        "train_domain.py",
        "--domain", args.domain,
        "--config", args.config,
    ]
    if args.max_samples:
        sys.argv.extend(["--max-samples", str(args.max_samples)])
    if args.epochs:
        sys.argv.extend(["--epochs", str(args.epochs)])
    if args.use_teacher:
        sys.argv.append("--use-teacher-data")
    if args.eval:
        sys.argv.append("--eval-after")

    train_main()


def cmd_evaluate(args):
    """Evaluate model performance."""
    from scripts.evaluate_all import main as eval_main

    sys.argv = [
        "evaluate_all.py",
        "--config", args.config,
        "--samples", str(args.samples),
    ]
    if args.domains:
        sys.argv.extend(["--domains"] + args.domains)
    if args.include_teacher:
        sys.argv.append("--include-teacher")
    if args.output:
        sys.argv.extend(["--output", args.output])

    eval_main()


def cmd_demo(args):
    """Run interactive demo."""
    from scripts.demo_interactive import main as demo_main

    sys.argv = ["demo_interactive.py", "--config", args.config]
    if args.teacher_only:
        sys.argv.append("--teacher-only")
    if args.no_training:
        sys.argv.append("--no-training")

    demo_main()


def cmd_serve(args):
    """Start API server."""
    print("Starting API server...")
    print("Note: API server implementation is a TODO item.")
    print(f"Would start on port {args.port}")

    # TODO: Implement FastAPI server
    # from src.api import create_app
    # import uvicorn
    # app = create_app(args.config)
    # uvicorn.run(app, host=args.host, port=args.port)


def cmd_info(args):
    """Show information about the framework."""
    print("=" * 60)
    print("  Adaptive Small Language Model Framework")
    print("=" * 60)
    print()

    print("Available Datasets:")
    print("-" * 40)
    for name, info in get_dataset_info().items():
        print(f"  {info['name']} ({name})")
        print(f"    Domain: {info['domain']}")
        print(f"    Size: {info['size']}")
        print(f"    Source: {info['source']}")
        print()

    print("Models:")
    print("-" * 40)
    config = load_config(args.config)
    print(f"  Teacher: {config.teacher.name}")
    print(f"  Student: {config.student.name}")
    print()

    print("Configuration:")
    print("-" * 40)
    print(f"  LoRA rank: {config.lora.r}")
    print(f"  LoRA alpha: {config.lora.lora_alpha}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Router threshold: {config.router.confidence_threshold}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Small Language Model Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --domain text_to_sql --use-teacher
  python main.py evaluate --domains text_to_sql math_reasoning
  python main.py demo
  python main.py info
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a LoRA adapter")
    train_parser.add_argument("--domain", required=True, help="Domain to train")
    train_parser.add_argument("--max-samples", type=int, help="Max training samples")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--use-teacher", action="store_true", help="Use teacher for data")
    train_parser.add_argument("--eval", action="store_true", help="Evaluate after training")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--domains", nargs="+", help="Domains to evaluate")
    eval_parser.add_argument("--samples", type=int, default=100, help="Samples per domain")
    eval_parser.add_argument("--include-teacher", action="store_true", help="Evaluate teacher too")
    eval_parser.add_argument("--output", type=str, help="Output file for results")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--teacher-only", action="store_true", help="Use teacher only")
    demo_parser.add_argument("--no-training", action="store_true", help="Disable online training")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show framework information")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Dispatch to appropriate command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
