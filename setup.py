"""
Setup script for the Adaptive SLM Framework.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-slm-framework",
    version="0.1.0",
    description="Adaptive Small Language Model Framework with LoRA adapters",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "datasets>=2.20.0",
        "accelerate>=0.34.0",
        "unsloth>=2024.12",
        "peft>=0.13.0",
        "openai>=1.50.0",
        "trl>=0.11.0",
        "bitsandbytes>=0.44.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "ruff>=0.5.0",
        ],
        "api": [
            "fastapi>=0.115.0",
            "uvicorn>=0.30.0",
            "pydantic>=2.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-slm=main:main",
        ],
    },
)
