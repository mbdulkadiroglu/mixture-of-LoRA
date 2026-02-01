"""
Teacher model interface using OpenAI GPT-5 mini API.
"""

import asyncio
from typing import AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from ..config import TeacherModelConfig


class TeacherModel:
    """
    Teacher model wrapper for GPT-5 mini via OpenAI API.

    This model serves as the oracle/teacher that provides high-quality
    responses for training the student model.
    """

    def __init__(self, config: TeacherModelConfig):
        """
        Initialize the teacher model.

        Args:
            config: Teacher model configuration.
        """
        self.config = config
        self.model_name = self._get_model_id(config.name)

        # Initialize OpenAI clients
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )

        logger.info(f"Initialized teacher model: {self.model_name}")

    def _get_model_id(self, name: str) -> str:
        """Map friendly name to API model ID."""
        model_mapping = {
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-mini-low": "gpt-5-mini",
            "gpt-5-mini-high": "gpt-5-mini",
            "gpt-5": "gpt-5",
        }
        return model_mapping.get(name, name)

    def _format_input(self, messages: list[dict]) -> str:
        """
        Format messages list into a single input string for the responses API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            Formatted input string.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """
        Generate a response synchronously.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            max_tokens: Maximum tokens in response (not used in new API).
            temperature: Sampling temperature (not used in new API).
            stop: Stop sequences (not used in new API).

        Returns:
            Generated response text.
        """
        try:
            input_text = self._format_input(messages)
            response = self.client.responses.create(
                model=self.model_name,
                input=input_text,
            )
            return response.output_text

        except Exception as e:
            logger.error(f"Teacher model generation failed: {e}")
            raise

    async def generate_async(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """
        Generate a response asynchronously.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            stop: Stop sequences.

        Returns:
            Generated response text.
        """
        try:
            input_text = self._format_input(messages)
            response = await self.async_client.responses.create(
                model=self.model_name,
                input=input_text,
            )
            return response.output_text

        except Exception as e:
            logger.error(f"Teacher model async generation failed: {e}")
            raise

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a response with streaming.

        Args:
            messages: List of message dictionaries.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Yields:
            Response text chunks.
        """
        try:
            input_text = self._format_input(messages)
            stream = await self.async_client.responses.create(
                model=self.model_name,
                input=input_text,
                stream=True,
            )

            async for chunk in stream:
                if hasattr(chunk, 'delta') and chunk.delta:
                    yield chunk.delta

        except Exception as e:
            logger.error(f"Teacher model streaming failed: {e}")
            raise

    def batch_generate(
        self,
        batch_messages: list[list[dict]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """
        Generate responses for a batch of message lists.

        Args:
            batch_messages: List of message lists.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of generated responses.
        """
        # Run async batch generation
        return asyncio.run(
            self._batch_generate_async(batch_messages, max_tokens, temperature)
        )

    async def _batch_generate_async(
        self,
        batch_messages: list[list[dict]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> list[str]:
        """Async implementation of batch generation."""
        tasks = [
            self.generate_async(messages, max_tokens, temperature)
            for messages in batch_messages
        ]
        return await asyncio.gather(*tasks)

    def evaluate_confidence(
        self,
        query: str,
        student_response: str,
        domain: str,
    ) -> tuple[float, str]:
        """
        Evaluate the confidence/quality of a student response.

        This is used by the router to decide if the student's response
        is good enough or if the teacher should provide a better one.

        Args:
            query: The original query.
            student_response: The student model's response.
            domain: The domain of the query.

        Returns:
            Tuple of (confidence_score, feedback).
        """
        eval_prompt = f"""You are evaluating the quality of a response from a smaller language model.

Domain: {domain}

Original Query:
{query}

Student Model Response:
{student_response}

Evaluate this response on a scale of 0.0 to 1.0 where:
- 1.0 = Perfect, completely correct and well-formed
- 0.8 = Very good, minor improvements possible
- 0.6 = Acceptable, some issues but usable
- 0.4 = Problematic, significant errors
- 0.2 = Poor, mostly incorrect
- 0.0 = Completely wrong or nonsensical

Respond with ONLY a JSON object in this format:
{{"score": <float>, "feedback": "<brief explanation>"}}"""

        messages = [{"role": "user", "content": eval_prompt}]

        try:
            response = self.generate(messages, max_tokens=200, temperature=0.1)

            # Parse the JSON response
            import json

            # Extract JSON from response (handle potential extra text)
            json_match = response.strip()
            if json_match.startswith("```"):
                json_match = json_match.split("```")[1]
                if json_match.startswith("json"):
                    json_match = json_match[4:]

            result = json.loads(json_match)
            return float(result["score"]), result.get("feedback", "")

        except Exception as e:
            logger.warning(f"Failed to parse teacher evaluation: {e}")
            # Return a conservative estimate
            return 0.5, "Evaluation parsing failed"

    def generate_training_response(
        self,
        query: str,
        domain: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a high-quality response specifically for training.

        Uses lower temperature and explicit instructions for consistency.

        Args:
            query: The user query.
            domain: The domain of the query.
            system_prompt: Optional domain-specific system prompt.

        Returns:
            High-quality response for training.
        """
        default_system_prompts = {
            "text_to_sql": """You are an expert SQL assistant. Generate precise, efficient SQL queries.
Always explain your reasoning step by step before providing the final SQL.
Format the final SQL query in a code block.""",
            "math_reasoning": """You are a mathematics tutor. Solve problems step by step.
Show all your work and explain each step clearly.
Provide the final numerical answer clearly marked.""",
            "code_generation": """You are an expert Python programmer. Write clean, efficient, well-documented code.
Include docstrings and type hints where appropriate.
Format code in Python code blocks.""",
            "general": """You are a helpful, accurate, and concise assistant.
Provide clear and well-structured responses.""",
        }

        system = system_prompt or default_system_prompts.get(domain, default_system_prompts["general"])

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        return self.generate(messages, temperature=0.3)
