"""
Student model interface using Qwen 2.5 14B with Unsloth.
"""

from pathlib import Path
from typing import Generator

import torch
from loguru import logger

from ..config import StudentModelConfig, LoRAConfig


class StudentModel:
    """
    Student model wrapper for Qwen 2.5 14B using Unsloth.

    This model is the small language model that learns from the teacher
    through domain-specific LoRA adapters.
    """

    def __init__(
        self,
        config: StudentModelConfig,
        lora_config: LoRAConfig | None = None,
        device_map: str = "auto",
    ):
        """
        Initialize the student model.

        Args:
            config: Student model configuration.
            lora_config: LoRA configuration for adapter training.
            device_map: Device mapping strategy.
        """
        self.config = config
        self.lora_config = lora_config or LoRAConfig()
        self.device_map = device_map

        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self._current_adapter = None

        logger.info(f"StudentModel initialized for: {config.name}")

    def load_model(self) -> None:
        """Load the base model using Unsloth."""
        from unsloth import FastLanguageModel

        logger.info(f"Loading model: {self.config.name}")

        # Determine dtype
        dtype = None
        if self.config.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.config.dtype == "float16":
            dtype = torch.float16

        # Load with Unsloth's optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.name,
            max_seq_length=self.config.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.config.load_in_4bit,
            device_map=self.device_map,
        )

        # Ensure proper pad token handling for Qwen
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded successfully. Device map: {self.device_map}")

    def setup_lora(self) -> None:
        """Configure LoRA for the model using Unsloth."""
        from unsloth import FastLanguageModel

        if self.model is None:
            raise RuntimeError("Model must be loaded before setting up LoRA")

        logger.info("Setting up LoRA configuration")

        self.peft_model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=42,
            max_seq_length=self.config.max_seq_length,
            use_rslora=self.lora_config.use_rslora,
        )

        logger.info(f"LoRA configured with r={self.lora_config.r}, alpha={self.lora_config.lora_alpha}")

    def load_adapter(self, adapter_path: str | Path) -> None:
        """
        Load a pre-trained LoRA adapter.

        Args:
            adapter_path: Path to the adapter directory.
        """
        from peft import PeftModel

        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        if self.model is None:
            self.load_model()

        logger.info(f"Loading adapter from: {adapter_path}")

        # Load the adapter
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            str(adapter_path),
            is_trainable=False,
        )
        self._current_adapter = str(adapter_path)

        logger.info(f"Adapter loaded: {adapter_path.name}")

    def save_adapter(self, output_path: str | Path) -> None:
        """
        Save the current LoRA adapter.

        Args:
            output_path: Path to save the adapter.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.peft_model is None:
            raise RuntimeError("No LoRA adapter to save")

        logger.info(f"Saving adapter to: {output_path}")

        # Save adapter weights
        self.peft_model.save_pretrained(str(output_path))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))

        logger.info("Adapter saved successfully")

    def merge_adapter(self) -> None:
        """Merge the LoRA adapter into the base model."""
        if self.peft_model is None:
            raise RuntimeError("No LoRA adapter to merge")

        logger.info("Merging adapter into base model")

        self.model = self.peft_model.merge_and_unload()
        self.peft_model = None
        self._current_adapter = None

        logger.info("Adapter merged successfully")

    def unload_adapter(self) -> None:
        """Unload the current adapter and restore base model."""
        if self.peft_model is not None:
            # Properly disable adapter - get_base_model() doesn't work because
            # PEFT modifies the model in-place. We need to reload the base model.
            del self.peft_model
            del self.model
            self.peft_model = None
            self.model = None
            self._current_adapter = None

            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reload clean base model
            self.load_model()
            logger.info("Adapter unloaded - base model reloaded")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stop_strings: list[str] | None = None,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling.
            do_sample: Whether to use sampling.
            stop_strings: Optional stop strings.
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty).

        Returns:
            Generated response text.
        """
        from unsloth import FastLanguageModel

        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Use the appropriate model (with or without adapter)
        model = self.peft_model if self.peft_model is not None else self.model

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        ).to(model.device)

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Handle stop strings
        if stop_strings:
            for stop in stop_strings:
                if stop in response:
                    response = response.split(stop)[0]

        return response.strip()

    def generate_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate a response in chat format.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response text.
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming output.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum new tokens.
            temperature: Sampling temperature.

        Yields:
            Generated tokens one at a time.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        from unsloth import FastLanguageModel

        if self.model is None:
            raise RuntimeError("Model not loaded")

        model = self.peft_model if self.peft_model is not None else self.model
        FastLanguageModel.for_inference(model)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Start generation in a thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they're generated
        for token in streamer:
            yield token

        thread.join()

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity for a given text.

        This can be used for routing decisions - lower perplexity
        suggests the model is more confident about the text.

        Args:
            text: Text to evaluate.

        Returns:
            Perplexity score.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        model = self.peft_model if self.peft_model is not None else self.model

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return torch.exp(loss).item()

    def get_trainable_model(self):
        """
        Get the model configured for training.

        Returns:
            The PEFT model ready for training.
        """
        from unsloth import FastLanguageModel

        if self.peft_model is None:
            raise RuntimeError("LoRA must be set up before training")

        FastLanguageModel.for_training(self.peft_model)
        return self.peft_model

    @property
    def current_adapter(self) -> str | None:
        """Get the currently loaded adapter path."""
        return self._current_adapter

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
