"""
Cascade student model — wraps StudentModel with log-prob extraction.

Key addition: generate_with_logprobs() extracts per-token log-probs and
entropy during generation, used by the router for confidence-based routing.

NOTE: Unsloth's fast inference path (paged attention) crashes with CUDA
index-out-of-bounds on multi-GPU device_map="auto". We bypass it by:
1. Not calling FastLanguageModel.for_inference() for generation
2. Using model._old_generate() to skip unsloth_fast_generate wrapper
3. Clearing _flag_for_generation from all modules before forward passes
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from .config import CascadeConfig


@dataclass
class GenerationResult:
    text: str
    token_log_probs: list[float]
    mean_log_prob: float
    min_log_prob: float
    mean_entropy: float
    num_tokens: int


class CascadeStudent:
    def __init__(self, config: CascadeConfig):
        self.config = config
        self._student = None  # Lazy init — need unsloth import order

    def load(self) -> None:
        """Load the base student model."""
        from src.models.student import StudentModel

        student_config = self.config.to_student_model_config()
        lora_config = self.config.to_lora_config()
        self._student = StudentModel(student_config, lora_config)
        self._student.load_model()
        logger.info("CascadeStudent: base model loaded")

    def load_adapter(self, path: str | Path) -> None:
        self._student.load_adapter(path)

    def unload_adapter(self) -> None:
        self._student.unload_adapter()

    def setup_for_training(self) -> None:
        """Setup LoRA for training (called before first training round)."""
        self._student.setup_lora()

    def save_adapter(self, path: str | Path) -> None:
        self._student.save_adapter(path)

    @property
    def inner(self):
        """Access the wrapped StudentModel (for trainer)."""
        return self._student

    @property
    def tokenizer(self):
        return self._student.tokenizer

    @property
    def is_loaded(self) -> bool:
        return self._student is not None and self._student.is_loaded

    def _disable_fast_inference(self, model) -> None:
        """
        Disable Unsloth fast generation flags safely.

        Some PEFT/LoRA wrapper paths assume `_flag_for_generation` exists on
        wrapped modules. Set existing flags to False (instead of deleting) to
        avoid AttributeError paths in mixed Unsloth/PEFT execution.
        """
        for module in model.modules():
            if hasattr(module, "_flag_for_generation"):
                try:
                    setattr(module, "_flag_for_generation", False)
                except Exception:
                    # Some wrappers may proxy attributes in ways that disallow
                    # direct assignment on the outer module object.
                    pass

    def generate_with_logprobs(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> GenerationResult:
        """
        Generate text and extract per-token log-probs + entropy.

        Two-pass approach:
        1. Generate text (bypassing Unsloth fast inference to avoid multi-GPU crash)
        2. Forward pass on prompt+output to extract logits for log-probs
        """
        model = self._student.peft_model or self._student.model

        # Ensure Unsloth fast inference path is OFF (crashes on multi-GPU)
        self._disable_fast_inference(model)
        model.eval()

        # Build prompt via chat template
        prompt = self._student.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._student.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self._student.config.max_seq_length - max_new_tokens,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # Pass 1: generate text.
        # Use _old_generate to bypass unsloth_fast_generate (which calls
        # for_inference() internally and re-enables the crashing fast path).
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self._student.tokenizer.pad_token_id,
            "eos_token_id": self._student.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k

        with torch.inference_mode():
            output_ids = model._old_generate(**generation_kwargs)

        generated_ids = output_ids[0, input_len:]

        if len(generated_ids) == 0:
            return GenerationResult(
                text="", token_log_probs=[], mean_log_prob=0.0,
                min_log_prob=0.0, mean_entropy=0.0, num_tokens=0,
            )

        # Pass 2: forward pass on full sequence to extract logits.
        # Ensure fast inference path is still off after generate.
        self._disable_fast_inference(model)
        full_ids = output_ids  # [1, prompt_len + gen_len]

        with torch.inference_mode():
            out = model(input_ids=full_ids, use_cache=False)
            logits = out.logits[0]  # [seq_len, vocab_size]

        token_log_probs = []
        entropies = []

        for i in range(len(generated_ids)):
            pos = input_len - 1 + i  # logits at pos predict token at pos+1
            step_logits = logits[pos].float()  # float32 for numerical stability
            log_probs = F.log_softmax(step_logits, dim=-1)
            probs = log_probs.exp()

            token_id = generated_ids[i]
            token_lp = log_probs[token_id].item()
            token_log_probs.append(token_lp)

            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)

        text = self._student.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        num_tokens = len(token_log_probs)

        return GenerationResult(
            text=text,
            token_log_probs=token_log_probs,
            mean_log_prob=sum(token_log_probs) / num_tokens,
            min_log_prob=min(token_log_probs),
            mean_entropy=sum(entropies) / num_tokens,
            num_tokens=num_tokens,
        )
