# Lessons Learned

## 2026-02-20: This project is about teacher→student distillation, not text-to-SQL

**Mistake**: Framed plans, code comments, and documentation around SQL execution verification and text-to-SQL specifics. Treated ground truth verification as the core approach. This led to over-engineering SQL-specific solutions and missing the real goal.

**Rule**: The project's core thesis is domain-agnostic online distillation — the small model learns from the large model's responses. Text-to-SQL is just a test domain. Ground truth / SQL execution is a development convenience for measuring progress, not a system component. Never frame implementation decisions around SQL verification as the primary approach. Always ask: "does this work without ground truth?"

## 2026-02-09: Always use full test set for evaluations

**Mistake**: Defaulted to `max_samples=100` when writing evaluation scripts, truncating results to a small subset instead of using the full test set (Spider: 1034, BIRD: 1534).

**Rule**: Never set `max_samples` in evaluation scripts unless the user explicitly requests a smaller sample size. Omit the parameter (or pass `None`) so the entire test split is used. This is also documented in CLAUDE.md under "Workflow Guidelines > Evaluation".

## 2026-02-23: Unsloth fast inference crashes on multi-GPU

**Mistake**: Used `FastLanguageModel.for_inference(model)` and `model.generate()` with `device_map="auto"` across 2 GPUs. Unsloth's paged attention fast inference path (`_flag_for_generation`) triggers CUDA index-out-of-bounds on multi-GPU setups. The `unsloth_fast_generate` wrapper calls `for_inference()` internally, so even disabling the flag before calling `model.generate()` doesn't help.

**Rule**: For inference with Unsloth-loaded models:
1. Use a single available GPU (`CUDA_VISIBLE_DEVICES=<0|1|2|3>`) — the 14B 4-bit model fits in ~10GB VRAM
2. Call `model._old_generate()` instead of `model.generate()` to bypass `unsloth_fast_generate` (which re-enables the crashing fast path)
3. Clear `_flag_for_generation` from all modules before any forward pass: `for m in model.modules(): delattr(m, '_flag_for_generation') if hasattr(m, '_flag_for_generation') else None`
4. Use `model.eval()` + `use_cache=False` for scoring forward passes
