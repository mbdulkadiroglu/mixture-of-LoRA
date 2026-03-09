# Training & Inference Pipeline Details (Spider + BIRD)

This document traces the exact data flow from raw dataset to trained LoRA adapter, and from evaluation query to scored prediction. Read this to verify no hidden assumptions.

---

## 1. Dataset Loading

### Spider

**Entry point:** `DatasetLoader.load_spider()` -> `_load_spider_local()` (if `spider_data/` exists)

**Raw source files:**
- `spider_data/train_spider.json` — each entry has `question`, `query`, `db_id`
- `spider_data/dev.json` — same format, used as test set

**Schema source:** `spider_data/tables.json` — parsed into `_schemas[db_id]` dict containing `tables`, `columns`, `column_types`, `primary_keys`, `foreign_keys`.

**Processing** (`_process_spider_examples()`):
For each raw entry, builds a `prompt` field:
```
Given the following database schema:

CREATE TABLE users (
  id NUMBER PRIMARY KEY,
  name TEXT
);

CREATE TABLE orders (
  order_id NUMBER PRIMARY KEY,
  user_id NUMBER REFERENCES users(id)
);

Convert this question to SQL: How many orders does each user have?
```

The schema uses CREATE TABLE format with PRIMARY KEY and REFERENCES annotations. If `db_id` is not found in `_schemas`, falls back to `"Convert this question to SQL: {question}"` with no schema.

**Output dataset columns:** `question`, `query`, `db_id`, `text`, `prompt`
- `prompt` = schema + question (the model input)
- `query` = gold SQL (the reference)
- `text` = pre-formatted chat template (used by raw SFT, NOT used by the `DataProcessor` path)

### BIRD

**Entry point:** `DatasetLoader.load_bird()` -> `_load_bird_hf_cleaned()` (default)

**Raw source (HuggingFace cleaned):**
- `birdsql/bird23-train-filtered` — 6601 train samples
- `birdsql/bird_sql_dev_20251106` — 1534 dev samples

Each entry has `question`, `SQL`, `db_id`, `evidence`, `difficulty`.

**Schema source:** `bird_data/**/dev_tables.json` or `train_tables.json` — parsed into `_bird_schemas[db_id]`.

**Processing** (`_process_bird_examples()`):
For each raw entry, builds a `prompt` field:
```
Database schema:
CREATE TABLE department (
  Department_ID NUMBER PRIMARY KEY,
  Name TEXT
);

Hint: The revenue numbers are stored in millions.

Question: How many departments are there?
```

Schema uses the same CREATE TABLE format as Spider (with PK, FK, and column description comments). The `Hint:` line is the BIRD `evidence` field — included if non-empty.

**Output dataset columns:** `question`, `query`, `SQL`, `db_id`, `evidence`, `difficulty`, `text`, `prompt`
- `prompt` = schema + evidence + question
- `SQL` = gold SQL
- `query` = also gold SQL (alias for compatibility)

---

## 2. Training Data Preparation

### From Dataset to TrainingExample

**Spider** — `DataProcessor.from_spider(data)`:
```python
TrainingExample(
    query = entry["prompt"]    # schema + question (if 'prompt' field exists)
    response = entry["query"]  # raw SQL, e.g. "SELECT COUNT(*) FROM singer"
    domain = "text_to_sql"
    metadata = {"db_id": "concert_singer"}
)
```

**BIRD** — `DataProcessor.from_bird(data)`:
```python
TrainingExample(
    query = entry["prompt"]    # schema + evidence + question
    response = entry["SQL"]    # raw SQL, e.g. "SELECT COUNT(*) FROM department"
    domain = "text_to_sql_bird"
    metadata = {"db_id": "department_management", "source": "bird"}
)
```

**Key fact:** Both produce **raw SQL** as the response. No markdown wrapping, no explanation text. This matches the system prompt instruction "output only valid SQLite SQL".

### From TrainingExample to Tokenized Text

`DataProcessor.format_example()` applies the Qwen chat template:

```
<|im_start|>system
You are an expert SQL assistant. Given a database schema and a natural language question, output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL.<|im_end|>
<|im_start|>user
Given the following database schema:

CREATE TABLE singer (
  Singer_ID NUMBER PRIMARY KEY,
  Name TEXT
);

Convert this question to SQL: How many singers are there?<|im_end|>
<|im_start|>assistant
SELECT COUNT(*) FROM singer<|im_end|>
```

**System prompt used** (same for both Spider and BIRD):
> You are an expert SQL assistant. Given a database schema and a natural language question, output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL.

**Note:** Spider uses domain `"text_to_sql"` and BIRD uses `"text_to_sql_bird"` — both map to the same system prompt string.

Examples exceeding `max_seq_length` (default 4096, config overrides to 8192) are dropped with a warning.

### From Formatted Text to HF Dataset

`DataProcessor.prepare_dataset()` produces a HuggingFace `Dataset` with columns:
- `text` — the full chat-templated string (system + user + assistant)
- `domain` — `"text_to_sql"` or `"text_to_sql_bird"`

This is what SFTTrainer consumes.

---

## 3. LoRA Training

**Entry point:** `train_domain.py` -> `framework.train_domain()` -> `LoRATrainer.train()`

### Model Setup

1. `StudentModel.load_model()` — loads `Qwen/Qwen2.5-14B-Instruct` via Unsloth `FastLanguageModel.from_pretrained()` in 4-bit quantization, bfloat16.
2. `StudentModel.setup_lora()` — applies LoRA with `r=32`, `alpha=32`, targeting `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Uses Unsloth's `get_peft_model()`.

### Training Loop

`LoRATrainer.train()`:
1. **Replay mixing** — if replay buffer is non-empty, mixes 20% old examples into the new dataset (`replay_ratio: 0.2`). This is for continual learning to prevent catastrophic forgetting.
2. **SFTTrainer config** — `SFTConfig` with:
   - batch_size=4, gradient_accumulation=4 (effective batch=16)
   - lr=2e-4, cosine scheduler, warmup_ratio=0.1
   - adamw_8bit optimizer, bf16
   - 3 epochs (default), max_seq_length from config
3. **Training** — standard HuggingFace `SFTTrainer.train()`. The dataset `text` column is the pre-formatted chat string.

### Adapter Saving

After training, adapter is saved to `data/lora_adapters/{domain}/latest/` with versioning via `AdapterManager.register_adapter()`.

**Both Spider and BIRD train a `text_to_sql` adapter** — BIRD examples use domain `"text_to_sql_bird"` for tracking but the adapter path could be the same or separate depending on how you invoke training.

---

## 4. Inference (Evaluation)

### Entry Point

`framework.evaluate_domain(domain)` -> `Evaluator.evaluate_dataset()`

### Adapter Loading

For `text_to_sql_bird` domain, the framework looks up the adapter using domain `"text_to_sql"`:
```python
adapter_domain = "text_to_sql" if domain == "text_to_sql_bird" else domain
adapter_path = self.adapter_manager.get_adapter_path(adapter_domain)
```
This means BIRD evaluation uses the same LoRA adapter as Spider.

### Query Construction at Inference

For each test example, `evaluate_dataset()` builds:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query},
]
prediction = model.generate_chat(messages, temperature=0.1)
```

Where:
- `system_prompt` = same prompt used in training (verified by tests)
- `query` = the `prompt` field from the dataset (schema + question, or schema + evidence + question for BIRD)
- `temperature` = 0.1 (near-deterministic for evaluation)

**The `generate_chat()` method** applies `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` — this produces:
```
<|im_start|>system
You are an expert SQL assistant. ...<|im_end|>
<|im_start|>user
{schema + question}<|im_end|>
<|im_start|>assistant

```
The model then generates tokens to complete the assistant turn.

### Reference Extraction

| Domain | Query field | Reference field |
|--------|-----------|----------------|
| `text_to_sql` (Spider) | `prompt` | `query` |
| `text_to_sql_bird` (BIRD) | `prompt` | `SQL` (falls back to `query`) |

### Scoring

**Execution-based** (preferred if databases are available):
- Spider: `SQLExecutor` looks up `spider_data/database/{db_id}/{db_id}.sqlite`, executes both predicted and reference SQL, compares result sets (order-independent).
- BIRD: `BIRDExecutor` looks up databases in `bird_data/**/dev_databases/{db_id}/{db_id}.sqlite` or `train_databases/`, same execution comparison.

**Fallback** (if no database files): `_sql_exact_match()` — extracts SQL from any markdown wrapping, normalizes (lowercase, whitespace, remove semicolons, normalize quotes), then compares strings.

---

## 5. Consistency Verification

These are the things that MUST match between training and inference:

| Aspect | Training (DataProcessor) | Inference (Evaluator / Framework) | Match? |
|--------|-------------------------|----------------------------------|--------|
| System prompt (text_to_sql) | "...output only the SQL query. Do not include any explanation, formatting, or markdown. Output only valid SQLite SQL." | Same string | YES |
| System prompt (text_to_sql_bird) | Same as text_to_sql | Same string | YES |
| Response format | Raw SQL (no markdown) | Model expected to output raw SQL | YES |
| Schema format (Spider) | CREATE TABLE with PK/FK | Same via `get_schema_string()` | YES |
| Schema format (BIRD) | CREATE TABLE with PK/FK + column descriptions | Same via `get_bird_schema_string()` | YES |
| Chat template | Qwen `apply_chat_template` | Same tokenizer, same `apply_chat_template` | YES |

**Verified by tests in `tests/test_pipeline_fixes.py`:**
- `TestSystemPromptConsistency` — asserts DataProcessor, Evaluator, and Framework all use identical prompt strings
- `TestSpiderResponseFormat` — asserts from_spider() produces raw SQL, no markdown
- `TestBirdResponseFormat` — asserts from_bird() produces raw SQL
- `TestSpiderSchemaFormat` — asserts CREATE TABLE format with PK/FK

---

## 6. Potential Gotchas & Assumptions

### BIRD domain naming
BIRD uses `domain="text_to_sql_bird"` for training examples but evaluation falls back to the `"text_to_sql"` adapter. This means:
- If you train only on Spider data, BIRD eval uses the Spider adapter.
- If you train with `--domain text_to_sql_bird`, the adapter is saved under `text_to_sql_bird/latest/` but evaluation will look it up under `text_to_sql` (due to the `adapter_domain` mapping in `evaluate_domain()`).
- **To train a joint adapter**, you would need to combine Spider and BIRD examples under the same domain name, or modify the adapter lookup logic.

### Schema availability
If `tables.json` (Spider) or `dev_tables.json` (BIRD) is not found, schemas are silently empty. The prompt degrades to just `"Convert this question to SQL: {question}"` with no schema — which will produce poor results. No error is raised.

### BIRD evidence field
The BIRD `evidence` field (hints about the data) is included in training prompts and evaluation prompts. If a test example has no evidence, the prompt just omits the `Hint:` line. This is fine — no empty "Hint:" is inserted.

### Replay buffer
When training BIRD after having trained Spider, the replay buffer may contain Spider examples. These get mixed in at 20% ratio. The examples have different domain tags (`text_to_sql` vs `text_to_sql_bird`) but the same system prompt, so this cross-pollination should be beneficial rather than harmful.

### Token length
Config YAML sets `max_seq_length: 8192` (overriding the Python default of 4096). BIRD prompts with full CREATE TABLE schemas can be long. Examples exceeding 8192 tokens are silently dropped during `prepare_dataset()`.

### Pre-formatted `text` field in dataset
The `DatasetLoader._process_spider_examples()` and `_process_bird_examples()` both create a `text` field with hardcoded Qwen chat tokens (`<|im_start|>user`...). This field is **NOT used** by the `DataProcessor` training path — it's only relevant if you bypass `DataProcessor` and feed the dataset directly to SFTTrainer. The `DataProcessor` path uses `prompt` + `query`/`SQL` fields and applies the tokenizer's own chat template.
