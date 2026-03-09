# RouterBench Dataset Analysis

**Source**: [withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench) (HuggingFace)
**Paper**: [RouterBench: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031) (ICML 2024)
**Authors**: Qitian Jason Hu et al. (UC Berkeley + Martian)
**Local path**: `data/routerbench_cache/datasets--withmartian--routerbench/`

---

## 1. Overview

RouterBench is the first comprehensive benchmark for evaluating **multi-LLM routing systems** -- systems that decide which LLM to send a query to, balancing quality against cost. It provides pre-computed responses and scores from 11 LLMs across 86 tasks, enabling offline evaluation of any routing strategy.

**Key stats**: 36,497 unique prompts, 11 models, 86 tasks, 401,467 total model-prompt pairs.

---

## 2. Files and Formats

| File | Size | Format | Shape | Description |
|------|------|--------|-------|-------------|
| `routerbench_raw.pkl` | 1,142.6 MB | Long | (401,467 x 8) | One row per (model, prompt) pair |
| `routerbench_0shot.pkl` | 95.0 MB | Wide | (36,497 x 37) | One row per prompt, 0-shot evaluation |
| `routerbench_5shot.pkl` | 163.2 MB | Wide | (36,511 x 37) | One row per prompt, 5-shot evaluation |

All files are **Python pickle** format. Load with `pd.read_pickle()`.

**Note**: 0-shot and 5-shot files have identical scores for GSM8K and MBPP (these benchmarks were evaluated with their own prompting strategies, independent of the 0/5-shot distinction). The 0/5-shot difference primarily affects MMLU and similar multiple-choice benchmarks.

---

## 3. Schema

### 3.1 Raw File (Long Format)

| Column | Type | Description |
|--------|------|-------------|
| `index` | int64 | Row index |
| `sample_id` | str | Unique ID: `{eval_name}.{split}.{index}` (e.g., `grade-school-math.dev.0`) |
| `prompt` | str | Full prompt text sent to the model |
| `eval_name` | str | Task/benchmark name (e.g., `grade-school-math`, `mbpp`, `hellaswag`) |
| `performance` | str (numeric) | Score: 0.0, 0.25, 0.5, 0.75, or 1.0 (also 0.1-0.9 for GPT-4-judged tasks) |
| `model_response` | str | The model's generated output |
| `model_name` | str | Which LLM generated this response |
| `cost` | float64 | USD cost of this inference |

### 3.2 Wide Files (0-shot / 5-shot)

| Column Pattern | Count | Type | Description |
|----------------|-------|------|-------------|
| `sample_id` | 1 | str | Unique prompt identifier |
| `prompt` | 1 | str | Full prompt text |
| `eval_name` | 1 | str | Task name |
| `{model_name}` | 11 | object | Performance score for each model |
| `{model_name}\|model_response` | 11 | str | Full response text for each model |
| `{model_name}\|total_cost` | 11 | float64 | USD cost for each model |
| `oracle_model_to_route_to` | 1 | str | Cheapest model that answered correctly |

---

## 4. Models (11)

| Model | Type | Avg Accuracy (0-shot) | Avg Cost/Query | $/Correct | Total Cost |
|-------|------|----------------------|----------------|-----------|------------|
| gpt-4-1106-preview | Proprietary | **0.7814** | $0.003293 | $0.004214 | $120.18 |
| zero-one-ai/Yi-34B-Chat | Open-source | 0.6475 | $0.000186 | $0.000287 | $6.77 |
| claude-v2 | Proprietary | 0.6358 | $0.002419 | $0.003804 | $88.27 |
| claude-v1 | Proprietary | 0.6301 | $0.002145 | $0.003404 | $78.27 |
| gpt-3.5-turbo-1106 | Proprietary | 0.6193 | $0.000243 | $0.000393 | $8.88 |
| claude-instant-v1 | Proprietary | 0.5984 | $0.000233 | $0.000389 | $8.50 |
| mistralai/mixtral-8x7b-chat | Open-source | 0.5471 | $0.000135 | $0.000246 | $4.91 |
| WizardLM/WizardLM-13B-V1.2 | Open-source | 0.4311 | $0.000073 | $0.000169 | $2.66 |
| meta/llama-2-70b-chat | Open-source | 0.3287 | $0.000203 | $0.000617 | $7.40 |
| mistralai/mistral-7b-chat | Open-source | 0.3061 | $0.000046 | $0.000149 | $1.67 |
| meta/code-llama-instruct-34b-chat | Open-source | 0.2022 | $0.000172 | $0.000851 | $6.28 |

**Cost insight**: GPT-4 is ~70x more expensive per query than Mistral-7B, but only ~2.5x more accurate. The cheapest model per correct answer is Mistral-7B ($0.000149), followed by WizardLM-13B ($0.000169).

---

## 5. Tasks and Benchmarks (86 tasks)

### 5.1 By Category

| Category | Tasks | Samples | Description |
|----------|-------|---------|-------------|
| Commonsense Reasoning | 3 | 12,779 | hellaswag, winogrande, arc-challenge |
| Mathematics | 5 | 8,218 | grade-school-math (GSM8K), mmlu-math variants |
| Knowledge (MMLU) | 57 | 14,042 | 57 MMLU subject domains |
| Code Generation | 1 | 427 | MBPP |
| Chinese Language | 16 | 785 | Character riddles, poetry, idioms, translations |
| Conversation | 3 | 80 | MT-Bench variants |
| Other | 9 | 1,424 | Summarization, bias detection, accounting |

### 5.2 Largest Tasks

| eval_name | Samples | Category | Evaluation |
|-----------|---------|----------|------------|
| hellaswag | 10,042 | Commonsense | Multiple choice (exact match) |
| grade-school-math | 7,450 | Mathematics | Numerical answer (partial credit) |
| mmlu-professional-law | 1,534 | Knowledge | Multiple choice |
| arc-challenge | 1,470 | Commonsense | Multiple choice |
| winogrande | 1,267 | Commonsense | Multiple choice |
| mmlu-moral-scenarios | 895 | Knowledge | Multiple choice |
| mmlu-miscellaneous | 783 | Knowledge | Multiple choice |
| mmlu-professional-psychology | 612 | Knowledge | Multiple choice |
| mmlu-high-school-psychology | 545 | Knowledge | Multiple choice |
| mbpp | 427 | Code | GPT-4 judged (binary + partial) |

---

## 6. Performance Scores

Scores are **not purely binary**. The scoring system uses partial credit:

| Score | Meaning | Count (raw) | % |
|-------|---------|-------------|---|
| 1.0 | Fully correct | 198,719 | 49.5% |
| 0.0 | Incorrect | 117,843 | 29.4% |
| 0.75 | Mostly correct | 46,301 | 11.5% |
| 0.25 | Partially correct | 24,621 | 6.1% |
| 0.5 | Half correct | 13,418 | 3.3% |
| 0.1-0.9 | Fine-grained (GPT-4-judged tasks) | 565 | 0.1% |

For multiple-choice tasks (MMLU, Hellaswag, ARC, Winogrande): scores are binary (0.0 or 1.0).
For GSM8K: scores include 0.25, 0.5, 0.75 partial credit.
For MBPP and MT-Bench: GPT-4-judged with partial credit.

---

## 7. Oracle Routing Analysis

The `oracle_model_to_route_to` column indicates the **cheapest model that answered correctly** for each prompt. This represents the theoretical best routing:

| Oracle Model | Count | % of Total | Interpretation |
|-------------|-------|------------|----------------|
| mistralai/mistral-7b-chat | 10,065 | 27.6% | Cheapest model suffices for 28% of queries |
| WizardLM/WizardLM-13B-V1.2 | 8,342 | 22.9% | Second-cheapest handles 23% |
| mistralai/mixtral-8x7b-chat | 6,140 | 16.8% | Mid-tier handles 17% |
| zero-one-ai/Yi-34B-Chat | 5,309 | 14.5% | |
| claude-instant-v1 | 1,580 | 4.3% | |
| **no_model_correct** | **1,308** | **3.6%** | **No model could answer these** |
| gpt-3.5-turbo-1106 | 1,069 | 2.9% | |
| gpt-4-1106-preview | 771 | 2.1% | GPT-4 only uniquely needed for 2.1% |
| meta/llama-2-70b-chat | 666 | 1.8% | |
| meta/code-llama-instruct-34b-chat | 477 | 1.3% | |
| claude-v1 | 425 | 1.2% | |
| claude-v2 | 345 | 0.9% | |

**Key insight**: GPT-4 is "seldom chosen" by the oracle -- only 2.1% of queries actually require it. A perfect router could use the cheapest models for 82% of queries while maintaining maximum quality.

---

## 8. Text Length Statistics

### Prompt Length
- **Mean**: 2,736 chars | **Median**: 2,594 chars | **Range**: 32 - 15,265 chars

### Response Length
- **Mean**: 152 chars | **Median**: 7 chars (most MCQ responses are single letters)

### By Task

| Task | Avg Prompt Length | Avg Response Length | Samples |
|------|------------------|--------------------|---------|
| hellaswag | 3,145 | 6 | 10,042 |
| grade-school-math (GSM8K) | 1,146 | 584 | 7,450 |
| mmlu-professional-law | 8,682 | 6 | 1,534 |
| arc-challenge | 2,061 | 6 | 1,470 |
| winogrande | 1,465 | 6 | 1,267 |
| mbpp | 97 | 910 | 427 |

### By Model (avg response length)

| Model | Avg Response |
|-------|-------------|
| WizardLM/WizardLM-13B-V1.2 | 226 chars |
| gpt-4-1106-preview | 185 chars |
| zero-one-ai/Yi-34B-Chat | 182 chars |
| mistralai/mistral-7b-chat | 161 chars |
| meta/llama-2-70b-chat | 154 chars |
| mistralai/mixtral-8x7b-chat | 148 chars |
| meta/code-llama-instruct-34b-chat | 141 chars |
| claude-v2 | 141 chars |
| claude-instant-v1 | 127 chars |
| gpt-3.5-turbo-1106 | 106 chars |
| claude-v1 | 93 chars |

---

## 9. Domain-Specific Deep Dives

### 9.1 GSM8K (grade-school-math) -- 7,450 samples

Directly maps to our `math_reasoning` domain.

| Model | Accuracy |
|-------|----------|
| claude-v2 | 0.6626 |
| gpt-4-1106-preview | 0.6588 |
| claude-v1 | 0.6508 |
| claude-instant-v1 | 0.6272 |
| gpt-3.5-turbo-1106 | 0.6048 |
| zero-one-ai/Yi-34B-Chat | 0.5481 |
| meta/llama-2-70b-chat | 0.5230 |
| mistralai/mixtral-8x7b-chat | 0.5190 |
| WizardLM/WizardLM-13B-V1.2 | 0.5063 |
| meta/code-llama-instruct-34b-chat | 0.4566 |
| mistralai/mistral-7b-chat | 0.4115 |

**Observation**: GSM8K is one of the most "democratic" tasks -- the gap between the best model (Claude-v2: 0.66) and the worst (Mistral-7B: 0.41) is relatively small, and GPT-4 doesn't dominate. This suggests student models can be competitive on math with fine-tuning.

### 9.2 MBPP (code generation) -- 427 samples

Directly maps to our `code_generation` domain.

| Model | Accuracy |
|-------|----------|
| gpt-4-1106-preview | 0.6862 |
| gpt-3.5-turbo-1106 | 0.6534 |
| claude-v2 | 0.6417 |
| claude-instant-v1 | 0.6042 |
| claude-v1 | 0.5972 |
| mistralai/mixtral-8x7b-chat | 0.5410 |
| meta/code-llama-instruct-34b-chat | 0.5176 |
| zero-one-ai/Yi-34B-Chat | 0.3864 |
| WizardLM/WizardLM-13B-V1.2 | 0.3700 |
| mistralai/mistral-7b-chat | 0.3443 |
| meta/llama-2-70b-chat | 0.3302 |

**Observation**: Larger spread than GSM8K. GPT-4 leads clearly. Code-Llama (34B, specialized for code) outperforms some larger general models (Yi-34B, LLama-70B), showing domain specialization matters.

---

## 10. Sample Data

### GSM8K Prompt Example
```
TASK: Solve the following grade school math problem and provide a numerical answer.
The following are examples of grade school math problems and answers:
Question: There are 15 trees in the grove. Grove workers will plant trees today.
After they are done, there will be 21 trees. How many trees did they plant?
Answer: 6
...
Question: Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether?
Answer:
```

### GSM8K Response Example (GPT-4)
```
In April, Natalia sold clips to 48 friends. In May, she sold half as many:
48 / 2 = 24 clips in May.
48 (April) + 24 (May) = 72 clips.
Natalia sold 72 clips altogether in April and May.
```

### MBPP Prompt Example
```
Write a function to find the shared elements from the given two lists.
```

### MBPP Response Example (GPT-4)
```python
def find_shared_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    shared_elements = set1.intersection(set2)
    return list(shared_elements)
```

### Hellaswag Prompt Example (multiple choice)
```
Then, the man writes over the snow covering the window of a car...
A) the man adds wax to the windshield...
B) a person boards a ski lift...
C) the man puts on a christmas coat...
D) the man continues removing the snow on his car.
Print only a single choice from "A" or "B" or "C" or "D" without explanation.
```

---

## 11. Paper Methodology

### AIQ Metric (Average Improvement in Quality)

The paper introduces **AIQ** -- a single scalar metric for comparing routing systems across the cost-quality tradeoff:

```
AIQ(R) = 1/(c_max - c_min) * integral[c_min to c_max] R_tilde(c) dc
```

Where `R_tilde(c)` is the non-decreasing convex hull of the router's quality at budget `c`. Higher AIQ = better cost-quality tradeoff.

### Routing Methods Evaluated

| Method | Type | Description |
|--------|------|-------------|
| **Zero Router** | Baseline | Random routing between models on the cost-quality Pareto frontier |
| **Oracle Router** | Upper bound | Always selects the best model per sample (perfect information) |
| **KNN Router** | Predictive | k-nearest neighbors on prompt embeddings → predict per-model quality |
| **MLP Router** | Predictive | 2-layer neural network on embeddings → predict per-model quality |
| **SVM Router** | Predictive | Support vector machine classifier on embeddings |
| **Cascading Router** | Non-predictive | Try cheapest model first, escalate if quality below threshold |

### Key Findings

1. **Cascading routers with low error rates (epsilon <= 0.1) significantly outperform all other approaches** -- including predictive methods.
2. Basic predictive routers (KNN, MLP) only match the best individual LLM but don't beat the Zero Router significantly.
3. The **quality of the confidence estimator** is the critical factor in cascading -- performance degrades rapidly above 0.2 error rate.
4. GPT-4 is "seldom chosen" by the Oracle -- cheaper models suffice for ~98% of queries.

---

## 12. Relevance to Our Project

### Direct Overlaps

| Our Domain | RouterBench Task | Samples | Usability |
|------------|-----------------|---------|-----------|
| `math_reasoning` | grade-school-math (GSM8K) | 7,450 | **High** -- same benchmark |
| `code_generation` | mbpp | 427 | **High** -- same benchmark |
| `text_to_sql` | *(none)* | 0 | **Not covered** |

### How to Use It

| Use Case | What to Use | Priority |
|----------|-------------|----------|
| **Adopt AIQ metric** for cost-efficiency analysis | Paper methodology | High |
| **Implement Oracle/Zero Router baselines** as bounds for our router | 0-shot data + oracle column | High |
| **Train router classifier** on GSM8K/MBPP: "does cheap model succeed?" | Score columns per model | Medium |
| **Validate cascading approach** -- paper confirms our architecture works | Paper findings | High |
| **Cost modeling template** -- project savings from routing | Cost columns | Medium |
| **Per-domain threshold tuning** -- different thresholds for math vs code | Per-eval_name analysis | Medium |

### What Doesn't Fit

| Gap | Impact | Workaround |
|-----|--------|------------|
| No text-to-SQL tasks | Primary domain missing | Generate our own data |
| Stale model set (GPT-4, Claude-v1) | Not our Qwen 14B or GPT-5 mini | Use methodology, not raw data |
| Static benchmark | No online learning signal | Generate dynamic data ourselves |
| 11-model routing | We need binary student/teacher | Collapse to "cheapest correct" vs "expensive" |

### Recommended Approach

1. **Use as methodology framework** -- adopt AIQ, Oracle/Zero baselines, cost-quality Pareto analysis
2. **Generate our own RouterBench-style data**: run Qwen 14B (base + LoRA) and GPT-5 mini on Spider/BIRD/GSM8K/MBPP, record `(prompt, student_perf, teacher_perf, cost)` tuples
3. **Use GSM8K/MBPP subsets** for router training experiments before investing in full data generation
4. **Cite the paper** for the routing formalization and cascading router validation

---

## 13. Data Quality

- **Zero nulls** in raw and 0-shot files
- **154 nulls** in 5-shot file (14 per score column -- 14 samples that only exist in 5-shot)
- No empty strings or corrupted values
- Performance scores stored as `object` type in wide files (need `pd.to_numeric()` conversion)
- Prompts and responses are plain strings (not lists, despite being stored as such in wide format)

---

## 14. Loading Code

```python
import pandas as pd

base = "data/routerbench_cache/datasets--withmartian--routerbench/snapshots/784021482c3f320c6619ed4b3bb3b41a21424fcb"

# Wide format (recommended for router training)
df = pd.read_pickle(f"{base}/routerbench_0shot.pkl")

# Get GSM8K subset
gsm8k = df[df["eval_name"] == "grade-school-math"]  # 7,450 rows

# Get MBPP subset
mbpp = df[df["eval_name"] == "mbpp"]  # 427 rows

# Convert scores to numeric
models = [c for c in df.columns if "|" not in c and c not in ["sample_id", "prompt", "eval_name", "oracle_model_to_route_to"]]
for m in models:
    df[m] = pd.to_numeric(df[m], errors="coerce")

# Binary routing labels: "can cheapest model handle it?"
df["cheap_model_correct"] = df["mistralai/mistral-7b-chat"] >= 0.75
df["expensive_needed"] = (df["gpt-4-1106-preview"] >= 0.75) & (df["mistralai/mistral-7b-chat"] < 0.5)

# Long format (for per-model analysis)
raw = pd.read_pickle(f"{base}/routerbench_raw.pkl")
```
