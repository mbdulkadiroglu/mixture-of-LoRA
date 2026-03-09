# Implementation Guide: From Pieces to Working System

## Important: What This Project Is

**The goal is teacher→student knowledge transfer.** The small model learns from the large model's responses. Text-to-SQL is a test domain, not the end goal. Ground truth verification (SQL execution) is a development convenience for measuring progress, not a core system component. The real system must work without ground truth — in production, the teacher's response IS the training signal.

## Where You Are

You have the **building blocks** but not the **wiring**:

- Student model (Qwen 14B + LoRA) generates responses -- **works, validated on Spider and BIRD**
- Teacher model (GPT-5 mini) generates responses -- **works**
- Router decides student vs teacher -- **works (4 strategies), but no real feedback**
- Trainer can fine-tune LoRA adapters -- **works for batch training**
- Replay buffer exists -- **works**
- Online training manager exists in code -- **never instantiated, never called**
- `process_query()` had hardcoded `success=True` for router -- **fixed (removed)**

**The core loop (query → student tries → cascade to teacher → teacher response trains student → router adapts) has never run end-to-end.**

---

## Part I: Research Questions You Must Answer First

These are not theoretical -- you can't make correct implementation decisions without resolving them. Each one has a concrete experiment you can run.

### RQ1: How much noise can the student tolerate in teacher training data?

**Why it matters:** The teacher (GPT-5 mini) is not perfect. If 30% of teacher responses are wrong, does training on them still help the student, or does noise cancel out the signal?

**Key insight:** In the real system, the teacher's response IS the training signal — there is no ground truth oracle. SQL execution verification is available for text-to-SQL and is useful for measuring teacher accuracy during development, but the framework must not depend on it. Other domains (math, code, general NLP) have no cheap verification oracle.

**Options for handling teacher noise:**
1. **Trust the teacher**: Train on all teacher responses. If the teacher is substantially better than the student, the signal-to-noise ratio may be good enough. This is the simplest and most domain-agnostic approach.
2. **Teacher self-verification**: Ask the teacher to check its own answer (expensive, 2x API calls, but domain-agnostic).
3. **Confidence filtering**: Only train on teacher responses where the teacher's own confidence is high. Requires calibrated confidence scores.
4. **Domain-specific verification** (text-to-sql only): Execute SQL to verify. Useful for development experiments, but not generalizable.

**Experiment to run:** Measure teacher accuracy on the test domain. Then train the student on unfiltered teacher responses and see if accuracy improves despite noise. This tells you whether verification is even necessary.

**Data so far:** GPT-5 mini achieves 73.5% on Spider (200 samples). The student (LoRA v4) achieves 64.9% on Spider. Since the teacher substantially outperforms the student, training on teacher responses should still provide a net positive signal even with ~26% noise.

### RQ2: How to do online training without catastrophic forgetting?

**Why it matters:** When you train the student on 50 new teacher examples, it might forget what it learned from the original 7000 Spider examples.

**What you already have:** Replay buffer with 20% mix ratio. This is the standard mitigation.

**Open questions:**
- Is 20% replay enough? Or does the student still degrade after multiple online rounds?
- Should you replay from the **original training data** (ground truth) or only from the replay buffer (which contains teacher data)?
- How many examples per online training round? Too few = noisy gradients, too many = expensive.
- How many epochs per online round? Currently hardcoded to 1 in `train_online()`.

**Experiment to run:**
1. Take your best Spider LoRA (v4, 73.98%)
2. Fine-tune it on 50 teacher-generated examples (1 epoch, replay_ratio=0.2)
3. Re-evaluate on full Spider test set
4. Repeat for 100, 200, 500 examples
5. Plot accuracy vs number of online training examples

This gives you the **forgetting curve** -- the most important empirical result for your system.

### RQ3: How should the router get real feedback?

**Why it matters:** The hardcoded `success=True` has been removed (Step 0, done), but the router still has no feedback signal. It can't learn to route better without knowing whether the student succeeded or failed.

**The challenge:** In a real domain-agnostic system, there is no ground truth at query time. The router needs a feedback mechanism that works without labeled data.

**Options (domain-agnostic first):**
1. **Teacher-as-judge feedback**: Use `teacher.evaluate_confidence()` (already implemented) to score the student's response. If the teacher was needed, the student "failed" by definition. This is the most general approach.
   - **Trade-off**: Costs an API call per query. The teacher might be wrong.
2. **Cascade-based implicit feedback**: If the router sent the query to the teacher (low confidence), that counts as a student failure. If the router sent it to the student (high confidence), it's an assumed success. Simple, free, no oracle needed.
3. **Periodic evaluation**: After each training round, run a benchmark eval and update router stats in bulk. Decouples feedback from individual queries.
4. **Domain-specific feedback** (text-to-sql only): Execute SQL to check if it runs without error. Useful for development but not generalizable.

**Recommendation:** Start with option 2 (cascade-based implicit feedback) — it's free and domain-agnostic. Supplement with option 3 (periodic eval) after training rounds.

### RQ4: When should retraining be triggered?

**Why it matters:** `should_retrain()` exists but is never called. Even if you wire it up, the current logic (retrain if success_rate drops below threshold - 0.1) is arbitrary.

**Options:**
1. **Batch threshold**: Retrain after collecting N teacher examples (e.g., 50). Simple, predictable.
2. **Performance drop**: Retrain when success_rate drops below X. Requires real feedback (see RQ3).
3. **Time-based**: Retrain every T queries or every H hours. Simple but wasteful.
4. **Cost-based**: Retrain when the cost of teacher calls exceeds the cost of one training round. Economically optimal.

**For now:** Start with batch threshold (option 1). It's the simplest and most debuggable. You can get fancy later.

### RQ5: How should the router adapt as the student improves?

**Why it matters:** After retraining, the student is better. The router should send more queries to the student and fewer to the teacher. But the current threshold is static at 0.7.

**Options:**
1. **Stats-based adaptation**: After each training round, run a quick eval (50 samples). Update the router's success_rate. The stats-based routing method already uses success_rate as confidence -- this would work automatically.
2. **Threshold decay**: After each successful training round, reduce the confidence threshold by 0.05. Simple linear schedule.
3. **Bandit formulation**: Thompson sampling -- maintain a Beta distribution over success probability per domain. Update with each observed outcome. Sample from it to decide routing.
4. **Perplexity recalibration**: After training, recalibrate the perplexity-to-confidence mapping on a validation set.

**Recommendation:** Start with option 1 (stats-based + periodic eval). It's simple and uses code you already have.

---

## Part II: Step-by-Step Implementation Roadmap

Each step is self-contained. **Do not start the next step until the current one is verified.**

### Step 0: Fix the Foundation (DONE)
**Goal:** Clean up known issues so the base is solid.

**Tasks:**
- [x] Fix model name in `config.yaml`, `src/config.py`, `.env` — all now say `unsloth/Qwen2.5-14B-Instruct`
- [x] Remove hardcoded `success=True` from `framework.py` — router stats no longer corrupted
- [x] Fix column-order-independent SQL result comparison in `sql_executor.py`
- [x] Verify baselines: Spider LoRA v4 = 64.89%, Teacher (GPT-5 mini) = 73.50% on Spider (200 samples)

---

### Step 1: Measure the Teacher (DONE for Spider)
**Goal:** Know the teacher-student gap. The bigger the gap, the more the student can learn from distillation.

**Results:**
- [x] GPT-5 mini on Spider (200 samples): 73.50% execution accuracy
- [x] Student LoRA v4 on Spider (1034 samples): 64.89% execution accuracy
- [x] Teacher-student gap: ~8.6 percentage points — meaningful room for distillation

**Interpretation:** The teacher substantially outperforms the student. Even if ~26% of teacher responses are wrong, training on them should still provide a net positive signal because the teacher's error distribution is different from (and better than) the student's. This is the standard assumption in knowledge distillation — the teacher doesn't need to be perfect, just better than the student.

**Remaining:** Measure teacher on BIRD and on non-SQL domains when those are set up.

---

### Step 2: First Distillation Experiment (3-4 hours)
**Goal:** Prove that training the student on teacher responses improves the student. This is the core thesis test.

**Tasks:**
- [ ] Collect teacher responses for N samples where the student failed (cascade)
- [ ] Train the student on those teacher responses (1 epoch, with replay buffer)
- [ ] Re-evaluate student — did accuracy improve?
- [ ] Optionally: compare training on ALL teacher responses vs only responses for queries the student got wrong

**Verification:** After one round of distillation:
- Student accuracy should increase (even +0.5% is a positive signal)
- Cross-domain performance should not significantly degrade (replay buffer working)

**This is the biggest gate in the project.** If teacher distillation doesn't improve the student, the thesis needs revision.

---

### Step 3: Wire Router Feedback (2-3 hours)
**Goal:** The router gets truthful success/failure signals so it can learn to route better.

**Tasks:**
- [ ] Implement cascade-based implicit feedback: if the query went to the teacher (low confidence), the student "failed". If the student handled it (high confidence), assume success.
- [ ] After each training round, run a periodic evaluation and update router stats in bulk
- [ ] The router should route more to the student as the student improves

**Verification:** Process queries through the cascade. Check that router stats reflect realistic success rates (not 100%). After a training round, the router should increase student confidence.

---

### Step 4: Wire the Automatic Cascade Loop (4-6 hours)
**Goal:** The cascade runs automatically: student tries → if uncertain, teacher responds → teacher response trains student → router adapts.

**Tasks:**
- [ ] Instantiate `OnlineTrainingManager` in `framework.initialize()`
- [ ] In `process_query()`, when teacher responds, collect the response as training data
- [ ] Configure batch threshold (e.g., retrain after 50 teacher examples collected)
- [ ] After each online training round:
  1. Save the new adapter version
  2. Run eval on test domain to measure improvement
  3. Update router stats with new eval results
  4. Log everything to a results JSON file
- [ ] Safety check: if post-training eval is worse than pre-training, rollback

**Verification:** Run the framework on 500 queries. It should cascade, collect, retrain, and adapt routing — all automatically.

---

### Step 5: Measure the Forgetting Curve (3-4 hours)
**Goal:** Understand how online teacher distillation affects existing knowledge. Answers RQ2.

**Tasks:**
- [ ] After Step 2's distillation, evaluate on the full test set (regression test)
- [ ] Repeat distillation for multiple rounds (2, 3, 4)
- [ ] After each round, evaluate on all domains
- [ ] Plot: `accuracy vs training_round`
- [ ] Vary replay_ratio (try 0.1, 0.2, 0.3) and compare curves

**Verification:** You get empirical answers to:
- Does accuracy plateau or keep improving with more teacher data?
- Does cross-domain performance degrade? By how much?
- What replay_ratio best balances new learning vs retention?

---

### Step 6: Expand to Non-SQL Domains (4-6 hours)
**Goal:** Demonstrate the framework is domain-agnostic by testing on domains without execution-based oracles.

**Tasks:**
- [ ] Set up math reasoning domain (MATH/GSM8K) with appropriate evaluation
- [ ] Set up code generation domain (HumanEval/MBPP) with evaluation
- [ ] Run the cascade on these domains — teacher distillation without SQL execution verification
- [ ] Compare: does teacher distillation help in domains where you can't verify teacher responses?

**Verification:** Student accuracy improves on non-SQL domains through teacher distillation. This proves the system is not dependent on SQL execution as a crutch.

---

### Step 7: Add Cost Tracking (2-3 hours)
**Goal:** Know exactly what the system costs to run. Needed for the paper.

**Tasks:**
- [ ] Track per-query costs:
  - Student inference: tokens generated, GPU time (estimate from batch size)
  - Teacher inference: tokens in/out (available from API response), API cost
  - Training: GPU time per round, number of examples used
- [ ] Add cost fields to `QueryResult`:
  ```python
  cost_student_tokens: int = 0
  cost_teacher_tokens: int = 0
  cost_teacher_dollars: float = 0.0
  cost_training_seconds: float = 0.0
  ```
- [ ] After each experiment, produce a cost summary:
  - Total cost if all queries went to teacher
  - Total cost with cascade (student + selective teacher)
  - Break-even point: after N training rounds, savings from not using teacher

**Verification:** Run 200 queries through the cascade. Compare total cost vs teacher-only baseline.

---

### Step 8: Adaptive Router Threshold (3-4 hours)
**Goal:** Router automatically sends more queries to the student as it improves. Answers RQ5.

**Tasks:**
- [ ] After each training round's eval (from Step 6), update the router:
  ```python
  def adapt_threshold(self, domain: str, new_eval_score: float):
      """Adjust confidence threshold based on student's improving accuracy."""
      # Example: threshold = 1 - eval_score (if student is 80% accurate, threshold = 0.2)
      # Or: use the eval score directly as the stats-based confidence
      self._domain_stats[domain]["success_rate"] = new_eval_score
  ```
- [ ] Log threshold changes over time
- [ ] Plot: `threshold vs training_round` and `queries_to_student_pct vs training_round`

**Verification:** Over multiple training rounds, the router should:
- Decrease the effective threshold (or increase student confidence)
- Route more queries to the student
- Teacher usage should decrease over time

---

### Step 9: Multi-Round End-to-End Evaluation (4-6 hours)
**Goal:** Run the full system for multiple rounds and produce the paper's key results.

**Tasks:**
- [ ] Design the experiment:
  - Start with base student (no LoRA) or best ground-truth LoRA
  - Process 1000 BIRD queries through the cascade
  - Let the system auto-train, adapt routing, collect data
  - Record everything: per-query decisions, costs, accuracy at each checkpoint
- [ ] Run 3 experimental conditions:
  1. **Student-only** (no cascade, no training): baseline
  2. **Teacher-only**: upper bound on quality, cost ceiling
  3. **Cascade with online training**: your system
- [ ] Produce key plots:
  - Accuracy over time (query number)
  - Cost over time (cumulative)
  - Routing distribution over time (% student vs % teacher)
  - Quality-cost Pareto frontier

**Verification:** Your system should:
- Start near student-only accuracy
- Improve toward teacher accuracy over time
- Cost significantly less than teacher-only
- Show a clear learning curve

---

### Step 10: Multi-Domain Routing Experiment (Optional, 6-8 hours)
**Goal:** Show that domain-specific adapters + routing outperforms a single general adapter.

**Tasks:**
- [ ] Run mixed queries from all domains through the cascade
- [ ] Show that the router correctly identifies domains and selects appropriate adapters
- [ ] Compare: domain-specific LoRA adapters vs single general adapter vs base model

---

## Part III: Research Question Summary

| # | Question | When to Answer | How to Answer | Blocking? |
|---|----------|---------------|---------------|-----------|
| RQ1 | How much teacher noise can the student tolerate? | Step 2 | Train on unfiltered teacher responses, measure improvement | Yes -- determines if distillation works at all |
| RQ2 | How to prevent catastrophic forgetting? | Step 5 | Plot forgetting curves with different replay ratios | Yes -- determines if online training is viable |
| RQ3 | How should the router get feedback without ground truth? | Step 3 | Cascade-based implicit feedback + periodic eval | Yes -- router can't adapt without this |
| RQ4 | When to trigger retraining? | Step 4 | Start with batch threshold, compare to performance-based | No -- batch threshold works for now |
| RQ5 | How should the router adapt? | Step 8 | Compare fixed vs adaptive threshold strategies | No -- can start with fixed |
| RQ6 | Does the framework generalize beyond text-to-SQL? | Step 6 | Test on math reasoning and code generation | Yes -- core claim is domain-agnosticism |
| RQ7 | What's the cost-efficiency story? | Step 7, 9 | Track all costs, compute break-even point | No -- needed for paper, not for prototype |

---

## Decision Tree: If Things Go Wrong

```
Teacher-student gap too small to distill?
  → Use a stronger teacher model (GPT-5 instead of GPT-5 mini)
  → Try harder evaluation benchmarks where the gap is larger
  → The framework still works — there's just less to gain

Distillation round makes student WORSE?
  → Is the teacher substantially worse than expected? Measure teacher accuracy.
  → Increase replay_ratio (try 0.3, 0.5)
  → Reduce learning rate for online training (try 1e-4 instead of 2e-4)
  → Use fewer online training epochs (1 is standard)
  → Try confidence filtering: only train on teacher responses where teacher confidence is high

Student forgets existing knowledge after distillation?
  → Increase replay_ratio
  → Use separate adapters per domain (you already support this)
  → Consider adapter merging

Router never routes to student?
  → Check confidence threshold (0.7 may be too high for a young adapter)
  → Start with threshold 0.5 and increase as student improves
  → Ensure router stats are getting real feedback (Step 3)

Framework only works on text-to-SQL?
  → This means you're accidentally depending on SQL execution verification
  → Remove that dependency — the framework must work with teacher responses as the only training signal
  → Test on math/code domains to prove generality
```

---

## Estimated Timeline

| Step | Effort | Cumulative | What You'll Know |
|------|--------|------------|-----------------|
| 0: Fix foundation | DONE | -- | Baselines are solid |
| 1: Measure teacher | DONE | -- | Teacher-student gap quantified (8.6% on Spider) |
| 2: First distillation | 3-4h | 4h | **Core thesis validated or invalidated** |
| 3: Router feedback | 2-3h | 7h | Router adapts without ground truth |
| 4: Automatic cascade loop | 4-6h | 13h | System runs hands-free |
| 5: Forgetting curve | 3-4h | 17h | Online training feasibility (RQ2) |
| 6: Non-SQL domains | 4-6h | 23h | Framework is truly domain-agnostic |
| 7: Cost tracking | 2-3h | 26h | Economic argument |
| 8: Adaptive router | 3-4h | 30h | Self-improving routing |
| 9: Full experiment | 4-6h | 36h | Paper-ready results |

**The critical milestone is Step 2.** If teacher distillation doesn't improve the student, the thesis needs revision. Run Step 2 as early as possible.
