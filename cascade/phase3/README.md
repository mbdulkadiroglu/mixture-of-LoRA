# Phase 3: Mitigations

Planned experiments to test strategies that prevent training signal degradation.

## Mitigations

### 3.1 — Teacher Consistency Filtering
Before accepting a teacher response for training, query the teacher N times
(consistency_samples=3). Only use responses where all N agree. This filters
ambiguous or unreliable teacher outputs.

### 3.2 — Confidence-Weighted Training
Weight training examples by the router's confidence in the student's inability.
Low-confidence escalations get high training weight (the model genuinely needs help).
High-confidence escalations get low weight (might be noise).

### 3.3 — Adaptive Threshold
Adjust the router threshold each round based on the student's eval accuracy.
As the student improves, raise the bar. If accuracy drops, lower it.

### 3.4 — Periodic Auditing
Reserve a fraction of queries (audit_rate) for gold-label evaluation of
teacher responses. Discard teacher outputs that are incorrect. This requires
gold labels but simulates a realistic quality assurance pipeline.

## Implementation Notes

- Requires CascadeConfig fields already defined (filter_teacher_consistency,
  confidence_weighting, adaptive_threshold, audit_rate)
- Modifications to CascadeRunner._run_round() and CascadeTrainer
- Compare against Phase 1 and Phase 2 baselines
