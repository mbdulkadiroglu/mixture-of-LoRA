# Phase 2: Regime Characterization

Planned experiments to characterize when cascade distillation fails.

## Experiments

### 2.1 — Teacher Corruption Sweep
Inject controlled errors into teacher SQL at varying rates (5%, 10%, 20%, 50%).
Measure how quickly student accuracy degrades and whether the router can detect
the shift via confidence metrics.

### 2.2 — Corruption Strategy Comparison
Compare the four corruption strategies (swap_columns, change_join, alter_where,
modify_aggregate) at a fixed corruption rate. Some errors may be easier for the
student to detect/reject than others.

### 2.3 — Threshold Sensitivity
Run the baseline (1.1) with thresholds at p10, p25, p50, p75 of the calibrated
confidence distribution. Identify the threshold range where the system converges
vs diverges.

## Implementation Notes

- Requires CascadeTeacher corruption methods (already stubbed)
- Use same CascadeRunner with different CascadeConfig settings
- Compare against Phase 1 baselines
