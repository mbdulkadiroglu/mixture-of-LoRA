"""
Phase 0 verification: 1 round, 5 queries smoke test.

Checks:
- SQLite has 5 rows in interactions table
- Training examples logged
- Adapter saved
- All confidence features populated (no NaN/None)
- GPU results JSON written
"""

import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def verify():
    load_dotenv(override=True)

    from cascade.config import CascadeConfig
    from cascade.runner import CascadeRunner

    config = CascadeConfig(
        experiment_name="phase0_verify",
        num_rounds=1,
        queries_per_round=5,
        eval_set_size=5,
        training_pool_size=100,
        training_epochs=1,
        output_dir="results/cascade/phase0",
        gpu_devices=os.getenv("CASCADE_GPU", "2"),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices

    runner = CascadeRunner(config)
    db_path = runner.run()

    # --- Verification checks ---
    logger.info("\n=== Phase 0 Verification ===")
    ok = True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check 1: interactions table
    rows = conn.execute("SELECT * FROM interactions WHERE round = 0").fetchall()
    n = len(rows)
    if n == 5:
        logger.info(f"[PASS] interactions: {n} rows")
    else:
        logger.error(f"[FAIL] interactions: expected 5, got {n}")
        ok = False

    # Check 2: confidence features populated
    for row in rows:
        for col in ["student_mean_logprob", "student_min_logprob", "student_mean_entropy", "student_num_tokens"]:
            if row[col] is None:
                logger.error(f"[FAIL] Null confidence feature: {col} at query_idx={row['query_idx']}")
                ok = False

    if all(row["student_mean_logprob"] is not None for row in rows):
        logger.info("[PASS] All confidence features populated")

    # Check 3: training examples logged (if any queries routed to teacher)
    teacher_routed = [r for r in rows if r["routed_to"] == "teacher"]
    train_rows = conn.execute("SELECT * FROM training_examples WHERE round = 0").fetchall()
    if teacher_routed:
        if len(train_rows) > 0:
            logger.info(f"[PASS] training_examples: {len(train_rows)} rows ({len(teacher_routed)} teacher-routed)")
        else:
            logger.error("[FAIL] No training examples despite teacher-routed queries")
            ok = False
    else:
        logger.info("[INFO] No queries routed to teacher — no training examples expected")

    # Check 4: adapter saved (if training happened)
    adapter_rows = conn.execute("SELECT * FROM adapter_versions WHERE round = 0").fetchall()
    if teacher_routed:
        if len(adapter_rows) > 0:
            adapter_path = adapter_rows[0]["adapter_path"]
            if adapter_path and Path(adapter_path).exists():
                logger.info(f"[PASS] Adapter saved: {adapter_path}")
            else:
                logger.warning(f"[WARN] Adapter path logged but not found: {adapter_path}")
        else:
            logger.error("[FAIL] No adapter version logged despite training")
            ok = False
    else:
        logger.info("[INFO] No training occurred — adapter check skipped")

    conn.close()

    # Check 5: GPU results JSON
    gpu_json = Path(db_path).with_name("phase0_verify_gpu_results.json")
    if gpu_json.exists():
        with open(gpu_json) as f:
            gpu_data = json.load(f)
        if len(gpu_data) > 0:
            logger.info(f"[PASS] GPU results JSON: {len(gpu_data)} entries")
        else:
            logger.error("[FAIL] GPU results JSON is empty")
            ok = False
    else:
        logger.error(f"[FAIL] GPU results JSON not found: {gpu_json}")
        ok = False

    if ok:
        logger.info("\n=== Phase 0 PASSED ===")
    else:
        logger.error("\n=== Phase 0 FAILED ===")

    return ok


if __name__ == "__main__":
    verify()
