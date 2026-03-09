#!/usr/bin/env python3
"""
Deep-dive analysis on specific error patterns in gpt-oss:120b BIRD dev results.
Focus on: value mismatches, evidence misuse, column aliasing, and format issues.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_PATH = Path("results/cascade/gpt-oss_120b_responses_bird_dev_fewshot.json")


def load_data():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data


def main():
    data = load_data()
    samples = data["per_sample"]
    wrong_samples = [s for s in samples if not s["correct"]]

    # =========================================================================
    # A. Value/literal mismatch analysis
    # =========================================================================
    print("=" * 70)
    print("A. VALUE/LITERAL MISMATCH ANALYSIS")
    print("=" * 70)

    # Extract string literals from gold and model SQL
    value_mismatch_examples = []
    for s in wrong_samples:
        gold = s.get("gold_sql", "")
        pred = s.get("model_sql", "") or ""
        ed = s.get("exec_details", {})
        pred_error = ed.get("pred_error")

        # Skip exec errors
        if pred_error:
            continue

        gold_vals = set(re.findall(r"'([^']*)'", gold))
        pred_vals = set(re.findall(r"'([^']*)'", pred))

        if gold_vals and pred_vals:
            gold_only = gold_vals - pred_vals
            pred_only = pred_vals - gold_vals
            if gold_only or pred_only:
                # Check for near-misses (case differences, padding, etc.)
                for gv in gold_only:
                    for pv in pred_only:
                        if gv.lower() == pv.lower():
                            value_mismatch_examples.append({
                                "type": "case_mismatch",
                                "idx": s["idx"],
                                "db_id": s["db_id"],
                                "gold_val": gv,
                                "pred_val": pv,
                                "question": s["question"][:100],
                            })
                        elif gv.strip('0') == pv.strip('0') or gv.lstrip('0') == pv.lstrip('0'):
                            value_mismatch_examples.append({
                                "type": "zero_padding",
                                "idx": s["idx"],
                                "db_id": s["db_id"],
                                "gold_val": gv,
                                "pred_val": pv,
                                "question": s["question"][:100],
                            })
                        elif gv.replace(' ', '') == pv.replace(' ', ''):
                            value_mismatch_examples.append({
                                "type": "whitespace_diff",
                                "idx": s["idx"],
                                "db_id": s["db_id"],
                                "gold_val": gv,
                                "pred_val": pv,
                                "question": s["question"][:100],
                            })

                if gold_only and pred_only:
                    # Complete value mismatch
                    for gv in gold_only:
                        matched = False
                        for pv in pred_only:
                            if gv.lower() == pv.lower() or gv.strip('0') == pv.strip('0'):
                                matched = True
                                break
                        if not matched:
                            # Check if evidence provides the mapping
                            evidence = s.get("evidence", "")
                            value_mismatch_examples.append({
                                "type": "value_mismatch",
                                "idx": s["idx"],
                                "db_id": s["db_id"],
                                "gold_val": gv,
                                "pred_val": str(pred_only)[:100],
                                "question": s["question"][:100],
                                "evidence": evidence[:200],
                            })

    type_counts = Counter(ex["type"] for ex in value_mismatch_examples)
    print(f"\n  Value mismatch types:")
    for t, c in type_counts.most_common():
        print(f"    {t:25s}: {c}")

    print(f"\n  Zero-padding examples:")
    for ex in [e for e in value_mismatch_examples if e["type"] == "zero_padding"][:5]:
        print(f"    idx={ex['idx']} db={ex['db_id']}: gold='{ex['gold_val']}' pred='{ex['pred_val']}'")
        print(f"      Q: {ex['question']}")

    print(f"\n  Case mismatch examples:")
    for ex in [e for e in value_mismatch_examples if e["type"] == "case_mismatch"][:10]:
        print(f"    idx={ex['idx']} db={ex['db_id']}: gold='{ex['gold_val']}' pred='{ex['pred_val']}'")

    # =========================================================================
    # B. Evidence-to-value mapping failures
    # =========================================================================
    print()
    print("=" * 70)
    print("B. EVIDENCE-TO-VALUE MAPPING FAILURES")
    print("=" * 70)

    # Common pattern: evidence says "X refers to column = 'encoded_value'"
    # but model uses the natural language term instead
    evidence_mapping_failures = []
    for s in wrong_samples:
        ev = s.get("evidence", "").strip()
        if not ev:
            continue
        pred = s.get("model_sql", "") or ""
        gold = s.get("gold_sql", "")
        ed = s.get("exec_details", {})
        if ed.get("pred_error"):
            continue

        # Look for patterns like "X refers to Y = 'value'" in evidence
        mappings = re.findall(r"(\w[\w\s]+?)\s+(?:refers to|stands for|means)\s+[\w.`]+\s*=\s*'([^']+)'", ev, re.IGNORECASE)
        for natural, encoded in mappings:
            # Check if gold uses the encoded value
            if encoded.lower() in gold.lower():
                # Check if model uses the natural language term instead of encoded
                if encoded.lower() not in pred.lower() and natural.strip().lower() in pred.lower():
                    evidence_mapping_failures.append({
                        "idx": s["idx"],
                        "db_id": s["db_id"],
                        "natural_term": natural.strip(),
                        "encoded_value": encoded,
                        "question": s["question"][:120],
                        "evidence": ev[:200],
                    })
                elif encoded.lower() not in pred.lower():
                    # Model doesn't use the encoded value at all
                    evidence_mapping_failures.append({
                        "idx": s["idx"],
                        "db_id": s["db_id"],
                        "natural_term": natural.strip(),
                        "encoded_value": encoded,
                        "question": s["question"][:120],
                        "evidence": ev[:200],
                        "note": "model uses neither natural nor encoded value",
                    })

    print(f"\n  Evidence mapping failures: {len(evidence_mapping_failures)}")
    for ex in evidence_mapping_failures[:15]:
        note = ex.get("note", "model used natural term instead of encoded value")
        print(f"    idx={ex['idx']} db={ex['db_id']}")
        print(f"      Natural: '{ex['natural_term']}' -> Encoded: '{ex['encoded_value']}'")
        print(f"      Note: {note}")
        print(f"      Q: {ex['question']}")
        print()

    # =========================================================================
    # C. Column selection analysis (model returns extra/wrong columns)
    # =========================================================================
    print()
    print("=" * 70)
    print("C. OUTPUT COLUMN DIFFERENCES")
    print("=" * 70)

    # Look at samples where pred_rows == ref_rows (same row count) but still wrong
    # This usually means column selection or ordering differences
    same_row_count_wrong = []
    for s in wrong_samples:
        ed = s.get("exec_details", {})
        if ed.get("pred_error"):
            continue
        pred_rows = ed.get("pred_rows", 0)
        ref_rows = ed.get("ref_rows", 0)
        if pred_rows == ref_rows and pred_rows > 0:
            same_row_count_wrong.append(s)

    print(f"\n  Wrong samples with matching row count: {len(same_row_count_wrong)} / {len(wrong_samples)}")

    # Analyze column differences
    col_diff_patterns = Counter()
    for s in same_row_count_wrong:
        gold = s.get("gold_sql", "").upper()
        pred = (s.get("model_sql", "") or "").upper()

        # Check if model adds extra columns
        gold_select = re.search(r'SELECT\s+(.*?)\s+FROM', gold, re.DOTALL)
        pred_select = re.search(r'SELECT\s+(.*?)\s+FROM', pred, re.DOTALL)

        if gold_select and pred_select:
            gold_cols = len(gold_select.group(1).split(','))
            pred_cols = len(pred_select.group(1).split(','))
            if pred_cols > gold_cols:
                col_diff_patterns["extra_columns"] += 1
            elif pred_cols < gold_cols:
                col_diff_patterns["missing_columns"] += 1
            else:
                # Same column count, possibly different expressions/aliases
                if 'AS' in pred and 'AS' not in gold:
                    col_diff_patterns["added_aliases"] += 1
                elif '||' in pred and '||' not in gold:
                    col_diff_patterns["concatenation_diff"] += 1
                elif 'CAST' in pred and 'CAST' not in gold:
                    col_diff_patterns["cast_diff"] += 1
                else:
                    col_diff_patterns["other_column_diff"] += 1

    print(f"\n  Column difference patterns (same row count):")
    for p, c in col_diff_patterns.most_common():
        print(f"    {p:30s}: {c}")

    # =========================================================================
    # D. Empty result deep dive
    # =========================================================================
    print()
    print("=" * 70)
    print("D. EMPTY RESULT DEEP DIVE")
    print("=" * 70)

    empty_result_samples = [s for s in wrong_samples
                            if s.get("exec_details", {}).get("pred_rows", 0) == 0
                            and s.get("exec_details", {}).get("ref_rows", 0) > 0
                            and not s.get("exec_details", {}).get("pred_error")]

    print(f"\n  Total empty result errors: {len(empty_result_samples)}")

    # Categorize empty result causes
    empty_causes = Counter()
    for s in empty_result_samples:
        gold = s.get("gold_sql", "")
        pred = s.get("model_sql", "") or ""

        gold_vals = set(re.findall(r"'([^']*)'", gold))
        pred_vals = set(re.findall(r"'([^']*)'", pred))

        if gold_vals != pred_vals:
            empty_causes["different_filter_values"] += 1
        elif len(re.findall(r'\bJOIN\b', pred.upper())) > len(re.findall(r'\bJOIN\b', gold.upper())):
            empty_causes["over_joining"] += 1
        elif 'WHERE' in pred.upper() and pred.upper().count('AND') > gold.upper().count('AND'):
            empty_causes["extra_where_conditions"] += 1
        else:
            empty_causes["other"] += 1

    print(f"\n  Empty result causes:")
    for cause, count in empty_causes.most_common():
        print(f"    {cause:30s}: {count}")

    # =========================================================================
    # E. Database-specific column naming issues
    # =========================================================================
    print()
    print("=" * 70)
    print("E. DATABASE-SPECIFIC ISSUES")
    print("=" * 70)

    # For the worst-performing databases, look at common patterns
    for db in ["financial", "california_schools", "toxicology"]:
        db_wrong = [s for s in wrong_samples if s.get("db_id") == db]
        db_all = [s for s in samples if s.get("db_id") == db]
        print(f"\n  --- {db} ({len(db_wrong)}/{len(db_all)} wrong) ---")

        # Common error patterns
        db_errors = Counter()
        for s in db_wrong:
            ed = s.get("exec_details", {})
            pred = s.get("model_sql", "") or ""
            gold = s.get("gold_sql", "")

            if ed.get("pred_error"):
                db_errors["exec_error"] += 1
            elif ed.get("pred_rows", 0) == 0 and ed.get("ref_rows", 0) > 0:
                db_errors["empty_result"] += 1
            else:
                db_errors["wrong_result"] += 1

        for e, c in db_errors.most_common():
            print(f"    {e}: {c}")

        # Show some examples
        for s in db_wrong[:3]:
            ed = s.get("exec_details", {})
            print(f"    Example idx={s['idx']} diff={s.get('difficulty')}")
            print(f"      Q: {s['question'][:100]}")
            if s.get("evidence"):
                print(f"      Ev: {s['evidence'][:100]}")
            print(f"      Gold: {s['gold_sql'][:120]}")
            print(f"      Pred: {(s.get('model_sql', '') or '')[:120]}")
            print(f"      pred_rows={ed.get('pred_rows')}, ref_rows={ed.get('ref_rows')}")
            print()

    # =========================================================================
    # F. Model verbosity / overthinking analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("F. MODEL VERBOSITY / OVERTHINKING ANALYSIS")
    print("=" * 70)

    # Compare SQL length ratio between model and gold
    length_ratios = {"correct": [], "wrong": []}
    for s in samples:
        gold_len = len(s.get("gold_sql", ""))
        pred_len = len(s.get("model_sql", "") or "")
        if gold_len > 0:
            ratio = pred_len / gold_len
            if s["correct"]:
                length_ratios["correct"].append(ratio)
            else:
                length_ratios["wrong"].append(ratio)

    avg_ratio_correct = sum(length_ratios["correct"]) / max(len(length_ratios["correct"]), 1)
    avg_ratio_wrong = sum(length_ratios["wrong"]) / max(len(length_ratios["wrong"]), 1)

    print(f"\n  Average SQL length ratio (model/gold):")
    print(f"    Correct: {avg_ratio_correct:.2f}x")
    print(f"    Wrong:   {avg_ratio_wrong:.2f}x")

    # Over-complicated queries
    over_complicated = 0
    under_complicated = 0
    for s in wrong_samples:
        gold = s.get("gold_sql", "")
        pred = s.get("model_sql", "") or ""
        if len(pred) > 2 * len(gold) and len(gold) > 50:
            over_complicated += 1
        elif len(pred) < 0.5 * len(gold) and len(gold) > 50:
            under_complicated += 1

    print(f"\n  Over-complicated (model > 2x gold length): {over_complicated}")
    print(f"  Under-complicated (model < 0.5x gold length): {under_complicated}")

    # =========================================================================
    # G. Financial database deep dive (worst performer at 14.2%)
    # =========================================================================
    print()
    print("=" * 70)
    print("G. FINANCIAL DATABASE DEEP DIVE (14.2% accuracy)")
    print("=" * 70)

    financial_wrong = [s for s in wrong_samples if s.get("db_id") == "financial"]
    financial_correct = [s for s in samples if s.get("db_id") == "financial" and s["correct"]]

    print(f"\n  Correct: {len(financial_correct)}, Wrong: {len(financial_wrong)}")

    # Difficulty distribution
    for d in ["simple", "moderate", "challenging"]:
        fd = [s for s in samples if s.get("db_id") == "financial" and s.get("difficulty") == d]
        fc = [s for s in fd if s["correct"]]
        if fd:
            print(f"  {d}: {len(fc)}/{len(fd)} = {len(fc)/len(fd)*100:.1f}%")

    # Check if financial has particularly long/complex gold queries
    financial_all = [s for s in samples if s.get("db_id") == "financial"]
    avg_gold_len = sum(len(s["gold_sql"]) for s in financial_all) / len(financial_all)
    avg_pred_len = sum(len(s.get("model_sql", "") or "") for s in financial_all) / len(financial_all)
    print(f"\n  Avg gold SQL length: {avg_gold_len:.0f} chars")
    print(f"  Avg model SQL length: {avg_pred_len:.0f} chars")

    # Check for common issues in financial
    financial_patterns = Counter()
    for s in financial_wrong:
        gold = s.get("gold_sql", "").upper()
        pred = (s.get("model_sql", "") or "").upper()
        ed = s.get("exec_details", {})

        if ed.get("pred_error"):
            financial_patterns["exec_error"] += 1
        elif ed.get("pred_rows", 0) == 0 and ed.get("ref_rows", 0) > 0:
            financial_patterns["empty_result"] += 1
        elif 'CASE' in gold and 'CASE' not in pred:
            financial_patterns["missing_CASE_WHEN"] += 1
        elif 'GROUP BY' in gold and 'GROUP BY' not in pred:
            financial_patterns["missing_GROUP_BY"] += 1
        elif len(re.findall(r'\bJOIN\b', gold)) > len(re.findall(r'\bJOIN\b', pred)):
            financial_patterns["missing_joins"] += 1
        elif len(pred) > 2 * len(gold):
            financial_patterns["over_complicated"] += 1
        else:
            financial_patterns["other_logic_error"] += 1

    print(f"\n  Error patterns in financial:")
    for p, c in financial_patterns.most_common():
        print(f"    {p}: {c}")

    # Show 5 interesting financial examples
    print(f"\n  Financial examples:")
    for i, s in enumerate(financial_wrong[:8]):
        ed = s.get("exec_details", {})
        print(f"\n    [{i+1}] idx={s['idx']} diff={s.get('difficulty')}")
        print(f"      Q: {s['question'][:120]}")
        print(f"      Ev: {s.get('evidence', '')[:150]}")
        print(f"      Gold: {s['gold_sql'][:200]}")
        print(f"      Pred: {(s.get('model_sql', '') or '')[:200]}")
        print(f"      pred_rows={ed.get('pred_rows')}, ref_rows={ed.get('ref_rows')}")

    # =========================================================================
    # H. Correct vs wrong: what makes correct queries succeed?
    # =========================================================================
    print()
    print("=" * 70)
    print("H. SUCCESS PATTERNS - What makes correct queries succeed?")
    print("=" * 70)

    correct_samples = [s for s in samples if s["correct"]]

    # Compare structural features
    for label, subset in [("Correct", correct_samples), ("Wrong", wrong_samples)]:
        avg_joins = sum(len(re.findall(r'\bJOIN\b', s["gold_sql"].upper())) for s in subset) / len(subset)
        avg_subq = sum(1 for s in subset if re.search(r'\(\s*SELECT\b', s["gold_sql"].upper())) / len(subset) * 100
        avg_groupby = sum(1 for s in subset if 'GROUP BY' in s["gold_sql"].upper()) / len(subset) * 100
        avg_case = sum(1 for s in subset if 'CASE' in s["gold_sql"].upper()) / len(subset) * 100
        avg_gold_len = sum(len(s["gold_sql"]) for s in subset) / len(subset)

        print(f"\n  {label} samples (n={len(subset)}):")
        print(f"    Avg JOINs in gold:     {avg_joins:.2f}")
        print(f"    % with subqueries:     {avg_subq:.1f}%")
        print(f"    % with GROUP BY:       {avg_groupby:.1f}%")
        print(f"    % with CASE:           {avg_case:.1f}%")
        print(f"    Avg gold SQL length:   {avg_gold_len:.0f} chars")


if __name__ == "__main__":
    main()
