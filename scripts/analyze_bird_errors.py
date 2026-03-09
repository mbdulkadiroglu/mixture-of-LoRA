#!/usr/bin/env python3
"""
Comprehensive error analysis of gpt-oss:120b few-shot results on BIRD dev.
Analyzes 744 incorrect predictions across multiple dimensions.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_PATH = Path("results/cascade/gpt-oss_120b_responses_bird_dev_fewshot.json")
OUTPUT_PATH = Path("results/cascade/bird_dev_fewshot_error_analysis.json")


def load_data():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data


def classify_error(sample):
    """Classify a wrong sample into error categories."""
    ed = sample.get("exec_details", {})
    pred_error = ed.get("pred_error")
    pred_rows = ed.get("pred_rows", 0)
    ref_rows = ed.get("ref_rows", 0)

    if pred_error is not None:
        return "exec_error"
    elif pred_rows == 0 and ref_rows > 0:
        return "empty_result"
    elif pred_rows > 0 and ref_rows == 0:
        return "unexpected_results"  # model returns rows when gold returns none
    else:
        return "wrong_result"


def classify_exec_error(error_msg):
    """Sub-classify execution errors."""
    if error_msg is None:
        return "none"
    msg = error_msg.lower()
    if "no such column" in msg:
        return "wrong_column"
    elif "no such table" in msg:
        return "wrong_table"
    elif "ambiguous column" in msg:
        return "ambiguous_column"
    elif "syntax error" in msg or "near" in msg:
        return "syntax_error"
    elif "no such function" in msg:
        return "wrong_function"
    elif "misuse of aggregate" in msg:
        return "aggregate_misuse"
    elif "mismatch" in msg:
        return "type_mismatch"
    elif "operand" in msg or "operator" in msg:
        return "operator_error"
    else:
        return f"other: {error_msg[:80]}"


def count_sql_features(sql):
    """Count SQL complexity features in a query."""
    if not sql:
        return {}
    sql_upper = sql.upper()
    # Count JOINs
    joins = len(re.findall(r'\bJOIN\b', sql_upper))
    has_subquery = 1 if re.search(r'\(\s*SELECT\b', sql_upper) else 0
    has_group_by = 1 if 'GROUP BY' in sql_upper else 0
    has_having = 1 if 'HAVING' in sql_upper else 0
    has_order_by = 1 if 'ORDER BY' in sql_upper else 0
    has_limit = 1 if 'LIMIT' in sql_upper else 0
    has_distinct = 1 if 'DISTINCT' in sql_upper else 0
    has_union = 1 if 'UNION' in sql_upper else 0
    has_case = 1 if 'CASE' in sql_upper else 0
    has_like = 1 if 'LIKE' in sql_upper else 0
    has_in = 1 if re.search(r'\bIN\s*\(', sql_upper) else 0
    has_between = 1 if 'BETWEEN' in sql_upper else 0
    has_except = 1 if 'EXCEPT' in sql_upper else 0
    has_intersect = 1 if 'INTERSECT' in sql_upper else 0
    # Aggregate functions
    agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_upper))

    return {
        "joins": joins,
        "has_subquery": has_subquery,
        "has_group_by": has_group_by,
        "has_having": has_having,
        "has_order_by": has_order_by,
        "has_limit": has_limit,
        "has_distinct": has_distinct,
        "has_union": has_union,
        "has_case": has_case,
        "has_like": has_like,
        "has_in": has_in,
        "has_between": has_between,
        "has_except": has_except,
        "has_intersect": has_intersect,
        "agg_count": agg_count,
    }


def analyze_wrong_examples(wrong_samples):
    """Detailed analysis of wrong result patterns for a subset."""
    patterns = Counter()
    examples = []

    for s in wrong_samples[:50]:  # analyze up to 50
        ed = s.get("exec_details", {})
        gold = s.get("gold_sql", "")
        pred = s.get("model_sql", "")
        question = s.get("question", "")
        evidence = s.get("evidence", "")

        gold_upper = gold.upper()
        pred_upper = pred.upper() if pred else ""

        issues = []

        # Check if model uses different tables
        gold_tables = set(re.findall(r'\bFROM\s+(\w+)', gold_upper)) | set(re.findall(r'\bJOIN\s+(\w+)', gold_upper))
        pred_tables = set(re.findall(r'\bFROM\s+(\w+)', pred_upper)) | set(re.findall(r'\bJOIN\s+(\w+)', pred_upper))
        # Remove aliases
        gold_tables = {t for t in gold_tables if t not in ('AS', 'SELECT', 'WHERE', 'ON', 'AND', 'OR', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS')}
        pred_tables = {t for t in pred_tables if t not in ('AS', 'SELECT', 'WHERE', 'ON', 'AND', 'OR', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS')}

        if gold_tables != pred_tables:
            missing = gold_tables - pred_tables
            extra = pred_tables - gold_tables
            if missing:
                issues.append(f"missing_table:{','.join(missing)}")
                patterns["missing_table"] += 1
            if extra:
                issues.append(f"extra_table:{','.join(extra)}")
                patterns["extra_table"] += 1

        # Check JOIN count difference
        gold_joins = len(re.findall(r'\bJOIN\b', gold_upper))
        pred_joins = len(re.findall(r'\bJOIN\b', pred_upper))
        if gold_joins != pred_joins:
            issues.append(f"join_count_diff:gold={gold_joins},pred={pred_joins}")
            if pred_joins < gold_joins:
                patterns["missing_join"] += 1
            else:
                patterns["extra_join"] += 1

        # Check aggregation differences
        gold_aggs = set(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', gold_upper))
        pred_aggs = set(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', pred_upper))
        if gold_aggs != pred_aggs:
            issues.append(f"agg_diff:gold={gold_aggs},pred={pred_aggs}")
            patterns["wrong_aggregation"] += 1

        # Check GROUP BY presence
        if 'GROUP BY' in gold_upper and 'GROUP BY' not in pred_upper:
            issues.append("missing_group_by")
            patterns["missing_group_by"] += 1
        elif 'GROUP BY' not in gold_upper and 'GROUP BY' in pred_upper:
            issues.append("extra_group_by")
            patterns["extra_group_by"] += 1

        # Check HAVING
        if 'HAVING' in gold_upper and 'HAVING' not in pred_upper:
            issues.append("missing_having")
            patterns["missing_having"] += 1

        # Check ORDER BY / LIMIT
        if 'ORDER BY' in gold_upper and 'ORDER BY' not in pred_upper:
            issues.append("missing_order_by")
            patterns["missing_order_by"] += 1
        if 'LIMIT' in gold_upper and 'LIMIT' not in pred_upper:
            issues.append("missing_limit")
            patterns["missing_limit"] += 1
        if 'LIMIT' not in gold_upper and 'LIMIT' in pred_upper:
            issues.append("extra_limit")
            patterns["extra_limit"] += 1

        # Check WHERE clause presence
        if 'WHERE' in gold_upper and 'WHERE' not in pred_upper:
            issues.append("missing_where")
            patterns["missing_where"] += 1

        # Check DISTINCT
        if 'DISTINCT' in gold_upper and 'DISTINCT' not in pred_upper:
            issues.append("missing_distinct")
            patterns["missing_distinct"] += 1
        if 'DISTINCT' not in gold_upper and 'DISTINCT' in pred_upper:
            issues.append("extra_distinct")
            patterns["extra_distinct"] += 1

        # Check subquery usage
        gold_subq = len(re.findall(r'\(\s*SELECT\b', gold_upper))
        pred_subq = len(re.findall(r'\(\s*SELECT\b', pred_upper))
        if gold_subq != pred_subq:
            issues.append(f"subquery_diff:gold={gold_subq},pred={pred_subq}")
            patterns["subquery_diff"] += 1

        # Check CASE/WHEN
        if 'CASE' in gold_upper and 'CASE' not in pred_upper:
            issues.append("missing_case")
            patterns["missing_case"] += 1

        # Check for empty vs non-empty results
        pred_rows = ed.get("pred_rows", 0)
        ref_rows = ed.get("ref_rows", 0)
        if pred_rows == 0 and ref_rows > 0:
            issues.append("empty_result")
            patterns["empty_result"] += 1
        elif pred_rows != ref_rows:
            issues.append(f"row_count_diff:pred={pred_rows},ref={ref_rows}")
            patterns["row_count_diff"] += 1

        if not issues:
            issues.append("subtle_logic_diff")
            patterns["subtle_logic_diff"] += 1

        examples.append({
            "idx": s.get("idx"),
            "db_id": s.get("db_id"),
            "question": question[:150],
            "evidence": evidence[:150] if evidence else "",
            "difficulty": s.get("difficulty"),
            "issues": issues,
            "gold_sql": gold[:300],
            "model_sql": pred[:300] if pred else "",
            "pred_rows": pred_rows,
            "ref_rows": ref_rows,
        })

    return patterns, examples


def main():
    data = load_data()
    samples = data["per_sample"]
    total = len(samples)

    correct_samples = [s for s in samples if s["correct"]]
    wrong_samples = [s for s in samples if not s["correct"]]
    n_correct = len(correct_samples)
    n_wrong = len(wrong_samples)

    print(f"Total samples: {total}")
    print(f"Correct: {n_correct} ({n_correct/total*100:.1f}%)")
    print(f"Wrong: {n_wrong} ({n_wrong/total*100:.1f}%)")
    print()

    # =========================================================================
    # 1. Accuracy by difficulty tier
    # =========================================================================
    print("=" * 70)
    print("1. ACCURACY BY DIFFICULTY TIER")
    print("=" * 70)
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        d = s.get("difficulty", "unknown")
        difficulty_stats[d]["total"] += 1
        if s["correct"]:
            difficulty_stats[d]["correct"] += 1

    difficulty_results = {}
    for d in ["simple", "moderate", "challenging", "unknown"]:
        if d in difficulty_stats:
            stats = difficulty_stats[d]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            difficulty_results[d] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "wrong": stats["total"] - stats["correct"],
                "accuracy": round(acc, 2),
            }
            print(f"  {d:15s}: {stats['correct']:4d}/{stats['total']:4d} = {acc:5.1f}%")

    # =========================================================================
    # 2. Accuracy by database
    # =========================================================================
    print()
    print("=" * 70)
    print("2. ACCURACY BY DATABASE")
    print("=" * 70)
    db_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        db = s.get("db_id", "unknown")
        db_stats[db]["total"] += 1
        if s["correct"]:
            db_stats[db]["correct"] += 1

    db_results = {}
    sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]["correct"] / max(x[1]["total"], 1))
    for db, stats in sorted_dbs:
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        db_results[db] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "wrong": stats["total"] - stats["correct"],
            "accuracy": round(acc, 2),
        }
        print(f"  {db:40s}: {stats['correct']:4d}/{stats['total']:4d} = {acc:5.1f}%")

    # =========================================================================
    # 3. Error categorization
    # =========================================================================
    print()
    print("=" * 70)
    print("3. ERROR CATEGORIZATION")
    print("=" * 70)
    error_cats = Counter()
    error_cat_samples = defaultdict(list)
    for s in wrong_samples:
        cat = classify_error(s)
        error_cats[cat] += 1
        error_cat_samples[cat].append(s)

    error_cat_results = {}
    for cat, count in error_cats.most_common():
        pct = count / n_wrong * 100
        error_cat_results[cat] = {"count": count, "pct_of_wrong": round(pct, 2)}
        print(f"  {cat:25s}: {count:4d} ({pct:5.1f}% of wrong)")

    # =========================================================================
    # 4. Common exec error patterns
    # =========================================================================
    print()
    print("=" * 70)
    print("4. COMMON EXEC ERROR PATTERNS")
    print("=" * 70)
    exec_error_patterns = Counter()
    exec_error_examples = []
    for s in wrong_samples:
        ed = s.get("exec_details", {})
        pred_error = ed.get("pred_error")
        if pred_error:
            pattern = classify_exec_error(pred_error)
            exec_error_patterns[pattern] += 1
            exec_error_examples.append({
                "idx": s.get("idx"),
                "db_id": s.get("db_id"),
                "pattern": pattern,
                "error": pred_error[:200],
                "model_sql": s.get("model_sql", "")[:200],
                "question": s.get("question", "")[:150],
            })

    exec_error_results = {}
    for pattern, count in exec_error_patterns.most_common():
        exec_error_results[pattern] = count
        print(f"  {pattern:30s}: {count}")

    print(f"\n  Total exec errors: {sum(exec_error_patterns.values())}")
    print(f"\n  Example exec errors:")
    for ex in exec_error_examples[:10]:
        print(f"    [{ex['pattern']}] db={ex['db_id']}: {ex['error'][:100]}")

    # =========================================================================
    # 5. Wrong result patterns
    # =========================================================================
    print()
    print("=" * 70)
    print("5. WRONG RESULT PATTERNS")
    print("=" * 70)

    # Filter to only wrong_result and empty_result categories
    non_exec_wrong = [s for s in wrong_samples if classify_error(s) in ("wrong_result", "empty_result", "unexpected_results")]
    print(f"  Analyzing {len(non_exec_wrong)} wrong/empty result samples...")

    wrong_patterns, wrong_examples = analyze_wrong_examples(non_exec_wrong)

    print(f"\n  Pattern frequencies (from top 50 analyzed):")
    for pattern, count in sorted(wrong_patterns.items(), key=lambda x: -x[1]):
        print(f"    {pattern:30s}: {count}")

    # Now do the analysis for ALL wrong non-exec samples
    all_wrong_patterns = Counter()
    for s in non_exec_wrong:
        ed = s.get("exec_details", {})
        gold = s.get("gold_sql", "").upper()
        pred = (s.get("model_sql", "") or "").upper()
        pred_rows = ed.get("pred_rows", 0)
        ref_rows = ed.get("ref_rows", 0)

        if pred_rows == 0 and ref_rows > 0:
            all_wrong_patterns["empty_result"] += 1
        if pred_rows != ref_rows and pred_rows > 0 and ref_rows > 0:
            all_wrong_patterns["row_count_mismatch"] += 1

        # Structural differences
        gold_joins = len(re.findall(r'\bJOIN\b', gold))
        pred_joins = len(re.findall(r'\bJOIN\b', pred))
        if pred_joins < gold_joins:
            all_wrong_patterns["missing_join_all"] += 1
        if pred_joins > gold_joins:
            all_wrong_patterns["extra_join_all"] += 1

        if 'GROUP BY' in gold and 'GROUP BY' not in pred:
            all_wrong_patterns["missing_group_by_all"] += 1
        if 'GROUP BY' not in gold and 'GROUP BY' in pred:
            all_wrong_patterns["extra_group_by_all"] += 1
        if 'HAVING' in gold and 'HAVING' not in pred:
            all_wrong_patterns["missing_having_all"] += 1
        if 'DISTINCT' in gold and 'DISTINCT' not in pred:
            all_wrong_patterns["missing_distinct_all"] += 1
        if 'DISTINCT' not in gold and 'DISTINCT' in pred:
            all_wrong_patterns["extra_distinct_all"] += 1
        if 'ORDER BY' in gold and 'ORDER BY' not in pred:
            all_wrong_patterns["missing_order_by_all"] += 1
        if 'LIMIT' in gold and 'LIMIT' not in pred:
            all_wrong_patterns["missing_limit_all"] += 1
        if 'LIMIT' not in gold and 'LIMIT' in pred:
            all_wrong_patterns["extra_limit_all"] += 1
        if 'WHERE' in gold and 'WHERE' not in pred:
            all_wrong_patterns["missing_where_all"] += 1
        if 'CASE' in gold and 'CASE' not in pred:
            all_wrong_patterns["missing_case_all"] += 1

        gold_aggs = set(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', gold))
        pred_aggs = set(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', pred))
        if gold_aggs != pred_aggs:
            all_wrong_patterns["agg_diff_all"] += 1

        gold_subq = len(re.findall(r'\(\s*SELECT\b', gold))
        pred_subq = len(re.findall(r'\(\s*SELECT\b', pred))
        if gold_subq > 0 and pred_subq == 0:
            all_wrong_patterns["missing_subquery_all"] += 1
        if gold_subq == 0 and pred_subq > 0:
            all_wrong_patterns["extra_subquery_all"] += 1

    print(f"\n  All non-exec wrong samples ({len(non_exec_wrong)}) structural analysis:")
    for pattern, count in sorted(all_wrong_patterns.items(), key=lambda x: -x[1]):
        pct = count / len(non_exec_wrong) * 100
        print(f"    {pattern:30s}: {count:4d} ({pct:5.1f}%)")

    # =========================================================================
    # 6. Evidence utilization
    # =========================================================================
    print()
    print("=" * 70)
    print("6. EVIDENCE UTILIZATION")
    print("=" * 70)

    with_evidence = [s for s in samples if s.get("evidence", "").strip()]
    without_evidence = [s for s in samples if not s.get("evidence", "").strip()]

    with_ev_correct = sum(1 for s in with_evidence if s["correct"])
    without_ev_correct = sum(1 for s in without_evidence if s["correct"])

    evidence_results = {
        "with_evidence": {
            "total": len(with_evidence),
            "correct": with_ev_correct,
            "accuracy": round(with_ev_correct / max(len(with_evidence), 1) * 100, 2),
        },
        "without_evidence": {
            "total": len(without_evidence),
            "correct": without_ev_correct,
            "accuracy": round(without_ev_correct / max(len(without_evidence), 1) * 100, 2),
        },
    }

    print(f"  With evidence:    {with_ev_correct:4d}/{len(with_evidence):4d} = {evidence_results['with_evidence']['accuracy']:.1f}%")
    print(f"  Without evidence: {without_ev_correct:4d}/{len(without_evidence):4d} = {evidence_results['without_evidence']['accuracy']:.1f}%")

    # Evidence accuracy by difficulty
    print(f"\n  With evidence, by difficulty:")
    for d in ["simple", "moderate", "challenging"]:
        subset = [s for s in with_evidence if s.get("difficulty") == d]
        if subset:
            c = sum(1 for s in subset if s["correct"])
            print(f"    {d:15s}: {c:4d}/{len(subset):4d} = {c/len(subset)*100:5.1f}%")

    print(f"\n  Without evidence, by difficulty:")
    for d in ["simple", "moderate", "challenging"]:
        subset = [s for s in without_evidence if s.get("difficulty") == d]
        if subset:
            c = sum(1 for s in subset if s["correct"])
            print(f"    {d:15s}: {c:4d}/{len(subset):4d} = {c/len(subset)*100:5.1f}%")

    # Check if wrong samples with evidence have evidence-related terms in the gold SQL
    wrong_with_evidence = [s for s in wrong_samples if s.get("evidence", "").strip()]
    print(f"\n  Wrong samples with evidence: {len(wrong_with_evidence)}")

    # For evidence samples, check if model SQL contains evidence keywords
    evidence_keyword_usage = {"used_in_model": 0, "not_used_in_model": 0, "total_checked": 0}
    evidence_misuse_examples = []
    for s in wrong_with_evidence[:100]:
        ev = s.get("evidence", "")
        model_sql = s.get("model_sql", "") or ""
        gold_sql = s.get("gold_sql", "")

        # Extract quoted string values from evidence
        quoted_values = re.findall(r"'([^']+)'", ev)
        if not quoted_values:
            continue

        evidence_keyword_usage["total_checked"] += 1
        used = False
        for val in quoted_values:
            if val.lower() in model_sql.lower():
                used = True
                break

        if used:
            evidence_keyword_usage["used_in_model"] += 1
        else:
            evidence_keyword_usage["not_used_in_model"] += 1
            evidence_misuse_examples.append({
                "idx": s.get("idx"),
                "db_id": s.get("db_id"),
                "question": s.get("question", "")[:120],
                "evidence": ev[:200],
                "gold_sql": gold_sql[:200],
                "model_sql": model_sql[:200],
            })

    print(f"\n  Evidence keyword usage in wrong samples (checked {evidence_keyword_usage['total_checked']}):")
    print(f"    Evidence values found in model SQL: {evidence_keyword_usage['used_in_model']}")
    print(f"    Evidence values NOT found in model SQL: {evidence_keyword_usage['not_used_in_model']}")

    if evidence_misuse_examples:
        print(f"\n  Examples where model DIDN'T use evidence values:")
        for ex in evidence_misuse_examples[:5]:
            print(f"    idx={ex['idx']} db={ex['db_id']}")
            print(f"      Q: {ex['question']}")
            print(f"      Evidence: {ex['evidence']}")
            print(f"      Gold: {ex['gold_sql'][:120]}")
            print(f"      Pred: {ex['model_sql'][:120]}")
            print()

    # =========================================================================
    # 7. SQL complexity analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("7. SQL COMPLEXITY ANALYSIS (based on gold SQL)")
    print("=" * 70)

    # Accuracy by number of JOINs in gold SQL
    join_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        gold = s.get("gold_sql", "")
        n_joins = len(re.findall(r'\bJOIN\b', gold.upper()))
        join_stats[n_joins]["total"] += 1
        if s["correct"]:
            join_stats[n_joins]["correct"] += 1

    join_results = {}
    print(f"\n  Accuracy by number of JOINs (gold SQL):")
    for n in sorted(join_stats.keys()):
        stats = join_stats[n]
        acc = stats["correct"] / stats["total"] * 100
        join_results[str(n)] = {"total": stats["total"], "correct": stats["correct"], "accuracy": round(acc, 2)}
        print(f"    {n} JOINs: {stats['correct']:4d}/{stats['total']:4d} = {acc:5.1f}%")

    # Accuracy by SQL features
    features_to_check = [
        "has_subquery", "has_group_by", "has_having", "has_order_by",
        "has_limit", "has_distinct", "has_union", "has_case", "has_like",
        "has_in", "has_between", "has_except", "has_intersect"
    ]

    feature_results = {}
    print(f"\n  Accuracy by SQL feature presence (gold SQL):")
    for feat in features_to_check:
        with_feat = {"total": 0, "correct": 0}
        without_feat = {"total": 0, "correct": 0}
        for s in samples:
            gold = s.get("gold_sql", "")
            feats = count_sql_features(gold)
            if feats.get(feat, 0):
                with_feat["total"] += 1
                if s["correct"]:
                    with_feat["correct"] += 1
            else:
                without_feat["total"] += 1
                if s["correct"]:
                    without_feat["correct"] += 1

        if with_feat["total"] > 0:
            acc_with = with_feat["correct"] / with_feat["total"] * 100
            acc_without = without_feat["correct"] / without_feat["total"] * 100 if without_feat["total"] > 0 else 0
            feature_results[feat] = {
                "with": {"total": with_feat["total"], "correct": with_feat["correct"], "accuracy": round(acc_with, 2)},
                "without": {"total": without_feat["total"], "correct": without_feat["correct"], "accuracy": round(acc_without, 2)},
                "delta": round(acc_with - acc_without, 2),
            }
            print(f"    {feat:20s}: WITH={acc_with:5.1f}% ({with_feat['total']:4d})  WITHOUT={acc_without:5.1f}% ({without_feat['total']:4d})  delta={acc_with - acc_without:+.1f}%")

    # Accuracy by aggregate count
    agg_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        gold = s.get("gold_sql", "")
        feats = count_sql_features(gold)
        n_aggs = feats.get("agg_count", 0)
        bucket = min(n_aggs, 3)  # 0, 1, 2, 3+
        agg_stats[bucket]["total"] += 1
        if s["correct"]:
            agg_stats[bucket]["correct"] += 1

    print(f"\n  Accuracy by aggregate function count (gold SQL):")
    for n in sorted(agg_stats.keys()):
        stats = agg_stats[n]
        acc = stats["correct"] / stats["total"] * 100
        label = f"{n}+" if n == 3 else str(n)
        print(f"    {label} aggregates: {stats['correct']:4d}/{stats['total']:4d} = {acc:5.1f}%")

    # Overall complexity score
    complexity_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        gold = s.get("gold_sql", "")
        feats = count_sql_features(gold)
        complexity = feats.get("joins", 0) + feats.get("has_subquery", 0) * 2 + feats.get("has_group_by", 0) + feats.get("has_having", 0) + feats.get("agg_count", 0)
        bucket = min(complexity, 7)
        complexity_stats[bucket]["total"] += 1
        if s["correct"]:
            complexity_stats[bucket]["correct"] += 1

    print(f"\n  Accuracy by complexity score (JOINs + 2*subquery + GROUP_BY + HAVING + agg_count):")
    for n in sorted(complexity_stats.keys()):
        stats = complexity_stats[n]
        acc = stats["correct"] / stats["total"] * 100
        label = f"{n}+" if n == 7 else str(n)
        print(f"    score {label}: {stats['correct']:4d}/{stats['total']:4d} = {acc:5.1f}%")

    # =========================================================================
    # 8. Token length analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("8. TOKEN LENGTH ANALYSIS")
    print("=" * 70)

    # Use character length as proxy for token length
    length_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    all_lengths_correct = []
    all_lengths_wrong = []

    for s in samples:
        model_sql = s.get("model_sql", "") or ""
        length = len(model_sql)
        if s["correct"]:
            all_lengths_correct.append(length)
        else:
            all_lengths_wrong.append(length)

        # Bucket by length
        if length < 100:
            bucket = "<100"
        elif length < 200:
            bucket = "100-200"
        elif length < 300:
            bucket = "200-300"
        elif length < 500:
            bucket = "300-500"
        elif length < 800:
            bucket = "500-800"
        else:
            bucket = "800+"

        length_stats[bucket]["total"] += 1
        if s["correct"]:
            length_stats[bucket]["correct"] += 1

    bucket_order = ["<100", "100-200", "200-300", "300-500", "500-800", "800+"]
    length_results = {}
    print(f"\n  Accuracy by model SQL character length:")
    for bucket in bucket_order:
        if bucket in length_stats:
            stats = length_stats[bucket]
            acc = stats["correct"] / stats["total"] * 100
            length_results[bucket] = {"total": stats["total"], "correct": stats["correct"], "accuracy": round(acc, 2)}
            print(f"    {bucket:12s}: {stats['correct']:4d}/{stats['total']:4d} = {acc:5.1f}%")

    avg_correct = sum(all_lengths_correct) / max(len(all_lengths_correct), 1)
    avg_wrong = sum(all_lengths_wrong) / max(len(all_lengths_wrong), 1)
    print(f"\n  Average SQL length (chars):")
    print(f"    Correct: {avg_correct:.0f}")
    print(f"    Wrong:   {avg_wrong:.0f}")

    # Also check if model response had markdown/extra text
    verbose_count = {"correct": 0, "wrong": 0}
    for s in samples:
        raw = s.get("model_response_raw", "") or ""
        if "```" in raw or raw.count("\n") > 20:
            if s["correct"]:
                verbose_count["correct"] += 1
            else:
                verbose_count["wrong"] += 1

    print(f"\n  Verbose responses (containing ``` or >20 lines):")
    print(f"    Among correct: {verbose_count['correct']}")
    print(f"    Among wrong:   {verbose_count['wrong']}")

    # =========================================================================
    # 9. Detailed examples of wrong results (top 20)
    # =========================================================================
    print()
    print("=" * 70)
    print("9. DETAILED EXAMPLES (20 wrong result samples)")
    print("=" * 70)

    # Pick diverse examples
    wrong_result_samples = [s for s in wrong_samples if classify_error(s) == "wrong_result"]
    empty_result_samples = [s for s in wrong_samples if classify_error(s) == "empty_result"]

    print(f"\n  === WRONG RESULT examples (SQL runs, wrong answer) ===")
    for i, s in enumerate(wrong_result_samples[:15]):
        ed = s.get("exec_details", {})
        print(f"\n  [{i+1}] idx={s['idx']} db={s['db_id']} difficulty={s.get('difficulty')}")
        print(f"      Q: {s['question'][:120]}")
        if s.get("evidence"):
            print(f"      Evidence: {s['evidence'][:120]}")
        print(f"      Gold:  {s['gold_sql'][:160]}")
        print(f"      Model: {s.get('model_sql', '')[:160]}")
        print(f"      pred_rows={ed.get('pred_rows')}, ref_rows={ed.get('ref_rows')}")

    print(f"\n  === EMPTY RESULT examples (model returns 0 rows, gold returns N) ===")
    for i, s in enumerate(empty_result_samples[:5]):
        ed = s.get("exec_details", {})
        print(f"\n  [{i+1}] idx={s['idx']} db={s['db_id']} difficulty={s.get('difficulty')}")
        print(f"      Q: {s['question'][:120]}")
        if s.get("evidence"):
            print(f"      Evidence: {s['evidence'][:120]}")
        print(f"      Gold:  {s['gold_sql'][:160]}")
        print(f"      Model: {s.get('model_sql', '')[:160]}")
        print(f"      pred_rows=0, ref_rows={ed.get('ref_rows')}")

    # =========================================================================
    # 10. Cross-dimensional analysis: hardest combinations
    # =========================================================================
    print()
    print("=" * 70)
    print("10. HARDEST COMBINATIONS (db + difficulty)")
    print("=" * 70)

    combo_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in samples:
        key = f"{s.get('db_id', '?')}|{s.get('difficulty', '?')}"
        combo_stats[key]["total"] += 1
        if s["correct"]:
            combo_stats[key]["correct"] += 1

    # Sort by accuracy ascending, filter to combos with >= 5 samples
    combos_sorted = sorted(
        [(k, v) for k, v in combo_stats.items() if v["total"] >= 5],
        key=lambda x: x[1]["correct"] / x[1]["total"]
    )

    print(f"\n  Worst 20 combos (min 5 samples):")
    for k, v in combos_sorted[:20]:
        db, diff = k.split("|")
        acc = v["correct"] / v["total"] * 100
        print(f"    {db:40s} {diff:12s}: {v['correct']:3d}/{v['total']:3d} = {acc:5.1f}%")

    # =========================================================================
    # Save results to JSON
    # =========================================================================
    analysis_output = {
        "summary": {
            "total": total,
            "correct": n_correct,
            "wrong": n_wrong,
            "accuracy": round(n_correct / total * 100, 2),
        },
        "accuracy_by_difficulty": difficulty_results,
        "accuracy_by_database": db_results,
        "error_categorization": error_cat_results,
        "exec_error_patterns": exec_error_results,
        "exec_error_examples": exec_error_examples[:15],
        "wrong_result_patterns_top50": dict(wrong_patterns.most_common()),
        "wrong_result_patterns_all": dict(all_wrong_patterns.most_common()),
        "wrong_result_detailed_examples": wrong_examples[:20],
        "evidence_utilization": evidence_results,
        "evidence_keyword_usage": evidence_keyword_usage,
        "evidence_misuse_examples": evidence_misuse_examples[:10],
        "sql_complexity_by_joins": join_results,
        "sql_complexity_by_features": feature_results,
        "token_length_analysis": length_results,
        "avg_sql_length": {"correct": round(avg_correct, 1), "wrong": round(avg_wrong, 1)},
        "hardest_combos": [
            {"db": k.split("|")[0], "difficulty": k.split("|")[1], **v}
            for k, v in combos_sorted[:20]
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\n\nFull analysis saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
