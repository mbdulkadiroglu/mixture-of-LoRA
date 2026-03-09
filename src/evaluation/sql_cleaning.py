"""
Shared SQL text cleaning helpers for evaluation and training.
"""

import json
import re


SQL_START_PATTERN = re.compile(
    r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\s+.+",
    re.IGNORECASE | re.DOTALL,
)
SQL_BLOCK_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql_from_json(text: str) -> tuple[str, str]:
    """
    Extract SQL and reasoning from JSON-structured model output.

    Expects {"reasoning": "...", "sql": "..."} format.
    Falls back to extract_sql_from_text if JSON parsing fails.

    Returns:
        (sql, reasoning) tuple. reasoning is empty string on fallback.
    """
    # Strip markdown code blocks if present
    json_matches = JSON_BLOCK_PATTERN.findall(text)
    if json_matches:
        candidate = json_matches[-1].strip()
    else:
        candidate = text.strip()

    # Try parsing as JSON
    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "sql" in data:
            sql = data["sql"].strip()
            reasoning = data.get("reasoning", "")
            return sql, reasoning
    except (json.JSONDecodeError, TypeError):
        pass

    # Try finding JSON object in the text with a regex
    json_obj_match = re.search(r'\{[^{}]*"sql"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
    if json_obj_match:
        try:
            data = json.loads(json_obj_match.group(0))
            sql = data["sql"].strip()
            reasoning = data.get("reasoning", "")
            return sql, reasoning
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: extract SQL normally
    return extract_sql_from_text(text), ""


def _split_at_unquoted_semicolon(text: str) -> tuple[str, str | None]:
    """
    Split text at the first semicolon outside single/double-quoted strings.

    Returns (head, tail_without_semicolon). If no such semicolon exists,
    returns (text, None).
    """
    in_single = False
    in_double = False
    i = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if ch == "'" and not in_double:
            # SQL escaped single quote inside single-quoted string: ''
            if in_single and nxt == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            # SQL escaped double quote inside double-quoted identifier: ""
            if in_double and nxt == '"':
                i += 2
                continue
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            return text[:i], text[i + 1:]

        i += 1

    return text, None


def extract_sql_from_text(text: str) -> str:
    """
    Extract/normalize SQL from model output that may include formatting noise.
    """
    matches = SQL_BLOCK_PATTERN.findall(text)
    if matches:
        sql = matches[-1].strip()
    else:
        match = SQL_START_PATTERN.search(text)
        sql = match.group(0).strip() if match else text.strip()

    sql = sql.removeprefix("```sql").removesuffix("```").strip()
    sql = sql.removeprefix("```").strip()

    head, tail = _split_at_unquoted_semicolon(sql)
    if tail is not None:
        sql = head.strip()
    else:
        for marker in ["\n\n", "\nExplanation:", "\nNote:", "\n--"]:
            if marker in sql:
                sql = sql.split(marker)[0].strip()
                break

    return sql.strip()
