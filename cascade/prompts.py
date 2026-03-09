"""
Shared prompt/message helpers for cascade modules.
"""

SPIDER_SQL_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a "
    "natural language question, output only the SQL query. Do not "
    "include any explanation, formatting, or markdown. Output only "
    "valid SQLite SQL."
)

# Few-shot examples for BIRD: chosen for pattern diversity and generalizability.
# Each demonstrates evidence-to-SQL mapping across different SQL constructs.
BIRD_FEW_SHOT_EXAMPLES = [
    # Example 1: Simple JOIN + WHERE with evidence mapping a term to a column value
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE sales (\n"
            "  stor_id TEXT [PRIMARY KEY],\n"
            "  ord_num TEXT [PRIMARY KEY],\n"
            "  qty INTEGER,\n"
            "  payterms TEXT,\n"
            "  title_id TEXT [FOREIGN KEY -> titles.title_id]\n"
            ");\n"
            "CREATE TABLE titles (\n"
            "  title_id TEXT [PRIMARY KEY],\n"
            "  title TEXT,\n"
            "  price REAL,\n"
            "  pubdate DATETIME\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "publication date refers to pubdate; payment terms refers to payterms; "
            "payterms = 'ON invoice'\n\n"
            "Question: List the title, price and publication date for all sales "
            "with 'ON invoice' payment terms.\n\n"
            "SQL Query:"
        ),
        "assistant": (
            "SELECT T2.title, T2.price, T2.pubdate "
            "FROM sales AS T1 "
            "INNER JOIN titles AS T2 ON T1.title_id = T2.title_id "
            "WHERE T1.payterms = 'ON invoice'"
        ),
    },
    # Example 2: JOIN + GROUP BY + HAVING + COUNT(DISTINCT)
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE taughtBy (\n"
            "  course_id INTEGER,\n"
            "  p_id INTEGER\n"
            ");\n"
            "CREATE TABLE person (\n"
            "  p_id INTEGER [PRIMARY KEY],\n"
            "  professor INTEGER,\n"
            "  student INTEGER\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "IDs of the professors refers to taughtBy.p_id and professor = 1; "
            "teaches more than 3 courses refers to count(course_id) > 3\n\n"
            "Question: Please list the IDs of the professors that teaches "
            "more than 3 courses.\n\n"
            "SQL Query:"
        ),
        "assistant": (
            "SELECT T1.p_id "
            "FROM taughtBy AS T1 "
            "INNER JOIN person AS T2 ON T1.p_id = T2.p_id "
            "WHERE T2.professor = 1 "
            "GROUP BY T1.p_id "
            "HAVING COUNT(DISTINCT T1.course_id) > 3"
        ),
    },
    # Example 3: GROUP BY + ORDER BY + LIMIT (simple top-N, no subquery needed)
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE District (\n"
            "  district_no INTEGER [PRIMARY KEY],\n"
            "  district_name TEXT,\n"
            "  fax INTEGER\n"
            ");\n"
            "CREATE TABLE Crime (\n"
            "  report_no INTEGER [PRIMARY KEY],\n"
            "  case_number TEXT,\n"
            "  date TEXT,\n"
            "  district_no INTEGER [FOREIGN KEY -> District.district_no]\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "fax number refers to fax; the most number of crimes refers to "
            "max(count(case_number)); in January 2018 refers to date like '%1/2018%'\n\n"
            "Question: What is the fax number for the district with the most "
            "number of crimes in January, 2018?\n\n"
            "SQL Query:"
        ),
        "assistant": (
            "SELECT T1.fax "
            "FROM District AS T1 "
            "INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no "
            "WHERE T2.date LIKE '%1/2018%' "
            "GROUP BY T2.district_no "
            "ORDER BY COUNT(case_number) DESC LIMIT 1"
        ),
    },
    # Example 4: CASE WHEN for percentage calculation with encoded values
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE employee (\n"
            "  emp_id TEXT [PRIMARY KEY],\n"
            "  fname TEXT,\n"
            "  job_id INTEGER [FOREIGN KEY -> jobs.job_id]\n"
            ");\n"
            "CREATE TABLE jobs (\n"
            "  job_id INTEGER [PRIMARY KEY],\n"
            "  job_desc TEXT\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "Editor or Designer are job descriptions which refers to job_desc; "
            "percentage = DIVIDE(count(job_desc = 'Editor' or job_desc = 'Designer'), "
            "count(all employees)) * 100\n\n"
            "Question: Calculate the percentage of the employees who are "
            "Editor or Designer?\n\n"
            "SQL Query:"
        ),
        "assistant": (
            "SELECT CAST(SUM(CASE WHEN T2.job_desc IN ('Editor', 'Designer') "
            "THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) "
            "FROM employee AS T1 "
            "INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id"
        ),
    },
    # Example 5: Multi-JOIN (3 tables) with ORDER BY + LIMIT instead of subquery
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE price (\n"
            "  ID INTEGER [PRIMARY KEY],\n"
            "  price REAL\n"
            ");\n"
            "CREATE TABLE production (\n"
            "  ID INTEGER [FOREIGN KEY -> price.ID],\n"
            "  country INTEGER [FOREIGN KEY -> country.origin]\n"
            ");\n"
            "CREATE TABLE country (\n"
            "  origin INTEGER [PRIMARY KEY],\n"
            "  country TEXT\n"
            ");\n"
            "CREATE TABLE data (\n"
            "  ID INTEGER [PRIMARY KEY],\n"
            "  car_name TEXT\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "name of the car refers to car_name; the most expensive refers to "
            "max(price); produced by the USA refers to country = 'USA'\n\n"
            "Question: What is the name of the most expensive car that was "
            "produced by the USA?\n\n"
            "SQL Query:"
        ),
        "assistant": (
            "SELECT T4.car_name "
            "FROM price AS T1 "
            "INNER JOIN production AS T2 ON T1.ID = T2.ID "
            "INNER JOIN country AS T3 ON T3.origin = T2.country "
            "INNER JOIN data AS T4 ON T4.ID = T1.ID "
            "WHERE T3.country = 'USA' "
            "ORDER BY T1.price DESC LIMIT 1"
        ),
    },
]

# JSON structured output: forces reasoning before SQL via JSON schema.
# The model MUST output {"reasoning": "...", "sql": "..."} — reasoning first
# ensures the model thinks before generating SQL.
BIRD_JSON_FEW_SHOT_EXAMPLES = [
    # Example 1: Simple JOIN + WHERE with evidence mapping
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE sales (\n"
            "  stor_id TEXT [PRIMARY KEY],\n"
            "  ord_num TEXT [PRIMARY KEY],\n"
            "  qty INTEGER,\n"
            "  payterms TEXT,\n"
            "  title_id TEXT [FOREIGN KEY -> titles.title_id]\n"
            ");\n"
            "CREATE TABLE titles (\n"
            "  title_id TEXT [PRIMARY KEY],\n"
            "  title TEXT,\n"
            "  price REAL,\n"
            "  pubdate DATETIME\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "publication date refers to pubdate; payment terms refers to payterms; "
            "payterms = 'ON invoice'\n\n"
            "Question: List the title, price and publication date for all sales "
            "with 'ON invoice' payment terms."
        ),
        "assistant": (
            '{"reasoning": "Need title, price, pubdate from titles table. '
            "Filter by payterms = 'ON invoice' in sales table. "
            'JOIN sales.title_id -> titles.title_id.", '
            '"sql": "SELECT T2.title, T2.price, T2.pubdate '
            "FROM sales AS T1 "
            "INNER JOIN titles AS T2 ON T1.title_id = T2.title_id "
            "WHERE T1.payterms = 'ON invoice'\"}"
        ),
    },
    # Example 2: JOIN + GROUP BY + HAVING + COUNT(DISTINCT)
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE taughtBy (\n"
            "  course_id INTEGER,\n"
            "  p_id INTEGER\n"
            ");\n"
            "CREATE TABLE person (\n"
            "  p_id INTEGER [PRIMARY KEY],\n"
            "  professor INTEGER,\n"
            "  student INTEGER\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "IDs of the professors refers to taughtBy.p_id and professor = 1; "
            "teaches more than 3 courses refers to count(course_id) > 3\n\n"
            "Question: Please list the IDs of the professors that teaches "
            "more than 3 courses."
        ),
        "assistant": (
            '{"reasoning": "Return p_id. Filter professors with professor = 1 in person table. '
            "JOIN taughtBy.p_id -> person.p_id. GROUP BY p_id, "
            "HAVING COUNT(DISTINCT course_id) > 3 to count unique courses per professor.\", "
            '"sql": "SELECT T1.p_id '
            "FROM taughtBy AS T1 "
            "INNER JOIN person AS T2 ON T1.p_id = T2.p_id "
            "WHERE T2.professor = 1 "
            "GROUP BY T1.p_id "
            "HAVING COUNT(DISTINCT T1.course_id) > 3\"}"
        ),
    },
    # Example 3: GROUP BY + ORDER BY + LIMIT (top-N without subquery)
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE District (\n"
            "  district_no INTEGER [PRIMARY KEY],\n"
            "  district_name TEXT,\n"
            "  fax INTEGER\n"
            ");\n"
            "CREATE TABLE Crime (\n"
            "  report_no INTEGER [PRIMARY KEY],\n"
            "  case_number TEXT,\n"
            "  date TEXT,\n"
            "  district_no INTEGER [FOREIGN KEY -> District.district_no]\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "fax number refers to fax; the most number of crimes refers to "
            "max(count(case_number)); in January 2018 refers to date like '%1/2018%'\n\n"
            "Question: What is the fax number for the district with the most "
            "number of crimes in January, 2018?"
        ),
        "assistant": (
            '{"reasoning": "Return fax from District. Filter January 2018 with date LIKE \'%1/2018%\'. '
            "JOIN Crime.district_no -> District.district_no. "
            "GROUP BY district_no, ORDER BY COUNT DESC, LIMIT 1 for the most crimes.\", "
            '"sql": "SELECT T1.fax '
            "FROM District AS T1 "
            "INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no "
            "WHERE T2.date LIKE '%1/2018%' "
            "GROUP BY T2.district_no "
            "ORDER BY COUNT(case_number) DESC LIMIT 1\"}"
        ),
    },
    # Example 4: CASE WHEN for percentage calculation
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE employee (\n"
            "  emp_id TEXT [PRIMARY KEY],\n"
            "  fname TEXT,\n"
            "  job_id INTEGER [FOREIGN KEY -> jobs.job_id]\n"
            ");\n"
            "CREATE TABLE jobs (\n"
            "  job_id INTEGER [PRIMARY KEY],\n"
            "  job_desc TEXT\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "Editor or Designer are job descriptions which refers to job_desc; "
            "percentage = DIVIDE(count(job_desc = 'Editor' or job_desc = 'Designer'), "
            "count(all employees)) * 100\n\n"
            "Question: Calculate the percentage of the employees who are "
            "Editor or Designer?"
        ),
        "assistant": (
            '{"reasoning": "Percentage calculation: (matching / total) * 100. '
            "Match job_desc IN ('Editor', 'Designer'). "
            "JOIN employee.job_id -> jobs.job_id. "
            "Use CASE WHEN inside SUM for numerator, COUNT for denominator, CAST as REAL.\", "
            '"sql": "SELECT CAST(SUM(CASE WHEN T2.job_desc IN (\'Editor\', \'Designer\') '
            "THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) "
            "FROM employee AS T1 "
            "INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id\"}"
        ),
    },
    # Example 5: Multi-JOIN (4 tables) with ORDER BY + LIMIT
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE price (\n"
            "  ID INTEGER [PRIMARY KEY],\n"
            "  price REAL\n"
            ");\n"
            "CREATE TABLE production (\n"
            "  ID INTEGER [FOREIGN KEY -> price.ID],\n"
            "  country INTEGER [FOREIGN KEY -> country.origin]\n"
            ");\n"
            "CREATE TABLE country (\n"
            "  origin INTEGER [PRIMARY KEY],\n"
            "  country TEXT\n"
            ");\n"
            "CREATE TABLE data (\n"
            "  ID INTEGER [PRIMARY KEY],\n"
            "  car_name TEXT\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "name of the car refers to car_name; the most expensive refers to "
            "max(price); produced by the USA refers to country = 'USA'\n\n"
            "Question: What is the name of the most expensive car that was "
            "produced by the USA?"
        ),
        "assistant": (
            '{"reasoning": "Return car_name from data. Filter country = \'USA\'. '
            "Trace JOIN path: price.ID -> production.ID, "
            "production.country -> country.origin, data.ID -> price.ID. "
            "ORDER BY price DESC LIMIT 1 for most expensive.\", "
            '"sql": "SELECT T4.car_name '
            "FROM price AS T1 "
            "INNER JOIN production AS T2 ON T1.ID = T2.ID "
            "INNER JOIN country AS T3 ON T3.origin = T2.country "
            "INNER JOIN data AS T4 ON T4.ID = T1.ID "
            "WHERE T3.country = 'USA' "
            "ORDER BY T1.price DESC LIMIT 1\"}"
        ),
    },
]

# Chain-of-thought versions: same examples but with reasoning before SQL.
BIRD_COT_FEW_SHOT_EXAMPLES = [
    # Example 1: Simple JOIN + WHERE with evidence mapping
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE sales (\n"
            "  stor_id TEXT [PRIMARY KEY],\n"
            "  ord_num TEXT [PRIMARY KEY],\n"
            "  qty INTEGER,\n"
            "  payterms TEXT,\n"
            "  title_id TEXT [FOREIGN KEY -> titles.title_id]\n"
            ");\n"
            "CREATE TABLE titles (\n"
            "  title_id TEXT [PRIMARY KEY],\n"
            "  title TEXT,\n"
            "  price REAL,\n"
            "  pubdate DATETIME\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "publication date refers to pubdate; payment terms refers to payterms; "
            "payterms = 'ON invoice'\n\n"
            "Question: List the title, price and publication date for all sales "
            "with 'ON invoice' payment terms.\n\n"
            "Think step by step, then write the SQL query."
        ),
        "assistant": (
            "Step 1: Identify needed columns. The question asks for title, price, "
            "and publication date. From evidence: publication date = pubdate.\n"
            "Step 2: Identify tables. title, price, pubdate are in titles. "
            "payterms is in sales. Need to JOIN these two tables.\n"
            "Step 3: JOIN path. sales.title_id -> titles.title_id.\n"
            "Step 4: Filter. Evidence says payterms = 'ON invoice'.\n\n"
            "SELECT T2.title, T2.price, T2.pubdate "
            "FROM sales AS T1 "
            "INNER JOIN titles AS T2 ON T1.title_id = T2.title_id "
            "WHERE T1.payterms = 'ON invoice'"
        ),
    },
    # Example 2: JOIN + GROUP BY + HAVING + COUNT(DISTINCT)
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE taughtBy (\n"
            "  course_id INTEGER,\n"
            "  p_id INTEGER\n"
            ");\n"
            "CREATE TABLE person (\n"
            "  p_id INTEGER [PRIMARY KEY],\n"
            "  professor INTEGER,\n"
            "  student INTEGER\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "IDs of the professors refers to taughtBy.p_id and professor = 1; "
            "teaches more than 3 courses refers to count(course_id) > 3\n\n"
            "Question: Please list the IDs of the professors that teaches "
            "more than 3 courses.\n\n"
            "Think step by step, then write the SQL query."
        ),
        "assistant": (
            "Step 1: Identify needed columns. Return p_id from taughtBy.\n"
            "Step 2: Filter professors. Evidence says professor = 1 in person table.\n"
            "Step 3: JOIN path. taughtBy.p_id -> person.p_id to filter professors.\n"
            "Step 4: Aggregation. Need to count courses per professor. GROUP BY p_id, "
            "HAVING COUNT > 3. Since a professor may teach the same course multiple "
            "times, use COUNT(DISTINCT course_id) to count unique courses.\n\n"
            "SELECT T1.p_id "
            "FROM taughtBy AS T1 "
            "INNER JOIN person AS T2 ON T1.p_id = T2.p_id "
            "WHERE T2.professor = 1 "
            "GROUP BY T1.p_id "
            "HAVING COUNT(DISTINCT T1.course_id) > 3"
        ),
    },
    # Example 3: GROUP BY + ORDER BY + LIMIT (top-N without subquery)
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE District (\n"
            "  district_no INTEGER [PRIMARY KEY],\n"
            "  district_name TEXT,\n"
            "  fax INTEGER\n"
            ");\n"
            "CREATE TABLE Crime (\n"
            "  report_no INTEGER [PRIMARY KEY],\n"
            "  case_number TEXT,\n"
            "  date TEXT,\n"
            "  district_no INTEGER [FOREIGN KEY -> District.district_no]\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "fax number refers to fax; the most number of crimes refers to "
            "max(count(case_number)); in January 2018 refers to date like '%1/2018%'\n\n"
            "Question: What is the fax number for the district with the most "
            "number of crimes in January, 2018?\n\n"
            "Think step by step, then write the SQL query."
        ),
        "assistant": (
            "Step 1: Identify needed columns. Return fax from District.\n"
            "Step 2: Filter. Evidence says January 2018 = date LIKE '%1/2018%'.\n"
            "Step 3: JOIN path. Crime.district_no -> District.district_no.\n"
            "Step 4: Aggregation. \"Most crimes\" = count crimes per district. "
            "GROUP BY district_no, ORDER BY COUNT DESC, LIMIT 1. "
            "This is simpler than using a subquery.\n\n"
            "SELECT T1.fax "
            "FROM District AS T1 "
            "INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no "
            "WHERE T2.date LIKE '%1/2018%' "
            "GROUP BY T2.district_no "
            "ORDER BY COUNT(case_number) DESC LIMIT 1"
        ),
    },
    # Example 4: CASE WHEN for percentage calculation
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE employee (\n"
            "  emp_id TEXT [PRIMARY KEY],\n"
            "  fname TEXT,\n"
            "  job_id INTEGER [FOREIGN KEY -> jobs.job_id]\n"
            ");\n"
            "CREATE TABLE jobs (\n"
            "  job_id INTEGER [PRIMARY KEY],\n"
            "  job_desc TEXT\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "Editor or Designer are job descriptions which refers to job_desc; "
            "percentage = DIVIDE(count(job_desc = 'Editor' or job_desc = 'Designer'), "
            "count(all employees)) * 100\n\n"
            "Question: Calculate the percentage of the employees who are "
            "Editor or Designer?\n\n"
            "Think step by step, then write the SQL query."
        ),
        "assistant": (
            "Step 1: This is a percentage calculation. I need (matching / total) * 100.\n"
            "Step 2: The evidence says to match job_desc IN ('Editor', 'Designer').\n"
            "Step 3: JOIN path. employee.job_id -> jobs.job_id to access job_desc.\n"
            "Step 4: Use CASE WHEN inside SUM for the numerator, COUNT for denominator. "
            "CAST as REAL to get decimal division.\n\n"
            "SELECT CAST(SUM(CASE WHEN T2.job_desc IN ('Editor', 'Designer') "
            "THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) "
            "FROM employee AS T1 "
            "INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id"
        ),
    },
    # Example 5: Multi-JOIN (4 tables) with ORDER BY + LIMIT
    {
        "user": (
            "Database Schema:\n"
            "CREATE TABLE price (\n"
            "  ID INTEGER [PRIMARY KEY],\n"
            "  price REAL\n"
            ");\n"
            "CREATE TABLE production (\n"
            "  ID INTEGER [FOREIGN KEY -> price.ID],\n"
            "  country INTEGER [FOREIGN KEY -> country.origin]\n"
            ");\n"
            "CREATE TABLE country (\n"
            "  origin INTEGER [PRIMARY KEY],\n"
            "  country TEXT\n"
            ");\n"
            "CREATE TABLE data (\n"
            "  ID INTEGER [PRIMARY KEY],\n"
            "  car_name TEXT\n"
            ");\n\n"
            "**IMPORTANT — Evidence (use this to interpret the question):**\n"
            "name of the car refers to car_name; the most expensive refers to "
            "max(price); produced by the USA refers to country = 'USA'\n\n"
            "Question: What is the name of the most expensive car that was "
            "produced by the USA?\n\n"
            "Think step by step, then write the SQL query."
        ),
        "assistant": (
            "Step 1: Identify needed columns. Return car_name from data table.\n"
            "Step 2: Filter. Evidence says country = 'USA' in country table.\n"
            "Step 3: Trace JOIN path across 4 tables:\n"
            "  price.ID -> production.ID (to link price with production)\n"
            "  production.country -> country.origin (to filter by USA)\n"
            "  data.ID -> price.ID (to get car_name)\n"
            "Step 4: \"Most expensive\" = ORDER BY price DESC LIMIT 1.\n\n"
            "SELECT T4.car_name "
            "FROM price AS T1 "
            "INNER JOIN production AS T2 ON T1.ID = T2.ID "
            "INNER JOIN country AS T3 ON T3.origin = T2.country "
            "INNER JOIN data AS T4 ON T4.ID = T1.ID "
            "WHERE T3.country = 'USA' "
            "ORDER BY T1.price DESC LIMIT 1"
        ),
    },
]


def build_query_messages(dataset: str, prompt: str) -> list[dict]:
    """
    Build chat messages for a dataset-specific query prompt.
    """
    if dataset == "spider":
        return [
            {"role": "system", "content": SPIDER_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    if dataset == "bird":
        # Few-shot messages followed by the actual query
        messages = []
        for ex in BIRD_FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    if dataset == "bird_json":
        # JSON structured output: forces reasoning before SQL
        # Replace the tail instructions to request JSON output
        json_prompt = prompt.replace(
            "7. Return ONLY the columns asked for — no extra columns, aliases, or formatting.\n"
            "8. Return ONLY the SQL query. No explanations, no markdown, no code blocks.\n\n"
            "SQL Query:",
            "7. Return ONLY the columns asked for — no extra columns, aliases, or formatting.\n\n"
            'Respond with a JSON object: {"reasoning": "<your step-by-step reasoning>", "sql": "<your SQL query>"}. '
            "No other text."
        )
        messages = []
        for ex in BIRD_JSON_FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": json_prompt})
        return messages

    if dataset == "bird_cot":
        # Chain-of-thought: few-shot with reasoning, then query
        # Swap the prompt tail to request step-by-step reasoning
        cot_prompt = prompt.replace(
            "8. Return ONLY the SQL query. No explanations, no markdown, no code blocks.\n\nSQL Query:",
            "8. Think step by step: identify columns, tables, JOIN paths, and filters before writing SQL.\n\nThink step by step, then write the SQL query."
        )
        messages = []
        for ex in BIRD_COT_FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": cot_prompt})
        return messages

    # Fallback: user-only prompt
    return [{"role": "user", "content": prompt}]
