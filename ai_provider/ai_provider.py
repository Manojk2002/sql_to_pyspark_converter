"""
AI LLM Client — Multi-provider AI integration for SQL-to-PySpark SQL conversion.

Converts SQL Stored Procedures & SQL Queries to PySpark SQL using the
7-step AI Framework Pipeline (Analyse → Map → Rewrite → Procedural →
Transactions → Optimise → Validate).

Supported AI providers (set AI_PROVIDER in .env — auto-detected if omitted):
  anthropic   Claude 3.5 Haiku / Sonnet  (requires ANTHROPIC_API_KEY)
  openai      GPT-4.1-mini / GPT-4.1  (requires OPENAI_API_KEY)
  huggingface Free HuggingFace Inference API  (requires HF_TOKEN)
  gemini      Google Gemini 1.5 Flash free tier  (requires GOOGLE_API_KEY)
  ollama      Local Ollama server — completely free (no API key required)

Environment variables:
    AI_PROVIDER      : anthropic | openai | huggingface | gemini | ollama  (auto if unset)
    ANTHROPIC_API_KEY: Anthropic API key  (console.anthropic.com)
    ANTHROPIC_MODEL  : Claude model  (default: claude-3-5-haiku-20241022)
    OPENAI_API_KEY   : OpenAI API key
    OPENAI_MODEL     : OpenAI model  (default: gpt-4.1-mini)
    OPENAI_BASE_URL  : custom endpoint for Azure OpenAI or proxies
    HF_TOKEN         : HuggingFace token  (free at huggingface.co/settings/tokens)
    HF_MODEL         : HF model  (default: Qwen/Qwen2.5-Coder-7B-Instruct)
    GOOGLE_API_KEY   : Google AI Studio key  (free at aistudio.google.com)
    GEMINI_MODEL     : Gemini model  (default: gemini-1.5-flash)
    OLLAMA_BASE_URL  : Ollama server URL  (default: http://localhost:11434)
    OLLAMA_MODEL     : Ollama model  (default: codellama)
"""

import os
import re
import sys
import pathlib
import textwrap
import json
import urllib.request
import urllib.error

# ── Optional .env loading ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on OS environment variables


# ── Configuration ─────────────────────────────────────────────────────────────
_PROVIDER        = os.getenv("AI_PROVIDER", "").strip().lower()   # auto if empty
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
_OAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_OAI_BASE_URL    = os.getenv("OPENAI_BASE_URL")
_HF_MODEL        = os.getenv("HF_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
_GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "codellama")


# ── Provider detection ────────────────────────────────────────────────────────

def _detect_provider() -> str:
    """Return the best available provider based on configured env vars.

    Priority order (free-first):
      1. ollama      — local, completely free, no account needed
      2. huggingface — free tier (rate-limited)
      3. gemini      — free tier (1M tokens/month)
      4. openai      — paid
      5. anthropic   — paid
    Explicit AI_PROVIDER env var overrides auto-detection.
    """
    if _PROVIDER:
        return _PROVIDER
    # ── Free providers first ──────────────────────────────────────────────────
    try:
        req = urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=2)
        if req.status == 200:
            return "ollama"
    except Exception:
        pass
    if os.getenv("HF_TOKEN"):
        return "huggingface"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    # ── Paid providers (fallback only) ────────────────────────────────────────
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "ollama"   # default target even if not yet reachable


def is_available() -> bool:
    """Return True if at least one AI provider is configured and its package available."""
    provider = _detect_provider()
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False
    if provider == "huggingface":
        return bool(os.getenv("HF_TOKEN"))
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            return False
        try:
            import google.genai  # noqa: F401
            return True
        except ImportError:
            return False
    if provider == "ollama":
        # Check if local Ollama server is reachable
        try:
            req = urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=2)
            return req.status == 200
        except Exception:
            return False
    return False


def get_provider_info() -> dict:
    """Return a dict describing the active provider and model."""
    provider = _detect_provider()
    model_map = {
        "anthropic":   _ANTHROPIC_MODEL,
        "openai":      _OAI_MODEL,
        "huggingface": _HF_MODEL,
        "gemini":      _GEMINI_MODEL,
        "ollama":      _OLLAMA_MODEL,
    }
    return {
        "provider": provider,
        "model":    model_map.get(provider, "unknown"),
        "available": is_available(),
    }


# ── Output helpers ───────────────────────────────────────────────────────────

def _clean_ai_output(text: str) -> str:
    """Extract only the Python code from AI output.

    Models often wrap code in markdown fences and append prose explanations
    despite being told not to. This function:
      1. Finds the FIRST ```python / ``` / ~~~ fence block and returns its content.
      2. If no fence is found, strips any trailing prose that starts with a
         markdown heading (##, ###) or a numbered list that follows a blank line.
      3. Removes any stray opening/closing fence lines that survived.
    """
    text = text.strip()

    # ── Strategy 1: extract content of the FIRST code fence ──────────────────
    # Matches ```python, ```py, ``` or ~~~python / ~~~ (greedy=False so we get
    # only the first block even if the model emits multiple fences).
    fence_re = re.compile(
        r'(?:```(?:python|py)?|~~~(?:python|py)?)\s*\n(.*?)(?:\n```|\n~~~)',
        re.DOTALL,
    )
    m = fence_re.search(text)
    if m:
        return m.group(1).strip()

    # ── Strategy 2: no fence — strip explanation prose after the code ─────────
    # Explanation sections typically start with a blank line followed by ##/###
    # or a "---" separator or a numbered/bulleted list.
    lines = text.splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##") or stripped.startswith("---") or stripped == "":
            # Check if the next non-empty line is a prose heading / list
            rest = "\n".join(lines[i:]).lstrip()
            if re.match(r'^(##|---|[0-9]+\.|[-*]\s)', rest):
                cut = i
                break

    text = "\n".join(lines[:cut]).strip()

    # ── Strategy 3: drop any bare fence markers that leaked through ───────────
    clean_lines = []
    for line in text.splitlines():
        if re.match(r'^```|^~~~', line.strip()):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


# ── Provider implementations ──────────────────────────────────────────────────

def _chat_openai(system_prompt: str, user_prompt: str, temperature: float) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    kwargs: dict = {"api_key": api_key}
    if _OAI_BASE_URL:
        kwargs["base_url"] = _OAI_BASE_URL
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=_OAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _chat_huggingface(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call HuggingFace Serverless Inference API (free tier)."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Get a free token at huggingface.co/settings/tokens")

    # Try huggingface_hub InferenceClient first (cleaner API)
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=_HF_MODEL, token=token)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=max(temperature, 0.01),
            max_tokens=4096,
        )
        return (response.choices[0].message.content or "").strip()
    except ImportError:
        pass  # fall through to requests-based call

    # Fallback: direct HTTP call (no extra package needed)
    api_url = f"https://api-inference.huggingface.co/models/{_HF_MODEL}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": max(temperature, 0.01),
        "max_tokens": 4096,
    }).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HuggingFace API error {exc.code}: {body}") from exc


def _chat_gemini(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call Google Gemini API (free 1 M tokens/month via AI Studio)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Get a free key at aistudio.google.com")

    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=4096,
            ),
        )
        return (response.text or "").strip()
    except ImportError:
        pass  # fall through to REST call

    # Fallback: direct REST call (no package needed)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {exc.code}: {body}") from exc


def _chat_ollama(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call local Ollama server (completely free, runs on your machine)."""
    url = f"{_OLLAMA_URL}/api/chat"
    payload = json.dumps({
        "model": _OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature":    0.05,  # near-deterministic → consistent, faster
            "top_k":          10,   # narrow token pool → faster sampling
            "top_p":          0.9,
            "num_predict":    800,  # cap output tokens (800 ≈ 400 lines of code)
            "num_ctx":        2048, # halved context → 2× faster attention
            "repeat_penalty": 1.1, # avoid repetitive output
        },
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message", {}).get("content") or "").strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {_OLLAMA_URL}. "
            "Install Ollama from ollama.com and run: ollama pull codellama"
        ) from exc


def _stream_ollama(system_prompt: str, user_prompt: str, temperature: float):
    """Stream tokens from Ollama one chunk at a time (generator).

    Yields str chunks as Ollama produces them.
    Raises RuntimeError if Ollama is not reachable.
    """
    url = f"{_OLLAMA_URL}/api/chat"
    payload = json.dumps({
        "model": _OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature":    temperature,
            "top_k":          10,
            "top_p":          0.9,
            "num_predict":    800,
            "num_ctx":        2048,
            "repeat_penalty": 1.1,
        },
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    yield token
                if obj.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {_OLLAMA_URL}. "
            "Start Ollama from the system tray or run: ollama serve"
        ) from exc


def _chat_anthropic(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call Anthropic Claude API using direct HTTP (no extra package required)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Get a key at console.anthropic.com"
        )
    payload = json.dumps({
        "model": _ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data["content"][0]["text"] or "").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error {exc.code}: {body}") from exc


def _chat(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Dispatch chat completion to the active AI provider."""
    provider = _detect_provider()
    if provider == "ollama":
        return _chat_ollama(system_prompt, user_prompt, temperature)
    if provider == "huggingface":
        return _chat_huggingface(system_prompt, user_prompt, temperature)
    if provider == "gemini":
        return _chat_gemini(system_prompt, user_prompt, temperature)
    if provider == "openai":
        return _chat_openai(system_prompt, user_prompt, temperature)
    if provider == "anthropic":
        return _chat_anthropic(system_prompt, user_prompt, temperature)
    raise RuntimeError(f"Unknown AI_PROVIDER: '{provider}'.")


# ── System prompts ────────────────────────────────────────────────────────────

_CONVERT_SYSTEM = textwrap.dedent("""\
    You are an expert SQL-to-PySpark SQL converter. Output ONLY valid Python code. No markdown fences, no explanations.

    CORE RULES — apply to EVERY conversion:
    1. First line must be an import or docstring. Never output backticks or code fences.
    2. Use spark.sql("...") for ALL SQL statements — SELECT, INSERT, UPDATE, DELETE, MERGE, CTEs, DDL.
       Never use DataFrame API chains (.join(), .filter(), .groupBy(), .select()).
    3. Temp tables: #t → spark.sql("CREATE OR REPLACE TEMP VIEW t AS SELECT ...")
    4. Stored procedure params → typed Python function args (e.g. emp_id: int = None).
    5. IF/ELSE control flow → Python if/elif/else blocks with spark.sql() calls inside.
    6. WHILE loops → Python while loops with spark.sql() inside; never loop over .collect().
    7. CURSORS → replace with a single bulk spark.sql() operation:
         Row-by-row UPDATE → spark.sql("UPDATE db.t SET col=val WHERE cond")
         Row-by-row INSERT → spark.sql("INSERT INTO db.t SELECT ... FROM ...")
         Row-by-row DELETE → spark.sql("DELETE FROM db.t WHERE cond")
         Complex cursor → spark.sql(...).foreachPartition(fn)  # only as last resort
         Always add: # NOTE: SQL cursor replaced with bulk Spark SQL operation
    8. Transactions (BEGIN/COMMIT/ROLLBACK) → remove; add # NOTE: Transaction replaced by Delta ACID.
    9. Function naming:
        - Stored procedure (CREATE PROCEDURE usp_X) → keep the SP name: def usp_x(spark, ...)
        - Plain SQL query (SELECT/INSERT/etc., no CREATE PROCEDURE) → descriptive name
          WITHOUT usp_/sp_ prefix: def get_employees(spark) NOT def usp_get_employees(spark)
    10. Wrap all logic in: def <name>(spark: SparkSession, <params>) -> DataFrame:
    11. Do NOT add .show() inside the function body. Production code never displays data.
    12. Return result_df (last meaningful DataFrame).
    13. Only import logging and define logger if you actually call logger.info()/logger.warning().
        For simple queries with no logging calls, omit logging entirely.

    PLAIN QUERY EXAMPLE:
    Input:  SELECT emp_id, name FROM employees WHERE dept_id = 10
    Output:
        from pyspark.sql import SparkSession, DataFrame
        def get_employees(spark: SparkSession) -> DataFrame:
            result_df = spark.sql("SELECT emp_id, name FROM employees WHERE dept_id = 10")
            return result_df

    STORED PROCEDURE EXAMPLE:
        from pyspark.sql import SparkSession, DataFrame
        import logging
        logger = logging.getLogger(__name__)

        def usp_example(spark: SparkSession, dept_id: int = None) -> DataFrame:
            spark.sql("CREATE OR REPLACE TEMP VIEW active_emps AS SELECT emp_id, name FROM emp WHERE active = 1")
            if dept_id is not None:
                result_df = spark.sql(f"SELECT * FROM active_emps WHERE dept_id = {dept_id}")
            else:
                result_df = spark.sql("SELECT * FROM active_emps")
            logger.info("Query complete")
            return result_df

    Output ONLY the Python code — nothing else.
""")

_EXPLAIN_SYSTEM = textwrap.dedent("""\
    You are a PySpark and Databricks expert and technical writer.
    Your task: add clear, concise inline comments to existing PySpark code.

    Rules:
      • Add a comment above each logical block explaining WHAT it does and WHY.
      • Explain Databricks/Spark-specific optimisations (broadcast, cache, Delta, etc.).
      • If a section relates to one of the 7 conversion steps, note it: # STEP N: ...
      • Keep ALL existing code lines exactly as-is — only ADD comment lines.
      • Do NOT wrap the output in Markdown fences.
      • If an operation could be further optimised, add:  # OPTIMISATION NOTE: ...
""")

_OPTIMIZE_SYSTEM = textwrap.dedent("""\
    You are a Databricks performance optimisation expert.
    Your task: optimise the given PySpark code for production Spark/Databricks workloads.

    Apply ALL applicable optimisations from the list below:
      1. Broadcast joins: wrap small lookup/dimension DataFrames in F.broadcast().
      2. Partition pruning: add .filter() on partition columns before joins.
      3. Vectorisation: replace any Python for-loop over rows with withColumn/UDF.
      4. Caching: call .cache() on DataFrames that are accessed more than once;
         add .unpersist() after the last use.
      5. Delta output: replace .write.parquet() or .write.csv() with
         .write.format("delta").saveAsTable("db.table").
      6. AQE: already enabled on Databricks — remove manual broadcast threshold hints.
      7. Avoid collect() on large DataFrames; use .show() or write to Delta instead.
      8. Replace Python UDFs with equivalent built-in F. functions wherever possible.
      9. Add Z-ORDER hint comments for large Delta tables: # HINT: OPTIMIZE table ZORDER BY (col)
      10. Add a performance notes section at the bottom as a comment block.

    Return the fully optimised Python source code. Do NOT include Markdown fences.
""")

_ANALYZE_SYSTEM = textwrap.dedent("""\
    You are a SQL code analyst specialising in stored procedure migration to Databricks.
    Your task: perform Step 1 of the 7-step AI framework — deep SQL analysis.

    Produce a structured analysis report in plain text with these sections:

    ## OVERVIEW
    Brief description of what the SQL does.

    ## DDL STATEMENTS
    List every CREATE TABLE, ALTER TABLE, schema definition.

    ## DML STATEMENTS
    List every SELECT, INSERT, UPDATE, DELETE, MERGE with a brief description.

    ## CONTROL FLOW
    List every IF/ELSE, WHILE loop, cursor, EXEC/dynamic SQL block.

    ## DEPENDENCIES
    Tables, views, functions, linked servers, external references.

    ## CONVERSION COMPLEXITY
    Score 1-10 and explain the main complexity drivers.

    ## CONVERSION WARNINGS
    Flag anything that requires manual review (cursors, dynamic SQL, XML, etc.).

    ## SPARK OBJECT MAPPING (Step 2 preview)
    Quick mapping table: SQL Object → Spark Equivalent.

    Be thorough but concise. Use bullet points.
""")


# ── Public API ────────────────────────────────────────────────────────────────

def convert_sql_with_ai(
    sql: str,
    db_prefix: str = "my_db",
    dialect: str = "T-SQL",
) -> str:
    """
    Convert SQL to PySpark SQL using the 3-stage pipeline:

      Stage 1 — Preprocess  : core/preprocessor.py
        Clean T-SQL noise (GO, USE, brackets), extract SP name / params / hints.

      Stage 2 — LLM          : Ollama / other provider
        Send cleaned SQL with a structured prompt enriched by metadata.

      Stage 3 — Postprocess  : core/postprocessor.py
        Strip fences, validate Python AST, inject missing imports, dedup.

    Args:
        sql:       Raw SQL text (T-SQL / ANSI SQL / PL/pgSQL / etc.)
        db_prefix: Databricks catalog/database prefix for permanent tables
        dialect:   SQL dialect hint (e.g. 'T-SQL', 'PostgreSQL', 'ANSI')

    Returns:
        Generated PySpark SQL Python source code as a string.

    Raises:
        RuntimeError: if no AI provider is configured or the API call fails.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    # Stage 1: Preprocess
    from converter.sql_preprocessor import preprocess
    pre = preprocess(sql)

    # Build hint annotations for the prompt
    hint_lines = ""
    if pre.is_stored_procedure:
        hint_lines += f"  - Stored procedure name : {pre.sp_name}\n"
    if pre.parameters:
        params_str = ", ".join(
            f"{p['name']} {p['type']}" + (f" = {p['default']}" if p['default'] else "")
            for p in pre.parameters
        )
        hint_lines += f"  - Parameters            : {params_str}\n"
    if pre.temp_tables:
        hint_lines += f"  - Temp tables detected  : {', '.join(pre.temp_tables)}\n"
    if pre.dialect_hints:
        hint_lines += f"  - Complexity flags      : {', '.join(pre.dialect_hints)}\n"

    context_block = f"\nSQL Context:\n{hint_lines}" if hint_lines else ""

    user_prompt = textwrap.dedent(f"""\
        Database prefix for permanent tables: "{db_prefix}"
        {context_block}
        SQL to convert:
        ---
        {pre.cleaned_sql}
        ---
    """)

    # ── Stage 2: LLM ────────────────────────────────────────────────────────
    raw_output = _chat(_CONVERT_SYSTEM, user_prompt)

    # ── Stage 3: Postprocess ─────────────────────────────────────────────────
    from converter.code_postprocessor import postprocess
    result = postprocess(raw_output)

    # Return the validated code; if syntax is invalid, still return best-effort
    return result.code


def stream_sql_with_ai(
    sql: str,
    db_prefix: str = "my_db",
    dialect: str = "T-SQL",
):
    """
    Stream SQL → PySpark SQL conversion token-by-token (generator).

    Only supported for the Ollama provider (local streaming).
    Falls back to a single-chunk yield for all other providers.

    Yields:
        str chunks as the model generates them.
    """
    user_prompt = textwrap.dedent(f"""\
        Convert the following {dialect} SQL to PySpark SQL.
        Database prefix: "{db_prefix}"

        SQL Input:
        ---
        {sql}
        ---

        Requirements:
        - Use spark.sql("...") for ALL SQL queries (PySpark SQL format).
        - Wrap logic in a typed Python function with spark: SparkSession as first arg.
        - Cursors → single batch spark.sql() UPDATE/INSERT (never row-by-row).
        - Temp tables → spark.sql("CREATE OR REPLACE TEMP VIEW ...").
        - Transactions → Delta Lake ACID (add comment, no BEGIN/COMMIT).
        - Output ONLY runnable Python code — no Markdown fences.
    """)
    provider = _detect_provider()
    if provider == "ollama":
        yield from _stream_ollama(_CONVERT_SYSTEM, user_prompt, temperature=0.2)
    else:
        # Non-streaming fallback: yield the full result as one chunk
        yield _clean_ai_output(_chat(_CONVERT_SYSTEM, user_prompt))


def analyze_sql_with_ai(sql: str, dialect: str = "T-SQL") -> str:
    """
    Deep SQL analysis (Step 1 of the 7-step framework) using the active AI provider.

    Returns a structured analysis report covering DDL, DML, control flow,
    dependencies, complexity, and conversion warnings.

    Args:
        sql:     Raw SQL text.
        dialect: SQL dialect hint.

    Returns:
        Structured analysis report as a string.
    """
    user_prompt = textwrap.dedent(f"""\
        Perform a deep analysis of this {dialect} SQL following the 7-step framework Step 1.

        SQL Input:
        ---
        {sql}
        ---

        Produce the full structured analysis report.
    """)
    return _clean_ai_output(_chat(_ANALYZE_SYSTEM, user_prompt, temperature=0.1))


def explain_pyspark_code(code: str) -> str:
    """
    Add inline comments/explanations to existing PySpark SQL code using the active AI provider.

    Args:
        code: PySpark Python source code.

    Returns:
        The same code with detailed inline comments added.
    """
    user_prompt = (
        "Add detailed inline comments to this PySpark code, "
        "noting which of the 7 conversion steps each block implements:\n\n"
        + code
    )
    return _clean_ai_output(_chat(_EXPLAIN_SYSTEM, user_prompt))


def optimize_pyspark_code(code: str) -> str:
    """
    Optimise PySpark SQL code for Spark/Databricks performance (Step 6) using the active AI provider.

    Applies: broadcast joins, partition pruning, caching, Delta output,
    vectorisation, Z-ORDER hints, and AQE best practices.

    Args:
        code: PySpark Python source code.

    Returns:
        The optimised PySpark Python source code.
    """
    user_prompt = (
        "Optimise this PySpark code for Databricks following the 10-point "
        "optimisation checklist:\n\n"
        + code
    )
    return _clean_ai_output(_chat(_OPTIMIZE_SYSTEM, user_prompt))
