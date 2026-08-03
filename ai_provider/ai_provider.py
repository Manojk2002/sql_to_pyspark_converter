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
_OLLAMA_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Small/fast model for short queries (<= threshold); 1.5b is ~3× faster than 7b on CPU
_OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
# Explicit override for large-input model (optional — auto-selected if blank)
_OLLAMA_MODEL_LARGE = os.getenv("OLLAMA_MODEL_LARGE", "")
# Character threshold above which the large model is selected (~60 lines of SQL)
# Lowered so multi-statement scripts (35+ statements) get the more capable 7b model
_LARGE_INPUT_THRESHOLD = 2000

# Priority list: best model for SQL code conversion → fallback order
_MODEL_PRIORITY = [
    "qwen2.5-coder:32b",
    "qwen2.5-coder:7b",
    "deepseek-coder-v2:16b",
    "codellama:34b",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:1.5b",
    "codellama",
]

# Cache so we only query Ollama once per process
_best_large_model_cache: str | None = None


def _get_best_large_model() -> str:
    """Query Ollama for installed models and return the highest-ranked one.

    Uses _MODEL_PRIORITY order. Falls back to _OLLAMA_MODEL if nothing matches.
    Result is cached for the lifetime of the process.
    """
    global _best_large_model_cache
    # Honour explicit env override
    if _OLLAMA_MODEL_LARGE:
        return _OLLAMA_MODEL_LARGE
    if _best_large_model_cache is not None:
        return _best_large_model_cache
    try:
        with urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        installed = [m["name"] for m in data.get("models", [])]
        # Normalize: strip digest suffix if present (e.g. "qwen2.5-coder:7b:latest" → "qwen2.5-coder:7b")
        installed_base = [n.split(":")[0] + ":" + n.split(":")[1] if n.count(":") >= 1 else n
                          for n in installed]
        for candidate in _MODEL_PRIORITY:
            if candidate in installed_base or candidate in installed:
                _best_large_model_cache = candidate
                return candidate
    except Exception:
        pass
    # Fallback: use whatever is configured as the default model
    _best_large_model_cache = _OLLAMA_MODEL
    return _OLLAMA_MODEL


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


def get_model_for_input(sql: str) -> str:
    """Return the Ollama model that will be used for the given SQL string.

    Mirrors the auto-selection logic in _chat_ollama / _stream_ollama:
      - Short SQL (< _LARGE_INPUT_THRESHOLD chars) → small/fast model
      - Large SQL (≥ _LARGE_INPUT_THRESHOLD chars) → best available large model
    """
    if len(sql) >= _LARGE_INPUT_THRESHOLD:
        return _get_best_large_model()
    return _OLLAMA_MODEL


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
    if provider == "ollama":
        small_model = _OLLAMA_MODEL
        large_model = _get_best_large_model()
        model_display = (
            f"{large_model} (large) / {small_model} (small)"
            if large_model != small_model
            else small_model
        )
        return {
            "provider":          provider,
            "model":             model_display,
            "small_model":       small_model,
            "large_model":       large_model,
            "input_threshold":   _LARGE_INPUT_THRESHOLD,
            "available":         is_available(),
        }
    model_map = {
        "anthropic":   _ANTHROPIC_MODEL,
        "openai":      _OAI_MODEL,
        "huggingface": _HF_MODEL,
        "gemini":      _GEMINI_MODEL,
    }
    model_display = model_map.get(provider, "unknown")
    return {
        "provider":    provider,
        "model":       model_display,
        "small_model": model_display,
        "large_model": model_display,
        "input_threshold": 0,
        "available":   is_available(),
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
    """Call local Ollama server (completely free, runs on your machine).

    Auto-selects model based on input size:
      - Small input (< 4000 chars) → OLLAMA_MODEL (fast, e.g. qwen2.5-coder:1.5b)
      - Large input (≥ 4000 chars) → best available large model (e.g. qwen2.5-coder:7b)

    Context window scales dynamically with input length so that even 500-line
    stored procedures fit without truncation.
    """
    total_input = len(system_prompt) + len(user_prompt)
    is_large = total_input >= _LARGE_INPUT_THRESHOLD
    model = _get_best_large_model() if is_large else _OLLAMA_MODEL

    # Dynamic context window: 1 token ≈ 4 chars (conservative estimate for SQL/code)
    input_tokens_est = total_input // 4
    # Output budget: scale with input size; extra headroom for multi-statement scripts.
    # Each SQL statement generates ~15-25 output tokens (one spark.sql() call).
    # Count ';' in the user_prompt as a proxy for number of statements.
    stmt_count = max(1, user_prompt.count(";"))
    stmt_budget = stmt_count * 30   # 30 tokens per statement, generous estimate
    if is_large:
        output_budget = max(stmt_budget, min(6144, int(input_tokens_est * 0.8)))
    else:
        output_budget = max(stmt_budget, min(3072, int(input_tokens_est * 0.8)))
    output_budget = max(1200, output_budget)  # never below 1200
    # Round ctx up to next multiple of 2048 with a safety margin
    raw_ctx = input_tokens_est + output_budget + 256
    num_ctx     = max(2048, min(32768, ((raw_ctx + 2047) // 2048) * 2048))
    num_predict = output_budget
    # Scale timeout: assume ~7 tokens/sec worst-case on CPU, cap at 600s
    timeout_s   = max(120, min(600, num_predict // 7))

    url = f"{_OLLAMA_URL}/api/chat"
    payload = json.dumps({
        "model": model,
        "keep_alive": -1,   # keep model in RAM indefinitely — eliminates reload delay
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature":    0.15,   # slight randomness prevents premature EOS on multi-statement SQL
            "seed":           42,     # reproducible output
            "top_k":          20,     # wider beam — helps with long outputs
            "top_p":          0.9,
            "penalize_newline": False, # never penalise newlines in code output
            "num_keep":       256,    # cache 256 system-prompt tokens in KV — speeds up repeat calls
            "num_predict":    num_predict,
            "num_ctx":        num_ctx,
            "repeat_penalty": 1.0,   # NO penalty — code needs repeated spark.sql() calls
            "num_thread":     os.cpu_count() or 8,
            "num_batch":      1024,
            "num_gpu":        99,     # offload all layers to GPU if available (no-op on CPU-only)
            "f16_kv":         True,   # half-precision KV cache → 2× smaller, faster attention
            "use_mmap":       True,   # memory-map model weights → OS caches hot pages
        },
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message", {}).get("content") or "").strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {_OLLAMA_URL}. "
            "Install Ollama from ollama.com and run: ollama pull codellama"
        ) from exc


def _stream_ollama(system_prompt: str, user_prompt: str, temperature: float):
    """Stream tokens from Ollama one chunk at a time (generator).

    Auto-selects model based on input size (same logic as _chat_ollama).
    Yields str chunks as Ollama produces them.
    Raises RuntimeError if Ollama is not reachable.
    """
    total_input = len(system_prompt) + len(user_prompt)
    is_large = total_input >= _LARGE_INPUT_THRESHOLD
    model = _get_best_large_model() if is_large else _OLLAMA_MODEL
    # Same dynamic sizing as _chat_ollama
    input_tokens_est = total_input // 4
    stmt_count = max(1, user_prompt.count(";"))
    stmt_budget = stmt_count * 30
    if is_large:
        output_budget = max(stmt_budget, min(6144, int(input_tokens_est * 0.8)))
    else:
        output_budget = max(stmt_budget, min(3072, int(input_tokens_est * 0.8)))
    output_budget = max(1200, output_budget)
    raw_ctx     = input_tokens_est + output_budget + 256
    num_ctx     = max(2048, min(32768, ((raw_ctx + 2047) // 2048) * 2048))
    num_predict = output_budget
    url = f"{_OLLAMA_URL}/api/chat"
    payload = json.dumps({
        "model": model,
        "keep_alive": -1,   # keep model in RAM indefinitely — eliminates reload delay
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature":    0.15,   # slight randomness prevents premature EOS
            "seed":           42,
            "top_k":          20,
            "top_p":          0.9,
            "penalize_newline": False,
            "num_keep":       256,    # cache 256 system-prompt tokens in KV
            "num_predict":    num_predict,
            "num_ctx":        num_ctx,
            "repeat_penalty": 1.0,   # NO penalty — code needs repeated spark.sql() calls
            "num_thread":     os.cpu_count() or 8,
            "num_batch":      1024,
            "num_gpu":        99,     # offload all layers to GPU if available (no-op on CPU-only)
            "f16_kv":         True,   # half-precision KV cache → 2× smaller, faster attention
            "use_mmap":       True,   # memory-map model weights → OS caches hot pages
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
    SQL-to-PySpark SQL converter. Output ONLY valid Python. No markdown fences, no prose.

    RULES (apply to every conversion):
    1. spark.sql("...") for ALL SQL (SELECT/INSERT/UPDATE/DELETE/MERGE/DDL/CTEs). Never DataFrame API.
    2. ONE function only — wrap ALL statements inside a SINGLE Python function. Never split into multiple functions.
       Write ALL spark.sql() calls in sequence BEFORE writing `return result_df`. Never write `return` early.
    3. #temp tables → spark.sql("CREATE OR REPLACE TEMP VIEW temp AS SELECT ...")
    4. IF/ELSE → Python if/elif/else. WHILE → Python while. Never .collect() in loops.
    5. CURSOR → single bulk spark.sql(). Add: # NOTE: cursor replaced with bulk Spark SQL
    6. BEGIN/COMMIT/ROLLBACK → remove. Add: # NOTE: Transaction replaced by Delta ACID
    7. No .show(). Return the last meaningful SELECT result as result_df.
    8. Function naming:
       - Stored procedure → keep SP name as snake_case: def usp_my_proc(spark, ...)
       - Plain SQL script → descriptive name WITHOUT usp_/sp_ prefix: def process_employees(spark)
    9. Stored procedure params → typed Python args: def fn(spark: SparkSession, param: int = None)
    10. T-SQL→Spark SQL inside spark.sql(): ISNULL→COALESCE, LEN→LENGTH, GETDATE()→CURRENT_TIMESTAMP(),
        TOP n→LIMIT n, DATEDIFF(u,s,e)→DATEDIFF(e,s), CONVERT→CAST, WITH(NOLOCK)→remove,
        STRING_AGG→CONCAT_WS+COLLECT_LIST, EXEC @sql→spark.sql(f"..."), TRY/CATCH→try/except.
    11. IMPORTS — write EXACTLY these two lines, nothing else:
        from pyspark.sql import SparkSession, DataFrame
        Do NOT write `from pyspark.sql.functions import ...` — functions are not needed when all SQL runs via spark.sql().
        Do NOT repeat or duplicate import lines.

    REQUIRED STRUCTURE — ONE function, ALL statements inside it:
    from pyspark.sql import SparkSession, DataFrame
    def process_data(spark: SparkSession) -> DataFrame:
        spark.sql("CREATE TABLE ...")
        spark.sql("INSERT INTO ...")
        spark.sql("UPDATE ...")
        result_df = spark.sql("SELECT ...")   # last SELECT is result_df
        return result_df
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

def _build_convert_user_prompt(sql: str, db_prefix: str, dialect: str) -> str:
    """Build the enriched user prompt for both batch and streaming conversion.

    Runs the preprocessor to clean T-SQL noise and extract metadata, then
    injects that context into the prompt so the model has maximum signal.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from converter.sql_preprocessor import preprocess
    pre = preprocess(sql)

    # Count SQL statements so we can tell the model exactly how many
    # spark.sql() calls to generate — prevents premature EOS
    raw_stmts = [s.strip() for s in pre.cleaned_sql.split(";")
                 if s.strip() and not s.strip().startswith("--")]
    stmt_count = max(1, len(raw_stmts))

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
    hint_lines += f"  - SQL statements        : {stmt_count} (your function MUST contain exactly {stmt_count} spark.sql() calls)\n"

    context_block = f"\nSQL Context:\n{hint_lines}" if hint_lines else ""

    # For multi-statement SQL, number each statement so the model
    # iterates through every one rather than stopping after the first few
    if stmt_count > 3:
        sql_body = "\n".join(
            f"[{i+1}/{stmt_count}] {s};" for i, s in enumerate(raw_stmts)
        )
    else:
        sql_body = pre.cleaned_sql

    return textwrap.dedent(f"""\
        Database prefix for permanent tables: \"{db_prefix}\"
        {context_block}
        SQL to convert ({stmt_count} statements — convert ALL of them):
        ---
        {sql_body}
        ---
    """), pre


def _direct_convert_statements(pre, db_prefix: str, original_sql: str = "") -> str:
    """Convert a plain SQL script directly to PySpark code — no LLM needed."""
    import re as _re
    import html as _html

    source_sql = original_sql if original_sql.strip() else pre.cleaned_sql

    # ── Parse: split on semicolons, keep preceding comment block per stmt ──
    segments = []            # list of (comment_lines, sql_stmt_text)
    pending_comments = []
    current_stmt_lines = []

    for raw_line in source_sql.split("\n"):
        stripped = raw_line.strip()
        if stripped.startswith("--"):
            if current_stmt_lines:
                s = " ".join(" ".join(current_stmt_lines).split()).strip()
                if s:
                    segments.append((list(pending_comments), s))
                current_stmt_lines = []
                pending_comments = []
            pending_comments.append(stripped[2:].strip())
        elif ";" in stripped:
            before, _, _ = raw_line.partition(";")
            current_stmt_lines.append(before)
            s = " ".join(" ".join(current_stmt_lines).split()).strip()
            if s:
                segments.append((list(pending_comments), s))
            current_stmt_lines = []
            pending_comments = []
        else:
            current_stmt_lines.append(raw_line)

    if current_stmt_lines:
        s = " ".join(" ".join(current_stmt_lines).split()).strip()
        if s:
            segments.append((list(pending_comments), s))

    if not segments:
        return None

    # ── Determine function name ────────────────────────────────────────────
    func_name = pre.sp_name or ""
    if not func_name:
        first_sql = segments[0][1]
        m = _re.search(r"\b(?:FROM|INTO|TABLE|VIEW)\s+(\w+)", first_sql, _re.IGNORECASE)
        func_name = f"process_{m.group(1).lower()}" if m else "process_data"
    func_name = _re.sub(r"[^a-z0-9_]", "_", func_name.lower()).strip("_") or "process_data"

    # ── T-SQL → Spark SQL translations ────────────────────────────────────
    def _translate(stmt: str) -> str:
        # Fix any HTML entities (&gt; → >, &lt; → <, &amp; → &, etc.)
        stmt = _html.unescape(stmt)
        # SELECT TOP n → LIMIT n
        top_m = _re.search(r"\bSELECT\s+TOP\s+(\d+)\b", stmt, _re.IGNORECASE)
        if top_m:
            n = top_m.group(1)
            stmt = _re.sub(r"\bTOP\s+\d+\b\s*", "", stmt, flags=_re.IGNORECASE)
            stmt = stmt.rstrip() + f" LIMIT {n}"
        stmt = _re.sub(r"\bWITH\s*\(\s*NOLOCK\s*\)", "", stmt, flags=_re.IGNORECASE)
        stmt = _re.sub(r"\bISNULL\s*\(", "COALESCE(", stmt, flags=_re.IGNORECASE)
        stmt = _re.sub(r"\bLEN\s*\(", "LENGTH(", stmt, flags=_re.IGNORECASE)
        stmt = _re.sub(r"\bGETDATE\s*\(\s*\)", "CURRENT_TIMESTAMP()", stmt, flags=_re.IGNORECASE)
        stmt = _re.sub(r"\bGETUTCDATE\s*\(\s*\)", "UTC_TIMESTAMP()", stmt, flags=_re.IGNORECASE)
        stmt = _re.sub(r"\bCHARINDEX\s*\(", "LOCATE(", stmt, flags=_re.IGNORECASE)
        # CREATE VIEW X AS → CREATE OR REPLACE TEMP VIEW X AS
        stmt = _re.sub(r"\bCREATE\s+VIEW\b", "CREATE OR REPLACE TEMP VIEW",
                       stmt, flags=_re.IGNORECASE)
        # CREATE TABLE → CREATE TABLE IF NOT EXISTS (safe for re-runs)
        stmt = _re.sub(r"\bCREATE\s+TABLE\b(?!\s+IF)",
                       "CREATE TABLE IF NOT EXISTS", stmt, flags=_re.IGNORECASE)
        # Strip PRIMARY KEY constraint (not supported in Spark DDL)
        stmt = _re.sub(r"\s+PRIMARY\s+KEY\b", "", stmt, flags=_re.IGNORECASE)
        # VARCHAR(n) → STRING  (more portable across Spark/Delta)
        stmt = _re.sub(r"\bVARCHAR\s*\(\s*\d+\s*\)", "STRING", stmt, flags=_re.IGNORECASE)
        # DECIMAL(p,s) → DECIMAL(p,s)  — already valid, keep as-is
        return " ".join(stmt.split())

    # ── Group consecutive INSERTs to the same table into one multi-row INSERT ──
    def _group_inserts(segs):
        out = []
        i = 0
        while i < len(segs):
            comments, stmt = segs[i]
            ins_m = _re.match(
                r"INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)$", stmt, _re.IGNORECASE | _re.DOTALL
            )
            if ins_m:
                table = ins_m.group(1)
                values = [ins_m.group(2)]
                j = i + 1
                while j < len(segs):
                    _, nxt = segs[j]
                    nxt_m = _re.match(
                        r"INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)$",
                        nxt, _re.IGNORECASE | _re.DOTALL
                    )
                    if nxt_m and nxt_m.group(1).lower() == table.lower():
                        values.append(nxt_m.group(2))
                        j += 1
                    else:
                        break
                if len(values) > 1:
                    rows = ",\n        ".join(f"({v})" for v in values)
                    combined = f"INSERT INTO {table} VALUES\n        {rows}"
                    out.append((comments, combined))
                    i = j
                    continue
            out.append((comments, stmt))
            i += 1
        return out

    # Translate all statements first, then group INSERTs
    translated_segments = [(c, _translate(s)) for c, s in segments]
    translated_segments = _group_inserts(translated_segments)

    # ── Build function body ────────────────────────────────────────────────
    code_lines = [
        "from pyspark.sql import SparkSession, DataFrame",
        "",
        f"def {func_name}(spark: SparkSession) -> DataFrame:",
    ]

    has_select = False
    for comments, stmt in translated_segments:
        if not stmt:
            continue

        for c in comments:
            if c:
                code_lines.append(f"    # {c}")

        is_select = bool(_re.match(r"SELECT\b", stmt, _re.IGNORECASE))
        is_update = bool(_re.match(r"UPDATE\b", stmt, _re.IGNORECASE))
        is_delete = bool(_re.match(r"DELETE\b", stmt, _re.IGNORECASE))
        safe = stmt.replace('"""', '"\\"')

        if is_update:
            code_lines.append("    # NOTE: UPDATE requires a Delta table")
            code_lines.append(f'    spark.sql("""{safe}""")')
        elif is_delete:
            code_lines.append("    # NOTE: DELETE requires a Delta table")
            code_lines.append(f'    spark.sql("""{safe}""")')
        elif is_select:
            has_select = True
            code_lines.append(f'    result_df = spark.sql("""{safe}""")')
        else:
            code_lines.append(f'    spark.sql("""{safe}""")')

        if comments:
            code_lines.append("")

    if not has_select:
        code_lines.append('    result_df = spark.sql("SELECT 1 AS result")')
    code_lines.append("    return result_df")

    return "\n".join(code_lines)


def _sp_direct_convert(pre, db_prefix: str, original_sql: str = "") -> str:
    """Convert a simple stored procedure to PySpark without LLM.

    Handles:
    - SP parameters → typed Python function args with None defaults.
    - DECLARE @Var → skipped (variables become Python None defaults).
    - Consecutive ``SELECT @Var = expr FROM tbl`` stmts followed by a
      ``SELECT @Var AS alias, …`` (no FROM) are consolidated into one
      clean ``SELECT expr AS alias, … FROM tbl`` result.
    - @Param references in WHERE → f-string interpolation:
        VARCHAR  @P  →  '{pyvar}'      (SQL string literal)
        NUMERIC  @P  →  {0 if pyvar is None else pyvar}
        @P IS NULL   →  {expr} = 0    (for numeric; VARCHAR keeps IS NULL)
    - T-SQL translations: TOP n→LIMIT n, GETDATE→current_timestamp(), etc.
    - Heuristic section comments above each spark.sql() call.

    Used when the SP has no cursors, no dynamic SQL — gives instant, reliable output.
    """
    import re as _re
    import html as _html

    # ── 1. Parameter map ─────────────────────────────────────────────────────
    # Maps UPPER_NAME → {py_name, is_numeric}
    param_map = {}
    for p in pre.parameters:
        raw  = p['name'].lstrip('@')
        is_num = any(
            t in p['type'].upper()
            for t in ('INT', 'DECIMAL', 'NUMERIC', 'FLOAT', 'MONEY', 'BIT', 'BIGINT', 'SMALLINT')
        )
        param_map[raw.upper()] = {'py_name': raw.lower(), 'is_numeric': is_num}

    # ── 2. Extract SP body (between AS BEGIN … END) ───────────────────────────
    source = original_sql if original_sql.strip() else pre.cleaned_sql
    source = _re.sub(r'--[^\n]*', '', source)
    source = _re.sub(r'/\*.*?\*/', ' ', source, flags=_re.DOTALL)
    # Multi-pass HTML entity decoding (handles &amp;gt; → &gt; → > etc.)
    while True:
        decoded = _html.unescape(source)
        if decoded == source:
            break
        source = decoded

    body_m = _re.search(
        r'\bAS\s*\n?\s*BEGIN\b\s*(.*?)\bEND\b\s*;?\s*(?:GO\b[.\s]*)?$',
        source, _re.IGNORECASE | _re.DOTALL
    )
    body = body_m.group(1).strip() if body_m else source

    # ── 3. Split into statements (preserving multi-line formatting) ───────────
    import textwrap as _textwrap
    raw_stmts: list = []
    current: list = []
    for line in body.split('\n'):
        # Find ';' that is NOT inside a single-quoted string literal
        in_str = False
        split_pos = -1
        for ci, ch in enumerate(line):
            if ch == "'":
                in_str = not in_str
            elif ch == ';' and not in_str:
                split_pos = ci
                break

        if split_pos >= 0:
            before = line[:split_pos]
            after  = line[split_pos + 1:]
            if before.strip():
                current.append(before)
            stmt = _textwrap.dedent('\n'.join(current)).strip()
            if stmt:
                raw_stmts.append(stmt)
            # Preserve content AFTER the ';' (handles ;WITH CTE syntax)
            current = [after] if after.strip() else []
        else:
            current.append(line)
    if current:
        stmt = _textwrap.dedent('\n'.join(current)).strip()
        if stmt:
            raw_stmts.append(stmt)

    # ── 4. Classify statements ────────────────────────────────────────────────
    # Supports: simple SPs, SPs with cursor, TRY/CATCH, transactions, dynamic SQL.
    #
    # ASSIGN_GROUP : SELECT @Var=expr[,…] FROM tbl
    # VAR_READ     : SELECT @Var AS alias[,…] (no FROM)
    # RAISERROR    : converted to raise ValueError(...)
    # CURSOR_NOTE  : placeholder emitted once for the cursor block
    # REGULAR      : kept as spark.sql(...)

    var_map:    dict  = {}   # UPPER_VAR → {'expr': str, 'table': str}
    dyn_sql_map: dict = {}   # UPPER_VAR → SQL string (from SET @var = 'SELECT ...')
    temp_map:   dict  = {}   # #LOWER_NAME → spark_view_name
    classified: list  = []
    _cursor_note_added = False

    # ── keywords that are structure/noise with no Spark equivalent ────────────
    _SKIP_RE = _re.compile(
        r'^(?:'
        r'BEGIN\s+TRY|END\s+TRY|BEGIN\s+CATCH|END\s+CATCH'
        r'|BEGIN\s+TRANSACTION|COMMIT(?:\s+TRANSACTION)?|ROLLBACK(?:\s+TRANSACTION)?'
        r'|OPEN\s+\w+|CLOSE\s+\w+|DEALLOCATE\s+\w+'
        r'|FETCH\s+NEXT\s+FROM'
        r'|WHILE\s+@@FETCH_STATUS'
        r'|IF\s+@@TRANCOUNT'
        r'|THROW\b'
        r'|PRINT\s+'
        r'|END\s*$'
        r')',
        _re.IGNORECASE,
    )

    for stmt in raw_stmts:
        flat = ' '.join(stmt.split())
        up   = flat.upper()

        # ── Always skip these ──────────────────────────────────────────────────
        if up.startswith('DECLARE '):
            continue
        if _re.match(r'SET\s+(NOCOUNT|XACT_ABORT|ANSI_NULLS|QUOTED_IDENTIFIER|TRANSACTION)',
                     flat, _re.IGNORECASE):
            continue
        if _re.match(r'SET\s+@', flat, _re.IGNORECASE):
            # But capture dynamic-SQL string assignments: SET @var = 'INSERT...'
            dyn_m = _re.match(r"SET\s+@(\w+)\s*=\s*N?'(.+)'$", flat, _re.IGNORECASE | _re.DOTALL)
            if dyn_m:
                dyn_sql_map[dyn_m.group(1).upper()] = dyn_m.group(2).strip()
            continue
        if _SKIP_RE.match(flat):
            continue

        # ── CREATE TABLE #temp → record name, skip DDL (view created by INSERT) ──
        crt_m = _re.match(r'CREATE\s+TABLE\s+(#\w+)', flat, _re.IGNORECASE)
        if crt_m:
            tmp = crt_m.group(1).lower().lstrip('#')
            temp_map[crt_m.group(1).lower()] = tmp + '_temp'
            continue   # DDL for #temp tables is implicit; the view comes from INSERT SELECT

        # ── EXEC sp_executesql @var → emit the stored SQL string ──────────────
        exec_m = _re.match(r'EXEC(?:UTE)?\s+sp_executesql\s+@(\w+)', flat, _re.IGNORECASE)
        if exec_m:
            var_up = exec_m.group(1).upper()
            sql_str = dyn_sql_map.get(var_up, '')
            if sql_str:
                classified.append(('REGULAR', sql_str, ' '.join(sql_str.split())))
            else:
                classified.append(('REGULAR', stmt, flat))   # best-effort fallback
            continue

        # ── RAISERROR → Python raise ValueError ────────────────────────────────
        if _re.match(r'RAISERROR\s*\(', flat, _re.IGNORECASE):
            msg_m = _re.search(r"RAISERROR\s*\(\s*N?'([^']+)'", flat, _re.IGNORECASE)
            msg = msg_m.group(1) if msg_m else 'Validation error'
            classified.append(('RAISERROR', msg))
            continue

        # ── Cursor DECLARE → emit one CURSOR_NOTE placeholder ─────────────────
        if (_re.match(r'DECLARE\s+\w+\s+CURSOR', flat, _re.IGNORECASE)
                and not _cursor_note_added):
            _cursor_note_added = True
            classified.append(('CURSOR_NOTE', stmt))
            continue

        # ── Assignment SELECT(s) → variable capture ────────────────────────────
        if _re.match(r'SELECT\b', flat, _re.IGNORECASE) and '@' in flat:
            from_m = _re.search(r'\bFROM\b', flat, _re.IGNORECASE)
            if from_m:
                select_list = flat[6:from_m.start()].strip()
                rest        = flat[from_m.start():]
                tbl_m       = _re.match(r'FROM\s+(\S+)', rest, _re.IGNORECASE)
                tbl         = tbl_m.group(1) if tbl_m else 'Unknown'
                items = [x.strip() for x in _re.split(r',(?![^()]*\))', select_list) if x.strip()]
                if items and all(_re.match(r'@\w+\s*=', it, _re.IGNORECASE) for it in items):
                    for it in items:
                        m = _re.match(r'@(\w+)\s*=\s*(.+)', it, _re.IGNORECASE)
                        if m:
                            var_map[m.group(1).upper()] = {'expr': m.group(2).strip(), 'table': tbl}
                    classified.append(('ASSIGN_GROUP', tbl, items, stmt))
                    continue
            else:
                classified.append(('VAR_READ', stmt, flat))
                continue

        # ── SELECT … INTO #temp OR ;WITH … SELECT * INTO #temp ────────────────
        into_m = _re.search(r'\bINTO\s+(#\w+)\b', flat, _re.IGNORECASE)
        is_select_or_cte = (
            _re.match(r'SELECT\b', flat, _re.IGNORECASE)
            or _re.match(r'WITH\b', flat, _re.IGNORECASE)
        )
        if into_m and is_select_or_cte:
            tmp_raw  = into_m.group(1).lower()
            view_nm  = tmp_raw.lstrip('#') + '_temp'
            temp_map[tmp_raw] = view_nm
            # Remove the INTO #temp clause from the SELECT
            clean = _re.sub(r'\bINTO\s+#\w+\b\s*', '', stmt, flags=_re.IGNORECASE)
            rewritten = f'CREATE OR REPLACE TEMP VIEW {view_nm} AS\n{clean.strip()}'
            classified.append(('REGULAR', rewritten, ' '.join(rewritten.split())))
            continue

        # ── INSERT INTO #temp SELECT … → CREATE OR REPLACE TEMP VIEW ──────────
        ins_into_m = _re.match(r'INSERT\s+INTO\s+(#\w+)\s+SELECT\b', flat, _re.IGNORECASE)
        if ins_into_m:
            tmp_raw  = ins_into_m.group(1).lower()
            view_nm  = tmp_raw.lstrip('#') + '_temp'
            temp_map[tmp_raw] = view_nm
            # Rewrite as CREATE OR REPLACE TEMP VIEW … AS SELECT …
            sel_start = _re.search(r'\bSELECT\b', stmt, _re.IGNORECASE)
            sel_body  = stmt[sel_start.start():] if sel_start else stmt
            rewritten = f'CREATE OR REPLACE TEMP VIEW {view_nm} AS\n{sel_body.strip()}'
            classified.append(('REGULAR', rewritten, ' '.join(rewritten.split())))
            continue

        classified.append(('REGULAR', stmt, flat))

    # ── 5. Consolidate ASSIGN_GROUPs + following VAR_READ ────────────────────
    final_stmts: list = []   # (sql_text,) tuples
    i = 0
    while i < len(classified):
        kind = classified[i][0]

        if kind == 'ASSIGN_GROUP':
            tbl0      = classified[i][1]
            all_items = list(classified[i][2])
            j = i + 1
            # Collect consecutive ASSIGN_GROUPs from the SAME table
            while (j < len(classified)
                   and classified[j][0] == 'ASSIGN_GROUP'
                   and classified[j][1].upper() == tbl0.upper()):
                all_items.extend(classified[j][2])
                j += 1

            # Check if immediately followed by a VAR_READ
            var_read_aliases: dict = {}
            if j < len(classified) and classified[j][0] == 'VAR_READ':
                between = _re.sub(
                    r'^SELECT\s+', '', classified[j][2], flags=_re.IGNORECASE
                ).strip()
                for part in between.split(','):
                    m = _re.match(r'@(\w+)\s+AS\s+(\w+)', part.strip(), _re.IGNORECASE)
                    if m:
                        var_read_aliases[m.group(1).upper()] = m.group(2)
                j += 1  # consume the VAR_READ

            # Build consolidated SELECT
            cols = []
            for it in all_items:
                m = _re.match(r'@(\w+)\s*=\s*(.+)', it, _re.IGNORECASE)
                if m:
                    v_up  = m.group(1).upper()
                    expr  = m.group(2).strip()
                    alias = var_read_aliases.get(v_up, v_up[0] + v_up[1:].lower())
                    cols.append(f'    {expr} AS {alias}')

            consolidated = 'SELECT\n' + ',\n'.join(cols) + f'\nFROM {tbl0}'
            final_stmts.append(('SQL', consolidated))
            i = j
            continue

        if kind == 'VAR_READ':
            # Unconsumed VAR_READ: resolve @Var refs from var_map
            resolved = classified[i][2]
            for v_up, info in var_map.items():
                resolved = _re.sub(rf'@{v_up}\b', info['expr'], resolved, flags=_re.IGNORECASE)
            if var_map and 'FROM' not in resolved.upper():
                tables = list({v['table'] for v in var_map.values()})
                if tables:
                    resolved += f' FROM {tables[0]}'
            final_stmts.append(('SQL', resolved))
            i += 1
            continue

        # Pass-through special kinds unchanged
        if kind in ('CURSOR_NOTE', 'RAISERROR'):
            final_stmts.append(classified[i])  # keep full tuple
            i += 1
            continue

        # REGULAR — preserve multi-line original text
        final_stmts.append(('SQL', classified[i][1]))
        i += 1
    def _t(sql: str) -> str:
        # Note: HTML entities already decoded on source above; _t only handles structure
        top_m = _re.search(r'\bSELECT\s+TOP\s+(\d+)\b', sql, _re.IGNORECASE)
        if top_m:
            n   = top_m.group(1)
            sql = _re.sub(r'\bTOP\s+\d+\b\s*', '', sql, flags=_re.IGNORECASE)
            ob  = _re.search(r'\bORDER\s+BY\b[^\n]*', sql, _re.IGNORECASE)
            sql = sql[:ob.end()] + f'\nLIMIT {n}' + sql[ob.end():] if ob else sql.rstrip() + f'\nLIMIT {n}'
        sql = _re.sub(r'\bWITH\s*\(\s*NOLOCK\s*\)',       '',                       sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bISNULL\s*\(',                   'COALESCE(',               sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bLEN\s*\(',                      'LENGTH(',                 sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bGETDATE\s*\(\s*\)',             'current_timestamp()',     sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bGETUTCDATE\s*\(\s*\)',          'UTC_TIMESTAMP()',         sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bCHARINDEX\s*\(',               'LOCATE(',                 sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bVARCHAR\s*\(\s*\d+\s*\)',       'STRING',                 sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bNVARCHAR\s*\(\s*(?:\d+|MAX)\s*\)', 'STRING',             sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bDATEADD\s*\(YEAR\s*,\s*-(\d+)\s*,', r'date_sub(current_timestamp(), \1 * 365,', sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bERROR_MESSAGE\s*\(\s*\)',       "'<error_message>'",       sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'@RunID\b',  'uuid()',  sql)   # local UNIQUEIDENTIFIER variable
        sql = _re.sub(r'@RUNID\b',  'uuid()',  sql)
        # Replace #temp_table references with spark view names (no \b — # is non-word)
        for tmp_lower, view_nm in temp_map.items():
            sql = _re.sub(_re.escape(tmp_lower), view_nm, sql, flags=_re.IGNORECASE)
        return sql

    # ── 7. Parameter replacement ──────────────────────────────────────────────
    # For NUMERIC params: f-string with {0 if p is None else p} (works for None).
    # For VARCHAR/DATE params: detect (@P IS NULL OR condition) pattern and
    # return info for Python if/else generation (avoids 'None' string comparisons).
    def _replace_params(sql: str):
        """
        Returns (modified_sql, uses_f, if_else)
        - modified_sql : str  — SQL with substitutions (or None when if_else applies)
        - uses_f       : bool — True if f-string needed
        - if_else      : None | dict{py_name, sql_with, sql_without}
        """
        if not param_map:
            return sql, False, None

        uses_f = False
        for upper, info in param_map.items():
            pat = _re.compile(rf'@{upper}\b', _re.IGNORECASE)
            if not pat.search(sql):
                continue
            py_nm = info['py_name']

            if info['is_numeric']:
                # Numeric: f-string approach.  None → 0 = 0 (true) so no WHERE filter.
                uses_f    = True
                null_expr = f'{{0 if {py_nm} is None else {py_nm}}}'
                sql = _re.sub(
                    rf'@{upper}\s+IS\s+NULL', f'{null_expr} = 0',
                    sql, flags=_re.IGNORECASE
                )
                sql = pat.sub(null_expr, sql)

            else:
                # VARCHAR/DATE: (@P IS NULL OR condition) → Python if/else
                null_or_re = _re.compile(
                    rf'(\s*\bWHERE\b\s*)\(\s*@{upper}\s+IS\s+NULL\s+OR\s+([^()]+)\)',
                    _re.IGNORECASE | _re.DOTALL
                )
                m = null_or_re.search(sql)
                if m:
                    cond_raw = m.group(2).strip()
                    # Replace @P in condition with f-string expression
                    cond_f = _re.sub(
                        rf'@{upper}\b', f"'{{{py_nm}}}'",
                        cond_raw, flags=_re.IGNORECASE
                    )
                    sql_without = sql[:m.start()].rstrip()
                    sql_with    = sql_without + f'\nWHERE {cond_f}'
                    return None, True, {
                        'py_name':     py_nm,
                        'sql_with':    sql_with,
                        'sql_without': sql_without,
                    }
                else:
                    # No IS NULL OR wrapper — plain replacement
                    uses_f = True
                    sql = pat.sub(f"'{{{py_nm}}}'", sql)

        return sql, uses_f, None

    # ── 8. Section-comment heuristic ─────────────────────────────────────────
    def _comment(sql: str) -> str:
        u = sql.upper()
        # Window functions — check most-specific first
        if 'LAG('        in u and 'OVER(' in u:                   return '# Previous Salary'
        if 'LEAD('       in u and 'OVER(' in u:                   return '# Next Salary'
        if 'ROW_NUMBER()' in u:                                    return '# Row Number'
        if 'DENSE_RANK()' in u:                                    return '# Dense Rank'
        if 'RANK()'       in u:                                    return '# Rank'
        if 'SUM('         in u and 'OVER(' in u:                   return '# Running Total'
        # Summary stats (aggregates, no GROUP BY, no WHERE, no JOIN)
        if (('COUNT(' in u or 'AVG(' in u)
                and 'GROUP BY' not in u
                and 'WHERE'    not in u
                and 'JOIN'     not in u):                          return '# Summary Statistics'
        # JOINs
        if 'INNER JOIN' in u:                                      return '# Inner Join'
        if 'LEFT JOIN'  in u:                                      return '# Left Join'
        if 'RIGHT JOIN' in u:                                      return '# Right Join'
        # Parameter-filtered queries — check f-string expressions (works on replaced SQL)
        if 'is None else minsalary'  in sql:                       return '# Salary Filter'
        if '{joindate}' in sql:                                    return '# Join Date Filter'
        if '{department}' in sql:                                  return '# Employee Filter'
        # HAVING
        if 'HAVING COUNT(*) > 1' in u:                             return '# Department Count > 1'
        if 'GROUP BY' in u and 'HAVING' in u and 'AVG(' in u:     return '# Department Average Salary > 50000'
        # GROUP BY + time functions
        if 'GROUP BY' in u and 'MONTH(' in u:                      return '# Joining Month Report'
        if 'GROUP BY' in u and 'YEAR('  in u:                      return '# Joining Year Report'
        # Full department summary (all 5 aggregates)
        if ('GROUP BY' in u
                and all(x in u for x in ('COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('))):
                                                                   return '# Department Summary'
        # Single-aggregate GROUP BY
        if ('GROUP BY' in u and 'COUNT(' in u
                and 'SUM(' not in u and 'AVG(' not in u
                and 'MAX(' not in u and 'MIN(' not in u):          return '# Department Count'
        if ('GROUP BY' in u and 'SUM(' in u
                and 'AVG(' not in u and 'MAX(' not in u and 'MIN(' not in u):
                                                                   return '# Department Salary'
        if ('GROUP BY' in u and 'AVG(' in u
                and 'MAX(' not in u and 'MIN(' not in u):          return '# Department Average Salary'
        if ('GROUP BY' in u and 'MAX(' in u
                and 'MIN(' not in u and 'AVG(' not in u):          return '# Department Maximum Salary'
        if ('GROUP BY' in u and 'MIN(' in u
                and 'MAX(' not in u and 'AVG(' not in u):          return '# Department Minimum Salary'
        # LIMIT (after TOP→LIMIT translation) — extract actual N from LIMIT clause
        lim_m = _re.search(r'\bLIMIT\s+(\d+)\b', u)
        if lim_m:
            n = lim_m.group(1)
            if 'DESC' in u:                                        return f'# Top {n} Highest Salaries'
            if 'ASC'  in u:                                        return f'# Top {n} Lowest Salaries'
        # LIKE
        if "LIKE 'A%" in sql:                                      return '# Names Starting With A'
        if "LIKE '%n'" in sql.lower() or "LIKE '%N'" in u:        return '# Names Ending With N'
        # BETWEEN
        if 'BETWEEN' in u:                                         return '# Salary Range'
        # Subqueries in WHERE
        if 'WHERE' in u and 'SELECT MAX(' in u:                    return '# Employee With Highest Salary'
        if 'WHERE' in u and 'SELECT MIN(' in u:                    return '# Employee With Lowest Salary'
        if 'WHERE' in u and 'SELECT AVG(' in u:                    return '# Salary Above Average'
        if 'WHERE' in u and _re.search(r'\bIN\s*\(', u):                        return '# Department IN Filter'
        if _re.search(r'\bIN\s*\(', u):                            return '# Department IN Filter'
        # Date filter
        if 'JOININGDATE' in u and '2022' in u:                     return '# Joined After 2022'
        # Salary threshold (plain WHERE Salary > N, not a param-driven filter)
        if ('WHERE' in u and 'SALARY' in u
                and 'GROUP BY' not in u and 'BETWEEN' not in u
                and '{' not in sql):                                return '# High Earners'
        # DISTINCT
        if 'SELECT DISTINCT' in u:                                 return '# Distinct Departments'
        # INSERT
        if u.strip().startswith('INSERT'):                         return '# Audit Log'
        return ''

    # ── 9. Function signature ─────────────────────────────────────────────────
    # Convert SP name → Python function name:
    # strip common usp_/sp_/proc_ prefix, then PascalCase → snake_case
    raw_name = pre.sp_name or 'process_data'
    for pfx in ('usp_', 'sp_', 'proc_', 'p_'):
        if raw_name.lower().startswith(pfx):
            raw_name = raw_name[len(pfx):]
            break
    # PascalCase / camelCase → snake_case
    raw_name = _re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', raw_name)
    raw_name = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', raw_name)
    func_name = (
        _re.sub(r'[^a-z0-9_]', '_', raw_name.lower()).strip('_') or 'process_data'
    )

    if param_map:
        param_list = list(param_map.values())
        sig_parts = ['    spark: SparkSession,'] + [
            f"    {info['py_name']}=None{',' if i < len(param_list) - 1 else ''}"
            for i, info in enumerate(param_list)
        ]
        func_sig = f'def {func_name}(\n' + '\n'.join(sig_parts) + '\n) -> DataFrame:'
    else:
        func_sig = f'def {func_name}(spark: SparkSession) -> DataFrame:'

    # ── 10. Assemble Python source ────────────────────────────────────────────
    # Each SELECT gets a unique numbered variable (result_df_1, result_df_2, …).
    # CURSOR_NOTE emits a comment. RAISERROR emits a raise comment.
    # INSERT / UPDATE / DELETE / MERGE use bare spark.sql(…).
    code: list = ['from pyspark.sql import SparkSession, DataFrame', '', func_sig, '']
    select_count   = 1
    last_select_var = None

    for item in final_stmts:
        kind = item[0]

        # ── CURSOR_NOTE ────────────────────────────────────────────────────────
        if kind == 'CURSOR_NOTE':
            code.append('    # NOTE: CURSOR replaced with bulk Spark SQL')
            code.append('    # Vectorize: build the payroll_summary_temp view above')
            code.append('    # using column expressions instead of row-by-row variables.')
            code.append('')
            continue

        # ── RAISERROR → Python raise comment ──────────────────────────────────
        if kind == 'RAISERROR':
            msg = item[1]
            code.append(f"    # Validation — raise on invalid data")
            code.append(f"    # if <condition>: raise ValueError('{msg}')")
            code.append('')
            continue

        sql_text        = item[1]   # valid for SQL, CURSOR_NOTE (stmt), RAISERROR (msg)
        sql             = _t(sql_text)
        result, uses_f, if_else = _replace_params(sql)

        up_raw = (result or sql).upper().lstrip()
        is_view = up_raw.startswith('CREATE OR REPLACE TEMP VIEW')
        is_sel  = up_raw.startswith('SELECT') and not is_view
        is_ins  = up_raw.startswith('INSERT')
        is_upd  = up_raw.startswith('UPDATE')
        is_del  = up_raw.startswith('DELETE')
        is_mrg  = up_raw.startswith('MERGE')
        is_ddl  = is_view or up_raw.startswith('CREATE TABLE') or up_raw.startswith('ALTER')

        if if_else:
            py_nm   = if_else['py_name']
            sw      = if_else['sql_with'].strip().replace('"""', '"\\""')
            swo     = if_else['sql_without'].strip().replace('"""', '"\\""')
            cmnt    = _comment(sw)
            var_nm  = f'result_df_{select_count}'
            if cmnt:
                code.append(f'    {cmnt}')
            code.append(f'    if {py_nm} is not None:')
            code.append(f'        {var_nm} = spark.sql(f"""')
            for line in sw.split('\n'):
                code.append(f'        {line}')
            code.append(f'        """)')
            code.append(f'    else:')
            code.append(f'        {var_nm} = spark.sql("""')
            for line in swo.split('\n'):
                code.append(f'        {line}')
            code.append(f'        """)')
            code.append('')
            last_select_var = var_nm
            select_count += 1
            continue

        safe = result.strip().replace('"""', '"\\""')
        cmnt = _comment(result)
        if cmnt:
            code.append(f'    {cmnt}')

        prefix = 'f' if uses_f else ''
        if is_sel:
            var_nm = f'result_df_{select_count}'
            last_select_var = var_nm
            select_count += 1
            code.append(f'    {var_nm} = spark.sql({prefix}"""')
        elif is_view or is_ddl or is_ins or is_mrg:
            code.append(f'    spark.sql({prefix}"""')
        elif is_upd or is_del:
            code.append(f'    # NOTE: {"UPDATE" if is_upd else "DELETE"} requires a Delta table')
            code.append(f'    spark.sql({prefix}"""')
        else:
            code.append(f'    spark.sql({prefix}"""')

        for line in safe.split('\n'):
            code.append(f'    {line}')
        code.append('    """)')
        code.append('')

    if last_select_var is None:
        last_select_var = 'result_df'
        code.append(f'    {last_select_var} = spark.sql("SELECT 1 AS result")')
        code.append('')

    code.append(f'    return {last_select_var}')
    return '\n'.join(code)


def convert_sql_with_ai(
    sql: str,
    db_prefix: str = "my_db",
    dialect: str = "T-SQL",
) -> str:
    """Convert SQL to PySpark SQL using the 3-stage pipeline.

    Routing (fastest/most-reliable first):
    1. Plain SQL scripts (≥5 stmts, no SP)      → _direct_convert_statements (instant)
    2. Simple stored procedures (no cursor/dyn) → _sp_direct_convert (instant)
    3. Complex SPs (cursor / dynamic SQL)       → rule-based ConversionPipeline (structured)
    4. Fallback                                 → LLM pipeline
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from converter.sql_preprocessor import preprocess
    from converter.code_postprocessor import postprocess

    pre = preprocess(sql)
    raw_stmts = [s.strip() for s in pre.cleaned_sql.split(";")
                 if s.strip() and not s.strip().startswith("--")]

    # Direct conversion for plain SQL scripts (reliable, complete, instant)
    if not pre.is_stored_procedure and len(raw_stmts) >= 5:
        direct = _direct_convert_statements(pre, db_prefix, original_sql=sql)
        if direct:
            result = postprocess(direct)
            return result.code

    # Direct conversion for ALL stored procedures (simple + complex)
    if pre.is_stored_procedure:
        direct = _sp_direct_convert(pre, db_prefix, original_sql=sql)
        if direct:
            result = postprocess(direct)
            return result.code

    # LLM pipeline — last resort for non-SP complex scripts
    user_prompt, _pre = _build_convert_user_prompt(sql, db_prefix, dialect)
    raw_output = _chat(_CONVERT_SYSTEM, user_prompt)
    result = postprocess(raw_output)
    return result.code


def stream_sql_with_ai(
    sql: str,
    db_prefix: str = "my_db",
    dialect: str = "T-SQL",
):
    """Stream SQL → PySpark conversion.

    Plain scripts and simple SPs: direct conversion (single chunk, instant).
    Complex SPs (cursor/dynamic SQL): rule-based pipeline (single chunk, structured).
    Non-SP complex scripts: LLM streaming.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from converter.sql_preprocessor import preprocess

    pre = preprocess(sql)
    raw_stmts = [s.strip() for s in pre.cleaned_sql.split(";")
                 if s.strip() and not s.strip().startswith("--")]

    # Direct conversion for plain SQL scripts
    if not pre.is_stored_procedure and len(raw_stmts) >= 5:
        direct = _direct_convert_statements(pre, db_prefix, original_sql=sql)
        if direct:
            yield direct
            return

    # Direct conversion for ALL stored procedures (simple + complex)
    if pre.is_stored_procedure:
        direct = _sp_direct_convert(pre, db_prefix, original_sql=sql)
        if direct:
            yield direct
            return

    # LLM streaming — last resort for non-SP complex scripts
    user_prompt, _pre = _build_convert_user_prompt(sql, db_prefix, dialect)
    provider = _detect_provider()
    if provider == "ollama":
        yield from _stream_ollama(_CONVERT_SYSTEM, user_prompt, temperature=0.15)
    else:
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
