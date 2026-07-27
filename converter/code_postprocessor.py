"""
PySpark Post-processor — Stage 3 of the conversion pipeline.

Validates and auto-fixes the raw LLM output before it is returned to the user.

Pipeline position:
  [SQL Input] → [Preprocessor] → [Ollama LLM] → [Postprocessor] → [PySpark Output]

What this module does:
  1. Strip markdown code fences (```python … ```) and any prose explanation
  2. Parse the Python code with ast.parse() to detect syntax errors
  3. Auto-inject missing standard imports (SparkSession, functions as F, etc.)
  4. Ensure the main function has a return statement for the result DataFrame
  5. Remove stray duplicate import lines
  6. Return a PostprocessResult with the fixed code + any warnings
"""

import ast
import re
import textwrap
from dataclasses import dataclass, field


# ─── Result container ──────────────────────────────────────────────────────────

@dataclass
class PostprocessResult:
    code: str                               # Final clean PySpark Python code
    syntax_valid: bool = True               # True if ast.parse() succeeded
    syntax_error: str = ""                  # AST error message if invalid
    warnings: list[str] = field(default_factory=list)  # Non-fatal notes
    imports_injected: list[str] = field(default_factory=list)  # Added imports


# ─── Required imports every PySpark script must have ─────────────────────────

_REQUIRED_IMPORTS = [
    ("from pyspark.sql import SparkSession",          "SparkSession"),
    ("from pyspark.sql import functions as F",        "functions"),
    ("from pyspark.sql.types import *",               "IntegerType"),   # heuristic
]

_DELTA_IMPORTS = [
    ("from delta.tables import DeltaTable",           "DeltaTable"),
]


# ─── Markdown fence stripper ──────────────────────────────────────────────────

_FENCE_RE = re.compile(
    r'(?:```(?:python|py)?|~~~(?:python|py)?)\s*\n(.*?)(?:\n```|\n~~~)',
    re.DOTALL,
)
_STRAY_FENCE = re.compile(r'^```|^~~~', re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Extract code from first markdown fence block; strip prose below it."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # No fence — strip trailing explanation (lines starting with ## or ---)
    lines = text.splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("##") or s == "---" or re.match(r'^[0-9]+\.', s):
            # Check there's an empty line before (prose section)
            if i > 0 and not lines[i - 1].strip():
                cut = i
                break

    text = "\n".join(lines[:cut]).strip()
    # Remove any remaining stray fence markers
    text = _STRAY_FENCE.sub("", text).strip()
    return text


# ─── Import injector ──────────────────────────────────────────────────────────

def _inject_imports(code: str, warnings: list[str], injected: list[str]) -> str:
    """Add missing standard PySpark imports at the top of the file."""
    needs_delta = bool(re.search(r"\bDeltaTable\b", code))

    candidates = list(_REQUIRED_IMPORTS)
    if needs_delta:
        candidates += _DELTA_IMPORTS

    additions = []
    for import_line, symbol in candidates:
        # Skip if already present (any form)
        if symbol in code or import_line in code:
            continue
        # Only inject functions if F. is used in the code
        if symbol == "functions" and "F." not in code:
            continue
        # Only inject types if type names are referenced
        if symbol == "IntegerType" and not re.search(
            r"\b(?:Integer|String|Double|Long|Boolean|Array|Map)Type\b", code
        ):
            continue
        additions.append(import_line)
        injected.append(import_line)

    if not additions:
        return code

    # Insert after the module docstring (if any) or at line 0
    lines = code.splitlines()
    insert_at = 0
    # Skip over module docstring
    if lines and lines[0].lstrip().startswith('"""'):
        for i, ln in enumerate(lines):
            if i > 0 and '"""' in ln:
                insert_at = i + 1
                break
    # Skip over existing import block
    for i in range(insert_at, len(lines)):
        if lines[i].startswith(("import ", "from ")):
            insert_at = i
            break

    # Find the end of the existing import block
    end_imports = insert_at
    for i in range(insert_at, len(lines)):
        if lines[i].startswith(("import ", "from ")):
            end_imports = i + 1
        elif lines[i].strip() == "" and end_imports > insert_at:
            break
        elif not lines[i].strip() and i == insert_at:
            continue
        elif lines[i].strip() and not lines[i].startswith(("import ", "from ", "#")):
            break

    new_lines = lines[:end_imports] + additions + lines[end_imports:]
    warnings.append(f"Injected missing imports: {additions}")
    return "\n".join(new_lines)


# ─── Duplicate import remover ────────────────────────────────────────────────

def _dedup_imports(code: str) -> str:
    """Remove duplicate import lines while preserving order."""
    lines = code.splitlines()
    seen_imports = set()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            if stripped in seen_imports:
                continue
            seen_imports.add(stripped)
        result.append(line)
    return "\n".join(result)


# ── Cursor warning injector ──────────────────────────────────────────────────

_CURSOR_NOTE = """
# =============================================================================
# CURSOR CONVERSION NOTE
# =============================================================================
# SQL RDBMS cursors and PySpark work fundamentally differently:
#
#  RDBMS Cursor              |  PySpark Equivalent
#  --------------------------|--------------------------------------------------
#  DECLARE cur CURSOR FOR    |  source_df = spark.sql("SELECT ...")
#  OPEN / FETCH NEXT         |  (no equivalent — Spark processes in bulk)
#  row-by-row UPDATE         |  spark.sql("UPDATE db.t SET col=val WHERE ...")  # Delta
#  row-by-row INSERT         |  spark.sql("INSERT INTO db.t SELECT ...")
#  row-by-row DELETE         |  spark.sql("DELETE FROM db.t WHERE ...")          # Delta
#  row-by-row transform      |  df.withColumn("col", expr(...))
#  CLOSE / DEALLOCATE        |  (no equivalent — no-op)
#
#  ⚠ NEVER use .collect() + Python loop — it loads all rows into driver memory.
#  ⚠ Use foreachPartition() only as a last resort for non-SQL logic.
# =============================================================================
""".strip()


def _inject_cursor_note(code: str) -> str:
    """If the code contains cursor-related patterns, inject a reference note."""
    import re
    cursor_patterns = [
        r"#.*cursor",
        r"#.*CURSOR",
        r"foreachPartition",
        r"foreach\(",
        r"\.collect\(\)",
    ]
    if any(re.search(p, code, re.IGNORECASE) for p in cursor_patterns):
        # Insert after the last import line
        lines = code.splitlines()
        last_import = 0
        for i, ln in enumerate(lines):
            if ln.startswith(("import ", "from ")):
                last_import = i
        insert_at = last_import + 1
        lines.insert(insert_at, "")
        lines.insert(insert_at + 1, _CURSOR_NOTE)
        lines.insert(insert_at + 2, "")
        return "\n".join(lines)
    return code



def _check_return(code: str, warnings: list[str]) -> None:
    """Warn if the main conversion function has no return statement."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for a return statement anywhere in the function body
                has_return = any(
                    isinstance(n, ast.Return) and n.value is not None
                    for n in ast.walk(node)
                )
                if not has_return:
                    warnings.append(
                        f"Function '{node.name}' has no return statement — "
                        "add 'return result_df' at the end."
                    )
    except SyntaxError:
        pass  # handled separately by syntax check

# ─── Unused logging remover ───────────────────────────────────────────────────────────

_LOGGING_LINE_RE = re.compile(
    r"^(?:import logging|logging\.basicConfig[^\n]*|logger\s*=\s*logging\.getLogger[^\n]*)\n?",
    re.MULTILINE,
)


def _strip_unused_logging(code: str) -> str:
    """Remove logging imports and logger setup if logger is never actually called."""
    if re.search(r"\blogger\.(info|warning|error|debug|exception)\s*\(", code):
        return code  # logger is used — keep it
    return _LOGGING_LINE_RE.sub("", code)

# ─── Public API ───────────────────────────────────────────────────────────────

def postprocess(raw_output: str) -> PostprocessResult:
    """
    Validate and auto-fix raw LLM output into clean, runnable PySpark code.

    Steps:
      1. Strip markdown fences and prose explanation
      2. Deduplicate import lines
      3. Inject any missing standard imports
      4. Validate Python syntax with ast.parse()
      5. Check for a return statement in the main function
      6. Return PostprocessResult with code + diagnostics

    Args:
        raw_output: Raw string from the LLM (may contain fences / prose)

    Returns:
        PostprocessResult with validated/fixed code and any warnings.
    """
    warnings: list[str] = []
    injected: list[str] = []

    # Step 1: Strip fences and prose
    code = _strip_fences(raw_output)

    # Guard: remove SSE artifacts that may have been passed in raw SSE text
    code = re.sub(r"\[RESULT\]\s*\{.*", "", code, flags=re.DOTALL).strip()
    code = re.sub(r"data:\s*\[(?:RESULT|DONE|ERROR)\].*", "", code, flags=re.DOTALL).strip()
    code = re.sub(r'",\s*"sp_name"\s*:\s*"[^"]*"\s*\}?\s*$', "", code).strip()

    # Guard: if code came through JSON serialisation, unescape \" → " and \n → newline
    if '\\"' in code and '"""' not in code:
        try:
            import json as _json
            code = _json.loads('"' + code + '"')   # interpret as JSON string value
        except Exception:
            code = code.replace('\\"', '"').replace("\\n", "\n")

    if not code.strip():
        return PostprocessResult(
            code="# ERROR: LLM returned empty output.",
            syntax_valid=False,
            syntax_error="Empty output from model.",
            warnings=["Model returned no code."],
        )

    # Step 2: Deduplicate imports
    code = _dedup_imports(code)

    # Step 3: Inject missing imports
    code = _inject_imports(code, warnings, injected)

    # Step 3b: Strip unused logging imports
    code = _strip_unused_logging(code)

    # Step 3c: If cursor patterns are present, inject reference note
    code = _inject_cursor_note(code)

    # Step 4: Validate Python syntax
    syntax_valid = True
    syntax_error = ""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        syntax_valid = False
        syntax_error = f"SyntaxError at line {exc.lineno}: {exc.msg}"
        warnings.append(f"Syntax error in generated code — {syntax_error}")

    # Step 5: Return-statement check (only if syntax is valid)
    if syntax_valid:
        _check_return(code, warnings)

    return PostprocessResult(
        code=code,
        syntax_valid=syntax_valid,
        syntax_error=syntax_error,
        warnings=warnings,
        imports_injected=injected,
    )
