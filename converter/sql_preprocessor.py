"""
T-SQL Pre-processor — Stage 1 of the conversion pipeline.

Cleans raw T-SQL / SQL Server syntax before it is sent to the Ollama LLM so
the model receives a tidy, unambiguous input and produces better output.

Pipeline position:
  [SQL Input] → [Preprocessor] → [Ollama LLM] → [Postprocessor] → [PySpark Output]

What this module does:
  1. Remove SQL Server noise: GO, USE, SET NOCOUNT, EXEC sp_executesql, etc.
  2. Strip bracket-quoted identifiers: [MyTable] → MyTable
  3. Normalise whitespace and blank lines
  4. Extract SP name and parameters as metadata for the prompt
  5. Return a PreprocessResult with the cleaned SQL + metadata dict
"""

import re
from dataclasses import dataclass, field


# ─── Result container ──────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    cleaned_sql: str                        # Cleaned SQL ready for the LLM
    sp_name: str = ""                       # Stored procedure name (if detected)
    parameters: list[dict] = field(default_factory=list)  # [{name, type, default}]
    temp_tables: list[str] = field(default_factory=list)  # temp table names
    is_stored_procedure: bool = False
    dialect_hints: list[str] = field(default_factory=list)  # e.g. ["cursor", "dynamic_sql"]


# ─── Regex patterns ────────────────────────────────────────────────────────────

# Block and line comments
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT  = re.compile(r"--[^\n]*")

# SQL Server noise to remove entirely
_NOISE_PATTERNS = [
    re.compile(r"^\s*GO\s*$",                           re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*USE\s+\S+\s*;?\s*$",              re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*SET\s+NOCOUNT\s+(?:ON|OFF)\s*;?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*SET\s+XACT_ABORT\s+(?:ON|OFF)\s*;?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*SET\s+ANSI_NULLS\s+(?:ON|OFF)\s*;?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*SET\s+QUOTED_IDENTIFIER\s+(?:ON|OFF)\s*;?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*EXEC(?:UTE)?\s+sp_addextendedproperty.*?;?\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL),
    re.compile(r"^\s*PRINT\s+N?'[^']*'\s*;?\s*$",      re.IGNORECASE | re.MULTILINE),
]

# Bracket-quoted identifiers: [name] → name  (keeps schema.table intact)
_BRACKET_ID = re.compile(r"\[([^\]]+)\]")

# SP header patterns
_SP_HEADER = re.compile(
    r"CREATE\s+(?:OR\s+ALTER\s+)?(?:PROCEDURE|PROC)\s+([\w\.\[\]]+)",
    re.IGNORECASE,
)
_SP_PARAM = re.compile(
    r"(@[\w]+)\s+(?:AS\s+)?(\w+(?:\s+\w+)?)(?:\([^)]*\))?"
    r"(?:\s*=\s*([^,\n@]+?))?(?:\s+(?:OUT(?:PUT)?|READONLY))?\s*(?=,|@|AS\s+BEGIN|BEGIN|$)",
    re.IGNORECASE,
)

# Temp table names
_TEMP_TABLE = re.compile(r"#[\w]+", re.IGNORECASE)

# Dialect complexity hints
_HINT_CURSOR  = re.compile(r"\bDECLARE\s+\w+\s+CURSOR\b",       re.IGNORECASE)
_HINT_DYNAMIC = re.compile(r"\bEXEC(?:UTE)?\s*\(",               re.IGNORECASE)
_HINT_MERGE   = re.compile(r"\bMERGE\s+INTO\b|\bMERGE\b",        re.IGNORECASE)
_HINT_WINDOW  = re.compile(r"\b(?:ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|NTILE)\s*\(", re.IGNORECASE)
_HINT_CTE     = re.compile(r"\bWITH\s+\w+\s+AS\s*\(",            re.IGNORECASE)


# ─── Public API ────────────────────────────────────────────────────────────────

def preprocess(sql: str) -> PreprocessResult:
    """
    Clean raw T-SQL and extract metadata for the LLM prompt.

    Steps performed:
      1. Strip block and line comments
      2. Remove SQL Server noise (GO, USE, SET …)
      3. Un-bracket identifiers: [name] → name
      4. Collapse excessive blank lines (max 2 → 1)
      5. Extract SP name, parameters, temp table names
      6. Detect dialect complexity hints (cursor, dynamic SQL, etc.)

    Returns a PreprocessResult with cleaned_sql and metadata.
    """
    text = sql

    # 1. Strip comments (preserve structure)
    text = _BLOCK_COMMENT.sub(" ", text)
    text = _LINE_COMMENT.sub("", text)

    # 2. Remove SQL Server noise lines
    for pat in _NOISE_PATTERNS:
        text = pat.sub("", text)

    # 3. Un-bracket identifiers
    text = _BRACKET_ID.sub(r"\1", text)

    # 4. Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 5. Extract metadata from the ORIGINAL sql (before noise removal)
    sp_match = _SP_HEADER.search(sql)
    sp_name  = _BRACKET_ID.sub(r"\1", sp_match.group(1)).strip() if sp_match else ""
    is_sp    = bool(sp_match)

    params = []
    if is_sp:
        # Find parameter block between SP header and AS/BEGIN
        header_end = sp_match.end()
        as_begin   = re.search(r"\bAS\s*\n?\s*BEGIN\b|\bAS\s*BEGIN\b", sql[header_end:], re.IGNORECASE)
        param_block = sql[header_end : header_end + (as_begin.start() if as_begin else 500)]
        for m in _SP_PARAM.finditer(param_block):
            params.append({
                "name":    m.group(1),
                "type":    m.group(2).strip(),
                "default": (m.group(3) or "").strip() or None,
            })

    temp_tables = list({m.group() for m in _TEMP_TABLE.finditer(text)})

    # 6. Detect dialect complexity hints
    hints = []
    if _HINT_CURSOR.search(sql):   hints.append("cursor")
    if _HINT_DYNAMIC.search(sql):  hints.append("dynamic_sql")
    if _HINT_MERGE.search(sql):    hints.append("merge")
    if _HINT_WINDOW.search(sql):   hints.append("window_functions")
    if _HINT_CTE.search(sql):      hints.append("cte")
    if temp_tables:                hints.append("temp_tables")

    return PreprocessResult(
        cleaned_sql=text,
        sp_name=sp_name,
        parameters=params,
        temp_tables=temp_tables,
        is_stored_procedure=is_sp,
        dialect_hints=hints,
    )
