"""
SQL Parser — extracts structured information from SQL Stored Procedures and queries.
Handles T-SQL (SQL Server), PL/pgSQL (PostgreSQL), and ANSI SQL.
"""
# saved
 
import re
from dataclasses import dataclass, field
from typing import Optional
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────────────────────
 
@dataclass
class SPParameter:
    name: str                       # e.g. "@dept_id"
    sql_type: str                   # e.g. "INT"
    precision: str = ""
    scale: str = ""
    default: Optional[str] = None   # e.g. "NULL" or "0"
    direction: str = "IN"           # IN | OUT | INOUT
 
 
@dataclass
class DeclaredVariable:
    name: str
    sql_type: str
    default: Optional[str] = None
 
 
@dataclass
class TempTable:
    name: str        # e.g. "#temp_results"
    columns: list[tuple[str, str]] = field(default_factory=list)  # [(col, type)]
    is_global: bool = False
 
 
@dataclass
class SQLStatement:
    stmt_type: str      # SELECT | INSERT | UPDATE | DELETE | MERGE | IF | WHILE |
                        # CREATE_TABLE | DROP_TABLE | DECLARE | EXEC | PRINT | OTHER
    raw_sql: str        # original SQL text
    alias: str = ""     # alias for INTO temp tables
    metadata: dict = field(default_factory=dict)
 
 
@dataclass
class ParsedSQL:
    """Top-level result of parsing a SQL Stored Procedure or query batch."""
    is_stored_procedure: bool = False
    sp_name: str = ""
    parameters: list[SPParameter] = field(default_factory=list)
    variables: list[DeclaredVariable] = field(default_factory=list)
    temp_tables: list[TempTable] = field(default_factory=list)
    statements: list[SQLStatement] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)   # permanent table names
    raw_body: str = ""
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Helper regex patterns
# ──────────────────────────────────────────────────────────────────────────────
 
# Matches /* ... */ block comments (non-greedy, dotall)
_BLOCK_COMMENT  = re.compile(r"/\*.*?\*/", re.DOTALL)
# Matches -- line comments
_LINE_COMMENT   = re.compile(r"--[^\n]*")
# SP header: CREATE [OR ALTER] PROCEDURE name
_SP_HEADER      = re.compile(
    r"CREATE\s+(?:OR\s+ALTER\s+)?(?:PROCEDURE|PROC)\s+([\w\.\[\]]+)",
    re.IGNORECASE,
)
# Parameter list: @name [AS] TYPE[(p,s)] [= default] [OUT|OUTPUT|READONLY]
_PARAM_RE       = re.compile(
    r"(@[\w]+)\s+(?:AS\s+)?(\w+(?:\s+\w+)?)(?:\(([^)]*)\))?"   # name + type + optional (p,s)
    r"(?:\s*=\s*([^,\n@]+?))?(?:\s+(OUT(?:PUT)?|READONLY))?\s*(?=,|$)",
    re.IGNORECASE | re.MULTILINE,
)
# DECLARE variable
_DECLARE_RE     = re.compile(
    r"DECLARE\s+(@[\w]+)\s+(?:AS\s+)?(\w+(?:\s+\w+)?)(?:\(([^)]*)\))?"
    r"(?:\s*=\s*(.+?))?(?=;|DECLARE|$)",
    re.IGNORECASE | re.DOTALL,
)
# SET @variable = value
_SET_VAR_RE     = re.compile(
    r"SET\s+(@[\w]+)\s*=\s*(.+?)(?=;|$|\n)",
    re.IGNORECASE,
)
# CREATE TABLE #temp (...)
_CREATE_TEMP_RE = re.compile(
    r"(?:CREATE\s+TABLE\s+|SELECT\s+.+?\s+INTO\s+)(#[\w]+)\b",
    re.IGNORECASE | re.DOTALL,
)
# FROM/JOIN table references (capture permanent table names)
_TABLE_REF_RE   = re.compile(
    r"(?:FROM|JOIN)\s+((?!#)[\w\.\[\]]+)\s*(?:AS\s+)?(\w+)?",
    re.IGNORECASE,
)
# SELECT ... INTO #temp
_SELECT_INTO_RE = re.compile(
    r"\bSELECT\b(.+?)\bINTO\b\s+(#[\w]+)\b(.+?)(?=;|$)",
    re.IGNORECASE | re.DOTALL,
)
# Basic statement type detector
_STMT_START_RE  = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|EXEC(?:UTE)?|PRINT|IF|WHILE|BEGIN|END|DECLARE|SET|WITH)\b",
    re.IGNORECASE,
)
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Core parser class
# ──────────────────────────────────────────────────────────────────────────────
 
class SQLParser:
    """Parse SQL Stored Procedures and standalone queries into a structured form."""
 
    # ── public API ────────────────────────────────────────────────────────────
 
    def parse(self, sql_text: str) -> ParsedSQL:
        """Main entry point. Returns a ParsedSQL dataclass."""
        cleaned  = self._strip_comments(sql_text)
        result   = ParsedSQL(raw_body=cleaned)
 
        sp_match = _SP_HEADER.search(cleaned)
        if sp_match:
            result.is_stored_procedure = True
            result.sp_name = sp_match.group(1).strip("[]")
            params_text, body = self._split_header_and_body(cleaned, sp_match)
            result.parameters  = self._parse_parameters(params_text)
            result.raw_body    = body
        else:
            result.raw_body = cleaned
 
        result.variables   = self._parse_declares(result.raw_body)
        result.temp_tables = self._detect_temp_tables(result.raw_body)
        result.statements  = self._extract_statements(result.raw_body)
        result.dependencies = self._extract_table_deps(result.raw_body)
        return result
 
    # ── private helpers ───────────────────────────────────────────────────────
 
    @staticmethod
    def _strip_comments(sql: str) -> str:
        sql = _BLOCK_COMMENT.sub(" ", sql)
        sql = _LINE_COMMENT.sub("", sql)
        # Collapse excessive blank lines
        sql = re.sub(r"\n{3,}", "\n\n", sql)
        return sql.strip()
 
    @staticmethod
    def _split_header_and_body(sql: str, sp_match) -> tuple[str, str]:
        """Return (params_text_before_AS, body_after_BEGIN)."""
        after_name = sql[sp_match.end():]
        # Find AS keyword that precedes the body
        as_match = re.search(r"\bAS\b", after_name, re.IGNORECASE)
        if as_match:
            params_text = after_name[:as_match.start()]
            rest        = after_name[as_match.end():]
        else:
            params_text = ""
            rest        = after_name
 
        # Strip outer BEGIN...END wrapper if present
        begin_match = re.search(r"\bBEGIN\b", rest, re.IGNORECASE)
        if begin_match:
            body = rest[begin_match.end():]
            # Strip trailing END
            body = re.sub(r"\bEND\s*;?\s*$", "", body, flags=re.IGNORECASE).strip()
        else:
            body = rest.strip()
 
        return params_text, body
 
    @staticmethod
    def _parse_parameters(params_text: str) -> list[SPParameter]:
        params: list[SPParameter] = []
        # Normalize: remove parentheses wrapping entire param block
        params_text = params_text.strip().strip("()")
        for m in _PARAM_RE.finditer(params_text):
            name      = m.group(1)
            raw_type  = m.group(2).strip().upper()
            paren_grp = m.group(3) or ""  # e.g. "10,2" or "100"
            default   = (m.group(4) or "").strip()
            direction = "OUT" if m.group(5) else "IN"
 
            # Parse precision/scale from paren group
            p, s = "", ""
            if "," in paren_grp:
                parts = paren_grp.split(",")
                p, s  = parts[0].strip(), parts[1].strip()
            elif paren_grp:
                p = paren_grp.strip()
 
            params.append(SPParameter(
                name=name, sql_type=raw_type,
                precision=p, scale=s,
                default=default if default else None,
                direction=direction,
            ))
        return params
 
    @staticmethod
    def _parse_declares(body: str) -> list[DeclaredVariable]:
        variables: list[DeclaredVariable] = []
        for m in _DECLARE_RE.finditer(body):
            name      = m.group(1)
            raw_type  = (m.group(2) or "").strip().upper()
            default   = (m.group(4) or "").strip()
            variables.append(DeclaredVariable(
                name=name, sql_type=raw_type,
                default=default if default else None,
            ))
        return variables
 
    @staticmethod
    def _detect_temp_tables(body: str) -> list[TempTable]:
        temps: list[TempTable] = []
        seen: set[str] = set()
        for m in _CREATE_TEMP_RE.finditer(body):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                temps.append(TempTable(name=name, is_global=name.startswith("##")))
        return temps
 
    @staticmethod
    def _extract_table_deps(body: str) -> list[str]:
        deps: list[str] = []
        seen: set[str] = set()
        for m in _TABLE_REF_RE.finditer(body):
            tbl = m.group(1).strip("[]")
            if tbl.upper() not in ("SELECT", "VALUES", "DUAL") and tbl not in seen:
                seen.add(tbl)
                deps.append(tbl)
        return deps
 
    def _extract_statements(self, body: str) -> list[SQLStatement]:
        """
        Split body into individual SQL statements and classify each one.
        Uses a simplified block-tracking approach for IF/WHILE/BEGIN blocks.
        """
        statements: list[SQLStatement] = []
        # Split on semicolons at the top level (not inside parentheses/strings)
        raw_stmts = self._split_on_semicolons(body)
        for raw in raw_stmts:
            raw = raw.strip()
            if not raw:
                continue
            stmt_type = self._classify_stmt(raw)
            alias     = self._extract_alias(raw, stmt_type)
            statements.append(SQLStatement(
                stmt_type=stmt_type,
                raw_sql=raw,
                alias=alias,
            ))
        return statements
 
    @staticmethod
    def _split_on_semicolons(sql: str) -> list[str]:
        """Split SQL on semicolons, but respect BEGIN...END blocks."""
        parts: list[str]  = []
        depth: int        = 0
        current: list[str]= []
 
        tokens = re.split(r"(\bBEGIN\b|\bEND\b|;)", sql, flags=re.IGNORECASE)
        for tok in tokens:
            upper = tok.strip().upper()
            if upper == "BEGIN":
                depth += 1
                current.append(tok)
            elif upper == "END":
                if depth > 0:
                    depth -= 1
                current.append(tok)
            elif tok == ";" and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(tok)
        if current:
            parts.append("".join(current))
        return parts
 
    @staticmethod
    def _classify_stmt(stmt: str) -> str:
        # First: try matching a known SQL keyword at the very start
        m = _STMT_START_RE.match(stmt)
        if m:
            kw = m.group(1).upper()
            mapping = {
                "SELECT":  "SELECT",
                "INSERT":  "INSERT",
                "UPDATE":  "UPDATE",
                "DELETE":  "DELETE",
                "MERGE":   "MERGE",
                "CREATE":  "CREATE_TABLE",
                "DROP":    "DROP_TABLE",
                "EXEC":    "EXEC",
                "EXECUTE": "EXEC",
                "PRINT":   "PRINT",
                "IF":      "IF",
                "WHILE":   "WHILE",
                "BEGIN":   "BEGIN_BLOCK",
                "END":     "END_BLOCK",
                "DECLARE": "DECLARE",
                "SET":     "SET",
                "WITH":    "CTE",
            }
            return mapping.get(kw, "OTHER")
 
        # Second: check if a SQL keyword appears ANYWHERE in the statement
        # (handles cases like "-- comment\nSELECT ..." or leading whitespace/junk)
        _ANY_SQL_KW = re.compile(
            r"\b(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE|CREATE\s+TABLE|DROP\s+TABLE)\b",
            re.IGNORECASE,
        )
        m2 = _ANY_SQL_KW.search(stmt)
        if m2:
            kw2 = m2.group(1).upper().split()[0]
            kw_map = {
                "SELECT": "SELECT", "INSERT": "INSERT", "UPDATE": "UPDATE",
                "DELETE": "DELETE", "MERGE": "MERGE",
                "CREATE": "CREATE_TABLE", "DROP": "DROP_TABLE",
            }
            return kw_map.get(kw2, "OTHER")
 
        # Third: anything else → OTHER (will be handled by convert_unknown)
        return "OTHER"
 
    @staticmethod
    def _extract_alias(stmt: str, stmt_type: str) -> str:
        """For SELECT...INTO #temp, extract the temp table name."""
        if stmt_type == "SELECT":
            m = re.search(r"\bINTO\s+(#[\w]+)\b", stmt, re.IGNORECASE)
            if m:
                return m.group(1)
        return ""
 