"""
PySpark Code Converter — transforms each SQL statement into PySpark code.
Uses sqlglot for SQL parsing; falls back to spark.sql() passthrough for complex queries.
Follows Step 3 of the 7-step AI Framework Pipeline.
"""
 
import re
import textwrap
from converter.sql_parser import SQLStatement, SPParameter, DeclaredVariable
from sql_mappings.sql_types import resolve_type, resolve_python_type
from sql_mappings.sql_functions import SCALAR_FUNCTION_MAP
 
try:
    import sqlglot
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
 
def _clean_name(name: str) -> str:
    """Strip @, #, [], spaces from SQL identifiers → valid Python names."""
    name = name.strip().lstrip("@#").strip("[]")
    return re.sub(r"\W+", "_", name).lower()
 
 
def _py_default(sql_default: str | None) -> str:
    if sql_default is None:
        return ""
    val = sql_default.strip().upper()
    if val in ("NULL", ""):
        return "None"
    if val == "TRUE":
        return "True"
    if val == "FALSE":
        return "False"
    return sql_default.strip()
 
 
def _indent(text: str, spaces: int = 4) -> str:
    return textwrap.indent(text, " " * spaces)
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Parameter → Python signature
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_parameters(params: list[SPParameter]) -> list[str]:
    """Return a list of Python function argument strings."""
    args: list[str] = []
    for p in params:
        py_name  = _clean_name(p.name)
        py_type  = resolve_python_type(p.sql_type)
        default  = _py_default(p.default)
        if default:
            args.append(f"{py_name}: {py_type} = {default}")
        else:
            args.append(f"{py_name}: {py_type}")
    return args
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  DECLARE / SET variable → Python variable
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_declare(var: DeclaredVariable) -> str:
    py_name  = _clean_name(var.name)
    default  = _py_default(var.default)
    if not default:
        default = "None"
    return f"{py_name} = {default}  # SQL: DECLARE {var.name} {var.sql_type}"
 
 
# SQL Server system SET options that have no PySpark equivalent — silently suppressed
_SUPPRESS_SET_RE = re.compile(
    r"SET\s+(NOCOUNT|XACT_ABORT|ANSI_NULLS|ANSI_PADDING|ANSI_WARNINGS|"
    r"QUOTED_IDENTIFIER|NOEXEC|ARITHABORT|CONCAT_NULL_YIELDS_NULL|"
    r"NUMERIC_ROUNDABORT|IMPLICIT_TRANSACTIONS)\s+(ON|OFF)",
    re.IGNORECASE,
)


def convert_set_statement(raw_sql: str) -> str:
    """Convert SET @var = expr to Python assignment."""
    # Suppress SQL Server system SET options — no PySpark equivalent
    if _SUPPRESS_SET_RE.match(raw_sql.strip()):
        return ""

    m = re.match(r"SET\s+(@[\w]+)\s*=\s*(.+)", raw_sql.strip(), re.IGNORECASE | re.DOTALL)
    if not m:
        return f"# TODO: {raw_sql.strip()}"
    py_name = _clean_name(m.group(1))
    value   = m.group(2).strip().rstrip(";")
    # Convert common SQL expressions
    value   = _translate_expression(value)
    return f"{py_name} = {value}  # SQL: SET {m.group(1)} = {m.group(2).strip()}"
 
 
def _translate_expression(expr: str) -> str:
    """Best-effort translation of simple SQL expressions to Python."""
    expr = expr.strip()
    # Replace GETDATE() / GETUTCDATE()
    expr = re.sub(r"GETDATE\(\)|GETUTCDATE\(\)|CURRENT_TIMESTAMP",
                  "datetime.datetime.now()", expr, flags=re.IGNORECASE)
    # Replace DATEADD(day, n, col)
    expr = re.sub(
        r"DATEADD\s*\(\s*day\s*,\s*([^,]+),\s*([^)]+)\)",
        lambda m: f"({m.group(2).strip()} + datetime.timedelta(days={m.group(1).strip()}))",
        expr, flags=re.IGNORECASE,
    )
    # Replace NULL → None
    expr = re.sub(r"\bNULL\b", "None", expr, flags=re.IGNORECASE)
    return expr
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  @Param substitution helpers
# ──────────────────────────────────────────────────────────────────────────────

def _handle_nullable_filters(sql: str) -> tuple[str, list[str]]:
    """Detect nullable-param filter patterns and convert to Python conditional variables.

    Handles both T-SQL:  AND (@Region IS NULL OR sr.Region = @Region)
    and sqlglot output:  AND (${Region} IS NULL OR sr.Region = ${Region})

    Produces:
      pre: _region_filter = f"AND sr.Region = '{region}'" if region is not None else ""
      sql: {_region_filter}
    """
    pre_lines: list[str] = []

    # Match both @Param and ${Param} forms; use DOTALL since sqlglot adds newlines
    pattern = re.compile(
        r"(?:AND\s+)?\(\s*(?:@|\$\{)([\w]+)(?:\})?\s+IS\s+NULL"
        r"\s+OR\s+([\w\.]+)\s*=\s*(?:@|\$\{)\1(?:\})?\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    def _replace(m_):
        py_name = _clean_name("@" + m_.group(1))
        col     = m_.group(2)
        var     = f"_{py_name}_filter"
        pre_lines.append(
            f'{var} = f"AND {col} = \'{{{py_name}}}\'" if {py_name} is not None else ""'
        )
        return "{" + var + "}"

    modified = pattern.sub(_replace, sql)
    return modified, pre_lines


def _substitute_params(sql: str) -> tuple[str, bool]:
    """Replace SQL parameter refs with quoted Python f-string placeholders.

    Handles both T-SQL (@Param) and Spark SQL named params (${Param}) produced
    by sqlglot transpilation.
    Returns (modified_sql, has_params).
    """
    if "@" not in sql and "${" not in sql:
        return sql, False
    # Match @Param or ${Param}
    _param_re = re.compile(r"(?:@([\w]+)|\$\{([\w]+)\})")
    if not _param_re.search(sql):
        return sql, False

    def _fmt(m_: re.Match) -> str:
        name = m_.group(1) or m_.group(2)  # capture from @Param or ${Param}
        return "'" + "{" + _clean_name("@" + name) + "}" + "'"

    result = _param_re.sub(_fmt, sql)
    return result, True


def _build_spark_sql(sql: str, var: str = "result_df",
                     view_name: str = "", pre_lines: list[str] | None = None) -> str:
    """Wrap a SQL string in spark.sql(). Handles f-string, indentation, pre-filter vars."""
    pre_lines = pre_lines or []
    sql_clean, has_params = _substitute_params(sql)
    q = "f" if (has_params or pre_lines) else ""

    output: list[str] = []
    if pre_lines:
        output.extend(pre_lines)
    output.append(f'{var} = spark.sql({q}"""')
    for line in sql_clean.rstrip(";").split("\n"):
        output.append(f"    {line}" if line.strip() else "")
    output.append('""")')
    if view_name:
        output.append(f'{var}.createOrReplaceTempView("{view_name}")')
    return "\n".join(output)


# ──────────────────────────────────────────────────────────────────────────────
#  SELECT statement → PySpark
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_select(stmt: SQLStatement, db_prefix: str = "my_db") -> str:
    """
    Convert a SELECT statement to PySpark.
    Uses sqlglot for structural analysis when available; otherwise uses spark.sql().
    """
    raw   = stmt.raw_sql.strip()
    alias = stmt.alias  # temp table name (for SELECT...INTO)
 
    # Check if it's a SELECT...INTO (temp table creation)
    into_match = re.search(r"\bINTO\s+(#[\w]+)\b", raw, re.IGNORECASE)
 
    if _HAS_SQLGLOT:
        return _convert_select_sqlglot(raw, alias, db_prefix, into_match)
    else:
        return _convert_select_spark_sql(raw, alias, db_prefix, into_match)
 
 
def _convert_select_spark_sql(raw: str, alias: str, db_prefix: str, into_match) -> str:
    """Fallback: wrap in spark.sql(). Remove INTO clause first."""
    if into_match:
        clean_sql = re.sub(r"\s*INTO\s+#[\w]+", "", raw, flags=re.IGNORECASE).strip()
        view_name = alias.lstrip("#")
        sql, pre  = _handle_nullable_filters(clean_sql)
        return _build_spark_sql(sql, var=f"_{view_name}_df",
                                view_name=view_name, pre_lines=pre)
    sql, pre = _handle_nullable_filters(raw)
    return _build_spark_sql(sql, pre_lines=pre)
def _convert_select_sqlglot(raw: str, alias: str, db_prefix: str, into_match) -> str:
    """Use sqlglot to transpile T-SQL → Spark SQL dialect, then wrap in spark.sql()."""
    try:
        # pretty=True preserves multi-line formatting; error_level=RAISE falls back on failure
        transpiled = sqlglot.transpile(raw, read="tsql", write="spark",
                                       pretty=True, error_level="raise")
        spark_sql = transpiled[0] if transpiled else raw
    except Exception:
        spark_sql = raw  # fall back to original SQL if transpilation fails

    if into_match:
        clean_sql = re.sub(r"\s*INTO\s+#[\w]+", "", spark_sql, flags=re.IGNORECASE).strip()
        view_name = alias.lstrip("#")
        sql, pre  = _handle_nullable_filters(clean_sql)
        return _build_spark_sql(sql, var=f"_{view_name}_df",
                                view_name=view_name, pre_lines=pre)

    sql, pre = _handle_nullable_filters(spark_sql)
    return _build_spark_sql(sql, pre_lines=pre)
 

 
 
# ──────────────────────────────────────────────────────────────────────────────
#  INSERT / UPDATE / DELETE / MERGE
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_insert(stmt: SQLStatement, db_prefix: str = "my_db") -> str:
    raw = stmt.raw_sql.strip()
    # INSERT INTO table SELECT ...
    m = re.match(
        r"INSERT\s+(?:INTO\s+)?([#\w\.]+)(?:\s*\([^)]*\))?\s*(SELECT.+)",
        raw, re.IGNORECASE | re.DOTALL
    )
    if m:
        target   = m.group(1).strip()
        sel_part = m.group(2).strip().rstrip(";")

        # Pretty-print the SELECT using sqlglot for consistent indentation
        if _HAS_SQLGLOT:
            try:
                transpiled = sqlglot.transpile(sel_part, read="tsql", write="spark",
                                               pretty=True, error_level="raise")
                sel_part = transpiled[0] if transpiled else sel_part
            except Exception:
                pass  # keep original if transpilation fails

        sel_sql, pre = _handle_nullable_filters(sel_part)

        if target.startswith("#"):
            view_name = target.lstrip("#")
            var_name  = _clean_name(view_name) + "_df"
            return _build_spark_sql(sel_sql, var=var_name,
                                    view_name=view_name, pre_lines=pre)
        else:
            sel_clean, has_params = _substitute_params(sel_sql)
            q = "f" if (has_params or pre) else ""
            pre_block = "\n".join(pre) + "\n" if pre else ""
            return (
                f"# INSERT INTO {target}\n"
                f"{pre_block}"
                f'_insert_df = spark.sql({q}"""\n    {sel_clean}\n""")\n'
                f'_insert_df.write.format("delta").mode("append").saveAsTable("{db_prefix}.{target}")'
            )

    # INSERT INTO table VALUES (...)
    raw_clean, has_params = _substitute_params(raw.rstrip(";"))
    q = "f" if has_params else ""
    return (
        f"# INSERT with VALUES\n"
        f"# NOTE: For bulk inserts prefer reading from a DataFrame.\n"
        f'spark.sql({q}"""\n    {raw_clean}\n""")'
    )


def convert_update(stmt: SQLStatement, db_prefix: str = "my_db") -> str:
    raw = stmt.raw_sql.strip()
    m   = re.match(
        r"UPDATE\s+([\w\.#]+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$",
        raw, re.IGNORECASE | re.DOTALL
    )
    if m:
        table     = m.group(1).strip()
        set_part  = m.group(2).strip()
        where_part= (m.group(3) or "").strip().rstrip(";")
        is_temp   = table.startswith("#")
        view_name = table.lstrip("#")
 
        # Parse SET col = value pairs
        set_pairs = {}
        for pair in re.split(r",(?![^(]*\))", set_part):
            if "=" in pair:
                col, val = pair.split("=", 1)
                col_py = col.strip().strip("[]")
                val_py = val.strip().rstrip(";")
                set_pairs[col_py] = f'F.expr("{val_py}")'
 
        set_dict_str = "{\n" + "".join(f'    "{k}": {v},\n' for k, v in set_pairs.items()) + "}"
 
        if is_temp:
            # Temp table → recreate view after update
            lines = [
                f"# UPDATE {table}",
                f'_upd_df = spark.table("{view_name}")',
            ]
            if where_part:
                lines.append(f'_upd_df = _upd_df.filter("{where_part}")')
            for col, val in set_pairs.items():
                lines.append(f'_upd_df = _upd_df.withColumn("{col}", {val})')
            lines.append(f'_upd_df.createOrReplaceTempView("{view_name}")')
            return "\n".join(lines)
        else:
            cond_str = f'"{where_part}"' if where_part else "None"
            return (
                f"# UPDATE {table} → Delta Lake DeltaTable.update()\n"
                f'from delta.tables import DeltaTable\n'
                f'_dt = DeltaTable.forName(spark, "{db_prefix}.{table}")\n'
                f'_dt.update(\n'
                f'    condition = F.expr({cond_str}),\n'
                f'    set       = {set_dict_str}\n'
                f')'
            )
    return f'# TODO (UPDATE): spark.sql("""{raw}""")'
 
 
def convert_delete(stmt: SQLStatement, db_prefix: str = "my_db") -> str:
    raw = stmt.raw_sql.strip()
    m   = re.match(
        r"DELETE\s+(?:FROM\s+)?([\w\.#]+)(?:\s+WHERE\s+(.+))?$",
        raw, re.IGNORECASE | re.DOTALL
    )
    if m:
        table      = m.group(1).strip()
        where_part = (m.group(2) or "").strip().rstrip(";")
        cond_str   = f'F.expr("{where_part}")' if where_part else "None"
        return (
            f"# DELETE FROM {table}\n"
            f"from delta.tables import DeltaTable\n"
            f'_dt = DeltaTable.forName(spark, "{db_prefix}.{table}")\n'
            f'_dt.delete(condition={cond_str})'
        )
    return f'# TODO (DELETE): spark.sql("""{raw}""")'
 
 
def convert_merge(stmt: SQLStatement, db_prefix: str = "my_db") -> str:
    raw = stmt.raw_sql.strip()
    m_target = re.search(r"MERGE\s+(?:INTO\s+)?([\w\.]+)\s+(?:AS\s+)?(\w+)?", raw, re.IGNORECASE)
    target   = m_target.group(1) if m_target else "target_table"
    return (
        f"# MERGE INTO {target}\n"
        f"# Delta Lake DeltaTable.merge() — replace conditions below with actual logic\n"
        f"from delta.tables import DeltaTable\n"
        f'_target = DeltaTable.forName(spark, "{db_prefix}.{target}")\n'
        f'_source = spark.table("{db_prefix}.source_table")  # TODO: replace with actual source\n'
        f"(\n"
        f'    _target.alias("tgt")\n'
        f'    .merge(_source.alias("src"), "tgt.id = src.id")  # TODO: merge condition\n'
        f'    .whenMatchedUpdateAll()\n'
        f'    .whenNotMatchedInsertAll()\n'
        f'    .execute()\n'
        f")"
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Control Flow
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_if_else(stmt: SQLStatement) -> str:
    raw = stmt.raw_sql.strip()
    # Extract condition between IF and BEGIN
    m = re.match(
        r"IF\s+(.+?)\s*BEGIN(.+?)END(?:\s+ELSE\s+BEGIN(.+?)END)?",
        raw, re.IGNORECASE | re.DOTALL
    )
    if m:
        condition  = m.group(1).strip()
        true_block = m.group(2).strip()
        else_block = (m.group(3) or "").strip()
        py_cond    = _translate_sql_condition(condition)
        lines      = [f"if {py_cond}:"]
        for line in true_block.split("\n"):
            lines.append(f"    {line}")
        if else_block:
            lines.append("else:")
            for line in else_block.split("\n"):
                lines.append(f"    {line}")
        return "\n".join(lines)
 
    # Inline IF (no BEGIN/END)
    m2 = re.match(r"IF\s+(.+?)\s*\n(.+)", raw, re.IGNORECASE | re.DOTALL)
    if m2:
        condition  = m2.group(1).strip()
        body       = m2.group(2).strip()
        py_cond    = _translate_sql_condition(condition)
        return f"if {py_cond}:\n    {body}"
 
    return (
        f"# SQL IF/ELSE — manual conversion required\n"
        f"# Original:\n"
        + "\n".join(f"# {l}" for l in raw.split("\n"))
    )
 
 
def _translate_sql_condition(cond: str) -> str:
    """Translate a SQL boolean condition to Python."""
    cond = cond.strip()
    # @variable → python_variable
    cond = re.sub(r"@([\w]+)", lambda m: m.group(1).lower(), cond)
    # IS NULL / IS NOT NULL
    cond = re.sub(r"IS\s+NULL",     "is None",     cond, flags=re.IGNORECASE)
    cond = re.sub(r"IS\s+NOT\s+NULL","is not None", cond, flags=re.IGNORECASE)
    # SQL AND/OR/NOT → Python
    cond = re.sub(r"\bAND\b", "and", cond, flags=re.IGNORECASE)
    cond = re.sub(r"\bOR\b",  "or",  cond, flags=re.IGNORECASE)
    cond = re.sub(r"\bNOT\b", "not", cond, flags=re.IGNORECASE)
    # EXISTS(SELECT ...) → True (placeholder)
    cond = re.sub(r"\bEXISTS\s*\(.+?\)", "True  # TODO: replace EXISTS check",
                  cond, flags=re.IGNORECASE | re.DOTALL)
    # (SELECT COUNT(*) ...) > 0
    m_cnt = re.search(r"\(\s*SELECT\s+COUNT\s*\(\*\)\s+FROM\s+(\w+)\s*\)", cond, re.IGNORECASE)
    if m_cnt:
        table = m_cnt.group(1)
        cond  = re.sub(r"\(\s*SELECT\s+COUNT\s*\(\*\).*?\)", f"spark.table('{table}').count()",
                       cond, flags=re.IGNORECASE | re.DOTALL)
    return cond
 
 
def convert_while(stmt: SQLStatement) -> str:
    raw = stmt.raw_sql.strip()
    m   = re.match(r"WHILE\s+(.+?)\s*BEGIN(.+?)END", raw, re.IGNORECASE | re.DOTALL)
    if m:
        condition  = _translate_sql_condition(m.group(1).strip())
        body       = m.group(2).strip()
        lines      = [
            "# ⚠ WARNING: SQL WHILE loop detected.",
            "# In PySpark, prefer vectorized transformations over loops.",
            "# The loop below is a direct translation — consider refactoring.",
            f"while {condition}:",
        ]
        for line in body.split("\n"):
            lines.append(f"    {line.strip()}")
        return "\n".join(lines)
    return f"# TODO (WHILE): {raw}"
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Cursor conversion — intent-aware, proper PySpark patterns
#
#  SQL cursors (RDBMS) vs PySpark — key differences:
#  ┌─────────────────────┬───────────────────────────────────────────────────┐
#  │ SQL Cursor          │ PySpark Equivalent                                │
#  ├─────────────────────┼───────────────────────────────────────────────────┤
#  │ DECLARE + FOR SELECT│ DataFrame: spark.sql() / spark.table()            │
#  │ FETCH NEXT INTO @v  │ _row.col_name  (within foreachPartition)          │
#  │ WHILE @@FETCH_STATUS│ foreachPartition(func)  — runs on Spark workers   │
#  │ Row-by-row UPDATE   │ DeltaTable.merge() / df.withColumn()  (vectorized)│
#  │ Row-by-row INSERT   │ df.write.format("delta").mode("append")           │
#  │ Row-by-row transform│ df.withColumn() / pandas_udf                      │
#  │ Side-effect per row │ df.foreachPartition(func)                         │
#  │ collect() on driver │ AVOID — foreachPartition runs on executors        │
#  └─────────────────────┴───────────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────────

class _CursorInfo:
    """Structured data extracted from a SQL CURSOR block."""
    __slots__ = (
        "name", "df_var", "select_sql", "fetch_vars",
        "while_body", "has_update", "has_insert",
        "has_delete", "has_exec", "update_table", "insert_table",
    )

    def __init__(self):
        self.name: str          = "cur"
        self.df_var: str        = "cur_df"
        self.select_sql: str    = "SELECT * FROM source_table  -- TODO: replace"
        self.fetch_vars: list   = []   # ["var1", "var2"]  (stripped of @)
        self.while_body: str    = ""   # body of WHILE @@FETCH_STATUS loop
        self.has_update: bool   = False
        self.has_insert: bool   = False
        self.has_delete: bool   = False
        self.has_exec: bool     = False
        self.update_table: str  = ""
        self.insert_table: str  = ""


def _parse_cursor_info(raw_sql: str) -> _CursorInfo:
    """Extract all useful parts from a DECLARE CURSOR or WHILE @@FETCH_STATUS block."""
    info = _CursorInfo()

    # Cursor name
    m = re.search(r"DECLARE\s+([\w]+)\s+CURSOR", raw_sql, re.IGNORECASE)
    if m:
        info.name = m.group(1)

    info.df_var = f"{info.name.lower()}_df"

    # FOR SELECT clause  (in DECLARE block)
    m = re.search(
        r"\bFOR\s+(SELECT\b.+?)(?=\bOPEN\b|\bFETCH\b|\bWHILE\b|\bCLOSE\b|$)",
        raw_sql, re.IGNORECASE | re.DOTALL,
    )
    if m:
        info.select_sql = m.group(1).strip().rstrip(";")

    # FETCH NEXT … INTO @var1, @var2
    m = re.search(
        r"FETCH\s+NEXT\s+FROM\s+\w+\s+INTO\s+([@\w\s,]+)",
        raw_sql, re.IGNORECASE,
    )
    if m:
        info.fetch_vars = [
            v.strip().lstrip("@").lower()
            for v in m.group(1).split(",")
        ]

    # WHILE @@FETCH_STATUS = 0 BEGIN … END body
    m = re.search(
        r"WHILE\s+@@FETCH_STATUS\s*=\s*0\s*BEGIN(.+?)END",
        raw_sql, re.IGNORECASE | re.DOTALL,
    )
    if m:
        body = m.group(1)
        # Strip FETCH NEXT lines (loop-increment in SQL)
        body = re.sub(
            r"FETCH\s+NEXT\s+FROM\s+\w+[^\n;]*[;\n]?", "",
            body, flags=re.IGNORECASE,
        )
        info.while_body = body.strip()

    # Detect intent from body
    body_up = info.while_body.upper()
    info.has_update = bool(re.search(r"\bUPDATE\b", body_up))
    info.has_insert = bool(re.search(r"\bINSERT\b", body_up))
    info.has_delete = bool(re.search(r"\bDELETE\b", body_up))
    info.has_exec   = bool(re.search(r"\bEXEC\b|\bEXECUTE\b", body_up))

    # Try to extract target table for UPDATE / INSERT
    if info.has_update:
        mu = re.search(r"UPDATE\s+([\w\.#]+)\s+SET", info.while_body, re.IGNORECASE)
        if mu:
            info.update_table = mu.group(1)

    if info.has_insert:
        mi = re.search(r"INSERT\s+INTO\s+([\w\.#]+)", info.while_body, re.IGNORECASE)
        if mi:
            info.insert_table = mi.group(1)

    return info


def _indent_body(body: str, indent: int = 8) -> list[str]:
    """Indent each line of a multi-line body for embedding in generated code."""
    pad = " " * indent
    return [f"{pad}# {line}" if line.strip() else "" for line in body.split("\n")]


def convert_cursor(raw_sql: str) -> str:
    """
    Convert a SQL CURSOR block to idiomatic PySpark.

    Strategy (by detected body intent):
      UPDATE in body  → DeltaTable.merge() vectorized + foreachPartition fallback
      INSERT in body  → df.write.format("delta") vectorized + foreachPartition fallback
      Side effects    → df.foreachPartition()  (executes on workers, NOT driver)
      Pure transforms → df.withColumn() hint  + foreachPartition fallback

    NOTE: foreachPartition() runs on Spark executors — it does NOT bring data
    to the driver like collect() does.  For large tables always prefer the
    fully vectorized approach (merge / withColumn / write).
    """
    info = _parse_cursor_info(raw_sql)

    divider = "# " + "─" * 70
    lines = [
        divider,
        f"# CURSOR → PySpark  |  cursor: '{info.name}'",
        divider,
        "#",
        "# SQL Cursor: row-by-row, sequential, executes on DB server.",
        "# PySpark   : distributed, columnar — NEVER use collect() on large data.",
        "#",
    ]

    # ── STEP 1: Load cursor source as DataFrame ──────────────────────────────
    lines += [
        "# STEP 1: Load cursor SELECT as a Spark DataFrame (replaces DECLARE … FOR SELECT)",
        f'{info.df_var} = spark.sql("""',
        f"    {info.select_sql.replace(chr(10), chr(10) + '    ')}",
        '""")',
        "",
    ]

    # ── STEP 2: Vectorized approach (preferred) ───────────────────────────────
    if info.has_update:
        lines += _cursor_vectorized_update(info)
    elif info.has_insert:
        lines += _cursor_vectorized_insert(info)
    else:
        lines += _cursor_vectorized_transform(info)

    # ── STEP 3: foreachPartition fallback ─────────────────────────────────────
    lines += _cursor_foreach_fallback(info)

    return "\n".join(lines)


def _cursor_vectorized_update(info: _CursorInfo) -> list[str]:
    target = info.update_table or "db.target_table  # TODO: set correct table"
    key    = info.fetch_vars[0] if info.fetch_vars else "id"
    return [
        "# STEP 2 — PREFERRED VECTORIZED APPROACH (replaces WHILE @@FETCH_STATUS loop)",
        "# Cursor body performs UPDATE → use DeltaTable.merge() or withColumn()",
        "# This runs fully distributed — no row-by-row loop needed.",
        "#",
        "# from delta.tables import DeltaTable",
        f"# _target_dt = DeltaTable.forName(spark, '{target}')",
        "# (",
        f"#     _target_dt.alias('tgt')",
        f"#     .merge({info.df_var}.alias('src'), 'tgt.{key} = src.{key}')  # TODO: merge key",
        "#     .whenMatchedUpdateAll()   # or .whenMatchedUpdate(set={{...}})",
        "#     .whenNotMatchedInsertAll()",
        "#     .execute()",
        "# )",
        "#",
        "# Alternative — column-level update without merge:",
        f"# {info.df_var} = {info.df_var}.withColumn('col', F.expr('new_value'))  # TODO",
        "#",
    ]


def _cursor_vectorized_insert(info: _CursorInfo) -> list[str]:
    target = info.insert_table or "db.target_table  # TODO: set correct table"
    return [
        "# STEP 2 — PREFERRED VECTORIZED APPROACH (replaces WHILE @@FETCH_STATUS loop)",
        "# Cursor body performs INSERT → write entire DataFrame to Delta in one operation.",
        "# This is dramatically faster than row-by-row inserts.",
        "#",
        f"# {info.df_var} = {info.df_var}.withColumn(",
        "#     'category',  # TODO: add any CASE WHEN transformations here",
        "#     F.when(F.col('salary') > 50000, F.lit('Senior')).otherwise(F.lit('Junior'))",
        "# )",
        f"# {info.df_var}.write.format('delta').mode('append').saveAsTable('{target}')",
        "#",
    ]


def _cursor_vectorized_transform(info: _CursorInfo) -> list[str]:
    return [
        "# STEP 2 — PREFERRED VECTORIZED APPROACH (replaces WHILE @@FETCH_STATUS loop)",
        "# Cursor body performs row-level transformations.",
        "# Replace with withColumn() chains or a pandas_udf for complex logic.",
        "#",
        f"# {info.df_var} = (",
        f"#     {info.df_var}",
        "#     .withColumn('result_col', F.expr('...'))  # TODO: translate body logic",
        "#     .withColumn('flag', F.when(F.col('value') > 0, F.lit(1)).otherwise(F.lit(0)))",
        "# )",
        "#",
        "# For complex per-row logic use a pandas_udf (vectorized UDF):",
        "# from pyspark.sql.functions import pandas_udf",
        "# import pandas as pd",
        "# @pandas_udf('string')",
        "# def transform_func(s: pd.Series) -> pd.Series:",
        "#     return s.apply(lambda x: ...)  # TODO",
        f"# {info.df_var} = {info.df_var}.withColumn('result', transform_func(F.col('col')))",
        "#",
    ]


def _cursor_foreach_fallback(info: _CursorInfo) -> list[str]:
    """Generate foreachPartition fallback (always better than collect())."""
    fn = f"_process_{info.name.lower()}"

    # Build row-field access lines from FETCH INTO variables
    if info.fetch_vars:
        field_lines = [
            f"        {v} = _row['{v}']  # SQL: FETCH NEXT … INTO @{v}"
            for v in info.fetch_vars
        ]
    else:
        field_lines = [
            "        # Access row fields as: _row['column_name']  or  _row.column_name",
        ]

    # Embed original body as comments
    body_comments = _indent_body(info.while_body, indent=8) if info.while_body else [
        "        # (cursor body not detected in this block)",
    ]

    lines = [
        "# STEP 3 — FALLBACK: foreachPartition  (use when vectorized approach is not feasible)",
        "# foreachPartition runs on Spark EXECUTORS (workers), NOT on the driver.",
        "# Each partition is processed independently — still row-by-row within a partition,",
        "# but distributed across the cluster. Far better than collect() for large data.",
        "#",
        "# Key difference from SQL Cursor:",
        "#   SQL Cursor  : single thread on DB server, sequential row access",
        "#   foreachPartition: parallel across Spark workers, one function call per partition",
        "#",
        f"def {fn}(partition):",
        f'    """',
        f"    Processes one Spark partition.",
        f"    Equivalent to: WHILE @@FETCH_STATUS = 0 BEGIN ... END",
        f"    Runs on executor nodes — do NOT reference SparkContext here.",
        f'    """',
    ]

    if info.has_exec:
        lines += [
            "    # External call per row (EXEC in original cursor)",
            "    # Use a connection pool or batch the calls for performance.",
        ]

    lines += [
        "    for _row in partition:",
        "        # ── Map FETCH INTO @variables ─────────────────────────────",
        *field_lines,
        "        # ── Original cursor body (translate below) ────────────────",
        *body_comments,
        "        # TODO: implement logic above using _row fields",
        "",
        f"# Execute — partitions run in parallel across Spark cluster:",
        f"{info.df_var}.foreachPartition({fn})",
        "",
        "# ── REMINDER: Vectorized alternatives are always preferred ────────────────",
        f"# {info.df_var}.withColumn(...)            → column transformation",
        f"# {info.df_var}.groupBy(...).agg(...)      → aggregation",
        f"# DeltaTable.merge(...)                    → conditional UPDATE/INSERT",
        f"# {info.df_var}.write.format('delta')...  → bulk INSERT",
    ]

    return lines
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Temp table / DROP TABLE
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_create_temp(stmt: SQLStatement) -> str:
    raw = stmt.raw_sql.strip()
    m   = re.search(r"CREATE\s+TABLE\s+(#[\w]+)", raw, re.IGNORECASE)
    if m:
        name = m.group(1).lstrip("#")
        return f'# NOTE: {m.group(1)} → Spark temp view "{name}" (created by createOrReplaceTempView below)'
    return f"# CREATE TABLE (temp): {raw[:80]}..."
 
 
def convert_drop_temp(stmt: SQLStatement) -> str:
    raw = stmt.raw_sql.strip()
    m   = re.search(r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(#[\w]+)", raw, re.IGNORECASE)
    if m:
        name = m.group(1).lstrip("#")
        return f'spark.catalog.dropTempView("{name}")'
    return f"# DROP TABLE: {raw[:80]}..."
 
 
def convert_print(stmt: SQLStatement) -> str:
    raw = stmt.raw_sql.strip()
    m   = re.match(r"PRINT\s+'?(.+?)'?\s*$", raw, re.IGNORECASE)
    msg = m.group(1) if m else raw.replace("PRINT ", "")
    return f'print("{msg}")  # SQL: {raw}'
 
 
def convert_exec(stmt: SQLStatement) -> str:
    raw = stmt.raw_sql.strip()
    return (
        f"# EXEC / Dynamic SQL detected — manual review required.\n"
        f'# spark.sql("""TODO: translate dynamic SQL""")  # Original: {raw[:100]}'
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Universal fallback — handles any unrecognised or partial SQL
# ──────────────────────────────────────────────────────────────────────────────
 
def convert_unknown(stmt: SQLStatement) -> str:
    """
    Catch-all converter for any SQL that could not be classified.
    Strategy:
      1. Try sqlglot to transpile to Spark SQL dialect.
      2. If that fails, wrap the raw text in spark.sql() as-is.
    Either way the user always gets runnable PySpark output.
    """
    raw = stmt.raw_sql.strip().rstrip(";")
    if not raw:
        return ""
 
    # ── Try sqlglot transpilation to Spark SQL ────────────────────────────────
    transpiled = _try_sqlglot_transpile(raw)
 
    if transpiled:
        lines = [
            "# Auto-transpiled to Spark SQL dialect",
            'result_df = spark.sql("""',
        ]
        for line in transpiled.split("\n"):
            lines.append(f"    {line}")
        lines.append('""")')
    else:
        # ── Raw passthrough — wrap as-is ──────────────────────────────────────
        lines = [
            "# Could not fully parse statement — using spark.sql() passthrough.",
            "# Review the query below; Spark SQL accepts most ANSI SQL syntax.",
            'result_df = spark.sql("""',
        ]
        for line in raw.split("\n"):
            lines.append(f"    {line}")
        lines.append('""")')
        lines.append("result_df.show()")
 
    return "\n".join(lines)
 
 
def _try_sqlglot_transpile(sql: str) -> str | None:
    """Attempt to transpile SQL to Spark dialect using sqlglot. Returns None on failure."""
    if not _HAS_SQLGLOT:
        return None
    try:
        import sqlglot
        results = sqlglot.transpile(sql, read="tsql", write="spark")
        if results and results[0].strip():
            return results[0].strip()
    except Exception:
        pass
    try:
        import sqlglot
        # Try ANSI dialect as second attempt
        results = sqlglot.transpile(sql, read="", write="spark")
        if results and results[0].strip():
            return results[0].strip()
    except Exception:
        pass
    return None
 
