"""
SQL Function → PySpark Function Mapping Rules
Step 2 of the AI Framework Pipeline: map SQL functions to Spark equivalents.
"""
 
# Scalar function mappings: SQL_FUNC → (pyspark_expr, notes)
SCALAR_FUNCTION_MAP: dict[str, tuple[str, str]] = {
    # Null handling
    "ISNULL":           ("F.coalesce",          "ISNULL(col, default) → F.coalesce(F.col('col'), F.lit(default))"),
    "NULLIF":           ("F.nullif",             "Direct equivalent"),
    "COALESCE":         ("F.coalesce",           "Direct equivalent"),
    "NVL":              ("F.coalesce",           "Oracle NVL → F.coalesce"),
    # String functions
    "LEN":              ("F.length",             "LEN(col) → F.length(F.col('col'))"),
    "LENGTH":           ("F.length",             "Direct equivalent"),
    "SUBSTRING":        ("F.substring",          "SUBSTRING(col, start, len) → F.substring(col, start, len)"),
    "SUBSTR":           ("F.substring",          "Direct equivalent"),
    "LEFT":             ("F.substring",          "LEFT(col, n) → F.substring(col, 1, n)"),
    "RIGHT":            ("F.expr",               "RIGHT(col, n) → F.expr(\"right(col, n)\")"),
    "UPPER":            ("F.upper",              "Direct equivalent"),
    "LOWER":            ("F.lower",              "Direct equivalent"),
    "LTRIM":            ("F.ltrim",              "Direct equivalent"),
    "RTRIM":            ("F.rtrim",              "Direct equivalent"),
    "TRIM":             ("F.trim",               "Direct equivalent"),
    "REPLACE":          ("F.regexp_replace",     "REPLACE(col, from, to) → F.regexp_replace(col, from, to)"),
    "CHARINDEX":        ("F.instr",              "CHARINDEX(search, col) → F.instr(col, search) [arg order reversed!]"),
    "PATINDEX":         ("F.regexp_extract",     "PATINDEX('%pattern%', col) → use F.regexp_extract"),
    "CONCAT":           ("F.concat",             "Direct equivalent"),
    "CONCAT_WS":        ("F.concat_ws",          "Direct equivalent"),
    "STRING_AGG":       ("F.concat_ws",          "STRING_AGG(col, sep) → F.concat_ws(sep, F.collect_list(col))"),
    "SPACE":            ("F.expr",               "SPACE(n) → F.expr(\"repeat(' ', n)\")"),
    "REPLICATE":        ("F.expr",               "REPLICATE(str, n) → F.expr(\"repeat(str, n)\")"),
    "REVERSE":          ("F.reverse",            "Direct equivalent"),
    "STUFF":            ("F.overlay",            "STUFF(col, start, len, new) → F.overlay(col, new, start, len)"),
    "FORMAT":           ("F.format_string",      "FORMAT(value, format) → F.format_string or F.date_format"),
    "STR":              ("F.format_string",      "STR(num) → F.col('num').cast(StringType())"),
    "UNICODE":          ("F.expr",               "UNICODE(col) → F.expr(\"ascii(col)\")"),
    "CHAR":             ("F.chr",                "CHAR(n) → F.chr(n)"),
    "ASCII":            ("F.ascii",              "Direct equivalent"),
    # Date/Time functions
    "GETDATE":          ("F.current_timestamp()", "GETDATE() → F.current_timestamp()"),
    "GETUTCDATE":       ("F.current_timestamp()", "GETUTCDATE() → F.current_timestamp()"),
    "NOW":              ("F.current_timestamp()", "NOW() → F.current_timestamp()"),
    "SYSDATE":          ("F.current_timestamp()", "Oracle SYSDATE → F.current_timestamp()"),
    "CURRENT_TIMESTAMP":("F.current_timestamp()", "Direct equivalent"),
    "CURRENT_DATE":     ("F.current_date()",     "Direct equivalent"),
    "DATEADD":          ("F.date_add",           "DATEADD(unit, n, col) — unit must be 'day' for F.date_add; use F.expr for others"),
    "DATEDIFF":         ("F.datediff",           "DATEDIFF(end, start) → F.datediff(F.col('end'), F.col('start'))"),
    "YEAR":             ("F.year",               "Direct equivalent"),
    "MONTH":            ("F.month",              "Direct equivalent"),
    "DAY":              ("F.dayofmonth",         "DAY(col) → F.dayofmonth(col)"),
    "DAYOFWEEK":        ("F.dayofweek",          "Direct equivalent"),
    "DAYOFYEAR":        ("F.dayofyear",          "Direct equivalent"),
    "HOUR":             ("F.hour",               "Direct equivalent"),
    "MINUTE":           ("F.minute",             "Direct equivalent"),
    "SECOND":           ("F.second",             "Direct equivalent"),
    "DATEPART":         ("F.extract",            "DATEPART(unit, col) → F.extract('unit', col) or use F.year/month/etc."),
    "DATENAME":         ("F.date_format",        "DATENAME(unit, col) → F.date_format(col, format_str)"),
    "EOMONTH":          ("F.last_day",           "EOMONTH(col) → F.last_day(col)"),
    "TO_DATE":          ("F.to_date",            "Direct equivalent"),
    "TO_TIMESTAMP":     ("F.to_timestamp",       "Direct equivalent"),
    "DATE_FORMAT":      ("F.date_format",        "Direct equivalent"),
    # Numeric functions
    "ABS":              ("F.abs",                "Direct equivalent"),
    "CEILING":          ("F.ceil",               "CEILING(col) → F.ceil(col)"),
    "CEIL":             ("F.ceil",               "Direct equivalent"),
    "FLOOR":            ("F.floor",              "Direct equivalent"),
    "ROUND":            ("F.round",              "Direct equivalent"),
    "POWER":            ("F.pow",                "POWER(col, n) → F.pow(col, n)"),
    "SQRT":             ("F.sqrt",               "Direct equivalent"),
    "LOG":              ("F.log",                "Direct equivalent"),
    "LOG10":            ("F.log",                "LOG10(col) → F.log(10.0, col)"),
    "EXP":              ("F.exp",                "Direct equivalent"),
    "SIGN":             ("F.signum",             "SIGN(col) → F.signum(col)"),
    "MOD":              ("F.mod",                "MOD(a, b) → F.col('a') % F.col('b')"),
    "RAND":             ("F.rand",               "RAND() → F.rand()"),
    # Aggregate functions
    "COUNT":            ("F.count",              "Direct equivalent"),
    "COUNT_BIG":        ("F.count",              "COUNT_BIG → F.count (returns LongType)"),
    "SUM":              ("F.sum",                "Direct equivalent"),
    "AVG":              ("F.avg",                "Direct equivalent"),
    "MAX":              ("F.max",                "Direct equivalent"),
    "MIN":              ("F.min",                "Direct equivalent"),
    "STDEV":            ("F.stddev",             "STDEV → F.stddev"),
    "STDEVP":           ("F.stddev_pop",         "STDEVP → F.stddev_pop"),
    "VAR":              ("F.variance",           "VAR → F.variance"),
    "VARP":             ("F.var_pop",            "VARP → F.var_pop"),
    # Window functions
    "ROW_NUMBER":       ("F.row_number()",       "Use with Window.partitionBy().orderBy()"),
    "RANK":             ("F.rank()",             "Use with Window.orderBy()"),
    "DENSE_RANK":       ("F.dense_rank()",       "Use with Window.orderBy()"),
    "NTILE":            ("F.ntile",              "NTILE(n) → F.ntile(n)"),
    "LAG":              ("F.lag",                "LAG(col, offset, default) → F.lag(col, offset, default)"),
    "LEAD":             ("F.lead",               "LEAD(col, offset, default) → F.lead(col, offset, default)"),
    "FIRST_VALUE":      ("F.first",              "FIRST_VALUE(col) → F.first(col)"),
    "LAST_VALUE":       ("F.last",               "LAST_VALUE(col) → F.last(col)"),
    "CUME_DIST":        ("F.cume_dist()",        "Direct equivalent"),
    "PERCENT_RANK":     ("F.percent_rank()",     "Direct equivalent"),
    # Type conversion
    "CAST":             (".cast()",              "CAST(col AS type) → F.col('col').cast(SparkType())"),
    "CONVERT":          (".cast()",              "CONVERT(type, col) → F.col('col').cast(SparkType())"),
    "TRY_CAST":         (".cast()",              "TRY_CAST → use .cast() (Spark returns null on failure by default)"),
    "PARSE":            ("F.to_date",            "PARSE(col AS DATE) → F.to_date(col, format)"),
    # Conditional
    "IIF":              ("F.when",               "IIF(cond, true_val, false_val) → F.when(cond, true_val).otherwise(false_val)"),
    "CHOOSE":           ("F.when",               "CHOOSE(n, val1, val2,...) → chained F.when().when().otherwise()"),
    # JSON/XML
    "JSON_VALUE":       ("F.get_json_object",    "JSON_VALUE(col, path) → F.get_json_object(col, '$.path')"),
    "JSON_QUERY":       ("F.get_json_object",    "Direct equivalent via F.get_json_object"),
    "OPENJSON":         ("F.from_json",          "OPENJSON → F.from_json(col, schema)"),
}
 
# Aggregate functions set (for identifying them in GROUP BY context)
AGGREGATE_FUNCTIONS = {
    "COUNT", "COUNT_BIG", "SUM", "AVG", "MAX", "MIN",
    "STDEV", "STDEVP", "VAR", "VARP",
    "STRING_AGG", "XMLAGG", "LISTAGG",
    "FIRST_VALUE", "LAST_VALUE", "ARRAY_AGG",
}
 
# Window functions set
WINDOW_FUNCTIONS = {
    "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE",
    "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE",
    "CUME_DIST", "PERCENT_RANK",
}
 
 
def lookup_function(name: str) -> tuple[str, str] | None:
    """Return (pyspark_equivalent, note) for a SQL function name, or None."""
    return SCALAR_FUNCTION_MAP.get(name.upper())
 