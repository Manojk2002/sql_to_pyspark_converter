"""
SQL-to-PySpark Type Mapping Rules
Step 2 of the AI Framework Pipeline: map SQL types to Spark types.
"""
 
# SQL Server / ANSI SQL → PySpark data types
SQL_TYPE_MAP: dict[str, str] = {
    # Integer types
    "INT":              "IntegerType()",
    "INTEGER":          "IntegerType()",
    "BIGINT":           "LongType()",
    "SMALLINT":         "ShortType()",
    "TINYINT":          "ByteType()",
    # Decimal / Numeric — needs precision/scale from original
    "DECIMAL":          "DecimalType",
    "NUMERIC":          "DecimalType",
    "MONEY":            "DecimalType(19, 4)",
    "SMALLMONEY":       "DecimalType(10, 4)",
    # Floating point
    "FLOAT":            "DoubleType()",
    "REAL":             "FloatType()",
    "DOUBLE":           "DoubleType()",
    "DOUBLE PRECISION": "DoubleType()",
    # String types
    "VARCHAR":          "StringType()",
    "NVARCHAR":         "StringType()",
    "CHAR":             "StringType()",
    "NCHAR":            "StringType()",
    "TEXT":             "StringType()",
    "NTEXT":            "StringType()",
    "SYSNAME":          "StringType()",
    "UNIQUEIDENTIFIER": "StringType()",
    # Date / Time
    "DATE":             "DateType()",
    "DATETIME":         "TimestampType()",
    "DATETIME2":        "TimestampType()",
    "SMALLDATETIME":    "TimestampType()",
    "TIME":             "StringType()",       # No native Time in Spark
    "TIMESTAMP":        "TimestampType()",
    # Boolean
    "BIT":              "BooleanType()",
    "BOOLEAN":          "BooleanType()",
    # Binary
    "VARBINARY":        "BinaryType()",
    "BINARY":           "BinaryType()",
    "IMAGE":            "BinaryType()",
    # Other
    "XML":              "StringType()",
    "JSON":             "StringType()",
    "GEOGRAPHY":        "StringType()",
    "GEOMETRY":         "StringType()",
}
 
# Python annotation equivalents (for function signatures)
SQL_TYPE_PYTHON_ANNOTATION: dict[str, str] = {
    "INT":              "int",
    "INTEGER":          "int",
    "BIGINT":           "int",
    "SMALLINT":         "int",
    "TINYINT":          "int",
    "DECIMAL":          "float",
    "NUMERIC":          "float",
    "MONEY":            "float",
    "SMALLMONEY":       "float",
    "FLOAT":            "float",
    "REAL":             "float",
    "DOUBLE":           "float",
    "VARCHAR":          "str",
    "NVARCHAR":         "str",
    "CHAR":             "str",
    "NCHAR":            "str",
    "TEXT":             "str",
    "DATE":             "str",    # pass as 'YYYY-MM-DD' string
    "DATETIME":         "str",
    "DATETIME2":        "str",
    "BIT":              "bool",
    "BOOLEAN":          "bool",
    "UNIQUEIDENTIFIER": "str",
}
 
 
def resolve_type(sql_type: str, precision: str = "", scale: str = "") -> str:
    """Resolve a SQL type string to its PySpark equivalent."""
    base = sql_type.strip().upper().split("(")[0].strip()
    mapped = SQL_TYPE_MAP.get(base, "StringType()")
 
    # Handle DecimalType with precision/scale
    if mapped == "DecimalType":
        if precision and scale:
            return f"DecimalType({precision}, {scale})"
        elif precision:
            return f"DecimalType({precision}, 0)"
        else:
            return "DecimalType(18, 2)"  # sensible default
 
    return mapped
 
 
def resolve_python_type(sql_type: str) -> str:
    """Resolve a SQL type to a Python type annotation string."""
    base = sql_type.strip().upper().split("(")[0].strip()
    return SQL_TYPE_PYTHON_ANNOTATION.get(base, "str")
 