"""Quick pipeline smoke-test (not a pytest file — run directly)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from converter.sql_preprocessor import preprocess
from converter.code_postprocessor import postprocess

# ── Test 1: Preprocessor ──────────────────────────────────────────────────────
sql = "\n".join([
    "USE MyDB;",
    "GO",
    "SET NOCOUNT ON;",
    "CREATE PROCEDURE dbo.usp_GetEmps @dept_id INT = NULL",
    "AS BEGIN",
    "  SELECT [emp_id], [name], [salary]",
    "  FROM [dbo].[employees] e",
    "  INNER JOIN [dbo].[departments] d ON e.dept_id = d.id",
    "  WHERE e.dept_id = @dept_id ORDER BY [salary] DESC;",
    "END",
    "GO",
])

pre = preprocess(sql)
print("=== PREPROCESSOR ===")
print("SP name   :", pre.sp_name)
print("Params    :", pre.parameters)
print("Hints     :", pre.dialect_hints)
print("Is SP     :", pre.is_stored_procedure)
print("Cleaned SQL:")
print(pre.cleaned_sql)
assert "USE MyDB" not in pre.cleaned_sql,   "USE not stripped"
assert "GO" not in pre.cleaned_sql,          "GO not stripped"
assert "SET NOCOUNT" not in pre.cleaned_sql, "SET NOCOUNT not stripped"
assert "[emp_id]" not in pre.cleaned_sql,    "Brackets not stripped"
assert pre.sp_name == "dbo.usp_GetEmps",     "SP name not extracted"
print("PASS\n")

# ── Test 2: Postprocessor — strips fence + validates syntax ────────────────────
raw_with_fence = (
    "```python\n"
    "from pyspark.sql import SparkSession\n"
    "\n"
    "def run_query(spark: SparkSession):\n"
    "    result_df = spark.sql('SELECT * FROM my_db.employees')\n"
    "    print('Row count:', result_df.count())\n"
    "    result_df.show(10, truncate=False)\n"
    "    return result_df\n"
    "```\n"
    "\n"
    "### Explanation:\n"
    "1. The function does X.\n"
    "2. It uses spark.sql.\n"
)

post = postprocess(raw_with_fence)
print("=== POSTPROCESSOR ===")
print("Syntax valid:", post.syntax_valid)
print("Warnings    :", post.warnings)
print("Injected    :", post.imports_injected)
print("Code snippet:")
print(post.code[:200])
assert "```" not in post.code,            "Fence not stripped"
assert "Explanation" not in post.code,    "Prose not stripped"
assert post.syntax_valid,                 "Syntax invalid"
assert "from pyspark.sql import SparkSession" in post.code
print("PASS\n")

# ── Test 3: Postprocessor — injects missing import ─────────────────────────────
raw_no_import = (
    "def run_query(spark):\n"
    "    result_df = spark.sql('SELECT 1')\n"
    "    return result_df\n"
)
post2 = postprocess(raw_no_import)
print("=== IMPORT INJECTION ===")
print("Injected:", post2.imports_injected)
assert "from pyspark.sql import SparkSession" in post2.code, "SparkSession import not injected"
print("PASS\n")

# ── Test 4: Postprocessor — detects syntax error ───────────────────────────────
bad_code = "def bad(spark:\n    return spark.sql('SELECT 1'\n"
post3 = postprocess(bad_code)
print("=== SYNTAX ERROR DETECTION ===")
print("Syntax valid :", post3.syntax_valid)
print("Syntax error :", post3.syntax_error)
assert not post3.syntax_valid, "Should have detected syntax error"
print("PASS\n")

print("All pipeline tests PASSED.")
