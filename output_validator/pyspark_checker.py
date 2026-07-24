"""
Output validator — compares SQL results vs PySpark results (local Spark mode).
"""
 
import sys
from dataclasses import dataclass, field
 
 
@dataclass
class ValidationResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
 
    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "✓ PASS" if c["passed"] else "✗ FAIL"
            lines.append(f"  {status}  {c['name']}: {c['detail']}")
        overall = "ALL CHECKS PASSED" if self.passed else "SOME CHECKS FAILED"
        return f"[{overall}]\n" + "\n".join(lines)
 
 
class PySparkValidator:
    """
    Validates PySpark output against expected results.
    Runs in local Spark mode (no Databricks required).
    """
 
    def __init__(self, spark=None):
        self._spark = spark
 
    @property
    def spark(self):
        if self._spark is None:
            try:
                from pyspark.sql import SparkSession  # type: ignore[import-untyped]
                self._spark = (
                    SparkSession.builder
                    .appName("SQL_to_PySpark_Validator")
                    .master("local[2]")
                    .getOrCreate()
                )
                self._spark.sparkContext.setLogLevel("ERROR")
            except ImportError:
                raise RuntimeError(
                    "PySpark not installed. Run: pip install pyspark"
                )
        return self._spark
 
    def validate_row_count(self, df, expected: int, name: str = "row_count") -> dict:
        actual = df.count()
        return {
            "name":   name,
            "passed": actual == expected,
            "detail": f"expected={expected}, actual={actual}",
        }
 
    def validate_schema(self, df, expected_columns: list[str], name: str = "schema") -> dict:
        actual_cols = [c.lower() for c in df.columns]
        missing     = [c for c in expected_columns if c.lower() not in actual_cols]
        return {
            "name":   name,
            "passed": len(missing) == 0,
            "detail": f"missing columns: {missing}" if missing else "all columns present",
        }
 
    def validate_no_nulls(self, df, columns: list[str], name: str = "no_nulls") -> dict:
        from pyspark.sql import functions as F  # type: ignore[import-untyped]
        total_nulls = 0
        for col in columns:
            if col in df.columns:
                total_nulls += df.filter(F.col(col).isNull()).count()
        return {
            "name":   name,
            "passed": total_nulls == 0,
            "detail": f"{total_nulls} null values found in checked columns",
        }
 
    def validate_data_equality(self, df_actual, df_expected, name: str = "data_equality") -> dict:
        """Symmetric difference check — rows in one but not the other."""
        missing = df_expected.subtract(df_actual).count()
        extra   = df_actual.subtract(df_expected).count()
        passed  = (missing == 0 and extra == 0)
        return {
            "name":   name,
            "passed": passed,
            "detail": f"missing={missing} rows, extra={extra} rows",
        }
 
    def run_full_validation(self, df_actual, df_expected,
                            key_columns: list[str] = None) -> ValidationResult:
        checks = []
 
        # Row count
        exp_count = df_expected.count()
        checks.append(self.validate_row_count(df_actual, exp_count, "row_count"))
 
        # Schema
        expected_cols = df_expected.columns
        checks.append(self.validate_schema(df_actual, expected_cols, "schema"))
 
        # Data equality
        checks.append(self.validate_data_equality(df_actual, df_expected, "data_equality"))
 
        # No nulls in key columns
        if key_columns:
            checks.append(self.validate_no_nulls(df_actual, key_columns, "no_nulls_in_keys"))
 
        passed = all(c["passed"] for c in checks)
        return ValidationResult(passed=passed, checks=checks)
 