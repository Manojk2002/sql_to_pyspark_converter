"""
AI Framework Pipeline — 7-Step SQL-to-PySpark Conversion Orchestrator.

This module is the central entry point for the full conversion pipeline.
Each step maps directly to the AI framework specification:

  Step 1: Analyze  — SQLParser + SQLAnalyzer → DDL/DML/control-flow/deps
  Step 2: Map      — SQL object → Spark equivalent mapping table
  Step 3: Rewrite  — DataFrame API code generation (SELECT/WHERE/JOIN/etc.)
  Step 4: Procedural — IF/ELSE, WHILE, cursors → Python/PySpark
  Step 5: Transactions — Delta ACID, temp tables → createOrReplaceTempView
  Step 6: Optimize — Databricks performance hints (broadcast, cache, Z-ORDER)
  Step 7: Validate — Row-count + schema validation scaffold

Usage:
    pipeline = ConversionPipeline(db_prefix="my_db")
    result   = pipeline.run(sql_text)
    print(result.pyspark_code)           # final generated code
    print(result.analysis_summary())     # human-readable report
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import date

from converter.sql_parser import SQLParser, ParsedSQL
from converter.sql_analyzer import SQLAnalyzer, AnalysisReport
from converter.code_generator import PySparkGenerator
from converter.spark_optimizer import DatabricksOptimizer


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Holds the output of every step in the 7-step pipeline."""

    # Input
    sql_input:      str = ""
    db_prefix:      str = "my_db"

    # Step 1 - Analysis
    parsed: ParsedSQL | None = None
    report: AnalysisReport | None = None

    # Step 2 — Object mapping (human-readable table)
    object_mapping: list[tuple[str, str, str]] = field(default_factory=list)
    # [(sql_object, object_type, spark_equivalent), ...]

    # Step 3-5 — Generated PySpark code (before optimisation)
    raw_pyspark_code: str = ""

    # Step 6 — Optimised code + optimisation notes
    pyspark_code:   str = ""
    optimizations:  list[str] = field(default_factory=list)

    # Step 7 — Validation scaffold (appended to generated code)
    validation_code: str = ""

    # Errors / warnings accumulated during the run
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def analysis_summary(self) -> str:
        """Return a formatted multi-section analysis report."""
        if self.parsed is None or self.report is None:
            return "No analysis available."

        p = self.parsed
        r = self.report
        lines = [
            "=" * 68,
            "  SQL → PySpark AI FRAMEWORK — Analysis Report",
            f"  Date: {date.today().isoformat()}",
            "=" * 68,
            "",
            "## STEP 1 — SQL ANALYSIS",
            f"  Source type       : {'Stored Procedure' if p.is_stored_procedure else 'SQL Query'}",
            f"  Name              : {p.sp_name or 'N/A'}",
            f"  Parameters        : {len(p.parameters)}",
            f"  Declared variables: {len(p.variables)}",
            f"  Statements        : {len(p.statements)}",
            f"  Complexity score  : {r.complexity_score}/10",
            "",
            "  — DDL statements —",
        ]
        if r.ddl_statements:
            for s in r.ddl_statements:
                lines.append(f"    • {s.stmt_type}: {s.raw_sql[:80].strip()}...")
        else:
            lines.append("    (none)")

        lines += ["", "  — DML statements —"]
        if r.dml_statements:
            for s in r.dml_statements:
                lines.append(f"    • {s.stmt_type}: {s.raw_sql[:80].strip()}...")
        else:
            lines.append("    (none)")

        lines += ["", "  — Control flow —"]
        if r.control_flow_statements:
            for s in r.control_flow_statements:
                lines.append(f"    • {s.stmt_type}: {s.raw_sql[:80].strip()}...")
        else:
            lines.append("    (none)")

        lines += [
            "",
            "## STEP 2 — SQL OBJECT MAPPING",
        ]
        if self.object_mapping:
            lines.append(f"  {'SQL Object':<30} {'Type':<18} {'Spark Equivalent'}")
            lines.append(f"  {'-'*30} {'-'*18} {'-'*30}")
            for sql_obj, obj_type, spark_eq in self.object_mapping:
                lines.append(f"  {sql_obj:<30} {obj_type:<18} {spark_eq}")
        else:
            lines.append("  (no objects mapped)")

        lines += [
            "",
            "## STEPS 3-5 — CONVERSION DETAILS",
            f"  Has cursors       : {r.has_cursors}",
            f"  Has transactions  : {r.has_transactions}",
            f"  Has dynamic SQL   : {r.has_dynamic_sql}",
            f"  Has window funcs  : {r.has_window_functions}",
            f"  CTE patterns      : {', '.join(r.cte_patterns) or 'none'}",
            f"  Temp tables       : {', '.join(t.name for t in p.temp_tables) or 'none'}",
            f"  Table deps        : {', '.join(r.dependencies) or 'none'}",
            "",
            "## STEP 6 — DATABRICKS OPTIMISATIONS APPLIED",
        ]
        if self.optimizations:
            for opt in self.optimizations:
                lines.append(f"  ✓ {opt}")
        else:
            lines.append("  (no additional optimisations)")

        lines += ["", "## CONVERSION WARNINGS"]
        if r.conversion_warnings:
            for w in r.conversion_warnings:
                lines.append(f"  ⚠  {w}")
        else:
            lines.append("  (no warnings)")

        if self.errors:
            lines += ["", "## ERRORS"]
            for e in self.errors:
                lines.append(f"  ✗ {e}")

        lines.append("")
        lines.append("=" * 68)
        return "\n".join(lines)

    def step_summary(self) -> dict:
        """Return a JSON-serialisable dict with one entry per step."""
        return {
            "step1_analysis": {
                "sp_name":          self.parsed.sp_name if self.parsed else None,
                "is_sp":            self.parsed.is_stored_procedure if self.parsed else False,
                "param_count":      len(self.parsed.parameters) if self.parsed else 0,
                "stmt_count":       len(self.parsed.statements) if self.parsed else 0,
                "complexity_score": self.report.complexity_score if self.report else 0,
            },
            "step2_mapping":   [
                {"sql": s, "type": t, "spark": e}
                for s, t, e in self.object_mapping
            ],
            "step3_5_code":    self.raw_pyspark_code,
            "step6_optimised": self.pyspark_code,
            "step6_opts":      self.optimizations,
            "step7_validation":self.validation_code,
            "warnings":        self.report.conversion_warnings if self.report else [],
            "errors":          self.errors,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class ConversionPipeline:
    """
    Orchestrates the full 7-step SQL-to-PySpark AI framework pipeline.

    Example:
        pipeline = ConversionPipeline(db_prefix="sales_db")
        result   = pipeline.run(sql_text)
        print(result.pyspark_code)
        print(result.analysis_summary())
    """

    def __init__(self, db_prefix: str = "my_db"):
        self.db_prefix  = db_prefix
        self._parser    = SQLParser()
        self._analyzer  = SQLAnalyzer()
        self._optimizer = DatabricksOptimizer()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, sql: str) -> PipelineResult:
        """
        Execute all 7 steps and return a PipelineResult.

        Args:
            sql: Raw SQL text (stored procedure or standalone query).

        Returns:
            PipelineResult with outputs from every step.
        """
        result = PipelineResult(sql_input=sql, db_prefix=self.db_prefix)

        # Step 1: Analyse
        result = self._step1_analyze(sql, result)
        if result.errors:
            return result

        # Step 2: Map SQL objects
        result = self._step2_map_objects(result)

        # Steps 3-5: Generate PySpark code
        result = self._step3_5_generate(result)

        # Step 6: Optimise for Databricks
        result = self._step6_optimize(result)

        # Step 7: Append validation scaffold
        result = self._step7_validate(result)

        return result

    # ── Step 1: Analyse ───────────────────────────────────────────────────────

    def _step1_analyze(self, sql: str, result: PipelineResult) -> PipelineResult:
        """
        Step 1 — Analyse the SQL.
        Break down into DDL, DML, control flow. Identify dependencies.
        """
        try:
            result.parsed = self._parser.parse(sql)
            result.report = self._analyzer.analyze(result.parsed)
            result.warnings.extend(result.report.conversion_warnings)
        except Exception as exc:
            result.errors.append(f"Step 1 (Parse/Analyse) failed: {exc}")
        return result

    # ── Step 2: Map SQL objects ───────────────────────────────────────────────

    def _step2_map_objects(self, result: PipelineResult) -> PipelineResult:
        """
        Step 2 — Map SQL objects to Spark/Delta equivalents.
        Tables, views, variables, parameters, temp tables.
        """
        p = result.parsed
        r = result.report
        if p is None or r is None:
            return result

        mapping: list[tuple[str, str, str]] = []

        # SP parameters → Python function args
        for param in p.parameters:
            from sql_mappings.sql_types import resolve_python_type
            py_type = resolve_python_type(param.sql_type)
            mapping.append((
                param.name,
                "SP Parameter",
                f"Python arg: {param.name.lstrip('@')}: {py_type}",
            ))

        # Declared variables → Python variables
        for var in p.variables:
            mapping.append((
                var.name,
                "DECLARE variable",
                f"Python var: {var.name.lstrip('@').lower()} = ...",
            ))

        # Permanent tables → Delta Tables
        for tbl in r.dependencies:
            mapping.append((
                tbl,
                "Permanent table",
                f'spark.table("{self.db_prefix}.{tbl}")',
            ))

        # Temp tables → createOrReplaceTempView
        for tmp in p.temp_tables:
            view_name = tmp.name.lstrip("#")
            mapping.append((
                tmp.name,
                "Temp table",
                f'df.createOrReplaceTempView("{view_name}")',
            ))

        # CTE patterns → spark.sql with WITH clause or DataFrame API
        for cte in r.cte_patterns:
            mapping.append((
                cte,
                "CTE",
                f'spark.sql("WITH {cte} AS (...)")  or  DataFrame chain',
            ))

        result.object_mapping = mapping
        return result

    # ── Steps 3-5: Generate PySpark code ─────────────────────────────────────

    def _step3_5_generate(self, result: PipelineResult) -> PipelineResult:
        """
        Steps 3-5 — Rewrite queries, replace procedural logic, handle
        transactions and temp tables.

        Uses PySparkGenerator which internally handles:
          • Step 3: SELECT/INSERT/UPDATE/DELETE/MERGE → DataFrame API
          • Step 4: IF/ELSE, WHILE, cursors → Python conditionals/iterations
          • Step 5: Transactions → Delta ACID comments; temp tables → TempViews
        """
        try:
            gen = PySparkGenerator(db_prefix=self.db_prefix)
            result.raw_pyspark_code = gen.generate(result.parsed, result.report)
        except Exception as exc:
            result.errors.append(f"Steps 3-5 (Code generation) failed: {exc}")
            result.raw_pyspark_code = ""
        return result

    # ── Step 6: Optimise for Databricks ──────────────────────────────────────

    def _step6_optimize(self, result: PipelineResult) -> PipelineResult:
        """
        Step 6 — Apply Databricks performance optimisations:
          • Broadcast joins for small lookup tables
          • Partition hints and filter push-down
          • Cache recommendations for reused DataFrames
          • Delta Lake output format
          • Z-ORDER hints as comments
        """
        if not result.raw_pyspark_code:
            result.pyspark_code = result.raw_pyspark_code
            return result

        try:
            optimised, notes = self._optimizer.optimize(
                result.raw_pyspark_code,
                report=result.report,
            )
            result.pyspark_code  = optimised
            result.optimizations = notes
        except Exception as exc:
            # Optimisation is best-effort — fall back to raw code
            result.pyspark_code  = result.raw_pyspark_code
            result.warnings.append(f"Step 6 (Optimise) skipped: {exc}")
        return result

    # ── Step 7: Validate ──────────────────────────────────────────────────────

    def _step7_validate(self, result: PipelineResult) -> PipelineResult:
        """
        Step 7 — Append a validation scaffold to the generated code.
        Generates row-count, schema, and sample-data checks.
        """
        if not result.pyspark_code:
            return result

        p = result.parsed
        r = result.report
        deps = r.dependencies if r else []

        validation_lines = [
            "",
            "",
            "# " + "=" * 68,
            "# STEP 7 — VALIDATION SCAFFOLD",
            "# " + "=" * 68,
            "# Run this section to verify the conversion output.",
            "# Adjust expected values for your dataset.",
            "",
            "def _validate_conversion(spark, result_df, expected_row_count: int = -1):",
            '    """Validate PySpark output: row count, schema, and sample data."""',
            "    import logging",
            "    log = logging.getLogger(__name__)",
            "",
            "    # ── Row count check ──────────────────────────────────────────",
            "    actual_rows = result_df.count()",
            "    if expected_row_count >= 0:",
            "        assert actual_rows == expected_row_count, (",
            '            f"Row count mismatch: expected {expected_row_count}, got {actual_rows}"',
            "        )",
            '        log.info("✓ Row count: %d", actual_rows)',
            "    else:",
            '        log.info("ℹ Row count (no expected set): %d", actual_rows)',
            "",
            "    # ── Schema check ─────────────────────────────────────────────",
            "    schema_str = result_df.schema.simpleString()",
            '    log.info("ℹ Schema: %s", schema_str)',
            "",
            "    # ── Null check on all columns ─────────────────────────────────",
            "    from pyspark.sql import functions as F",
            "    null_counts = result_df.select([",
            "        F.count(F.when(F.col(c).isNull(), c)).alias(c)",
            "        for c in result_df.columns",
            "    ])",
            "    null_row = null_counts.first()",
            "    for col_name in result_df.columns:",
            "        n = getattr(null_row, col_name, 0)",
            "        if n and n > 0:",
            '            log.warning("⚠ Column \'%s\' has %d null(s)", col_name, n)',
            "",
            "    # ── Sample data ───────────────────────────────────────────────",
            '    print("\\n── Sample output (first 20 rows) ────────────────────")',
            "    result_df.show(20, truncate=False)",
            '    print("─" * 60)',
            "",
        ]

        # Add dependency existence checks
        if deps:
            validation_lines += [
                "    # ── Source table checks ──────────────────────────────────",
            ]
            for dep in deps[:5]:  # limit to first 5
                validation_lines.append(
                    f'    assert spark.catalog.tableExists("{self.db_prefix}.{dep}"), '
                    f'"Table {self.db_prefix}.{dep} not found"'
                )

        validation_lines += [
            "    return True",
            "",
            "",
            "# ── Run validation ───────────────────────────────────────────────",
            "# Uncomment to run validation after executing the main function:",
            "# if __name__ == '__main__':",
        ]

        if p and p.is_stored_procedure:
            fn_name = _to_fn_name(p.sp_name) if p.sp_name else "run_query"
            validation_lines.append(f"#     output_df = {fn_name}(spark)")
        else:
            validation_lines.append("#     output_df = run_query(spark)")

        validation_lines.append(
            "#     _validate_conversion(spark, output_df, expected_row_count=-1)"
        )

        result.validation_code = "\n".join(validation_lines)
        result.pyspark_code   += result.validation_code
        return result


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_fn_name(sp_name: str) -> str:
    import re
    name = re.sub(r"[\W]+", "_", sp_name.strip().lstrip("[]").rstrip("[]"))
    return name.lower().strip("_") or "run_query"
