"""
Databricks Optimizer — Step 6 of the AI Framework Pipeline.

Analyses generated PySpark code and applies Databricks performance optimisations:

  1. Broadcast joins    — F.broadcast() for small lookup/dimension tables
  2. Filter push-down   — .filter() before joins on partition keys
  3. DataFrame caching  — .cache() for DataFrames reused > 1 time
  4. Delta output       — .write.format("delta") for persistent outputs
  5. Z-ORDER hints      — comment recommendations for large Delta tables
  6. Avoid collect()    — warn against collect() on large DataFrames
  7. AQE note           — remind that AQE is on by default in Databricks
"""

from __future__ import annotations

import re
from typing import Optional
from converter.sql_analyzer import AnalysisReport


# ── Small-table name heuristics (broadcast candidates) ────────────────────────
_SMALL_TABLE_PATTERNS = re.compile(
    r"\b(dim_|lookup_|ref_|code_|type_|status_|config_|map_|master_)",
    re.IGNORECASE,
)

# Detect spark.table() or DataFrame reads
_TABLE_READ_RE = re.compile(
    r'(\w+)\s*=\s*spark\.table\("([^"]+)"\)',
)

# Detect .join( calls without broadcast
_JOIN_RE = re.compile(
    r'(\w+_df)\s*=\s*(\w+_df)\.join\(\s*(\w+_df)',
)

# Detect .write calls that are not already Delta
_WRITE_RE = re.compile(
    r'\.write(?:\.mode\("[^"]+"\))?\.(?:parquet|csv|json|orc)\(',
    re.IGNORECASE,
)

# Detect DataFrame variable usage frequency
_DF_USAGE_RE = re.compile(r'\b(\w+_df)\b')

# Detect collect() on DataFrames
_COLLECT_RE = re.compile(r'(\w+_df)\.collect\(\)')


class DatabricksOptimizer:
    """
    Rule-based code optimiser for Step 6 of the conversion pipeline.

    Applies deterministic text transformations to inject Databricks
    best-practices into the generated PySpark source code.
    """

    def optimize(
        self,
        code: str,
        report: Optional[AnalysisReport] = None,
    ) -> tuple[str, list[str]]:
        """
        Optimise generated PySpark code for Databricks.

        Args:
            code:   Generated PySpark source code.
            report: AnalysisReport from Step 1 (used for context hints).

        Returns:
            (optimised_code, list_of_optimisation_notes)
        """
        notes: list[str] = []

        code, n = self._inject_broadcast_joins(code)
        notes.extend(n)

        code, n = self._inject_cache_hints(code)
        notes.extend(n)

        code, n = self._fix_write_format(code)
        notes.extend(n)

        code, n = self._warn_collect(code)
        notes.extend(n)

        code, n = self._add_aqe_note(code)
        notes.extend(n)

        if report:
            code, n = self._add_zorder_hints(code, report)
            notes.extend(n)

        code, n = self._add_optimization_header(code, notes)
        notes.extend(n)

        return code, notes

    # ── Broadcast join injection ───────────────────────────────────────────────

    def _inject_broadcast_joins(self, code: str) -> tuple[str, list[str]]:
        """Wrap small lookup/dimension DataFrames in F.broadcast()."""
        notes: list[str] = []
        lines = code.split("\n")
        new_lines: list[str] = []

        # Find all table reads and mark small ones
        small_df_vars: set[str] = set()
        for line in lines:
            m = _TABLE_READ_RE.search(line)
            if m:
                df_var = m.group(1)
                table_name = m.group(2)
                if _SMALL_TABLE_PATTERNS.search(table_name):
                    small_df_vars.add(df_var)

        if not small_df_vars:
            return code, notes

        # Inject broadcast() in join calls involving small DataFrames
        for line in lines:
            modified = line
            jm = _JOIN_RE.search(line)
            if jm:
                join_arg = jm.group(3)
                if join_arg in small_df_vars and f"broadcast({join_arg}" not in line:
                    modified = line.replace(
                        f".join({join_arg}",
                        f".join(F.broadcast({join_arg})",
                    )
                    notes.append(
                        f"Applied F.broadcast() to small table DataFrame '{join_arg}'"
                    )
            new_lines.append(modified)

        return "\n".join(new_lines), notes

    # ── DataFrame caching ─────────────────────────────────────────────────────

    def _inject_cache_hints(self, code: str) -> tuple[str, list[str]]:
        """Add .cache() calls for DataFrames referenced more than twice."""
        notes: list[str] = []

        # Count how many times each _df variable appears
        usages: dict[str, int] = {}
        for m in _DF_USAGE_RE.finditer(code):
            var = m.group(1)
            usages[var] = usages.get(var, 0) + 1

        # Candidates: used 3+ times (definition + 2 reads) and not already cached
        candidates = [
            v for v, cnt in usages.items()
            if cnt >= 3 and f"{v}.cache()" not in code
        ]

        if not candidates:
            return code, notes

        lines = code.split("\n")
        new_lines: list[str] = []
        injected: set[str] = set()

        for line in lines:
            new_lines.append(line)
            # Inject .cache() right after the first assignment of the candidate
            for var in candidates:
                if var in injected:
                    continue
                assign_pattern = re.compile(rf"^\s*{re.escape(var)}\s*=\s*")
                if assign_pattern.match(line) and "cache()" not in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(
                        " " * indent
                        + f"{var}.cache()  "
                        + f"# STEP 6: cache — reused {usages[var]}x"
                    )
                    injected.add(var)
                    notes.append(
                        f"Added .cache() to '{var}' (referenced {usages[var]} times)"
                    )

        return "\n".join(new_lines), notes

    # ── Delta write format ────────────────────────────────────────────────────

    def _fix_write_format(self, code: str) -> tuple[str, list[str]]:
        """Replace non-Delta write formats with Delta format."""
        notes: list[str] = []
        new_code = code

        matches = list(_WRITE_RE.finditer(code))
        for m in reversed(matches):  # reverse to preserve offsets
            original = m.group(0)
            format_name = re.search(r'\.(parquet|csv|json|orc)\(', original, re.IGNORECASE)
            if format_name:
                old_fmt = format_name.group(1)
                replacement = original.replace(
                    f".{old_fmt}(",
                    '.format("delta").saveAsTable(',
                )
                new_code = new_code[:m.start()] + replacement + new_code[m.end():]
                notes.append(
                    f"Converted .{old_fmt}() write to .format('delta').saveAsTable() "
                    f"for Delta Lake reliability"
                )

        return new_code, notes

    # ── collect() warnings ────────────────────────────────────────────────────

    def _warn_collect(self, code: str) -> tuple[str, list[str]]:
        """Add warning comments above collect() calls."""
        notes: list[str] = []
        lines = code.split("\n")
        new_lines: list[str] = []

        for line in lines:
            if _COLLECT_RE.search(line) and "# ⚠" not in line:
                indent = len(line) - len(line.lstrip())
                new_lines.append(
                    " " * indent
                    + "# ⚠ STEP 6: .collect() brings ALL rows to driver — "
                    + "avoid on large DataFrames; use .show() or write to Delta instead."
                )
                notes.append("Flagged .collect() call — potential performance risk on large data")
            new_lines.append(line)

        return "\n".join(new_lines), notes

    # ── AQE note ──────────────────────────────────────────────────────────────

    def _add_aqe_note(self, code: str) -> tuple[str, list[str]]:
        """Add a note that AQE is enabled by default on Databricks."""
        if "AQE" in code or "adaptive" in code.lower():
            return code, []

        note_block = (
            "\n# STEP 6 NOTE: Adaptive Query Execution (AQE) is ON by default on Databricks.\n"
            "# No explicit spark.conf.set('spark.sql.adaptive.enabled', 'true') needed.\n"
            "# AQE automatically handles: skew joins, dynamic coalescing, and join reordering.\n"
        )

        # Insert after imports block (find first non-import, non-comment line)
        lines = code.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("import") \
               and not stripped.startswith("from ") and not stripped.startswith('"""'):
                insert_idx = i
                break

        lines.insert(insert_idx, note_block)
        return "\n".join(lines), ["Added AQE note (enabled by default on Databricks)"]

    # ── Z-ORDER hints ─────────────────────────────────────────────────────────

    def _add_zorder_hints(
        self,
        code: str,
        report: AnalysisReport,
    ) -> tuple[str, list[str]]:
        """Append Z-ORDER optimisation hints for large dependency tables."""
        notes: list[str] = []
        if not report.dependencies:
            return code, notes

        hint_lines = [
            "",
            "# ── STEP 6: Z-ORDER Optimisation Hints ─────────────────────────────────────",
            "# Run these commands in Databricks to improve query performance on large tables:",
        ]
        for dep in report.dependencies:
            hint_lines.append(
                f"# OPTIMIZE {self._db_prefix_hint(dep)} ZORDER BY (<partition_key_column>);"
            )
            notes.append(f"Added Z-ORDER hint for table '{dep}'")

        hint_lines.append(
            "# Replace <partition_key_column> with the most frequently filtered column."
        )
        hint_lines.append("")

        return code + "\n".join(hint_lines), notes

    @staticmethod
    def _db_prefix_hint(table_name: str) -> str:
        # Already qualified → use as-is; else suggest schema prefix
        return table_name if "." in table_name else f"<db>.{table_name}"

    # ── Optimisation header block ─────────────────────────────────────────────

    @staticmethod
    def _add_optimization_header(
        code: str,
        notes: list[str],
    ) -> tuple[str, list[str]]:
        """Prepend a compact optimisation summary comment block."""
        if not notes:
            return code, []

        header_lines = [
            "# " + "─" * 68,
            "# STEP 6 — DATABRICKS OPTIMISATIONS APPLIED:",
        ]
        for note in notes:
            header_lines.append(f"#   ✓ {note}")
        header_lines.append("# " + "─" * 68)
        header_lines.append("")

        # Find where the imports end and insert the header after them
        lines = code.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_idx = i + 1

        lines.insert(insert_idx + 1, "\n".join(header_lines))
        return "\n".join(lines), []
