"""
SQL Analyzer

Classifies parsed SQL into categorized components for the converter.
"""

import re

from dataclasses import dataclass, field
from converter.sql_parser import ParsedSQL, SQLStatement


@dataclass
class AnalysisReport:

    sp_name: str
    parameters: list
    variables: list
    temp_tables: list
    dependencies: list[str]  # permanent tables used

    ddl_statements: list[SQLStatement]
    dml_statements: list[SQLStatement]
    control_flow_statements: list[SQLStatement]
    transaction_statements: list[SQLStatement]

    cursor_patterns: list[str]
    cte_patterns: list[str]

    has_cursors: bool = False
    has_transactions: bool = False
    has_dynamic_sql: bool = False
    has_window_functions: bool = False

    complexity_score: int = 0  # 0-10 scale

    conversion_warnings: list[str] = field(default_factory=list)


class SQLAnalyzer:
    """
    Takes a ParsedSQL and produces an AnalysisReport.
    """

    # Patterns indicating presence of specific features

    CURSOR_RE = re.compile(
        r"\bCURSOR\b|\bFETCH\b|\bOPEN\s+\w+\b|\bCLOSE\s+\w+\b",
        re.IGNORECASE
    )

    TRANSACTION_RE = re.compile(
        r"\bBEGIN\s+TRAN\b|\bCOMMIT\b|\bROLLBACK\b|\bSAVEPOINT\b",
        re.IGNORECASE
    )

    DYNAMIC_SQL_RE = re.compile(
        r"\bEXEC\s*\(|\bSP_EXECUTESQL\b|\bEXECUTE\s*\(",
        re.IGNORECASE
    )

    WINDOW_FN_RE = re.compile(
        r"\bOVER\s*\(",
        re.IGNORECASE
    )

    LINKED_SRV_RE = re.compile(
        r"\[\w+\]\.\[\w+\]\.\[\w+\]\.\[\w+\]",
        re.IGNORECASE
    )

    CTE_RE = re.compile(
        r"\bWITH\s+([\w]+)\s+AS\s*\(",
        re.IGNORECASE
    )

    DDL_TYPES = {
        "CREATE_TABLE",
        "DROP_TABLE"
    }

    DML_TYPES = {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "CTE"
    }

    CTRL_TYPES = {
        "IF",
        "WHILE",
        "EXEC",
        "BEGIN_BLOCK",
        "END_BLOCK"
    }

    TXN_KEYWORDS = {
        "BEGIN TRAN",
        "COMMIT",
        "ROLLBACK"
    }

    def analyze(self, parsed: ParsedSQL) -> AnalysisReport:

        body = parsed.raw_body

        ddl = [
            s for s in parsed.statements
            if s.stmt_type in self.DDL_TYPES
        ]

        dml = [
            s for s in parsed.statements
            if s.stmt_type in self.DML_TYPES
        ]

        ctrl = [
            s for s in parsed.statements
            if s.stmt_type in self.CTRL_TYPES
        ]

        txn = self._find_transaction_stmts(parsed.statements)

        cursors = self._extract_cursor_patterns(body)

        cte_names = [
            m.group(1)
            for m in self.CTE_RE.finditer(body)
        ]

        warnings = self._build_warnings(body, cursors)

        complexity = self._score_complexity(
            parsed,
            cursors,
            warnings
        )

        return AnalysisReport(

            sp_name=parsed.sp_name,
            parameters=parsed.parameters,
            variables=parsed.variables,
            temp_tables=parsed.temp_tables,
            dependencies=parsed.dependencies,

            ddl_statements=ddl,
            dml_statements=dml,
            control_flow_statements=ctrl,
            transaction_statements=txn,

            cursor_patterns=cursors,
            cte_patterns=cte_names,

            has_cursors=bool(cursors),
            has_transactions=bool(
                self.TRANSACTION_RE.search(body)
            ),
            has_dynamic_sql=bool(
                self.DYNAMIC_SQL_RE.search(body)
            ),
            has_window_functions=bool(
                self.WINDOW_FN_RE.search(body)
            ),

            complexity_score=complexity,
            conversion_warnings=warnings
        )

    # ----------------------------------------------------
    # helpers
    # ----------------------------------------------------

    def _find_transaction_stmts(
        self,
        stmts: list[SQLStatement]
    ) -> list[SQLStatement]:
        result = []

        for s in stmts:

            upper = s.raw_sql.strip().upper()

            if any(
                k in upper
                for k in (
                    "BEGIN TRAN",
                    "COMMIT",
                    "ROLLBACK",
                    "SAVEPOINT"
                )
            ):
                result.append(s)

        return result

    def _extract_cursor_patterns(
        self,
        body: str
    ) -> list[str]:
        cursors = []

        for m in re.finditer(
            r"DECLARE\s+([\w]+)\s+CURSOR",
            body,
            re.IGNORECASE
        ):
            cursors.append(m.group(1))

        return cursors

    def _build_warnings(
        self,
        body: str,
        cursors: list[str]
    ) -> list[str]:
        warnings = []

        if cursors:
            warnings.append(
                f"CURSOR detected ({', '.join(cursors)}): "
                "SQL cursors (row-by-row, driver-side) differ fundamentally from PySpark. "
                "Converted to foreachPartition() which runs on Spark executors, not the driver. "
                "Prefer: UPDATE->DeltaTable.merge(), INSERT->df.write.format('delta'), transform->df.withColumn()."
            )

        if self.DYNAMIC_SQL_RE.search(body):
            warnings.append(
                "Dynamic SQL (EXEC/sp_executesql) detected: "
                "Requires manual review. Convert to parameterized spark.sql() calls."
            )

        if self.LINKED_SRV_RE.search(body):
            warnings.append(
                "Linked Server reference detected: "
                "Replace with Delta Lake external tables or JDBC data sources."
            )

        if re.search(
            r"\bNOLOCK\b|\bWITH\s*\(NOLOCK\)",
            body,
            re.IGNORECASE
        ):
            warnings.append(
                "NOLOCK hint detected: "
                "Remove - Delta Lake ACID guarantees consistency without dirty reads."
            )

        if re.search(
            r"\bXML\b|\bFOR\s+XML\b",
            body,
            re.IGNORECASE
        ):
            warnings.append(
                "XML operation detected: "
                "Use F.from_xml() or convert to JSON with F.to_json()."
            )

        if re.search(
            r"\bOPENROWSET\b|\bOPENQUERY\b",
            body,
            re.IGNORECASE
        ):
            warnings.append(
                "OPENROWSET/OPENQUERY detected: "
                "Replace with spark.read.jdbc() or external Delta table definition."
            )

        if re.search(r"@@\w+", body):
            warnings.append(
                "SQL Server system variable (@@variable) detected: "
                "Replace with Spark equivalents (e.g., @@ROWCOUNT -> .count())."
            )

        return warnings

    @staticmethod
    def _score_complexity(
        parsed: ParsedSQL,
        cursors: list,
        warnings: list
    ) -> int:

        score = 0

        score += min(len(parsed.parameters), 3)

        score += min(
            len(parsed.temp_tables) * 2,
            4
        )

        score += min(
            len(cursors) * 3,
            6
        )

        score += min(
            len(warnings),
            4
        )

        score += (
            1 if any(
                s.stmt_type == "MERGE"
                for s in parsed.statements
            )
            else 0
        )

        score += (
            1 if any(
                s.stmt_type == "WHILE"
                for s in parsed.statements
            )
            else 0
        )

        score += (
            1 if any(
                s.stmt_type == "CTE"
                for s in parsed.statements
            )
            else 0
        )

        return min(score, 10)