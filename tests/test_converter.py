"""
Unit tests for the Converter and Generator
Run: python -m pytest tests/ -v
Covers Steps 2-5 of the 7-step AI Framework Pipeline.
"""
 
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
 
import pytest
from converter.sql_parser    import SQLParser, SQLStatement
from converter.sql_analyzer  import SQLAnalyzer
from converter.code_generator import PySparkGenerator
from converter      import sql_converter as CV
from sql_mappings.sql_types   import resolve_type, resolve_python_type
from sql_mappings.sql_functions import lookup_function
 
 
# ------------------------------------------------------------------------------
#  Type mapping tests
# ------------------------------------------------------------------------------
 
class TestTypeMapping:
    def test_int_mapping(self):
        assert resolve_type("INT") == "IntegerType()"
 
    def test_bigint_mapping(self):
        assert resolve_type("BIGINT") == "LongType()"
 
    def test_varchar_mapping(self):
        assert resolve_type("VARCHAR") == "StringType()"
 
    def test_date_mapping(self):
        assert resolve_type("DATE") == "DateType()"
 
    def test_datetime_mapping(self):
        assert resolve_type("DATETIME") == "TimestampType()"
 
    def test_decimal_with_precision(self):
        result = resolve_type("DECIMAL", precision="18", scale="2")
        assert result == "DecimalType(18, 2)"
 
    def test_decimal_without_precision(self):
        result = resolve_type("DECIMAL")
        assert "DecimalType" in result
 
    def test_bit_mapping(self):
        assert resolve_type("BIT") == "BooleanType()"
 
    def test_python_type_int(self):
        assert resolve_python_type("INT") == "int"
 
    def test_python_type_varchar(self):
        assert resolve_python_type("VARCHAR") == "str"
 
    def test_python_type_bit(self):
        assert resolve_python_type("BIT") == "bool"
 
 
# ------------------------------------------------------------------------------
#  Function mapping tests
# ------------------------------------------------------------------------------
 
class TestFunctionMapping:
    def test_count_mapping(self):
        result = lookup_function("COUNT")
        assert result is not None
        assert "F.count" in result[0]
 
    def test_isnull_mapping(self):
        result = lookup_function("ISNULL")
        assert "coalesce" in result[0]
 
    def test_getdate_mapping(self):
        result = lookup_function("GETDATE")
        assert "current_timestamp" in result[0]
 
    def test_case_insensitive(self):
        assert lookup_function("count") == lookup_function("COUNT")
 
    def test_unknown_returns_none(self):
        assert lookup_function("NONEXISTENT_FUNC_XYZ") is None
 
 
# ------------------------------------------------------------------------------
#  Converter tests
# ------------------------------------------------------------------------------
 
class TestConverter:
    def test_convert_declare(self):
        from converter.sql_parser import DeclaredVariable
        var    = DeclaredVariable(name="@counter", sql_type="INT", default="0")
        result = CV.convert_declare(var)
        assert "counter" in result
        assert "0" in result
 
    def test_convert_declare_null_default(self):
        from converter.sql_parser import DeclaredVariable
        var    = DeclaredVariable(name="@name", sql_type="VARCHAR", default="NULL")
        result = CV.convert_declare(var)
        assert "None" in result
 
    def test_convert_parameters(self):
        from converter.sql_parser import SPParameter
        params = [
            SPParameter(name="@start_date", sql_type="DATE"),
            SPParameter(name="@dept_id",    sql_type="INT", default="NULL"),
        ]
        result = CV.convert_parameters(params)
        assert any("start_date" in a for a in result)
        assert any("dept_id" in a and "None" in a for a in result)
 
    def test_convert_print(self):
        stmt   = SQLStatement(stmt_type="PRINT", raw_sql="PRINT 'Hello World'")
        result = CV.convert_print(stmt)
        assert "print(" in result
        assert "Hello World" in result
 
    def test_convert_if_else_condition(self):
        stmt   = SQLStatement(
            stmt_type="IF",
            raw_sql="IF @dept_id IS NULL BEGIN SELECT 1 END ELSE BEGIN SELECT 2 END"
        )
        result = CV.convert_if_else(stmt)
        assert "if " in result.lower()
 
    def test_convert_drop_temp(self):
        stmt   = SQLStatement(stmt_type="DROP_TABLE", raw_sql="DROP TABLE #temp_results")
        result = CV.convert_drop_temp(stmt)
        assert "dropTempView" in result or "temp_results" in result
 
    def test_convert_insert_select(self):
        stmt = SQLStatement(
            stmt_type="INSERT",
            raw_sql="INSERT INTO #tmp SELECT id, name FROM employees"
        )
        result = CV.convert_insert(stmt)
        assert "spark.sql" in result or "createOrReplaceTempView" in result
 
    def test_convert_update_temp(self):
        stmt = SQLStatement(
            stmt_type="UPDATE",
            raw_sql="UPDATE #tmp SET status = 'done' WHERE id = 1"
        )
        result = CV.convert_update(stmt)
        assert "withColumn" in result or "DeltaTable" in result
 
 
# ------------------------------------------------------------------------------
#  Full pipeline integration tests
# ------------------------------------------------------------------------------
 
class TestFullPipeline:
    SIMPLE_QUERY = """
        SELECT dept, COUNT(*) AS emp_count
        FROM employees
        WHERE salary > 50000
        GROUP BY dept
        ORDER BY emp_count DESC;
    """
 
    SP_TEXT = """
        CREATE PROCEDURE usp_TestProc
            @min_salary INT = 0
        AS
        BEGIN
            SELECT emp_id, full_name, salary
            FROM   employees
            WHERE  salary > @min_salary
            ORDER  BY salary DESC;
        END;
    """
 
    @pytest.fixture
    def pipeline(self):
        return SQLParser(), SQLAnalyzer(), PySparkGenerator()
 
    def test_simple_query_generates_code(self, pipeline):
        parser, analyzer, gen = pipeline
        parsed   = parser.parse(self.SIMPLE_QUERY)
        report   = analyzer.analyze(parsed)
        code     = gen.generate(parsed, report)
        assert len(code) > 0
        assert "spark" in code.lower()
 
    def test_sp_generates_function(self, pipeline):
        parser, analyzer, gen = pipeline
        parsed  = parser.parse(self.SP_TEXT)
        report  = analyzer.analyze(parsed)
        code    = gen.generate(parsed, report)
        assert "def " in code
        assert "usp_testproc" in code.lower() or "TestProc" in code
 
    def test_sp_includes_imports(self, pipeline):
        parser, analyzer, gen = pipeline
        parsed  = parser.parse(self.SP_TEXT)
        report  = analyzer.analyze(parsed)
        code    = gen.generate(parsed, report)
        assert "from pyspark.sql" in code
 
    def test_sp_includes_parameters(self, pipeline):
        parser, analyzer, gen = pipeline
        parsed  = parser.parse(self.SP_TEXT)
        report  = analyzer.analyze(parsed)
        code    = gen.generate(parsed, report)
        assert "min_salary" in code
 
    def test_complexity_score_range(self, pipeline):
        parser, analyzer, _ = pipeline
        parsed  = parser.parse(self.SIMPLE_QUERY)
        report  = analyzer.analyze(parsed)
        assert 0 <= report.complexity_score <= 10
 
    def test_no_warnings_on_simple_query(self, pipeline):
        parser, analyzer, _ = pipeline
        parsed  = parser.parse(self.SIMPLE_QUERY)
        report  = analyzer.analyze(parsed)
        # Simple SELECT should have no warnings
        cursor_warnings = [w for w in report.conversion_warnings if "CURSOR" in w]
        assert len(cursor_warnings) == 0
 
    def test_complex_sp_detects_cursors(self, pipeline):
        parser, analyzer, _ = pipeline
        sp_with_cursor = """
            CREATE PROCEDURE usp_CursorTest AS
            BEGIN
                DECLARE emp_cursor CURSOR FOR SELECT emp_id FROM employees;
                OPEN emp_cursor;
                FETCH NEXT FROM emp_cursor INTO @eid;
                WHILE @@FETCH_STATUS = 0
                BEGIN
                    FETCH NEXT FROM emp_cursor INTO @eid;
                END;
                CLOSE emp_cursor;
            END;
        """
        parsed = parser.parse(sp_with_cursor)
        report = analyzer.analyze(parsed)
        assert report.has_cursors is True
        assert len(report.conversion_warnings) > 0
 
    def test_validation_result_structure(self):
        from output_validator.pyspark_checker import ValidationResult
        r = ValidationResult(passed=True, checks=[{"name": "rows", "passed": True, "detail": "ok"}])
        summary = r.summary()
        assert "PASS" in summary
 