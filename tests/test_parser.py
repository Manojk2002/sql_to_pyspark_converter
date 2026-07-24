"""
Unit tests for the SQL Parser
Run: python -m pytest tests/ -v
Covers Step 1 of the 7-step AI Framework Pipeline.
"""
 
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
 
import pytest
from converter.sql_parser import SQLParser, SPParameter
 
 
@pytest.fixture
def parser():
    return SQLParser()
 
 
class TestStripComments:
    def test_strips_line_comments(self, parser):
        sql    = "SELECT * FROM t -- this is a comment\nWHERE id = 1"
        result = parser._strip_comments(sql)
        assert "--" not in result
        assert "WHERE id = 1" in result
 
    def test_strips_block_comments(self, parser):
        sql    = "SELECT /* block comment */ * FROM t"
        result = parser._strip_comments(sql)
        assert "/*" not in result
        assert "SELECT" in result
        assert "FROM t" in result
 
 
class TestStoredProcedureParsing:
    SP_SIMPLE = """
        CREATE PROCEDURE usp_GetSales
            @start_date DATE,
            @end_date   DATE,
            @dept_id    INT = NULL
        AS
        BEGIN
            SELECT * FROM sales WHERE sale_date >= @start_date;
        END;
    """
 
    def test_detects_sp(self, parser):
        result = parser.parse(self.SP_SIMPLE)
        assert result.is_stored_procedure is True
 
    def test_extracts_sp_name(self, parser):
        result = parser.parse(self.SP_SIMPLE)
        assert result.sp_name == "usp_GetSales"
 
    def test_extracts_parameters(self, parser):
        result = parser.parse(self.SP_SIMPLE)
        assert len(result.parameters) == 3
 
    def test_parameter_names(self, parser):
        result = parser.parse(self.SP_SIMPLE)
        names  = [p.name for p in result.parameters]
        assert "@start_date" in names
        assert "@end_date"   in names
        assert "@dept_id"    in names
 
    def test_parameter_types(self, parser):
        result = parser.parse(self.SP_SIMPLE)
        types  = {p.name: p.sql_type for p in result.parameters}
        assert types["@start_date"] == "DATE"
        assert types["@end_date"]   == "DATE"
        assert types["@dept_id"]    == "INT"
 
    def test_parameter_default(self, parser):
        result  = parser.parse(self.SP_SIMPLE)
        dept_p  = next(p for p in result.parameters if p.name == "@dept_id")
        assert dept_p.default == "NULL"
 
    def test_extracts_statements(self, parser):
        result = parser.parse(self.SP_SIMPLE)
        types  = [s.stmt_type for s in result.statements]
        assert "SELECT" in types
 
 
class TestStandaloneQuery:
    QUERY = """
        SELECT dept, COUNT(*) AS emp_count
        FROM employees
        WHERE salary > 50000
        GROUP BY dept
        ORDER BY emp_count DESC;
    """
 
    def test_not_a_sp(self, parser):
        result = parser.parse(self.QUERY)
        assert result.is_stored_procedure is False
 
    def test_detects_select(self, parser):
        result = parser.parse(self.QUERY)
        types  = [s.stmt_type for s in result.statements]
        assert "SELECT" in types
 
    def test_detects_table_dependency(self, parser):
        result = parser.parse(self.QUERY)
        assert "employees" in result.dependencies
 
 
class TestTempTableDetection:
    SP_WITH_TEMP = """
        CREATE PROCEDURE usp_Test AS
        BEGIN
            CREATE TABLE #tmp_results (id INT, val VARCHAR(50));
            SELECT * INTO #other_temp FROM source_table;
        END;
    """
 
    def test_detects_temp_tables(self, parser):
        result = parser.parse(self.SP_WITH_TEMP)
        names  = [t.name for t in result.temp_tables]
        assert "#tmp_results" in names
 
    def test_detects_select_into_temp(self, parser):
        result = parser.parse(self.SP_WITH_TEMP)
        names  = [t.name for t in result.temp_tables]
        assert "#other_temp" in names
 
 
class TestDeclareVariables:
    SP_WITH_DECL = """
        CREATE PROCEDURE usp_Vars AS
        BEGIN
            DECLARE @counter INT = 0;
            DECLARE @name    VARCHAR(100);
        END;
    """
 
    def test_detects_declared_variables(self, parser):
        result = parser.parse(self.SP_WITH_DECL)
        names  = [v.name for v in result.variables]
        assert "@counter" in names
 
    def test_variable_default(self, parser):
        result   = parser.parse(self.SP_WITH_DECL)
        counter  = next(v for v in result.variables if v.name == "@counter")
        assert counter.default == "0"
 
 
class TestControlFlowDetection:
    SP_WITH_IF = """
        CREATE PROCEDURE usp_CF AS
        BEGIN
            IF (SELECT COUNT(*) FROM t) > 0
            BEGIN
                SELECT * FROM t;
            END
            ELSE
            BEGIN
                PRINT 'empty';
            END
        END;
    """
 
    def test_detects_if(self, parser):
        result = parser.parse(self.SP_WITH_IF)
        types  = [s.stmt_type for s in result.statements]
        assert "IF" in types
 