"""
Ollama Demo Script — SQL to PySpark SQL Conversion
===================================================
Run this to demonstrate Ollama converting SQL to PySpark SQL
without needing to open the web browser.

Usage:
    python ollama/demo_conversion.py
    python ollama/demo_conversion.py --sample 2   (use sample 02)
    python ollama/demo_conversion.py --sql "SELECT * FROM orders"
"""

import sys
import time
import pathlib
import argparse

# Ensure project root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

SAMPLES = {
    1: """-- Simple query
SELECT
    d.DepartmentName,
    COUNT(e.EmployeeID) AS EmployeeCount,
    AVG(e.Salary)       AS AvgSalary
FROM Employees e
INNER JOIN Departments d ON e.DepartmentID = d.DepartmentID
WHERE e.IsActive = 1
GROUP BY d.DepartmentName
ORDER BY AvgSalary DESC;""",

    2: """-- Stored procedure with cursor (key conversion test)
CREATE PROCEDURE usp_RaiseSalaries
    @RaisePercent DECIMAL(5,2),
    @DepartmentID INT = NULL
AS
BEGIN
    DECLARE @EmpID     INT;
    DECLARE @OldSalary DECIMAL(18,2);

    DECLARE emp_cursor CURSOR FOR
        SELECT EmployeeID, Salary
        FROM   Employees
        WHERE  IsActive = 1
          AND  (@DepartmentID IS NULL OR DepartmentID = @DepartmentID);

    OPEN emp_cursor;
    FETCH NEXT FROM emp_cursor INTO @EmpID, @OldSalary;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        UPDATE Employees
        SET    Salary = @OldSalary * (1 + @RaisePercent / 100.0)
        WHERE  EmployeeID = @EmpID;

        FETCH NEXT FROM emp_cursor INTO @EmpID, @OldSalary;
    END;

    CLOSE emp_cursor;
    DEALLOCATE emp_cursor;
END;""",

    3: """-- Sales summary SP with temp table and window function
CREATE PROCEDURE usp_GetSalesSummary
    @StartDate DATE,
    @EndDate   DATE
AS
BEGIN
    SELECT
        sr.FullName,
        SUM(o.OrderAmount) AS TotalSales,
        RANK() OVER (ORDER BY SUM(o.OrderAmount) DESC) AS SalesRank
    FROM SalesReps sr
    INNER JOIN Orders o ON sr.SalesRepID = o.SalesRepID
    WHERE o.OrderDate BETWEEN @StartDate AND @EndDate
    GROUP BY sr.SalesRepID, sr.FullName
    ORDER BY TotalSales DESC;
END;""",
}


def check_ollama():
    """Verify Ollama is running and the model is available."""
    from ai_provider.ai_provider import get_provider_info, is_available
    info = get_provider_info()
    print(f"\n{'='*60}")
    print(f"  AI Provider : {info['provider']}")
    print(f"  Model       : {info['model']}")
    print(f"  Status      : {'✓ Ready' if info['available'] else '✗ Not available'}")
    print(f"{'='*60}\n")
    if not info['available']:
        print("ERROR: Ollama is not running.")
        print("  Start it: run  .\\ollama\\start.ps1")
        sys.exit(1)
    return info


def convert(sql: str, db_prefix: str = "my_db") -> str:
    """Convert SQL to PySpark SQL via Ollama."""
    from ai_provider.ai_provider import convert_sql_with_ai
    print("SQL Input:")
    print("-" * 50)
    print(sql.strip())
    print("-" * 50)
    print("\nSending to Ollama... (this may take 30-90s on CPU)\n")
    t0 = time.time()
    result = convert_sql_with_ai(sql, db_prefix=db_prefix)
    elapsed = time.time() - t0
    print(f"\nPySpark SQL Output  [{elapsed:.1f}s]:")
    print("=" * 60)
    print(result)
    print("=" * 60)
    return result


def main():
    parser = argparse.ArgumentParser(description="Demo: SQL → PySpark SQL via Ollama")
    parser.add_argument("--sample", type=int, choices=[1, 2, 3], default=2,
                        help="Sample to convert (1=simple query, 2=cursor SP, 3=window SP)")
    parser.add_argument("--sql", type=str, default=None,
                        help="Custom SQL to convert instead of a sample")
    parser.add_argument("--db", type=str, default="my_db",
                        help="Database prefix for table names (default: my_db)")
    args = parser.parse_args()

    check_ollama()

    sql = args.sql if args.sql else SAMPLES[args.sample]
    label = "custom SQL" if args.sql else f"Sample {args.sample}"
    print(f"Converting {label} with db_prefix='{args.db}'...\n")
    convert(sql, db_prefix=args.db)


if __name__ == "__main__":
    main()
