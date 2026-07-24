-- Sample 01: Simple SQL Query
-- Demonstrates SELECT with JOIN, WHERE, GROUP BY, HAVING, ORDER BY

SELECT
    d.DepartmentName,
    COUNT(e.EmployeeID)   AS EmployeeCount,
    AVG(e.Salary)         AS AvgSalary,
    MAX(e.Salary)         AS MaxSalary,
    MIN(e.Salary)         AS MinSalary
FROM
    Employees   e
    INNER JOIN Departments d ON e.DepartmentID = d.DepartmentID
WHERE
    e.IsActive = 1
    AND e.HireDate >= '2020-01-01'
GROUP BY
    d.DepartmentName
HAVING
    COUNT(e.EmployeeID) > 2
ORDER BY
    AvgSalary DESC;
