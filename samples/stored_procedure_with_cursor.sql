-- Sample 03: Complex Stored Procedure
-- Raises salaries by department with cursor, transaction, and audit log

CREATE PROCEDURE usp_RaiseSalaries
    @RaisePercent   DECIMAL(5,2),
    @DepartmentID   INT = NULL,
    @EffectiveDate  DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;

    IF @EffectiveDate IS NULL
        SET @EffectiveDate = GETDATE();

    IF @RaisePercent <= 0 OR @RaisePercent > 50
    BEGIN
        RAISERROR('RaisePercent must be between 0 and 50.', 16, 1);
        RETURN;
    END;

    CREATE TABLE #SalaryAudit (
        EmployeeID   INT,
        OldSalary    DECIMAL(18,2),
        NewSalary    DECIMAL(18,2),
        ChangeDate   DATE
    );

    DECLARE @EmpID       INT;
    DECLARE @OldSalary   DECIMAL(18,2);
    DECLARE @NewSalary   DECIMAL(18,2);

    DECLARE emp_cursor CURSOR FOR
        SELECT EmployeeID, Salary
        FROM   Employees
        WHERE  IsActive = 1
          AND  (@DepartmentID IS NULL OR DepartmentID = @DepartmentID);

    BEGIN TRANSACTION;

    BEGIN TRY
        OPEN emp_cursor;
        FETCH NEXT FROM emp_cursor INTO @EmpID, @OldSalary;

        WHILE @@FETCH_STATUS = 0
        BEGIN
            SET @NewSalary = @OldSalary * (1 + @RaisePercent / 100.0);

            UPDATE Employees
            SET    Salary       = @NewSalary,
                   ModifiedDate = @EffectiveDate
            WHERE  EmployeeID = @EmpID;

            INSERT INTO #SalaryAudit VALUES (@EmpID, @OldSalary, @NewSalary, @EffectiveDate);

            FETCH NEXT FROM emp_cursor INTO @EmpID, @OldSalary;
        END;

        CLOSE emp_cursor;
        DEALLOCATE emp_cursor;

        INSERT INTO SalaryAuditLog (EmployeeID, OldSalary, NewSalary, ChangeDate)
        SELECT EmployeeID, OldSalary, NewSalary, ChangeDate FROM #SalaryAudit;

        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        CLOSE emp_cursor;
        DEALLOCATE emp_cursor;
        THROW;
    END CATCH;

    SELECT
        COUNT(*)                        AS EmployeesUpdated,
        SUM(OldSalary)                  AS TotalOldPayroll,
        SUM(NewSalary)                  AS TotalNewPayroll,
        SUM(NewSalary) - SUM(OldSalary) AS TotalIncrease
    FROM #SalaryAudit;

    DROP TABLE #SalaryAudit;
END;
