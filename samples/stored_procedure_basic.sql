-- Sample 02: Stored Procedure with temp table and window function
-- Calculates sales summary for a date range, ranked by region

CREATE PROCEDURE usp_GetSalesSummary
    @StartDate  DATE,
    @EndDate    DATE,
    @Region     NVARCHAR(50) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    CREATE TABLE #SalesSummary (
        SalesRepID   INT,
        SalesRepName NVARCHAR(100),
        Region       NVARCHAR(50),
        TotalSales   DECIMAL(18,2),
        OrderCount   INT
    );

    INSERT INTO #SalesSummary (SalesRepID, SalesRepName, Region, TotalSales, OrderCount)
    SELECT
        sr.SalesRepID,
        sr.FullName,
        sr.Region,
        SUM(o.OrderAmount)  AS TotalSales,
        COUNT(o.OrderID)    AS OrderCount
    FROM
        SalesReps   sr
        INNER JOIN Orders o ON sr.SalesRepID = o.SalesRepID
    WHERE
        o.OrderDate BETWEEN @StartDate AND @EndDate
        AND (@Region IS NULL OR sr.Region = @Region)
    GROUP BY
        sr.SalesRepID, sr.FullName, sr.Region;

    SELECT
        SalesRepID,
        SalesRepName,
        Region,
        TotalSales,
        OrderCount,
        RANK() OVER (PARTITION BY Region ORDER BY TotalSales DESC) AS RegionRank
    FROM   #SalesSummary
    ORDER BY TotalSales DESC;

    DROP TABLE #SalesSummary;
END;
