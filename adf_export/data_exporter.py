"""
ADF Data Export — Read data from SQL database and export to Excel, CSV, Parquet.

Uses SQLAlchemy + pandas to connect to any SQL database (PostgreSQL, SQL Server,
MySQL, SQLite, etc.) and export query results to multiple file formats.
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# Output directory for exported files
EXPORT_DIR = pathlib.Path(__file__).parent.parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

# Supported database types and their SQLAlchemy drivers
DB_DRIVERS = {
    "postgresql":  "postgresql+psycopg2",
    "sqlserver":   "mssql+pyodbc",
    "azuresql":    "mssql+pyodbc",
    "mysql":       "mysql+pymysql",
    "mariadb":     "mariadb+mariadbconnector",
    "sqlite":      "sqlite",
    "oracle":      "oracle+cx_oracle",
    "redshift":    "redshift+redshift_connector",
    "snowflake":   "snowflake+snowflake-sqlalchemy",
    "bigquery":    "bigquery",
}

# Default ports for each database type
DB_DEFAULT_PORTS = {
    "postgresql": "5432",
    "sqlserver":  "1433",
    "azuresql":   "1433",
    "mysql":      "3306",
    "mariadb":    "3306",
    "sqlite":     "",
    "oracle":     "1521",
    "redshift":   "5439",
    "snowflake":  "443",
    "bigquery":   "",
}


@dataclass
class ExportResult:
    """Result of a data export operation."""
    success: bool
    records_extracted: int = 0
    columns_extracted: int = 0
    files_created: list = field(default_factory=list)
    error: Optional[str] = None


def build_connection_string(
    db_type: str,
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
) -> str:
    """Build a SQLAlchemy connection string from components."""
    db_type_lower = db_type.lower().strip()
    driver = DB_DRIVERS.get(db_type_lower)

    if not driver:
        raise ValueError(
            f"Unsupported database type: '{db_type}'. "
            f"Supported: {', '.join(DB_DRIVERS.keys())}"
        )

    if db_type_lower == "sqlite":
        # Resolve relative paths to absolute so SQLite always finds the right file
        db_path = pathlib.Path(database)
        if not db_path.is_absolute():
            db_path = pathlib.Path(__file__).parent.parent / database
        return f"sqlite:///{db_path.resolve()}"

    if db_type_lower in ("sqlserver", "azuresql"):
        # SQL Server / Azure SQL need ODBC driver specification
        return (
            f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        )

    if db_type_lower == "oracle":
        return f"oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}"

    if db_type_lower == "snowflake":
        return f"snowflake://{username}:{password}@{host}/{database}"

    if db_type_lower == "bigquery":
        return f"bigquery://{database}"

    return f"{driver}://{username}:{password}@{host}:{port}/{database}"


def export_data(
    db_type: str,
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
    query: str,
    formats: list,
    output_name: str = "export_data",
) -> ExportResult:
    """
    Connect to a SQL database, execute the query, and export results.

    Args:
        db_type: Database type (postgresql, sqlserver, mysql, sqlite, oracle)
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        query: SQL SELECT query to execute
        formats: List of export formats ('xlsx', 'csv', 'parquet')
        output_name: Base filename for exported files (without extension)

    Returns:
        ExportResult with details of the export operation.
    """
    if not query or not query.strip():
        return ExportResult(success=False, error="No SQL query provided.")

    if not formats:
        return ExportResult(success=False, error="No export formats selected.")

    # Sanitize output filename
    safe_name = "".join(
        c if c.isalnum() or c in ("_", "-") else "_"
        for c in output_name.strip()
    ) or "export_data"

    try:
        conn_string = build_connection_string(
            db_type, host, port, database, username, password
        )
        engine = create_engine(conn_string)

        # Execute query and load into DataFrame
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        result = ExportResult(
            success=True,
            records_extracted=len(df),
            columns_extracted=len(df.columns),
        )

        # Export to requested formats
        for fmt in formats:
            fmt_lower = fmt.lower().strip()
            if fmt_lower == "xlsx":
                filepath = EXPORT_DIR / f"{safe_name}.xlsx"
                df.to_excel(str(filepath), index=False)
                result.files_created.append(str(filepath))

            elif fmt_lower == "csv":
                filepath = EXPORT_DIR / f"{safe_name}.csv"
                df.to_csv(str(filepath), index=False)
                result.files_created.append(str(filepath))

            elif fmt_lower == "parquet":
                filepath = EXPORT_DIR / f"{safe_name}.parquet"
                df.to_parquet(str(filepath), index=False, engine="pyarrow")
                result.files_created.append(str(filepath))

            else:
                result.files_created.append(f"SKIPPED: unknown format '{fmt}'")

        return result

    except Exception as exc:
        return ExportResult(success=False, error=str(exc))


def test_connection(
    db_type: str,
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
) -> dict:
    """Test database connectivity without executing a query."""
    try:
        conn_string = build_connection_string(
            db_type, host, port, database, username, password
        )
        engine = create_engine(conn_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"success": True, "message": "Connection successful!"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
