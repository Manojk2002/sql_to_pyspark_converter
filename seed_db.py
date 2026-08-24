"""
Seed Script — Load CSV exports into the local SQLite database (mydb).

Run this once after cloning or if mydb is reset/deleted:
    python seed_db.py
"""

import pathlib
import sqlite3

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parent
DB_PATH      = PROJECT_ROOT / "mydb"
EXPORTS_DIR  = PROJECT_ROOT / "exports"

# Map: SQLite table name → CSV file in exports/
TABLES = {
    "employee_details":    "employee_details.csv",
    "health_check":        "health_check.csv",
    "provider_performance": "provider_performance.csv",
    "verify_test":         "verify_test.csv",
}


def seed():
    conn = sqlite3.connect(str(DB_PATH))
    print(f"Connected to: {DB_PATH}\n")

    loaded, skipped = 0, 0
    for table, csv_file in TABLES.items():
        csv_path = EXPORTS_DIR / csv_file
        if not csv_path.exists():
            print(f"  SKIP  {table:25} — file not found: {csv_path}")
            skipped += 1
            continue

        df = pd.read_csv(str(csv_path))
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  OK    {table:25} — {len(df)} rows, cols: {list(df.columns)}")
        loaded += 1

    conn.commit()
    conn.close()

    print(f"\nDone. {loaded} table(s) loaded, {skipped} skipped.")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    seed()
