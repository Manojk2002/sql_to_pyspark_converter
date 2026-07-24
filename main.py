"""
SQL -> PySpark AI Framework Pipeline — Command Line Entry Point

Usage:
  python main.py                          # interactive prompt
  python main.py path/to/query.sql        # convert a SQL file
  python main.py path/to/query.sql --db my_catalog  # with custom DB prefix
  python main.py --app                    # launch Flask web UI
"""

import sys
import pathlib
import argparse


def _run_pipeline(sql: str, db_prefix: str) -> None:
    from converter.pipeline import ConversionPipeline
    pipeline = ConversionPipeline(db_prefix=db_prefix)
    result   = pipeline.run(sql)

    print(result.analysis_summary())

    out_dir  = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    sp_name  = (result.parsed.sp_name or "query").replace(".", "_").strip("[]")
    out_file = out_dir / f"{sp_name}_pyspark.py"
    out_file.write_text(result.pyspark_code, encoding="utf-8")
    print(f"\nGenerated: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL -> PySpark AI Framework Converter"
    )
    parser.add_argument(
        "sql_file", nargs="?",
        help="Path to a .sql file to convert (omit for interactive mode)"
    )
    parser.add_argument(
        "--db", default="my_db",
        help="Database/catalog prefix for table references (default: my_db)"
    )
    parser.add_argument(
        "--app", action="store_true",
        help="Launch the Flask web UI instead of CLI conversion"
    )
    args = parser.parse_args()

    if args.app:
        from app import app
        print("\n" + "=" * 60)
        print("  SQL -> PySpark AI Converter -- Web UI")
        print("  Open: http://localhost:5000")
        print("  Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        app.run(debug=True, host="127.0.0.1", port=5000)
        return

    if args.sql_file:
        sql_path = pathlib.Path(args.sql_file)
        if not sql_path.exists():
            print(f"Error: file not found: {sql_path}", file=sys.stderr)
            sys.exit(1)
        sql = sql_path.read_text(encoding="utf-8", errors="replace")
        _run_pipeline(sql, db_prefix=args.db)
    else:
        print("SQL -> PySpark AI Framework (7-Step Pipeline)")
        print("Paste your SQL below, then press Ctrl+Z (Windows) / Ctrl+D (Unix) + Enter:")
        sql = sys.stdin.read()
        if sql.strip():
            _run_pipeline(sql, db_prefix=args.db)
        else:
            print("No input provided.")


if __name__ == "__main__":
    main()

