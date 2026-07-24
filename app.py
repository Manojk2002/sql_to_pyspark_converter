"""
Flask Web Application � SQL to PySpark AI Converter
Run: python app.py
Open: http://localhost:5000
7-Step AI Framework Pipeline: Analyse ? Map ? Rewrite ? Procedural ? Transactions ? Optimise ? Validate
"""
 
import os
import sys
import json
import pathlib
 
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
 
# -- Ensure project root is on sys.path ---------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent))
 
from converter.sql_parser    import SQLParser
from converter.sql_analyzer  import SQLAnalyzer
from converter.code_generator import PySparkGenerator
from ai_provider.ai_provider  import (
    convert_sql_with_ai, stream_sql_with_ai,
    explain_pyspark_code, optimize_pyspark_code,
    is_available as ai_available, get_provider_info,
)
 
app = Flask(__name__, template_folder="web_ui/templates")
app.config["JSON_SORT_KEYS"]   = False
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit
 
ALLOWED_EXTENSIONS = {".sql", ".txt"}
 
SAMPLES_DIR = pathlib.Path(__file__).parent / "samples"
OUTPUT_DIR  = pathlib.Path(__file__).parent / "output"
UPLOADS_DIR = pathlib.Path(__file__).parent / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
 
_parser    = SQLParser()
_analyzer  = SQLAnalyzer()
 
# Quick-reference mapping for the UI
QUICK_REF = [
    ("SELECT col",          "df.select('col')"),
    ("WHERE cond",          "df.filter(F.col('col') > val)"),
    ("INNER JOIN",          "df.join(other, cond, 'inner')"),
    ("LEFT JOIN",           "df.join(other, cond, 'left')"),
    ("GROUP BY + COUNT(*)", "df.groupBy('col').agg(F.count('*'))"),
    ("SUM(col)",            "F.sum('col')"),
    ("AVG(col)",            "F.avg('col')"),
    ("ORDER BY col DESC",   "df.orderBy(F.desc('col'))"),
    ("DISTINCT",            "df.distinct()"),
    ("TOP n / LIMIT n",     "df.limit(n)"),
    ("UNION ALL",           "df1.union(df2)"),
    ("CASE WHEN",           "F.when(cond, val).otherwise(other)"),
    ("ISNULL / COALESCE",   "F.coalesce(F.col('a'), F.lit(0))"),
    ("CAST(col AS INT)",    "F.col('col').cast(IntegerType())"),
    ("ROW_NUMBER() OVER",   "F.row_number().over(Window.spec)"),
    ("CREATE TABLE #tmp",   "df.createOrReplaceTempView('tmp')"),
    ("BEGIN TRANSACTION",   "DeltaTable.forName(spark, tbl)"),
    ("MERGE INTO",          "DeltaTable.merge().execute()"),
    ("UPDATE ... SET",      "DeltaTable.update(cond, set={})"),
    ("DELETE FROM",         "DeltaTable.delete(cond)"),
]
 
 
# -- Routes --------------------------------------------------------------------
 
def _extract_quick_view(code: str, sql_input: str) -> str:
    """
    Extract just the converted logic (function body) from the full generated file.
    Returns clean, dedented PySpark code the user cares about immediately.
    """
    lines = code.split("\n")
 
    # Find the "Main logic" marker
    main_idx = next((i for i, l in enumerate(lines) if "# -- Main logic" in l), None)
 
    # Find entry point / validation section (marks end of function body)
    val_idx = next(
        (i for i, l in enumerate(lines)
         if "# ENTRY POINT" in l or "# VALIDATION HELPERS" in l or "# -- Validation helpers" in l),
        len(lines),
    )
    # Also stop at the === separator line immediately before those sections
    while val_idx > 0 and lines[val_idx - 1].strip().startswith("# ==="):
        val_idx -= 1
 
    if main_idx is None:
        # No marker � try to find first _df or spark. line
        main_idx = next(
            (i for i, l in enumerate(lines) if "_df = spark" in l or "result_df = spark" in l),
            None,
        )
        if main_idx is None:
            return code  # fall back to full code
 
    body_lines = lines[main_idx + 1 : val_idx]
 
    # Dedent one level (4 spaces) and strip trailing blank lines
    dedented = []
    for line in body_lines:
        dedented.append(line[4:] if line.startswith("    ") else line)
 
    # Remove leading/trailing blank lines
    while dedented and not dedented[0].strip():
        dedented.pop(0)
    while dedented and not dedented[-1].strip():
        dedented.pop()
    # Drop trailing bare return statement (looks odd outside a function)
    while dedented and dedented[-1].strip().startswith("return "):
        dedented.pop()
    while dedented and not dedented[-1].strip():
        dedented.pop()
 
    if not dedented:
        # Truly empty body � return a passthrough
        clean = sql_input.strip().rstrip(";")
        return (
            '# Could not parse SQL � using spark.sql() passthrough\n'
            'result_df = spark.sql("""\n'
            + "\n".join(f"    {l}" for l in clean.split("\n"))
            + '\n""")\nresult_df.show()'
        )
 
    return "\n".join(dedented)
 
 
@app.route("/")
def index():
    samples = sorted(f.name for f in SAMPLES_DIR.glob("*.sql")) if SAMPLES_DIR.exists() else []
    resp = Response(
        render_template("index.html", samples=samples, quick_ref=QUICK_REF),
        mimetype="text/html",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp
 
 
@app.route("/convert", methods=["POST"])
def convert():
    data      = request.get_json(force=True)
    sql_text  = (data.get("sql") or "").strip()
 
    if not sql_text:
        return jsonify({"error": "No SQL provided"}), 400
 
    try:
        parsed   = _parser.parse(sql_text)
        report   = _analyzer.analyze(parsed)
        gen      = PySparkGenerator()
        code     = gen.generate(parsed, report)
 
        # Save to output/
        safe_name = (parsed.sp_name or "query").replace(".", "_").replace("[", "").replace("]", "")
        out_file  = OUTPUT_DIR / f"{safe_name}_pyspark.py"
        out_file.write_text(code, encoding="utf-8")
 
        analysis = {
            "sp_name":          parsed.sp_name,
            "is_sp":            parsed.is_stored_procedure,
            "param_count":      len(parsed.parameters),
            "temp_table_count": len(parsed.temp_tables),
            "stmt_count":       len(parsed.statements),
            "dep_count":        len(report.dependencies),
            "warning_count":    len(report.conversion_warnings),
            "complexity_score": report.complexity_score,
            "has_cursors":      report.has_cursors,
            "has_transactions": report.has_transactions,
            "has_dynamic_sql":  report.has_dynamic_sql,
            "has_window":       report.has_window_functions,
            "dependencies":     report.dependencies,
        }
 
        quick_code = _extract_quick_view(code, sql_text)
 
        return jsonify({
            "code":       code,
            "quick_code": quick_code,
            "sp_name":    parsed.sp_name or "query",
            "analysis":   analysis,
            "warnings":   report.conversion_warnings,
        })
 
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500
 
 
@app.route("/sample/<filename>")
def get_sample(filename):
    # Security: only allow .sql files inside the samples directory
    safe = pathlib.Path(filename).name
    if not safe.endswith(".sql"):
        return jsonify({"error": "Invalid file"}), 400
    path = SAMPLES_DIR / safe
    if not path.exists():
        return jsonify({"error": "Sample not found"}), 404
    return jsonify({"content": path.read_text(encoding="utf-8")})
 
 
@app.route("/health")
def health():
    return jsonify({"status": "ok", "sqlglot": _check_sqlglot()})
 
 
@app.route("/upload", methods=["POST"])
def upload_file():
    """Accept a .sql file upload and return its text content."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
 
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400
 
    ext = pathlib.Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Only .sql / .txt files are supported (got '{ext}')"}), 400
 
    safe_name = secure_filename(f.filename)
    save_path = UPLOADS_DIR / safe_name
    f.save(str(save_path))
 
    try:
        content = save_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return jsonify({"error": f"Could not read file: {exc}"}), 500
 
    return jsonify({"filename": safe_name, "content": content, "size": len(content)})
 
 
def _check_sqlglot() -> bool:
    try:
        import sqlglot
        return True
    except ImportError:
        return False
 
 
# -- AI endpoints -------------------------------------------------------------
 
@app.route("/ai-status")
def ai_status():
    """Return the active AI provider, model, and availability status."""
    info = get_provider_info()
    return jsonify({
        "available": info["available"],
        "provider":  info["provider"],
        "model":     info["model"],
    })
 
 
@app.route("/ai-convert", methods=["POST"])
def ai_convert():
    """
    Convert SQL to PySpark SQL using the active AI provider (GPT-4.1-mini,
    HuggingFace, Gemini, or Ollama).  Falls back to rule-based converter if
    no AI provider is configured.
    """
    data      = request.get_json(force=True)
    sql_text  = (data.get("sql") or "").strip()
    dialect   = (data.get("dialect") or "T-SQL").strip()

    if not sql_text:
        return jsonify({"error": "No SQL provided"}), 400

    if not ai_available():
        info = get_provider_info()
        return jsonify({
            "error": (
                f"AI provider '{info['provider']}' is not configured or not reachable. "
                "See .env.example for setup instructions."
            )
        }), 503
 
    try:
        # -- Full 3-stage pipeline: Preprocess ? LLM ? Postprocess ------------
        from converter.sql_preprocessor import preprocess
        from converter.code_postprocessor import postprocess

        pre    = preprocess(sql_text)
        code   = convert_sql_with_ai(sql_text, db_prefix="", dialect=dialect)
        post   = postprocess(code)  # re-validate final code
        code   = post.code

        safe_name = pre.sp_name.replace(".", "_").replace(" ", "_") or "ai_query"
        out_file  = OUTPUT_DIR / f"{safe_name}_pyspark.py"
        out_file.write_text(code, encoding="utf-8")

        info = get_provider_info()
        return jsonify({
            "code":             code,
            "quick_code":       code,
            "sp_name":          safe_name,
            "source":           f"{info['provider']}/{info['model']}",
            "syntax_valid":     post.syntax_valid,
            "syntax_error":     post.syntax_error,
            "warnings":         post.warnings,
            "imports_injected": post.imports_injected,
            "dialect_hints":    pre.dialect_hints,
        })
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/ai-convert-stream", methods=["POST"])
def ai_convert_stream():
    """
    SSE endpoint � runs the full 3-stage pipeline and pushes one clean [RESULT] event.
    No raw tokens are streamed; this eliminates all display artifacts in any JS version.
    Progress steps are sent as lightweight [STEP] events so old/new JS can show feedback.
    """
    data      = request.get_json(force=True)
    sql_text  = (data.get("sql") or "").strip()
    dialect   = (data.get("dialect") or "T-SQL").strip()

    if not sql_text:
        return jsonify({"error": "No SQL provided"}), 400

    if not ai_available():
        info = get_provider_info()
        return jsonify({"error": f"AI provider '{info['provider']}' not reachable."}), 503

    def generate():
        import json as _json
        try:
            # -- Stage 1: Preprocess ------------------------------------------
            yield "data: [STEP] Analysing SQL structure (Stage 1/3)\n\n"
            from converter.sql_preprocessor import preprocess
            pre = preprocess(sql_text)

            # -- Stage 2: LLM -------------------------------------------------
            yield "data: [STEP] Sending to Ollama AI (Stage 2/3 - please wait)\n\n"
            code = convert_sql_with_ai(sql_text, db_prefix="", dialect=dialect)

            # -- Stage 3: Postprocess -----------------------------------------
            yield "data: [STEP] Validating and cleaning output (Stage 3/3)\n\n"
            from converter.code_postprocessor import postprocess
            post = postprocess(code)
            code = post.code

            safe_name = pre.sp_name.replace(".", "_").replace(" ", "_") or "ai_query"
            out_file  = OUTPUT_DIR / f"{safe_name}_pyspark.py"
            out_file.write_text(code, encoding="utf-8")

            # Single compact [RESULT] event � ensure_ascii keeps it on one line
            result_payload = _json.dumps(
                {"code": code, "sp_name": safe_name,
                 "syntax_valid": post.syntax_valid,
                 "warnings": post.warnings},
                ensure_ascii=True,
            )
            yield f"data: [RESULT] {result_payload}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            import traceback as _tb
            yield f"data: [ERROR] {_json.dumps(str(exc))}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/ai-explain", methods=["POST"])
def ai_explain():
    """Add AI-generated inline comments/explanations to existing PySpark SQL code."""
    data = request.get_json(force=True)
    code = (data.get("code") or "").strip()

    if not code:
        return jsonify({"error": "No code provided"}), 400
    if not ai_available():
        return jsonify({"error": "AI provider not configured. See .env.example."}), 503
 
    try:
        explained = explain_pyspark_code(code)
        return jsonify({"code": explained})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
 
 
@app.route("/ai-optimize", methods=["POST"])
def ai_optimize():
    """Ask the active AI provider to optimise PySpark SQL code for Spark/Databricks performance."""
    data = request.get_json(force=True)
    code = (data.get("code") or "").strip()
 
    if not code:
        return jsonify({"error": "No code provided"}), 400
    if not ai_available():
        return jsonify({"error": "AI provider not configured. See .env.example."}), 503
 
    try:
        optimized = optimize_pyspark_code(code)
        return jsonify({"code": optimized})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
 
 
# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SQL ? PySpark AI Converter � Web UI")
    print("  Open: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    # Exclude output/ and uploads/ from the reloader so saving generated .py
    # files does not trigger a full server restart.
    import os as _os
    _os.environ.setdefault("WERKZEUG_RELOADER_EXTRA_FILES", "")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        extra_files=[],
        exclude_patterns=["output/*", "uploads/*"],
    )
