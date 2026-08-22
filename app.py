"""
Flask Web Application - SQL to PySpark AI Converter
Run: python app.py
Open: http://localhost:5000
"""

import os
import sys
import json
import time
import pathlib
import traceback
import subprocess
import threading
import urllib.request

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from converter.sql_parser import SQLParser
from converter.sql_analyzer import SQLAnalyzer
from converter.code_generator import PySparkGenerator
from converter.sql_preprocessor import preprocess
from converter.code_postprocessor import postprocess
from ai_provider.ai_provider import (
    convert_sql_with_ai,
    stream_sql_with_ai,
    is_available as ai_available,
    get_provider_info,
    get_model_for_input,
    explain_pyspark_code,
    optimize_pyspark_code,
)
from adf_export.data_exporter import export_data, test_connection, EXPORT_DIR


# ── Ollama auto-start ─────────────────────────────────────────────────────────

_OLLAMA_PATHS = [
    os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
    r"C:\Program Files\Ollama\ollama.exe",
    "ollama",  # if already on PATH
]
_OLLAMA_API = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_FAST_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")


def _ollama_reachable(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{_OLLAMA_API}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def _start_ollama_serve():
    """Launch `ollama serve` in the background if not already running."""
    if _ollama_reachable():
        return True
    for path in _OLLAMA_PATHS:
        if path == "ollama" or os.path.exists(path):
            try:
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                subprocess.Popen(
                    [path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=flags,
                )
                print("  [Ollama] Starting server...", flush=True)
                for _ in range(15):          # wait up to 15 s
                    time.sleep(1)
                    if _ollama_reachable():
                        print("  [Ollama] Server ready.", flush=True)
                        return True
                print("  [Ollama] Timed out waiting for server.", flush=True)
                return False
            except Exception as e:
                print(f"  [Ollama] Could not start: {e}", flush=True)
    return False


def _prewarm_ollama():
    """Load the model into RAM/GPU so the first real request is instant."""
    if not _ollama_reachable():
        return
    try:
        payload = json.dumps({
            "model": _OLLAMA_FAST_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 1},
            "keep_alive": -1,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_API}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60):
            pass
        print(f"  [Ollama] Model '{_OLLAMA_FAST_MODEL}' warmed up — ready!", flush=True)
    except Exception as e:
        print(f"  [Ollama] Pre-warm skipped: {e}", flush=True)


def _ensure_ollama_background():
    """Start Ollama and pre-warm model in a background thread (non-blocking)."""
    def _run():
        if _start_ollama_serve():
            _prewarm_ollama()
    threading.Thread(target=_run, daemon=True).start()


app = Flask(__name__, template_folder="web_ui/templates")
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

ALLOWED_EXTENSIONS = {".sql", ".txt"}

SAMPLES_DIR = pathlib.Path(__file__).parent / "samples"
OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
UPLOADS_DIR = pathlib.Path(__file__).parent / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

_parser = SQLParser()
_analyzer = SQLAnalyzer()

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
 
 
# --- Routes ------------------------------------------------------------------

def _extract_quick_view(code: str, sql_input: str) -> str:
    """Extract just the converted logic from the full generated file.

    Strips the file header, imports, and validation scaffold so the user
    sees only the PySpark SQL statements they care about.
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
        # No marker - try to find first spark.sql() line
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
        clean = sql_input.strip().rstrip(";")
        return (
            '# Could not parse SQL - using spark.sql() passthrough\n'
            'result_df = spark.sql("""\n'
            + "\n".join(f"    {l}" for l in clean.split("\n"))
            + '\n""")'
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
    data = request.get_json(force=True)
    sql_text = (data.get("sql") or "").strip()

    if not sql_text:
        return jsonify({"error": "No SQL provided"}), 400

    try:
        parsed = _parser.parse(sql_text)
        report = _analyzer.analyze(parsed)
        gen = PySparkGenerator()
        code = gen.generate(parsed, report)

        safe_name = (parsed.sp_name or "query").replace(".", "_").replace("[", "").replace("]", "")
        out_file = OUTPUT_DIR / f"{safe_name}_pyspark.py"
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
 
 
# --- AI endpoints ------------------------------------------------------------

def _rule_based_convert(sql_text: str, as_dict: bool = False):
    """Run the rule-based converter and return a JSON response or dict."""
    parsed = _parser.parse(sql_text)
    report = _analyzer.analyze(parsed)
    gen = PySparkGenerator()
    code = gen.generate(parsed, report)
    safe_name = (parsed.sp_name or "query").replace(".", "_").replace("[", "").replace("]", "")
    out_file = OUTPUT_DIR / f"{safe_name}_pyspark.py"
    out_file.write_text(code, encoding="utf-8")
    quick_code = _extract_quick_view(code, sql_text)
    result = {
        "code":       code,
        "quick_code": quick_code,
        "sp_name":    safe_name,
        "model_used": "rule-based",
        "warnings":   report.conversion_warnings,
        "source":     "rule-based",
    }
    if as_dict:
        return result
    return jsonify(result)

@app.route("/ai-status")
def ai_status():
    """Return the active AI provider, model, and availability status."""
    info = get_provider_info()
    return jsonify({
        "available":        info["available"],
        "provider":         info["provider"],
        "model":            info["model"],
        "small_model":      info.get("small_model", info["model"]),
        "large_model":      info.get("large_model", info["model"]),
        "input_threshold":  info.get("input_threshold", 0),
    })
 
 
@app.route("/ai-convert", methods=["POST"])
def ai_convert():
    """
    Convert SQL to PySpark SQL using the active AI provider.
    Falls back to rule-based converter if no AI provider is reachable.
    """
    data      = request.get_json(force=True)
    sql_text  = (data.get("sql") or "").strip()
    dialect   = (data.get("dialect") or "T-SQL").strip()

    if not sql_text:
        return jsonify({"error": "No SQL provided"}), 400

    if not ai_available():
        # Fall back to rule-based converter
        return _rule_based_convert(sql_text)
 
    try:
        # preprocess once for sp_name / dialect_hints metadata
        pre  = preprocess(sql_text)
        # convert_sql_with_ai runs preprocess+LLM+postprocess internally
        code = convert_sql_with_ai(sql_text, db_prefix="", dialect=dialect)

        safe_name = pre.sp_name.replace(".", "_").replace(" ", "_") or "ai_query"
        out_file  = OUTPUT_DIR / f"{safe_name}_pyspark.py"
        out_file.write_text(code, encoding="utf-8")

        info = get_provider_info()
        model_used = get_model_for_input(sql_text) if info["provider"] == "ollama" else info["model"]
        return jsonify({
            "code":             code,
            "quick_code":       code,
            "sp_name":          safe_name,
            "source":           f"{info['provider']}/{model_used}",
            "model_used":       model_used,
            "syntax_valid":     True,
            "syntax_error":     None,
            "warnings":         [],
            "imports_injected": [],
            "dialect_hints":    pre.dialect_hints,
        })
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/ai-convert-stream", methods=["POST"])
def ai_convert_stream():
    """SSE endpoint - runs the full 3-stage pipeline and pushes one [RESULT] event.

    Progress is reported via lightweight [STEP] events before the final result.
    """
    data = request.get_json(force=True)
    sql_text = (data.get("sql") or "").strip()
    dialect = (data.get("dialect") or "T-SQL").strip()

    if not sql_text:
        return jsonify({"error": "No SQL provided"}), 400

    if not ai_available():
        # AI offline — run rule-based converter and emit as a stream result
        def _fallback_generate():
            yield "data: [STEP] AI provider offline — using rule-based converter\n\n"
            try:
                result = _rule_based_convert(sql_text, as_dict=True)
                payload = json.dumps({
                    "code":         result["code"],
                    "sp_name":      result["sp_name"],
                    "syntax_valid": True,
                    "warnings":     result.get("warnings", []),
                    "model_used":   "rule-based",
                }, ensure_ascii=True)
                yield f"data: [RESULT] {payload}\n\n"
            except Exception as exc:
                yield f"data: [ERROR] {json.dumps(str(exc))}\n\n"
            yield "data: [DONE]\n\n"
        return Response(
            stream_with_context(_fallback_generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def generate():
        try:
            yield "data: [STEP] Analysing SQL structure (Stage 1/3)\n\n"
            pre = preprocess(sql_text)

            info = get_provider_info()
            model_used = get_model_for_input(sql_text) if info["provider"] == "ollama" else info["model"]
            yield f"data: [STEP] Generating — {model_used} (Stage 2/3)\n\n"

            # Stream tokens live from the model instead of blocking
            raw_chunks = []
            for chunk in stream_sql_with_ai(sql_text, db_prefix="", dialect=dialect):
                raw_chunks.append(chunk)
                yield f"data: [TOKEN] {json.dumps(chunk, ensure_ascii=True)}\n\n"

            yield "data: [STEP] Validating and cleaning output (Stage 3/3)\n\n"
            post = postprocess("".join(raw_chunks))

            safe_name = (pre.sp_name or "ai_query").replace(".", "_").replace(" ", "_")
            out_file = OUTPUT_DIR / f"{safe_name}_pyspark.py"
            out_file.write_text(post.code, encoding="utf-8")

            result_payload = json.dumps(
                {
                    "code":         post.code,
                    "sp_name":      safe_name,
                    "syntax_valid": post.syntax_valid,
                    "warnings":     post.warnings,
                    "model_used":   model_used,
                },
                ensure_ascii=True,
            )
            yield f"data: [RESULT] {result_payload}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield f"data: [ERROR] {json.dumps(str(exc))}\n\n"

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
        return jsonify({"code": code, "warning": "AI provider offline — code returned unchanged."})
 
    try:
        explained = explain_pyspark_code(code)
        return jsonify({"code": explained})
    except Exception as exc:
        return jsonify({"code": code, "warning": str(exc)})
 
 
@app.route("/ai-optimize", methods=["POST"])
def ai_optimize():
    """Ask the active AI provider to optimise PySpark SQL code for Spark/Databricks performance."""
    data = request.get_json(force=True)
    code = (data.get("code") or "").strip()
 
    if not code:
        return jsonify({"error": "No code provided"}), 400
    if not ai_available():
        return jsonify({"code": code, "warning": "AI provider offline — code returned unchanged."})
 
    try:
        optimized = optimize_pyspark_code(code)
        return jsonify({"code": optimized})
    except Exception as exc:
        return jsonify({"code": code, "warning": str(exc)})
 
 
# --- ADF Data Export endpoints ------------------------------------------------

@app.route("/adf")
def adf_page():
    """Render the ADF Data Export page."""
    return render_template("adf_export.html")


@app.route("/adf/test-connection", methods=["POST"])
def adf_test_connection():
    """Test database connectivity."""
    data = request.get_json(force=True)
    result = test_connection(
        db_type=data.get("db_type", ""),
        host=data.get("host", ""),
        port=data.get("port", ""),
        database=data.get("database", ""),
        username=data.get("username", ""),
        password=data.get("password", ""),
    )
    return jsonify(result)


@app.route("/adf/export", methods=["POST"])
def adf_export():
    """Execute query and export data to selected formats."""
    data = request.get_json(force=True)
    result = export_data(
        db_type=data.get("db_type", ""),
        host=data.get("host", ""),
        port=data.get("port", ""),
        database=data.get("database", ""),
        username=data.get("username", ""),
        password=data.get("password", ""),
        query=data.get("query", ""),
        formats=data.get("formats", []),
        output_name=data.get("output_name", "export_data"),
    )
    return jsonify({
        "success": result.success,
        "records_extracted": result.records_extracted,
        "columns_extracted": result.columns_extracted,
        "files_created": result.files_created,
        "error": result.error,
    })


@app.route("/adf/download/<filename>")
def adf_download(filename):
    """Download an exported file."""
    from flask import send_from_directory
    safe = pathlib.Path(filename).name
    filepath = EXPORT_DIR / safe
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(str(EXPORT_DIR), safe, as_attachment=True)


# --- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SQL -> PySpark AI Converter - Web UI")
    print("  Open: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Auto-start Ollama + pre-warm model in background (non-blocking)
    _ensure_ollama_background()

    os.environ.setdefault("WERKZEUG_RELOADER_EXTRA_FILES", "")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        extra_files=[],
        exclude_patterns=["output/*", "uploads/*"],
    )
