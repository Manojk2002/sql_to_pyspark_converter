# Ollama — Local AI for SQL → PySpark SQL Conversion

## What is Ollama?

Ollama is a **free, open-source** tool that runs large language models (AI) **locally on your own machine**.
No internet required after setup. No API key. No subscription. No cost per query.

It acts as a private AI server running at `http://localhost:11434`.

---

## How Ollama is used in this project

```
You (browser)
    │
    │  paste SQL stored procedure / query
    ▼
Flask App (app.py — http://localhost:5000)
    │
    │  POST /ai-convert
    ▼
ai/llm_client.py  ←── AI_PROVIDER=ollama (.env)
    │
    │  POST http://localhost:11434/api/chat
    │  model: qwen2.5-coder:7b
    │  system: 7-step conversion instructions
    │  user: your SQL
    ▼
Ollama Server (running locally)
    │
    │  qwen2.5-coder:7b generates PySpark SQL
    ▼
spark.sql("...") code returned to browser
```

---

## Installed Model

| Property     | Value                        |
|--------------|------------------------------|
| Model Name   | `qwen2.5-coder:7b`           |
| Model ID     | `dae161e27b0e`               |
| Size on Disk | 4.7 GB                       |
| Context      | 4096 tokens                  |
| Speciality   | Code generation (SQL, Python)|
| Cost         | $0 — runs on your CPU        |
| Developer    | Alibaba / Qwen Team          |

**Why this model?**
`Qwen2.5-Coder-7B` is ranked among the top open-source code models.
It understands both SQL syntax and Python/PySpark, making it ideal for this conversion task.

---

## Project Configuration

`.env` file (project root):
```
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
```

`ai/llm_client.py` auto-detects Ollama by pinging `localhost:11434`.
If Ollama is running, it is selected automatically — no configuration needed.

---

## Starting Ollama

Ollama installs itself as a **Windows startup application** — it starts automatically when you log in.

To start it manually:
```powershell
# Option 1: From the system tray icon (recommended)
# Look for the Ollama llama icon in the Windows taskbar system tray

# Option 2: PowerShell script (see start.ps1 in this folder)
.\ollama\start.ps1

# Option 3: Direct executable
& "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
```

---

## Checking Ollama Status

```powershell
# List installed models
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama list

# Check which models are currently loaded/running
ollama ps

# Test the API directly
Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
```

---

## Available Free Models (alternatives)

| Model                  | Size   | Best For              |
|------------------------|--------|-----------------------|
| `qwen2.5-coder:7b`     | 4.7 GB | SQL/Python code ✓ installed |
| `qwen2.5-coder:1.5b`   | 986 MB | Fast, lightweight      |
| `codellama`            | 3.8 GB | Code generation        |
| `deepseek-coder:6.7b`  | 3.8 GB | Code generation        |
| `mistral`              | 4.1 GB | General purpose        |

To switch models, update `.env`:
```
OLLAMA_MODEL=codellama
```
Then pull the model: `ollama pull codellama`

---

## Cost Comparison

| Provider              | Cost per Conversion | Internet | Privacy |
|-----------------------|---------------------|----------|---------|
| **Ollama (this setup)** | **$0.00**         | ❌ None  | ✅ 100% |
| OpenAI GPT-4.1-mini   | ~$0.001             | ✅ Yes   | ⚠️ Sent to OpenAI |
| Google Gemini Flash   | $0 (1M/month free) | ✅ Yes   | ⚠️ Sent to Google |
| Anthropic Claude Haiku| ~$0.0008            | ✅ Yes   | ⚠️ Sent to Anthropic |

---

## Conversion Quality

Ollama with `qwen2.5-coder:7b` correctly handles:
- ✅ `SELECT / JOIN / GROUP BY / HAVING` → `spark.sql()`
- ✅ `DECLARE CURSOR / FETCH NEXT` → batch `spark.sql(UPDATE ...)` or `DeltaTable.update()`
- ✅ `CREATE PROCEDURE` → Python function with `spark: SparkSession`
- ✅ `BEGIN TRANSACTION / COMMIT` → Delta Lake ACID comments
- ✅ `CREATE TABLE #tmp` → `spark.sql("CREATE OR REPLACE TEMP VIEW ...")`
- ✅ `IF/ELSE / WHILE` → Python `if/else` / `while`
- ✅ `MERGE INTO` → `spark.sql("MERGE INTO ...")`
- ✅ Window functions (`ROW_NUMBER OVER`) → preserved in `spark.sql()`

**Note on speed:** The 7B model runs on CPU at ~10–30 tokens/second.
A typical stored procedure takes 30–90 seconds to convert.
GPU acceleration (if available) reduces this to 5–15 seconds.
