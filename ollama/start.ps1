# Ollama Startup Script
# Run this from the project root to start Ollama and verify it is ready.
# Usage: .\ollama\start.ps1

$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
$OllamaCmd = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Ollama Startup — SQL to PySpark SQL Converter"            -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is installed
if (-not (Test-Path $OllamaExe)) {
    Write-Host "[ERROR] Ollama not found at: $OllamaExe" -ForegroundColor Red
    Write-Host "  Install from: https://ollama.com" -ForegroundColor Yellow
    exit 1
}

# Check if already running
try {
    $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "[OK] Ollama is already running." -ForegroundColor Green
} catch {
    Write-Host "[INFO] Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath $OllamaExe -WindowStyle Hidden
    Write-Host "      Waiting for Ollama to initialise..."
    Start-Sleep -Seconds 5

    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
        Write-Host "[OK] Ollama started successfully." -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Ollama failed to start. Try launching from the system tray." -ForegroundColor Red
        exit 1
    }
}

# Add ollama to PATH for this session
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"

# List installed models
Write-Host ""
Write-Host "Installed Models:" -ForegroundColor Cyan
& $OllamaCmd list

# Check if required model is available
Write-Host ""
$model = "qwen2.5-coder:7b"
$installed = & $OllamaCmd list 2>&1
if ($installed -match [regex]::Escape($model)) {
    Write-Host "[OK] Model '$model' is installed and ready." -ForegroundColor Green
} else {
    Write-Host "[WARN] Model '$model' not found. Pulling now (~4.7 GB)..." -ForegroundColor Yellow
    & $OllamaCmd pull $model
}

# Final status
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Ollama is ready at: http://localhost:11434"               -ForegroundColor Green
Write-Host "  Model: $model"                                            -ForegroundColor Green
Write-Host "  Start the web app: python app.py"                         -ForegroundColor Green
Write-Host "  Open browser:      http://localhost:5000"                 -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
