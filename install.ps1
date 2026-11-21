# ReefFlow installer for Windows (PowerShell).
#
#   Right-click > "Run with PowerShell", or in a PowerShell window:
#       powershell -ExecutionPolicy Bypass -File install.ps1
#
# What it does (internet needed once; everything afterwards runs offline):
#   1. installs the 'uv' Python manager for the current user if it is missing
#   2. creates .venv with Python 3.12 (Taichi has no wheels for newer Python)
#   3. installs ReefFlow and its dependencies into .venv
#   4. writes ReefFlow.bat (starts the browser interface) and runs a quick self-check
#
# Options:
#   -SystemCerts   use the Windows certificate store (needed on networks that
#                  inspect TLS, e.g. many corporate networks)
#   -Dev           also install the test tools (pytest)
param([switch]$SystemCerts, [switch]$Dev)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Checking for uv"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "uv not found - installing it for the current user (https://astral.sh/uv)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    $uv = Get-Command uv -ErrorAction Stop
}
Write-Host "using $($uv.Source)"

$certArg = @()
if ($SystemCerts) { $certArg = @("--system-certs") }

Step "Creating the Python 3.12 environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & uv venv .venv --python 3.12 @certArg
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

Step "Installing ReefFlow"
$target = "."
if ($Dev) { $target = ".[dev]" }
& uv pip install @certArg --python .venv\Scripts\python.exe -e $target
if ($LASTEXITCODE -ne 0) { throw "package installation failed (on a TLS-inspecting network, rerun with -SystemCerts)" }

Step "Writing ReefFlow.bat"
@"
@echo off
rem Starts the ReefFlow browser interface (runs locally; no internet needed).
cd /d "%~dp0"
".venv\Scripts\reefflow.exe" ui %*
"@ | Set-Content -Encoding ASCII ReefFlow.bat

Step "Self-check (GPU detection and a tiny solver run)"
& .venv\Scripts\python.exe -m reefflow.selfcheck
if ($LASTEXITCODE -ne 0) { Write-Warning "self-check reported a problem (see above)" }

Write-Host "`nDone. Double-click ReefFlow.bat to start, or use the command line: .venv\Scripts\reefflow --help" -ForegroundColor Green
