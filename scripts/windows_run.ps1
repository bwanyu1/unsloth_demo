<#
Windows (PowerShell) quick run for this repo.

Prerequisites
- Python 3.10.x installed and on PATH
- (Recommended) uv package manager installed

Usage
PS> ./scripts/windows_run.ps1
Optional env vars:
  $env:MODEL_NAME, $env:MAX_STEPS, $env:BATCH, $env:GRAD_ACC, $env:LR, $env:FP16

Notes
- On Windows native, Unsloth wheels are typically unavailable. The app
  automatically falls back to transformers+peft.
- To use Unsloth on Windows, prefer WSL2 + NVIDIA GPU (see README section).
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-UvInstalled {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "[info] uv not found. Installing to user profile..." -ForegroundColor Yellow
        # Official installer
        Invoke-Expression (&([ScriptBlock]::Create((Invoke-WebRequest -UseBasicParsing https://astral.sh/uv/install.ps1).Content)))
        $env:Path = "$env:USERPROFILE\.cargo\bin;$env:USERPROFILE\.local\bin;$env:Path"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv installation failed. Please open a new shell or install manually: https://docs.astral.sh/uv/"
        }
    }
}

Push-Location "$PSScriptRoot/.."
try {
    Ensure-UvInstalled
    Set-Location "app"
    $hasNvidiaSmi = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasNvidiaSmi -and -not $env:MODEL_NAME) {
        # Unsloth が通常依存のため uv sync だけで OK。既定モデルを GPU 用に設定。
        $env:MODEL_NAME = 'unsloth/llama-2-7b-bnb-4bit'
    }

    Write-Host "[step] Resolving environment (platform-marked dependencies)…" -ForegroundColor Cyan
    uv sync

    Write-Host "[step] Launching training (Unsloth on Win/Linux x86_64, else fallback)…" -ForegroundColor Cyan
    uv run python itai.py
}
finally {
    Pop-Location
}
