#!/usr/bin/env bash
set -euo pipefail

# WSL2 + NVIDIA GPU path for Windows to run Unsloth (Linux x86_64)
# Requires: WSL2 Ubuntu, NVIDIA drivers on Windows, Docker optional

cd "$(dirname "$0")/.."
cd app

echo "[step] Sync env (Unsloth is a regular dependency for x86_64 Linux)…"
uv sync

echo "[step] Run with Unsloth default 4-bit model…"
MODEL_NAME=${MODEL_NAME:-unsloth/llama-2-7b-bnb-4bit} uv run python itai.py
