# unsloth_demoIt_ai — Quickstart

概要
- macOS でも動くように、`unsloth` が使えない環境では自動で `transformers + peft` にフォールバックします。
- 学習データは JSON 設定ファイル（`dataset.json`）で指定できます。Alpaca / ShareGPT 形式に対応。

前提
- Python 3.10 系（`>=3.10,<3.11`）
- `uv` 推奨（無ければ `pip` でも可）

セットアップ（macOS / CPU / MPS）
```
cd app
uv sync
uv run python itai.py
```
環境変数で一部を調整できます：
- `MODEL_NAME` デフォルト: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `MAX_STEPS`、`BATCH`、`GRAD_ACC`、`LR`、`FP16`（true/false）

 Linux x86_64 + CUDA (GPU) — Unsloth 版
```
cd app
 uv sync
 MODEL_NAME=unsloth/llama-2-7b-bnb-4bit uv run python itai.py
```

Docker 開発
```
docker compose up -d --build
docker compose exec dev bash
cd /workspace/app
uv sync            # macOS でも GPU なし構成で OK
uv run python itai.py
```

Windows（ネイティブ）
- 目標: Windows 上で簡単に学習を動かす（Unsloth が無い場合は自動フォールバック）。
- 前提: Python 3.10 系、PowerShell、（推奨）uv。

手順（PowerShell）
```
git clone <this repo>
cd unsloth_demo
./scripts/windows_run.ps1
```
環境変数例（PowerShell）
```
$env:MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
$env:MAX_STEPS = "200"; $env:BATCH = "1"; $env:GRAD_ACC = "4"
$env:LR = "2e-4"; $env:FP16 = "false"
./scripts/windows_run.ps1
```
注意
- Windows ネイティブ環境では `unsloth` の公式ホイールが提供されないことが多く、
  その場合は実行時に自動で `transformers + peft` に切替わります（学習自体は可能）。

CUDA（Windows）で Unsloth を使う
- NVIDIA Driver/Toolkit 導入済みであれば、通常依存として `unsloth` が解決対象になります。
- 自動検出（`nvidia-smi` が見つかった場合）は `MODEL_NAME` 既定値を Unsloth 用に設定します：
  - `./scripts/windows_run.ps1`
- 手動で実行する場合：
  ```
  cd app
  uv sync
  $env:MODEL_NAME = "unsloth/llama-2-7b-bnb-4bit"
  uv run python itai.py
  ```
注: OS マーカーで Windows/AMD64 を許可していますが、CUDA/PyTorch 構成が未整備だと `unsloth` のインストール/実行は失敗します。

Windows で Unsloth を使う（推奨: WSL2 + GPU）
1) Windows に WSL2 Ubuntu を導入し、NVIDIA ドライバ（Windows 側）を最新化。
2) Ubuntu 側で `uv` をインストール。
3) 本リポジトリを WSL 側にクローン。
4) 下記スクリプトでセットアップ＆実行：
```
bash scripts/wsl_gpu_setup_and_run.sh
```
WSL2 であれば Linux x86_64 扱いとなり、`uv sync` で `unsloth` が導入されます。

Docker（Windows）で GPU を使う場合のヒント
- Docker Desktop + WSL2 integration を有効化し、NVIDIA GPU を共有（`--gpus all`）。
- コンテナ内で `uv sync --extra gpu` を実行。
- 既存の `docker-compose.yml` を利用しつつ、実行時に `docker compose run --gpus all dev bash` などで GPU を渡してください。

データ指定（JSON 設定）

`app/dataset.json` または `data/dataset.json` を用意します（上が優先）。

Alpaca 形式の例
```json
{
  "format": "alpaca",
  "train": "data/instruction_data.json",
  "val_ratio": 0.05,
  "max_length": 512,
  "fields": {"instruction": "instruction", "input": "input", "output": "output"}
}
```

ShareGPT 形式の例（conversations）
```json
{
  "format": "sharegpt",
  "train": "data/sharegpt.json",
  "validation": "data/sharegpt_val.json",
  "field": "conversations",
  "max_length": 512
}
```

補足
- `train` と `validation` の両方を指定しない場合、`val_ratio` で分割します。
- 相対パスは `dataset.json` が置かれた場所からも解決されます。
- `format: alpaca` は `instruction`/`input`/`output` の 3 フィールドを想定（`fields` で名前変更可）。
- `format: sharegpt` は `{conversations: [{from|role, value|content}, ...]}` を簡易対応し、最初の user→assistant のペアで SFT します。

レガシー（後方互換）
- `DATA_FILE=/path/to/your.jsonl` を環境変数で指定、または `instruction_data.json` を `app/` / `data/` に置く方法も引き続き利用できます。

トラブルシュート
- 「Unsloth が見つからない」: macOS では自動で transformers+peft に切替わります。
- 「instruction_data.json が無い」: `app/` または `data/` に置いてください。
- Python バージョンエラー: 3.10 系を使用してください。

Windows 補足
- `uv` が見つからない場合は PowerShell で以下：
  ```
  iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
  ```
- CUDA を使うには対応 GPU/Driver が必要です。WSL2 での実行を推奨します。
