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
uv sync --extra gpu
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
