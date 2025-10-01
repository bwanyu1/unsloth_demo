import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)


def dataset_path() -> str:
    candidates = [
        "instruction_data.json",
        str(Path(__file__).resolve().parent.parent / "data" / "instruction_data.json"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise SystemExit(
        "instruction_data.json が見つかりません。app/ または data/ 配下に配置してください。\n"
        "例: data/instruction_data.json"
    )


def run_with_unsloth():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.environ.get("MODEL_NAME", "unsloth/llama-2-7b-bnb-4bit"),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.05
    )
    return model, tokenizer


def run_fallback_transformers():
    import torch
    from peft import LoraConfig, get_peft_model

    model_name = os.environ.get(
        "MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=(torch.float16 if device in {"cuda", "mps"} else torch.float32),
    )
    if device == "cuda":
        model = model.to("cuda")
    elif device == "mps":
        model = model.to("mps")

    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    return model, tokenizer


def main():
    try:
        model, tokenizer = run_with_unsloth()
        using = "unsloth"
    except Exception:
        print(
            "[info] Unsloth が利用できないため、macOS/CPU/MPS 用の transformers+peft 構成に切替えます。"
        )
        model, tokenizer = run_fallback_transformers()
        using = "transformers+peft"

    data_file = dataset_path()
    dataset = load_dataset("json", data_files=data_file)

    def format_example(example):
        prompt = (
            f"### Instruction\n{example.get('instruction','')}\n\n"
            f"### Input\n{example.get('input','')}\n\n"
            f"### Response\n{example.get('output','')}"
        )
        return {
            "input_ids": tokenizer(
                prompt, truncation=True, padding="max_length", max_length=512
            ).input_ids,
            "labels": tokenizer(
                example.get("output", ""),
                truncation=True,
                padding="max_length",
                max_length=512,
            ).input_ids,
        }

    dataset = dataset.map(format_example)

    args = TrainingArguments(
        per_device_train_batch_size=int(os.environ.get("BATCH", 1 if using != "unsloth" else 2)),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACC", 4)),
        warmup_steps=int(os.environ.get("WARMUP", 10)),
        max_steps=int(os.environ.get("MAX_STEPS", 200)),
        learning_rate=float(os.environ.get("LR", 2e-4)),
        fp16=os.environ.get("FP16", "auto").lower()
        == "true",  # 明示指定時のみ有効化
        logging_steps=10,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=50,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset["train"],
    )
    trainer.train()

    Path("lora-instruction").mkdir(parents=True, exist_ok=True)
    model.save_pretrained("lora-instruction")
    tokenizer.save_pretrained("lora-instruction")


if __name__ == "__main__":
    main()
