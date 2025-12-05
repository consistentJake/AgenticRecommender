#!/usr/bin/env python
"""
LoRA finetuning for Qwen2.5-0.5B-Instruct using TRL's SFTTrainer.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen2_5_0_5b_lora_simplifier"
TRAIN_PATH = "data/train.json"
EVAL_PATH = "data/eval.json"
MAX_SEQ_LEN = 512


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_json(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_datasets(train_path: str, eval_path: str):
    train_data = load_json(train_path)
    eval_data = load_json(eval_path)

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)
    return train_ds, eval_ds


def make_preprocess_fn(tokenizer, max_length: int):
    def preprocess(example):
        messages = example["messages"]
        # Turn chat-style messages into a single prompt string
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Right-pad / truncate to a fixed length
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return preprocess


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load base model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build datasets
    train_ds, eval_ds = build_datasets(TRAIN_PATH, EVAL_PATH)

    preprocess_fn = make_preprocess_fn(tokenizer, MAX_SEQ_LEN)
    tokenized_train = train_ds.map(preprocess_fn, batched=False)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=False)

    # LoRA configuration (PEFT)
    lora_config = LoraConfig(
        r=16,                        # rank
        lora_alpha=64,               # scaling
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training hyperparameters (adapt for your GPU)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=f"{OUTPUT_DIR}/logs",

        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,   # effective batch size 2 * 16 = 32

        warmup_steps=50,
        max_steps=200,                    # fixed number of steps (no epochs)
        learning_rate=5e-5,

        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=50,

        bf16=torch.cuda.is_available(),  # if you want FP16/BF16, adjust as needed
    )

    # TRL SFTTrainer handles PEFT integration
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Adapter + tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
