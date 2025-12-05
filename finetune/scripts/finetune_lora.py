#!/usr/bin/env python
"""LoRA finetuning for Qwen3-0.6B on MovieLens using TRL's SFTTrainer."""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


BASE_MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "output/qwen3-movielens-qlora"
DATA_DIR = Path("data/movielens_qwen3")
TRAIN_PATH = DATA_DIR / "train.json"
EVAL_PATH = DATA_DIR / "eval.json"
MAX_SEQ_LEN = 2048

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1.5e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_EVAL_STEPS = 200

LORA_RANK = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]  # mirrors `lora_target: all` from the config


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


def to_chat_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert supervised sample to chat-style messages expected by Qwen."""
    messages: List[Dict[str, str]] = []

    system_prompt = example.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # History is present but empty in current data; keep for forward compatibility.
    for turn in example.get("history") or []:
        role = turn.get("role")
        content = turn.get("content")
        if role and content:
            messages.append({"role": role, "content": content})

    user_content = example.get("instruction", "")
    example_input = example.get("input")
    if example_input:
        user_content = f"{user_content}\n\n{example_input}"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example.get("output", "")})

    return messages


def main():
    device = get_device()
    print(f"Using device: {device}")

    quantization_config = None
    device_map = None
    torch_dtype = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to(device)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build datasets
    train_ds, eval_ds = build_datasets(TRAIN_PATH, EVAL_PATH)

    # LoRA configuration (PEFT)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training hyperparameters (adapt for your GPU)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=f"{OUTPUT_DIR}/logs",

        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        max_grad_norm=1.0,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS,

        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
    )

    def formatting_func(batch: Dict[str, List[Any]]) -> List[str]:
        prompts: List[str] = []
        for idx in range(len(batch["instruction"])):
            example = {key: value[idx] for key, value in batch.items()}
            prompts.append(
                tokenizer.apply_chat_template(
                    to_chat_messages(example),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return prompts

    # TRL SFTTrainer handles PEFT integration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=training_args,
        max_seq_length=MAX_SEQ_LEN,
        formatting_func=formatting_func,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Adapter + tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
