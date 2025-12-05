#!/usr/bin/env python
"""LoRA finetuning for Qwen3-0.6B on MovieLens using TRL's SFTTrainer.

Configuration is loaded from a YAML file (default: configs/qwen3_movielens_qlora.yaml).
Most training/LoRA hyperparameters can be overridden there instead of hardcoding values.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer


BASE_MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "output/qwen3-movielens-qlora"
DATA_DIR = Path("data/movielens_qwen3")
TRAIN_PATH = DATA_DIR / "train.json"
EVAL_PATH = DATA_DIR / "eval.json"
MAX_SEQ_LEN = 1024
MAX_EVAL_SAMPLES: Optional[int] = 1000

PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1.5e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_EVAL_STEPS = 2000

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


def maybe_shrink_eval(eval_ds: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None:
        return eval_ds
    return eval_ds.select(range(min(len(eval_ds), max_samples)))


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _strip_split_suffix(name: str) -> str:
    for suffix in ("_train", "_eval", "_test"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_dataset_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    dataset_dir = Path(cfg.get("dataset_dir", "data"))

    # Explicit overrides win.
    if "train_file" in cfg and "eval_file" in cfg:
        return Path(cfg["train_file"]), Path(cfg["eval_file"])

    dataset_key = cfg.get("dataset", "movielens_qwen3_train")
    eval_dataset_key = cfg.get("eval_dataset", "movielens_qwen3_eval")
    base_key = _strip_split_suffix(dataset_key)

    info_path = dataset_dir / "dataset_info.json"
    if info_path.exists():
        info = load_json(info_path)
        if base_key in info:
            train_rel = info[base_key].get("file_name")
            eval_rel = info[base_key].get("file_name_eval")
            if train_rel and eval_rel:
                return dataset_dir / train_rel, dataset_dir / eval_rel

    # Fallback: assume folder matches base_key with train/eval splits.
    return (
        dataset_dir / base_key / "train.json",
        dataset_dir / base_key / "eval.json",
    )


def resolve_lora_targets(cfg_value: Any) -> List[str]:
    if cfg_value is None or cfg_value == "all":
        return LORA_TARGET_MODULES
    if isinstance(cfg_value, str):
        return [part.strip() for part in cfg_value.split(",") if part.strip()]
    if isinstance(cfg_value, list):
        return [str(x) for x in cfg_value]
    return LORA_TARGET_MODULES


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen3_movielens_qlora.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    # Enable 4-bit QLoRA on CUDA; otherwise fall back to full precision.
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
        cfg.get("model_name_or_path", BASE_MODEL),
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to(device)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("model_name_or_path", BASE_MODEL),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build datasets
    train_path, eval_path = resolve_dataset_paths(cfg)
    train_ds, eval_ds = build_datasets(str(train_path), str(eval_path))
    eval_ds = maybe_shrink_eval(eval_ds, cfg.get("max_eval_samples", MAX_EVAL_SAMPLES))

    # LoRA configuration (PEFT)
    lora_config = LoraConfig(
        r=cfg.get("lora_rank", LORA_RANK),
        lora_alpha=cfg.get("lora_alpha", LORA_ALPHA),
        target_modules=resolve_lora_targets(cfg.get("lora_target")),
        lora_dropout=cfg.get("lora_dropout", LORA_DROPOUT),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training hyperparameters (adapt for your GPU)
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", OUTPUT_DIR),
        logging_dir=f"{cfg.get('output_dir', OUTPUT_DIR)}/logs",

        per_device_train_batch_size=cfg.get(
            "per_device_train_batch_size", PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        per_device_eval_batch_size=cfg.get(
            "per_device_eval_batch_size", PER_DEVICE_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=cfg.get(
            "gradient_accumulation_steps", GRADIENT_ACCUMULATION_STEPS
        ),

        num_train_epochs=cfg.get("num_train_epochs", NUM_TRAIN_EPOCHS),
        learning_rate=cfg.get("learning_rate", LEARNING_RATE),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", WARMUP_RATIO),
        weight_decay=cfg.get("weight_decay", 0.0),
        adam_beta1=cfg.get("adam_beta1", 0.9),
        adam_beta2=cfg.get("adam_beta2", 0.999),
        adam_epsilon=cfg.get("adam_epsilon", 1.0e-8),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),

        logging_steps=cfg.get("logging_steps", LOGGING_STEPS),
        save_steps=cfg.get("save_steps", cfg.get("eval_steps", SAVE_EVAL_STEPS)),
        save_total_limit=cfg.get("save_total_limit", 2),
        evaluation_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", SAVE_EVAL_STEPS),
        predict_with_generate=True,
        generation_max_length=cfg.get("cutoff_len", MAX_SEQ_LEN),
        remove_unused_columns=False,

        bf16=cfg.get("bf16", torch.cuda.is_available()),
        fp16=cfg.get("fp16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
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

    def _normalize_label(text: str) -> Optional[int]:
        cleaned = text.strip().lower()
        if cleaned.startswith("yes"):
            return 1
        if cleaned.startswith("no"):
            return 0
        return None

    def compute_metrics(eval_preds: Tuple[Any, Any]) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pairs = []
        for pred_text, label_text in zip(pred_texts, label_texts):
            p = _normalize_label(pred_text)
            t = _normalize_label(label_text)
            if p is not None and t is not None:
                pairs.append((p, t))

        if not pairs:
            return {"accuracy": 0.0, "f1": 0.0}

        y_pred, y_true = zip(*pairs)
        y_pred_arr = np.array(y_pred)
        y_true_arr = np.array(y_true)

        accuracy = float((y_pred_arr == y_true_arr).mean())

        tp = float(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
        fp = float(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
        fn = float(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = float(0.0 if denom == 0 else (2 * tp) / denom)

        return {"accuracy": accuracy, "f1": f1}

    # TRL SFTTrainer handles PEFT integration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=training_args,
        max_seq_length=cfg.get("cutoff_len", MAX_SEQ_LEN),
        formatting_func=formatting_func,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.get("early_stopping_patience", 2)
            )
        ],
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    out_dir = Path(cfg.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Training complete. Adapter + tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
