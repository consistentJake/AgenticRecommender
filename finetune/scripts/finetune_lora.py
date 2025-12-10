#!/usr/bin/env python
"""LoRA finetuning for Qwen3-0.6B on MovieLens using TRL's SFTTrainer.

Configuration is loaded from a YAML file (default: configs/qwen3_movielens_qlora.yaml).
Most training/LoRA hyperparameters can be overridden there instead of hardcoding values.
"""

import argparse
import json
import multiprocessing as mp
import os
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
    TrainerCallback,
    TrainerControl,
    TrainerState,
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


def resolve_dataset_dir(raw_dir: Path, config_path: Path) -> Path:
    if raw_dir.is_absolute() or raw_dir.exists():
        return raw_dir
    cfg_root = config_path.parent.parent  # e.g., finetune/ from finetune/configs/...
    candidate = cfg_root / raw_dir
    if candidate.exists():
        return candidate
    return raw_dir


def resolve_dataset_paths(cfg: Dict[str, Any], config_path: Path) -> Tuple[Path, Path]:
    dataset_dir = resolve_dataset_dir(Path(cfg.get("dataset_dir", "data")), config_path)

    # Explicit overrides win.
    if "train_file" in cfg and "eval_file" in cfg:
        train_file = Path(cfg["train_file"])
        eval_file = Path(cfg["eval_file"])
        if not train_file.is_absolute():
            train_file = dataset_dir / train_file
        if not eval_file.is_absolute():
            eval_file = dataset_dir / eval_file
        return train_file, eval_file

    dataset_key = cfg.get("dataset", "movielens_qwen3_train")
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


def resolve_attention_impl(cfg: Dict[str, Any]) -> Optional[str]:
    attn_val = cfg.get("flash_attn")
    if attn_val is None:
        return None
    if isinstance(attn_val, str):
        attn_val = attn_val.lower()
        if attn_val == "auto":
            return "flash_attention_2"
        if attn_val in {"flash_attention_2", "sdpa", "eager"}:
            return attn_val
        if attn_val in {"false", "none", "off"}:
            return None
    return None


def resolve_report_to(cfg: Dict[str, Any]) -> List[str]:
    value = cfg.get("report_to")
    if value is None:
        return ["tensorboard"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return ["tensorboard"]


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


def preprocess_datasets_parallel(
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer: Any,
    formatting_func: Any,
    cutoff_len: int,
    cache_dir: Optional[Path] = None,
    num_proc: Optional[int] = None,
    clear_cache: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Preprocess datasets with multiprocessing for faster tokenization.

    Args:
        train_ds: Training dataset
        eval_ds: Evaluation dataset
        tokenizer: Tokenizer instance
        formatting_func: Function to format examples into chat template
        cutoff_len: Maximum sequence length
        cache_dir: Directory to cache preprocessed datasets (None = no caching)
        num_proc: Number of processes to use (defaults to CPU count)
        clear_cache: If True, delete existing cache and regenerate datasets

    Returns:
        Preprocessed (train_ds, eval_ds) tuple
    """
    if num_proc is None:
        num_proc = mp.cpu_count()

    # Check if cached preprocessed datasets exist
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        train_cache = cache_dir / "train_preprocessed"
        eval_cache = cache_dir / "eval_preprocessed"

        # Clear cache if requested
        if clear_cache:
            import shutil
            if train_cache.exists():
                print(f"\nClearing train cache: {train_cache}")
                shutil.rmtree(train_cache)
            if eval_cache.exists():
                print(f"Clearing eval cache: {eval_cache}")
                shutil.rmtree(eval_cache)
            print("Cache cleared successfully!\n")

        if train_cache.exists() and eval_cache.exists():
            print(f"\nLoading preprocessed datasets from cache: {cache_dir}")
            print(f"  Train samples: checking cached dataset...")
            print(f"  Eval samples: checking cached dataset...")
            from datasets import load_from_disk
            train_ds = load_from_disk(str(train_cache))
            eval_ds = load_from_disk(str(eval_cache))
            print(f"  Train samples loaded: {len(train_ds)}")
            print(f"  Eval samples loaded: {len(eval_ds)}")
            print("Cached datasets loaded successfully!\n")
            return train_ds, eval_ds

    print(f"\nPreprocessing datasets with {num_proc} processes for faster tokenization...")

    # Apply formatting function to convert to chat template format
    print("Applying formatting function to train dataset...")
    train_ds = train_ds.map(
        lambda example: {"text": formatting_func(example)},
        num_proc=num_proc,
        desc="Formatting train dataset"
    )

    print("Applying formatting function to eval dataset...")
    eval_ds = eval_ds.map(
        lambda example: {"text": formatting_func(example)},
        num_proc=num_proc,
        desc="Formatting eval dataset"
    )

    # Add EOS token
    print("Adding EOS tokens to train dataset...")
    train_ds = train_ds.map(
        lambda example: {"text": example["text"] + tokenizer.eos_token},
        num_proc=num_proc,
        desc="Adding EOS to train"
    )

    print("Adding EOS tokens to eval dataset...")
    eval_ds = eval_ds.map(
        lambda example: {"text": example["text"] + tokenizer.eos_token},
        num_proc=num_proc,
        desc="Adding EOS to eval"
    )

    # Tokenize with multiprocessing and properly mask labels
    def tokenize_func(example):
        # Tokenize the full text (prompt + response)
        full_tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )

        # Find where the assistant's response starts
        # The chat template includes the prompt + assistant response
        # We need to mask (set to -100) all tokens before the assistant's actual response

        # Get the text before and after assistant marker
        text = example["text"]

        # For Qwen chat template, find where assistant content starts
        # The format is: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        assistant_marker = "<|im_start|>assistant\n"

        if assistant_marker in text:
            # Split at the assistant marker
            before_assistant = text.split(assistant_marker)[0] + assistant_marker

            # Tokenize just the prompt part to know how many tokens to mask
            prompt_tokens = tokenizer(
                before_assistant,
                truncation=False,
                padding=False,
            )
            prompt_len = len(prompt_tokens["input_ids"])

            # Create labels: -100 for prompt, actual token IDs for response
            labels = [-100] * prompt_len + full_tokens["input_ids"][prompt_len:]

            # Ensure labels and input_ids have same length
            if len(labels) > len(full_tokens["input_ids"]):
                labels = labels[:len(full_tokens["input_ids"])]
            elif len(labels) < len(full_tokens["input_ids"]):
                # Shouldn't happen, but pad with actual tokens if needed
                labels = labels + full_tokens["input_ids"][len(labels):]
        else:
            # Fallback: use all tokens (shouldn't happen with proper chat template)
            labels = full_tokens["input_ids"].copy()

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels
        }

    print("Tokenizing train dataset...")
    train_ds = train_ds.map(
        tokenize_func,
        num_proc=num_proc,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset"
    )

    print("Tokenizing eval dataset...")
    eval_ds = eval_ds.map(
        tokenize_func,
        num_proc=num_proc,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval dataset"
    )

    print("Dataset preprocessing complete!\n")

    # Save preprocessed datasets to cache for future runs
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_cache = cache_dir / "train_preprocessed"
        eval_cache = cache_dir / "eval_preprocessed"

        print(f"Saving preprocessed datasets to cache: {cache_dir}")
        train_ds.save_to_disk(str(train_cache))
        eval_ds.save_to_disk(str(eval_cache))
        print("Cache saved successfully!\n")

    return train_ds, eval_ds


class TrainingLossPlateauCallback(TrainerCallback):
    """Stop training when training loss plateaus (stops improving significantly).

    Args:
        patience: Number of logging steps to wait for improvement
        min_delta: Minimum change in loss to qualify as improvement (default 0.01)
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait_count = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return control

        # Only check training loss
        current_loss = logs.get("loss")
        if current_loss is None:
            return control

        # Initialize best_loss on first log
        if self.best_loss is None:
            self.best_loss = current_loss
            print(f"\n[TrainingLossPlateauCallback] Initial loss: {current_loss:.4f}")
            return control

        # Check if loss improved significantly
        if current_loss < (self.best_loss - self.min_delta):
            improvement = self.best_loss - current_loss
            self.best_loss = current_loss
            self.wait_count = 0
            print(f"\n[TrainingLossPlateauCallback] Loss improved by {improvement:.4f} to {current_loss:.4f}. Resetting patience.")
        else:
            self.wait_count += 1
            print(f"\n[TrainingLossPlateauCallback] No significant improvement ({self.wait_count}/{self.patience}). Best: {self.best_loss:.4f}, Current: {current_loss:.4f}")

            if self.wait_count >= self.patience:
                print(f"\n[TrainingLossPlateauCallback] Training loss plateaued. Stopping training.")
                control.should_training_stop = True

        return control


class MultiMetricEarlyStoppingCallback(TrainerCallback):
    """Stop training when ALL tracked metrics (training loss, accuracy, f1) stop improving.

    This callback monitors:
    1. Training loss (logged during training) - most important
    2. Evaluation accuracy - second priority
    3. Evaluation F1 - third priority

    Training stops only when ALL metrics haven't improved for the specified patience.

    Args:
        patience: Number of evaluation steps to wait without improvement in ALL metrics
        min_delta_loss: Minimum change in training loss to qualify as improvement (default 0.001)
        min_delta_metrics: Minimum change in eval metrics to qualify as improvement (default 0.001)
    """
    def __init__(self, patience: int = 5, min_delta_loss: float = 0.001, min_delta_metrics: float = 0.001):
        self.patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_metrics = min_delta_metrics

        # Track best values for each metric
        self.best_train_loss = None
        self.best_accuracy = None
        self.best_f1 = None

        # Track the last training loss logged (will check at eval time)
        self.current_train_loss = None

        # Track steps since last improvement for each metric
        self.train_loss_no_improvement_count = 0
        self.accuracy_no_improvement_count = 0
        self.f1_no_improvement_count = 0

        self.eval_count = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when training logs are generated - capture training loss."""
        if logs is None:
            return control

        # Capture the latest training loss
        train_loss = logs.get("loss")
        if train_loss is not None:
            self.current_train_loss = train_loss

        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after each evaluation - check all metrics."""
        if metrics is None:
            return control

        self.eval_count += 1

        # Get current eval metrics
        current_accuracy = metrics.get("eval_accuracy")
        current_f1 = metrics.get("eval_f1")

        # Check training loss (lower is better) - PRIORITY 1
        if self.current_train_loss is not None:
            if self.best_train_loss is None:
                self.best_train_loss = self.current_train_loss
                print(f"\n[MultiMetric] Initial training loss: {self.current_train_loss:.4f}")
            elif self.current_train_loss < (self.best_train_loss - self.min_delta_loss):
                improvement = self.best_train_loss - self.current_train_loss
                self.best_train_loss = self.current_train_loss
                self.train_loss_no_improvement_count = 0
                print(f"\n[MultiMetric] Training loss improved by {improvement:.4f} to {self.current_train_loss:.4f}")
            else:
                self.train_loss_no_improvement_count += 1

        # Check accuracy (higher is better) - PRIORITY 2
        if current_accuracy is not None:
            if self.best_accuracy is None:
                self.best_accuracy = current_accuracy
                print(f"\n[MultiMetric] Initial accuracy: {current_accuracy:.4f}")
            elif current_accuracy > (self.best_accuracy + self.min_delta_metrics):
                improvement = current_accuracy - self.best_accuracy
                self.best_accuracy = current_accuracy
                self.accuracy_no_improvement_count = 0
                print(f"\n[MultiMetric] Accuracy improved by {improvement:.4f} to {current_accuracy:.4f}")
            else:
                self.accuracy_no_improvement_count += 1

        # Check F1 (higher is better) - PRIORITY 3
        if current_f1 is not None:
            if self.best_f1 is None:
                self.best_f1 = current_f1
                print(f"\n[MultiMetric] Initial F1: {current_f1:.4f}")
            elif current_f1 > (self.best_f1 + self.min_delta_metrics):
                improvement = current_f1 - self.best_f1
                self.best_f1 = current_f1
                self.f1_no_improvement_count = 0
                print(f"\n[MultiMetric] F1 improved by {improvement:.4f} to {current_f1:.4f}")
            else:
                self.f1_no_improvement_count += 1

        # Print status summary
        print(f"\n[MultiMetric] === Evaluation {self.eval_count} Summary ===")
        print(f"[MultiMetric] No improvement counts (stop at {self.patience}):")
        print(f"  - Training Loss: {self.train_loss_no_improvement_count}/{self.patience}")
        print(f"  - Accuracy:      {self.accuracy_no_improvement_count}/{self.patience}")
        print(f"  - F1:            {self.f1_no_improvement_count}/{self.patience}")
        print(f"[MultiMetric] Current values:")
        print(f"  - Training Loss: {self.current_train_loss:.4f}")
        print(f"  - Accuracy:      {current_accuracy:.4f}")
        print(f"  - F1:            {current_f1:.4f}")
        print(f"[MultiMetric] Best values:")
        print(f"  - Training Loss: {self.best_train_loss:.4f}")
        print(f"  - Accuracy:      {self.best_accuracy:.4f}")
        print(f"  - F1:            {self.best_f1:.4f}")
        print(f"[MultiMetric] ================================\n")

        # Stop only if ALL metrics haven't improved for patience evaluations
        if (self.train_loss_no_improvement_count >= self.patience and
            self.accuracy_no_improvement_count >= self.patience and
            self.f1_no_improvement_count >= self.patience):
            print(f"\n[MultiMetric] ALL metrics plateaued for {self.patience} evaluations. Stopping training.")
            control.should_training_stop = True

        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen3_movielens_qlora.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear preprocessing cache and regenerate eval data (useful when changing max_eval_samples)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = get_device()

    # Print configuration at the start
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Clear cache: {args.clear_cache}")
    print()

    # Model configuration
    print("Model Configuration:")
    print(f"  model_name_or_path: {cfg.get('model_name_or_path', BASE_MODEL)}")
    print(f"  flash_attn: {cfg.get('flash_attn', 'auto')}")
    print(f"  bf16: {cfg.get('bf16', torch.cuda.is_available())}")
    print(f"  fp16: {cfg.get('fp16', False)}")
    print()

    # Dataset configuration
    print("Dataset Configuration:")
    print(f"  dataset: {cfg.get('dataset', 'movielens_qwen3_train')}")
    print(f"  dataset_dir: {cfg.get('dataset_dir', 'data')}")
    print(f"  max_eval_samples: {cfg.get('max_eval_samples', MAX_EVAL_SAMPLES)}")
    print(f"  cutoff_len: {cfg.get('cutoff_len', MAX_SEQ_LEN)}")
    print(f"  preprocessing_cache_dir: {cfg.get('preprocessing_cache_dir', '.cache/preprocessed')}")
    print(f"  num_proc: {cfg.get('num_proc', mp.cpu_count())}")
    print()

    # LoRA configuration
    print("LoRA Configuration:")
    print(f"  lora_rank: {cfg.get('lora_rank', LORA_RANK)}")
    print(f"  lora_alpha: {cfg.get('lora_alpha', LORA_ALPHA)}")
    print(f"  lora_dropout: {cfg.get('lora_dropout', LORA_DROPOUT)}")
    print(f"  lora_target: {cfg.get('lora_target', 'all')}")
    print()

    # Training configuration
    print("Training Configuration:")
    print(f"  output_dir: {cfg.get('output_dir', OUTPUT_DIR)}")
    print(f"  num_train_epochs: {cfg.get('num_train_epochs', NUM_TRAIN_EPOCHS)}")
    print(f"  per_device_train_batch_size: {cfg.get('per_device_train_batch_size', PER_DEVICE_TRAIN_BATCH_SIZE)}")
    print(f"  per_device_eval_batch_size: {cfg.get('per_device_eval_batch_size', PER_DEVICE_EVAL_BATCH_SIZE)}")
    print(f"  gradient_accumulation_steps: {cfg.get('gradient_accumulation_steps', GRADIENT_ACCUMULATION_STEPS)}")
    print(f"  learning_rate: {cfg.get('learning_rate', LEARNING_RATE)}")
    print(f"  lr_scheduler_type: {cfg.get('lr_scheduler_type', 'cosine')}")
    print(f"  warmup_ratio: {cfg.get('warmup_ratio', WARMUP_RATIO)}")
    print(f"  weight_decay: {cfg.get('weight_decay', 0.0)}")
    print(f"  max_grad_norm: {cfg.get('max_grad_norm', 1.0)}")
    print(f"  gradient_checkpointing: {cfg.get('gradient_checkpointing', False)}")
    print()

    # Evaluation and logging
    print("Evaluation & Logging Configuration:")
    print(f"  logging_steps: {cfg.get('logging_steps', LOGGING_STEPS)}")
    print(f"  eval_strategy: {cfg.get('eval_strategy', 'steps')}")
    print(f"  eval_steps: {cfg.get('eval_steps', SAVE_EVAL_STEPS)}")
    print(f"  save_steps: {cfg.get('save_steps', cfg.get('eval_steps', SAVE_EVAL_STEPS))}")
    print(f"  save_total_limit: {cfg.get('save_total_limit', 2)}")
    print(f"  metric_for_best_model: f1")
    print(f"  report_to: {resolve_report_to(cfg)}")
    print()

    # Early stopping configuration
    print("Early Stopping Configuration:")
    print(f"  multi_metric_patience: {cfg.get('multi_metric_patience', 5)}")
    print(f"  multi_metric_min_delta_loss: {cfg.get('multi_metric_min_delta_loss', 0.001)}")
    print(f"  multi_metric_min_delta_metrics: {cfg.get('multi_metric_min_delta_metrics', 0.001)}")
    print()

    # Optimizer configuration
    print("Optimizer Configuration:")
    print(f"  adam_beta1: {cfg.get('adam_beta1', 0.9)}")
    print(f"  adam_beta2: {cfg.get('adam_beta2', 0.999)}")
    print(f"  adam_epsilon: {cfg.get('adam_epsilon', 1.0e-8)}")
    print()

    print("=" * 80)
    print()

    # Enable 4-bit QLoRA on CUDA; otherwise fall back to full precision.
    quantization_config = None
    device_map = None
    torch_dtype = None
    attn_impl = resolve_attention_impl(cfg)
    attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}
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
        **attn_kwargs,
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
    train_path, eval_path = resolve_dataset_paths(cfg, args.config)
    train_ds, eval_ds = build_datasets(str(train_path), str(eval_path))
    eval_ds = maybe_shrink_eval(eval_ds, cfg.get("max_eval_samples", MAX_EVAL_SAMPLES))

    # Define formatting function for parallel preprocessing
    def formatting_func(example: Dict[str, Any]) -> str:
        return tokenizer.apply_chat_template(
            to_chat_messages(example),
            tokenize=False,
            add_generation_prompt=False,
        )

    # Preprocess datasets with multiprocessing for faster tokenization
    # Use cache to avoid re-tokenizing on subsequent runs
    num_proc = cfg.get("num_proc", mp.cpu_count())
    cache_dir = cfg.get("preprocessing_cache_dir", ".cache/preprocessed")
    train_ds, eval_ds = preprocess_datasets_parallel(
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        cutoff_len=cfg.get("cutoff_len", MAX_SEQ_LEN),
        cache_dir=Path(cache_dir),
        num_proc=num_proc,
        clear_cache=args.clear_cache,
    )

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
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", SAVE_EVAL_STEPS),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,

        bf16=cfg.get("bf16", torch.cuda.is_available()),
        fp16=cfg.get("fp16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        report_to=resolve_report_to(cfg),
    )

    def _extract_answer(text: str) -> Optional[str]:
        """Extract Yes/No answer from model response, matching inference logic."""
        import re

        # Extract only the assistant's actual response after the prompt
        if "assistant" in text:
            text = text.split("assistant")[-1]

        # Remove <think> tags and their content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Clean up whitespace
        text = text.strip().lower()

        # Look for yes/no in the cleaned response
        if text.startswith("yes"):
            return "yes"
        elif text.startswith("no"):
            return "no"
        elif "yes" in text and "no" not in text:
            return "yes"
        elif "no" in text and "yes" not in text:
            return "no"
        else:
            return None

    def _normalize_label(text: str) -> Optional[int]:
        answer = _extract_answer(text)
        if answer == "yes":
            return 1
        elif answer == "no":
            return 0
        return None

    def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Convert logits to predictions immediately to save GPU memory.

        Instead of storing full logits (batch × seq × vocab_size) in GPU memory,
        we convert to token IDs (batch × seq) right away, reducing memory by ~100x.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        # Convert to token IDs by taking argmax
        return torch.argmax(logits, dim=-1)

    def compute_metrics(eval_preds: Tuple[Any, Any]) -> Dict[str, float]:
        """Compute accuracy and F1 from teacher-forced predictions.

        During evaluation, we only get token-by-token predictions (teacher forcing),
        not full generated responses. We extract only the assistant's response tokens
        (where labels != -100) and check if they start with yes/no.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Predictions are already token IDs from preprocess_logits_for_metrics
        # No need to take argmax - preds is already (batch, sequence)

        y_pred_list = []
        y_true_list = []
        debug_samples = []

        # Process each example in the batch
        for idx, (pred_seq, label_seq) in enumerate(zip(preds, labels)):
            # Labels are -100 for prompt tokens (ignored in loss)
            # and actual token IDs for assistant response tokens
            valid_positions = label_seq != -100

            if not valid_positions.any():
                continue

            # Extract only the assistant's response tokens
            pred_tokens = pred_seq[valid_positions]
            true_tokens = label_seq[valid_positions]

            # Decode just the assistant's response
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            true_text = tokenizer.decode(true_tokens, skip_special_tokens=True)

            # Extract yes/no labels
            pred_label = _normalize_label(pred_text)
            true_label = _normalize_label(true_text)

            # Debug: collect first 3 samples to see what's happening
            if idx < 3:
                debug_samples.append({
                    'pred_text': pred_text[:100],
                    'true_text': true_text[:100],
                    'pred_label': pred_label,
                    'true_label': true_label
                })

            if pred_label is not None and true_label is not None:
                y_pred_list.append(pred_label)
                y_true_list.append(true_label)

        # Print debug info on first evaluation
        if debug_samples and not hasattr(compute_metrics, '_debug_printed'):
            print("\n=== DEBUG: First 3 eval samples ===")
            for i, sample in enumerate(debug_samples):
                print(f"\nSample {i}:")
                print(f"  Pred text: {sample['pred_text']}")
                print(f"  True text: {sample['true_text']}")
                print(f"  Pred label: {sample['pred_label']}")
                print(f"  True label: {sample['true_label']}")
            print(f"\nTotal valid pairs: {len(y_pred_list)} / {len(preds)}")
            print("=" * 50 + "\n")
            compute_metrics._debug_printed = True

        if not y_pred_list:
            return {"accuracy": 0.0, "f1": 0.0}

        y_pred_arr = np.array(y_pred_list)
        y_true_arr = np.array(y_true_list)

        accuracy = float((y_pred_arr == y_true_arr).mean())

        tp = float(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
        fp = float(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
        fn = float(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = float(0.0 if denom == 0 else (2 * tp) / denom)

        return {"accuracy": accuracy, "f1": f1}

    # TRL SFTTrainer handles PEFT integration
    # Note: Datasets are pre-tokenized via multiprocessing, so no formatting_func needed
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            MultiMetricEarlyStoppingCallback(
                patience=cfg.get("multi_metric_patience", 5),
                min_delta_loss=cfg.get("multi_metric_min_delta_loss", 0.001),
                min_delta_metrics=cfg.get("multi_metric_min_delta_metrics", 0.001)
            )
        ],
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    out_dir = Path(cfg.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Training complete. Adapter + tokenizer saved to: {out_dir}")


if __name__ == "__main__":
    main()
