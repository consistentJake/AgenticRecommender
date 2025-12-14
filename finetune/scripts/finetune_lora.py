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

# Set PyTorch CUDA memory allocator config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
import yaml
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

# Import shared utilities
from utils import (
    BASE_MODEL as UTILS_BASE_MODEL,
    load_model_and_tokenizer,
    get_device,
    load_yaml_config,
    load_json,
    build_datasets,
    to_chat_messages,
    to_generation_messages,
    tokenize_func,
    preprocess_datasets_parallel,
)


BASE_MODEL = UTILS_BASE_MODEL  # Import from utils to ensure consistency
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
QV_LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Query and value only
FN_LORA_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]  # mirrors `lora_target: all` from the config


# get_device, load_json, build_datasets imported from utils


def maybe_shrink_eval(eval_ds: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None:
        return eval_ds
    return eval_ds.select(range(min(len(eval_ds), max_samples)))


# load_yaml_config imported from utils


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
    if isinstance(cfg_value, str) and cfg_value.lower() == "qv":
        return QV_LORA_TARGET_MODULES
    if isinstance(cfg_value, str) and cfg_value.lower() == "fn":
        return FN_LORA_TARGET_MODULES
    if isinstance(cfg_value, str):
        return [part.strip() for part in cfg_value.split(",") if part.strip()]
    if isinstance(cfg_value, list):
        return [str(x) for x in cfg_value]
    return LORA_TARGET_MODULES


def resolve_report_to(cfg: Dict[str, Any]) -> List[str]:
    value = cfg.get("report_to")
    if value is None:
        return ["tensorboard"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return ["tensorboard"]


# to_chat_messages, tokenize_func, preprocess_datasets_parallel imported from utils


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
        print(f"  - Training Loss: {self.current_train_loss:.4f}" if self.current_train_loss is not None else "  - Training Loss: N/A")
        print(f"  - Accuracy:      {current_accuracy:.4f}" if current_accuracy is not None else "  - Accuracy:      N/A")
        print(f"  - F1:            {current_f1:.4f}" if current_f1 is not None else "  - F1:            N/A")
        print(f"[MultiMetric] Best values:")
        print(f"  - Training Loss: {self.best_train_loss:.4f}" if self.best_train_loss is not None else "  - Training Loss: N/A")
        print(f"  - Accuracy:      {self.best_accuracy:.4f}" if self.best_accuracy is not None else "  - Accuracy:      N/A")
        print(f"  - F1:            {self.best_f1:.4f}" if self.best_f1 is not None else "  - F1:            N/A")
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
    print(f"  use_qlora: {cfg.get('use_qlora', torch.cuda.is_available())}")
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

    model, tokenizer = load_model_and_tokenizer(cfg, device, disable_cache=True)

    # Build datasets
    train_path, eval_path = resolve_dataset_paths(cfg, args.config)
    train_ds, eval_ds = build_datasets(str(train_path), str(eval_path))
    eval_ds = maybe_shrink_eval(eval_ds, cfg.get("max_eval_samples", MAX_EVAL_SAMPLES))
    raw_eval_ds_for_gen = eval_ds

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
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing", False) else None,
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

    def run_generative_eval(model, tokenizer, raw_eval_ds, num_samples: int = 32):
        """Lightweight generative eval to mirror real inference behavior."""
        if num_samples <= 0 or len(raw_eval_ds) == 0:
            return

        sample_ds = raw_eval_ds.select(range(min(len(raw_eval_ds), num_samples)))
        preds: List[int] = []
        labels: List[int] = []

        print(f"\nRunning generative eval on {len(sample_ds)} samples (greedy, max_new_tokens=8)...")
        for example in sample_ds:
            # Use to_generation_messages to build prompt WITHOUT assistant response
            messages = to_generation_messages(example)
            chat_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                chat_str,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(out[0], skip_special_tokens=True)
            pred_label = _normalize_label(response)
            true_label = _normalize_label(example.get("output", ""))

            if pred_label is not None and true_label is not None:
                preds.append(pred_label)
                labels.append(true_label)

        if not preds:
            print("No valid pairs for generative eval.")
            return

        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        gen_accuracy = float((preds_arr == labels_arr).mean())

        tp = float(((preds_arr == 1) & (labels_arr == 1)).sum())
        fp = float(((preds_arr == 1) & (labels_arr == 0)).sum())
        fn = float(((preds_arr == 0) & (labels_arr == 1)).sum())
        denom = (2 * tp + fp + fn)
        gen_f1 = float(0.0 if denom == 0 else (2 * tp) / denom)

        print(f"Generative eval — accuracy: {gen_accuracy:.4f}, f1: {gen_f1:.4f}, samples: {len(preds)}\n")

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

    # Optional small generative eval to mirror inference behavior.
    run_generative_eval(
        trainer.model,
        tokenizer,
        raw_eval_ds_for_gen,
        num_samples=cfg.get("gen_eval_samples", 0),
    )

    # Save LoRA adapter + tokenizer
    out_dir = Path(cfg.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Training complete. Adapter + tokenizer saved to: {out_dir}")


if __name__ == "__main__":
    main()
