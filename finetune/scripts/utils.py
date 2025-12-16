#!/usr/bin/env python
"""Shared utility functions for MovieLens fine-tuning and inference scripts."""

import json
import multiprocessing as mp
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ==============================================================================
# Constants
# ==============================================================================

# Model Configuration
BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "output/qwen3-movielens-qlora"

# Generation Parameters
MAX_NEW_TOKENS = 256  # Increased to allow model to complete reasoning and answer
TEMPERATURE = 0.7
TOP_P = 0.9

# System Prompts
DEFAULT_SYSTEM_PROMPT = (
    "You are a movie recommendation assistant. Given a user's recent history "
    "and a candidate movie. Please begin your analysis with 'Yes' or 'No'."
)


# ==============================================================================
# Device Detection
# ==============================================================================

def get_device() -> str:
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> Any:
    """Load JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: str) -> List[Dict]:
    """Load data from JSON or JSONL file with automatic fallback."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        # Try to load as JSON first
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL format
        f.seek(0)
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ==============================================================================
# Data Formatting
# ==============================================================================

def format_prompt(example: Dict) -> str:
    """Format example into prompt, supporting both training and test formats."""
    # Training data format (JSON with instruction/input)
    if "input" in example:
        return example["input"]

    # Test data format (JSONL with history_titles, candidate_title)
    if "history_titles" in example and "candidate_title" in example:
        history_str = "\n".join(example["history_titles"])
        prompt = (
            f"User's last 15 watched movies:\n"
            f"{history_str}\n\n"
            f"Candidate movie:\n"
            f"{example['candidate_title']}\n\n"
            f"Should we recommend this movie to the user? Answer Yes or No."
        )
        return prompt

    raise ValueError(f"Unknown data format. Example keys: {example.keys()}")


def to_chat_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert supervised sample to chat-style messages expected by Qwen.

    This includes the assistant's response and is used for training.
    For generation/inference, use to_generation_messages() instead.
    """
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


def to_generation_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert example to chat messages for generation (no assistant response).

    This creates the prompt for the model to generate a response.
    Use this for inference/evaluation, not training.

    Args:
        example: Dict with 'system', 'instruction', 'input', and optionally 'history'

    Returns:
        List of message dicts ready for chat template with add_generation_prompt=True
    """
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

    return messages


# ==============================================================================
# Dataset Building
# ==============================================================================

def build_datasets(train_path: str, eval_path: str) -> Tuple[Dataset, Dataset]:
    """Build Dataset objects from JSON or JSONL files."""
    train_data = load_data(train_path)
    eval_data = load_data(eval_path)

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)
    return train_ds, eval_ds


# ==============================================================================
# Tokenization
# ==============================================================================

def tokenize_func(
    example: Dict[str, Any],
    tokenizer: Any,
    cutoff_len: int,
    assistant_marker: str = "<|im_start|>assistant\n"
) -> Dict[str, List[int]]:
    """Tokenize example with proper label masking for supervised fine-tuning.

    This function implements the critical logic for supervised fine-tuning:
    - Tokenizes the full conversation (prompt + response)
    - Masks prompt tokens with -100 so they're ignored in loss calculation
    - Preserves assistant response tokens for training

    Args:
        example: Dict with 'text' key containing formatted chat template
        tokenizer: Transformers tokenizer instance
        cutoff_len: Maximum sequence length (truncation point)
        assistant_marker: String marking start of assistant's response.
                         Default is Qwen's format: "<|im_start|>assistant\n"

    Returns:
        Dict with:
            - input_ids: Token IDs for full sequence
            - attention_mask: Attention mask (1s for real tokens)
            - labels: Token IDs with prompt masked as -100

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>> example = {"text": "<|im_start|>user\nHello<|im_end|>..."}
        >>> result = tokenize_func(example, tokenizer, cutoff_len=1024)
        >>> result.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """
    # Tokenize the full text (prompt + response) without truncation so we can
    # apply consistent left-side trimming to both inputs and labels.
    full_tokens = tokenizer(
        example["text"],
        truncation=False,
        padding=False,
    )
    full_ids = full_tokens["input_ids"]
    full_attn = full_tokens["attention_mask"]

    # Find where the assistant's response starts
    # The chat template includes the prompt + assistant response
    # We need to mask (set to -100) all tokens before the assistant's actual response

    # Get the text before and after assistant marker
    text = example["text"]

    # For Qwen chat template, find where assistant content starts
    # The format is: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n

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
        labels = [-100] * prompt_len + full_ids[prompt_len:]
    else:
        # Fallback: use all tokens (shouldn't happen with proper chat template)
        labels = full_ids.copy()

    # Apply left-side truncation manually so inputs and labels stay aligned.
    if len(full_ids) > cutoff_len:
        start = len(full_ids) - cutoff_len
        full_ids = full_ids[start:]
        full_attn = full_attn[start:]
        labels = labels[start:]

    return {
        "input_ids": full_ids,
        "attention_mask": full_attn,
        "labels": labels
    }


# ==============================================================================
# Model / Tokenizer Loading (shared between training & integration tests)
# ==============================================================================

def resolve_attention_impl(cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve attention implementation string."""
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


def load_model_and_tokenizer(
    cfg: Dict[str, Any],
    device: str,
    disable_cache: bool = True,
) -> Tuple[Any, Any]:
    """Load model/tokenizer with the same settings used in training."""
    model_name = cfg.get("model_name_or_path", BASE_MODEL)
    attn_impl = resolve_attention_impl(cfg)
    if attn_impl and not torch.cuda.is_available():
        # Disable flash attention on CPU-only environments.
        attn_impl = None
    attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

    quantization_config = None
    device_map = None
    torch_dtype = None
    if cfg.get("use_qlora", torch.cuda.is_available()) and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **attn_kwargs,
    )
    if device_map is None:
        model = model.to(device)
    if disable_cache:
        model.config.use_cache = False

    # Load tokenizer with optional special tokenization handling
    enable_special_tokenization = cfg.get("enable_special_tokenization", False)

    if enable_special_tokenization:
        if "Qwen" in model_name:
            # Special tokenization for Qwen models
            # Use custom unk_token for special object reference handling
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            tokenizer.unk_token = "<|object_ref_end|>"
            tokenizer.unk_token_id = 151647
        else:
            # Standard tokenization
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False if not enable_special_tokenization or "Qwen" not in model_name else None,
            )
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.unk_token is None:
                tokenizer.add_special_tokens({"unk_token": "<unk>"})

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    return model, tokenizer


def preprocess_datasets_parallel(
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer: Any,
    formatting_func: Any,
    cutoff_len: int,
    cache_dir: Optional[Path] = None,
    num_proc: Optional[int] = None,
    clear_cache: bool = False,
    prepare_for_packing: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Preprocess datasets with multiprocessing for faster tokenization.

    This function supports two modes:

    1. **Pre-tokenized mode** (prepare_for_packing=False, default):
       - Fully tokenizes datasets with proper label masking
       - Caches tokenized data for fast subsequent runs
       - Sequences are variable length (NOT padded to cutoff_len)
       - Use this when packing is disabled in SFTTrainer
       - Best for: Faster training startup with cached data

    2. **Packing mode** (prepare_for_packing=True):
       - Only formats to text (no tokenization)
       - Returns datasets with 'text' field for SFTTrainer to handle
       - Enables SFTTrainer's packing feature to concatenate multiple examples
       - Use with SFTTrainer(packing=True, formatting_func=...)
       - Best for: Maximum GPU efficiency, less padding waste

    Args:
        train_ds: Training dataset
        eval_ds: Evaluation dataset
        tokenizer: Tokenizer instance
        formatting_func: Function to format examples into chat template
        cutoff_len: Maximum sequence length
        cache_dir: Directory to cache preprocessed datasets (None = no caching)
        num_proc: Number of processes to use (defaults to CPU count)
        clear_cache: If True, delete existing cache and regenerate datasets
        prepare_for_packing: If True, only format (no tokenization) for SFTTrainer packing

    Returns:
        Preprocessed (train_ds, eval_ds) tuple

    Example:
        >>> # Pre-tokenized mode (current behavior, with caching)
        >>> train_ds, eval_ds = preprocess_datasets_parallel(
        ...     train_ds, eval_ds, tokenizer, formatting_func,
        ...     cutoff_len=1024, prepare_for_packing=False
        ... )
        >>> trainer = SFTTrainer(train_dataset=train_ds, ...)  # No packing

        >>> # Packing mode (for better GPU utilization)
        >>> train_ds, eval_ds = preprocess_datasets_parallel(
        ...     train_ds, eval_ds, tokenizer, formatting_func,
        ...     cutoff_len=1024, prepare_for_packing=True
        ... )
        >>> trainer = SFTTrainer(
        ...     train_dataset=train_ds,
        ...     packing=True,
        ...     max_seq_length=1024,
        ...     ...
        ... )
    """
    from functools import partial

    if num_proc is None:
        num_proc = mp.cpu_count()

    # Preserve assistant labels by preferring to drop earliest prompt tokens when
    # we exceed cutoff_len.
    tokenizer.truncation_side = "left"

    # Use different cache directories for packing vs non-packing modes
    # This prevents cache conflicts when switching between modes
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if prepare_for_packing:
            train_cache = cache_dir / "train_formatted_for_packing"
            eval_cache = cache_dir / "eval_formatted_for_packing"
        else:
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
            mode_str = "formatted (for packing)" if prepare_for_packing else "preprocessed (tokenized)"
            print(f"\nLoading {mode_str} datasets from cache: {cache_dir}")
            print(f"  Train samples: checking cached dataset...")
            print(f"  Eval samples: checking cached dataset...")
            from datasets import load_from_disk
            train_ds = load_from_disk(str(train_cache))
            eval_ds = load_from_disk(str(eval_cache))
            print(f"  Train samples loaded: {len(train_ds)}")
            print(f"  Eval samples loaded: {len(eval_ds)}")
            print(f"Cached datasets loaded successfully!\n")
            return train_ds, eval_ds

    mode_str = "formatting (for packing)" if prepare_for_packing else "preprocessing with tokenization"
    print(f"\n{mode_str.capitalize()} datasets with {num_proc} processes...")

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

    # If preparing for packing mode, stop here (no tokenization)
    # SFTTrainer will handle tokenization and packing
    if prepare_for_packing:
        print("Formatting complete! (Tokenization will be handled by SFTTrainer with packing)\n")

        # Save formatted datasets to cache for future runs
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            train_cache = cache_dir / "train_formatted_for_packing"
            eval_cache = cache_dir / "eval_formatted_for_packing"

            print(f"Saving formatted datasets to cache: {cache_dir}")
            train_ds.save_to_disk(str(train_cache))
            eval_ds.save_to_disk(str(eval_cache))
            print("Cache saved successfully!\n")

        return train_ds, eval_ds

    # Pre-tokenized mode: Tokenize with multiprocessing and properly mask labels
    # Use partial to bind tokenizer and cutoff_len to tokenize_func
    tokenize_fn = partial(
        tokenize_func,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len
    )

    print("Tokenizing train dataset...")
    train_ds = train_ds.map(
        tokenize_fn,
        num_proc=num_proc,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset"
    )

    print("Tokenizing eval dataset...")
    eval_ds = eval_ds.map(
        tokenize_fn,
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


# ==============================================================================
# Generation
# ==============================================================================

def generate(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    do_sample: bool = False
) -> str:
    """Generate model response for a given prompt.

    Args:
        prompt: User prompt text
        model: Loaded model instance
        tokenizer: Tokenizer instance
        device: Device string ("cuda", "mps", or "cpu")
        system_prompt: System message (defaults to DEFAULT_SYSTEM_PROMPT)
    max_new_tokens: Maximum tokens to generate (defaults to MAX_NEW_TOKENS=64)
    temperature: Sampling temperature (defaults to TEMPERATURE=0.7)
    top_p: Nucleus sampling parameter (defaults to TOP_P=0.9)
    do_sample: Whether to use sampling (vs greedy). Defaults to deterministic decoding.

    Returns:
        Full decoded model response including prompt
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    if temperature is None:
        temperature = TEMPERATURE
    if top_p is None:
        top_p = TOP_P

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_batch(
    prompts: List[str],
    model: Any,
    tokenizer: Any,
    device: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    do_sample: bool = False
) -> List[str]:
    """Generate model responses for a batch of prompts (more efficient).

    Args:
        prompts: List of user prompt texts
        model: Loaded model instance
        tokenizer: Tokenizer instance
        device: Device string ("cuda", "mps", or "cpu")
        system_prompt: System message (defaults to DEFAULT_SYSTEM_PROMPT)
        max_new_tokens: Maximum tokens to generate (defaults to MAX_NEW_TOKENS)
        temperature: Sampling temperature (defaults to TEMPERATURE)
        top_p: Nucleus sampling parameter (defaults to TOP_P)
        do_sample: Whether to use sampling (vs greedy). Defaults to deterministic decoding.

    Returns:
        List of full decoded model responses including prompts
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    if temperature is None:
        temperature = TEMPERATURE
    if top_p is None:
        top_p = TOP_P

    # Build messages for each prompt
    batch_messages = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        batch_messages.append(messages)

    # Apply chat template to all prompts
    chat_strs = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in batch_messages
    ]

    # Tokenize all prompts with padding
    inputs = tokenizer(
        chat_strs,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    # Generate responses for the batch
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode all outputs
    responses = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]

    return responses


# ==============================================================================
# Evaluation
# ==============================================================================

def extract_answer(response: str) -> str:
    """Extract Yes/No answer from model response.

    Args:
        response: Model's text response

    Returns:
        "Yes", "No", or "Unknown"
    """
    # Extract only the assistant's actual response after the prompt
    # Split by "assistant" to get the generated part
    if "assistant" in response:
        response = response.split("assistant")[-1]

    # Remove <think> tags and their content
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Clean up whitespace
    response = response.strip()
    response_lower = response.lower()

    # Look for yes/no in the cleaned response
    if response_lower.startswith("yes"):
        return "Yes"
    elif response_lower.startswith("no"):
        return "No"
    elif "yes" in response_lower and "no" not in response_lower:
        return "Yes"
    elif "no" in response_lower and "yes" not in response_lower:
        return "No"
    else:
        return "Unknown"


def compute_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score.

    Args:
        predictions: List of predicted labels ("Yes", "No", "Unknown")
        labels: List of ground truth labels ("Yes", "No")

    Returns:
        Dict with metrics: accuracy, precision, recall, f1, total, correct
    """
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0.0

    # Compute F1 for "Yes" class
    tp = sum(1 for p, l in zip(predictions, labels) if p == "Yes" and l == "Yes")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "Yes" and l == "No")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "No" and l == "Yes")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "No" and l == "No")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(labels),
        "correct": correct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ==============================================================================
# Training Log Archival
# ==============================================================================

def archive_training_log(log_path: str, timestamp_format: str = "%y%m%d%H%M") -> Optional[str]:
    """Archive training log file with timestamp.

    Creates a copy of the training log file with a timestamp appended to the filename.
    The timestamp format is YYMMDDHHmm by default (e.g., 2412141530 for Dec 14, 2024 15:30).

    Args:
        log_path: Path to the training log file (e.g., "training.log" or "training-resume.log")
        timestamp_format: strftime format string for timestamp (default: "%y%m%d%H%M")

    Returns:
        Path to the archived log file if successful, None if the log file doesn't exist

    Example:
        >>> # At end of training
        >>> archived = archive_training_log("training.log")
        >>> # Creates: training.log.2412141530
        >>>
        >>> # Or with custom format
        >>> archived = archive_training_log("training.log", "%Y%m%d_%H%M%S")
        >>> # Creates: training.log.20241214_153045
    """
    log_path = Path(log_path)

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return None

    # Generate timestamp
    timestamp = datetime.now().strftime(timestamp_format)

    # Create archived filename: original_name.timestamp
    archived_path = Path(f"{log_path}.{timestamp}")

    # Copy the log file
    try:
        shutil.copy2(log_path, archived_path)
        print(f"Training log archived: {archived_path}")
        return str(archived_path)
    except Exception as e:
        print(f"Error archiving log file: {e}")
        return None
