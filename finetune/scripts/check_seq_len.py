#!/usr/bin/env python
"""Unified sequence-length analysis tool for finetuning datasets.

Features:
- Supports both JSON and JSONL formats (auto-detected)
- Analyzes single split or all splits (train/eval/test)
- Uses shared to_chat_messages from utils.py for consistency
- Reports length percentiles and truncation analysis
- Shows example transformations for debugging

Examples:
    # Analyze training data only (default)
    python check_seq_len.py --config configs/qwen3_7b_delivery_hero_qlora.yaml

    # Analyze all splits (train/eval/test)
    python check_seq_len.py --config configs/qwen3_7b_delivery_hero_qlora.yaml --all-splits

    # Show detailed examples
    python check_seq_len.py --config configs/qwen3_7b_delivery_hero_qlora.yaml --show-examples 3

    # Character-only mode (no tokenizer)
    python check_seq_len.py --config configs/qwen3_7b_delivery_hero_qlora.yaml --char-only
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

# Import shared utility functions
from utils import to_chat_messages


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSON or JSONL file automatically."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Try JSONL first (one JSON object per line)
    try:
        data = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # If first line fails, it's not JSONL - try regular JSON
                        if line_num == 1:
                            break
                        raise
            if data:  # Successfully loaded as JSONL
                return data
    except json.JSONDecodeError:
        pass

    # Fall back to regular JSON
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]


def _strip_split_suffix(name: str) -> str:
    for suffix in ("_train", "_eval", "_test"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_dataset_dir(raw_dir: Path, config_path: Path) -> Path:
    if raw_dir.is_absolute() or raw_dir.exists():
        return raw_dir
    cfg_root = config_path.parent.parent  # e.g., finetune/
    candidate = cfg_root / raw_dir
    if candidate.exists():
        return candidate
    return raw_dir


def resolve_dataset_paths(
    cfg: Dict[str, Any],
    config_path: Path,
    split: str = "train"
) -> Path:
    """Resolve path for a specific dataset split.

    Args:
        cfg: Config dictionary
        config_path: Path to config file
        split: Dataset split name ("train", "eval", or "test")

    Returns:
        Path to dataset file
    """
    dataset_dir = resolve_dataset_dir(Path(cfg.get("dataset_dir", "data")), config_path)

    # Check for explicit file paths in config
    if split == "train" and "train_file" in cfg:
        train_file = Path(cfg["train_file"])
        if not train_file.is_absolute():
            train_file = dataset_dir / train_file
        return train_file
    elif split == "eval" and "eval_file" in cfg:
        eval_file = Path(cfg["eval_file"])
        if not eval_file.is_absolute():
            eval_file = dataset_dir / eval_file
        return eval_file

    # Check dataset_info.json
    dataset_key = cfg.get("dataset" if split == "train" else f"{split}_dataset", f"dataset_{split}")
    base_key = _strip_split_suffix(dataset_key)

    info_path = dataset_dir / "dataset_info.json"
    if info_path.exists():
        try:
            info = load_json_or_jsonl(info_path)
            if isinstance(info, list):
                info = info[0]
            if base_key in info:
                if split == "train":
                    file_rel = info[base_key].get("file_name")
                elif split == "eval":
                    file_rel = info[base_key].get("file_name_eval")
                else:
                    file_rel = info[base_key].get(f"file_name_{split}")

                if file_rel:
                    return dataset_dir / file_rel
        except Exception:
            pass

    # Try common patterns for split files
    # Pattern 1: split_dir/split.json (e.g., train/train.json)
    split_file_in_dir = dataset_dir / split / f"{split}.json"
    if split_file_in_dir.exists():
        return split_file_in_dir

    # Pattern 2: split.jsonl in root (e.g., train.jsonl)
    split_file_jsonl = dataset_dir / f"{split}.jsonl"
    if split_file_jsonl.exists():
        return split_file_jsonl

    # Pattern 3: split.json in root (e.g., train.json)
    split_file_json = dataset_dir / f"{split}.json"
    if split_file_json.exists():
        return split_file_json

    # Pattern 4: base_key/split.json (e.g., movielens/train.json)
    if base_key:
        base_split_file = dataset_dir / base_key / f"{split}.json"
        if base_split_file.exists():
            return base_split_file

    # If nothing found, return the default path (will error later if it doesn't exist)
    return dataset_dir / split / f"{split}.json"


def describe_lengths(lengths: List[int], percentiles: List[float]) -> Dict[str, float]:
    arr = np.array(lengths)
    stats: Dict[str, float] = {
        "count": len(arr),
        "min": float(arr.min()) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else 0.0,
    }
    for p in percentiles:
        stats[f"p{int(p)}"] = float(np.percentile(arr, p)) if len(arr) else 0.0
    return stats


def show_examples(
    data: List[Dict[str, Any]],
    tokenizer: Any,
    num_examples: int = 2,
    cutoff_len: int = 1024,
):
    """Display example transformations from JSON to tokenized input."""
    print("\n" + "=" * 80)
    print("EXAMPLE TRANSFORMATIONS: JSON → Chat Messages → Formatted String → Tokens")
    print("=" * 80)

    for idx, example in enumerate(data[:num_examples]):
        print(f"\n{'─' * 80}")
        print(f"EXAMPLE {idx + 1}")
        print(f"{'─' * 80}")

        # Step 1: Original JSON
        print("\n[STEP 1] Original JSON Record:")
        print(f"  instruction: {example.get('instruction', '')[:80]}...")
        print(f"  input: {example.get('input', '')[:100]}...")
        print(f"  output: {example.get('output', '')}")
        print(f"  system: {example.get('system', '')[:80]}...")

        # Step 2: Chat messages
        messages = to_chat_messages(example)
        print("\n[STEP 2] Converted to Chat Messages:")
        for i, msg in enumerate(messages):
            content_preview = msg["content"][:100].replace("\n", "\\n")
            print(f"  {i+1}. role={msg['role']}: {content_preview}...")

        # Step 3: Chat template
        chat_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print("\n[STEP 3] Applied Chat Template (Qwen format with special tokens):")
        # Show first 500 chars with line breaks preserved
        preview = chat_str[:500]
        for line in preview.split('\n'):
            print(f"  {line}")
        if len(chat_str) > 500:
            print(f"  ... (truncated, total {len(chat_str)} chars)")

        # Step 4: Tokenization
        tokens = tokenizer(
            chat_str,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        token_ids = tokens["input_ids"]

        print(f"\n[STEP 4] Tokenized:")
        print(f"  Total tokens: {len(token_ids)}")
        print(f"  First 20 token IDs: {token_ids[:20]}")
        print(f"  Last 20 token IDs: {token_ids[-20:]}")

        # Step 5: Truncation analysis
        print(f"\n[STEP 5] Truncation Analysis (cutoff_len={cutoff_len}):")
        if len(token_ids) > cutoff_len:
            print(f"  ⚠️  WILL BE TRUNCATED! ({len(token_ids)} > {cutoff_len})")
            print(f"  Tokens removed: {len(token_ids) - cutoff_len}")

            # Show what gets removed with left truncation
            removed_tokens = token_ids[:len(token_ids) - cutoff_len]
            kept_tokens = token_ids[len(token_ids) - cutoff_len:]

            removed_text = tokenizer.decode(removed_tokens)
            print(f"\n  TEXT THAT WILL BE REMOVED (first 200 chars):")
            print(f"  {removed_text[:200]}...")

            print(f"\n  TEXT THAT WILL BE KEPT (first 200 chars):")
            kept_text = tokenizer.decode(kept_tokens)
            print(f"  {kept_text[:200]}...")
        else:
            print(f"  ✓ No truncation needed ({len(token_ids)} <= {cutoff_len})")

        # Step 6: Label masking
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in chat_str:
            before_assistant = chat_str.split(assistant_marker)[0] + assistant_marker
            prompt_tokens = tokenizer(before_assistant, truncation=False, padding=False)
            prompt_len = len(prompt_tokens["input_ids"])
            response_len = len(token_ids) - prompt_len

            print(f"\n[STEP 6] Label Masking for Training:")
            print(f"  Prompt tokens (masked with -100): {prompt_len}")
            print(f"  Response tokens (trained): {response_len}")
            print(f"  Masking ratio: {prompt_len / len(token_ids) * 100:.1f}% masked")


def analyze_split(
    split_name: str,
    split_path: Path,
    tokenizer: Any,
    cutoff_len: int,
    max_samples: Optional[int],
    percentiles: List[float],
    char_only: bool = False,
) -> Dict[str, Any]:
    """Analyze sequence lengths for a single dataset split."""

    print(f"\nLoading {split_name} data from: {split_path}")
    data = load_json_or_jsonl(split_path)
    print(f"Loaded {len(data)} samples")

    # Compute statistics
    lengths: List[int] = []
    limit = max_samples or len(data)
    for example in data[:limit]:
        messages = to_chat_messages(example)
        if not char_only and tokenizer is not None:
            chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer(
                chat,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            lengths.append(len(tokens["input_ids"]))
        else:
            chat = "\n".join([m["content"] for m in messages])
            lengths.append(len(chat))

    stats = describe_lengths(lengths, percentiles)
    mode = "tokens" if (not char_only and tokenizer is not None) else "characters"

    print("\n" + "=" * 80)
    print(f"SEQUENCE LENGTH STATISTICS - {split_name.upper()}")
    print("=" * 80)
    print(f"\nDataset: {split_path}")
    print(f"Measured: {int(stats['count'])} samples ({mode})")
    print(f"Cutoff length: {cutoff_len}")
    print(f"\nStatistics:")
    for k, v in stats.items():
        if k == "count":
            continue
        print(f"  {k}: {v:.2f}")

    # Truncation analysis
    num_truncated = sum(1 for l in lengths if l > cutoff_len)
    pct_truncated = num_truncated / len(lengths) * 100 if lengths else 0

    print(f"\nTruncation Analysis:")
    print(f"  Samples > {cutoff_len} {mode}: {num_truncated} ({pct_truncated:.1f}%)")
    print(f"  Samples <= {cutoff_len} {mode}: {len(lengths) - num_truncated} ({100 - pct_truncated:.1f}%)")

    if pct_truncated > 10:
        print(f"\n⚠️  WARNING: {pct_truncated:.1f}% of samples will be truncated!")
        print(f"  Consider increasing cutoff_len or reviewing truncation strategy.")
    elif pct_truncated > 0:
        print(f"\n  ℹ️  {pct_truncated:.1f}% of samples will be truncated (acceptable range).")
    else:
        print(f"\n  ✓ No truncation needed for any samples.")

    # Left-truncation warning for recommendation task
    if num_truncated > 0:
        print(f"\n⚠️  IMPORTANT: Current implementation uses LEFT-SIDE truncation!")
        print(f"  For recommendation tasks, this removes EARLY history/context.")
        print(f"  Consider:")
        print(f"    1. Increasing cutoff_len to preserve full context")
        print(f"    2. Using right-side truncation (but may cut assistant response)")
        print(f"    3. Reducing history length in data preparation")

    return {
        'split': split_name,
        'path': str(split_path),
        'count': int(stats['count']),
        'min': stats['min'],
        'max': stats['max'],
        'mean': stats['mean'],
        'p50': stats[f'p50'],
        'p90': stats[f'p90'],
        'p95': stats[f'p95'],
        'p99': stats[f'p99'],
        'num_truncated': num_truncated,
        'pct_truncated': pct_truncated,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze sequence lengths for finetuning datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("finetune/configs/qwen3_7b_delivery_hero_qlora.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Analyze all splits (train/eval/test) instead of just training",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to inspect per split.",
    )
    parser.add_argument(
        "--char-only",
        action="store_true",
        help="Skip tokenizer and report character lengths instead of token counts.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="50,90,95,99,100",
        help="Comma-separated percentiles to report (integers).",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=2,
        help="Number of examples to show from training split (0 to disable)",
    )
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=None,
        help="Cutoff length for truncation analysis (from config if not specified)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_name = cfg.get("model_name_or_path", "Qwen/Qwen3-8B")
    cutoff_len = args.cutoff_len or cfg.get("cutoff_len", 1024)

    pct_list = [int(x) for x in args.percentiles.split(",") if x.strip()]

    tokenizer = None
    if not args.char_only:
        if AutoTokenizer is None:
            raise ImportError(
                "transformers not installed; install it or rerun with --char-only"
            )
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Determine which splits to analyze
    splits_to_analyze = ["train", "eval", "test"] if args.all_splits else ["train"]

    # Resolve paths for all splits
    split_paths = {}
    for split in splits_to_analyze:
        try:
            split_path = resolve_dataset_paths(cfg, args.config, split)
            if split_path.exists():
                split_paths[split] = split_path
            else:
                print(f"Warning: {split} split not found at {split_path}")
        except Exception as e:
            print(f"Warning: Could not resolve {split} split: {e}")

    if not split_paths:
        raise FileNotFoundError("No dataset splits found!")

    # Show examples from training split if requested
    if args.show_examples > 0 and tokenizer is not None and "train" in split_paths:
        train_data = load_json_or_jsonl(split_paths["train"])
        show_examples(train_data, tokenizer, args.show_examples, cutoff_len)

    # Analyze each split
    results = []
    for split in splits_to_analyze:
        if split in split_paths:
            result = analyze_split(
                split,
                split_paths[split],
                tokenizer,
                cutoff_len,
                args.max_samples,
                pct_list,
                args.char_only,
            )
            results.append(result)

    # Print summary table if analyzing multiple splits
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY - ALL SPLITS")
        print("=" * 80)
        print(f"\n{'Split':<10} {'Count':<8} {'Min':<6} {'Max':<6} {'Mean':<7} {'P95':<6} {'Truncated%':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['split']:<10} {r['count']:<8} {r['min']:<6.0f} {r['max']:<6.0f} "
                  f"{r['mean']:<7.1f} {r['p95']:<6.1f} {r['pct_truncated']:<12.2f}")
        print()


if __name__ == "__main__":
    main()
