#!/usr/bin/env python
"""Quick sequence-length audit for the MovieLens finetuning data.

Reads the same YAML config used for training, resolves the dataset paths,
formats samples with the chat template, and reports length percentiles.

By default it uses the tokenizer to measure tokenized length; use --char-only
to skip loading the tokenizer and just count characters.
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


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def resolve_dataset_paths(cfg: Dict[str, Any], config_path: Path) -> Tuple[Path, Path]:
    dataset_dir = resolve_dataset_dir(Path(cfg.get("dataset_dir", "data")), config_path)

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

    return (
        dataset_dir / base_key / "train.json",
        dataset_dir / base_key / "eval.json",
    )


def to_chat_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []

    system_prompt = example.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("finetune/configs/qwen3_movielens_qlora.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to inspect (train split).",
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
        help="Number of examples to show (0 to disable)",
    )
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=None,
        help="Cutoff length for truncation analysis (from config if not specified)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_name = cfg.get("model_name_or_path", "Qwen/Qwen3-0.6B")
    cutoff_len = args.cutoff_len or cfg.get("cutoff_len", 1024)
    train_path, _ = resolve_dataset_paths(cfg, args.config)
    data = load_json(train_path)

    pct_list = [int(x) for x in args.percentiles.split(",") if x.strip()]

    tokenizer = None
    if not args.char_only:
        if AutoTokenizer is None:
            raise ImportError(
                "transformers not installed; install it or rerun with --char-only"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Show examples first if requested
    if args.show_examples > 0 and tokenizer is not None:
        show_examples(data, tokenizer, args.show_examples, cutoff_len)

    # Compute statistics
    lengths: List[int] = []
    limit = args.max_samples or len(data)
    for example in data[:limit]:
        messages = to_chat_messages(example)
        if tokenizer is not None:
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

    stats = describe_lengths(lengths, pct_list)
    mode = "tokens" if tokenizer is not None else "characters"

    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH STATISTICS")
    print("=" * 80)
    print(f"\nDataset: {train_path}")
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
    print(f"  Samples > {cutoff_len} tokens: {num_truncated} ({pct_truncated:.1f}%)")
    print(f"  Samples <= {cutoff_len} tokens: {len(lengths) - num_truncated} ({100 - pct_truncated:.1f}%)")

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
        print(f"  For recommendation tasks, this removes EARLY movie history.")
        print(f"  Consider:")
        print(f"    1. Increasing cutoff_len to preserve full context")
        print(f"    2. Using right-side truncation (but may cut assistant response)")
        print(f"    3. Reducing history_len in data preparation")


if __name__ == "__main__":
    main()
