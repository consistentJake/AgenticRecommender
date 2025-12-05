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
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_name = cfg.get("model_name_or_path", "Qwen/Qwen3-0.6B")
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
    print(f"Measured {stats['count']} samples from {train_path} ({mode}).")
    for k, v in stats.items():
        if k == "count":
            continue
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
