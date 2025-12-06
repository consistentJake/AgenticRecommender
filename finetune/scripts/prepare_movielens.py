"""MovieLens → Qwen3 chat training data pipeline.

This script:
1. (Optionally) downloads the MovieLens "latest-small" dataset.
2. Builds sequential recommendation examples using the last N interactions.
3. Converts them to Alpaca/Qwen-style chat records for LLaMA-Factory.
4. Emits raw test rows for the evaluation script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a movie recommendation assistant. Given a user's recent history "
    "and a candidate movie, respond with exactly one word: Yes or No."
)
DEFAULT_INSTRUCTION = (
    "Predict whether the user will like the candidate movie. Answer only "
    "with Yes or No."
)


@dataclass
class Sample:
    split: str  # train/val/test
    user_id: int
    history_titles: List[str]
    candidate_title: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=False,
        help="Directory containing MovieLens CSV files (ratings.csv, movies.csv).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="(Deprecated: auto-downloads when --source is not provided) Download ml-latest-small.zip to a temp folder automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination folder for processed dataset.",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=15,
        help="Number of historical interactions to feed into the prompt.",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Ratings >= threshold → Yes, otherwise No.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of each user's timeline reserved for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of each user's timeline reserved for held-out test.",
    )
    parser.add_argument(
        "--max-per-user",
        type=int,
        default=None,
        help="Optional cap on examples per user (after history warmup).",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Optional cap on total eval records after splitting (uses shuffled order).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt stored alongside each Alpaca record.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction prompt for each Alpaca record.",
    )
    return parser.parse_args()


def maybe_download_movielens(tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_root / "ml-latest-small.zip"
    if zip_path.exists():
        print(f"[info] Reusing cached file at {zip_path}")
    else:
        print(f"[info] Downloading MovieLens to {zip_path}")
        with requests.get(MOVIELENS_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as fout:
                shutil.copyfileobj(resp.raw, fout)
    extract_dir = tmp_root / "ml-latest-small"
    if not extract_dir.exists():
        print(f"[info] Extracting archive to {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(tmp_root)
    return extract_dir


def load_movielens(source_dir: Path) -> Tuple[pd.DataFrame, Dict[int, str]]:
    ratings_path = source_dir / "ratings.csv"
    movies_path = source_dir / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"Missing MovieLens CSV files in {source_dir}. "
            "Provide --source or enable --download."
        )
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    movie_map = dict(zip(movies["movieId"], movies["title"]))
    return ratings, movie_map


def iter_user_events(
    ratings: pd.DataFrame,
) -> Iterable[Tuple[int, List[Tuple[int, float, int]]]]:
    ratings = ratings.sort_values(["userId", "timestamp"])
    grouped = ratings.groupby("userId", sort=False)
    for user_id, df in grouped:
        events = [
            (int(row.movieId), float(row.rating), int(row.timestamp))
            for row in df.itertuples(index=False)
        ]
        yield int(user_id), events


def make_history_strings(
    history: List[Tuple[int, float, int]], movie_map: Dict[int, str]
) -> List[str]:
    lines = []
    for idx, (movie_id, rating, _) in enumerate(history, start=1):
        title = movie_map.get(movie_id, f"Movie {movie_id}")
        lines.append(f"{idx}. {title} (rating ≈ {rating:.1f})")
    return lines


def classify_label(rating: float, threshold: float) -> str:
    return "Yes" if rating >= threshold else "No"


def split_bounds(length: int, val_ratio: float, test_ratio: float) -> Tuple[int, int]:
    if length <= 0:
        return length, length
    test_count = max(1, int(math.floor(length * test_ratio)))
    val_count = max(1, int(math.floor(length * val_ratio)))
    test_start = length - test_count
    val_start = max(test_start - val_count, 0)
    return val_start, test_start


def build_samples(
    ratings: pd.DataFrame,
    movie_map: Dict[int, str],
    history_len: int,
    threshold: float,
    val_ratio: float,
    test_ratio: float,
    max_per_user: int | None,
) -> List[Sample]:
    samples: List[Sample] = []
    for user_id, events in iter_user_events(ratings):
        if len(events) <= history_len:
            continue
        usable = events[history_len:]
        if max_per_user:
            usable = usable[:max_per_user]
        val_start, test_start = split_bounds(len(usable), val_ratio, test_ratio)
        for idx, current in enumerate(usable):
            history_slice = events[idx : idx + history_len]
            history_titles = make_history_strings(history_slice, movie_map)
            movie_id, rating, _ = current
            candidate_title = movie_map.get(movie_id, f"Movie {movie_id}")
            label = classify_label(rating, threshold)
            if idx >= test_start:
                split = "test"
            elif idx >= val_start:
                split = "val"
            else:
                split = "train"
            samples.append(
                Sample(
                    split=split,
                    user_id=user_id,
                    history_titles=history_titles,
                    candidate_title=candidate_title,
                    label=label,
                )
            )
    return samples


def format_alpaca(sample: Sample, history_len: int, instruction: str, system: str) -> Dict:
    history_str = "\n".join(sample.history_titles)
    input_text = (
        f"User's last {history_len} watched movies:\n{history_str}\n\n"
        f"Candidate movie:\n{sample.candidate_title}\n\n"
        "Should we recommend this movie to the user? Answer Yes or No."
    )
    return {
        "instruction": instruction,
        "input": input_text,
        "output": sample.label,
        "system": system,
        "history": [],
    }


def write_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.source:
        # Use provided source directory
        source_dir = args.source
    else:
        # Auto-download to temp directory when --source is not provided
        print("[info] No --source provided, auto-downloading MovieLens dataset...")
        tmp_dir = Path(tempfile.mkdtemp(prefix="movielens_"))
        source_dir = maybe_download_movielens(tmp_dir)

    ratings, movie_map = load_movielens(source_dir)
    samples = build_samples(
        ratings=ratings,
        movie_map=movie_map,
        history_len=args.history_len,
        threshold=args.rating_threshold,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_per_user=args.max_per_user,
    )
    if not samples:
        raise RuntimeError("No samples generated. Check history length or filters.")

    random.shuffle(samples)
    train_records = [
        format_alpaca(s, args.history_len, args.instruction, args.system_prompt)
        for s in samples
        if s.split == "train"
    ]
    val_records = [
        format_alpaca(s, args.history_len, args.instruction, args.system_prompt)
        for s in samples
        if s.split == "val"
    ]
    test_rows = [
        {
            "user_id": s.user_id,
            "history_titles": s.history_titles,
            "candidate_title": s.candidate_title,
            "label": s.label,
        }
        for s in samples
        if s.split == "test"
    ]

    if not train_records or not val_records or not test_rows:
        raise RuntimeError(
            "Some splits are empty. Adjust val/test ratios or history requirements."
        )

    if args.max_eval and len(val_records) > args.max_eval:
        # Shuffle already applied to samples; slicing keeps determinism under the same seed.
        val_records = val_records[: args.max_eval]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train.json", train_records)
    write_json(output_dir / "eval.json", val_records)
    write_jsonl(output_dir / "test_raw.jsonl", test_rows)

    meta = {
        "history_len": args.history_len,
        "rating_threshold": args.rating_threshold,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "max_eval": args.max_eval,
        "num_train": len(train_records),
        "num_eval": len(val_records),
        "num_test": len(test_rows),
        "source_dir": str(source_dir),
    }
    write_json(output_dir / "meta.json", meta)

    print(
        json.dumps(
            {
                "train": len(train_records),
                "eval": len(val_records),
                "test": len(test_rows),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
