"""Delivery Hero → Alpaca chat training data pipeline.

This script:
1. Loads Delivery Hero orders, vendors, and products data
2. Builds sequential recommendation examples using purchase history
3. Converts them to Alpaca-style chat records for LLaMA-Factory
4. Emits train/val/test splits

Format matches prepare_movielens.py for compatibility with training pipeline.
"""

import pandas as pd
import json
import argparse
import math
from typing import List, Dict, Tuple
import random
from pathlib import Path
from dataclasses import dataclass

# Default paths
DEFAULT_SOURCE = Path("/Users/zhenkai/Documents/personal/Projects/AgenticRecommender/agentic_recommender/datasets/source")
DEFAULT_OUTPUT = Path("/Users/zhenkai/Documents/personal/Projects/AgenticRecommender/agentic_recommender/datasets")

DEFAULT_SYSTEM_PROMPT = (
    "You are a food delivery recommendation assistant. "
    "Given a user's recent purchase history and a candidate product, "
    "decide whether the user is likely to purchase this product next. "
    "You must answer ONLY with 'Yes' or 'No'."
)
DEFAULT_INSTRUCTION = (
    "Predict whether the user will purchase the candidate product. "
    "Answer only with Yes or No."
)


@dataclass
class Sample:
    split: str  # train/val/test
    customer_id: str
    history_items: List[Dict]
    candidate: Dict
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory containing Delivery Hero data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination folder for processed dataset.",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=10,
        help="Number of historical purchases to include in prompt.",
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
        help="Fraction of each user's timeline reserved for test.",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=1,
        help="Number of negative samples per positive sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for chat format.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction prompt for Alpaca format.",
    )
    parser.add_argument(
        "--sample-customers",
        type=int,
        default=None,
        help="Sample only N customers for testing (default: all customers).",
    )
    return parser.parse_args()


def time_to_bucket(time_str: str) -> str:
    """Convert time string to time bucket (mor/aft/eve)."""
    try:
        hour = int(time_str.split(':')[0])
        if 0 <= hour < 12:
            return "mor"
        elif 12 <= hour < 18:
            return "aft"
        else:
            return "eve"
    except:
        return "aft"  # default


def load_data(source_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load orders, vendors, and products data."""
    print(f"[info] Loading data from {source_dir}")

    orders_train = pd.read_csv(source_dir / "orders_se_train.txt")
    orders_test = pd.read_csv(source_dir / "orders_se_test.txt")
    vendors = pd.read_csv(source_dir / "vendors_se.txt")
    products = pd.read_csv(source_dir / "products_se.txt")

    print(f"[info] Train orders: {len(orders_train)}")
    print(f"[info] Test orders: {len(orders_test)}")
    print(f"[info] Vendors: {len(vendors)}")
    print(f"[info] Products: {len(products)}")

    # Combine train and test for unified processing
    orders = pd.concat([orders_train, orders_test], ignore_index=True)

    return orders, vendors, products


def merge_order_data(orders_df, vendors_df, products_df):
    """Merge orders with vendors and products."""
    print("[info] Merging datasets...")

    # Merge orders with vendors
    merged = orders_df.merge(
        vendors_df[['vendor_id', 'primary_cuisine', 'geohash']],
        on='vendor_id',
        how='left',
        suffixes=('_user', '_vendor')
    )

    # Merge with products
    merged = merged.merge(
        products_df[['vendor_id', 'product_id', 'name', 'unit_price']],
        on=['vendor_id', 'product_id'],
        how='left'
    )

    # Clean column names
    merged = merged.rename(columns={
        'geohash_user': 'user_geo',
        'geohash_vendor': 'vendor_geo',
        'name': 'product_name',
        'primary_cuisine': 'cuisine'
    })

    # Convert time to bucket
    merged['time_bucket'] = merged['order_time'].apply(time_to_bucket)

    # Extract day number from order_day (e.g., "11 days" -> 11)
    merged['day_num'] = merged['order_day'].str.extract(r'(\d+)').astype(int)

    # Handle missing values
    merged['product_name'] = merged['product_name'].fillna('Unknown Product')
    merged['cuisine'] = merged['cuisine'].fillna('unknown')
    merged['unit_price'] = merged['unit_price'].fillna(0.0)
    merged['vendor_geo'] = merged['vendor_geo'].fillna('unknown')

    return merged


def format_history_list(history_items: List[Dict]) -> str:
    """Format purchase history as numbered list (similar to MovieLens)."""
    if not history_items:
        return "No previous purchase history."

    lines = []
    for idx, item in enumerate(history_items, start=1):
        line = (
            f"{idx}. {item['product_name']} "
            f"(day={item['day_num']}, time={item['time_bucket']}, "
            f"cuisine={item['cuisine']}, price=${item['unit_price']:.2f})"
        )
        lines.append(line)

    return "\n".join(lines)


def split_bounds(length: int, val_ratio: float, test_ratio: float) -> Tuple[int, int]:
    """Calculate split boundaries for train/val/test."""
    if length <= 0:
        return length, length
    test_count = max(1, int(math.floor(length * test_ratio)))
    val_count = max(1, int(math.floor(length * val_ratio)))
    test_start = length - test_count
    val_start = max(test_start - val_count, 0)
    return val_start, test_start


def build_samples(
    merged_df: pd.DataFrame,
    history_len: int,
    val_ratio: float,
    test_ratio: float,
    neg_ratio: int,
) -> List[Sample]:
    """Build positive and negative samples with train/val/test splits."""
    print("[info] Building samples...")

    samples = []
    customer_groups = merged_df.groupby('customer_id')
    total_customers = len(customer_groups)

    print(f"[info] Processing {total_customers} customers...")

    # Pre-collect all unique products for negative sampling
    print("[info] Collecting unique products for negative sampling...")
    all_products = merged_df[['product_id', 'product_name', 'vendor_geo', 'cuisine', 'unit_price']].drop_duplicates()
    all_products_list = all_products.to_dict('records')
    print(f"[info] Found {len(all_products_list)} unique products")

    processed_customers = 0
    for customer_id, group in customer_groups:
        processed_customers += 1
        if processed_customers % 1000 == 0:
            print(f"[progress] Processed {processed_customers}/{total_customers} customers, generated {len(samples)} samples so far...")
        # Sort by order day and time
        group = group.sort_values(['day_num', 'order_time']).reset_index(drop=True)

        # Skip customers without enough history
        if len(group) <= history_len:
            continue

        # Usable events (after warm-up history)
        usable = group.iloc[history_len:].reset_index(drop=True)
        val_start, test_start = split_bounds(len(usable), val_ratio, test_ratio)

        # Get user's purchased products for negative sampling
        user_product_ids = set(group['product_id'].unique())

        for idx in range(len(usable)):
            # Get history slice
            history_slice = group.iloc[idx:idx + history_len].to_dict('records')

            # Current order (positive sample)
            current_order = usable.iloc[idx].to_dict()

            # Determine split
            if idx >= test_start:
                split = "test"
            elif idx >= val_start:
                split = "val"
            else:
                split = "train"

            # Add positive sample
            samples.append(
                Sample(
                    split=split,
                    customer_id=customer_id,
                    history_items=history_slice,
                    candidate=current_order,
                    label="Yes"
                )
            )

            # Add negative samples
            negative_candidates = [
                p for p in all_products_list
                if p['product_id'] not in user_product_ids
            ]

            if negative_candidates and neg_ratio > 0:
                n_neg = min(neg_ratio, len(negative_candidates))
                neg_samples = random.sample(negative_candidates, n_neg)

                for neg_product in neg_samples:
                    samples.append(
                        Sample(
                            split=split,
                            customer_id=customer_id,
                            history_items=history_slice,
                            candidate=neg_product,
                            label="No"
                        )
                    )

    print(f"[info] Generated {len(samples)} samples")
    return samples


def format_alpaca(sample: Sample, history_len: int, instruction: str, system: str) -> Dict:
    """Convert sample to Alpaca format (instruction/input/output/system/history)."""
    history_str = format_history_list(sample.history_items)
    candidate = sample.candidate

    input_text = (
        f"User's last {history_len} food orders:\n{history_str}\n\n"
        f"Candidate product:\n"
        f"- {candidate['product_name']} "
        f"(cuisine: {candidate['cuisine']}, price: ${candidate['unit_price']:.2f})\n\n"
        "Should we recommend this product to the user? Answer Yes or No."
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": sample.label,
        "system": system,
        "history": [],
    }


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Write samples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict) -> None:
    """Write metadata to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Main conversion pipeline."""
    args = parse_args()
    random.seed(args.seed)

    print("=" * 80)
    print("Delivery Hero → Alpaca Dataset Conversion")
    print("=" * 80)
    print(f"[config] History length: {args.history_len}")
    print(f"[config] Val ratio: {args.val_ratio}")
    print(f"[config] Test ratio: {args.test_ratio}")
    print(f"[config] Negative ratio: {args.neg_ratio}")
    print(f"[config] Sample customers: {args.sample_customers if args.sample_customers else 'all'}")
    print("=" * 80)

    # Load and merge data
    orders, vendors, products = load_data(args.source)

    # Optional: sample customers for testing
    if args.sample_customers:
        print(f"[info] Sampling {args.sample_customers} customers for testing...")
        unique_customers = orders['customer_id'].unique()
        sampled_customers = random.sample(list(unique_customers), min(args.sample_customers, len(unique_customers)))
        orders = orders[orders['customer_id'].isin(sampled_customers)]
        print(f"[info] After sampling: {len(orders)} orders from {len(sampled_customers)} customers")

    merged = merge_order_data(orders, vendors, products)
    print(f"[info] Merged dataset size: {len(merged)} rows")

    # Build samples with train/val/test splits
    print("[info] Starting sample generation...")
    samples = build_samples(
        merged_df=merged,
        history_len=args.history_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_ratio=args.neg_ratio,
    )

    if not samples:
        raise RuntimeError("No samples generated. Check history length or data quality.")

    # Shuffle samples
    print("[info] Shuffling samples...")
    random.shuffle(samples)

    # Convert to Alpaca format and split by dataset
    print("[info] Converting to Alpaca format and splitting datasets...")
    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "val"]
    test_samples = [s for s in samples if s.split == "test"]

    print(f"[info] Train samples: {len(train_samples)}")
    print(f"[info] Val samples: {len(val_samples)}")
    print(f"[info] Test samples: {len(test_samples)}")

    print("[info] Formatting train records...")
    train_records = [
        format_alpaca(s, args.history_len, args.instruction, args.system_prompt)
        for s in train_samples
    ]
    print("[info] Formatting val records...")
    val_records = [
        format_alpaca(s, args.history_len, args.instruction, args.system_prompt)
        for s in val_samples
    ]
    print("[info] Formatting test records...")
    test_records = [
        format_alpaca(s, args.history_len, args.instruction, args.system_prompt)
        for s in test_samples
    ]

    if not train_records or not val_records or not test_records:
        raise RuntimeError(
            "Some splits are empty. Adjust val/test ratios or history requirements."
        )

    # Write outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Writing train.jsonl ({len(train_records)} records)...")
    write_jsonl(output_dir / "train.jsonl", train_records)
    print(f"[info] Writing eval.jsonl ({len(val_records)} records)...")
    write_jsonl(output_dir / "eval.jsonl", val_records)
    print(f"[info] Writing test.jsonl ({len(test_records)} records)...")
    write_jsonl(output_dir / "test.jsonl", test_records)

    # Write metadata
    meta = {
        "dataset": "delivery_hero",
        "history_len": args.history_len,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "neg_ratio": args.neg_ratio,
        "num_train": len(train_records),
        "num_eval": len(val_records),
        "num_test": len(test_records),
        "source_dir": str(args.source),
    }
    write_json(output_dir / "meta.json", meta)

    # Count positive/negative samples
    train_pos = sum(1 for r in train_records if r['output'] == 'Yes')
    val_pos = sum(1 for r in val_records if r['output'] == 'Yes')
    test_pos = sum(1 for r in test_records if r['output'] == 'Yes')

    # Print summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(json.dumps({
        "train": {"total": len(train_records), "positive": train_pos, "negative": len(train_records) - train_pos},
        "eval": {"total": len(val_records), "positive": val_pos, "negative": len(val_records) - val_pos},
        "test": {"total": len(test_records), "positive": test_pos, "negative": len(test_records) - test_pos},
        "output_dir": str(output_dir),
    }, indent=2))

    # Show a sample
    print("\n" + "=" * 80)
    print("Sample from train set:")
    print("=" * 80)
    print(json.dumps(train_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
