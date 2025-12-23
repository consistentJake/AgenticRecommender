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
        "--country-code",
        type=str,
        default="sg",
        help="Country code for data files (e.g., 'sg', 'se', 'tw'). Default: sg",
    )
    parser.add_argument(
        "--min-history-len",
        type=int,
        default=3,
        help="Minimum number of historical purchases required.",
    )
    parser.add_argument(
        "--max-history-len",
        type=int,
        default=5,
        help="Maximum number of historical purchases to include in prompt.",
    )
    parser.add_argument(
        "--enable-sliding-windows",
        action="store_true",
        help="Enable sliding window technique to generate multiple samples per user.",
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


def get_hour_from_time(time_str: str) -> int:
    """Extract hour from time string."""
    try:
        return int(time_str.split(':')[0])
    except:
        return 12  # default noon


def get_day_of_week(day_num: int) -> str:
    """Convert day number to day of week (Mon-Sun)."""
    # Assuming day_num is days from some starting point, we'll use modulo 7
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    return day_names[day_num % 7]


def load_data(source_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load orders, vendors, and products data."""
    print(f"[info] Loading data from {source_dir}")

    orders_train = pd.read_csv(source_dir / "orders_sg_train.txt")
    orders_test = pd.read_csv(source_dir / "orders_sg_test.txt")
    vendors = pd.read_csv(source_dir / "vendors_sg.txt")
    products = pd.read_csv(source_dir / "products_sg.txt")

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
    
    # Extract hour from order_time
    merged['hour'] = merged['order_time'].apply(get_hour_from_time)

    # Extract day number from order_day (e.g., "11 days" -> 11)
    merged['day_num'] = merged['order_day'].str.extract(r'(\d+)').astype(int)
    
    # Get day of week
    merged['day_of_week'] = merged['day_num'].apply(get_day_of_week)

    # Handle missing values
    merged['product_name'] = merged['product_name'].fillna('Unknown Product')
    merged['cuisine'] = merged['cuisine'].fillna('unknown')
    merged['unit_price'] = merged['unit_price'].fillna(0.0)
    merged['vendor_geo'] = merged['vendor_geo'].fillna('unknown')

    return merged


def format_history_list(history_items: List[Dict]) -> str:
    """Format purchase history as table format (oldest to newest)."""
    if not history_items:
        return "No previous purchase history."

    # Create table header
    lines = [
        "User recent orders (oldest → newest):\n",
        "| idx | day | hour | cuisine     | price |",
        "|-----|-----|------|-------------|",
    ]
    
    # Add table rows
    for idx, item in enumerate(history_items, start=1):
        line = (
            f"| {idx:<3} | {item['day_of_week']:<3} | {item['hour']:<4} | "
            f"{item['cuisine']:<11} | {item['unit_price']:>5.2f} |"
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


def analyze_customer_history_lengths(merged_df: pd.DataFrame) -> None:
    """Analyze the distribution of historic event lengths for each customer."""
    print("\n" + "=" * 80)
    print("CUSTOMER HISTORY LENGTH ANALYSIS")
    print("=" * 80)
    
    customer_groups = merged_df.groupby('customer_id')
    history_lengths = []
    
    for customer_id, group in customer_groups:
        history_lengths.append(len(group))
    
    # Calculate statistics
    total_customers = len(history_lengths)
    min_len = min(history_lengths)
    max_len = max(history_lengths)
    avg_len = sum(history_lengths) / len(history_lengths)
    
    print(f"Total customers: {total_customers:,}")
    print(f"Min historic events per customer: {min_len}")
    print(f"Max historic events per customer: {max_len}")
    print(f"Average historic events per customer: {avg_len:.2f}")
    
    # Calculate percentiles
    sorted_lengths = sorted(history_lengths)
    p25 = sorted_lengths[int(0.25 * len(sorted_lengths))]
    p50 = sorted_lengths[int(0.50 * len(sorted_lengths))]
    p75 = sorted_lengths[int(0.75 * len(sorted_lengths))]
    p90 = sorted_lengths[int(0.90 * len(sorted_lengths))]
    p95 = sorted_lengths[int(0.95 * len(sorted_lengths))]
    
    print(f"\nPercentiles:")
    print(f"25th percentile: {p25}")
    print(f"50th percentile (median): {p50}")
    print(f"75th percentile: {p75}")
    print(f"90th percentile: {p90}")
    print(f"95th percentile: {p95}")
    
    # Show data preservation for different sequence lengths
    print(f"\nData Preservation Analysis:")
    print(f"{'Seq Length':<12} {'Customers Kept':<15} {'% Preserved':<12} {'Customers Lost':<15}")
    print("-" * 60)
    
    for seq_len in [5, 10, 15, 20, 25, 30, 40, 50]:
        customers_kept = sum(1 for length in history_lengths if length > seq_len)
        customers_lost = total_customers - customers_kept
        pct_preserved = (customers_kept / total_customers) * 100
        print(f"{seq_len:<12} {customers_kept:<15,} {pct_preserved:<12.1f}% {customers_lost:<15,}")
    
    # Distribution by bins
    print(f"\nHistoric Event Length Distribution:")
    bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, float('inf'))]
    for min_val, max_val in bins:
        if max_val == float('inf'):
            count = sum(1 for length in history_lengths if length >= min_val)
            label = f"{min_val}+"
        else:
            count = sum(1 for length in history_lengths if min_val <= length <= max_val)
            label = f"{min_val}-{max_val}"
        pct = (count / total_customers) * 100
        print(f"{label:<10}: {count:>6,} customers ({pct:>5.1f}%)")
    
    print("=" * 80 + "\n")


def build_samples(
    merged_df: pd.DataFrame,
    min_history_len: int,
    max_history_len: int,
    enable_sliding_windows: bool,
    val_ratio: float,
    test_ratio: float,
    neg_ratio: int,
) -> List[Sample]:
    """Build positive and negative samples with train/val/test splits."""
    print("[info] Building samples...")

    # First analyze customer history lengths before filtering
    analyze_customer_history_lengths(merged_df)

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

        # Skip customers without minimum history
        if len(group) < min_history_len:
            continue

        # Get user's purchased products and product names for negative sampling
        user_product_ids = set(group['product_id'].unique())
        user_product_names = set(group['product_name'].unique())

        if enable_sliding_windows:
            # Generate multiple samples using sliding windows
            # For each possible prediction point, create a sample
            for pred_idx in range(min_history_len, len(group)):
                # Determine history slice (use up to max_history_len)
                history_start = max(0, pred_idx - max_history_len)
                history_slice = group.iloc[history_start:pred_idx].to_dict('records')
                
                # Current order (positive sample)
                current_order = group.iloc[pred_idx].to_dict()

                # Determine usable samples for splitting
                total_predictions = len(group) - min_history_len
                current_pred_idx = pred_idx - min_history_len
                val_start, test_start = split_bounds(total_predictions, val_ratio, test_ratio)

                # Determine split
                if current_pred_idx >= test_start:
                    split = "test"
                elif current_pred_idx >= val_start:
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

                # Add negative samples - exclude products with same ID OR same name
                negative_candidates = [
                    p for p in all_products_list
                    if (p['product_id'] not in user_product_ids and 
                        p['product_name'] not in user_product_names)
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
        else:
            # Original approach: one sample per user using latest history
            if len(group) < max_history_len + 1:  # Need history + 1 for prediction
                continue
                
            # Use the first max_history_len items as history
            history_slice = group.iloc[:max_history_len].to_dict('records')
            # Predict the next item
            current_order = group.iloc[max_history_len].to_dict()

            # For single sample per user, distribute across splits
            # Use customer hash to deterministically assign splits
            import hashlib
            customer_hash = int(hashlib.md5(str(customer_id).encode()).hexdigest(), 16)
            hash_mod = customer_hash % 100
            
            if hash_mod < int(test_ratio * 100):
                split = "test"
            elif hash_mod < int((test_ratio + val_ratio) * 100):
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
                if (p['product_id'] not in user_product_ids and 
                    p['product_name'] not in user_product_names)
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


def format_alpaca(sample: Sample, max_history_len: int, system: str) -> Dict:
    """Convert sample to Alpaca format (input/output/system/history) - optimized for chat template."""
    history_str = format_history_list(sample.history_items)
    candidate = sample.candidate

    input_text = (
        f"{history_str}\n\n"
        f"Candidate product:\n"
        f"- {candidate['product_name']} "
        f"(cuisine: {candidate['cuisine']}, price: ${candidate['unit_price']:.2f})"
    )

    return {
        "instruction": input_text,
        "input": "",
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
    print(f"[config] Min history length: {args.min_history_len}")
    print(f"[config] Max history length: {args.max_history_len}")
    print(f"[config] Enable sliding windows: {args.enable_sliding_windows}")
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
        min_history_len=args.min_history_len,
        max_history_len=args.max_history_len,
        enable_sliding_windows=args.enable_sliding_windows,
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
        format_alpaca(s, args.max_history_len, args.system_prompt)
        for s in train_samples
    ]
    print("[info] Formatting val records...")
    val_records = [
        format_alpaca(s, args.max_history_len, args.system_prompt)
        for s in val_samples
    ]
    print("[info] Formatting test records...")
    test_records = [
        format_alpaca(s, args.max_history_len, args.system_prompt)
        for s in test_samples
    ]

    # For small datasets without sliding windows, allow empty val/test splits
    if args.enable_sliding_windows and (not train_records or not val_records or not test_records):
        raise RuntimeError(
            "Some splits are empty. Adjust val/test ratios or history requirements."
        )
    elif not args.enable_sliding_windows and not train_records:
        raise RuntimeError(
            "No training samples generated. Check history requirements."
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
        "min_history_len": args.min_history_len,
        "max_history_len": args.max_history_len,
        "enable_sliding_windows": args.enable_sliding_windows,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "neg_ratio": args.neg_ratio,
        "num_train": len(train_records),
        "num_eval": len(val_records),
        "num_test": len(test_records),
        "source_dir": str(args.source),
    }
    write_json(output_dir / "meta.json", meta)

    # Write dataset_info.json for LLaMA-Factory
    dataset_info = {
        "train": {
            "file_name": "train.jsonl",
            "file_name_eval": "eval.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
                "system": "system"
            }
        },
        "eval": {
            "file_name": "eval.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
                "system": "system"
            }
        },
        "test": {
            "file_name": "test.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
                "system": "system"
            }
        }
    }
    print("[info] Writing dataset_info.json for LLaMA-Factory...")
    write_json(output_dir / "dataset_info.json", dataset_info)

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
    print("\n[info] Generated files:")
    print(f"  - {output_dir / 'train.jsonl'}")
    print(f"  - {output_dir / 'eval.jsonl'}")
    print(f"  - {output_dir / 'test.jsonl'}")
    print(f"  - {output_dir / 'meta.json'}")
    print(f"  - {output_dir / 'dataset_info.json'} (for LLaMA-Factory)")
    print(f"\n[info] Ready to use with LLaMA-Factory!")
    print(f"  Set dataset_dir: {output_dir}")
    print(f"  Set dataset: train")
    print(f"  Set eval_dataset: eval")

    # Show a sample
    print("\n" + "=" * 80)
    print("Sample from train set:")
    print("=" * 80)
    print(json.dumps(train_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Example usage commands:
#
# 1. Basic conversion with optimized settings (recommended):
#    python convert_to_jsonl.py --min-history-len 3 --max-history-len 5 --enable-sliding-windows
#
# 2. Convert with custom paths:
#    python convert_to_jsonl.py --source /path/to/delivery_hero/data --output-dir /path/to/output
#
# 3. Sample customers for testing with sliding windows:
#    python convert_to_jsonl.py --sample-customers 1000 --min-history-len 3 --max-history-len 5 --enable-sliding-windows
#
# 4. Without sliding windows (one sample per user):
#    python convert_to_jsonl.py --min-history-len 3 --max-history-len 5 --sample-customers 1000
#
# 5. Adjust ratios and negative sampling:
#    python convert_to_jsonl.py --min-history-len 3 --max-history-len 5 --val-ratio 0.1 --test-ratio 0.2 --neg-ratio 2
#
# 6. Custom prompts:
#    python convert_to_jsonl.py --system-prompt "You are a recommendation system..."
#
# 7. Full example with multiple parameters:
#    python agentic_recommender/datasets/convert_to_jsonl.py \
#        --source agentic_recommender/datasets/delivery_hero \
#        --output-dir agentic_recommender/datasets \
#        --min-history-len 3 \
#        --max-history-len 5 \
#        --enable-sliding-windows \
#        --val-ratio 0.05 \
#        --test-ratio 0.1 \
#        --neg-ratio 1 \
#        --sample-customers 1000 \
#        --seed 42
