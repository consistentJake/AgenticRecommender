"""Analyze Delivery Hero JSONL dataset statistics.

This script provides comprehensive statistics on the generated training data.
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze_dataset(records: List[Dict], split_name: str) -> Dict:
    """Analyze a single dataset split."""
    print(f"\n{'=' * 80}")
    print(f"{split_name.upper()} Dataset Analysis")
    print(f"{'=' * 80}")

    # Basic stats
    total = len(records)
    positive = sum(1 for r in records if r['output'] == 'Yes')
    negative = total - positive

    print(f"\n[Basic Statistics]")
    print(f"Total samples: {total:,}")
    print(f"Positive samples (Yes): {positive:,} ({positive/total*100:.2f}%)")
    print(f"Negative samples (No): {negative:,} ({negative/total*100:.2f}%)")
    print(f"Pos/Neg ratio: 1:{negative/positive:.2f}")

    # Analyze input lengths
    input_lengths = [len(r['input']) for r in records]
    print(f"\n[Input Length Statistics]")
    print(f"Min length: {min(input_lengths):,} chars")
    print(f"Max length: {max(input_lengths):,} chars")
    print(f"Avg length: {sum(input_lengths)/len(input_lengths):,.0f} chars")

    # Analyze history counts (count number of orders in history)
    history_counts = []
    for r in records:
        # Count numbered items in history
        history_str = r['input'].split("Candidate product:")[0]
        history_items = [line for line in history_str.split('\n') if line.strip() and line.strip()[0].isdigit()]
        history_counts.append(len(history_items))

    print(f"\n[History Length Statistics]")
    print(f"Min history items: {min(history_counts)}")
    print(f"Max history items: {max(history_counts)}")
    print(f"Avg history items: {sum(history_counts)/len(history_counts):.2f}")

    # Analyze cuisines mentioned in inputs
    cuisines = []
    for r in records:
        input_text = r['input']
        # Extract cuisines from the input
        import re
        cuisine_matches = re.findall(r'cuisine:\s*(\w+)', input_text)
        cuisines.extend(cuisine_matches)

    cuisine_counts = Counter(cuisines)
    print(f"\n[Top 10 Cuisines]")
    for cuisine, count in cuisine_counts.most_common(10):
        print(f"{cuisine:15s}: {count:6,} ({count/len(cuisines)*100:5.2f}%)")

    # Analyze price ranges
    prices = []
    for r in records:
        input_text = r['input']
        import re
        price_matches = re.findall(r'price:\s*\$(\d+\.\d+)', input_text)
        prices.extend([float(p) for p in price_matches])

    print(f"\n[Price Statistics]")
    print(f"Min price: ${min(prices):.2f}")
    print(f"Max price: ${max(prices):.2f}")
    print(f"Avg price: ${sum(prices)/len(prices):.2f}")
    print(f"Median price: ${sorted(prices)[len(prices)//2]:.2f}")

    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'avg_input_length': sum(input_lengths) / len(input_lengths),
        'avg_history_length': sum(history_counts) / len(history_counts),
        'top_cuisines': cuisine_counts.most_common(5),
        'avg_price': sum(prices) / len(prices),
    }


def compare_splits(train_stats: Dict, eval_stats: Dict, test_stats: Dict):
    """Compare statistics across splits."""
    print(f"\n{'=' * 80}")
    print("Cross-Split Comparison")
    print(f"{'=' * 80}")

    print(f"\n{'Metric':<30} {'Train':>15} {'Eval':>15} {'Test':>15}")
    print("-" * 80)
    print(f"{'Total samples':<30} {train_stats['total']:>15,} {eval_stats['total']:>15,} {test_stats['total']:>15,}")
    print(f"{'Positive samples':<30} {train_stats['positive']:>15,} {eval_stats['positive']:>15,} {test_stats['positive']:>15,}")
    print(f"{'Negative samples':<30} {train_stats['negative']:>15,} {eval_stats['negative']:>15,} {test_stats['negative']:>15,}")
    print(f"{'Avg input length':<30} {train_stats['avg_input_length']:>15.0f} {eval_stats['avg_input_length']:>15.0f} {test_stats['avg_input_length']:>15.0f}")
    print(f"{'Avg history length':<30} {train_stats['avg_history_length']:>15.2f} {eval_stats['avg_history_length']:>15.2f} {test_stats['avg_history_length']:>15.2f}")
    print(f"{'Avg price':<30} ${train_stats['avg_price']:>14.2f} ${eval_stats['avg_price']:>14.2f} ${test_stats['avg_price']:>14.2f}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/zhenkai/Documents/personal/Projects/AgenticRecommender/agentic_recommender/datasets"),
        help="Directory containing the JSONL files",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Delivery Hero Dataset Analysis")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")

    # Load datasets
    train_path = args.data_dir / "train.jsonl"
    eval_path = args.data_dir / "eval.jsonl"
    test_path = args.data_dir / "test.jsonl"

    if not train_path.exists() or not eval_path.exists() or not test_path.exists():
        print("\n[ERROR] Missing dataset files. Please run convert_to_jsonl.py first.")
        return

    print("\n[info] Loading datasets...")
    train_records = load_jsonl(train_path)
    eval_records = load_jsonl(eval_path)
    test_records = load_jsonl(test_path)

    # Analyze each split
    train_stats = analyze_dataset(train_records, "train")
    eval_stats = analyze_dataset(eval_records, "eval")
    test_stats = analyze_dataset(test_records, "test")

    # Compare splits
    compare_splits(train_stats, eval_stats, test_stats)

    # Load and display metadata
    meta_path = args.data_dir / "meta.json"
    if meta_path.exists():
        print(f"\n{'=' * 80}")
        print("Dataset Metadata")
        print(f"{'=' * 80}")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(json.dumps(meta, indent=2))

    print(f"\n{'=' * 80}")
    print("Analysis Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
