#!/usr/bin/env python
"""Inspect preprocessed dataset cache files and display formatted examples."""

import argparse
from pathlib import Path
from datasets import Dataset

def inspect_arrow_file(cache_file: Path, num_examples: int = 5, show_full: bool = False):
    """Load and display examples from an Arrow cache file.

    Args:
        cache_file: Path to the .arrow cache file
        num_examples: Number of examples to display
        show_full: If True, show full text; if False, truncate long texts
    """
    print("="*80)
    print(f"Inspecting cache file: {cache_file}")
    print("="*80)

    # The cache_file should be the dataset directory itself
    cache_dir = cache_file

    try:
        # Load dataset from disk
        from datasets import load_from_disk
        dataset = load_from_disk(str(cache_dir))

        print(f"\nDataset info:")
        print(f"  - Total examples: {len(dataset)}")
        print(f"  - Features: {list(dataset.features.keys())}")
        print(f"  - First feature types: {dataset.features}")
        print()

        # Display examples
        print("="*80)
        print(f"Showing {min(num_examples, len(dataset))} example(s):")
        print("="*80)

        for i in range(min(num_examples, len(dataset))):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i+1}/{min(num_examples, len(dataset))}")
            print(f"{'='*80}")

            example = dataset[i]

            # Display all fields in the example
            for key, value in example.items():
                print(f"\n[{key}]:")
                print("-"*80)

                if isinstance(value, str):
                    if show_full or len(value) <= 500:
                        print(value)
                    else:
                        print(value[:500] + f"\n... [truncated, {len(value)} chars total]")
                elif isinstance(value, list):
                    if show_full or len(value) <= 10:
                        for idx, item in enumerate(value):
                            print(f"  [{idx}]: {item}")
                    else:
                        for idx, item in enumerate(value[:10]):
                            print(f"  [{idx}]: {item}")
                        print(f"  ... [truncated, {len(value)} items total]")
                else:
                    print(value)

                print("-"*80)

        print(f"\n{'='*80}")
        print("Summary Statistics:")
        print(f"{'='*80}")

        # If 'text' field exists, show some stats
        if 'text' in dataset.features:
            texts = [example['text'] for example in dataset]
            text_lengths = [len(text) for text in texts]
            print(f"\nText field statistics:")
            print(f"  - Average length: {sum(text_lengths) / len(text_lengths):.1f} characters")
            print(f"  - Min length: {min(text_lengths)} characters")
            print(f"  - Max length: {max(text_lengths)} characters")

            # Show a sample of the first text to understand format
            print(f"\nFirst text preview (first 300 chars):")
            print("-"*80)
            print(texts[0][:300])
            if len(texts[0]) > 300:
                print(f"... [truncated, {len(texts[0])} chars total]")
            print("-"*80)

        # If 'input_ids' field exists, show token stats
        if 'input_ids' in dataset.features:
            token_counts = [len(example['input_ids']) for example in dataset]
            print(f"\nToken statistics (input_ids):")
            print(f"  - Average tokens: {sum(token_counts) / len(token_counts):.1f}")
            print(f"  - Min tokens: {min(token_counts)}")
            print(f"  - Max tokens: {max(token_counts)}")

            # Show token IDs for first example
            print(f"\nFirst example token IDs (first 20):")
            print(f"  {dataset[0]['input_ids'][:20]}")

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTrying alternative approach (reading Arrow file directly)...")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Read arrow file
            table = pa.ipc.open_file(str(cache_file)).read_all()
            print(f"\nArrow table schema:")
            print(table.schema)
            print(f"\nTotal rows: {len(table)}")

            # Convert to pandas for easier viewing
            df = table.to_pandas()
            print(f"\nFirst {num_examples} rows:")
            print(df.head(num_examples))

        except Exception as e2:
            print(f"Error with alternative approach: {e2}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Inspect preprocessed dataset cache files"
    )
    parser.add_argument(
        "cache_file",
        type=Path,
        help="Path to the cache .arrow file or cache directory"
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=3,
        help="Number of examples to display (default: 3)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full text without truncation"
    )

    args = parser.parse_args()

    # If given a .arrow file, use its parent directory
    if args.cache_file.is_file() and args.cache_file.suffix == '.arrow':
        cache_path = args.cache_file.parent
    else:
        cache_path = args.cache_file

    # Make sure path is absolute
    cache_path = cache_path.resolve()

    inspect_arrow_file(cache_path, args.num_examples, args.full)


if __name__ == "__main__":
    main()
