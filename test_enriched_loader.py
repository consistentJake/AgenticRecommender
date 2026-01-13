#!/usr/bin/env python3
"""
Test script for enriched_loader.py

Usage:
    python test_enriched_loader.py
"""

from agentic_recommender.data.enriched_loader import load_singapore_data
from pathlib import Path


def main():
    print("=" * 60)
    print("Testing EnrichedDataLoader")
    print("=" * 60)

    # Load the data
    data_dir = "/Users/zhenkai/Downloads/data_sg"
    print(f"\n1. Loading data from: {data_dir}")

    loader = load_singapore_data(data_dir=data_dir)

    # Load merged data
    print("\n2. Loading merged dataset...")
    merged = loader.load_merged()

    print(f"   Total rows: {len(merged)}")
    print(f"   Columns: {list(merged.columns)}")

    # Get statistics
    print("\n3. Dataset statistics:")
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Show sample data
    print("\n4. Sample data (first 5 rows):")
    print(merged.head())

    # Show column info
    print("\n5. Column information:")
    print(merged.info())

    # Test customer-specific queries
    print("\n6. Testing customer queries...")
    sample_customer = merged['customer_id'].iloc[0]
    customer_orders = loader.get_customer_orders(sample_customer)
    print(f"   Customer {sample_customer} has {len(customer_orders)} order items")

    # Test order-specific queries
    sample_order = merged['order_id'].iloc[0]
    order_items = loader.get_order_items(sample_order)
    print(f"   Order {sample_order} has {len(order_items)} items")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
