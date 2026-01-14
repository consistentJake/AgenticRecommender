"""
Pytest configuration and fixtures for agentic recommender tests.

These fixtures use REAL data samples from Singapore dataset.
"""

import pytest
import pandas as pd
from pathlib import Path


# Path to original Singapore dataset
ORIGINAL_DATA_DIR = Path("/Users/zhenkai/Downloads/data_sg")


@pytest.fixture(scope="session")
def data_dir():
    """Return path to original data directory."""
    return ORIGINAL_DATA_DIR


@pytest.fixture(scope="session")
def all_orders(data_dir):
    """Load all training orders (cached for session)."""
    return pd.read_csv(data_dir / "orders_sg_train.txt")


@pytest.fixture(scope="session")
def all_vendors(data_dir):
    """Load all vendors."""
    return pd.read_csv(data_dir / "vendors_sg.txt")


@pytest.fixture(scope="session")
def all_products(data_dir):
    """Load all products."""
    return pd.read_csv(data_dir / "products_sg.txt")


@pytest.fixture
def sample_orders(all_orders):
    """Load first 10000 orders for fast testing."""
    return all_orders.head(10000).copy()


@pytest.fixture
def sample_customer_id(all_orders):
    """Get a customer with moderate history (10-50 orders)."""
    customer_counts = all_orders['customer_id'].value_counts()
    eligible = customer_counts[(customer_counts >= 10) & (customer_counts <= 50)]
    return eligible.index[0]


@pytest.fixture
def sample_customer_orders(all_orders, all_vendors, all_products, sample_customer_id):
    """Get merged orders for a specific test customer."""
    # First filter to customer's orders
    customer_orders = all_orders[all_orders['customer_id'] == sample_customer_id].copy()

    # Merge with vendors to get cuisine
    merged = customer_orders.merge(
        all_vendors[['vendor_id', 'primary_cuisine', 'geohash', 'chain_id']],
        on='vendor_id',
        how='left',
        suffixes=('', '_vendor')
    )

    # Merge with products to get unit_price
    merged = merged.merge(
        all_products[['vendor_id', 'product_id', 'unit_price']],
        on=['vendor_id', 'product_id'],
        how='left'
    )

    merged = merged.rename(columns={
        'geohash': 'user_geohash',
        'geohash_vendor': 'vendor_geohash',
        'primary_cuisine': 'cuisine'
    })

    # Parse hour from order_time
    if 'order_time' in merged.columns:
        merged['hour'] = merged['order_time'].str.split(':').str[0].astype(int)

    # Parse day_num from order_day
    if 'order_day' in merged.columns:
        merged['day_num'] = merged['order_day'].str.extract(r'(\d+)')[0].astype(int)

    return merged


@pytest.fixture
def merged_orders(sample_orders, all_vendors, all_products):
    """Merged orders with vendor and product info."""
    merged = sample_orders.merge(
        all_vendors[['vendor_id', 'primary_cuisine', 'geohash', 'chain_id']],
        on='vendor_id',
        how='left',
        suffixes=('', '_vendor')
    )
    merged = merged.merge(
        all_products[['vendor_id', 'product_id', 'name', 'unit_price']],
        on=['vendor_id', 'product_id'],
        how='left'
    )

    # Rename columns for consistency
    merged = merged.rename(columns={
        'geohash': 'user_geohash',
        'geohash_vendor': 'vendor_geohash',
        'primary_cuisine': 'cuisine'
    })

    # Parse hour from order_time
    if 'order_time' in merged.columns:
        merged['hour'] = merged['order_time'].str.split(':').str[0].astype(int)

    # Parse day_num from order_day
    if 'order_day' in merged.columns:
        merged['day_num'] = merged['order_day'].str.extract(r'(\d+)')[0].astype(int)

    return merged


@pytest.fixture
def unique_cuisines(all_vendors):
    """Get list of unique cuisines."""
    return all_vendors['primary_cuisine'].dropna().unique().tolist()
