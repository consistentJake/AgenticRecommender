"""
Enriched data loader for Singapore food delivery dataset.

Preserves ALL information from original dataset:
- customer_id, order_id (for CF and co-purchase patterns)
- geohash (user and vendor locations)
- chain_id (brand preferences)
- temporal patterns (hour, weekday)
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: Path
    orders_file: str = ""
    orders_test_file: str = ""
    vendors_file: str = ""
    products_file: str = ""

    # Processing options
    parse_time: bool = True
    compute_day_name: bool = True

    def __post_init__(self):
        """Auto-detect file names from directory if not explicitly set."""
        if not self.orders_file or not self.vendors_file or not self.products_file:
            self._auto_detect_files()

    def _auto_detect_files(self):
        """Detect file names by looking for orders_*_train.txt pattern."""
        import glob
        data_dir = Path(self.data_dir)

        # Find orders train file: orders_*_train.txt
        train_files = list(data_dir.glob("orders_*_train.txt"))
        if train_files:
            self.orders_file = train_files[0].name
            # Derive region suffix (e.g., "sg" from "orders_sg_train.txt")
            suffix = self.orders_file.replace("orders_", "").replace("_train.txt", "")
            if not self.orders_test_file:
                self.orders_test_file = f"orders_{suffix}_test.txt"
            if not self.vendors_file:
                self.vendors_file = f"vendors_{suffix}.txt"
            if not self.products_file:
                self.products_file = f"products_{suffix}.txt"
        else:
            # Fallback to original defaults
            self.orders_file = self.orders_file or "orders_se_train.txt"
            self.orders_test_file = self.orders_test_file or "orders_se_test.txt"
            self.vendors_file = self.vendors_file or "vendors_se.txt"
            self.products_file = self.products_file or "products_se.txt"


class EnrichedDataLoader:
    """
    Load and merge Singapore food delivery data with all fields preserved.

    Key preserved fields:
    - customer_id: User identity for CF
    - order_id: Order grouping for co-purchase patterns
    - geohash (user): User location
    - geohash (vendor): Restaurant location
    - chain_id: Chain restaurant grouping
    - vendor_id: Specific restaurant
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, config: DataConfig):
        self.config = config
        self._orders: Optional[pd.DataFrame] = None
        self._orders_test: Optional[pd.DataFrame] = None
        self._vendors: Optional[pd.DataFrame] = None
        self._products: Optional[pd.DataFrame] = None
        self._merged: Optional[pd.DataFrame] = None
        self._merged_test: Optional[pd.DataFrame] = None

    def load_orders(self) -> pd.DataFrame:
        """Load orders data."""
        if self._orders is None:
            path = self.config.data_dir / self.config.orders_file
            self._orders = pd.read_csv(path)

            if self.config.parse_time:
                self._orders = self._parse_time_columns(self._orders)

            if self.config.compute_day_name:
                self._orders['day_name'] = self._orders['day_of_week'].apply(
                    lambda x: self.DAY_NAMES[x] if 0 <= x <= 6 else 'Unknown'
                )

        return self._orders

    def load_vendors(self) -> pd.DataFrame:
        """Load vendors data."""
        if self._vendors is None:
            path = self.config.data_dir / self.config.vendors_file
            self._vendors = pd.read_csv(path)
            # Fill missing chain_id with empty string
            self._vendors['chain_id'] = self._vendors['chain_id'].fillna('')

        return self._vendors

    def load_products(self) -> pd.DataFrame:
        """Load products data."""
        if self._products is None:
            path = self.config.data_dir / self.config.products_file
            self._products = pd.read_csv(path)
            self._products['unit_price'] = self._products['unit_price'].fillna(0.0)
            self._products['name'] = self._products['name'].fillna('Unknown Product')

        return self._products

    def load_merged(self) -> pd.DataFrame:
        """
        Load and merge all data with preserved fields.

        Output columns:
        - customer_id, order_id, vendor_id, product_id
        - user_geohash, vendor_geohash
        - chain_id, cuisine
        - day_of_week, day_name, hour, order_day
        - product_name, unit_price
        """
        if self._merged is not None:
            return self._merged

        orders = self.load_orders()
        vendors = self.load_vendors()
        products = self.load_products()

        # Merge orders with vendors
        merged = orders.merge(
            vendors[['vendor_id', 'primary_cuisine', 'geohash', 'chain_id']],
            on='vendor_id',
            how='left',
            suffixes=('', '_vendor')
        )

        # Rename geohash columns for clarity
        merged = merged.rename(columns={
            'geohash': 'user_geohash',
            'geohash_vendor': 'vendor_geohash',
            'primary_cuisine': 'cuisine'
        })

        # Merge with products
        merged = merged.merge(
            products[['vendor_id', 'product_id', 'name', 'unit_price']],
            on=['vendor_id', 'product_id'],
            how='left'
        )
        merged = merged.rename(columns={'name': 'product_name'})

        # Fill missing values
        merged['cuisine'] = merged['cuisine'].fillna('unknown')
        merged['vendor_geohash'] = merged['vendor_geohash'].fillna('unknown')
        merged['chain_id'] = merged['chain_id'].fillna('')
        merged['product_name'] = merged['product_name'].fillna('Unknown Product')
        merged['unit_price'] = merged['unit_price'].fillna(0.0)

        self._merged = merged
        return self._merged

    def load_test_orders(self) -> pd.DataFrame:
        """Load test orders data."""
        if self._orders_test is None:
            path = self.config.data_dir / self.config.orders_test_file
            if not path.exists():
                raise FileNotFoundError(f"Test orders file not found: {path}")

            self._orders_test = pd.read_csv(path)

            if self.config.parse_time:
                self._orders_test = self._parse_time_columns(self._orders_test)

            if self.config.compute_day_name:
                self._orders_test['day_name'] = self._orders_test['day_of_week'].apply(
                    lambda x: self.DAY_NAMES[x] if 0 <= x <= 6 else 'Unknown'
                )

        return self._orders_test

    def load_merged_test(self) -> pd.DataFrame:
        """
        Load and merge test data with vendor/product info.

        Uses same merge logic as load_merged() but with test orders.
        """
        if self._merged_test is not None:
            return self._merged_test

        orders = self.load_test_orders()
        vendors = self.load_vendors()
        products = self.load_products()

        # Merge orders with vendors
        merged = orders.merge(
            vendors[['vendor_id', 'primary_cuisine', 'geohash', 'chain_id']],
            on='vendor_id',
            how='left',
            suffixes=('', '_vendor')
        )

        # Rename geohash columns for clarity
        merged = merged.rename(columns={
            'geohash': 'user_geohash',
            'geohash_vendor': 'vendor_geohash',
            'primary_cuisine': 'cuisine'
        })

        # Merge with products
        merged = merged.merge(
            products[['vendor_id', 'product_id', 'name', 'unit_price']],
            on=['vendor_id', 'product_id'],
            how='left'
        )
        merged = merged.rename(columns={'name': 'product_name'})

        # Fill missing values
        merged['cuisine'] = merged['cuisine'].fillna('unknown')
        merged['vendor_geohash'] = merged['vendor_geohash'].fillna('unknown')
        merged['chain_id'] = merged['chain_id'].fillna('')
        merged['product_name'] = merged['product_name'].fillna('Unknown Product')
        merged['unit_price'] = merged['unit_price'].fillna(0.0)

        self._merged_test = merged
        return self._merged_test

    def _parse_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse time-related columns."""
        df = df.copy()

        # Extract hour from order_time (format: "HH:MM:SS")
        if 'order_time' in df.columns:
            df['hour'] = df['order_time'].str.split(':').str[0].astype(int)

        # Extract day number from order_day (format: "N days")
        if 'order_day' in df.columns:
            df['day_num'] = df['order_day'].str.extract(r'(\d+)')[0].astype(int)

        return df

    def get_customer_orders(self, customer_id: str) -> pd.DataFrame:
        """Get all orders for a specific customer."""
        merged = self.load_merged()
        return merged[merged['customer_id'] == customer_id].copy()

    def get_order_items(self, order_id: str) -> pd.DataFrame:
        """Get all items in a specific order."""
        merged = self.load_merged()
        return merged[merged['order_id'] == order_id].copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        merged = self.load_merged()

        return {
            'total_rows': len(merged),
            'unique_customers': merged['customer_id'].nunique(),
            'unique_orders': merged['order_id'].nunique(),
            'unique_vendors': merged['vendor_id'].nunique(),
            'unique_cuisines': merged['cuisine'].nunique(),
            'unique_products': merged['product_id'].nunique(),
            'avg_items_per_order': len(merged) / merged['order_id'].nunique(),
        }


def load_singapore_data(
    data_dir: str = "/Users/zhenkai/Downloads/data_se"
) -> EnrichedDataLoader:
    """
    Convenience function to load Singapore dataset.

    Args:
        data_dir: Path to data directory

    Returns:
        EnrichedDataLoader instance
    """
    config = DataConfig(data_dir=Path(data_dir))
    return EnrichedDataLoader(config)
