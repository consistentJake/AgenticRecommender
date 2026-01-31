"""
Repeated Dataset Filter for repeat-order evaluation.

Filters training and test data to only include orders where the user
re-orders from a vendor they have ordered from before (in training).
"""

import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import pandas as pd


class RepeatDatasetFilter:
    """Filter training/test data for repeated order evaluation."""

    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "repeat_filter"

    def filter(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        min_history_items: int = 5,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Filter datasets for repeat-order evaluation.

        1. Group training orders by customer_id (each unique order_id = 1 datapoint)
        2. Keep users with >= min_history_items unique orders
        3. For each user, collect set of vendor_ids from training
        4. Filter test orders: only keep test orders whose vendor_id is in user's training vendor set
        5. Return filtered DataFrames + stats dict

        Args:
            train_df: Training data DataFrame
            test_df: Test data DataFrame
            min_history_items: Minimum unique orders per user in training
            use_cache: Whether to use disk cache

        Returns:
            Tuple of (filtered_train_df, filtered_test_df, stats_dict)
        """
        cache_key = self._cache_key(train_df, test_df, min_history_items)

        if use_cache:
            cached = self.load_cache(cache_key)
            if cached is not None:
                print(f"[RepeatDatasetFilter] Loaded from cache (key={cache_key[:8]}...)")
                return cached

        print(f"[RepeatDatasetFilter] Filtering datasets...")
        print(f"  Training rows: {len(train_df):,}")
        print(f"  Test rows: {len(test_df):,}")

        # Step 1-2: Get users with sufficient training history
        train_order_counts = train_df.groupby('customer_id')['order_id'].nunique()
        valid_users = set(train_order_counts[train_order_counts >= min_history_items].index)
        print(f"  Users with >= {min_history_items} training orders: {len(valid_users):,}")

        # Filter training data to valid users
        filtered_train = train_df[train_df['customer_id'].isin(valid_users)].copy()

        # Step 3: Build vendor set per user from training
        user_train_vendors: Dict[Any, Set[str]] = {}
        for customer_id in valid_users:
            customer_train = filtered_train[filtered_train['customer_id'] == customer_id]
            user_train_vendors[customer_id] = set(customer_train['vendor_id'].unique().astype(str))

        # Step 4: Filter test orders - only keep orders where vendor is in user's training vendors
        test_users = set(test_df['customer_id'].unique())
        overlap_users = valid_users & test_users

        keep_mask = pd.Series(False, index=test_df.index)
        for idx, row in test_df.iterrows():
            customer_id = row['customer_id']
            vendor_id = str(row['vendor_id'])
            if customer_id in user_train_vendors and vendor_id in user_train_vendors[customer_id]:
                keep_mask.at[idx] = True

        filtered_test = test_df[keep_mask].copy()

        # Compute stats
        test_users_after = set(filtered_test['customer_id'].unique())
        stats = {
            'train_rows_before': len(train_df),
            'train_rows_after': len(filtered_train),
            'test_rows_before': len(test_df),
            'test_rows_after': len(filtered_test),
            'users_with_min_history': len(valid_users),
            'users_in_both_before': len(test_users & set(train_df['customer_id'].unique())),
            'users_in_both_after': len(test_users_after),
            'test_orders_after': filtered_test['order_id'].nunique(),
            'min_history_items': min_history_items,
            'repeat_rate': len(filtered_test) / len(test_df) if len(test_df) > 0 else 0,
        }

        print(f"  Filtered test rows: {stats['test_rows_after']:,} "
              f"({stats['repeat_rate']:.1%} of original)")
        print(f"  Users in filtered test: {stats['users_in_both_after']:,}")
        print(f"  Test orders (unique): {stats['test_orders_after']:,}")

        if use_cache:
            self.save_cache(cache_key, filtered_train, filtered_test, stats)

        return filtered_train, filtered_test, stats

    def _cache_key(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        min_history_items: int,
    ) -> str:
        """Compute cache key from data shape and params."""
        key_str = f"{len(train_df)}_{len(test_df)}_{min_history_items}"
        key_str += f"_{train_df['customer_id'].nunique()}_{test_df['customer_id'].nunique()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def save_cache(
        self,
        cache_key: str,
        filtered_train: pd.DataFrame,
        filtered_test: pd.DataFrame,
        stats: Dict,
    ) -> bool:
        """Save filtered data to cache."""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = self.CACHE_DIR / f"{cache_key}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'filtered_train': filtered_train,
                    'filtered_test': filtered_test,
                    'stats': stats,
                }, f)
            print(f"[RepeatDatasetFilter] Saved to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[RepeatDatasetFilter] Failed to save cache: {e}")
            return False

    def load_cache(
        self, cache_key: str
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
        """Load filtered data from cache."""
        try:
            cache_path = self.CACHE_DIR / f"{cache_key}.pkl"
            if not cache_path.exists():
                return None
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data['filtered_train'], data['filtered_test'], data['stats']
        except Exception as e:
            print(f"[RepeatDatasetFilter] Failed to load cache: {e}")
            return None


def build_repeat_test_samples(
    filtered_train_df: pd.DataFrame,
    filtered_test_df: pd.DataFrame,
    n_samples: int = -1,
    deterministic: bool = True,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build test samples from filtered repeat dataset.

    Each test sample represents one test order where the user has
    previously ordered from the same vendor in training.

    Args:
        filtered_train_df: Training data (already filtered for min history)
        filtered_test_df: Test data (already filtered for repeat vendors)
        n_samples: Number of samples to return (-1 = all)
        deterministic: Sort by order_id for reproducibility
        seed: Random seed (used only if deterministic=False)

    Returns:
        List of test sample dicts
    """
    random.seed(seed)

    # Build training history per user
    user_histories: Dict[Any, List[Dict]] = {}
    valid_users = set(filtered_test_df['customer_id'].unique())

    for customer_id in valid_users:
        customer_train = filtered_train_df[filtered_train_df['customer_id'] == customer_id]

        # Sort by time
        if 'day_num' in customer_train.columns:
            customer_train = customer_train.sort_values(['day_num', 'hour'])

        orders = []
        seen_orders = set()

        for _, row in customer_train.iterrows():
            order_id = row['order_id']
            if order_id in seen_orders:
                continue
            seen_orders.add(order_id)

            vendor_id = str(row.get('vendor_id', ''))
            cuisine = row.get('cuisine', 'unknown')
            orders.append({
                'order_id': order_id,
                'vendor_id': vendor_id,
                'cuisine': cuisine,
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
                'vendor_geohash': str(row.get('vendor_geohash', 'unknown')),
                'item': f"{vendor_id}||{cuisine}",
            })

        user_histories[customer_id] = orders

    # Build test samples from test orders
    samples = []
    test_order_ids = filtered_test_df['order_id'].unique()

    for test_order_id in test_order_ids:
        order_data = filtered_test_df[filtered_test_df['order_id'] == test_order_id]
        customer_id = order_data['customer_id'].iloc[0]

        if customer_id not in user_histories:
            continue

        first_row = order_data.iloc[0]
        vendor_id = str(first_row.get('vendor_id', ''))
        cuisine = first_row.get('cuisine', 'unknown')
        ground_truth = f"{vendor_id}||{cuisine}"

        sample = {
            'customer_id': customer_id,
            'order_history': user_histories[customer_id],
            'ground_truth': ground_truth,
            'ground_truth_vendor_id': vendor_id,
            'ground_truth_cuisine': cuisine,
            'target_hour': int(first_row.get('hour', 12)),
            'target_day_of_week': int(first_row.get('day_of_week', 0)),
            'target_vendor_geohash': str(first_row.get('vendor_geohash', 'unknown')),
            'order_id': test_order_id,
        }

        samples.append(sample)

    # Sort or shuffle
    if deterministic:
        samples.sort(key=lambda x: str(x['order_id']))
    else:
        random.shuffle(samples)

    # Limit samples
    total_available = len(samples)
    if 0 < n_samples < total_available:
        samples = samples[:n_samples]

    mode = "deterministic" if deterministic else "random"
    print(f"[build_repeat_test_samples] Created {len(samples)} test samples "
          f"(from {total_available} available, {mode})")

    return samples
