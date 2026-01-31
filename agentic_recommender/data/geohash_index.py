"""
Geohash-based vendor index for candidate generation.

Pre-computes mapping: vendor_geohash → primary_cuisine → [vendor_ids]
Used in repeat evaluation to find candidate vendors by location and cuisine.
"""

import hashlib
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import pandas as pd


class GeohashVendorIndex:
    """Pre-computed index: vendor_geohash → primary_cuisine → [vendor_ids]."""

    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "geohash_index"

    def __init__(self):
        # geohash -> cuisine -> [vendor_ids]
        self.geohash_cuisine_vendors: Dict[str, Dict[str, List[str]]] = {}
        # vendor_id -> (vendor_geohash, cuisine)
        self.vendor_metadata: Dict[str, tuple] = {}
        self._built = False

    def build(self, train_df: pd.DataFrame, use_cache: bool = True) -> 'GeohashVendorIndex':
        """
        Build geohash→cuisine→vendors index from training data.

        Extracts unique (vendor_id, vendor_geohash, cuisine) tuples
        and builds lookup structures.

        Args:
            train_df: Training data DataFrame
            use_cache: Whether to use disk cache

        Returns:
            self (for chaining)
        """
        cache_key = self._cache_key(train_df)

        if use_cache:
            if self._load_cache(cache_key):
                return self

        print(f"[GeohashVendorIndex] Building index from {len(train_df):,} rows...")

        # Extract unique vendor tuples
        vendor_info = train_df[['vendor_id', 'vendor_geohash', 'cuisine']].drop_duplicates()

        self.geohash_cuisine_vendors.clear()
        self.vendor_metadata.clear()

        for _, row in vendor_info.iterrows():
            vendor_id = str(row['vendor_id'])
            geohash = str(row.get('vendor_geohash', 'unknown'))
            cuisine = str(row.get('cuisine', 'unknown'))

            # Build geohash -> cuisine -> vendors
            if geohash not in self.geohash_cuisine_vendors:
                self.geohash_cuisine_vendors[geohash] = {}
            if cuisine not in self.geohash_cuisine_vendors[geohash]:
                self.geohash_cuisine_vendors[geohash][cuisine] = []
            if vendor_id not in self.geohash_cuisine_vendors[geohash][cuisine]:
                self.geohash_cuisine_vendors[geohash][cuisine].append(vendor_id)

            # Build vendor metadata
            self.vendor_metadata[vendor_id] = (geohash, cuisine)

        self._built = True

        n_geohashes = len(self.geohash_cuisine_vendors)
        n_vendors = len(self.vendor_metadata)
        print(f"[GeohashVendorIndex] Built: {n_geohashes} geohashes, {n_vendors} vendors")

        if use_cache:
            self._save_cache(cache_key)

        return self

    def get_vendors(
        self,
        vendor_geohash: str,
        cuisines: List[str],
        max_candidates: int = 20,
    ) -> List[str]:
        """
        Look up vendors by geohash and cuisine filter.

        Args:
            vendor_geohash: Geohash to look up
            cuisines: List of cuisines to filter by
            max_candidates: Maximum vendor_ids to return

        Returns:
            List of vendor_ids matching the geohash and cuisines
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        geohash_data = self.geohash_cuisine_vendors.get(vendor_geohash, {})
        if not geohash_data:
            return []

        candidates = []
        seen = set()

        # Iterate through requested cuisines in order (preserves priority)
        for cuisine in cuisines:
            vendors = geohash_data.get(cuisine, [])
            for vid in vendors:
                if vid not in seen:
                    candidates.append(vid)
                    seen.add(vid)
                    if len(candidates) >= max_candidates:
                        return candidates

        return candidates

    def get_vendor_cuisine(self, vendor_id: str) -> Optional[str]:
        """Get the cuisine for a vendor_id."""
        meta = self.vendor_metadata.get(vendor_id)
        return meta[1] if meta else None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_vendors_in_index = sum(
            len(vendors)
            for cuisines in self.geohash_cuisine_vendors.values()
            for vendors in cuisines.values()
        )
        return {
            'built': self._built,
            'n_geohashes': len(self.geohash_cuisine_vendors),
            'n_vendors': len(self.vendor_metadata),
            'total_vendor_entries': total_vendors_in_index,
        }

    def _cache_key(self, train_df: pd.DataFrame) -> str:
        """Compute cache key from data shape."""
        key_str = (
            f"{len(train_df)}_{train_df['vendor_id'].nunique()}_"
            f"{train_df['customer_id'].nunique()}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()

    def _save_cache(self, cache_key: str) -> bool:
        """Save index to disk cache."""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = self.CACHE_DIR / f"{cache_key}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'geohash_cuisine_vendors': self.geohash_cuisine_vendors,
                    'vendor_metadata': self.vendor_metadata,
                }, f)
            print(f"[GeohashVendorIndex] Saved to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[GeohashVendorIndex] Failed to save cache: {e}")
            return False

    def _load_cache(self, cache_key: str) -> bool:
        """Load index from disk cache."""
        try:
            cache_path = self.CACHE_DIR / f"{cache_key}.pkl"
            if not cache_path.exists():
                return False
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.geohash_cuisine_vendors = data['geohash_cuisine_vendors']
            self.vendor_metadata = data['vendor_metadata']
            self._built = True
            print(f"[GeohashVendorIndex] Loaded from cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[GeohashVendorIndex] Failed to load cache: {e}")
            return False
