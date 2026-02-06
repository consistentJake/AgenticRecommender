"""VendorProfilerAgent — wraps GeohashVendorIndex for candidate filtering."""

import logging
from typing import Dict, Any, List

from .base import RecommendationAgent


class VendorProfilerAgent(RecommendationAgent):
    """Wraps GeohashVendorIndex for geohash-based candidate vendor selection."""

    def __init__(self, name: str = "vendor_profiler"):
        super().__init__(name)
        self.geohash_index = None

    def initialize(self, train_df, no_cache: bool = False) -> None:
        """Build GeohashVendorIndex from training data.

        Extracts from workflow_runner.py Stage 9 Step 2.
        """
        from ..data.geohash_index import GeohashVendorIndex

        self.geohash_index = GeohashVendorIndex()
        self.geohash_index.build(train_df, use_cache=not no_cache)
        self._initialized = True
        self._logger.info("Geohash index stats: %s", self.geohash_index.get_stats())

    def process(
        self,
        target_geohash: str,
        predicted_cuisines: List[str],
        max_candidates: int = 20,
    ) -> List[str]:
        """Get candidate vendors by geohash + cuisine filter.

        Returns vendor IDs formatted as "vendor_id||cuisine" strings.
        Extracts from repeat_evaluator.py lines 553-567.
        """
        candidate_vendor_ids = self.geohash_index.get_vendors(
            target_geohash,
            predicted_cuisines,
            max_candidates=max_candidates,
        )

        candidate_vendors = []
        for vid in candidate_vendor_ids:
            cuisine = self.geohash_index.get_vendor_cuisine(vid)
            candidate_vendors.append(f"{vid}||{cuisine}" if cuisine else vid)

        return candidate_vendors

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        if self.geohash_index:
            stats.update(self.geohash_index.get_stats())
        return stats
