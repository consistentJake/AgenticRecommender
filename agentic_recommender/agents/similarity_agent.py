"""SimilarityAgent — wraps LightGCN embedding manager for cuisine similarity."""

import logging
from typing import Dict, Any, List, Tuple

from .base import RecommendationAgent


class SimilarityAgent(RecommendationAgent):
    """Wraps LightGCN for customer-to-cuisine similarity scoring."""

    def __init__(self, name: str = "similarity"):
        super().__init__(name)
        self.lightgcn = None

    def initialize(
        self,
        train_df,
        lightgcn_config,
        dataset_name: str,
        no_cache: bool = False,
    ) -> None:
        """Train or load LightGCN on customer->cuisine interactions.

        Extracts from workflow_runner.py Stage 9 Step 3.
        """
        from ..similarity.lightGCN import LightGCNEmbeddingManager

        # Build cuisine-level interactions (deduplicated customer->cuisine pairs)
        cuisine_interactions = []
        seen_pairs = set()
        for _, row in train_df.iterrows():
            cid = str(row['customer_id'])
            cuisine = str(row.get('cuisine', 'unknown'))
            pair = (cid, cuisine)
            if pair not in seen_pairs:
                cuisine_interactions.append(pair)
                seen_pairs.add(pair)

        self._logger.info("Cuisine interactions: %d", len(cuisine_interactions))

        lightgcn_manager = LightGCNEmbeddingManager(lightgcn_config)
        lightgcn_manager.load_or_train(
            dataset_name=dataset_name,
            interactions=cuisine_interactions,
            method="repeat",
            prediction_target="cuisine",
            force_retrain=no_cache,
            verbose=True,
        )
        self.lightgcn = lightgcn_manager
        self._initialized = True

    def process(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return LightGCN top cuisines for a user."""
        return self.lightgcn.get_top_cuisines_for_user(user_id, top_k=top_k)

    @property
    def cache_key(self):
        return self.lightgcn.cache_key if self.lightgcn else None

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        if self.lightgcn:
            stats.update(self.lightgcn.get_stats())
        return stats
