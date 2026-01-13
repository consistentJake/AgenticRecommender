"""
Concrete similarity method implementations.

Available methods:
- SwingMethod: Alibaba's Swing algorithm (anti-noise)
- CosineMethod: Vector-based cosine similarity
- JaccardMethod: Set overlap similarity
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import math

from .base import SimilarityMethod, SimilarityConfig


@dataclass
class SwingConfig(SimilarityConfig):
    """Swing-specific configuration."""
    alpha1: float = 5.0   # User activity smoothing
    alpha2: float = 1.0   # Item popularity smoothing
    beta: float = 0.3     # Power weight for user activity


class SwingMethod(SimilarityMethod):
    """
    Alibaba's Swing algorithm for user-user similarity.

    Anti-noise property: Penalizes popular items and active users.

    Formula:
    sim(u1, u2) = Σ(i ∈ common_items) 1 / ((|I(u1)|+α1)^β × (|I(u2)|+α1)^β × (|U(i)|+α2))
    """

    def __init__(self, config: SwingConfig = None):
        super().__init__(config or SwingConfig())
        self.user_items: Dict[str, Set[str]] = {}
        self.item_users: Dict[str, Set[str]] = {}

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'SwingMethod':
        """Build user-item and item-user indices."""
        self.user_items.clear()
        self.item_users.clear()
        self.clear_cache()

        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

            if item_id not in self.item_users:
                self.item_users[item_id] = set()
            self.item_users[item_id].add(user_id)

        self._fitted = True
        return self

    def compute_similarity(self, user1: str, user2: str) -> float:
        """Compute Swing similarity between two users."""
        # Check cache
        cache_key = (min(user1, user2), max(user1, user2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())
        common = items1 & items2

        if not common:
            return 0.0

        cfg = self.config
        similarity = 0.0

        for item in common:
            item_pop = len(self.item_users.get(item, set()))
            weight = 1.0 / (
                ((len(items1) + cfg.alpha1) ** cfg.beta) *
                ((len(items2) + cfg.alpha1) ** cfg.beta) *
                (item_pop + cfg.alpha2)
            )
            similarity += weight

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        return list(self.user_items.keys())

    def get_method_name(self) -> str:
        return "swing"

    def get_user_items(self, user_id: str) -> Set[str]:
        """Get items for a user."""
        return self.user_items.get(user_id, set()).copy()

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats.update({
            'num_users': len(self.user_items),
            'num_items': len(self.item_users),
            'total_interactions': sum(len(items) for items in self.user_items.values()),
        })
        return stats


@dataclass
class CosineConfig(SimilarityConfig):
    """Cosine similarity configuration."""
    normalize: bool = True


class CosineMethod(SimilarityMethod):
    """
    Cosine similarity based on user-item vectors.

    sim(u1, u2) = (v1 · v2) / (||v1|| × ||v2||)
    """

    def __init__(self, config: CosineConfig = None):
        super().__init__(config or CosineConfig())
        self.user_vectors: Dict[str, Dict[str, float]] = {}
        self.all_items: Set[str] = set()

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'CosineMethod':
        """Build user vectors from interactions."""
        self.user_vectors.clear()
        self.all_items.clear()
        self.clear_cache()

        # Count interactions per user-item pair
        for user_id, item_id in interactions:
            if user_id not in self.user_vectors:
                self.user_vectors[user_id] = {}
            self.user_vectors[user_id][item_id] = \
                self.user_vectors[user_id].get(item_id, 0) + 1
            self.all_items.add(item_id)

        # Normalize if configured
        if self.config.normalize:
            for user_id in self.user_vectors:
                total = sum(self.user_vectors[user_id].values())
                if total > 0:
                    for item_id in self.user_vectors[user_id]:
                        self.user_vectors[user_id][item_id] /= total

        self._fitted = True
        return self

    def compute_similarity(self, user1: str, user2: str) -> float:
        """Compute cosine similarity."""
        cache_key = (min(user1, user2), max(user1, user2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        v1 = self.user_vectors.get(user1, {})
        v2 = self.user_vectors.get(user2, {})

        if not v1 or not v2:
            return 0.0

        # Compute dot product
        common_items = set(v1.keys()) & set(v2.keys())
        dot_product = sum(v1[i] * v2[i] for i in common_items)

        # Compute norms
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        return list(self.user_vectors.keys())

    def get_method_name(self) -> str:
        return "cosine"

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats.update({
            'num_users': len(self.user_vectors),
            'num_items': len(self.all_items),
        })
        return stats


@dataclass
class JaccardConfig(SimilarityConfig):
    """Jaccard similarity configuration."""
    pass


class JaccardMethod(SimilarityMethod):
    """
    Jaccard similarity based on item set overlap.

    sim(u1, u2) = |I(u1) ∩ I(u2)| / |I(u1) ∪ I(u2)|
    """

    def __init__(self, config: JaccardConfig = None):
        super().__init__(config or JaccardConfig())
        self.user_items: Dict[str, Set[str]] = {}

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'JaccardMethod':
        """Build user-item sets."""
        self.user_items.clear()
        self.clear_cache()

        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

        self._fitted = True
        return self

    def compute_similarity(self, user1: str, user2: str) -> float:
        """Compute Jaccard similarity."""
        cache_key = (min(user1, user2), max(user1, user2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())

        if not items1 or not items2:
            return 0.0

        intersection = len(items1 & items2)
        union = len(items1 | items2)

        similarity = intersection / union if union > 0 else 0.0

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        return list(self.user_items.keys())

    def get_method_name(self) -> str:
        return "jaccard"

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats.update({
            'num_users': len(self.user_items),
        })
        return stats
