"""
Concrete similarity method implementations.

Available methods:
- SwingMethod: Alibaba's Swing algorithm (anti-noise)
- CosineMethod: Vector-based cosine similarity
- JaccardMethod: Set overlap similarity
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
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

    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "swing_user"

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

    def get_top_similar_users(
        self, user_id: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k most similar users via Swing.

        Args:
            user_id: Query user
            top_k: Number of similar users to return

        Returns:
            List of (user_id, similarity_score) tuples, sorted descending
        """
        old_top_k = self.config.top_k
        self.config.top_k = top_k
        result = self.get_similar(user_id)
        self.config.top_k = old_top_k
        return result

    def save_to_cache(self, dataset_name: str, method: str = "repeat") -> bool:
        """Save fitted model to disk cache.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            method: Evaluation method for cache naming

        Returns:
            True if save succeeded
        """
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = self._get_cache_path(dataset_name, method)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'user_items': self.user_items,
                    'item_users': self.item_users,
                    'config': self.config,
                }, f)
            print(f"[SwingMethod] Saved to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[SwingMethod] Failed to save cache: {e}")
            return False

    def load_from_cache(self, dataset_name: str, method: str = "repeat") -> bool:
        """Load fitted model from disk cache.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            method: Evaluation method for cache naming

        Returns:
            True if cache load succeeded
        """
        try:
            cache_path = self._get_cache_path(dataset_name, method)
            if not cache_path.exists():
                return False
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.user_items = data['user_items']
            self.item_users = data['item_users']
            self._fitted = True
            print(f"[SwingMethod] Loaded from cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[SwingMethod] Failed to load cache: {e}")
            return False

    def _get_cache_path(self, dataset_name: str, method: str = "repeat") -> Path:
        """Get cache file path."""
        if method:
            return self.CACHE_DIR / f"{dataset_name}_{method}_swing_user.pkl"
        return self.CACHE_DIR / f"{dataset_name}_swing_user.pkl"

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


# =============================================================================
# CUISINE-CUISINE SWING SIMILARITY
# =============================================================================

@dataclass
class CuisineSwingConfig(SimilarityConfig):
    """Configuration for Cuisine-Cuisine Swing similarity."""
    alpha1: float = 5.0   # Cuisine popularity smoothing
    alpha2: float = 1.0   # User activity smoothing
    beta: float = 0.3     # Power weight


class CuisineSwingMethod(SimilarityMethod):
    """
    Cuisine-to-Cuisine Swing similarity (also supports vendor_cuisine items).

    Operates at ITEM level (cuisine or vendor_cuisine).

    Formula:
    sim(i1, i2) = Σ(u ∈ common_users) 1 / ((|U(i1)|+α1)^β × (|U(i2)|+α1)^β × (|I(u)|+α2))

    Where:
    - U(i) = users who ordered item i
    - I(u) = items ordered by user u
    """

    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "swing"

    def __init__(self, config: CuisineSwingConfig = None, prediction_target: str = "cuisine"):
        super().__init__(config or CuisineSwingConfig())
        self.prediction_target = prediction_target
        self.cuisine_users: Dict[str, Set[str]] = {}  # item -> set of user_ids
        self.user_cuisines: Dict[str, Set[str]] = {}  # user_id -> set of items

    def save_to_cache(self, dataset_name: str, method: str = "full") -> bool:
        """
        Save fitted model to cache.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            method: Evaluation method ("method1", "method2", or "full")

        Returns:
            True if save succeeded

        Cache naming:
            - {dataset}_{method}_{target}_swing.pkl (with method and non-cuisine target)
            - {dataset}_{method}_swing.pkl (with method, cuisine target)
            - {dataset}_{target}_swing.pkl (no method, non-cuisine target)
            - {dataset}_swing.pkl (no method, cuisine target - legacy)
        """
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = self._get_cache_path(dataset_name, method)

            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'cuisine_users': self.cuisine_users,
                    'user_cuisines': self.user_cuisines,
                    'config': self.config,
                    'prediction_target': self.prediction_target,
                }, f)
            print(f"[CuisineSwingMethod] Saved to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[CuisineSwingMethod] Failed to save cache: {e}")
            return False

    def _get_cache_path(self, dataset_name: str, method: str = "full") -> Path:
        """
        Get cache file path for dataset, method, and prediction target.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            method: Evaluation method ("method1", "method2", or "full")

        Returns:
            Path to cache file
        """
        # Build cache filename with prediction_target for non-default targets
        if method and method != "full":
            if self.prediction_target and self.prediction_target != "cuisine":
                return self.CACHE_DIR / f"{dataset_name}_{method}_{self.prediction_target}_swing.pkl"
            return self.CACHE_DIR / f"{dataset_name}_{method}_swing.pkl"
        else:
            if self.prediction_target and self.prediction_target != "cuisine":
                return self.CACHE_DIR / f"{dataset_name}_{self.prediction_target}_swing.pkl"
            return self.CACHE_DIR / f"{dataset_name}_swing.pkl"

    def load_from_cache(self, dataset_name: str, method: str = "full") -> bool:
        """
        Load fitted model from cache.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            method: Evaluation method ("method1", "method2", or "full")

        Returns:
            True if cache load succeeded
        """
        try:
            cache_path = self._get_cache_path(dataset_name, method)

            if not cache_path.exists():
                return False

            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            self.cuisine_users = data['cuisine_users']
            self.user_cuisines = data['user_cuisines']
            # Load prediction_target if saved (backward compat)
            if 'prediction_target' in data:
                self.prediction_target = data['prediction_target']
            self._fitted = True
            print(f"[CuisineSwingMethod] Loaded from cache: {cache_path}")
            return True
        except Exception as e:
            print(f"[CuisineSwingMethod] Failed to load cache: {e}")
            return False

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'CuisineSwingMethod':
        """
        Build indices from user-cuisine interactions.

        Args:
            interactions: List of (user_id, cuisine) tuples
        """
        self.cuisine_users.clear()
        self.user_cuisines.clear()
        self.clear_cache()

        for user_id, cuisine in interactions:
            # Build cuisine -> users index
            if cuisine not in self.cuisine_users:
                self.cuisine_users[cuisine] = set()
            self.cuisine_users[cuisine].add(user_id)

            # Build user -> cuisines index
            if user_id not in self.user_cuisines:
                self.user_cuisines[user_id] = set()
            self.user_cuisines[user_id].add(cuisine)

        self._fitted = True
        return self

    def compute_similarity(self, cuisine1: str, cuisine2: str) -> float:
        """
        Compute Swing similarity between two cuisines.

        Based on shared users, penalized by cuisine popularity and user activity.
        """
        if cuisine1 == cuisine2:
            return 1.0

        # Check cache
        cache_key = (min(cuisine1, cuisine2), max(cuisine1, cuisine2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        users1 = self.cuisine_users.get(cuisine1, set())
        users2 = self.cuisine_users.get(cuisine2, set())
        common_users = users1 & users2

        if not common_users:
            return 0.0

        cfg = self.config
        similarity = 0.0

        for user in common_users:
            user_activity = len(self.user_cuisines.get(user, set()))
            weight = 1.0 / (
                ((len(users1) + cfg.alpha1) ** cfg.beta) *
                ((len(users2) + cfg.alpha1) ** cfg.beta) *
                (user_activity + cfg.alpha2)
            )
            similarity += weight

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def get_similar_cuisines(
        self,
        cuisine: str,
        top_k: int = None,
        exclude: Set[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k similar cuisines to the given cuisine.

        Args:
            cuisine: Query cuisine
            top_k: Number of similar cuisines to return (default: config.top_k)
            exclude: Set of cuisines to exclude from results

        Returns:
            List of (cuisine, similarity_score) tuples, sorted descending
        """
        top_k = top_k or self.config.top_k
        exclude = exclude or set()
        exclude.add(cuisine)

        similarities = []
        for other_cuisine in self.cuisine_users.keys():
            if other_cuisine in exclude:
                continue

            sim = self.compute_similarity(cuisine, other_cuisine)
            if sim >= self.config.min_threshold:
                similarities.append((other_cuisine, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def get_candidates_for_history(
        self,
        cuisine_history: List[str],
        items_per_seed: int = 5,
        total_candidates: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Generate candidate cuisines based on a user's cuisine history.

        For each unique cuisine in history, get top-k similar cuisines.
        Combine all, deduplicate by keeping highest score, return top N.

        Args:
            cuisine_history: List of cuisines the user has ordered
            items_per_seed: Top-k similar cuisines per history cuisine
            total_candidates: Total candidates to return

        Returns:
            List of (cuisine, score) tuples, sorted by score descending
        """
        # Deduplicate history
        unique_cuisines = list(dict.fromkeys(cuisine_history))  # Preserves order

        # Collect all similar cuisines
        candidate_scores: Dict[str, float] = {}
        history_set = set(unique_cuisines)

        for seed_cuisine in unique_cuisines:
            similar = self.get_similar_cuisines(
                seed_cuisine,
                top_k=items_per_seed,
                exclude=history_set  # Don't recommend cuisines already in history
            )
            for sim_cuisine, score in similar:
                if sim_cuisine in candidate_scores:
                    candidate_scores[sim_cuisine] = max(candidate_scores[sim_cuisine], score)
                else:
                    candidate_scores[sim_cuisine] = score

        # Sort by score and return top N
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: -x[1]
        )
        return sorted_candidates[:total_candidates]

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        """Get all cuisines as candidates."""
        return list(self.cuisine_users.keys())

    def get_method_name(self) -> str:
        return "cuisine_swing"

    def get_all_cuisines(self) -> List[str]:
        """Get list of all cuisines."""
        return list(self.cuisine_users.keys())

    def get_cuisine_popularity(self, cuisine: str) -> int:
        """Get number of users who ordered this cuisine."""
        return len(self.cuisine_users.get(cuisine, set()))

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats.update({
            'num_cuisines': len(self.cuisine_users),
            'num_users': len(self.user_cuisines),
            'total_interactions': sum(len(cuisines) for cuisines in self.user_cuisines.values()),
            'avg_cuisines_per_user': (
                sum(len(cuisines) for cuisines in self.user_cuisines.values()) / len(self.user_cuisines)
                if self.user_cuisines else 0.0
            ),
            'avg_users_per_cuisine': (
                sum(len(users) for users in self.cuisine_users.values()) / len(self.cuisine_users)
                if self.cuisine_users else 0.0
            ),
        })
        return stats
