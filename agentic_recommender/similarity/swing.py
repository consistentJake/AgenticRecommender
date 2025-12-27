"""
Swing similarity algorithm for collaborative filtering.

Based on Alibaba's Swing algorithm with anti-noise properties.

Formula:
sim(u1, u2) = Σ(i ∈ common_items) w(u1, u2, i)

where:
w(u1, u2, i) = 1 / ((|I(u1)| + α1)^β × (|I(u2)| + α1)^β × (|U(i)| + α2))

Parameters:
- α1 = 5.0: User activity smoothing
- α2 = 1.0: Item popularity smoothing
- β = 0.3: Power weight for user activity
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class SwingConfig:
    """Configuration for Swing similarity."""
    alpha1: float = 5.0
    alpha2: float = 1.0
    beta: float = 0.3
    similarity_threshold: float = 0.1
    top_k: int = 5


class SwingSimilarity:
    """
    Swing similarity calculator for user-to-user collaborative filtering.

    Usage:
        # Initialize
        swing = SwingSimilarity(SwingConfig())

        # Fit with interactions
        interactions = [
            ('user1', 'pizza'),
            ('user1', 'burgare'),
            ('user2', 'pizza'),
            ...
        ]
        swing.fit(interactions)

        # Get similar users
        similar = swing.get_similar_users('user1')
        # Returns: [('user2', 0.75), ('user3', 0.42), ...]
    """

    def __init__(self, config: SwingConfig = None):
        """
        Initialize Swing similarity calculator.

        Args:
            config: Configuration object
        """
        self.config = config or SwingConfig()

        # Indices
        self.user_items: Dict[str, Set[str]] = {}  # user_id -> set of item_ids
        self.item_users: Dict[str, Set[str]] = {}  # item_id -> set of user_ids

        # Cache for computed similarities
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def fit(self, interactions: List[Tuple[str, str]]):
        """
        Build user-item and item-user indices from interactions.

        Args:
            interactions: List of (user_id, item_id) tuples
        """
        self.user_items.clear()
        self.item_users.clear()
        self._similarity_cache.clear()

        for user_id, item_id in interactions:
            # Add to user_items
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

            # Add to item_users
            if item_id not in self.item_users:
                self.item_users[item_id] = set()
            self.item_users[item_id].add(user_id)

    def compute_user_similarity(self, user1: str, user2: str) -> float:
        """
        Compute Swing similarity between two users.

        Args:
            user1: First user ID
            user2: Second user ID

        Returns:
            Similarity score (0.0 to ~1.0)
        """
        # Check cache
        cache_key = (min(user1, user2), max(user1, user2))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Get user item sets
        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())

        # Find common items
        common_items = items1 & items2

        if not common_items:
            self._similarity_cache[cache_key] = 0.0
            return 0.0

        # Calculate Swing similarity
        α1 = self.config.alpha1
        α2 = self.config.alpha2
        β = self.config.beta

        similarity = 0.0

        for item in common_items:
            # Get number of users who interacted with this item
            item_users = self.item_users.get(item, set())
            num_item_users = len(item_users)

            # Weight components
            user1_weight = (len(items1) + α1) ** β
            user2_weight = (len(items2) + α1) ** β
            item_weight = num_item_users + α2

            # Add contribution
            contribution = 1.0 / (user1_weight * user2_weight * item_weight)
            similarity += contribution

        # Cache result
        self._similarity_cache[cache_key] = similarity

        return similarity

    def get_similar_users(
        self,
        user_id: str,
        exclude_users: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k similar users above threshold.

        Args:
            user_id: Target user ID
            exclude_users: Optional set of user IDs to exclude

        Returns:
            List of (user_id, similarity_score) tuples, sorted by similarity desc
        """
        exclude = exclude_users or set()
        exclude.add(user_id)  # Always exclude self

        similarities = []

        # Compute similarity with all other users
        for other_user in self.user_items.keys():
            if other_user in exclude:
                continue

            sim = self.compute_user_similarity(user_id, other_user)

            # Filter by threshold
            if sim >= self.config.similarity_threshold:
                similarities.append((other_user, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return similarities[:self.config.top_k]

    def get_user_items(self, user_id: str) -> Set[str]:
        """
        Get items for a user.

        Args:
            user_id: User ID

        Returns:
            Set of item IDs
        """
        return self.user_items.get(user_id, set()).copy()

    def get_item_users(self, item_id: str) -> Set[str]:
        """
        Get users for an item.

        Args:
            item_id: Item ID

        Returns:
            Set of user IDs
        """
        return self.item_users.get(item_id, set()).copy()

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the similarity index.

        Returns:
            Dictionary with stats
        """
        return {
            'num_users': len(self.user_items),
            'num_items': len(self.item_users),
            'total_interactions': sum(len(items) for items in self.user_items.values()),
            'cache_size': len(self._similarity_cache),
            'config': {
                'alpha1': self.config.alpha1,
                'alpha2': self.config.alpha2,
                'beta': self.config.beta,
                'threshold': self.config.similarity_threshold,
                'top_k': self.config.top_k
            }
        }

    def clear_cache(self):
        """Clear similarity cache."""
        self._similarity_cache.clear()
