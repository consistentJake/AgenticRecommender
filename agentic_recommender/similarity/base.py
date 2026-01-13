"""
Base classes for modular similarity calculation.

Provides abstract interface and factory for different similarity methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Type


@dataclass
class SimilarityConfig:
    """Base configuration for similarity methods."""
    top_k: int = 10
    min_threshold: float = 0.0
    cache_enabled: bool = True


class SimilarityMethod(ABC):
    """
    Abstract base class for similarity calculation methods.

    All similarity methods must implement:
    - fit(): Build index from interaction data
    - compute_similarity(): Calculate similarity between two entities
    - get_similar(): Get top-k similar entities
    """

    def __init__(self, config: SimilarityConfig = None):
        self.config = config or SimilarityConfig()
        self._cache: Dict[Tuple[str, str], float] = {}
        self._fitted = False

    @abstractmethod
    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'SimilarityMethod':
        """
        Build similarity index from interactions.

        Args:
            interactions: List of (entity1_id, entity2_id) tuples
                         For user-item: (user_id, item_id)

        Returns:
            self (for chaining)
        """
        pass

    @abstractmethod
    def compute_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute similarity between two entities.

        Args:
            entity1: First entity ID
            entity2: Second entity ID

        Returns:
            Similarity score (typically 0.0 to 1.0)
        """
        pass

    def get_similar(
        self,
        entity_id: str,
        exclude: Set[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k similar entities.

        Args:
            entity_id: Query entity ID
            exclude: Set of entity IDs to exclude

        Returns:
            List of (entity_id, similarity_score) tuples, sorted desc
        """
        exclude = exclude or set()
        exclude.add(entity_id)

        similarities = []
        for other_id in self._get_candidate_entities(entity_id):
            if other_id in exclude:
                continue

            sim = self.compute_similarity(entity_id, other_id)
            if sim >= self.config.min_threshold:
                similarities.append((other_id, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:self.config.top_k]

    @abstractmethod
    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        """Get candidate entities for similarity computation."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of this similarity method."""
        pass

    def clear_cache(self):
        """Clear similarity cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            'method': self.get_method_name(),
            'fitted': self._fitted,
            'cache_size': len(self._cache),
            'config': {
                'top_k': self.config.top_k,
                'min_threshold': self.config.min_threshold,
                'cache_enabled': self.config.cache_enabled,
            }
        }
