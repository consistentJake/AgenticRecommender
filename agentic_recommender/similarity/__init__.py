"""
Similarity algorithms for collaborative filtering.

Available methods:
- SwingMethod: Alibaba's Swing algorithm (anti-noise)
- CosineMethod: Vector-based cosine similarity
- JaccardMethod: Set overlap similarity
"""

from .base import SimilarityMethod, SimilarityConfig
from .methods import (
    SwingMethod, SwingConfig,
    CosineMethod, CosineConfig,
    JaccardMethod, JaccardConfig,
)

__all__ = [
    # Base classes
    'SimilarityMethod',
    'SimilarityConfig',

    # Concrete methods
    'SwingMethod',
    'SwingConfig',
    'CosineMethod',
    'CosineConfig',
    'JaccardMethod',
    'JaccardConfig',
]
