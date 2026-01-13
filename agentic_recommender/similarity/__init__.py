"""
Similarity algorithms for collaborative filtering.

Available methods:
- SwingMethod: Alibaba's Swing algorithm (anti-noise)
- CosineMethod: Vector-based cosine similarity
- JaccardMethod: Set overlap similarity

Use SimilarityFactory for easy creation and switching.
"""

from .base import SimilarityMethod, SimilarityConfig
from .methods import (
    SwingMethod, SwingConfig,
    CosineMethod, CosineConfig,
    JaccardMethod, JaccardConfig,
)
from .factory import SimilarityFactory

# Keep backward compatibility with old Swing import
from .swing import SwingSimilarity

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

    # Factory
    'SimilarityFactory',

    # Legacy (backward compatibility)
    'SwingSimilarity',
]
