"""
Similarity algorithms for collaborative filtering.

Available methods:
- SwingMethod: Alibaba's Swing algorithm (anti-noise)
- CosineMethod: Vector-based cosine similarity
- JaccardMethod: Set overlap similarity

Item algorithms for configurable item representation:
- CuisineItemAlgorithm: Item = cuisine only
- VendorCuisineItemAlgorithm: Item = vendor_id||cuisine
"""

from .base import SimilarityMethod, SimilarityConfig
from .methods import (
    SwingMethod, SwingConfig,
    CosineMethod, CosineConfig,
    JaccardMethod, JaccardConfig,
)
from .item_algorithm import (
    ItemAlgorithmType,
    ItemAlgorithm,
    CuisineItemAlgorithm,
    VendorCuisineItemAlgorithm,
    create_item_algorithm,
    get_cache_suffix_for_target,
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

    # Item algorithms
    'ItemAlgorithmType',
    'ItemAlgorithm',
    'CuisineItemAlgorithm',
    'VendorCuisineItemAlgorithm',
    'create_item_algorithm',
    'get_cache_suffix_for_target',
]
