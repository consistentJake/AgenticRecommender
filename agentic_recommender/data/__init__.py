"""Data adapters and utilities for agentic recommendation system."""

from .enriched_loader import (
    EnrichedDataLoader,
    DataConfig,
    load_singapore_data,
)
from .representations import (
    EnrichedUser,
    CuisineProfile,
    CuisineRegistry,
)

__all__ = [
    # Enriched loader
    'EnrichedDataLoader',
    'DataConfig',
    'load_singapore_data',

    # Representations
    'EnrichedUser',
    'CuisineProfile',
    'CuisineRegistry',
]
