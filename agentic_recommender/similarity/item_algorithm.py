"""
Item Algorithm Abstraction for Configurable Item Representation.

Supports different item granularities for recommendation:
- CUISINE: Item = primary_cuisine (e.g., "Thai")
- VENDOR_CUISINE: Item = vendor_id||cuisine (e.g., "V123||Thai")

This allows comparing recommendation performance at different levels.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import pandas as pd


class ItemAlgorithmType(Enum):
    """Types of item representation algorithms."""
    CUISINE = "cuisine"              # Algorithm 1: Item = cuisine only
    VENDOR_CUISINE = "vendor_cuisine"  # Algorithm 2: Item = vendor_id||cuisine


class ItemAlgorithm(ABC):
    """
    Abstract base class for item representation algorithms.

    Defines how items are extracted from order data and named in caches.
    """

    SEPARATOR = "||"  # Separator for composite items

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of this algorithm (used in logging)."""
        pass

    @abstractmethod
    def extract_item_from_row(self, row: pd.Series) -> str:
        """
        Extract item identifier from a DataFrame row.

        Args:
            row: A row from the orders DataFrame containing vendor_id, cuisine, etc.

        Returns:
            String item identifier
        """
        pass

    @abstractmethod
    def get_cache_suffix(self) -> str:
        """
        Get suffix for cache file naming.

        Returns:
            String suffix like "cuisine" or "vendor_cuisine"
        """
        pass

    def parse_item(self, item: str) -> dict:
        """
        Parse an item string back into its components.

        Args:
            item: Item string (e.g., "Thai" or "V123||Thai")

        Returns:
            Dict with parsed components (varies by algorithm)
        """
        return {"item": item}


class CuisineItemAlgorithm(ItemAlgorithm):
    """
    Algorithm 1: Item = cuisine only.

    This is the traditional approach where items are just cuisine names.
    Example: "Thai", "Chinese", "Italian"
    """

    def get_algorithm_name(self) -> str:
        return "cuisine"

    def extract_item_from_row(self, row: pd.Series) -> str:
        """Extract cuisine from row."""
        return str(row.get('cuisine', 'unknown'))

    def get_cache_suffix(self) -> str:
        return "cuisine"

    def parse_item(self, item: str) -> dict:
        """Parse cuisine item."""
        return {"cuisine": item}


class VendorCuisineItemAlgorithm(ItemAlgorithm):
    """
    Algorithm 2: Item = vendor_id||cuisine.

    This creates more granular items that capture vendor-cuisine combinations.
    Example: "V123||Thai", "V456||Chinese"

    Benefits:
    - Distinguishes between different vendors serving the same cuisine
    - More items = finer-grained recommendations
    - Can capture vendor-specific user preferences
    """

    def get_algorithm_name(self) -> str:
        return "vendor_cuisine"

    def extract_item_from_row(self, row: pd.Series) -> str:
        """Extract vendor_id||cuisine composite item from row."""
        vendor_id = str(row.get('vendor_id', 'unknown'))
        cuisine = str(row.get('cuisine', 'unknown'))
        return f"{vendor_id}{self.SEPARATOR}{cuisine}"

    def get_cache_suffix(self) -> str:
        return "vendor_cuisine"

    def parse_item(self, item: str) -> dict:
        """Parse vendor_cuisine item into components."""
        if self.SEPARATOR in item:
            parts = item.split(self.SEPARATOR, 1)
            return {
                "vendor_id": parts[0],
                "cuisine": parts[1] if len(parts) > 1 else "unknown"
            }
        # Fallback: treat as cuisine only
        return {"vendor_id": "unknown", "cuisine": item}


def create_item_algorithm(prediction_target: str) -> ItemAlgorithm:
    """
    Factory function to create the appropriate item algorithm.

    Args:
        prediction_target: One of "cuisine", "vendor_cuisine", "vendor", "product"

    Returns:
        ItemAlgorithm instance

    Note:
        - "cuisine" -> CuisineItemAlgorithm
        - "vendor_cuisine" -> VendorCuisineItemAlgorithm
        - "vendor" and "product" are handled separately in the codebase
    """
    if prediction_target == ItemAlgorithmType.VENDOR_CUISINE.value:
        return VendorCuisineItemAlgorithm()
    else:
        # Default to cuisine for "cuisine", "vendor", "product"
        # (vendor/product have their own column mappings)
        return CuisineItemAlgorithm()


def get_cache_suffix_for_target(prediction_target: str) -> str:
    """
    Get cache suffix for a prediction target.

    Args:
        prediction_target: "cuisine", "vendor_cuisine", "vendor", or "product"

    Returns:
        Cache suffix string
    """
    if prediction_target == ItemAlgorithmType.VENDOR_CUISINE.value:
        return "vendor_cuisine"
    return prediction_target  # cuisine, vendor, product
