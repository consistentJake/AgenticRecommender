"""
Factory for creating and switching similarity methods.
"""

from typing import Dict, List, Optional, Type, Any

from .base import SimilarityMethod, SimilarityConfig
from .methods import (
    SwingMethod, SwingConfig,
    CosineMethod, CosineConfig,
    JaccardMethod, JaccardConfig,
)


class SimilarityFactory:
    """
    Factory for creating and switching similarity methods.

    Usage:
        factory = SimilarityFactory()

        # Get specific method
        swing = factory.create("swing", SwingConfig(alpha1=5.0))

        # Get default method
        default = factory.get_default()

        # Switch method at runtime
        factory.set_default("cosine")

        # List available methods
        methods = factory.available_methods()
    """

    # Registry of available methods
    METHODS: Dict[str, tuple] = {
        "swing": (SwingMethod, SwingConfig),
        "cosine": (CosineMethod, CosineConfig),
        "jaccard": (JaccardMethod, JaccardConfig),
    }

    def __init__(self, default_method: str = "swing"):
        """
        Initialize factory.

        Args:
            default_method: Name of default similarity method
        """
        if default_method not in self.METHODS:
            raise ValueError(f"Unknown method: {default_method}. Available: {list(self.METHODS.keys())}")

        self._default_method = default_method
        self._instances: Dict[str, SimilarityMethod] = {}

    def create(
        self,
        method_name: str,
        config: SimilarityConfig = None
    ) -> SimilarityMethod:
        """
        Create a new similarity method instance.

        Args:
            method_name: Name of the method ("swing", "cosine", "jaccard")
            config: Optional configuration object

        Returns:
            New SimilarityMethod instance
        """
        if method_name not in self.METHODS:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.METHODS.keys())}")

        method_class, config_class = self.METHODS[method_name]
        config = config or config_class()

        return method_class(config)

    def get_or_create(
        self,
        method_name: str,
        config: SimilarityConfig = None
    ) -> SimilarityMethod:
        """
        Get cached instance or create new one.

        Args:
            method_name: Name of the method
            config: Optional configuration (only used if creating new)

        Returns:
            Cached or new SimilarityMethod instance
        """
        if method_name not in self._instances:
            self._instances[method_name] = self.create(method_name, config)
        return self._instances[method_name]

    def get_default(self) -> SimilarityMethod:
        """
        Get the default similarity method.

        Returns:
            Default SimilarityMethod instance
        """
        return self.get_or_create(self._default_method)

    def set_default(self, method_name: str):
        """
        Set the default similarity method.

        Args:
            method_name: Name of the method to use as default
        """
        if method_name not in self.METHODS:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.METHODS.keys())}")
        self._default_method = method_name

    def get_default_name(self) -> str:
        """Get name of current default method."""
        return self._default_method

    @classmethod
    def available_methods(cls) -> List[str]:
        """
        List available similarity methods.

        Returns:
            List of method names
        """
        return list(cls.METHODS.keys())

    @classmethod
    def get_method_info(cls, method_name: str) -> Dict[str, Any]:
        """
        Get information about a method.

        Args:
            method_name: Name of the method

        Returns:
            Dict with method info
        """
        if method_name not in cls.METHODS:
            return {"error": f"Unknown method: {method_name}"}

        method_class, config_class = cls.METHODS[method_name]
        default_config = config_class()

        return {
            "name": method_name,
            "class": method_class.__name__,
            "config_class": config_class.__name__,
            "default_config": {
                k: v for k, v in default_config.__dict__.items()
            }
        }

    def clear_all_caches(self):
        """Clear caches for all instantiated methods."""
        for method in self._instances.values():
            method.clear_cache()

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all instantiated methods."""
        return {
            name: method.get_stats()
            for name, method in self._instances.items()
        }
