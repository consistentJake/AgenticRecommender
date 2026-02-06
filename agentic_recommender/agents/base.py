"""Base class for recommendation pipeline agents."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any


class RecommendationAgent(ABC):
    """Base class for recommendation pipeline agents.

    Each agent has:
    - initialize(): one-time setup (model training/loading)
    - process(): handle a single request with typed I/O
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._logger = logging.getLogger(f"agents.{name}")
        self._initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """One-time setup (model training/loading)."""

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """Process a single request."""

    def get_stats(self) -> Dict[str, Any]:
        return {'name': self.name, 'initialized': self._initialized}
