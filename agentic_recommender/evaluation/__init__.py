"""
Evaluation module for agentic recommender system.

Provides Top-K Hit Ratio evaluation for sequential recommendation.
"""

from .topk import (
    TopKMetrics,
    SequentialRecommendationEvaluator,
    TopKTestDataBuilder,
)

__all__ = [
    'TopKMetrics',
    'SequentialRecommendationEvaluator',
    'TopKTestDataBuilder',
]
