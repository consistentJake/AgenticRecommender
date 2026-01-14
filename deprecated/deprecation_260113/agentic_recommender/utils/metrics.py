"""
Evaluation metrics for sequential recommendation.
Based on LLM-Sequential-Recommendation evaluation metrics.
"""

import numpy as np
from typing import List, Union, Dict


def hit_rate_at_k(predictions: List[int], ground_truth: Union[int, List[int]], k: int) -> float:
    """
    Calculate Hit Rate@K metric.
    
    Args:
        predictions: List of predicted item IDs in ranking order
        ground_truth: Ground truth item ID(s)
        k: Cut-off rank position
        
    Returns:
        1.0 if ground truth in top-k predictions, 0.0 otherwise
        
    Reference: LLM_Sequential_Recommendation_Analysis.md:244-246
    """
    if isinstance(ground_truth, int):
        ground_truth = [ground_truth]
    
    return 1.0 if any(item in ground_truth for item in predictions[:k]) else 0.0


def ndcg_at_k(predictions: List[int], ground_truth: Union[int, List[int]], k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    
    Args:
        predictions: List of predicted item IDs in ranking order
        ground_truth: Ground truth item ID(s)
        k: Cut-off rank position
        
    Returns:
        NDCG@K score between 0 and 1
        
    Reference: LLM_Sequential_Recommendation_Analysis.md:237-242
    """
    if isinstance(ground_truth, int):
        ground_truth = [ground_truth]
    
    # Calculate DCG
    dcg = sum([1/np.log2(i+2) for i, item in enumerate(predictions[:k]) 
               if item in ground_truth])
    
    # Calculate IDCG
    idcg = sum([1/np.log2(i+2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(predictions: List[int], ground_truth: Union[int, List[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        predictions: List of predicted item IDs in ranking order
        ground_truth: Ground truth item ID(s)
        
    Returns:
        Reciprocal rank of first correct prediction, 0.0 if none found
    """
    if isinstance(ground_truth, int):
        ground_truth = [ground_truth]
    
    for i, item in enumerate(predictions):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(predictions: List[int], ground_truth: Union[int, List[int]], k: int) -> float:
    """Calculate Precision@K"""
    if isinstance(ground_truth, int):
        ground_truth = [ground_truth]
    
    top_k = predictions[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / k if k > 0 else 0.0


def recall_at_k(predictions: List[int], ground_truth: Union[int, List[int]], k: int) -> float:
    """Calculate Recall@K"""
    if isinstance(ground_truth, int):
        ground_truth = [ground_truth]
    
    top_k = predictions[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0


class MetricsCalculator:
    """
    Calculator for comprehensive evaluation metrics.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values
    
    def calculate_all(self, predictions: List[int], ground_truth: Union[int, List[int]]) -> Dict[str, float]:
        """Calculate all metrics for given predictions"""
        metrics = {}
        
        # MRR (no k parameter)
        metrics['mrr'] = mrr(predictions, ground_truth)
        
        # Metrics for each k value
        for k in self.k_values:
            metrics[f'hr@{k}'] = hit_rate_at_k(predictions, ground_truth, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, ground_truth, k)
            metrics[f'precision@{k}'] = precision_at_k(predictions, ground_truth, k)
            metrics[f'recall@{k}'] = recall_at_k(predictions, ground_truth, k)
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple predictions"""
        if not all_metrics:
            return {}
        
        # Get all metric names
        metric_names = all_metrics[0].keys()
        
        # Calculate averages
        aggregated = {}
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            aggregated[metric] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated


def evaluate_recommendations(predictions_list: List[List[int]], 
                           ground_truths: List[Union[int, List[int]]],
                           k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Evaluate a batch of recommendations.
    
    Args:
        predictions_list: List of prediction lists for each sample
        ground_truths: List of ground truth items for each sample
        k_values: K values to evaluate
        
    Returns:
        Dictionary of aggregated metrics
    """
    calculator = MetricsCalculator(k_values)
    
    # Calculate metrics for each sample
    all_metrics = []
    for preds, gt in zip(predictions_list, ground_truths):
        metrics = calculator.calculate_all(preds, gt)
        all_metrics.append(metrics)
    
    # Aggregate results
    return calculator.aggregate_metrics(all_metrics)