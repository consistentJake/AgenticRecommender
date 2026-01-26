"""
Basket-aware metrics for multi-item prediction evaluation.

These metrics handle cases where the ground truth is a set of items (basket)
rather than a single item, common in next-basket recommendation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Tuple


@dataclass
class BasketMetrics:
    """
    Metrics for basket (multi-item) prediction.

    For each sample, ground truth is a set of items (e.g., cuisines in an order).
    Predictions are a ranked list of items.
    """
    # Core basket metrics
    hit_at_k: float = 0.0        # % of samples with at least one correct prediction
    recall_at_k: float = 0.0     # Average |R_K ∩ G| / |G| across samples
    precision_at_k: float = 0.0  # Average |R_K ∩ G| / K across samples
    ndcg_at_k: float = 0.0       # Normalized DCG with binary relevance
    mrr: float = 0.0             # Mean Reciprocal Rank of first correct prediction

    # Per-K breakdowns
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0

    # Statistics
    total_samples: int = 0
    valid_samples: int = 0
    avg_basket_size: float = 0.0
    avg_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hit@k': self.hit_at_k,
            'recall@k': self.recall_at_k,
            'precision@k': self.precision_at_k,
            'ndcg@k': self.ndcg_at_k,
            'mrr': self.mrr,
            'recall@1': self.recall_at_1,
            'recall@3': self.recall_at_3,
            'recall@5': self.recall_at_5,
            'recall@10': self.recall_at_10,
            'hit@1': self.hit_at_1,
            'hit@3': self.hit_at_3,
            'hit@5': self.hit_at_5,
            'hit@10': self.hit_at_10,
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,
            'avg_basket_size': self.avg_basket_size,
            'avg_time_ms': self.avg_time_ms,
        }

    def __str__(self) -> str:
        return f"""Basket Evaluation Results:
  Hit@K:        {self.hit_at_k:.2%}
  Recall@K:     {self.recall_at_k:.4f}
  Precision@K:  {self.precision_at_k:.4f}
  NDCG@K:       {self.ndcg_at_k:.4f}
  MRR:          {self.mrr:.4f}
  ---
  Hit@1:        {self.hit_at_1:.2%}
  Hit@3:        {self.hit_at_3:.2%}
  Hit@5:        {self.hit_at_5:.2%}
  Hit@10:       {self.hit_at_10:.2%}
  ---
  Recall@1:     {self.recall_at_1:.4f}
  Recall@3:     {self.recall_at_3:.4f}
  Recall@5:     {self.recall_at_5:.4f}
  Recall@10:    {self.recall_at_10:.4f}
  ---
  Avg Basket Size: {self.avg_basket_size:.2f}
  Samples: {self.valid_samples}/{self.total_samples}"""


def compute_basket_hit(predictions: List[str], ground_truth: Set[str], k: int) -> int:
    """
    Compute hit for basket prediction.

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)
        k: Number of top predictions to consider

    Returns:
        1 if any prediction in top-k is in ground truth, 0 otherwise
    """
    if not predictions or not ground_truth:
        return 0

    top_k = set(predictions[:k])
    return 1 if len(top_k & ground_truth) > 0 else 0


def compute_basket_recall(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute recall for basket prediction.

    Recall@K = |R_K ∩ G| / |G|

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)
        k: Number of top predictions to consider

    Returns:
        Fraction of ground truth items found in top-k predictions
    """
    if not predictions or not ground_truth:
        return 0.0

    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def compute_basket_precision(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute precision for basket prediction.

    Precision@K = |R_K ∩ G| / K

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)
        k: Number of top predictions to consider

    Returns:
        Fraction of top-k predictions that are in ground truth
    """
    if not predictions or not ground_truth:
        return 0.0

    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)
    return hits / k


def compute_basket_ndcg(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute NDCG for basket prediction with binary relevance.

    DCG = Σ(i=1 to k) rel_i / log2(i+1)
    where rel_i = 1 if prediction[i] in ground_truth, 0 otherwise

    NDCG = DCG / IDCG
    where IDCG is computed assuming all ground truth items ranked first

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)
        k: Number of top predictions to consider

    Returns:
        NDCG score between 0 and 1
    """
    if not predictions or not ground_truth:
        return 0.0

    # Compute DCG
    dcg = 0.0
    for i, pred in enumerate(predictions[:k]):
        if pred in ground_truth:
            # Binary relevance: 1 if in ground truth, 0 otherwise
            # Position is 1-indexed, so use i+1
            dcg += 1.0 / math.log2(i + 2)  # log2(i+2) because i starts at 0

    # Compute IDCG (ideal DCG)
    # Best case: all ground truth items ranked first
    n_relevant = min(len(ground_truth), k)
    idcg = 0.0
    for i in range(n_relevant):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_basket_mrr(predictions: List[str], ground_truth: Set[str]) -> float:
    """
    Compute Mean Reciprocal Rank for basket prediction.

    MRR = 1 / rank_of_first_correct_prediction

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)

    Returns:
        Reciprocal rank of first correct prediction, or 0 if none found
    """
    if not predictions or not ground_truth:
        return 0.0

    for i, pred in enumerate(predictions):
        if pred in ground_truth:
            return 1.0 / (i + 1)

    return 0.0


def compute_basket_f1(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute F1 score for basket prediction.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)
        k: Number of top predictions to consider

    Returns:
        F1 score between 0 and 1
    """
    precision = compute_basket_precision(predictions, ground_truth, k)
    recall = compute_basket_recall(predictions, ground_truth, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def aggregate_basket_metrics(
    results: List[Dict[str, Any]],
    k: int = 10,
) -> BasketMetrics:
    """
    Aggregate basket metrics across multiple samples.

    Args:
        results: List of result dicts, each containing:
            - 'predictions': List[str] - ranked predictions
            - 'ground_truth_items': Set[str] - ground truth basket
            - 'time_ms': float (optional) - time taken
        k: Default K value for @K metrics

    Returns:
        Aggregated BasketMetrics
    """
    if not results:
        return BasketMetrics()

    n = len(results)
    valid_results = [r for r in results if r.get('predictions') and r.get('ground_truth_items')]
    n_valid = len(valid_results)

    if n_valid == 0:
        return BasketMetrics(total_samples=n)

    # Compute per-sample metrics
    hits_at_k = []
    recalls_at_k = []
    precisions_at_k = []
    ndcgs_at_k = []
    mrrs = []

    # Per-K metrics
    hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []
    recalls_at_1, recalls_at_3, recalls_at_5, recalls_at_10 = [], [], [], []

    basket_sizes = []
    times = []

    for r in valid_results:
        preds = r['predictions']
        gt = r['ground_truth_items']

        # Main metrics at K
        hits_at_k.append(compute_basket_hit(preds, gt, k))
        recalls_at_k.append(compute_basket_recall(preds, gt, k))
        precisions_at_k.append(compute_basket_precision(preds, gt, k))
        ndcgs_at_k.append(compute_basket_ndcg(preds, gt, k))
        mrrs.append(compute_basket_mrr(preds, gt))

        # Per-K hits
        hits_at_1.append(compute_basket_hit(preds, gt, 1))
        hits_at_3.append(compute_basket_hit(preds, gt, 3))
        hits_at_5.append(compute_basket_hit(preds, gt, 5))
        hits_at_10.append(compute_basket_hit(preds, gt, 10))

        # Per-K recalls
        recalls_at_1.append(compute_basket_recall(preds, gt, 1))
        recalls_at_3.append(compute_basket_recall(preds, gt, 3))
        recalls_at_5.append(compute_basket_recall(preds, gt, 5))
        recalls_at_10.append(compute_basket_recall(preds, gt, 10))

        # Basket size
        basket_sizes.append(len(gt))

        # Time
        if 'time_ms' in r:
            times.append(r['time_ms'])

    # Aggregate
    return BasketMetrics(
        hit_at_k=sum(hits_at_k) / n_valid,
        recall_at_k=sum(recalls_at_k) / n_valid,
        precision_at_k=sum(precisions_at_k) / n_valid,
        ndcg_at_k=sum(ndcgs_at_k) / n_valid,
        mrr=sum(mrrs) / n_valid,
        recall_at_1=sum(recalls_at_1) / n_valid,
        recall_at_3=sum(recalls_at_3) / n_valid,
        recall_at_5=sum(recalls_at_5) / n_valid,
        recall_at_10=sum(recalls_at_10) / n_valid,
        hit_at_1=sum(hits_at_1) / n_valid,
        hit_at_3=sum(hits_at_3) / n_valid,
        hit_at_5=sum(hits_at_5) / n_valid,
        hit_at_10=sum(hits_at_10) / n_valid,
        total_samples=n,
        valid_samples=n_valid,
        avg_basket_size=sum(basket_sizes) / n_valid,
        avg_time_ms=sum(times) / len(times) if times else 0.0,
    )


def compute_all_basket_metrics(
    predictions: List[str],
    ground_truth: Set[str],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute all basket metrics for a single sample.

    Args:
        predictions: Ranked list of predicted items
        ground_truth: Set of ground truth items (basket)
        k: Default K value for @K metrics

    Returns:
        Dictionary of metric name -> value
    """
    return {
        f'hit@{k}': compute_basket_hit(predictions, ground_truth, k),
        f'recall@{k}': compute_basket_recall(predictions, ground_truth, k),
        f'precision@{k}': compute_basket_precision(predictions, ground_truth, k),
        f'ndcg@{k}': compute_basket_ndcg(predictions, ground_truth, k),
        'mrr': compute_basket_mrr(predictions, ground_truth),
        f'f1@{k}': compute_basket_f1(predictions, ground_truth, k),
        # Additional K values
        'hit@1': compute_basket_hit(predictions, ground_truth, 1),
        'hit@3': compute_basket_hit(predictions, ground_truth, 3),
        'hit@5': compute_basket_hit(predictions, ground_truth, 5),
        'hit@10': compute_basket_hit(predictions, ground_truth, 10),
        'recall@1': compute_basket_recall(predictions, ground_truth, 1),
        'recall@3': compute_basket_recall(predictions, ground_truth, 3),
        'recall@5': compute_basket_recall(predictions, ground_truth, 5),
        'recall@10': compute_basket_recall(predictions, ground_truth, 10),
    }
