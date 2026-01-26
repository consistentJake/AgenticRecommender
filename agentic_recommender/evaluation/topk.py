"""
Top-K Hit Ratio evaluation for sequential recommendation.

Instead of Yes/No classification, we evaluate:
- Given: User's order history (sequence of orders)
- Task: Predict the NEXT cuisine the user will order
- Metric: Is the ground truth in the top-K predictions?

Metrics:
- Hit@K: % of times ground truth is in top-K predictions
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


@dataclass
class TopKMetrics:
    """Metrics for Top-K evaluation."""

    k: int

    # Core metrics
    hit_rate: float = 0.0        # Hit@K
    mrr: float = 0.0             # Mean Reciprocal Rank
    ndcg: float = 0.0            # Normalized Discounted Cumulative Gain

    # Detailed breakdown
    hits_at_1: float = 0.0
    hits_at_3: float = 0.0
    hits_at_5: float = 0.0
    hits_at_10: float = 0.0

    # Statistics
    total_samples: int = 0
    valid_samples: int = 0       # Samples with valid predictions
    avg_prediction_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            f"hit@{self.k}": self.hit_rate,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit@1": self.hits_at_1,
            "hit@3": self.hits_at_3,
            "hit@5": self.hits_at_5,
            "hit@10": self.hits_at_10,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "avg_time_ms": self.avg_prediction_time_ms,
        }

    def __str__(self) -> str:
        """Format as readable string."""
        lines = [
            f"Top-K Evaluation Results (K={self.k})",
            f"  Hit@1:  {self.hits_at_1:.2%}",
            f"  Hit@3:  {self.hits_at_3:.2%}",
            f"  Hit@5:  {self.hits_at_5:.2%}",
            f"  Hit@10: {self.hits_at_10:.2%}",
            f"  MRR:    {self.mrr:.4f}",
            f"  NDCG:   {self.ndcg:.4f}",
            f"  Samples: {self.valid_samples}/{self.total_samples}",
            f"  Avg time: {self.avg_prediction_time_ms:.2f}ms",
        ]
        return "\n".join(lines)


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    sample_idx: int
    customer_id: str
    ground_truth: str
    predictions: List[Tuple[str, float]]  # (cuisine, confidence)
    rank: int  # 0 if not found, 1-indexed if found
    time_ms: float


class SequentialRecommendationEvaluator:
    """
    Evaluator for sequential recommendation with Top-K Hit Ratio.

    Workflow:
    1. For each test user:
       a. Get user's order history (all but last order)
       b. Ground truth = cuisine of last order
       c. LLM generates top-K cuisine predictions
       d. Check if ground truth is in predictions

    2. Aggregate metrics across all test users
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(
        self,
        llm_provider,
        cuisine_list: List[str] = None,
        k_values: List[int] = None,
    ):
        """
        Initialize evaluator.

        Args:
            llm_provider: LLM provider instance
            cuisine_list: List of available cuisines
            k_values: K values to evaluate (default: [1, 3, 5, 10])
        """
        self.llm = llm_provider
        self.cuisine_list = cuisine_list or []
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        verbose: bool = False,
        max_samples: int = None,
    ) -> TopKMetrics:
        """
        Evaluate on test samples.

        Args:
            test_samples: List of test cases, each with:
                - customer_id: str
                - order_history: List[Dict] with keys: cuisine, day_of_week, hour, price
                - ground_truth_cuisine: str
            verbose: Print progress
            max_samples: Limit number of samples

        Returns:
            TopKMetrics with evaluation results
        """
        if max_samples:
            test_samples = test_samples[:max_samples]

        results: List[EvaluationResult] = []
        total_time = 0.0

        for i, sample in enumerate(test_samples):
            start_time = time.time()

            # Get top-K predictions from LLM
            predictions = self._get_predictions(
                customer_id=sample['customer_id'],
                order_history=sample['order_history'],
                k=max(self.k_values),
                target_hour=sample.get('target_hour'),
                target_day_of_week=sample.get('target_day_of_week'),
            )

            elapsed = (time.time() - start_time) * 1000  # ms
            total_time += elapsed

            # Find rank of ground truth
            ground_truth = sample['ground_truth_cuisine']
            rank = self._find_rank(predictions, ground_truth)

            results.append(EvaluationResult(
                sample_idx=i,
                customer_id=sample['customer_id'],
                ground_truth=ground_truth,
                predictions=predictions,
                rank=rank,
                time_ms=elapsed
            ))

            if verbose and (i + 1) % 10 == 0:
                print(f"Evaluated {i+1}/{len(test_samples)} samples...")

        # Compute metrics
        return self._compute_metrics(results)

    def _get_predictions(
        self,
        customer_id: str,
        order_history: List[Dict],
        k: int,
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> List[Tuple[str, float]]:
        """
        Get top-K cuisine predictions from LLM.

        Returns:
            List of (cuisine, confidence) tuples
        """
        prompt = self._build_prediction_prompt(customer_id, order_history, k, target_hour, target_day_of_week)
        system_prompt = self._get_system_prompt()

        response = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=500,
        )

        return self._parse_predictions(response, k)

    def _build_prediction_prompt(
        self,
        customer_id: str,
        order_history: List[Dict],
        k: int,
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> str:
        """Build prompt for top-K prediction.

        Args:
            customer_id: Customer identifier
            order_history: List of past orders with cuisine, hour, day_of_week
            k: Number of top cuisines to predict
            target_hour: Hour when user wants to order (0-23)
            target_day_of_week: Day when user wants to order (0=Mon, 6=Sun)
        """
        # Format order history (last 10 orders)
        history_lines = []
        for i, order in enumerate(order_history[-10:], 1):
            day_name = self.DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            cuisine = order.get('cuisine', 'unknown')
            price = order.get('price', 0.0)
            vendor_id = order.get('vendor_id', '')
            if vendor_id:
                history_lines.append(
                    f"{i}. {cuisine} from vendor {vendor_id} | {day_name} {hour}:00 | ${price:.2f}"
                )
            else:
                history_lines.append(
                    f"{i}. {cuisine} | {day_name} {hour}:00 | ${price:.2f}"
                )
        history_str = "\n".join(history_lines)

        # Limit cuisine list for prompt
        cuisines_str = ", ".join(self.cuisine_list[:30]) if self.cuisine_list else "various cuisines"

        # Build target time context
        if target_hour is not None and target_day_of_week is not None:
            target_day_name = self.DAY_NAMES[target_day_of_week]
            time_context = f"\n## Prediction Context:\nThe user wants to order on {target_day_name} at {target_hour}:00.\n"
        else:
            time_context = ""

        return f"""Based on this user's order history, predict the top {k} cuisines they are most likely to order next.

## Order History (oldest to newest):
{history_str}
{time_context}
## Available Cuisines:
{cuisines_str}

## Task:
Predict the TOP {k} cuisines this user is most likely to order next, ranked by probability.

Consider:
1. User's cuisine preferences (which cuisines they order most)
2. Sequential patterns (what typically follows recent orders)
3. Time patterns - especially the target ordering time if provided

Return your predictions as a JSON object:
{{
    "predictions": [
        {{"cuisine": "cuisine_name", "confidence": 0.95}},
        {{"cuisine": "cuisine_name", "confidence": 0.82}},
        ...
    ]
}}

Return ONLY the JSON object, no other text."""

    def _get_system_prompt(self) -> str:
        """Get system prompt for prediction."""
        return (
            "You are a sequential recommendation agent for a food delivery app. "
            "Given a user's order history, predict what cuisine they will order next. "
            "Base predictions on patterns in their history. "
            "Always respond with valid JSON only."
        )

    def _parse_predictions(
        self,
        response: str,
        k: int
    ) -> List[Tuple[str, float]]:
        """Parse LLM response into predictions."""
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            predictions = data.get('predictions', [])

            result = []
            for pred in predictions[:k]:
                cuisine = pred.get('cuisine', '').lower().strip()
                confidence = float(pred.get('confidence', 0.5))
                if cuisine:
                    result.append((cuisine, confidence))

            return result

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Try to extract any cuisine mentions from response
            return self._fallback_parse(response, k)

    def _fallback_parse(
        self,
        response: str,
        k: int
    ) -> List[Tuple[str, float]]:
        """Fallback parsing when JSON fails."""
        result = []
        response_lower = response.lower()

        # Check if any known cuisines are mentioned
        for cuisine in self.cuisine_list:
            if cuisine.lower() in response_lower:
                result.append((cuisine.lower(), 0.5))
                if len(result) >= k:
                    break

        return result

    def _find_rank(
        self,
        predictions: List[Tuple[str, float]],
        ground_truth: str
    ) -> int:
        """
        Find rank of ground truth in predictions.

        Returns:
            1-indexed rank, or 0 if not found
        """
        ground_truth_lower = ground_truth.lower().strip()

        for i, (cuisine, _) in enumerate(predictions):
            if cuisine.lower().strip() == ground_truth_lower:
                return i + 1

        return 0

    def _compute_metrics(
        self,
        results: List[EvaluationResult]
    ) -> TopKMetrics:
        """Compute evaluation metrics from results."""
        n = len(results)
        if n == 0:
            return TopKMetrics(k=max(self.k_values))

        # Valid samples (got some predictions)
        valid_results = [r for r in results if len(r.predictions) > 0]
        n_valid = len(valid_results)

        if n_valid == 0:
            return TopKMetrics(
                k=max(self.k_values),
                total_samples=n,
                valid_samples=0
            )

        # Hit rates at various K
        def hit_at_k(k: int) -> float:
            return sum(1 for r in valid_results if 0 < r.rank <= k) / n_valid

        # MRR (Mean Reciprocal Rank)
        mrr = sum(
            1.0 / r.rank if r.rank > 0 else 0
            for r in valid_results
        ) / n_valid

        # NDCG (simplified - binary relevance)
        ndcg = sum(
            1.0 / math.log2(r.rank + 1) if r.rank > 0 else 0
            for r in valid_results
        ) / n_valid

        # Average time
        avg_time = sum(r.time_ms for r in results) / n

        return TopKMetrics(
            k=max(self.k_values),
            hit_rate=hit_at_k(max(self.k_values)),
            mrr=mrr,
            ndcg=ndcg,
            hits_at_1=hit_at_k(1),
            hits_at_3=hit_at_k(3),
            hits_at_5=hit_at_k(5),
            hits_at_10=hit_at_k(10),
            total_samples=n,
            valid_samples=n_valid,
            avg_prediction_time_ms=avg_time,
        )


class TopKTestDataBuilder:
    """
    Build test data for Top-K evaluation from order data.

    Strategy:
    1. For each user with N orders (N >= min_history)
    2. Use first N-1 orders as history
    3. Use Nth order's cuisine as ground truth
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(
        self,
        orders_df: pd.DataFrame,
        vendors_df: pd.DataFrame = None,
        min_history: int = 5,
    ):
        """
        Initialize builder.

        Args:
            orders_df: Orders DataFrame (with customer_id, order_id, vendor_id, etc.)
            vendors_df: Optional vendors DataFrame (with vendor_id, primary_cuisine)
            min_history: Minimum number of orders required for a user
        """
        self.orders = orders_df.copy()
        self.vendors = vendors_df
        self.min_history = min_history

        # Merge cuisine if not present
        if 'cuisine' not in self.orders.columns and self.vendors is not None:
            self.orders = self.orders.merge(
                self.vendors[['vendor_id', 'primary_cuisine']],
                on='vendor_id',
                how='left'
            )
            self.orders = self.orders.rename(columns={'primary_cuisine': 'cuisine'})

        # Parse hour if needed
        if 'hour' not in self.orders.columns and 'order_time' in self.orders.columns:
            self.orders['hour'] = self.orders['order_time'].str.split(':').str[0].astype(int)

    def build_test_samples(
        self,
        n_samples: int = 100,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Build test samples for evaluation.

        Args:
            n_samples: Number of test samples to create. Use -1 for all eligible samples.
            seed: Random seed for reproducibility

        Returns:
            List of test samples with:
            - customer_id: str
            - order_history: List[Dict]
            - ground_truth_cuisine: str
        """
        import random
        random.seed(seed)

        # Group by customer
        customer_order_counts = self.orders.groupby('customer_id')['order_id'].nunique()

        # Filter customers with enough history
        eligible_customers = customer_order_counts[
            customer_order_counts >= self.min_history
        ].index.tolist()

        if len(eligible_customers) == 0:
            print(f"Warning: No customers with >= {self.min_history} orders")
            return []

        # Sample customers (n_samples < 0 means use all eligible)
        if n_samples < 0:
            sampled = eligible_customers
            random.shuffle(sampled)  # Shuffle for consistency with seed
        else:
            sampled = random.sample(
                eligible_customers,
                min(n_samples, len(eligible_customers))
            )

        samples = []
        for customer_id in sampled:
            sample = self._build_sample_for_customer(customer_id)
            if sample:
                samples.append(sample)

        return samples

    def _build_sample_for_customer(
        self,
        customer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Build single test sample for a customer."""

        customer_data = self.orders[self.orders['customer_id'] == customer_id]

        # Get unique orders sorted by time
        if 'day_num' in customer_data.columns:
            sort_cols = ['day_num', 'hour'] if 'hour' in customer_data.columns else ['day_num']
        else:
            sort_cols = ['order_id']

        customer_data = customer_data.sort_values(sort_cols)

        # Build order list (one entry per order, not per item)
        orders_list = []
        seen_orders = set()

        for _, row in customer_data.iterrows():
            order_id = row['order_id']
            if order_id in seen_orders:
                continue
            seen_orders.add(order_id)

            order_items = customer_data[customer_data['order_id'] == order_id]

            orders_list.append({
                'order_id': order_id,
                'vendor_id': str(row.get('vendor_id', '')),
                'cuisine': row.get('cuisine', 'unknown'),
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
                'price': float(order_items['unit_price'].sum()) if 'unit_price' in order_items else 0.0,
            })

        if len(orders_list) < self.min_history:
            return None

        # Use last order as ground truth
        # Include target hour/weekday from the ground truth order for prediction context
        last_order = orders_list[-1]
        ground_truth = last_order['cuisine']
        history = orders_list[:-1]

        return {
            'customer_id': customer_id,
            'order_history': history,
            'ground_truth_cuisine': ground_truth,
            'target_hour': last_order['hour'],
            'target_day_of_week': last_order['day_of_week'],
        }

    def get_unique_cuisines(self) -> List[str]:
        """Get list of unique cuisines in dataset."""
        if 'cuisine' in self.orders.columns:
            return self.orders['cuisine'].dropna().unique().tolist()
        return []


def run_topk_evaluation(
    api_key: str,
    data_dir: str = "/Users/zhenkai/Downloads/data_sg",
    n_samples: int = 50,
    verbose: bool = True,
) -> TopKMetrics:
    """
    Run full Top-K evaluation pipeline.

    Args:
        api_key: OpenRouter API key
        data_dir: Path to Singapore dataset
        n_samples: Number of test samples
        verbose: Print progress

    Returns:
        TopKMetrics with results
    """
    from pathlib import Path
    from agentic_recommender.models.llm_provider import OpenRouterProvider

    print("=" * 60)
    print("TOP-K HIT RATIO EVALUATION")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data_path = Path(data_dir)
    orders = pd.read_csv(data_path / "orders_sg_train.txt")
    vendors = pd.read_csv(data_path / "vendors_sg.txt")

    print(f"   Orders: {len(orders):,}")
    print(f"   Vendors: {len(vendors):,}")

    # Build test samples
    print(f"\n2. Building {n_samples} test samples...")
    builder = TopKTestDataBuilder(orders, vendors, min_history=5)
    test_samples = builder.build_test_samples(n_samples=n_samples)
    print(f"   Created {len(test_samples)} samples")

    # Get cuisine list
    cuisines = builder.get_unique_cuisines()
    print(f"   Unique cuisines: {len(cuisines)}")

    # Initialize LLM
    print("\n3. Initializing LLM (OpenRouter + Gemini 2.5 Flash)...")
    llm = OpenRouterProvider(
        api_key=api_key,
        model_name="google/gemini-2.0-flash-001"
    )

    # Run evaluation
    print(f"\n4. Running evaluation on {len(test_samples)} samples...")
    evaluator = SequentialRecommendationEvaluator(
        llm_provider=llm,
        cuisine_list=cuisines,
        k_values=[1, 3, 5, 10]
    )

    metrics = evaluator.evaluate(test_samples, verbose=verbose)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(metrics)
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    # Use API key from APIs.md
    API_KEY = "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"
    run_topk_evaluation(api_key=API_KEY, n_samples=20)
