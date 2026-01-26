"""
Retrieve-then-Rerank Evaluation for Cuisine Prediction.

Two-stage approach:
1. RETRIEVE: Generate candidate cuisines using user history + similar users (CF)
2. RERANK: LLM picks from candidates K times, evaluate if ground truth is picked

This is more realistic than asking LLM to pick from all 78 cuisines.
"""

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd

from ..similarity.methods import SwingMethod, CosineMethod, SwingConfig, CosineConfig, CuisineSwingMethod, CuisineSwingConfig
from ..similarity.lightGCN import LightGCNEmbeddingManager, LightGCNConfig


@dataclass
class RerankConfig:
    """Configuration for retrieve-rerank evaluation."""
    # Candidate generation
    n_candidates: int = 20          # Total candidates to generate
    n_from_history: int = 10        # Max cuisines from user's own history
    n_similar_users: int = 10       # Number of similar users to consider
    similarity_method: str = "swing"  # swing, cosine, or jaccard

    # LLM evaluation
    k_picks: int = 5                # Ask LLM to pick K times
    temperature: float = 0.7        # LLM temperature (higher = more variety)

    # Test settings
    n_samples: int = 10             # Number of test samples
    min_history: int = 5            # Minimum orders for a user
    seed: int = 42


@dataclass
class RerankMetrics:
    """Metrics for rerank evaluation."""
    # Core metrics
    recall_at_k: float = 0.0        # % of samples where ground truth was picked
    precision_at_k: float = 0.0     # % of picks that matched ground truth
    first_hit_avg: float = 0.0      # Average position of first correct pick

    # Candidate quality metrics
    ground_truth_in_candidates: float = 0.0  # % where GT was in candidate set
    avg_candidate_rank: float = 0.0  # Average rank of GT in candidates

    # Statistics
    total_samples: int = 0
    valid_samples: int = 0
    avg_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'recall@k': self.recall_at_k,
            'precision@k': self.precision_at_k,
            'first_hit_avg': self.first_hit_avg,
            'gt_in_candidates': self.ground_truth_in_candidates,
            'avg_candidate_rank': self.avg_candidate_rank,
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,
            'avg_time_ms': self.avg_time_ms,
        }

    def __str__(self) -> str:
        return f"""Rerank Evaluation Results:
  Recall@K:     {self.recall_at_k:.2%}
  Precision@K:  {self.precision_at_k:.2%}
  First Hit:    {self.first_hit_avg:.2f} (avg position)
  GT in Candidates: {self.ground_truth_in_candidates:.2%}
  Samples: {self.valid_samples}/{self.total_samples}
  Avg Time: {self.avg_time_ms:.2f}ms"""


class CuisineCandidateGenerator:
    """
    Generate candidate cuisines for a user using:
    1. User's own order history (most frequent cuisines)
    2. Similar users' preferences (collaborative filtering)
    """

    def __init__(self, config: RerankConfig):
        self.config = config
        self.similarity_model = None
        self.user_cuisines: Dict[str, List[str]] = {}  # user -> list of cuisines ordered
        self.all_cuisines: Set[str] = set()

    def fit(self, orders_df: pd.DataFrame):
        """
        Build user-cuisine mappings and similarity model.

        Args:
            orders_df: DataFrame with customer_id, order_id, cuisine columns
        """
        # Build user -> cuisines mapping
        self.user_cuisines.clear()
        self.all_cuisines.clear()

        # Get unique cuisines per order (not per item)
        order_cuisines = orders_df.groupby(['customer_id', 'order_id'])['cuisine'].first()

        for (customer_id, order_id), cuisine in order_cuisines.items():
            if customer_id not in self.user_cuisines:
                self.user_cuisines[customer_id] = []
            self.user_cuisines[customer_id].append(cuisine)
            self.all_cuisines.add(cuisine)

        # Build similarity model (user-cuisine interactions)
        interactions = []
        for customer_id, cuisines in self.user_cuisines.items():
            for cuisine in cuisines:
                interactions.append((customer_id, cuisine))

        # Initialize similarity method
        if self.config.similarity_method == "swing":
            self.similarity_model = SwingMethod(SwingConfig(
                top_k=self.config.n_similar_users
            ))
        else:
            self.similarity_model = CosineMethod(CosineConfig(
                top_k=self.config.n_similar_users
            ))

        self.similarity_model.fit(interactions)

        print(f"[CandidateGenerator] Fitted on {len(self.user_cuisines)} users, "
              f"{len(self.all_cuisines)} cuisines, {len(interactions)} interactions")

    def generate_candidates(
        self,
        customer_id: str,
        ground_truth: str = None,
        exclude_last_n: int = 1,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate candidate cuisines for a user.

        Args:
            customer_id: User to generate candidates for
            ground_truth: The actual cuisine (for evaluation)
            exclude_last_n: Exclude last N cuisines from history (for fair eval)

        Returns:
            Tuple of (candidate_list, debug_info)
        """
        debug_info = {
            'from_history': [],
            'from_similar_users': [],
            'ground_truth_added': False,
            'ground_truth_rank': -1,
        }

        user_history = self.user_cuisines.get(customer_id, [])

        # Exclude last N orders (the ground truth)
        if exclude_last_n > 0:
            user_history = user_history[:-exclude_last_n]

        candidates = []
        seen = set()

        # 1. Get cuisines from user's own history (most frequent first)
        cuisine_counts = Counter(user_history)
        history_cuisines = [c for c, _ in cuisine_counts.most_common(self.config.n_from_history)]

        for cuisine in history_cuisines:
            if cuisine not in seen:
                candidates.append(cuisine)
                seen.add(cuisine)
                debug_info['from_history'].append(cuisine)

        # 2. Get cuisines from similar users
        if self.similarity_model and len(candidates) < self.config.n_candidates:
            similar_users = self.similarity_model.get_similar(customer_id)

            similar_cuisine_counts = Counter()
            for sim_user_id, sim_score in similar_users:
                sim_cuisines = self.user_cuisines.get(sim_user_id, [])
                for cuisine in sim_cuisines:
                    if cuisine not in seen:
                        # Weight by similarity score
                        similar_cuisine_counts[cuisine] += sim_score

            # Add top cuisines from similar users
            for cuisine, _ in similar_cuisine_counts.most_common():
                if len(candidates) >= self.config.n_candidates:
                    break
                if cuisine not in seen:
                    candidates.append(cuisine)
                    seen.add(cuisine)
                    debug_info['from_similar_users'].append(cuisine)

        # 3. Fill remaining with popular cuisines if needed
        if len(candidates) < self.config.n_candidates:
            remaining = list(self.all_cuisines - seen)
            random.shuffle(remaining)
            for cuisine in remaining:
                if len(candidates) >= self.config.n_candidates:
                    break
                candidates.append(cuisine)
                seen.add(cuisine)

        # 4. IMPORTANT: Add ground truth if not present
        # ============================================
        # NOTE: We add ground truth to candidates for FAIR EVALUATION.
        # Without this, the candidate generation itself could fail to include
        # the correct answer, making it impossible for the LLM to succeed.
        #
        # In production, you wouldn't have ground truth, but for evaluation
        # we need to measure: "Given a candidate set that CONTAINS the answer,
        # can the LLM correctly identify it?"
        #
        # This separates two concerns:
        # 1. Candidate generation quality (does GT appear in top-20?)
        # 2. LLM reranking quality (can LLM pick GT from candidates?)
        #
        # We track ground_truth_rank to measure candidate generation quality.
        # ============================================
        if ground_truth:
            if ground_truth in candidates:
                debug_info['ground_truth_rank'] = candidates.index(ground_truth) + 1
            else:
                # Add ground truth to a random position (not first, not last)
                insert_pos = random.randint(1, max(1, len(candidates) - 1))
                candidates.insert(insert_pos, ground_truth)
                debug_info['ground_truth_added'] = True
                debug_info['ground_truth_rank'] = insert_pos + 1

                # Keep only n_candidates
                if len(candidates) > self.config.n_candidates:
                    # Remove a random non-GT candidate
                    removable = [i for i, c in enumerate(candidates) if c != ground_truth]
                    if removable:
                        candidates.pop(random.choice(removable))

        return candidates, debug_info

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'num_users': len(self.user_cuisines),
            'num_cuisines': len(self.all_cuisines),
            'similarity_method': self.config.similarity_method,
            'similarity_stats': self.similarity_model.get_stats() if self.similarity_model else None,
        }


class RerankEvaluator:
    """
    Evaluate LLM's ability to pick the correct cuisine from candidates.

    Flow:
    1. Generate candidate cuisines for user
    2. Ask LLM to pick ONE cuisine K times
    3. Check if ground truth appears in any of the K picks
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(
        self,
        llm_provider,
        candidate_generator: CuisineCandidateGenerator,
        config: RerankConfig,
    ):
        self.llm = llm_provider
        self.generator = candidate_generator
        self.config = config

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Tuple[RerankMetrics, List[Dict[str, Any]]]:
        """
        Run evaluation on test samples.

        Args:
            test_samples: List of test cases with:
                - customer_id
                - order_history
                - ground_truth_cuisine
            verbose: Print progress

        Returns:
            Tuple of (metrics, detailed_results)
        """
        results = []
        total_time = 0.0

        for i, sample in enumerate(test_samples):
            if verbose:
                print(f"\n[{i+1}/{len(test_samples)}] Customer: {sample['customer_id']}")

            start_time = time.time()

            # Generate candidates
            candidates, candidate_info = self.generator.generate_candidates(
                customer_id=sample['customer_id'],
                ground_truth=sample['ground_truth_cuisine'],
            )

            if verbose:
                print(f"  Candidates: {len(candidates)} cuisines")
                print(f"  GT rank in candidates: {candidate_info['ground_truth_rank']}")
                print(f"  GT added manually: {candidate_info['ground_truth_added']}")

            # Ask LLM K times
            picks = []
            llm_interactions = []
            for k in range(self.config.k_picks):
                llm_result = self._get_llm_pick(
                    order_history=sample['order_history'],
                    candidates=candidates,
                    target_hour=sample.get('target_hour'),
                    target_day_of_week=sample.get('target_day_of_week'),
                )
                pick = llm_result['parsed_pick']
                picks.append(pick)
                llm_interactions.append(llm_result)

                if verbose:
                    match = "✓" if pick == sample['ground_truth_cuisine'] else "✗"
                    print(f"  Pick {k+1}: {pick} {match}")

            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed

            # Calculate metrics for this sample
            ground_truth = sample['ground_truth_cuisine']
            hits = [1 if p == ground_truth else 0 for p in picks]
            first_hit = next((i+1 for i, h in enumerate(hits) if h == 1), 0)

            result = {
                'sample_idx': i,
                'customer_id': sample['customer_id'],
                'ground_truth': ground_truth,
                'candidates': candidates,
                'candidate_info': candidate_info,
                'picks': picks,
                'llm_interactions': llm_interactions,
                'hits': hits,
                'first_hit': first_hit,
                'recall': 1 if any(hits) else 0,
                'precision': sum(hits) / len(hits),
                'time_ms': elapsed,
            }
            results.append(result)

            if verbose:
                print(f"  Ground truth: {ground_truth}")
                print(f"  Recall: {result['recall']}, Precision: {result['precision']:.2f}")

        # Aggregate metrics
        valid_results = [r for r in results if len(r['picks']) > 0]
        n = len(valid_results)

        if n == 0:
            return RerankMetrics(total_samples=len(test_samples)), results

        metrics = RerankMetrics(
            recall_at_k=sum(r['recall'] for r in valid_results) / n,
            precision_at_k=sum(r['precision'] for r in valid_results) / n,
            first_hit_avg=sum(r['first_hit'] for r in valid_results if r['first_hit'] > 0) / max(1, sum(1 for r in valid_results if r['first_hit'] > 0)),
            ground_truth_in_candidates=sum(1 for r in valid_results if not r['candidate_info']['ground_truth_added']) / n,
            avg_candidate_rank=sum(r['candidate_info']['ground_truth_rank'] for r in valid_results) / n,
            total_samples=len(test_samples),
            valid_samples=n,
            avg_time_ms=total_time / len(test_samples),
        )

        return metrics, results

    def _get_llm_pick(
        self,
        order_history: List[Dict],
        candidates: List[str],
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> Dict[str, Any]:
        """Ask LLM to pick ONE cuisine from candidates.

        Returns:
            Dict with 'prompt', 'response', 'parsed_pick' keys
        """
        prompt = self._build_prompt(order_history, candidates, target_hour, target_day_of_week)

        response = self.llm.generate(
            prompt,
            temperature=self.config.temperature,
            max_tokens=100,
        )

        parsed_pick = self._parse_pick(response, candidates)

        return {
            'prompt': prompt,
            'response': response,
            'parsed_pick': parsed_pick,
        }

    def _build_prompt(
        self,
        order_history: List[Dict],
        candidates: List[str],
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> str:
        """Build prompt for LLM to pick one cuisine.

        Args:
            order_history: List of past orders with cuisine, hour, day_of_week
            candidates: List of cuisine options to pick from
            target_hour: Hour when user wants to order (0-23)
            target_day_of_week: Day when user wants to order (0=Mon, 6=Sun)
        """
        # Format order history
        history_lines = []
        for i, order in enumerate(order_history[-10:], 1):
            day_name = self.DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            cuisine = order.get('cuisine', 'unknown')
            vendor_id = order.get('vendor_id', '')
            if vendor_id:
                history_lines.append(f"{i}. {cuisine} from vendor {vendor_id} ({day_name} {hour}:00)")
            else:
                history_lines.append(f"{i}. {cuisine} ({day_name} {hour}:00)")
        history_str = "\n".join(history_lines)

        # Shuffle candidates to avoid position bias
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        candidates_str = ", ".join(shuffled)

        # Build target time context
        if target_hour is not None and target_day_of_week is not None:
            target_day_name = self.DAY_NAMES[target_day_of_week]
            time_context = f"\n## Prediction Context:\nThe user wants to order on {target_day_name} at {target_hour}:00.\n"
        else:
            time_context = ""

        return f"""Based on this user's order history, pick the ONE cuisine they will most likely order next.

## Order History (oldest to newest):
{history_str}
{time_context}
## Available Options:
{candidates_str}

Pick exactly ONE cuisine from the options above. Reply with just the cuisine name, nothing else."""

    def _parse_pick(self, response: str, candidates: List[str]) -> str:
        """Parse LLM response to extract cuisine pick."""
        response = response.strip().lower()

        # Direct match
        for candidate in candidates:
            if candidate.lower() == response:
                return candidate

        # Partial match
        for candidate in candidates:
            if candidate.lower() in response:
                return candidate

        # Fallback: return first candidate mentioned
        for candidate in candidates:
            if candidate.lower() in response.lower():
                return candidate

        # Last resort: return empty or first candidate
        return candidates[0] if candidates else ""


def build_test_samples(
    orders_df: pd.DataFrame,
    n_samples: int = 10,
    min_history: int = 5,
    seed: int = 42,
    prediction_target: str = "cuisine",
    return_basket: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build test samples for evaluation (Method 1: Leave-Last-Out).

    For each sample:
    - order_history: all orders except the last one
    - ground_truth_cuisine: cuisine of the last order (single item, backward compat)
    - ground_truth_items: Set of all items in last order (if return_basket=True)

    Args:
        orders_df: DataFrame with order data
        n_samples: Number of test samples to create. Use -1 for all eligible samples.
        min_history: Minimum number of orders required for a user
        seed: Random seed for reproducibility
        prediction_target: "cuisine", "vendor", or "product" level prediction
        return_basket: If True, return all items in last order as ground_truth_items (Set)
    """
    random.seed(seed)

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Map prediction target to column
    target_column = {
        'cuisine': 'cuisine',
        'vendor': 'vendor_id',
        'product': 'product_id',
    }.get(prediction_target, 'cuisine')

    # Find eligible customers
    customer_order_counts = orders_df.groupby('customer_id')['order_id'].nunique()
    eligible = customer_order_counts[customer_order_counts >= min_history].index.tolist()

    if len(eligible) == 0:
        print(f"Warning: No customers with >= {min_history} orders")
        return []

    # n_samples < 0 means use all eligible samples
    if n_samples < 0:
        sampled = eligible
        random.shuffle(sampled)  # Shuffle for consistency with seed
    else:
        sampled = random.sample(eligible, min(n_samples, len(eligible)))

    samples = []
    for customer_id in sampled:
        customer_data = orders_df[orders_df['customer_id'] == customer_id]

        # Sort by time
        if 'day_num' in customer_data.columns:
            customer_data = customer_data.sort_values(['day_num', 'hour'])

        # Get unique orders with their items
        orders = []
        order_items: Dict[Any, Set[str]] = {}  # order_id -> set of items
        seen_orders = set()

        for _, row in customer_data.iterrows():
            order_id = row['order_id']
            item = str(row.get(target_column, 'unknown'))

            # Track items per order for basket
            if order_id not in order_items:
                order_items[order_id] = set()
            order_items[order_id].add(item)

            if order_id in seen_orders:
                continue
            seen_orders.add(order_id)

            orders.append({
                'order_id': order_id,
                'vendor_id': str(row.get('vendor_id', '')),
                'cuisine': row.get('cuisine', 'unknown'),
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
            })

        if len(orders) < min_history:
            continue

        # Split: history = all but last, ground truth = last
        last_order = orders[-1]
        last_order_id = last_order['order_id']

        sample = {
            'customer_id': customer_id,
            'order_history': orders[:-1],
            'ground_truth_cuisine': last_order['cuisine'],  # Backward compat
            'target_hour': last_order['hour'],
            'target_day_of_week': last_order['day_of_week'],
        }

        if return_basket:
            # Get all items in the last order as ground truth basket
            ground_truth_basket = order_items.get(last_order_id, {last_order['cuisine']})
            sample['ground_truth_items'] = ground_truth_basket
            sample['ground_truth_primary'] = last_order['cuisine']
            sample['order_id'] = last_order_id
            sample['basket_size'] = len(ground_truth_basket)

        samples.append(sample)

    return samples


def build_test_samples_from_test_file(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prediction_target: str = "cuisine",
    seed: int = 42,
    n_samples: int = -1,
) -> List[Dict[str, Any]]:
    """
    Build test samples using full training history + test orders (Method 2).

    Each order in test_df becomes one test case.
    Ground truth = set of items (cuisines) in that order (basket).

    IMPORTANT: Only includes users that appear in BOTH train and test data.
    Cold-start users (only in test) are skipped.

    Args:
        train_df: Training data DataFrame (full history)
        test_df: Test data DataFrame (orders to predict)
        prediction_target: "cuisine", "vendor", or "product"
        seed: Random seed
        n_samples: Number of samples to return (-1 = all samples)

    Returns:
        List of samples with:
        - customer_id
        - order_history: ALL orders from training data
        - ground_truth_items: Set[str] of items in test order
        - ground_truth_primary: str - primary item (for backward compat)
        - target_hour, target_day_of_week
        - order_id (for grouping)
        - basket_size
    """
    random.seed(seed)

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Map prediction target to column
    target_column = {
        'cuisine': 'cuisine',
        'vendor': 'vendor_id',
        'product': 'product_id',
    }.get(prediction_target, 'cuisine')

    # Get users in training data
    train_users = set(train_df['customer_id'].unique())

    # Get users in test data
    test_users = set(test_df['customer_id'].unique())

    # Only process users that appear in BOTH
    valid_users = train_users & test_users
    cold_start_users = test_users - train_users

    if cold_start_users:
        print(f"[build_test_samples_from_test_file] Skipping {len(cold_start_users)} cold-start users (only in test)")

    print(f"[build_test_samples_from_test_file] Processing {len(valid_users)} users with training history")

    # Build training history for each user
    user_histories: Dict[Any, List[Dict]] = {}

    for customer_id in valid_users:
        customer_train = train_df[train_df['customer_id'] == customer_id]

        # Sort by time
        if 'day_num' in customer_train.columns:
            customer_train = customer_train.sort_values(['day_num', 'hour'])

        # Get unique orders
        orders = []
        seen_orders = set()

        for _, row in customer_train.iterrows():
            order_id = row['order_id']
            if order_id in seen_orders:
                continue
            seen_orders.add(order_id)

            orders.append({
                'order_id': order_id,
                'vendor_id': str(row.get('vendor_id', '')),
                'cuisine': row.get('cuisine', 'unknown'),
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
            })

        user_histories[customer_id] = orders

    # Build test samples from test orders
    samples = []

    # Get unique test orders
    test_order_ids = test_df['order_id'].unique()

    for test_order_id in test_order_ids:
        order_data = test_df[test_df['order_id'] == test_order_id]
        customer_id = order_data['customer_id'].iloc[0]

        # Skip cold-start users
        if customer_id not in valid_users:
            continue

        # Get all items in this test order (basket)
        ground_truth_items = set(str(item) for item in order_data[target_column].unique())

        # Get first row for metadata
        first_row = order_data.iloc[0]

        sample = {
            'customer_id': customer_id,
            'order_history': user_histories[customer_id],
            'ground_truth_items': ground_truth_items,
            'ground_truth_primary': str(first_row.get(target_column, 'unknown')),
            'ground_truth_cuisine': str(first_row.get('cuisine', 'unknown')),  # Backward compat
            'target_hour': int(first_row.get('hour', 12)),
            'target_day_of_week': int(first_row.get('day_of_week', 0)),
            'order_id': test_order_id,
            'basket_size': len(ground_truth_items),
        }

        samples.append(sample)

    # Shuffle for randomness
    random.shuffle(samples)

    # Limit samples if requested
    total_available = len(samples)
    if n_samples > 0 and n_samples < total_available:
        samples = samples[:n_samples]

    print(f"[build_test_samples_from_test_file] Created {len(samples)} test samples (from {total_available} available)")

    return samples


# =============================================================================
# ENHANCED TWO-ROUND RERANKING EVALUATION
# =============================================================================

@dataclass
class EnhancedRerankConfig:
    """Configuration for enhanced two-round rerank evaluation."""
    # Candidate generation (cuisine-to-cuisine swing)
    n_candidates: int = 20
    items_per_seed: int = 5  # Similar cuisines per history cuisine

    # LightGCN settings
    dataset_name: str = "data_se"
    lightgcn_epochs: int = 50
    lightgcn_embedding_dim: int = 64

    # LLM settings
    temperature_round1: float = 0.3
    temperature_round2: float = 0.2
    max_tokens_round1: int = 4096  # Increased to avoid truncation
    max_tokens_round2: int = 4096  # Increased to avoid truncation

    # Test settings
    n_samples: int = 10
    min_history: int = 5
    seed: int = 42

    # Evaluation method
    evaluation_method: str = "method1"  # "method1" (leave-last-out) or "method2" (train-test split)

    # Basket prediction settings
    prediction_target: str = "cuisine"  # "cuisine", "vendor", or "product"
    enable_basket_metrics: bool = True  # Multi-item evaluation
    filter_seen_items: bool = True      # Exclude items from history


@dataclass
class EnhancedRerankMetrics:
    """Metrics for enhanced two-round rerank evaluation."""
    # NDCG metrics
    ndcg_at_1: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0

    # MRR metrics
    mrr_at_1: float = 0.0
    mrr_at_3: float = 0.0
    mrr_at_5: float = 0.0
    mrr_at_10: float = 0.0

    # Hit Rate metrics (single-item)
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0

    # Round comparison
    round1_hit_at_5: float = 0.0
    final_hit_at_5: float = 0.0
    improvement: float = 0.0

    # Basket metrics (multi-item ground truth)
    basket_hit_at_1: float = 0.0
    basket_hit_at_3: float = 0.0
    basket_hit_at_5: float = 0.0
    basket_hit_at_10: float = 0.0
    basket_recall_at_1: float = 0.0
    basket_recall_at_3: float = 0.0
    basket_recall_at_5: float = 0.0
    basket_recall_at_10: float = 0.0
    basket_precision_at_1: float = 0.0
    basket_precision_at_3: float = 0.0
    basket_precision_at_5: float = 0.0
    basket_precision_at_10: float = 0.0
    basket_ndcg_at_1: float = 0.0
    basket_ndcg_at_3: float = 0.0
    basket_ndcg_at_5: float = 0.0
    basket_ndcg_at_10: float = 0.0
    basket_mrr: float = 0.0
    avg_basket_size: float = 0.0

    # Statistics
    total_samples: int = 0
    valid_samples: int = 0
    avg_time_ms: float = 0.0
    gt_in_candidates: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ndcg@1': self.ndcg_at_1,
            'ndcg@3': self.ndcg_at_3,
            'ndcg@5': self.ndcg_at_5,
            'ndcg@10': self.ndcg_at_10,
            'mrr@1': self.mrr_at_1,
            'mrr@3': self.mrr_at_3,
            'mrr@5': self.mrr_at_5,
            'mrr@10': self.mrr_at_10,
            'hit@1': self.hit_at_1,
            'hit@3': self.hit_at_3,
            'hit@5': self.hit_at_5,
            'hit@10': self.hit_at_10,
            'round1_hit@5': self.round1_hit_at_5,
            'final_hit@5': self.final_hit_at_5,
            'improvement': self.improvement,
            # Basket metrics
            'basket_hit@1': self.basket_hit_at_1,
            'basket_hit@3': self.basket_hit_at_3,
            'basket_hit@5': self.basket_hit_at_5,
            'basket_hit@10': self.basket_hit_at_10,
            'basket_recall@1': self.basket_recall_at_1,
            'basket_recall@3': self.basket_recall_at_3,
            'basket_recall@5': self.basket_recall_at_5,
            'basket_recall@10': self.basket_recall_at_10,
            'basket_precision@1': self.basket_precision_at_1,
            'basket_precision@3': self.basket_precision_at_3,
            'basket_precision@5': self.basket_precision_at_5,
            'basket_precision@10': self.basket_precision_at_10,
            'basket_ndcg@1': self.basket_ndcg_at_1,
            'basket_ndcg@3': self.basket_ndcg_at_3,
            'basket_ndcg@5': self.basket_ndcg_at_5,
            'basket_ndcg@10': self.basket_ndcg_at_10,
            'basket_mrr': self.basket_mrr,
            'avg_basket_size': self.avg_basket_size,
            # Statistics
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,
            'avg_time_ms': self.avg_time_ms,
            'gt_in_candidates': self.gt_in_candidates,
        }

    def __str__(self) -> str:
        basket_str = ""
        if self.avg_basket_size > 0:
            basket_str = f"""
  --- Basket Metrics ---
  Basket Hit@5:      {self.basket_hit_at_5:.2%}
  Basket Hit@10:     {self.basket_hit_at_10:.2%}
  Basket Recall@5:   {self.basket_recall_at_5:.4f}
  Basket Recall@10:  {self.basket_recall_at_10:.4f}
  Basket NDCG@5:     {self.basket_ndcg_at_5:.4f}
  Basket NDCG@10:    {self.basket_ndcg_at_10:.4f}
  Basket MRR:        {self.basket_mrr:.4f}
  Avg Basket Size:   {self.avg_basket_size:.2f}"""

        return f"""Enhanced Rerank Evaluation Results:
  NDCG@5:       {self.ndcg_at_5:.4f}
  NDCG@10:      {self.ndcg_at_10:.4f}
  MRR@5:        {self.mrr_at_5:.4f}
  MRR@10:       {self.mrr_at_10:.4f}
  Hit@1:        {self.hit_at_1:.2%}
  Hit@3:        {self.hit_at_3:.2%}
  Hit@5:        {self.hit_at_5:.2%}
  Hit@10:       {self.hit_at_10:.2%}
  Round1 Hit@5: {self.round1_hit_at_5:.2%}
  Final Hit@5:  {self.final_hit_at_5:.2%}
  Improvement:  {self.improvement:.2%}{basket_str}
  GT in Candidates: {self.gt_in_candidates:.2%}
  Samples: {self.valid_samples}/{self.total_samples}"""


class CuisineBasedCandidateGenerator:
    """
    Generate candidate cuisines using cuisine-to-cuisine swing similarity.

    Process:
    1. Get user's n-1 purchase history (excluding ground truth)
    2. Deduplicate to unique cuisines
    3. For each unique cuisine, get top-k similar cuisines
    4. Combine all, deduplicate by max score, return top 20
    """

    def __init__(self, config: EnhancedRerankConfig = None):
        self.config = config or EnhancedRerankConfig()
        self.swing_model = CuisineSwingMethod(CuisineSwingConfig(
            top_k=self.config.items_per_seed * 2  # Get extra for dedup
        ))
        self._fitted = False

    def fit(self, orders_df: pd.DataFrame):
        """
        Build cuisine-cuisine swing model from orders.

        Args:
            orders_df: DataFrame with customer_id, order_id, cuisine columns
        """
        # Build (user_id, cuisine) interactions
        interactions = []

        # Get unique cuisine per order (not per item)
        order_cuisines = orders_df.groupby(['customer_id', 'order_id'])['cuisine'].first()

        for (customer_id, _), cuisine in order_cuisines.items():
            interactions.append((str(customer_id), cuisine))

        print(f"[CuisineBasedGenerator] Building swing model from {len(interactions)} interactions...")
        self.swing_model.fit(interactions)
        self._fitted = True

        stats = self.swing_model.get_stats()
        print(f"[CuisineBasedGenerator] Fitted: {stats['num_cuisines']} cuisines, {stats['num_users']} users")

    def generate_candidates(
        self,
        cuisine_history: List[str],
        ground_truth: str = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate candidate cuisines from user's cuisine history.

        Args:
            cuisine_history: List of cuisines the user ordered (n-1, excluding last)
            ground_truth: The actual next cuisine (for evaluation)

        Returns:
            Tuple of (candidate_list, debug_info)
        """
        debug_info = {
            'history_cuisines': [],
            'candidates_per_seed': {},
            'ground_truth_added': False,
            'ground_truth_rank': -1,
        }

        if not cuisine_history:
            return [], debug_info

        # Deduplicate history while preserving order (most recent first could be weighted)
        unique_cuisines = list(dict.fromkeys(cuisine_history))
        debug_info['history_cuisines'] = unique_cuisines

        # Get candidates using swing similarity
        candidates_with_scores = self.swing_model.get_candidates_for_history(
            cuisine_history=unique_cuisines,
            items_per_seed=self.config.items_per_seed,
            total_candidates=self.config.n_candidates
        )

        # Store debug info
        for seed in unique_cuisines:
            similar = self.swing_model.get_similar_cuisines(seed, top_k=self.config.items_per_seed)
            debug_info['candidates_per_seed'][seed] = [(c, round(s, 4)) for c, s in similar]

        candidates = [c for c, _ in candidates_with_scores]

        # Add ground truth if not present (for fair evaluation)
        if ground_truth:
            if ground_truth in candidates:
                debug_info['ground_truth_rank'] = candidates.index(ground_truth) + 1
            else:
                # Insert at random position (not first/last)
                insert_pos = random.randint(1, max(1, len(candidates) - 1)) if candidates else 0
                candidates.insert(insert_pos, ground_truth)
                debug_info['ground_truth_added'] = True
                debug_info['ground_truth_rank'] = insert_pos + 1

                # Keep only n_candidates
                if len(candidates) > self.config.n_candidates:
                    # Remove a random non-GT candidate from the end
                    for i in range(len(candidates) - 1, -1, -1):
                        if candidates[i] != ground_truth:
                            candidates.pop(i)
                            break

        return candidates, debug_info

    def get_all_interactions(self, orders_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Extract all (user_id, cuisine) interactions from orders."""
        interactions = []
        order_cuisines = orders_df.groupby(['customer_id', 'order_id'])['cuisine'].first()
        for (customer_id, _), cuisine in order_cuisines.items():
            interactions.append((str(customer_id), cuisine))
        return interactions

    def get_stats(self) -> Dict[str, Any]:
        return {
            'fitted': self._fitted,
            'swing_stats': self.swing_model.get_stats() if self._fitted else None,
        }


class EnhancedRerankEvaluator:
    """
    Two-round LLM reranking evaluation with LightGCN reflection.

    Round 1: LLM reranks all 20 candidates based on user history
    Round 2: LLM reflects on Round 1 ranking + LightGCN signals for final ranking
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(
        self,
        llm_provider,
        candidate_generator: CuisineBasedCandidateGenerator,
        lightgcn_manager: LightGCNEmbeddingManager,
        config: EnhancedRerankConfig,
    ):
        self.llm = llm_provider
        self.generator = candidate_generator
        self.lightgcn = lightgcn_manager
        self.config = config

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Tuple[EnhancedRerankMetrics, List[Dict[str, Any]]]:
        """
        Run two-round evaluation on test samples.

        Args:
            test_samples: List with customer_id, order_history, ground_truth_cuisine
            verbose: Print progress

        Returns:
            Tuple of (metrics, detailed_results)
        """
        results = []
        total_time = 0.0

        for i, sample in enumerate(test_samples):
            if verbose:
                print(f"\n[{i+1}/{len(test_samples)}] Customer: {sample['customer_id']}")

            start_time = time.time()

            # Extract cuisine history from order history
            cuisine_history = [o.get('cuisine', 'unknown') for o in sample['order_history']]

            # Generate candidates
            candidates, candidate_info = self.generator.generate_candidates(
                cuisine_history=cuisine_history,
                ground_truth=sample['ground_truth_cuisine'],
            )

            if verbose:
                print(f"  Candidates: {len(candidates)}")
                print(f"  GT rank: {candidate_info['ground_truth_rank']}, added: {candidate_info['ground_truth_added']}")

            # Round 1: LLM reranking
            round1_result = self._round1_rerank(
                order_history=sample['order_history'],
                candidates=candidates,
                target_hour=sample.get('target_hour'),
                target_day_of_week=sample.get('target_day_of_week'),
            )

            if verbose:
                gt = sample['ground_truth_cuisine']
                r1_rank = self._find_rank(round1_result['ranking'], gt)
                print(f"  Round 1 rank: {r1_rank}")

            # Get LightGCN scores
            lightgcn_scores = self.lightgcn.get_user_cuisines_similarities(
                sample['customer_id'],
                candidates
            )

            # Round 2: Reflection
            round2_result = self._round2_reflect(
                order_history=sample['order_history'],
                round1_ranking=round1_result['ranking'],
                lightgcn_scores=lightgcn_scores,
                candidates=candidates,
            )

            if verbose:
                r2_rank = self._find_rank(round2_result['ranking'], gt)
                print(f"  Round 2 (final) rank: {r2_rank}")

            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed

            # Calculate metrics for this sample
            ground_truth = sample['ground_truth_cuisine']
            r1_rank = self._find_rank(round1_result['ranking'], ground_truth)
            final_rank = self._find_rank(round2_result['ranking'], ground_truth)

            result = {
                'sample_idx': i,
                'customer_id': sample['customer_id'],
                'ground_truth': ground_truth,
                'candidates': candidates,
                'candidate_info': candidate_info,
                # Round 1 results
                'round1_prompt': round1_result.get('prompt', ''),
                'round1_raw_response': round1_result.get('raw_response', ''),
                'round1_ranking': round1_result['ranking'],
                'round1_reasoning': round1_result.get('reasoning', ''),
                # LightGCN scores
                'lightgcn_scores': lightgcn_scores[:10],  # Top 10 for logging
                # Round 2 results
                'round2_prompt': round2_result.get('prompt', ''),
                'round2_raw_response': round2_result.get('raw_response', ''),
                'final_ranking': round2_result['ranking'],
                'final_reflection': round2_result.get('reflection', ''),
                # Metrics
                'round1_rank': r1_rank,
                'final_rank': final_rank,
                'time_ms': elapsed,
            }
            results.append(result)

            if verbose:
                print(f"  Time: {elapsed:.2f}ms")

        # Aggregate metrics
        metrics = self._compute_metrics(results, test_samples, total_time)
        return metrics, results

    def _round1_rerank(
        self,
        order_history: List[Dict],
        candidates: List[str],
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> Dict[str, Any]:
        """
        Round 1: LLM reranks all candidates based on user history.

        Returns dict with 'ranking' (list), 'reasoning' (str), 'prompt' (str), 'raw_response' (str).
        """
        prompt = self._build_round1_prompt(
            order_history, candidates, target_hour, target_day_of_week
        )

        response = self.llm.generate(
            prompt,
            temperature=self.config.temperature_round1,
            max_tokens=self.config.max_tokens_round1,
        )

        result = self._parse_round1_response(response, candidates)
        # Store full prompt and raw response for debugging/analysis
        result['prompt'] = prompt
        result['raw_response'] = response
        return result

    def _round2_reflect(
        self,
        order_history: List[Dict],
        round1_ranking: List[str],
        lightgcn_scores: List[Tuple[str, float]],
        candidates: List[str],
    ) -> Dict[str, Any]:
        """
        Round 2: LLM reflects on Round 1 + LightGCN signals.

        Returns dict with 'ranking' (list), 'reflection' (str), 'prompt' (str), 'raw_response' (str).
        """
        prompt = self._build_round2_prompt(
            order_history, round1_ranking, lightgcn_scores
        )

        response = self.llm.generate(
            prompt,
            temperature=self.config.temperature_round2,
            max_tokens=self.config.max_tokens_round2,
        )

        result = self._parse_round2_response(response, candidates)
        # Store full prompt and raw response for debugging/analysis
        result['prompt'] = prompt
        result['raw_response'] = response
        return result

    def _build_round1_prompt(
        self,
        order_history: List[Dict],
        candidates: List[str],
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> str:
        """Build Round 1 reranking prompt."""
        # Format history
        history_lines = []
        for i, order in enumerate(order_history, 1):
            day_name = self.DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            cuisine = order.get('cuisine', 'unknown')
            history_lines.append(f"{i}. {cuisine} ({day_name} {hour}:00)")

        history_str = "\n".join(history_lines)

        # Shuffle candidates for position bias reduction
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        candidates_str = ", ".join(shuffled)

        # Target time context
        time_context = ""
        if target_hour is not None and target_day_of_week is not None:
            day_name = self.DAY_NAMES[target_day_of_week]
            time_context = f"Target time: {day_name} at {target_hour}:00\n"

        return f"""Based on this user's complete order history, RE-RANK all {len(candidates)} candidate cuisines from most likely to least likely.

## Complete Order History ({len(order_history)} orders, oldest to newest):
{history_str}

## Prediction Context:
{time_context}
## Candidates to Rank:
{candidates_str}

You must rank ALL {len(candidates)} cuisines. Return your response as JSON:
{{"ranking": ["most_likely", "second_most_likely", ..., "least_likely"], "reasoning": "brief explanation"}}"""

    def _build_round2_prompt(
        self,
        order_history: List[Dict],
        round1_ranking: List[str],
        lightgcn_scores: List[Tuple[str, float]],
    ) -> str:
        """Build Round 2 reflection prompt."""
        # History summary (last 5)
        recent = order_history[-5:]
        history_summary = ", ".join([o.get('cuisine', 'unknown') for o in recent])

        # Round 1 ranking (top 10)
        r1_str = ", ".join(round1_ranking[:10])

        # LightGCN scores (top 10)
        lgcn_lines = [f"{i+1}. {c}: {s:.3f}" for i, (c, s) in enumerate(lightgcn_scores[:10])]
        lgcn_str = "\n".join(lgcn_lines)

        # LightGCN ranking
        lgcn_ranking = [c for c, _ in lightgcn_scores[:10]]
        lgcn_ranking_str = ", ".join(lgcn_ranking)

        return f"""Review the initial ranking and collaborative filtering signals to produce a FINAL ranking.

## User's Recent History (last 5):
{history_summary}

## Initial LLM Ranking (Round 1, top 10):
{r1_str}

## Collaborative Filtering Signals (LightGCN user-cuisine similarity):
{lgcn_str}

## LightGCN-based Ranking:
{lgcn_ranking_str}

Consider both your initial intuition AND the collaborative filtering signals from similar users.
Produce your FINAL ranking of all cuisines.

Return JSON: {{"final_ranking": ["most_likely", ..., "least_likely"], "reflection": "how you balanced the signals"}}"""

    def _parse_round1_response(
        self,
        response: str,
        candidates: List[str]
    ) -> Dict[str, Any]:
        """Parse Round 1 LLM response."""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[^{}]*"ranking"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                ranking = parsed.get('ranking', [])
                reasoning = parsed.get('reasoning', '')

                # Validate and fill missing
                ranking = self._validate_ranking(ranking, candidates)
                return {'ranking': ranking, 'reasoning': reasoning}
        except:
            pass

        # Fallback: extract cuisine names from response
        ranking = self._extract_cuisines_from_text(response, candidates)
        return {'ranking': ranking, 'reasoning': ''}

    def _parse_round2_response(
        self,
        response: str,
        candidates: List[str]
    ) -> Dict[str, Any]:
        """Parse Round 2 LLM response."""
        try:
            import re
            json_match = re.search(r'\{[^{}]*"final_ranking"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                ranking = parsed.get('final_ranking', [])
                reflection = parsed.get('reflection', '')

                ranking = self._validate_ranking(ranking, candidates)
                return {'ranking': ranking, 'reflection': reflection}
        except:
            pass

        ranking = self._extract_cuisines_from_text(response, candidates)
        return {'ranking': ranking, 'reflection': ''}

    def _validate_ranking(
        self,
        ranking: List[str],
        candidates: List[str]
    ) -> List[str]:
        """Validate and complete ranking with all candidates."""
        # Normalize
        ranking_lower = [r.lower().strip() for r in ranking if isinstance(r, str)]
        candidate_map = {c.lower(): c for c in candidates}

        valid_ranking = []
        seen = set()

        for r in ranking_lower:
            if r in candidate_map and r not in seen:
                valid_ranking.append(candidate_map[r])
                seen.add(r)

        # Add missing candidates at the end
        for c in candidates:
            if c.lower() not in seen:
                valid_ranking.append(c)

        return valid_ranking

    def _extract_cuisines_from_text(
        self,
        text: str,
        candidates: List[str]
    ) -> List[str]:
        """Extract cuisine mentions from text as fallback."""
        text_lower = text.lower()
        found = []
        seen = set()

        for candidate in candidates:
            if candidate.lower() in text_lower and candidate.lower() not in seen:
                found.append(candidate)
                seen.add(candidate.lower())

        # Add remaining
        for c in candidates:
            if c.lower() not in seen:
                found.append(c)

        return found

    def _find_rank(self, ranking: List[str], target: str) -> int:
        """Find rank of target in ranking (1-indexed, 0 if not found)."""
        target_lower = target.lower()
        for i, item in enumerate(ranking):
            if item.lower() == target_lower:
                return i + 1
        return 0

    def _compute_metrics(
        self,
        results: List[Dict],
        test_samples: List[Dict],
        total_time: float
    ) -> EnhancedRerankMetrics:
        """Compute aggregate metrics including basket metrics."""
        from .basket_metrics import (
            compute_basket_hit,
            compute_basket_recall,
            compute_basket_precision,
            compute_basket_ndcg,
            compute_basket_mrr,
        )

        valid = [r for r in results if r['final_rank'] > 0]
        n = len(valid)

        if n == 0:
            return EnhancedRerankMetrics(total_samples=len(test_samples))

        # NDCG helper (single-item)
        def dcg(rank, k):
            if rank <= 0 or rank > k:
                return 0.0
            return 1.0 / (math.log2(rank + 1))

        def ndcg(ranks, k):
            # Binary relevance: 1 if in top-k, 0 otherwise
            dcg_scores = [dcg(r, k) for r in ranks if r > 0]
            ideal_dcg = 1.0  # Perfect rank = 1
            return sum(dcg_scores) / (len(ranks) * ideal_dcg) if ranks else 0.0

        # MRR helper (single-item)
        def mrr(ranks, k):
            scores = [1.0/r if 0 < r <= k else 0.0 for r in ranks]
            return sum(scores) / len(scores) if scores else 0.0

        # Hit rate helper (single-item)
        def hit_rate(ranks, k):
            hits = [1 if 0 < r <= k else 0 for r in ranks]
            return sum(hits) / len(hits) if hits else 0.0

        final_ranks = [r['final_rank'] for r in valid]
        r1_ranks = [r['round1_rank'] for r in valid]

        # Compute basket metrics if samples have ground_truth_items
        basket_hit_1, basket_hit_3, basket_hit_5, basket_hit_10 = 0.0, 0.0, 0.0, 0.0
        basket_recall_1, basket_recall_3, basket_recall_5, basket_recall_10 = 0.0, 0.0, 0.0, 0.0
        basket_precision_1, basket_precision_3, basket_precision_5, basket_precision_10 = 0.0, 0.0, 0.0, 0.0
        basket_ndcg_1, basket_ndcg_3, basket_ndcg_5, basket_ndcg_10 = 0.0, 0.0, 0.0, 0.0
        basket_mrr_val = 0.0
        avg_basket_size = 0.0

        # Check if we have basket ground truth in test samples
        has_basket = any('ground_truth_items' in s for s in test_samples)

        if has_basket:
            basket_hits_1, basket_hits_3, basket_hits_5, basket_hits_10 = [], [], [], []
            basket_recalls_1, basket_recalls_3, basket_recalls_5, basket_recalls_10 = [], [], [], []
            basket_precisions_1, basket_precisions_3, basket_precisions_5, basket_precisions_10 = [], [], [], []
            basket_ndcgs_1, basket_ndcgs_3, basket_ndcgs_5, basket_ndcgs_10 = [], [], [], []
            basket_mrrs = []
            basket_sizes = []

            # Match results with test samples by customer_id
            sample_map = {s['customer_id']: s for s in test_samples}

            for r in valid:
                customer_id = r['customer_id']
                sample = sample_map.get(customer_id)

                if sample and 'ground_truth_items' in sample:
                    gt_items = sample['ground_truth_items']
                    predictions = r.get('final_ranking', [])

                    # Compute for k=1, 3, 5, 10
                    basket_hits_1.append(compute_basket_hit(predictions, gt_items, 1))
                    basket_hits_3.append(compute_basket_hit(predictions, gt_items, 3))
                    basket_hits_5.append(compute_basket_hit(predictions, gt_items, 5))
                    basket_hits_10.append(compute_basket_hit(predictions, gt_items, 10))
                    basket_recalls_1.append(compute_basket_recall(predictions, gt_items, 1))
                    basket_recalls_3.append(compute_basket_recall(predictions, gt_items, 3))
                    basket_recalls_5.append(compute_basket_recall(predictions, gt_items, 5))
                    basket_recalls_10.append(compute_basket_recall(predictions, gt_items, 10))
                    basket_precisions_1.append(compute_basket_precision(predictions, gt_items, 1))
                    basket_precisions_3.append(compute_basket_precision(predictions, gt_items, 3))
                    basket_precisions_5.append(compute_basket_precision(predictions, gt_items, 5))
                    basket_precisions_10.append(compute_basket_precision(predictions, gt_items, 10))
                    basket_ndcgs_1.append(compute_basket_ndcg(predictions, gt_items, 1))
                    basket_ndcgs_3.append(compute_basket_ndcg(predictions, gt_items, 3))
                    basket_ndcgs_5.append(compute_basket_ndcg(predictions, gt_items, 5))
                    basket_ndcgs_10.append(compute_basket_ndcg(predictions, gt_items, 10))
                    basket_mrrs.append(compute_basket_mrr(predictions, gt_items))
                    basket_sizes.append(len(gt_items))

            if basket_hits_1:
                n_basket = len(basket_hits_1)
                basket_hit_1 = sum(basket_hits_1) / n_basket
                basket_hit_3 = sum(basket_hits_3) / n_basket
                basket_hit_5 = sum(basket_hits_5) / n_basket
                basket_hit_10 = sum(basket_hits_10) / n_basket
                basket_recall_1 = sum(basket_recalls_1) / n_basket
                basket_recall_3 = sum(basket_recalls_3) / n_basket
                basket_recall_5 = sum(basket_recalls_5) / n_basket
                basket_recall_10 = sum(basket_recalls_10) / n_basket
                basket_precision_1 = sum(basket_precisions_1) / n_basket
                basket_precision_3 = sum(basket_precisions_3) / n_basket
                basket_precision_5 = sum(basket_precisions_5) / n_basket
                basket_precision_10 = sum(basket_precisions_10) / n_basket
                basket_ndcg_1 = sum(basket_ndcgs_1) / n_basket
                basket_ndcg_3 = sum(basket_ndcgs_3) / n_basket
                basket_ndcg_5 = sum(basket_ndcgs_5) / n_basket
                basket_ndcg_10 = sum(basket_ndcgs_10) / n_basket
                basket_mrr_val = sum(basket_mrrs) / n_basket
                avg_basket_size = sum(basket_sizes) / n_basket

        metrics = EnhancedRerankMetrics(
            ndcg_at_1=sum(dcg(r, 1) for r in final_ranks) / n,
            ndcg_at_3=sum(dcg(r, 3) for r in final_ranks) / n,
            ndcg_at_5=sum(dcg(r, 5) for r in final_ranks) / n,
            ndcg_at_10=sum(dcg(r, 10) for r in final_ranks) / n,
            mrr_at_1=mrr(final_ranks, 1),
            mrr_at_3=mrr(final_ranks, 3),
            mrr_at_5=mrr(final_ranks, 5),
            mrr_at_10=mrr(final_ranks, 10),
            hit_at_1=hit_rate(final_ranks, 1),
            hit_at_3=hit_rate(final_ranks, 3),
            hit_at_5=hit_rate(final_ranks, 5),
            hit_at_10=hit_rate(final_ranks, 10),
            round1_hit_at_5=hit_rate(r1_ranks, 5),
            final_hit_at_5=hit_rate(final_ranks, 5),
            improvement=hit_rate(final_ranks, 5) - hit_rate(r1_ranks, 5),
            # Basket metrics
            basket_hit_at_1=basket_hit_1,
            basket_hit_at_3=basket_hit_3,
            basket_hit_at_5=basket_hit_5,
            basket_hit_at_10=basket_hit_10,
            basket_recall_at_1=basket_recall_1,
            basket_recall_at_3=basket_recall_3,
            basket_recall_at_5=basket_recall_5,
            basket_recall_at_10=basket_recall_10,
            basket_precision_at_1=basket_precision_1,
            basket_precision_at_3=basket_precision_3,
            basket_precision_at_5=basket_precision_5,
            basket_precision_at_10=basket_precision_10,
            basket_ndcg_at_1=basket_ndcg_1,
            basket_ndcg_at_3=basket_ndcg_3,
            basket_ndcg_at_5=basket_ndcg_5,
            basket_ndcg_at_10=basket_ndcg_10,
            basket_mrr=basket_mrr_val,
            avg_basket_size=avg_basket_size,
            # Statistics
            total_samples=len(test_samples),
            valid_samples=n,
            avg_time_ms=total_time / len(test_samples) if test_samples else 0,
            gt_in_candidates=sum(1 for r in results if not r['candidate_info']['ground_truth_added']) / len(results),
        )

        return metrics
