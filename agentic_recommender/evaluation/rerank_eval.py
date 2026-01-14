"""
Retrieve-then-Rerank Evaluation for Cuisine Prediction.

Two-stage approach:
1. RETRIEVE: Generate candidate cuisines using user history + similar users (CF)
2. RERANK: LLM picks from candidates K times, evaluate if ground truth is picked

This is more realistic than asking LLM to pick from all 78 cuisines.
"""

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd

from ..similarity.methods import SwingMethod, CosineMethod, SwingConfig, CosineConfig


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
            for k in range(self.config.k_picks):
                pick = self._get_llm_pick(
                    order_history=sample['order_history'],
                    candidates=candidates,
                    target_hour=sample.get('target_hour'),
                    target_day_of_week=sample.get('target_day_of_week'),
                )
                picks.append(pick)

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
    ) -> str:
        """Ask LLM to pick ONE cuisine from candidates."""
        prompt = self._build_prompt(order_history, candidates, target_hour, target_day_of_week)

        response = self.llm.generate(
            prompt,
            temperature=self.config.temperature,
            max_tokens=100,
        )

        return self._parse_pick(response, candidates)

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
) -> List[Dict[str, Any]]:
    """
    Build test samples for evaluation.

    For each sample:
    - order_history: all orders except the last one
    - ground_truth_cuisine: cuisine of the last order
    """
    random.seed(seed)

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Find eligible customers
    customer_order_counts = orders_df.groupby('customer_id')['order_id'].nunique()
    eligible = customer_order_counts[customer_order_counts >= min_history].index.tolist()

    if len(eligible) == 0:
        print(f"Warning: No customers with >= {min_history} orders")
        return []

    sampled = random.sample(eligible, min(n_samples, len(eligible)))

    samples = []
    for customer_id in sampled:
        customer_data = orders_df[orders_df['customer_id'] == customer_id]

        # Sort by time
        if 'day_num' in customer_data.columns:
            customer_data = customer_data.sort_values(['day_num', 'hour'])

        # Get unique orders
        orders = []
        seen_orders = set()
        for _, row in customer_data.iterrows():
            order_id = row['order_id']
            if order_id in seen_orders:
                continue
            seen_orders.add(order_id)

            orders.append({
                'order_id': order_id,
                'cuisine': row.get('cuisine', 'unknown'),
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
            })

        if len(orders) < min_history:
            continue

        # Split: history = all but last, ground truth = last
        # Include target hour/weekday from the ground truth order for prediction context
        last_order = orders[-1]
        samples.append({
            'customer_id': customer_id,
            'order_history': orders[:-1],
            'ground_truth_cuisine': last_order['cuisine'],
            'target_hour': last_order['hour'],
            'target_day_of_week': last_order['day_of_week'],
        })

    return samples
