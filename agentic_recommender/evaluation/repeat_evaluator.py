"""
Async evaluator for repeated dataset two-round evaluation.

Round 1: Predict top 3 primary cuisines (LLM + LightGCN on customer→cuisine)
Round 2: Rank candidate vendors filtered by geohash + Round 1 cuisines
         (LLM + Swing user-user CF)

Ground truth = vendor_id||primary_cuisine
Metrics: Hit@1/3/5, NDCG@1/3/5, MRR
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    from tqdm.asyncio import tqdm as async_tqdm
except ImportError:
    async_tqdm = None

from ..llm.async_provider import AsyncLLMProvider

logger = logging.getLogger(__name__)


DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


@dataclass
class RepeatEvalConfig:
    """Configuration for repeat evaluation."""
    # Filter
    min_history_items: int = 5
    # LightGCN
    lightgcn_top_k_cuisines: int = 10
    lightgcn_epochs: int = 50
    lightgcn_embedding_dim: int = 64
    # Round 1
    round1_predict_top_k: int = 3
    temperature_round1: float = 0.3
    max_tokens_round1: int = 4096
    # Candidate selection
    max_candidate_vendors: int = 20
    # Round 2 / Swing
    top_similar_users: int = 5
    max_records_per_similar_user: int = 5
    temperature_round2: float = 0.2
    max_tokens_round2: int = 4096
    # General
    enable_thinking: bool = True
    prediction_target: str = "vendor_cuisine"
    dataset_name: str = "data_se"
    n_samples: int = 20
    deterministic_sampling: bool = True
    # Async
    enable_async: bool = True
    max_workers: int = 25
    checkpoint_interval: int = 50
    retry_attempts: int = 3


class AsyncRepeatEvaluator:
    """Async evaluator for repeated dataset two-round evaluation."""

    def __init__(
        self,
        async_provider: AsyncLLMProvider,
        lightgcn_manager,
        swing_model,
        geohash_index,
        train_df,
        config: RepeatEvalConfig,
    ):
        """
        Initialize async repeat evaluator.

        Args:
            async_provider: Async LLM provider for API calls
            lightgcn_manager: LightGCNEmbeddingManager (trained on customer→cuisine)
            swing_model: SwingMethod (trained on customer→vendor_id||cuisine)
            geohash_index: GeohashVendorIndex for candidate lookup
            train_df: Training DataFrame (for building similar user records)
            config: RepeatEvalConfig with settings
        """
        self.provider = async_provider
        self.lightgcn = lightgcn_manager
        self.swing = swing_model
        self.geohash_index = geohash_index
        self.train_df = train_df
        self.config = config

        # Pre-build user training records for similar user lookups
        self._user_records: Dict[str, List[Dict]] = {}
        self._build_user_records()

        # State
        self.results_file: Optional[Path] = None
        self._completed_ids: Set[str] = set()
        self._write_lock = asyncio.Lock()
        self._progress_count = 0
        self._start_time = 0.0

    def _build_user_records(self):
        """Pre-build training records per user for similar user lookups."""
        seen_orders: Dict[str, set] = {}

        if 'day_num' in self.train_df.columns:
            df_sorted = self.train_df.sort_values(['customer_id', 'day_num', 'hour'])
        else:
            df_sorted = self.train_df

        for _, row in df_sorted.iterrows():
            cid = str(row['customer_id'])
            oid = row['order_id']

            if cid not in seen_orders:
                seen_orders[cid] = set()
                self._user_records[cid] = []

            if oid in seen_orders[cid]:
                continue
            seen_orders[cid].add(oid)

            self._user_records[cid].append({
                'vendor_id': str(row.get('vendor_id', '')),
                'cuisine': row.get('cuisine', 'unknown'),
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
            })

    async def evaluate_async(
        self,
        test_samples: List[Dict[str, Any]],
        output_path: Path,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate samples with parallel LLM requests.

        Args:
            test_samples: List of test samples
            output_path: Directory to save results
            verbose: Print progress

        Returns:
            Dict with results list and metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.results_file = output_path / "detailed_results.jsonl"
        self._start_time = time.time()

        # Resume support
        self._completed_ids = self._load_completed_ids()
        pending = [s for s in test_samples if str(s['customer_id']) + "_" + str(s['order_id']) not in self._completed_ids]

        if verbose:
            print("")
            print("=" * 60)
            print("  ASYNC REPEAT EVALUATOR")
            print("=" * 60)
            print(f"  Total samples:     {len(test_samples)}")
            print(f"  Already completed: {len(self._completed_ids)}")
            print(f"  Pending:           {len(pending)}")
            print(f"  Concurrent workers: {self.config.max_workers}")
            print(f"  Results file:      {self.results_file}")
            print("=" * 60)
            print("")

        if not pending:
            print("[AsyncRepeatEvaluator] All samples already completed.")
            return self._read_all_results()

        # Process pending samples
        async with self.provider:
            queue = asyncio.Queue()
            for i, sample in enumerate(pending):
                await queue.put((i, sample))

            workers = [
                asyncio.create_task(self._worker(queue, worker_id, len(pending), verbose))
                for worker_id in range(min(self.config.max_workers, len(pending)))
            ]

            if verbose and async_tqdm:
                with async_tqdm(total=len(pending), desc="Evaluating") as pbar:
                    prev_count = 0
                    while self._progress_count < len(pending):
                        await asyncio.sleep(0.5)
                        new_count = self._progress_count
                        if new_count > prev_count:
                            pbar.update(new_count - prev_count)
                            prev_count = new_count
            else:
                await queue.join()

            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        elapsed = time.time() - self._start_time
        if verbose:
            print("")
            print("=" * 60)
            print("  ASYNC REPEAT EVALUATION COMPLETE")
            print("=" * 60)
            print(f"  Total time:    {elapsed:.1f}s")
            print(f"  Samples done:  {len(pending)}")
            if len(pending) > 0:
                print(f"  Throughput:    {len(pending) / elapsed:.2f} samples/sec")
                print(f"  Avg per sample: {elapsed / len(pending) * 1000:.1f}ms")
            print("=" * 60)

            model_info = self.provider.get_model_info()
            print("")
            print("-" * 60)
            print("  LLM REQUEST TIMING STATISTICS")
            print("-" * 60)
            print(f"  Total LLM calls:   {model_info.get('total_calls', 0)}")
            print(f"  Failed calls:      {model_info.get('failed_calls', 0)}")
            if 'timing' in model_info:
                timing = model_info['timing']
                print(f"  Avg request time:  {timing['avg_seconds']:.2f}s")
                print(f"  P50 (median):      {timing['p50_seconds']:.2f}s")
                print(f"  P95:               {timing['p95_seconds']:.2f}s")
            print("-" * 60)

        return self._read_all_results()

    async def _worker(self, queue, worker_id, total, verbose):
        """Worker that processes samples from queue."""
        while True:
            try:
                idx, sample = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                result = await self._process_sample(idx, sample, verbose)
                await self._write_result(result)
                self._progress_count += 1

                if verbose and self._progress_count % self.config.checkpoint_interval == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._progress_count / elapsed
                    remaining = total - self._progress_count
                    eta = remaining / rate if rate > 0 else 0
                    print(
                        f"[Worker Pool] Progress: {self._progress_count}/{total} "
                        f"({self._progress_count/total*100:.1f}%) | "
                        f"Rate: {rate:.2f} samples/sec | "
                        f"ETA: {eta:.0f}s"
                    )
            except Exception as e:
                logger.error(f"Worker {worker_id} failed on {sample.get('customer_id')}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                queue.task_done()

    async def _process_sample(
        self,
        idx: int,
        sample: Dict[str, Any],
        verbose: bool,
    ) -> Dict[str, Any]:
        """Process single sample through two-round pipeline."""
        start_time = time.time()
        customer_id = str(sample['customer_id'])

        # Get LightGCN top cuisines for this user
        lightgcn_top_cuisines = self.lightgcn.get_top_cuisines_for_user(
            customer_id,
            top_k=self.config.lightgcn_top_k_cuisines,
        )

        # Round 1: Predict top cuisines
        round1_prompt = self._build_round1_prompt(sample, lightgcn_top_cuisines)

        round1_response = await self.provider.generate(
            round1_prompt,
            temperature=self.config.temperature_round1,
            max_tokens=self.config.max_tokens_round1,
            enable_thinking=self.config.enable_thinking,
        )

        # Parse Round 1
        all_cuisines = list(set(
            o.get('cuisine', 'unknown') for o in sample['order_history']
        ))
        round1_cuisines = self._parse_round1_cuisine_response(
            round1_response, all_cuisines, lightgcn_top_cuisines
        )

        # Candidate selection: geohash lookup with top cuisines
        target_geohash = sample.get('target_vendor_geohash', 'unknown')
        candidate_vendor_ids = self.geohash_index.get_vendors(
            target_geohash,
            round1_cuisines,
            max_candidates=self.config.max_candidate_vendors,
        )

        # Build candidate list as vendor_id||cuisine
        candidate_vendors = []
        for vid in candidate_vendor_ids:
            cuisine = self.geohash_index.get_vendor_cuisine(vid)
            candidate_vendors.append(f"{vid}||{cuisine}" if cuisine else vid)

        # Get similar users from Swing
        similar_users_info = self._get_similar_users_records(
            customer_id, round1_cuisines
        )

        # Round 2: Rank candidate vendors
        round2_prompt = self._build_round2_prompt(
            sample, round1_cuisines, candidate_vendors, similar_users_info
        )

        round2_response = await self.provider.generate(
            round2_prompt,
            temperature=self.config.temperature_round2,
            max_tokens=self.config.max_tokens_round2,
            enable_thinking=self.config.enable_thinking,
        )

        # Parse Round 2
        final_ranking = self._parse_round2_vendor_response(
            round2_response, candidate_vendors
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Find ground truth rank
        ground_truth = sample['ground_truth']
        ground_truth_rank = self._find_rank(final_ranking, ground_truth)

        return {
            'sample_idx': idx,
            'customer_id': customer_id,
            'order_id': str(sample.get('order_id', '')),
            'ground_truth': ground_truth,
            'ground_truth_vendor_id': sample.get('ground_truth_vendor_id', ''),
            'ground_truth_cuisine': sample.get('ground_truth_cuisine', ''),
            'target_hour': sample.get('target_hour'),
            'target_day_of_week': sample.get('target_day_of_week'),
            'target_vendor_geohash': target_geohash,

            'lightgcn_top_cuisines': lightgcn_top_cuisines[:10],

            'round1_prompt': round1_prompt,
            'round1_raw_response': round1_response,
            'round1_predicted_cuisines': round1_cuisines,

            'candidate_vendors': candidate_vendors,
            'candidate_count': len(candidate_vendors),

            'similar_users': [
                {'user_id': u['user_id'], 'similarity': u['similarity'],
                 'record_count': len(u['records'])}
                for u in similar_users_info
            ],

            'round2_prompt': round2_prompt,
            'round2_raw_response': round2_response,
            'final_ranking': final_ranking,

            'ground_truth_rank': ground_truth_rank,
            'time_ms': elapsed_ms,
        }

    def _build_round1_prompt(
        self,
        sample: Dict,
        lightgcn_scores: List[Tuple[str, float]],
    ) -> str:
        """Build Round 1 prompt: predict top cuisines."""
        import random as _random

        order_history = sample['order_history']
        top_k = self.config.round1_predict_top_k

        # Format history
        history_lines = []
        for i, order in enumerate(order_history, 1):
            day_name = DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            vid = order.get('vendor_id', '?')
            cuisine = order.get('cuisine', '?')
            history_lines.append(f"{i}. {vid}||{cuisine} ({day_name} {hour}:00)")

        history_str = "\n".join(history_lines)

        # Target time
        target_day = DAY_NAMES[sample.get('target_day_of_week', 0)]
        target_hour = sample.get('target_hour', 12)

        # LightGCN cuisine scores
        lgcn_lines = [f"{i+1}. {c}: {s:.3f}" for i, (c, s) in enumerate(lightgcn_scores[:10])]
        lgcn_str = "\n".join(lgcn_lines)

        return f"""You are a food delivery recommendation system. Based on this user's order history, predict the top {top_k} most likely PRIMARY CUISINES for their next order.

## Order History ({len(order_history)} orders, oldest to newest):
Each entry is: vendor_id||cuisine (day_of_week time)
{history_str}

## Predict for: {target_day} at {target_hour}:00

## Collaborative Filtering Scores (cuisine similarity from similar users):
{lgcn_str}

Consider:
- Temporal patterns: day-of-week and meal-time preferences
- Cuisine frequency: cuisines ordered more often are more likely
- CF scores: higher scores indicate cuisines popular among similar users
- Recency: recent cuisine choices may reflect current preferences

Return exactly {top_k} cuisines as JSON:
{{"cuisines": ["most_likely_cuisine", "second_most_likely", "third_most_likely"], "reasoning": "brief explanation"}}"""

    def _build_round2_prompt(
        self,
        sample: Dict,
        round1_cuisines: List[str],
        candidate_vendors: List[str],
        similar_users_info: List[Dict],
    ) -> str:
        """Build Round 2 prompt: rank candidate vendors."""
        import random as _random

        order_history = sample['order_history']

        # Format history (last 10 for brevity)
        recent = order_history[-10:]
        history_lines = []
        for i, order in enumerate(recent, 1):
            day_name = DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            vid = order.get('vendor_id', '?')
            cuisine = order.get('cuisine', '?')
            history_lines.append(f"{i}. {vid}||{cuisine} ({day_name} {hour}:00)")
        history_str = "\n".join(history_lines)

        # Target time
        target_day = DAY_NAMES[sample.get('target_day_of_week', 0)]
        target_hour = sample.get('target_hour', 12)

        # Round 1 cuisines
        cuisines_str = ", ".join(round1_cuisines)

        # Shuffle candidates to reduce position bias
        shuffled_candidates = candidate_vendors.copy()
        _random.shuffle(shuffled_candidates)
        candidates_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(shuffled_candidates)])

        # Similar users info
        similar_lines = []
        for user_info in similar_users_info:
            sim_score = user_info['similarity']
            for rec in user_info['records']:
                vid = rec.get('vendor_id', '?')
                cuisine = rec.get('cuisine', '?')
                day_name = DAY_NAMES[rec.get('day_of_week', 0)]
                hour = rec.get('hour', 12)
                similar_lines.append(f"  - {vid}||{cuisine} ({day_name} {hour}:00) [sim={sim_score:.3f}]")

        similar_str = "\n".join(similar_lines) if similar_lines else "  (no similar user records available)"

        return f"""You are a food delivery recommendation system. Rank the candidate vendors from most to least likely for this user's next order.

## User's Recent Order History (last {len(recent)} orders):
Each entry is: vendor_id||cuisine (day_of_week time)
{history_str}

## Predict for: {target_day} at {target_hour}:00

## Round 1 Predicted Cuisines (in order of likelihood):
{cuisines_str}

## Candidate Vendors to Rank:
Each candidate is: vendor_id||cuisine
{candidates_str}

## Similar Users' Recent Orders (collaborative filtering):
{similar_str}

Consider:
- Vendor loyalty: vendors the user has ordered from before are more likely
- Round 1 cuisine predictions: vendors matching top cuisines should rank higher
- Temporal patterns: day-of-week and meal-time preferences
- Similar users: vendors popular among similar users may be good choices
- Rank ALL {len(shuffled_candidates)} candidates

Return JSON:
{{"final_ranking": ["vendor_id||cuisine", ...], "reflection": "brief reasoning"}}"""

    def _parse_round1_cuisine_response(
        self,
        response: str,
        all_cuisines: List[str],
        lightgcn_scores: List[Tuple[str, float]],
    ) -> List[str]:
        """Extract top K cuisine predictions from Round 1 response."""
        import re

        top_k = self.config.round1_predict_top_k

        try:
            json_match = re.search(r'\{[^{}]*"cuisines"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                cuisines = parsed.get('cuisines', [])
                if isinstance(cuisines, list) and len(cuisines) > 0:
                    # Validate against known cuisines (case-insensitive)
                    cuisine_map = {c.lower(): c for c in all_cuisines}
                    # Also include LightGCN cuisines
                    for c, _ in lightgcn_scores:
                        cuisine_map[c.lower()] = c

                    valid = []
                    seen = set()
                    for c in cuisines:
                        c_lower = c.strip().lower()
                        if c_lower in cuisine_map and c_lower not in seen:
                            valid.append(cuisine_map[c_lower])
                            seen.add(c_lower)
                    if valid:
                        return valid[:top_k]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: use top cuisines from LightGCN
        fallback = [c for c, _ in lightgcn_scores[:top_k]]
        return fallback if fallback else all_cuisines[:top_k]

    def _parse_round2_vendor_response(
        self,
        response: str,
        candidates: List[str],
    ) -> List[str]:
        """Extract vendor ranking from Round 2 response."""
        import re

        try:
            json_match = re.search(r'\{[^{}]*"final_ranking"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                ranking = parsed.get('final_ranking', [])
                if isinstance(ranking, list):
                    ranking = self._validate_ranking(ranking, candidates)
                    return ranking
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: try to extract vendor mentions from text
        return self._extract_items_from_text(response, candidates)

    def _validate_ranking(
        self,
        ranking: List[str],
        candidates: List[str],
    ) -> List[str]:
        """Validate and complete ranking with all candidates."""
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

    def _extract_items_from_text(
        self,
        text: str,
        candidates: List[str],
    ) -> List[str]:
        """Extract candidate mentions from text as fallback."""
        text_lower = text.lower()
        found = []
        seen = set()

        for candidate in candidates:
            if candidate.lower() in text_lower and candidate.lower() not in seen:
                found.append(candidate)
                seen.add(candidate.lower())

        for c in candidates:
            if c.lower() not in seen:
                found.append(c)

        return found

    def _get_similar_users_records(
        self,
        user_id: str,
        top_cuisines: List[str],
    ) -> List[Dict]:
        """
        Get similar users and their training records filtered by top cuisines.

        Args:
            user_id: Query user
            top_cuisines: Cuisines to filter records by

        Returns:
            List of dicts with user_id, similarity, records
        """
        cuisine_set = set(c.lower() for c in top_cuisines)
        similar_users = self.swing.get_top_similar_users(
            user_id, top_k=self.config.top_similar_users
        )

        result = []
        for sim_uid, sim_score in similar_users:
            records = self._user_records.get(sim_uid, [])
            # Filter by top cuisines
            filtered = [
                r for r in records
                if r.get('cuisine', '').lower() in cuisine_set
            ]
            # Limit records per user
            filtered = filtered[-self.config.max_records_per_similar_user:]
            result.append({
                'user_id': sim_uid,
                'similarity': sim_score,
                'records': filtered,
            })

        return result

    def _find_rank(self, ranking: List[str], target: str) -> int:
        """Find rank of target in ranking (1-indexed, 0 if not found)."""
        target_lower = target.lower()
        for i, item in enumerate(ranking):
            if item.lower() == target_lower:
                return i + 1
        return 0

    def _load_completed_ids(self) -> Set[str]:
        """Load already completed sample IDs from JSONL file."""
        completed = set()
        if self.results_file and self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            cid = str(result.get('customer_id', ''))
                            oid = str(result.get('order_id', ''))
                            completed.add(f"{cid}_{oid}")
            except Exception as e:
                logger.warning(f"Error loading completed IDs: {e}")
        return completed

    async def _write_result(self, result: Dict[str, Any]):
        """Append result to JSONL file."""
        async with self._write_lock:
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

    def _read_all_results(self) -> Dict[str, Any]:
        """Read all results from JSONL file."""
        results = []
        if self.results_file and self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        return {
            'results': results,
            'completed_samples': len(results),
            'results_file': str(self.results_file),
        }


def compute_repeat_metrics(
    detailed_results: List[Dict[str, Any]],
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compute Hit@K, NDCG@K, MRR metrics from detailed results.

    Args:
        detailed_results: List of per-sample result dicts
        k_values: K values for metrics (default: [1, 3, 5])

    Returns:
        Dict with metrics
    """
    if k_values is None:
        k_values = [1, 3, 5]

    if not detailed_results:
        return {'error': 'No results to compute metrics from'}

    metrics = {}

    # Filter to samples that had candidates
    valid_results = [r for r in detailed_results if r.get('candidate_count', 0) > 0]
    total = len(detailed_results)
    valid = len(valid_results)

    metrics['total_samples'] = total
    metrics['valid_samples'] = valid
    metrics['no_candidate_samples'] = total - valid

    if valid == 0:
        for k in k_values:
            metrics[f'hit@{k}'] = 0.0
            metrics[f'ndcg@{k}'] = 0.0
        metrics['mrr'] = 0.0
        return metrics

    # Compute metrics
    ranks = [r.get('ground_truth_rank', 0) for r in valid_results]

    for k in k_values:
        # Hit@K: 1 if ground truth is in top K
        hits = sum(1 for rank in ranks if 0 < rank <= k)
        metrics[f'hit@{k}'] = hits / valid

        # NDCG@K
        ndcg_sum = sum(
            1.0 / math.log2(rank + 1) if 0 < rank <= k else 0.0
            for rank in ranks
        )
        metrics[f'ndcg@{k}'] = ndcg_sum / valid

    # MRR
    rr_sum = sum(
        1.0 / rank if rank > 0 else 0.0
        for rank in ranks
    )
    metrics['mrr'] = rr_sum / valid

    # Additional stats
    found = [r for r in ranks if r > 0]
    metrics['ground_truth_found_rate'] = len(found) / valid if valid > 0 else 0.0
    metrics['avg_rank_when_found'] = sum(found) / len(found) if found else 0.0
    metrics['avg_candidates'] = sum(
        r.get('candidate_count', 0) for r in valid_results
    ) / valid
    metrics['avg_time_ms'] = sum(
        r.get('time_ms', 0) for r in detailed_results
    ) / total

    return metrics
