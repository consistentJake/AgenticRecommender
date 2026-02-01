"""
Async evaluator for repeated dataset two-round evaluation.

Round 1: Predict top 3 primary cuisines (LLM + LightGCN on customer→cuisine)
Round 2: Rank candidate vendors filtered by geohash + Round 1 cuisines
         (LLM + Swing user-user CF)

Ground truth = vendor_id||primary_cuisine
Metrics: Hit@1/3/5, NDCG@1/3/5, MRR
"""

import asyncio
import hashlib
import json
import logging
import math
import pickle
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

USER_RECORDS_CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "repeat_user_records"
USER_LOOKUPS_CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "repeat_user_lookups"


def _hash_df(df) -> str:
    """Compute a hash key from DataFrame shape and content summary."""
    key_parts = [
        str(df.shape),
        str(df['customer_id'].nunique()),
        str(df['order_id'].nunique()),
    ]
    if 'vendor_id' in df.columns:
        key_parts.append(str(df['vendor_id'].nunique()))
    return hashlib.md5("_".join(key_parts).encode()).hexdigest()


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
    max_total_similar_records: int = 20  # Cap total records across all similar users
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
    # Request timeout (per LLM attempt, in seconds)
    request_timeout: float = 30.0


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
        use_cache: bool = True,
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
            use_cache: Whether to use disk cache for user records and lookups
        """
        self.provider = async_provider
        self.lightgcn = lightgcn_manager
        self.swing = swing_model
        self.geohash_index = geohash_index
        self.train_df = train_df
        self.config = config
        self._use_cache = use_cache

        # Pre-build user training records for similar user lookups
        self._user_records: Dict[str, List[Dict]] = {}
        self._build_user_records()

        # Pre-computed per-user lookups (populated by precompute_user_lookups)
        self._user_lookups: Optional[Dict[str, Dict]] = None

        # State
        self.results_file: Optional[Path] = None
        self._detailed_json_path: Optional[Path] = None
        self._all_results: List[Dict] = []
        self._completed_ids: Set[str] = set()
        self._write_lock = asyncio.Lock()
        self._progress_count = 0
        self._start_time = 0.0

    def _build_user_records(self):
        """Pre-build training records per user for similar user lookups."""
        dataset_name = self.config.dataset_name

        # Try loading from cache
        if self._use_cache and dataset_name:
            df_key = _hash_df(self.train_df)
            cache_path = USER_RECORDS_CACHE_DIR / f"{dataset_name}_user_records.pkl"
            try:
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if data.get('cache_key') == df_key:
                        self._user_records = data['user_records']
                        logger.info(
                            f"Loaded user records from cache: {len(self._user_records)} users "
                            f"(key={df_key[:8]}...)"
                        )
                        print(
                            f"[AsyncRepeatEvaluator] Loaded user records from cache: "
                            f"{len(self._user_records)} users (key={df_key[:8]}...)"
                        )
                        return
            except Exception as e:
                logger.warning(f"Failed to load user records cache: {e}")

        # Build from scratch
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

        # Save to cache
        if self._use_cache and dataset_name:
            try:
                USER_RECORDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path = USER_RECORDS_CACHE_DIR / f"{dataset_name}_user_records.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'cache_key': df_key,
                        'user_records': self._user_records,
                    }, f)
                logger.info(f"Saved user records cache: {len(self._user_records)} users → {cache_path}")
                print(f"[AsyncRepeatEvaluator] Saved user records cache: {len(self._user_records)} users")
            except Exception as e:
                logger.warning(f"Failed to save user records cache: {e}")

    def precompute_user_lookups(self, test_samples: List[Dict[str, Any]]) -> None:
        """Pre-compute LightGCN + Swing lookups for all test users.

        Caches LightGCN top cuisines, Swing similar users, and their full
        (unfiltered) records per test user. The Round 1 cuisine filtering
        still happens at runtime via _filter_similar_users_by_cuisines().

        Args:
            test_samples: List of test sample dicts (need customer_id)
        """
        dataset_name = self.config.dataset_name
        user_ids = sorted(set(str(s['customer_id']) for s in test_samples))

        # Try loading from cache
        if self._use_cache and dataset_name:
            cache_key_parts = [
                "_".join(user_ids),
                str(self.config.lightgcn_top_k_cuisines),
                str(self.config.top_similar_users),
                str(self.lightgcn.cache_key or ""),
            ]
            cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()
            cache_path = USER_LOOKUPS_CACHE_DIR / f"{dataset_name}_user_lookups.pkl"

            try:
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if data.get('cache_key') == cache_key:
                        self._user_lookups = data['lookups']
                        logger.info(
                            f"Loaded user lookups from cache: {len(self._user_lookups)} users "
                            f"(key={cache_key[:8]}...)"
                        )
                        print(
                            f"[AsyncRepeatEvaluator] Loaded user lookups from cache: "
                            f"{len(self._user_lookups)} users (key={cache_key[:8]}...)"
                        )
                        return
            except Exception as e:
                logger.warning(f"Failed to load user lookups cache: {e}")

        # Build from scratch
        print(f"[AsyncRepeatEvaluator] Pre-computing lookups for {len(user_ids)} users...")
        lookups = {}
        for i, uid in enumerate(user_ids):
            top_cuisines = self.lightgcn.get_top_cuisines_for_user(
                uid, top_k=self.config.lightgcn_top_k_cuisines
            )
            similar_users = self.swing.get_top_similar_users(
                uid, top_k=self.config.top_similar_users
            )
            sim_records = {
                sim_uid: self._user_records.get(sim_uid, [])
                for sim_uid, _ in similar_users
            }
            lookups[uid] = {
                'lightgcn_top_cuisines': top_cuisines,
                'similar_users': similar_users,
                'similar_users_full_records': sim_records,
            }

            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(user_ids)} users")

        self._user_lookups = lookups
        print(f"[AsyncRepeatEvaluator] Pre-computed lookups for {len(lookups)} users")

        # Save to cache
        if self._use_cache and dataset_name:
            try:
                USER_LOOKUPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'cache_key': cache_key,
                        'lookups': lookups,
                    }, f)
                logger.info(f"Saved user lookups cache: {len(lookups)} users → {cache_path}")
                print(f"[AsyncRepeatEvaluator] Saved user lookups cache: {len(lookups)} users")
            except Exception as e:
                logger.warning(f"Failed to save user lookups cache: {e}")

    def _filter_similar_users_by_cuisines(
        self,
        similar_users: List[Tuple[str, float]],
        similar_users_full_records: Dict[str, List[Dict]],
        top_cuisines: List[str],
    ) -> List[Dict]:
        """
        Filter pre-fetched similar users' records by Round 1 predicted cuisines.

        Same logic as _get_similar_users_records() but operates on pre-fetched
        data instead of calling Swing/user_records directly.

        Args:
            similar_users: List of (user_id, similarity_score) from Swing
            similar_users_full_records: Dict of user_id → full order records
            top_cuisines: Cuisines from Round 1 to filter by

        Returns:
            List of dicts with user_id, similarity, records
        """
        cuisine_set = set(c.lower() for c in top_cuisines)
        result = []
        total_records = 0
        max_total = self.config.max_total_similar_records

        for sim_uid, sim_score in similar_users:
            records = similar_users_full_records.get(sim_uid, [])
            filtered = [
                r for r in records
                if r.get('cuisine', '').lower() in cuisine_set
            ]
            filtered = filtered[-self.config.max_records_per_similar_user:]
            remaining = max_total - total_records
            if remaining <= 0:
                break
            if len(filtered) > remaining:
                filtered = filtered[-remaining:]
            total_records += len(filtered)
            result.append({
                'user_id': sim_uid,
                'similarity': sim_score,
                'records': filtered,
            })

        return result

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
        self._detailed_json_path = output_path / "stage9_repeat_detailed.json"
        self._start_time = time.time()

        # Resume support
        self._completed_ids = self._load_completed_ids()
        # Load existing results into memory for incremental JSON writing
        self._all_results = self._load_all_results_from_jsonl()
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

            if 'failure_diagnostics' in model_info:
                diag = model_info['failure_diagnostics']
                print("")
                print("-" * 60)
                print("  FAILURE DIAGNOSTICS")
                print("-" * 60)
                print(f"  Total retry failures: {diag['total_failures']}")
                print(f"  By error type:     {diag['by_error_type']}")
                if diag['by_http_status']:
                    print(f"  By HTTP status:    {diag['by_http_status']}")
                ps = diag['prompt_size_stats']
                print(f"  Prompt size (chars): min={ps['min_chars']}  max={ps['max_chars']}  "
                      f"avg={ps['avg_chars']}  median={ps['median_chars']}")
                at = diag['attempt_elapsed_stats']
                print(f"  Time at failure:   min={at['min_s']}s  max={at['max_s']}s  "
                      f"avg={at['avg_s']}s  timeouts={at['timeouts_count']}")
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

        # Get LightGCN top cuisines for this user (from pre-computed cache or live)
        lookup = self._user_lookups.get(customer_id) if self._user_lookups else None
        if lookup:
            lightgcn_top_cuisines = lookup['lightgcn_top_cuisines']
        else:
            lightgcn_top_cuisines = self.lightgcn.get_top_cuisines_for_user(
                customer_id,
                top_k=self.config.lightgcn_top_k_cuisines,
            )

        # Round 1: Predict top cuisines
        round1_prompt = self._build_round1_prompt(sample, lightgcn_top_cuisines)
        round1_error = False

        round1_start = time.time()
        try:
            round1_response = await self.provider.generate(
                round1_prompt,
                temperature=self.config.temperature_round1,
                max_tokens=self.config.max_tokens_round1,
                enable_thinking=self.config.enable_thinking,
            )
        except Exception as e:
            round1_response = f"ERROR: {e}"
            round1_error = True
            logger.warning(f"Round 1 LLM failed for {customer_id}: {e}")
        round1_llm_ms = (time.time() - round1_start) * 1000

        # Check if provider returned an ERROR string (all retries exhausted)
        if round1_response.startswith("ERROR:"):
            round1_error = True

        # Parse Round 1
        all_cuisines = list(set(
            o.get('cuisine', 'unknown') for o in sample['order_history']
        ))
        round1_cuisines = self._parse_round1_cuisine_response(
            round1_response, all_cuisines, lightgcn_top_cuisines
        )

        # Candidate selection: filter vendors by geohash + predicted cuisines.
        # We use vendor_geohash to scope candidates to the same delivery area,
        # mirroring a real app scenario where we only recommend vendors that
        # can actually deliver to the user's location.
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

        # Get similar users' records filtered by Round 1 cuisines
        if lookup:
            similar_users_info = self._filter_similar_users_by_cuisines(
                lookup['similar_users'],
                lookup['similar_users_full_records'],
                round1_cuisines,
            )
        else:
            similar_users_info = self._get_similar_users_records(
                customer_id, round1_cuisines
            )

        # Round 2: Rank candidate vendors
        round2_prompt = self._build_round2_prompt(
            sample, round1_cuisines, candidate_vendors, similar_users_info
        )
        round2_error = False

        round2_start = time.time()
        try:
            round2_response = await self.provider.generate(
                round2_prompt,
                temperature=self.config.temperature_round2,
                max_tokens=self.config.max_tokens_round2,
                enable_thinking=self.config.enable_thinking,
            )
        except Exception as e:
            round2_response = f"ERROR: {e}"
            round2_error = True
            logger.warning(f"Round 2 LLM failed for {customer_id}: {e}")
        round2_llm_ms = (time.time() - round2_start) * 1000

        # Check if provider returned an ERROR string (all retries exhausted)
        if round2_response.startswith("ERROR:"):
            round2_error = True

        # Parse Round 2
        final_ranking = self._parse_round2_vendor_response(
            round2_response, candidate_vendors
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Find ground truth rank
        ground_truth = sample['ground_truth']
        ground_truth_rank = self._find_rank(final_ranking, ground_truth)

        # Round 1 cuisine rank: position of GT cuisine in round1_predicted_cuisines
        ground_truth_cuisine = sample.get('ground_truth_cuisine', '').lower()
        round1_gt_cuisine_rank = 0
        for i, c in enumerate(round1_cuisines):
            if c.lower() == ground_truth_cuisine:
                round1_gt_cuisine_rank = i + 1
                break

        # LightGCN cuisine rank: position of GT cuisine in lightgcn_top_cuisines
        lightgcn_gt_cuisine_rank = 0
        for i, (cuisine, score) in enumerate(lightgcn_top_cuisines[:10]):
            if cuisine.lower() == ground_truth_cuisine:
                lightgcn_gt_cuisine_rank = i + 1
                break

        # Build order history tuples: (vendor_id, cuisine, "DayName HH:00")
        history_tuples = [
            (o.get('vendor_id', ''), o.get('cuisine', ''),
             f"{DAY_NAMES[o.get('day_of_week', 0)]} {o.get('hour', 12)}:00")
            for o in sample['order_history']
        ]

        # Build full similar user records with tuples
        similar_users_detail = []
        for u in similar_users_info:
            record_tuples = [
                (r.get('vendor_id', ''), r.get('cuisine', ''),
                 f"{DAY_NAMES[r.get('day_of_week', 0)]} {r.get('hour', 12)}:00")
                for r in u['records']
            ]
            similar_users_detail.append({
                'user_id': u['user_id'],
                'similarity': u['similarity'],
                'record_count': len(u['records']),
                'records': record_tuples,
            })

        # Mark as error if either round failed completely
        has_error = round1_error or round2_error

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

            'order_history_tuples': history_tuples,

            'round1_prompt': round1_prompt,
            'round1_raw_response': round1_response,
            'round1_predicted_cuisines': round1_cuisines,
            'round1_llm_ms': round1_llm_ms,
            'round1_error': round1_error,

            'candidate_vendors': candidate_vendors,
            'candidate_count': len(candidate_vendors),

            'similar_users': similar_users_detail,

            'round2_prompt': round2_prompt,
            'round2_raw_response': round2_response,
            'final_ranking': final_ranking,
            'round2_llm_ms': round2_llm_ms,
            'round2_error': round2_error,

            'ground_truth_rank': ground_truth_rank,
            'round1_ground_truth_cuisine_rank': round1_gt_cuisine_rank,
            'lightgcn_ground_truth_cuisine_rank': lightgcn_gt_cuisine_rank,
            'time_ms': elapsed_ms,
            'error': has_error,
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
{{"cuisines": [{", ".join(f'"cuisine_{i+1}"' for i in range(top_k))}], "reasoning": "brief explanation"}}"""

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
        total_records = 0
        max_total = self.config.max_total_similar_records

        for sim_uid, sim_score in similar_users:
            records = self._user_records.get(sim_uid, [])
            # Filter by top cuisines
            filtered = [
                r for r in records
                if r.get('cuisine', '').lower() in cuisine_set
            ]
            # Limit records per user
            filtered = filtered[-self.config.max_records_per_similar_user:]
            # Enforce total cap across all similar users
            remaining = max_total - total_records
            if remaining <= 0:
                break
            if len(filtered) > remaining:
                filtered = filtered[-remaining:]
            total_records += len(filtered)
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
        """Append result to JSONL file and update detailed JSON."""
        async with self._write_lock:
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')
            # Keep in-memory list for incremental detailed JSON
            self._all_results.append(result)
            # Write full detailed JSON after each sample
            if self._detailed_json_path:
                with open(self._detailed_json_path, 'w') as f:
                    json.dump(self._all_results, f, indent=2, default=str)

    def _load_all_results_from_jsonl(self) -> List[Dict]:
        """Load all existing results from JSONL into memory."""
        results = []
        if self.results_file and self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        return results

    def _read_all_results(self) -> Dict[str, Any]:
        """Read all results from in-memory list."""
        return {
            'results': self._all_results,
            'completed_samples': len(self._all_results),
            'results_file': str(self.results_file),
        }


def _compute_ranking_metrics(
    ranks: List[int],
    valid: int,
    k_values: List[int],
    prefix: str = '',
) -> Dict[str, float]:
    """
    Compute Hit@K, NDCG@K, MRR from a list of ranks.

    Args:
        ranks: List of 1-indexed ranks (0 = not found)
        valid: Number of valid samples (denominator)
        k_values: K values for Hit@K and NDCG@K
        prefix: Prefix for metric keys (e.g. 'round1_')

    Returns:
        Dict of metric_name → value
    """
    m: Dict[str, float] = {}

    if valid == 0:
        for k in k_values:
            m[f'{prefix}hit@{k}'] = 0.0
            m[f'{prefix}ndcg@{k}'] = 0.0
        m[f'{prefix}mrr'] = 0.0
        return m

    for k in k_values:
        hits = sum(1 for rank in ranks if 0 < rank <= k)
        m[f'{prefix}hit@{k}'] = hits / valid

        ndcg_sum = sum(
            1.0 / math.log2(rank + 1) if 0 < rank <= k else 0.0
            for rank in ranks
        )
        m[f'{prefix}ndcg@{k}'] = ndcg_sum / valid

    rr_sum = sum(1.0 / rank if rank > 0 else 0.0 for rank in ranks)
    m[f'{prefix}mrr'] = rr_sum / valid

    return m


def compute_repeat_metrics(
    detailed_results: List[Dict[str, Any]],
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compute Hit@K, NDCG@K, MRR metrics from detailed results.

    Computes three sets of metrics:
    - round1_*: GT cuisine rank in Round 1 predicted cuisines
    - lightgcn_*: GT cuisine rank in LightGCN top cuisines
    - (unprefixed): GT vendor rank in Round 2 final ranking (backward-compat)

    Also computes improvement deltas between rounds.

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

    metrics: Dict[str, Any] = {}

    # Filter out error samples, then filter to samples that had candidates
    non_error = [r for r in detailed_results if not r.get('error', False)]
    error_count = len(detailed_results) - len(non_error)
    valid_results = [r for r in non_error if r.get('candidate_count', 0) > 0]
    total = len(non_error)
    valid = len(valid_results)

    metrics['total_samples'] = total
    metrics['valid_samples'] = valid
    metrics['no_candidate_samples'] = total - valid
    metrics['error_samples'] = error_count

    if valid == 0:
        for prefix in ['round1_', 'lightgcn_', '']:
            for k in k_values:
                metrics[f'{prefix}hit@{k}'] = 0.0
                metrics[f'{prefix}ndcg@{k}'] = 0.0
            metrics[f'{prefix}mrr'] = 0.0
        return metrics

    # Round 1 cuisine metrics
    round1_ranks = [r.get('round1_ground_truth_cuisine_rank', 0) for r in valid_results]
    metrics.update(_compute_ranking_metrics(round1_ranks, valid, k_values, prefix='round1_'))

    # LightGCN cuisine metrics
    lightgcn_ranks = [r.get('lightgcn_ground_truth_cuisine_rank', 0) for r in valid_results]
    metrics.update(_compute_ranking_metrics(lightgcn_ranks, valid, k_values, prefix='lightgcn_'))

    # Round 2 vendor metrics (backward-compatible, unprefixed)
    final_ranks = [r.get('ground_truth_rank', 0) for r in valid_results]
    metrics.update(_compute_ranking_metrics(final_ranks, valid, k_values, prefix=''))

    # Improvement deltas (final vs round1 / lightgcn) for each k
    for k in k_values:
        metrics[f'improvement_r1_to_final_hit@{k}'] = (
            metrics[f'hit@{k}'] - metrics[f'round1_hit@{k}']
        )
        metrics[f'improvement_lgcn_to_final_hit@{k}'] = (
            metrics[f'hit@{k}'] - metrics[f'lightgcn_hit@{k}']
        )

    # Additional stats (backward-compat)
    found = [r for r in final_ranks if r > 0]
    metrics['ground_truth_found_rate'] = len(found) / valid if valid > 0 else 0.0
    metrics['avg_rank_when_found'] = sum(found) / len(found) if found else 0.0
    metrics['avg_candidates'] = sum(
        r.get('candidate_count', 0) for r in valid_results
    ) / valid
    metrics['avg_time_ms'] = sum(
        r.get('time_ms', 0) for r in non_error
    ) / total if total > 0 else 0.0

    return metrics
