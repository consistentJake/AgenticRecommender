"""RecommendationManager — central coordinator for the two-round pipeline."""

import logging
import time
from typing import Dict, Any, List, Tuple

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


class RecommendationManager:
    """Central coordinator for the two-round recommendation pipeline.

    Replaces the inline _process_sample() in AsyncRepeatEvaluator.
    """

    def __init__(self, user_profiler, vendor_profiler, similarity, cuisine_predictor, vendor_ranker):
        self.user_profiler = user_profiler
        self.vendor_profiler = vendor_profiler
        self.similarity = similarity
        self.cuisine_predictor = cuisine_predictor
        self.vendor_ranker = vendor_ranker
        self._logger = logging.getLogger("agents.manager")

    async def process_sample(
        self,
        idx: int,
        sample: Dict[str, Any],
        llm_provider,
        config,
    ) -> Dict[str, Any]:
        """Orchestrate one sample through the pipeline.

        Flow (matching design doc's Agent Collaboration Flow):
        1. SimilarityAgent -> LightGCN top cuisines
        2. CuisinePredictorAgent -> Round 1: predict cuisines (LLM)
        3. VendorProfilerAgent -> candidate vendors (geohash + R1 cuisines)
        4. UserProfilerAgent -> similar users' records (filtered by R1 cuisines)
        5. VendorRankerAgent -> Round 2: rank vendors (LLM)
        6. Build result dict with ranks, metrics, diagnostics

        Returns the same result dict structure as the original _process_sample().
        """
        start_time = time.time()
        customer_id = str(sample['customer_id'])

        # 1. Get LightGCN top cuisines (from pre-computed cache or live)
        lookup = self.user_profiler.get_lookup(customer_id)
        if lookup:
            lightgcn_top_cuisines = lookup['lightgcn_top_cuisines']
        else:
            lightgcn_top_cuisines = self.similarity.process(
                customer_id, top_k=config.lightgcn_top_k_cuisines
            )

        # 2. Round 1: Predict top cuisines
        round1_result = await self.cuisine_predictor.process(
            sample, lightgcn_top_cuisines, llm_provider
        )
        round1_cuisines = round1_result['predicted_cuisines']

        # 3. Candidate selection: filter vendors by geohash + predicted cuisines
        target_geohash = sample.get('target_vendor_geohash', 'unknown')
        candidate_vendors = self.vendor_profiler.process(
            target_geohash, round1_cuisines, max_candidates=config.max_candidate_vendors
        )

        # 4. Get similar users' records filtered by Round 1 cuisines
        if lookup:
            similar_users_info = self.user_profiler.process(
                lookup['similar_users'],
                lookup['similar_users_full_records'],
                round1_cuisines,
                config,
            )
        else:
            # Fallback: compute on the fly (shouldn't normally happen with precompute)
            similar_users = self.user_profiler.swing.get_top_similar_users(
                customer_id, top_k=config.top_similar_users
            )
            sim_records = {
                sim_uid: self.user_profiler._user_records.get(sim_uid, [])
                for sim_uid, _ in similar_users
            }
            similar_users_info = self.user_profiler.process(
                similar_users, sim_records, round1_cuisines, config
            )

        # 5. Round 2: Rank candidate vendors
        round2_result = await self.vendor_ranker.process(
            sample, round1_cuisines, candidate_vendors, similar_users_info, llm_provider
        )
        final_ranking = round2_result['final_ranking']

        elapsed_ms = (time.time() - start_time) * 1000

        # 6. Build result dict (same structure as original _process_sample)
        ground_truth = sample['ground_truth']
        ground_truth_rank = self._find_rank(final_ranking, ground_truth)

        ground_truth_cuisine = sample.get('ground_truth_cuisine', '').lower()
        round1_gt_cuisine_rank = 0
        for i, c in enumerate(round1_cuisines):
            if c.lower() == ground_truth_cuisine:
                round1_gt_cuisine_rank = i + 1
                break

        lightgcn_gt_cuisine_rank = 0
        for i, (cuisine, score) in enumerate(lightgcn_top_cuisines[:10]):
            if cuisine.lower() == ground_truth_cuisine:
                lightgcn_gt_cuisine_rank = i + 1
                break

        # Build order history tuples
        history_tuples = [
            (o.get('vendor_id', ''), o.get('cuisine', ''),
             f"{DAY_NAMES[o.get('day_of_week', 0)]} {o.get('hour', 12)}:00")
            for o in sample['order_history']
        ]

        # Build similar user detail tuples
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

        has_error = round1_result['error'] or round2_result['error']

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

            'round1_prompt': round1_result['prompt'],
            'round1_raw_response': round1_result['raw_response'],
            'round1_predicted_cuisines': round1_cuisines,
            'round1_llm_ms': round1_result['llm_ms'],
            'round1_error': round1_result['error'],

            'candidate_vendors': candidate_vendors,
            'candidate_count': len(candidate_vendors),

            'similar_users': similar_users_detail,

            'round2_prompt': round2_result['prompt'],
            'round2_raw_response': round2_result['raw_response'],
            'final_ranking': final_ranking,
            'round2_llm_ms': round2_result['llm_ms'],
            'round2_error': round2_result['error'],

            'ground_truth_rank': ground_truth_rank,
            'round1_ground_truth_cuisine_rank': round1_gt_cuisine_rank,
            'lightgcn_ground_truth_cuisine_rank': lightgcn_gt_cuisine_rank,
            'time_ms': elapsed_ms,
            'error': has_error,
        }

    @staticmethod
    def _find_rank(ranking: List[str], target: str) -> int:
        """Find rank of target in ranking (1-indexed, 0 if not found)."""
        target_lower = target.lower()
        for i, item in enumerate(ranking):
            if item.lower() == target_lower:
                return i + 1
        return 0
