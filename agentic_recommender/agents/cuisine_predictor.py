"""CuisinePredictorAgent — Round 1 LLM: cuisine prediction."""

import json
import logging
import re
import time
from collections import Counter
from typing import Dict, Any, List, Tuple

from .base import RecommendationAgent

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


class CuisinePredictorAgent(RecommendationAgent):
    """Round 1 LLM agent: predict top cuisines for a user's next order."""

    def __init__(self, name: str = "cuisine_predictor"):
        super().__init__(name)
        self._config = None

    def initialize(self, config) -> None:
        """Store config (RepeatEvalConfig)."""
        self._config = config
        self._initialized = True

    async def process(
        self,
        sample: Dict[str, Any],
        lightgcn_scores: List[Tuple[str, float]],
        llm_provider,
    ) -> Dict[str, Any]:
        """Run Round 1: predict top cuisines via LLM.

        Returns dict with predicted_cuisines, prompt, raw_response, error, llm_ms.
        """
        round1_prompt = self._build_round1_prompt(sample, lightgcn_scores)
        round1_error = False

        round1_start = time.time()
        try:
            round1_response = await llm_provider.generate(
                round1_prompt,
                temperature=self._config.temperature_round1,
                max_tokens=self._config.max_tokens_round1,
                enable_thinking=self._config.enable_thinking,
            )
        except Exception as e:
            round1_response = f"ERROR: {e}"
            round1_error = True
            self._logger.warning("Round 1 LLM failed for %s: %s", sample.get('customer_id'), e)
        round1_llm_ms = (time.time() - round1_start) * 1000

        if round1_response.startswith("ERROR:"):
            round1_error = True

        # Parse
        all_cuisines = list(set(
            o.get('cuisine', 'unknown') for o in sample['order_history']
        ))
        round1_cuisines = self._parse_round1_cuisine_response(
            round1_response, all_cuisines, lightgcn_scores
        )

        # Frequency ensemble
        if self._config.enable_frequency_ensemble:
            cuisine_counts = Counter(o.get('cuisine', 'unknown') for o in sample['order_history'])
            top2_freq = [c for c, _ in cuisine_counts.most_common(2)]
            for freq_cuisine in top2_freq:
                if freq_cuisine.lower() not in [c.lower() for c in round1_cuisines]:
                    round1_cuisines[-1] = freq_cuisine
                    break

        return {
            'predicted_cuisines': round1_cuisines,
            'prompt': round1_prompt,
            'raw_response': round1_response,
            'error': round1_error,
            'llm_ms': round1_llm_ms,
        }

    def _build_round1_prompt(
        self,
        sample: Dict,
        lightgcn_scores: List[Tuple[str, float]],
    ) -> str:
        """Build Round 1 prompt: predict top cuisines.

        Same logic as AsyncRepeatEvaluator._build_round1_prompt().
        """
        order_history = sample['order_history']
        top_k = self._config.round1_predict_top_k

        # Format history
        history_lines = []
        for i, order in enumerate(order_history, 1):
            day_name = DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            vid = order.get('vendor_id', '?')
            cuisine = order.get('cuisine', '?')
            history_lines.append(f"{i}. {vid}||{cuisine} ({day_name} {hour}:00)")
        history_str = "\n".join(history_lines)

        # Most recent orders highlight (last 5)
        n_recent = min(5, len(order_history))
        recent_orders = order_history[-n_recent:]
        recent_lines = []
        for i, order in enumerate(recent_orders, 1):
            day_name = DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            vid = order.get('vendor_id', '?')
            cuisine = order.get('cuisine', '?')
            recent_lines.append(f"  {i}. {vid}||{cuisine} ({day_name} {hour}:00)")
        recent_str = "\n".join(recent_lines)

        # Cuisine frequency rankings
        cuisine_counts = Counter(o.get('cuisine', 'unknown') for o in order_history)
        sorted_cuisines = cuisine_counts.most_common()
        freq_lines = [
            f"  {rank}. {cuisine} — {count}/{len(order_history)} orders ({count/len(order_history)*100:.0f}%)"
            for rank, (cuisine, count) in enumerate(sorted_cuisines, 1)
        ]
        freq_str = "\n".join(freq_lines)

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

## Most Recent Orders (last {n_recent} — these reflect the user's CURRENT preferences):
{recent_str}

## Cuisine Frequency Rankings (from entire history):
{freq_str}

## Predict for: {target_day} at {target_hour}:00

## Collaborative Filtering Scores (cuisine similarity from similar users):
{lgcn_str}

IMPORTANT: This is a REPEAT order prediction task. The user tends to reorder from places they've used before.

Consider (in priority order):
1. Cuisine frequency: cuisines ordered more often are very likely to be reordered — check the frequency rankings above
2. Recency: recent orders indicate CURRENT active preferences
3. Temporal patterns: match day-of-week and meal-time to similar past orders
4. CF scores: use as a tiebreaker when frequency and recency are similar

Return exactly {top_k} cuisines as JSON:
{{"cuisines": [{", ".join(f'"cuisine_{i+1}"' for i in range(top_k))}], "reasoning": "brief explanation"}}"""

    def _parse_round1_cuisine_response(
        self,
        response: str,
        all_cuisines: List[str],
        lightgcn_scores: List[Tuple[str, float]],
    ) -> List[str]:
        """Extract top K cuisine predictions from Round 1 response.

        Same logic as AsyncRepeatEvaluator._parse_round1_cuisine_response().
        """
        top_k = self._config.round1_predict_top_k

        try:
            json_match = re.search(r'\{[^{}]*"cuisines"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                cuisines = parsed.get('cuisines', [])
                if isinstance(cuisines, list) and len(cuisines) > 0:
                    cuisine_map = {c.lower(): c for c in all_cuisines}
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

        fallback = [c for c, _ in lightgcn_scores[:top_k]]
        return fallback if fallback else all_cuisines[:top_k]
