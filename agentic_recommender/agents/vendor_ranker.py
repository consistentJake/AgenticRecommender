"""VendorRankerAgent — Round 2 LLM (Critic): vendor ranking."""

import json
import logging
import re
import time
from typing import Dict, Any, List

from .base import RecommendationAgent

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


class VendorRankerAgent(RecommendationAgent):
    """Round 2 LLM agent: rank candidate vendors for a user's next order."""

    def __init__(self, name: str = "vendor_ranker"):
        super().__init__(name)
        self._config = None

    def initialize(self, config) -> None:
        """Store config (RepeatEvalConfig)."""
        self._config = config
        self._initialized = True

    async def process(
        self,
        sample: Dict[str, Any],
        round1_cuisines: List[str],
        candidate_vendors: List[str],
        similar_users_info: List[Dict],
        llm_provider,
    ) -> Dict[str, Any]:
        """Run Round 2: rank candidate vendors via LLM.

        Returns dict with final_ranking, prompt, raw_response, error, llm_ms.
        """
        round2_prompt = self._build_round2_prompt(
            sample, round1_cuisines, candidate_vendors, similar_users_info
        )
        round2_error = False

        round2_start = time.time()
        try:
            round2_response = await llm_provider.generate(
                round2_prompt,
                temperature=self._config.temperature_round2,
                max_tokens=self._config.max_tokens_round2,
                enable_thinking=self._config.enable_thinking_round2,
            )
        except Exception as e:
            round2_response = f"ERROR: {e}"
            round2_error = True
            self._logger.warning("Round 2 LLM failed for %s: %s", sample.get('customer_id'), e)
        round2_llm_ms = (time.time() - round2_start) * 1000

        if round2_response.startswith("ERROR:"):
            round2_error = True

        # Parse
        final_ranking = self._parse_round2_vendor_response(
            round2_response, candidate_vendors
        )

        return {
            'final_ranking': final_ranking,
            'prompt': round2_prompt,
            'raw_response': round2_response,
            'error': round2_error,
            'llm_ms': round2_llm_ms,
        }

    def _build_round2_prompt(
        self,
        sample: Dict,
        round1_cuisines: List[str],
        candidate_vendors: List[str],
        similar_users_info: List[Dict],
    ) -> str:
        """Build Round 2 prompt: rank candidate vendors.

        Same logic as AsyncRepeatEvaluator._build_round2_prompt().
        """
        import random as _random

        order_history = sample['order_history']

        # Format history (last 10 for brevity), mark most recent 3
        recent = order_history[-10:]
        n_recent_mark = min(3, len(recent))
        history_lines = []
        for i, order in enumerate(recent, 1):
            day_name = DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            vid = order.get('vendor_id', '?')
            cuisine = order.get('cuisine', '?')
            recency_mark = " [MOST RECENT]" if i > len(recent) - n_recent_mark else ""
            history_lines.append(f"{i}. {vid}||{cuisine} ({day_name} {hour}:00){recency_mark}")
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

IMPORTANT: This is a REPEAT order prediction task — the user is expected to reorder from a vendor they've used before. Vendors from their most recent orders are the strongest candidates.

## User's Recent Order History (last {len(recent)} orders, oldest to newest):
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

Consider (in priority order):
1. Recent vendor reuse: vendors the user ordered from MOST RECENTLY are much more likely to be reordered — prioritize the [MOST RECENT] orders heavily
2. Vendor loyalty: vendors ordered from multiple times are strong candidates
3. Round 1 cuisine predictions: among recently-used vendors, prefer those matching top cuisines
4. Temporal patterns: match day-of-week and meal-time to similar past orders
5. Similar users: use as a tiebreaker for vendors not in the user's history
6. Rank ALL {len(shuffled_candidates)} candidates

Return JSON:
{{"final_ranking": ["vendor_id||cuisine", ...], "reflection": "brief reasoning"}}"""

    def _parse_round2_vendor_response(
        self,
        response: str,
        candidates: List[str],
    ) -> List[str]:
        """Extract vendor ranking from Round 2 response.

        Same logic as AsyncRepeatEvaluator._parse_round2_vendor_response().
        """
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

        return self._extract_items_from_text(response, candidates)

    def _validate_ranking(self, ranking: List[str], candidates: List[str]) -> List[str]:
        """Validate and complete ranking with all candidates."""
        ranking_lower = [r.lower().strip() for r in ranking if isinstance(r, str)]
        candidate_map = {c.lower(): c for c in candidates}

        valid_ranking = []
        seen = set()
        for r in ranking_lower:
            if r in candidate_map and r not in seen:
                valid_ranking.append(candidate_map[r])
                seen.add(r)

        for c in candidates:
            if c.lower() not in seen:
                valid_ranking.append(c)

        return valid_ranking

    def _extract_items_from_text(self, text: str, candidates: List[str]) -> List[str]:
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
