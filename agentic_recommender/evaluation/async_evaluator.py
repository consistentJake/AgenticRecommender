"""
Async parallel evaluator for enhanced rerank evaluation.

Features:
- Worker pool for concurrent LLM requests
- JSONL streaming to disk (memory efficient)
- Resume support (skip completed samples)
- Progress tracking with tqdm
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

try:
    from tqdm.asyncio import tqdm as async_tqdm
except ImportError:
    async_tqdm = None

from ..llm.async_provider import AsyncLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class AsyncEvalConfig:
    """Configuration for async evaluation."""
    max_workers: int = 10          # Concurrent LLM requests
    checkpoint_interval: int = 50   # Log progress every N samples
    retry_attempts: int = 3         # Retries per failed request
    retry_delay: float = 1.0        # Base delay between retries

    # LLM settings
    temperature_round1: float = 0.3
    temperature_round2: float = 0.2
    max_tokens_round1: int = 4096
    max_tokens_round2: int = 4096
    enable_thinking: bool = False

    # Prediction target
    prediction_target: str = "cuisine"


class AsyncRerankEvaluator:
    """
    Async evaluator for enhanced two-round rerank evaluation.

    Processes samples in parallel using worker pool.
    Results stream to JSONL file for memory efficiency.
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(
        self,
        async_provider: AsyncLLMProvider,
        candidate_generator,
        lightgcn_manager,
        config: AsyncEvalConfig,
    ):
        """
        Initialize async evaluator.

        Args:
            async_provider: Async LLM provider for API calls
            candidate_generator: CuisineBasedCandidateGenerator instance
            lightgcn_manager: LightGCNEmbeddingManager instance
            config: AsyncEvalConfig with settings
        """
        self.provider = async_provider
        self.generator = candidate_generator
        self.lightgcn = lightgcn_manager
        self.config = config

        # State
        self.results_file: Optional[Path] = None
        self._completed_ids: Set[str] = set()
        self._write_lock = asyncio.Lock()
        self._progress_count = 0
        self._start_time = 0.0

    async def evaluate_async(
        self,
        test_samples: List[Dict[str, Any]],
        output_path: Path,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate samples with parallel LLM requests.

        Results stream to JSONL file, not held in memory.

        Args:
            test_samples: List of test samples
            output_path: Directory to save results
            verbose: Print progress

        Returns:
            Dict with metrics and stats
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.results_file = output_path / "detailed_results.jsonl"
        self._start_time = time.time()

        # Resume support: load already completed sample IDs
        self._completed_ids = self._load_completed_ids()
        pending = [s for s in test_samples if str(s['customer_id']) not in self._completed_ids]

        if verbose:
            print("")
            print("=" * 60)
            print("  ASYNC PARALLEL EVALUATOR")
            print("=" * 60)
            print(f"  Total samples:     {len(test_samples)}")
            print(f"  Already completed: {len(self._completed_ids)}")
            print(f"  Pending:           {len(pending)}")
            print(f"  Concurrent workers: {self.config.max_workers}")
            print(f"  Results file:      {self.results_file}")
            print("=" * 60)
            print("")

        if not pending:
            print("[AsyncEvaluator] All samples already completed. Reading from file.")
            return self._compute_metrics_from_file(test_samples)

        # Process pending samples
        async with self.provider:
            # Create task queue
            queue = asyncio.Queue()
            for i, sample in enumerate(pending):
                await queue.put((i, sample))

            # Spawn workers
            workers = [
                asyncio.create_task(self._worker(queue, worker_id, len(pending), verbose))
                for worker_id in range(min(self.config.max_workers, len(pending)))
            ]

            # Progress monitoring
            if verbose and async_tqdm:
                # Use tqdm for progress
                with async_tqdm(total=len(pending), desc="Evaluating") as pbar:
                    prev_count = 0
                    # Wait until all items processed (progress_count equals total)
                    # Note: Can't check workers.done() - they run forever until cancelled
                    while self._progress_count < len(pending):
                        await asyncio.sleep(0.5)
                        new_count = self._progress_count
                        if new_count > prev_count:
                            pbar.update(new_count - prev_count)
                            prev_count = new_count
            else:
                # Wait for completion
                await queue.join()

            # Cancel workers
            for w in workers:
                w.cancel()

            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

        elapsed = time.time() - self._start_time
        if verbose:
            print("")
            print("=" * 60)
            print("  ASYNC EVALUATION COMPLETE")
            print("=" * 60)
            print(f"  Total time:    {elapsed:.1f}s")
            print(f"  Samples done:  {len(pending)}")
            print(f"  Throughput:    {len(pending) / elapsed:.2f} samples/sec")
            print(f"  Avg per sample: {elapsed / len(pending) * 1000:.1f}ms")
            print(f"  Speedup vs sequential: ~{self.config.max_workers}x (with {self.config.max_workers} workers)")
            print("=" * 60)

            # Print LLM request timing statistics
            model_info = self.provider.get_model_info()
            print("")
            print("-" * 60)
            print("  LLM REQUEST TIMING STATISTICS")
            print("-" * 60)
            print(f"  Total LLM calls:   {model_info.get('total_calls', 0)}")
            print(f"  Failed calls:      {model_info.get('failed_calls', 0)}")
            print(f"  Total tokens:      {model_info.get('total_tokens', 0)}")

            if 'timing' in model_info:
                timing = model_info['timing']
                print(f"  Min request time:  {timing['min_seconds']:.2f}s")
                print(f"  Max request time:  {timing['max_seconds']:.2f}s")
                print(f"  Avg request time:  {timing['avg_seconds']:.2f}s")
                print(f"  P50 (median):      {timing['p50_seconds']:.2f}s")
                print(f"  P90:               {timing['p90_seconds']:.2f}s")
                print(f"  P95:               {timing['p95_seconds']:.2f}s")
                print(f"  P99:               {timing['p99_seconds']:.2f}s")
            print("-" * 60)

        # Compute final metrics from disk
        return self._compute_metrics_from_file(test_samples)

    async def _worker(
        self,
        queue: asyncio.Queue,
        worker_id: int,
        total: int,
        verbose: bool,
    ):
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

                # Periodic progress log
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
            finally:
                queue.task_done()

    async def _process_sample(
        self,
        idx: int,
        sample: Dict[str, Any],
        verbose: bool,
    ) -> Dict[str, Any]:
        """Process single sample: candidate gen + round1 + lightgcn + round2."""
        start_time = time.time()

        # Extract item history from order history based on prediction_target
        if self.config.prediction_target == 'vendor_cuisine':
            cuisine_history = [
                o.get('item', f"{o.get('vendor_id', 'unknown')}||{o.get('cuisine', 'unknown')}")
                for o in sample['order_history']
            ]
        else:
            cuisine_history = [o.get('cuisine', 'unknown') for o in sample['order_history']]

        # Generate candidates (sync, fast)
        candidates, candidate_info = self.generator.generate_candidates(
            cuisine_history=cuisine_history,
            ground_truth=sample['ground_truth_cuisine'],
        )

        # Round 1 LLM call (async)
        round1_result = await self._async_round1(
            order_history=sample['order_history'],
            candidates=candidates,
            target_hour=sample.get('target_hour'),
            target_day_of_week=sample.get('target_day_of_week'),
        )

        # Get LightGCN scores (sync, fast - just embedding lookups)
        lightgcn_scores = self.lightgcn.get_user_cuisines_similarities(
            sample['customer_id'],
            candidates
        )
        lightgcn_ranking = [c for c, _ in lightgcn_scores]

        # Round 2 LLM call (async)
        round2_result = await self._async_round2(
            order_history=sample['order_history'],
            round1_ranking=round1_result['ranking'],
            lightgcn_scores=lightgcn_scores,
            candidates=candidates,
        )

        elapsed = (time.time() - start_time) * 1000

        # Calculate ranks
        ground_truth = sample['ground_truth_cuisine']
        r1_rank = self._find_rank(round1_result['ranking'], ground_truth)
        lightgcn_rank = self._find_rank(lightgcn_ranking, ground_truth)
        final_rank = self._find_rank(round2_result['ranking'], ground_truth)

        return {
            'sample_idx': idx,
            'customer_id': str(sample['customer_id']),
            'ground_truth': ground_truth,
            'candidates': candidates,
            'candidate_info': candidate_info,
            # Round 1 results
            'round1_prompt': round1_result.get('prompt', ''),
            'round1_raw_response': round1_result.get('raw_response', ''),
            'round1_ranking': round1_result['ranking'],
            'round1_reasoning': round1_result.get('reasoning', ''),
            # LightGCN results
            'lightgcn_scores': lightgcn_scores[:10],
            'lightgcn_ranking': lightgcn_ranking,
            'lightgcn_rank': lightgcn_rank,
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

    async def _async_round1(
        self,
        order_history: List[Dict],
        candidates: List[str],
        target_hour: int = None,
        target_day_of_week: int = None,
    ) -> Dict[str, Any]:
        """Round 1: LLM reranks candidates."""
        prompt = self._build_round1_prompt(
            order_history, candidates, target_hour, target_day_of_week
        )

        response = await self.provider.generate(
            prompt,
            temperature=self.config.temperature_round1,
            max_tokens=self.config.max_tokens_round1,
            enable_thinking=self.config.enable_thinking,
        )

        result = self._parse_round1_response(response, candidates)
        result['prompt'] = prompt
        result['raw_response'] = response
        return result

    async def _async_round2(
        self,
        order_history: List[Dict],
        round1_ranking: List[str],
        lightgcn_scores: List[tuple],
        candidates: List[str],
    ) -> Dict[str, Any]:
        """Round 2: LLM reflects on Round 1 + LightGCN signals."""
        prompt = self._build_round2_prompt(
            order_history, round1_ranking, lightgcn_scores
        )

        response = await self.provider.generate(
            prompt,
            temperature=self.config.temperature_round2,
            max_tokens=self.config.max_tokens_round2,
            enable_thinking=self.config.enable_thinking,
        )

        result = self._parse_round2_response(response, candidates)
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
        import random

        # Format history
        history_lines = []
        for i, order in enumerate(order_history, 1):
            day_name = self.DAY_NAMES[order.get('day_of_week', 0)]
            hour = order.get('hour', 12)
            if self.config.prediction_target == 'vendor_cuisine':
                item = order.get('item', f"{order.get('vendor_id', 'unknown')}||{order.get('cuisine', 'unknown')}")
                history_lines.append(f"{i}. {item}||({day_name}, {hour})")
            else:
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
        lightgcn_scores: List[tuple],
    ) -> str:
        """Build Round 2 reflection prompt."""
        # History summary (last 5)
        recent = order_history[-5:]
        if self.config.prediction_target == 'vendor_cuisine':
            history_summary = ", ".join([
                f"{o.get('item', o.get('vendor_id','?')+'||'+o.get('cuisine','?'))}||({self.DAY_NAMES[o.get('day_of_week',0)]}, {o.get('hour',12)})"
                for o in recent
            ])
        else:
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
        import re

        try:
            json_match = re.search(r'\{[^{}]*"ranking"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                ranking = parsed.get('ranking', [])
                reasoning = parsed.get('reasoning', '')
                ranking = self._validate_ranking(ranking, candidates)
                return {'ranking': ranking, 'reasoning': reasoning}
        except:
            pass

        ranking = self._extract_cuisines_from_text(response, candidates)
        return {'ranking': ranking, 'reasoning': ''}

    def _parse_round2_response(
        self,
        response: str,
        candidates: List[str]
    ) -> Dict[str, Any]:
        """Parse Round 2 LLM response."""
        import re

        try:
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

    def _load_completed_ids(self) -> Set[str]:
        """Load already completed sample IDs from JSONL file."""
        completed = set()
        if self.results_file and self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            completed.add(str(result.get('customer_id', '')))
            except Exception as e:
                logger.warning(f"Error loading completed IDs: {e}")
        return completed

    async def _write_result(self, result: Dict[str, Any]):
        """Append result to JSONL file (thread-safe via lock)."""
        async with self._write_lock:
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

    def _compute_metrics_from_file(
        self,
        test_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Read JSONL file and compute aggregate metrics."""
        results = []
        if self.results_file and self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

        # Import metrics computation from rerank_eval
        # We'll return raw results and let the caller compute metrics
        return {
            'results': results,
            'total_samples': len(test_samples),
            'completed_samples': len(results),
            'results_file': str(self.results_file),
        }


async def run_async_evaluation(
    test_samples: List[Dict[str, Any]],
    candidate_generator,
    lightgcn_manager,
    output_path: Path,
    api_key: str,
    model_name: str = None,
    max_workers: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run async evaluation.

    Args:
        test_samples: Test samples to evaluate
        candidate_generator: CuisineBasedCandidateGenerator instance
        lightgcn_manager: LightGCNEmbeddingManager instance
        output_path: Directory to save results
        api_key: OpenRouter API key
        model_name: Model to use
        max_workers: Maximum concurrent requests
        **kwargs: Additional config options

    Returns:
        Dict with results and metrics
    """
    from ..llm.async_provider import AsyncLLMProvider

    config = AsyncEvalConfig(
        max_workers=max_workers,
        temperature_round1=kwargs.get('temperature_round1', 0.3),
        temperature_round2=kwargs.get('temperature_round2', 0.2),
        max_tokens_round1=kwargs.get('max_tokens_round1', 4096),
        max_tokens_round2=kwargs.get('max_tokens_round2', 4096),
        enable_thinking=kwargs.get('enable_thinking', False),
        prediction_target=kwargs.get('prediction_target', 'cuisine'),
    )

    provider = AsyncLLMProvider(
        api_key=api_key,
        model_name=model_name,
        max_concurrent=max_workers,
    )

    evaluator = AsyncRerankEvaluator(
        async_provider=provider,
        candidate_generator=candidate_generator,
        lightgcn_manager=lightgcn_manager,
        config=config,
    )

    return await evaluator.evaluate_async(
        test_samples=test_samples,
        output_path=output_path,
        verbose=True,
    )
