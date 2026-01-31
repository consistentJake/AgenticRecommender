"""
Async LLM provider for parallel request processing.

Uses aiohttp for concurrent HTTP requests with semaphore-based rate limiting.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

try:
    import aiohttp
except ImportError:
    aiohttp = None


logger = logging.getLogger(__name__)


class AsyncLLMProvider:
    """
    Async LLM provider using aiohttp for concurrent requests.

    Features:
    - Semaphore-based concurrency control
    - Connection pooling
    - Exponential backoff retry
    - Qwen3 thinking mode toggle support

    Usage:
        async with AsyncLLMProvider(api_key="...", max_concurrent=10) as provider:
            result = await provider.generate("Hello!")
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "qwen/qwen3-32b"

    def __init__(
        self,
        api_key: str,
        model_name: str = None,
        max_concurrent: int = 10,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize async LLM provider.

        Args:
            api_key: OpenRouter API key
            model_name: Model to use (default: qwen/qwen3-32b)
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds (per attempt)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Base delay between retries (exponential backoff)
        """
        if aiohttp is None:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")

        self.api_key = api_key
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None

        # Metrics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.failed_calls = 0

        # Per-request timing tracking
        self.request_times: List[float] = []  # List of individual request durations (seconds)
        self.min_request_time = float('inf')
        self.max_request_time = 0.0

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AgenticRecommender",
            "X-Title": "Agentic Recommender System",
        }

    async def __aenter__(self) -> "AsyncLLMProvider":
        """Enter async context: create session with connection pooling."""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            keepalive_timeout=30,
        )
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            sock_connect=10,
            sock_read=self.timeout,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context: close session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        system_prompt: str = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text using async HTTP request.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system message
            enable_thinking: For Qwen3 models, enable/disable thinking mode
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        async with self.semaphore:
            return await self._generate_with_retry(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                **kwargs
            )

    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """Generate with exponential backoff retry."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                # asyncio.wait_for as safety net in case aiohttp timeout doesn't trigger
                return await asyncio.wait_for(
                    self._generate_single(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        enable_thinking=enable_thinking,
                        **kwargs
                    ),
                    timeout=self.timeout + 5,  # slightly longer than aiohttp timeout
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.retry_attempts}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.failed_calls += 1
                    logger.error(f"Request failed after {self.retry_attempts} attempts: {e}")

        return f"ERROR: {last_error}"

    async def _generate_single(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """Single async LLM request."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        start_time = time.time()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = prompt

        # For Qwen3 models, add thinking control suffix
        if "qwen" in self.model_name.lower() and "qwen3" in self.model_name.lower():
            if not enable_thinking:
                user_content += " /no_think"

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with self.session.post(self.BASE_URL, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

            text = ""
            if data.get("choices"):
                text = data["choices"][0]["message"]["content"]

            # Track metrics
            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration

            # Track per-request timing
            self.request_times.append(duration)
            self.min_request_time = min(self.min_request_time, duration)
            self.max_request_time = max(self.max_request_time, duration)

            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            if tokens:
                self.total_tokens += tokens
            else:
                # Estimate tokens
                self.total_tokens += len(prompt.split()) + len(text.split())

            return text.strip()

    async def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of prompts
            **kwargs: Generation parameters

        Returns:
            List of responses (same order as prompts)
        """
        tasks = [self.generate(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model and usage information including detailed timing statistics."""
        info = {
            "provider": "AsyncOpenRouter",
            "model_name": self.model_name,
            "max_concurrent": self.max_concurrent,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_time_per_call": self.total_time / max(self.total_calls, 1),
        }

        # Add detailed timing statistics if we have request times
        if self.request_times:
            sorted_times = sorted(self.request_times)
            n = len(sorted_times)

            info["timing"] = {
                "min_seconds": self.min_request_time,
                "max_seconds": self.max_request_time,
                "avg_seconds": sum(self.request_times) / n,
                "p50_seconds": sorted_times[n // 2],
                "p90_seconds": sorted_times[int(n * 0.90)] if n >= 10 else sorted_times[-1],
                "p95_seconds": sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
                "p99_seconds": sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
                "total_requests": n,
            }

        return info

    def reset_metrics(self):
        """Reset performance metrics."""
        self.total_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.request_times = []
        self.min_request_time = float('inf')
        self.max_request_time = 0.0


def create_async_provider(
    api_key: str = None,
    model_name: str = None,
    max_concurrent: int = 10,
    **kwargs
) -> AsyncLLMProvider:
    """
    Factory function to create async LLM provider.

    Args:
        api_key: OpenRouter API key (or uses OPENROUTER_API_KEY env var)
        model_name: Model to use
        max_concurrent: Maximum concurrent requests
        **kwargs: Additional provider arguments

    Returns:
        AsyncLLMProvider instance
    """
    import os

    resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_key:
        raise ValueError("API key required (pass api_key or set OPENROUTER_API_KEY)")

    return AsyncLLMProvider(
        api_key=resolved_key,
        model_name=model_name,
        max_concurrent=max_concurrent,
        **kwargs
    )
