"""
Integration test for OpenAI-compatible provider (e.g., Alibaba DashScope).

Requires DASHSCOPE_API_KEY env var to be set. Skips otherwise.
"""

import asyncio
import os
from pathlib import Path

import pytest
import yaml


DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "agentic_recommender"
    / "workflow"
    / "workflow_config_qwen32_linux_sg_dashscope.yaml"
)


def _load_llm_config():
    """Load the openai_compatible section from the DashScope config."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["llm"]["openai_compatible"]


@pytest.mark.skipif(
    not DASHSCOPE_API_KEY,
    reason="DASHSCOPE_API_KEY env var not set",
)
def test_dashscope_generate():
    """Make a simple generate() call via DashScope and verify response."""
    from agentic_recommender.llm.async_provider import AsyncLLMProvider

    oc = _load_llm_config()

    async def _run():
        provider = AsyncLLMProvider(
            api_key=DASHSCOPE_API_KEY,
            model_name=oc["model_name"],
            base_url=oc["base_url"],
            max_concurrent=1,
            timeout=30.0,
            retry_attempts=2,
        )
        async with provider as p:
            response = await p.generate("Who are you? Reply in one sentence.")
        return response, provider

    response, provider = asyncio.run(_run())

    assert isinstance(response, str)
    assert len(response) > 0
    assert not response.startswith("ERROR:")

    info = provider.get_model_info()
    assert "AsyncOpenAICompatible" in info["provider"]
    assert info["total_calls"] == 1


@pytest.mark.skipif(
    not DASHSCOPE_API_KEY,
    reason="DASHSCOPE_API_KEY env var not set",
)
def test_dashscope_headers_no_openrouter_fields():
    """Verify OpenRouter-specific headers are not sent for DashScope."""
    from agentic_recommender.llm.async_provider import AsyncLLMProvider

    oc = _load_llm_config()
    provider = AsyncLLMProvider(
        api_key=DASHSCOPE_API_KEY,
        model_name=oc["model_name"],
        base_url=oc["base_url"],
    )

    headers = provider.headers
    assert "Authorization" in headers
    assert "Content-Type" in headers
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


def test_openrouter_headers_present():
    """Verify OpenRouter-specific headers are present for default base URL."""
    from agentic_recommender.llm.async_provider import AsyncLLMProvider

    provider = AsyncLLMProvider(
        api_key="test-key",
        model_name="test-model",
    )

    headers = provider.headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers


def test_config_file_exists():
    """Verify the DashScope config file exists and is valid YAML."""
    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
    oc = _load_llm_config()
    assert oc["base_url"] is not None
    assert oc["model_name"] is not None
    assert oc["api_key_env_var"] == "DASHSCOPE_API_KEY"
