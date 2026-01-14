"""
Test Stage 3: LLM providers.

Run with: pytest agentic_recommender/tests/test_llm_provider.py -v

Note: Tests with @pytest.mark.api require API keys.
"""

import pytest
import os

from agentic_recommender.models.llm_provider import (
    LLMProvider,
    MockLLMProvider,
    OpenRouterProvider,
    GeminiProvider,
    create_llm_provider,
)


# OpenRouter API key for testing (from APIs.md)
OPENROUTER_API_KEY = "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"


class TestMockLLMProvider:
    """Test mock provider (no API needed)."""

    def test_mock_returns_response(self):
        """Mock provider should return a response."""
        provider = MockLLMProvider()
        response = provider.generate("Test prompt")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_tracks_calls(self):
        """Mock provider should track call count."""
        provider = MockLLMProvider()
        provider.generate("Test 1")
        provider.generate("Test 2")

        assert provider.call_count == 2

    def test_mock_stores_last_prompt(self):
        """Mock provider should store last prompt."""
        provider = MockLLMProvider()
        provider.generate("My test prompt")

        assert provider.last_prompt == "My test prompt"

    def test_mock_uses_predefined_responses(self):
        """Mock provider should use predefined responses when matched."""
        responses = {
            "hello": "World!",
            "test": "This is a test response"
        }
        provider = MockLLMProvider(responses=responses)

        response = provider.generate("Say hello")
        assert "World!" in response

    def test_mock_get_model_info(self):
        """Mock provider should return model info."""
        provider = MockLLMProvider()
        provider.generate("test")

        info = provider.get_model_info()
        assert info['provider'] == 'Mock'
        assert info['total_calls'] == 1


class TestOpenRouterProvider:
    """Test OpenRouter provider initialization."""

    def test_init_requires_api_key(self):
        """Should raise without API key."""
        # Clear env var if set
        old_key = os.environ.pop("OPENROUTER_API_KEY", None)

        try:
            with pytest.raises(ValueError):
                OpenRouterProvider(api_key=None)
        finally:
            if old_key:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_init_with_api_key(self):
        """Should initialize with provided API key."""
        provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)

        assert provider.api_key == OPENROUTER_API_KEY
        assert provider.model_name == "google/gemini-2.0-flash-001"

    def test_init_with_custom_model(self):
        """Should use custom model name."""
        provider = OpenRouterProvider(
            api_key=OPENROUTER_API_KEY,
            model_name="anthropic/claude-3-haiku"
        )

        assert provider.model_name == "anthropic/claude-3-haiku"

    def test_get_model_info(self):
        """Should return model info."""
        provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)
        info = provider.get_model_info()

        assert info['provider'] == 'OpenRouter'
        assert info['model_name'] == "google/gemini-2.0-flash-001"
        assert info['total_calls'] == 0

    def test_reset_metrics(self):
        """Should reset metrics."""
        provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)
        provider.total_calls = 10
        provider.total_tokens = 1000

        provider.reset_metrics()

        assert provider.total_calls == 0
        assert provider.total_tokens == 0


@pytest.mark.api
class TestOpenRouterProviderAPI:
    """Test OpenRouter provider with actual API calls.

    These tests require a valid API key and network access.
    Run with: pytest -m api
    """

    @pytest.fixture
    def provider(self):
        """Create provider with API key."""
        return OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    def test_simple_generation(self, provider):
        """Should generate a response."""
        response = provider.generate(
            "Say 'hello' and nothing else.",
            max_tokens=10
        )

        assert not response.startswith("ERROR")
        assert len(response) > 0

    def test_generation_with_system_prompt(self, provider):
        """Should use system prompt."""
        response = provider.generate(
            "What are you?",
            system_prompt="You are a helpful food recommendation assistant.",
            max_tokens=50
        )

        assert not response.startswith("ERROR")

    def test_json_mode(self, provider):
        """Should return valid JSON in json_mode."""
        import json

        response = provider.generate(
            "Return a JSON object with key 'cuisine' and value 'pizza'",
            json_mode=True,
            max_tokens=50
        )

        assert not response.startswith("ERROR")
        # Try to parse as JSON
        try:
            parsed = json.loads(response)
            assert isinstance(parsed, dict)
        except json.JSONDecodeError:
            # Some models wrap JSON in markdown
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                assert isinstance(parsed, dict)

    def test_metrics_tracked(self, provider):
        """Should track metrics after generation."""
        provider.generate("Hello", max_tokens=10)

        assert provider.total_calls == 1
        assert provider.total_time > 0
        assert provider.total_tokens > 0

    def test_batch_generation(self, provider):
        """Should handle batch generation."""
        prompts = [
            "Say 'one'",
            "Say 'two'",
        ]

        responses = provider.generate_batch(prompts, max_tokens=10)

        assert len(responses) == 2
        assert all(not r.startswith("ERROR") for r in responses)


class TestCreateLLMProvider:
    """Test factory function."""

    def test_create_mock_provider(self):
        """Factory should create mock provider."""
        provider = create_llm_provider(provider_type="mock")

        assert isinstance(provider, MockLLMProvider)

    def test_create_invalid_raises(self):
        """Factory should raise for unknown type."""
        with pytest.raises(ValueError):
            create_llm_provider(provider_type="invalid_provider_xyz")


# Validation function
def validate_openrouter_provider():
    """Validate OpenRouter provider with real API call."""
    print("=" * 60)
    print("VALIDATION: OpenRouter Provider")
    print("=" * 60)

    try:
        provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)
        print(f"\nProvider initialized:")
        print(f"  Model: {provider.model_name}")

        print("\n--- Test 1: Simple Generation ---")
        response = provider.generate(
            "Name 3 popular cuisines in Singapore. Be brief.",
            max_tokens=100
        )
        print(f"Response: {response[:200]}...")

        print("\n--- Test 2: With System Prompt ---")
        response = provider.generate(
            "What should I order for dinner?",
            system_prompt="You are a Singapore food recommendation assistant. Suggest local dishes.",
            max_tokens=100
        )
        print(f"Response: {response[:200]}...")

        print("\n--- Test 3: JSON Mode ---")
        response = provider.generate(
            "List 3 cuisines as JSON array",
            json_mode=True,
            max_tokens=100
        )
        print(f"Response: {response}")

        print("\n--- Metrics ---")
        info = provider.get_model_info()
        print(f"  Total calls: {info['total_calls']}")
        print(f"  Total tokens: {info['total_tokens']}")
        print(f"  Avg time: {info['avg_time_per_call']:.2f}s")

        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure the API key is valid.")


if __name__ == "__main__":
    validate_openrouter_provider()
