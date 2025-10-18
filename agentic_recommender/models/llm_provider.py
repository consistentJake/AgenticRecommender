"""
LLM provider implementations for agentic recommendation system.
Supports Gemini API integration configured via `configs/config`.
"""

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config"

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-exp"
DEFAULT_OPENROUTER_MODEL = "google/gemini-flash-1.5"

# Default API keys sourced from APIs.md for local development.
DEFAULT_GEMINI_KEY = None
DEFAULT_OPENROUTER_KEY = None


from ..utils.logging import get_component_logger, get_general_log_file, get_logger


PROVIDER_LOGGER = get_component_logger("models.llm_provider")

DEFAULT_CONFIG = {
    "mode": "gemini",
    "gemini": {
        "api_key": None,
        "model_name": DEFAULT_GEMINI_MODEL,
    },
    "openrouter": {
        "api_key": None,
        "model_name": DEFAULT_OPENROUTER_MODEL,
    },
}


def _load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from `configs/config` with sensible defaults."""
    config = deepcopy(DEFAULT_CONFIG)

    if not CONFIG_PATH.exists():
        return config

    try:
        raw_text = CONFIG_PATH.read_text().strip()
    except OSError:
        return config

    if not raw_text:
        return config

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        provider = raw_text.lower()
        if provider in {"gemini", "openrouter", "mock"}:
            config["mode"] = provider
        return config

    if isinstance(parsed, dict):
        llm_section = parsed.get("llm", parsed)

        mode = llm_section.get("mode")
        if isinstance(mode, str) and mode.strip():
            config["mode"] = mode.strip().lower()

        for provider_key in ("gemini", "openrouter"):
            provider_section = llm_section.get(provider_key)
            if isinstance(provider_section, dict):
                for field in ("api_key", "model_name"):
                    if field in provider_section and provider_section[field] is not None:
                        config[provider_key][field] = provider_section[field]

        # Allow simple overrides that apply to the active mode.
        if llm_section.get("api_key") is not None:
            target = "openrouter" if config["mode"] == "openrouter" else "gemini"
            config[target]["api_key"] = llm_section["api_key"]
        if llm_section.get("model_name") is not None:
            target = "openrouter" if config["mode"] == "openrouter" else "gemini"
            config[target]["model_name"] = llm_section["model_name"]

    if config["mode"] not in {"gemini", "openrouter", "mock"}:
        config["mode"] = "gemini"

    return config


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 512, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class GeminiProvider(LLMProvider):
    """
    Gemini API provider for LLM inference.
    Supports both direct Gemini API and OpenRouter API.
    Uses Gemini 2.0 Flash for fast, cost-effective generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_openrouter: Optional[bool] = None,
    ):
        config = _load_llm_config()
        resolved_use_openrouter = (
            use_openrouter if use_openrouter is not None else config["mode"] == "openrouter"
        )

        provider_key = "openrouter" if resolved_use_openrouter else "gemini"
        provider_config = config.get(provider_key, {})

        resolved_model_name = (
            model_name
            or provider_config.get("model_name")
            or (DEFAULT_OPENROUTER_MODEL if resolved_use_openrouter else DEFAULT_GEMINI_MODEL)
        )

        resolved_api_key = (
            api_key
            or provider_config.get("api_key")
            or (DEFAULT_OPENROUTER_KEY if resolved_use_openrouter else DEFAULT_GEMINI_KEY)
        )

        if not resolved_api_key:
            raise ValueError("api_key required for Gemini provider")

        self.api_key = resolved_api_key
        self.model_name = resolved_model_name
        self.use_openrouter = resolved_use_openrouter
        self._genai = None
        self._requests = None
        self.model = None

        if self.use_openrouter:
            try:
                import requests  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency check
                raise ImportError(
                    "requests not installed. Run: pip install requests"
                ) from exc

            self._requests = requests
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AgenticRecommender/system",
                "X-Title": "Agentic Recommender System",
            }

            # Convert model name for OpenRouter if needed
            if self.model_name.startswith("gemini"):
                if "2.0" in self.model_name or "flash" in self.model_name:
                    self.model_name = "google/gemini-flash-1.5"
                else:
                    self.model_name = "google/gemini-pro-1.5"
        else:
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency check
                raise ImportError(
                    "google-generativeai not installed. Run: pip install google-generativeai"
                ) from exc

            self._genai = genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

        # Performance tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0

        provider_mode = "OpenRouter" if self.use_openrouter else "Gemini SDK"
        self._log_event(
            f"[GeminiProvider] Initialised using {provider_mode} (model={self.model_name})"
        )

    def _log_event(self, message: str, level: int = logging.INFO) -> None:
        """Log provider events to configured outputs."""
        PROVIDER_LOGGER.log(level, message)

    def _enrich_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attach provider-specific details to logging metadata."""
        meta = dict(metadata or {})
        meta.setdefault("model_name", self.model_name)
        meta.setdefault("provider_mode", "openrouter" if self.use_openrouter else "direct")
        return meta

    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 512, json_mode: bool = False, **kwargs) -> str:
        """
        Generate text using Gemini API (direct or via OpenRouter).
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            json_mode: Whether to enforce JSON output
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        log_metadata = kwargs.pop("log_metadata", None)
        meta = self._enrich_metadata(log_metadata)

        mode_label = "OpenRouter" if self.use_openrouter else "Gemini"
        self._log_event(
            f"[GeminiProvider] Generating via {mode_label} (model={self.model_name})"
        )

        if self.use_openrouter:
            return self._generate_openrouter(prompt, temperature, max_tokens, json_mode, meta, **kwargs)
        else:
            return self._generate_direct(prompt, temperature, max_tokens, json_mode, meta, **kwargs)
    
    def _generate_openrouter(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        log_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate using OpenRouter API with structured logging."""

        provider_label = "OpenRouter"
        meta = dict(log_metadata or {})
        meta.setdefault("provider", provider_label)
        logger = get_logger()
        request_id = logger.log_llm_request(provider_label, prompt, meta)

        requests_module = self._requests
        start_time = time.time()

        if requests_module is None:
            error_msg = "OpenRouter support not initialised; requests missing"
            self._log_event(f"❌ {error_msg}", level=logging.ERROR)
            logger.log_llm_response(provider_label, request_id, f"ERROR: {error_msg}", meta, duration_ms=0.0, tokens_used=0)
            raise RuntimeError(error_msg)

        try:
            messages = [{"role": "user", "content": prompt}]

            if json_mode:
                messages[0]["content"] += "\n\nRespond only with valid JSON."

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.9),
            }

            if json_mode and "gpt" in self.model_name.lower():
                payload["response_format"] = {"type": "json_object"}

            response = requests_module.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            response_data = response.json()

            if response_data.get("choices"):
                text = response_data["choices"][0]["message"]["content"]
            else:
                text = ""

            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration

            usage = response_data.get("usage", {})
            tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
            if tokens and tokens > 0:
                self.total_tokens += tokens
            else:
                tokens = len(prompt.split()) + len(text.split())
                self.total_tokens += tokens

            logger.log_llm_response(
                provider_label,
                request_id,
                text.strip(),
                meta,
                duration_ms=duration * 1000,
                tokens_used=tokens,
            )

            return text.strip()

        except requests_module.exceptions.RequestException as e:  # type: ignore[union-attr]
            duration = time.time() - start_time
            error_msg = f"OpenRouter API request error: {str(e)}"
            self._log_event(f"❌ {error_msg}", level=logging.ERROR)
            logger.log_llm_response(
                provider_label,
                request_id,
                f"ERROR: {error_msg}",
                meta,
                duration_ms=duration * 1000,
                tokens_used=0,
            )
            return f"ERROR: {error_msg}"
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"OpenRouter API error: {str(e)}"
            self._log_event(f"❌ {error_msg}", level=logging.ERROR)
            logger.log_llm_response(
                provider_label,
                request_id,
                f"ERROR: {error_msg}",
                meta,
                duration_ms=duration * 1000,
                tokens_used=0,
            )
            return f"ERROR: {error_msg}"

    def _generate_direct(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        log_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate using direct Gemini API with structured logging."""

        provider_label = "Gemini SDK"
        meta = dict(log_metadata or {})
        meta.setdefault("provider", provider_label)
        logger = get_logger()
        request_id = logger.log_llm_request(provider_label, prompt, meta)
        start_time = time.time()

        try:
            if self._genai is None or self.model is None:
                raise RuntimeError("Gemini SDK not initialised; google-generativeai missing")

            generation_config = self._genai.types.GenerationConfig(
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                max_output_tokens=max_tokens,
            )

            prompt_payload = prompt + "\n\nRespond only with valid JSON." if json_mode else prompt

            response = self.model.generate_content(
                prompt_payload,
                generation_config=generation_config,
            )

            if response.candidates:
                text = response.candidates[0].content.parts[0].text
            else:
                text = ""

            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration

            estimated_tokens = len(prompt_payload.split()) + len(text.split())
            self.total_tokens += estimated_tokens

            logger.log_llm_response(
                provider_label,
                request_id,
                text.strip(),
                meta,
                duration_ms=duration * 1000,
                tokens_used=estimated_tokens,
            )

            return text.strip()

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Gemini API error: {str(e)}"
            self._log_event(f"❌ {error_msg}", level=logging.ERROR)
            logger.log_llm_response(
                provider_label,
                request_id,
                f"ERROR: {error_msg}",
                meta,
                duration_ms=duration * 1000,
                tokens_used=0,
            )
            return f"ERROR: {error_msg}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model and usage information"""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        
        provider_name = 'Gemini (OpenRouter)' if self.use_openrouter else 'Gemini (Direct)'
        
        return {
            'provider': provider_name,
            'model_name': self.model_name,
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'total_time': self.total_time,
            'avg_time_per_call': avg_time,
            'api_mode': 'openrouter' if self.use_openrouter else 'direct'
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate mock response with structured logging."""

        log_metadata = kwargs.pop("log_metadata", None)
        meta = dict(log_metadata or {})
        meta.setdefault("model_name", "mock-llm")
        provider_label = "MockLLM"

        logger = get_logger()
        request_id = logger.log_llm_request(provider_label, prompt, meta)
        start_time = time.time()

        self.call_count += 1
        self.last_prompt = prompt

        response_text = self._generate_mock_response(prompt, temperature, max_tokens, **kwargs)

        duration = time.time() - start_time
        estimated_tokens = len(prompt.split()) + len(response_text.split())

        logger.log_llm_response(
            provider_label,
            request_id,
            response_text,
            meta,
            duration_ms=duration * 1000,
            tokens_used=estimated_tokens,
        )

        return response_text

    def _generate_mock_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> str:
        """Internal helper implementing the mock response heuristics."""

        # Check for predefined responses
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        lower_prompt = prompt.lower()

        if "available actions" in lower_prompt:
            if (
                "\"analysis\":" in lower_prompt
                and ("tech accessories" in lower_prompt or "user shows" in lower_prompt)
            ) or (
                "scratchpad" in lower_prompt
                and "analyse[user" in lower_prompt
                and "analysis" in lower_prompt
            ):
                return "Finish[wireless_mouse]"

            if "demo_user" in prompt:
                if "gaming_laptop" in prompt or "mechanical_keyboard" in prompt:
                    return "Finish[monitor]"
                if "foundation" in prompt or "concealer" in prompt or "lipstick" in prompt:
                    return "Finish[mascara]"
                if "fiction_book" in prompt or "bookmark" in prompt:
                    return "Finish[notebook]"
                return "Analyse[user, demo_user]"

            import re

            user_match = re.search(r'"user_id":\s*"([^"]+)"', prompt)
            if user_match:
                user_id = user_match.group(1)
                return f"Analyse[user, {user_id}]"
            return "Finish[wireless_mouse]"

        if "analyze this user's preferences" in lower_prompt or "user information:" in lower_prompt:
            return (
                "User shows strong preference for tech accessories and complementary items. "
                "Sequential pattern: laptop → peripherals → productivity tools. Likely to want "
                "wireless_mouse next based on tech workflow completion."
            )

        if "think" in lower_prompt or "analyze" in lower_prompt:
            return "I need to analyze the user's sequential behavior to make a good recommendation."

        return "Mock LLM response"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'provider': 'Mock',
            'model_name': 'mock-llm',
            'total_calls': self.call_count
        }


def create_llm_provider(provider_type: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers based on configuration overrides.

    Args:
        provider_type: Type of provider ("gemini", "openrouter", "mock", "config").
            When ``None`` or ``"config"``, the value from ``configs/config`` is used.
        **kwargs: Provider-specific arguments taking precedence over config values.

    Returns:
        LLM provider instance
    """
    config = _load_llm_config()

    requested_type = provider_type.lower() if isinstance(provider_type, str) else None
    if requested_type == "config":
        requested_type = None

    target_type = (requested_type or config["mode"]).lower()

    if target_type == "mock":
        return MockLLMProvider(kwargs.get('responses'))

    if target_type == "openrouter":
        api_key = kwargs.get('api_key') or config["openrouter"].get("api_key") or DEFAULT_OPENROUTER_KEY
        model_name = (
            kwargs.get('model_name')
            or config["openrouter"].get("model_name")
            or DEFAULT_OPENROUTER_MODEL
        )
        return GeminiProvider(
            api_key=api_key,
            model_name=model_name,
            use_openrouter=True,
        )

    if target_type == "gemini":
        api_key = kwargs.get('api_key') or config["gemini"].get("api_key") or DEFAULT_GEMINI_KEY
        model_name = (
            kwargs.get('model_name')
            or config["gemini"].get("model_name")
            or DEFAULT_GEMINI_MODEL
        )
        return GeminiProvider(
            api_key=api_key,
            model_name=model_name,
            use_openrouter=False,
        )

    raise ValueError(f"Unknown provider type: {provider_type or target_type}")


def get_default_openrouter_provider(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> GeminiProvider:
    """Get Gemini provider configured for OpenRouter using config fallbacks."""
    config = _load_llm_config()
    resolved_api_key = api_key or config["openrouter"].get("api_key") or DEFAULT_OPENROUTER_KEY
    resolved_model_name = (
        model_name
        or config["openrouter"].get("model_name")
        or DEFAULT_OPENROUTER_MODEL
    )
    return GeminiProvider(
        api_key=resolved_api_key,
        model_name=resolved_model_name,
        use_openrouter=True,
    )


def get_default_gemini_provider() -> GeminiProvider:
    """Get Gemini provider honouring the mode specified in `configs/config`."""
    config = _load_llm_config()

    if config["mode"] == "mock":
        raise ValueError("LLM config is set to 'mock'; Gemini provider is unavailable")

    use_openrouter = config["mode"] == "openrouter"
    provider_key = "openrouter" if use_openrouter else "gemini"
    provider_config = config[provider_key]

    api_key = provider_config.get("api_key") or (
        DEFAULT_OPENROUTER_KEY if use_openrouter else DEFAULT_GEMINI_KEY
    )
    model_name = provider_config.get("model_name") or (
        DEFAULT_OPENROUTER_MODEL if use_openrouter else DEFAULT_GEMINI_MODEL
    )

    return GeminiProvider(
        api_key=api_key,
        model_name=model_name,
        use_openrouter=use_openrouter,
    )


def get_default_openrouter_gemini_provider() -> GeminiProvider:
    """Get Gemini provider with OpenRouter settings derived from config."""
    return get_default_openrouter_provider()
