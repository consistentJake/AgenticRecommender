"""
Centralized LLM interface for agent system.

This module provides a unified interface for LLM interactions that:
1. Abstracts provider details (Claude, Gemini, OpenRouter)
2. Handles prompt formatting and template rendering
3. Manages provider switching transparently
4. Integrates with logging and error handling
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from ..models.llm_provider import (
    LLMProvider,
    GeminiProvider,
    MockLLMProvider,
    create_llm_provider
)
from ..utils.logging import get_logger


class ModelTier(Enum):
    """Model tiers for different use cases."""
    FAST = "fast"      # Quick tasks, lower cost
    STANDARD = "standard"  # General purpose
    POWERFUL = "powerful"  # Complex reasoning


@dataclass
class LLMRequest:
    """Structured LLM request with metadata."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    json_mode: bool = False

    # Metadata for logging and tracking
    agent_name: Optional[str] = None
    stage: Optional[str] = None
    task_context: Dict[str, Any] = field(default_factory=dict)

    # Advanced options
    model_tier: ModelTier = ModelTier.STANDARD
    preferred_provider: Optional[str] = None


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    text: str
    provider: str
    model_name: str
    duration_ms: float
    tokens_used: int
    request_id: str
    success: bool = True
    error: Optional[str] = None


class LLMInterface:
    """
    Centralized interface for LLM interactions.

    Usage:
        llm = LLMInterface()

        # Simple request
        response = llm.generate("What is 2+2?")

        # Structured request with metadata
        request = LLMRequest(
            prompt="Analyze user behavior",
            system_prompt="You are an analyst",
            agent_name="Analyst",
            stage="user_analysis",
            temperature=0.3
        )
        response = llm.generate_structured(request)
    """

    def __init__(self, provider: Optional[LLMProvider] = None, config: Dict[str, Any] = None):
        """
        Initialize LLM interface.

        Args:
            provider: Optional LLM provider instance. If None, creates default.
            config: Optional configuration dict
        """
        self.config = config or {}
        self.provider = provider or self._create_default_provider()
        self.logger = get_logger()

        # Track usage
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def _create_default_provider(self) -> LLMProvider:
        """Create default provider based on configuration."""
        provider_type = self.config.get('provider_type', 'config')
        try:
            return create_llm_provider(provider_type)
        except Exception as e:
            self.logger.log_agent_action(
                agent_name="LLMInterface",
                action_type="warning",
                message=f"Failed to create provider {provider_type}, using mock: {e}"
            )
            return MockLLMProvider()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Simple generate method for backwards compatibility.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            json_mode: Whether to enforce JSON output
            **kwargs: Additional metadata

        Returns:
            Generated text
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            agent_name=kwargs.get('agent_name'),
            stage=kwargs.get('stage'),
            task_context=kwargs.get('task_context', {})
        )

        response = self.generate_structured(request)
        return response.text

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """
        Generate with structured request/response.

        Args:
            request: LLMRequest with all parameters

        Returns:
            LLMResponse with result and metadata
        """
        start_time = time.time()

        # Build full prompt
        full_prompt = request.prompt
        if request.system_prompt:
            # Prepend system prompt to user prompt for providers that don't support system
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

        # Build log metadata
        log_metadata = {
            'agent': request.agent_name or 'unknown',
            'stage': request.stage or 'generate',
            'model_tier': request.model_tier.value,
            'task_context': request.task_context
        }

        # Log request
        request_id = self.logger.log_llm_request(
            provider=self.provider.get_model_info().get('provider', 'unknown'),
            prompt=full_prompt,
            metadata=log_metadata
        )

        try:
            # Generate
            text = self.provider.generate(
                prompt=full_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                json_mode=request.json_mode,
                log_metadata=log_metadata
            )

            duration = (time.time() - start_time) * 1000

            # Estimate tokens (rough approximation)
            tokens = len(full_prompt.split()) + len(text.split())

            # Update stats
            self.total_requests += 1
            self.total_tokens += tokens
            self.total_time += duration

            # Log response
            self.logger.log_llm_response(
                provider=self.provider.get_model_info().get('provider', 'unknown'),
                request_id=request_id,
                response=text,
                metadata=log_metadata,
                duration_ms=duration,
                tokens_used=tokens
            )

            return LLMResponse(
                text=text,
                provider=self.provider.get_model_info().get('provider', 'unknown'),
                model_name=self.provider.get_model_info().get('model_name', 'unknown'),
                duration_ms=duration,
                tokens_used=tokens,
                request_id=request_id,
                success=True
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            error_msg = str(e)

            # Log error
            self.logger.log_llm_response(
                provider=self.provider.get_model_info().get('provider', 'unknown'),
                request_id=request_id,
                response=f"ERROR: {error_msg}",
                metadata=log_metadata,
                duration_ms=duration,
                tokens_used=0
            )

            return LLMResponse(
                text="",
                provider=self.provider.get_model_info().get('provider', 'unknown'),
                model_name=self.provider.get_model_info().get('model_name', 'unknown'),
                duration_ms=duration,
                tokens_used=0,
                request_id=request_id,
                success=False,
                error=error_msg
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0

        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_time_ms': self.total_time,
            'avg_time_per_request_ms': avg_time,
            'provider': self.provider.get_model_info()
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0


# Global singleton instance
_global_llm_interface: Optional[LLMInterface] = None


def get_llm_interface(config: Dict[str, Any] = None) -> LLMInterface:
    """
    Get global LLM interface instance.

    Args:
        config: Optional configuration dict

    Returns:
        Global LLMInterface instance
    """
    global _global_llm_interface

    if _global_llm_interface is None:
        _global_llm_interface = LLMInterface(config=config)

    return _global_llm_interface


def set_llm_provider(provider: LLMProvider):
    """
    Set the LLM provider for global interface.

    Args:
        provider: LLM provider instance
    """
    global _global_llm_interface

    if _global_llm_interface is None:
        _global_llm_interface = LLMInterface(provider=provider)
    else:
        _global_llm_interface.provider = provider
