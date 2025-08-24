"""
LLM provider implementations for agentic recommendation system.
Supports Gemini API integration.
"""

import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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
    Uses Gemini 2.0 Flash for fast, cost-effective generation.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Performance tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 512, json_mode: bool = False, **kwargs) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            json_mode: Whether to enforce JSON output
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 40),
                max_output_tokens=max_tokens,
            )
            
            # Add JSON instruction if needed
            if json_mode:
                prompt += "\n\nRespond only with valid JSON."
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text
            if response.candidates:
                text = response.candidates[0].content.parts[0].text
            else:
                text = ""
            
            # Update metrics
            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(prompt.split()) + len(text.split())
            self.total_tokens += estimated_tokens
            
            return text.strip()
            
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"ERROR: {error_msg}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model and usage information"""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        
        return {
            'provider': 'Gemini',
            'model_name': self.model_name,
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'total_time': self.total_time,
            'avg_time_per_call': avg_time
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
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 512, **kwargs) -> str:
        """Generate mock response"""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Check for predefined responses
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default responses based on prompt content
        if "think" in prompt.lower() or "analyze" in prompt.lower():
            return "I need to analyze the user's sequential behavior to make a good recommendation."
        elif "action" in prompt.lower():
            if "analyse" in prompt.lower():
                return "Analyse[user, 524]"
            else:
                return "Finish[Mock recommendation]"
        else:
            return "Mock LLM response"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'provider': 'Mock',
            'model_name': 'mock-llm',
            'total_calls': self.call_count
        }


def create_llm_provider(provider_type: str = "gemini", **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.
    
    Args:
        provider_type: Type of provider ("gemini", "mock")
        **kwargs: Provider-specific arguments
        
    Returns:
        LLM provider instance
    """
    if provider_type.lower() == "gemini":
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key required for Gemini provider")
        return GeminiProvider(api_key, kwargs.get('model_name', 'gemini-2.0-flash-exp'))
    
    elif provider_type.lower() == "mock":
        return MockLLMProvider(kwargs.get('responses'))
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Default API key from APIs.md
DEFAULT_GEMINI_KEY = "AIzaSyC-Ku57fAg294TDJ0iIgpkp7X8fs191W-M"

def get_default_gemini_provider() -> GeminiProvider:
    """Get Gemini provider with default API key"""
    return GeminiProvider(DEFAULT_GEMINI_KEY)