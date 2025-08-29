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

try:
    import requests
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False


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
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp", use_openrouter: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.use_openrouter = use_openrouter
        
        if use_openrouter:
            # OpenRouter mode - use requests
            if not OPENROUTER_AVAILABLE:
                raise ImportError("requests not installed. Run: pip install requests")
            
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AgenticRecommender/system",
                "X-Title": "Agentic Recommender System"
            }
            
            # Convert model name for OpenRouter if needed
            if model_name.startswith("gemini"):
                if "2.0" in model_name or "flash" in model_name:
                    self.model_name = "google/gemini-flash-1.5"
                else:
                    self.model_name = "google/gemini-pro-1.5"
        else:
            # Direct Gemini API mode
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            
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
        start_time = time.time()
        
        if self.use_openrouter:
            return self._generate_openrouter(prompt, temperature, max_tokens, json_mode, start_time, **kwargs)
        else:
            return self._generate_direct(prompt, temperature, max_tokens, json_mode, start_time, **kwargs)
    
    def _generate_openrouter(self, prompt: str, temperature: float, max_tokens: int, 
                           json_mode: bool, start_time: float, **kwargs) -> str:
        """Generate using OpenRouter API"""
        try:
            # Prepare messages in OpenAI chat format
            messages = [{"role": "user", "content": prompt}]
            
            # Add JSON instruction if needed
            if json_mode:
                messages[0]["content"] += "\n\nRespond only with valid JSON."
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get('top_p', 0.9),
            }
            
            # Add response format for JSON mode (if supported by model)
            if json_mode and "gpt" in self.model_name.lower():
                payload["response_format"] = {"type": "json_object"}
            
            # Make API request
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                text = response_data['choices'][0]['message']['content']
            else:
                text = ""
            
            # Update metrics
            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration
            
            # Use actual token counts if provided
            usage = response_data.get('usage', {})
            if usage:
                tokens = usage.get('total_tokens', 0)
                if tokens > 0:
                    self.total_tokens += tokens
                else:
                    # Fallback to estimation
                    estimated_tokens = len(prompt.split()) + len(text.split())
                    self.total_tokens += estimated_tokens
            else:
                # Fallback to estimation
                estimated_tokens = len(prompt.split()) + len(text.split())
                self.total_tokens += estimated_tokens
            
            return text.strip()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"OpenRouter API request error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"ERROR: {error_msg}"
        except Exception as e:
            error_msg = f"OpenRouter API error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"ERROR: {error_msg}"
    
    def _generate_direct(self, prompt: str, temperature: float, max_tokens: int, 
                        json_mode: bool, start_time: float, **kwargs) -> str:
        """Generate using direct Gemini API"""
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
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 512, **kwargs) -> str:
        """Generate mock response"""
        self.call_count += 1
        self.last_prompt = prompt
        
        
        # Check for predefined responses
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default responses based on prompt content (check most specific first)
        if "available actions" in prompt.lower():
            # This is the action phase - need to return proper action format
            
            # Check if this is a final recommendation step (has analysis in context or scratchpad)
            if ("\"analysis\":" in prompt.lower() and ("tech accessories" in prompt.lower() or "user shows" in prompt.lower())) or \
               ("scratchpad" in prompt.lower() and "analyse[user" in prompt.lower() and "analysis" in prompt.lower()):
                # Final recommendation step - recommend the best item for tech sequence  
                return "Finish[wireless_mouse]"
            
            if "demo_user" in prompt:
                # For demo scenarios with specific candidates  
                if "gaming_laptop" in prompt or "mechanical_keyboard" in prompt:
                    return "Finish[monitor]"  # Good recommendation for tech sequence
                elif "foundation" in prompt or "concealer" in prompt or "lipstick" in prompt:
                    return "Finish[mascara]"  # Good recommendation for beauty sequence  
                elif "fiction_book" in prompt or "bookmark" in prompt:
                    return "Finish[notebook]"  # Good recommendation for book sequence
                else:
                    return "Analyse[user, demo_user]"
            else:
                # Extract user_id from context if possible
                import re
                user_match = re.search(r'"user_id":\s*"([^"]+)"', prompt)
                if user_match:
                    user_id = user_match.group(1)
                    return f"Analyse[user, {user_id}]"
                else:
                    return "Finish[wireless_mouse]"  # Default recommendation for tech sequence
        elif "analyze this user's preferences" in prompt.lower() or "user information:" in prompt.lower():
            # This is an analyst analysis request
            return "User shows strong preference for tech accessories and complementary items. Sequential pattern: laptop → peripherals → productivity tools. Likely to want wireless_mouse next based on tech workflow completion."
        elif "think" in prompt.lower() or "analyze" in prompt.lower():
            return "I need to analyze the user's sequential behavior to make a good recommendation."
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
        provider_type: Type of provider ("gemini", "openrouter", "mock")
        **kwargs: Provider-specific arguments
        
    Returns:
        LLM provider instance
    """
    if provider_type.lower() == "gemini":
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key required for Gemini provider")
        return GeminiProvider(
            api_key=api_key, 
            model_name=kwargs.get('model_name', 'gemini-2.0-flash-exp'),
            use_openrouter=False
        )
    
    elif provider_type.lower() == "openrouter":
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key required for OpenRouter provider")
        return GeminiProvider(
            api_key=api_key,
            model_name=kwargs.get('model_name', 'google/gemini-flash-1.5'),
            use_openrouter=True
        )
    
    elif provider_type.lower() == "mock":
        return MockLLMProvider(kwargs.get('responses'))
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def get_default_openrouter_provider(api_key: str, model_name: str = "google/gemini-flash-1.5") -> GeminiProvider:
    """Get Gemini provider configured for OpenRouter"""
    return GeminiProvider(api_key=api_key, model_name=model_name, use_openrouter=True)


# Default API key from APIs.md
DEFAULT_GEMINI_KEY = "AIzaSyC-Ku57fAg294TDJ0iIgpkp7X8fs191W-M"
DEFAULT_OPENROUTER_KEY = "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"

def get_default_gemini_provider() -> GeminiProvider:
    """Get Gemini provider with default API key"""
    return GeminiProvider(DEFAULT_GEMINI_KEY)

def get_default_openrouter_gemini_provider() -> GeminiProvider:
    """Get Gemini provider with default OpenRouter API key"""
    return GeminiProvider(api_key=DEFAULT_OPENROUTER_KEY, model_name="google/gemini-flash-1.5", use_openrouter=True)