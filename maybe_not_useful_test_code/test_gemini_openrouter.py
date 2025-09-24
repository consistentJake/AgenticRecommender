#!/usr/bin/env python3
"""
Test GeminiProvider with OpenRouter integration.
Shows how to use the same GeminiProvider class with both direct API and OpenRouter.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_recommender.models.llm_provider import GeminiProvider, create_llm_provider


def test_gemini_openrouter_integration():
    """Test GeminiProvider with OpenRouter mode"""
    print("🧪 Testing GeminiProvider with OpenRouter Integration")
    print("=" * 55)
    
    # Get OpenRouter API key from environment
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("   Please set your OpenRouter API key:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return False
    
    try:
        print("1. Creating GeminiProvider with OpenRouter...")
        
        # Method 1: Direct instantiation
        provider = GeminiProvider(
            api_key=openrouter_key,
            model_name="google/gemini-flash-1.5", 
            use_openrouter=True
        )
        
        print(f"   ✅ Provider created: {provider.get_model_info()['provider']}")
        print(f"   Model: {provider.model_name}")
        
        # Method 2: Factory function
        print("\n2. Creating with factory function...")
        factory_provider = create_llm_provider(
            "openrouter",
            api_key=openrouter_key,
            model_name="google/gemini-flash-1.5"
        )
        
        print(f"   ✅ Factory provider: {factory_provider.get_model_info()['provider']}")
        
        # Test generation
        print(f"\n3. Testing text generation...")
        test_prompt = "What is the capital of France? Answer with just the city name."
        response = provider.generate(test_prompt, temperature=0.3, max_tokens=20)
        
        if "ERROR" not in response:
            print(f"   ✅ Response: {response}")
        else:
            print(f"   ❌ Error: {response}")
            return False
        
        # Test JSON mode
        print(f"\n4. Testing JSON mode...")
        json_prompt = "Create a JSON object with 'city' and 'country' for Paris."
        json_response = provider.generate(json_prompt, temperature=0.3, max_tokens=50, json_mode=True)
        
        if "ERROR" not in json_response:
            print(f"   ✅ JSON Response: {json_response}")
        else:
            print(f"   ❌ JSON Error: {json_response}")
        
        # Show detailed metrics
        print(f"\n5. Provider metrics:")
        metrics = provider.get_model_info()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_direct_vs_openrouter():
    """Compare direct Gemini API vs OpenRouter for the same model"""
    print("\n⚖️ Comparing Direct Gemini vs OpenRouter")
    print("=" * 45)
    
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    direct_key = os.getenv('GEMINI_API_KEY')
    
    if not openrouter_key:
        print("⚠️ OPENROUTER_API_KEY not available for comparison")
        return
    
    if not direct_key:
        print("⚠️ GEMINI_API_KEY not available for comparison")
    
    test_prompt = "Recommend a laptop for a software developer. Be concise."
    
    # Test OpenRouter
    print("1. Testing OpenRouter...")
    try:
        openrouter_provider = GeminiProvider(
            api_key=openrouter_key,
            model_name="google/gemini-flash-1.5",
            use_openrouter=True
        )
        
        import time
        start_time = time.time()
        openrouter_response = openrouter_provider.generate(test_prompt, temperature=0.5, max_tokens=100)
        openrouter_time = time.time() - start_time
        
        print(f"   ✅ Response: {openrouter_response[:100]}...")
        print(f"   Time: {openrouter_time:.3f}s")
        
    except Exception as e:
        print(f"   ❌ OpenRouter failed: {e}")
        openrouter_response = "ERROR"
        openrouter_time = 0
    
    # Test Direct API (if available)
    if direct_key:
        print("\n2. Testing Direct Gemini API...")
        try:
            direct_provider = GeminiProvider(
                api_key=direct_key,
                model_name="gemini-2.0-flash-exp",
                use_openrouter=False
            )
            
            start_time = time.time()
            direct_response = direct_provider.generate(test_prompt, temperature=0.5, max_tokens=100)
            direct_time = time.time() - start_time
            
            print(f"   ✅ Response: {direct_response[:100]}...")
            print(f"   Time: {direct_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Direct API failed: {e}")
            direct_response = "ERROR"
            direct_time = 0
        
        # Summary
        print(f"\n📊 Comparison Summary:")
        print(f"   OpenRouter: {'✅ Success' if 'ERROR' not in openrouter_response else '❌ Failed'} ({openrouter_time:.3f}s)")
        if direct_key:
            print(f"   Direct API: {'✅ Success' if 'ERROR' not in direct_response else '❌ Failed'} ({direct_time:.3f}s)")


def test_orchestrator_with_openrouter():
    """Test the full orchestrator workflow with OpenRouter"""
    print("\n🎭 Testing Orchestrator with OpenRouter")
    print("=" * 40)
    
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_key:
        print("⚠️ OPENROUTER_API_KEY not available")
        return
    
    try:
        from agentic_recommender.core import AgentOrchestrator, RecommendationRequest
        
        # Create provider with OpenRouter
        provider = GeminiProvider(
            api_key=openrouter_key,
            model_name="google/gemini-flash-1.5",
            use_openrouter=True
        )
        
        # Create orchestrator
        orchestrator = AgentOrchestrator(provider)
        
        # Test recommendation
        request = RecommendationRequest(
            user_id="test_user",
            user_sequence=["smartphone", "phone_case"],
            candidates=["screen_protector", "wireless_charger", "bluetooth_headphones", "power_bank"]
        )
        
        print(f"   User sequence: {' → '.join(request.user_sequence)}")
        print(f"   Candidates: {', '.join(request.candidates)}")
        
        # Get recommendation
        response = orchestrator.recommend(request, max_iterations=2)
        
        print(f"\n   ✅ Recommendation: {response.recommendation}")
        print(f"   📈 Confidence: {response.confidence:.3f}")
        print(f"   🧠 Reasoning: {response.reasoning[:150]}...")
        
        # Show provider metrics
        metrics = provider.get_model_info()
        print(f"\n   📊 OpenRouter Metrics:")
        print(f"   Calls: {metrics['total_calls']}, Tokens: {metrics['total_tokens']}, Avg Time: {metrics['avg_time_per_call']:.3f}s")
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test execution"""
    print("GeminiProvider OpenRouter Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Basic provider functionality
        success = test_gemini_openrouter_integration()
        
        # Test 2: Direct vs OpenRouter comparison
        test_comparison_direct_vs_openrouter()
        
        # Test 3: Orchestrator integration
        if success:
            test_orchestrator_with_openrouter()
        
        print(f"\n🎯 Summary:")
        print(f"   GeminiProvider now supports both direct Gemini API and OpenRouter!")
        print(f"   Use: GeminiProvider(api_key, model_name, use_openrouter=True)")
        print(f"   Or:  create_llm_provider('openrouter', api_key=key)")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")


if __name__ == "__main__":
    main()