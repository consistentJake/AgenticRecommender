#!/usr/bin/env python3
"""
Test OpenRouter integration with Gemini Flash model.
This test demonstrates how to use OpenRouter provider with the agentic system.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_recommender.models.llm_provider import OpenRouterProvider, create_llm_provider
from agentic_recommender.core import AgentOrchestrator, RecommendationRequest
from agentic_recommender.system import create_pipeline


def test_openrouter_provider():
    """Test basic OpenRouter provider functionality"""
    print("🧪 Testing OpenRouter Provider")
    print("=" * 40)
    
    # Get OpenRouter API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("   Please set your OpenRouter API key:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return False
    
    try:
        # Test provider creation
        print("1. Creating OpenRouter provider...")
        provider = OpenRouterProvider(api_key=api_key)
        print(f"   ✅ Provider created: {provider.model_name}")
        
        # Test different models
        models_to_test = [
            "google/gemini-flash-1.5",
            "google/gemini-pro-1.5", 
            "openai/gpt-3.5-turbo"
        ]
        
        for model_name in models_to_test:
            print(f"\n2. Testing model: {model_name}")
            test_provider = OpenRouterProvider(api_key=api_key, model_name=model_name)
            
            # Simple generation test
            prompt = "What is the capital of France? Answer in one word."
            response = test_provider.generate(prompt, temperature=0.3, max_tokens=10)
            
            if "ERROR" not in response:
                print(f"   ✅ Response: {response}")
            else:
                print(f"   ❌ Error: {response}")
        
        # Test JSON mode
        print(f"\n3. Testing JSON mode...")
        json_prompt = "Return a JSON object with the following structure: {\"recommendation\": \"item_name\", \"confidence\": 0.85}"
        json_response = provider.generate(json_prompt, temperature=0.3, max_tokens=100, json_mode=True)
        
        if "ERROR" not in json_response:
            print(f"   ✅ JSON Response: {json_response}")
        else:
            print(f"   ❌ JSON Error: {json_response}")
        
        # Show metrics
        print(f"\n4. Provider metrics:")
        metrics = provider.get_model_info()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter provider test failed: {e}")
        return False


def test_openrouter_with_orchestrator():
    """Test OpenRouter provider with AgentOrchestrator"""
    print("\n🎭 Testing OpenRouter with AgentOrchestrator")
    print("=" * 45)
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return False
    
    try:
        # Create OpenRouter provider
        print("1. Creating OpenRouter provider for orchestrator...")
        provider = create_llm_provider("openrouter", api_key=api_key, model_name="google/gemini-flash-1.5")
        
        # Create orchestrator
        print("2. Creating AgentOrchestrator...")
        orchestrator = AgentOrchestrator(provider)
        
        # Test recommendation request
        print("3. Testing recommendation workflow...")
        request = RecommendationRequest(
            user_id="test_user",
            user_sequence=["laptop", "mouse"],
            candidates=["keyboard", "monitor", "webcam", "headphones"]
        )
        
        print(f"   User sequence: {' → '.join(request.user_sequence)}")
        print(f"   Candidates: {', '.join(request.candidates)}")
        
        # Get recommendation
        response = orchestrator.recommend(request, max_iterations=2)
        
        print(f"\n4. OpenRouter Orchestrator Results:")
        print(f"   ✅ Recommendation: {response.recommendation}")
        print(f"   📈 Confidence: {response.confidence:.3f}")
        print(f"   🧠 Reasoning: {response.reasoning}")
        print(f"   ⚡ Iterations: {response.metadata.get('iterations', 'N/A')}")
        
        # Show system stats
        stats = orchestrator.get_system_stats()
        print(f"\n5. System Performance:")
        print(f"   Total requests: {stats['orchestrator']['total_requests']}")
        print(f"   Success rate: {stats['orchestrator']['success_rate']:.3f}")
        
        # Show provider metrics
        provider_info = provider.get_model_info()
        print(f"\n6. OpenRouter Provider Metrics:")
        print(f"   Model: {provider_info['model_name']}")
        print(f"   Total calls: {provider_info['total_calls']}")
        print(f"   Total tokens: {provider_info['total_tokens']}")
        print(f"   Avg time per call: {provider_info['avg_time_per_call']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_comparison():
    """Compare different providers if multiple API keys are available"""
    print("\n⚖️ Provider Comparison Test")
    print("=" * 35)
    
    # Check available providers
    providers_to_test = []
    
    # OpenRouter
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_key:
        providers_to_test.append(("OpenRouter", "openrouter", {"api_key": openrouter_key}))
    
    # Gemini Direct
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        providers_to_test.append(("Gemini Direct", "gemini", {"api_key": gemini_key}))
    
    # Mock provider
    providers_to_test.append(("Mock", "mock", {}))
    
    if len(providers_to_test) < 2:
        print("⚠️ Need at least 2 providers for comparison")
        print("   Available providers:", [p[0] for p in providers_to_test])
        return
    
    print(f"Comparing {len(providers_to_test)} providers...")
    
    # Test prompt
    test_prompt = "Given a user who bought a laptop and mouse, recommend the next item from: keyboard, monitor, webcam. Answer with just the item name."
    
    results = []
    
    for provider_name, provider_type, kwargs in providers_to_test:
        print(f"\n🔍 Testing {provider_name}...")
        
        try:
            # Create provider
            provider = create_llm_provider(provider_type, **kwargs)
            
            # Test generation
            start_time = time.time()
            response = provider.generate(test_prompt, temperature=0.3, max_tokens=20)
            end_time = time.time()
            
            results.append({
                'provider': provider_name,
                'response': response,
                'time': end_time - start_time,
                'success': "ERROR" not in response
            })
            
            print(f"   Response: {response}")
            print(f"   Time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            results.append({
                'provider': provider_name,
                'response': f"ERROR: {e}",
                'time': 0,
                'success': False
            })
            print(f"   ❌ Failed: {e}")
    
    # Summary
    print(f"\n📊 Comparison Summary:")
    print("Provider          | Success | Time    | Response")
    print("-" * 55)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        response_preview = result['response'][:25] + "..." if len(result['response']) > 25 else result['response']
        print(f"{result['provider']:<17} | {status}      | {result['time']:.3f}s | {response_preview}")


def main():
    """Main test execution"""
    print("OpenRouter Integration Testing")
    print("=" * 50)
    
    try:
        # Test 1: Basic OpenRouter provider
        success1 = test_openrouter_provider()
        
        # Test 2: OpenRouter with orchestrator
        success2 = test_openrouter_with_orchestrator() if success1 else False
        
        # Test 3: Provider comparison
        test_provider_comparison()
        
        print(f"\n🎯 Test Results:")
        print(f"   OpenRouter Provider: {'✅ PASSED' if success1 else '❌ FAILED'}")
        print(f"   Orchestrator Integration: {'✅ PASSED' if success2 else '❌ FAILED'}")
        
        if success1 and success2:
            print("\n🎉 All tests passed! OpenRouter integration is working.")
        else:
            print("\n⚠️ Some tests failed. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time  # Import time for comparison test
    main()