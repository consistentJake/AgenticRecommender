#!/usr/bin/env python3
"""
Quick test script for OpenRouter integration with AgentOrchestrator.
Uses the default OpenRouter API key from llm_provider.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_recommender.models.llm_provider import get_default_openrouter_gemini_provider
from agentic_recommender.core import AgentOrchestrator, RecommendationRequest


def quick_openrouter_test():
    """Quick test of OpenRouter with AgentOrchestrator"""
    print("🚀 Quick OpenRouter Test")
    print("=" * 30)
    
    try:
        # Use default OpenRouter provider
        print("1. Creating OpenRouter provider with default key...")
        provider = get_default_openrouter_gemini_provider()
        print(f"   ✅ Provider: {provider.get_model_info()['provider']}")
        print(f"   Model: {provider.model_name}")
        
        # Create orchestrator
        print("\n2. Creating AgentOrchestrator...")
        orchestrator = AgentOrchestrator(provider)
        
        # Test simple recommendation
        print("\n3. Testing recommendation workflow...")
        request = RecommendationRequest(
            user_id="test_user",
            user_sequence=["smartphone", "phone_case"], 
            candidates=["screen_protector", "wireless_charger", "bluetooth_headphones"]
        )
        
        print(f"   User sequence: {' → '.join(request.user_sequence)}")
        print(f"   Candidates: {', '.join(request.candidates)}")
        
        # Get recommendation
        print("   🎭 Running orchestrator...")
        response = orchestrator.recommend(request, max_iterations=2)
        
        print(f"\n4. ✅ Results:")
        print(f"   Recommendation: {response.recommendation}")
        print(f"   Confidence: {response.confidence:.3f}")
        print(f"   Reasoning: {response.reasoning}")
        
        # Show metrics
        metrics = provider.get_model_info()
        print(f"\n5. 📊 Performance:")
        print(f"   API calls: {metrics['total_calls']}")
        print(f"   Total tokens: {metrics['total_tokens']}")
        print(f"   Average time: {metrics['avg_time_per_call']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution"""
    print("OpenRouter Quick Test")
    print("=" * 40)
    print("Using DEFAULT_OPENROUTER_KEY from llm_provider.py")
    print()
    
    success = quick_openrouter_test()
    
    if success:
        print("\n🎉 OpenRouter integration is working!")
        print("💡 You can now use OpenRouter in your agentic system!")
    else:
        print("\n⚠️ Test failed. Check your API key and connection.")


if __name__ == "__main__":
    main()