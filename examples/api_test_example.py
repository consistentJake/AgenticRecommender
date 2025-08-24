"""
Example of how to test the system with real API calls.
Shows how to set up and run tests with a real Gemini API key.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models.llm_provider import GeminiProvider


def test_with_api_key():
    """Example of testing with real API key"""
    print("🔑 API Key Testing Example")
    print("=" * 30)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ No GEMINI_API_KEY found!")
        print("\n🛠️  To test with real API:")
        print("1. Get a Gemini API key from: https://aistudio.google.com/app/apikey")
        print("2. Set the environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("3. Run this script again:")
        print("   python examples/api_test_example.py")
        print("\n💡 Alternative: Run with inline key:")
        print("   GEMINI_API_KEY='your-key' python examples/api_test_example.py")
        return False
    
    print(f"✅ Found API key: {api_key[:8]}...")
    
    try:
        print("\n🤖 Creating pipeline with real Gemini API...")
        
        # Create pipeline with real API
        gemini = GeminiProvider(api_key)
        pipeline = create_pipeline(gemini, "beauty", "reflexion")
        
        # Load synthetic data for testing
        pipeline.load_dataset("dummy_path.json", use_synthetic=True)
        
        print("\n🎯 Testing recommendation with real API...")
        
        # Create test request
        request = RecommendationRequest(
            user_id="api_test_user",
            user_sequence=["gaming_laptop", "wireless_mouse"],
            candidates=["mechanical_keyboard", "monitor", "headphones", "webcam"],
            ground_truth="mechanical_keyboard"
        )
        
        print(f"   User sequence: {' → '.join(request.user_sequence)}")
        print(f"   Candidates: {', '.join(request.candidates)}")
        print("   Making API calls...")
        
        # Get recommendation (this will make real API calls)
        response = pipeline.orchestrator.recommend(request, max_iterations=2)
        
        print(f"\n📊 API Test Results:")
        print(f"   ✅ Recommendation: {response.recommendation}")
        print(f"   📈 Confidence: {response.confidence:.3f}")
        print(f"   🧠 Reasoning: {response.reasoning}")
        print(f"   ⚡ Iterations: {response.metadata.get('iterations', 'N/A')}")
        
        if response.metadata.get('reflection'):
            reflection = response.metadata['reflection']
            if isinstance(reflection, dict):
                print(f"   🪞 Reflection: {reflection.get('reason', 'No reason')}")
        
        # Show system performance
        stats = pipeline.orchestrator.get_system_stats()
        print(f"\n📈 System Performance:")
        print(f"   Total requests: {stats['orchestrator']['total_requests']}")
        print(f"   Success rate: {stats['orchestrator']['success_rate']:.3f}")
        
        agent_stats = stats['agents']
        for agent_name, perf in agent_stats.items():
            if perf['total_calls'] > 0:
                print(f"   {agent_name}: {perf['total_calls']} calls, {perf['avg_time_per_call']:.3f}s avg")
        
        print(f"\n🎉 API integration test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ API test failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Common issues and solutions
        print(f"\n🛠️  Common solutions:")
        print(f"   - Check if API key is valid and active")
        print(f"   - Verify internet connection")
        print(f"   - Check if Gemini API quota is available")
        print(f"   - Try again in a few moments")
        
        return False


def run_api_integration_demo():
    """Run comprehensive API integration demo"""
    print("🚀 Complete API Integration Demo")
    print("=" * 35)
    
    success = test_with_api_key()
    
    if success:
        print("\n✅ INTEGRATION TEST PASSED")
        print("   The agentic recommendation system successfully:")
        print("   🤖 Made real API calls to Gemini")
        print("   🎯 Generated recommendations")
        print("   🧠 Coordinated multiple agents") 
        print("   📊 Tracked performance metrics")
        print("   🪞 Generated reflections")
        
    else:
        print("\n⚠️  INTEGRATION TEST SKIPPED")
        print("   Set up API key to test with real LLM calls")
    
    print(f"\n📚 System Features Demonstrated:")
    print(f"   🏗️  Multi-agent architecture (Manager, Analyst, Reflector)")
    print(f"   🔄 Agent communication protocol")
    print(f"   📊 Dataset processing and evaluation")
    print(f"   🤖 Real LLM API integration")
    print(f"   📈 Performance monitoring")
    print(f"   🧪 Comprehensive error handling")
    
    return success


if __name__ == "__main__":
    success = run_api_integration_demo()
    sys.exit(0 if success else 1)