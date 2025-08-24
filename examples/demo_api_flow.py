"""
Demo showing the API integration flow and what happens with real API calls.
Uses enhanced mock to simulate real API responses.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models.llm_provider import MockLLMProvider


def create_realistic_api_mock():
    """Create a mock that simulates real API responses"""
    return MockLLMProvider({
        "analyze": "I should analyze the user's purchasing sequence. The pattern shows laptop → mouse, which indicates they're building a complete workstation setup.",
        "user": "User analysis: This user demonstrates tech-focused purchasing behavior. Sequential pattern shows logical progression from main device (laptop) to peripherals (mouse). Next likely purchases would complete the workstation setup.",
        "action": "Analyse[user, api_test_user]", 
        "finish": "Finish[mechanical_keyboard]",
        "reflection": '{"correctness": true, "reason": "Good sequential recommendation - keyboard logically completes laptop + mouse setup", "improvement": "Continue analyzing sequential completion patterns"}'
    })


def demo_api_integration_flow():
    """Demonstrate what happens with API integration"""
    print("🚀 API Integration Flow Demonstration")
    print("=" * 42)
    
    print("This demo simulates what happens when using real API calls:")
    print("💡 Using enhanced mock responses that mimic real LLM behavior")
    
    print("\n1. 🏗️ PIPELINE SETUP")
    print("-" * 20)
    
    # Create pipeline with realistic mock
    mock_llm = create_realistic_api_mock()
    pipeline = create_pipeline(mock_llm, "beauty", "reflexion")
    
    # Load synthetic dataset
    pipeline.load_dataset("dummy_path.json", use_synthetic=True)
    dataset_stats = pipeline.dataset.get_statistics()
    print(f"✅ Pipeline created with {dataset_stats['num_sessions']} sessions")
    
    print("\n2. 🎯 RECOMMENDATION REQUEST")
    print("-" * 27)
    
    # Create realistic request
    request = RecommendationRequest(
        user_id="demo_api_user",
        user_sequence=["gaming_laptop", "wireless_mouse"],
        candidates=["mechanical_keyboard", "monitor", "headphones", "webcam", "mousepad"],
        ground_truth="mechanical_keyboard"
    )
    
    print("📋 Request Details:")
    print(f"   User ID: {request.user_id}")
    print(f"   Sequence: {' → '.join(request.user_sequence)}")
    print(f"   Candidates: {', '.join(request.candidates)}")
    print(f"   Ground Truth: {request.ground_truth}")
    
    print("\n3. 🤖 AGENT WORKFLOW (Simulated API Calls)")
    print("-" * 44)
    
    print("🧠 Starting recommendation workflow...")
    print("   → Manager will think about the situation")
    print("   → Manager will decide on action") 
    print("   → If analysis needed, Analyst will be called")
    print("   → Manager will make final recommendation")
    print("   → Reflector will evaluate the result")
    
    # Execute with detailed logging
    response = pipeline.orchestrator.recommend(request, max_iterations=3)
    
    print("\n4. 📊 API RESPONSE ANALYSIS")
    print("-" * 26)
    
    print(f"✅ Recommendation Results:")
    print(f"   🎯 Recommended Item: {response.recommendation}")
    print(f"   📈 Confidence Score: {response.confidence:.3f}")
    print(f"   🧠 Reasoning: {response.reasoning}")
    print(f"   ⚡ Iterations Used: {response.metadata.get('iterations', 'N/A')}")
    
    # Show reflection if available
    if response.metadata.get('reflection'):
        reflection = response.metadata['reflection']
        if isinstance(reflection, dict):
            print(f"   🪞 Reflection:")
            print(f"      Correctness: {reflection.get('correctness', 'N/A')}")
            print(f"      Reason: {reflection.get('reason', 'N/A')}")
            print(f"      Improvement: {reflection.get('improvement', 'N/A')}")
    
    # Compare with ground truth
    correct = response.recommendation == request.ground_truth
    print(f"   ✅ Accuracy: {'CORRECT' if correct else 'INCORRECT'} (vs ground truth: {request.ground_truth})")
    
    print("\n5. 📈 SYSTEM PERFORMANCE")
    print("-" * 23)
    
    stats = pipeline.orchestrator.get_system_stats()
    print("📊 Performance Metrics:")
    print(f"   Total Requests: {stats['orchestrator']['total_requests']}")
    print(f"   Success Rate: {stats['orchestrator']['success_rate']:.3f}")
    print(f"   Session ID: {stats['orchestrator']['current_session']}")
    
    print("\n   🤖 Agent Performance:")
    for agent_name, perf in stats['agents'].items():
        calls = perf['total_calls']
        avg_time = perf['avg_time_per_call']
        if calls > 0:
            print(f"      {agent_name:10}: {calls} calls, {avg_time:.3f}s avg")
        else:
            print(f"      {agent_name:10}: No direct calls (orchestrated)")
    
    print(f"\n   📡 Communication:")
    print(f"      Messages: {stats['communication']['messages_in_session']}")
    
    print("\n6. 🔄 WORKFLOW TRACE")
    print("-" * 17)
    
    print("📝 Agent Communication History:")
    comm_history = pipeline.orchestrator.conversation_history[-5:]  # Last 5 messages
    for i, entry in enumerate(comm_history, 1):
        agent = entry['agent']
        msg_type = entry['type']
        content = entry['content'][:60] + "..." if len(entry['content']) > 60 else entry['content']
        print(f"   {i}. [{agent}] {msg_type}: {content}")
    
    print("\n7. 🎉 INTEGRATION SUMMARY")
    print("-" * 24)
    
    print("✅ Successfully Demonstrated:")
    print("   🏗️  Multi-agent coordination")
    print("   🤖 LLM API integration pattern")
    print("   📊 Performance monitoring")
    print("   🪞 Reflection and improvement")
    print("   📈 Comprehensive logging")
    print("   🎯 End-to-end recommendation")
    
    print(f"\n💡 With Real API Key:")
    print(f"   - Replace MockLLMProvider with GeminiProvider(api_key)")
    print(f"   - All API calls would go to Google's Gemini API") 
    print(f"   - Responses would be actual LLM-generated content")
    print(f"   - Performance metrics would include real API latency")
    print(f"   - Token usage would be tracked and reported")
    
    return response


if __name__ == "__main__":
    response = demo_api_integration_flow()
    print(f"\n🏁 Demo completed with recommendation: {response.recommendation}")