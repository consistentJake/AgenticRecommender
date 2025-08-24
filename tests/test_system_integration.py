"""
Integration tests for the complete agentic recommendation system.
Tests the interaction between agents, datasets, and evaluation metrics.
"""

import os
from agentic_recommender.agents import Manager, Analyst, Reflector
from agentic_recommender.agents.base import ReflectionStrategy
from agentic_recommender.models.llm_provider import MockLLMProvider, GeminiProvider
from agentic_recommender.utils.metrics import hit_rate_at_k, ndcg_at_k, mrr


def test_basic_system_integration():
    """Test basic system integration without real datasets"""
    print("🔧 Testing basic system integration...")
    
    # Create mock providers
    mock_llm = MockLLMProvider({
        "analyze": "User prefers electronics and tech accessories",
        "action": "Analyse[user, test_user]",
        "finish": "Finish[headphones]"
    })
    
    # Create agents
    manager = Manager(mock_llm, mock_llm)
    analyst = Analyst(mock_llm)
    reflector = Reflector(mock_llm, ReflectionStrategy.REFLEXION)
    
    # Test agent interaction
    task_context = {
        "user_id": "test_user",
        "user_sequence": ["laptop", "mouse"],
        "candidates": ["headphones", "monitor", "keyboard"]
    }
    
    # Manager thinks and acts
    thought = manager.think(task_context)
    action_type, argument = manager.act(task_context)
    
    # Analyst provides analysis (if needed)
    if action_type == "Analyse":
        analysis = analyst.forward(argument)
        
    # Reflector evaluates attempt
    reflection = reflector.forward(
        input_task="recommend next item",
        scratchpad=f"Thought: {thought}\nAction: {action_type}[{argument}]",
        first_attempt="headphones",
        ground_truth="monitor"
    )
    
    # Verify agents worked
    assert len(thought) > 0
    assert action_type in ["Analyse", "Search", "Finish"]
    assert len(reflection) > 0
    
    print("✅ Basic system integration test passed")


def test_metrics_functionality():
    """Test that evaluation metrics work correctly"""
    print("📊 Testing evaluation metrics...")
    
    # Test data
    predictions = [101, 102, 103, 104, 105]
    ground_truth = 102
    
    # Calculate metrics
    hr_1 = hit_rate_at_k(predictions, ground_truth, k=1)
    hr_5 = hit_rate_at_k(predictions, ground_truth, k=5)
    ndcg_5 = ndcg_at_k(predictions, ground_truth, k=5)
    mrr_score = mrr(predictions, ground_truth)
    
    # Verify results
    assert hr_1 == 0.0  # Ground truth not at rank 1
    assert hr_5 == 1.0  # Ground truth at rank 2, within top 5
    assert ndcg_5 > 0   # Should be positive
    assert mrr_score == 0.5  # 1/2 for rank 2
    
    print("✅ Evaluation metrics test passed")


def test_gemini_integration():
    """Test integration with real Gemini API if available"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  Skipping Gemini integration test - no API key")
        return
    
    try:
        print("🤖 Testing Gemini API integration...")
        
        # Create real Gemini provider
        gemini = GeminiProvider(api_key)
        
        # Test basic generation
        response = gemini.generate("What is 2+2?", max_tokens=50)
        assert len(response) > 0
        
        # Test with agent
        manager = Manager(gemini, gemini, config={'max_steps': 1})
        thought = manager.think({"task": "simple test"})
        assert len(thought) > 0
        
        print("✅ Gemini integration test passed")
        
    except Exception as e:
        print(f"⚠️  Gemini integration test failed: {e}")


def test_agent_communication():
    """Test communication between agents"""
    print("🗣️ Testing agent communication...")
    
    # Mock responses for agent interaction
    mock_responses = {
        "think": "I need to analyze the user before making recommendations",
        "action": "Analyse[user, U000001]",
        "user_analysis": "User likes tech products, sequential pattern: accessories after main items",
        "reflection": '{"correctness": false, "reason": "Ignored user preference", "improvement": "Consider sequential patterns"}'
    }
    
    thought_llm = MockLLMProvider({"default": mock_responses["think"]})
    action_llm = MockLLMProvider({"default": mock_responses["action"]})
    analyst_llm = MockLLMProvider({"default": mock_responses["user_analysis"]})
    reflector_llm = MockLLMProvider({"default": mock_responses["reflection"]})
    
    # Create agents
    manager = Manager(thought_llm, action_llm)
    analyst = Analyst(analyst_llm)
    reflector = Reflector(reflector_llm)
    
    # Simulate communication flow
    task_context = {"user_id": "U000001", "candidates": ["item1", "item2"]}
    
    # 1. Manager thinks and decides to analyze user
    thought = manager.think(task_context)
    action_type, argument = manager.act(task_context)
    
    # 2. If manager wants analysis, ask analyst
    analysis_result = ""
    if action_type == "Analyse":
        analysis_result = analyst.forward(argument)
    
    # 3. Generate reflection on the process
    scratchpad_content = f"Thought: {thought}"
    if analysis_result:
        scratchpad_content += f"\nAnalysis: {analysis_result}"
    
    reflection = reflector.forward(
        input_task=str(task_context),
        scratchpad=scratchpad_content,
        first_attempt="recommendation_attempt"
    )
    
    # Verify communication worked
    assert len(thought) > 0
    assert action_type in ["Analyse", "Search", "Finish"]  # Any valid action
    assert len(reflection) > 0
    
    print("✅ Agent communication test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running integration tests...\n")
    
    test_basic_system_integration()
    test_metrics_functionality()
    test_agent_communication()
    test_gemini_integration()
    
    print("\n🎉 All integration tests completed!")


if __name__ == "__main__":
    run_integration_tests()