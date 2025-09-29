"""
Test the complete recommendation pipeline.
"""

import os
from agentic_recommender.system import RecommendationPipeline, create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models.llm_provider import MockLLMProvider


def test_pipeline_creation():
    """Test pipeline factory and initialization"""
    print("🏗️ Testing pipeline creation...")
    
    mock_llm = MockLLMProvider({
        "default": "Mock analysis result",
        "action": "Finish[test_item]"
    })
    
    # Test factory function
    pipeline = create_pipeline(
        llm_provider=mock_llm,
        dataset_type="beauty",
        reflection_strategy="reflexion"
    )
    
    assert pipeline.dataset_type == "beauty"
    assert pipeline.orchestrator is not None
    
    print("✅ Pipeline creation test passed")


def test_demo_prediction():
    """Test demo prediction without dataset"""
    print("🎯 Testing demo prediction...")
    
    mock_llm = MockLLMProvider({
        "analyze": "User prefers tech accessories based on sequence",
        "action": "Finish[wireless_mouse]"
    })
    
    pipeline = create_pipeline(mock_llm, "beauty", "reflexion")
    
    # Run demo prediction
    response = pipeline.run_demo_prediction(
        user_sequence=["laptop", "keyboard", "monitor"],
        candidates=["wireless_mouse", "headphones", "webcam"]
    )
    
    assert response.recommendation in ["wireless_mouse", "headphones", "webcam", "Unable to parse action"]
    assert 0 <= response.confidence <= 1.0
    assert len(response.reasoning) > 0
    
    print("✅ Demo prediction test passed")


def test_orchestrator_integration():
    """Test orchestrator with mock data"""
    print("🤝 Testing orchestrator integration...")
    
    mock_llm = MockLLMProvider({
        "finish": "Finish[recommended_item]",
        "analysis": "User analysis complete"
    })
    
    pipeline = create_pipeline(mock_llm, "beauty", "last_trial")
    
    # Create request
    request = RecommendationRequest(
        user_id="test_user",
        user_sequence=["item1", "item2"],
        candidates=["item3", "item4", "item5"],
        ground_truth="item3"
    )
    
    # Get recommendation
    response = pipeline.orchestrator.recommend(request, max_iterations=2)
    
    assert response.recommendation is not None
    assert response.confidence >= 0
    assert len(response.reasoning) > 0
    assert 'session_id' in response.metadata
    
    print("✅ Orchestrator integration test passed")


def test_system_stats():
    """Test system statistics collection"""
    print("📊 Testing system statistics...")
    
    mock_llm = MockLLMProvider({"default": "test response"})
    pipeline = create_pipeline(mock_llm, "beauty", "none")
    
    # Run a few operations
    for i in range(3):
        request = RecommendationRequest(
            user_id=f"user_{i}",
            user_sequence=[f"item_{i}"],
            candidates=[f"rec_{i}", f"rec_{i+1}"]
        )
        pipeline.orchestrator.recommend(request, max_iterations=1)
    
    # Get statistics
    stats = pipeline.orchestrator.get_system_stats()
    
    assert stats['orchestrator']['total_requests'] == 3
    assert stats['orchestrator']['success_rate'] > 0
    assert 'agents' in stats
    assert 'communication' in stats
    
    print("✅ System statistics test passed")


def run_pipeline_tests():
    """Run all pipeline tests"""
    print("🧪 Running pipeline tests...\n")
    
    test_pipeline_creation()
    test_demo_prediction() 
    test_orchestrator_integration()
    test_system_stats()
    
    print("\n🎉 All pipeline tests completed!")


if __name__ == "__main__":
    run_pipeline_tests()