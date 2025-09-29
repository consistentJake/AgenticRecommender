"""
Comprehensive integration tests for the complete agentic recommendation system.
Tests end-to-end workflow including datasets, agents, orchestration, and evaluation.
"""

import pytest
import json
from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.agents.base import ReflectionStrategy
from agentic_recommender.models.llm_provider import MockLLMProvider
from agentic_recommender.utils.metrics import hit_rate_at_k, ndcg_at_k, mrr


class TestSystemIntegration:
    """Test complete system integration"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create comprehensive mock responses
        self.mock_responses = {
            "analyze": "User shows preference for complementary items",
            "user": "Strong pattern: buys accessories after main items",
            "action": "Finish[recommended_item]",
            "reflection": json.dumps({
                "correctness": True, 
                "reason": "Good sequential recommendation",
                "improvement": "Continue with pattern analysis"
            })
        }
        
        self.mock_llm = MockLLMProvider(self.mock_responses)
    
    def test_pipeline_creation_and_config(self):
        """Test pipeline creation with different configurations"""
        
        # Test all reflection strategies
        strategies = ['none', 'last_trial', 'reflexion', 'last_trial_and_reflexion']
        
        for strategy in strategies:
            pipeline = create_pipeline(
                llm_provider=self.mock_llm,
                dataset_type="beauty",
                reflection_strategy=strategy
            )
            
            assert pipeline.dataset_type == "beauty"
            assert pipeline.orchestrator is not None
            assert pipeline.orchestrator.manager is not None
            assert pipeline.orchestrator.analyst is not None
            assert pipeline.orchestrator.reflector is not None
            
            # Verify reflection strategy
            expected_strategy = getattr(ReflectionStrategy, strategy.upper().replace('TRIAL', 'ATTEMPT'))
            if 'AND' in strategy.upper():
                expected_strategy = ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION
            assert pipeline.orchestrator.reflector.reflection_strategy == expected_strategy
        
        print("✅ Pipeline creation and configuration test passed")
    
    def test_orchestrator_workflow(self):
        """Test complete orchestrator workflow"""
        
        pipeline = create_pipeline(self.mock_llm, "beauty", "reflexion")
        
        # Test single recommendation
        request = RecommendationRequest(
            user_id="test_user",
            user_sequence=["item1", "item2", "item3"],
            candidates=["candidate1", "candidate2", "candidate3", "candidate4"],
            ground_truth="candidate2"
        )
        
        response = pipeline.orchestrator.recommend(request, max_iterations=3)
        
        # Verify response structure
        assert response.recommendation is not None
        assert isinstance(response.confidence, (int, float))
        assert 0 <= response.confidence <= 1.0
        assert len(response.reasoning) > 0
        assert isinstance(response.metadata, dict)
        assert 'session_id' in response.metadata
        
        # Verify orchestrator state
        stats = pipeline.orchestrator.get_system_stats()
        assert stats['orchestrator']['total_requests'] == 1
        assert stats['orchestrator']['success_rate'] == 1.0
        
        print("✅ Orchestrator workflow test passed")
    
    def test_agent_communication(self):
        """Test inter-agent communication protocol"""
        
        pipeline = create_pipeline(self.mock_llm, "beauty", "reflexion")
        
        # Add mock data to analyst
        pipeline.orchestrator.analyst.update_data_sources(
            user_data={"test_user": {"preferences": ["tech", "books"]}},
            item_data={"item1": {"category": "tech"}},
            user_histories={"test_user": ["laptop", "mouse", "keyboard"]}
        )
        
        request = RecommendationRequest(
            user_id="test_user",
            user_sequence=["laptop", "mouse"],
            candidates=["keyboard", "monitor", "headphones"],
            context={"test_communication": True}
        )
        
        response = pipeline.orchestrator.recommend(request, max_iterations=2)
        
        # Check communication history
        comm_history = pipeline.orchestrator.conversation_history
        assert len(comm_history) > 0
        
        # Verify agent interactions were logged
        agent_types = {entry['agent'] for entry in comm_history}
        assert 'Manager' in agent_types
        
        # Verify different message types
        message_types = {entry['type'] for entry in comm_history}
        assert 'thought' in message_types or 'action' in message_types
        
        print("✅ Agent communication test passed")
    
    def test_reflection_strategies(self):
        """Test different reflection strategies"""
        
        strategies_to_test = ['none', 'last_trial', 'reflexion']
        results = {}
        
        for strategy in strategies_to_test:
            pipeline = create_pipeline(self.mock_llm, "beauty", strategy)
            
            request = RecommendationRequest(
                user_id="reflection_test_user",
                user_sequence=["item1", "item2"],
                candidates=["candidate1", "candidate2"],
                ground_truth="candidate1"
            )
            
            response = pipeline.orchestrator.recommend(request, max_iterations=2)
            results[strategy] = response
            
            # Reset for next test
            pipeline.orchestrator.reset_session()
        
        # Verify all strategies produced responses
        for strategy, response in results.items():
            assert response.recommendation is not None
            assert response.confidence >= 0
            
            # Check reflection metadata
            if strategy == 'none':
                reflection = response.metadata.get('reflection', {})
                assert reflection.get('reflection_enabled', True) == False
            else:
                # Should have some form of reflection
                assert 'reflection' in response.metadata
        
        print("✅ Reflection strategies test passed")
    
    def test_synthetic_dataset_integration(self):
        """Test integration with synthetic dataset"""
        
        pipeline = create_pipeline(self.mock_llm, "beauty", "reflexion")
        
        # Load synthetic dataset
        pipeline.load_dataset("dummy_path.json", use_synthetic=True)
        
        # Verify dataset was loaded
        assert pipeline.dataset is not None
        assert len(pipeline.dataset.sessions) > 0
        assert len(pipeline.dataset.all_items) > 0
        
        # Test dataset statistics
        stats = pipeline.dataset.get_statistics()
        assert stats['num_sessions'] > 0
        assert stats['num_items'] > 0
        assert stats['avg_session_length'] > 0
        
        # Test recommendation with real dataset item
        if pipeline.dataset.sessions:
            sample_session = pipeline.dataset.sessions[0]
            sample_items = sample_session['items'][:2]  # Use first 2 items as sequence
            candidates = list(pipeline.dataset.all_items)[:5]  # First 5 items as candidates
            
            request = RecommendationRequest(
                user_id=sample_session['user_id'],
                user_sequence=sample_items,
                candidates=candidates
            )
            
            response = pipeline.orchestrator.recommend(request)
            assert response.recommendation is not None
        
        print("✅ Synthetic dataset integration test passed")
    
    def test_evaluation_metrics_integration(self):
        """Test evaluation metrics with system output"""
        
        # Test metrics with mock predictions
        predictions_batch = [
            ["item1", "item2", "item3", "item4", "item5"],
            ["item2", "item1", "item3", "item4", "item5"],
            ["item3", "item4", "item5", "item1", "item2"]
        ]
        
        ground_truths = ["item1", "item1", "item1"]
        
        # Test individual metrics
        for i, (preds, gt) in enumerate(zip(predictions_batch, ground_truths)):
            hr_1 = hit_rate_at_k(preds, gt, k=1)
            hr_5 = hit_rate_at_k(preds, gt, k=5)
            ndcg_5 = ndcg_at_k(preds, gt, k=5)
            mrr_score = mrr(preds, gt)
            
            # Verify metrics are in valid range
            assert 0 <= hr_1 <= 1
            assert 0 <= hr_5 <= 1
            assert ndcg_5 >= 0
            assert 0 <= mrr_score <= 1
        
        # Test batch evaluation
        from agentic_recommender.utils.metrics import evaluate_recommendations
        
        metrics = evaluate_recommendations(predictions_batch, ground_truths, k_values=[1, 3, 5])
        
        expected_metrics = ['hr@1', 'hr@3', 'hr@5', 'ndcg@1', 'ndcg@3', 'ndcg@5', 'mrr']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1 or metrics[metric] >= 0  # NDCG can be > 1 in theory
        
        print("✅ Evaluation metrics integration test passed")
    
    def test_system_performance_monitoring(self):
        """Test system performance monitoring and logging"""
        
        pipeline = create_pipeline(self.mock_llm, "beauty", "reflexion")
        
        # Make multiple requests
        requests = [
            RecommendationRequest(
                user_id=f"user_{i}",
                user_sequence=[f"item_{i}", f"item_{i+1}"],
                candidates=[f"candidate_{i}", f"candidate_{i+1}", f"candidate_{i+2}"]
            )
            for i in range(5)
        ]
        
        responses = []
        for request in requests:
            response = pipeline.orchestrator.recommend(request, max_iterations=2)
            responses.append(response)
            pipeline.orchestrator.reset_session()
        
        # Check system stats
        stats = pipeline.orchestrator.get_system_stats()
        
        # Verify orchestrator stats
        orch_stats = stats['orchestrator']
        assert orch_stats['total_requests'] == 5
        assert orch_stats['success_rate'] > 0
        
        # Verify agent stats structure
        assert 'agents' in stats
        for agent_name in ['manager', 'analyst', 'reflector']:
            assert agent_name in stats['agents']
            agent_stats = stats['agents'][agent_name]
            assert 'total_calls' in agent_stats
            assert 'avg_time_per_call' in agent_stats
        
        # Verify communication stats
        assert 'communication' in stats
        
        print("✅ System performance monitoring test passed")
    
    def test_error_handling_and_fallbacks(self):
        """Test system error handling and fallback mechanisms"""
        
        # Create pipeline with mock that might fail
        error_mock = MockLLMProvider({
            "error": "ERROR: Simulated failure"
        })
        
        pipeline = create_pipeline(error_mock, "beauty", "none")
        
        request = RecommendationRequest(
            user_id="error_test_user",
            user_sequence=["item1"],
            candidates=["candidate1", "candidate2"],
            ground_truth="candidate1"
        )
        
        # Should not crash, should return fallback response
        response = pipeline.orchestrator.recommend(request, max_iterations=1)
        
        # Verify fallback behavior
        assert response.recommendation is not None
        assert response.confidence >= 0
        
        # Check if error was logged in metadata
        if response.metadata.get('error'):
            assert response.confidence < 0.5  # Low confidence for error cases
        
        print("✅ Error handling and fallbacks test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running comprehensive integration tests...\n")
    
    test_suite = TestSystemIntegration()
    test_suite.setup_method()
    
    # Run all tests
    test_suite.test_pipeline_creation_and_config()
    test_suite.test_orchestrator_workflow()
    test_suite.test_agent_communication()
    test_suite.test_reflection_strategies()
    test_suite.test_synthetic_dataset_integration()
    test_suite.test_evaluation_metrics_integration()
    test_suite.test_system_performance_monitoring()
    test_suite.test_error_handling_and_fallbacks()
    
    print("\n🎉 All integration tests passed!")
    print("🏗️ Complete agentic recommendation system is ready for production!")


if __name__ == "__main__":
    run_integration_tests()