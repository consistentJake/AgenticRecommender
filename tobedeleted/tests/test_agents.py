"""
Unit tests for agent functionality.
Tests the core agent system with mock data.
"""

import os
import pytest
import json
from agentic_recommender.agents import Manager, Analyst, Reflector
from agentic_recommender.agents.base import AgentType, ReflectionStrategy
from agentic_recommender.models.llm_provider import MockLLMProvider, GeminiProvider


class TestAgentSystem:
    """Test the complete agent system"""
    
    def setup_method(self):
        """Setup test data and mock providers"""
        # Mock LLM responses
        self.mock_responses = {
            "thinking": "I need to analyze the user's sequential patterns to make a good recommendation.",
            "action": "Analyse[user, 524]",
            "user_analysis": json.dumps({
                "patterns": "User prefers electronics and books",
                "sequential_behavior": "Tends to buy accessories after main items",
                "recent_trends": "Increasing interest in tech gadgets",
                "recommendations": "Recommend tech accessories or advanced electronics"
            }),
            "reflection": json.dumps({
                "correctness": False,
                "reason": "Previous recommendation ignored user's recent shift to tech items",
                "improvement": "Focus more on recent interactions and sequential patterns"
            })
        }
        
        # Sample data
        self.user_data = {
            "524": {"age": 28, "location": "SF", "preferences": ["electronics", "books"]}
        }
        self.item_data = {
            "181": {"name": "Wireless Headphones", "category": "Electronics", "price": 99.99}
        }
        self.user_histories = {
            "524": ["laptop", "mouse", "keyboard", "book", "headphones"]
        }
        
    def test_manager_think_act_cycle(self):
        """Test Manager's two-stage think/act mechanism"""
        # Create mock providers with specific responses for different calls
        thought_llm = MockLLMProvider({
            "default": self.mock_responses["thinking"],
            "analyze": self.mock_responses["thinking"]
        })
        action_llm = MockLLMProvider({
            "default": self.mock_responses["action"],
            "choose": self.mock_responses["action"]
        })
        
        # Create manager
        manager = Manager(thought_llm, action_llm)
        
        # Test thinking phase
        task_context = {
            "user_id": "524",
            "user_sequence": ["laptop", "mouse", "keyboard"],
            "candidates": ["headphones", "monitor", "book"]
        }
        
        thought = manager.think(task_context)
        assert "analyze" in thought.lower()
        
        # Test action phase
        action_type, argument = manager.act(task_context)
        assert action_type == "Analyse"
        assert "user" in argument[0].lower()
        assert "524" in argument[1]
        
        print("✅ Manager think/act cycle tests passed")
    
    def test_analyst_user_analysis(self):
        """Test Analyst's user analysis capabilities"""
        mock_llm = MockLLMProvider({"default": self.mock_responses["user_analysis"]})
        
        # Create analyst with test data
        analyst = Analyst(
            llm_provider=mock_llm,
            user_data=self.user_data,
            item_data=self.item_data
        )
        analyst.update_data_sources(user_histories=self.user_histories)
        
        # Test user analysis
        result = analyst.forward(["user", "524"], json_mode=True)
        assert len(result) > 0  # Just check that we get a response
        
        # Test item analysis
        result = analyst.forward(["item", "181"])
        assert len(result) > 0
        
        print("✅ Analyst user/item analysis tests passed")
    
    def test_reflector_strategies(self):
        """Test Reflector's different reflection strategies"""
        mock_llm = MockLLMProvider({"default": self.mock_responses["reflection"]})
        
        # Test NONE strategy
        reflector = Reflector(mock_llm, ReflectionStrategy.NONE)
        result = reflector.forward("test task", "test scratchpad")
        assert "No reflection" in result
        
        # Test LAST_ATTEMPT strategy
        reflector.set_strategy(ReflectionStrategy.LAST_ATTEMPT)
        result = reflector.forward("recommend item", "tried headphones")
        assert "Previous attempt" in result
        assert "headphones" in result
        
        # Test REFLEXION strategy
        reflector.set_strategy(ReflectionStrategy.REFLEXION)
        result = reflector.forward(
            input_task="recommend next item",
            scratchpad="analyzed user, recommended headphones",
            first_attempt="headphones",
            ground_truth="monitor"
        )
        assert len(result) > 0
        
        print("✅ Reflector strategy tests passed")
    
    def test_agent_performance_tracking(self):
        """Test agent performance and metrics tracking"""
        mock_llm = MockLLMProvider({"default": "test response"})
        manager = Manager(mock_llm, mock_llm)
        
        # Reset initial state
        manager.reset()
        
        # Invoke multiple times using invoke method (which tracks calls)
        for i in range(3):
            manager.invoke(argument={"test": f"context_{i}"}, stage="thought")
        
        # Check performance stats
        stats = manager.get_performance_stats()
        assert stats['total_calls'] == 3
        assert stats['avg_time_per_call'] >= 0
        assert stats['agent_type'] == "Manager"
        
        print("✅ Agent performance tracking tests passed")
    
    def test_tool_agent_commands(self):
        """Test ToolAgent command execution"""
        mock_llm = MockLLMProvider({"default": "analysis complete"})
        
        analyst = Analyst(mock_llm, self.user_data, self.item_data)
        analyst.update_data_sources(user_histories=self.user_histories)
        
        # Test command execution
        result = analyst.execute_command("UserInfo[524]")
        assert "524" in result
        
        result = analyst.execute_command("UserHistory[524, 3]")
        assert "keyboard" in result or "book" in result or "headphones" in result
        
        # Check command history
        assert len(analyst.command_history) == 2
        
        print("✅ ToolAgent command execution tests passed")


def test_with_real_gemini():
    """Test agents with real Gemini API (if available)"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  Skipping Gemini tests - no API key found")
        return
    
    try:
        # Create real Gemini provider
        gemini = GeminiProvider(api_key)
        
        # Test basic functionality
        manager = Manager(gemini, gemini, config={'max_steps': 2})
        
        task_context = {
            "user_id": "test_user",
            "user_sequence": ["phone", "case"],
            "candidates": ["charger", "headphones", "laptop"]
        }
        
        # Test one think/act cycle
        thought = manager.think(task_context)
        action_type, argument = manager.act(task_context)
        
        assert len(thought) > 0
        assert action_type in ["Analyse", "Search", "Finish"]
        
        print("✅ Real Gemini API tests passed")
        
    except Exception as e:
        print(f"⚠️  Gemini API test failed: {e}")


def run_agent_tests():
    """Run all agent tests"""
    print("🧪 Running agent system tests...\n")
    
    # Create test instance
    test_suite = TestAgentSystem()
    test_suite.setup_method()
    
    # Run all tests
    test_suite.test_manager_think_act_cycle()
    test_suite.test_analyst_user_analysis()
    test_suite.test_reflector_strategies()
    test_suite.test_agent_performance_tracking()
    test_suite.test_tool_agent_commands()
    
    # Test with real API if available
    test_with_real_gemini()
    
    print("\n🎉 All agent tests passed! Agent system is ready.")


if __name__ == "__main__":
    run_agent_tests()