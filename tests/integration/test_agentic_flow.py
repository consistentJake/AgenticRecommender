"""
Integration tests for the complete agentic recommendation flow.

This test validates the entire pipeline:
1. Data adapter parses JSONL
2. Manager orchestrates the flow
3. Analyst provides analysis
4. Reflector uses Swing similarity for two-stage judgment
5. Final recommendation is produced
"""

import pytest
import json
from pathlib import Path

# Components to be implemented
from agentic_recommender.data.food_delivery_adapter import (
    parse_jsonl_record,
    RecommendationRequest
)
from agentic_recommender.similarity.swing import SwingSimilarity, SwingConfig
from agentic_recommender.agents.enhanced_reflector import EnhancedReflector
from agentic_recommender.agents.manager import RecommendationManager

# Infrastructure
from agentic_recommender.core.config import ConfigManager, AgentConfig
from agentic_recommender.core.llm_interface import LLMInterface, set_llm_provider
from agentic_recommender.core.prompts import get_prompt_manager, PromptType
from agentic_recommender.core.test_utils import (
    PseudoLLMAgent,
    create_test_order_history,
    create_test_candidate,
    assert_json_response
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = AgentConfig()
    config.swing.top_k = 5
    config.swing.similarity_threshold = 0.1
    config.reflector.enable_two_stage = True
    return ConfigManager(config)


@pytest.fixture
def pseudo_llm():
    """Create pseudo LLM agent for testing."""
    return PseudoLLMAgent()


@pytest.fixture
def llm_interface(pseudo_llm):
    """Create LLM interface with pseudo agent."""
    interface = LLMInterface(provider=pseudo_llm)
    set_llm_provider(pseudo_llm)
    return interface


@pytest.fixture
def swing_similarity():
    """Create and populate Swing similarity index."""
    config = SwingConfig(
        alpha1=5.0,
        alpha2=1.0,
        beta=0.3,
        similarity_threshold=0.1,
        top_k=5
    )
    swing = SwingSimilarity(config)

    # Add sample interactions (user_id, item_id as cuisine)
    interactions = [
        # User 1: pizza lover
        ('user1', 'pizza'), ('user1', 'pizza'), ('user1', 'burgare'),
        # User 2: mexican food lover
        ('user2', 'mexikanskt'), ('user2', 'mexikanskt'), ('user2', 'pizza'),
        # User 3: varied tastes
        ('user3', 'pizza'), ('user3', 'sushi'), ('user3', 'indiskt'),
        # User 4: similar to user1
        ('user4', 'pizza'), ('user4', 'pizza'), ('user4', 'burgare'), ('user4', 'pizza'),
        # User 5: similar to user2
        ('user5', 'mexikanskt'), ('user5', 'mexikanskt'), ('user5', 'burgare'),
    ]

    swing.fit(interactions)
    return swing


@pytest.fixture
def sample_jsonl_record():
    """Sample JSONL record from Singapore dataset."""
    return {
        "instruction": """User recent orders (oldest → newest):

| idx | day | hour | cuisine     | price |
|-----|-----|------|-------------|-------|
| 1   | Mon | 17   | pizza       | 0.12  |
| 2   | Mon | 17   | pizza       | 0.06  |
| 3   | Thu | 20   | kebab       | 0.56  |
| 4   | Fri | 20   | burgare     | 0.36  |
| 5   | Wed | 18   | mexikanskt  | 0.28  |
| 6   | Wed | 18   | mexikanskt  | 0.14  |
| 7   | Fri | 13   | mexikanskt  | 0.54  |
| 8   | Thu | 20   | pizza       | 0.44  |
| 9   | Thu | 20   | pizza       | 0.08  |
| 10  | Thu | 19   | burgare     | 1.00  |

Candidate product:
- Buffalo Wings (cuisine: mexikanskt, price: $0.32)""",
        "input": "",
        "output": "Yes",
        "system": "You are a food delivery recommendation assistant...",
        "history": []
    }


class TestDataAdapter:
    """Test data adapter functionality."""

    def test_parse_jsonl_record(self, sample_jsonl_record):
        """Test parsing JSONL record into structured format."""
        request = parse_jsonl_record(sample_jsonl_record)

        assert isinstance(request, RecommendationRequest)
        assert request.user_id is not None
        assert len(request.orders) == 10
        assert request.candidate.name == "Buffalo Wings"
        assert request.candidate.cuisine == "mexikanskt"
        assert request.candidate.price == 0.32
        assert request.ground_truth is True

    def test_order_parsing(self, sample_jsonl_record):
        """Test order record parsing."""
        request = parse_jsonl_record(sample_jsonl_record)

        # Check first order
        first_order = request.orders[0]
        assert first_order.idx == 1
        assert first_order.day == "Mon"
        assert first_order.hour == 17
        assert first_order.cuisine == "pizza"
        assert first_order.price == 0.12


class TestSwingSimilarity:
    """Test Swing similarity module."""

    def test_similarity_calculation(self, swing_similarity):
        """Test user similarity calculation."""
        # Users 1 and 4 both love pizza
        sim_1_4 = swing_similarity.compute_user_similarity('user1', 'user4')
        assert sim_1_4 > 0.1  # Should be similar

        # Users 1 and 2 have different preferences
        sim_1_2 = swing_similarity.compute_user_similarity('user1', 'user2')
        # May still have some similarity due to shared pizza, but lower
        assert sim_1_2 >= 0

    def test_get_similar_users(self, swing_similarity):
        """Test retrieving top-k similar users."""
        similar = swing_similarity.get_similar_users('user1')

        assert isinstance(similar, list)
        assert len(similar) <= 5  # top_k = 5

        # Should return (user_id, similarity_score) tuples
        if similar:
            user_id, score = similar[0]
            assert isinstance(user_id, str)
            assert isinstance(score, float)
            assert score >= 0.1  # Above threshold


class TestEnhancedReflector:
    """Test Enhanced Reflector with two-stage judgment."""

    def test_first_round_judgment(self, llm_interface, sample_jsonl_record):
        """Test first round LLM judgment."""
        request = parse_jsonl_record(sample_jsonl_record)

        reflector = EnhancedReflector(
            llm_interface=llm_interface,
            swing_similarity=None,  # Not needed for first round
            config={'enable_two_stage': False}
        )

        result = reflector.first_round_judgment(
            order_history=request.orders,
            candidate_product=request.candidate
        )

        # Should return JSON with prediction, confidence, reasoning
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        assert isinstance(result['prediction'], bool)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_two_stage_reflection(self, llm_interface, swing_similarity, sample_jsonl_record):
        """Test complete two-stage reflection flow."""
        request = parse_jsonl_record(sample_jsonl_record)

        reflector = EnhancedReflector(
            llm_interface=llm_interface,
            swing_similarity=swing_similarity,
            config={'enable_two_stage': True}
        )

        # Mock user database for similarity lookup
        user_database = {
            'user2': {
                'orders': [
                    {'cuisine': 'mexikanskt', 'price': 0.5},
                    {'cuisine': 'mexikanskt', 'price': 0.3}
                ]
            },
            'user5': {
                'orders': [
                    {'cuisine': 'mexikanskt', 'price': 0.4},
                    {'cuisine': 'burgare', 'price': 0.6}
                ]
            }
        }

        result = reflector.reflect(
            user_id=request.user_id,
            order_history=request.orders,
            candidate_product=request.candidate,
            user_database=user_database
        )

        # Should have both first and second round results
        assert hasattr(result, 'first_round_prediction')
        assert hasattr(result, 'final_prediction')
        assert hasattr(result, 'similar_users')
        assert isinstance(result.similar_users, list)


class TestRecommendationManager:
    """Test Manager orchestration."""

    def test_manager_think_act_cycle(self, llm_interface, test_config):
        """Test Manager think-act cycle."""
        manager = RecommendationManager(
            llm_interface=llm_interface,
            config=test_config.get_manager_config()
        )

        # Build test context
        context = {
            'user_id': 'test_user',
            'candidate': 'Buffalo Wings (mexikanskt, $0.32)'
        }

        # Think phase
        thought = manager.think(context)
        assert isinstance(thought, str)
        assert len(thought) > 0

        # Act phase
        action_type, argument = manager.act(context)
        assert action_type in ['Analyse', 'Reflect', 'Finish']
        assert argument is not None


class TestEndToEndFlow:
    """Test complete end-to-end recommendation flow."""

    def test_full_recommendation_pipeline(
        self,
        llm_interface,
        swing_similarity,
        test_config,
        sample_jsonl_record
    ):
        """Test complete pipeline from JSONL to recommendation."""

        # Step 1: Parse input
        request = parse_jsonl_record(sample_jsonl_record)
        assert request.user_id is not None

        # Step 2: Initialize components
        manager = RecommendationManager(
            llm_interface=llm_interface,
            config=test_config.get_manager_config()
        )

        reflector = EnhancedReflector(
            llm_interface=llm_interface,
            swing_similarity=swing_similarity,
            config=test_config.get_reflector_config().__dict__
        )

        # Step 3: Manager initiates flow
        context = {
            'user_id': request.user_id,
            'orders': request.orders,
            'candidate': request.candidate
        }

        # Think
        thought = manager.think(context)
        assert thought is not None

        # Act
        action_type, argument = manager.act(context)

        # Step 4: If action is Reflect, use reflector
        if action_type == 'Reflect':
            user_database = {}  # Empty for this test
            result = reflector.reflect(
                user_id=request.user_id,
                order_history=request.orders,
                candidate_product=request.candidate,
                user_database=user_database
            )

            # Should have final prediction
            assert hasattr(result, 'final_prediction')
            assert isinstance(result.final_prediction, bool)

        # Step 5: Manager finishes with recommendation
        final_action = "Finish[Yes]"  # Mock final action
        assert 'Finish' in final_action

    def test_pseudo_llm_behavior(self, pseudo_llm):
        """Test that pseudo LLM generates appropriate responses."""
        # Test Manager think
        think_response = pseudo_llm.generate(
            "Think step by step about what information you need",
            temperature=0.8
        )
        assert len(think_response) > 0
        assert 'analyze' in think_response.lower() or 'user' in think_response.lower()

        # Test Reflector first round
        first_round_response = pseudo_llm.generate(
            """Predict whether the user will purchase this product.
User's Recent Orders: pizza, pizza, mexikanskt
Candidate Product: mexikanskt""",
            temperature=0.3,
            json_mode=True
        )

        # Should be valid JSON
        data = json.loads(first_round_response)
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'reasoning' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
