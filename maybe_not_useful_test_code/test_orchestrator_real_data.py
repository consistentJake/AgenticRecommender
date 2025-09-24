#!/usr/bin/env python3
"""
Test AgentOrchestrator.recommend() with real Beauty dataset data.
This test demonstrates the complete workflow with actual user sequences and item data.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_recommender.core import AgentOrchestrator, RecommendationRequest
from agentic_recommender.models.llm_provider import (
    GeminiProvider, MockLLMProvider, 
    DEFAULT_GEMINI_KEY, DEFAULT_OPENROUTER_KEY,
    get_default_openrouter_gemini_provider
)
from agentic_recommender.datasets import BeautyDataset
from agentic_recommender.agents.reflector import ReflectionStrategy
from agentic_recommender.utils.metrics import evaluate_recommendations


class OrchestoratorRealDataTester:
    """Test AgentOrchestrator with real Beauty dataset"""
    
    def __init__(self, use_real_api: bool = True, use_openrouter: bool = False):
        """
        Initialize tester
        
        Args:
            use_real_api: If True, use Gemini API. If False, use MockLLMProvider
            use_openrouter: If True, use OpenRouter API instead of direct Gemini
        """
        self.use_real_api = use_real_api
        self.use_openrouter = use_openrouter
        self.dataset = None
        self.orchestrator = None
        self.test_results = []
        
    def setup_llm_provider(self):
        """Setup LLM provider (Gemini Direct, OpenRouter, or Mock)"""
        if self.use_real_api:
            if self.use_openrouter:
                # Use OpenRouter with default key
                api_key = os.getenv('OPENROUTER_API_KEY', DEFAULT_OPENROUTER_KEY)
                if not api_key or api_key == "your-api-key-here":
                    print("⚠️ No OpenRouter API key found. Using Mock LLM instead.")
                    self.use_real_api = False
                    return MockLLMProvider()
                else:
                    print("🔗 Using OpenRouter API with Gemini Flash")
                    return GeminiProvider(
                        api_key=api_key,
                        model_name="google/gemini-flash-1.5",
                        use_openrouter=True
                    )
            else:
                # Use direct Gemini API
                api_key = os.getenv('GEMINI_API_KEY', DEFAULT_GEMINI_KEY)
                if not api_key or api_key == "your-api-key-here":
                    print("⚠️ No Gemini API key found. Using Mock LLM instead.")
                    self.use_real_api = False
                    return MockLLMProvider()
                else:
                    print("🔗 Using direct Gemini API")
                    return GeminiProvider(api_key=api_key)
        else:
            print("🎭 Using Mock LLM provider")
            return MockLLMProvider({
                "default": "Based on the user's beauty preferences, I recommend foundation.",
                "think": "The user has shown interest in skincare and makeup products.",
                "analyze": "This user prefers premium beauty products with good reviews.",
                "finish": "Finish[B000015]"
            })
    
    def setup_beauty_dataset(self, use_real_data: bool = False):
        """
        Setup Beauty dataset
        
        Args:
            use_real_data: If True, try to load real Amazon Beauty data
        """
        print("\n📊 Setting up Beauty Dataset...")
        print("-" * 35)
        
        try:
            if use_real_data:
                # Try to use real Beauty dataset files
                data_path = "/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/data/inputs/reviews_Beauty.json"
                metadata_path = "/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/data/inputs/meta_Beauty.json"
                
                self.dataset = BeautyDataset(
                    data_path=data_path,
                    metadata_path=metadata_path
                )
                print(f"📄 Trying to load real data from: {data_path}")
            else:
                # Use synthetic data
                self.dataset = BeautyDataset(data_path="dummy_path.json")
                print("🧪 Using synthetic Beauty data")
            
            # Process the dataset
            print("⚙️ Processing dataset...")
            self.dataset.process_data()
            
            # Get statistics
            stats = self.dataset.get_statistics()
            print(f"✅ Dataset loaded successfully!")
            print(f"   Sessions: {stats['num_sessions']}")
            print(f"   Users: {stats['num_users']}")
            print(f"   Items: {stats['num_items']}")
            print(f"   Total interactions: {stats['total_interactions']}")
            print(f"   Avg session length: {stats['avg_session_length']:.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset setup failed: {e}")
            return False
    
    def setup_orchestrator(self, reflection_strategy: ReflectionStrategy = ReflectionStrategy.REFLEXION):
        """Setup AgentOrchestrator with specified reflection strategy"""
        print(f"\n🎭 Setting up AgentOrchestrator...")
        print(f"   Reflection strategy: {reflection_strategy.value}")
        
        try:
            llm_provider = self.setup_llm_provider()
            self.orchestrator = AgentOrchestrator(
                llm_provider=llm_provider,
                reflection_strategy=reflection_strategy
            )
            
            # Update orchestrator with dataset information
            if self.dataset and hasattr(self.dataset, 'item_to_name'):
                print("📝 Updating orchestrator with dataset info...")
                self.orchestrator.update_agent_data(
                    user_data={},  # Could add user profiles here
                    item_data=self.dataset.item_to_name,
                    user_histories={}  # Could add interaction histories here
                )
            
            print("✅ Orchestrator setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Orchestrator setup failed: {e}")
            return False
    
    def get_test_sessions(self, num_sessions: int = 5) -> List[Dict[str, Any]]:
        """Get test sessions from the dataset"""
        if not self.dataset or not hasattr(self.dataset, 'sessions'):
            return []
        
        # Get random sessions that have enough items for testing
        valid_sessions = [s for s in self.dataset.sessions if len(s['items']) >= 3]
        
        if len(valid_sessions) == 0:
            print("⚠️ No valid sessions found for testing")
            return []
        
        # Sample random sessions
        test_sessions = random.sample(
            valid_sessions, 
            min(num_sessions, len(valid_sessions))
        )
        
        print(f"📋 Selected {len(test_sessions)} test sessions")
        return test_sessions
    
    def test_orchestrator_recommend(self, session: Dict[str, Any], max_iterations: int = 3) -> Dict[str, Any]:
        """
        Test orchestrator.recommend() with a real session
        
        Args:
            session: Dataset session with user items
            max_iterations: Max iterations for orchestrator
            
        Returns:
            Test result dictionary
        """
        try:
            # Prepare evaluation data (leave-one-out)
            pred_sequence, target = self.dataset.prepare_to_predict(session)
            candidates, target_idx = self.dataset.create_candidate_pool(session)
            
            # Create recommendation request
            request = RecommendationRequest(
                user_id=session['user_id'],
                user_sequence=pred_sequence,
                candidates=candidates,
                ground_truth=target
            )
            
            print(f"\n🔍 Testing Session: {session['user_id']}")
            print(f"   Sequence length: {len(pred_sequence)}")
            print(f"   Last 3 items: {' → '.join(pred_sequence[-3:])}")
            print(f"   Candidates: {len(candidates)}")
            print(f"   Ground truth: {target}")
            
            # Get recommendation from orchestrator
            print("🎭 Running orchestrator.recommend()...")
            response = self.orchestrator.recommend(request, max_iterations=max_iterations)
            
            # Evaluate result
            is_correct = response.recommendation == target
            confidence = response.confidence
            reasoning = response.reasoning
            
            # Get recommendation rank in candidates
            try:
                recommendation_rank = candidates.index(response.recommendation) + 1
            except ValueError:
                recommendation_rank = len(candidates) + 1  # Not found
            
            result = {
                'session_id': session['user_id'],
                'sequence_length': len(pred_sequence),
                'num_candidates': len(candidates),
                'ground_truth': target,
                'recommendation': response.recommendation,
                'confidence': confidence,
                'is_correct': is_correct,
                'recommendation_rank': recommendation_rank,
                'reasoning': reasoning,
                'metadata': response.metadata,
                'success': True
            }
            
            print(f"✅ Result:")
            print(f"   Recommendation: {response.recommendation}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Accuracy: {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"   Rank: {recommendation_rank}/{len(candidates)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Test failed for session {session.get('user_id', 'unknown')}: {e}")
            return {
                'session_id': session.get('user_id', 'unknown'),
                'success': False,
                'error': str(e)
            }
        
        finally:
            # Reset orchestrator for next test
            if self.orchestrator:
                self.orchestrator.reset_session()
    
    def run_comprehensive_test(self, num_sessions: int = 5, use_real_data: bool = False):
        """
        Run comprehensive test of AgentOrchestrator with real data
        
        Args:
            num_sessions: Number of sessions to test
            use_real_data: Whether to try loading real Beauty dataset files
        """
        print("🚀 AgentOrchestrator Real Data Test")
        print("=" * 45)
        
        # Setup dataset
        if not self.setup_beauty_dataset(use_real_data):
            print("❌ Dataset setup failed. Cannot proceed.")
            return
        
        # Setup orchestrator
        if not self.setup_orchestrator():
            print("❌ Orchestrator setup failed. Cannot proceed.")
            return
        
        # Get test sessions
        test_sessions = self.get_test_sessions(num_sessions)
        if not test_sessions:
            print("❌ No test sessions available. Cannot proceed.")
            return
        
        print(f"\n🧪 Testing {len(test_sessions)} sessions...")
        print("-" * 35)
        
        # Run tests
        self.test_results = []
        for i, session in enumerate(test_sessions, 1):
            print(f"\n[{i}/{len(test_sessions)}] ", end="")
            result = self.test_orchestrator_recommend(session)
            self.test_results.append(result)
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and display test results"""
        print(f"\n📊 Test Results Analysis")
        print("=" * 30)
        
        successful_tests = [r for r in self.test_results if r['success']]
        failed_tests = [r for r in self.test_results if not r['success']]
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for test in failed_tests:
                print(f"   Session {test['session_id']}: {test.get('error', 'Unknown error')}")
        
        if successful_tests:
            # Calculate accuracy
            correct_predictions = sum(1 for r in successful_tests if r['is_correct'])
            accuracy = correct_predictions / len(successful_tests)
            
            # Calculate average metrics
            avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests)
            avg_rank = sum(r['recommendation_rank'] for r in successful_tests) / len(successful_tests)
            
            print(f"\n✅ Success Metrics:")
            print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{len(successful_tests)})")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Average recommendation rank: {avg_rank:.1f}")
            
            # Show hit rate at different positions
            hit_at_1 = sum(1 for r in successful_tests if r['recommendation_rank'] == 1)
            hit_at_3 = sum(1 for r in successful_tests if r['recommendation_rank'] <= 3)
            hit_at_5 = sum(1 for r in successful_tests if r['recommendation_rank'] <= 5)
            
            print(f"   Hit@1: {hit_at_1/len(successful_tests):.3f}")
            print(f"   Hit@3: {hit_at_3/len(successful_tests):.3f}")
            print(f"   Hit@5: {hit_at_5/len(successful_tests):.3f}")
        
        # Show system performance
        if self.orchestrator:
            stats = self.orchestrator.get_system_stats()
            print(f"\n⚡ System Performance:")
            print(f"   Total requests: {stats['orchestrator']['total_requests']}")
            print(f"   Success rate: {stats['orchestrator']['success_rate']:.3f}")
            print(f"   Avg response time: {stats['orchestrator']['avg_response_time']:.3f}s")
            
            # Agent performance
            agent_stats = stats['agents']
            for agent_name, perf in agent_stats.items():
                if perf['total_calls'] > 0:
                    print(f"   {agent_name}: {perf['total_calls']} calls, {perf['avg_time_per_call']:.3f}s avg")
    
    def run_strategy_comparison(self, num_sessions: int = 3):
        """Compare different reflection strategies"""
        print("🎯 Reflection Strategy Comparison")
        print("=" * 40)
        
        strategies = [
            ReflectionStrategy.NONE,
            ReflectionStrategy.REFLEXION,
            ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION
        ]
        
        strategy_results = {}
        
        for strategy in strategies:
            print(f"\n🔍 Testing strategy: {strategy.value}")
            print("-" * 30)
            
            # Setup orchestrator with this strategy
            if not self.setup_orchestrator(strategy):
                continue
            
            # Get fresh test sessions
            test_sessions = self.get_test_sessions(num_sessions)
            if not test_sessions:
                continue
            
            # Run tests
            results = []
            for session in test_sessions:
                result = self.test_orchestrator_recommend(session, max_iterations=2)
                if result['success']:
                    results.append(result)
            
            if results:
                accuracy = sum(1 for r in results if r['is_correct']) / len(results)
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                strategy_results[strategy.value] = {
                    'accuracy': accuracy,
                    'confidence': avg_confidence,
                    'num_tests': len(results)
                }
                
                print(f"   Results: {accuracy:.3f} accuracy, {avg_confidence:.3f} confidence")
        
        # Compare strategies
        print(f"\n📈 Strategy Comparison Summary:")
        for strategy_name, results in strategy_results.items():
            print(f"   {strategy_name:25}: {results['accuracy']:.3f} acc, {results['confidence']:.3f} conf ({results['num_tests']} tests)")


def main():
    """Main test execution"""
    print("AgentOrchestrator Real Data Testing")
    print("=" * 50)
    
    # Configuration
    USE_REAL_API = True      # Set to False to use Mock LLM
    USE_OPENROUTER = True    # Set to True to use OpenRouter instead of direct Gemini
    USE_REAL_DATA = True     # Set to True to try loading real Beauty dataset files
    NUM_SESSIONS = 2         # Number of sessions to test
    
    try:
        # Initialize tester
        tester = OrchestoratorRealDataTester(
            use_real_api=USE_REAL_API,
            use_openrouter=USE_OPENROUTER
        )
        
        # Show configuration
        api_mode = "OpenRouter" if USE_OPENROUTER else "Direct Gemini" if USE_REAL_API else "Mock"
        data_mode = "Real Beauty data" if USE_REAL_DATA else "Synthetic data"
        print(f"🔧 Configuration: {api_mode} API, {data_mode}, {NUM_SESSIONS} sessions")
        
        # Run comprehensive test
        tester.run_comprehensive_test(
            num_sessions=NUM_SESSIONS,
            use_real_data=USE_REAL_DATA
        )
        
        # Run strategy comparison (smaller set)
        print("\n" + "="*50)
        tester.run_strategy_comparison(num_sessions=3)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()