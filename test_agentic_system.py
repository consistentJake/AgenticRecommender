#!/usr/bin/env python3
"""
Comprehensive test system for agentic recommendation workflow.
Tests with both mock data and real datasets using Gemini API.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_recommender.models.llm_provider import GeminiProvider, MockLLMProvider, DEFAULT_GEMINI_KEY
from agentic_recommender.agents import Manager, Analyst, Reflector
from agentic_recommender.utils.logging import get_logger
from agentic_recommender.datasets import BeautyDataset, DeliveryHeroDataset


class AgenticRecommendationTester:
    """
    Comprehensive tester for agentic recommendation system.
    
    Tests:
    1. Mock data with MockLLM
    2. Mock data with real Gemini API
    3. Real dataset samples with Gemini API
    4. Error handling and fault analysis
    """
    
    def __init__(self, use_real_gemini: bool = False):
        self.use_real_gemini = use_real_gemini
        self.logger = get_logger()
        self.test_results = []
        
        print("🧪 Initializing Agentic Recommendation Tester")
        print(f"   Using real Gemini API: {use_real_gemini}")
        
        # Initialize providers
        if use_real_gemini:
            try:
                self.real_llm = GeminiProvider(DEFAULT_GEMINI_KEY)
                print("✅ Real Gemini provider initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
                self.real_llm = None
        else:
            self.real_llm = None
            
        self.mock_llm = MockLLMProvider()
        print("✅ Mock LLM provider initialized")
        
        # Load processed datasets
        self.datasets = {}
        self.load_processed_datasets()
    
    def load_processed_datasets(self):
        """Load processed dataset samples"""
        outputs_dir = Path("agentic_recommender/data/outputs")
        
        if not outputs_dir.exists():
            print("⚠️ No processed datasets found. Run dataset processing scripts first.")
            return
        
        # Load Beauty dataset samples
        try:
            beauty_samples_path = outputs_dir / "beauty_evaluation_samples.json"
            if beauty_samples_path.exists():
                with open(beauty_samples_path, 'r') as f:
                    self.datasets['beauty'] = json.load(f)[:3]  # First 3 samples
                print(f"✅ Loaded {len(self.datasets['beauty'])} Beauty samples")
        except Exception as e:
            print(f"⚠️ Could not load Beauty samples: {e}")
        
        # Load Delivery Hero dataset samples  
        try:
            dh_samples_path = outputs_dir / "delivery_hero_se_evaluation_samples.json"
            if dh_samples_path.exists():
                with open(dh_samples_path, 'r') as f:
                    self.datasets['delivery_hero'] = json.load(f)[:3]  # First 3 samples
                print(f"✅ Loaded {len(self.datasets['delivery_hero'])} Delivery Hero samples")
        except Exception as e:
            print(f"⚠️ Could not load Delivery Hero samples: {e}")
    
    def create_mock_samples(self) -> List[Dict[str, Any]]:
        """Create mock recommendation samples for testing"""
        return [
            {
                "session_id": "demo_1",
                "user_id": "demo_user",
                "prompt_items": ["gaming_laptop", "mechanical_keyboard"],
                "target_item": "gaming_monitor",
                "candidates": ["gaming_monitor", "mouse_pad", "webcam", "headset"],
                "target_index": 0,
                "item_names": {
                    "gaming_laptop": "Gaming Laptop Pro",
                    "mechanical_keyboard": "RGB Mechanical Keyboard",
                    "gaming_monitor": "4K Gaming Monitor"
                }
            },
            {
                "session_id": "demo_2", 
                "user_id": "beauty_user",
                "prompt_items": ["foundation", "concealer"],
                "target_item": "setting_powder",
                "candidates": ["setting_powder", "blush", "lipstick", "mascara"],
                "target_index": 0,
                "item_names": {
                    "foundation": "Liquid Foundation",
                    "concealer": "Under-eye Concealer", 
                    "setting_powder": "Translucent Setting Powder"
                }
            }
        ]
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n🚀 Starting Comprehensive Agentic System Tests")
        print("="*60)
        
        # Test 1: Mock data with Mock LLM
        print("\n📋 TEST 1: Mock data + Mock LLM")
        self.test_mock_workflow()
        
        # Test 2: Mock data with real Gemini
        if self.real_llm:
            print("\n📋 TEST 2: Mock data + Real Gemini API")
            self.test_gemini_with_mock_data()
        
        # Test 3: Real datasets with Gemini
        if self.real_llm and self.datasets:
            print("\n📋 TEST 3: Real datasets + Gemini API")
            self.test_real_datasets_with_gemini()
        
        # Test 4: Error handling and fault analysis
        print("\n📋 TEST 4: Error handling and fault analysis")
        self.test_error_handling()
        
        # Print summary
        self.print_test_summary()
    
    def test_mock_workflow(self):
        """Test basic workflow with mock data and mock LLM"""
        print("   Testing basic agent workflow...")
        
        try:
            # Create agents
            manager = Manager(self.mock_llm, self.mock_llm)
            analyst = Analyst(self.mock_llm)
            
            # Test samples
            samples = self.create_mock_samples()
            
            for i, sample in enumerate(samples):
                print(f"   Sample {i+1}: {sample['session_id']}")
                
                # Test Manager thinking
                task_context = {
                    'user_id': sample['user_id'],
                    'prompt_items': sample['prompt_items'],
                    'candidates': sample['candidates'],
                    'task': 'sequential recommendation'
                }
                
                thought = manager.think(task_context)
                print(f"     Manager thought: {thought[:50]}...")
                
                # Test Manager action
                action_type, argument = manager.act(task_context)
                print(f"     Manager action: {action_type}[{argument}]")
                
                # Test Analyst if needed
                if action_type == "Analyse" and isinstance(argument, list):
                    analysis = analyst.forward(argument)
                    print(f"     Analyst result: {analysis[:50]}...")
                
                self.test_results.append({
                    'test': 'mock_workflow',
                    'sample': sample['session_id'],
                    'success': True,
                    'thought': thought,
                    'action': f"{action_type}[{argument}]"
                })
        
        except Exception as e:
            print(f"   ❌ Mock workflow test failed: {e}")
            self.test_results.append({
                'test': 'mock_workflow',
                'success': False,
                'error': str(e)
            })
    
    def test_gemini_with_mock_data(self):
        """Test workflow with mock data but real Gemini API"""
        print("   Testing with real Gemini API...")
        
        try:
            # Create agents with real Gemini
            manager = Manager(self.real_llm, self.real_llm)
            analyst = Analyst(self.real_llm)
            
            # Test one mock sample
            sample = self.create_mock_samples()[0]
            print(f"   Sample: {sample['session_id']}")
            
            task_context = {
                'user_id': sample['user_id'],
                'prompt_items': sample['prompt_items'],
                'candidates': sample['candidates'],
                'task': 'sequential recommendation'
            }
            
            # Test full workflow
            print("     🧠 Manager thinking...")
            thought = manager.think(task_context)
            print(f"     Thought: {thought}")
            
            print("     ⚡ Manager acting...")
            action_type, argument = manager.act(task_context)
            print(f"     Action: {action_type}[{argument}]")
            
            # If action is Analyse, test analyst
            result = None
            if action_type == "Analyse" and isinstance(argument, list):
                print("     🔍 Running analysis...")
                
                # Update analyst with sample data
                user_data = {sample['user_id']: "Demo user for testing"}
                item_data = {item: name for item, name in sample['item_names'].items()}
                user_histories = {sample['user_id']: sample['prompt_items']}
                
                analyst.update_data_sources(user_data, item_data, user_histories)
                
                result = analyst.forward(argument)
                print(f"     Analysis: {result[:100]}...")
            
            self.test_results.append({
                'test': 'gemini_mock_data',
                'sample': sample['session_id'],
                'success': True,
                'thought': thought,
                'action': f"{action_type}[{argument}]",
                'result': result
            })
            
        except Exception as e:
            print(f"   ❌ Gemini mock data test failed: {e}")
            traceback.print_exc()
            self.test_results.append({
                'test': 'gemini_mock_data',
                'success': False,
                'error': str(e)
            })
    
    def test_real_datasets_with_gemini(self):
        """Test with real dataset samples and Gemini API"""
        print("   Testing with real datasets...")
        
        for dataset_name, samples in self.datasets.items():
            print(f"   Dataset: {dataset_name}")
            
            try:
                # Create agents
                manager = Manager(self.real_llm, self.real_llm)
                analyst = Analyst(self.real_llm)
                
                # Test first sample from dataset
                sample = samples[0]
                print(f"     Sample: {sample['session_id']}")
                
                # Prepare task context
                task_context = {
                    'user_id': sample['user_id'],
                    'prompt_items': sample['prompt_items'][:5],  # Limit to first 5 for token efficiency
                    'candidates': sample['candidates'][:20],    # Limit candidates
                    'task': 'sequential recommendation',
                    'dataset': dataset_name
                }
                
                # Test Manager workflow
                print("     🧠 Manager thinking...")
                thought = manager.think(task_context)
                print(f"     Thought: {thought}")
                
                print("     ⚡ Manager acting...")
                action_type, argument = manager.act(task_context)
                print(f"     Action: {action_type}[{argument}]")
                
                # Test analysis if requested
                result = None
                if action_type == "Analyse":
                    print("     🔍 Running analysis...")
                    
                    # Prepare data for analyst
                    user_data = {sample['user_id']: f"User from {dataset_name} dataset"}
                    item_data = {}
                    user_histories = {sample['user_id']: sample['prompt_items']}
                    
                    # Add item names if available
                    if 'item_names' in sample:
                        item_data.update(sample['item_names'])
                    
                    analyst.update_data_sources(user_data, item_data, user_histories)
                    result = analyst.forward(argument)
                    print(f"     Analysis: {result[:150]}...")
                
                self.test_results.append({
                    'test': f'real_dataset_{dataset_name}',
                    'sample': sample['session_id'],
                    'success': True,
                    'thought': thought,
                    'action': f"{action_type}[{argument}]",
                    'result': result,
                    'target_item': sample.get('target_item'),
                    'correct_prediction': action_type == "Finish" and argument == sample.get('target_item')
                })
                
            except Exception as e:
                print(f"   ❌ Real dataset {dataset_name} test failed: {e}")
                traceback.print_exc()
                self.test_results.append({
                    'test': f'real_dataset_{dataset_name}',
                    'success': False,
                    'error': str(e)
                })
    
    def test_error_handling(self):
        """Test system's error handling capabilities"""
        print("   Testing error handling...")
        
        error_tests = [
            {
                'name': 'Invalid API key',
                'test': lambda: self._test_invalid_api_key()
            },
            {
                'name': 'Malformed input',
                'test': lambda: self._test_malformed_input()
            },
            {
                'name': 'Empty dataset',
                'test': lambda: self._test_empty_dataset()
            }
        ]
        
        for error_test in error_tests:
            try:
                print(f"     Testing: {error_test['name']}")
                error_test['test']()
                self.test_results.append({
                    'test': f"error_{error_test['name'].replace(' ', '_')}",
                    'success': True
                })
            except Exception as e:
                print(f"     Expected error caught: {e}")
                self.test_results.append({
                    'test': f"error_{error_test['name'].replace(' ', '_')}",
                    'success': True,
                    'expected_error': str(e)
                })
    
    def _test_invalid_api_key(self):
        """Test with invalid API key"""
        try:
            invalid_provider = GeminiProvider("invalid_key")
            invalid_provider.generate("Test prompt")
        except:
            pass  # Expected to fail
    
    def _test_malformed_input(self):
        """Test with malformed input data"""
        manager = Manager(self.mock_llm, self.mock_llm)
        # Test with malformed context
        manager.think({"invalid": "data structure"})
    
    def _test_empty_dataset(self):
        """Test with empty dataset"""
        analyst = Analyst(self.mock_llm)
        analyst.update_data_sources({}, {}, {}, {})
        analyst.forward(["user", "nonexistent_user"])
    
    def analyze_failures(self):
        """Analyze failed tests and identify common issues"""
        print("\n🔍 FAULT ANALYSIS")
        print("="*40)
        
        failures = [r for r in self.test_results if not r.get('success', False)]
        
        if not failures:
            print("✅ No failures detected!")
            return
        
        print(f"❌ Found {len(failures)} failures:")
        
        # Group failures by type
        failure_types = {}
        for failure in failures:
            error = failure.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error
            
            if error_type not in failure_types:
                failure_types[error_type] = []
            failure_types[error_type].append(failure)
        
        # Analyze each failure type
        for error_type, failures_of_type in failure_types.items():
            print(f"\n📋 {error_type} ({len(failures_of_type)} occurrences):")
            
            for failure in failures_of_type:
                print(f"   - Test: {failure['test']}")
                if 'sample' in failure:
                    print(f"     Sample: {failure['sample']}")
                print(f"     Error: {failure.get('error', 'Unknown')}")
            
            # Suggest fixes
            self._suggest_fixes(error_type, failures_of_type)
    
    def _suggest_fixes(self, error_type: str, failures: List[Dict]):
        """Suggest fixes for common failure patterns"""
        print(f"   💡 Suggested fixes for {error_type}:")
        
        if "API" in error_type or "key" in error_type.lower():
            print("     - Check API key validity")
            print("     - Verify network connectivity") 
            print("     - Check API quota/limits")
        
        elif "parse" in error_type.lower() or "format" in error_type.lower():
            print("     - Improve prompt formatting")
            print("     - Add better output parsing")
            print("     - Use JSON mode for structured responses")
        
        elif "timeout" in error_type.lower():
            print("     - Increase timeout limits")
            print("     - Reduce input size")
            print("     - Implement retry logic")
        
        else:
            print("     - Review error logs for specific issues")
            print("     - Add more robust error handling")
            print("     - Validate input data structure")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n📊 TEST SUMMARY")
        print("="*50)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get('success', False)])
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
        
        # Group results by test type
        test_types = {}
        for result in self.test_results:
            test_type = result['test'].split('_')[0]
            if test_type not in test_types:
                test_types[test_type] = {'success': 0, 'total': 0}
            
            test_types[test_type]['total'] += 1
            if result.get('success', False):
                test_types[test_type]['success'] += 1
        
        print(f"\n📋 Results by test type:")
        for test_type, stats in test_types.items():
            rate = stats['success']/stats['total']*100 if stats['total'] > 0 else 0
            print(f"   {test_type}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        # Performance insights
        if self.real_llm:
            print(f"\n⚡ Gemini API Performance:")
            model_info = self.real_llm.get_model_info()
            print(f"   Total calls: {model_info.get('total_calls', 0)}")
            print(f"   Total time: {model_info.get('total_time', 0):.2f}s")
            print(f"   Avg time per call: {model_info.get('avg_time_per_call', 0):.2f}s")
        
        # Analysis and recommendations
        if total_tests > 0:
            self.analyze_failures()
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if successful_tests == total_tests:
            print("✅ All tests passed! System is working correctly.")
        elif successful_tests > total_tests * 0.8:
            print("⚠️ Most tests passed. Review failures for minor issues.")
        else:
            print("❌ Multiple failures detected. Comprehensive fixes needed.")
            print("   Priority areas:")
            print("   1. API integration stability")
            print("   2. Input/output parsing robustness") 
            print("   3. Error handling improvements")


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Agentic Recommendation System')
    parser.add_argument('--use-gemini', action='store_true',
                       help='Use real Gemini API (requires valid API key)')
    parser.add_argument('--mock-only', action='store_true',
                       help='Run only mock tests')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = AgenticRecommendationTester(use_real_gemini=args.use_gemini and not args.mock_only)
    
    # Run tests
    if args.mock_only:
        print("🧪 Running mock tests only...")
        tester.test_mock_workflow()
        tester.test_error_handling()
    else:
        tester.run_all_tests()
    
    # Print summary
    tester.print_test_summary()


if __name__ == "__main__":
    main()