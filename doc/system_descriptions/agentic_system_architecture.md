# Agentic Recommendation System Architecture and Flow

## Overview

The agentic recommendation system is a multi-agent architecture inspired by MACRec (Multi-Agent Collaborative Recommendation) that combines sequential recommendation capabilities with LLM-powered reasoning agents. The system processes user interaction sequences and generates personalized recommendations through coordinated agent workflows.

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Agent Layer    │    │  System Layer   │
│                 │    │                 │    │                 │
│ • Beauty Dataset│    │ • Manager       │    │ • Orchestrator  │
│ • Delivery Hero │◄──►│ • Analyst       │◄──►│ • Pipeline      │
│ • Base Dataset  │    │ • Reflector     │    │ • Metrics       │
│                 │    │ • Base Agents   │    │ • Logging       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Model Layer   │
                       │                 │
                       │ • Gemini LLM    │
                       │ • Mock LLM      │
                       │ • Provider Base │
                       └─────────────────┘
```

## 1. Core Agent Architecture

### Base Agent Framework
**File**: `agentic_recommender/agents/base.py`

The foundation provides common functionality for all agents:

```python
class Agent(ABC):
    """Abstract base agent with performance tracking and logging"""
    def __init__(self, agent_type: AgentType, llm_provider: LLMProvider):
        self.performance_tracker = PerformanceTracker()
        self.logger = get_logger(f"Agent.{agent_type.value}")
        
    @abstractmethod
    def forward(self, task: str) -> str:
        """Process task and return response"""
        pass
```

**Agent Types**: Manager, Analyst, Reflector, Searcher, Interpreter

### Manager Agent
**File**: `agentic_recommender/agents/manager.py`

Central orchestrator with two-stage LLM architecture:

```python
class Manager(Agent):
    def __init__(self, thought_llm: LLMProvider, action_llm: LLMProvider):
        self.thought_llm = thought_llm  # High temp (0.8) for reasoning
        self.action_llm = action_llm    # Low temp (0.3) for structured output
```

**Two-Phase Workflow**:
1. **Think Phase**: Creative reasoning and planning
2. **Act Phase**: Structured action generation

**Available Actions**:
- `Analyse[user/item, id]` - Request user/item analysis
- `Search[query]` - Search for relevant information
- `Finish[result]` - Provide final recommendation

### Analyst Agent
**File**: `agentic_recommender/agents/analyst.py`

Specialized in user and item analysis with comprehensive tools:

**Available Tools**:
```python
# User Analysis Tools
UserInfo[user_id]           # Get user profile
UserHistory[user_id, k]     # Get recent k interactions
SequenceAnalysis[sequence]  # Analyze interaction patterns

# Item Analysis Tools  
ItemInfo[item_id]           # Get item details
ItemHistory[item_id, k]     # Get item interaction history
```

### Reflector Agent
**File**: `agentic_recommender/agents/reflector.py`

Provides iterative improvement through self-assessment:

**Reflection Strategies**:
```python
class ReflectionStrategy(Enum):
    NONE = "none"                              # No reflection
    LAST_ATTEMPT = "last_attempt"              # Use previous context
    REFLEXION = "reflexion"                    # LLM-generated reflection
    LAST_ATTEMPT_AND_REFLEXION = "combined"   # Both approaches
```

**Output Format**:
```json
{
    "reflection": "Analysis of what went wrong and why",
    "is_correct": false,
    "improvement_suggestions": [
        "Analyze user's preference for gaming items",
        "Consider sequence patterns more carefully"
    ]
}
```

## 2. System Flow and Data Pipeline

### Complete Data Flow

```
Raw Data → Dataset Processing → Session Creation → Agent Orchestration → Recommendations
    ↓              ↓                 ↓                    ↓                    ↓
Reviews/       5-core Filter    User Sequences    Multi-Agent Workflow   Ranked Items
Orders         Product Names    Timestamps        Think→Act→Analyse       Evaluation
```

### Dataset Processing Pipeline

**File**: `agentic_recommender/datasets/base_dataset.py`

```python
class SequentialDataset:
    def process_data(self):
        # 1. Load raw interaction data
        raw_data = self._load_raw_data()
        
        # 2. Create user sessions with temporal ordering
        sessions = self._create_sessions(raw_data)
        
        # 3. Process item metadata (names, categories)
        item_names = self._process_metadata()
        
        # 4. Generate statistics and splits
        self._generate_statistics()
```

### Agent Orchestration Workflow

**File**: `agentic_recommender/core/orchestrator.py`

```python
def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
    iteration = 0
    max_iterations = 10
    
    while iteration < max_iterations:
        # 1. Manager thinking phase
        thought = self.manager.think(task_context)
        
        # 2. Manager action decision
        action_type, argument = self.manager.act(task_context)
        
        # 3. Route to appropriate agent
        if action_type == "Analyse":
            result = self.analyst.forward(argument)
        elif action_type == "Search":
            result = self._handle_search_request(argument)
        elif action_type == "Finish":
            break
            
        # 4. Update context with results
        task_context['last_analysis'] = result
        iteration += 1
    
    return self._create_response(task_context)
```

## 3. Testing and Usage

### Quick Start Testing

**File**: `examples/complete_pipeline_demo.py`

```python
# Initialize system with LLM provider
from agentic_recommender.models.llm_provider import GeminiProvider
from agentic_recommender.system.pipeline import create_pipeline

# Setup
llm_provider = GeminiProvider()
pipeline = create_pipeline(llm_provider, "beauty", "reflexion")

# Load dataset (synthetic for demo)
pipeline.load_dataset(use_synthetic=True)

# Run prediction
response = pipeline.run_demo_prediction(
    user_sequence=["moisturizer", "cleanser", "serum"],
    candidates=["foundation", "lipstick", "mascara", "sunscreen"]
)

print(f"Recommended: {response.recommended_items}")
print(f"Reasoning: {response.reasoning}")
```

### Comprehensive Testing

**File**: `test_agentic_system.py`

```python
# Full system test with multiple scenarios
tester = AgenticRecommendationTester()

# Test scenarios
scenarios = [
    ("mock", "beauty", False),      # Mock LLM with synthetic data
    ("gemini", "beauty", False),    # Real Gemini with synthetic data
    ("gemini", "deliveryhero", True) # Real Gemini with real data
]

for llm_type, dataset_type, use_real_data in scenarios:
    results = tester.run_test_scenario(llm_type, dataset_type, use_real_data)
    print(f"Results: {results}")
```

### System Integration Tests

**File**: `tests/test_system_integration.py`

```python
def test_agent_coordination():
    # Test inter-agent communication
    manager = Manager(llm, llm)
    analyst = Analyst(llm)
    reflector = Reflector(llm, ReflectionStrategy.REFLEXION)
    
    # Execute workflow
    thought = manager.think("Recommend items for user who likes tech")
    action_type, argument = manager.act(thought)
    
    if action_type == "Analyse":
        analysis = analyst.forward(argument)
    
    reflection = reflector.forward(task, scratchpad, attempt)
    assert reflection is not None
```

## 4. Evaluation Framework

### Metrics Implementation
**File**: `agentic_recommender/utils/metrics.py`

```python
def evaluate_recommendations(predictions_list, ground_truths, k_values=[1, 3, 5, 10]):
    """
    Evaluate recommendation performance using standard metrics
    
    Returns:
        Dict with metrics like:
        {
            'hr@1': 0.23, 'hr@3': 0.35, 'hr@5': 0.42,
            'ndcg@1': 0.23, 'ndcg@3': 0.31, 'ndcg@5': 0.36,
            'mrr': 0.34, 'precision@5': 0.08, 'recall@5': 0.42
        }
    """
```

**Supported Metrics**:
- **Hit Rate@K**: Binary relevance at rank K
- **NDCG@K**: Normalized discounted cumulative gain  
- **MRR**: Mean reciprocal rank
- **Precision@K** and **Recall@K**: Traditional IR metrics

### Dataset Evaluation Setup

```python
# Prepare evaluation data
pred_sequence, target = dataset.prepare_to_predict(session)
candidates, target_idx = dataset.create_candidate_pool(session)

# Create evaluation request
request = RecommendationRequest(
    user_id=session['user_id'],
    user_sequence=pred_sequence,
    candidates=candidates,
    ground_truth=target
)
```

## 5. System Configuration

### LLM Provider Setup

**Gemini Integration**:
```python
from agentic_recommender.models.llm_provider import GeminiProvider

# Configure with API key
provider = GeminiProvider(
    api_key="your-gemini-api-key",
    model_name="gemini-2.0-flash-exp"
)
```

**Mock Provider for Testing**:
```python
from agentic_recommender.models.llm_provider import MockLLMProvider

provider = MockLLMProvider()  # Returns predefined responses
```

### Reflection Strategy Configuration

```python
# Available strategies
strategies = [
    ReflectionStrategy.NONE,                    # No reflection
    ReflectionStrategy.REFLEXION,               # LLM-based reflection
    ReflectionStrategy.LAST_ATTEMPT,            # Context preservation
    ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION  # Combined approach
]

# Create pipeline with specific strategy
pipeline = create_pipeline(llm_provider, dataset_name, "reflexion")
```

## Key File References

### Core System Files
- `agentic_recommender/agents/base.py` - Agent framework
- `agentic_recommender/agents/manager.py` - Central orchestrator
- `agentic_recommender/agents/analyst.py` - Analysis specialist  
- `agentic_recommender/agents/reflector.py` - Self-improvement agent
- `agentic_recommender/core/orchestrator.py` - Agent coordination
- `agentic_recommender/system/pipeline.py` - Complete pipeline

### Dataset Processing
- `agentic_recommender/datasets/base_dataset.py` - Dataset framework
- `agentic_recommender/datasets/beauty_dataset.py` - Beauty dataset processor
- `agentic_recommender/datasets/delivery_hero_dataset.py` - Food delivery data

### Testing and Examples  
- `examples/complete_pipeline_demo.py` - Complete workflow demo
- `test_agentic_system.py` - Comprehensive system testing
- `tests/test_system_integration.py` - Integration testing

### Utilities
- `agentic_recommender/models/llm_provider.py` - LLM integration
- `agentic_recommender/utils/metrics.py` - Evaluation metrics
- `agentic_recommender/utils/logging.py` - System logging

## Getting Started

1. **Install Dependencies**: Set up Python environment with required packages
2. **Configure LLM**: Set up Gemini API key or use Mock provider
3. **Run Demo**: Execute `examples/complete_pipeline_demo.py`
4. **Run Tests**: Execute `test_agentic_system.py` for comprehensive testing
5. **Custom Dataset**: Extend `SequentialDataset` for your own data

The system provides a complete framework for agentic sequential recommendation with clear extensibility points for additional agents, datasets, and evaluation strategies.