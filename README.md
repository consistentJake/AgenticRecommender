# Agentic Sequential Recommendation System

A comprehensive multi-agent recommendation system combining **MACRec's agent architecture**, **LLM-Sequential-Recommendation's dataset processing**, and **ThinkRec's dual-objective training** approach.

## 🚀 Quick Start


```bash

source venv/bin/activate
# Install dependencies
pip install -r requirements.txt

# Run complete system demo
python examples/complete_pipeline_demo.py

# Run specific tests
python tests/test_agents.py
python tests/test_integration.py
```

## 🏗️ System Architecture

### Core Components

1. **Multi-Agent System** (MACRec-inspired)
   - **Manager**: Central orchestrator with think/act mechanism
   - **Analyst**: User and item sequential pattern analysis  
   - **Reflector**: Iterative improvement through reflection

2. **Dataset Processing** (LLM-Sequential-Recommendation compatible)
   - Beauty dataset (Amazon 5-core)
   - Delivery Hero dataset
   - Leave-one-out evaluation setup

3. **Evaluation Framework**
   - Hit Rate@K, NDCG@K, MRR metrics
   - Comprehensive performance tracking
   - Reflection-based improvements

### Agent Communication Protocol

```python
# Example workflow
from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models.llm_provider import GeminiProvider

# Initialize system
llm = GeminiProvider(api_key="your-api-key")
pipeline = create_pipeline(llm, "beauty", "reflexion")

# Load dataset
pipeline.load_dataset("path/to/data.json", use_synthetic=True)

# Make recommendation
request = RecommendationRequest(
    user_id="user_123",
    user_sequence=["laptop", "mouse", "keyboard"],
    candidates=["monitor", "headphones", "webcam"]
)

response = pipeline.orchestrator.recommend(request)
print(f"Recommendation: {response.recommendation}")
print(f"Confidence: {response.confidence}")
print(f"Reasoning: {response.reasoning}")
```

## 📊 Key Features

- **Multi-Agent Coordination**: Manager orchestrates Analyst and Reflector for comprehensive reasoning
- **Reflection Strategies**: Four strategies (none, last_trial, reflexion, combined)
- **Sequential Analysis**: Deep understanding of user interaction patterns
- **Comprehensive Logging**: Full traceability of agent decisions and performance
- **Flexible Datasets**: Support for Beauty and Delivery Hero datasets
- **Robust Evaluation**: Leave-one-out setup with multiple metrics

## 🧪 Testing

The system includes comprehensive test coverage across multiple levels:

```bash
# Component-level tests
python tests/test_agents.py              # Agent functionality
python tests/test_datasets.py           # Dataset processing  
python tests/test_pipeline.py           # Pipeline integration
python tests/test_system_integration.py # End-to-end system
python tests/test_api_integration.py    # API integration

# Comprehensive verification
python tests/test_all_components.py     # Full system verification
```

### Test Coverage Details

#### Unit Tests (`tests/test_agents.py`)
- **Manager Agent**: Think/act cycle, action parsing, performance tracking
- **Analyst Agent**: User/item analysis, tool execution, data source updates  
- **Reflector Agent**: Multiple reflection strategies, reflection parsing
- **Base Classes**: Agent types, performance metrics, state management

#### Integration Tests (`tests/test_system_integration.py`)
- **Agent Communication**: Manager → Analyst → Reflector workflow
- **Multi-agent Coordination**: Complete recommendation pipeline
- **LLM Integration**: MockLLMProvider and GeminiProvider testing
- **End-to-end Workflow**: Request → Processing → Response cycle

#### Component Tests (`tests/test_all_components.py`)
- **Metrics System**: HR@K, NDCG@K, MRR calculation verification
- **Logging System**: Performance tracking, action logging
- **Dataset Processing**: Synthetic data generation, evaluation splits
- **Orchestrator**: Request handling, response structure validation
- **Complete Pipeline**: Factory functions, demo predictions

Each test file includes specific scenarios:
- ✅ **Functional Correctness**: All components work as designed
- ✅ **Error Handling**: Graceful failure and fallback mechanisms  
- ✅ **Performance Tracking**: Metrics collection and reporting
- ✅ **Integration Points**: Proper component interaction

## 📁 Project Structure

```
agentic_recommender/
├── agents/             # Multi-agent system
│   ├── base.py         # Base agent classes (Agent, ToolAgent, AgentType, ReflectionStrategy)
│   ├── manager.py      # Manager agent (two-stage think/act mechanism)
│   ├── analyst.py      # Analyst agent (user/item analysis with tools)
│   └── reflector.py    # Reflector agent (multiple reflection strategies)
├── core/               # Orchestration layer
│   └── orchestrator.py # AgentOrchestrator (coordinates multi-agent workflow)
├── datasets/           # Data processing pipeline
│   ├── base_dataset.py # SequentialDataset base class
│   └── beauty_dataset.py # BeautyDataset & DeliveryHeroDataset
├── models/             # LLM provider abstraction
│   └── llm_provider.py # LLMProvider, GeminiProvider, MockLLMProvider
├── system/             # Complete pipeline integration
│   └── pipeline.py     # RecommendationPipeline (end-to-end system)
├── training/           # Training utilities  
│   └── data_preparation.py # Dataset preparation for training
└── utils/              # Shared utilities
    ├── metrics.py      # Evaluation metrics (HR@K, NDCG@K, MRR)
    └── logging.py      # Performance logging and tracking
```

### Core Components Detail

#### Agents (`agentic_recommender/agents/`)
- **`base.py:51`** - `Agent` abstract base class with performance tracking
- **`base.py:108`** - `ToolAgent` for command execution (used by Analyst)
- **`manager.py:17`** - `Manager` with dual LLM architecture (think_llm + action_llm)
- **`analyst.py:15`** - `Analyst` with data analysis tools (UserInfo, ItemInfo, etc.)
- **`reflector.py:20`** - `Reflector` supporting 4 reflection strategies

#### Core Orchestration (`agentic_recommender/core/`)
- **`orchestrator.py:36`** - `AgentOrchestrator` coordinates agent workflow
- **`orchestrator.py:18`** - `RecommendationRequest` input structure
- **`orchestrator.py:28`** - `RecommendationResponse` output structure

#### System Pipeline (`agentic_recommender/system/`)
- **`pipeline.py:19`** - `RecommendationPipeline` integrates all components
- **`pipeline.py:246`** - `create_pipeline()` factory function

## 🎯 Example Use Cases

### 1. E-commerce Recommendation
```python
# Tech product sequence
user_sequence = ["gaming_laptop", "mechanical_keyboard", "gaming_mouse"]  
candidates = ["wireless_headphones", "monitor", "webcam"]

response = pipeline.run_demo_prediction(user_sequence, candidates)
# → Recommends "monitor" based on sequential patterns
```

### 2. Beauty Product Recommendation  
```python
# Beauty routine sequence
user_sequence = ["foundation", "concealer", "lipstick"]
candidates = ["mascara", "eyeshadow", "blush"]

response = pipeline.run_demo_prediction(user_sequence, candidates)
# → Recommends "mascara" completing the look
```

## 📈 Performance Monitoring

The system provides comprehensive performance tracking through multiple layers:

### Agent-Level Metrics (`Agent.get_performance_stats()`)
```python
{
    'agent_type': 'Manager',
    'total_calls': 15,
    'avg_time_per_call': 0.234,
    'step_count': 8,
    'is_finished': True
}
```

### Orchestrator-Level Stats (`AgentOrchestrator.get_system_stats()`)  
```python
{
    'orchestrator': {
        'total_requests': 25,
        'successful_requests': 24,
        'success_rate': 0.96,
        'current_session': 'session_1756034452'
    },
    'agents': {
        'manager': {...},  # Individual agent stats
        'analyst': {...},
        'reflector': {...}
    },
    'communication': {
        'messages_in_session': 12,
        'last_session_id': 'session_1756034452'
    }
}
```

### Logging System (`utils/logging.py`)
- **Action Logging**: Every agent action with context and timing
- **Communication Tracking**: Inter-agent message flow
- **Performance Metrics**: Response times, token usage estimates
- **Session Management**: Conversation history and workflow traces

### Evaluation Framework (`utils/metrics.py`)
- **Hit Rate@K**: `hit_rate_at_k(predictions, ground_truth, k)`
- **NDCG@K**: `ndcg_at_k(predictions, ground_truth, k)` 
- **MRR**: `mrr(predictions, ground_truth)`
- **Batch Evaluation**: `evaluate_recommendations(pred_list, gt_list, k_values)`

### Real-time Monitoring
The system logs all activities to structured JSONL files:
```bash
logs/session_1756034452.jsonl  # Session-specific logs
```

Each log entry includes:
- Timestamp and agent identification
- Action type and parameters
- Execution duration and context
- Performance metrics and metadata

## 🔧 Configuration

### Reflection Strategies
- `none`: No reflection
- `last_trial`: Store previous attempt context
- `reflexion`: LLM-based reflection and improvement
- `last_trial_and_reflexion`: Combined approach

### LLM Providers
- `GeminiProvider`: Google's Gemini API
- `MockLLMProvider`: For testing and development

### Dataset Options
- `beauty`: Amazon Beauty 5-core dataset
- `delivery_hero`: Food delivery dataset
- Synthetic data generation for testing

## 🧠 Research Foundation

This implementation combines insights from three cutting-edge research papers:

### 1. **MACRec** - Multi-Agent Collaborative Reasoning
- **Paper**: `papers/MacRec.pdf`
- **Implementation Reference**: `previousWorks/MACRec/`
- **Key Contributions**:
  - Two-stage Manager architecture (think/act mechanism)
  - Tool-based Analyst agent with specialized commands
  - Multiple reflection strategies (none/last_trial/reflexion/combined)
  - Agent orchestration and communication protocols

### 2. **LLM-Sequential-Recommendation**
- **Implementation Reference**: `previousWorks/LLM-Sequential-Recommendation/`  
- **Key Contributions**:
  - Sequential dataset processing pipeline
  - Leave-one-out evaluation methodology
  - Comprehensive evaluation metrics (HR@K, NDCG@K, MRR)
  - Amazon Beauty and delivery datasets integration

### 3. **ThinkRec** - Reasoning-Enhanced Recommendation
- **Paper**: `papers/thinkRec.pdf`
- **Implementation Reference**: `previousWorks/ThinkRec/`
- **Key Contributions**:
  - Dual-objective training (accuracy + reasoning quality)
  - Sequential pattern analysis and reasoning chains
  - User preference understanding through reflection

### Implementation Architecture Mapping

| Research Component | Our Implementation | Code Reference |
|-------------------|-------------------|----------------|
| MACRec Manager | `Manager` class | `agents/manager.py:17` |
| MACRec Analyst | `Analyst` class | `agents/analyst.py:15` |
| MACRec Reflector | `Reflector` class | `agents/reflector.py:20` |
| LLM-SeqRec Datasets | `BeautyDataset` | `datasets/beauty_dataset.py:15` |
| LLM-SeqRec Metrics | Evaluation functions | `utils/metrics.py:10` |
| ThinkRec Reasoning | Orchestrator workflow | `core/orchestrator.py:149` |

### Analysis Documents
- **`MACRec_Analysis.md`**: Comprehensive MACRec analysis and implementation notes
- **`LLM_Sequential_Recommendation_Analysis.md`**: Dataset and evaluation framework analysis  
- **`ThinkRec_Technical_Analysis.md`**: Reasoning and training methodology analysis

## 🚦 Getting Started

1. **Clone and Install**:
   ```bash
   git clone <repository>
   pip install -r requirements.txt
   ```

2. **Set API Key** (optional for real LLM):
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

3. **Run Demo**:
   ```bash
   python examples/complete_pipeline_demo.py
   ```

4. **Explore Examples**:
   - `examples/demo_system.py` - Basic agent interaction
   - `examples/complete_pipeline_demo.py` - Full system demo

## 📚 Documentation

- `CLAUDE.md` - Project instructions and references
- `MACRec_Analysis.md` - Detailed technical analysis
- `Design_Document.md` - System design and architecture
- `papers/` - Research paper references

## 🎉 System Status

✅ **Complete Implementation**:
- Multi-agent system with orchestration
- Dataset processing and evaluation  
- Comprehensive testing and logging
- End-to-end recommendation pipeline
- Reflection-based improvements
- Performance monitoring

The system is **ready for research and production use** with full test coverage and comprehensive documentation.