# Agentic Sequential Recommendation System

A comprehensive multi-agent recommendation system combining **MACRec's agent architecture**, **LLM-Sequential-Recommendation's dataset processing**, and **ThinkRec's dual-objective training** approach.

## 🚀 Quick Start

```bash
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

The system includes comprehensive test coverage:

```bash
# Run all tests
python tests/test_agents.py          # Agent functionality
python tests/test_datasets.py       # Dataset processing  
python tests/test_pipeline.py       # Pipeline integration
python tests/test_integration.py    # End-to-end system
```

## 📁 Project Structure

```
agentic_recommender/
├── agents/          # Multi-agent system
│   ├── base.py      # Base agent classes
│   ├── manager.py   # Manager (think/act)
│   ├── analyst.py   # Sequential analysis
│   └── reflector.py # Reflection strategies
├── core/            # Orchestration
│   └── orchestrator.py # Agent coordination
├── datasets/        # Data processing
│   ├── base_dataset.py
│   └── beauty_dataset.py
├── models/          # LLM providers
│   └── llm_provider.py
├── system/          # Complete pipeline
│   └── pipeline.py
└── utils/           # Metrics & logging
    ├── metrics.py
    └── logging.py
```

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

The system provides comprehensive performance tracking:

- **Agent Performance**: Call counts, response times, success rates
- **Communication Logs**: Full agent conversation history  
- **Reflection Insights**: Quality assessments and improvements
- **Evaluation Metrics**: HR@K, NDCG@K, MRR across test splits

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

Based on cutting-edge research:

1. **MACRec** (Multi-Agent Collaborative Reasoning): Agent architecture and communication protocol
2. **LLM-Sequential-Recommendation**: Dataset processing and evaluation framework  
3. **ThinkRec**: Dual-objective training approach for reasoning

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