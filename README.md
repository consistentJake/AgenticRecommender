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

## 📝 Session Notes: LoRA Finetuning Data Flow

Context from `finetune/scripts/finetune_lora.py` on how supervised JSON samples are fed to Qwen during LoRA finetuning:
- Each JSON record provides `system`, `history`, `instruction`, `input`, and `output`; the script turns this into chat turns via `to_chat_messages`.
- The user turn is `instruction` plus `input` (if present); the assistant turn is exactly `output`. Optional `system` and any `history` are prepended.
- `tokenizer.apply_chat_template` renders those turns into Qwen chat text (e.g., `<|im_start|>system ... <|im_end|>`, `<|im_start|>user ...`, `<|im_start|>assistant\nNo<|im_end|>`), no generation prompt added.
- `preprocess_datasets_parallel` appends EOS, tokenizes, and masks all prompt tokens with `-100`, keeping labels only on the assistant response tokens.
- SFTTrainer then trains on these token/label pairs; metrics extract Yes/No from decoded assistant responses to compute accuracy/F1.


## Jan 12 Reform: Modular Recommendation System

This reform introduces a test-first approach with modular components for similarity calculation, enriched data representations, and Top-K evaluation.

**Full design documentation**: `agentic_recommender/agents/doc/ENHANCED_DESIGN_V2.md`

### New Module Structure

```
agentic_recommender/
├── data/                       # Data loading and representations
│   ├── enriched_loader.py      # Singapore dataset loader (preserves full schema)
│   └── representations.py      # EnrichedUser, CuisineProfile, CuisineRegistry
├── similarity/                 # Modular similarity calculation
│   ├── base.py                 # Abstract SimilarityMethod interface
│   ├── methods.py              # Swing, Cosine, Jaccard implementations
│   └── factory.py              # SimilarityFactory for easy method switching
├── evaluation/                 # Top-K evaluation framework
│   └── topk.py                 # TopKMetrics, SequentialRecommendationEvaluator
├── models/
│   └── llm_provider.py         # Added OpenRouterProvider (Gemini 2.5 Flash)
└── tests/                      # Comprehensive test suite (94 tests)
    ├── conftest.py             # Pytest fixtures with real Singapore data
    ├── test_data_loader.py     # Data loader and representation tests
    ├── test_similarity.py      # Similarity method tests
    ├── test_llm_provider.py    # LLM provider tests (including API tests)
    └── test_topk_evaluation.py # Top-K evaluation tests
```

### New Files Explained

#### Data Module (`agentic_recommender/data/`)

| File | Purpose |
|------|---------|
| `enriched_loader.py` | `EnrichedDataLoader` - Loads Singapore dataset preserving all fields (customer_id, order_id, geohash, chain_id, vendor_id). Parses timestamps into hour/day_num. |
| `representations.py` | `EnrichedUser` - User representation with cuisine preferences, temporal patterns (peak hours/weekdays), vendor loyalty, basket size. `CuisineProfile` - Cuisine profiles with peak ordering hours/days, meal time distribution. `CuisineRegistry` - Fast lookup for cuisine temporal patterns. |

#### Similarity Module (`agentic_recommender/similarity/`)

| File | Purpose |
|------|---------|
| `base.py` | Abstract `SimilarityMethod` class defining interface: `fit()`, `compute_similarity()`, `get_similar()`. Includes caching and configuration. |
| `methods.py` | Three similarity implementations: **SwingMethod** (Alibaba's anti-noise CF algorithm), **CosineMethod** (vector-based similarity), **JaccardMethod** (set overlap similarity). |
| `factory.py` | `SimilarityFactory` - Factory pattern for creating/switching similarity methods at runtime. Supports `factory.create("swing")`, `factory.set_default("cosine")`. |

#### Evaluation Module (`agentic_recommender/evaluation/`)

| File | Purpose |
|------|---------|
| `topk.py` | `TopKMetrics` - Dataclass for Hit@1/3/5/10, MRR, NDCG metrics. `SequentialRecommendationEvaluator` - Evaluates LLM predictions against ground truth. `TopKTestDataBuilder` - Creates test samples from order history (last order = ground truth). |

#### LLM Provider (`agentic_recommender/models/llm_provider.py`)

| Addition | Purpose |
|----------|---------|
| `OpenRouterProvider` | LLM provider using OpenRouter API with Gemini 2.5 Flash (`google/gemini-2.0-flash-001`). Supports JSON mode, batch generation, and metrics tracking. |

### Test Suite (`agentic_recommender/tests/`)

**94 tests** covering all new modules with real Singapore data:

| File | Tests | Coverage |
|------|-------|----------|
| `conftest.py` | - | Pytest fixtures: `all_orders`, `all_vendors`, `all_products`, `sample_customer_orders`, `merged_orders` |
| `test_data_loader.py` | 29 | EnrichedDataLoader (12), EnrichedUser (7), CuisineProfile (5), CuisineRegistry (5) |
| `test_similarity.py` | 25 | SwingMethod (8), CosineMethod (4), JaccardMethod (4), SimilarityFactory (9) |
| `test_llm_provider.py` | 17 | MockLLMProvider (5), OpenRouterProvider (5), API tests (5), Factory (2) |
| `test_topk_evaluation.py` | 20 | TopKMetrics (3), EvaluationResult (1), TopKTestDataBuilder (5), Evaluator (8), MetricsComputation (3) |

### Running Tests

```bash
# Run all 94 tests
python -m pytest agentic_recommender/tests/ -v

# Run specific test modules
python -m pytest agentic_recommender/tests/test_data_loader.py -v
python -m pytest agentic_recommender/tests/test_similarity.py -v
python -m pytest agentic_recommender/tests/test_llm_provider.py -v
python -m pytest agentic_recommender/tests/test_topk_evaluation.py -v

# Run with coverage
python -m pytest agentic_recommender/tests/ -v --cov=agentic_recommender
```

### Quick Validation Commands

```bash
# Test data loading
python -c "
from agentic_recommender.data import EnrichedDataLoader
loader = EnrichedDataLoader('/Users/zhenkai/Downloads/data_sg')
print(loader.get_stats())
"

# Test similarity methods
python -c "
from agentic_recommender.similarity import SimilarityFactory
print('Available methods:', SimilarityFactory.available_methods())
"

# Test LLM provider (real API call)
python -c "
from agentic_recommender.models.llm_provider import OpenRouterProvider
llm = OpenRouterProvider(api_key='your-api-key')
print(llm.generate('What is 2+2?'))
"

# Run full Top-K evaluation
python agentic_recommender/tests/test_topk_evaluation.py
```

### Key Design Decisions

1. **Top-K Hit Ratio** instead of Yes/No classification - More meaningful for sequential recommendation
2. **Modular Similarity** - Easy to switch between Swing/Cosine/Jaccard via factory pattern
3. **Enriched Representations** - Preserve all original data fields; add temporal patterns (peak hours/days)
4. **Test-First Approach** - 94 tests using real Singapore data fixtures
5. **OpenRouter + Gemini 2.5 Flash** - Fast and cost-effective for testing


## Enhanced Two-Round LLM Reranking System (Stage 8)

A cuisine-level recommendation pipeline with two-round LLM reranking and LightGCN collaborative filtering.

### Pipeline Flow

```
Data Loading → Cuisine-Cuisine Swing Candidates (top 20)
    → Round 1: LLM Reranking → LightGCN User-Cuisine Similarity
    → Round 2: Reflection (final reranking) → Metrics
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `CuisineSwingMethod` | `similarity/methods.py` | Cuisine-to-cuisine swing similarity for candidate generation |
| `LightGCNEmbeddingManager` | `similarity/lightGCN.py` | User-cuisine embeddings with disk caching |
| `CuisineBasedCandidateGenerator` | `evaluation/rerank_eval.py` | Generates 20 candidates from user's cuisine history |
| `EnhancedRerankEvaluator` | `evaluation/rerank_eval.py` | Two-round LLM evaluation with LightGCN reflection |

### Metrics

- **NDCG@K** (K=5, 10)
- **MRR@K** (K=5, 10)
- **Hit Rate@K** (K=1, 3, 5, 10)
- Round 1 vs Final comparison

### Configuration

Edit `workflow_config_linux.yaml`:

```yaml
run_enhanced_rerank_evaluation:
  enabled: true
  settings:
    n_candidates: 20
    items_per_seed: 5
    dataset_name: "data_se"
    lightgcn_epochs: 50
    lightgcn_embedding_dim: 64
    temperature_round1: 0.3
    temperature_round2: 0.2
    n_samples: 10
    min_history: 5
```

### Run Command

```bash
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages run_enhanced_rerank_evaluation
```

### LightGCN Cache

Embeddings are cached at `~/.cache/agentic_recommender/lightgcn/{dataset_name}_embeddings.pkl` for fast reloading.

---

## 🔍 Validation & Testing

### Workflow Stage Verification

The workflow runner has multiple stages that can be tested independently. Each stage produces output files that can be verified.

#### Stage Summary

| Stage | Name | Output Files | Description |
|-------|------|--------------|-------------|
| 1 | `load_data` | `stage1_merged_data.parquet`, `stage1_stats.json` | Load and merge raw data |
| 2 | `build_users` | `stage2_enriched_users.json`, `stage2_users_summary.json` | Build EnrichedUser representations |
| 3 | `build_cuisines` | `stage3_cuisine_profiles.json` | Build CuisineProfile for temporal patterns |
| 8 | `run_enhanced_rerank_evaluation` | `stage8_enhanced_rerank_results.json`, `stage8_enhanced_rerank_detailed.json` | Two-round LLM reranking with LightGCN |

#### Running Individual Stages

```bash
# Stage 1: Data Loading
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages load_data

# Stage 2: Build Users (requires Stage 1)
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages load_data build_users

# Stage 3: Build Cuisines (requires Stage 1-2)
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages load_data build_users build_cuisines

# Stage 8: Enhanced Rerank Evaluation
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages run_enhanced_rerank_evaluation
```

#### Full Pipeline Verification Script

```python
import json
import pandas as pd
import os

def verify_all_stages():
    errors = []

    # Stage 1
    try:
        df = pd.read_parquet('outputs/stage1_merged_data.parquet')
        assert len(df) > 0 and 'customer_id' in df.columns
        print(f"✓ Stage 1: {len(df)} rows")
    except Exception as e:
        errors.append(f"Stage 1: {e}")

    # Stage 2
    try:
        with open('outputs/stage2_users_summary.json') as f:
            summary = json.load(f)
        assert summary['total_users'] > 0
        print(f"✓ Stage 2: {summary['total_users']} users")
    except Exception as e:
        errors.append(f"Stage 2: {e}")

    # Stage 3
    try:
        with open('outputs/stage3_cuisine_profiles.json') as f:
            cuisines = json.load(f)
        assert len(cuisines) > 0
        print(f"✓ Stage 3: {len(cuisines)} cuisines")
    except Exception as e:
        errors.append(f"Stage 3: {e}")

    # Stage 8
    try:
        with open('outputs/stage8_enhanced_rerank_results.json') as f:
            results = json.load(f)
        print(f"✓ Stage 8: NDCG@5={results['ndcg@5']:.4f}, Hit@5={results['hit@5']:.4f}")
    except Exception as e:
        errors.append(f"Stage 8: {e}")

    return len(errors) == 0

verify_all_stages()
```

---

### LightGCN Embedding Validation

A dedicated script validates LightGCN embeddings by checking if a user's top-ranked cuisines match their purchase history.

#### Script Location

```
scripts/evaluate_lightgcn_embeddings.py
```

#### Usage

```bash
python scripts/evaluate_lightgcn_embeddings.py
```

#### What It Does

1. **Loads cached embeddings** from `~/.cache/agentic_recommender/lightgcn/`
2. **Loads user purchase history** from `outputs/stage1_merged_data.parquet`
3. **For each sample user**:
   - Shows their actual cuisine purchase history with counts
   - Calculates LightGCN similarity scores for all cuisines
   - Ranks cuisines by score and shows top-K
   - Computes Recall@K and Precision@K

#### Sample Output

```
======================================================================
EVALUATING USER: cd4e9f9533
======================================================================

📋 PURCHASE HISTORY:
   Total orders: 20
   Unique cuisines: 10
   Cuisine breakdown:
      mexikanskt: 5 orders (25.0%)
      turkiskt: 3 orders (15.0%)
      ...

🔮 LIGHTGCN TOP-10 PREDICTIONS:
    1. turkiskt: 4.5996 ✓ (purchased)
    2. pizza: 4.2781 ✓ (purchased)
    3. kebab: 4.1570 ✓ (purchased)
    ...

📊 EVALUATION METRICS (Top-10):
   Recall@10: 100.00% (10/10 purchased cuisines in top-10)
   Precision@10: 100.00% (10/10 of top-10 are purchased)
```

#### Expected Results for Good Embeddings

| Metric | Expected | Warning Sign |
|--------|----------|--------------|
| Recall@10 | > 70% | < 30% indicates model not learning |
| Mean Rank | < 5 | > 10 indicates weak personalization |
| Precision@10 | 30-60% | 100% may indicate overfitting |

#### Key Metrics Explained

- **Recall@K**: Fraction of purchased cuisines appearing in top-K predictions
- **Precision@K**: Fraction of top-K predictions that are actually purchased
- **Mean Rank**: Average position of purchased cuisines in the ranking

Note: Some new cuisines in top-K is expected and desired - these are recommendations!

---

### LightGCN Cache Verification

```bash
# Check cache exists
ls -la ~/.cache/agentic_recommender/lightgcn/

# Verify cache structure
python3 << 'EOF'
import pickle
import os

cache_path = os.path.expanduser('~/.cache/agentic_recommender/lightgcn/data_se_embeddings.pkl')
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

print(f"Users: {cache['user_embeddings'].shape}")
print(f"Cuisines: {cache['cuisine_embeddings'].shape}")
print(f"Cache key: {cache['cache_key']}")
EOF

# Test cache loading (should say "Cache key matches, loading embeddings")
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages run_enhanced_rerank_evaluation 2>&1 | grep -E "(Training|Cache)"
```