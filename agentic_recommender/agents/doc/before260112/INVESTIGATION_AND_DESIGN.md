# Investigation and Design Process

## Document Purpose

This document captures the complete investigation process, findings, and redesigned agentic flow for the food delivery recommendation system.

---

## Part 1: Investigation Process

### 1.1 Task Requirements Analysis

**Source**: `/agentic_recommender/agents/task`

The task requested:
1. Review existing agents design
2. Examine the Singapore food delivery dataset
3. Adapt code to match dataset format (reference: `finetune_lora.py` chat templates)
4. Implement Swing similarity for finding top-5 similar users (with threshold)
5. Redesign Reflector with similar user evidence + LLM judgment
6. Create unified LLM module with Claude API priority
7. Document the design

---

### 1.2 Current Codebase Analysis

#### 1.2.1 Agent Architecture

**Files Examined**:
- `agentic_recommender/agents/base.py`
- `agentic_recommender/agents/manager.py`
- `agentic_recommender/agents/analyst.py`
- `agentic_recommender/agents/reflector.py`

**Key Findings**:

| Component | Location | Purpose |
|-----------|----------|---------|
| `Agent` | base.py:34 | Abstract base with invoke(), reset(), performance tracking |
| `ToolAgent` | base.py:136 | Agent with tool execution capability |
| `ReflectionStrategy` | base.py:232 | Enum: NONE, LAST_ATTEMPT, REFLEXION, LAST_ATTEMPT_AND_REFLEXION |
| `Manager` | manager.py:18 | Two-stage LLM (thought_llm + action_llm) |
| `Analyst` | analyst.py:16 | Tools: UserInfo, ItemInfo, UserHistory, ItemHistory |
| `Reflector` | reflector.py:17 | Self-reflection with multiple strategies |

**Current Manager Flow**:
```
think() → Generates reasoning using thought_llm (temp=0.8)
    ↓
act() → Generates structured action using action_llm (temp=0.3)
    ↓
_parse_action() → Extracts action type and arguments
    ↓
Available Actions: Analyse[user/item, id], Search[query], Finish[result]
```

**Current Reflector Flow**:
```
forward() → Routes to strategy method
    ↓
_reflect_with_llm() → Builds prompt, calls LLM, parses JSON response
    ↓
Output: {"correctness": bool, "reason": str, "improvement": str}
```

#### 1.2.2 LLM Provider Architecture

**File**: `agentic_recommender/models/llm_provider.py`

**Current Providers**:
| Provider | Class | Notes |
|----------|-------|-------|
| Gemini Direct | `GeminiProvider` | Uses `google-generativeai` SDK |
| OpenRouter | `GeminiProvider` | `use_openrouter=True` |
| Mock | `MockLLMProvider` | For testing |

**Configuration**: Loaded from `configs/config` JSON file

**Gap Identified**: No Claude API support (required by task)

---

### 1.3 Dataset Analysis

**File**: `datasets/sg/train.jsonl`

**Sample Record**:
```json
{
  "instruction": "User recent orders (oldest → newest):\n\n| idx | day | hour | cuisine     | price |\n|-----|-----|------|-------------|\n| 1   | Mon | 17   | pizza       |  0.12 |\n| 2   | Mon | 17   | pizza       |  0.06 |\n...\n\nCandidate product:\n- Buffalo Wings (cuisine: mexikanskt, price: $0.32)",
  "input": "",
  "output": "Yes",
  "system": "You are a food delivery recommendation assistant...",
  "history": []
}
```

**Data Characteristics**:

| Field | Type | Description |
|-------|------|-------------|
| instruction | string | Markdown table of orders + candidate |
| input | string | Always empty |
| output | string | "Yes" or "No" |
| system | string | System prompt |
| history | array | Always empty (future multi-turn support) |

**Order Fields Extracted**:
- `idx`: Order sequence number (1-10)
- `day`: Day of week (Mon-Sun)
- `hour`: Hour of day (0-23)
- `cuisine`: Food category (pizza, burgare, mexikanskt, etc.)
- `price`: Normalized price (0.0-1.0 scale)

**Candidate Fields**:
- `name`: Product name (e.g., "Buffalo Wings")
- `cuisine`: Category
- `price`: Normalized price

---

### 1.4 Finetune Pipeline Analysis

**Files Examined**:
- `finetune/scripts/finetune_lora.py`
- `finetune/scripts/utils.py`

**Chat Template Pattern** (utils.py:164-197):

```python
def to_chat_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """For training - includes assistant response."""
    messages = _build_messages_up_to_user(example)
    messages.append({"role": "assistant", "content": example.get("output", "")})
    return messages

def to_generation_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """For inference - no assistant response."""
    return _build_messages_up_to_user(example)
```

**Message Structure**:
```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": instruction + input},
    {"role": "assistant", "content": output}  # Only for training
]
```

**Tokenization** (utils.py:218-301):
- Uses `<|im_start|>assistant\n` as marker for label masking
- Prompt tokens masked with -100 (ignored in loss)
- Response tokens preserved for training

---

### 1.5 Swing Similarity Research

**Sources Consulted**:
- Alibaba Cloud Documentation
- Academic papers on collaborative filtering
- Web search for mathematical formulas

**Swing Algorithm Overview**:

Swing is a graph-based similarity algorithm developed by Alibaba that considers user pairs who have interacted with common items. Unlike traditional methods (Jaccard, Cosine), it:
- Penalizes users who interact with many items (reduces noise from power users)
- Considers the co-occurrence pattern between user pairs
- Provides anti-noise properties for sparse data

**Mathematical Formula**:

For item-to-item similarity:
```
sim(i, j) = Σ(u ∈ U(i) ∩ U(j)) Σ(v ∈ U(i) ∩ U(j), v≠u) w(u, v, i, j)

where:
w(u, v, i, j) = 1 / ((|I(u)| + α₁)^β × (|I(v)| + α₁)^β × (|I(u) ∩ I(v)| + α₂))
```

For user-to-user similarity (adapted):
```
sim(u1, u2) = Σ(i ∈ I(u1) ∩ I(u2)) w(u1, u2, i)

where:
w(u1, u2, i) = 1 / ((|I(u1)| + α₁)^β × (|I(u2)| + α₁)^β × (|U(i)| + α₂))
```

**Recommended Parameters** (from Alibaba production):
| Parameter | Value | Purpose |
|-----------|-------|---------|
| α₁ | 5.0 | Smoothing for user activity |
| α₂ | 1.0 | Smoothing for intersection term |
| β | 0.3 | Power weight for user activity |
| threshold | 0.1 | Minimum similarity to consider |
| top_k | 5 | Number of similar users to return |

---

## Part 2: Redesigned Agentic Flow

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC RECOMMENDATION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐   │
│  │   INPUT     │     │   DATA      │     │      UNIFIED LLM            │   │
│  │   LAYER     │────>│   ADAPTER   │     │      PROVIDER               │   │
│  └─────────────┘     └─────────────┘     │  ┌─────────────────────┐    │   │
│        │                   │             │  │ Claude (Priority 1) │    │   │
│        v                   v             │  ├─────────────────────┤    │   │
│  ┌─────────────────────────────────┐     │  │ Gemini (Priority 2) │    │   │
│  │         MANAGER AGENT           │     │  ├─────────────────────┤    │   │
│  │  ┌─────────┐    ┌─────────┐     │     │  │ OpenRouter (Pri. 3) │    │   │
│  │  │  THINK  │───>│   ACT   │     │<───>│  └─────────────────────┘    │   │
│  │  └─────────┘    └─────────┘     │     └─────────────────────────────┘   │
│  └─────────────────────────────────┘                                        │
│        │                   │                                                │
│        v                   v                                                │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │  ANALYST    │     │            ENHANCED REFLECTOR                    │   │
│  │  AGENT      │     │  ┌─────────────┐    ┌───────────────────────┐   │   │
│  │             │     │  │ FIRST ROUND │    │    SWING SIMILARITY   │   │   │
│  │  Tools:     │     │  │ LLM JUDGE   │───>│    USER RETRIEVAL     │   │   │
│  │  - UserInfo │     │  └─────────────┘    └───────────────────────┘   │   │
│  │  - History  │     │        │                      │                 │   │
│  │  - Sequence │     │        v                      v                 │   │
│  └─────────────┘     │  ┌─────────────────────────────────────────┐   │   │
│                      │  │         SECOND ROUND LLM JUDGE          │   │   │
│                      │  │  (Initial prediction + Similar users)   │   │   │
│                      │  └─────────────────────────────────────────┘   │   │
│                      │        │                                        │   │
│                      │        v                                        │   │
│                      │  ┌─────────────────────────────────────────┐   │   │
│                      │  │     FINAL DECISION + REASONING          │   │   │
│                      │  └─────────────────────────────────────────┘   │   │
│                      └─────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Sequence

```
┌──────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌─────────┐
│Client│  │ Manager │  │ Analyst │  │ Reflector│  │   Swing   │  │   LLM   │
└──┬───┘  └────┬────┘  └────┬────┘  └────┬─────┘  └─────┬─────┘  └────┬────┘
   │           │            │            │              │             │
   │ Request   │            │            │              │             │
   │──────────>│            │            │              │             │
   │           │            │            │              │             │
   │           │ think()    │            │              │             │
   │           │────────────────────────────────────────────────────->│
   │           │            │            │              │             │
   │           │<─────────────────────────────────────────────────────│
   │           │ thought    │            │              │             │
   │           │            │            │              │             │
   │           │ act()      │            │              │             │
   │           │────────────────────────────────────────────────────->│
   │           │            │            │              │             │
   │           │<─────────────────────────────────────────────────────│
   │           │ Analyse[user]           │              │             │
   │           │            │            │              │             │
   │           │ invoke()   │            │              │             │
   │           │───────────>│            │              │             │
   │           │            │ analyze    │              │             │
   │           │            │──────────────────────────────────────-->│
   │           │            │<────────────────────────────────────────│
   │           │<───────────│            │              │             │
   │           │ analysis   │            │              │             │
   │           │            │            │              │             │
   │           │ reflect()  │            │              │             │
   │           │───────────────────────->│              │             │
   │           │            │            │              │             │
   │           │            │            │ Stage 1: First Round       │
   │           │            │            │─────────────────────────-->│
   │           │            │            │<────────────────────────────│
   │           │            │            │ initial prediction         │
   │           │            │            │              │             │
   │           │            │            │ get_similar()│             │
   │           │            │            │─────────────>│             │
   │           │            │            │<─────────────│             │
   │           │            │            │ top-5 users  │             │
   │           │            │            │              │             │
   │           │            │            │ Stage 2: Second Round      │
   │           │            │            │─────────────────────────-->│
   │           │            │            │<────────────────────────────│
   │           │            │            │ final decision + reasoning │
   │           │            │            │              │             │
   │           │<────────────────────────│              │             │
   │           │ reflection result       │              │             │
   │           │            │            │              │             │
   │           │ Finish[recommendation]  │              │             │
   │           │            │            │              │             │
   │<──────────│            │            │              │             │
   │ Response  │            │            │              │             │
```

### 2.3 Detailed Component Design

#### 2.3.1 Data Adapter

**Purpose**: Parse JSONL format into structured types for agent processing

```python
@dataclass
class OrderRecord:
    idx: int
    day: str        # Mon, Tue, Wed, Thu, Fri, Sat, Sun
    hour: int       # 0-23
    cuisine: str    # pizza, burgare, mexikanskt, etc.
    price: float    # Normalized 0.0-1.0

@dataclass
class CandidateProduct:
    name: str
    cuisine: str
    price: float

@dataclass
class RecommendationRequest:
    user_id: str                    # Assigned during preprocessing
    orders: List[OrderRecord]       # Recent order history
    candidate: CandidateProduct     # Product to evaluate
    ground_truth: Optional[bool]    # For evaluation
```

**Parsing Logic**:
```python
def parse_instruction(instruction: str) -> Tuple[List[OrderRecord], CandidateProduct]:
    """
    Parse instruction text:
    1. Extract markdown table rows
    2. Parse each row into OrderRecord
    3. Extract candidate product line
    4. Return structured data
    """
```

#### 2.3.2 Swing Similarity Module

**Purpose**: Find top-k similar users for collaborative filtering evidence

```python
class SwingSimilarity:
    def __init__(self, config: SwingConfig):
        self.alpha1 = config.alpha1  # 5.0
        self.alpha2 = config.alpha2  # 1.0
        self.beta = config.beta      # 0.3
        self.threshold = config.similarity_threshold  # 0.1
        self.top_k = config.top_k    # 5

    def fit(self, interactions: List[Tuple[str, str]]):
        """Build user-item and item-user indices."""

    def compute_user_similarity(self, user1: str, user2: str) -> float:
        """
        Swing formula:
        sim(u1, u2) = Σ(i ∈ common_items) 1/((|I(u1)|+α1)^β × (|I(u2)|+α1)^β × (|U(i)|+α2))
        """

    def get_similar_users(self, user_id: str) -> List[Tuple[str, float]]:
        """Return top-k users above threshold, sorted by similarity."""
```

**Index Structure**:
```
user_items: Dict[user_id, Set[item_id]]
item_users: Dict[item_id, Set[user_id]]
similarity_cache: Dict[Tuple[user_id, user_id], float]
```

#### 2.3.3 Enhanced Reflector

**Purpose**: Two-stage LLM judgment with similar user evidence

**Stage 1: First Round Judgment**
```
Input:
  - User's order history (10 orders)
  - Candidate product

Process:
  - Build prompt with history table + candidate
  - LLM predicts: Yes/No + confidence + reasoning

Output:
  {
    "prediction": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Based on user's preference for..."
  }
```

**Stage 2: Similar User Retrieval**
```
Input:
  - Current user ID
  - Candidate product

Process:
  - Call SwingSimilarity.get_similar_users()
  - For each similar user above threshold:
    - Retrieve their order history
    - Check if they bought similar cuisine/product

Output:
  List[SimilarUserEvidence]:
    - user_id: str
    - similarity_score: float
    - recent_cuisines: Set[str]
    - bought_similar: bool
```

**Stage 3: Second Round Judgment**
```
Input:
  - User's order history
  - Candidate product
  - First round prediction + confidence + reasoning
  - Similar user evidence (up to 5 users)

Process:
  - Build prompt with all evidence
  - LLM reviews initial prediction considering peer behavior
  - LLM explains how similar users influenced decision

Output:
  {
    "prediction": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Revised prediction because similar users..."
  }
```

#### 2.3.4 Unified LLM Provider

**Purpose**: Multi-provider support with automatic fallback

**Provider Priority**:
| Priority | Provider | Model | Use Case |
|----------|----------|-------|----------|
| 1 | Claude API | claude-sonnet-4-20250514 | Primary reasoning |
| 2 | Gemini Direct | gemini-2.0-flash-exp | Fallback |
| 3 | OpenRouter | google/gemini-flash-1.5 | Secondary fallback |

**Interface**:
```python
class UnifiedLLMProvider:
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        preferred_provider: ProviderType = None,
        **kwargs
    ) -> str:
        """
        Generate text using best available provider.
        Falls back through priority chain on failure.
        """
```

**Configuration**:
```python
# configs/llm_config.json
{
  "providers": [
    {
      "type": "claude",
      "api_key": "${ANTHROPIC_API_KEY}",
      "model": "claude-sonnet-4-20250514",
      "priority": 1
    },
    {
      "type": "gemini",
      "api_key": "${GEMINI_API_KEY}",
      "model": "gemini-2.0-flash-exp",
      "priority": 2
    }
  ],
  "fallback_enabled": true,
  "retry_count": 2
}
```

### 2.4 Complete Recommendation Flow

```
1. INPUT PROCESSING
   │
   ├─ Parse JSONL record
   ├─ Extract order history (10 orders)
   ├─ Extract candidate product
   └─ Assign/lookup user_id

2. MANAGER: THINK PHASE
   │
   ├─ Analyze user order patterns
   ├─ Identify cuisine preferences
   ├─ Note price sensitivity
   └─ Plan next action

3. MANAGER: ACT PHASE
   │
   └─ Action: Analyse[user, user_id]

4. ANALYST: USER ANALYSIS
   │
   ├─ Execute UserInfo tool
   ├─ Execute UserHistory tool
   ├─ Generate preference summary
   └─ Return analysis to Manager

5. MANAGER: REFLECT DECISION
   │
   └─ Action: Reflect[analysis, candidate]

6. ENHANCED REFLECTOR: STAGE 1
   │
   ├─ Build first-round prompt
   ├─ Call LLM for initial judgment
   └─ Parse prediction + confidence

7. SWING SIMILARITY
   │
   ├─ Lookup user in similarity index
   ├─ Compute similarity scores
   ├─ Filter by threshold (>0.1)
   └─ Return top-5 similar users

8. ENHANCED REFLECTOR: STAGE 2
   │
   ├─ Gather similar user evidence
   │   ├─ Their recent cuisines
   │   └─ Their decisions for similar products
   │
   ├─ Build second-round prompt
   │   ├─ Include first-round analysis
   │   └─ Include similar user evidence
   │
   ├─ Call LLM for refined judgment
   └─ Parse final prediction + reasoning

9. MANAGER: FINISH
   │
   └─ Action: Finish[Yes/No + reasoning]

10. OUTPUT
    │
    └─ Return recommendation with explanation
```

### 2.5 Prompt Templates

#### First Round Prompt
```
You are a food delivery recommendation system. Analyze the user's order history
and predict whether they will purchase the candidate product.

## User's Recent Orders
| # | Day | Hour | Cuisine | Price |
|---|-----|------|---------|-------|
| 1 | Mon | 17   | pizza   | $0.12 |
| 2 | Mon | 17   | pizza   | $0.06 |
...

## Candidate Product
- Name: Buffalo Wings
- Cuisine: mexikanskt
- Price: $0.32

## Task
Predict whether the user is likely to purchase this product. Consider:
1. Cuisine preferences (do they order this type of food?)
2. Price sensitivity (is this within their typical spending?)
3. Time patterns (when do they typically order?)
4. Sequential behavior (what do they order after certain cuisines?)

Return JSON: {"prediction": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
```

#### Second Round Prompt
```
You are a food delivery recommendation system performing a REFINED prediction.

## User's Recent Orders
[Same as above]

## Candidate Product
[Same as above]

## First Round Analysis
- Initial Prediction: Yes
- Confidence: 65%
- Reasoning: User has ordered mexikanskt cuisine before...

## Evidence from Similar Users
**Similar User 1** (Similarity: 78%)
- Recent cuisines: pizza, mexikanskt, burgare
- Action for similar products: Bought similar

**Similar User 2** (Similarity: 65%)
- Recent cuisines: pizza, pizza, pizza
- Action for similar products: Did not buy similar

**Similar User 3** (Similarity: 52%)
- Recent cuisines: mexikanskt, indiskt
- Action for similar products: Bought similar

## Task
Review your initial prediction considering the behavior of similar users.
- If similar users consistently bought/rejected similar items, weigh that heavily
- If similar user behavior contradicts your initial prediction, reconsider
- Provide your final decision with reasoning

Return JSON: {"prediction": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
```

---

## Part 3: Implementation Checklist

### Phase 1: Data Adapter
- [ ] Create `agentic_recommender/data/food_delivery_adapter.py`
- [ ] Implement `parse_instruction()` function
- [ ] Implement `OrderRecord`, `CandidateProduct`, `RecommendationRequest` dataclasses
- [ ] Create user ID assignment logic (hash of order pattern or sequential)
- [ ] Write unit tests

### Phase 2: Swing Similarity
- [ ] Create `agentic_recommender/similarity/swing.py`
- [ ] Implement `SwingConfig` dataclass
- [ ] Implement `SwingSimilarity.fit()` method
- [ ] Implement `SwingSimilarity.compute_user_similarity()` method
- [ ] Implement `SwingSimilarity.get_similar_users()` method
- [ ] Add caching layer for similarity scores
- [ ] Write unit tests with sample interactions

### Phase 3: Enhanced Reflector
- [ ] Create `agentic_recommender/agents/enhanced_reflector.py`
- [ ] Implement `SimilarUserEvidence` dataclass
- [ ] Implement `ReflectionResult` dataclass
- [ ] Implement `_first_round_judgment()` method
- [ ] Implement `_get_similar_user_evidence()` method
- [ ] Implement `_second_round_judgment()` method
- [ ] Implement prompt builders
- [ ] Write unit tests

### Phase 4: Unified LLM Provider
- [ ] Create `agentic_recommender/models/unified_llm_provider.py`
- [ ] Implement `ProviderType` enum
- [ ] Implement `ProviderConfig` dataclass
- [ ] Implement `UnifiedLLMConfig` dataclass
- [ ] Add Claude API provider (`pip install anthropic`)
- [ ] Implement fallback chain logic
- [ ] Write unit tests with mock providers

### Phase 5: Integration
- [ ] Update Manager agent to use new Reflector
- [ ] Update configuration loading for new LLM provider
- [ ] Create end-to-end test with sample data
- [ ] Add performance benchmarking
- [ ] Update documentation

---

## Part 4: Open Questions for Review

Before implementation, please confirm:

1. **User ID Strategy**: How should we assign user IDs from the dataset?
   - Option A: Hash of first N orders (deterministic)
   - Option B: Sequential assignment during preprocessing
   - Option C: Use additional metadata if available

2. **Similarity Pre-computation**: Should we pre-compute all user similarities?
   - Option A: Full pre-computation (faster runtime, more memory)
   - Option B: On-demand with LRU cache (slower first query, less memory)

3. **LLM Provider Priority**: Confirm Claude API is available?
   - Need `ANTHROPIC_API_KEY` environment variable
   - Fallback to Gemini if Claude unavailable?

4. **Threshold Tuning**: Should thresholds be configurable?
   - Similarity threshold: 0.1 (default)
   - Confidence threshold for overriding: 0.8?
   - Minimum similar users required: 1?

5. **Evaluation Metrics**: Which metrics should we track?
   - Accuracy (overall)
   - Precision/Recall/F1 for "Yes" class
   - Agreement rate between first and second round
   - Similar user hit rate
