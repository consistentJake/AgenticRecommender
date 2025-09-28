# LLM Multi-Agent Recommender Flow

The diagram below captures the end-to-end control flow across the pipeline, orchestrator, and subordinate agents, as implemented in `agentic_recommender/system/pipeline.py` and `agentic_recommender/core/orchestrator.py`.

```mermaid
flowchart TD
    A[Pipeline entry point<br/>run_demo_prediction / run_evaluation] --> B[load_dataset]
    B --> C[Dataset processed<br/>sessions, item metadata]
    C --> D[orchestrator.update_agent_data]
    D --> E[Create RecommendationRequest]
    E --> F[AgentOrchestrator.recommend]
    F --> G[Init session state<br/>conversation history, counters]
    G --> H{Manager iteration
(max_iterations ≤ 5)}
    H -->|think()| I[Manager.think builds
context-driven prompt]
    I --> J[LLM reasoning output]
    J --> K[Log thought & append to scratchpad]
    K --> L[Manager.act builds
AVAILABLE ACTIONS prompt]
    L --> M[action_llm.generate]
    M --> N[Parse Action type]
    N --> O{Action?}
    O -->|Analyse[user/item]| P[Analyst.forward]
    P --> Q[Analyst tools read user/item history]
    Q --> R[LLM analysis result]
    R --> S[Update context.last_analysis
Log Analyst output]
    O -->|Search[query]| T[Mock search handler]
    T --> U[Return placeholder
search result]
    U --> V[Update context.last_search]
    O -->|Finish[result]| W[Set final_recommendation]
    S --> H
    V --> H
    H -->|Iterations exhausted
without Finish| X[Fallback to first candidate]
    W --> Y[RecommendationResponse.pack]
    X --> Y
    Y --> Z[Reflector.forward]
    Z --> AA{Reflection strategy}
    AA -->|NONE| AB[Skip reflection metadata]
    AA -->|REFLEXION / variants| AC[LLM reflection prompt
with scratchpad]
    AC --> AD[Parse JSON feedback → metadata]
    AB --> AE
    AD --> AE[Attach reflection metadata]
    AE --> AF[Logger records success]
    AF --> AG[Return RecommendationResponse]
    AG --> AH[Pipeline collects ranking /
metrics]
```

Key state transitions:
- `conversation_history` accrues every agent communication and feeds the reflector scratchpad.
- `accumulated_context` aggregates latest analyses/search outcomes for the manager.
- Reflection metadata is embedded in the response for downstream evaluation and learning.