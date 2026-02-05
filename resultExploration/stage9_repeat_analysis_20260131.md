# Stage 9 Repeat Evaluation Analysis — data_sg (202601311905)

**Run:** `outputs/data_sg/202601311905/stage9_repeat_results.json`
**Samples:** 200 total, 190 valid, 10 errors

## Current Performance

| Metric | Round 1 (cuisine) | LightGCN (cuisine) | Final (vendor) |
|--------|-------------------|---------------------|----------------|
| Hit@1 | 0.537 | 0.253 | 0.479 |
| Hit@3 | **0.816** | 0.632 | 0.558 |
| Hit@5 | 0.816 | 0.795 | 0.563 |
| MRR | 0.661 | 0.472 | 0.521 |

## Key Finding: The LLM Underperforms a Simple Frequency Baseline

A naive "top-3 most frequent cuisines in history" baseline achieves **92.6%** cuisine Hit@3, vs the LLM's **81.6%**. The LLM hurts in 28 cases but only helps in 7.

Examples of LLM failures:

- **Sample 42:** GT=`coffee` appears **13/33 orders (39%)** — the #1 most ordered cuisine — but LLM predicts `chinese, singaporean, noodles`
- **Sample 140:** GT=`dim sum` appears **7/26 orders (27%)** — also #1 — but LLM predicts `roti prata, western`
- **Sample 50:** GT=`bubble tea` appears **5/29 orders (17%)** — #1 — but LLM predicts `western, fried chicken, ayam penyet`

The LLM over-weights CF scores and temporal "reasoning" at the expense of obvious frequency dominance in the history.

## Root Causes of R1 Misses (35 samples)

| Factor | R1 Hit (n=155) | R1 Miss (n=35) |
|--------|---------------|----------------|
| GT cuisine frequency in history | 6.3 avg | 3.5 avg |
| GT cuisine % of history | 41.0% | 16.1% |
| GT cuisine frequency rank | 1.6 | 2.9 |
| Cuisine recency (0=most recent) | 1.8 orders ago | 6.9 orders ago |
| Unique cuisines in history | 5.9 | 10.1 |

The LLM struggles with: **lower-frequency cuisines, older orders, and diverse histories**.

## Recency Impact on R1 Cuisine Hit@3

| GT cuisine recency | Samples | R1 Hit@3 | Final Hit@3 |
|-------------------|---------|----------|-------------|
| Most recent (0-2 orders ago) | 131 | **0.908** | 0.695 |
| Medium (3-5 orders ago) | 35 | 0.743 | 0.457 |
| Remote (6+ orders ago) | 24 | **0.417** | 0.292 |

## Pipeline Loss Breakdown

Even when R1 gets the cuisine right (155 samples):

- GT vendor found in candidate list: **72.9%** (113/155) — 27.1% lost at retrieval
- Among those found, R2 final Hit@3: **93.8%** — R2 ranking is actually good

The two bottlenecks are: **(1) R1 cuisine prediction, (2) candidate retrieval.**

## LightGCN Complementarity

Of 35 R1 misses:

- LightGCN has GT in top 3: **10** (28.6%)
- LightGCN has GT in top 5: **17** (48.6%)
- LightGCN also misses: **7** (20.0%)

7 hardest cases where both R1 and LightGCN miss involve cuisines like `coffee`, `hokkien mee`, `noodles`, `chicken rice`, `dim sum` — all present in history but at low frequency relative to the dominant cuisines.

## Quantified Improvement Opportunities

| Strategy | Projected cuisine Hit@3 | Gain |
|----------|------------------------|------|
| Current R1 | 0.816 | — |
| Use frequency-only baseline instead | **0.926** | +0.110 |
| Merge R1 + LightGCN top-3 | 0.868 | +0.052 |
| Merge R1 + LightGCN top-5 | 0.905 | +0.089 |
| Expand R1 to predict 5 cuisines | > 0.816 (currently capped) | TBD |

## Actionable Recommendations to Improve R1 Hit@3

1. **Anchor the LLM on frequency** — The prompt currently lists history as raw orders. Add explicit cuisine frequency counts/rankings so the LLM can't miss that `coffee` is 13/33 orders. Inject a line like `Top cuisines by frequency: 1. coffee (13), 2. chinese (5), ...`

2. **Ensemble R1 with frequency baseline** — Post-process: if the LLM's top-3 doesn't include any of the user's top-2 most frequent cuisines, force-include them. This alone could bridge most of the 11% gap.

3. **Fuse LightGCN at the cuisine level** — Currently LightGCN scores are provided as text in the prompt but the LLM ignores them for many cases. Instead, take the union of R1 top-3 and LightGCN top-3 cuisines (projected: 86.8%).

4. **Predict 5 cuisines instead of 3** — 19 samples returned only 2 predictions. Expanding to 5 would capture more long-tail cuisines, especially for diverse users (10+ unique cuisines).

5. **Fix the candidate retrieval gap** — Even with perfect R1, 27.1% of GT vendors are missing from the candidate list. This is the second-largest bottleneck after R1 accuracy.

## R1 Hit Rate by GT Cuisine Frequency Rank

| GT cuisine freq rank in history | Samples | R1 cuisine Hit@3 |
|--------------------------------|---------|-------------------|
| Top-1 (most frequent) | 105 | 0.924 |
| Top-2 | 51 | 0.804 |
| Top-3 | 20 | 0.500 |
| Rank 4-5 | 7 | 0.571 |
| Rank 6+ | 7 | 0.429 |

## Temporal Patterns

Hit rate is relatively stable across days of the week (0.78–0.89). By hour, late afternoon / evening (16:00–22:00) shows slightly lower hit rates (0.67–0.80) compared to morning/lunch (0.80–1.00), though sample sizes per hour are small.

## LLM vs Frequency Baseline: Case-by-Case

- **LLM wins (7 cases):** LLM correctly predicts low-frequency cuisines (rank 4–9) that frequency baseline misses, leveraging CF scores or temporal signals.
- **LLM loses (28 cases):** LLM ignores high-frequency cuisines (often rank 1–3) in favor of CF-boosted or temporally-inferred cuisines that turn out to be wrong.
- **Both agree (155 cases):** Mostly straightforward cases where the dominant cuisine is obvious.

**Net: the LLM's "reasoning" hurts more than it helps for R1 cuisine prediction in the repeat evaluation setting.**
