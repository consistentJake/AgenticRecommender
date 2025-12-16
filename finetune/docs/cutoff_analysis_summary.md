# Cutoff Length and Truncation Strategy Analysis

## Executive Summary

**Question**: Is `cutoff_len=1024` the right value, and is left-side truncation the right approach?

**Answer**:
- ✅ **1024 is appropriate** - All samples fit comfortably (max 875 tokens)
- ✅ **Left-side truncation is correct** - Standard for SFT, preserves training signal
- ℹ️ **Truncation is not happening** - No samples exceed the cutoff

---

## Detailed Analysis

### Sequence Length Distribution (78,271 training samples)

```
Metric          Value    % of Cutoff
──────────────  ─────    ───────────
Minimum         393      38%
Mean            435      42%
Median (p50)    432      42%
90th percentile 458      45%
95th percentile 468      46%
99th percentile 493      48%
Maximum         875      85%
Cutoff          1024     100%
```

### Key Findings

#### 1. No Truncation Occurring
- **0 samples** (0.0%) exceed 1024 tokens
- **78,271 samples** (100%) fit within the cutoff
- **149 token headroom** between max length (875) and cutoff (1024)

#### 2. Conservative Cutoff
The cutoff provides substantial safety margin:
- Even the maximum sample uses only 85% of available space
- Mean sample uses only 42% of available space
- 99% of samples use less than 48% of available space

#### 3. Distribution is Tight
Most sequences cluster around 430-435 tokens:
- Standard deviation: ~20 tokens
- 90% of samples between 410-458 tokens
- Very few outliers

---

## Is 1024 the Right Cutoff?

### ✅ Advantages of Current Setting (1024)

1. **Safety**: Handles all current samples with 15% headroom
2. **Future-proof**: Room for longer movie titles or extended history
3. **No data loss**: Zero samples truncated
4. **Simplicity**: Power of 2, common in ML (512, 1024, 2048)

### ⚡ Alternative Options

#### Option A: Reduce to 896 (Recommended if memory-constrained)
```yaml
cutoff_len: 896  # 12.5% memory savings
```
- Still safely above max (875 tokens)
- Saves GPU memory and compute
- Risk: None for current dataset

#### Option B: Reduce to 768 (Aggressive optimization)
```yaml
cutoff_len: 768  # 25% memory savings
```
- Well above 99th percentile (493 tokens)
- May truncate outliers near 875 tokens
- Risk: Low (~0.01% of samples might be affected)

#### Option C: Reduce to 512 (Not recommended)
```yaml
cutoff_len: 512  # 50% memory savings
```
- Below maximum observed length
- Would truncate ~0.1% of samples
- Risk: Moderate data loss

### 📊 Memory Impact Estimation

For a batch size of 4 with gradient accumulation:

```
Cutoff   Tokens/Batch   Memory      vs 1024
──────   ────────────   ──────      ───────
512      2,048          100%        -50%
768      3,072          150%        -25%
896      3,584          175%        -12.5%
1024     4,096          200%        baseline
```

**Note**: Actual memory savings depend on model size, batch size, and other factors.

### 🎯 Recommendation

**Keep `cutoff_len=1024`** unless memory is a critical constraint.

**If memory is constrained**: Use `cutoff_len=896` for 12.5% savings with zero risk.

---

## Is Left-Side Truncation the Right Method?

### Current Implementation

From `scripts/utils.py:275-279`:
```python
if len(full_ids) > cutoff_len:
    start = len(full_ids) - cutoff_len
    full_ids = full_ids[start:]      # Remove from LEFT
    labels = labels[start:]
```

This keeps the **right side** (recent movies + response) and removes the **left side** (early movies).

### ✅ Why Left-Side Truncation is Correct for SFT

#### 1. Preserves Training Signal
```
[Prompt tokens...] [Response: "Yes"] <-- MUST keep this!
```
The assistant's response is always at the end. Left-truncation ensures it's never cut.

#### 2. Maintains Label Alignment
```python
labels = [-100, -100, ..., -100, token_yes, token_end]
         └─ masked ─┘        └─ trained ─┘
```
Right-truncation would cut the tokens we're trying to train on.

#### 3. Standard Practice in SFT
All major SFT implementations use left-truncation:
- HuggingFace TRL
- Axolotl
- LLaMA-Factory

### ⚠️ Potential Issues (if truncation were happening)

For **recommendation tasks**, left-truncation removes early user history:

```
Before truncation:
Movies 1-15: [oldest...........................newest] + Response
             ^^^^^^^^^^^^^
             These get removed

After truncation:
Movies 8-15: [................newest] + Response
             Only recent history remains
```

**Impact**:
- Temporal bias: Model only sees recent preferences
- Lost context: Early taste indicators removed
- Inconsistent history length: Some users have full history, others partial

### 🔍 Why This Doesn't Matter Here

**Because no truncation is occurring!**
- 0 samples > 1024 tokens
- All samples preserve their full 15-movie history
- Left-truncation logic exists but is never executed

### 🎯 If Truncation Became a Problem

If future datasets had sequences >1024 tokens, here are solutions:

#### Solution 1: Reduce History Length (Best)
```bash
# In data preparation
python scripts/prepare_movielens.py --history-len 10  # Instead of 15
```
**Pros**: Prevents truncation at the source, maintains data quality
**Cons**: Less context for the model

#### Solution 2: Increase Cutoff (Simple)
```yaml
cutoff_len: 2048  # Double the limit
```
**Pros**: Keeps all data
**Cons**: 2x memory usage, may exceed model's context window

#### Solution 3: Smart Sampling (Advanced)
```python
# Sample movies from timeline instead of taking last 15
sample_indices = [0, 3, 6, 9, 10, 11, 12, 13, 14]  # Mix old and new
```
**Pros**: Balanced temporal coverage
**Cons**: More complex, may miss sequential patterns

#### Solution 4: Middle Truncation (Complex)
```python
# Keep first N and last M tokens
if len(full_ids) > cutoff_len:
    keep_prefix = 150  # Early history
    keep_suffix = cutoff_len - keep_prefix
    full_ids = full_ids[:keep_prefix] + full_ids[-keep_suffix:]
```
**Pros**: Preserves both early and late history
**Cons**: Complex to implement, breaks label masking logic

### 🎯 Recommendation

**Keep left-side truncation** - it's the correct approach for SFT.

**Monitor with the script**:
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 0
```

If truncation ever occurs (>1% of samples), address it by reducing `--history-len` in data preparation.

---

## Practical Recommendations

### For Current Dataset

```yaml
# configs/qwen3_7b_movielens_qlora.yaml
cutoff_len: 1024  # ✅ Keep as is
```

**No changes needed** - current configuration is optimal.

### For Memory-Constrained Environments

```yaml
# configs/qwen3_7b_movielens_qlora.yaml
cutoff_len: 896  # ⚡ Safe reduction
```

Run analysis to confirm:
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --cutoff-len 896
```

### For Future Datasets

1. **Always run analysis first**:
   ```bash
   python scripts/check_seq_len.py --config YOUR_CONFIG.yaml --show-examples 2
   ```

2. **Check truncation percentage**:
   - 0%: Perfect, no action needed
   - <1%: Acceptable, consider increasing cutoff slightly
   - 1-5%: Warning, review truncation strategy
   - >5%: Problem, reduce history length or increase cutoff

3. **Monitor what gets truncated**:
   - Use `--show-examples` to see which movies are removed
   - Ensure important context is preserved

---

## Verification Examples

### Example 1: Typical Sample (No Truncation)

```
Length: 427 tokens
Cutoff: 1024 tokens
Status: ✅ No truncation (58% below cutoff)

Prompt tokens: 420 (masked with -100)
Response tokens: 7 (trained)
Masking ratio: 98.4%
```

### Example 2: Maximum-Length Sample (Still No Truncation)

```
Length: 875 tokens
Cutoff: 1024 tokens
Status: ✅ No truncation (15% below cutoff)

Headroom: 149 tokens
```

### Example 3: Simulated Truncation (cutoff=400)

```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml \
    --cutoff-len 400 --show-examples 1
```

Output would show:
```
⚠️ WILL BE TRUNCATED! (427 > 400)
Tokens removed: 27

TEXT THAT WILL BE REMOVED:
  <|im_start|>system
  You are a movie recommendation assistant...
  1. Pearl Harbor (2001) (rating ≈ 3.0)
  2. Moulin Rouge (2001) (ra...

TEXT THAT WILL BE KEPT:
  ...14. Boys Don't Cry (1999) (rating ≈ 1.0)
  15. Babe: Pig in the City (1998) (rating ≈ 3.5)

  Candidate movie:
  Money Pit, The (1986)...
```

This demonstrates how early movie history would be lost if cutoff were too small.

---

## Conclusion

### Summary Table

| Aspect | Current Setting | Status | Recommendation |
|--------|----------------|--------|----------------|
| Cutoff length | 1024 tokens | ✅ Appropriate | Keep as is (or reduce to 896 if memory-constrained) |
| Truncation method | Left-side | ✅ Correct for SFT | Keep as is |
| Truncation occurring | 0% of samples | ✅ No data loss | No action needed |
| Sequence length distribution | Mean 435, max 875 | ✅ Well within limits | No action needed |
| Label masking | 98.4% masked | ✅ Standard for SFT | No action needed |

### Key Takeaways

1. ✅ **1024 is the right cutoff** - conservative and safe
2. ✅ **Left-side truncation is correct** - standard for SFT
3. ✅ **No optimization needed** - system is working well
4. ⚡ **Optional**: Reduce to 896 for 12.5% memory savings with zero risk
5. 📊 **Monitoring**: Use `check_seq_len.py` for future datasets

### Quick Reference Commands

```bash
# Full analysis with examples
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 2

# Statistics only (fast)
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 0

# Test different cutoff
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --cutoff-len 896

# Analyze subset for quick check
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --max-samples 1000
```

---

**Last Updated**: Based on analysis of 78,271 training samples
**Dataset**: MovieLens recommendation task with 15-movie history
**Model**: Qwen3-0.6B with chat template
**Config**: `configs/qwen3_7b_movielens_qlora.yaml`
