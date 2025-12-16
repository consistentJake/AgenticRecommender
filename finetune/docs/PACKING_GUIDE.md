# Packing Mode Guide

This guide explains how to use sequence packing in SFTTrainer to improve GPU utilization and training efficiency.

## Understanding Packing

**What is packing?**
Packing concatenates multiple short training examples into a single sequence (up to `cutoff_len`), reducing wasted GPU computation on padding tokens.

**Example:**
```
Without packing (3 sequences, cutoff_len=1024):
Seq 1: [300 tokens] + [724 padding tokens] = 1024 tokens
Seq 2: [450 tokens] + [574 padding tokens] = 1024 tokens
Seq 3: [280 tokens] + [744 padding tokens] = 1024 tokens
Total: 3072 tokens (only 1030 useful, 2042 wasted on padding = 66% waste)

With packing (1 packed sequence):
Packed: [300 + 450 + 280 tokens] + [remaining padding] = 1024 tokens
Total: 1024 tokens (1030 useful packed across batches, minimal padding waste)
```

## Current Tokenization Behavior

Your tokenization does **NOT** pre-pad sequences to `cutoff_len`:
- Sequences **longer** than `cutoff_len` → truncated to exactly `cutoff_len`
- Sequences **shorter** than `cutoff_len` → kept at original length (NOT padded)

This means **packing CAN work** with your data because sequences are variable length.

## Two Available Modes

### Mode 1: Pre-tokenized (No Packing) - **Current Default**

**When to use:**
- ✅ You want fast training startup with cached tokenized data
- ✅ Sequence lengths are relatively uniform (less benefit from packing)
- ✅ You prefer predictable behavior and explicit control over tokenization

**How it works:**
1. Datasets are fully tokenized during preprocessing
2. Tokenized data is cached to disk (`.cache/preprocessed/`)
3. Subsequent runs load from cache instantly
4. SFTTrainer receives pre-tokenized sequences (no packing)

**Configuration:**
```yaml
# In your config file (e.g., configs/qwen3_7b_movielens_qlora.yaml)
packing: false  # or omit this line (default is false)
```

**Training command:**
```bash
python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml
```

### Mode 2: Packing (Better GPU Utilization) - **New Option**

**When to use:**
- ✅ You have many short sequences with variable lengths
- ✅ You want maximum GPU efficiency and less padding waste
- ✅ You're willing to trade slightly slower startup for better throughput

**How it works:**
1. Datasets are only formatted (converted to chat template + EOS)
2. Formatted data is cached to disk (`.cache/preprocessed/*_formatted_for_packing/`)
3. SFTTrainer handles tokenization AND packs multiple examples per sequence
4. Better GPU utilization due to reduced padding

**Configuration:**
```yaml
# In your config file (e.g., configs/qwen3_7b_movielens_qlora.yaml)
packing: true  # Enable packing mode
```

**Training command:**
```bash
python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml --clear-cache
```

**Note:** Use `--clear-cache` when switching between modes to regenerate the appropriate cache.

## Cache Management

The system uses **separate cache directories** for each mode to prevent conflicts:

- **Pre-tokenized mode:** `.cache/preprocessed/train_preprocessed/`
- **Packing mode:** `.cache/preprocessed/train_formatted_for_packing/`

### Clearing Cache

Clear cache when:
- Switching between packing modes
- Changing `cutoff_len`
- Modifying the formatting function
- Changing `max_eval_samples`

```bash
# Clear cache and regenerate
python scripts/finetune_lora.py --config your_config.yaml --clear-cache
```

## Performance Comparison

### Expected Benefits of Packing

Based on typical MovieLens data:
- **Pre-tokenized mode:** ~30-40% padding waste (depending on sequence length distribution)
- **Packing mode:** <5% padding waste, better GPU utilization

### Trade-offs

| Aspect | Pre-tokenized (No Packing) | Packing Mode |
|--------|---------------------------|--------------|
| **Startup Speed** | ✅ Fast (loads from cache) | ⚠️ Slower first run (tokenizes during training) |
| **GPU Efficiency** | ⚠️ Some padding waste | ✅ Minimal padding waste |
| **Throughput** | ⚠️ Standard | ✅ Better (more examples per batch) |
| **Memory Usage** | Standard | Slightly better |
| **Reproducibility** | ✅ Exact sequence order | ⚠️ Packed sequences vary |
| **Cache Size** | Larger (tokenized) | Smaller (text only) |
| **Subsequent Runs** | ✅✅ Very fast | ✅ Fast (caches formatted text) |

## Recommendations

### Use Pre-tokenized Mode (Default) If:
1. Your sequences are relatively uniform in length (>70% of cutoff_len)
2. You frequently restart training and value instant startup
3. You need exact reproducibility of training sequences

### Use Packing Mode If:
1. You have many short sequences (<50% of cutoff_len)
2. You're doing long training runs where startup time is negligible
3. You want maximum GPU efficiency and throughput
4. Your dataset has high variance in sequence lengths

## Switching Between Modes

1. **Edit your config file:**
   ```yaml
   packing: true  # or false
   ```

2. **Clear the cache:**
   ```bash
   python scripts/finetune_lora.py --config your_config.yaml --clear-cache
   ```

3. **Train as usual:**
   ```bash
   python scripts/finetune_lora.py --config your_config.yaml
   ```

## Verifying Packing is Working

When you run training, check the output:

**Pre-tokenized mode:**
```
Dataset Configuration:
  packing: False
  ...
Trainer configured with pre-tokenized datasets (no packing)
```

**Packing mode:**
```
Dataset Configuration:
  packing: True
  ...
Formatting complete! (Tokenization will be handled by SFTTrainer with packing)
Trainer configured with packing=True, max_seq_length=1024
```

## Example Configs

### Pre-tokenized Config (Current Default)
```yaml
# configs/qwen3_7b_movielens_qlora.yaml
cutoff_len: 1024
packing: false  # or omit this line
preprocessing_cache_dir: .cache/preprocessed
```

### Packing Config
```yaml
# configs/qwen3_7b_movielens_qlora_packing.yaml
cutoff_len: 1024
packing: true  # Enable packing
preprocessing_cache_dir: .cache/preprocessed
```

## Troubleshooting

### Issue: "Cache exists but getting errors"
**Solution:** Clear cache and regenerate
```bash
python scripts/finetune_lora.py --config your_config.yaml --clear-cache
```

### Issue: "Packing doesn't seem to help performance"
**Cause:** Your sequences might be uniformly long (close to cutoff_len)
**Solution:** Check your sequence length distribution:
```python
from datasets import load_from_disk
ds = load_from_disk(".cache/preprocessed/train_preprocessed")
lengths = [len(ex['input_ids']) for ex in ds]
print(f"Mean: {sum(lengths)/len(lengths)}, Max: {max(lengths)}")
```

If mean length is >70% of cutoff_len, packing won't provide significant benefits.

### Issue: "Out of memory with packing"
**Cause:** Packing can fit more examples per sequence, increasing memory usage
**Solution:** Reduce `per_device_train_batch_size` or `gradient_accumulation_steps`

## Summary

✅ **Both modes preserve your caching** - no work is thrown away
✅ **You can switch between modes** - just change config and clear cache
✅ **Pre-tokenized is the safe default** - use packing when you need max efficiency
✅ **Packing works with your data** - sequences are variable length, not pre-padded

Choose based on your specific use case and sequence length distribution!
