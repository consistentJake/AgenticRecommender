# Chunked Inference for Base vs LoRA Comparison

## Overview

The `compare_base_vs_lora.py` script supports chunked processing mode, which optimizes memory usage and allows LoRA inference to potentially reuse intermediate results from base model inference.

**Also see:** [LORA_INFERENCE_OPTIMIZATION.md](./LORA_INFERENCE_OPTIMIZATION.md) for details on additional performance optimizations (direct PEFT usage, prompt caching, etc.).

## How It Works

### Traditional Approach (Full Processing)
```
1. Load base model
2. Run inference on ALL samples with base model
3. Load LoRA adapter
4. Run inference on ALL samples with LoRA model
5. Compare results
```

**Memory Requirements:**
- Base model: stays in VRAM throughout
- LoRA adapter: added after all base inference
- Results: accumulate in RAM for all samples

### Chunked Processing Approach
```
For each chunk of X samples:
  1. Run base model inference on chunk
  2. Load LoRA adapter
  3. Run LoRA inference on same chunk
  4. Write results to disk immediately
  5. Unload LoRA adapter
  6. Clear GPU cache
  7. Continue to next chunk
```

**Benefits:**
1. **Lower Memory Usage**: Results are written immediately, not accumulated
2. **Better Cache Locality**: Base and LoRA inference on same chunk are close together
3. **LoRA Reuses Base State**: LoRA is loaded while base model activations might still be in cache
4. **Incremental Progress**: Results are saved continuously (safer for long runs)
5. **Flexible Resource Management**: Unload LoRA between chunks to free VRAM

## Configuration

Add the `chunk_size` parameter to your config file's `inference` section:

```yaml
inference:
  batch_size: 8
  chunk_size: 100  # Process 100 samples at a time
  max_samples: 1000
  test_file: data/movielens_qwen3/test_raw.jsonl
```

### Parameters

- **`chunk_size`** (optional): Number of samples to process together before switching models
  - If not specified or `null`: Process all data at once (traditional approach)
  - Recommended values:
    - Small datasets (<1000 samples): Can use full processing
    - Medium datasets (1000-10000): Try 200-500
    - Large datasets (>10000): Try 500-1000
  - Consider your available VRAM and sample complexity

- **`batch_size`**: Number of samples to process in a single forward pass
  - This is nested within `chunk_size`
  - Example: `chunk_size=100, batch_size=8` means:
    - Process 100 samples per chunk
    - Within each chunk, process in batches of 8

## Example Configurations

### Small Dataset (Full Processing)
```yaml
inference:
  batch_size: 4
  # No chunk_size specified - process all at once
  max_samples: 500
```

### Medium Dataset (Chunked Processing)
```yaml
inference:
  batch_size: 8
  chunk_size: 200
  max_samples: 2000
```

### Large Dataset (Chunked Processing)
```yaml
inference:
  batch_size: 8
  chunk_size: 500
  # No max_samples - process entire dataset
```

## Memory Considerations

### Calculating Chunk Size

Consider these factors:
1. **Available VRAM**: More VRAM allows larger chunks
2. **Model Size**: Larger models need smaller chunks
3. **Sequence Length**: Longer sequences need smaller chunks
4. **Batch Size**: Larger batches need smaller chunks

### Formula (rough estimate)
```
chunk_size = (Available_VRAM_GB - Model_Size_GB) / (Sequence_Length * Batch_Size * 0.001)
```

### Example
- Available VRAM: 16GB
- Model Size: 8GB (with quantization)
- Sequence Length: 2048 tokens
- Batch Size: 8

```
chunk_size ≈ (16 - 8) / (2048 * 8 * 0.001) ≈ 488
```

Round down to a nice number like 400 for safety.

## Performance Tips

1. **Batch Size First**: Maximize `batch_size` for your VRAM, then tune `chunk_size`
2. **Even Division**: Make `chunk_size` divisible by `batch_size` for cleaner batching
3. **Monitor VRAM**: Watch GPU memory usage to find optimal chunk size
4. **Experiment**: Different workloads benefit from different chunk sizes

## Output Behavior

Both full and chunked processing modes:
- Write results to disk continuously during inference
- Save to the same output files
- Produce identical final results

The only differences are:
- Memory usage patterns
- When LoRA adapter is loaded/unloaded
- Progress tracking messages

## Example Usage

```bash
# Use config with chunk_size specified
python scripts/compare_base_vs_lora.py --config configs/qwen3_7b_movielens_qlora.yaml
```

You'll see output like:
```
================================================================================
CHUNKED PROCESSING MODE
================================================================================
Processing data in chunks of 200 samples
For each chunk:
  1. Run base model inference
  2. Load LoRA adapter and run LoRA inference
  3. Unload LoRA adapter and clear cache
  4. Write results to disk
This approach minimizes memory usage and allows LoRA to reuse base model state.
================================================================================

================================================================================
Processing Chunk 1/5
Samples 0 to 199 (200 samples)
================================================================================

============================================================
Step 1: BASE MODEL inference (chunk 1/5)
============================================================
Base inference (chunk 1): 100%|████████████| 25/25 [00:45<00:00]

============================================================
Step 2: LORA MODEL inference (chunk 1/5)
============================================================
Loading LoRA adapter from output/qwen3-7b-movielens-qlora-2...
LoRA adapter loaded successfully.
LoRA inference (chunk 1): 100%|████████████| 25/25 [00:43<00:00]

============================================================
Step 3: Cleanup before next chunk
============================================================
LoRA adapter unloaded.
GPU cache cleared.
Chunk 1 complete. Ready for next chunk.
```

## Troubleshooting

### Out of Memory Errors
- Reduce `chunk_size`
- Reduce `batch_size`
- Use gradient checkpointing (in training config)
- Enable more aggressive quantization

### Slow Processing
- Increase `batch_size` (if VRAM allows)
- Increase `chunk_size` (if VRAM allows)
- Chunk size too small causes overhead from frequent LoRA loading/unloading

### LoRA Loading Overhead
- If you have enough VRAM, increase `chunk_size` or disable chunking
- Each chunk requires one LoRA load/unload cycle
