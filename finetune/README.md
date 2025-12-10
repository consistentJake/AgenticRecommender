# Finetuning Qwen3 with LoRA/QLoRA

This directory contains scripts and configs for LoRA (QLoRA on CUDA) finetuning of `Qwen/Qwen3-0.6B` on the MovieLens-style binary recommendation task.

## Prerequisites

### Basic Installation
Install core dependencies:
```bash
pip install -r finetune/requirements_sft.txt
```

This includes: `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `tensorboard`, `accelerate`, and other required packages.

### Flash Attention 2 (Recommended for Speed)

Flash Attention 2 provides **2-4x faster training** with lower memory usage. The config is set to `flash_attn: auto` by default.

**Requirements:**
- NVIDIA GPU with compute capability 8.0+ (Ampere, Ada, Hopper: A100, RTX 3090/4090, H100, etc.)
- CUDA 11.6 or higher
- PyTorch with CUDA support

**Installation:**

**Method 1: Try standard installation first**
```bash
pip install flash-attn --no-build-isolation
```

**Method 2: If you encounter CUDA_HOME errors (recommended fix)**

**Issue:** If installation fails with error:
```
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

This happens when the CUDA toolkit (nvcc compiler) is not installed, even though PyTorch with CUDA support is available.

**Solution:** Download and install the pre-built wheel directly:

```bash
# Step 1: Check your PyTorch version and CXX11 ABI setting
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# Step 2: Download the appropriate pre-built wheel
# For PyTorch 2.5.x + CUDA 12.x + Python 3.11 + CXX11 ABI False:
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Step 3: Install the wheel
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Step 4: Clean up
rm flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

**Note:** If your environment differs (different PyTorch version, Python version, or ABI), check the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases) for the correct wheel matching your configuration.

**Verify installation:**
```bash
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}'); from flash_attn import flash_attn_func; print('Success!')"
```

**Fallback:** If Flash Attention fails to install or your GPU doesn't support it, the training will automatically fall back to PyTorch's native SDPA (Scaled Dot Product Attention), which is still efficient on modern GPUs.

### Data
- Ensure `finetune/data/movielens_qwen3/train.json` and `eval.json` exist
- Paths are resolved via `dataset_info.json` or the YAML config

## Config
- Main config: `finetune/configs/qwen3_movielens_qlora.yaml`.
- Most knobs (model path, output_dir, batch sizes, cutoff_len, LoRA params, eval/save steps, gradient checkpointing, early stopping, etc.) are read from this YAML. Override or supply a different config via `--config`.

## Training Best Practices

Our training configuration follows these optimized guidelines:

### 1. Evaluation Strategy
- **Eval frequency**: `eval_steps = 2000` to balance monitoring with training efficiency
  - Reduces evaluation overhead while maintaining visibility into model progress
  - Can also use `eval_strategy = "epoch"` for smaller datasets
- **Eval batch size**: `per_device_eval_batch_size = 16` (maximize based on VRAM)
  - Larger batches speed up evaluation significantly
- **Eval dataset size**: Limited to 1000 samples (`max_eval_samples = 1000`) for quick frequent checks
  - Keeps eval time reasonable while providing reliable metrics

### 2. Batch Size & Gradient Accumulation
- **Train batch size**: `per_device_train_batch_size = 2` (can increase to 4 if VRAM allows)
- **Gradient accumulation**: `gradient_accumulation_steps = 8`
- **Effective batch size**: 16-32 samples
  - Provides training stability without excessive memory usage
  - Balances convergence speed with resource constraints

### 3. Sequence Length & Memory Optimization
- **Max sequence length**: `cutoff_len = 1024` (set close to p95 data length)
  - Avoids wasting compute on excessive padding
  - Adjust based on your actual data distribution
- **Gradient checkpointing**: `gradient_checkpointing = false`
  - Disabled for faster training when VRAM permits
  - Enable (`true`) if encountering OOM errors

### 4. Learning Rate & Schedule
- **Learning rate**: `learning_rate = 1.5e-4` (optimal for LoRA fine-tuning)
- **Warmup**: `warmup_ratio = 0.03` (gradual ramp-up prevents early instability)
- **Schedule**: `lr_scheduler_type = cosine` (smooth decay)
- **Optimizer**: AdamW with standard betas (0.9, 0.999)

### 5. Metrics & Early Stopping
- **Metrics computed**: Loss, Accuracy, F1 score
  - Accuracy measures exact match on Yes/No predictions
  - F1 score handles class imbalance in binary recommendations
- **Early stopping**: `early_stopping_patience = 3`
  - Stops training if eval metrics plateau/degrade
  - Prevents overfitting and saves compute time

### 6. LoRA Configuration
- **Rank**: `lora_rank = 16` (good balance of capacity vs. efficiency)
- **Alpha**: `lora_alpha = 64` (4x rank for stable training)
- **Dropout**: `lora_dropout = 0.05` (light regularization)
- **Target modules**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)

### 7. Flash Attention Configuration
- **Setting**: `flash_attn: auto` enables Flash Attention 2 automatically when available
- **Performance**: 2-4x faster training with 30-50% lower memory usage
- **Automatic fallback**: If Flash Attention 2 is not installed or GPU doesn't support it, falls back to PyTorch SDPA
- **Manual control**: Set to `flash_attention_2`, `sdpa`, `eager`, or `false` to override auto-detection

## Training

### Standard Training (Foreground)
```bash
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml
```

### Background Training (Survives SSH Disconnection)

For remote servers where you need to disconnect SSH, use `nohup`:

```bash
# Start training in background
nohup python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml > training.log 2>&1 &

# Get the process ID
echo "Training started with PID: $!"

# Check if training is running
ps aux | grep finetune_lora.py | grep -v grep

# Monitor progress in real-time
tail -f training.log

# Monitor without verbose progress bars
tail -f training.log | grep -E "loss|accuracy|f1|Step|Epoch|complete"

# Check latest progress
tail -50 training.log | grep -E "loss|Step"

# Kill training if needed
pkill -f "finetune_lora.py"
```

**After disconnecting SSH and reconnecting:**
```bash
# Check if still running
pgrep -f "finetune_lora.py"

# Monitor progress
tail -f training.log

# Check GPU usage
nvidia-smi

# View specific metrics
tail -100 training.log | grep "{'loss"
```

### Alternative: Using tmux (More Interactive)
```bash
# Start tmux session
tmux new -s training

# Run training inside tmux
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml

# Detach from tmux: Press Ctrl+B, then D
# (Training continues in background)

# Reconnect to session later
tmux attach -t training

# List all sessions
tmux ls

# Kill session when done
tmux kill-session -t training
```

### Training Output
- Outputs (adapter + tokenizer + logs) land in `output/qwen3-movielens-qlora` by default.
- Training/eval metrics include loss, accuracy, and F1 on the Yes/No labels. Early stopping patience is configurable in the YAML.
- Preprocessed datasets are cached in `.cache/preprocessed/` for faster subsequent runs (instant loading vs ~13 seconds of tokenization)

### Checkpointing & Model Saving

**Checkpoint Location:**
```
output/qwen3-movielens-qlora/
├── checkpoint-4000/          # Checkpoint at step 4000
├── checkpoint-8000/          # Checkpoint at step 8000
├── adapter_model.safetensors # Best model (final save)
├── adapter_config.json       # LoRA configuration
├── tokenizer files...
└── logs/                     # TensorBoard logs
```

**Configuration:**
- `save_steps: 4000` - Save checkpoint every 4000 steps
- `save_total_limit: 2` - Keep only last 2 checkpoints (auto-deletes older ones)
- `load_best_model_at_end: true` - Best model is loaded and saved to output_dir after training

**Each checkpoint contains:**
- `adapter_model.safetensors` (39MB) - LoRA adapter weights at that step
- `optimizer.pt` (78MB) - Optimizer state for resuming training
- `scheduler.pt` - Learning rate scheduler state
- `rng_state.pth` - Random number generator state (for reproducibility)
- `trainer_state.json` - Training metrics, loss history, eval scores
- Tokenizer files (vocab.json, tokenizer.json, etc.)

**Resuming from a checkpoint:**
```bash
python scripts/finetune_lora.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --resume_from_checkpoint output/qwen3-movielens-qlora/checkpoint-4000
```

**Best model selection:**
- Trainer tracks the best checkpoint based on `metric_for_best_model: "f1"`
- At training end, the best checkpoint is loaded and saved to the main output directory
- Use the adapter files in `output/qwen3-movielens-qlora/` for inference (not the checkpoint subdirectories)

**Checkpoint inspection:**
```bash
# View training metrics from a checkpoint
cat output/qwen3-movielens-qlora/checkpoint-4000/trainer_state.json | jq '.log_history[-5:]'

# Check which checkpoint is best
cat output/qwen3-movielens-qlora/checkpoint-4000/trainer_state.json | jq '.best_model_checkpoint'

# List all checkpoints
ls -lht output/qwen3-movielens-qlora/checkpoint-*/
```

### Training Phases

Training proceeds through these phases:

1. **Model Loading** (~30-60 seconds)
   - Loads Qwen3-0.6B model with 4-bit quantization (QLoRA)
   - Initializes LoRA adapters
   - GPU memory usage: ~20-21 GB

2. **Data Preprocessing** (~2-3 minutes for 78k examples)
   - **Applying formatting function**: Converts examples to chat template format (CPU-bound)
   - **Adding EOS tokens**: Appends end-of-sequence tokens (CPU-bound, fast)
   - **Tokenizing dataset**: Converts text to token IDs (CPU-bound, slowest phase)
   - Progress bars show examples/second for each phase
   - **Note**: Tokenization is CPU-only and cannot be GPU-accelerated

3. **Training Loop** (varies by config)
   - GPU utilization jumps to 90-100%
   - Progress shows step/epoch, loss, learning rate
   - Evaluation runs every 2000 steps (default)
   - Checkpoints saved to `output/qwen3-movielens-qlora/checkpoint-{step}`

### Monitoring Training Progress

**Quick status check (GPU and process):**
```bash
nvidia-smi && echo "---" && ps aux | grep finetune_lora.py | grep -v grep
```
What to look for:
- **GPU memory usage**: ~21 GB indicates model is loaded
- **GPU utilization**: 21% during tokenization, 90-100% during training
- **Process CPU usage**: 100% during data preprocessing
- **Process PID and runtime**: Shows process is active

**Check if training started (look for step output):**
```bash
ps aux | grep finetune_lora.py | grep -v grep && tail -100 /proc/$(pgrep -f "finetune_lora.py")/fd/2 2>/dev/null | grep -E "Step|loss|Training"
```

**Monitor training output in background job:**
If you ran training in background with `run_in_background=true`, check output:
```bash
# List all background jobs
jobs -l

# Check specific background job output (replace job_id with actual ID)
# This shows the current status and any new output
fg %job_id  # Or use job management commands
```

**View current progress (filters out verbose progress bars):**
```bash
tail -200 /proc/$(pgrep -f "finetune_lora.py")/fd/2 2>/dev/null | grep -v "examples/s\|Tokenizing\|Applying formatting\|Adding EOS" | tail -20
```

**Check training output directory:**
```bash
ls -lht output/qwen3-movielens-qlora/
```
Shows checkpoints and logs as they're created.

**Monitor in real-time (shows last 30 lines every 5 seconds):**
```bash
watch -n 5 'tail -30 /proc/$(pgrep -f "finetune_lora.py")/fd/2 2>/dev/null | grep -E "Step|Epoch|loss|accuracy|f1|Training|Evaluating" || echo "Waiting for training output..."'
```

**View full output with progress bars:**
```bash
tail -f /proc/$(pgrep -f "finetune_lora.py")/fd/2 2>/dev/null
```

**Check process details:**
```bash
# Get process ID
pgrep -f "finetune_lora.py"

# Check process status and resource usage
top -p $(pgrep -f "finetune_lora.py")

# Check how long the process has been running
ps -p $(pgrep -f "finetune_lora.py") -o pid,etime,cmd
```

**Kill training if needed:**
```bash
# Gracefully terminate
pkill -f "finetune_lora.py"

# Force kill if necessary (use with caution)
pkill -9 -f "finetune_lora.py"

# Clear GPU memory after killing
nvidia-smi  # Verify no processes remain
``` 

**How to Know Tokenization Progress:**

Check the last line of output for the tokenization progress bar:
```bash
pgrep -f "finetune_lora.py" > /dev/null && tail -1 /proc/$(pgrep -f "finetune_lora.py")/fd/2 2>/dev/null | grep "Tokenizing"
```

Example output:
```
Tokenizing train dataset:  42%|████▏     | 32817/78271 [00:21<00:29, 1561.92 examples/s]
```
This shows: 42% complete, ~29 seconds remaining

**Signs that tokenization is finished:**
- Progress bar shows `100%|██████████| 78271/78271`
- Next you'll see: `Applying formatting function to eval dataset`
- Then: `The tokenizer has new PAD/BOS/EOS tokens...`
- Finally: Training loop starts with `0/14676 [00:00<?, ?it/s]`

**Check if training has started:**
```bash
pgrep -f "finetune_lora.py" > /dev/null && tail -50 /proc/$(pgrep -f "finetune_lora.py")/fd/2 2>/dev/null | grep -E "loss|Step|Training" | head -5
```

Once training begins, you'll see step output with loss values:
```
{'loss': 0.XXXX, 'learning_rate': 1.XXe-04, 'epoch': 0.XX}
  1%|          | 10/14676 [00:XX<XX:XX,  X.XX it/s]
```

**Debugging Tips:**
- **"No running processes found" in nvidia-smi**: This is a known nvidia-smi display bug during the tokenization phase. The process is running correctly - verify by checking GPU memory usage (~21 GB) and process status with `ps aux`.
- **High CPU, low GPU usage**: Normal during data preprocessing (tokenization). GPU usage will spike to 90-100% once training begins.
- **Process stuck**: Check stderr output with `tail -100 /proc/$(pgrep -f "finetune_lora.py")/fd/2` for error messages.
- **Tokenization taking long**: Normal - processes ~1,500 examples/sec on CPU. For 78k examples, expect ~50-60 seconds.

## Inference
- Compare base vs adapter:
  ```bash
  python finetune/scripts/infer_lora.py
  ```
- `ADAPTER_DIR` in the script points at the default training output. Adjust if you train to a different `output_dir`.

## Monitoring (TensorBoard)
- Training logs are written to `<output_dir>/logs`. Start TensorBoard:
  ```bash
  tensorboard --logdir output/qwen3-movielens-qlora/logs --port 6006
  ```
- Open http://localhost:6006 to view loss curves, accuracy/F1, and eval checkpoints. Point `--logdir` to a parent folder (e.g., `output`) to compare multiple runs.

## Quick sequence-length audit
- Token-based (uses the same tokenizer as training):
  ```bash
  python finetune/scripts/check_seq_len.py --config finetune/configs/qwen3_movielens_qlora.yaml --max-samples 2000
  ```
- Char-only (no model/tokenizer download):
  ```bash
  python finetune/scripts/check_seq_len.py --config finetune/configs/qwen3_movielens_qlora.yaml --char-only --max-samples 2000
  ```
- Reports min/max/mean and percentiles (p50/p90/p95/p99/p100) for the train split using the same chat template formatting. Use this to set `cutoff_len` near your p95 length.

## Tuning for Your Hardware
- **Low VRAM (8GB)**: Keep `per_device_train_batch_size = 1-2`, enable `gradient_checkpointing = true`
- **Medium VRAM (16GB)**: Use `per_device_train_batch_size = 2-4`, can disable gradient checkpointing
- **High VRAM (24GB+)**: Increase batch sizes further and eval batch size to 32+


## Evaluation 
  1. compare_base_vs_lora.py

  - load_data(): Handles both JSON and JSONL formats
  - format_prompt(): Supports both training (instruction/input) and test (history_titles/candidate_title) formats
  - generate(): Accepts optional system_prompt parameter
  - Inference loops: Extract labels from either output or label fields
  - Result saving: Includes format-specific fields (user_id, candidate_title, instruction, input)
  - Disagreement analysis: Works with both formats

  2. analyze_differences.py

  - display_example(): Shows different fields based on data format:
    - Test format: displays user_id, candidate_title, history_titles
    - Training format: displays input prompt (truncated to 300 chars)

  Usage Examples:

  On training data:
  python scripts/compare_base_vs_lora.py \
    --test_file data/movielens_qwen3/train.json \
    --output_dir infer/train_comparison \
    --max_samples 100

  On test data (existing behavior):
  python scripts/compare_base_vs_lora.py \
    --test_file data/movielens_qwen3/test_raw.jsonl \
    --output_dir infer/test_comparison \
    --max_samples 50

  Then analyze results:
  python scripts/analyze_differences.py \
    --infer_dir infer/train_comparison \
    --show_examples 10

## Training Metrics Issue: Why Accuracy/F1 Were Showing 0

### Problem
During training, evaluation metrics (accuracy and F1) were showing 0.0, even though the model was learning and inference showed good performance (accuracy ~68-70%, F1 ~64-66%).

### Root Cause
The `compute_metrics` function in `finetune_lora.py` was decoding the **entire sequence** (including system prompt, user message, and assistant response) instead of just the assistant's answer:

**Decoded text looked like:**
```
"system\nYou are a movie recommendation assistant...\nuser\nUser's last 15 watched movies:...\nassistant\n<think>\n\n</think>\n\nNo"
```

The original `_normalize_label` function checked if text **started with** "yes" or "no":
```python
def _normalize_label(text: str) -> Optional[int]:
    cleaned = text.strip().lower()
    if cleaned.startswith("yes"):  # ← This fails! Text starts with "system"
        return 1
    if cleaned.startswith("no"):
        return 0
    return None
```

**Result:** All predictions returned `None`, got filtered out, and metrics calculated as 0.0.

### Solution
Updated `compute_metrics` to use the same answer extraction logic as inference scripts:

1. **Extract only assistant's response**: Split by "assistant" to isolate the generated text
2. **Remove `<think>` tags**: Use regex to strip thinking tags and their content
3. **Search for yes/no**: Look for the answer in the cleaned text

**New implementation (finetune_lora.py:452-484):**
```python
def _extract_answer(text: str) -> Optional[str]:
    """Extract Yes/No answer from model response, matching inference logic."""
    import re

    # Extract only the assistant's actual response after the prompt
    if "assistant" in text:
        text = text.split("assistant")[-1]

    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Clean up whitespace
    text = text.strip().lower()

    # Look for yes/no in the cleaned response
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    elif "yes" in text and "no" not in text:
        return "yes"
    elif "no" in text and "yes" not in text:
        return "no"
    else:
        return None

def _normalize_label(text: str) -> Optional[int]:
    answer = _extract_answer(text)
    if answer == "yes":
        return 1
    elif answer == "no":
        return 0
    return None
```

### Impact
- Training metrics now show **real accuracy and F1 scores** during evaluation
- Metrics match inference results (both ~68-70% accuracy, ~64-66% F1)
- Early stopping and model selection now work correctly based on actual performance
- No impact on final model quality - the model was learning correctly, we just couldn't see it in the metrics

### Key Takeaway
When implementing custom metrics for generative models, ensure your metric calculation logic matches your inference/evaluation logic, especially when:
- Decoding full sequences vs. generated portions only
- Models use special tokens or reasoning tags (like `<think>`)
- Looking for specific answer formats in generated text