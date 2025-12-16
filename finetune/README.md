# Finetuning Qwen3 with LoRA/QLoRA

This directory contains scripts and configs for LoRA (QLoRA on CUDA) finetuning of `Qwen/Qwen3-0.6B` on the MovieLens-style binary recommendation task.

## Prerequisites

### Basic Installation
Install core dependencies:
```

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

## Code Architecture

### Shared Utilities (`scripts/utils.py`)
All common functions and constants are centralized in `scripts/utils.py` to eliminate duplication and ensure consistency:

**Constants:**
- `BASE_MODEL = "Qwen/Qwen3-0.6B"` - Model identifier used across all scripts
- `ADAPTER_DIR = "output/qwen3-movielens-qlora"` - Default LoRA adapter location
- `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P` - Generation parameters
- `DEFAULT_SYSTEM_PROMPT` - System prompt for movie recommendations

**Core Functions:**
- `get_device()` - Device detection (MPS/CUDA/CPU)
- `load_yaml_config()`, `load_json()`, `load_data()` - Configuration and data loading
- `format_prompt()`, `to_chat_messages()` - Data formatting for chat templates
- `build_datasets()`, `tokenize_func()`, `preprocess_datasets_parallel()` - Dataset preparation
- `generate()` - Model inference with configurable parameters
- `extract_answer()`, `compute_metrics()` - Evaluation utilities

### Unified Inference Script (`scripts/infer_lora.py`)
Single script supporting two modes:

**Demo Mode** (no arguments):
```bash
python scripts/infer_lora.py
```
Runs hardcoded example comparing base vs LoRA models.

**Batch Mode** (with `--output_dir`):
```bash
python scripts/infer_lora.py --test_file data/test.jsonl --output_dir output/results
```
Processes test files, computes metrics, saves detailed results.

Options: `--adapter_dir`, `--max_samples`, `--compare_base`

### Testing
Comprehensive unit tests in `tests/test_utils.py`:
```bash
pytest tests/test_utils.py -v
```
Tests cover all utility functions with special focus on `tokenize_func()` for training consistency.

Integration smoke tests now also cover:
- Left-side truncation preserving assistant labels under `cutoff_len`
- Loss masking with `ignore_index=-100`
- Prompt → generate → extract → metrics pipeline (mocked model)

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
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml --clear-cache
```

#### With lightweight generative eval
Add to your YAML (e.g., `configs/qwen3_movielens_qlora.yaml`):
```yaml
gen_eval_samples: 32
```
Then run:
```bash
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml --clear-cache
```
This runs a small greedy generation eval (max_new_tokens=8) after training to mirror inference behavior.

### Background Training (Survives SSH Disconnection)

For remote servers where you need to disconnect SSH, use `nohup`:

```bash
# Start training in background
nohup python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml > training.log 2>&1 &

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

### Resuming Training from Checkpoints

If training was interrupted or you want to continue training with more epochs, use the `--resume` flag:

```bash
# Resume from latest checkpoint in output_dir
python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml --resume

# Resume and train for 2 additional epochs beyond what's in the config
# (e.g., if config has 3 epochs, this will train to 5 total)
python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml --resume --extra-epochs 2

# Example: Continue training in background
nohup python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml --resume --extra-epochs 2 > training_resume.log 2>&1 &
```

**How it works:**
- The script automatically finds the latest checkpoint in `output_dir` (e.g., `checkpoint-12000`)
- The checkpoint number (12000) represents the **last completed training step**
- Training continues from that checkpoint with all optimizer states preserved
- **Step counter continues from where it left off** (e.g., if checkpoint-12000, next step will be 12001)
- If no checkpoint is found, training starts from scratch
- Use `--extra-epochs` to train beyond the original `num_train_epochs` setting

**Example output when resuming:**
```
================================================================================
RESUMING TRAINING FROM CHECKPOINT
================================================================================
Checkpoint: checkpoint-12000
Last completed step: 12000
Next step will be: 12001
Epoch: 2.45
================================================================================
```

## Inference (Deterministic Yes/No)

- Demo mode (prints base vs LoRA on a canned prompt):
```bash
python scripts/infer_lora.py --adapter_dir output/qwen3-movielens-qlora
```

- Batch mode on MovieLens test split with greedy decoding (max_new_tokens=8):
```bash
python scripts/infer_lora.py \
  --test_file data/movielens_qwen3/test_raw.jsonl \
  --output_dir output/infer \
  --adapter_dir output/qwen3-movielens-qlora
```
Outputs `lora_predictions.jsonl` and `metrics_summary.json`.

- Quick check on the first eval sample (training-format JSON):
```bash
python scripts/infer_lora.py \
  --test_file data/movielens_qwen3/eval.json \
  --max_samples 1 \
  --output_dir output/infer_eval1 \
  --adapter_dir output/qwen3-movielens-qlora
```

## Integration Test (optional real model)

Use the same loader/config as training. Either pass a model id/path directly:
```bash
pytest tests/test_integration_model.py --integration-model-path Qwen/Qwen3-8B -v -s
```
or point to a config that contains `model_name_or_path` (default: `configs/qwen3_7b_movielens_qlora.yaml`):
```bash
pytest tests/test_integration_model.py --integration-config-path configs/qwen3_7b_movielens_qlora.yaml -v -s
```
Add `-s` to surface the logged raw input JSON, templated text, and model response. This uses the shared model/tokenizer loader (quantization, flash attention, padding) and runs a forward pass with masked labels plus a short greedy generation. Skips if no model can be resolved.


### pack the files
git archive --format=tar.gz -o committed-code.tar.gz HEAD



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

### Preprocessing Cache

The training script automatically caches preprocessed datasets to dramatically speed up subsequent training runs:

**What gets cached:**
- Chat template formatting (applying `tokenizer.apply_chat_template`)
- EOS token addition
- Full tokenization (converting text to token IDs)
- Label masking (setting prompt tokens to -100)

**Cache location:**
- Default: `.cache/preprocessed/train_preprocessed` and `.cache/preprocessed/eval_preprocessed`
- Configurable via `preprocessing_cache_dir` in the YAML config

**Time savings:**
- **First run**: ~2-3 minutes for formatting and tokenizing 78k examples
- **Subsequent runs**: ~1-2 seconds to load from cache (near instant)

**Cache behavior:**
- On first training run, the script preprocesses datasets and saves them to cache
- On subsequent runs, if cache exists, it loads directly from disk and **skips all preprocessing steps**
- You'll see `"Loading preprocessed datasets from cache..."` instead of `"Applying formatting function to train dataset..."`

**When to clear cache:**
- Changed `max_eval_samples` (eval dataset size changed)
- Modified dataset files (`train.json` or `eval.json`)
- Changed tokenizer or chat template
- Changed `cutoff_len` (max sequence length)
- Modified any preprocessing logic

**Clear cache and regenerate:**
```bash
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml --clear-cache
```

**Manual cache deletion:**
```bash
rm -rf .cache/preprocessed/
```

**Note:** The cache is dataset-specific but not config-specific. If you switch between different configs that use the same dataset files, they'll share the same cache.

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
    tensorboard --logdir output/qwen3-7b-movielens-qlora/logs --port 6007
        tensorboard --logdir output/qwen3-7b-movielens-qlora-2/logs --port 6007

        tensorboard --logdir output/qwen3-7b-movielens-qlora-speical-token/logs --port 6007


  ```
- Open http://localhost:6006 to view loss curves, accuracy/F1, and eval checkpoints. Point `--logdir` to a parent folder (e.g., `output`) to compare multiple runs.

## Dataset Sequence Length Analysis

The `scripts/check_seq_len.py` tool provides comprehensive analysis of your dataset's sequence lengths, helping you:
- Verify that `cutoff_len` is appropriately sized
- Understand how JSON data is transformed into tokenized inputs
- Identify truncation issues before training
- See concrete examples of the full data pipeline

### Basic Usage

**Full analysis with example transformations:**
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 2
```

**Quick statistics (no examples):**
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 0
```

**Analyze subset for quick check:**
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --max-samples 1000
```

**Test different cutoff lengths:**
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --cutoff-len 896
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --cutoff-len 512
```

**Character-based analysis (no tokenizer needed):**
```bash
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --char-only
```

### What the Script Shows

**When using `--show-examples`, you'll see the complete transformation pipeline:**

1. **Original JSON Record**: The raw data format (instruction, input, output, system)
2. **Chat Messages**: Converted to role-based format (system, user, assistant)
3. **Applied Chat Template**: Formatted string with Qwen special tokens (`<|im_start|>`, `<|im_end|>`)
4. **Tokenized**: Token IDs and counts
5. **Truncation Analysis**: What would be removed if sequence exceeds cutoff
6. **Label Masking**: How many tokens are masked (-100) vs. trained

**Example Output:**
```
================================================================================
EXAMPLE TRANSFORMATIONS: JSON → Chat Messages → Formatted String → Tokens
================================================================================

────────────────────────────────────────────────────────────────────────────
EXAMPLE 1
────────────────────────────────────────────────────────────────────────────

[STEP 1] Original JSON Record:
  instruction: Predict whether the user will like the candidate movie...
  input: User's last 15 watched movies:
1. Pearl Harbor (2001) (rating ≈ 3.0)
...
  output: No
  system: You are a movie recommendation assistant...

[STEP 2] Converted to Chat Messages:
  1. role=system: You are a movie recommendation assistant...
  2. role=user: Predict whether the user will like...
  3. role=assistant: No...

[STEP 3] Applied Chat Template (Qwen format with special tokens):
  <|im_start|>system
  You are a movie recommendation assistant...
  <|im_end|>
  <|im_start|>user
  Predict whether the user will like the candidate movie...
  <|im_end|>
  <|im_start|>assistant
  No<|im_end|>

[STEP 4] Tokenized:
  Total tokens: 427
  First 20 token IDs: [151644, 8948, 198, 2610, 525, ...]
  Last 20 token IDs: [..., 2753, 151645, 198]

[STEP 5] Truncation Analysis (cutoff_len=1024):
  ✓ No truncation needed (427 <= 1024)

[STEP 6] Label Masking for Training:
  Prompt tokens (masked with -100): 420
  Response tokens (trained): 7
  Masking ratio: 98.4% masked

================================================================================
SEQUENCE LENGTH STATISTICS
================================================================================

Dataset: data/movielens_qwen3/train.json
Measured: 78271 samples (tokens)
Cutoff length: 1024

Statistics:
  min: 393.00
  max: 875.00
  mean: 435.02
  p50: 432.00
  p90: 458.00
  p95: 468.00
  p99: 493.00
  p100: 875.00

Truncation Analysis:
  Samples > 1024 tokens: 0 (0.0%)
  Samples <= 1024 tokens: 78271 (100.0%)

  ✓ No truncation needed for any samples.
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to YAML config | `finetune/configs/qwen3_movielens_qlora.yaml` |
| `--show-examples` | Number of transformation examples to show | `2` |
| `--max-samples` | Limit number of samples to analyze | All samples |
| `--cutoff-len` | Override cutoff length for testing | From config |
| `--char-only` | Skip tokenizer, count characters instead | False |
| `--percentiles` | Comma-separated percentiles to report | `50,90,95,99,100` |

### Interpreting Results

**Setting `cutoff_len`:**
- Ideal: Set to ~1.1-1.2x your p95 length for safety margin
- Too low: If p99 > cutoff_len, many samples will be truncated
- Too high: Wastes memory and compute on padding

**Truncation Warnings:**
- **0% truncated**: Perfect, no data loss
- **<1% truncated**: Generally acceptable
- **1-5% truncated**: Review truncation strategy
- **>5% truncated**: Consider reducing `history_len` in data preparation or increasing `cutoff_len`

**For MovieLens Dataset:**
```
Mean: 435 tokens (42% of cutoff)
p95:  468 tokens (46% of cutoff)
Max:  875 tokens (85% of cutoff)

Conclusion: cutoff_len=1024 is appropriate with 15% headroom.
Could reduce to 896 for 12.5% memory savings with zero risk.
```

### When to Run This Tool

1. **Before first training**: Verify your `cutoff_len` setting
2. **After changing data preparation**: Check impact on sequence lengths
3. **When debugging**: Understand exactly how data is formatted
4. **For optimization**: Find the minimum safe `cutoff_len` to save memory

### Documentation

See `/workspace/docs/` for detailed documentation:
- `dataset_preparation_and_tokenization.md`: Complete pipeline explanation
- `cutoff_analysis_summary.md`: Analysis of cutoff length and truncation strategy

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
  nohup python scripts/compare_base_vs_lora.py --config configs/qwen3_7b_movielens_qlora.yaml > compare_base_vs_lora.log 2>&1 &

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


### Memory calculation

Let me give you the detailed VRAM breakdown for Qwen3-8B with QLoRA, batch_size=2, seq_len=1024, gradient_checkpointing=True:

  Memory Breakdown

  1. Base Model (4-bit Quantized): ~4.5 GiB

  8B parameters × 0.5 bytes/param (4-bit) = 4.0 GiB
  + Quantization metadata (scales, zeros)  = 0.5 GiB
  ────────────────────────────────────────────────
  Total: 4.5 GiB

  2. LoRA Adapters (trainable): ~0.03 GiB

  With lora_target: qv (only q_proj and v_proj):
  Per layer: 2 projections × (4096 × rank + rank × 4096)
           = 2 × (4096 × 16 + 16 × 4096)
           = 2 × 131,072 params

  32 layers: 32 × 262,144 = 8,388,608 params
  In bf16: 8.4M × 2 bytes = 16.8 MB ≈ 0.017 GiB

  (Note: If you used lora_target: all, this would be ~0.1 GiB)

  3. Optimizer States (AdamW for LoRA only): ~0.10 GiB

  AdamW stores 3 copies (params + momentum + variance):
  - Gradients:      8.4M × 4 bytes = 33.6 MB
  - First moment:   8.4M × 4 bytes = 33.6 MB
  - Second moment:  8.4M × 4 bytes = 33.6 MB
  ────────────────────────────────────────────────
  Total: 100.8 MB ≈ 0.10 GiB

  4. Activations (WITH gradient checkpointing): ~0.6 GiB


    5. Logits Tensor (forward pass peak): ~2.5 GiB

  Logits: batch × seq × vocab_size
        = 2 × 1024 × 151,643 × 4 bytes (fp32)
        = 1.24 GiB

  During loss computation in SFTTrainer:
    shift_logits = logits[..., :-1, :].contiguous()  # Creates copy
    shift_labels = labels[..., 1:]
    
    Peak: Original (1.24 GB) + Shifted contiguous copy (1.24 GB)
  ────────────────────────────────────────────────
  Peak during loss: ~2.5 GiB

  6. CUDA Context & Overhead: ~1.5 GiB

  - CUDA runtime context:         500 MB
  - cuBLAS workspace:              300 MB
  - Flash Attention workspace:     200 MB
  - PyTorch allocator fragmentation: 500 MB
  ────────────────────────────────────────────────
  Total: ~1.5 GiB

  Total Memory Usage

  | Component                     | Memory     |
  |-------------------------------|------------|
  | 1. Base Model (4-bit)         | 4.5 GiB    |
  | 2. LoRA Adapters              | 0.03 GiB   |
  | 3. Optimizer States           | 0.10 GiB   |
  | 4. Activations (checkpointed) | 0.6 GiB    |
  | 5. Logits (peak)              | 2.5 GiB    |
  | 6. CUDA Overhead              | 1.5 GiB    |
  | Subtotal                      | 9.2 GiB    |
  | + Safety margin (10%)         | 0.9 GiB    |
  | Total Estimate                | ~10-11 GiB |

  Verification from Your Log

  Your training log with bs=16, no checkpointing showed:
  22.39 GiB allocated by PyTorch

  Let's verify the scaling:

  | Config                | Logits   | Activations | Model+LoRA | Total         |
  |-----------------------|----------|-------------|------------|---------------|
  | bs=16, no checkpoint  | 9.9 GiB  | ~8 GiB      | 4.6 GiB    | ~22.5 GiB ✅  |
  | bs=2, with checkpoint | 1.24 GiB | ~0.6 GiB    | 4.6 GiB    | ~10-11 GiB ✅ |

  Memory savings:
  - Logits: 8x reduction (16→2 batch)
  - Activations: 13x reduction (gradient checkpointing)
  - Total: 2.2x reduction

  Why Gradient Checkpointing Saves So Much Memory

  Forward Pass Activations for 36 layers:

  WITHOUT checkpointing:
  ┌─────────────────────────────────────┐
  │ Layer 1  → Store 16.8 MB            │
  │ Layer 2  → Store 16.8 MB            │
  │ ...                                 │
  │ Layer 36 → Store 16.8 MB            │
  │ Total: 605 MB STORED IN VRAM        │ ❌
  │ + MLP intermediates: +1.8 GiB       │
  └─────────────────────────────────────┘

  WITH checkpointing (saves every 6th layer):
  ┌─────────────────────────────────────┐
  │ Layer 1  → Discard                  │
  │ Layer 6  → CHECKPOINT (16.8 MB)     │
  │ Layer 12 → CHECKPOINT (16.8 MB)     │
  │ Layer 18 → CHECKPOINT (16.8 MB)     │
  │ Layer 24 → CHECKPOINT (16.8 MB)     │
  │ Layer 30 → CHECKPOINT (16.8 MB)     │
  │ Layer 36 → CHECKPOINT (16.8 MB)     │
  │ Total: 100 MB STORED IN VRAM        │ ✅
  │ (Recompute others during backward)  │
  └─────────────────────────────────────┘

  During backward pass, PyTorch recomputes the discarded activations on-the-fly, which:
  - Saves ~2 GiB of VRAM
  - Costs ~25% more computation time
  - But allows 2x bigger batch size = net speedup!

## Sequence Packing: Why Training Sample Count Appears Reduced

### What is Sequence Packing?

When `packing: true` is enabled in your config, TRL's `SFTTrainer` concatenates multiple short sequences together into single training examples to maximize GPU utilization. This is particularly efficient for datasets with variable-length sequences.

### Why Does Sample Count Change?

**Before Packing:**
- Original dataset: 78,271 individual samples
- Each sample is a separate training example
- Many samples are shorter than `max_length=1024`, wasting compute on padding

**After Packing:**
- Multiple samples are packed into each training example up to `max_length=1024` tokens
- Result: ~7,344 packed training examples
- **Average packing ratio**: 78,271 ÷ 7,344 ≈ 10.7 samples per packed example

### Is This Normal?

Yes, this is completely normal and expected! The reduced number you see is **not** a data loss—it's an efficiency optimization.

**Benefits of Packing:**
- ✅ **Better GPU utilization**: Minimizes padding waste
- ✅ **Faster training**: Fewer forward/backward passes
- ✅ **Same data coverage**: All 78,271 samples are still being trained on
- ✅ **More efficient learning**: The model sees more tokens per training step

**Example:**
```
Without packing (78,271 examples):
  Sample 1: [350 tokens] + [674 padding tokens] = 1024
  Sample 2: [420 tokens] + [604 padding tokens] = 1024
  → Total: 770 tokens trained, 1278 tokens wasted

With packing (7,344 examples):
  Packed 1: [350 + 420 + 254 tokens] = 1024
  → Total: 1024 tokens trained, 0 tokens wasted
```

### Configuration

Packing is controlled in the config YAML:
```yaml
# Enable sequence packing
packing: true  # default: false
cutoff_len: 1024  # Max sequence length for packing
```

**When to use packing:**
- ✅ Variable-length sequences (like our MovieLens data: 393-875 tokens)
- ✅ Mean sequence length << max_length (our data: mean=435, max_length=1024)
- ✅ Want to maximize GPU efficiency

**When NOT to use packing:**
- ❌ Sequences are already near max_length
- ❌ Need to preserve exact sample boundaries for specific evaluation
- ❌ Debugging individual samples

### Implementation Details (TRL 0.24.0+)

The packing parameters are passed via `SFTConfig` (not `TrainingArguments`):

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    max_length=1024,          # Max sequence length
    packing=True,              # Enable packing
    # ... other training args
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=sft_config,           # Pass SFTConfig as args
    formatting_func=formatting_func,  # Required for packing
)
```

**Note**: In TRL 0.24.0+, the `packing` parameter was moved from `SFTTrainer.__init__()` to `SFTConfig`. Using the old API will result in:
```
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'packing'
```

## TensorBoard Troubleshooting: "No Dashboards Active" Error

### Problem
When opening TensorBoard, you may see the error:
```
No dashboards are active for the current data set.

Probable causes:
  - You haven't written any data to your event files.
  - TensorBoard can't find your event files.
```

### Root Cause
This error is **almost always a TensorBoard cache issue**, not a logging problem. The training code logs correctly to TensorBoard event files, but:
1. TensorBoard cached an earlier state when the log directory was empty
2. Browser cached the empty dashboard view
3. Port conflict with existing TensorBoard instance

**Important**: Enabling packing (`packing: true`) does NOT break TensorBoard logging. Both packing and non-packing modes use the same HuggingFace Trainer logging infrastructure.

### How to Fix

**Step 1: Verify logs are actually being written**
```bash
# Check that event files exist and have data
ls -lh /workspace/output/qwen3-7b-movielens-qlora-special-token/logs/

# Should show files like:
# events.out.tfevents.1765756160.d2ff652b0591.2363578.0  (17K)
```

If files exist with size >10KB, your logging is working fine and this is just a cache issue.

**Step 2: Kill existing TensorBoard processes**
```bash
pkill -9 -f tensorboard
```

**Step 3: Clear TensorBoard cache**
```bash
rm -rf /tmp/.tensorboard-info/ ~/.tensorboard/
```

**Step 4: Restart TensorBoard (use port 6007 if 6006 is occupied)**
```bash
# Start TensorBoard in background
nohup tensorboard --logdir=/workspace/output/qwen3-7b-movielens-qlora-special-token --port=6007 --bind_all --reload_interval=5 > /tmp/tensorboard.log 2>&1 &

# Verify it started
ps aux | grep "tensorboard.*6007" | grep -v grep
```

**Step 5: Access TensorBoard and hard refresh**
1. Open `http://localhost:6007/` in your browser
2. Do a hard refresh: **Ctrl+F5** (Windows/Linux) or **Cmd+Shift+R** (Mac)
3. If still showing empty, try an incognito/private window

### Verification

After restarting, verify TensorBoard can see your metrics:
```bash
# Query the API to check available metrics
curl -s "http://localhost:6007/data/plugin/scalars/tags" | python3 -m json.tool | grep -E 'train|eval'
```

You should see metrics like:
- `train/loss`, `train/grad_norm`, `train/learning_rate`
- `eval/loss`, `eval/accuracy`, `eval/f1`
- And many more (typically 20+ metrics)

### Alternative: Point to Parent Directory

Instead of pointing to the `logs/` subdirectory, point to the parent:
```bash
tensorboard --logdir=/workspace/output/qwen3-7b-movielens-qlora-special-token --port=6007
```

TensorBoard will automatically find the `logs/` subdirectory.

### Common Mistakes

**❌ Wrong:** Assuming packing broke logging
- Packing only affects training step count, not TensorBoard logging

**❌ Wrong:** Pointing to a non-existent directory
- Double-check the path matches your `output_dir` in the config

**✅ Correct:** Clear cache and restart TensorBoard when you see "No dashboards active"
