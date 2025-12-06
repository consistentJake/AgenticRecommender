# Finetune Commands

Quick setup/run cheat sheet for the Qwen3 LoRA/QLoRA finetuning pipeline. Run all commands from the repo root.

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r finetune/requirements_sft.txt

# (Optional) Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

## Data
```bash
# Prepare MovieLens data (downloads ml-latest-small)
python finetune/scripts/prepare_movielens.py \
  --output-dir finetune/data/movielens_qwen3 \
  --history-len 15 \
  --rating-threshold 4.0
  --max-eval 1000
```
Dataset files should end up at `finetune/data/movielens_qwen3/train.json` and `eval.json`.

## Train
```bash
python finetune/scripts/finetune_lora.py \
  --config finetune/configs/qwen3_movielens_qlora.yaml
```
Outputs (adapter, tokenizer, logs) land in `output/qwen3-movielens-qlora/`.

## Monitor Learning Curves
```bash
tensorboard --logdir output/qwen3-movielens-qlora/logs --port 6006
# then open http://localhost:6006
```
