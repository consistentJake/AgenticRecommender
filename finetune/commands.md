# Finetune Commands

Quick setup/run cheat sheet for the Qwen3 LoRA/QLoRA finetuning pipeline. Run all commands from the repo root.

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate


curl -fsSL https://claude.ai/install.sh | bash
pip install -r finetune/requirements_sft.txt
# (Optional) Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
  touch ~/.no_auto_tmux
  
## Compress

git archive --format=tar -o finetune.tar HEAD:finetune


## Data
```bash
# Prepare MovieLens data (downloads ml-latest-small)
cd finetune 
python scripts/prepare_movielens.py \
  --output-dir data/movielens_qwen3 \
  --history-len 15 \
  --rating-threshold 4.0 \
  --max-eval 1000

python scripts/prepare_movielens.py   --output-dir data/movielens_qwen3   --history-len 15   --rating-threshold 4.0   --max-eval 100
```
Dataset files should end up at `finetune/data/movielens_qwen3/train.json` and `eval.json`.

## Train
```bash
python scripts/finetune_lora.py --config configs/qwen3_7b_movielens_qlora.yaml
```
Outputs (adapter, tokenizer, logs) land in `output/qwen3-movielens-qlora/`.

## Monitor Learning Curves
```bash
tensorboard --logdir output/qwen3-movielens-qlora/logs --port 6006
# then open http://localhost:6006
```
## ssh key

in my linux 

