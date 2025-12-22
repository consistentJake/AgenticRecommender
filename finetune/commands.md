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
python finetune/scripts/prepare_movielens.py \
  --output-dir finetune/data/movielens_qwen3 \
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


source /venv/py311-cu128/bin/activate

# Use the complete Qwen model from /root/.cache/huggingface
export HF_HOME=/root/.cache/huggingface
nohup python scripts/finetune_lora.py --config configs/qwen3_7b_delivery_hero_qlora.yaml > training.log 2>&1 &


## Monitor Learning Curves
```bash
tensorboard --logdir output/qwen3-movielens-qlora/logs --port 6006
# then open http://localhost:6006
```


https://drive.google.com/file/d/11OsafYu26ISaUfGEXzFSwNyKDJxKCy8e/view?usp=drive_link
gdown 11OsafYu26ISaUfGEXzFSwNyKDJxKCy8e
unzip data_se.zip


## fixing bistandbyptes 


pip uninstall -y bitsandbytes torch torchvision torchaudio

# install a cu128 torch build (nightly index has cu128 wheels)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# reinstall bitsandbytes
pip install -U bitsandbytes

# verify
python -m bitsandbytes


### run the testing
because we download the qwen3-8b model in advance

cd /workspace/AgenticRecommender/finetune
  source /venv/py311-cu128/bin/activate
  export HF_HOME=/root/.cache/huggingface
  nohup python scripts/finetune_lora.py --config configs/qwen3_7b_delivery_hero_qlora.yaml > training.log 2>&1 &

