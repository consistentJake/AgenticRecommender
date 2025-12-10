# Finetune Commands

Quick setup/run cheat sheet for the Qwen3 LoRA/QLoRA finetuning pipeline. Run all commands from the repo root.


cat ~/.ssh/id_ed25519.pub

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH5BccNvTJiR0S/VHcufq+T07A0MP90gJZDCef8Kvn3p zhenkaiwang@hotmail.com


git archive --format=tar HEAD:finetune -o finetune.gz


git archive --format=tar.gz -o finetune-code.tar.gz HEAD && ls -lh finetune-code.tar.gz


sudo apt-get update
sudo apt-get install unzip

touch ~/.no_auto_tmux

mkdir finetune
unzip finetune.zip -d finetune

### verify first 
curl -fsSL https://claude.ai/install.sh | bash

  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

### verify downlaoding teh model 
python -c "from transformers import AutoModel; 
  AutoModel.from_pretrained('Qwen/Qwen3-0.6B')"
  


  To disconnect without closing your processes, press ctrl+b, release, then d.
To disable auto-tmux, run `touch ~/.no_auto_tmux` and reconnect. See also https://tmuxcheatsheet.com/


## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r finetune/requirements_sft.txt

# (Optional) Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```


  HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/finetune_lora.py \
    --config configs/qwen3_movielens_qlora.yaml

## Data
```bash
# Prepare MovieLens data (downloads ml-latest-small)
python scripts/prepare_movielens.py \
  --output-dir finetune/data/movielens_qwen3 \
  --history-len 15 \
  --rating-threshold 4.0 \
  --max-eval 1000
```
Dataset files should end up at `finetune/data/movielens_qwen3/train.json` and `eval.json`.

## Train
```bash
python scripts/finetune_lora.py \
  --config configs/qwen3_movielens_qlora.yaml
```
Outputs (adapter, tokenizer, logs) land in `output/qwen3-movielens-qlora/`.

## Monitor Learning Curves
```bash
tensorboard --logdir output/qwen3-movielens-qlora/logs --port 6006
# then open http://localhost:6006
```
