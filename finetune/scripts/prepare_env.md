## install claude

curl -fsSL https://claude.ai/install.sh | bash

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

ssh-keygen -t ed25519 -C "zhenkaiwang@hotmail.com"

cd /home/workspace
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

source .venv/bin/activate
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name qwen3-movielens-qlora


## ssh key
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH5BccNvTJiR0S/VHcufq+T07A0MP90gJZDCef8Kvn3p zhenkaiwang@hotmail.com

passcode: nba