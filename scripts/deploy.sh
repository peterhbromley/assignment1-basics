#!/usr/bin/env bash
# Run locally: bash scripts/deploy.sh ubuntu@INSTANCE_IP
# Uses your SSH config/agent for authentication.
set -euo pipefail

if [[ $# -ne 1 || "$1" == -* ]]; then
    echo "Usage: bash scripts/deploy.sh user@host (or an SSH config alias)" >&2
    exit 1
fi

host="$1"
cd "$(dirname "$0")/.."

# An explicit file list keeps local environments, raw data, and runs off the upload.
files=(llm scripts pyproject.toml uv.lock README.md
    artifacts/tinystories_bpe.pkl
    data/tokenized/tinystories-bpe-10k/train.npy
    data/tokenized/tinystories-bpe-10k/valid.npy)
for file in "${files[@]}"; do
    [[ -e "$file" ]] || { echo "Missing local file: $file" >&2; exit 1; }
done

rsync -avR --progress --exclude='__pycache__/' "${files[@]}" "$host:assignment1-basics/"

ssh "$host" 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd "$HOME/assignment1-basics"
uv sync --locked
uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA ready:", torch.cuda.get_device_name(0))'
REMOTE

echo "Ready in ~/assignment1-basics on $host. For W&B, run: uv run wandb login"
