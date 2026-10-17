#!/usr/bin/env bash
# Run ON the GPU host, once, for the lerobot engine of the SO-101 server (the A/B option for compile and
# the LoRA path). Builds ${LEROBOT_DIR:-~/lerobot-flux3}: LeRobot main at the commit pinned below, a
# Python 3.12 venv, torch 2.11.0 + torchvision 0.26.0 for CUDA 12.8, the flux3 and peft extras of that
# commit's pyproject, and NATTEN for the video VAE. Safe to re-run.
#   bash infra/gpu/so101/bootstrap-lerobot.sh
set -euo pipefail
LEROBOT_COMMIT="${LEROBOT_COMMIT:-146d8474932c8b566717eb2738ba459337ab71c4}"   # main, verified 2026-10-17
TORCH=("torch==2.11.0" "torchvision==0.26.0")
NATTEN="natten==0.21.6+torch2110cu128"                                          # must match torch 2.11.0 + cu128
DIR="${LEROBOT_DIR:-$HOME/lerobot-flux3}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

[ -d "$DIR/.git" ] || git clone https://github.com/huggingface/lerobot.git "$DIR"
cd "$DIR"; git fetch -q origin; git checkout -q "$LEROBOT_COMMIT"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
PY="$DIR/.venv/bin/python"
# torch first, from the CUDA 12.8 index. lerobot requires torch>=2.7,<2.12, so the next step keeps it.
uv pip install --python "$PY" "${TORCH[@]}" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -e ".[flux3,peft]"
# NATTEN is not on PyPI for this torch/CUDA pair; reinstall it after anything that resolves the env.
uv pip install --python "$PY" "$NATTEN" -f https://whl.natten.org/
"$PY" -c 'import websockets, msgpack' 2>/dev/null || uv pip install --python "$PY" "websockets==15.0.1" "msgpack==1.2.2"
"$PY" - <<'PY'
from importlib.metadata import version

import natten
import torch
from lerobot.policies.flux3 import Flux3Policy  # noqa: F401  (fails here if an extra is missing)

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "natten", natten.__version__, "lerobot", version("lerobot"), "peft", version("peft"))
assert (torch.version.cuda or "").startswith("12.8"), "torch is not the CUDA 12.8 build the NATTEN wheel needs"
PY
echo "OK: lerobot @ ${LEROBOT_COMMIT:0:7} in $DIR. Next: bash infra/gpu/so101/serve.sh --engine lerobot --selftest"
