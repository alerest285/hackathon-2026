#!/usr/bin/env bash
# Run ON the GPU host that trains, once. Safe to re-run.
#   bash train/setup.sh
# Builds the LeRobot env the serving engine already uses (infra/gpu/so101/bootstrap-lerobot.sh, same commit, same
# dir) and adds the training extras BFL's guide lists (training, flux3, peft, diffusion). Then fetches the policy
# and encoders at the decision-06 pins into the HF cache under /workspace, so an adapter trained here names a base
# path that is identical on any pod with the same HF_HOME, and the server loads it unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LEROBOT_DIR="${LEROBOT_DIR:-/workspace/lerobot-flux3}" HF_HOME="${HF_HOME:-/workspace/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv-cache}"
POLICY_REPO=black-forest-labs/flux-3-action-so101 POLICY_REV=c9e13b2aca6a0a472b3ea03fd90cff32d3e85849
BASE_REPO=black-forest-labs/flux-3-action-base BASE_REV=62878e2925e59b7a89ec14463ce89932624c490d

bash "$ROOT/infra/gpu/so101/bootstrap-lerobot.sh"
PY="$LEROBOT_DIR/.venv/bin/python"
NATTEN=$(grep -oE 'natten==[^"]+' "$ROOT/infra/gpu/so101/bootstrap-lerobot.sh" | head -1)
TORCH=$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')
cd "$LEROBOT_DIR"
# torch is already installed from the cu128 index; pin it so resolving the extras cannot swap it for a CPU build.
uv pip install --python "$PY" -e ".[training,flux3,peft,diffusion]" "torch==$TORCH"
uv pip install --python "$PY" "$NATTEN" -f https://whl.natten.org/   # again after every resolve (BFL setup.md)
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }

"$PY" - "$POLICY_REPO" "$POLICY_REV" "$BASE_REPO" "$BASE_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
pr, prev, br, brev = sys.argv[1:]
policy = snapshot_download(pr, revision=prev)
base = snapshot_download(br, revision=brev, allow_patterns=["video_vae.safetensors", "text_encoder/*"])
with open("/workspace/train-env.sh", "w") as f:
    f.write(f'export POLICY_DIR="{policy}"\nexport ENCODERS_DIR="{base}"\n')
print("policy", policy, "\nencoders", base)
PY
"$PY" -c 'import accelerate, peft, natten, torch, lerobot.scripts.lerobot_train; from lerobot.policies.flux3 import Flux3Policy; print("OK torch", torch.__version__, "cuda", torch.cuda.is_available(), "natten", natten.__version__)'
echo "OK: paths in /workspace/train-env.sh. Next: bash train/train.sh probe <dataset dir under /workspace/datasets>"
