#!/usr/bin/env bash
# Run ON the GPU host, once. Installs FLUX Action exactly as BFL's docs/setup.md says,
# at the commit pinned below, into ~/flux-action. Safe to re-run.
#   bash infra/gpu/bootstrap.sh
set -euo pipefail
FLUX_COMMIT="${FLUX_COMMIT:-830f1ce178d7aea52f18fde31d8e8e36ce6bae21}"   # main on 2026-10-02
NATTEN="natten==0.21.6+torch2100cu128"                                    # must match torch 2.10.0 + cu128
DIR="${FLUX_DIR:-$HOME/flux-action}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v ffmpeg >/dev/null || sudo apt-get install -y ffmpeg
ffmpeg -hide_banner -decoders 2>/dev/null | grep -qi av1 || echo "WARNING: ffmpeg has no AV1 decoder; the recorded DROID sample needs one."
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

[ -d "$DIR/.git" ] || git clone https://github.com/black-forest-labs/flux-action.git "$DIR"
cd "$DIR"; git fetch -q origin; git checkout -q "$FLUX_COMMIT"
uv sync --locked --extra encoders --extra data --extra serve
# NATTEN is outside uv.lock and is wiped by every uv sync: always reinstall right after.
uv pip install --python .venv/bin/python "$NATTEN" -f https://whl.natten.org/
uv run python -c 'import torch, natten; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "natten", natten.__version__)'
echo "OK: flux-action @ ${FLUX_COMMIT:0:7} in $DIR. If a weights repo asks for access: uv run hf auth login"
