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
# The FLUX weights are gated (auto-approved license click). Check access before anyone waits on a download.
if ! uv run hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face login needed for the gated FLUX weights:"
  echo "  1. with your HF account, open and accept the license on"
  echo "     https://huggingface.co/black-forest-labs/flux-3-action-so101 (and -base, -droid)"
  echo "  2. create a read token at https://huggingface.co/settings/tokens"
  echo "  3. here: cd $DIR && uv run hf auth login"
fi
for r in flux-3-action-so101 flux-3-action-base; do
  if uv run python -c "from huggingface_hub import model_info; model_info('black-forest-labs/$r', files_metadata=False); from huggingface_hub import hf_hub_download as d; d('black-forest-labs/$r','config.json')" >/dev/null 2>&1
  then echo "weights access: $r ok"; else echo "weights access: $r NOT YET (accept the license + hf auth login)"; fi
done
echo "OK: flux-action @ ${FLUX_COMMIT:0:7} in $DIR."
