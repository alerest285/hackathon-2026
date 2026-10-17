#!/usr/bin/env bash
# Run ON the GPU host, inside tmux. Serves black-forest-labs/flux-3-action-so101 over so101.v1 on
# 127.0.0.1:8000 (docs/architecture.md). The Mac reaches it through infra/aws/ssh.sh, which forwards 8000.
#   bash infra/gpu/so101/serve.sh --selftest                 build, warm up, time 50 plans, PASS/FAIL
#   bash infra/gpu/so101/serve.sh                            serve: engine flux-action, port 8000
#   bash infra/gpu/so101/serve.sh --engine lerobot --compile [any other server.py flag]
# Interpreter: flux-action uses ${FLUX_DIR:-~/flux-action}/.venv (built by infra/gpu/bootstrap.sh),
# lerobot uses ${LEROBOT_DIR:-~/lerobot-flux3}/.venv (bootstrap-lerobot.sh), fake uses whichever exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUX_PY="${FLUX_DIR:-$HOME/flux-action}/.venv/bin/python"
LEROBOT_PY="${LEROBOT_DIR:-$HOME/lerobot-flux3}/.venv/bin/python"

engine=flux-action
prev=""
for arg in "$@"; do
  [ "$prev" = "--engine" ] && engine="$arg"
  case "$arg" in --engine=*) engine="${arg#--engine=}" ;; esac
  prev="$arg"
done
case "$engine" in
  flux-action) py="$FLUX_PY"; boot="infra/gpu/bootstrap.sh" ;;
  lerobot) py="$LEROBOT_PY"; boot="infra/gpu/so101/bootstrap-lerobot.sh" ;;
  fake) if [ -x "$FLUX_PY" ]; then py="$FLUX_PY"; else py="$LEROBOT_PY"; fi; boot="infra/gpu/bootstrap.sh" ;;
  *) echo "error: unknown engine '$engine' (flux-action | lerobot | fake)" >&2; exit 2 ;;
esac
[ -x "$py" ] || { echo "error: no Python at $py. Build it first: bash $boot" >&2; exit 1; }
echo "engine $engine, python $py"

# The server needs websockets + msgpack (and Pillow or OpenCV for jpeg/png). In the flux-action env they
# come with --extra serve, and every `uv sync` without it removes them again, like NATTEN, so this is
# checked on every start. Versions as in flux-action's uv.lock.
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
need=()
"$py" -c 'import websockets' 2>/dev/null || need+=("websockets==15.0.1")
"$py" -c 'import msgpack' 2>/dev/null || need+=("msgpack==1.2.2")
"$py" -c 'import PIL' 2>/dev/null || "$py" -c 'import cv2' 2>/dev/null || need+=("pillow")
if [ "${#need[@]}" -gt 0 ]; then
  echo "installing ${need[*]} into $py"
  uv pip install --python "$py" "${need[@]}"
fi

if [ "$engine" != fake ]; then
  "$py" -c 'import natten' 2>/dev/null \
    || echo "WARNING: natten is missing from $py (uv sync removes it). Re-run: bash $boot" >&2
  token_file="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ ! -s "$token_file" ]; then
    echo "WARNING: no Hugging Face token (HF_TOKEN or 'hf auth login'); the gated weights will not download." >&2
  fi
fi
exec "$py" "$HERE/server.py" "$@"
