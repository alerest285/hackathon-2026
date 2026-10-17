#!/usr/bin/env bash
# Run ON the GPU host (inside tmux). Serves the DROID policy on :8000 over the OpenPI WebSocket
# protocol, with query logging. Reach it from your laptop via infra/aws/ssh.sh (forwards 8000).
#   bash infra/gpu/serve.sh [subfolder]     default variants/gd-fp8r (guidance-distilled FP8r)
set -euo pipefail
cd "${FLUX_DIR:-$HOME/flux-action}"
SUB="${1:-variants/gd-fp8r}"
args=(--checkpoint black-forest-labs/flux-3-action-droid --port 8000 --log-dir "outputs/robolab-logs/$(date -u +%Y%m%dT%H%M%SZ)")
[ "$SUB" = root ] || args+=(--subfolder "$SUB")
exec uv run --extra serve flux-action serve-robolab "${args[@]}"
