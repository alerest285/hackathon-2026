#!/usr/bin/env bash
# One command on the Mac at the arm. It:
#   1. checks ssh to the GPU pod,
#   2. starts the SO-101 policy server there if it is not running (tmux "so101-serve", pre-warmed with presets.txt),
#   3. opens an ssh tunnel  localhost:18000 -> pod:8000,
#   4. installs robot/'s environment on the first run (uv),
#   5. starts the console and opens http://127.0.0.1:8765 in your browser.
#
#   console/start.sh --port /dev/tty.usbmodemXXXX --scene 0 --wrist 1            dry run on the real arm (safe)
#   console/start.sh --port /dev/tty.usbmodemXXXX --scene 0 --wrist 1 --live     the arm moves
#   console/start.sh --fake-robot --fake-cameras                                 no hardware, real server
# Ctrl-C stops the console and the tunnel. The arm holds its last command; the server keeps running on the pod.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The pod address is NOT in the repo (public). Put it in ~/hackathon-data/pod.env (never committed):
#   POD_HOST=<ip>   POD_SSH_PORT=<port>      (ask the team chat; it changes when the pod restarts)
[ -f "${HACK_POD_ENV:-$HOME/hackathon-data/pod.env}" ] && . "${HACK_POD_ENV:-$HOME/hackathon-data/pod.env}"
POD_HOST="${POD_HOST:-}"; POD_SSH_PORT="${POD_SSH_PORT:-}"; LOCAL_PORT="${LOCAL_PORT:-18000}"
[ -n "$POD_HOST" ] && [ -n "$POD_SSH_PORT" ] || { echo "error: set POD_HOST and POD_SSH_PORT in ~/hackathon-data/pod.env (ask the team chat)" >&2; exit 1; }
ROBOT_DIR="${HACK_ROBOT_DIR:-$ROOT/robot}"; export HACK_ROBOT_DIR="$ROBOT_DIR"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p "$POD_SSH_PORT" "root@$POD_HOST")
die() { echo "error: $*" >&2; exit 1; }
say() { echo "[start] $*"; }

[ -d "$ROBOT_DIR/so101_client" ] || die "robot/ client not found at $ROBOT_DIR. Run 'git pull' (it lands with T06)."
command -v uv >/dev/null || die "uv missing: curl -LsSf https://astral.sh/uv/install.sh | sh"

# Disk: report only, never delete here. Old recordings may hold their only camera data as per-frame PNGs.
data="${HACK_DATA:-$HOME/hackathon-data}"
if [ -d "$data" ]; then
  big=$(find "$data" -type d -name images -prune -exec du -sm {} + 2>/dev/null | sort -rn | head -5)
  [ -z "$big" ] || { say "disk: per-frame image folders under $data (MB, path). Review, then remove explicitly with console/cleanup.sh:"; echo "$big" | sed 's/^/          /'; }
fi
rec_root="$data"; prev=""; for arg in "$@"; do [ "$prev" = "--record-root" ] && rec_root="$arg"; prev="$arg"; done
probe="$rec_root"; while [ ! -d "$probe" ]; do probe="$(dirname "$probe")"; done   # never create it: LeRobot needs it absent
free_mb=$(df -Pm "$probe" | awk 'NR==2 {print $4}')   # the disk the recording goes to, not $HOME
say "disk: ${free_mb} MB free"
case " $* " in *" --record "*) [ "$free_mb" -ge 5000 ] || die "only ${free_mb} MB free: not recording. Free space (console/README.md 'Disk') or drop --record." ;; esac

say "1/5 ssh to pod $POD_HOST:$POD_SSH_PORT"
"${SSH[@]}" true 2>/dev/null || die "cannot ssh to the pod. Is your public key on your GitHub account and installed on the pod?
  Did the pod restart (new port)? Check the Runpod console, then: POD_SSH_PORT=<port> console/start.sh ..."

say "2/5 policy server on the pod"
warm=""; while IFS= read -r l; do case "$l" in ""|\#*) ;; *) warm="$warm --warmup-task \\\"$l\\\"" ;; esac; done < "$ROOT/console/presets.txt"
"${SSH[@]}" "set -e
  cd /workspace/hackathon-2026 && git pull -q || true
  d=/workspace/hackathon-2026; [ -f \$d/infra/gpu/so101/server.py ] || d=/workspace/so101-server
  if tmux has-session -t so101-serve 2>/dev/null; then echo '  already running'; else
    tmux new-session -d -s so101-serve \"cd \$d && FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache bash infra/gpu/so101/serve.sh --log-dir /workspace/so101-logs $warm 2>&1 | tee -a /workspace/so101-serve.log\"
    echo '  starting (loads the model + warm-up, ~2 min)'; fi
  for i in \$(seq 72); do ss -ltn | grep -q '127.0.0.1:8000' && { echo '  listening'; exit 0; }; sleep 5; done
  echo '  server did not come up; last log lines:'; tail -15 /workspace/so101-serve.log; exit 1"

say "3/5 tunnel localhost:$LOCAL_PORT -> pod:8000"
if lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  say "  port $LOCAL_PORT already forwarded, reusing it"; tunnel=""
else
  "${SSH[@]}" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L "$LOCAL_PORT:localhost:8000" &
  tunnel=$!; sleep 2; kill -0 "$tunnel" 2>/dev/null || die "tunnel failed"
fi
trap '[ -n "${tunnel:-}" ] && kill "$tunnel" 2>/dev/null; true' EXIT

say "4/5 robot environment (first run installs LeRobot 0.6.1, a few minutes)"
(cd "$ROBOT_DIR" && uv sync -q --python 3.13)

say "5/5 console on http://127.0.0.1:8765"
[ -n "${HACK_NO_BROWSER:-}" ] || ( sleep 5; command -v open >/dev/null && open "http://127.0.0.1:8765" ) &
# lerobot[dataset] adds the LeRobotDataset writer for recording, without changing robot/'s own environment.
cd "$ROBOT_DIR" && uv run -q --python 3.13 --with "lerobot[dataset]==0.6.1" python "$ROOT/console/app.py" --server "ws://localhost:$LOCAL_PORT" "$@"
