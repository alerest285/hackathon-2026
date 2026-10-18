#!/usr/bin/env bash
# Recording dashboard: teleop data collection with buttons (no terminal keys). Opens http://127.0.0.1:8766.
#   console/record.sh --follower <port> --leader <port> --scene 0 --wrist 1 \
#       --task 'pick the soda can and place it inside the box' --dataset so100_soda_ale [--episode-s 20]
# Same calibration ids as teleop.sh: --robot-id my_follower_arm_e82 --leader-id my_leader_e82 (defaults).
# Ctrl-C (or the Stop button) finalizes the dataset; saved episodes are never lost.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/robot"
[ -n "${HACK_NO_BROWSER:-}" ] || ( sleep 6; command -v open >/dev/null && open "http://127.0.0.1:8766" ) &
exec uv run --python 3.13 --with "lerobot[dataset]==0.6.1" python "$ROOT/console/record_app.py" "$@"
