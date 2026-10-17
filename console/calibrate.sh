#!/usr/bin/env bash
# Calibrate the follower arm (LeRobot's own interactive walkthrough). Stop the console first: only one program can
# own the arm's USB port.
#   console/calibrate.sh --port /dev/tty.usbmodemXXXX [--robot-id so101_hanoi]
# Follow the prompts: put the arm in the middle of its range and press ENTER, then move EVERY joint through its full
# range and press ENTER. The file lands in ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot-id>.json,
# which the console shows. Keep the same --robot-id as start.sh (default so101_hanoi).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
port="" id="so101_hanoi" type="${ROBOT_TYPE:-so100_follower}"
while [ $# -gt 0 ]; do case "$1" in --port) port="$2"; shift 2 ;; --robot-id) id="$2"; shift 2 ;; *) echo "unknown $1" >&2; exit 2 ;; esac; done
[ -n "$port" ] || { echo "usage: console/calibrate.sh --port /dev/tty.usbmodemXXXX   (find it: cd robot && uv run --python 3.13 lerobot-find-port)" >&2; exit 2; }
cd "$ROOT/robot" && uv sync -q --python 3.13
exec uv run --python 3.13 lerobot-calibrate --robot.type="$type" --robot.port="$port" --robot.id="$id"
