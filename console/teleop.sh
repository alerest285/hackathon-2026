#!/usr/bin/env bash
# Native LeRobot teleop and demo recording with the leader arm, in the same layout as the console's recordings
# (cameras "scene" and "wrist" at 640x480 @ 30 fps, observation.state / action named <joint>.pos).
# Stop the console first: only one program can own the follower's USB port.
#
#   console/teleop.sh move   --follower /dev/tty.usbmodemF --leader /dev/tty.usbmodemL     drive the arm by hand (resets)
#   console/teleop.sh record --follower ... --leader ... --task "pick up the small disk and place it on the middle peg" [--episodes 10]
#
# Demos go to ~/hackathon-data/so100_hanoi_demos (repo_id alerest285/so100_hanoi_demos), separate from the zero-shot
# runs, and are what milestone 4 (LoRA) trains on. Calibrate both arms once first:
#   cd robot && uv run --python 3.13 lerobot-calibrate --robot.type=so100_follower --robot.port=<F> --robot.id=so101_hanoi   (same id as the console)
#   cd robot && uv run --python 3.13 lerobot-calibrate --teleop.type=so100_leader --teleop.port=<L> --teleop.id=so100_leader
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; ROBOT_DIR="${HACK_ROBOT_DIR:-$ROOT/robot}"
mode="${1:-}"; shift || true
follower="" leader="" task="" episodes=10 scene=0 wrist=1 robot_id="${ROBOT_ID:-so101_hanoi}" leader_id="${LEADER_ID:-so100_leader}"
rtype="${ROBOT_TYPE:-so100_follower}"; ttype="${TELEOP_TYPE:-so100_leader}"
while [ $# -gt 0 ]; do case "$1" in
  --follower) follower="$2"; shift 2 ;; --leader) leader="$2"; shift 2 ;; --task) task="$2"; shift 2 ;;
  --episodes) episodes="$2"; shift 2 ;; --scene) scene="$2"; shift 2 ;; --wrist) wrist="$2"; shift 2 ;;
  *) echo "unknown option $1" >&2; exit 2 ;; esac; done
[ -n "$follower" ] && [ -n "$leader" ] || { sed -n 2,13p "$0"; exit 2; }
cams="{scene: {type: opencv, index_or_path: $scene, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $wrist, width: 640, height: 480, fps: 30}}"
common=(--robot.type="$rtype" --robot.port="$follower" --robot.id="$robot_id" --robot.cameras="$cams"
        --teleop.type="$ttype" --teleop.port="$leader" --teleop.id="$leader_id")
cd "$ROBOT_DIR"
case "$mode" in
  move)   exec uv run --python 3.13 --with "lerobot[dataset]==0.6.1" lerobot-teleoperate "${common[@]}" --display_data=false ;;
  record) [ -n "$task" ] || { echo "record needs --task" >&2; exit 2; }
          exec uv run --python 3.13 --with "lerobot[dataset]==0.6.1" lerobot-record "${common[@]}" \
            --dataset.repo_id=alerest285/so100_hanoi_demos --dataset.root="$HOME/hackathon-data/so100_hanoi_demos" \
            --dataset.single_task="$task" --dataset.num_episodes="$episodes" --dataset.fps=30 \
            --dataset.push_to_hub=false --dataset.streaming_encoding=true --dataset.encoder_threads=2 --display_data=false ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
