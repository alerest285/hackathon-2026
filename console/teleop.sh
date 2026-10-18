#!/usr/bin/env bash
# Native LeRobot teleop and demo recording with the leader arm, in the same layout as the console's recordings
# (cameras "scene" and "wrist" at 640x480 @ 30 fps, observation.state / action named <joint>.pos).
# Stop the console first: only one program can own the follower's USB port.
#
#   console/teleop.sh move   --follower /dev/tty.usbmodemF --leader /dev/tty.usbmodemL     drive the arm by hand (resets)
#   console/teleop.sh record --follower ... --leader ... --task "pick up the small disk and place it on the middle peg" [--episodes 10]
#                            [--episode-s 15] [--reset-s 10] [--dataset NAME]
# An existing dataset is RESUMED (new episodes appended), so one prompt per session is fine: run it once per Hanoi step.
#
# Demos go to ~/hackathon-data/so100_hanoi_demos (repo_id alerest285/so100_hanoi_demos), separate from the zero-shot
# runs, and are what milestone 4 (LoRA) trains on. Calibrate both arms once first:
#   cd robot && uv run --python 3.13 lerobot-calibrate --robot.type=so100_follower --robot.port=<F> --robot.id=so101_hanoi   (same id as the console)
#   cd robot && uv run --python 3.13 lerobot-calibrate --teleop.type=so100_leader --teleop.port=<L> --teleop.id=so100_leader
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; ROBOT_DIR="${HACK_ROBOT_DIR:-$ROOT/robot}"
mode="${1:-}"; shift || true
follower="" leader="" task="" episodes=10 scene=0 wrist=1 episode_s=15 reset_s=10 dsname="so100_hanoi_demos" robot_id="${ROBOT_ID:-so101_hanoi}" leader_id="${LEADER_ID:-so100_leader}"
rtype="${ROBOT_TYPE:-so100_follower}"; ttype="${TELEOP_TYPE:-so100_leader}"
while [ $# -gt 0 ]; do case "$1" in
  --follower) follower="$2"; shift 2 ;; --leader) leader="$2"; shift 2 ;; --task) task="$2"; shift 2 ;;
  --episodes) episodes="$2"; shift 2 ;; --scene) scene="$2"; shift 2 ;; --wrist) wrist="$2"; shift 2 ;;
  --episode-s) episode_s="$2"; shift 2 ;; --reset-s) reset_s="$2"; shift 2 ;; --dataset) dsname="$2"; shift 2 ;;
  *) echo "unknown option $1" >&2; exit 2 ;; esac; done
[ -n "$follower" ] && [ -n "$leader" ] || { sed -n 2,13p "$0"; exit 2; }
# warmup_s 3: the Global Shutter scene camera needs ~1.7 s for its first frame on macOS (LeRobot's default 1 s times out).
warm="${CAM_WARMUP_S:-3}"
cams="{scene: {type: opencv, index_or_path: $scene, width: 640, height: 480, fps: 30, warmup_s: $warm}, wrist: {type: opencv, index_or_path: $wrist, width: 640, height: 480, fps: 30, warmup_s: $warm}}"
# use_degrees=true on both arms: joints in degrees, gripper 0..100, the units the model and the console use.
common=(--robot.type="$rtype" --robot.port="$follower" --robot.id="$robot_id" --robot.use_degrees=true --robot.cameras="$cams"
        --teleop.type="$ttype" --teleop.port="$leader" --teleop.id="$leader_id" --teleop.use_degrees=true)
cd "$ROBOT_DIR"
case "$mode" in
  move)   exec uv run --python 3.13 --with "lerobot[dataset]==0.6.1" lerobot-teleoperate "${common[@]}" --display_data=false ;;
  record) [ -n "$task" ] || { echo "record needs --task" >&2; exit 2; }
          root="$HOME/hackathon-data/$dsname"; resume=()
          # lerobot-record refuses to create into an existing root; append to it instead.
          [ -f "$root/meta/info.json" ] && { resume=(--resume=true); echo "resuming $root"; }
          exec uv run --python 3.13 --with "lerobot[dataset]==0.6.1" lerobot-record "${common[@]}" ${resume[@]+"${resume[@]}"} \
            --dataset.repo_id="alerest285/$dsname" --dataset.root="$root" \
            --dataset.episode_time_s="$episode_s" --dataset.reset_time_s="$reset_s" \
            --dataset.single_task="$task" --dataset.num_episodes="$episodes" --dataset.fps=30 \
            --dataset.push_to_hub=false --dataset.streaming_encoding=true --dataset.encoder_threads=2 --display_data=false ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
