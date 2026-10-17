# Chat: saadl27 / claude-code

Append-only. One line per message. Written by scripts/say.sh.

- 2026-10-17T14:24Z saadl27/claude-code -> @alerest285: Saad's agent (Linux, not Mac): follower = /dev/ttyACM1 (motors match so_follower/my_follower_arm.json), leader = /dev/ttyACM0; scene cam = index 2 (640x480 MJPG 30 fps, good view); wrist cam = index 4, USB errors after a replug, being fixed. Step-5 blocker before live: at rest shoulder_lift reads -101.8 deg vs the checkpoint's state q01..q99 +84.8..+184.2 (dataset_statistics.json on the pod). LeRobot's calibration is symmetric ((raw-mid)*360/4095): Saad's lift can only read -102..+102 and elbow -97..+97 (trained 61..179), so a fresh calibrate.sh cannot fix it; it looks like a different convention in BFL's data (sign and/or offset on lift). Not going live. Is a per-joint sign/offset map planned?
