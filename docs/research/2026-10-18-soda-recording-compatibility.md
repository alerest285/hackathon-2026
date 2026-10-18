# Soda recording compatibility check — 2026-10-18

`codex-arm-review`, supporting Claude's recording setup on Ale's Mac. Checked
without opening serial ports or cameras, moving the arm, or running a GPU job.

## Requested merge check: PASS

Claude's command uses native LeRobot 0.6.1 through `console/teleop.sh record`:

- `ROBOT_ID=my_follower_arm_e82`, `LEADER_ID=my_leader_e82`.
- `so100_follower` / `so100_leader`, `use_degrees=true` on both.
- Scene camera 0, wrist camera 1 on this Mac; 640×480 RGB at 30 Hz.
- Dataset `so100_soda_ale`, episode time 20 seconds, reset time 10 seconds.
- Exact task: `pick the soda can and place it inside the box`.

The local follower calibration SHA-256 is
`4a9562e4ab96a2dd08af05b8eaf40181c66787996e6879223310c66425e7d2bb`,
matching Erfan's file. The leader snapshot is
`17726ab05a9c201f66435df65e3c93f9aaef4b06642b87a571a06ec536bdab31`.
Claude independently confirmed follower motor readback; this review read the
files only. Camera indices are session-specific on macOS.

Claude's `2bd633c` camera warmup fix was also verified against the installed
LeRobot parser and actual `connect()`/`async_read()` code with a fake capture,
thread and clock. A simulated 1.7-second first frame fails at the old one-second
default and succeeds at the new three-second default (and a five-second
`CAM_WARMUP_S` override). The setting extends the actual first-frame event wait;
dataset features remain unchanged. No camera was opened by the test.

Native recording features were generated through the installed LeRobot pipeline
and compared with the actual pod source metadata at
`/workspace/datasets/so100_simple_pick_and_place_scene/meta/info.json`.
`codebase_version`, `robot_type`, `fps`, and the entire `features` dictionary
match, including video metadata:

- Dataset v3.0, robot type `so_follower`, 30 fps.
- `observation.images.scene` and `.wrist`: shape `[480, 640, 3]`, AV1 video.
- `observation.state` and `action`: float32 `[6]`, ordered
  `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`,
  `wrist_roll.pos`, `gripper.pos`.
- Body joints use degrees; gripper uses LeRobot's 0–100 scale.

An offline roundtrip wrote two synthetic native datasets and merged them through
the real `train/prepare.py --teleop` path with an identity map. Result: two
episodes, 115 frames, one exact soda caption, unchanged actions, and both videos
decode. This checks format and preparation behavior, not demonstration quality.

Merge the **raw** renamed Erfan source and the finalized new native recording.
Do not feed the already prepared `lora_so100_simple_pick_and_place_hfsim_v1`
back into preparation. Use a stamped identity map for Erfan's calibration and
the HF-sim-normalized base; do not apply the earlier qmap. The old pod identity
map has a null calibration hash, so it does not document the verified identity
of these new recordings. Train into a new dataset/run name after adding data.

## Separate short-episode issue

Native `lerobot-record` does not use `console.Recorder` and therefore does not
inherit its ten-frame guard. A synthetic one-frame episode followed by a
50-frame episode reproduced `ArrowInvalid` during dataset finalization
(`stats/observation.images.scene/min` length mismatch). Normal 20-second
recordings passed. The console guard and preparation's null `episode_index`
handling both passed their separate checks. This native edge case was reported
to Claude; it is not a camera/schema incompatibility.

## Calibration helper review

`so100_save_motor_calibration.py` now uses the Feetech bus directly, cleans up a
partial connection without torque changes, and creates the destination
exclusively. This removes the high-level device destructor's torque-changing
error path and the check-then-overwrite race. Nine fake-bus tests cover both
namespaces, all six motor records, connection/read/cleanup failures, and existing
or concurrently created destination files. The real installed
`read_calibration()` was exercised with fake register reads and forbidden writes.

The related `so100_replay.py` check found that high-level `robot.connect()`
configured motors before checking their calibration. Replay now connects the
bus, checks calibration, then configures. Partial connection failures close
without changing torque. A failed observation after a successful command is
logged as a paired NaN row with `measured_valid=false`; cleanup and log-writing
errors cannot replace the original exception. Fourteen fake-device tests passed.

Run the helper checks from the repository with the LeRobot 0.6.1 environment:

```sh
python -m pytest docs/research/test_so100_save_motor_calibration.py docs/research/test_so100_replay.py -q
```

These are offline error-path checks. They do not execute a physical replay or
recalibrate either arm.
