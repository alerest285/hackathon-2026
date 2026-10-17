# Saad's first live zero-shot run: SO-100, T21 joint map, 2026-10-17 15:16Z

saadl27/claude-code, for alerest285's failure log (T24). Facts come from the policy server's log, the console's
unsaved recording and the files on Saad's laptop. The last section lists what is not established.

**Result: failed.** The arm moved for 32 s under 48 plans, never went toward the cube, and the operator stopped it.
The episode was never saved. The arm followed its commands closely, so the failure lies in the plans, not the execution.

## Setup

| | |
|---|---|
| Code | `5506604` (HEAD of Saad's clone at 15:16Z). It has `3fb5132` NativeGuard (25 deg tracking, 15 deg jump, 3 write failures), but the console passed it `limits=None` until `8e303ef` (15:22Z). The loop clamp applied `limits.json`. Old PNG recorder |
| Command | as given to Saad at 15:14Z (Saad to confirm): `console/start.sh --port /dev/ttyACM1 --scene 2 --wrist 4 --robot-id my_follower_arm --limits ~/hackathon-data/limits.json --step-max 3 --live` |
| Instruction | `put the blue cube in the box` (the target is a round white tray with a green rim) |
| Joint map | `console/fit_reference.py` (T21), readings from 17:07 local: signs `[+1, -1, +1, +1, -1]`, offsets `[0.17, 80.23, 74.02, 86.84, -93.21]`, gripper identity. Pan and wrist_flex hand-checked; experimental |
| Limits (our deg) | pan [-117.4, 76.8], lift [-100.9, 26.7], elbow [-51.4, 95.6], wrist_flex [-141.8, 25.1], wrist_roll [-175.0, 72.0] |
| Server | so101-serve eager (before `--compile`), Euler 4, guidance 3/3. First plan 6326 ms (new caption), then p50 513 ms, p95 568 ms. Requests every 18 ticks p50 (0.61 s) |
| Scene camera | USB3 "Global Shutter Camera", index 2. Top-down, high angle over a white desk; the arm sits at the far edge facing the camera. 640x480 MJPG, 16-30 fps (auto exposure, indoor light) |
| Wrist camera | USB2.0_CAM1 (Sonix), index 4, 640x480 MJPG 30 fps, sees both gripper fingers. Focus set by hand |
| Sent to the model | both cameras as 256x256 JPEG q95 (`frame_hw [256, 256]` in the server log) |

## What the arm did

Measured state from the server log, in our degrees (LeRobot 0.6.1). Time is from the first plan; the last block is
the final plan's end minus the measured state, i.e. where the plan wanted to go.

| t (s) | tick | pan | lift | elbow | wrist_flex | wrist_roll | gripper | plan end - state: pan, lift, elbow, wflex, wroll, grip |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 9 | -5.9 | -102.3 | 97.7 | -9.0 | -5.4 | 1.3 | -1.5, +0.2, +0.7, -16.6, -16.8, +33.1 |
| 3.3 | 281 | 3.0 | -99.8 | 94.9 | -40.9 | -41.5 | 48.0 | +33.4, -14.5, -5.4, -48.5, -13.8, +5.5 |
| 6.5 | 367 | 9.4 | -98.2 | 90.2 | -71.4 | -28.0 | 53.1 | -9.5, -9.6, +10.4, -28.8, -1.0, +1.7 |
| 9.4 | 454 | -40.4 | -98.6 | 95.9 | -105.4 | -41.9 | 58.6 | +34.1, +61.2, -47.4, +64.0, +4.6, +13.1 |
| 12.0 | 537 | 4.8 | -100.1 | 95.4 | -42.7 | -53.9 | 62.1 | -5.1, -15.0, +18.5, -9.6, -0.7, +3.6 |
| 14.3 | 616 | 4.5 | -100.6 | 64.4 | -16.3 | -38.8 | 58.9 | +24.5, -40.4, +32.7, -55.2, +3.2, +0.2 |
| 17.0 | 686 | 8.2 | -94.5 | 95.9 | -120.7 | -45.6 | 63.0 | +1.9, -10.0, +8.9, -6.4, -10.8, +7.4 |
| 19.5 | 772 | -16.4 | -66.2 | 71.7 | -92.0 | -81.4 | 72.3 | -5.9, -0.2, -42.6, +112.4, -5.7, +21.0 |
| 22.0 | 837 | 9.8 | -58.7 | 41.2 | -39.0 | -62.5 | 93.6 | -10.9, -24.9, +39.5, -31.1, +5.7, +8.5 |
| 24.8 | 928 | -64.4 | -100.5 | 92.4 | -32.5 | -26.8 | 38.3 | -2.1, -14.1, +17.7, -19.2, -0.2, +11.4 |
| 27.6 | 1015 | -34.6 | -94.7 | 96.0 | -64.6 | 1.3 | 54.4 | +7.7, -11.9, +13.8, -14.3, +6.4, +0.9 |
| 30.0 | 1087 | -18.6 | -91.7 | 48.0 | -65.0 | -8.7 | 51.5 | -18.7, -10.8, +10.2, +15.4, -36.4, +8.6 |
| 31.8 | 1141 | -35.7 | -100.2 | 75.9 | -51.5 | -83.1 | 96.4 | +2.4, -8.0, +17.4, +28.0, +13.1, -1.1 |

- **Ranges (our deg):** pan -64..+53, lift -102..-36, elbow 31..98, wrist_flex -123..0, wrist_roll -83..+2, gripper 1..99.
- **Tracking:** |last command - measured| was p50 0.7 deg, p95 3.8 deg, max 13.8 deg (wrist_flex). The arm did what it was told.
- **Behavior, from the scene video:**
  - The gripper opened within about 3 s of the first plan.
  - The wrist flexed hard, down to -123.
  - The base swung left and right twice (-64 to +53).
  - The shoulder rose at most to -36 (66 deg off rest), around 19-22 s.
  - The arm stayed within about 15 cm of its base and never approached the cube, which sat about 25 cm in front on the
    robot's right, or the tray. No grasp.
- **Plans:** consecutive plans disagree strongly. At 9.4 s a plan's end moved lift +61 and wrist_flex +64; at 19.5 s,
  wrist_flex +112.
- **Not measured:** jerks at chunk seams (there is no tick log).
- **Before the plans:** in the first 6 s the arm held at rest while the first plan, with its new caption, was computed.
  The operator's hand adjusted the cube at about 2-5 s.

## Data

- `2026-10-17-saad-live-run-1.plans.jsonl` (next to this file): all 48 plans with t, tick, timing, state, last command
  and the 42x6 plan, in our degrees through the T21 map. Source: pod `/workspace/so101-logs/so101-20261017T140022Z.jsonl`,
  peer 127.0.0.1:36348.
- **Video**, on Saad's laptop only (not in git): `~/hackathon-data/so100_hanoi_flux_zeroshot/unsaved-episode-000000-{scene,wrist}.mp4`,
  real time at 60 fps. The old recorder saved two frames per 30 Hz tick: 2278 PNGs per camera in 37.9 s
  (15:16:03.8-15:16:41.7Z, file mtimes).
- **Not available:** a `robot/runs/<stamp>/` summary or tick log (the console does not write one), and the unsaved
  episode's numeric rows.
- **The 15:21:41Z session** (5 plans, `pick up the cube and place it in the bowl`) is not this arm. Its state never moved
  and reads lift +180 through this map. Probably a fake or test client.

## Not established (inputs for T24)

- **Joint conventions beyond the two hand checks.** The lift and elbow signs come from rest-pose geometry, the
  wrist_roll sign from issue #2. The offsets match the dataset's rest pose by construction.
- **Scene and wording.** The training data's cameras and kitchen differ from this desk, and the instruction said
  "box" for a round tray.
- **Latency.** The first plan took 6.3 s; the plan cycle of about 0.6 s exceeds the 533 ms stall-free budget.
  The server has since been compiled (about 370 ms per plan).
