# console/: drive the arm (SO-100 / SO-101) with FLUX from a browser page

One command on the Mac that has the arm and both cameras plugged in:

```sh
console/start.sh --port /dev/tty.usbmodemXXXX --scene 0 --wrist 1            # dry run: reads the arm, moves nothing
console/start.sh --port /dev/tty.usbmodemXXXX --scene 0 --wrist 1 --live     # the arm moves
```

It checks ssh to the GPU pod, starts the policy server there if needed, opens a tunnel (localhost:18000 → pod:8000),
installs `robot/`'s environment on first use, and opens **http://127.0.0.1:8765**. On the page:

| Control | Does |
|---|---|
| instruction box + **Run (new episode)** | sends the instruction; the history is reset to the current observation |
| **Next instruction** | the next Hanoi step: drops queued actions and keeps the history (decision 06) |
| **Pause / hold** | the arm holds its position, no new requests |
| preset buttons | from `presets.txt`, which also pre-warms the server so the first plan is fast |
| camera previews | scene (fixed) must look at the workspace; wrist must be on the gripper. If they are swapped, swap `--scene`/`--wrist` |
| joint table | measured, commanded, and the **range the checkpoint was trained on**. Red rows are outside it |

## Recording (native LeRobot format)

Every run is recorded as a **LeRobotDataset** episode, the same layout `lerobot-record` writes:
`observation.images.scene` / `observation.images.wrist` (480×640 video), `observation.state` and `action` (6 × `<joint>.pos`,
action = the command sent, or that a dry run would send), the instruction as the task, 30 fps, robot type `so100_follower`
(`--robot-type so101_follower` for an SO-101; LeRobot drives both with the same class).

- **Run (new episode)** starts an episode, and **Next instruction** continues the same one (multi-step Hanoi = one episode).
- **Save: success / Save: fail / Discard** closes it. The verdict goes to `console_episodes.jsonl` next to the dataset.
- Ctrl-C saves the open episode as `unlabeled` and finalizes the dataset (tested: it loads with `LeRobotDataset`).
- Location: `~/hackathon-data/so100_hanoi_flux_zeroshot` (repo_id `alerest285/so100_hanoi_flux_zeroshot`, local only).
  Push when the team agrees: `huggingface-cli upload ... --repo-type dataset`. `--no-record` turns recording off.

## Leader arm: resets and demonstrations (native LeRobot)

Stop the console first (only one program can own the follower's port), then:

```sh
console/teleop.sh move   --follower /dev/tty.usbmodemF --leader /dev/tty.usbmodemL --scene 0 --wrist 1     # move it by hand
console/teleop.sh record --follower ... --leader ... --task "pick up the small disk and place it on the middle peg" --episodes 10
```

This is plain `lerobot-teleoperate` / `lerobot-record` with `so100_follower` + `so100_leader` and the **same camera names**, so
demos (`~/hackathon-data/so100_hanoi_demos`) and zero-shot runs share one format for milestone 4 (LoRA).
Override the types with `ROBOT_TYPE=so101_follower TELEOP_TYPE=so101_leader`.

Ctrl-C in the terminal stops the console and the tunnel. The arm keeps its last command (torque stays on).
**The physical power switch is the real emergency stop.** Keep a hand on it during `--live` runs.

## Before the first live run (once per arm)

Follow `robot/README.md` steps 1–4 (from the T06 agent):

1. `cd robot && uv run --python 3.13 lerobot-find-port`, then find the cameras: `uv run --python 3.13 python camera_check.py`.
2. Calibrate the follower (and the leader for teleop): `uv run --python 3.13 lerobot-calibrate --robot.type=so100_follower --robot.port=<port> --robot.id=so101_hanoi`
   (the id is just the calibration file name; keep `so101_hanoi` so the console and teleop share it).
3. **Check the calibration against the model's training range.** Run the console in dry run, put the arm in its rest pose,
   and read the joint table. The checkpoint was trained with shoulder_lift ≈ 85…184° and elbow_flex ≈ 61…179°
   (server `hello`, `state_q01`/`state_q99`). If your values are far outside (red rows), the calibration convention differs
   from BFL's, and zero-shot will not work until it matches. Tell the team in chat; do not go live.
4. Optional joint limits: `uv run --python 3.13 python readout.py --save-limits limits.json --margin 3`, then
   add `--limits robot/limits.json` to `start.sh`.

## No hardware

`console/start.sh --fake-robot --fake-cameras` runs the whole path with a simulated arm and test images against the real
server. Tested 17.10: 690 ticks at 30 Hz, 36 chunks accepted, 0 errors, turnaround ≈ 560–590 ms (server ≈ 510 ms).

## If something fails

| Symptom | Fix |
|---|---|
| `cannot ssh to the pod` | your key is not on the pod, or the pod restarted with a new port: `POD_SSH_PORT=<new> console/start.sh ...` |
| `robot/ client not found` | `git pull`; `robot/` comes from task T06 |
| server did not come up | read the printed log lines; ask in chat (`scripts/say.sh`) |
| the arm holds briefly now and then | turnaround is just above the 533 ms stall-free budget; compiling the server (T10) fixes it |
| page says "console not reachable" | the terminal running `start.sh` stopped; restart it |

Settings: `POD_HOST`, `POD_SSH_PORT`, `LOCAL_PORT` (default 18000; local 8000 is often taken by other tools).
