# robot/: the SO-101 client (T06)

The Mac at the robot station runs this. It reads the scene camera, the wrist camera and the SO-101 joint
state at 30 Hz, keeps the 8-tick window the model reads, fetches the next 42-action chunk from the
`so101.v1` server (`infra/gpu/so101/`, T10) before the current one runs out, and stitches the chunks in
joint-delta space so the arm never stalls or jumps. `eval/` (T07) drives it in the same process. The
contract is `docs/architecture.md`, section "SO-101 path", plus decisions 05 and 06.

Dry run is the default. Nothing is written to the motors without `--live`.

## Setup

```sh
cd robot
uv sync --python 3.13          # the Mac's python3 is 3.15, which has no torch wheels; LeRobot 0.6.1 needs torch
```

LeRobot 0.6.1 is installed in `robot/.venv`, so its tools run from here:

```sh
uv run --python 3.13 lerobot-find-port                 # unplug/replug the arm's USB: prints /dev/tty.usbmodem...
uv run --python 3.13 lerobot-find-cameras opencv       # or camera_check.py below
uv run --python 3.13 lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodemXXXX --robot.id=so101_hanoi
```

Calibration writes `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_hanoi.json` and the
motors' offsets. Use the same `--robot-id` for every command here. The client never calibrates by itself.
A missing file, or motors that disagree with it, is an error that prints the command to run. On macOS the
terminal needs camera permission: System Settings > Privacy & Security > Camera.

## Check order

Decision 06 sets this order. Each step must pass before the next:

| # | Command | Passes when |
|---|---|---|
| 1 | `uv run --python 3.13 python camera_check.py` then `... camera_check.py --scene 0 --wrist 1` | both cameras run at about 30 fps, and in `runs/camera-check-<stamp>/composed.png` the left half is the fixed scene camera and the right half is the wrist camera. A swapped pair cannot be fixed by renaming: swap the indices. |
| 2 | `../infra/aws/ssh.sh -N &` then `uv run --python 3.13 python ping.py --n 20 --infer 20` | `PASS`, with the round trip split into server and network time |
| 3 | `uv run --python 3.13 python run.py --task "..." --fake-robot --scene 0 --wrist 1` | `PASS` against the live server, with real frames and a fake arm |
| 4 | `uv run --python 3.13 python run.py --task "..." --port /dev/tty.usbmodemXXXX --robot-id so101_hanoi` | `PASS` in a dry run on the arm: the state is read, the commands are only logged |
| 5 | the same with `--live --limits limits.json` | one supervised episode, with someone at the arm |

`readout.py` helps before step 5. It prints the six joint values at 10 Hz. With `--torque-off` the arm can be
moved by hand, and `--save-limits limits.json --margin 3` writes joint limits for `run.py --limits`, which
`--live` requires. The
gripper column gives the open and closed values for eval's step-end thresholds (also the fully closed value
on air, eval's `empty_threshold`); which end is closed depends on the calibration.

Fakes stand in for any part: `--fake-robot`, `--fake-cameras`, `--fake-net [hold|sine|pickplace]`
(with `--latency-ms`, `--jitter-ms`). `uv run --python 3.13 python run.py --task "pick up the small disk and
place it on the right peg" --fake-robot --fake-cameras --fake-net` runs with no hardware and no server. With
`--fake-robot` the commands go to the fake arm, which follows them.

## PASS criteria (run.py)

| Check | Limit | Why |
|---|---|---|
| tick lateness p99 | ≤ 2 ms | the 30 Hz deadlines hold |
| holds after the first chunk | 0 (prefetch) | the arm never waits for the network; sync mode holds by design and is reported as n/a. Holds after failed state reads are counted apart, as `bus_holds` |
| turnaround p95 | ≤ 533 ms | 16 ticks keeps every executed action inside the 32 the checkpoint was meant to execute (decision 06) |
| chunks accepted | ≥ 1 | the pipe works |

Turnaround runs from the capture of the newest frame in the request to the arrival of the reply. It covers
frame age, upload, queue, compute and download.

Each run writes `runs/<stamp>/` (gitignored):

- `ticks.jsonl`: one line per tick with `k, t, state, cmd, held, written, lateness_ms, frame_age_ms, remaining, task, chunk_id`.
- `plans.jsonl`: one line per reply, with `id, tick, d, accepted, start, reason (ok|late|stale), turnaround_ms, server_ms, rtt_ms, actions`.
- `summary.json`: the arguments, the server hello, `ControlLoop.stats()` and the checks.

## Safety

- Dry run is the default. `--live` asks for ENTER before the loop starts (`--yes` skips the prompt).
- Ctrl-C stops the loop. The last command stays the motors' goal, so the arm holds, and the port closes with
  torque on. LeRobot 0.6.1 would turn torque off on disconnect (`disable_torque_on_disconnect=True`) and drop
  the arm; this client switches that off.
- Connecting runs LeRobot's `configure()`, which turns torque off for a moment while it writes the motor
  settings. Start with the arm resting. The goal register is pinned to the present position before and after,
  so turning torque back on moves nothing, even after `readout.py --torque-off` left an old pose there.
- Every executed command is clamped: joint limits first (`--limits`, lo < hi), then at most `--step-max` (6°)
  per tick per joint, and the gripper to 0..100. A joint that starts outside its limits is never pulled in
  faster than the plan nor moved further out (the log says which joints). Holds resend the last command
  unchanged.
- A failed state read does not stop the chunk: the command does not need a fresh state. Three failures in a
  row hold the arm (`bus_holds`), and 30 stop the loop.
- The history only holds commands that went out. A failed write leaves the last sent command in force and
  the next tick anchors on it; three failed writes in a row stop the loop for good (`ControlLoop.error`).
- `SO101.write` checks a native guard (`so101_client/safety.py`) in LeRobot degrees, below any joint map:
  a goal more than 25° from the measured joint, a jump of more than 15° between writes, or a goal outside
  `--limits` (plus 2°) raises `SafetyStop`. Nothing is written and the loop stops at once. The gripper is
  left out, since a grasp holds a steady error. The thresholds are `NativeGuard(max_error_deg,
  max_step_deg, limits)`.
- A dry run writes nothing, so its history keeps the held command and the measured state, as on the real
  arm. The logged `cmd` is what it would have sent: each accepted chunk, planned from the held pose. It
  cannot run away the way re-integrating unsent commands did (codex audit, `docs/research/
  so101-calibration-audit-2026-10-17.md`: +654° in 13 s), but it is a preview of one plan, not the live
  trajectory.
- For `--live`, someone stands at the arm with a hand on the power switch.
- `readout.py --torque-off` releases the motors, and the arm falls unless someone holds it. Torque stays
  off when it exits.

## Network and the Mac

- Use wired Ethernet. Wi-Fi spikes of 0.5 s were measured, which is the whole stall-free budget.
- The server binds 127.0.0.1:8000 on the GPU host. `infra/aws/ssh.sh` forwards it to `ws://localhost:8000`.
- While the loop runs it spawns `caffeinate -dimsu -w <pid>`, so the Mac neither sleeps nor dims. Keep the
  lid open.
- The loop sets Python's thread switch interval to 1 ms, so a camera or network thread cannot hold the
  control thread for long.

## How it works

| Module | Does |
|---|---|
| `so101_client/protocol.py` | `so101.v1` msgpack frames; arrays as `{"__nd__": true, "dtype", "shape", "data"}`; every message carries `type` |
| `so101_client/history.py` | `Ring(8)`: entry j pairs the state and frames of tick t−7+j with the command sent the tick before; `window()` keeps the frames at indices 0 and 7 |
| `so101_client/chunks.py` | pure: `to_deltas`, `integrate`, `clamp`, `Stitcher` (join at d, 6-tick crossfade, drop if the start index > 35), `Scheduler` (prefetch P = 16 or remaining ≤ d_max + 8; sync 32-then-replan), `Planner` (the two together, per tick, and the tick the current hold began) |
| `so101_client/net.py` | `NetClient`: websockets on its own thread and event loop, TCP_NODELAY, reconnect with backoff; requests in flight fail with an error on disconnect |
| `so101_client/cameras.py` | `Cameras`: one thread per camera (AVFoundation on macOS), 640×480 kept for taps, 256×256 JPEG q95 / PNG / raw encoded on the camera thread |
| `so101_client/robot_io.py` | `SO101` over LeRobot 0.6.1 `so101_follower`: `sync_read` Present_Position, `sync_write` Goal_Position after the native guard |
| `so101_client/safety.py` | `NativeGuard` (max error vs the measured joint, max step between writes, limits) and `SafetyStop` |
| `so101_client/fakes.py` | `FakeRobot` (first-order lag, τ = 0.1 s), `FakeCameras` (moving pattern), `FakeNet` (hold, sine, pickplace) |
| `so101_client/loop.py` | `ControlLoop`, `LoopConfig`, `TickInfo`: the 30 Hz loop from docs/architecture.md "Client loop" |

A chunk's joints are turned back into per-tick deltas (D0 = a0 − cmd_hist[−1], Di = ai − ai−1). The client
joins the chunk at index d = k − r while a chunk runs. After a hold that began at tick h it starts at index
max(0, h − r), which is 0 when the request went out during the hold (start, task change, sync mode); after a
starvation hold it skips the actions for ticks the arm already moved through on the previous chunk, which
would otherwise be replayed as overshoot. It adds each delta to the command it last sent, which is what BFL's
`select_action` does tick by tick. A position jump at a seam is therefore impossible, and the crossfade
smooths the change in velocity. With one request in flight, prefetch stays stall-free while the turnaround is
at most 21 ticks, since a chunk joined at d leaves 42 − d actions. d_max leaves out stale replies and the first
reply of each task (it can include encoding a new caption), which stands in only until another reply arrives.

`LoopConfig` adds four optional fields to the shared signature. `request_timeout_s` (10 s) is generous
because the first request for a new instruction includes encoding the caption on the server.
`max_frame_age_ms` (1000) blocks new requests while a camera's newest frame is older than that.
`error_backoff_ticks` is 15 and `caffeinate` is True. `ControlLoop.start()` connects whatever robot, cameras and net are not up yet. `stop()` releases only
those, so a caller that connected them keeps ownership. Besides the shared API, eval reads `running`, `error`,
`pending` (requests not yet answered) and `stats()`.

## Tests

```sh
cd robot && uv run --python 3.13 pytest -q
```

No hardware or GPU is needed. The tests cover the chunk arithmetic (re-integration, seams, crossfade, drop
rule, gripper, clamp), the window alignment, the prefetch and sync cadences (no overshoot after a starvation
hold, one slow first reply does not set the cadence), and the loop with all three fakes (holds only before the
first chunk, `set_task` drops the chunk and ignores the stale reply, pause holds, sync executes 32 then
replans, a failed state read is not starvation). They also run `NetClient` against a small `so101.v1` peer on
127.0.0.1, `Cameras` on generated video files, and `SO101` against the installed LeRobot 0.6.1 (the config it
builds, the calibration check, and the goal pinned before and after `configure()`). `eval/tests/test_e2e_socket.py` runs this client against the real T10 server
(`infra/gpu/so101/server.py --engine fake`) over a localhost socket.
