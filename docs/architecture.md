# Architecture

**Last changed:** 2026-10-17 · Changes need a lock on this file and a decision record (AGENTS.md section 2).

## Components

```
 robot or sim machine                          AWS p5.4xlarge spot (1x H100) eu-north-1
 ┌──────────────────────────┐   ssh -L 8000   ┌────────────────────────────────────┐
 │ robot/   adapter         │ ──────────────► │ flux-action serve-robolab :8000    │
 │  cameras + joint state   │  WebSocket,     │  (BFL repo, pinned in infra/gpu/)  │
 │  -> request, action ->   │  msgpack+numpy  │  weights in the HF cache           │
 │  execute first k of 32   │ ◄────────────── │                                    │
 │ eval/   episodes, video  │  action (32,8)  └────────────────────────────────────┘
 └──────────────────────────┘
```

| Directory | Owns | Task |
|---|---|---|
| `infra/aws/` | account steps, quota, budget, launch/stop/ssh of the one shared GPU | T02, T03 |
| `infra/gpu/` | bootstrap, smoke tests and serve scripts that run on the GPU host | T03–T05 |
| `infra/gpu/so101/` | our SO-101 policy server (`so101.v1` protocol below), its engines and self-test | T10 |
| `infra/client/` | the client side of the WebSocket protocol, plus a ping/latency tool | T05 |
| `robot/` | the SO-101 client on the Mac: cameras, joint bus, 8-tick history, chunk prefetch and stitching, 30 Hz loop | T06 |
| `eval/` | the Hanoi sequencer, step end, episode runner, success logging, video capture | T07 |
| `docs/`, `tasks/`, `locks/`, `scripts/` | coordination (AGENTS.md) | — |

The FLUX code is **not vendored**. `infra/gpu/bootstrap.sh` clones
`black-forest-labs/flux-action` at a pinned commit next to this repo on the GPU host.

## Interfaces (DROID policy, from BFL's setup guide)

**Offline (`flux-action infer`).** Input is an NPZ with `images.wrist`, `images.left` and `images.right`,
each uint8 `(360, 640, 3)` RGB, plus `state` as float32 `(8,)`: 7 joint positions in radians and
the gripper closed fraction. Output is `actions.npy` float32 `(1, 32, 8)` holding absolute joint
targets plus the gripper, and `report.json` with settings, timing and GPU memory.

**Served (`serve-robolab`, OpenPI WebSocket protocol).** The server sends metadata on
connect, then takes one msgpack request per call:

| Key | Content |
|---|---|
| `observation/image` | uint8 `(540, 640, 3)` composite: the wrist view above two half-size exterior views. Alternatively send the three separate keys `observation/wrist_image_left`, `observation/exterior_image_1_left`, `observation/exterior_image_2_left` |
| `observation/joint_position` | 7 joint positions in radians |
| `observation/gripper_position` | gripper closed fraction |
| `prompt` | task instruction |

The response holds `action` as float32 `(32, 8)` and `server_timing`.

**Control loop.** Execute the first k actions of each chunk, observe again, and replan.
The reference DROID setting is `n_action_steps: 32`; k is ours to tune and is recorded as a decision.

## SO-101 path (decisions 05 and 06)

```
 Mac at the robot station                              GPU host: eu-north-1 p5.4xlarge, or any Linux + CUDA box
 ┌────────────────────────────────────┐               ┌──────────────────────────────────────────────┐
 │ eval/  Hanoi sequencer, step end,  │               │ infra/gpu/so101/server.py   127.0.0.1:8000   │
 │        attempts, video             │               │  stateless: each request carries its window  │
 │   │ set_task / observers           │  ssh -L 8000  │  engine flux-action (default): load_policy   │
 │ robot/ ControlLoop 30 Hz           │ ────────────► │  engine lerobot: Flux3Policy (+ LoRA later)  │
 │   cameras scene + wrist, SO-101    │  so101.v1     │  engine fake: no GPU, for tests              │
 │   bus, 8-tick ring, prefetch       │ ◄──────────── │                                              │
 └────────────────────────────────────┘  chunk (42,6) └──────────────────────────────────────────────┘
```

**The model contract (verified in flux-action `830f1ce` and LeRobot main `146d847`).** Two cameras, `scene` and
`wrist`. State and commands have 6 channels in this order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper`. The first five are in degrees (LeRobot `use_degrees=true`); the gripper is LeRobot `RANGE_0_100`.
The model reads an 8-tick window. Entry j pairs the measured state and frames at tick t−7+j with the command
sent at the tick before. Only the frames at window indices 0 and 7 are VAE-encoded. At an episode start the window is
8 copies of the first observation, and the command is the measured state. The model returns 42 commands for ticks
t…t+41: joint deltas integrated from `command_history[-1]`, gripper absolute. The reference loop
(`select_action`) executes 32 of them, adding each delta to the last sent command, then replans.

### Wire protocol `so101.v1`

One WebSocket per client, binary msgpack frames, no compression. Arrays are msgpack maps
`{"__nd__": true, "dtype": "<f4" | "|u1", "shape": [...], "data": <bytes, C order>}`.

| Message | Direction | Fields |
|---|---|---|
| `hello` | server → client, on connect | `protocol: "so101.v1"`, `engine`, `checkpoint`, `revision`, `chunk: 42`, `exec: 32`, `fps: 30`, `n_obs: 8`, `snapshots: [0, 7]`, `cameras: ["scene", "wrist"]`, `state_names`, `guidance`, `steps`, `warm: bool` |
| `ping` → `pong` | both | `t` (client clock, echoed), `server_t` |
| `infer` | client → server | `id` (int), `tick` (control tick of the newest observation), `task` (str), `state` f32 (8, 6), `cmd_hist` f32 (8, 6), `images: {"scene": [i0, i7], "wrist": [i0, i7]}`, `image_format: "jpeg" \| "png" \| "raw"` (encoded bytes, or a u1 (H, W, 3) RGB array for raw) |
| `chunk` | server → client | `id`, `tick`, `actions` f32 (42, 6) absolute for ticks `tick…tick+41`, `timing: {queue_ms, decode_ms, infer_ms, total_ms}` |
| `error` | server → client | `id`, `error` (str). The connection stays open. |

The server runs one inference at a time on a worker thread and keeps answering pings while it computes.
It binds 127.0.0.1:8000 only, so the Mac reaches it through `ssh -L 8000:localhost:8000` (`infra/aws/ssh.sh`
forwards 8000). It is never exposed publicly. The client sends 256×256 frames by default. The model's own
side-by-side resize leaves those unchanged, and other sizes are resized by the model.

### Server engines (`infra/gpu/so101/engines/`)

`infer(state (8,6) f32, cmd_hist (8,6) f32, frames {cam: u1 (2, H, W, 3) RGB}, task) -> (42, 6) f32` absolute.

- `flux-action` (default, decision 03): `flux_action.inference.so101.load_policy(repo, revision=…)`, then
  `set_text_encoder_offload(True)` before `.to("cuda")`, then `predict_action_chunk` on the explicit window
  `{"state": (1,8,6), "command_history": (1,8,6), "images.scene": u1 (1,8,3,H,W), "images.wrist": …, "task": [task]}`.
  The two frames sit at indices 0 and 7, and slots 1–6 repeat them (they are never encoded).
- `lerobot`: LeRobot main `Flux3Policy` with its saved processors, through the stateless
  `observation.command_history` branch. Same inputs and outputs. It can load a LoRA adapter (milestone 4).
- `fake`: no torch, for tests. Modes `hold`, `sine`, `pickplace` (a gripper close-hold-open cycle), plus an optional simulated latency.

### Client loop (`robot/`)

Each tick k at 30 Hz, with absolute deadlines (sleep until about 6 ms before each one, then spin):

1. Read the state (`sync_read`), take the newest frame of each camera, and append `(state_k, cmd_{k-1}, frames_k)` to the 8-deep ring.
2. If a chunk for request tick r has arrived, d = k − r. Turn it into deltas (Δ0 = a0 − cmd_hist[−1], Δi = ai − ai−1,
   joints 0–4; gripper absolute) and continue from index d, integrating from the last sent command, so there is no jump.
   Crossfade the joint deltas over K = 6 ticks (weights (j+1)/(K+1)). Drop the chunk if d > 41 − K.
3. Choose cmd_k (the next delta, or a hold that resends the last command), clamp it (per-tick |Δ| ≤ `step_max`,
   joint limits, gripper 0–100), and write it unless this is a dry run. Dry run is the default; `--live` writes.
4. **prefetch** (default): with nothing in flight, request when k − last_request ≥ P (16), or when the remaining
   actions ≤ d_max + K + 2, where d_max is the worst turnaround of the last 100 plans, in ticks. **sync**: BFL's
   cadence. Execute 32, hold without appending to the history, request from the current window, start the reply at index 0.
5. After any hold (start, starvation, task change), the next chunk starts at index 0.

Camera capture, JPEG encoding and all socket I/O run on their own threads; the control thread never blocks on them.

### `robot/` ↔ `eval/` (Python, same process on the Mac)

`ControlLoop(robot, cameras, net, cfg)` with `start()`, `set_task(task, reset_history)`, `pause()` (hold, no requests),
`add_observer(fn)` (called on the control thread every tick with a `TickInfo`; must return fast),
`add_frame_tap(fn(camera, frame_bgr, t))` (full-resolution frames, called on the camera threads) and `stop()`.
`TickInfo` carries `k, t, state (6,), cmd (6,), held, lateness_ms, frame_age_ms {cam: ms}, remaining, task`, and `chunk`
(set on the tick a chunk is accepted: `id, tick, d, turnaround_ms, server_ms`). `set_task(..., reset_history=False)`
drops the queued actions and keeps the history (LeRobot's `set_task` behaviour), and eval uses it between Hanoi steps.
`reset_history=True` refills the ring with the current observation and sets the last command to the measured state;
eval uses it at an episode start and after an intervention. `FakeRobot` and `FakeCameras` stand in for the hardware.

### `eval/` output

`eval/runs/<UTC stamp>-<label>/` (gitignored) holds `run.json` (config, git commit, server hello) and `summary.md` (the
results table), plus one `epNN/` per episode with `episode.json`, `attempts.jsonl` (one line per step attempt:
prompt, times, end reason `key|auto|timeout`, success, interventions), `ticks.jsonl`, `chunks.npz` (every chunk
received, with its id, tick and d) and `scene.mp4` / `wrist.mp4`.
