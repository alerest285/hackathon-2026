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
| `robot/` | the SO-101 client on the Mac: cameras, joint bus, native guard, 8-tick history, chunk prefetch and stitching, 30 Hz loop | T06 |
| `eval/` | the Hanoi sequencer, step end, episode runner, success logging, video capture, attempts as a LeRobot dataset | T07 |
| `console/` | the localhost operator UI over the `robot/` loop, its LeRobot recorder, the joint map (`mapping.py`) and its fit (`fit_reference.py`) | T18–T22 |
| `train/` | LoRA fine-tuning on our demos (decision 09) | T23 |
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

## SO-101 path (decisions 05, 06 and 10)

```
 Mac at the robot station                                GPU host: Runpod H100 now (decision 08), any Linux + CUDA
 ┌────────────────────────────────────────┐             ┌──────────────────────────────────────────────┐
 │ eval/ Hanoi runner · console/ web UI   │             │ infra/gpu/so101/server.py   127.0.0.1:8000   │
 │   │ set_task / observers / frame taps  │             │  stateless: each request carries its window  │
 │ robot/ ControlLoop 30 Hz   model space │ ssh -L 8000 │  engine flux-action (default): load_policy   │
 │   8-tick ring, prefetch, loop clamp    │ ──────────► │  engine lerobot: Flux3Policy (+ LoRA, T23)   │
 │ MappedRobot (console/mapping.py)  ─────│ so101.v1    │  engine fake: no GPU, for tests              │
 │ SO101 + NativeGuard        native deg  │ ◄────────── │                                              │
 │ cameras scene + wrist                  │ chunk (42,6)└──────────────────────────────────────────────┘
 └────────────────────────────────────────┘
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

One WebSocket per client, binary msgpack frames, no compression. Every message is a msgpack map with
`"type": "hello" | "ping" | "pong" | "infer" | "chunk" | "error"`. Arrays are msgpack maps
`{"__nd__": true, "dtype": "<f4" | "|u1", "shape": [...], "data": <bytes, C order>}`.

| Message | Direction | Fields |
|---|---|---|
| `hello` | server → client, on connect | `protocol: "so101.v1"`, `engine`, `checkpoint`, `revision`, `chunk: 42`, `exec: 32`, `fps: 30`, `n_obs: 8`, `snapshots: [0, 7]`, `cameras: ["scene", "wrist"]`, `state_names`, `guidance: [video, action]`, `steps`, `warm: bool`, `warm_tasks: [str]` (captions encoded at startup), plus engine extras the client ignores (`sampler`, `seed`, `dtype`, `compile`, encoder ids, `state_q01/q99`, `action_q01/q99`) |
| `ping` → `pong` | both | `t` (client clock, echoed), `server_t` |
| `infer` | client → server | `id` (int), `tick` (control tick of the newest observation), `task` (str), `state` f32 (8, 6), `cmd_hist` f32 (8, 6), `images: {"scene": [i0, i7], "wrist": [i0, i7]}`, `image_format: "jpeg" \| "png" \| "raw"` (encoded bytes, or a u1 (H, W, 3) RGB array for raw) |
| `chunk` | server → client | `id`, `tick`, `actions` f32 (42, 6) absolute for ticks `tick…tick+41`, `timing: {queue_ms, decode_ms, infer_ms, total_ms}` |
| `error` | server → client | `id`, `error` (str). The connection stays open. A request is checked on arrival, so its error can come before chunks of earlier requests. |

The server runs one inference at a time on a worker thread and keeps answering pings while it computes.
It binds 127.0.0.1:8000 only, so the Mac reaches it through `ssh -L 8000:localhost:8000`. It is never exposed
publicly. The client sends 256×256 frames by default. The model's own side-by-side resize leaves those unchanged,
and other sizes are resized by the model. Every value in `state`, `cmd_hist` and `actions` is in model space
(decision 10).

### Server engines (`infra/gpu/so101/engines/`)

`infer(state (8,6) f32, cmd_hist (8,6) f32, frames {cam: u1 (2, H, W, 3) RGB}, task) -> (42, 6) f32` absolute.

- `flux-action` (default, decision 03): `flux_action.inference.so101.load_policy(repo, revision=…)`, then
  `set_text_encoder_offload(True)` before `.to("cuda")`, then `predict_action_chunk` on the explicit window
  `{"state": (1,8,6), "command_history": (1,8,6), "images.scene": u1 (1,8,3,H,W), "images.wrist": …, "task": [task]}`.
  The two frames sit at indices 0 and 7, and slots 1–6 repeat them (they are never encoded).
- `lerobot`: LeRobot main `Flux3Policy` with its saved processors, through the stateless
  `observation.command_history` branch. Same inputs and outputs. `--adapter <dir>` loads a PEFT LoRA (T23).
- `fake`: no torch, for tests. Modes `hold`, `sine`, `pickplace` (a gripper close-hold-open cycle,
  `--fake-period-s`), plus an optional simulated latency.

Server flags that change the numbers: `--compile` wraps the DiT forward in `torch.compile` and falls back to eager
if the compiled warmup fails. On the H100 it took plans from p50 514 ms to about 351 ms with the same sampling.
`--guidance VIDEO ACTION` and `--steps N` override the sampler (3/3 and 4 by default; 1/1 halves the compute and
needs a decision record). `--norm-stats <json>` replaces the state/action quantiles for zero-shot (with `--adapter` it
only checks them, since an adapter keeps its training normalization); the hello reports the source and its sha256. `--warmup-task` can be repeated; `eval/run_hanoi.py --print-warmup` prints one per Hanoi
prompt. The encoder revision pinned in the checkpoint's config wins. A conflicting `--base-revision` stops the
server before any weights load.

### Client loop (`robot/`)

Each tick k at 30 Hz, with absolute deadlines (sleep until about 6 ms before each one, then spin):

1. Read the state (`sync_read`), take the newest frame of each camera, and append `(state_k, cmd_{k-1}, frames_k)`
   to the 8-deep ring. `cmd_{k-1}` is the last command that reached the motors (below). A failed read keeps the
   chunk going on the last good state. 3 failed reads in a row hold the arm (`bus_holds`), and 30 stop the loop.
2. If a chunk for request tick r has arrived, d = k − r. Turn it into deltas (Δ0 = a0 − cmd_hist[−1], Δi = ai − ai−1,
   joints 0–4; gripper absolute) and continue from index d, integrating from the last sent command, so there is no
   jump. Crossfade the joint deltas over K = 6 ticks (weights (j+1)/(K+1)). Drop the chunk if d > 41 − K.
3. Choose cmd_k (the next delta, or a hold that resends the last command) and clamp it: joint limits, then
   |Δ| ≤ `step_max` (6°) per tick per joint, gripper 0–100. A joint already outside its limits may move back in at
   the planned pace but never further out. Write it unless this is a dry run (dry run is the default; `--live`
   writes). Then the safety rules of decision 10 apply:
   - A failed write leaves the last sent command in force, and the next tick anchors on it.
   - 3 failed writes in a row, or one `SafetyStop` from the native guard, stop the loop for good (`ControlLoop.error`).
   - A dry run keeps the held command in the ring. Its logged command is a preview: each accepted chunk is replayed
     from the held pose, restarting inside the limits.
4. **prefetch** (default): with nothing in flight, request when k − last_request ≥ P (16), or when the remaining
   actions ≤ d_max + K + 2. d_max is the worst turnaround, in ticks, of the last 100 replies. Stale replies and each
   task's first reply (which can include encoding a new caption) are left out; the first reply stands in only while
   there is no other sample. With one request in flight, prefetch stays stall-free up to a turnaround of about 21
   ticks (700 ms). **sync**: BFL's cadence. Execute 32, hold without appending to the history, request from the
   current window, start the reply at index 0.
5. After a hold that began at tick h, the next chunk for request tick r starts at index max(0, h − r), without a
   crossfade. That is 0 when the request went out during the hold (start, task change, sync mode). After a
   starvation hold it skips the actions for ticks the arm already moved through on the previous chunk.

Camera capture, JPEG encoding and all socket I/O run on their own threads; the control thread never blocks on them.

### Units, joint map and safety (decision 10)

The loop, its history, the server and the chunks are in model space, the checkpoint's convention. Everything stored
(LeRobot datasets, eval's `ticks.jsonl`, limits files) and every safety bound is in native LeRobot 0.6.1 degrees,
with the gripper 0..100. `MappedRobot(SO101, JointMap)` from `console/mapping.py` converts at the arm, using
`~/hackathon-data/joint_mapping.json` (written by `console/fit_reference.py`, T21). console/ and eval/ apply it;
`robot/run.py` does not (raw degrees, bring-up only). Before a map drives the arm, its `meta.robot_id` must match
the arm and its `meta.calibration_sha256` must match the arm's calibration file. Live runs refuse a mismatch.

Under the map, `SO101.write` checks a `NativeGuard` (`robot/so101_client/safety.py`) in native degrees and raises
`SafetyStop` without writing on any of these:

- |goal − measured| > 25°
- a jump > 15° between writes
- a goal outside the native limits + 2°

The gripper is exempt. `--live` needs native limits everywhere: `robot/run.py --limits`, `console/start.sh
--limits`, eval `--limits` or `loop.joint_limits`. The loop clamps with the same limits mapped into model space.

### `robot/` ↔ `eval/` and `console/` (Python, same process on the Mac)

`ControlLoop(robot, cameras, net, cfg)` provides:

- `start()`, `set_task(task, reset_history)`, `pause()` (hold, no requests) and `stop()`.
- `add_observer(fn)`: called on the control thread every tick with a `TickInfo`; it must return fast.
- `add_frame_tap(fn(camera, frame_bgr, t))`: full-resolution frames, called on the camera threads.
- Read-only `running`, `error` (set when the loop stopped on a fault), `stats()` and `pending` (requests not yet
  answered, failed or given up).

`TickInfo` carries `k, t, state (6,), cmd (6,), held, lateness_ms, frame_age_ms {cam: ms}, remaining, task, chunk`
and `written`, in the loop's units:

- `cmd` is the command in force: written this tick, kept after a failed write, or a dry run's preview.
- `written` is true when `cmd` reached the motors this tick.
- `chunk` is set on every tick that handles a reply with actions (accepted, late or stale). Its keys are `id, tick,
  d, accepted, start, reason (ok|late|stale), turnaround_ms, server_ms, rtt_ms, actions`.

`set_task(..., reset_history=False)` drops the queued actions and keeps the history (LeRobot's `set_task`
behaviour), and eval uses it between Hanoi steps. `reset_history=True` refills the ring with the current observation
and sets the last command to the measured state. eval uses it at an episode start and after an intervention,
including a resume after a pause. `FakeRobot`, `FakeCameras` and `FakeNet` stand in for the hardware and the server.

### `eval/` output

`eval/runs/<UTC stamp>-<label>[-sanity|-recover][-demo]/` (gitignored) holds:

- `run.json`: config, git commit, server hello, `LoopConfig`, the joint map, native limits and guard, a note on units,
  the dataset location and the loop stats.
- `summary.md` / `summary.json`: the results table.
- One `epNN/` per episode with:
  - `episode.json`
  - `attempts.jsonl`: one line per step attempt with prompt, times, end reason `key|auto|timeout|quit|error`, error,
    success, verdict_by, pauses and interventions. An `error` end is an infrastructure fault, kept out of the totals.
  - `ticks.jsonl`: state and cmd in native degrees.
  - `chunks.npz`: every chunk received, in model space, with its id, tick, d, start, reason and attempt.
  - `scene.mp4` / `wrist.mp4`, plus `*.times.csv` to align frames with ticks.

Live runs also record each step attempt as one LeRobotDataset episode, task = its prompt, in native units through
`console/recorder.py`, to `~/hackathon-data/so100_hanoi_eval`. Its sidecars are `console_episodes.jsonl` (run_mode,
drops, valid) and `hanoi_episodes.jsonl` (verdict, run, step, attempt, interventions).
