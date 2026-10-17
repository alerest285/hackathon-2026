# Architecture

**Last changed:** 2026-10-17 · Changes need a lock on this file and a decision record (AGENTS.md section 2).

## Components

```
 robot or sim machine                          AWS p5.4xlarge spot (1x H100, DLAMI)
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
| `infra/client/` | the client side of the WebSocket protocol, plus a ping/latency tool | T05 |
| `robot/` | adapter from our embodiment (real or sim) to the request format | T06 |
| `eval/` | episode runner, success logging, video capture | T07 |
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

**SO-101 policy (if T01 picks it).** The inputs are two cameras, `scene` and `wrist`, plus a state of 6
values: shoulder pan, shoulder lift, elbow flex, wrist flex and wrist roll in degrees, and the
gripper in percent. The output is `(1, 42, 6)`. In a loop, call `policy.select_action(batch)` on every
30 Hz tick; it executes 32 actions and then replans. Call `policy.reset()` at each episode boundary.
There is no server for it in BFL's repo, so one would be our component.
