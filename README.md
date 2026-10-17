# hackathon-2026: zero-shot FLUX 3 Action on an SO-101

We run Black Forest Labs' open 7B world action model, [FLUX 3 Action](https://huggingface.co/blog/black-forest-labs/flux-3-action),
zero-shot on a physical SO-101 arm. The task is a 3-disk Tower of Hanoi, one instruction per move (decision 05).
Four people work on it in parallel, each with their own coding agent. This repo is set up so those agents
never edit the same files and always start from the latest state.

**Private repo.**

| Member | GitHub | Pod SSH | Runpod team |
|---|---|---|---|
| Alejo | alerest285 | yes | owner |
| Saad | saadl27 | yes (2 keys) | invite pending (<redacted-email>) |
| Joelle | joelleayoub | yes | invite pending (<redacted-email>) |
| Erfan | ErfanRzt | **no key on GitHub yet**: add one, then ask alerest285 | invite pending (<redacted-email>) |

Accept your GitHub invite first: https://github.com/alerest285/hackathon-2026/invitations

## Where we are (17.10.2026)

| Milestone | State |
|---|---|
| 1. Inference on a GPU | **done.** SO-101 policy on an H100: 0.58 s per 42-step action chunk (uncompiled), 25.9 GB of GPU memory, 2 min to load |
| 2. Policy served to the Mac at the robot | in progress: T10 server, T11 interfaces |
| 3. Hanoi zero-shot, 10 episodes | next: T06 robot adapter, T07 episode runner |
| 4. LoRA on our own demonstrations | stretch |

The newest status file in [`docs/status/`](docs/status) is always the current truth.

## The GPU

One H100 80 GB on Runpod, in Paris (EU-FR-1). Pod `rmhs4cu2cejemv`, **$3.99/h, running continuously**.

```sh
ssh root@<pod-host> -p <ssh-port>          # keys installed: alerest285, saadl27, joelleayoub
```

| Path on the pod | What |
|---|---|
| `/workspace/flux-action` | BFL's code, pinned at `830f1ce`, with its env (`uv run ...`) |
| `/workspace/hf` | weights: `flux-3-action-so101` + the `flux-3-action-base` encoders (gated; alerest's HF token is on the pod) |
| `/workspace/hackathon-2026` | this repo; `git pull` before running anything |

Always set `FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache`. Run long jobs inside
`tmux` and log GPU use in your log. Only `/workspace` survives a pod restart, and the SSH port changes when the pod
restarts: check the Runpod console or ask alerest285. The pod pulls this private repo through a read-only deploy key.

AWS is the official compute, and the hackathon reimburses it. It is set up (`infra/aws/`, eu-north-1) but waits on
AWS approving the GPU quota. Once approved, `infra/aws/first-boot.sh` brings up the same thing there (decision 07).

## Run the zero-shot inference yourself (5 minutes)

Everything is already installed on the pod. From your laptop:

```sh
ssh root@<pod-host> -p <ssh-port>
cd /workspace && git -C hackathon-2026 pull
export FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache
nvidia-smi                      # check nobody else is running a heavy job (see "Sharing the GPU")

# 1. synthetic frame: checks the stack works, prints load time, latency, GPU memory
bash hackathon-2026/infra/gpu/smoke-so101.sh 5

# 2. REAL SO-101 frames (public dataset, red cube into pot): FLUX's 42 actions vs what the human did next
cd /workspace/flux-action
uv run --extra encoders --extra data --with pandas python /workspace/hackathon-2026/infra/gpu/real-so101.py --frame 150 \
  --out /workspace/outputs/$USER-real-f150
```

Expect: load ~2 min, then ~0.6–1 s per action chunk of shape (42, 6), ~26 GB of GPU memory.
`real-so101.py` saves the input image (`input_scene_left_wrist_right.png`), `pred.npy` and `recorded.npy` in `--out`.
Measured 17.10 (alerest285's log): frame 150 follows the operator's reach to within 3–17° per joint; frame 300 diverges.

Your own frames: an `.npz` with `images.scene` and `images.wrist` (HWC uint8 RGB, **same shape**) and `state`
(6 × float32 in **degrees**: pan, lift, elbow, wrist flex, wrist roll, gripper %). Then:

```sh
uv run --extra encoders python examples/so101/inference.py --weights black-forest-labs/flux-3-action-so101 \
  --observation my_obs.npz --prompt "pick up the small disk and place it on the right peg" --output /workspace/outputs/$USER-mine.npy
```

Rig rules learned the hard way: the scene camera is the fixed overview (left in the tiled image) and the wrist camera is
on the gripper (right). Never swap them. Both must have the same resolution, and joint states are in degrees, not LeRobot's −100…100 range.

## Drive the real arm (Saad): follower arm only

**For Saad's agent: do these in order on the Mac with the SO-100 follower arm and both cameras plugged in.**
No leader arm is needed. Stop at any step that fails and report it with `scripts/say.sh @alerest285 "..."`.

```sh
git pull
# 0. tools: uv (curl -LsSf https://astral.sh/uv/install.sh | sh) and ssh access to the pod (your GitHub key is installed)

# 1. find the follower's USB port: unplug + replug the arm's USB when asked
cd robot && uv sync --python 3.13 && uv run --python 3.13 lerobot-find-port && cd ..

# 2. find the cameras: scene = fixed camera looking at the table, wrist = camera on the gripper
cd robot && uv run --python 3.13 python camera_check.py --scene 0 --wrist 1 && cd ..
#    open robot/runs/camera-check-*/composed.png: LEFT must be the scene, RIGHT the wrist. Wrong? swap the numbers.

# 3. calibrate the follower (interactive, a human moves the arm): middle pose + ENTER, then every joint through its full range + ENTER
console/calibrate.sh --port /dev/tty.usbmodemXXXX

# 3b. map YOUR arm's joint ranges onto the checkpoint's (fixes the calibration mismatch). Torque goes OFF: hold the arm.
#     For 45 s move it through the task workspace: rest pose, reach every spot on the table, high/low, gripper fully
#     open and closed, wrist rolled both ways. Writes ~/hackathon-data/joint_mapping.json, which the console loads.
cd robot && uv run --python 3.13 python ../console/fit_mapping.py --port /dev/tty.usbmodemXXXX && cd ..

# 4. start the console in DRY RUN (the arm is read, never moved). The page opens at http://127.0.0.1:8765
console/start.sh --port /dev/tty.usbmodemXXXX --scene 0 --wrist 1
```

5. **On the page, check before anything moves:**
   - **Follower calibration** card shows `ok`, the file path and all six motors. `MISSING` means: redo step 3.
   - Camera previews: scene left, wrist right.
   - The **mapping on** pill is green. `mapping MISSING` means: do step 3b.
   - Joint table: **arm (deg)** is your arm, **model space** is what the model sees after the mapping. Model space must sit inside
     the **trained range**; red rows mean a pose outside your sweep: refit 3b covering it.
   - **Direction check:** in dry run with an instruction running, watch the **command** column while moving nothing. If a joint's
     command pulls clearly the wrong way (e.g. the arm would fold when it should reach), refit with that joint flipped:
     `... fit_mapping.py --port ... --flip elbow_flex` (several joints allowed), and tell the team which ones.
   - Type `pick up the small disk and place it on the middle peg`, press **Run**. In dry run the **command** column changes and
     the arm stays still. The turnaround reads about 500–600 ms.
6. **Live**, with a person at the arm and a hand on the power switch: Ctrl-C, then
   `console/start.sh --port /dev/tty.usbmodemXXXX --scene 0 --wrist 1 --live`. **Run**, watch, **Pause** to hold,
   **Save: success / fail** at the end of each attempt. Every attempt is recorded as a LeRobot dataset episode in
   `~/hackathon-data/so100_hanoi_flux_zeroshot`.

Recalibrate any time: Ctrl-C the console, `console/calibrate.sh --port ...`, start again. The policy server runs on the pod
(tmux `so101-serve`, 514 ms per plan), and `start.sh` starts it if it is down. Details and troubleshooting: [`console/README.md`](console/README.md).

## Sharing the GPU

One H100, four people. The model takes ~26 GB of 80, so two inference jobs fit side by side. Training or compiling does not.

- Run anything longer than a minute in **tmux named after you** (`tmux new -s saad-smoke`). `tmux ls` shows who is running what.
- Before a heavy job, check `nvidia-smi`. If another job holds more than 40 GB, ask in the team chat first.
- Write outputs to `/workspace/outputs/<you>-…`. Never write into someone else's folder, and never into `/workspace/flux-action` or `/workspace/hf` (shared install and weights).
- Do not `uv sync`, `pip install` into, or `git checkout` inside `/workspace/flux-action`: it would break everyone's environment
  (NATTEN gets wiped). One-off packages: `uv run --with <pkg>`. Install changes go through a task and a decision.
- Do not stop, restart or terminate the pod without asking alerest285. It runs continuously, and a restart changes the SSH port.
- Log GPU work in your log: `scripts/log.sh --task T.. "what ran, how long, numbers"`.

## Join in 3 commands

```sh
gh repo clone alerest285/hackathon-2026 && cd hackathon-2026
scripts/setup.sh claude-code      # or codex | cursor | gemini. Sets your name = GitHub login, turns hooks on
scripts/sync.sh                   # pull + the board: locks, tasks, latest logs, decisions
```

Then tell your agent: **"Read AGENTS.md, run scripts/sync.sh, start listening to the agent chat, and pick an unclaimed task."**

## Running several agents without overlap

These rules hold for every agent, from every person. The hooks enforce the first three.

1. **One name per agent.** Before starting each agent: `export HACK_AGENT=<you>-<job>` (e.g. `saad-robot`, `saad-eval`).
   A lock belongs to member + agent, so two of your own agents cannot hold the same file either.
2. **Claim before editing.** `scripts/claim.sh T06` locks the task's paths. If it says "overlaps … held by …", pick another task
   or ask that person. Never widen your glob to grab someone else's files.
3. **Commit only what you claimed.** The pre-commit hook blocks the rest; CI flags anything that slips through.
4. **Separate checkouts for parallel agents.** Two agents in one folder share uncommitted files. Use a second clone or
   `git worktree add ../hack-<job>` for each extra agent.
5. **Pull often, push small.** `scripts/sync.sh` at the start of every session and before every push. Commit at least hourly.
6. **Release when done:** `scripts/release.sh T06 done "what shipped"`, or `todo "where I stopped"` so the next agent can continue.
7. **New work = new task:** `scripts/new-task.sh "title" 'narrow/path/**'`, then claim it. Ids never collide.

Who holds what right now is always on `scripts/sync.sh` (ACTIVE LOCKS). On 17.10 alerest285's Hanoi agent holds the SO-101
server (`infra/gpu/so101/**`, T10), the robot adapter (`robot/**`, T06), the episode runner (`eval/**`, T07) and `docs/architecture.md` (T11).
Free work for others: T04/T05 (DROID check, served-policy ping), or new tasks under your own paths.

## Agents talking to each other

Our agents (alerest285's, saadl27's and everyone else's) keep a running conversation through the repo:

```sh
scripts/say.sh @saadl27 "camera indices: scene=/dev/video0, wrist=/dev/video2, both 480x640"   # to one member
scripts/say.sh "T10 server up on the pod, port 8765; client in infra/gpu/so101/"                # to everyone
scripts/listen.sh          # live stream: [chat FOR YOU] / [chat] / [board] lines, polls every 20 s, fetch-only
```

Claude Code agents run `scripts/listen.sh` under the **Monitor tool** for the whole session (CLAUDE.md tells them to), so
every message arrives in their conversation within ~20 s. Lines addressed to them say `FOR YOU`, and they answer
before continuing. Messages live in `docs/chat/<login>--<agent>.md`: one file per agent, append-only, never conflicting.
To read the history: `cat docs/chat/*.md | grep '^- ' | sort | tail -30`.

## How the agents stay out of each other's way

| Rule | How |
|---|---|
| Claim before you edit | `scripts/claim.sh T06` pushes a lock file to `main`. Only one push can land first, so a claim is atomic. Overlapping claims are refused |
| Enforced | a pre-commit hook (every tool), a Claude Code edit guard, and CI re-checking each push |
| Fresh context | `scripts/sync.sh` at every session start shows who holds what, the open tasks, everyone's latest log lines, and any upstream docs that changed |
| Docs flow downstream | `AGENTS.md` → `docs/project.md` → `docs/architecture.md` → the task file → code. Fix the doc before the code contradicts it |
| Write as you go | your own log (`scripts/log.sh`), handoff notes in the task file, `scripts/new-decision.sh` for choices |

| Command | Does |
|---|---|
| `scripts/claim.sh T06` | locks a task's files for you (3 h, renew by running it again) |
| `scripts/release.sh T06 done "note"` | frees them and records the handoff (`todo` / `blocked` also work) |
| `scripts/log.sh --task T06 "..."` | one line in `docs/log/<you>.md` |
| `scripts/new-task.sh "title" 'path/**'` | a new task with a collision-free id |
| `scripts/new-decision.sh "title"` | a new numbered decision record |

## Read in this order

[`AGENTS.md`](AGENTS.md) (the contract) · [`docs/project.md`](docs/project.md) (goal, milestones) ·
[`docs/architecture.md`](docs/architecture.md) (components, exact interfaces) · [`docs/decisions/`](docs/decisions) (why) ·
[`tasks/`](tasks) (work, with "Done when" checks)

Never commit keys, tokens, `.env`, weights or datasets, even though the repo is private.

## Licenses

Our code is MIT. FLUX 3 Action code is Apache-2.0. The weights are under the
[FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/main/LICENSE.md),
downloaded on the GPU and never stored here.
