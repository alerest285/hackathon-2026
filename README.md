# hackathon-2026: zero-shot FLUX 3 Action on an SO-101

We run Black Forest Labs' open 7B world action model, [FLUX 3 Action](https://huggingface.co/blog/black-forest-labs/flux-3-action),
zero-shot on a physical SO-101 arm. The task is a 3-disk Tower of Hanoi, one instruction per move (decision 05).
Three people work on it in parallel, each with their own coding agent. This repo is set up so those agents
never edit the same files and always start from the latest state.

**Private repo.** Members: alerest285, saadl27, joelleayoub.

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
ssh root@<pod-host> -p <ssh-port>          # alerest285, saadl27 and joelleayoub keys are installed
```

| Path on the pod | What |
|---|---|
| `/workspace/flux-action` | BFL's code, pinned at `830f1ce`, with its env (`uv run ...`) |
| `/workspace/hf` | weights: `flux-3-action-so101` + the `flux-3-action-base` encoders (gated; alerest's HF token is on the pod) |
| `/workspace/hackathon-2026` | this repo; `git pull` before running anything |

Always set `FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache`. Run long jobs inside
`tmux` and log GPU use in your log. Only `/workspace` survives a pod restart, and the SSH port changes when the pod
restarts: check the Runpod console or ask alerest285. Re-check the inference anytime:

```sh
cd /workspace && FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache \
  bash hackathon-2026/infra/gpu/smoke-so101.sh 5
```

AWS is the official compute, and the hackathon reimburses it. It is set up (`infra/aws/`, eu-north-1) but waits on
AWS approving the GPU quota. Once approved, `infra/aws/first-boot.sh` brings up the same thing there (decision 07).

## Join in 3 commands

Accept the GitHub invite first (check your email or https://github.com/alerest285/hackathon-2026/invitations).

```sh
gh repo clone alerest285/hackathon-2026 && cd hackathon-2026
scripts/setup.sh claude-code      # or codex | cursor | gemini. Sets your name = GitHub login, turns hooks on
scripts/sync.sh                   # pull + the board: locks, tasks, latest logs, decisions
```

Then tell your agent: **"Read AGENTS.md and pick a task."** Running several agents? Give each one its own name
(`export HACK_AGENT=claude-robot`), because locks belong to member + agent.

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
