# Project: zero-shot FLUX 3 Action

**Status:** milestone 1 in progress (zero-shot inference) · **Created:** 2026-10-17 · **Repo owner:** alerest285

## What we are doing

[FLUX 3 Action](https://huggingface.co/blog/black-forest-labs/flux-3-action) is Black Forest
Labs' open-weights 7B world action model. It takes camera frames, the robot state and a text
instruction. It returns a chunk of 32 future actions, denoised jointly with the future video
frames. It ranks first on RoboLab-120 with 42.9 %. We run it **zero-shot** on one task
first, meaning the released checkpoints with no fine-tuning. Fine-tuning comes later, if at all.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | **Inference infra** | A released checkpoint runs on our AWS H100 and returns `(1, 32, 8)` actions for a recorded DROID observation. Latency and VRAM are logged. |
| 2 | **Served policy** | `flux-action serve-robolab` is up and a client on another machine gets an action chunk over the WebSocket. |
| 3 | **Zero-shot task** | The chosen task runs closed-loop on the chosen embodiment (real robot or sim) for N episodes, with success counted and videos saved. |
| 4 | Stretch | LoRA on SO-101, or a second task. Decided after milestone 3. |

## Open questions (resolve as decision records)

1. **Which embodiment and task for milestone 3?** This is task T01. The released policies cover two:
   - **DROID**: a Franka arm with 3 cameras. Also the setup used for **RoboLab**, a simulated benchmark, so it needs no physical robot. Isaac Sim needs an RTX-class GPU, which an H200 is not. Unverified; check the RoboLab docs.
   - **SO-101**: the cheap LeRobot arm, about 200 teleop pick-and-place episodes. It needs the physical arm and two cameras: scene and wrist.
2. **VRAM and latency.** Not published. We measure them on our H100 in T04 (`report.json` records GPU memory).

## Constraints

- **Compute is ours to provide**: one AWS `p5.4xlarge` (1× H100 80 GB) on spot in **eu-north-1**, **$150 total**,
  inference only (decisions 02 and 04, `infra/aws/README.md`). That is about 47 GPU-hours at $3.16/h, so stop the box when it is idle.
  Spot needs the AWS project on the paid plan; the free plan blocks it. The fallback is on-demand at $7.36/h, about 20 h.

## Out of scope for now

Training from the base trunk, games, the drone, custom embodiments.

## Sources

The facts we rely on, with links, are in `docs/research/flux-3-action.md`. The coordination
method is in `docs/research/coordination-methods.md`.
