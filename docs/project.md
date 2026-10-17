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
| 1 | **Inference infra** | `flux-3-action-so101` runs on our H100 and returns `(1, 42, 6)` actions for a recorded or synthetic SO-101 observation. Latency and VRAM are logged. (The DROID smoke test, T04, is an optional environment check.) |
| 2 | **Served policy** | Our SO-101 server on the GPU host returns action chunks to the Mac that drives the arm, with the round trip logged and split into network and server time. |
| 3 | **Zero-shot task** | The 3-disk Tower of Hanoi (decision 05) runs closed-loop on the SO-101 for 10 strict episodes, with steps completed counted and videos saved. |
| 4 | Stretch | LoRA on SO-101 with Hanoi demonstrations, compared with milestone 3 under the same protocol. |

## Open questions (resolve as decision records)

1. ~~Which embodiment and task for milestone 3?~~ Settled: the physical SO-101 and a 3-disk Tower of Hanoi, one move per instruction. Prompts, success rule and N = 10 are in decision 05.
2. ~~VRAM and latency~~ measured 17.10: 25.9 GB, 0.58 s per chunk eager, 123 s load (status 2026-10-17-1545).

## Constraints

- **Compute is ours to provide**, and the hackathon reimburses AWS. **Now:** one Runpod H100 80 GB in Paris, $3.99/h,
  running continuously (decision 08). **Official route:** an AWS `p5.4xlarge` on-demand in eu-north-1 at $7.36/h (decisions 04 and 07),
  which waits on AWS approving the GPU quota.

## Out of scope for now

Training from the base trunk, games, the drone, custom embodiments.

## Sources

The facts we rely on, with links, are in `docs/research/flux-3-action.md`. The coordination
method is in `docs/research/coordination-methods.md`.
