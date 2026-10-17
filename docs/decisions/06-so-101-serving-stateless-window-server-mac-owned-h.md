# 06. SO-101 serving: stateless window server, Mac-owned history, chunk prefetch

**Date:** 2026-10-17 · **Status:** accepted, numbers pending (T10) · **Author:** alerest285 (claude-code-hanoi)

## Context

The SO-101 hangs off a Mac over USB; the GPU sits in eu-north-1 (decision 04), 35–38 ms from Zurich, with
Wi-Fi spikes measured to 0.5 s. No ready-made server can run `flux-3-action-so101` remotely. All of these were checked in
source (LeRobot main `146d847`, flux-action `830f1ce`):

- LeRobot's `lerobot-policy-server` (`--inference.type=remote`) and `--inference.type=rtc` call
  `chunk_inference_spec()`, which raises for `n_obs_steps != 1` (`policies/pretrained.py:323`). The SO-101
  checkpoint uses 8. The server fails only after loading 25 GB.
- flux-action's `serve-robolab` is DROID-only: it needs a `manifest.json`, asserts a 540×640 composite, and calls
  `predict_from_composite`, which asserts the DROID layout.
- The checkpoint keeps its history in processor state: one observation per call, and the last sent command
  anchors the deltas. Forcing it through a once-per-chunk server would space the history about 1 s apart and freeze the anchor.
- Both stacks accept an **explicit window** instead: `state` and `command_history` (8, 6) plus 8 frames per camera,
  of which only indices 0 and 7 are encoded (flux-action `processing/history.py`, LeRobot `processor_flux3.py:360-400`).

Compute per plan is 4 Euler steps × 2 guidance passes = 8 forwards of a ~7B DiT over ~2k tokens. BFL's compiled,
FP8 and distilled paths cover only DROID geometry, so SO-101 runs eager. The estimate is 0.25–0.6 s per plan on an
H100. That is unmeasured: BFL's 79 ms figure is a single-forward games config. Replanning synchronously would
freeze the arm for compute + round trip after every 1.07 s slice.

## Decision

- **Our server, `infra/gpu/so101/server.py`, speaks `so101.v1`** (docs/architecture.md). It is stateless: every
  request carries its own 8-tick window, and the Mac owns the history. It binds 127.0.0.1:8000 behind `ssh -L`.
- **Engine `flux-action` is the default** (decision 03 holds), with these pins:
  - flux-action `830f1ce`
  - `black-forest-labs/flux-3-action-so101` at `c9e13b2aca6a0a472b3ea03fd90cff32d3e85849`
  - encoders `black-forest-labs/flux-3-action-base` at `62878e2925e59b7a89ec14463ce89932624c490d`

  The text encoder is offloaded to CPU, and the released sampling is kept (Euler 4, guidance 3/3, seed 42).
  **Engine `lerobot`** (LeRobot main `146d847`) is the A/B option for `compile_model` and the only path that can load a LoRA adapter later.
- **The client prefetches.** One request is in flight. It is sent every P = 16 ticks, or earlier when the remaining
  actions ≤ worst recent turnaround + 8 ticks. The reply is joined at index d (ticks elapsed since its observation),
  its joint deltas are re-integrated from the last sent command (the reference `select_action` does the same per tick),
  the seam is crossfaded over 6 ticks, and the gripper stays absolute. **Sync mode** (BFL's cadence: execute 32, hold, replan)
  stays in the client as the A/B baseline and fallback.
- **Stall-free budget:** turnaround ≤ 16 ticks (533 ms) keeps every executed action inside the 32 the checkpoint was
  meant to execute. The hard limit is 35 ticks; beyond that a reply is dropped. Turnaround = frame age + upload + queue +
  compute + download.
- **Images:** 256×256 JPEG q95, 4 frames per request (about 100 KB). `png` and `raw` exist to measure what JPEG costs.
- **Order of measurement:** GPU self-test (p50/p95 per plan) → ping over the tunnel → fake-robot replay against the live
  server → dry run on the arm → live.

## Consequences

- Guidance 1/1 would halve compute but changes the released behaviour. Use it only if 3/3 misses the budget, and record it as its own decision.
- The seam has no RTC conditioning, so prefetch can behave differently from the reference loop. The sanity runs (decision 05)
  A/B prefetch against sync before the 10 strict episodes.
- Refines decision 05: between Hanoi steps the queued actions are dropped and the history is kept (LeRobot `set_task`
  behaviour). A full reset happens at an episode start and after an intervention.
- The server runs anywhere with Linux + CUDA + the flux-action env, so track compute or another GPU host can stand in for AWS.
