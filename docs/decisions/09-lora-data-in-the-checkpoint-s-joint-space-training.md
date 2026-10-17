# 09. LoRA data in the checkpoint's joint space; training never shares the live GPU

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285 (claude-train)

## Context

Milestone 4 adapts `flux-3-action-so101` @ `c9e13b2a` with a LoRA through LeRobot `146d847` (decision 03 and 06),
following BFL's `docs/so101-lora.md`. Two facts from the source shape it:

- The LoRA path keeps the checkpoint's own normalization. For this checkpoint (`conditioning="history"`), LeRobot
  drops the dataset-stat overrides and loads the history quantiles saved with the checkpoint
  (`processor_flux3.py`, `make_flux3_pre_post_processors_from_pretrained`). Our recordings (console `--record`,
  `teleop.sh`, eval) are in the arm's native degrees and gripper 0..100, while inference sends model-space values
  through `joint_mapping.json` (console/mapping.py). Trained as recorded, every joint would be normalized against the
  wrong range.
- Each training sample decodes 50 frames per camera (8 history ticks + 42 future frames) and runs a 7B DiT forward
  and backward with activation checkpointing. Peak VRAM and s/step are not published. The serving pod already
  carries `so101-serve` (17–26 GB) and must answer within the 533 ms stall-free budget (decision 06).

## Decision

- **Every training set goes through `train/prepare.py`.** It converts `observation.state` and `action` with
  `JointMap.to_model`, using the map of the arm that recorded each source, rewrites the per-episode and global
  stats, merges sources (model space is shared across arms) and writes `prepare_manifest.json` with each map's sha256,
  `robot_id` and calibration hash. `train/train.sh` refuses a dataset without that manifest.
- **An adapter is served behind the same map it was trained with.** A refit map means prepare and train again.
- **Admission fails closed:** runs (`--source`) need a sidecar row with `run_mode` exactly `live` and `valid` exactly
  true, verdict success, no interventions, no writer or encoder drops. Every episode needs ≥ 50 frames and both mp4s
  covering every row exactly (checked on the files). Leader-arm demos have no sidecar and must be declared `--teleop`.
- **Provenance:** each dataset carries its map files (named by sha256); each run gets a sibling `<run>.provenance/`
  before the first step. `train.sh resume` restages only a dataset whose manifest hash matches.
- **Gate before a full run:** `train.sh smoke` (save → resume → raw and EMA adapters loaded by the server) must pass
  once per setup. `probe` measures speed only.
- **Recipe:** `train/lora.json` = BFL's preset (rank/alpha 32, LR 1e-4 / heads ×5, EMA 0.999, BF16, checkpointing,
  effective batch 8, 2,500 updates) with `eval_split` 0.1 instead of 0.2 (small sets), `max_eval_samples` 64 (else
  every eval runs the whole held-out split), uint8 frames and prefetch 2 (loader RAM), 12–16 workers, wandb off. `--batch` changes only speed: accumulation, steps, save and eval scale to keep the effective batch.
- **Training never runs on a GPU that serves a live arm.** `train.sh` refuses while another process holds the GPU,
  unless `--share`. Train on a second pod, or on the serving pod while no one drives the arm. Datasets are staged on
  `/dev/shm`, because `/workspace` on Runpod is a network volume.
- **Weights are fetched into `HF_HOME=/workspace/hf` at the decision-06 pins**, so the base path an adapter records
  is the same on any pod set up by `train/setup.sh`, and `server.py --engine lerobot --adapter` loads it unchanged.

## Consequences

- Probe numbers (peak VRAM, update and data time per microstep, ETA) are owed: `train.sh probe` on an idle GPU.
  They decide `--batch` and whether a second pod is worth its $3.99/h.
- The joint map is now part of the model: lose or refit it and the adapter is stale. Keep the map file with the run.
- The serve path for adapters (`--engine lerobot --adapter`) is untested on GPU until the first checkpoint exists.
