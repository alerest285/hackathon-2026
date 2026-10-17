# LoRA plan cross-check (T23): the sources agree with decision 09; four gaps would cost time

2026-10-17, alerest285 / claude-code-hanoi (a read-only research agent). Nothing was edited and nothing ran on the GPU.
An independent check of decision 09 and `train/` against the pinned sources.

**Short paths:**

- **FA** = black-forest-labs/flux-action @ 830f1ce
- **LR** = huggingface/lerobot main @ 146d847, with **LS** = `LR/src/lerobot`
- **L061** = the installed LeRobot 0.6.1

Repo paths are relative to the root. The line numbers are as of this commit.

## 1. BFL's LoRA path goes through LeRobot, not flux-action

- **flux-action has no LoRA code.** Its docs send task LoRA to LeRobot's trainer and adapter loader
  (FA docs/so101-lora.md:3-6), and its standalone loader loads strictly with no adapter hook
  (FA src/flux_action/inference/so101.py:190-198).
- **Command** (so101-lora.md:152-158): `python -m lerobot.scripts.lerobot_train --config_path=examples/flux3/lora.json
  --policy.path=$policy_dir --policy.device=cuda --policy.video_vae_id=... --policy.text_encoder_id=... --dataset.repo_id=...
  --output_dir=...`. A new adapter comes from the top-level `peft` block. `--policy.use_peft=true` means "load an existing
  adapter" (161-164).
- **Packages:** `.[training,flux3,peft,diffusion]`, NATTEN `0.21.6+torch2110cu128` and FFmpeg (so101-lora.md:85-108;
  LR docs/source/flux3.mdx:16-29).
- **BFL's pinned training repo is not reachable.** `huggingface/collab-lerobot-nv-bfl-flux3@c027ee36` (so101-lora.md:8-9,
  93-96) returns 404 on GitHub. LeRobot main is the only public route, and its doc says this revision "still needs GPU and
  robot validation" (flux3.mdx:265-269).
- **Data:** LeRobot v3 at 30 Hz, six absolute commands and six measured states, gripper last, the instruction in `task`
  (flux3.mdx:33-35). 0.6.1 writes v3.0 (L061 datasets/dataset_metadata.py:60) and main reads it
  (LS datasets/dataset_metadata.py:64). Our camera keys match the checkpoint's.
- **Nothing is precomputed.** The VAE encodes every 50-frame window while training (LS policies/flux3/modeling_flux3.py:463-475),
  and captions are cached in memory (399-409).
- **Output:** `<output_dir>/checkpoints/<step>/pretrained_model/` (adapter, config, processors), `pretrained_model_ema/`
  next to it, and a `checkpoints/last` link (LS common/train_utils.py:260-285; scripts/lerobot_train.py:1026-1037).
- **0.6.1 has no flux3 policy.** It cannot train this model, but keep recording on 0.6.1: main rewrote the motor bus,
  which could change the degrees the joint map was fitted on (inferred).

## 2. Normalization: map into the checkpoint's convention first (as decision 09 does)

- **Fine-tuning keeps the checkpoint's q01/q99.** For this checkpoint LeRobot loads the saved quantiles and drops
  dataset statistics (LS policies/flux3/processor_flux3.py:70, 75-78), and a test checks that conflicting statistics are
  ignored (LR tests/policies/flux3/test_flux3_lora.py:512-526). Other statistics would need a new exported base
  (LR examples/flux3/export_base.py:92-124).
- **Why map rather than retrain the statistics:**
  - The quantiles "assume its joint order, units and calibration" (flux3.mdx:181-182).
  - They are VoicAndrei/so100_kitchen's (FA configs/so101/community_corpus.json:316 = console/mapping.py:21-22).
  - Lift and wrist roll run reversed on our arm, and min/max rescaling cannot reverse a direction
    (processor_flux3.py:276-283).
  - The same frozen map at training and at inference turns a small map error into a fixed offset that the LoRA and
    the fully trained heads can learn around (inferred).

## 3. Recipe, memory, time

- **BFL preset** (LR examples/flux3/lora.json):
  - rank and alpha 32
  - LR 1e-4, with the heads at 5×
  - AdamW, weight decay 0, gradient clip 1, no warmup or decay
  - BF16, batch 2 × accumulation 4, 10,000 microsteps (2,500 updates), a save every 500
  - EMA 0.999, gradient checkpointing, caption dropout 0.1
- **What trains:** LoRA on every q/k/v/attn_out/mlp_in/mlp_out. The robot heads train in full (modeling_flux3.py:243-249).
- **Not published:** memory and speed. Estimate 35–45 GB and 0.4–0.8 s per microstep, so 2,500 updates take about
  1.1–2.2 h (inferred). `train.sh probe` measures them.
- **Gap 1: the held-out eval runs over the whole split** at every eval step without `--max_eval_samples`
  (LS configs/train.py:146-149). Evaluating every 125 updates would add 1–2 h (inferred). Add `--max_eval_samples=64`.
- **Gap 2: loader memory.** Frames arrive as float32, about 0.37 GB per sample (inferred), with 12–16 workers × 4
  prefetched batches on top of the dataset in /dev/shm. Add `--dataset.return_uint8=true --prefetch_factor=2`.
- **Stop the server while training.** Memory may fit, but saturating the GPU breaks the 533 ms budget (decision 06).

## 4. LeRobot route, not flux-action

flux-action only offers full fine-tuning: fresh heads, statistics recomputed from your data, and a 512-window batch
on 8 GPUs (FA so101-lora.md:186-237). That is hundreds of H100-hours (inferred).

## 5. Serving the adapter

- **Command:** `serve.sh --engine lerobot --adapter <checkpoint>/pretrained_model`. The base comes from the adapter's
  config, and local encoder paths pass the revision check (engines/base.py).
- **Gap 3, fixed here:** serve.sh only looked for `~/lerobot-flux3`, while train/setup.sh builds
  `/workspace/lerobot-flux3`. serve.sh now falls back to /workspace.
- **Adapter loading has never run on real weights,** and the probe saves no checkpoint. Self-test the first real
  checkpoint (step 500) early.

## 6. Demos

- **BFL publishes no demo count.**
- **Each training window needs 8 frames before and 42 after** (LS configuration_flux3.py:292-299), so an N-frame
  episode gives N − 50 windows.
- **What to record:** the 7 solution steps (6 prompts; steps 1 and 7 share one), with prompts exactly as in
  `eval/config/hanoi.toml`. At least 10, ideally 15–20 per step (inferred).
- **Episode shape** (inferred): 6–15 s, start moving within about 0.5 s, and end with about 2 s of stillness after
  the release. That stillness teaches the rest the eval's auto end needs.
- **Gap 4:** prepare.py should require `run_mode == live and valid is True`, since rows from before the sidecar fields
  pass today (codex-ft-supervisor raised the same point).
- **teleop.sh cannot add a second session to the same dataset** (no `--resume`, so it fails with FileExistsError at
  L061 datasets/dataset_metadata.py:832). Episodes default to 60 s with a 60 s reset (L061 configs/dataset.py:35-37).

## 7. Risks, ranked, each checkable in under 10 minutes

1. **Time budget.** Check with `train.sh probe`, plus held-out windows ≈ 10 % × (length − 50).
2. **Loader memory.** Check `df -h /dev/shm; free -g` during the probe.
3. **Unvalidated LoRA and adapter serving.** Run the probe, then a `--selftest` on the first checkpoint.
4. **The experimental map is now part of the model.** prepare.py's coverage table should show under 10 % of states
   outside the range, and the manifest should match the map eval loads.
5. **Legacy sidecars admitted** (gap 4).
6. **EMA lag on a short run.** At 0.999 about 8 % of the start weights remain after 2,500 updates (inferred), and BFL
   calls SO-101 EMA results "mixed" (flux3.mdx:93-96). Evaluate the raw adapter first.
7. **AV1 decoding on the pod** (0.6.1 records AV1 by default). The first probe batch covers it; the fallback is
   `--dataset.video_backend=pyav`.
