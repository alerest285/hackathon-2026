# train/: LoRA on flux-3-action-so101 from our demos (milestone 4)

The recipe is BFL's (`flux-action` docs/so101-lora.md): LeRobot's trainer, rank/alpha 32 LoRA, heads trained in full,
EMA 0.999, BF16, activation checkpointing, effective batch 8, 2,500 optimizer updates. LeRobot is pinned to
`146d847`, the same commit the server's `lerobot` engine uses (decision 06). Why the data goes through our joint map
first, and why training gets its own GPU: decision 09.

## The loop

| # | Where | Command | Result |
|---|---|---|---|
| 1 | Mac with the arm | `console/teleop.sh record ... --task "<exact prompt>"` (leader arm), or console runs with `--record` | LeRobot datasets in the arm's own degrees |
| 2 | Mac | `uv run --python 3.13 --with "lerobot[dataset]==0.6.1" python train/prepare.py --teleop <demos> [--source <eval or console runs>] [DIR=<map> per arm] --mapping ~/hackathon-data/joint_mapping.json --out ~/hackathon-data/lora_v1` | one dataset in model space + `prepare_manifest.json` |
| 3 | Mac | `rsync -a --no-owner --no-group ~/hackathon-data/lora_v1/ root@<pod>:/workspace/datasets/lora_v1/` (the network volume refuses chown) | dataset on the GPU host |
| 4 | GPU host, once | `rsync -a --no-owner --no-group -R train/ infra/gpu/so101/ root@<pod>:/workspace/claude-train/` then `bash /workspace/claude-train/train/setup.sh` | LeRobot env with training extras, pinned weights, `/workspace/train-env.sh` |
| 5 | GPU host | `bash train/train.sh probe lora_v1 --batch 2`, then `--batch 4` | **performance only**, saves nothing: s/step, data wait, peak VRAM, hours and $ |
| 5b | GPU host, once per setup | `bash train/train.sh smoke lora_v1 --batch <B>` | save → resume → raw and EMA adapters loaded by the server's `--selftest`; must PASS before a full run |
| 6 | GPU host | `bash train/train.sh run lora_v1 --batch <fastest that fits>` | tmux `lora-<run>`, checkpoints every 125 updates, `<run>.provenance/` beside it |
| 7 | serving host | `bash infra/gpu/so101/serve.sh --engine lerobot --adapter <run>/checkpoints/<step>/pretrained_model_ema` | the adapter behind the same `so101.v1` server |
| 8 | Mac | the console or eval, **same joint map, same prompts, same protocol** as the zero-shot baseline | the milestone-4 comparison |

## What makes a good dataset

- **Exact prompts.** The task string is the caption the adapter learns. Type it the same way at inference, character
  for character. `prepare.py` prints the per-task counts; `--hanoi-only` keeps only prompts `eval/config/hanoi.toml` can issue.
- **One arm, one map per source.** Every source is converted with the `joint_mapping.json` of the arm that recorded it
  (`--source DIR=MAP.json`). The manifest records each map's sha256, `robot_id` and calibration hash. Serve the adapter
  behind that same map. A refit map after training means re-running `prepare.py` and training again.
- **Same cameras as at test time.** Scene camera fixed and overlooking the workspace (left tile), wrist camera on the
  gripper (right tile). Do not move the scene camera between demos and evaluation.
- **At least 50 frames (1.7 s) per episode.** One window is 8 history ticks + 42 future frames; shorter episodes are dropped.
- **Only clean episodes, fail closed.** Leader-arm demos are declared `--teleop`; console and eval runs are `--source` and need a sidecar row with `run_mode` and `valid`, or they are dropped. `prepare.py` drops dry/fake console runs (the arm did not follow the commands), failed or
  unlabeled runs (`--keep success`, default), episodes with writer or encoder drops, and any episode whose scene or
  wrist video does not cover all its rows (checked on the mp4 itself, not only the counters).
- **Teleop like the policy should act.** Start from rest, move at a steady pace, one move per episode, finish at rest.
  The model predicts command deltas, so jerky demos teach jerky motion.

How many demos is not published for SO-101 LoRA. Start small (the 7 Hanoi solution moves, or one cube task), train,
and test a checkpoint on the arm before recording more: the probe gives the hours, so the first adapter is cheap.

## Picking a checkpoint

Every save holds `pretrained_model` (raw) and `pretrained_model_ema`. Try the raw adapter first: EMA 0.999 still carries
about 8 % of the starting weights after 2,500 updates, and BFL calls EMA on SO-101 "mixed". Then the EMA one under the
same protocol. The logged `eval_loss` (64 held-out samples, 10 % of episodes per task) uses the RAW weights at this
pin (lerobot_train.py:987-1007); `ema.use_for_eval` only affects simulator rollouts (:1061), which we do not run. It is a sanity signal, not a ranking: offline error does not establish task success. First checkpoint to self-test in the
server: `serve.sh --engine lerobot --adapter <run>/checkpoints/000500/pretrained_model --selftest --n 5`.

## Resume

`bash train/train.sh resume <run>`. The dataset is staged in RAM, which a pod restart wipes, so resume first checks
that `/workspace/datasets/<name>/prepare_manifest.json` still has the sha256 recorded in `<run>.provenance/run.json`,
checks the saved maps against their hashes, restages to the same path the run was trained from, then continues from
`checkpoints/last` (optimizer and EMA state restored). A changed dataset is refused: train a new run instead.

## Provenance

Every run (not probes) gets `/workspace/outputs/lora/<run>.provenance/`, written before the first step and never moved:
the dataset's `prepare_manifest.json`, the joint-map files it was converted with (named by sha256), and `run.json`.
Serve an adapter only behind the map in its provenance (decision 10).

## Speed levers, in the order to try them

1. `--batch 4` (or 8) with the same effective batch: fewer, larger microsteps. The probe shows whether it fits.
2. More loader workers, if the probe prints that data loading is a bottleneck. Each sample decodes 50 frames per camera
   (kept uint8 until the policy packs them); the dataset is staged on `/dev/shm` because `/workspace` is a network
   volume. If AV1 decoding fails or crawls with torchcodec, add `--dataset.video_backend=pyav` to the `.cmd` file.
3. Evaluation costs about 110 s the first time (its loader starts) and about 20 s after that, so roughly 8 min per
   full run (measured 17.10, smoke on wog1l8ijpl3jp5). Not worth tuning.
4. Fewer updates (`--updates 1250`) for a first adapter on a small dataset, then a full run on the one that works.

## Tests

`uv run --python 3.13 --with "lerobot[dataset]==0.6.1" --with pytest python -m pytest train/tests -q` (Mac, about a minute):
writes small datasets with the real LeRobot, then checks the map, the filters, the merge and the rewritten stats.
