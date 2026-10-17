# Training restart and hotfix review — 2026-10-17

Reviewer: `alerest285/codex-arm-review`. Training implementation: `claude-train`;
pod recovery: `claude-code`; sole GPU launcher: `saadl27/claude-code`.

The replacement H100 pod is training after a successful save/resume/adapter smoke
test. Saad subsequently stopped the qmap run and selected native LeRobot units
with HF-sim normalization; the replacement run is healthy with checkpoints
through step 3,500 at 22:41Z. This record covers the restart and training hotfixes;
it does not establish task success on the physical arm or completion of training.

## Environment and data

- Pod `wog1l8ijpl3jp5`, persistent volume `xdg2i10kmu`, SSH
  `root@<pod-host>:<ssh-port>`. The previous pod stopped when its balance ran out;
  its host had no GPU available for restart.
- LeRobot pin `146d8474932c8b566717eb2738ba459337ab71c4`, torch
  `2.11.0+cu128`, natten `0.21.6`; CUDA setup completed successfully.
- Initial qmap dataset `/workspace/datasets/lora_so100_pick_cube_v1`: 82 episodes,
  51,922 frames at 30 Hz; 73 train / 9 held out, 45,874 training frames.
- Exact instruction: `Put the green objects at the right of the black line and
  the orange at the left of the black line`.
- Manifest SHA-256:
  `ce9a7ad2da8ea4b29660edc89789f0278cd32a2f6ed3a4cf2c02a691f998a699`.
- Frozen qmap SHA-256:
  `e0cab00787346a866d961d11a16dbce53eda8c992441493c883a6f157128131f`.
  The bundled file matches its hash; robot ID `my_follower_arm`, calibration
  `d6d7ccc82ab2153e5aef742fcf2d96c903fe9677ad321f19a85606c4c9d619fa`.
  The prepared dataset was transferred without remapping. This verifies the
  selected data contract, not the physical accuracy of the map.

## Hotfixes checked

- `bf19469`: parse the pinned trainer's timing fields independently of their
  order, include preprocessing in the compute ETA, expose loader worker counts,
  warm the environment file cache, and enable W&B metrics. Preparation rejects
  tasks whose holdout would leave no training episode, already prepared inputs,
  and the former unproven `--no-map` route.
- `3c0b7d7`: disable checkpoint artifact uploads when W&B is enabled.
- `6397217`, followed by `09d2bb9`: inherit the W&B key by environment name
  instead of including its value in tmux arguments; configure and create the
  tmux session in one invocation so an empty server cannot exit between calls.
  Fresh and existing tmux servers were tested with dummy credentials.
- `3dca2fb`, `28c25d1`: apply explicit worker/W&B resume overrides and refuse
  missing or corrupt provenance maps.
- `0ecf67a`: enabling W&B on an untracked checkpoint or changing destination
  supplies a new explicit run ID with `resume=allow`; the same tracked
  destination retains its saved ID.
- `bdb048d`, `867349e`: select a complete alternate base with `--policy DIR`,
  record that base and encoder directory, hash its configuration and normalizer
  files, and refuse resume if those files change or disappear. This enabled
  Saad's HF-sim normalization restart.
- `2f1cd8f`: reject `--policy` on resume explicitly; the saved base must be
  retained. Existing, different, and missing override paths all reject before
  launch, while normal resume and fresh-run overrides still pass.

Local checks passed: seven preparation tests; real pinned config parsing;
launcher argument, timing, worker and provenance checks; six W&B resume cases
through the actual pinned logger with an offline SDK stub; CPU optimizer,
scheduler and RNG save/resume; and tiny real-PEFT raw/EMA exports loaded through
the production inference engine with finite `(42, 6)` outputs. GPU smoke below
provides the actual-model check.

The first-microstep scheduler warning was reproduced against the pinned update
function and Accelerate. With this constant-LR recipe, eight microsteps produce
exactly two optimizer and EMA updates and match an effective-batch-eight
reference. Learning rates remain `1e-4` and `5e-4`; no scheduler patch is needed
for this recipe.

## GPU smoke evidence

Run `lora_so100_pick_cube_v1-b2x4-1017-195215-f2f6`, batch 2 × accumulation 4,
three loader workers. Evidence is in `/workspace/train-logs/saad-smoke.out` and
the run's `.log` under `/workspace/train-logs/`.

- First batch: about 116 seconds after training started, versus 8.6 minutes on
  the previous pod with 12–16 workers. Steady update time about 0.90 seconds per
  microstep, data wait 0.004 seconds.
- First phase: 16 microsteps, trainer exit 0, 406 seconds wall time, peak 33,857
  MiB. Training loss 0.209 at step 4 and 0.166 at step 8.
- Checkpoints saved at 8 and 16; held-out loss 0.1045 and 0.1244 respectively.
- Resumed successfully to step 24, saved again, held-out loss 0.1374.
- Production server selftests passed on both `000024/pretrained_model` and
  `000024/pretrained_model_ema`, followed by the wrapper's `smoke PASS`.

At this LeRobot pin, the dataset `eval_loss` uses **raw** weights. The
`ema.use_for_eval` flag applies to environment rollouts, not dataset loss. Both
raw and EMA adapters are saved. The synthetic smoke instruction is a loader and
finite-output check; deployment must use the dataset's exact sorting caption
and frozen qmap.

## Initial qmap run (stopped by Saad)

Started at 20:08:59Z from repository commit `d90c6fb`, six loader workers:

- Run: `lora_so100_pick_cube_v1-b2x4-1017-200859-092d`.
- W&B: <https://wandb.ai/slahlou-ethz/flux3-so100-lora/runs/e8co8cvy>.
- Log: `/workspace/train-logs/lora_so100_pick_cube_v1-b2x4-1017-200859-092d.log`.
- Output: `/workspace/outputs/lora/lora_so100_pick_cube_v1-b2x4-1017-200859-092d`;
  frozen contract in the sibling `.provenance` directory.
- 10,000 microsteps = 2,500 optimizer updates; raw/EMA checkpoint every 500
  microsteps = 125 updates. W&B artifact uploads disabled.
- Stopped by Saad at microstep 4,272 to replace its normalization setup.
  Checkpoints `000500` through `004000` remain available; this was an intentional
  stop, not a training exception.

## Current HF-sim run

Started at 21:30:30Z from commit `c1716bd`:

- Run: `lora_so100_pick_cube_hfsim_v1-b2x4-1017-213030-6ec3`.
- W&B: <https://wandb.ai/slahlou-ethz/flux3-so100-lora/runs/it5chf78>.
- Log: `/workspace/train-logs/lora_so100_pick_cube_hfsim_v1-b2x4-1017-213030-6ec3.log`.
- Output: `/workspace/outputs/lora/lora_so100_pick_cube_hfsim_v1-b2x4-1017-213030-6ec3`.
- Base: `/workspace/ckpt/flux-3-action-so101-hfsim-norm`, original pinned weights
  with state/action normalizer files replaced by the selected HF-sim statistics.
- Same 82 episodes and caption, now native LeRobot joint values with a stamped
  identity map. Manifest SHA-256
  `fa5615b4c90132c3c198c34be9d189ec3f0bdbc118488a6e005bc8ef81a36a92`;
  identity map SHA-256
  `2985122800c29787db85d5580ef6e88321f01d64a9e852a2051bf970dd42c92e`.
  The bundled map hashes correctly and has equal source/destination quantiles
  for every joint, with Saad's current calibration identity.
- At 22:41:25Z: step 3,800 / 10,000, loss 0.149, update time 0.932 seconds,
  data wait 0.004 seconds; latest checkpoint `003500`.
- Checkpoint `003500` inspected without loading a GPU model: raw and EMA each
  contain 642 adapter tensors (471,775,248 bytes), point to the HF-sim base, and
  include their processor JSON files. All three saved normalizers have exactly
  the same tensor names, dtypes, shapes, and value bytes as the HF-sim base, in
  both variants. Both saved configs record W&B run `it5chf78` with artifacts
  disabled. Optimizer state/groups, RNG, scheduler, training step, and EMA state
  files are present.
- Frozen `run.json` includes the alternate base path and seven config/processor
  hashes; all seven were independently checked against the files on the pod.
  Both source and frozen manifest copies match the recorded hash. Offline
  launcher tests confirm that altered/deleted normalizers, altered config, and
  a missing base stop resume before launch. Dataset coverage metadata still
  references the original snapshot's
  quantiles; it is not coverage against the HF-sim normalizers. Loss values across
  the two runs have different normalization scales and should not be ranked
  directly.

Saad owns launch and monitoring. He explicitly chose `--share` for concurrent
serving tests; this replaced the earlier exclusive-GPU plan. No competing
training or inference job was started by the reviewer. Deployment of the current
adapter must retain its HF-sim base/processors and use native units without the
old qmap (or use the equivalent identity map).
