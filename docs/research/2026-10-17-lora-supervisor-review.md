# LoRA preparation and inference supervision

Reviewer: alerest285 / codex-ft-supervisor. Implementation owner: claude-train, T23.
Review started 2026-10-17. This review uses a separate checkout and does not change implementation files or launch jobs on the shared GPU.

## Status

CPU review of `224a900` and `bf19469` passed: **7 preparation integration tests, 21 inference protocol tests**, and independent negative admission, short-video, and trainer-exit reproducers. These results do **not** establish GPU checkpoint reload, resume, or robot task success. GPU verification is in progress on the replacement pod.

At 19:50 UTC the replacement pod `wog1l8ijpl3jp5` had `lora_so100_pick_cube_v1`: 82 episodes, 51,922 frames, 28.8 minutes, one sorting-task caption. The recipe's 0.1 holdout gives 73 train / 9 eval episodes. Its manifest identifies map `e0cab00787346a866d961d11a16dbce53eda8c992441493c883a6f157128131f` and calibration `d6d7ccc82ab2153e5aef742fcf2d96c903fe9677ad321f19a85606c4c9d619fa`. Preserve this map and the exact caption for inference; the dataset's historical name is misleading.

The original pod became unavailable after a balance outage and could not restart on its host, according to the infrastructure owner. That owner created the replacement with a persistent network volume. Saad's agent owns the smoke/full-run launch; this reviewer makes read-only GPU checks. At 19:50 the H100 was idle and setup was completing. Serving remains stopped for training.

## Findings sent to the owner

| Finding | Evidence | Resolution |
|---|---|---|
| Training and serving must use the same joint convention | Recorder stores native values; `MappedRobot` sends model values; pinned processors retain checkpoint statistics. | `prepare.py` maps both state and commanded action. Decisions 09 and 10 describe the contract. Map artifacts travel with data and runs. |
| Recorder counters alone cannot prove video coverage | Arm reviewer reproduced 12 numeric rows, scene 6 frames, wrist 12 frames, with the old dropped-frame counter still zero. | Preparation checks video spans and packet counts as well as mode, validity, drops, and interventions. Packet coverage does not prove full-file decoded-image integrity. |
| Missing provenance admitted non-demonstrations | Missing run mode/validity and missing sidecar rows were accepted. | `224a900` rejects these by default and requires explicit `--teleop` for sidecarless demonstrations. Independent negative checks pass. |
| Unknown/null sidecar values still accepted | Unknown mode with valid=true, or live mode with valid=null, passed an intermediate revision. | `224a900` requires exact live mode and Boolean true by default; independently verified. Diagnostic overrides remain explicit. |
| One missing video frame passed validation | Actual H.264 scene 59 frames / wrist 60 frames with 60 numeric rows returned no shortfall. | `224a900` rejects this reproducer with exact integer-span and file-coverage checks. |
| Training failures could look successful | Detached launcher reported tee's status; probe did not return the trainer status. | `224a900` preserves trainer status. A controlled detached trainer exit 7 produces EXIT 7, including after `bf19469`. |
| Resume referenced volatile storage | Saved config points at /dev/shm, which disappears on restart. | `224a900` restages from persistent storage to the recorded path, checks manifest and copied-map hashes. This does not hash every dataset payload file. GPU resume remains to be executed. |
| Tiny tasks could lose every training example | Pinned LeRobot holds out ceil(n * split) episodes per task. | `bf19469` checks this during preparation; regression test passed. Explicit --eval-split must still match the trainer recipe. |
| Conversion could be skipped or repeated without provenance | --no-map could label raw input as model space; prepared input could be mapped twice. | `bf19469` removes --no-map and rejects sources with prepare_manifest.json; regression test passed. |
| Provenance could be missing or overwritten | Early watcher waited only 600 s for a trainer-created output directory; run names could collide. | `224a900` creates immutable sibling provenance before launch, copies maps, uses seconds plus random run IDs, and refuses collisions. |
| Performance probe did not test checkpoint lifecycle | Probe disables saving. | `224a900` adds a separate save → resume → server selftest workflow for raw and EMA packages. Runtime completion pending. |
| Timing parser assumed the wrong metric order | Pinned trainer prints data/prep/update fields in a different order. | `bf19469` parses fields independently and includes preparation time. |
| W&B launch omitted requested artifact disabling | Pinned WandBConfig.disable_artifact defaults false. | Fixed in `3c0b7d7`; reviewed at `09d2bb9`. Run and explicit resume logging disable artifact upload. |

Additional owner/arm-reviewer fixes through `09d2bb9`: W&B credentials are transferred to tmux by environment-variable name, not command-line value; a fresh-server lifecycle regression was caught and fixed using one tmux command sequence. The owner tested fresh and existing servers with dummy credentials. Explicit resume accepts worker/W&B overrides, and `28c25d1` refuses missing provenance maps. These fixes were source-reviewed here; the runtime tmux credential test belongs to the other reviewers.

## Verification performed

- Inference protocol suite: `python -m pytest infra/gpu/so101/tests -q`, **21 passed in 1.95 s** against the `bf19469` checkout. Uses temporary localhost WebSockets and a fake engine; not a GPU/adapter reload test.
- Preparation suite at `bf19469`: `python -m pytest train/tests -q`, **7 passed in 63.21 s**. Uses real LeRobot 0.6.1 datasets, mapping, filtering, merging, rewritten statistics, and video decoding.
- Independent audit: [check_lora_data_admission.py](check_lora_data_admission.py), **PASS**. Tests admission separately from video validation: dry/fake/damaged, missing-row, missing-mode, unknown-mode and null-validity cases. Then creates actual 59/60-frame H.264 files and checks coverage.
- Detached launcher extraction with a controlled failing trainer: **PASS**, EXIT 7 preserved. Temporary harness /private/tmp/ft_supervisor_launcher.py.
- Read-only H100 process/memory and dataset-manifest inspection. No server restart, training launch, installation, or hardware operation by this reviewer.

Saad's agent reports the original-pod probe reached 40 microsteps, exit 0, 33.2 GB peak, steady 0.87–0.95 s/microstep, loss 0.209 → 0.136. Its smoke reached the first save/eval before the outage. Those logs were left on the inaccessible host and were not independently inspected; they do not establish resume or inference reload.

## Evidence still required

1. Completed GPU smoke: full raw and EMA packages, fresh-process loading through the actual lerobot server engine, and finite (42, 6) output.
2. Resume from saved optimizer/EMA state, including recovery after volatile dataset storage is lost.
3. First full-run checkpoint and finite held-out evaluation; continued exclusive GPU use during training.
4. Physical validation of the chosen map and controlled adapter evaluation on the robot. A low training loss or a server selftest does not establish task success.

The pinned [LeRobot FLUX3 guide](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/docs/source/flux3.mdx) is the upstream reference for processors, raw/EMA packages, episode holdout and microstep accounting. Local decisions 06, 09 and 10 govern the project's pins, units and GPU sharing.
