# Erfan agent: training and inference infrastructure handoff

Requested by Ale on 2026-10-18 (Zurich). Coordinator: `alerest285/codex-ft-supervisor`.
Recipient: `ErfanRzt`, proposed agent name `erfan-infra`, Runpod email `<redacted-email>`.
Snapshot updated 2026-10-18 02:23 UTC; recheck runtime state before any changes.

## Access and first response

- GitHub API confirms `ErfanRzt` has **write** access to `alerest285/hackathon-2026`; no repository invitation is pending.
- There is no Erfan agent chat file yet. From your own checkout, run `scripts/setup.sh erfan-infra`, then `scripts/sync.sh`, and keep `scripts/listen.sh` running.
- Reply with `scripts/say.sh @alerest285/codex-ft-supervisor 'erfan-infra online; <public-key location or public key>; <SSH validation result>'`.
- GitHub currently publishes no SSH keys for `ErfanRzt`. Generate or reuse a key on **your own machine**, keep its private part there, and add the public key to your GitHub profile. The infrastructure owner has set up a watcher to append keys from that authenticated profile to the pod without restarting it. A pasted repo-chat name alone does not authenticate a key; keys supplied another way need identity verification. Do not send API tokens or private keys through chat, issues, or git.
- SSH access permits setup and operation inside the existing pod. Cloud management is separate: the account owner reports Runpod organization API 403 (organizations not enabled), and the available Runpod integration has no team-invite action. A Runpod team invitation for `<redacted-email>` with **Dev** role is still outstanding. An old README saying “invite pending” is not evidence that an invitation was sent.

## Current infrastructure

| Item | Current value |
|---|---|
| Active pod | `wog1l8ijpl3jp5`, H100 80 GB, EU-FR-1 |
| SSH | `ssh -p <ssh-port> root@<pod-host>` |
| Persistent storage | Network volume `xdg2i10kmu`, mounted at `/workspace`, 250 GB |
| Repository on pod | `/workspace/hackathon-2026` (read-only GitHub deploy key) |
| Training/adapter Python | `/workspace/lerobot-flux3/.venv/bin/python` |
| Zero-shot flux-action environment | `/workspace/flux-action` |
| Pinned paths/environment | `/workspace/train-env.sh`; HF cache `/workspace/hf` |
| Current training dataset | `/workspace/datasets/lora_so100_pick_cube_hfsim_v1` |
| Current run | `lora_so100_pick_cube_hfsim_v1-b2x4-1017-213030-6ec3` |
| Training log | `/workspace/train-logs/<run>.log` |
| Checkpoints | `/workspace/outputs/lora/<run>/checkpoints/` |
| Provenance | `/workspace/outputs/lora/<run>.provenance/` |
| Current base | `/workspace/ckpt/flux-3-action-so101-hfsim-norm` |
| Serving session | `so101-serve`, port 8000 over SSH tunnel |

The old pod `rmhs4cu2cejemv` and port <ssh-port> are obsolete. The replacement network volume persists across hosts; `/dev/shm` and the container root do not.

The HF-sim run **completed at 00:23 UTC on October 18**, 10000 microsteps / 2500 optimizer updates, final raw held-out eval loss 0.0931, `EXIT 0`, last checkpoint 010000. This was independently checked on the pod at 02:23 UTC. The `so101-serve` session remains running. Do not duplicate the completed run or disrupt the active server. Coordinate new jobs, restarts and environment changes with the owners below. Earlier training shared the GPU by Saad's explicit decision; latency measured during that overlap is not an exclusive-GPU benchmark.

## First checks after SSH is authorized

Run read-only checks and post the result:

```sh
ssh -p <ssh-port> root@<pod-host> \
  'hostname; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; tmux list-sessions; df -h /workspace /dev/shm'
```

Confirm the completed training log ends with `EXIT 0`, the selected checkpoint contains raw and EMA adapters plus `training_state`, and `/workspace/train-env.sh` points at existing paths. Check for any newer run before changes. Do not print credential files or environment values containing tokens.

## Existing setup and operations to reuse

Read `AGENTS.md`, `docs/project.md`, `docs/architecture.md`, decisions 06/08/09/10 and the relevant task before changing implementation. Claim narrow paths with the existing lock protocol; do not edit `train/**` or serving code while another agent holds them.

On a **fresh, designated training host**, `train/setup.sh` creates the pinned training environment, downloads the pinned model/encoders and writes `/workspace/train-env.sh`. `infra/gpu/bootstrap.sh` sets up flux-action zero-shot inference. Both are already installed on the current pod. Use the repository scripts rather than inventing another environment.

Training modes are `probe` (performance only), `smoke` (save → resume → raw/EMA inference), `run`, and `resume`. The reviewed baseline is batch 2, accumulation 4, effective batch 8, rank/alpha 32, bf16, EMA 0.999, 2500 optimizer updates. Launch after coordinating the GPU slot; pass `--policy /workspace/ckpt/flux-3-action-so101-hfsim-norm` for the current normalization. W&B accepts `--wandb ENTITY/PROJECT`, with credentials provided through the environment and checkpoint artifact upload disabled. `train.sh resume <run>` validates provenance and restores the dataset to its recorded staging path before resuming.

Prepared datasets must include `prepare_manifest.json` and copied, hash-named maps. Raw recordings use native degrees. Ordinary console/eval sources must be live, valid, successful and complete; sidecarless leader demonstrations must be explicitly declared `--teleop`. Use at least two retained episodes per task with the default holdout. Never prepare an already prepared dataset again.

## Inference and coordinate contract

The current HF-sim run uses **native joint coordinates and an identity map**. Its adapter loads its named HF-sim base and the processors saved with the adapter. Do not apply the original qmap to it. Old qmap adapters require their original qmap; these are separate experiments.

After the serving owner agrees to a switch, the command shape is:

```sh
bash infra/gpu/so101/serve.sh --engine lerobot \
  --adapter /workspace/outputs/lora/<run>/checkpoints/<step>/pretrained_model \
  --compile --warmup-task '<exact training caption>' --log-dir /workspace/so101-logs
```

First self-test an adapter with `--selftest --n 5` in a coordinated GPU slot. Compare raw and EMA under the same protocol. Check the startup/hello normalization against the adapter's processors. `--norm-stats` can override zero-shot normalization but only asserts equality for an adapter.

Current exact caption:

> Put the green objects at the right of the black line and the orange at the left of the black line

Tunnel from a laptop with `ssh -N -L 8000:127.0.0.1:8000 -p <ssh-port> root@<pod-host>`. Robot trials remain with the hardware operator and native limits/calibration checks. `eval/run_task.py` records attempts, successes and completed sorting steps; its current HF-sim config uses native coordinates. Do not treat a server self-test or lower loss as physical task success.

## Owners and completion evidence

| Scope | Contact |
|---|---|
| Pod/account bring-up | `alerest285/claude-code` |
| Training implementation, T23 | `alerest285/claude-train` |
| Active run and dataset decisions | `saadl27/claude-code` |
| Serving, robot loop and evaluation | `alerest285/claude-code-hanoi` |
| Calibration/normalization | `saadl27/saad-calib` |
| Independent review / SSH onboarding coordination | `alerest285/codex-ft-supervisor` |

Onboarding is complete when Erfan's agent acknowledges, its public key is installed, it independently verifies SSH/environment access, and it posts an agreed scope with the current owners. New-pod creation/start/stop additionally requires confirmed Runpod Dev access. No new pod, paid job, or service restart has been performed by this handoff.

Existing audit: [LoRA supervision and GPU evidence](2026-10-17-lora-supervisor-review.md).
