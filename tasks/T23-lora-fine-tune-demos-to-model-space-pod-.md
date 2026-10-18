---
id: T23
title: LoRA fine-tune: demos to model space, pod training, serve the adapter
status: todo
owner: 
depends: 
paths:
  - train/**
---

## Goal

Fine-tune flux-3-action-so101 with BFL's LoRA recipe on our own demos, with the data converted into the checkpoint's
joint space by the same map used at inference, and serve the adapter through the existing so101.v1 server (decision 09).

## Done when

- [x] CPU, Mac: `uv run --python 3.13 --with "lerobot[dataset]==0.6.1" --with pytest python -m pytest train/tests -q`
      passes (map conversion per arm, merge, rewritten stats, fail-closed admission, exact video coverage).
- [x] GPU host env: `bash train/setup.sh` ends with `OK torch 2.11.0+cu128 cuda True natten 0.21.6` (done 17.10 on rmhs4cu2cejemv).
- [ ] GPU, idle: `train.sh probe <set>` reports s/step, data wait and peak VRAM (pending: GPU busy serving).
- [ ] GPU, idle: `train.sh smoke <set>` prints `smoke PASS` (save, resume, raw + EMA adapters through server --selftest).
- [ ] First real adapter: checkpoint 000500 passes `serve.sh --engine lerobot --adapter ... --selftest --n 5`, then the
      arm under the milestone-3 protocol, raw before EMA.

## Upstream

- docs/architecture.md

## Notes and handoffs (append, newest last)

- 2026-10-18 02:3xZ claude-train: pipeline done and reviewed (codex-arm-review + codex-ft-supervisor, docs/status/2026-10-17-training-restart-review.md, docs/research/2026-10-17-lora-supervisor-review.md). GPU items done: setup, probe, smoke PASS (save/resume/raw+EMA served). Two adapters trained on wog1l8ijpl3jp5: qmap run …200859-092d (stopped 4272, ckpts 000500-004000) and hfsim run …213030-6ec3 (10000/10000, EXIT 0, base /workspace/ckpt/flux-3-action-so101-hfsim-norm). Offline eval plateaued .082-.094 from step 2500 (final .093). OPEN (Saad): arm A/B with eval/run_task.py, same caption, raw before EMA, candidates 004000 / 007500 or 009500 / 010000. Serve: infra/gpu/so101/serve.sh --engine lerobot --adapter <ckpt>/pretrained_model (hfsim: native, no map). Then tick the last Done-when box.
- 2026-10-18T03:14Z · alerest285 · todo: pipeline done + reviewed; 2 adapters trained; only open item: arm A/B (Saad), see handoff
