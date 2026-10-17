---
id: T23
title: LoRA fine-tune: demos to model space, pod training, serve the adapter
status: doing
owner: alerest285
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

