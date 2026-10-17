---
id: T04
title: DROID smoke test on a recorded observation
status: todo
owner: 
depends: T03
paths:
  - infra/gpu/smoke-droid.sh
---

## Goal

Milestone 1. Run the released DROID policy on BFL's public recorded sample, with no robot, and measure it.

## Done when

- [ ] `bash infra/gpu/smoke-droid.sh` prints PASS: actions `(1, 32, 8)`
- [ ] the same passes with `variants/gd-fp8r` and `variants/sd-fp8r`
- [ ] a decision record lists, per package: latency (eager and `--compile-dit`), peak GPU memory from `report.json`, and which package we serve

## Upstream

- docs/research/flux-3-action.md · docs/architecture.md (offline interface)

## Notes and handoffs (append, newest last)

