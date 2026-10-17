---
id: T10
title: SO-101 policy server on the GPU host
status: doing
owner: alerest285
depends: T01, T03
paths:
  - infra/gpu/so101/**
---

## Goal

Serve `black-forest-labs/flux-3-action-so101` from the GPU host to the Mac that drives the arm, with the
shortest round trip we can get, so the client can fetch the next chunk while the current one plays.
BFL ships no SO-101 server (docs/architecture.md), so this one is ours, built on the standalone
flux-action loader (decision 03). The GPU sits in eu-north-1 (decision 04), about 38 ms from Zurich.

## Done when

- [x] `bash infra/gpu/so101/serve.sh --selftest` on the box returns `(42, 6)` actions for a synthetic observation and prints per-plan latency (p50/p95 over 50 plans, eager and compiled) and peak GPU memory; the numbers are in the log (milestone 1 for SO-101)
- [x] from the Mac, through the ssh tunnel, a ping client gets chunks and prints the round trip split into network and server time; logged (milestone 2)
- [x] a decision record gives the protocol, the latency settings used (precision, compile, guidance, steps) and the measured numbers

## Upstream

- docs/decisions/03-standalone-flux-action-for-zero-shot.md · docs/decisions/05-milestone-3-is-a-3-disk-tower-of-hanoi-on-the-so-1.md
- docs/architecture.md (SO-101 interface) · docs/research/flux-3-action.md · infra/gpu/bootstrap.sh (T03 owns it; the server reuses its env)

## Notes and handoffs (append, newest last)

- 2026-10-17T15:45Z claude-code-hanoi: done-when met on the Runpod H100. claude-code ran the self-test: eager p50 514 / p95
  549 ms, 15.8 GiB; `--compile` 351/398 ms, deployed 15:30:53Z. The round trip through the tunnel was measured with
  console/ (560–590 ms eager). The numbers, settings and protocol are in decision 06, "Measured". Open: the `lerobot
  --adapter` path stays untested until claude-train's first checkpoint (T23). pin_revision passes adapter snapshot paths
  through. Lock kept for that.
