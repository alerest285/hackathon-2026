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

- [ ] `bash infra/gpu/so101/serve.sh --selftest` on the box returns `(42, 6)` actions for a synthetic observation and prints per-plan latency (p50/p95 over 50 plans, eager and compiled) and peak GPU memory; the numbers are in the log (milestone 1 for SO-101)
- [ ] from the Mac, through the ssh tunnel, a ping client gets chunks and prints the round trip split into network and server time; logged (milestone 2)
- [ ] a decision record gives the protocol, the latency settings used (precision, compile, guidance, steps) and the measured numbers

## Upstream

- docs/decisions/03-standalone-flux-action-for-zero-shot.md · docs/decisions/05-milestone-3-is-a-3-disk-tower-of-hanoi-on-the-so-1.md
- docs/architecture.md (SO-101 interface) · docs/research/flux-3-action.md · infra/gpu/bootstrap.sh (T03 owns it; the server reuses its env)

## Notes and handoffs (append, newest last)

