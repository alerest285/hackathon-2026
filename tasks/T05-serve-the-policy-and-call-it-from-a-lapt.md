---
id: T05
title: Serve the policy and call it from a laptop
status: todo
owner: 
depends: T04
paths:
  - infra/gpu/serve.sh
  - infra/client/**
---

## Goal

Milestone 2. Run `serve-robolab` on the box in tmux. A teammate's laptop then gets action chunks
through the ssh tunnel with `uv run infra/client/ping.py`.

## Done when

- [ ] ping.py prints PASS from a machine other than the box, with black frames and with `--npz` (the recorded observation copied over)
- [ ] round-trip latency is logged, split into network and `server_timing`
- [ ] docs/architecture.md is corrected if the real protocol differs from the docs (lock it first)

## Upstream

- docs/architecture.md (served interface) · docs/decisions/03-standalone-flux-action-for-zero-shot.md

## Notes and handoffs (append, newest last)

