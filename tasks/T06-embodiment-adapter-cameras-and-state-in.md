---
id: T06
title: Embodiment adapter: cameras and state in, actions out
status: todo
owner: 
depends: T01, T05
paths:
  - robot/**
---

## Goal

Build the bridge for whatever T01 picks: grab synchronized frames and joint state, build the request
(docs/architecture.md), execute the first k actions, then replan. For SO-101 this means the standalone
`load_policy` loop at 30 Hz, which may need its own server on the box (add it under `infra/gpu/`
with a new task).

## Done when

- [ ] a dry run (actions logged, not executed) at the control rate, with timing per tick
- [ ] one supervised live episode, with the e-stop in reach

## Upstream

- the T01 decision · docs/architecture.md

## Notes and handoffs (append, newest last)

