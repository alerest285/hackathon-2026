---
id: T01
title: Pick the zero-shot task and embodiment
status: todo
owner: 
depends: 
paths:
  - docs/project.md
---

## Goal

Choose what milestone 3 runs: DROID in RoboLab sim, a physical DROID-style Franka, or a physical
SO-101. Also choose the task instruction(s) and what counts as success. This is a human call;
agents can prepare the comparison (hardware we can access, the cameras each needs, sim GPU needs).

## Done when

- [ ] A decision record (`scripts/new-decision.sh`) names the embodiment, the task prompt(s), the success criterion and N episodes
- [ ] `docs/project.md` open question 1 is replaced by a pointer to that decision

## Upstream

- docs/project.md · docs/research/flux-3-action.md · docs/architecture.md (SO-101 vs DROID interfaces)

## Notes and handoffs (append, newest last)

