---
id: T03
title: Launch the H100 and bootstrap flux-action
status: doing
owner: alerest285
depends: T02
paths:
  - infra/aws/*.sh
  - infra/aws/user-data.sh
  - infra/gpu/bootstrap.sh
---

## Goal

First boot of the shared GPU. `infra/aws/up.sh`, then on the box `infra/gpu/bootstrap.sh`, and
`torch.cuda.is_available()` is True with NATTEN imported. Fix whatever the scripts got wrong
(they were written before any AWS account existed; see the README's "Not verified yet").
The holder of this lock owns the instance's lifecycle.

## Done when

- [ ] up → ssh → bootstrap prints `cuda True` and a natten version
- [ ] down → up restarts the same instance, with the env still there
- [ ] the log has the instance id, zone, spot price/h, and boot-to-ready time

## Upstream

- docs/decisions/02-aws-spot-h100.md · docs/decisions/03-standalone-flux-action-for-zero-shot.md · infra/aws/README.md

## Notes and handoffs (append, newest last)

