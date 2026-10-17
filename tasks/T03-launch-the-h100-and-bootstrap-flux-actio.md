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

### Handoff 2026-10-17 14:30 (alerest285 / claude-code)

- Scripts are final for eu-north-1 and checked against the live project without launching: AMI `ami-0b00ba21bb3194a5a` (/dev/sda1),
  p5.4xlarge offered in a/b/c, on-demand dry run allowed, spot denied until the plan upgrade. `up.sh` dry-runs first and
  stopped cleanly (0 instances).
- Start the moment T02 clears: `infra/aws/up.sh` (spot) or `HACK_MARKET=ondemand infra/aws/up.sh`. Then `infra/aws/ssh.sh`.
  On the box: clone, `bash hackathon-2026/infra/gpu/bootstrap.sh`, accept the HF licenses + `uv run hf auth login`. Log id, zone, $/h.
- Not yet observed on a real box: user-data (GitHub keys, idle stop), stop → start of a persistent spot instance, bootstrap on Ubuntu 24.04.
- 2026-10-17T13:06Z · alerest285 · todo: scripts final + dry-run verified on the live project; first boot waits on T02. See handoff.
- 2026-10-17T13:09Z · alerest285 · todo: on-demand default, full launch dry run passes; start with infra/aws/up.sh once quota >= 16
