# 07. On-demand H100: spot is blocked on new-experience projects

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285 · **Supersedes:** the spot parts of 02 and 04

## Context

Decision 04 assumed the paid plan would allow spot. alerest285 upgraded the project (plan = PAID, $100 credits kept), and the
dry run still failed: `--instance-market-options MarketType=spot` → explicit SCP deny. AWS's managed paid-plan policy contains
`DenySpotMarketType` (`ec2:InstanceMarketType = spot`) and `DenyRequestSpotInstances`, the same as the free plan
([policy page](https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html)). The only way out is
"Activate advanced AWS features". That is irreversible: you become org admin, with a root email and password, the spend limits go away, and the starter rules must be removed.

## Decision

- **On-demand `p5.4xlarge` (1× H100 80 GB), $7.36/h in eu-north-1.** `HACK_MARKET` now defaults to `ondemand`.
  Spot is kept in the scripts for a future account that allows it.
- Quota needed: `Running On-Demand P instances` ≥ 16 vCPU (case open since 14:00).
- The idle stop (45 min) and `down.sh` matter more now: every idle hour costs $7.36.

## Consequences

- $150 is about 20 GPU-hours ($100 credits + $50). Budget per milestone: bootstrap and smoke ~1 h, the SO-101 selftest and latency ~1 h,
  and the rest for Hanoi runs. Keep the box stopped between sessions; the disk alone is ~$0.02/h.
- On-demand is never interrupted, so demo timing is safe.
- Revisit if hours run short: an L40S `g6e.2xlarge` on-demand at $2.38/h (needs a G quota and a 48 GB fit test), or activating advanced features for spot.
