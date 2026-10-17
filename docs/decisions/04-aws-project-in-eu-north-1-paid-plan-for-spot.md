# 04. AWS project in eu-north-1; spot needs the paid plan

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285 · **Amends:** 02 (region and plan; the instance choice stands)

## Context

The AWS account is a project in the new AWS experience (social sign-in), and its region is **eu-north-1**. A project can
only create resources in that one region. It started on the **free plan** with $100 credits, which expire 17.04.2027.
Decision 02 assumed spot in us-east-1. Measured on 17.10 without launching anything:

| Check | Result |
|---|---|
| `p5.4xlarge` in eu-north-1 | offered in a, b and c |
| Price | spot $3.16/h (1c), on-demand $7.36/h |
| `run-instances --dry-run`, on-demand p5.4xlarge, free plan | allowed |
| same with `--instance-market-options MarketType=spot`, any type, free plan | **denied, explicit SCP deny** |
| GPU quotas (P spot, P on-demand, G spot) | 0. P spot → 16 and P on-demand → 16 requested, both PENDING |

## Decision

- Region **eu-north-1**, CLI profile `hackathon`. Scripts default to both (`infra/aws/env.sh`).
- alerest285 **upgrades the project to the paid plan** so spot works. The instance stays a `p5.4xlarge` persistent spot, as in decision 02.
- `HACK_MARKET=ondemand` stays as the fallback if spot is refused or interrupted at demo time.
- `up.sh` runs a free dry run before launching, so a plan or permission problem fails cleanly.
- The AWS Agent Toolkit (MCP server + skills) is set up on alerest285's Mac. Its rules are in AGENTS.md section 9.

## Consequences

- $150 at $3.16/h is about 47 GPU-hours, against about 20 on on-demand.
- If the quota is refused, the next request is "All G and VT Spot Instance Requests" for an L40S `g6e.2xlarge` ($2.38/h on-demand). Only two requests can be open at once.
- Revisit if the upgrade still denies spot (then go on-demand) or if eu-north-1 has no spot capacity at launch time.
