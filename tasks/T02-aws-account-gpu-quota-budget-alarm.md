---
id: T02
title: AWS account, GPU quota, budget alarm
status: doing
owner: alerest285
depends: 
paths:
  - infra/aws/README.md
  - infra/aws/members.txt
---

## Goal

Get an AWS account to the point where `infra/aws/up.sh` can launch one p5.4xlarge on spot,
with the $150 budget alarm armed. **Start today: the quota approval is the slow step.**

## Done when

- [ ] `aws sts get-caller-identity` works for whoever operates the box
- [ ] `infra/aws/quota.sh` shows at least 16 for P5 spot (or for "All P Spot Instance Requests") in the chosen region
- [ ] `infra/aws/budget.sh` has run, and the alert emails are confirmed
- [ ] every teammate's GitHub login is in `infra/aws/members.txt`

## Upstream

- docs/decisions/02-aws-spot-h100.md · infra/aws/README.md

Human steps (sign-up, card, MFA) cannot be done by an agent. Log the quota request id and its time.

## Notes and handoffs (append, newest last)

### Handoff 2026-10-17 14:30 (alerest285 / claude-code)

- Done: project signed in (eu-north-1, profile `hackathon`); Agent Toolkit + aws-mcp on alerest285's Mac; budget `hackathon-gpu` $150 live with alerts.
- **Waiting on AWS:** quota cases `All P Spot Instance Requests` → 16 and `Running On-Demand P instances` → 16, both CASE_OPENED 14:00.
  Check: `infra/aws/quota.sh`. If refused: request `All G and VT Spot Instance Requests` (L-3819A6DF) for an L40S (decision 04).
- **Waiting on alerest285:** upgrade to the paid plan (AWS Settings → Billing); the free plan denies all spot. Then set a $150 spend limit there.
- Still open: teammates' GitHub logins in `members.txt` (ssh), and AWS team-member invites only for those who run AWS commands.
- 2026-10-17T13:06Z · alerest285 · blocked: waiting on AWS quota review (2 cases open) + paid-plan upgrade by alerest285; budget done
- 2026-10-17T13:09Z · alerest285/claude-code: plan PAID; spot denied on both plans, so on-demand (decision 07). Only blocker left: on-demand P quota (case open). Owner: set a $150 spend limit in AWS Settings.
- 2026-10-17T13:09Z · alerest285 · blocked: only the on-demand P quota (case open) is left; plan PAID, budget live
