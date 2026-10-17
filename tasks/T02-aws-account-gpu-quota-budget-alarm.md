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

