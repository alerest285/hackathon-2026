# 08. Runpod H100 while the AWS GPU quota is pending

**Date:** 2026-10-17 · **Status:** accepted · **Author:** alerest285

## Context

The hackathon asks teams to use AWS and reimburses it. Our AWS project's GPU quota is 0, and the request
(case 179223860800987) was still open after hours. The team needed a GPU now.

## Decision

- One Runpod pod `rmhs4cu2cejemv`: H100 80 GB HBM3, SECURE cloud, EU-FR-1 (Paris), **$3.99/h, running continuously**.
  It has a 200 GB persistent `/workspace`, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, and SSH on 22/tcp.
- SSH keys for alerest285 (Mac), saadl27 (2) and joelleayoub (1) come from their GitHub keys through `PUBLIC_KEY`. Everyone is root.
- `infra/gpu/bootstrap.sh` and `smoke-so101.sh` work unchanged with `FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf`.
- Measured: load 123 s, 0.58 s per chunk eager, 25.9 GB peak.

## Consequences

- $3.99/h is about $96 per day. Runpod has no spend limit; the account balance is the cap.
- Runpod console control (start, stop) for teammates needs a Runpod team. The pod would have to be recreated under the team.
- When the AWS quota lands, decide whether to move. AWS is reimbursed; Runpod is not, unless the organizers agree.
