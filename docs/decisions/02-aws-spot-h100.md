# 02. Compute: one AWS p5.4xlarge (1× H100) on persistent spot, $150

**Date:** 2026-10-17 · **Status:** accepted, region and plan amended by 04 · **Author:** alerest285

## Context

We provide our own compute. The budget is $150, the work is inference only for now, and the team chose AWS. FLUX Action needs Linux,
an NVIDIA GPU and CUDA 12.8. BFL's reference environment is an H200, and VRAM is not published. A 7B DiT plus Qwen3-VL-4B plus the video VAE should fit
in 80 GB for inference; T04 verifies this.

## Decision

- `p5.4xlarge`: 1× H100 80 GB, 16 vCPU. us-east-1 spot was about $2.6–3.8/h (aggregators, Oct 2026) against $6.88 on-demand.
- **Persistent spot, interruption = stop**, 250 GB gp3 root. Stopping keeps the ~30–40 GB of weights and env.
- AMI: AWS Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 24.04 (driver 570, P5 supported), resolved through SSM.
- SSH on port 22 from anywhere, **key-only**, with keys taken from teammates' GitHub accounts (`infra/aws/members.txt`).
  The policy server is reached only through an `ssh -L 8000` tunnel and is never exposed publicly.
- Guards: a $150 AWS Budget with email alerts, an on-box idle stop after 45 min, and a log line for every up and down.

## Consequences

- **The quota is the critical path.** New accounts have 0 P-spot vCPUs, and approval can take days. Request it first (`infra/aws/quota.sh --request`).
- Spot can be interrupted mid-demo. If the demo slot matters, switch to on-demand for that hour (about $7).
- Revisit if quota is refused, or if spot prices stay above ~$5/h.
