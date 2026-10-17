# Status 2026-10-17 14:30: AWS project connected, waiting on the plan upgrade and quota

**TL;DR.** The AWS CLI and Agent Toolkit are signed in to alerest285's project (eu-north-1, profile `hackathon`). The H100 image,
prices and permissions are checked without launching anything. Two blockers remain before the GPU can start: the **paid-plan
upgrade**, because the free plan denies spot, and the **P-quota approval**, requested and pending. Details in decision 04.

## Next, ranked

1. alerest285: upgrade the project (AWS Settings → Billing) and set a $150 spend limit
2. Wait for the quota (check with `infra/aws/quota.sh`), then `infra/aws/up.sh`, which runs T03
3. T01, the embodiment choice, is still open
