# AWS GPU: one H100 on-demand, $150 budget

One shared `p5.4xlarge` (1× H100 80 GB, 16 vCPU) in **eu-north-1**, **on-demand at $7.36/h**. Stop it between sessions;
its 200 GB disk keeps the weights. Inference only. $150 is about **20 GPU-hours**. Spot would be $3.16/h, but
new-experience projects block spot on every plan (decision 07). Check live: `infra/aws/cost.sh`.

## Steps

The project is **alerest285's AWS project, eu-north-1 (Stockholm), CLI profile `hackathon`**. It uses the new AWS
experience: management (billing, spend limits, team members) happens in [AWS Settings](https://settings.aws.com/).
A project can only create resources in its one region. Decision 04 has the reasons and the measured facts.

| # | Step | State (17.10) |
|---|---|---|
| 0 | SSH: `up.sh` imports your `~/.ssh/id_ed25519.pub` as the EC2 key. alerest285 has no keys on GitHub, so this is how you get in. Teammates: their GitHub keys via `members.txt` | built in |
| 1 | Sign in: `aws login --profile hackathon` (browser). Credentials last 12 h and renew for 90 days without the browser | done |
| 2 | AWS Agent Toolkit: `aws configure agent-toolkit --yes --region us-east-1 --profile hackathon`, then add `"env": {"AWS_MCP_PROXY_PROFILES": "hackathon"}` to each `aws-mcp` entry | done on alerest285's Mac |
| 3 | Paid plan (AWS Settings → Billing). It turned out spot stays denied on the paid plan too, so we run on-demand (decision 07) | done 17.10 |
| 4 | **GPU quota.** It was 0. The one that matters is **"Running On-Demand P instances" (L-417A185B) ≥ 16**. P spot was also requested but is now moot | pending AWS review (cases opened 14:00) |
| 5 | Spend guard: budget `hackathon-gpu` $150/month, alerts at 50/80/100 % + forecast to alerest285's mail (`infra/aws/budget.sh`). Also set a **$150 spend limit** in AWS Settings → Billing (it pauses the project when reached) | **both done 17.10**: spend limit $150 on project "Just Ship It" (near the limit AWS denies `RunInstances`/`StartInstances`, so the box cannot be restarted past it) |
| 6 | Teammates: invite them as **team members** in AWS Settings (by email) only if they must run AWS commands. For ssh-only access, adding their GitHub login to `members.txt` is enough | as needed |

Check the state any time:

```sh
aws freetier get-account-plan-state --profile hackathon --query accountPlanType     # FREE or PAID
infra/aws/quota.sh                                                                   # P quotas in eu-north-1
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --profile hackathon --region eu-north-1 \
  --query 'RequestedQuotas[].[QuotaName,DesiredValue,Status]' --output table
```

`up.sh` does a free `--dry-run` before any launch, so a permission problem stops it with a message instead of failing halfway.

## Daily use

| Command | Effect |
|---|---|
| `infra/aws/first-boot.sh` | **The first time:** up → wait for ssh → copy your HF token → clone + bootstrap → SO-101 smoke (load, latency, VRAM) → stop. `--droid` adds the DROID check, `--keep` leaves it running. Log in `outputs/` |
| `infra/aws/up.sh` | Starts the stopped instance, or launches a new one in the cheapest zone. Prints host + price |
| `infra/aws/ssh.sh` | ssh as `ubuntu`, with localhost:8000 forwarded to the policy server. Teammates without AWS access: `HACK_GPU_HOST=<host> infra/aws/ssh.sh` |
| `infra/aws/down.sh` | Stops compute billing. The disk stays, ~$0.03/h |
| `infra/aws/cost.sh` | on-demand price, GPU-hours left in the budget, instance state, month-to-date spend |
| `infra/aws/destroy.sh` | End of the hackathon: terminates the instance and deletes the disk (asks first) |

On the box, first time: `git clone https://github.com/alerest285/hackathon-2026 && bash hackathon-2026/infra/gpu/bootstrap.sh`.
The FLUX weights are **gated** (auto-approved): accept the license on each `black-forest-labs/flux-3-action-*` page with your
Hugging Face account, then run `uv run hf auth login` on the box. bootstrap.sh checks access and tells you what is missing.
Then the SO-101 selftest (T10), or `bash hackathon-2026/infra/gpu/smoke-droid.sh` for the DROID check. Serve inside tmux with `bash infra/gpu/serve.sh`.

## Budget guards

- **Idle stop.** The box stops itself after 45 minutes with no SSH connection and an idle GPU
  (`user-data.sh`). Turn it off on the box with `sudo touch /etc/no-idle-stop`.
- **Every idle hour costs $7.36.** Run `infra/aws/down.sh` when you finish, even with the idle stop as a backstop.
- **Log every up and down** with its price (AGENTS.md section 8).
- Whoever holds the T03 lock owns the instance's lifecycle. Everyone else asks before stopping it.

## Verified on 17.10.2026 (eu-north-1, without launching anything)

- `p5.4xlarge` is offered in eu-north-1a/b/c. Spot was $3.16/h (1c) and on-demand $7.36/h (pricing API).
- The DLAMI SSM parameter resolves to `ami-0b00ba21bb3194a5a`, "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261006". Its root device is `/dev/sda1`.
- Dry run, free plan and again after the paid upgrade: on-demand p5.4xlarge **allowed**; spot **denied** by the project's SCP (`DenySpotMarketType`) for every instance type.
- Default subnets exist in all three zones.

## Not verified yet

- The first real boot: user-data keys, the idle stop, and stop → start keeping the env. That is task T03.
