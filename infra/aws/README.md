# AWS GPU: one H100 on spot, $150 budget

One shared `p5.4xlarge` (1× H100 80 GB, 16 vCPU), persistent spot with stop-on-interrupt,
so the weights survive on its 250 GB disk. Inference only. Spot was about $2.6–3.8/h in
us-east-1 in Oct 2026, against $6.88 on-demand, so $150 buys roughly 40–55 GPU-hours. Check live: `infra/aws/cost.sh`.

## Steps (in order; step 3 is the slow one, so start it first)

1. **Account (a human, ~10 min).** Sign up at https://aws.amazon.com (needs a card and a phone).
   Turn on MFA for the root user. Then create an admin user under IAM Identity Center (or an
   IAM user with an access key) and stop using root.
2. **CLI.** `aws configure sso` (Identity Center) or `aws configure` (access key), region `us-east-1`.
   Check with `aws sts get-caller-identity`.
3. **GPU quota (can take hours to days).** New accounts have **0** vCPUs for P-type spot
   instances, so no H100 will launch until AWS approves an increase.
   `infra/aws/quota.sh --request` asks for 16 vCPUs, which is one p5.4xlarge, for both spot and on-demand.
   In the support case, say it is for a short ML inference project with a $150 budget.
   If spot is still refused near the deadline, on-demand at $6.88/h gives about 20 h.
4. **Budget alarm.** `infra/aws/budget.sh you@mail [teammate@mail ...]` sends emails at 50/80/100 % of $150.
5. **Members.** Put every teammate's GitHub login in `members.txt`. Their public keys from
   `github.com/<login>.keys` are installed at first boot. Edit the list before the first `up.sh`.

## Daily use

| Command | Effect |
|---|---|
| `infra/aws/up.sh` | Starts the stopped instance, or launches a new one in the cheapest zone. Prints host + price |
| `infra/aws/ssh.sh` | ssh as `ubuntu`, with localhost:8000 forwarded to the policy server. Teammates without AWS access: `HACK_GPU_HOST=<host> infra/aws/ssh.sh` |
| `infra/aws/down.sh` | Stops compute billing. The disk stays, ~$0.03/h |
| `infra/aws/cost.sh` | Spot prices, instance state, month-to-date spend |
| `infra/aws/destroy.sh` | End of the hackathon: cancels the spot request, terminates, deletes the disk |

On the box, first time: `git clone <this repo> && bash hackathon-2026/infra/gpu/bootstrap.sh`,
then `bash hackathon-2026/infra/gpu/smoke-droid.sh`. Serve inside tmux with `bash infra/gpu/serve.sh`.

## Budget guards

- **Idle stop.** The box stops itself after 45 minutes with no SSH connection and an idle GPU
  (`user-data.sh`). Turn it off on the box with `sudo touch /etc/no-idle-stop`.
- **Spot interruption** stops the box rather than terminating it. `up.sh` starts it again when capacity returns.
- **Log every up and down** with its price (AGENTS.md section 8).
- Whoever holds the T03 lock owns the instance's lifecycle. Everyone else asks before stopping it.

## Not verified yet

The scripts are syntax-checked only: no AWS account existed when they were written. Specifically:
the DLAMI SSM parameter (`base-oss-nvidia-driver-gpu-ubuntu-24.04`, which lists P5 as supported),
the root device name `/dev/sda1`, and starting a stopped persistent spot instance. T03 is
where these get checked; fix the scripts and log what was wrong.
