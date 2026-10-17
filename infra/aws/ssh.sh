#!/usr/bin/env bash
# SSH into the GPU with port 8000 forwarded (the policy server). Works without AWS credentials
# if you set HACK_GPU_HOST (whoever ran up.sh shares it in the team chat, never in the repo).
#   infra/aws/ssh.sh [extra ssh args]
set -eu; . "$(dirname "$0")/env.sh"
host="${HACK_GPU_HOST:-}"
if [ -z "$host" ]; then
  need_aws; id="$(instance_id)"; [ -n "$id" ] || die "no instance. Run infra/aws/up.sh"
  host="$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
fi
exec ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -L 8000:localhost:8000 "ubuntu@$host" "$@"
