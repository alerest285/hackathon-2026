#!/usr/bin/env bash
# Stop the GPU. Compute billing stops; the disk and the weights on it stay (~$0.03/h for 250 GB gp3).
set -eu; . "$(dirname "$0")/env.sh"; need_aws
id="$(instance_id)"; [ -n "$id" ] || die "no instance named $HACK_NAME."
aws ec2 stop-instances --instance-ids "$id" >/dev/null && echo "stopping $id. Log it: scripts/log.sh \"gpu down $id\""
