#!/usr/bin/env bash
# End of hackathon: cancel the persistent spot request (else AWS relaunches it) and terminate.
# Deletes the disk with the weights. Asks first.
set -eu; . "$(dirname "$0")/env.sh"; need_aws
id="$(instance_id)"; [ -n "$id" ] || die "no instance named $HACK_NAME."
printf 'Terminate %s and delete its disk? Type the instance id to confirm: ' "$id"; read -r ok
[ "$ok" = "$id" ] || die "not confirmed."
req="$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].SpotInstanceRequestId' --output text)"
[ "$req" = None ] || aws ec2 cancel-spot-instance-requests --spot-instance-request-ids "$req" >/dev/null
aws ec2 terminate-instances --instance-ids "$id" >/dev/null && echo "terminating $id."
