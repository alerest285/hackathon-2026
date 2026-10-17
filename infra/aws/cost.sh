#!/usr/bin/env bash
# Where the money is: current spot prices, instance state, month-to-date spend.
set -eu; . "$(dirname "$0")/env.sh"; need_aws
echo "Spot $HACK_TYPE now ($AWS_REGION):"
aws ec2 describe-spot-price-history --instance-types "$HACK_TYPE" --product-descriptions Linux/UNIX \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text | sed 's/^/  /'
id="$(instance_id)"
[ -z "$id" ] || aws ec2 describe-instances --instance-ids "$id" \
  --query 'Reservations[0].Instances[0].[InstanceId,State.Name,LaunchTime]' --output text | sed 's/^/Instance: /'
echo "Month to date (Cost Explorer, ~\$0.01 per call, lags up to 24 h):"
end="$(date -u -v+1d +%Y-%m-%d 2>/dev/null || date -u -d tomorrow +%Y-%m-%d)"
aws ce get-cost-and-usage --time-period "Start=$(date -u +%Y-%m-01),End=$end" \
  --granularity MONTHLY --metrics UnblendedCost --query 'ResultsByTime[0].Total.UnblendedCost.[Amount,Unit]' --output text | sed 's/^/  /'
echo "Budget: \$$HACK_BUDGET_USD. At ~\$3/h spot that is ~45 GPU-hours."
