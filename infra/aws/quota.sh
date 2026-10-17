#!/usr/bin/env bash
# Show (and optionally request) the GPU quotas a new account needs. Default for P spot is 0.
#   infra/aws/quota.sh              show
#   infra/aws/quota.sh --request    ask for 16 vCPUs of P5 spot + P on-demand (one p5.4xlarge each)
set -eu; . "$(dirname "$0")/env.sh"; need_aws
q() { aws service-quotas list-service-quotas --service-code ec2 \
  --query "Quotas[?contains(QuotaName,'P5 Spot') || QuotaName=='All P Spot Instance Requests' || QuotaName=='Running On-Demand P instances'].[QuotaCode,QuotaName,Value]" --output text; }
echo "Region $AWS_REGION:"; q | sed 's/^/  /'
[ "${1:-}" = "--request" ] || exit 0
q | while IFS="$(printf '\t')" read -r code name val; do
  v=${val%.*}; [ "$v" -ge 16 ] && { echo "  $name already $v"; continue; }
  aws service-quotas request-service-quota-increase --service-code ec2 --quota-code "$code" --desired-value 16 \
    --query 'RequestedQuota.[QuotaName,Status]' --output text | sed 's/^/  requested: /'
done
echo "Track it: aws service-quotas list-requested-service-quota-change-history --service-code ec2"
