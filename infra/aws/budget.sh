#!/usr/bin/env bash
# Monthly cost budget with email alerts at 50 / 80 / 100 % of actual spend and 100 % forecast.
#   infra/aws/budget.sh you@example.com [more@example.com ...]
set -eu; . "$(dirname "$0")/env.sh"; need_aws
[ $# -ge 1 ] || die "usage: infra/aws/budget.sh email [email ...]"
acct="$(aws sts get-caller-identity --query Account --output text)"
subs=""; for e in "$@"; do subs="$subs{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$e\"},"; done; subs="[${subs%,}]"
n() { echo "{\"Notification\":{\"NotificationType\":\"$1\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":$2,\"ThresholdType\":\"PERCENTAGE\"},\"Subscribers\":$subs}"; }
aws budgets create-budget --account-id "$acct" \
  --budget "{\"BudgetName\":\"hackathon-gpu\",\"BudgetLimit\":{\"Amount\":\"$HACK_BUDGET_USD\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" \
  --notifications-with-subscribers "[$(n ACTUAL 50),$(n ACTUAL 80),$(n ACTUAL 100),$(n FORECASTED 100)]"
echo "OK: budget hackathon-gpu = \$$HACK_BUDGET_USD/month, alerts to $*"
