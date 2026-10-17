# Shared settings for the AWS scripts. Override any of them in your shell.
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
HACK_NAME="${HACK_NAME:-hack-2026-gpu}"
HACK_TYPE="${HACK_TYPE:-p5.4xlarge}"          # 1x H100 80 GB, 16 vCPU
HACK_DISK_GB="${HACK_DISK_GB:-250}"
HACK_AMI_PARAM="${HACK_AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"
HACK_IDLE_MIN="${HACK_IDLE_MIN:-45}"          # instance stops itself after this many idle minutes
HACK_BUDGET_USD="${HACK_BUDGET_USD:-150}"
die() { echo "error: $*" >&2; exit 1; }
need_aws() { aws sts get-caller-identity >/dev/null 2>&1 || die "AWS CLI not logged in. See infra/aws/README.md step 2."; }
instance_id() { # newest non-terminated instance with our Name tag
  aws ec2 describe-instances --filters "Name=tag:Name,Values=$HACK_NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId' --output text 2>/dev/null | sed 's/^None$//'
}
