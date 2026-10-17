# Shared settings for the AWS scripts. Override any of them in your shell.
# The AWS project lives in eu-north-1 (Stockholm) and cannot create resources in any other region.
export AWS_PROFILE="${AWS_PROFILE:-hackathon}"   # created by `aws login --profile hackathon`
export AWS_REGION="${AWS_REGION:-eu-north-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
HACK_NAME="${HACK_NAME:-hack-2026-gpu}"
HACK_TYPE="${HACK_TYPE:-p5.4xlarge}"          # 1x H100 80 GB, 16 vCPU
HACK_MARKET="${HACK_MARKET:-spot}"            # spot (~$3.2/h, needs the paid plan) | ondemand ($7.36/h)
HACK_DISK_GB="${HACK_DISK_GB:-250}"
HACK_AMI_PARAM="${HACK_AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"
HACK_IDLE_MIN="${HACK_IDLE_MIN:-45}"          # instance stops itself after this many idle minutes
HACK_BUDGET_USD="${HACK_BUDGET_USD:-150}"
die() { echo "error: $*" >&2; exit 1; }
need_aws() { aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials missing or expired. Run: aws login --profile $AWS_PROFILE"; }
market_opts() { # args for run-instances
  [ "$HACK_MARKET" = spot ] && echo "--instance-market-options MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}"; }
ondemand_price() {
  aws pricing get-products --region us-east-1 --service-code AmazonEC2 --filters "Type=TERM_MATCH,Field=instanceType,Value=$HACK_TYPE" \
    "Type=TERM_MATCH,Field=regionCode,Value=$AWS_REGION" "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" "Type=TERM_MATCH,Field=capacitystatus,Value=Used" "Type=TERM_MATCH,Field=marketoption,Value=OnDemand" \
    --query 'PriceList[0]' --output text 2>/dev/null | sed -n 's/.*"USD" *: *"\([0-9.]*\)".*/\1/p' | head -1; }
instance_id() { # newest non-terminated instance with our Name tag
  aws ec2 describe-instances --filters "Name=tag:Name,Values=$HACK_NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId' --output text 2>/dev/null | sed 's/^None$//'
}
