#!/usr/bin/env bash
# Start the team GPU. Re-uses a stopped instance (weights stay on its disk); otherwise launches
# a persistent spot p5.4xlarge in the cheapest zone that offers it. Prints the ssh line.
set -eu; . "$(dirname "$0")/env.sh"; need_aws
here="$(cd "$(dirname "$0")" && pwd)"

id="$(instance_id)"
if [ -n "$id" ]; then
  st="$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)"
  case "$st" in
    stopped) echo "starting stopped $id..."; aws ec2 start-instances --instance-ids "$id" >/dev/null ;;
    running|pending) echo "$id is already $st." ;;
    *) die "$id is $st; wait a minute and retry." ;;
  esac
else
  # Free, nothing created: catches the free plan's spot block and missing permissions up front.
  pre="$(aws ec2 run-instances --dry-run --image-id "$(aws ssm get-parameter --name "$HACK_AMI_PARAM" --query Parameter.Value --output text)" \
         --instance-type "$HACK_TYPE" $(market_opts) 2>&1 || true)"
  case "$pre" in
    *DryRunOperation*) ;;
    *"explicit deny"*) [ "$HACK_MARKET" = spot ] && die "spot is denied for this project. On the free plan it always is:
  upgrade in AWS Settings > Billing, or run HACK_MARKET=ondemand infra/aws/up.sh (\$$(ondemand_price)/h)."
                       die "launch denied by the project's policy: $pre" ;;
    *) die "dry run failed: $pre" ;;
  esac
  ami="$(aws ssm get-parameter --name "$HACK_AMI_PARAM" --query Parameter.Value --output text)"
  since="$(date -u -v-3H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '-3 hours' +%Y-%m-%dT%H:%M:%S)"
  [ "$HACK_MARKET" = spot ] && az="$(aws ec2 describe-spot-price-history --instance-types "$HACK_TYPE" --product-descriptions Linux/UNIX \
        --start-time "$since" --query 'sort_by(SpotPriceHistory,&to_number(SpotPrice))[0].AvailabilityZone' --output text)" || \
    az="$(aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values="$HACK_TYPE" \
          --query 'InstanceTypeOfferings[0].Location' --output text)"
  [ -n "$az" ] && [ "$az" != None ] || die "no spot price for $HACK_TYPE in $AWS_REGION. Try AWS_REGION=us-west-2."
  subnet="$(aws ec2 describe-subnets --filters Name=availability-zone,Values="$az" Name=default-for-az,Values=true \
           --query 'Subnets[0].SubnetId' --output text)"
  vpc="$(aws ec2 describe-subnets --subnet-ids "$subnet" --query 'Subnets[0].VpcId' --output text)"
  sg="$(aws ec2 describe-security-groups --filters Name=group-name,Values=hack-2026-ssh Name=vpc-id,Values="$vpc" \
       --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null | sed 's/^None$//')"
  if [ -z "$sg" ]; then
    sg="$(aws ec2 create-security-group --group-name hack-2026-ssh --description 'SSH only, key auth' --vpc-id "$vpc" --query GroupId --output text)"
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
  fi
  [ -f "$HACK_SSH_PUBKEY" ] || die "no ssh public key found. Create one: ssh-keygen -t ed25519"
  keyname="hack-2026-$(git config --get hack.member 2>/dev/null || whoami)"
  aws ec2 describe-key-pairs --key-names "$keyname" >/dev/null 2>&1 || \
    aws ec2 import-key-pair --key-name "$keyname" --public-key-material "fileb://$HACK_SSH_PUBKEY" >/dev/null
  members="$(grep -v '^#' "$here/members.txt" | sed '/^ *$/d' | tr '\n' ' ')"
  [ -n "$members" ] || members="none"
  ud="$(sed "s/__MEMBERS__/$members/; s/__IDLE_MIN__/$HACK_IDLE_MIN/g" "$here/user-data.sh")"
  echo "launching $HACK_TYPE $HACK_MARKET in $az (ami $ami)..."
  id="$(aws ec2 run-instances --image-id "$ami" --instance-type "$HACK_TYPE" --subnet-id "$subnet" \
    --key-name "$keyname" --security-group-ids "$sg" --associate-public-ip-address \
    $(market_opts) \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$HACK_DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --metadata-options HttpTokens=required \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$HACK_NAME}]" "ResourceType=volume,Tags=[{Key=Name,Value=$HACK_NAME}]" \
    --user-data "$ud" --query 'Instances[0].InstanceId' --output text)"
fi
aws ec2 wait instance-running --instance-ids "$id"
host="$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
az_now="$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)"
if [ "$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].InstanceLifecycle' --output text)" = spot ]; then
  price="$(aws ec2 describe-spot-price-history --instance-types "$HACK_TYPE" --product-descriptions Linux/UNIX \
    --availability-zone "$az_now" --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query 'SpotPriceHistory[0].SpotPrice' --output text)"
else price="$(ondemand_price) on-demand"; fi
echo "RUNNING: $id  $az_now  host: $host  ~\$${price}/h"
echo "ssh:     HACK_GPU_HOST=$host infra/aws/ssh.sh"
echo "LOG IT:  scripts/log.sh \"gpu up $id $HACK_TYPE ~\$${price}/h\""
echo "First time on the box: git clone https://github.com/alerest285/hackathon-2026 && bash hackathon-2026/infra/gpu/bootstrap.sh"
echo "It stops itself after $HACK_IDLE_MIN idle minutes. Stop it yourself with infra/aws/down.sh."
