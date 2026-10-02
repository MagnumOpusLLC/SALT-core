#!/usr/bin/env bash
# Start only an existing, type-checked Salt EC2 instance. Never provisions one.
set -euo pipefail

usage() {
    printf '%s\n' 'Usage: salt-ec2 {status|start} {c6i|g4ad} [--region REGION] [--profile PROFILE] [--instance-id ID] [--wait]'
    printf '%s\n' '  c6i  = exactly one existing c6i.2xlarge in the selected account/region'
    printf '%s\n' '  g4ad = exactly one existing g4ad.xlarge in the selected account/region'
    printf '%s\n' '  status is read-only; start requires explicit invocation and never creates an instance.'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (($# < 2)); then usage >&2; exit 2; fi
action=$1; target=$2; shift 2
case "$action" in status|start) ;; *) usage >&2; exit 2 ;; esac
case "$target" in
    c6i) expected_type=c6i.2xlarge; instance_id= ;;
    g4ad) expected_type=g4ad.xlarge; instance_id= ;;
    *) usage >&2; exit 2 ;;
esac
region=us-east-1
profile=
wait_for_running=0
while (($#)); do
    case "$1" in
        --region)
            (($# >= 2)) || { printf 'Missing --region value\n' >&2; exit 2; }
            region=$2; shift 2 ;;
        --profile)
            (($# >= 2)) || { printf 'Missing --profile value\n' >&2; exit 2; }
            profile=$2; shift 2 ;;
        --instance-id)
            (($# >= 2)) || { printf 'Missing --instance-id value\n' >&2; exit 2; }
            instance_id=$2; shift 2 ;;
        --wait) wait_for_running=1; shift ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ $action == status && $wait_for_running == 1 ]]; then
    printf '%s\n' '--wait requires start' >&2; exit 2
fi
[[ $region =~ ^[a-z0-9-]+$ ]] || { printf 'Invalid region\n' >&2; exit 2; }
[[ -z $profile || ( $profile != -* && $profile =~ ^[A-Za-z0-9_.@-]+$ ) ]] || {
    printf 'Invalid profile name\n' >&2; exit 2;
}
[[ -z $instance_id || $instance_id =~ ^i-([0-9a-f]{8}|[0-9a-f]{17})$ ]] || {
    printf 'Invalid instance ID\n' >&2; exit 2;
}
command -v aws >/dev/null || { printf 'AWS CLI is required\n' >&2; exit 1; }
command -v jq >/dev/null || { printf 'jq is required\n' >&2; exit 1; }
export AWS_EC2_METADATA_DISABLED=true
aws_args=(--region "$region" --no-cli-pager --cli-connect-timeout 8 --cli-read-timeout 20)
if [[ -n $profile ]]; then aws_args+=(--profile "$profile"); fi
instance_query='Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,Dns:PublicDnsName}'
if [[ -n $instance_id ]]; then
    found=$(aws "${aws_args[@]}" ec2 describe-instances --instance-ids "$instance_id" \
        --query "$instance_query" --output json)
else
    found=$(aws "${aws_args[@]}" ec2 describe-instances \
        --filters "Name=instance-type,Values=$expected_type" \
                  'Name=instance-state-name,Values=pending,running,stopping,stopped' \
        --query "$instance_query" --output json)
fi
count=$(jq 'length' <<< "$found")
if [[ $count != 1 ]]; then
    printf 'Expected exactly one %s in %s; found %s. Select the correct --region/--profile or pass --instance-id. No instance was started.\n' \
        "$expected_type" "$region" "$count" >&2
    exit 1
fi
actual_id=$(jq -r '.[0].Id' <<< "$found")
actual_type=$(jq -r '.[0].Type' <<< "$found")
state=$(jq -r '.[0].State' <<< "$found")
dns=$(jq -r '.[0].Dns // ""' <<< "$found")
if [[ $actual_type != "$expected_type" || $actual_id != i-* ]]; then
    printf 'Instance identity/type mismatch; no instance was started.\n' >&2; exit 1
fi
printf 'Instance %s (%s) in %s: %s%s\n' "$actual_id" "$actual_type" "$region" "$state" "${dns:+ [$dns]}"
if [[ $action == status ]]; then exit 0; fi
case "$state" in
    running) printf 'Already running; no start API call.\n' ;;
    pending) printf 'Already pending; no start API call.\n' ;;
    stopped)
        response=$(aws "${aws_args[@]}" ec2 start-instances --instance-ids "$actual_id" \
            --query 'StartingInstances[0].{Id:InstanceId,Previous:PreviousState.Name,Current:CurrentState.Name}' --output json)
        returned_id=$(jq -r '.Id' <<< "$response")
        [[ $returned_id == "$actual_id" ]] || { printf 'Start response ID mismatch\n' >&2; exit 1; }
        printf 'Start accepted for %s; checking the instance state.\n' "$actual_id" ;;
    *) printf 'State %s is not startable; no start API call.\n' "$state" >&2; exit 1 ;;
esac
if [[ $wait_for_running == 1 ]]; then
    aws "${aws_args[@]}" ec2 wait instance-running --instance-ids "$actual_id"
fi
observed=$(aws "${aws_args[@]}" ec2 describe-instances --instance-ids "$actual_id" \
    --query "$instance_query" --output json)
[[ $(jq 'length' <<< "$observed") == 1 && $(jq -r '.[0].Id' <<< "$observed") == "$actual_id" ]] || {
    printf 'Could not read back the exact instance after start.\n' >&2; exit 1;
}
observed_state=$(jq -r '.[0].State' <<< "$observed")
observed_dns=$(jq -r '.[0].Dns // ""' <<< "$observed")
printf 'Observed %s: %s%s\n' "$actual_id" "$observed_state" "${observed_dns:+ [$observed_dns]}"
if [[ $wait_for_running == 1 && $observed_state != running ]]; then
    printf 'Wait returned without a running state.\n' >&2; exit 1
fi
