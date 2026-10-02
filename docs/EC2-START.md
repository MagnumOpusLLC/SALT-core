# Start existing EC2 hosts from a Mac

`tools/ec2-start-existing.sh` controls **existing** `c6i.2xlarge` and
`g4ad.xlarge` instances. It never creates, changes the type of, or provisions an
instance. `status` only describes; `start` makes an EC2 start call only after
resolving exactly one instance of the requested type and confirming that it is
stopped. A running or pending instance receives no duplicate start request.
The default region is `us-east-1`; supply `--region` and optionally `--profile`
for a different AWS account/region. If more than one matching instance exists,
select one explicitly with `--instance-id` (the type is still checked). If none
exists, the command fails without creating anything.

## Mac setup

AWS CLI v2 and `jq` must already be installed and an AWS profile must have EC2
read/start permission. Use your normal local AWS credential configuration; never
put credentials in this repository or the command line. From the checkout root:

```sh
command -v aws
command -v jq
mkdir -p "$HOME/bin"
chmod 755 tools/ec2-start-existing.sh
ln -s "$(pwd)/tools/ec2-start-existing.sh" "$HOME/bin/salt-ec2"
```

Ensure `$HOME/bin` is already on `PATH`; do not replace an existing command
without inspecting it. Or call `bash tools/ec2-start-existing.sh` directly
without changing permissions or creating a symlink.

```sh
salt-ec2 status c6i
salt-ec2 status g4ad
salt-ec2 start c6i
salt-ec2 start g4ad --region us-east-1 --profile YOUR_PROFILE
salt-ec2 start g4ad --instance-id "$EXISTING_INSTANCE_ID" --wait
```

`--wait` requests AWS's `instance-running` waiter and verifies the exact ID
again before reporting `running`. Without it, the command reports the state
observed immediately after the accepted start call; `pending` is not presented
as ready. An instance marked `running` is not proof of SSH, model assets,
server health, or inference readiness. Starting EC2 incurs cloud charges; the
script does not start either host during setup or `status` checks.
Set `EXISTING_INSTANCE_ID` to an actual ID from the selected account before
using the explicit-ID example; the script never guesses or repairs an ID.
