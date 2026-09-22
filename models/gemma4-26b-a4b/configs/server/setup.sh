#!/usr/bin/env bash
set -euo pipefail
umask 077

ENDPOINT=${GEMMA4_ENDPOINT:-http://127.0.0.1:8090}

usage() {
    cat <<'EOF'
Usage: bash models/gemma4-26b-a4b/configs/server/setup.sh ACTION [ARG...]

Scheduler/existing-server API client:
  status                       GET /healthz
  models                       GET /v1/models
  create-session               POST /v1/sessions
  session SESSION              GET /v1/session
  chat SESSION REQUEST.json    POST /v1/chat/completions
  checkpoint SESSION           POST /v1/session/checkpoint
  bootstrap SESSION FILE.json  POST /v1/session/bootstrap from checkpoint response
  reset SESSION                POST /v1/session/reset
  release SESSION              POST /v1/session/release

Environment:
  GEMMA4_ENDPOINT              existing HTTP endpoint (default http://127.0.0.1:8090)
  GEMMA4_SOURCE_DIR, GEMMA4_POOL, GEMMA4_RECEIPT,
  GEMMA4_AUTH_RECEIPT, GEMMA4_PYTHON, SALT_GEMMA_PLATFORM_RECIPE,
  GEMMA4_KV_CACHE_ROOT, GEMMA4_SHARED_KV, GEMMA4_DPR_ROOT and endpoint sizing
  variables are consumed by server/test-gemma4-launch.sh.

This utility never launches gemma4-server directly, imports gemma4_backend.py,
selects filesystem KV from a request, or owns session/KV/inference state.
EOF
}

fail() {
    printf 'gemma4 server setup: %s\n' "$*" >&2
    exit 2
}

require_session() {
    local value=${1:-}
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$ ]] ||
        fail "invalid session id"
}

json_get() {
    local path=$1
    shift
    curl --fail --silent --show-error --max-time 30 "$@" "$ENDPOINT$path"
    printf '\n'
}

json_post() {
    local path=$1 body=$2
    shift 2
    curl --fail-with-body --silent --show-error --max-time 1800 \
        -H 'Content-Type: application/json' "$@" \
        --data-binary "$body" "$ENDPOINT$path"
    printf '\n'
}

case ${1:-} in
verify|start)
    fail "$1 moved to server/test-gemma4-launch.sh; use that single setup/build/launch entry"
    ;;
status)
    shift
    [[ $# -eq 0 ]] || fail "status accepts no arguments"
    json_get /healthz
    ;;
models)
    shift
    [[ $# -eq 0 ]] || fail "models accepts no arguments"
    json_get /v1/models
    ;;
create-session)
    shift
    [[ $# -eq 0 ]] || fail "create-session accepts no arguments"
    json_post /v1/sessions '{}'
    ;;
session)
    session=${2:-}
    [[ $# -eq 2 ]] || fail "session requires SESSION"
    require_session "$session"
    json_get /v1/session -H "X-Salt-Session: $session"
    ;;
chat)
    session=${2:-}
    request=${3:-}
    [[ $# -eq 3 ]] || fail "chat requires SESSION REQUEST.json"
    require_session "$session"
    [[ -f "$request" && ! -L "$request" ]] || fail "request must be a regular non-symlink file"
    curl --fail-with-body --silent --show-error --max-time 1800 \
        -H 'Content-Type: application/json' \
        -H "X-Salt-Session: $session" \
        --data-binary @"$request" "$ENDPOINT/v1/chat/completions"
    printf '\n'
    ;;
checkpoint)
    session=${2:-}
    [[ $# -eq 2 ]] || fail "checkpoint requires SESSION"
    require_session "$session"
    json_post /v1/session/checkpoint '{}' -H "X-Salt-Session: $session"
    ;;
bootstrap)
    session=${2:-}
    checkpoint=${3:-}
    [[ $# -eq 3 ]] || fail "bootstrap requires SESSION FILE.json"
    require_session "$session"
    [[ -f "$checkpoint" && ! -L "$checkpoint" ]] ||
        fail "checkpoint must be a regular non-symlink file"
    python=${GEMMA4_PYTHON:-/usr/bin/python3}
    [[ "$python" == /* && -x "$python" ]] ||
        fail "GEMMA4_PYTHON must be an absolute executable path"
    "$python" - "$checkpoint" <<'PY' |
import json,sys
with open(sys.argv[1],encoding="utf-8") as stream:
    value=json.load(stream)
if (not isinstance(value,dict) or not isinstance(value.get("cache_id"),str) or
        not isinstance(value.get("manifest"),dict)):
    raise SystemExit("invalid session checkpoint file")
json.dump({"cache_id":value["cache_id"],"manifest":value["manifest"]},sys.stdout,separators=(",",":"))
PY
        curl --fail-with-body --silent --show-error --max-time 1800 \
            -H 'Content-Type: application/json' \
            -H "X-Salt-Session: $session" \
            --data-binary @- "$ENDPOINT/v1/session/bootstrap"
    printf '\n'
    ;;
reset|release)
    action=$1
    session=${2:-}
    [[ $# -eq 2 ]] || fail "$action requires SESSION"
    require_session "$session"
    json_post "/v1/session/$action" '{}' -H "X-Salt-Session: $session"
    ;;
-h|--help|help|'')
    usage
    ;;
*)
    usage >&2
    fail "unknown action: $1"
    ;;
esac
