#!/usr/bin/env bash
# Normal OpenAI-shaped Gemma endpoint over persistent KV/DPR engines.
set -euo pipefail

ROOT=${SALT_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
CONFIG="$ROOT/models/gemma4-26b-a4b/server.config"
GEMMA4_ENDPOINT_CHILD=${GEMMA4_ENDPOINT_CHILD:-$ROOT/server/model/gemma4-endpoint-child.py}
if [[ ! -f "$CONFIG" ||
      ! -f "$ROOT/server/scheduler.py" ||
      ! -f "$GEMMA4_ENDPOINT_CHILD" ]]; then
    printf 'gemma4 endpoint: invalid Salt repository root: %s\n' "$ROOT" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

: "${GEMMA4_SOURCE_DIR:?set GEMMA4_SOURCE_DIR to the authenticated MLX source directory}"
: "${GEMMA4_POOL:?set GEMMA4_POOL to the imported expert pool.bin}"
: "${SALT_GEMMA_PLATFORM_RECIPE:?set SALT_GEMMA_PLATFORM_RECIPE to a checked-in recipe}"

GEMMA4_PYTHON=${GEMMA4_PYTHON:-}
if [[ -z "$GEMMA4_PYTHON" ]]; then
    GEMMA4_PYTHON=$(command -v python3)
fi
if [[ "$GEMMA4_PYTHON" != /* || ! -x "$GEMMA4_PYTHON" ]]; then
    printf 'gemma4 endpoint: GEMMA4_PYTHON must be an absolute executable path\n' >&2
    exit 2
fi

export GEMMA4_PYTHON
export GEMMA4_HOST=${GEMMA4_HOST:-127.0.0.1}
export GEMMA4_PORT=${GEMMA4_PORT:-8090}
export GEMMA4_ENGINE_PROCESSES=${GEMMA4_ENGINE_PROCESSES:-1}
export GEMMA4_WORKERS
export GEMMA4_CONTEXT_TOKENS
export GEMMA4_MAX_OUTPUT_TOKENS
export GEMMA4_KV_BUDGET_GB
if [[ -n "${GEMMA4_MEMORY_LIMIT_GB:-}" ]]; then
    export GEMMA4_MEMORY_LIMIT_GB
else
    unset GEMMA4_MEMORY_LIMIT_GB
fi
export GEMMA4_TIMEOUT_S=${GEMMA4_TIMEOUT_S:-900}
export GEMMA4_SCHEDULER_WAIT_S=${GEMMA4_SCHEDULER_WAIT_S:-60.0}
export GEMMA4_SCHEDULER_PREFILL_MS_PER_TOKEN=${GEMMA4_SCHEDULER_PREFILL_MS_PER_TOKEN:-0.0}
export GEMMA4_SCHEDULER_MAX_HEAVY=${GEMMA4_SCHEDULER_MAX_HEAVY:-1}
export GEMMA4_SCHEDULER_HEAVY_WORK_RATIO=${GEMMA4_SCHEDULER_HEAVY_WORK_RATIO:-0.25}
export GEMMA4_CHILD_PORT_BASE=${GEMMA4_CHILD_PORT_BASE:-18090}

GEMMA4_MODEL_ID=${GEMMA4_MODEL_ID:-gemma-4-26b-a4b-it}
GEMMA4_RECEIPT=${GEMMA4_RECEIPT:-${GEMMA4_POOL}.import.json}
GEMMA4_AUTH_RECEIPT=${GEMMA4_AUTH_RECEIPT:-${GEMMA4_POOL}.auth.json}

umask 077
make -C "$ROOT" GEMMA4_COMPAT_PYTHON="$GEMMA4_PYTHON" \
    gemma4-server.build.json

child_args=()
if [[ -n "${GEMMA4_KV_CACHE_ROOT:-}" ]]; then
    child_args+=(--gemma-kv-cache-root "$GEMMA4_KV_CACHE_ROOT")
fi

if [[ -n "${GEMMA4_DPR_ROOT:-}" ]]; then
    GEMMA4_DPR_MODE=${GEMMA4_DPR_MODE:-dynamic}
    if [[ "$GEMMA4_DPR_MODE" != persist && "$GEMMA4_DPR_MODE" != dynamic ]]; then
        printf 'gemma4 endpoint: GEMMA4_DPR_MODE must be persist or dynamic\n' >&2
        exit 2
    fi
    child_args+=(
        --gemma-dpr-root "$GEMMA4_DPR_ROOT"
        --gemma-dpr-mode "$GEMMA4_DPR_MODE"
    )
    if [[ -n "${GEMMA4_DPR_BUDGET_GB:-}" ]]; then
        child_args+=(--gemma-dpr-budget-gb "$GEMMA4_DPR_BUDGET_GB")
    fi
    if [[ -n "${GEMMA4_DPR_SERIAL_MS_PER_TOKEN:-}" ]]; then
        child_args+=(
            --gemma-dpr-serial-ms-per-token
            "$GEMMA4_DPR_SERIAL_MS_PER_TOKEN"
        )
    fi
    if [[ -n "${GEMMA4_DPR_RETENTION_GB:-}" ]]; then
        child_args+=(--gemma-dpr-retention-gb "$GEMMA4_DPR_RETENTION_GB")
    fi
fi

prefill_args=()
if [[ -n "${GEMMA4_PREFILL_CHUNK_TOKENS:-}" ]]; then
    prefill_args=(
        --gemma-prefill-chunk-tokens "$GEMMA4_PREFILL_CHUNK_TOKENS"
    )
fi

exec env -u PYTHONPATH "$GEMMA4_PYTHON" \
    "$ROOT/server/scheduler.py" \
    --host "$GEMMA4_HOST" \
    --port "$GEMMA4_PORT" \
    --engine-processes "$GEMMA4_ENGINE_PROCESSES" \
    --scheduler-wait-s "$GEMMA4_SCHEDULER_WAIT_S" \
    --prefill-ms-per-token "$GEMMA4_SCHEDULER_PREFILL_MS_PER_TOKEN" \
    --max-heavy-active "$GEMMA4_SCHEDULER_MAX_HEAVY" \
    --heavy-work-ratio "$GEMMA4_SCHEDULER_HEAVY_WORK_RATIO" \
    --child-port-base "$GEMMA4_CHILD_PORT_BASE" \
    --python "$GEMMA4_PYTHON" \
    --serve-script "$GEMMA4_ENDPOINT_CHILD" \
    -- \
    --model "$GEMMA4_MODEL_ID" \
    --gemma-source-dir "$GEMMA4_SOURCE_DIR" \
    --gemma-pool "$GEMMA4_POOL" \
    --gemma-receipt "$GEMMA4_RECEIPT" \
    --gemma-auth-receipt "$GEMMA4_AUTH_RECEIPT" \
    --gemma-python "$GEMMA4_PYTHON" \
    --gemma-workers "$GEMMA4_WORKERS" \
    --gemma-context-tokens "$GEMMA4_CONTEXT_TOKENS" \
    --gemma-max-output-tokens "$GEMMA4_MAX_OUTPUT_TOKENS" \
    ${prefill_args[@]+"${prefill_args[@]}"} \
    --gemma-kv-budget-gb "$GEMMA4_KV_BUDGET_GB" \
    --gemma-timeout-s "$GEMMA4_TIMEOUT_S" \
    ${child_args[@]+"${child_args[@]}"} \
    "$@"
