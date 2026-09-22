#!/usr/bin/env bash
# Authenticated experimental Gemma 4 OpenAI-shaped server.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
CONFIG="$ROOT/models/gemma4-26b-a4b/server.config"

if [[ ! -f "$CONFIG" ]]; then
    printf 'gemma4 server: missing model config: %s\n' "$CONFIG" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

: "${GEMMA4_SOURCE_DIR:?set GEMMA4_SOURCE_DIR to the authenticated MLX source directory}"
: "${GEMMA4_POOL:?set GEMMA4_POOL to the imported expert pool.bin}"

GEMMA4_RECEIPT=${GEMMA4_RECEIPT:-${GEMMA4_POOL}.import.json}
GEMMA4_AUTH_RECEIPT=${GEMMA4_AUTH_RECEIPT:-${GEMMA4_POOL}.auth.json}
GEMMA4_PYTHON=${GEMMA4_PYTHON:-python3}

umask 077

make -C "$ROOT" gemma4-build-receipts

PREFILL_ARGS=()
if [[ -n "$GEMMA4_PREFILL_CHUNK_TOKENS" ]]; then
    PREFILL_ARGS=(--gemma-prefill-chunk-tokens "$GEMMA4_PREFILL_CHUNK_TOKENS")
fi

exec env -u PYTHONPATH "$GEMMA4_PYTHON" "$ROOT/server/serve.py" \
    --backend gemma4 \
    --host "${GEMMA4_HOST:-127.0.0.1}" \
    --port "${GEMMA4_PORT:-8090}" \
    --model "$GEMMA4_MODEL_ID" \
    --gemma-source-dir "$GEMMA4_SOURCE_DIR" \
    --gemma-pool "$GEMMA4_POOL" \
    --gemma-receipt "$GEMMA4_RECEIPT" \
    --gemma-auth-receipt "$GEMMA4_AUTH_RECEIPT" \
    --gemma-python "$GEMMA4_PYTHON" \
    --gemma-workers "$GEMMA4_WORKERS" \
    --gemma-context-tokens "$GEMMA4_CONTEXT_TOKENS" \
    --gemma-max-output-tokens "$GEMMA4_MAX_OUTPUT_TOKENS" \
    "${PREFILL_ARGS[@]}" \
    --gemma-kv-budget-gb "$GEMMA4_KV_BUDGET_GB" \
    --gemma-timeout-s "$GEMMA4_TIMEOUT_S" \
    "$@"
