#!/usr/bin/env bash
# Start the general OpenAI-compatible Salt API server.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
VENV=${SALT_SERVER_VENV:-$ROOT/.venv}
PYTHON=${SALT_SERVER_PYTHON:-$VENV/bin/python3}
SALT_BIN=${SALT_BIN:-$ROOT/salt}

raw_args=("$@")
args=()
preload_kv=""
preload_kv_set=0
for ((i = 0; i < ${#raw_args[@]}; i = i + 1)); do
    case "${raw_args[$i]}" in
        --gemma-preload-kv)
            if ((i + 1 >= ${#raw_args[@]})) || ((preload_kv_set)); then
                printf 'Salt server: --gemma-preload-kv requires one file\n' >&2
                exit 2
            fi
            preload_kv_set=1
            preload_kv=${raw_args[$((i + 1))]}
            if [[ -z "$preload_kv" ]]; then
                printf 'Salt server: --gemma-preload-kv requires one file\n' >&2
                exit 2
            fi
            i=$((i + 1))
            ;;
        --gemma-preload-kv=*)
            if ((preload_kv_set)); then
                printf 'Salt server: specify --gemma-preload-kv once\n' >&2
                exit 2
            fi
            preload_kv_set=1
            preload_kv=${raw_args[$i]#*=}
            if [[ -z "$preload_kv" ]]; then
                printf 'Salt server: --gemma-preload-kv requires one file\n' >&2
                exit 2
            fi
            ;;
        *) args+=("${raw_args[$i]}") ;;
    esac
done
rotation_backend=0
backend_count=0
has_expert_budget=0
has_memory_limit=0
has_server_runner=0
has_server_receipt=0
has_shared_kv=0
for ((i = 0; i < ${#args[@]}; i = i + 1)); do
    case "${args[$i]}" in
        --backend)
            if ((i + 1 >= ${#args[@]})); then
                printf 'Salt server: --backend requires a value\n' >&2
                exit 2
            fi
            backend_count=$((backend_count + 1))
            if [[ "${args[$((i + 1))]}" == "gemma4-rotation" ]]; then
                args[$((i + 1))]=gemma4
                rotation_backend=1
            elif [[ "${args[$((i + 1))]}" == "gemma4-rotation-cpu" ]]; then
                args[$((i + 1))]=gemma4
                rotation_backend=2
            fi
            i=$((i + 1))
            ;;
        --backend=*)
            backend_count=$((backend_count + 1))
            if [[ "${args[$i]}" == "--backend=gemma4-rotation" ]]; then
                args[$i]=--backend=gemma4
                rotation_backend=1
            elif [[ "${args[$i]}" == "--backend=gemma4-rotation-cpu" ]]; then
                args[$i]=--backend=gemma4
                rotation_backend=2
            fi
            ;;
        --gemma-expert-budget-gb|--gemma-memory-limit-gb|--gemma-server-runner|--gemma-server-build-receipt|--gemma-shared-kv)
            if ((i + 1 >= ${#args[@]})); then
                printf 'Salt server: %s requires a value\n' "${args[$i]}" >&2
                exit 2
            fi
            case "${args[$i]}" in
                --gemma-expert-budget-gb) has_expert_budget=1 ;;
                --gemma-memory-limit-gb) has_memory_limit=1 ;;
                --gemma-server-runner) has_server_runner=1 ;;
                --gemma-server-build-receipt) has_server_receipt=1 ;;
                --gemma-shared-kv) has_shared_kv=1 ;;
            esac
            i=$((i + 1))
            ;;
        --gemma-expert-budget-gb=*) has_expert_budget=1 ;;
        --gemma-memory-limit-gb=*) has_memory_limit=1 ;;
        --gemma-server-runner=*) has_server_runner=1 ;;
        --gemma-server-build-receipt=*) has_server_receipt=1 ;;
        --gemma-shared-kv=*) has_shared_kv=1 ;;
    esac
done

if ((backend_count > 1)); then
    printf 'Salt server: specify exactly one --backend\n' >&2
    exit 2
fi

if ((preload_kv_set)) && ((rotation_backend == 0)); then
    printf 'Salt server: --gemma-preload-kv is supported only by a Gemma rotation backend\n' >&2
    exit 2
fi

if ((rotation_backend)); then
    if ((rotation_backend == 2)); then
        ROTATION_RUNNER="$ROOT/rotation/bin/gemma4-server-cpu"
        ROTATION_RECIPE=mac
        ROTATION_CONFIG="$ROOT/rotation/engine-cpu.config"
        ROTATION_BUILD="bash rotation/build-server-cpu.sh"
        ROTATION_COMPUTE="pure-cpu"
    else
        ROTATION_RUNNER="$ROOT/rotation/bin/gemma4-server"
        ROTATION_RECIPE=mac-metal
        ROTATION_CONFIG="$ROOT/rotation/engine.config"
        ROTATION_BUILD="bash rotation/build-server.sh"
        ROTATION_COMPUTE="metal"
    fi
    ROTATION_RECEIPT="$ROTATION_RUNNER.build.json"
    if [[ "$(uname -s)" != "Darwin" ]]; then
        printf 'Salt server: Gemma rotation runners currently require Darwin\n' >&2
        exit 2
    fi
    if ((has_server_runner || has_server_receipt)); then
        printf 'Salt server: gemma4-rotation owns its dedicated runner and receipt\n' >&2
        exit 2
    fi
    if [[ -n "${SALT_GEMMA_PLATFORM_RECIPE:-}" &&
          "$SALT_GEMMA_PLATFORM_RECIPE" != "$ROTATION_RECIPE" ]]; then
        printf 'Salt server: selected rotation backend requires SALT_GEMMA_PLATFORM_RECIPE=%s\n' "$ROTATION_RECIPE" >&2
        exit 2
    fi
    if [[ ! -x "$ROTATION_RUNNER" || ! -f "$ROTATION_RECEIPT" ]]; then
        printf 'Salt server: rotation runner/receipt missing; run: %s\n' "$ROTATION_BUILD" >&2
        exit 2
    fi
    if [[ ! -f "$ROTATION_CONFIG" ]]; then
        printf 'Salt server: rotation engine configuration is missing: %s\n' "$ROTATION_CONFIG" >&2
        exit 2
    fi
    export SALT_GEMMA_PLATFORM_RECIPE="$ROTATION_RECIPE"
    args+=(--gemma-server-runner "$ROTATION_RUNNER")
    args+=(--gemma-server-build-receipt "$ROTATION_RECEIPT")

    if ((preload_kv_set)); then
        if ((has_shared_kv)); then
            printf 'Salt server: --gemma-preload-kv and --gemma-shared-kv are mutually exclusive\n' >&2
            exit 2
        fi
        preload_dir=$(cd "$(dirname "$preload_kv")" 2>/dev/null && pwd -P) || {
            printf 'Salt server: preload KV parent directory is invalid: %s\n' "$preload_kv" >&2
            exit 2
        }
        preload_kv="$preload_dir/$(basename "$preload_kv")"
        if [[ ! -f "$preload_kv" || -L "$preload_kv" ]]; then
            printf 'Salt server: preload KV must be a regular non-symlink file: %s\n' "$preload_kv" >&2
            exit 2
        fi
        preload_mode=$(stat -f '%Lp' "$preload_kv")
        if [[ ! "$preload_mode" =~ ^[0-7]{3,4}$ ]] ||
                (( (8#$preload_mode & 8#077) != 0 )); then
            printf 'Salt server: preload KV must have no group/other permissions: %s\n' "$preload_kv" >&2
            exit 2
        fi
        args+=(--gemma-shared-kv "$preload_kv")
    fi

    if (( ! has_expert_budget )); then
        if ((rotation_backend == 2)); then
            args+=(--gemma-expert-budget-gb 2)
        else
            args+=(--gemma-expert-budget-gb 3)
        fi
    fi
    if (( ! has_memory_limit )); then
        args+=(--gemma-memory-limit-gb 6)
    fi
    printf 'Salt server: Gemma 4 layer rotation | compute=%s | runner=%s | recipe=%s | model=gemma-4-26b-a4b-it | preload_kv=%s\n' \
        "$ROTATION_COMPUTE" "$ROTATION_RUNNER" "$ROTATION_RECIPE" "${preload_kv:-none}" >&2
fi

if [[ -z "${SALT_SERVER_PYTHON:-}" && ! -d "$VENV" ]]; then
    BOOTSTRAP_PYTHON=${SALT_BOOTSTRAP_PYTHON:-python3}
    command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1 || {
        printf 'Salt server: bootstrap Python not found: %s\n' \
            "$BOOTSTRAP_PYTHON" >&2
        exit 2
    }
    printf 'Salt server: creating virtual environment: %s\n' "$VENV" >&2
    if ! "$BOOTSTRAP_PYTHON" -m venv "$VENV"; then
        if [[ ! -x "$PYTHON" ]]; then
            "$BOOTSTRAP_PYTHON" -m venv --without-pip "$VENV"
        fi
        printf 'Salt server: ensurepip unavailable; using stdlib-only venv\n' >&2
    fi

    REQUIREMENTS=${SALT_SERVER_REQUIREMENTS:-$ROOT/server/requirements.txt}
    if [[ -f "$REQUIREMENTS" ]]; then
        "$PYTHON" -m pip --version >/dev/null 2>&1 || {
            printf 'Salt server: requirements exist but venv pip is unavailable\n' >&2
            exit 2
        }
        "$PYTHON" -m pip install --disable-pip-version-check \
            --requirement "$REQUIREMENTS"
    fi
fi

if [[ ! -x "$PYTHON" ]]; then
    printf 'Salt server: invalid project virtual environment: %s\n' "$VENV" >&2
    exit 2
fi

cd "$ROOT"
exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    "$ROOT/server/serve.py" \
    --host "${SALT_API_HOST:-127.0.0.1}" \
    --port "${SALT_API_PORT:-8090}" \
    --salt "$SALT_BIN" \
    "${args[@]}"
