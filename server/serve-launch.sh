#!/usr/bin/env bash
# Start the general OpenAI-compatible Salt API server.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
VENV=${SALT_SERVER_VENV:-$ROOT/.venv}
PYTHON=${SALT_SERVER_PYTHON:-$VENV/bin/python3}
SALT_BIN=${SALT_BIN:-$ROOT/salt}

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
    "$@"
