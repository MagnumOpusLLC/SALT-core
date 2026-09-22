#!/usr/bin/env bash
# setup_dev_env.sh -- the salt dev environment: the env vars, the
# build, the track check, the gate verify. Source or run; all vars
# are overridable (the ${VAR:-default} pattern).
set -uo pipefail

# ---- the env vars (override any) -----------------------------------
export SALT_REPO="${SALT_REPO:-$HOME/salt}"
export SALT_TRACK="${SALT_TRACK:-/Volumes/prod/hf-cache/mlx4-eng}"
export SALT_MODEL="${SALT_MODEL:-qwen36}"
export SALT_QUANT="${SALT_QUANT:-mlx4}"
export SALT_THREADS="${SALT_THREADS:-3}"
export SALT_CACHE_GB="${SALT_CACHE_GB:-2}"
export SALT_PIN_LAYERS="${SALT_PIN_LAYERS:-40}"
export SALT_MEM_LIMIT_GB="${SALT_MEM_LIMIT_GB:-8}"
export SALT_GEN="${SALT_GEN:-20}"

# the engine run defaults (the operating mode)
engine_cfg() { python3 "$SALT_REPO/tools/engine_config.py" "$1"; }
export SALT_PREFILL_CHUNK="${SALT_PREFILL_CHUNK:-$(engine_cfg SALT_PREFILL_CHUNK)}"
export SALT_PREFILL_B="${SALT_PREFILL_B:-$(engine_cfg SALT_PREFILL_B)}"
export SALT_M2_BATCH="${SALT_M2_BATCH:-$(engine_cfg SALT_M2_BATCH)}"
export SALT_M2_BATCH_B="${SALT_M2_BATCH_B:-$(engine_cfg SALT_M2_BATCH_B)}"
export SALT_CACHE_MODE="${SALT_CACHE_MODE:-$(engine_cfg SALT_CACHE_MODE)}"
export SALT_GREEDY="${SALT_GREEDY:-1}"

# the test schema
export SALT_GATE_PROMPT="${SALT_GATE_PROMPT:-200}"
export SALT_GATE_GEN="${SALT_GATE_GEN:-20}"

echo "== salt dev env"
echo "  repo:   $SALT_REPO"
echo "  track:  $SALT_TRACK"
echo "  model:  $SALT_MODEL / $SALT_QUANT"
echo "  run:    threads=$SALT_THREADS cache=${SALT_CACHE_GB}GB pin=$SALT_PIN_LAYERS mem=${SALT_MEM_LIMIT_GB}GB"

# ---- the build ------------------------------------------------------
if [ ! -f "$SALT_REPO/Makefile" ]; then
    echo "ERROR: no Makefile at $SALT_REPO (set SALT_REPO)" >&2
    exit 1
fi
echo "== build"
make -C "$SALT_REPO" salt || { echo "BUILD FAILED" >&2; exit 1; }
echo "build ok"

# ---- the track check ------------------------------------------------
if [ ! -d "$SALT_TRACK" ] || [ ! -f "$SALT_TRACK/trunk.bin" ]; then
    echo "ERROR: the track is missing at $SALT_TRACK (set SALT_TRACK)" >&2
    exit 1
fi
echo "track ok: $SALT_TRACK"

# ---- the gate verify ------------------------------------------------
echo "== kv-gate"
bash "$SALT_REPO/tools/test/kv-gate.sh" 2>&1 | grep -a "kv-gate:" | head -1
echo "== phase gate"
bash "$SALT_REPO/tools/test/phase-gate.sh" 2>&1 | grep -aE "PHASE GATE|kv-gate:" | head -3

echo "== dev env ready"
