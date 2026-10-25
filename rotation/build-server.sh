#!/usr/bin/env bash
# Build the isolated Gemma 4 layer-rotation native server and receipt.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${SALT_PROJECT_PYTHON:-$ROOT/.venv/bin/python3}
BIN_DIR="$ROOT/rotation/bin"
BINARY="$BIN_DIR/gemma4-server"
RECEIPT="$BINARY.build.json"
OBJECT="$BIN_DIR/gpu_metal.o"
umask 077
cd "$ROOT"
mkdir -p "$BIN_DIR"
trap 'rm -f "$OBJECT"' EXIT

if [[ ! -x "$PY" || "$PY" != /* ]]; then
  printf 'rotation build: SALT_PROJECT_PYTHON must be an absolute executable path\n' >&2
  exit 2
fi

COMPAT=$(env -u PYTHONPATH "$PY" server/model/gemma4_compat_projection.py --hex)
LEGACY5=aa4eb350cfcecbe829391c5d02e139de0f4c1331d0e3f45a0f2f2daf70637a24
LEGACY4=2744872d508028385e7f631ba6a3c8a7596c2a9d257f5abd56d021ba04f942e2
COMMON_FLAGS=(
  -std=c99 -O2 -Wall -Wextra -Werror -pthread -flto -ffp-contract=off
  "-DSALT_GEMMA4_KV_COMPAT_SHA256=\"$COMPAT\""
  "-DSALT_GEMMA4_KV_LEGACY_V5_SHA256=\"$LEGACY5\""
  "-DSALT_GEMMA4_KV_LEGACY_V4_SHA256=\"$LEGACY4\""
  -Irotation/engine/include -Iinclude -Irotation/engine
  -Irotation/engine/model -Imodels/gemma4-26b-a4b/src
)
FLAGS_TEXT="${COMMON_FLAGS[*]} -lm -framework Accelerate -framework Metal -framework Foundation"

printf 'ROTATION_BUILD_BEGIN head=%s compatibility=%s\n' "$(git rev-parse HEAD)" "$COMPAT"
clang -std=c99 -O2 -Wall -Wextra -flto \
  -Irotation/engine/include -Iinclude -Irotation/engine \
  -Irotation/engine/model -Imodels/gemma4-26b-a4b/src \
  -c rotation/engine/gpu_metal.m -o "$OBJECT"

cc "${COMMON_FLAGS[@]}" -o "$BINARY" \
  models/gemma4-26b-a4b/server.c \
  rotation/engine/model/inference.c rotation/engine/model/gemma4_text.c \
  rotation/engine/model/gemma4_vision.c rotation/engine/model/gemma4.c \
  rotation/engine/tokenizer.c rotation/engine/cache.c rotation/engine/kernels.c \
  rotation/engine/simd.c rotation/engine/nvfp4.c "$OBJECT" \
  rotation/engine/sha256.c rotation/engine/model.c models/registry.c \
  models/qwen36/model.c rotation/engine/model/model.c models/maple-preview/model.c \
  rotation/engine/state.c rotation/engine/gpu_resource.c rotation/engine/attn.c \
  rotation/engine/attn_batch.c rotation/engine/moe_group.c rotation/engine/text_exec.c \
  rotation/engine/text_token.c rotation/engine/area_scan.c rotation/engine/tensorops.c \
  rotation/engine/text_verify.c rotation/engine/text_verify_cpu.c rotation/engine/bitmath.c \
  rotation/engine/mem.c rotation/engine/dpr.c rotation/engine/dpr_store.c \
  rotation/engine/dpr_stats.c rotation/engine/sampling.c rotation/engine/gpu_residency.c \
  rotation/engine/trace.c rotation/layer_rotation.c rotation/inference_rotation.c \
  -lm -framework Accelerate -framework Metal -framework Foundation

env -u PYTHONPATH "$PY" server/model/gemma4_build.py create \
  --binary "$BINARY" --receipt "$RECEIPT" --compiler cc \
  --flags "$FLAGS_TEXT" --source-compatibility "$COMPAT" \
  --math-abi 'salt-bitmath-v2;gemma4-vision-accelerate-v1' \
  --blas-abi apple-accelerate-lp64
IDENTITY=$(env -u PYTHONPATH "$PY" server/model/gemma4_build.py verify \
  --binary "$BINARY" --receipt "$RECEIPT" \
  --source-compatibility "$COMPAT")
printf 'ROTATION_BUILD_PASS binary=%s receipt=%s identity=%s\n' \
  "$BINARY" "$RECEIPT" "$IDENTITY"
