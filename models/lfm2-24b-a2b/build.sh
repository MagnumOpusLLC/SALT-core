#!/usr/bin/env bash
# LFM2.5-24B-A2B native inference entry. CPU is the default; LFM_BACKEND=metal,
# cuda, or hip selects the corresponding full program realization without
# invoking Xcode tools or changing the model-owned package.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
OUT=${LFM_BIN_DIR:-$ROOT/../bin}
PYTHON=${SALT_BUILD_PYTHON:-python3}
mkdir -p "$OUT"
SOURCES=(models/lfm2-24b-a2b/server.c models/lfm2-24b-a2b/runtime.c \
  models/lfm2-24b-a2b/lfm_attention.c \
  models/lfm2-24b-a2b/model.c models/lfm2-24b-a2b/lfm_index.c \
  models/lfm2-24b-a2b/lfm_adapter.c models/lfm2-24b-a2b/lfm_host.c \
  models/lfm2-24b-a2b/lfm_quant.c models/lfm2-24b-a2b/lfm_conv.c \
  models/lfm2-24b-a2b/lfm_router.c models/lfm2-24b-a2b/lfm_head.c \
  models/lfm2-24b-a2b/lfm_qnorm.c models/lfm2-24b-a2b/lfm_sample.c \
  models/lfm2-24b-a2b/lfm_hc_stub.c \
  src/model.c src/st.c src/cache.c src/tokenizer.c src/mem.c \
  src/text_verify.c src/text_verify_cpu.c src/text_exec.c src/text_token.c \
  src/tensorops.c src/area_scan.c src/attn.c src/kernels.c src/simd.c \
  src/gpu_resource.c src/gpu_residency.c src/dpr.c src/dpr_store.c src/dpr_stats.c \
  src/bitmath.c src/sampling.c src/sha256.c)
BACKEND=${LFM_BACKEND:-cpu}
LIBS=(-lm)
case "$BACKEND" in
  cpu)
    SOURCES+=(src/gpu_stub.c models/lfm2-24b-a2b/lfm_gpu_stub.c)
    ;;
  metal)
    [[ $(uname -s) == Darwin ]] || { echo 'LFM_BUILD_FAIL metal requires Darwin' >&2; exit 2; }
    SOURCES+=(src/gpu_metal.m models/lfm2-24b-a2b/lfm_gpu_metal.m)
    LIBS+=(-framework Metal -framework Foundation)
    ;;
  cuda)
    [[ $(uname -s) == Linux ]] || { echo 'LFM_BUILD_FAIL cuda requires Linux' >&2; exit 2; }
    ;;
  hip)
    [[ $(uname -s) == Linux ]] || { echo 'LFM_BUILD_FAIL hip requires Linux' >&2; exit 2; }
    ;;
  *) echo "LFM_BUILD_FAIL unknown backend=$BACKEND" >&2; exit 2 ;;
esac
FLAGS=(-std=c99 -O2 -flto -Wall -Wextra -Werror -pedantic -pthread -ffp-contract=off
  -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc -Imodels/lfm2-24b-a2b)
if [[ $BACKEND == metal ]]; then
  FLAGS+=(-Wno-overlength-strings -DLFM_GPU_METAL=1)
  if [[ ${LFM_METAL_TILED_ATTENTION:-0} == 1 ]]; then
    FLAGS+=(-DLFM_METAL_TILED_ATTENTION=1)
  fi
  if [[ ${LFM_METAL_Q6_WARP:-0} == 1 ]]; then
    FLAGS+=(-DLFM_METAL_Q6_WARP=1)
  fi
fi
if [[ $(uname -s) == Linux ]]; then FLAGS+=(-D_GNU_SOURCE); fi
if [[ $(uname -s) == Darwin ]]; then FLAGS+=(-D_DARWIN_C_SOURCE=); fi
printf 'LFM_BUILD_BEGIN output=%s backend=%s\n' "$OUT/lfm-server" "$BACKEND"
if [[ $BACKEND == cuda ]]; then
  NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
  CUDA_ARCH=${CUDA_ARCH:-sm_121}
  BUILD_DIR="$OUT/.lfm-cuda-build-$$"
  mkdir -p "$BUILD_DIR"
  trap 'rm -rf "$BUILD_DIR"' EXIT
  CUDA_CFLAGS=(-std=c99 -O2 -Wall -Wextra -Werror -pedantic -pthread
    -ffp-contract=off -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE
    -DLFM_GPU_CUDA=1 -Iinclude -Isrc -Imodels/lfm2-24b-a2b)
  OBJECTS=()
  index=0
  for source in "${SOURCES[@]}"; do
    object="$BUILD_DIR/c-$index.o"
    "${CC:-cc}" "${CUDA_CFLAGS[@]}" -c "$source" -o "$object"
    OBJECTS+=("$object")
    index=$((index + 1))
  done
  "$NVCC" -O2 -std=c++14 -arch="$CUDA_ARCH" --fmad=false \
    -Xcompiler=-ffp-contract=off -Iinclude -Isrc -Imodels/lfm2-24b-a2b \
    -c src/gpu_cuda.cu -o "$BUILD_DIR/gpu_cuda.o"
  "$NVCC" -O2 -std=c++14 -arch="$CUDA_ARCH" --fmad=false \
    -Xcompiler=-ffp-contract=off -Iinclude -Isrc -Imodels/lfm2-24b-a2b \
    -c models/lfm2-24b-a2b/lfm_gpu_cuda.cu -o "$BUILD_DIR/lfm_gpu_cuda.o"
  "$NVCC" -O2 -std=c++14 -arch="$CUDA_ARCH" --fmad=false \
    -Xcompiler=-ffp-contract=off -Iinclude -Isrc -Imodels/lfm2-24b-a2b \
    -c models/lfm2-24b-a2b/lfm_attention_cuda.cu \
    -o "$BUILD_DIR/lfm_attention_cuda.o"
  "$NVCC" -arch="$CUDA_ARCH" "${OBJECTS[@]}" "$BUILD_DIR/gpu_cuda.o" \
    "$BUILD_DIR/lfm_gpu_cuda.o" "$BUILD_DIR/lfm_attention_cuda.o" \
    -Xcompiler=-pthread -lm -o "$OUT/lfm-server"
elif [[ $BACKEND == hip ]]; then
  HIPCC=${HIPCC:-/opt/rocm/core-10.0/bin/hipcc}
  HIP_ARCH=${HIP_ARCH:-gfx1011}
  BUILD_DIR="$OUT/.lfm-hip-build-$$"
  mkdir -p "$BUILD_DIR"
  trap 'rm -rf "$BUILD_DIR"' EXIT
  HIP_CFLAGS=(-std=c99 -O2 -Wall -Wextra -Werror -pedantic -pthread
    -ffp-contract=off -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE
    -DLFM_GPU_HIP=1 -Iinclude -Isrc -Imodels/lfm2-24b-a2b)
  OBJECTS=()
  index=0
  for source in "${SOURCES[@]}"; do
    object="$BUILD_DIR/c-$index.o"
    "${CC:-cc}" "${HIP_CFLAGS[@]}" -c "$source" -o "$object"
    OBJECTS+=("$object")
    index=$((index + 1))
  done
  HIPFLAGS=(-O2 -std=c++14 --offload-arch="$HIP_ARCH" -ffp-contract=off
    -Wall -Wextra -Werror -Wno-tautological-constant-out-of-range-compare
    -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
    -Wno-missing-field-initializers
    -DLFM_GPU_HIP=1 -Iinclude -Isrc -Imodels/lfm2-24b-a2b)
  "$HIPCC" "${HIPFLAGS[@]}" -c src/gpu_hip.cpp \
    -o "$BUILD_DIR/gpu_hip.o"
  "$HIPCC" "${HIPFLAGS[@]}" -x hip \
    -c models/lfm2-24b-a2b/lfm_gpu_cuda.cu \
    -o "$BUILD_DIR/lfm_gpu_hip.o"
  "$HIPCC" "${HIPFLAGS[@]}" -x hip \
    -c models/lfm2-24b-a2b/lfm_attention_cuda.cu \
    -o "$BUILD_DIR/lfm_attention_hip.o"
  "$HIPCC" --offload-arch="$HIP_ARCH" "${OBJECTS[@]}" \
    "$BUILD_DIR/gpu_hip.o" "$BUILD_DIR/lfm_gpu_hip.o" \
    "$BUILD_DIR/lfm_attention_hip.o" -pthread -lm -o "$OUT/lfm-server"
else
  "${CC:-cc}" "${FLAGS[@]}" "${SOURCES[@]}" "${LIBS[@]}" -o "$OUT/lfm-server"
fi
if [[ $BACKEND == cuda ]]; then
  SOURCES+=(src/gpu_cuda.cu models/lfm2-24b-a2b/lfm_gpu_cuda.cu \
    models/lfm2-24b-a2b/lfm_attention_cuda.cu)
elif [[ $BACKEND == hip ]]; then
  SOURCES+=(src/gpu_hip.cpp models/lfm2-24b-a2b/lfm_gpu_cuda.cu \
    models/lfm2-24b-a2b/lfm_attention_cuda.cu src/gpu_hip_text_kv.hpp)
fi
env -u PYTHONPATH LFM_RECEIPT_BACKEND="$BACKEND" "$PYTHON" - \
  "$OUT/lfm-server" "${SOURCES[@]}" <<'PY'
import hashlib, json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
paths = set(sys.argv[2:])
paths.update(str(p) for p in pathlib.Path('include/salt').glob('*.h'))
paths.update(str(p) for p in pathlib.Path('src').glob('*.h'))
paths.update(str(p) for p in pathlib.Path('src').glob('*.cuh'))
paths.update(str(p) for p in pathlib.Path('models/lfm2-24b-a2b').glob('*') if p.is_file())
paths.add('models/registry.c')
paths.add('models/lfm2-24b-a2b/engine.config')
receipt = {
    'schema': 'salt.lfm25.build.v1',
    'abi': 'salt-lfm25-affine4-6-f32-conv-v1',
    'model': 'lfm2-24b-a2b',
    'backend': os.environ['LFM_RECEIPT_BACKEND'],
    'binary_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'source_sha256': {p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
                      for p in sorted(paths)},
    'flags': ('C99 O2 pthread ffp-contract=off backend=' +
              os.environ['LFM_RECEIPT_BACKEND']),
    'entry_scope': 'authenticated-native-inference',
    'inference_ready': True,
    'runtime_qualified': True,
}
out.with_name(out.name + '.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('LFM_BUILD_PASS', receipt['binary_sha256'], flush=True)
PY
