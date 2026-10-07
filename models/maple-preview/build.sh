#!/usr/bin/env bash
# CPU-only native build: deliberately no Metal/Xcode targets or dependencies.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
OUT=${MAPLE_BIN_DIR:-$ROOT/../bin}
PYTHON=${SALT_BUILD_PYTHON:-python3}
mkdir -p "$OUT"
SOURCES=(models/maple-preview/server.c models/maple-preview/model.c src/model.c
  src/int2.c src/st.c src/cache.c src/tokenizer.c src/mem.c src/text_verify.c
  src/text_verify_cpu.c src/text_exec.c src/text_token.c src/tensorops.c
  src/area_scan.c src/attn.c src/kernels.c src/simd.c src/gpu_stub.c src/bitmath.c
  src/sampling.c src/dpr.c src/dpr_store.c src/dpr_stats.c src/sha256.c)
FLAGS=(-std=c99 -O2 -flto -Wall -Wextra -Werror -pedantic -pthread -ffp-contract=off
  -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc)
if [[ $(uname -s) == Linux ]]; then FLAGS+=(-D_GNU_SOURCE); fi
if [[ $(uname -s) == Darwin ]]; then FLAGS+=(-D_DARWIN_C_SOURCE=); fi
printf 'MAPLE_BUILD_BEGIN output=%s\n' "$OUT/maple-server"
"${CC:-cc}" "${FLAGS[@]}" "${SOURCES[@]}" -lm -o "$OUT/maple-server"
env -u PYTHONPATH "$PYTHON" - "$OUT/maple-server" "${SOURCES[@]}" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1])
paths=set(sys.argv[2:])
paths.update(str(p) for p in pathlib.Path('include/salt').glob('*.h'))
paths.update(str(p) for p in pathlib.Path('src').glob('*.h'))
paths.add('models/maple-preview/engine.config')
receipt={'schema':'salt.maple.build.v1','abi':'salt-maple-int2-f32-v1',
         'binary_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
         'source_sha256':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sorted(paths)},
         'flags':'C99 O2 pthread ffp-contract=off CPU-only', 'runtime_qualified':False}
out.with_name(out.name+'.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('MAPLE_BUILD_PASS',receipt['binary_sha256'])
PY
