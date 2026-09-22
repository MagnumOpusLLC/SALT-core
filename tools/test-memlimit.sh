#!/usr/bin/env bash
# Verify graceful memory-limit enforcement uses application footprint,
# reports the reason, exits 3, and leaves the unlimited control intact.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIX="$(mktemp -d "${TMPDIR:-/tmp}/salt-memlimit.XXXXXX")"
LIMIT_LOG="$FIX/limit.log"
HYBRID_LIMIT_LOG="$FIX/hybrid-limit.log"
CONTROL_LOG="$FIX/control.log"
PREFILL_CONTROL_LOG="$FIX/prefill-control.log"
trap 'rm -rf "$FIX"' EXIT

"$REPO_ROOT/make-fixture" --dir "$FIX/model" \
  --layers 4 --experts 16 --topk 3 --hidden 128 --latent 64 \
  --moe-inter 128 --expert-bytes 8192 --trunk-bytes 16384 --seed 7 \
  >/dev/null 2>&1

set +e
env SALT_GPU=0 SALT_PREFILL_CHUNK=0 SALT_CACHE_MODE=arena \
  "$REPO_ROOT/salt" "$FIX/model" \
  --trunk "$FIX/model/trunk.bin" --offsets "$FIX/model/trunk.offsets" \
  --gen 4 --mem-limit-gb 0.000001 >"$LIMIT_LOG" 2>&1
limit_rc=$?
set -e

if [ "$limit_rc" -ne 3 ]; then
  echo "FAIL memlimit: exit=$limit_rc, want 3" >&2
  exit 1
fi
if ! grep -q 'MEMORY LIMIT: application footprint' "$LIMIT_LOG" \
   || ! grep -q 'exit 3' "$LIMIT_LOG"; then
  echo "FAIL memlimit: application-footprint diagnostic missing" >&2
  exit 1
fi

# Phase-isolation cleanup must not reclassify a hard-limit stop. This bounded
# fixture has no eligible GPU chunk but initializes the explicit prefill policy,
# captures its phase boundary, and must preserve exit 3.
set +e
env SALT_GPU=1 SALT_GPU_RESIDENT=1 SALT_GPU_TRUNK=1 SALT_GPU_MOE=0 \
  SALT_GPU_TRUNK_PREFILL_ONLY=1 SALT_GPU_TRUNK_B=200 \
  SALT_PREFILL_CHUNK=0 SALT_CACHE_MODE=arena \
  "$REPO_ROOT/salt" "$FIX/model" \
  --trunk "$FIX/model/trunk.bin" --offsets "$FIX/model/trunk.offsets" \
  --gen 4 --mem-limit-gb 0.000001 >"$HYBRID_LIMIT_LOG" 2>&1
hybrid_limit_rc=$?
set -e
if [ "$hybrid_limit_rc" -ne 3 ]; then
  echo "FAIL memlimit: hybrid exit=$hybrid_limit_rc, want 3" >&2
  exit 1
fi
if ! grep -q 'MEMORY LIMIT: application footprint' "$HYBRID_LIMIT_LOG"; then
  echo "FAIL memlimit: hybrid application-footprint diagnostic missing" >&2
  exit 1
fi

# The model blueprint may describe its subordinate prefill-only policy while
# SALT_GPU=0; the master switch must leave that policy inert.
if ! env SALT_GPU=0 SALT_GPU_TRUNK=1 SALT_GPU_TRUNK_PREFILL_ONLY=1 \
  SALT_PREFILL_CHUNK=0 SALT_CACHE_MODE=arena \
  "$REPO_ROOT/salt" "$FIX/model" \
  --trunk "$FIX/model/trunk.bin" --offsets "$FIX/model/trunk.offsets" \
  --gen 4 --mem-limit-gb 0 >"$CONTROL_LOG" 2>&1; then
  echo "FAIL memlimit: unlimited control failed" >&2
  exit 1
fi

# An explicit GPU-prefill request with no eligible chunk must use the CPU path,
# not fail merely because the threshold was not reached.
if ! env SALT_GPU=1 SALT_GPU_RESIDENT=1 SALT_GPU_TRUNK=1 SALT_GPU_MOE=0 \
  SALT_GPU_TRUNK_PREFILL_ONLY=1 SALT_GPU_TRUNK_B=200 \
  SALT_PREFILL_CHUNK=0 SALT_CACHE_MODE=arena \
  "$REPO_ROOT/salt" "$FIX/model" \
  --trunk "$FIX/model/trunk.bin" --offsets "$FIX/model/trunk.offsets" \
  --gen 4 --mem-limit-gb 0 >"$PREFILL_CONTROL_LOG" 2>&1; then
  echo "FAIL memlimit: CPU fallback for ineligible GPU prefill failed" >&2
  exit 1
fi
if ! grep -q 'no eligible chunked prefill; using CPU' "$PREFILL_CONTROL_LOG"; then
  echo "FAIL memlimit: ineligible GPU-prefill fallback diagnostic missing" >&2
  exit 1
fi

echo "PASS memlimit"
