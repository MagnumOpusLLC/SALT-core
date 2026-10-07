CC      ?= cc
# macOS default -O2: the old -O0-on-Darwin rule existed for a clang
# codegen divergence in the DS-V4 fixture era, but the current gate
# (including e2e_text's byte-identical determinism check) passes at
# -O2, and the real Qwen3.5 model measured 9.14 s/token at -O0 vs
# 0.26 s/token at -O2 (35x). If the divergence ever reappears, the
# -O0 override is one flag away:
#   make CFLAGS="-std=c99 -O0 -Wall -Wextra -pthread"
ifeq ($(shell uname),Darwin)
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread -flto
GPU_SRC  = src/gpu_metal.m
GPU_OBJ  = src/gpu_metal.o
GPU_LIBS = -framework Metal -framework Foundation
VISION_LIBS = -framework Accelerate
GEMMA4_BUILD_MATH_ABI = salt-bitmath-v2;gemma4-vision-accelerate-v1
GEMMA4_BUILD_BLAS_ABI = apple-accelerate-lp64
OBJC  ?= clang
OBJCFLAGS ?= -std=c99 -O2 -Wall -Wextra -flto
METAL_LIB = salt-gpu.metallib
else
# Keep the portable engine in C99; _GNU_SOURCE below exposes Linux host APIs
# without changing the language dialect.
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread -flto
ifeq ($(CUDA),1)
NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_121
GPU_SRC  = src/gpu_cuda.o
GPU_OBJ  = src/gpu_cuda.o
GPU_LIBS = -L/usr/local/cuda/lib64 -lcudart -lstdc++
GEMMA4_BUILD_MATH_ABI = salt-bitmath-v2;gemma4-vision-adaptive-simd-v1;cuda-exact-v1;cuda-hmm-vision-exact-v1
else ifeq ($(ROCM),1)
ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/hipcc
ROCM_ARCH ?= gfx1011
GPU_SRC  = src/gpu_hip.o
GPU_OBJ  = src/gpu_hip.o
GPU_LIBS = -L$(ROCM_PATH)/lib -lamdhip64 -lstdc++
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
GPU_LIBS += -Wl,-rpath,$(ROCM_PATH)/lib
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
GEMMA4_BUILD_MATH_ABI = salt-bitmath-v2;gemma4-vision-adaptive-simd-v1;rocm-exact-v1
else
GPU_SRC  = src/gpu_stub.c
GPU_OBJ  =
GPU_LIBS =
GEMMA4_BUILD_MATH_ABI = salt-bitmath-v2;gemma4-vision-adaptive-simd-v1
endif
VISION_LIBS =
GEMMA4_BUILD_BLAS_ABI = none
endif
ARCHFLAGS ?=
CFLAGS += $(ARCHFLAGS)
OBJCFLAGS += $(ARCHFLAGS)
INC      = -Iinclude -Isrc -Imodels/gemma4-26b-a4b/src
# -D_GNU_SOURCE on non-Darwin: glibc gates MADV_DONTNEED and
# POSIX_FADV_* behind it (macOS defines them unconditionally). Must be
# a compile flag -- defining it inside cache.c after the header include
# is too late (salt/salt.h pulls in libc headers first).
ifneq ($(shell uname),Darwin)
CFLAGS  += -D_GNU_SOURCE
endif
# -ffp-contract=off: the bit-exact math layer (bitmath.c) must never
# let a compiler fuse mul+add into FMA -- the fusion decision differs
# per compiler/arch and would break cross-platform KV byte-identity.
CFLAGS  += -ffp-contract=off
HDR = include/salt/salt.h include/salt/kernels.h include/salt/moe.h \
      include/salt/simd.h include/salt/attn.h include/salt/head.h \
      include/salt/attn_batch.h include/salt/moe_group.h \
      include/salt/text_exec.h include/salt/area_scan.h include/salt/tensorops.h \
      include/salt/text_verify.h include/salt/text_token.h \
      include/salt/tokenizer.h include/salt/bitmath.h \
      include/salt/dpr.h include/salt/dpr_store.h include/salt/dpr_stats.h \
      include/salt/layer.h \
      include/salt/quant.h include/salt/model.h include/salt/state.h \
      include/salt/gpu.h \
      models/gemma4-26b-a4b/src/gemma4.h \
      models/gemma4-26b-a4b/src/gemma4_text.h \
      models/gemma4-26b-a4b/src/gemma4_vision.h \
      models/gemma4-26b-a4b/src/inference.h \
      include/salt/gpu_pool.h include/salt/gpu_resource.h src/json.h \
      src/compiler.h src/thread-lifecycle.h src/text_verify_internal.h \
      src/sha256.h
MODEL_REGISTRY_SRC = src/model.c models/registry.c \
      models/qwen36/model.c models/gemma4-26b-a4b/src/model.c models/maple-preview/model.c
QWEN36_MODEL_SRC = models/qwen36/moe.c models/qwen36/attn.c \
      models/qwen36/deltachunk.c models/qwen36/layer.c
QWEN36_MODEL_HDR = models/qwen36/attn.h models/qwen36/deltachunk.h
SRC = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
      src/kernels.c src/simd.c src/attn.c \
      src/attn_batch.c src/moe_group.c src/text_exec.c src/text_token.c \
      src/area_scan.c src/tensorops.c src/text_verify.c src/text_verify_cpu.c \
      src/head.c src/tokenizer.c src/bitmath.c src/sampling.c \
      src/quant.c src/int2.c $(MODEL_REGISTRY_SRC) src/state.c $(QWEN36_MODEL_SRC) \
      models/gemma4-26b-a4b/src/gemma4.c src/gpu_pool.c \
      src/gpu_resource.c src/gpu_layer.c src/dpr.c src/dpr_store.c \
      src/dpr_stats.c src/sha256.c $(GPU_SRC)

all: salt gemma4-server

include models/maple-preview/build.mk

salt: models/qwen36/main.c $(SRC) $(HDR) $(QWEN36_MODEL_HDR) $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) -DSALT_GIT=\"$(shell git rev-parse --short HEAD 2>/dev/null)\" -o $@ models/qwen36/main.c $(filter-out %.m %.mm $(GPU_OBJ),$(SRC)) $(GPU_OBJ) -lm $(GPU_LIBS)

GEMMA4_COMPAT_PYTHON ?= python3
GEMMA4_KV_COMPAT_TOOL = server/gemma4_compat.py
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
GEMMA4_KV_COMPAT_TOOL := server/model/gemma4_compat_projection.py
SRC += src/gpu_residency.c
HDR += include/salt/gpu_residency.h
HDR += include/salt/sampling.h
.PHONY: test-gpu-residency test-gemma4-kv-compat-projection \
	test-gemma4-endpoint-startup-kv-contract test-gemma4-rocm-source-contract \
	test-gemma4-rocm-residency-source-contract \
	test-gemma4-rocm-permanent-trunk-source-contract \
	test-gemma4-rocm-expert-residency-source-contract \
	test-gemma4-rocm-backend-published-kv-source-contract
test: test-public-smoke
test-gpu-residency:
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -o /tmp/salt-gpu-residency-test \
		tests/test_gpu_residency.c src/gpu_residency.c src/gpu_resource.c
	/tmp/salt-gpu-residency-test
	env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" tools/test/gpu-residency-source-test.py
	rm -f /tmp/salt-gpu-residency-test
test-gemma4-kv-compat-projection:
	env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" \
		tools/test/gemma4-kv-compat-projection-test.py
test-gemma4-endpoint-startup-kv-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-endpoint-startup-kv-test.py
test-gemma4-rocm-source-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-kv-ring-source-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-q4-tile-source-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-attention-position-source-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-selected-resource-fence-source-test.py
test-gemma4-rocm-residency-source-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-residency-source-test.py
test-gemma4-rocm-permanent-trunk-source-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-permanent-trunk-source-test.py
test-gemma4-rocm-expert-residency-source-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-expert-residency-source-test.py
test-gemma4-rocm-backend-published-kv-source-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-rocm-backend-published-kv-source-test.py
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
GEMMA4_KV_COMPAT_MANIFEST = models/gemma4-26b-a4b/kv-cache-compat.json
GEMMA4_KV_COMPAT_DEPS := $(shell env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" \
		$(GEMMA4_KV_COMPAT_TOOL) --manifest "$(GEMMA4_KV_COMPAT_MANIFEST)" --files)
GEMMA4_KV_COMPAT_SHA256 = $(shell env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" \
		$(GEMMA4_KV_COMPAT_TOOL) --manifest "$(GEMMA4_KV_COMPAT_MANIFEST)" --hex)
GEMMA4_KV_LEGACY_V5_SHA256 = $(shell env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" \
		$(GEMMA4_KV_COMPAT_TOOL) --manifest "$(GEMMA4_KV_COMPAT_MANIFEST)" --legacy-v5-hex)
GEMMA4_KV_LEGACY_V4_SHA256 = $(shell env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" \
		$(GEMMA4_KV_COMPAT_TOOL) --manifest "$(GEMMA4_KV_COMPAT_MANIFEST)" --legacy-v4-hex)
GEMMA4_KV_COMPAT_CFLAG = \
	-DSALT_GEMMA4_KV_COMPAT_SHA256=\"$(GEMMA4_KV_COMPAT_SHA256)\" \
	-DSALT_GEMMA4_KV_LEGACY_V5_SHA256=\"$(GEMMA4_KV_LEGACY_V5_SHA256)\" \
	-DSALT_GEMMA4_KV_LEGACY_V4_SHA256=\"$(GEMMA4_KV_LEGACY_V4_SHA256)\"
GEMMA4_BUILD_TOOL = server/gemma4_build.py
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
GEMMA4_BUILD_TOOL := server/model/gemma4_build.py
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
override GEMMA4_CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror -pthread \
	-flto -ffp-contract=off $(ARCHFLAGS)
ifneq ($(shell uname),Darwin)
override GEMMA4_CFLAGS += -D_GNU_SOURCE
endif

ifeq ($(shell uname),Darwin)
GEMMA4_GPU_SRC = src/gpu_metal.o
GEMMA4_GPU_LIBS = $(GPU_LIBS)
else ifeq ($(CUDA),1)
GEMMA4_GPU_SRC = src/gpu_cuda.o
GEMMA4_GPU_LIBS = $(GPU_LIBS)
else ifeq ($(ROCM),1)
GEMMA4_GPU_SRC = src/gpu_hip.o
GEMMA4_GPU_LIBS = $(GPU_LIBS)
else
GEMMA4_GPU_SRC = src/gpu_stub.c
GEMMA4_GPU_LIBS =
endif

GEMMA4_QA_SRC = models/gemma4-26b-a4b/src/gemma4_text.c \
		models/gemma4-26b-a4b/src/gemma4.c src/tokenizer.c \
		src/cache.c src/kernels.c src/simd.c src/nvfp4.c $(GEMMA4_GPU_SRC) src/sha256.c \
		$(MODEL_REGISTRY_SRC) src/state.c src/gpu_resource.c src/mem.c src/attn.c src/attn_batch.c src/moe_group.c src/text_exec.c src/text_token.c \
		src/area_scan.c src/tensorops.c src/text_verify.c src/text_verify_cpu.c \
		src/bitmath.c src/dpr.c
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
GEMMA4_QA_SRC += src/gpu_residency.c
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
gemma4-qa: tools/gemma4-qa.c models/gemma4-26b-a4b/src/gemma4_kv_file.h $(GEMMA4_QA_SRC) $(HDR) $(METAL_LIB) \
		$(GEMMA4_KV_COMPAT_TOOL) $(GEMMA4_KV_COMPAT_MANIFEST) \
		$(GEMMA4_KV_COMPAT_DEPS)
	$(CC) $(GEMMA4_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) -o $@ \
		tools/gemma4-qa.c $(GEMMA4_QA_SRC) -lm $(GEMMA4_GPU_LIBS)

GEMMA4_VISION_QA_SRC = models/gemma4-26b-a4b/src/gemma4_vision.c \
		models/gemma4-26b-a4b/src/gemma4.c src/kernels.c \
		src/simd.c $(GEMMA4_GPU_SRC) src/sha256.c src/bitmath.c src/attn.c \
		src/tensorops.c
gemma4-vision-qa: tools/gemma4-vision-qa.c $(GEMMA4_VISION_QA_SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/gemma4-vision-qa.c \
		$(GEMMA4_VISION_QA_SRC) -lm $(VISION_LIBS) $(GEMMA4_GPU_LIBS)

GEMMA4_MULTIMODAL_QA_SRC = models/gemma4-26b-a4b/src/gemma4_text.c \
		models/gemma4-26b-a4b/src/gemma4_vision.c \
		models/gemma4-26b-a4b/src/gemma4.c src/tokenizer.c src/cache.c src/kernels.c src/simd.c \
		src/nvfp4.c $(GEMMA4_GPU_SRC) src/sha256.c $(MODEL_REGISTRY_SRC) src/state.c src/gpu_resource.c \
		src/attn.c src/attn_batch.c \
		src/moe_group.c src/text_exec.c src/text_token.c src/area_scan.c src/tensorops.c src/text_verify.c src/text_verify_cpu.c \
		src/bitmath.c src/mem.c src/dpr.c src/dpr_store.c \
		src/dpr_stats.c src/sampling.c
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
GEMMA4_MULTIMODAL_QA_SRC += src/gpu_residency.c
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
gemma4-multimodal-qa: tools/gemma4-multimodal-qa.c \
		models/gemma4-26b-a4b/src/gemma4_kv_file.h $(GEMMA4_MULTIMODAL_QA_SRC) $(HDR) \
		$(GEMMA4_KV_COMPAT_TOOL) $(GEMMA4_KV_COMPAT_MANIFEST) \
		$(GEMMA4_KV_COMPAT_DEPS)
	$(CC) $(GEMMA4_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) -o $@ \
		tools/gemma4-multimodal-qa.c $(GEMMA4_MULTIMODAL_QA_SRC) \
		-lm $(VISION_LIBS) $(GEMMA4_GPU_LIBS)

GEMMA4_SERVER_SRC = models/gemma4-26b-a4b/server.c \
		models/gemma4-26b-a4b/src/inference.c
gemma4-server: $(GEMMA4_SERVER_SRC) \
		models/gemma4-26b-a4b/src/gemma4_kv_file.h \
		$(GEMMA4_MULTIMODAL_QA_SRC) $(HDR) \
		$(GEMMA4_KV_COMPAT_TOOL) $(GEMMA4_KV_COMPAT_MANIFEST) \
		$(GEMMA4_KV_COMPAT_DEPS)
	$(CC) $(GEMMA4_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) -o $@ \
		$(GEMMA4_SERVER_SRC) $(GEMMA4_MULTIMODAL_QA_SRC) \
		-lm $(VISION_LIBS) $(GEMMA4_GPU_LIBS)

gemma4-qa.build.json: gemma4-qa $(GEMMA4_BUILD_TOOL) $(GEMMA4_KV_COMPAT_DEPS)
	env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" $(GEMMA4_BUILD_TOOL) create \
		--binary gemma4-qa --receipt $@ --compiler "$(CC)" \
		--flags "$(GEMMA4_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) -lm $(GEMMA4_GPU_LIBS)" \
		--source-compatibility "$(GEMMA4_KV_COMPAT_SHA256)" \
		--math-abi "$(GEMMA4_BUILD_MATH_ABI)" --blas-abi "$(GEMMA4_BUILD_BLAS_ABI)"

gemma4-multimodal-qa.build.json: gemma4-multimodal-qa $(GEMMA4_BUILD_TOOL) \
		$(GEMMA4_KV_COMPAT_DEPS)
	env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" $(GEMMA4_BUILD_TOOL) create \
		--binary gemma4-multimodal-qa --receipt $@ --compiler "$(CC)" \
		--flags "$(GEMMA4_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) -lm $(VISION_LIBS) $(GEMMA4_GPU_LIBS)" \
		--source-compatibility "$(GEMMA4_KV_COMPAT_SHA256)" \
		--math-abi "$(GEMMA4_BUILD_MATH_ABI)" --blas-abi "$(GEMMA4_BUILD_BLAS_ABI)"

gemma4-server.build.json: gemma4-server $(GEMMA4_BUILD_TOOL) \
		$(GEMMA4_KV_COMPAT_DEPS)
	env -u PYTHONPATH "$(GEMMA4_COMPAT_PYTHON)" $(GEMMA4_BUILD_TOOL) create \
		--binary gemma4-server --receipt $@ --compiler "$(CC)" \
		--flags "$(GEMMA4_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) -lm $(VISION_LIBS) $(GEMMA4_GPU_LIBS)" \
		--source-compatibility "$(GEMMA4_KV_COMPAT_SHA256)" \
		--math-abi "$(GEMMA4_BUILD_MATH_ABI)" --blas-abi "$(GEMMA4_BUILD_BLAS_ABI)"

gemma4-build-receipts: gemma4-qa.build.json \
		gemma4-multimodal-qa.build.json gemma4-server.build.json

# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
gemma4-server-state-fixture: src/text_verify.c src/area_scan.c src/tensorops.c include/salt/text_verify.h
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
gemma4-server-state-fixture: $(GEMMA4_SERVER_SRC) \
		models/gemma4-26b-a4b/src/gemma4_kv_file.h \
		tools/test/gemma4-server-state-stub.c \
		models/gemma4-26b-a4b/src/gemma4_text.h \
		models/gemma4-26b-a4b/src/gemma4_vision.h include/salt/model.h include/salt/state.h \
		include/salt/dpr.h include/salt/dpr_store.h include/salt/dpr_stats.h \
		$(MODEL_REGISTRY_SRC) src/state.c src/dpr.c src/dpr_store.c src/dpr_stats.c \
		src/mem.c src/sha256.c src/sampling.c src/bitmath.c include/salt/sampling.h
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
ifeq (0,1)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
	$(CC) $(GEMMA4_CFLAGS) \
		-DSALT_GEMMA4_KV_COMPAT_SHA256=\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\" \
		$(INC) -o $@ $(GEMMA4_SERVER_SRC) \
		tools/test/gemma4-server-state-stub.c $(MODEL_REGISTRY_SRC) src/state.c \
		src/dpr.c src/dpr_store.c src/dpr_stats.c src/mem.c src/sha256.c \
		src/sampling.c src/bitmath.c -lm
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
else
	$(CC) $(GEMMA4_CFLAGS) \
		-DSALT_GEMMA4_KV_COMPAT_SHA256=\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\" \
		$(INC) -o $@ $(GEMMA4_SERVER_SRC) \
		tools/test/gemma4-server-state-stub.c $(MODEL_REGISTRY_SRC) src/state.c \
		src/dpr.c src/dpr_store.c src/dpr_stats.c src/mem.c src/sha256.c \
		src/sampling.c src/bitmath.c src/text_verify.c src/area_scan.c src/tensorops.c -lm
endif
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1

gemma4-dpr-store-fixture: tools/test/gemma4-dpr-store-fixture.c \
		include/salt/dpr.h include/salt/dpr_store.h \
		src/dpr.c src/dpr_store.c src/sha256.c src/sha256.h
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o $@ \
		tools/test/gemma4-dpr-store-fixture.c \
		src/dpr.c src/dpr_store.c src/sha256.c

gemma4-dpr-edge: tools/gemma4-dpr-edge.c include/salt/dpr.h \
		include/salt/dpr_store.h src/dpr.c src/dpr_store.c src/sha256.c src/sha256.h
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o $@ tools/gemma4-dpr-edge.c \
		src/dpr.c src/dpr_store.c src/sha256.c

test-gemma4-vision-image: tools/test/gemma4-vision-image-test.c \
		$(GEMMA4_VISION_QA_SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o /tmp/salt-gemma4-vision-image-test \
		tools/test/gemma4-vision-image-test.c $(GEMMA4_VISION_QA_SRC) \
		-lm $(VISION_LIBS) $(GEMMA4_GPU_LIBS)
	/tmp/salt-gemma4-vision-image-test

GEMMA4_E2E_PYTHON ?= python3
GEMMA4_E2E_JSON ?= /tmp/salt-gemma4-e2e-report.json
GEMMA4_E2E_REPORT ?= /tmp/salt-gemma4-e2e-report.md
GEMMA4_E2E_ARTIFACTS ?= /tmp/salt-gemma4-e2e-artifacts
GEMMA4_SERVER_E2E_ARGS ?=
GEMMA4_PHASE6_JSON ?= models/gemma4-26b-a4b/qualification/persistent-state-v2.json
GEMMA4_PHASE6_IMAGE ?= models/gemma4-26b-a4b/qualification/red-mug-384.ppm

test-gemma4-e2e-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-e2e-test.py

.PHONY: test-gemma4-server-live-e2e
test-gemma4-server-live-e2e:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-server-live-e2e.py $(GEMMA4_SERVER_E2E_ARGS)

test-gemma4-finegrain-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-finegrain-matrix.py check

test-gemma4-memory-accounting:
	$(CC) -std=c99 -O2 -Wall -Wextra -Werror -pedantic -Itools \
		tools/test/gemma4-memory-test.c -o /tmp/salt-gemma4-memory-test
	/tmp/salt-gemma4-memory-test

test-hot-expert-ledger:
	$(CC) -std=c99 -O2 -Wall -Wextra -Werror -pedantic $(INC) \
		tools/state-exam/tests/hot-expert-ledger-test.c \
		src/dpr.c src/sha256.c -o /tmp/salt-hot-expert-ledger-test
	/tmp/salt-hot-expert-ledger-test

test-gemma4-server-contract: gemma4-server-state-fixture \
		gemma4-dpr-store-fixture
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-server-launch-policy-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-server-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-http-framing-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-live-stream-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-persistent-session-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-dpr-runtime-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-dpr-native-engine-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-dpr-mentor-launch-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-scheduler-test.py

# Phase-1 gate: these contracts must become GREEN together.
test-gemma4-state-phase1:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-state-machine-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-history-reconciliation-test.py

test-gemma4-state-phase2: gemma4-server-state-fixture
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-replay-test.py

test-gemma4-state-phase3: gemma4-server-state-fixture
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-state-deferred-test.py

test-gemma4-mindset-kv:
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o /tmp/salt-gemma4-mindset-kv-test \
		tools/test/gemma4-mindset-kv-test.c \
		models/gemma4-26b-a4b/src/gemma4.c src/tokenizer.c src/cache.c src/kernels.c src/simd.c \
		src/nvfp4.c src/gpu_stub.c src/sha256.c $(MODEL_REGISTRY_SRC) \
		src/gpu_resource.c src/mem.c src/attn.c src/attn_batch.c \
		src/moe_group.c src/text_exec.c src/bitmath.c src/dpr.c -lm
	/tmp/salt-gemma4-mindset-kv-test

TEXT_VERIFY_CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror -pedantic \
	-ffp-contract=off -D_POSIX_C_SOURCE=200809L

test-text-verify:
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -c src/area_scan.c \
		-o /tmp/salt-area-scan.o
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -c src/tensorops.c \
		-o /tmp/salt-tensorops.o
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -c src/text_verify.c \
		-o /tmp/salt-text-verify.o
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -c src/text_verify_cpu.c \
		-o /tmp/salt-text-verify-cpu.o
	@if nm -u /tmp/salt-area-scan.o /tmp/salt-tensorops.o /tmp/salt-text-verify.o \
		/tmp/salt-text-verify-cpu.o | \
		grep -E ' (malloc|calloc|realloc|free|aligned_alloc|posix_memalign)$$' \
		>/dev/null; then \
		echo "text verify allocation-symbol scan: FAIL"; exit 1; \
	else \
		echo "text verify allocation-symbol scan: PASS"; \
	fi
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
ifeq (0,1)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -o /tmp/salt-text-verify-test \
		tests/test_text_verify.c /tmp/salt-area-scan.o /tmp/salt-tensorops.o /tmp/salt-text-verify.o \
		/tmp/salt-text-verify-cpu.o \
		src/attn.c src/kernels.c src/simd.c src/gpu_stub.c src/bitmath.c -lm
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
else
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -o /tmp/salt-text-verify-test \
		tests/test_text_verify.c /tmp/salt-area-scan.o /tmp/salt-tensorops.o /tmp/salt-text-verify.o \
		/tmp/salt-text-verify-cpu.o \
		src/attn.c src/kernels.c src/simd.c src/gpu_stub.c src/bitmath.c \
		src/sampling.c src/dpr.c src/dpr_store.c src/dpr_stats.c src/sha256.c -lm
endif
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
	/tmp/salt-text-verify-test

test-text-token:
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -c src/text_token.c \
		-o /tmp/salt-text-token.o
	@if nm -u /tmp/salt-text-token.o | \
		grep -E ' (malloc|calloc|realloc|free|aligned_alloc|posix_memalign)$$' \
		>/dev/null; then \
		echo "text token allocation-symbol scan: FAIL"; exit 1; \
	else \
		echo "text token allocation-symbol scan: PASS"; \
	fi
	$(CC) $(TEXT_VERIFY_CFLAGS) $(INC) -o /tmp/salt-text-token-test \
		tests/test_text_token.c /tmp/salt-text-token.o src/tensorops.c -lm
	/tmp/salt-text-token-test

test-wfq-pool: $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) -o /tmp/salt-wfq-pool-test \
		tests/test_wfq_pool.c $(filter-out %.m %.mm $(GPU_OBJ),$(SRC)) \
		$(GPU_OBJ) -lm $(GPU_LIBS)
	/tmp/salt-wfq-pool-test

test-q4-nparallel:
	$(CC) $(CFLAGS) $(INC) -o /tmp/salt-q4-nparallel-test \
		tests/verify_batch2.c src/kernels.c src/simd.c src/gpu_stub.c -lm
	/tmp/salt-q4-nparallel-test

test-gemma4-phase4: test-text-verify test-text-token test-wfq-pool test-q4-nparallel
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o /tmp/salt-text-exec-control-test \
		tests/test_text_exec.c src/text_exec.c
	/tmp/salt-text-exec-control-test
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o /tmp/salt-state-control-test \
		tools/test/state-control-test.c $(MODEL_REGISTRY_SRC) src/state.c -lm
	/tmp/salt-state-control-test
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o /tmp/salt-gemma4-math-contract \
		tests/test_gemma4_text.c models/gemma4-26b-a4b/src/gemma4.c \
		$(MODEL_REGISTRY_SRC) src/state.c \
		src/bitmath.c -lm
	/tmp/salt-gemma4-math-contract
	$(CC) $(GEMMA4_CFLAGS) $(INC) -o /tmp/salt-gemma4-bit-identity-test \
		tools/test/gemma4-bit-identity-test.c src/kernels.c src/simd.c \
		src/gpu_stub.c -lm
	/tmp/salt-gemma4-bit-identity-test
	$(CC) $(GEMMA4_CFLAGS) -fno-lto $(INC) \
		-Dpthread_create=test_pthread_create \
		-Dpthread_join=test_pthread_join -c src/kernels.c \
		-o /tmp/salt-q4-pthread-kernels.o
	$(CC) $(GEMMA4_CFLAGS) -fno-lto $(INC) -o /tmp/salt-q4-pthread-failure-test \
		tools/test/q4-pthread-failure-test.c \
		/tmp/salt-q4-pthread-kernels.o src/simd.c src/gpu_stub.c -lm
	SALT_ATTN_THREADS=4 /tmp/salt-q4-pthread-failure-test

GEMMA4_STRICT_CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror \
	-Werror=misleading-indentation -pedantic -pthread -ffp-contract=off \
	-D_POSIX_C_SOURCE=200809L $(ARCHFLAGS)
GEMMA4_PLATFORM_CFLAGS := $(GEMMA4_STRICT_CFLAGS)
ifeq ($(shell uname),Darwin)
GEMMA4_PLATFORM_CFLAGS += -D_DARWIN_C_SOURCE
else
GEMMA4_PLATFORM_CFLAGS += -D_GNU_SOURCE
endif

test-gemma4-phase5-strict:
	$(CC) $(GEMMA4_STRICT_CFLAGS) $(GEMMA4_KV_COMPAT_CFLAG) $(INC) \
		-fsyntax-only $(GEMMA4_SERVER_SRC) \
		$(filter-out src/cache.c %.o,$(GEMMA4_MULTIMODAL_QA_SRC))
	$(CC) $(GEMMA4_PLATFORM_CFLAGS) $(INC) -fsyntax-only src/cache.c

test-gemma4-qa-source-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-qa-topology-source-test.py

test-gemma4-phase5: gemma4-build-receipts
	$(MAKE) test-gemma4-phase5-strict
	env -u PYTHONPATH CC="$(CC)" "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-build-identity-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-compat-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-cache-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" \
		tools/test/gemma4-prefill-retention-source-test.py
	$(MAKE) test-gemma4-qa-source-contract
	$(MAKE) test-gemma4-mindset-kv
	$(MAKE) test-gemma4-phase4

test-gemma4-phase5-sanitize:
	@if [ "`uname`" = Darwin ]; then \
		echo "gemma4 Linux sanitizer gate: SKIP on Darwin"; \
	else \
		set -e; \
		SAN="-std=c99 -O1 -g -Wall -Wextra -Werror -pthread -ffp-contract=off -fno-omit-frame-pointer -fsanitize=address,undefined"; \
		ASAN_LIB=`$(CC) -print-file-name=libasan.so`; \
		ASAN_DIR=$${ASAN_LIB%/*}; \
		ASAN_REAL=`readlink -f "$$ASAN_LIB"`; \
		UBSAN_LIB=`$(CC) -print-file-name=libubsan.so`; \
		UBSAN_REAL=`readlink -f "$$UBSAN_LIB"`; \
		SAN_PRELOAD="$$ASAN_REAL:$$UBSAN_REAL"; \
		$(CC) $$SAN $(INC) -o /tmp/salt-state-control-san \
			tools/test/state-control-test.c $(MODEL_REGISTRY_SRC) src/state.c -lm; \
		LD_PRELOAD="$$SAN_PRELOAD" LD_LIBRARY_PATH="$$ASAN_DIR:$${LD_LIBRARY_PATH:-}" \
			/tmp/salt-state-control-san; \
		$(CC) $$SAN $(INC) -o /tmp/salt-gemma4-math-san \
			tests/test_gemma4_text.c models/gemma4-26b-a4b/src/gemma4.c \
			$(MODEL_REGISTRY_SRC) src/state.c src/bitmath.c -lm; \
		LD_PRELOAD="$$SAN_PRELOAD" LD_LIBRARY_PATH="$$ASAN_DIR:$${LD_LIBRARY_PATH:-}" \
			/tmp/salt-gemma4-math-san; \
		$(CC) $$SAN $(INC) -Dpthread_create=test_pthread_create \
			-Dpthread_join=test_pthread_join -c src/kernels.c \
			-o /tmp/salt-q4-pthread-kernels-san.o; \
		$(CC) $$SAN $(INC) -o /tmp/salt-q4-pthread-failure-san \
			tools/test/q4-pthread-failure-test.c \
			/tmp/salt-q4-pthread-kernels-san.o src/simd.c src/gpu_stub.c -lm; \
		LD_PRELOAD="$$SAN_PRELOAD" LD_LIBRARY_PATH="$$ASAN_DIR:$${LD_LIBRARY_PATH:-}" \
			SALT_ATTN_THREADS=4 \
			/tmp/salt-q4-pthread-failure-san; \
	fi

test-gemma4-phase6: gemma4-build-receipts
	@if [ -z "$(GEMMA4_SOURCE_DIR)" ] || [ -z "$(GEMMA4_POOL)" ] || \
		[ -z "$(GEMMA4_AUTH_RECEIPT)" ]; then \
		echo "gemma4 phase6: FAIL (set GEMMA4_SOURCE_DIR, GEMMA4_POOL, and GEMMA4_AUTH_RECEIPT)"; \
		exit 2; \
	fi
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/gemma4-phase6.py \
		--source-dir "$(GEMMA4_SOURCE_DIR)" --pool "$(GEMMA4_POOL)" \
		--receipt "$${GEMMA4_RECEIPT:-$(GEMMA4_POOL).import.json}" \
		--auth-receipt "$(GEMMA4_AUTH_RECEIPT)" \
		--image "$(GEMMA4_PHASE6_IMAGE)" --report "$(GEMMA4_PHASE6_JSON)"

# Aggregate state ladder. The historical target name is retained for compatibility.
test-gemma4-state-red:
	@rc=0; \
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-state-machine-test.py || rc=1; \
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-history-reconciliation-test.py || rc=1; \
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-replay-test.py || rc=1; \
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-state-deferred-test.py || rc=1; \
	exit $$rc

test-gemma4-http-framing-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-http-framing-test.py

test-gemma4-kv-cache-contract:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-compat-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-cache-test.py

# Expensive authenticated final gate: four fixed cases x exactly three serial
# process-isolated cold/full-prefix-poison/verified-KV-warm pairs. It remains
# separate from the regular suite.
test-gemma4-e2e: gemma4-qa gemma4-multimodal-qa \
		test-gemma4-e2e-contract test-gemma4-server-contract \
		test-gemma4-kv-cache-contract
	@if [ -z "$(GEMMA4_SOURCE_DIR)" ] || [ -z "$(GEMMA4_POOL)" ] || \
		[ -z "$(GEMMA4_AUTH_RECEIPT)" ]; then \
		echo "gemma4 e2e: FAIL (set GEMMA4_SOURCE_DIR, GEMMA4_POOL, and GEMMA4_AUTH_RECEIPT)"; \
		exit 2; \
	fi
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/gemma4-e2e.py \
		--python "$(GEMMA4_E2E_PYTHON)" \
		--source-dir "$(GEMMA4_SOURCE_DIR)" --pool "$(GEMMA4_POOL)" \
		--auth-receipt "$(GEMMA4_AUTH_RECEIPT)" \
		--json-report "$(GEMMA4_E2E_JSON)" \
		--report "$(GEMMA4_E2E_REPORT)" \
		--artifacts "$(GEMMA4_E2E_ARTIFACTS)"
src/gpu_metal.o: src/gpu_metal.m src/gpu_metal_ops_source.h \
                 include/salt/gpu.h include/salt/gpu_resource.h
	$(OBJC) $(OBJCFLAGS) $(INC) -c src/gpu_metal.m -o src/gpu_metal.o

salt-gpu.metallib: tools/build-metal-library.py src/gpu_metal.m \
                   src/gpu_metal_ops_source.h
	python3 tools/build-metal-library.py --output $@

src/gpu_cuda.o: src/gpu_cuda.cu include/salt/gpu.h include/salt/gpu_resource.h
	$(NVCC) -O2 -std=c++14 -arch=$(CUDA_ARCH) --fmad=false \
		-Xcompiler=-ffp-contract=off $(INC) -c src/gpu_cuda.cu -o src/gpu_cuda.o

src/gpu_hip.o: src/gpu_hip.cpp include/salt/gpu.h include/salt/gpu_resource.h
	$(HIPCC) -O2 -std=c++14 --offload-arch=$(ROCM_ARCH) -ffp-contract=off -fPIE \
		$(INC) -c src/gpu_hip.cpp -o src/gpu_hip.o

CUDA_TEXT_SUPPORT_OBJS := /tmp/salt-gpu-cuda-area-scan.o \
	/tmp/salt-gpu-cuda-tensorops.o /tmp/salt-gpu-cuda-text-verify.o

/tmp/salt-gpu-cuda-area-scan.o: src/area_scan.c
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c $< -o $@

/tmp/salt-gpu-cuda-tensorops.o: src/tensorops.c
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c $< -o $@

/tmp/salt-gpu-cuda-text-verify.o: src/text_verify.c
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c $< -o $@

ROCM_TEXT_SUPPORT_OBJS := /tmp/salt-gpu-rocm-area-scan.o \
	/tmp/salt-gpu-rocm-tensorops.o /tmp/salt-gpu-rocm-text-verify.o
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
ROCM_TEXT_SUPPORT_OBJS += /tmp/salt-gpu-rocm-residency.o
/tmp/salt-gpu-rocm-residency.o: src/gpu_residency.c include/salt/gpu_residency.h
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) -c $< -o $@
src/gpu_hip.o: include/salt/gpu_residency.h
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1

/tmp/salt-gpu-rocm-area-scan.o: src/area_scan.c
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) -c $< -o $@

/tmp/salt-gpu-rocm-tensorops.o: src/tensorops.c
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) -c $< -o $@

/tmp/salt-gpu-rocm-text-verify.o: src/text_verify.c
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) -c $< -o $@

test-gpu-cuda-exact: src/gpu_cuda.o $(CUDA_TEXT_SUPPORT_OBJS)
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c tools/test/gpu-cuda-exact-test.c \
		-o /tmp/salt-gpu-cuda-exact-test.o
	$(CC) $(GEMMA4_CFLAGS) -Wno-error=maybe-uninitialized $(INC) \
		-c src/kernels.c -o /tmp/salt-gpu-cuda-kernels.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/simd.c -o /tmp/salt-gpu-cuda-simd.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/gpu_resource.c \
		-o /tmp/salt-gpu-cuda-resource.o

	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/bitmath.c \
		-o /tmp/salt-gpu-cuda-bitmath.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c models/gemma4-26b-a4b/src/gemma4.c \
		-o /tmp/salt-gpu-cuda-gemma4.o
	$(NVCC) -arch=$(CUDA_ARCH) /tmp/salt-gpu-cuda-exact-test.o \
		/tmp/salt-gpu-cuda-kernels.o /tmp/salt-gpu-cuda-simd.o \
		/tmp/salt-gpu-cuda-resource.o $(CUDA_TEXT_SUPPORT_OBJS) \
		/tmp/salt-gpu-cuda-bitmath.o \
		/tmp/salt-gpu-cuda-gemma4.o \
		src/gpu_cuda.o -lpthread -lm \
		-o /tmp/salt-gpu-cuda-exact-test
	/tmp/salt-gpu-cuda-exact-test

test-gpu-rocm-exact: src/gpu_hip.o $(ROCM_TEXT_SUPPORT_OBJS)
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) \
		-c tools/test/gpu-cuda-exact-test.c \
		-o /tmp/salt-gpu-rocm-exact-test.o
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) \
		-Wno-error=maybe-uninitialized $(INC) \
		-c src/kernels.c -o /tmp/salt-gpu-rocm-kernels.o
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) \
		-c src/simd.c -o /tmp/salt-gpu-rocm-simd.o
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) -c src/gpu_resource.c \
		-o /tmp/salt-gpu-rocm-resource.o
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) -c src/bitmath.c \
		-o /tmp/salt-gpu-rocm-bitmath.o
	$(CC) $(filter-out -flto,$(GEMMA4_CFLAGS)) $(INC) \
		-c models/gemma4-26b-a4b/src/gemma4.c \
		-o /tmp/salt-gpu-rocm-gemma4.o
	$(HIPCC) --offload-arch=$(ROCM_ARCH) /tmp/salt-gpu-rocm-exact-test.o \
		/tmp/salt-gpu-rocm-kernels.o /tmp/salt-gpu-rocm-simd.o \
		/tmp/salt-gpu-rocm-resource.o $(ROCM_TEXT_SUPPORT_OBJS) \
		/tmp/salt-gpu-rocm-bitmath.o /tmp/salt-gpu-rocm-gemma4.o \
		src/gpu_hip.o -lpthread -lm -o /tmp/salt-gpu-rocm-exact-test
	@if [ "$(ROCM_TEST_RUN)" = "0" ]; then \
		echo "ROCm exact fixture: COMPILED (runtime skipped)"; \
	else \
		SALT_TEST_ROCM_COMPONENT_POOL=1 \
		SALT_TEST_ROCM_COMPONENT_ROLLBACK=1 \
		/tmp/salt-gpu-rocm-exact-test; \
	fi

# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
.PHONY: test-gpu-rocm-residency
test-gpu-rocm-residency: test-gpu-rocm-exact
	$(HIPCC) -O2 -std=c++14 --offload-arch=$(ROCM_ARCH) -ffp-contract=off \
		$(INC) -c tools/test/gpu-hip-residency-test.cpp \
		-o /tmp/salt-gpu-rocm-residency-test.o
	$(HIPCC) --offload-arch=$(ROCM_ARCH) \
		/tmp/salt-gpu-rocm-residency-test.o \
		/tmp/salt-gpu-rocm-kernels.o /tmp/salt-gpu-rocm-simd.o \
		/tmp/salt-gpu-rocm-resource.o $(ROCM_TEXT_SUPPORT_OBJS) \
		/tmp/salt-gpu-rocm-bitmath.o /tmp/salt-gpu-rocm-gemma4.o \
		src/gpu_hip.o -lpthread -lm -o /tmp/salt-gpu-rocm-residency-test
	SALT_NVFP4_TELEMETRY=0 /tmp/salt-gpu-rocm-residency-test
	rm -f /tmp/salt-gpu-rocm-residency-test \
		/tmp/salt-gpu-rocm-residency-test.o \
		/tmp/salt-gpu-rocm-exact-test /tmp/salt-gpu-rocm-exact-test.o \
		/tmp/salt-gpu-rocm-kernels.o /tmp/salt-gpu-rocm-simd.o \
		/tmp/salt-gpu-rocm-resource.o /tmp/salt-gpu-rocm-area-scan.o \
		/tmp/salt-gpu-rocm-tensorops.o /tmp/salt-gpu-rocm-text-verify.o \
		/tmp/salt-gpu-rocm-residency.o /tmp/salt-gpu-rocm-bitmath.o \
		/tmp/salt-gpu-rocm-gemma4.o
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
test-gpu-metal-exact: src/gpu_metal.o $(METAL_LIB)
	$(CC) $(GEMMA4_CFLAGS) $(INC) \
		tools/test/gpu-cuda-exact-test.c src/kernels.c src/simd.c \
		src/gpu_resource.c src/bitmath.c \
		models/gemma4-26b-a4b/src/gemma4.c src/gpu_metal.o \
		-lpthread -lm -framework Metal -framework Foundation \
		-o /tmp/salt-gpu-metal-exact-test
	SALT_METAL_LIBRARY=$(abspath $(METAL_LIB)) \
		SALT_METAL_Q4_SIMDGROUPS=0 SALT_METAL_Q4_MAX_DESCRIPTORS=128 \
		SALT_METAL_Q4_WEIGHT_STATIONARY_MIN_B=16 \
		SALT_TEST_PRODUCTION_SHAPES_ONLY=1 /tmp/salt-gpu-metal-exact-test

test-gpu-cuda-heterogeneous: src/gpu_cuda.o $(CUDA_TEXT_SUPPORT_OBJS)
	$(CC) $(GEMMA4_CFLAGS) $(INC) \
		-c tools/test/gpu-cuda-heterogeneous-test.c \
		-o /tmp/salt-gpu-cuda-heterogeneous-test.o
	$(CC) $(GEMMA4_CFLAGS) -Wno-error=maybe-uninitialized $(INC) \
		-c src/kernels.c -o /tmp/salt-gpu-cuda-heterogeneous-kernels.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/simd.c \
		-o /tmp/salt-gpu-cuda-heterogeneous-simd.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/gpu_resource.c \
		-o /tmp/salt-gpu-cuda-heterogeneous-resource.o

	$(NVCC) -arch=$(CUDA_ARCH) /tmp/salt-gpu-cuda-heterogeneous-test.o \
		/tmp/salt-gpu-cuda-heterogeneous-kernels.o \
		/tmp/salt-gpu-cuda-heterogeneous-simd.o \
		/tmp/salt-gpu-cuda-heterogeneous-resource.o \
		$(CUDA_TEXT_SUPPORT_OBJS) src/gpu_cuda.o \
		-lpthread -lm -o /tmp/salt-gpu-cuda-heterogeneous-test
	/tmp/salt-gpu-cuda-heterogeneous-test

test-gpu-cuda-nvfp4: src/gpu_cuda.o $(CUDA_TEXT_SUPPORT_OBJS)
	$(CC) $(GEMMA4_CFLAGS) $(INC) \
		-c tools/test/gpu-cuda-nvfp4-test.c \
		-o /tmp/salt-gpu-cuda-nvfp4-test.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/nvfp4.c \
		-o /tmp/salt-gpu-cuda-nvfp4-reference.o
	$(NVCC) -arch=$(CUDA_ARCH) /tmp/salt-gpu-cuda-nvfp4-test.o \
		/tmp/salt-gpu-cuda-nvfp4-reference.o $(CUDA_TEXT_SUPPORT_OBJS) \
		src/gpu_cuda.o \
		-lm -o /tmp/salt-gpu-cuda-nvfp4-test
	/tmp/salt-gpu-cuda-nvfp4-test

NVFP4_MODEL ?=
test-gpu-cuda-nvfp4-production: src/gpu_cuda.o $(CUDA_TEXT_SUPPORT_OBJS)
	$(CC) $(GEMMA4_CFLAGS) $(INC) \
		-c tools/test/gpu-cuda-nvfp4-production-test.c \
		-o /tmp/salt-gpu-cuda-nvfp4-production-test.o
	$(CC) $(GEMMA4_CFLAGS) $(INC) -c src/nvfp4.c \
		-o /tmp/salt-gpu-cuda-nvfp4-production-reference.o
	$(NVCC) -arch=$(CUDA_ARCH) /tmp/salt-gpu-cuda-nvfp4-production-test.o \
		/tmp/salt-gpu-cuda-nvfp4-production-reference.o \
		$(CUDA_TEXT_SUPPORT_OBJS) src/gpu_cuda.o \
		-lm -o /tmp/salt-gpu-cuda-nvfp4-production-test
	@test -n "$(NVFP4_MODEL)"
	/tmp/salt-gpu-cuda-nvfp4-production-test "$(NVFP4_MODEL)"

pack-trunk: tools/test/pack-trunk.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/test/pack-trunk.c -lm

make-fixture: tools/test/make-fixture.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/test/make-fixture.c -lm

bench-kernels: tools/test/bench-kernels.c $(SRC) $(HDR) $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/test/bench-kernels.c $(filter-out %.m %.mm $(GPU_OBJ),$(SRC)) $(GPU_OBJ) -lm $(GPU_LIBS)

validate-shapes: tools/test/validate-shapes.c $(SRC) $(HDR) $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/test/validate-shapes.c $(filter-out %.m %.mm $(GPU_OBJ),$(SRC)) $(GPU_OBJ) -lm $(GPU_LIBS)

# GPU expert offload microbenchmark (Darwin only): one expert layer,
# CPU 8-thread vs the existing per-matvec Metal API. Links only the
# kernels it needs (kernels.c + simd.c + the Metal shim) -- head.c and
# the rest of the engine are C99 and don't compile as C++.
BENCH_GPU_SRC = src/kernels.c src/simd.c $(GPU_OBJ)
bench-gpu-experts: tools/bench-gpu-experts.mm $(BENCH_GPU_SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ tools/bench-gpu-experts.mm $(BENCH_GPU_SRC) -lm $(GPU_LIBS)

test-gpu-ledger: test-gpu-projection
	python3 tools/test/gpu-pool-builder-test.py
	$(CC) $(CFLAGS) $(INC) tools/test/gpu-resource-binding-test.c \
		src/gpu_resource.c -o /tmp/salt-gpu-resource-binding-test
	/tmp/salt-gpu-resource-binding-test

test-gpu-projection: $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) \
		tools/test/gpu-ledger-projection-test.c src/gpu_pool.c \
		src/gpu_resource.c src/gpu_layer.c src/sha256.c \
		$(filter-out %.m %.mm,$(GPU_SRC)) $(GPU_OBJ) \
		-lm $(GPU_LIBS) -o /tmp/salt-gpu-ledger-projection-test
	@if [ -n "$(GPU_LEDGER_DIR)" ] && [ -n "$(GPU_POOL_BIN)" ]; then \
		for component in 0 1 2; do \
			/tmp/salt-gpu-ledger-projection-test \
				"$(GPU_LEDGER_DIR)" "$(GPU_POOL_BIN)" $$component || exit; \
		done; \
	else \
		echo "gpu-ledger projection: SKIP (set GPU_LEDGER_DIR and GPU_POOL_BIN)"; \
	fi

test-gpu-moe-shared: $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) tools/test/gpu-moe-shared-mapped-test.c \
		$(filter-out %.m %.mm $(GPU_OBJ),$(SRC)) $(GPU_OBJ) \
		-lm $(GPU_LIBS) -o /tmp/salt-gpu-moe-shared-mapped-test
	/tmp/salt-gpu-moe-shared-mapped-test

test-gpu-trunk: test-gpu-moe-shared $(GPU_OBJ)
	$(CC) $(CFLAGS) $(INC) tools/test/gpu-trunk-mapped-test.c \
		src/trunk.c src/gpu_resource.c src/kernels.c src/simd.c \
		$(filter-out %.m %.mm,$(GPU_SRC)) $(GPU_OBJ) \
		-lm $(GPU_LIBS) -o /tmp/salt-gpu-trunk-mapped-test
	/tmp/salt-gpu-trunk-mapped-test

# Actual-width bit-identity probe: compare the production layer-0 Q4
# projection at B=200 across CPU A12, CPU base, and source-backed Metal.
# The production files are read-only inputs and are required explicitly.
test-gpu-trunk-batch-exact: $(GPU_OBJ)
	@if [ -z "$(GPU_TRUNK_BIN)" ] || [ -z "$(GPU_TRUNK_OFFSETS)" ] || \
		[ -z "$(GPU_TRUNK_JSON)" ]; then \
		echo "gpu trunk batch exact: FAIL (set GPU_TRUNK_BIN, GPU_TRUNK_OFFSETS, GPU_TRUNK_JSON)"; \
		exit 2; \
	fi
	$(CC) $(CFLAGS) $(INC) tools/test/gpu-trunk-batch-exact-test.c \
		$(filter-out %.m %.mm $(GPU_OBJ),$(SRC)) $(GPU_OBJ) \
		-lm $(GPU_LIBS) -o /tmp/salt-gpu-trunk-batch-exact-test
	/tmp/salt-gpu-trunk-batch-exact-test \
		"$(GPU_TRUNK_BIN)" "$(GPU_TRUNK_OFFSETS)" "$(GPU_TRUNK_JSON)"

test-gpu-qualification:
	@if [ -z "$(GPU_LEDGER_DIR)" ] || [ -z "$(GPU_POOL_BIN)" ]; then \
		echo "gpu qualification: FAIL (set GPU_LEDGER_DIR and GPU_POOL_BIN)"; \
		exit 2; \
	fi
	SALT_REQUIRE_GPU=1 $(MAKE) -B test-gpu-trunk
	SALT_REQUIRE_GPU=1 $(MAKE) -B test-gpu-projection \
		GPU_LEDGER_DIR="$(GPU_LEDGER_DIR)" GPU_POOL_BIN="$(GPU_POOL_BIN)"

test-config:
	python3 tools/test/engine-config-test.py

test-spectrum-gpu-telemetry:
	python3 tools/test/spectrum-gpu-telemetry-test.py

test-public-smoke:
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/engine-config-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-tools-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gpu-metal-selected-capacity-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-compat-projection-test.py --production
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-compat-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gpu-residency-source-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-shared-kv-transform-source-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-http-framing-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-kv-cache-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-persistent-session-test.py
	env -u PYTHONPATH "$(GEMMA4_E2E_PYTHON)" tools/test/gemma4-server-test.py
	@set -eu; scratch=$$(mktemp -d "$${TMPDIR:-/tmp}/salt-public-smoke.XXXXXX"); \
		trap 'rm -f "$$scratch/state-control" "$$scratch/bit-identity" "$$scratch/gpu-resource"; rmdir "$$scratch"' EXIT; \
		$(CC) $(GEMMA4_CFLAGS) $(INC) -o "$$scratch/state-control" \
			tools/test/state-control-test.c $(MODEL_REGISTRY_SRC) src/state.c -lm; \
		"$$scratch/state-control"; \
		$(CC) $(GEMMA4_CFLAGS) $(INC) -o "$$scratch/bit-identity" \
			tools/test/gemma4-bit-identity-test.c src/kernels.c src/simd.c src/gpu_stub.c -lm; \
		"$$scratch/bit-identity"; \
		$(CC) $(GEMMA4_CFLAGS) $(INC) -o "$$scratch/gpu-resource" \
			tools/test/gpu-resource-binding-test.c src/gpu_resource.c -lm; \
		"$$scratch/gpu-resource"

# the disk/cpu loading experiment (A3B-shaped fixture, footprint report)
FIXDIR ?= /tmp/fix35
experiment: all
	bash tools/run-experiment.sh $(FIXDIR)

clean:
	rm -rf build salt gemma4-qa gemma4-vision-qa gemma4-multimodal-qa gemma4-server \
		gemma4-qa.build.json gemma4-multimodal-qa.build.json \
		gemma4-server.build.json gemma4-server-state-fixture \
		gemma4-dpr-store-fixture gemma4-dpr-edge \
		pack-trunk make-fixture $(GPU_OBJ) $(METAL_LIB)

.PHONY: all test test-public-smoke test-config test-gpu-ledger test-gpu-projection \
	test-hot-expert-ledger \
	test-gpu-moe-shared test-gpu-trunk test-gpu-trunk-batch-exact \
	test-gpu-rocm-exact \
	test-gpu-qualification test-gemma4-vision-image \
	test-gemma4-e2e-contract test-gemma4-finegrain-contract \
	test-gemma4-memory-accounting \
	test-gemma4-server-contract \
	test-gemma4-state-phase1 test-gemma4-state-phase2 \
	test-gemma4-state-phase3 test-gemma4-mindset-kv test-text-verify \
	test-text-token \
	test-gemma4-phase4 test-gemma4-phase5 \
	test-gemma4-phase5-strict test-gemma4-phase5-sanitize test-gemma4-phase6 \
	gemma4-build-receipts test-gemma4-state-red \
	test-gemma4-http-framing-contract test-gemma4-kv-cache-contract \
	test-gemma4-e2e clean
