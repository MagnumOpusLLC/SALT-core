#!/usr/bin/env python3
"""Static allocation and backend-surface contract for GPU residency Task 1."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HEADER=(ROOT/'include/salt/gpu_residency.h').read_text()
SOURCE=(ROOT/'src/gpu_residency.c').read_text()
GPU=(ROOT/'include/salt/gpu.h').read_text()

for forbidden in ('malloc(', 'calloc(', 'realloc(', 'free('):
    assert forbidden not in SOURCE, forbidden
for forbidden in ('hipEvent_t','hipStream_t','CUdeviceptr','MTLBuffer','cudaEvent_t'):
    assert forbidden not in HEADER, forbidden
for forbidden in ('salt/text_verify.h','salt/cache.h','salt/tensorops.h'):
    assert forbidden not in HEADER, forbidden

required=(
    'SALT_GPU_RESIDENCY_EMPTY',
    'SALT_GPU_RESIDENCY_POPULATING',
    'SALT_GPU_RESIDENCY_READY',
    'SaltGpuResidencyPlan',
    'SaltGpuResidencySource',
    'SaltGpuResidencyBackendOps',
    'SaltGpuResidencySlot',
    'SaltGpuResidencyBinding',
    'SaltGpuResidencyStats',
    'salt_gpu_residency_arena_requirement',
    'salt_gpu_residency_population_begin',
    'salt_gpu_residency_population_poll',
    'salt_gpu_residency_population_abort',
    'salt_gpu_residency_acquire',
    'salt_gpu_residency_release',
    'salt_gpu_residency_retire',
    'salt_gpu_residency_destroy',
)
for value in required:
    assert value in HEADER, value
assert 'SALT_GPU_RESIDENCY_DEVICE_LOCAL = 1' in HEADER
assert 'salt_gpu_residency_backend_ops(void);' in HEADER
assert 'SALT_GPU_WEIGHT_ADDRESS_DEVICE_RESIDENT' not in GPU
assert 'salt_gpu_residency_backend_ops(void);' not in GPU

for backend in ('gpu_stub.c','gpu_metal.m','gpu_cuda.cu'):
    text=(ROOT/'src'/backend).read_text()
    assert 'salt_gpu_residency_backend_ops(void)' in text, backend
    assert 'return NULL;' in text, backend
assert 'salt_gpu_residency_backend_ops(void)' in (ROOT/'src/gpu_hip.cpp').read_text()

print('GPU_RESIDENCY_SOURCE_TEST_PASS')
