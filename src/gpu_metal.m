// gpu_metal.m -- Darwin-only Objective-C adapter with a C99 language base.
// No Objective-C type crosses the salt/gpu.h C ABI. Optional Metal
// acceleration for the MLX-4bit output head compiles its kernel at runtime,
// so the Makefile needs no .metallib build step. Mirrors salt_q4_matvec's
// math exactly: per-64-element group BF16 scale (+ optional BF16 bias),
// q*s+b dequant, float32 accumulation. Weight buffers are cached across
// tokens (only x changes per token), so the per-token cost is one small
// upload + one 248K-row readback.
//
// Built only on Darwin (see Makefile). Non-Darwin builds use gpu_stub.c.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include "salt/gpu.h"
#include "salt/gpu_residency.h"
#include "salt/gpu_resource.h"
#include "salt/text_verify.h"
#include "salt/bitmath.h"
#include "gpu_metal_ops_source.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void metal_release(id object) {
    if (object) [object release];
}

static int metal_command_succeeded(id<MTLCommandBuffer> command) {
    if ([command status] == MTLCommandBufferStatusCompleted) return 0;
    NSError *error = [command error];
    fprintf(stderr, "gpu: Metal command failed%s%s\n",
            error ? ": " : "",
            error ? [[error localizedDescription] UTF8String] : "");
    return -1;
}

static void metal_disable_fast_math_legacy(MTLCompileOptions *opts) {
    /* Avoid a direct reference to the macOS-15-deprecated property while
     * retaining compatibility with runtimes where mathMode is unavailable. */
    [opts setValue:@NO forKey:@"fastMathEnabled"];
}

static void metal_exact_math(MTLCompileOptions *opts) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *))
        opts.mathMode = MTLMathModeSafe;
    else
        metal_disable_fast_math_legacy(opts);
#else
    metal_disable_fast_math_legacy(opts);
#endif
}

static int metal_buffer_gpu_address(id<MTLBuffer> buffer,
                                    uint64_t *address_out) {
    if (!address_out) return 0;
    *address_out = 0;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
    if (@available(macOS 13.0, *)) {
        *address_out = [buffer gpuAddress];
        return 1;
    }
#else
    (void)buffer;
#endif
    return 0;
}

static id<MTLDevice> _dev;
static id<MTLLibrary> _offline_library;
static id<MTLComputePipelineState> _pso;
static id<MTLCommandQueue> _queue;
/* cached weight buffers (never change between tokens) */
static id<MTLBuffer> _bv, _bs, _bb;
static id<MTLBuffer> _bx, _by;    /* per-token x / y */
static int _R = -1, _C = -1;
static int _has_bias = 0;
static id<MTLComputePipelineState> _transform_pso, _attention_pso;
static id<MTLComputePipelineState> _activation_pso, _q8_pso;
static id<MTLComputePipelineState> _text_embedding_pso, _text_route_maps_pso;
static id<MTLComputePipelineState> _text_routed_gather_pso;
static id<MTLComputePipelineState> _text_expert_reduce_pso;
static id<MTLComputePipelineState> _text_combine_pso;
static id<MTLComputePipelineState> _text_rms_rows_pso;
static id<MTLComputePipelineState> _text_residual_rows_pso;
static id<MTLComputePipelineState> _text_router_rows_pso;
static id<MTLComputePipelineState> _text_topk_rows_pso;
static id<MTLComputePipelineState> _text_softcap_pso;
static id<MTLComputePipelineState> _text_transform_view_pso;
static id<MTLComputePipelineState> _text_attention_view_pso;
static id<MTLComputePipelineState> _exact_rms_pso, _exact_topk_pso;
static id<MTLComputePipelineState> _exact_residual_pso, _exact_combine_pso;
static id<MTLComputePipelineState> _exact_softcap_pso, _exact_router_pso;
static id<MTLBuffer> _op_meta, _op_status;
typedef struct {
    id<MTLBuffer> buffer;
    void *contents;
    size_t nbytes;
} MetalSharedSlot;
#define METAL_SHARED_SLOTS 8
static MetalSharedSlot _shared_slots[METAL_SHARED_SLOTS];

static id<MTLLibrary> metal_library_for_source_mode(const char *source,
                                                    NSError **error,
                                                    int t4_proof,
                                                    int allow_offline) {
    if (allow_offline && _offline_library) return [_offline_library retain];
    if (!source) return nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    metal_exact_math(opts);
    opts.preprocessorMacros = @{
        @"SALT_T4_PROOF": t4_proof ? @1 : @0,
    };
    id<MTLLibrary> library = [_dev newLibraryWithSource:
        [NSString stringWithUTF8String:source] options:opts error:error];
    [opts release];
    return library;
}

static id<MTLLibrary> metal_library_for_source(const char *source,
                                               NSError **error) {
    return metal_library_for_source_mode(source, error, 0, 1);
}

static int metal_ops_prepare(void) {
    if (_transform_pso && _attention_pso && _activation_pso && _q8_pso &&
        _text_embedding_pso && _text_route_maps_pso &&
        _text_routed_gather_pso && _text_expert_reduce_pso &&
        _text_combine_pso &&
        _text_rms_rows_pso && _text_residual_rows_pso &&
        _text_router_rows_pso && _text_topk_rows_pso && _text_softcap_pso &&
        _text_transform_view_pso && _text_attention_view_pso)
        return 0;
    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> lib = metal_library_for_source_mode(
            salt_gpu_metal_ops_source, &err, 0, 0);
        if (!lib) {
            fprintf(stderr, "gpu: Metal operator compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return -1;
        }
        struct { const char *name; id<MTLComputePipelineState> *slot; } specs[] = {
            {"attention_transform", &_transform_pso},
            {"attention_exact", &_attention_pso},
            {"moe_activate", &_activation_pso},
            {"q8batch", &_q8_pso},
            {"text_embedding_q4", &_text_embedding_pso},
            {"text_route_maps", &_text_route_maps_pso},
            {"text_routed_gather", &_text_routed_gather_pso},
            {"text_expert_reduce", &_text_expert_reduce_pso},
            {"text_parallel_combine_rows", &_text_combine_pso},
            {"text_rmsnorm_rows", &_text_rms_rows_pso},
            {"text_residual_rows", &_text_residual_rows_pso},
            {"text_router_input_rows", &_text_router_rows_pso},
            {"text_topk_rows", &_text_topk_rows_pso},
            {"text_softcap", &_text_softcap_pso},
            {"attention_transform_view", &_text_transform_view_pso},
            {"attention_exact_view", &_text_attention_view_pso},
        };
        for (size_t i = 0; i < sizeof specs / sizeof specs[0]; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:specs[i].name]];
            if (!fn) { [lib release]; return -1; }
            *specs[i].slot = [_dev newComputePipelineStateWithFunction:fn
                                                                  error:&err];
            [fn release];
            if (!*specs[i].slot) { [lib release]; return -1; }
        }
        [lib release];
        _op_status = [_dev newBufferWithLength:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
        _op_meta = [_dev newBufferWithLength:2u * 1024u * 1024u
                                      options:MTLResourceStorageModeShared];
        if (!_op_status || !_op_meta) return -1;
    }
    return 0;
}

static int metal_exact_prepare(void) {
    id<MTLComputePipelineState> created[6] = {nil, nil, nil, nil, nil, nil};
    if (_exact_rms_pso && _exact_topk_pso && _exact_residual_pso &&
        _exact_combine_pso && _exact_softcap_pso && _exact_router_pso)
        return 0;
    if (metal_ops_prepare() != 0) return -1;
    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> lib = metal_library_for_source_mode(
            salt_gpu_metal_ops_source, &err, 1, 0);
        struct {
            const char *name;
            id<MTLComputePipelineState> *slot;
        } specs[] = {
            {"exact_rmsnorm", &_exact_rms_pso},
            {"exact_topk", &_exact_topk_pso},
            {"exact_residual_postnorm", &_exact_residual_pso},
            {"exact_parallel_combine", &_exact_combine_pso},
            {"exact_softcap", &_exact_softcap_pso},
            {"exact_router_input", &_exact_router_pso},
        };
        if (!lib) {
            fprintf(stderr, "gpu: Metal exact proof compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return -1;
        }
        for (size_t i = 0; i < sizeof specs / sizeof specs[0]; i++) {
            id<MTLFunction> fn;
            if (*specs[i].slot) continue;
            fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:specs[i].name]];
            if (!fn) goto fail;
            created[i] = [_dev newComputePipelineStateWithFunction:fn
                                                              error:&err];
            [fn release];
            if (!created[i]) goto fail;
        }
        for (size_t i = 0; i < sizeof specs / sizeof specs[0]; i++)
            if (created[i]) *specs[i].slot = created[i];
        [lib release];
        return 0;
fail:
        for (size_t i = 0; i < sizeof created / sizeof created[0]; i++)
            metal_release(created[i]);
        [lib release];
        return -1;
    }
}

static int metal_shared_range(const SaltGpuSharedBuffer *buffer,
                              size_t float_offset, size_t float_count,
                              id<MTLBuffer> *storage_out) {
    id<MTLBuffer> storage;
    size_t capacity;
    if (!buffer || !buffer->contents || !buffer->backend || !storage_out ||
        buffer->nbytes < sizeof(float)) return -1;
    storage = (id<MTLBuffer>)buffer->backend;
    capacity = buffer->nbytes / sizeof(float);
    if ([storage contents] != buffer->contents ||
        (size_t)[storage length] < buffer->nbytes ||
        float_offset > capacity || float_count > capacity - float_offset)
        return -1;
    *storage_out = storage;
    return 0;
}

static int metal_shared_pointer(const float *pointer, size_t count,
                                id<MTLBuffer> *buffer_out,
                                size_t *byte_offset_out) {
    uintptr_t p = (uintptr_t)(const void *)pointer;
    size_t bytes;
    if (!pointer || !buffer_out || !byte_offset_out ||
        count > SIZE_MAX / sizeof(float)) return -1;
    bytes = count * sizeof(float);
    for (int i = 0; i < METAL_SHARED_SLOTS; i++) {
        uintptr_t base;
        size_t delta;
        if (!_shared_slots[i].buffer || !_shared_slots[i].contents) continue;
        base = (uintptr_t)_shared_slots[i].contents;
        if (p < base || p - base > SIZE_MAX) continue;
        delta = (size_t)(p - base);
        if (delta <= _shared_slots[i].nbytes &&
            bytes <= _shared_slots[i].nbytes - delta) {
            *buffer_out = _shared_slots[i].buffer;
            *byte_offset_out = delta;
            return 0;
        }
    }
    return -1;
}

static const char *_kernel_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void q4mv(\n"
    "    device const uint *vals    [[buffer(0)]],\n"
    "    device const ushort *scales [[buffer(1)]],\n"
    "    device const ushort *biases [[buffer(2)]],\n"
    "    device const float *x       [[buffer(3)]],\n"
    "    device float *y             [[buffer(4)]],\n"
    "    constant uint &R            [[buffer(5)]],\n"
    "    constant uint &C            [[buffer(6)]],\n"
    "    uint r                      [[thread_position_in_grid]])\n"
    "{\n"
    "    if (r >= R) return;\n"
    "    float acc = 0.0f;\n"
    "    for (uint c = 0; c < C; c++) {\n"
    "        uint k = r * C + c;\n"
    "        uint g = k / 64u;\n"
    "        float s = as_type<float>(uint(scales[g]) << 16);\n"
    "        float b = as_type<float>(uint(biases[g]) << 16);\n"
    "        uint q = (vals[k >> 3u] >> ((k & 7u) << 2u)) & 0xFu;\n"
    "        // EXACT scalar staging (the CPU anchor's LUT path):\n"
    "        // lut[q] = (float)q * s; += b;  then acc += lut[q] * x[c]\n"
    "        float t = float(q) * s;\n"
    "        t += b;\n"
    "        float u = t * x[c];\n"
    "        acc += u;\n"
    "    }\n"
    "    y[r] = acc;\n"
    "}\n";

static int metal_session_buffers_prepare(void);
static int salt_gpu_free_impl(void);

int salt_gpu_startup_requirements(
    SaltGpuStartupRequirements *requirements, size_t requirements_size) {
    if (!requirements || requirements_size != sizeof *requirements) return -1;
    memset(requirements, 0, sizeof *requirements);
    requirements->shared_buffer_slots = METAL_SHARED_SLOTS;
    return 0;
}

int salt_gpu_init(void) {
    if (_dev) return 0;
    @autoreleasepool {
        _dev = MTLCreateSystemDefaultDevice();
        if (!_dev) return -1;
        NSError *err = nil;
        const char *library_path = getenv("SALT_METAL_LIBRARY");
        if (library_path && *library_path) {
            NSString *path = [[NSString stringWithUTF8String:library_path]
                stringByStandardizingPath];
            NSURL *url = [NSURL fileURLWithPath:path];
            _offline_library = [_dev newLibraryWithURL:url error:&err];
            if (!_offline_library) {
                fprintf(stderr, "gpu: precompiled Metal library failed: %s\n",
                        err ? [[err localizedDescription] UTF8String] : "?");
                [_dev release];
                _dev = nil;
                return -1;
            }
        }
        id<MTLLibrary> lib = metal_library_for_source(_kernel_src, &err);
        if (!lib) {
            fprintf(stderr, "gpu: kernel compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return -1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q4mv"];
        [lib release];
        if (!fn) {
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return -1;
        }
        _pso = [_dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        if (!_pso) {
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return -1;
        }
        _queue = [_dev newCommandQueue];
        if (!_queue) {
            [_pso release];
            _pso = nil;
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return -1;
        }
        if (metal_ops_prepare() != 0 || metal_session_buffers_prepare() != 0) {
            (void)salt_gpu_free_impl();
            return -1;
        }
        fprintf(stderr, "gpu: Metal ready (%s)\n",
                [[_dev name] UTF8String]);
    }
    return 0;
}

int salt_gpu_free(void) {
    return salt_gpu_free_impl();
}

int salt_gpu_shared_buffer_alloc(SaltGpuSharedBuffer *buffer, size_t nbytes) {
    id<MTLBuffer> storage;
    int shared_slot = -1;
    if (!buffer || !_dev || nbytes < 1) return -1;
    memset(buffer, 0, sizeof *buffer);
    if (nbytes > (size_t)NSUIntegerMax) return -1;
    @autoreleasepool {
        storage = [_dev newBufferWithLength:nbytes
                                    options:MTLResourceStorageModeShared];
        if (!storage || ![storage contents] || [storage length] < nbytes) {
            metal_release(storage);
            return -1;
        }
        buffer->contents = [storage contents];
        buffer->nbytes = (size_t)[storage length];
        buffer->backend = (void *)storage;
        for (int i = 0; i < METAL_SHARED_SLOTS; i++)
            if (!_shared_slots[i].buffer) { shared_slot = i; break; }
        if (shared_slot < 0) {
            memset(buffer, 0, sizeof *buffer);
            [storage release];
            return -1;
        }
        _shared_slots[shared_slot] = (MetalSharedSlot) {
            storage, [storage contents], (size_t)[storage length],
        };
    }
    return 0;
}

typedef struct MetalExactArgs {
    uint64_t off[8];
    uint32_t n, aux;
    float eps, scalar;
} MetalExactArgs;

static int metal_exact_view(const SaltGpuExactCell *cell, int view,
                            size_t count, id<MTLBuffer> *storage) {
    if (!cell || view < 0 || view >= 8 || !cell->views[view]) return -1;
    return metal_shared_range(cell->views[view], cell->offsets[view],
                              count, storage);
}

static int metal_exact_encode(id<MTLCommandBuffer> command,
                              const SaltGpuExactCell *cell) {
    MetalExactArgs args;
    id<MTLBuffer> views[8] = {nil, nil, nil, nil, nil, nil, nil, nil};
    id<MTLComputePipelineState> pipeline = nil;
    int used = 0;
    if (!command || !cell || cell->n < 1) return -1;
    memset(&args, 0, sizeof args);
    for (int i = 0; i < 8; i++) args.off[i] = cell->offsets[i];
    args.n = cell->n;
    args.aux = cell->aux;
    args.eps = cell->eps;
    args.scalar = cell->scalar;
    switch (cell->kind) {
    case SALT_GPU_EXACT_RMSNORM:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) || cell->aux > 1u ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->aux ? cell->n : 1u,
                             &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->n, &views[2]) != 0)
            return -1;
        pipeline = _exact_rms_pso; used = 3;
        break;
    case SALT_GPU_EXACT_ROUTER_INPUT:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) ||
            !(cell->scalar > 0.0f) || !isfinite(cell->scalar) ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->n, &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->n, &views[2]) != 0)
            return -1;
        pipeline = _exact_router_pso; used = 3;
        break;
    case SALT_GPU_EXACT_TOPK:
        if (cell->aux < 1u || cell->aux > cell->n ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->n, &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->aux, &views[2]) != 0 ||
            metal_exact_view(cell, 3, cell->aux, &views[3]) != 0)
            return -1;
        pipeline = _exact_topk_pso; used = 4;
        break;
    case SALT_GPU_EXACT_RESIDUAL_POSTNORM:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->n, &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->n, &views[2]) != 0 ||
            metal_exact_view(cell, 3, cell->n, &views[3]) != 0)
            return -1;
        pipeline = _exact_residual_pso; used = 4;
        break;
    case SALT_GPU_EXACT_PARALLEL_COMBINE:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) ||
            !isfinite(cell->scalar)) return -1;
        for (int i = 0; i < 8; i++)
            if (metal_exact_view(cell, i,
                    i == 6 ? (size_t)cell->n * 2u : cell->n,
                    &views[i]) != 0)
                return -1;
        pipeline = _exact_combine_pso; used = 8;
        break;
    case SALT_GPU_EXACT_SOFTCAP:
        if (!(cell->scalar > 0.0f) || !isfinite(cell->scalar) ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0)
            return -1;
        pipeline = _exact_softcap_pso; used = 1;
        break;
    default:
        return -1;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder || !pipeline) return -1;
        [encoder setComputePipelineState:pipeline];
        for (int i = 0; i < used; i++)
            [encoder setBuffer:views[i] offset:0 atIndex:(NSUInteger)i];
        [encoder setBuffer:_op_status offset:0 atIndex:(NSUInteger)used];
        [encoder setBytes:&args length:sizeof args
                  atIndex:(NSUInteger)(used + 1)];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
    }
    return 0;
}

int salt_gpu_exact_cells(const SaltGpuExactCell *cells, int count) {
    id<MTLCommandBuffer> command;
    if (!_dev || !cells || count < 1 || metal_exact_prepare() != 0 ||
        salt_gpu_sync() != 0)
        return -1;
    *(uint32_t *)[_op_status contents] = 0;
    command = [_queue commandBuffer];
    if (!command) return -1;
    for (int index = 0; index < count; index++)
        if (metal_exact_encode(command, &cells[index]) != 0)
            return -1;
    [command commit];
    [command waitUntilCompleted];
    return metal_command_succeeded(command) == 0 &&
        *(uint32_t *)[_op_status contents] == 0 ? 0 : -1;
}

int salt_gpu_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y) {
    if (!_dev || !_pso || R < 1 || C < 1) return -1;
    if (salt_gpu_mapped_only()) {
        const uint32_t *value_jobs[1] = {vals};
        const uint16_t *scale_jobs[1] = {scales};
        const uint16_t *bias_jobs[1] = {biases};
        const float *input_jobs[1] = {x};
        float *output_jobs[1] = {y};
        const void *keys[1] = {vals};
        int rows[1] = {R};
        if (salt_gpu_q4_batch(value_jobs, scale_jobs, bias_jobs,
                              input_jobs, output_jobs, keys, rows,
                              C, 1) != 0)
            return -1;
        return salt_gpu_sync();
    }
    @autoreleasepool {
        /* The singleton path is synchronous, so replacing a grown buffer
         * after the prior call completes is ownership-safe. */
        size_t vbytes = (size_t)R * (size_t)C / 2u;
        size_t sbytes = ((size_t)R * (size_t)C + 63u) / 64u * 2u;
        if (!_bv || vbytes > _bv.length || !_bs || sbytes > _bs.length ||
            (biases && (!_bb || sbytes > _bb.length))) {
            id<MTLBuffer> bv = [_dev newBufferWithLength:vbytes
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bs = [_dev newBufferWithLength:sbytes
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bb = biases
                ? [_dev newBufferWithLength:sbytes
                                    options:MTLResourceStorageModeShared]
                : nil;
            if (!bv || !bs || (biases && !bb)) {
                metal_release(bv); metal_release(bs); metal_release(bb);
                return -1;
            }
            metal_release(_bv); metal_release(_bs); metal_release(_bb);
            _bv = bv; _bs = bs; _bb = bb;
        }
        memcpy(_bv.contents, vals, vbytes);
        memcpy(_bs.contents, scales, sbytes);
        if (biases) memcpy(_bb.contents, biases, sbytes);
        _R = R; _C = C; _has_bias = (biases != NULL);
        if (!_bx || (size_t)C * sizeof(float) > _bx.length) {
            id<MTLBuffer> bx = [_dev newBufferWithLength:
                (size_t)C * sizeof(float) options:MTLResourceStorageModeShared];
            if (!bx) return -1;
            metal_release(_bx);
            _bx = bx;
        }
        if (!_by || (size_t)R * sizeof(float) > _by.length) {
            id<MTLBuffer> by = [_dev newBufferWithLength:
                (size_t)R * sizeof(float) options:MTLResourceStorageModeShared];
            if (!by) return -1;
            metal_release(_by);
            _by = by;
        }
        memcpy(_bx.contents, x, (size_t)C * sizeof(float));

        id<MTLCommandBuffer> cb = [_queue commandBuffer];
        if (!cb) return -1;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return -1;
        [enc setComputePipelineState:_pso];
        [enc setBuffer:_bv offset:0 atIndex:0];
        [enc setBuffer:_bs offset:0 atIndex:1];
        [enc setBuffer:_bb offset:0 atIndex:2];
        [enc setBuffer:_bx offset:0 atIndex:3];
        [enc setBuffer:_by offset:0 atIndex:4];
        uint rv = (uint)R, cv = (uint)C;
        [enc setBytes:&rv length:sizeof(rv) atIndex:5];
        [enc setBytes:&cv length:sizeof(cv) atIndex:6];
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake((NSUInteger)R, 1, 1);
        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return -1;
        memcpy(y, _by.contents, (size_t)R * sizeof(float));
    }
    return 0;
}

/* ---- batched expert matvec (the 13x design) --------------------- */
static const char *_batch_src2 =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "inline uint load32(device const uchar *p){return uint(p[0])|(uint(p[1])<<8)|(uint(p[2])<<16)|(uint(p[3])<<24);}\n"
    "inline ushort load16(device const uchar *p){return ushort(uint(p[0])|(uint(p[1])<<8));}\n"
    "kernel void q4batch2(\n"
    "    device const uchar  *vals    [[buffer(0)]],\n"
    "    device const uchar  *scales  [[buffer(1)]],\n"
    "    device const uchar  *biases  [[buffer(2)]],\n"
    "    device const float  *x       [[buffer(3)]],\n"
    "    device float        *ys      [[buffer(4)]],\n"
    "    device const ulong4 *desc    [[buffer(5)]],  // 64-bit source and destination offsets\n"
    "    constant uint &R             [[buffer(6)]],\n"
    "    constant uint &C             [[buffer(7)]],\n"
    "    constant uint &njobs         [[buffer(8)]],\n"
    "    uint idx                     [[thread_position_in_grid]])\n"
    "{\n"
    "    // grid = total rows across jobs; desc[j*2+1].w holds the\n"
    "    // job's row-start (the host's prefix sum), so jobs may have\n"
    "    // MIXED row counts in one call (the 3-in-1 gate+up+down fold).\n"
    "    uint j = njobs - 1u;\n"
    "    for (uint t = 1; t < njobs; t++)\n"
    "        if (desc[t * 2 + 1].w > idx) { j = t - 1u; break; }\n"
    "    uint r = idx - desc[j * 2 + 1].w;\n"
    "    // per-job base from the desc (arena appends in first-seen\n"
    "    // order across layers, so offsets are NOT j*stride)\n"
    "    ulong4 d = desc[j * 2];\n"
    "    ulong4 db = desc[j * 2 + 1];\n"
    "    device const uchar *vr = vals + d.x;\n"
    "    device const uchar *sr = scales + d.y;\n"
    "    device const uchar *br = biases + db.x;\n"
    "    // x/y live in the caller's CPU buffers (shared memory): the\n"
    "    // kernel reads x at d.z and writes y at d.w directly -- no\n"
    "    // host-side pack/scatter copies.\n"
    "    device const float *xj = x + d.z;\n"
    "    device float *yj = ys + d.w;\n"
    "    // Preserve the established scalar path for uncommon tail shapes.\n"
    "    // Production Q4 trunk projections are 32-column aligned.\n"
    "    if ((C & 31u) != 0u) {\n"
    "        float scalar = 0.0f;\n"
    "        for (uint tc = 0u; tc < C; tc++) {\n"
    "            uint tk = r * C + tc;\n"
    "            uint tg = tk / 64u;\n"
    "            float ts = as_type<float>(uint(load16(sr+tg*2u)) << 16);\n"
    "            float tb = as_type<float>(uint(load16(br+tg*2u)) << 16);\n"
    "            uint tq = (load32(vr+(tk >> 3u)*4u) >> ((tk & 7u) << 2u)) & 0xFu;\n"
    "            float tw = float(tq) * ts; tw += tb;\n"
    "            float tp = tw * xj[tc]; scalar += tp;\n"
    "        }\n"
    "        yj[r] = scalar;\n"
    "        return;\n"
    "    }\n"
    "    // Mirror q4_batch_worker_tiled exactly: one 32-column weight\n"
    "    // block updates eight float4 accumulators.  Dequant, product,\n"
    "    // accumulation, and final lane reduction remain separate ops.\n"
    "    float4 a0 = float4(0.0f), a1 = float4(0.0f);\n"
    "    float4 a2 = float4(0.0f), a3 = float4(0.0f);\n"
    "    float4 a4 = float4(0.0f), a5 = float4(0.0f);\n"
    "    float4 a6 = float4(0.0f), a7 = float4(0.0f);\n"
    "    uint c = 0u;\n"
    "    for (; c + 31u < C; c += 32u) {\n"
    "        uint k = r * C + c;\n"
    "        uint g = k / 64u;\n"
    "        float s = as_type<float>(uint(load16(sr+g*2u)) << 16);\n"
    "        float b = as_type<float>(uint(load16(br+g*2u)) << 16);\n"
    "        ulong word = (k >> 3u)*4u;\n"
    "        uint u0 = load32(vr+word+0u), u1 = load32(vr+word+4u);\n"
    "        uint u2 = load32(vr+word+8u), u3 = load32(vr+word+12u);\n"
    "        float4 q0 = float4((u0 >> 0u) & 15u, (u0 >> 4u) & 15u,\n"
    "                           (u0 >> 8u) & 15u, (u0 >> 12u) & 15u);\n"
    "        float4 q1 = float4((u0 >> 16u) & 15u, (u0 >> 20u) & 15u,\n"
    "                           (u0 >> 24u) & 15u, (u0 >> 28u) & 15u);\n"
    "        float4 q2 = float4((u1 >> 0u) & 15u, (u1 >> 4u) & 15u,\n"
    "                           (u1 >> 8u) & 15u, (u1 >> 12u) & 15u);\n"
    "        float4 q3 = float4((u1 >> 16u) & 15u, (u1 >> 20u) & 15u,\n"
    "                           (u1 >> 24u) & 15u, (u1 >> 28u) & 15u);\n"
    "        float4 q4 = float4((u2 >> 0u) & 15u, (u2 >> 4u) & 15u,\n"
    "                           (u2 >> 8u) & 15u, (u2 >> 12u) & 15u);\n"
    "        float4 q5 = float4((u2 >> 16u) & 15u, (u2 >> 20u) & 15u,\n"
    "                           (u2 >> 24u) & 15u, (u2 >> 28u) & 15u);\n"
    "        float4 q6 = float4((u3 >> 0u) & 15u, (u3 >> 4u) & 15u,\n"
    "                           (u3 >> 8u) & 15u, (u3 >> 12u) & 15u);\n"
    "        float4 q7 = float4((u3 >> 16u) & 15u, (u3 >> 20u) & 15u,\n"
    "                           (u3 >> 24u) & 15u, (u3 >> 28u) & 15u);\n"
    "        float4 w0 = q0 * s; w0 += b;\n"
    "        float4 w1 = q1 * s; w1 += b;\n"
    "        float4 w2 = q2 * s; w2 += b;\n"
    "        float4 w3 = q3 * s; w3 += b;\n"
    "        float4 w4 = q4 * s; w4 += b;\n"
    "        float4 w5 = q5 * s; w5 += b;\n"
    "        float4 w6 = q6 * s; w6 += b;\n"
    "        float4 w7 = q7 * s; w7 += b;\n"
    "        float4 p0 = w0 * float4(xj[c+0u], xj[c+1u], xj[c+2u], xj[c+3u]);\n"
    "        float4 p1 = w1 * float4(xj[c+4u], xj[c+5u], xj[c+6u], xj[c+7u]);\n"
    "        float4 p2 = w2 * float4(xj[c+8u], xj[c+9u], xj[c+10u], xj[c+11u]);\n"
    "        float4 p3 = w3 * float4(xj[c+12u], xj[c+13u], xj[c+14u], xj[c+15u]);\n"
    "        float4 p4 = w4 * float4(xj[c+16u], xj[c+17u], xj[c+18u], xj[c+19u]);\n"
    "        float4 p5 = w5 * float4(xj[c+20u], xj[c+21u], xj[c+22u], xj[c+23u]);\n"
    "        float4 p6 = w6 * float4(xj[c+24u], xj[c+25u], xj[c+26u], xj[c+27u]);\n"
    "        float4 p7 = w7 * float4(xj[c+28u], xj[c+29u], xj[c+30u], xj[c+31u]);\n"
    "        a0 += p0; a1 += p1; a2 += p2; a3 += p3;\n"
    "        a4 += p4; a5 += p5; a6 += p6; a7 += p7;\n"
    "    }\n"
    "    float4 s4 = float4(0.0f);\n"
    "    s4 += a0; s4 += a1; s4 += a2; s4 += a3;\n"
    "    s4 += a4; s4 += a5; s4 += a6; s4 += a7;\n"
    "    float2 tt = s4.xy + s4.zw;\n"
    "    float acc = tt.x + tt.y;\n"
    "    yj[r] = acc;\n"
    "}\n"
    "kernel void q4weightstationary(\n"
    "    device const uchar *vals [[buffer(0)]],device const uchar *scales [[buffer(1)]],\n"
    "    device const uchar *biases [[buffer(2)]],device const float *x [[buffer(3)]],\n"
    "    device float *ys [[buffer(4)]],device const ulong4 *desc [[buffer(5)]],\n"
    "    threadgroup float *tile [[threadgroup(0)]],uint2 tg [[threadgroup_position_in_grid]],\n"
    "    uint2 ti [[thread_position_in_threadgroup]])\n"
    "{\n"
    "    ulong4 d=desc[0],db=desc[1];uint R=uint(db.y),C=uint(db.z),B=uint(db.w);\n"
    "    uint rbase=tg.x*8u,tbase=tg.y*8u,r=rbase+ti.x,token=tbase+ti.y;\n"
    "    uint linear=ti.y*8u+ti.x;threadgroup float *wt=tile,*xt=tile+256u;\n"
    "    bool active=r<R&&token<B;float4 a0=float4(0.0f),a1=float4(0.0f);\n"
    "    float4 a2=float4(0.0f),a3=float4(0.0f),a4=float4(0.0f),a5=float4(0.0f);\n"
    "    float4 a6=float4(0.0f),a7=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){\n"
    "        if(linear<32u){uint wr=linear>>2u,wi=linear&3u,rr=rbase+wr;\n"
    "            uint to=wr*32u+wi*8u;if(rr<R){ulong k=ulong(rr)*C+c+wi*8u,g=k/64u;\n"
    "                uint u=load32(vals+d.x+(k>>3u)*4u);float s=as_type<float>(uint(load16(scales+d.y+g*2u))<<16);\n"
    "                float b=as_type<float>(uint(load16(biases+db.x+g*2u))<<16);\n"
    "                float4 q0=float4((u>>0u)&15u,(u>>4u)&15u,(u>>8u)&15u,(u>>12u)&15u);\n"
    "                float4 q1=float4((u>>16u)&15u,(u>>20u)&15u,(u>>24u)&15u,(u>>28u)&15u);\n"
    "                float4 w0=q0*s;w0+=b;float4 w1=q1*s;w1+=b;\n"
    "                wt[to]=w0.x;wt[to+1u]=w0.y;wt[to+2u]=w0.z;wt[to+3u]=w0.w;\n"
    "                wt[to+4u]=w1.x;wt[to+5u]=w1.y;wt[to+6u]=w1.z;wt[to+7u]=w1.w;\n"
    "            }else{for(uint q=0u;q<8u;q++)wt[to+q]=0.0f;}}\n"
    "        for(uint xi=linear;xi<256u;xi+=64u){uint tr=xi>>5u,cc=xi&31u,tt=tbase+tr;\n"
    "            xt[xi]=tt<B?x[d.z+ulong(tt)*C+c+cc]:0.0f;}\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if(active){threadgroup float *w=wt+ti.x*32u,*xx=xt+ti.y*32u;\n"
    "            float4 p0=float4(w[0],w[1],w[2],w[3])*float4(xx[0],xx[1],xx[2],xx[3]);\n"
    "            float4 p1=float4(w[4],w[5],w[6],w[7])*float4(xx[4],xx[5],xx[6],xx[7]);\n"
    "            float4 p2=float4(w[8],w[9],w[10],w[11])*float4(xx[8],xx[9],xx[10],xx[11]);\n"
    "            float4 p3=float4(w[12],w[13],w[14],w[15])*float4(xx[12],xx[13],xx[14],xx[15]);\n"
    "            float4 p4=float4(w[16],w[17],w[18],w[19])*float4(xx[16],xx[17],xx[18],xx[19]);\n"
    "            float4 p5=float4(w[20],w[21],w[22],w[23])*float4(xx[20],xx[21],xx[22],xx[23]);\n"
    "            float4 p6=float4(w[24],w[25],w[26],w[27])*float4(xx[24],xx[25],xx[26],xx[27]);\n"
    "            float4 p7=float4(w[28],w[29],w[30],w[31])*float4(xx[28],xx[29],xx[30],xx[31]);\n"
    "            a0+=p0;a1+=p1;a2+=p2;a3+=p3;a4+=p4;a5+=p5;a6+=p6;a7+=p7;}\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if(active){float4 s4=float4(0.0f);s4+=a0;s4+=a1;s4+=a2;s4+=a3;\n"
    "        s4+=a4;s4+=a5;s4+=a6;s4+=a7;float2 tt=s4.xy+s4.zw;\n"
    "        ys[d.w+ulong(token)*R+r]=tt.x+tt.y;}\n"
    "}\n"
    "kernel void q4coalesced(\n"
    "    device const uchar *vals [[buffer(0)]],device const uchar *scales [[buffer(1)]],\n"
    "    device const uchar *biases [[buffer(2)]],device const float *x [[buffer(3)]],\n"
    "    device float *ys [[buffer(4)]],device const ulong4 *desc [[buffer(5)]],\n"
    "    constant uint &ndesc [[buffer(6)]],uint3 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    uint r=pos.x,token=pos.y,di=pos.z;if(di>=ndesc)return;\n"
    "    ulong4 d=desc[di*2u],db=desc[di*2u+1u];uint R=uint(db.y),C=uint(db.z),B=uint(db.w);\n"
    "    if(r>=R||token>=B)return;device const uchar *vr=vals+d.x,*sr=scales+d.y,*br=biases+db.x;\n"
    "    device const float *xj=x+d.z+ulong(token)*C;device float *yj=ys+d.w+ulong(token)*R;\n"
    "    float4 a0=float4(0.0f),a1=float4(0.0f),a2=float4(0.0f),a3=float4(0.0f);\n"
    "    float4 a4=float4(0.0f),a5=float4(0.0f),a6=float4(0.0f),a7=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){uint k=r*C+c,g=k/64u;float s=as_type<float>(uint(load16(sr+g*2u))<<16);\n"
    "        float b=as_type<float>(uint(load16(br+g*2u))<<16);ulong word=(k>>3u)*4u;\n"
    "        uint u0=load32(vr+word),u1=load32(vr+word+4u),u2=load32(vr+word+8u),u3=load32(vr+word+12u);\n"
    "        float4 q0=float4((u0>>0u)&15u,(u0>>4u)&15u,(u0>>8u)&15u,(u0>>12u)&15u);\n"
    "        float4 q1=float4((u0>>16u)&15u,(u0>>20u)&15u,(u0>>24u)&15u,(u0>>28u)&15u);\n"
    "        float4 q2=float4((u1>>0u)&15u,(u1>>4u)&15u,(u1>>8u)&15u,(u1>>12u)&15u);\n"
    "        float4 q3=float4((u1>>16u)&15u,(u1>>20u)&15u,(u1>>24u)&15u,(u1>>28u)&15u);\n"
    "        float4 q4=float4((u2>>0u)&15u,(u2>>4u)&15u,(u2>>8u)&15u,(u2>>12u)&15u);\n"
    "        float4 q5=float4((u2>>16u)&15u,(u2>>20u)&15u,(u2>>24u)&15u,(u2>>28u)&15u);\n"
    "        float4 q6=float4((u3>>0u)&15u,(u3>>4u)&15u,(u3>>8u)&15u,(u3>>12u)&15u);\n"
    "        float4 q7=float4((u3>>16u)&15u,(u3>>20u)&15u,(u3>>24u)&15u,(u3>>28u)&15u);\n"
    "        float4 w0=q0*s;w0+=b;float4 w1=q1*s;w1+=b;float4 w2=q2*s;w2+=b;float4 w3=q3*s;w3+=b;\n"
    "        float4 w4=q4*s;w4+=b;float4 w5=q5*s;w5+=b;float4 w6=q6*s;w6+=b;float4 w7=q7*s;w7+=b;\n"
    "        float4 p0=w0*float4(xj[c],xj[c+1u],xj[c+2u],xj[c+3u]);\n"
    "        float4 p1=w1*float4(xj[c+4u],xj[c+5u],xj[c+6u],xj[c+7u]);\n"
    "        float4 p2=w2*float4(xj[c+8u],xj[c+9u],xj[c+10u],xj[c+11u]);\n"
    "        float4 p3=w3*float4(xj[c+12u],xj[c+13u],xj[c+14u],xj[c+15u]);\n"
    "        float4 p4=w4*float4(xj[c+16u],xj[c+17u],xj[c+18u],xj[c+19u]);\n"
    "        float4 p5=w5*float4(xj[c+20u],xj[c+21u],xj[c+22u],xj[c+23u]);\n"
    "        float4 p6=w6*float4(xj[c+24u],xj[c+25u],xj[c+26u],xj[c+27u]);\n"
    "        float4 p7=w7*float4(xj[c+28u],xj[c+29u],xj[c+30u],xj[c+31u]);\n"
    "        a0+=p0;a1+=p1;a2+=p2;a3+=p3;a4+=p4;a5+=p5;a6+=p6;a7+=p7;}\n"
    "    float4 s4=float4(0.0f);s4+=a0;s4+=a1;s4+=a2;s4+=a3;s4+=a4;s4+=a5;s4+=a6;s4+=a7;\n"
    "    float2 tt=s4.xy+s4.zw;yj[r]=tt.x+tt.y;\n"
    "}\n"
    "struct SelectedResources { array<device const uchar *,512> resource [[id(0)]]; };\n"
    "kernel void q4selected(\n"
    "    constant SelectedResources &resources [[buffer(0)]],device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]],device const ulong4 *desc [[buffer(3)]],\n"
    "    constant uint &ndesc [[buffer(4)]],device const int *logical_slots [[buffer(5)]],\n"
    "    device const uint *payload_offsets [[buffer(6)]],device const int *selected [[buffer(7)]],\n"
    "    device atomic_uint *status [[buffer(8)]],uint3 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    uint r=pos.x,token=pos.y,di=pos.z;if(di>=ndesc)return;ulong4 d=desc[di*3u],db=desc[di*3u+1u],dc=desc[di*3u+2u];\n"
    "    uint R=uint(db.y),C=uint(db.z),B=uint(db.w);if(r>=R||token>=B)return;uint slot=uint(dc.x);ulong payload=0u;if(dc.y!=0u){uint logical=uint(dc.z)+uint(selected[dc.w+token]);int mapped=logical_slots[logical];if(mapped<0){atomic_store_explicit(status,2u,memory_order_relaxed);return;}slot=uint(mapped);payload=payload_offsets[logical];}\n"
    "    device const uchar *base=resources.resource[slot];device const uchar *vr=base+payload+d.x,*sr=base+payload+d.y,*br=base+payload+db.x;\n"
    "    device const float *xj=x+d.z+ulong(token)*C;device float *yj=ys+d.w+ulong(token)*R;\n"
    "    float4 a0=float4(0.0f),a1=float4(0.0f),a2=float4(0.0f),a3=float4(0.0f);\n"
    "    float4 a4=float4(0.0f),a5=float4(0.0f),a6=float4(0.0f),a7=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){uint k=r*C+c,g=k/64u;float s=as_type<float>(uint(load16(sr+g*2u))<<16);\n"
    "        float b=as_type<float>(uint(load16(br+g*2u))<<16);ulong word=(k>>3u)*4u;\n"
    "        uint u0=load32(vr+word),u1=load32(vr+word+4u),u2=load32(vr+word+8u),u3=load32(vr+word+12u);\n"
    "        float4 q0=float4((u0>>0u)&15u,(u0>>4u)&15u,(u0>>8u)&15u,(u0>>12u)&15u);\n"
    "        float4 q1=float4((u0>>16u)&15u,(u0>>20u)&15u,(u0>>24u)&15u,(u0>>28u)&15u);\n"
    "        float4 q2=float4((u1>>0u)&15u,(u1>>4u)&15u,(u1>>8u)&15u,(u1>>12u)&15u);\n"
    "        float4 q3=float4((u1>>16u)&15u,(u1>>20u)&15u,(u1>>24u)&15u,(u1>>28u)&15u);\n"
    "        float4 q4=float4((u2>>0u)&15u,(u2>>4u)&15u,(u2>>8u)&15u,(u2>>12u)&15u);\n"
    "        float4 q5=float4((u2>>16u)&15u,(u2>>20u)&15u,(u2>>24u)&15u,(u2>>28u)&15u);\n"
    "        float4 q6=float4((u3>>0u)&15u,(u3>>4u)&15u,(u3>>8u)&15u,(u3>>12u)&15u);\n"
    "        float4 q7=float4((u3>>16u)&15u,(u3>>20u)&15u,(u3>>24u)&15u,(u3>>28u)&15u);\n"
    "        float4 w0=q0*s;w0+=b;float4 w1=q1*s;w1+=b;float4 w2=q2*s;w2+=b;float4 w3=q3*s;w3+=b;\n"
    "        float4 w4=q4*s;w4+=b;float4 w5=q5*s;w5+=b;float4 w6=q6*s;w6+=b;float4 w7=q7*s;w7+=b;\n"
    "        float4 p0=w0*float4(xj[c],xj[c+1u],xj[c+2u],xj[c+3u]);\n"
    "        float4 p1=w1*float4(xj[c+4u],xj[c+5u],xj[c+6u],xj[c+7u]);\n"
    "        float4 p2=w2*float4(xj[c+8u],xj[c+9u],xj[c+10u],xj[c+11u]);\n"
    "        float4 p3=w3*float4(xj[c+12u],xj[c+13u],xj[c+14u],xj[c+15u]);\n"
    "        float4 p4=w4*float4(xj[c+16u],xj[c+17u],xj[c+18u],xj[c+19u]);\n"
    "        float4 p5=w5*float4(xj[c+20u],xj[c+21u],xj[c+22u],xj[c+23u]);\n"
    "        float4 p6=w6*float4(xj[c+24u],xj[c+25u],xj[c+26u],xj[c+27u]);\n"
    "        float4 p7=w7*float4(xj[c+28u],xj[c+29u],xj[c+30u],xj[c+31u]);\n"
    "        a0+=p0;a1+=p1;a2+=p2;a3+=p3;a4+=p4;a5+=p5;a6+=p6;a7+=p7;}\n"
    "    float4 s4=float4(0.0f);s4+=a0;s4+=a1;s4+=a2;s4+=a3;s4+=a4;s4+=a5;s4+=a6;s4+=a7;\n"
    "    float2 tt=s4.xy+s4.zw;yj[r]=tt.x+tt.y;\n"
    "}\n"
    "kernel void q4selectedragged(\n"
    "    constant SelectedResources &resources [[buffer(0)]],device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]],device const ulong4 *desc [[buffer(3)]],\n"
    "    constant uint2 &counts [[buffer(4)]],device const int *logical_slots [[buffer(5)]],\n"
    "    device const uint *payload_offsets [[buffer(6)]],device const uint2 *tasks [[buffer(7)]],\n"
    "    device atomic_uint *status [[buffer(8)]],uint3 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    uint r=pos.x,task=pos.y,ndesc=counts.x,ntasks=counts.y;if(task>=ntasks)return;uint2 work=tasks[task];uint di=work.x,token=work.y;if(di>=ndesc)return;ulong4 d=desc[di*3u],db=desc[di*3u+1u],dc=desc[di*3u+2u];\n"
    "    uint R=uint(db.y),C=uint(db.z),B=uint(db.w);if(r>=R||token>=B)return;if(dc.y!=0u){atomic_store_explicit(status,2u,memory_order_relaxed);return;}uint slot=uint(dc.x);ulong payload=0u;\n"
    "    device const uchar *base=resources.resource[slot];device const uchar *vr=base+payload+d.x,*sr=base+payload+d.y,*br=base+payload+db.x;\n"
    "    device const float *xj=x+d.z+ulong(token)*C;device float *yj=ys+d.w+ulong(token)*R;\n"
    "    float4 a0=float4(0.0f),a1=float4(0.0f),a2=float4(0.0f),a3=float4(0.0f);\n"
    "    float4 a4=float4(0.0f),a5=float4(0.0f),a6=float4(0.0f),a7=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){uint k=r*C+c,g=k/64u;float s=as_type<float>(uint(load16(sr+g*2u))<<16);\n"
    "        float b=as_type<float>(uint(load16(br+g*2u))<<16);ulong word=(k>>3u)*4u;\n"
    "        uint u0=load32(vr+word),u1=load32(vr+word+4u),u2=load32(vr+word+8u),u3=load32(vr+word+12u);\n"
    "        float4 q0=float4((u0>>0u)&15u,(u0>>4u)&15u,(u0>>8u)&15u,(u0>>12u)&15u);\n"
    "        float4 q1=float4((u0>>16u)&15u,(u0>>20u)&15u,(u0>>24u)&15u,(u0>>28u)&15u);\n"
    "        float4 q2=float4((u1>>0u)&15u,(u1>>4u)&15u,(u1>>8u)&15u,(u1>>12u)&15u);\n"
    "        float4 q3=float4((u1>>16u)&15u,(u1>>20u)&15u,(u1>>24u)&15u,(u1>>28u)&15u);\n"
    "        float4 q4=float4((u2>>0u)&15u,(u2>>4u)&15u,(u2>>8u)&15u,(u2>>12u)&15u);\n"
    "        float4 q5=float4((u2>>16u)&15u,(u2>>20u)&15u,(u2>>24u)&15u,(u2>>28u)&15u);\n"
    "        float4 q6=float4((u3>>0u)&15u,(u3>>4u)&15u,(u3>>8u)&15u,(u3>>12u)&15u);\n"
    "        float4 q7=float4((u3>>16u)&15u,(u3>>20u)&15u,(u3>>24u)&15u,(u3>>28u)&15u);\n"
    "        float4 w0=q0*s;w0+=b;float4 w1=q1*s;w1+=b;float4 w2=q2*s;w2+=b;float4 w3=q3*s;w3+=b;\n"
    "        float4 w4=q4*s;w4+=b;float4 w5=q5*s;w5+=b;float4 w6=q6*s;w6+=b;float4 w7=q7*s;w7+=b;\n"
    "        float4 p0=w0*float4(xj[c],xj[c+1u],xj[c+2u],xj[c+3u]);\n"
    "        float4 p1=w1*float4(xj[c+4u],xj[c+5u],xj[c+6u],xj[c+7u]);\n"
    "        float4 p2=w2*float4(xj[c+8u],xj[c+9u],xj[c+10u],xj[c+11u]);\n"
    "        float4 p3=w3*float4(xj[c+12u],xj[c+13u],xj[c+14u],xj[c+15u]);\n"
    "        float4 p4=w4*float4(xj[c+16u],xj[c+17u],xj[c+18u],xj[c+19u]);\n"
    "        float4 p5=w5*float4(xj[c+20u],xj[c+21u],xj[c+22u],xj[c+23u]);\n"
    "        float4 p6=w6*float4(xj[c+24u],xj[c+25u],xj[c+26u],xj[c+27u]);\n"
    "        float4 p7=w7*float4(xj[c+28u],xj[c+29u],xj[c+30u],xj[c+31u]);\n"
    "        a0+=p0;a1+=p1;a2+=p2;a3+=p3;a4+=p4;a5+=p5;a6+=p6;a7+=p7;}\n"
    "    float4 s4=float4(0.0f);s4+=a0;s4+=a1;s4+=a2;s4+=a3;s4+=a4;s4+=a5;s4+=a6;s4+=a7;\n"
    "    float2 tt=s4.xy+s4.zw;yj[r]=tt.x+tt.y;\n"
    "}\n"
    "kernel void q4warpbatch(\n"
    "    device const uchar *vals [[buffer(0)]], device const uchar *scales [[buffer(1)]],\n"
    "    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],\n"
    "    device float *ys [[buffer(4)]], device const ulong4 *desc [[buffer(5)]],\n"
    "    constant uint &ndesc [[buffer(6)]], constant uint &simdgroups [[buffer(7)]],\n"
    "    constant uint &token_span [[buffer(8)]], threadgroup float *dequant [[threadgroup(0)]],\n"
    "    uint3 group [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]])\n"
    "{\n"
    "    uint di=group.z; if(di>=ndesc) return; ulong4 da=desc[di*2u],db=desc[di*2u+1u];\n"
    "    uint r=group.x,R=uint(db.y),C=uint(db.z),B=uint(db.w),token_base=group.y*token_span;\n"
    "    if(r>=R || token_base>=B) return; float acc[16]; for(uint i=0u;i<16u;i++)acc[i]=0.0f;\n"
    "    for(uint cb=0u;cb<C;cb+=32u){\n"
    "        if(warp==0u){ uint c=cb+lane; ulong k=ulong(r)*ulong(C)+ulong(c),g=k/64u;\n"
    "            uint packed=load32(vals+da.x+(k>>3u)*4u); uint q=(packed>>((k&7u)*4u))&15u;\n"
    "            float s=as_type<float>(uint(load16(scales+da.y+g*2u))<<16);\n"
    "            float b=as_type<float>(uint(load16(biases+db.x+g*2u))<<16);\n"
    "            float value=float(q)*s; dequant[lane]=value+b; }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup); uint index=0u;\n"
    "        for(uint token=token_base+warp;token<B && token<token_base+token_span;token+=simdgroups,index++){\n"
    "            uint c=cb+lane; float product=dequant[lane]*x[da.z+ulong(token)*ulong(C)+c];\n"
    "            acc[index]+=product; }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    uint base=lane&3u,index=0u;\n"
    "    for(uint token=token_base+warp;token<B && token<token_base+token_span;token+=simdgroups,index++){\n"
    "        float lane_sum=0.0f; for(uint a=0u;a<8u;a++)lane_sum+=simd_shuffle(acc[index],ushort(base+4u*a));\n"
    "        float l0=simd_shuffle(lane_sum,ushort(0)),l1=simd_shuffle(lane_sum,ushort(1));\n"
    "        float l2=simd_shuffle(lane_sum,ushort(2)),l3=simd_shuffle(lane_sum,ushort(3));\n"
    "        float t0=l0+l2,t1=l1+l3; if(lane==0u)ys[da.w+ulong(token)*ulong(R)+r]=t0+t1;\n"
    "    }\n"
    "}\n";

static id<MTLComputePipelineState> _bpso2;
static id<MTLComputePipelineState> _bweight_pso;
static id<MTLComputePipelineState> _bcoalesced_pso;
static id<MTLComputePipelineState> _bselected_pso;
static id<MTLComputePipelineState> _bselected_ragged_pso;
static id<MTLArgumentEncoder> _selected_arg_encoder;
static id<MTLBuffer> _selected_args, _selected_desc;
static id<MTLBuffer> _selected_task_map;
static id<MTLArgumentEncoder> _wave_selected_arg_encoder;
static id<MTLBuffer> _wave_selected_args, _selected_logical_slots;
static id<MTLBuffer> _selected_logical_payload_offsets;
static int32_t *_selected_logical_slot_host;
static uint32_t *_selected_logical_payload_host;
static int _selected_logical_capacity;
typedef struct MetalSelectedCacheResource {
    id<MTLBuffer> map;
    const uint8_t *base;
    size_t nbytes;
    uint64_t logical_resource_id;
} MetalSelectedCacheResource;
static MetalSelectedCacheResource *_selected_cache_resources;
static int _selected_cache_capacity;
static int _selected_retirement_fenced;
static id<MTLComputePipelineState> _btiled_pso;
static uint32_t _tiled_simdgroups, _tiled_token_span, _tiled_max_descriptors;
static uint32_t _weight_stationary_min_b;
static int _tiled_configured;
static id<MTLCommandQueue> _bqueue2;
static id<MTLBuffer> _bx2[2] = {nil, nil}, _by2[2] = {nil, nil};
static size_t _bx2_len[2] = {0, 0}, _by2_len[2] = {0, 0};
/* the trunk's zero-copy residency: the wrapped trunk mmap + its base
 * (the proj-batch's first sight registers offsets, not copies). */
static id<MTLBuffer> _trunk_map = nil;
static const uint8_t *_trunk_base = NULL;
static size_t _trunk_len = 0;
static int _trunk_address_map_reported = 0;
static int _trunk_address_audit_active = 0;
static int _trunk_address_slot_reported = 0;
static int _trunk_address_desc_reported = 0;

static int gpu_address_audit_enabled(void) {
    const char *p = getenv("SALT_GPU_ADDRESS_AUDIT");
    return p && *p && *p != '0';
}

/* arena: one concatenated buffer per component, grown on demand.
 * tensors are cached per vals-pointer; the cache entry holds the
 * byte offsets into the arena. */
typedef struct {
    const void *key;
    SaltGpuResourceRef resource;
    id<MTLBuffer> value_map, scale_map, bias_map;
    uint64_t value_base, scale_base, bias_base;
    size_t pooled_nbytes;
} TensorSlot;
#define BATCH_MAXSLOT 65536        /* 40 layers x 256 experts x 3 tensors */
#define BATCH_ARENA_V (512u << 20)   /* full 256-exp pool: ~453 MB */
#define BATCH_ARENA_S (64u << 20)    /* scales+biases: ~28 MB */

static TensorSlot _tslot[BATCH_MAXSLOT];
static int _tslot_n = 0;
static uint64_t _component_pool_bytes = 0;
static uint64_t _component_pool_peak_bytes = 0;
static int _mapped_only = 0;
static SaltGpuBatchStatsV2 _batch_stats;
static id<MTLBuffer> _arena_v, _arena_s, _arena_b;
static size_t _arena_v_used = 0, _arena_s_used = 0;
static id<MTLBuffer> _desc = nil;
static size_t _desc_len = 0;
static id<MTLBuffer> _q8_x = nil, _q8_y = nil;
#define METAL_SESSION_X_BYTES (16u * 1024u * 1024u)
#define METAL_SESSION_Y_BYTES (16u * 1024u * 1024u)
#define METAL_SESSION_DESC_BYTES (BATCH_MAXSLOT * 64u)
static const uint32_t *_moe_vals[BATCH_MAXSLOT];
static const uint16_t *_moe_scales[BATCH_MAXSLOT];
static const uint16_t *_moe_biases[BATCH_MAXSLOT];
static const float *_moe_xs[BATCH_MAXSLOT];
static float *_moe_ys[BATCH_MAXSLOT];
static const void *_moe_ids[BATCH_MAXSLOT];
static int _moe_rows[BATCH_MAXSLOT];
static int _direct_slots[BATCH_MAXSLOT], _direct_rows[BATCH_MAXSLOT];
static int _q8_slots[BATCH_MAXSLOT];
static int batch_pipeline_prepare(void);
#define METAL_SELECTED_DESC_STRIDE (3u * 32u)
#define METAL_SELECTED_DESC_DOWN_OFFSET \
    (2u * 128u * METAL_SELECTED_DESC_STRIDE)
#define METAL_SELECTED_DESC_BYTES \
    (3u * 128u * METAL_SELECTED_DESC_STRIDE)
#define METAL_SELECTED_GATE_UP_TASK_CAPACITY 2048u
#define METAL_SELECTED_DOWN_TASK_CAPACITY 1024u
#define METAL_SELECTED_TASK_CAPACITY \
    (METAL_SELECTED_GATE_UP_TASK_CAPACITY + \
     METAL_SELECTED_DOWN_TASK_CAPACITY)

typedef struct MetalSelectedTask {
    uint32_t descriptor;
    uint32_t token;
} MetalSelectedTask;

static int metal_selected_resources_prepare(void) {
    if (_selected_arg_encoder && _selected_args && _selected_desc &&
        _selected_task_map && _wave_selected_arg_encoder &&
        _wave_selected_args)
        return 0;
    metal_release(_selected_arg_encoder); _selected_arg_encoder = nil;
    metal_release(_selected_args); _selected_args = nil;
    metal_release(_selected_desc); _selected_desc = nil;
    metal_release(_selected_task_map); _selected_task_map = nil;
    metal_release(_wave_selected_arg_encoder);
    _wave_selected_arg_encoder = nil;
    metal_release(_wave_selected_args); _wave_selected_args = nil;
    @autoreleasepool {
        NSError *error = nil;
        id<MTLLibrary> library = metal_library_for_source(_batch_src2, &error);
        if (!library) return -1;
        id<MTLFunction> function =
            [library newFunctionWithName:@"q4selected"];
        [library release];
        if (!function) return -1;
        id<MTLArgumentEncoder> encoder =
            [function newArgumentEncoderWithBufferIndex:0];
        id<MTLArgumentEncoder> wave_encoder =
            [function newArgumentEncoderWithBufferIndex:0];
        [function release];
        id<MTLBuffer> arguments = encoder
            ? [_dev newBufferWithLength:encoder.encodedLength
                                 options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> wave_arguments = wave_encoder
            ? [_dev newBufferWithLength:wave_encoder.encodedLength
                                 options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> descriptors = [_dev newBufferWithLength:
            METAL_SELECTED_DESC_BYTES options:MTLResourceStorageModeShared];
        id<MTLBuffer> tasks = [_dev newBufferWithLength:
            (NSUInteger)METAL_SELECTED_TASK_CAPACITY *
                sizeof(MetalSelectedTask)
            options:MTLResourceStorageModeShared];
        if (!encoder || !wave_encoder || !arguments || !wave_arguments ||
            !descriptors || !tasks) {
            metal_release(encoder);
            metal_release(wave_encoder);
            metal_release(arguments);
            metal_release(wave_arguments);
            metal_release(descriptors);
            metal_release(tasks);
            return -1;
        }
        _selected_arg_encoder = encoder;
        _selected_args = arguments;
        _selected_desc = descriptors;
        _selected_task_map = tasks;
        _wave_selected_arg_encoder = wave_encoder;
        _wave_selected_args = wave_arguments;
        [_selected_arg_encoder setArgumentBuffer:_selected_args offset:0];
        [_wave_selected_arg_encoder setArgumentBuffer:
            _wave_selected_args offset:0];
    }
    return 0;
}

int salt_gpu_selected_resources_prepare(int capacity, int logical_capacity) {
    if (!_dev || capacity < 1 || capacity > 512 || logical_capacity < 1 ||
        logical_capacity > INT32_MAX || batch_pipeline_prepare() != 0)
        return -1;
    if (_selected_cache_resources)
        return capacity == _selected_cache_capacity &&
            logical_capacity == _selected_logical_capacity ? 0 : -1;
    if ((size_t)capacity > SIZE_MAX / sizeof *_selected_cache_resources ||
        (size_t)logical_capacity > NSUIntegerMax / sizeof(int32_t))
        return -1;
    _selected_cache_resources = (MetalSelectedCacheResource *)calloc(
        (size_t)capacity, sizeof *_selected_cache_resources);
    _selected_logical_slots = [_dev newBufferWithLength:
        (NSUInteger)logical_capacity * sizeof(int32_t)
        options:MTLResourceStorageModeShared];
    _selected_logical_payload_offsets = [_dev newBufferWithLength:
        (NSUInteger)logical_capacity * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    if (!_selected_cache_resources || !_selected_logical_slots ||
        !_selected_logical_payload_offsets || ![_selected_logical_slots contents] ||
        ![_selected_logical_payload_offsets contents]) {
        free(_selected_cache_resources);
        _selected_cache_resources = NULL;
        metal_release(_selected_logical_slots);
        _selected_logical_slots = nil;
        metal_release(_selected_logical_payload_offsets);
        _selected_logical_payload_offsets = nil;
        return -1;
    }
    _selected_logical_slot_host =
        (int32_t *)[_selected_logical_slots contents];
    _selected_logical_payload_host =
        (uint32_t *)[_selected_logical_payload_offsets contents];
    for (int logical = 0; logical < logical_capacity; logical++) {
        _selected_logical_slot_host[logical] = -1;
        _selected_logical_payload_host[logical] = 0;
    }
    _selected_cache_capacity = capacity;
    _selected_logical_capacity = logical_capacity;
    _selected_retirement_fenced = 0;
    return 0;
}

int salt_gpu_selected_resource_bind(int slot, uint64_t logical_resource_id,
                                    const void *base, size_t nbytes,
                                    const void *payload) {
    id<MTLBuffer> map = nil;
    uintptr_t base_value, payload_value;
    if (!_dev || !_selected_cache_resources || slot < 0 ||
        slot >= _selected_cache_capacity ||
        logical_resource_id >= (uint64_t)(uint32_t)_selected_logical_capacity ||
        !base || !payload || nbytes < 1 || !_wave_selected_arg_encoder ||
        !_selected_logical_slot_host || !_selected_logical_payload_host)
        return -1;
    base_value = (uintptr_t)base;
    payload_value = (uintptr_t)payload;
    if (payload_value < base_value || payload_value - base_value > nbytes ||
        payload_value - base_value > UINT32_MAX)
        return -1;
    MetalSelectedCacheResource *resource = &_selected_cache_resources[slot];
    if (resource->map)
        return resource->base == base && resource->nbytes == nbytes &&
            resource->logical_resource_id == logical_resource_id &&
            _selected_logical_slot_host[logical_resource_id] == slot &&
            _selected_logical_payload_host[logical_resource_id] ==
                (uint32_t)(payload_value - base_value) ? 0 : -1;
    if (_selected_logical_slot_host[logical_resource_id] != -1)
        return -1;
    @autoreleasepool {
        map = [_dev newBufferWithBytesNoCopy:(void *)base length:nbytes
            options:MTLResourceStorageModeShared deallocator:nil];
        if (!map)
            map = [_dev newBufferWithBytesNoCopy:(void *)base length:nbytes
                options:MTLResourceStorageModeManaged deallocator:nil];
        if (!map || [map contents] != base) {
            metal_release(map);
            return -1;
        }
    }
    resource->map = map;
    resource->base = (const uint8_t *)base;
    resource->nbytes = nbytes;
    resource->logical_resource_id = logical_resource_id;
    [_wave_selected_arg_encoder setBuffer:map offset:0
        atIndex:(NSUInteger)slot];

    _selected_logical_slot_host[logical_resource_id] = slot;
    _selected_logical_payload_host[logical_resource_id] =
        (uint32_t)(payload_value - base_value);
    _selected_retirement_fenced = 0;
    return 0;
}

int salt_gpu_selected_resource_bind_resident(
        int slot, uint64_t logical_resource_id,
        const void *payload, size_t nbytes,
        uintptr_t device_address, uint64_t generation) {
    (void)slot; (void)logical_resource_id; (void)payload; (void)nbytes;
    (void)device_address; (void)generation;
    return -1;
}

int salt_gpu_selected_resources_fence(void) {
    if (salt_gpu_sync() != 0) return -1;
    _selected_retirement_fenced = 1;
    return 0;
}

int salt_gpu_selected_resource_unbind(int slot,
                                      uint64_t logical_resource_id) {
    if (!_selected_cache_resources || slot < 0 ||
        slot >= _selected_cache_capacity ||
        logical_resource_id >= (uint64_t)(uint32_t)_selected_logical_capacity ||
        !_selected_logical_slot_host || !_selected_logical_payload_host)
        return -1;
    MetalSelectedCacheResource *resource = &_selected_cache_resources[slot];
    if (!resource->map) return 0;
    if (resource->logical_resource_id != logical_resource_id ||
        _selected_logical_slot_host[logical_resource_id] != slot ||
        (!_selected_retirement_fenced && salt_gpu_sync() != 0))
        return -1;
    _selected_logical_slot_host[logical_resource_id] = -1;
    _selected_logical_payload_host[logical_resource_id] = 0;
    [_wave_selected_arg_encoder setBuffer:nil offset:0
        atIndex:(NSUInteger)slot];
    metal_release(resource->map);
    memset(resource, 0, sizeof *resource);
    return 0;
}

static int metal_session_buffers_prepare(void) {
    if (_bx2[0] && _bx2[1] && _by2[0] && _by2[1] && _desc &&
        _q8_x && _q8_y) return 0;
    for (int i = 0; i < 2; i++) {
        if (!_bx2[i])
            _bx2[i] = [_dev newBufferWithLength:METAL_SESSION_X_BYTES
                                         options:MTLResourceStorageModeShared];
        if (!_by2[i])
            _by2[i] = [_dev newBufferWithLength:METAL_SESSION_Y_BYTES
                                         options:MTLResourceStorageModeShared];
        if (!_bx2[i] || !_by2[i]) return -1;
        _bx2_len[i] = METAL_SESSION_X_BYTES;
        _by2_len[i] = METAL_SESSION_Y_BYTES;
    }
    if (!_desc)
        _desc = [_dev newBufferWithLength:METAL_SESSION_DESC_BYTES
                                   options:MTLResourceStorageModeShared];
    if (!_q8_x)
        _q8_x = [_dev newBufferWithLength:METAL_SESSION_X_BYTES
                                   options:MTLResourceStorageModeShared];
    if (!_q8_y)
        _q8_y = [_dev newBufferWithLength:METAL_SESSION_Y_BYTES
                                   options:MTLResourceStorageModeShared];
    if (!_desc || !_q8_x || !_q8_y ||
        (_offline_library && metal_selected_resources_prepare() != 0))
        return -1;
    _desc_len = METAL_SESSION_DESC_BYTES;
    return 0;
}

static int mapped_range_offset(const uint8_t *base, size_t map_len,
                               const void *ptr, size_t nbytes,
                               uint64_t *offset) {
    uintptr_t b, p;
    uint64_t delta;
    if (!base || !ptr || !offset) return -1;
    b = (uintptr_t)(const void *)base;
    p = (uintptr_t)ptr;
    if (p < b) return -1;
    delta = (uint64_t)(p - b);
    if (delta > map_len || nbytes > map_len - (size_t)delta) return -1;
    *offset = delta;
    return 0;
}

static int batch_weight_sizes(int R, int C, size_t *vbytes,
                              size_t *sbytes) {
    uint64_t elements, packed_words, groups;
    if (R < 1 || C < 1 || !vbytes || !sbytes) return -1;
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    packed_words = (elements + 7u) / 8u;
    groups = (elements + 63u) / 64u;
    if (packed_words > SIZE_MAX / sizeof(uint32_t) ||
        groups > SIZE_MAX / sizeof(uint16_t)) return -1;
    *vbytes = (size_t)packed_words * sizeof(uint32_t);
    *sbytes = (size_t)groups * sizeof(uint16_t);
    return 0;
}

static int tensor_slot_store(const void *key, uint32_t kind,
                             uint32_t resource_id, uint64_t voff,
                             uint64_t soff, uint64_t boff) {
    TensorSlot *slot;
    if (!key || _tslot_n >= BATCH_MAXSLOT) return -1;
    slot = &_tslot[_tslot_n];
    memset(slot, 0, sizeof *slot);
    slot->key = key;
    slot->resource.kind = kind;
    slot->resource.resource_id = resource_id;
    slot->resource.value_offset = voff;
    slot->resource.scale_offset = soff;
    slot->resource.bias_offset = boff;
    return _tslot_n++;
}

static void tensor_slots_forget(uint32_t kind, uint32_t resource_id) {
    int out = 0;
    for (int i = 0; i < _tslot_n; i++) {
        const SaltGpuResourceRef *ref = &_tslot[i].resource;
        if (ref->kind == kind && ref->resource_id == resource_id) {
            metal_release(_tslot[i].value_map);
            metal_release(_tslot[i].scale_map);
            metal_release(_tslot[i].bias_map);
            if (_tslot[i].pooled_nbytes <= _component_pool_bytes)
                _component_pool_bytes -= _tslot[i].pooled_nbytes;
            continue;
        }
        if (out != i) _tslot[out] = _tslot[i];
        out++;
    }
    if (out < _tslot_n)
        memset(_tslot + out, 0,
               (size_t)(_tslot_n - out) * sizeof _tslot[0]);
    _tslot_n = out;
}

static int batch_ensure_arena(size_t vneed, size_t sneed) {
    size_t vtotal, stotal;
    if (vneed > SIZE_MAX - _arena_v_used ||
        sneed > SIZE_MAX - _arena_s_used)
        return -1;
    vtotal = _arena_v_used + vneed;
    stotal = _arena_s_used + sneed;
    if (!_arena_v) {
        id<MTLBuffer> av = [_dev newBufferWithLength:BATCH_ARENA_V
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> as = [_dev newBufferWithLength:BATCH_ARENA_S
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> ab = [_dev newBufferWithLength:BATCH_ARENA_S
                                             options:MTLResourceStorageModeShared];
        if (!av || !as || !ab) {
            metal_release(av); metal_release(as); metal_release(ab);
            return -1;
        }
        _arena_v = av; _arena_s = as; _arena_b = ab;
    }
    /* grow on demand: new buffer, copy, swap (Metal buffers are
     * immutable-sized; reallocate like realloc). */
    if (vtotal > _arena_v.length) {
        size_t nlen = _arena_v.length;
        while (nlen < vtotal) {
            if (nlen > SIZE_MAX / 2) return -1;
            nlen *= 2;
        }
        id<MTLBuffer> nv = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        if (!nv) return -1;
        memcpy(nv.contents, _arena_v.contents, _arena_v_used);
        metal_release(_arena_v);
        _arena_v = nv;
    }
    if (stotal > _arena_s.length) {
        size_t nlen = _arena_s.length;
        while (nlen < stotal) {
            if (nlen > SIZE_MAX / 2) return -1;
            nlen *= 2;
        }
        id<MTLBuffer> ns = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> nb = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        if (!ns || !nb) {
            metal_release(ns); metal_release(nb);
            return -1;
        }
        memcpy(ns.contents, _arena_s.contents, _arena_s_used);
        memcpy(nb.contents, _arena_b.contents, _arena_s_used);
        metal_release(_arena_s); metal_release(_arena_b);
        _arena_s = ns; _arena_b = nb;
    }
    return 0;
}

/* find or cache a tensor by STABLE id (expert layout pointer). The
 * vals pointer alone is not stable (cache slots rotate per token).
 * Returns the slot index, -1 on failure. */
static int batch_slot(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, const void *id, int R, int C) {
    size_t vbytes, sbytes;
    for (int i = 0; i < _tslot_n; i++) {
        if (_tslot[i].key != id) continue;
        if (_mapped_only &&
            _tslot[i].resource.kind == SALT_GPU_RESOURCE_ARENA) {
            _batch_stats.mapped_only_misses++;
            return -1;
        }
        return i;
    }
    if (_tslot_n >= BATCH_MAXSLOT) return -1;
    if (batch_weight_sizes(R, C, &vbytes, &sbytes) != 0) return -1;
    /* the trunk's zero-copy residency: the tensor's offset into the
     * wrapped trunk mmap -- NO arena copy, no duplicated RSS (the
     * vLLM-style residency: the weights live in the unified memory
     * once, the GPU references them). */
    if (_trunk_map && _trunk_base) {
        uint64_t voff, soff, boff;
        if (biases &&
            mapped_range_offset(_trunk_base, _trunk_len, vals, vbytes,
                                &voff) == 0 &&
            mapped_range_offset(_trunk_base, _trunk_len, scales, sbytes,
                                &soff) == 0 &&
            mapped_range_offset(_trunk_base, _trunk_len, biases, sbytes,
                                &boff) == 0) {
            int stored = tensor_slot_store(id, SALT_GPU_RESOURCE_TRUNK, 0,
                                           voff, soff, boff);
            if (stored >= 0 && _trunk_address_audit_active &&
                !_trunk_address_slot_reported) {
                fprintf(stderr,
                        "[gpu-address] slot sample=1 key=%p resource=trunk "
                        "host_base=%p host_v=%p host_s=%p host_b=%p "
                        "rel_v_bytes=%llu rel_s_bytes=%llu "
                        "rel_b_bytes=%llu\n",
                        id, (const void *)_trunk_base, (const void *)vals,
                        (const void *)scales, (const void *)biases,
                        (unsigned long long)voff,
                        (unsigned long long)soff,
                        (unsigned long long)boff);
                _trunk_address_slot_reported = 1;
            }
            return stored;
        }
    }
    if (_mapped_only) {
        _batch_stats.mapped_only_misses++;
        return -1;
    }
    if (batch_ensure_arena(vbytes, sbytes) != 0) return -1;
    memcpy((char *)_arena_v.contents + _arena_v_used, vals, vbytes);
    memcpy((char *)_arena_s.contents + _arena_s_used, scales, sbytes);
    if (biases)
        memcpy((char *)_arena_b.contents + _arena_s_used, biases, sbytes);
    else
        memset((char *)_arena_b.contents + _arena_s_used, 0, sbytes);
    int stored = tensor_slot_store(id, SALT_GPU_RESOURCE_ARENA, 0,
                                   _arena_v_used, _arena_s_used,
                                   _arena_s_used);
    if (stored < 0) return -1;
    _arena_v_used += vbytes;
    _arena_s_used += sbytes;
    return stored;
}

/* Bounded source registration pre-populates slots so first sight is a hit.
 * The host lease remains authoritative and live through synchronization. */
static id<MTLBuffer> _res_map = nil;   /* zero-copy pool wrap */
static const uint8_t *_res_base = NULL;
static size_t _res_len = 0;
static uint32_t _res_id = 0;

typedef struct {
    int used, active;
    int component_pool;
    uint32_t kind, resource_id;
    SaltGpuWeightAddressability policy;
    id<MTLBuffer> map;
    const uint8_t *base;
    size_t nbytes;
    uint64_t map_offset;
    size_t map_nbytes;
} MetalWeightResource;
#define METAL_WEIGHT_RESOURCES 16
static MetalWeightResource _weight_resources[METAL_WEIGHT_RESOURCES];
static uint64_t _weight_peak_window_bytes;

static MetalWeightResource *metal_weight_resource(uint32_t kind,
                                                   uint32_t resource_id) {
    for (int i = 0; i < METAL_WEIGHT_RESOURCES; i++)
        if (_weight_resources[i].used &&
            _weight_resources[i].kind == kind &&
            _weight_resources[i].resource_id == resource_id)
            return &_weight_resources[i];
    return NULL;
}

int salt_gpu_resident_trunk_map(const void *base, size_t nbytes) {
    if (!_dev || !base || nbytes < 1 || _trunk_map || _trunk_base ||
        _trunk_len)
        return -1;
    @autoreleasepool {
        _trunk_map = [_dev newBufferWithBytesNoCopy:(void *)base
                                              length:nbytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
        if (!_trunk_map) {
            _trunk_map = [_dev newBufferWithBytesNoCopy:(void *)base
                                                  length:nbytes
                                                 options:MTLResourceStorageModeManaged
                                             deallocator:nil];
        }
        if (!_trunk_map) return -1;
        _trunk_base = (const uint8_t *)base;
        _trunk_len = nbytes;
        if (gpu_address_audit_enabled() && !_trunk_address_map_reported) {
            uint64_t gpu_address = 0;
            int gpu_address_available =
                metal_buffer_gpu_address(_trunk_map, &gpu_address);
            fprintf(stderr,
                    "[gpu-address] map sample=1 host_base=%p "
                    "buffer_contents=%p bytes=%zu storage_mode=%lu "
                    "gpu_base=0x%llx gpu_address_available=%d no_copy=%d\n",
                    base, [_trunk_map contents], nbytes,
                    (unsigned long)[_trunk_map storageMode],
                    (unsigned long long)gpu_address,
                    gpu_address_available,
                    [_trunk_map contents] == base);
            _trunk_address_map_reported = 1;
            _trunk_address_audit_active = 1;
            _trunk_address_slot_reported = 0;
            _trunk_address_desc_reported = 0;
        }
    }
    return 0;
}

int salt_gpu_resident_trunk_unmap(void) {
    id<MTLBuffer> map;
    if (!_trunk_map && !_trunk_base && !_trunk_len) return 0;
    if (!_trunk_map || !_trunk_base || !_trunk_len ||
        salt_gpu_sync() != 0)
        return -1;
    tensor_slots_forget(SALT_GPU_RESOURCE_TRUNK, 0);
    map = _trunk_map;
    _trunk_map = nil;
    _trunk_base = NULL;
    _trunk_len = 0;
    _trunk_address_audit_active = 0;
    [map release];
    return 0;
}

int salt_gpu_expert_resource_bind(uint32_t resource_id,
                                  const void *base, size_t nbytes) {
    id<MTLBuffer> map = nil;
    if (!_dev || !base || nbytes < 1 || _res_map || _res_base || _res_len)
        return -1;
    @autoreleasepool {
        map = [_dev newBufferWithBytesNoCopy:(void *)base
                                      length:nbytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (!map) {
            /* the shared-mode no-copy can fail on mmap'd file-backed
             * pages (the GPU's access attributes aren't guaranteed).
             * Retry MANAGED -- still no copy, the GPU's writes (none
             * here -- the weights are read-only) sync on demand. */
            map = [_dev newBufferWithBytesNoCopy:(void *)base
                                          length:nbytes
                                         options:MTLResourceStorageModeManaged
                                     deallocator:nil];
        }
        if (!map) return -1;
        _res_map = map;
        _res_base = (const uint8_t *)base;
        _res_len = nbytes;
        _res_id = resource_id;
    }
    return 0;
}

int salt_gpu_expert_resource_unbind(uint32_t resource_id) {
    id<MTLBuffer> map;
    if (!_res_map || !_res_base || !_res_len || resource_id != _res_id)
        return -1;
    if (salt_gpu_sync() != 0) return -1;
    tensor_slots_forget(SALT_GPU_RESOURCE_EXPERT_LAYER, resource_id);
    map = _res_map;
    _res_map = nil;
    _res_base = NULL;
    _res_len = 0;
    _res_id = 0;
    [map release];
    return 0;
}

int salt_gpu_resident_pool_map(const void *base, size_t nbytes) {
    if (salt_gpu_expert_resource_bind(0, base, nbytes) != 0) {
        fprintf(stderr, "gpu: FATAL -- pool mmap no-copy wrap FAILED "
                "(%zu MB); resident mode disabled.\n", nbytes >> 20);
        return -1;
    }
    fprintf(stderr, "gpu: pool mmap wrapped zero-copy (%zu MB)\n",
            nbytes >> 20);
    return 0;
}

int salt_gpu_expert_resource_slot(uint32_t resource_id, const void *key,
                                  const uint32_t *vals,
                                  const uint16_t *scales,
                                  const uint16_t *biases, int R, int C) {
    size_t vbytes, sbytes;
    uint64_t voff, soff, boff;
    if (!_dev || !key || !vals || !scales || !biases || R < 1 || C < 1)
        return -1;
    if (_tslot_n >= BATCH_MAXSLOT) return -1;
    if (_res_map && _res_base && resource_id == _res_id) {
        /* zero-copy: the slot holds offsets INTO the wrapped mmap --
         * no arena copy, no extra RSS. The batch kernel reads the
         * map buffer directly. NOTE the biases live at rel_b in the
         * pool (NOT at the scales' soff -- the arena stored them
         * contiguously at soff, but the pool does not), so the bias
         * offset is tracked separately. */
        if (batch_weight_sizes(R, C, &vbytes, &sbytes) != 0 ||
            mapped_range_offset(_res_base, _res_len, vals, vbytes,
                                &voff) != 0 ||
            mapped_range_offset(_res_base, _res_len, scales, sbytes,
                                &soff) != 0 ||
            mapped_range_offset(_res_base, _res_len, biases, sbytes,
                                &boff) != 0)
            return -1;
        for (int i = 0; i < _tslot_n; i++) {
            const SaltGpuResourceRef *ref = &_tslot[i].resource;
            if (_tslot[i].key != key) continue;
            return ref->kind == SALT_GPU_RESOURCE_EXPERT_LAYER &&
                   ref->resource_id == resource_id &&
                   ref->value_offset == voff &&
                   ref->scale_offset == soff &&
                   ref->bias_offset == boff ? 0 : -1;
        }
        return tensor_slot_store(key, SALT_GPU_RESOURCE_EXPERT_LAYER,
                                 resource_id, voff, soff, boff) < 0 ? -1 : 0;
    }
    return -1;
}

int salt_gpu_resident_slot(const void *key, const uint32_t *vals,
                           const uint16_t *scales, const uint16_t *biases,
                           int R, int C) {
    if (_res_map && _res_base)
        return salt_gpu_expert_resource_slot(_res_id, key, vals, scales,
                                             biases, R, C);
    return -1;
}

typedef struct { uint64_t voff, soff, xoff, yoff; } BatchDesc4;
typedef struct { BatchDesc4 a, b, c; } SelectedDesc;
static int metal_selected_encode(id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> resource_arguments, id<MTLBuffer> input_buffer,
    id<MTLBuffer> output_buffer,
    NSUInteger descriptor_offset, int descriptor_count,
    int max_rows, int max_batch);

static int batch_resource_for_slots(const int *slots, int nslots,
                                    const SaltGpuResourceRef **resource) {
    const SaltGpuResourceRef *first;
    if (!slots || nslots < 1 || !resource || slots[0] < 0 ||
        slots[0] >= _tslot_n) return -1;
    first = &_tslot[slots[0]].resource;
    if (!salt_gpu_resource_ref_valid(first)) return -1;
    for (int j = 1; j < nslots; j++) {
        if (slots[j] < 0 || slots[j] >= _tslot_n ||
            !salt_gpu_resource_same(first, &_tslot[slots[j]].resource))
            return -1;
    }
    *resource = first;
    return 0;
}

static int batch_build_desc(BatchDesc4 *desc, const int *slots,
                            const int *rows, int C, int njobs,
                            int shared_x, uint64_t resource_base) {
    uint64_t row_prefix = 0;
    if (!desc || !slots || !rows || C < 1 || njobs < 1) return -1;
    for (int j = 0; j < njobs; j++) {
        const SaltGpuResourceRef *ref;
        uint64_t xoff;
        uint64_t voff, soff, boff;
        if (slots[j] < 0 || slots[j] >= _tslot_n || rows[j] < 1)
            return -1;
        ref = &_tslot[slots[j]].resource;
        xoff = shared_x ? 0 : (uint64_t)(uint32_t)j * (uint32_t)C;
        if ((uint64_t)(uint32_t)rows[j] > UINT64_MAX - row_prefix) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "[gpu-fail] desc64 j=%d voff=%llu soff=%llu boff=%llu "
                    "rows=%d prefix=%llu mod=%llu/%llu/%llu\n",
                    j, (unsigned long long)ref->value_offset,
                    (unsigned long long)ref->scale_offset,
                    (unsigned long long)ref->bias_offset, rows[j],
                    (unsigned long long)row_prefix,
                    (unsigned long long)(ref->value_offset % 4u),
                    (unsigned long long)(ref->scale_offset % 2u),
                    (unsigned long long)(ref->bias_offset % 2u));
            return -1;
        }
        if (ref->value_offset < resource_base ||
            ref->scale_offset < resource_base ||
            ref->bias_offset < resource_base) return -1;
        voff = ref->value_offset - resource_base;
        soff = ref->scale_offset - resource_base;
        boff = ref->bias_offset - resource_base;
        desc[2 * j].voff = voff;
        desc[2 * j].soff = soff;
        desc[2 * j].xoff = xoff;
        desc[2 * j].yoff = row_prefix;
        desc[2 * j + 1].voff = boff;
        desc[2 * j + 1].soff = 0;
        desc[2 * j + 1].xoff = 0;
        desc[2 * j + 1].yoff = row_prefix;
        row_prefix += (uint32_t)rows[j];
    }
    return 0;
}

static int batch_build_component_desc(
    BatchDesc4 *desc, int slot, int R, int C, int njobs,
    uint64_t value_base, uint64_t scale_base, uint64_t bias_base) {
    const SaltGpuResourceRef *ref;
    if (!desc || slot < 0 || slot >= _tslot_n || R < 1 || C < 1 ||
        njobs < 1)
        return -1;
    ref = &_tslot[slot].resource;
    if (ref->value_offset < value_base || ref->scale_offset < scale_base ||
        ref->bias_offset < bias_base)
        return -1;
    for (int job = 0; job < njobs; job++) {
        uint64_t xoff = (uint64_t)(uint32_t)job * (uint32_t)C;
        uint64_t yoff = (uint64_t)(uint32_t)job * (uint32_t)R;
        desc[2 * job] = (BatchDesc4) {
            ref->value_offset - value_base,
            ref->scale_offset - scale_base,
            xoff,
            yoff,
        };
        desc[2 * job + 1] = (BatchDesc4) {
            ref->bias_offset - bias_base,
            0,
            0,
            yoff,
        };
    }
    return 0;
}

static int batch_build_tiled_desc(BatchDesc4 *desc, const int *slots,
                                  const int *rows, int C, int njobs,
                                  uint64_t resource_base,
                                  int *descriptor_count_out,
                                  int *max_rows_out, int *max_batch_out) {
    uint64_t output_prefix = 0;
    int descriptor_count = 0, max_rows = 0, max_batch = 0;
    if (!desc || !slots || !rows || C < 1 || njobs < 1 ||
        !descriptor_count_out || !max_rows_out || !max_batch_out)
        return -1;
    if (_tiled_max_descriptors == 0 || njobs < 2 || (C & 31) != 0) return 1;
    for (int job = 0; job < njobs;) {
        const SaltGpuResourceRef *ref;
        int run = 1;
        if (slots[job] < 0 || slots[job] >= _tslot_n || rows[job] < 1)
            return -1;
        ref = &_tslot[slots[job]].resource;
        while (job + run < njobs && slots[job + run] == slots[job] &&
               rows[job + run] == rows[job]) run++;
        if (descriptor_count >= BATCH_MAXSLOT ||
            ref->value_offset < resource_base ||
            ref->scale_offset < resource_base ||
            ref->bias_offset < resource_base ||
            (uint64_t)(uint32_t)run * (uint32_t)rows[job] >
                UINT64_MAX - output_prefix)
            return -1;
        desc[2 * descriptor_count] = (BatchDesc4) {
            ref->value_offset - resource_base,
            ref->scale_offset - resource_base,
            (uint64_t)(uint32_t)job * (uint32_t)C,
            output_prefix,
        };
        desc[2 * descriptor_count + 1] = (BatchDesc4) {
            ref->bias_offset - resource_base,
            (uint64_t)(uint32_t)rows[job],
            (uint64_t)(uint32_t)C,
            (uint64_t)(uint32_t)run,
        };
        output_prefix += (uint64_t)(uint32_t)run * (uint32_t)rows[job];
        if (rows[job] > max_rows) max_rows = rows[job];
        if (run > max_batch) max_batch = run;
        descriptor_count++;
        job += run;
    }
    if (descriptor_count >= njobs ||
        (uint32_t)descriptor_count > _tiled_max_descriptors || max_batch < 2)
        return 1;
    *descriptor_count_out = descriptor_count;
    *max_rows_out = max_rows;
    *max_batch_out = max_batch;
    return 0;
}

static int metal_q4_operation_view(const SaltGpuResourceRef *resource,
                                   const int *slots, const int *rows,
                                   int C, int nslots,
                                   id<MTLBuffer> *view_out,
                                   uint64_t *base_offset_out) {
    MetalWeightResource *generic;
    uint64_t lo = UINT64_MAX, hi = 0;
    long page_l;
    if (!resource || !slots || !rows || nslots < 1 || C < 1 ||
        !view_out || !base_offset_out) return -1;
    *view_out = nil;
    *base_offset_out = 0;
    generic = metal_weight_resource(resource->kind, resource->resource_id);
    if (!generic) return 0;
    if (!generic->active) return -1;
    for (int j = 0; j < nslots; j++) {
        size_t vbytes, sbytes;
        const SaltGpuResourceRef *ref;
        uint64_t ends[3];
        if (slots[j] < 0 || slots[j] >= _tslot_n || rows[j] < 1 ||
            batch_weight_sizes(rows[j], C, &vbytes, &sbytes) != 0)
            return -1;
        ref = &_tslot[slots[j]].resource;
        if (!salt_gpu_resource_same(resource, ref) ||
            ref->value_offset > UINT64_MAX - vbytes ||
            ref->scale_offset > UINT64_MAX - sbytes ||
            ref->bias_offset > UINT64_MAX - sbytes)
            return -1;
        if (ref->value_offset < lo) lo = ref->value_offset;
        if (ref->scale_offset < lo) lo = ref->scale_offset;
        if (ref->bias_offset < lo) lo = ref->bias_offset;
        ends[0] = ref->value_offset + vbytes;
        ends[1] = ref->scale_offset + sbytes;
        ends[2] = ref->bias_offset + sbytes;
        for (int i = 0; i < 3; i++) if (ends[i] > hi) hi = ends[i];
    }
    if (lo == UINT64_MAX || hi <= lo || hi > generic->nbytes)
        return -1;
    if (generic->map) {
        if (lo < generic->map_offset ||
            generic->map_nbytes > UINT64_MAX - generic->map_offset ||
            hi > generic->map_offset + generic->map_nbytes)
            return -1;
        *view_out = [generic->map retain];
        *base_offset_out = generic->map_offset;
        return 0;
    }
    if ((page_l = sysconf(_SC_PAGESIZE)) <= 0) return -1;
    uint64_t page = (uint64_t)page_l;
    uint64_t aligned = lo - lo % page;
    uint64_t length = hi - aligned;
    if (length > (uint64_t)NSUIntegerMax) return -1;
    id<MTLBuffer> view = [_dev newBufferWithBytesNoCopy:
        (void *)(generic->base + aligned) length:(NSUInteger)length
        options:MTLResourceStorageModeShared deallocator:nil];
    if (!view)
        view = [_dev newBufferWithBytesNoCopy:
            (void *)(generic->base + aligned) length:(NSUInteger)length
            options:MTLResourceStorageModeManaged deallocator:nil];
    if (!view || [view contents] != generic->base + aligned) {
        metal_release(view);
        return -1;
    }
    *view_out = view;
    *base_offset_out = aligned;
    return 0;
}

static int metal_exact_component_view(
    MetalWeightResource *resource, uint64_t offset, size_t nbytes,
    id<MTLBuffer> *view_out, uint64_t *base_out) {
    long page_l;
    if (!resource || !resource->active || nbytes < 1 || !view_out || !base_out ||
        offset > resource->nbytes || nbytes > resource->nbytes - (size_t)offset ||
        (page_l = sysconf(_SC_PAGESIZE)) <= 0)
        return -1;
    uint64_t page = (uint64_t)page_l;
    uint64_t aligned = offset - offset % page;
    uint64_t length = offset - aligned + nbytes;
    if (length > (uint64_t)NSUIntegerMax) return -1;
    id<MTLBuffer> view = [_dev newBufferWithBytesNoCopy:
        (void *)(resource->base + aligned) length:(NSUInteger)length
        options:MTLResourceStorageModeShared deallocator:nil];
    if (!view)
        view = [_dev newBufferWithBytesNoCopy:
            (void *)(resource->base + aligned) length:(NSUInteger)length
            options:MTLResourceStorageModeManaged deallocator:nil];
    if (!view || [view contents] != resource->base + aligned) {
        metal_release(view);
        return -1;
    }
    *view_out = view;
    *base_out = aligned;
    return 0;
}

static int batch_bind_weight_buffers(id<MTLComputeCommandEncoder> enc,
                                     const SaltGpuResourceRef *resource,
                                     id<MTLBuffer> operation_view) {
    id<MTLBuffer> values = nil, scales = nil, biases = nil;
    MetalWeightResource *generic;
    if (!enc || !salt_gpu_resource_ref_valid(resource)) return -1;
    generic = metal_weight_resource(resource->kind, resource->resource_id);
    if (generic && !generic->active) return -1;
    if (operation_view) {
        values = scales = biases = operation_view;
    } else if (generic) {
        values = scales = biases = generic->map;
    } else {
    switch (resource->kind) {
    case SALT_GPU_RESOURCE_ARENA:
        if (resource->resource_id != 0) return -1;
        values = _arena_v; scales = _arena_s; biases = _arena_b;
        break;
    case SALT_GPU_RESOURCE_TRUNK:
        if (resource->resource_id != 0) return -1;
        values = scales = biases = _trunk_map;
        break;
    case SALT_GPU_RESOURCE_EXPERT_LAYER:
        if (resource->resource_id != _res_id) return -1;
        values = scales = biases = _res_map;
        break;
    default:
        return -1;
    }
    }
    if (!values || !scales || !biases) return -1;
    [enc setBuffer:values offset:0 atIndex:0];
    [enc setBuffer:scales offset:0 atIndex:1];
    [enc setBuffer:biases offset:0 atIndex:2];
    return 0;
}

static int metal_env_u32(const char *name, uint32_t minimum,
                         uint32_t maximum, uint32_t fallback,
                         uint32_t *value_out) {
    const char *text = getenv(name);
    char *end = NULL;
    unsigned long value;
    if (!value_out) return -1;
    if (!text || !*text) {
        *value_out = fallback;
        return 0;
    }
    value = strtoul(text, &end, 10);
    if (!end || *end || value < minimum || value > maximum) return -1;
    *value_out = (uint32_t)value;
    return 0;
}

static int metal_tiled_configure(void) {
    if (_tiled_configured) return 0;
    if (metal_env_u32("SALT_METAL_Q4_SIMDGROUPS", 0, 32, 0,
            &_tiled_simdgroups) != 0 ||
        metal_env_u32("SALT_METAL_Q4_MAX_DESCRIPTORS", 0, 128, 0,
            &_tiled_max_descriptors) != 0 ||
        metal_env_u32("SALT_METAL_Q4_WEIGHT_STATIONARY_MIN_B", 0, 512, 0,
            &_weight_stationary_min_b) != 0 ||
        (_tiled_simdgroups > 0 && _tiled_max_descriptors == 0))
        return -1;
    _tiled_token_span = _tiled_simdgroups * 16u;
    _tiled_configured = 1;
    return 0;
}

static int batch_pipeline_prepare(void) {
    if (metal_tiled_configure() != 0 ||
        metal_selected_resources_prepare() != 0) return -1;
    if (_bpso2 && _bweight_pso && _bcoalesced_pso && _bselected_pso &&
        _bselected_ragged_pso && _btiled_pso &&
        _selected_arg_encoder && _selected_args && _selected_desc && _bqueue2)
        return 0;
    metal_release(_bpso2); _bpso2 = nil;
    metal_release(_bweight_pso); _bweight_pso = nil;
    metal_release(_bcoalesced_pso); _bcoalesced_pso = nil;
    metal_release(_bselected_pso); _bselected_pso = nil;
    metal_release(_bselected_ragged_pso); _bselected_ragged_pso = nil;
    metal_release(_btiled_pso); _btiled_pso = nil;
    metal_release(_bqueue2); _bqueue2 = nil;
    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> lib = metal_library_for_source_mode(
            _batch_src2, &err, 0, 0);
        if (!lib) {
            fprintf(stderr, "gpu-metal: batch library failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q4batch2"];
        id<MTLFunction> weight_fn =
            [lib newFunctionWithName:@"q4weightstationary"];
        id<MTLFunction> coalesced_fn =
            [lib newFunctionWithName:@"q4coalesced"];
        id<MTLFunction> selected_fn =
            [lib newFunctionWithName:@"q4selected"];
        id<MTLFunction> selected_ragged_fn =
            [lib newFunctionWithName:@"q4selectedragged"];
        id<MTLFunction> tiled_fn = [lib newFunctionWithName:@"q4warpbatch"];
        if (!fn || !weight_fn || !coalesced_fn || !selected_fn ||
            !selected_ragged_fn || !tiled_fn) {
            fprintf(stderr,
                "gpu-metal: batch function lookup failed base=%d weight=%d "
                "coalesced=%d selected=%d ragged=%d tiled=%d names=%s\n",
                fn != nil, weight_fn != nil, coalesced_fn != nil,
                selected_fn != nil, selected_ragged_fn != nil,
                tiled_fn != nil,
                [[[lib functionNames] componentsJoinedByString:@","] UTF8String]);
            [lib release];
            metal_release(fn); metal_release(weight_fn);
            metal_release(coalesced_fn);
            metal_release(selected_fn); metal_release(selected_ragged_fn);
            metal_release(tiled_fn);
            return -1;
        }
        [lib release];
        id<MTLComputePipelineState> pso =
            [_dev newComputePipelineStateWithFunction:fn error:&err];
        id<MTLComputePipelineState> weight_pso =
            [_dev newComputePipelineStateWithFunction:weight_fn error:&err];
        id<MTLComputePipelineState> coalesced_pso =
            [_dev newComputePipelineStateWithFunction:coalesced_fn error:&err];
        id<MTLComputePipelineState> selected_pso =
            [_dev newComputePipelineStateWithFunction:selected_fn error:&err];
        id<MTLComputePipelineState> selected_ragged_pso =
            [_dev newComputePipelineStateWithFunction:selected_ragged_fn
                                                error:&err];
        id<MTLComputePipelineState> tiled_pso =
            [_dev newComputePipelineStateWithFunction:tiled_fn error:&err];
        [fn release];
        [weight_fn release];
        [coalesced_fn release];
        [selected_fn release];
        [selected_ragged_fn release];
        [tiled_fn release];
        id<MTLCommandQueue> queue = [_dev newCommandQueue];
        if (!pso || !weight_pso || !coalesced_pso || !selected_pso ||
            !selected_ragged_pso || !tiled_pso || !queue ||
            tiled_pso.threadExecutionWidth != 32 ||
            (_tiled_simdgroups > 0 &&
             tiled_pso.maxTotalThreadsPerThreadgroup <
                (NSUInteger)_tiled_simdgroups * 32u)) {
            fprintf(stderr, "gpu-metal: batch pipeline failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown");
            metal_release(pso);
            metal_release(weight_pso);
            metal_release(coalesced_pso);
            metal_release(selected_pso);
            metal_release(selected_ragged_pso);
            metal_release(tiled_pso);
            metal_release(queue);
            return -1;
        }
        _bpso2 = pso;
        _bweight_pso = weight_pso;
        _bcoalesced_pso = coalesced_pso;
        _bselected_pso = selected_pso;
        _bselected_ragged_pso = selected_ragged_pso;
        _btiled_pso = tiled_pso;
        _bqueue2 = queue;
    }
    return 0;
}

/* The legacy deferred mode owns one staged result. Indexed direct-output
 * submissions own command-local x/descriptor buffers and may queue until
 * the first consumer fence; synchronization never scatters those outputs. */
static int _defer = 0;
static id<MTLCommandBuffer> _pend_cb = nil;
#define DIRECT_PENDING_MAX 16
static id<MTLCommandBuffer> _direct_pending[DIRECT_PENDING_MAX];
static int _direct_pending_n = 0;
/* Deferred completion outlives salt_gpu_q4_batch()'s caller-owned
 * descriptor arrays.  Keep backend-owned copies: proj_batch's Rj is
 * stack storage and its caller frees the ys pointer table immediately
 * after this function returns. */
static float *_pend_ys[BATCH_MAXSLOT];
static int _pend_Rj[BATCH_MAXSLOT];
static int _pend_n = 0, _pend_bi = 0;

void salt_gpu_set_mapped_only(int on) {
    _mapped_only = on != 0;
}

int salt_gpu_mapped_only(void) {
    return _mapped_only;
}

void salt_gpu_batch_stats_get(SaltGpuBatchStats *stats) {
    if (!stats) return;
    stats->arena_batches = _batch_stats.arena_batches;
    stats->trunk_batches = _batch_stats.trunk_batches;
    stats->expert_layer_batches = _batch_stats.expert_layer_batches;
    stats->mapped_only_misses = _batch_stats.mapped_only_misses;
    stats->direct_output_batches = _batch_stats.direct_output_batches;
    stats->direct_output_jobs = _batch_stats.direct_output_jobs;
}

int salt_gpu_batch_stats_get_v2(SaltGpuBatchStatsV2 *stats,
                                size_t stats_size) {
    if (!stats || stats_size != sizeof *stats) return -1;
    *stats = _batch_stats;
    return 0;
}

static int salt_gpu_proj_batch_indexed_impl(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *key, int exact_views) {
    const SaltGpuResourceRef *resource = NULL;
    id<MTLBuffer> ybuffer;
    int tensor_slot = -1;
    uint64_t total_rows, output_span;
    size_t output_end;
    int *slots = _direct_slots, *rows = _direct_rows;
    int rc = -1;
    const char *dispatch = getenv("SALT_GPU_DISPATCH");

    if (!_dev || !vals || !scales || !biases || !xs || !output ||
        !output->contents ||
        !output->backend || !key || R < 1 || C < 1 || B < 2 ||
        first_job < 0 || job_count < 1 || first_job > B - job_count ||
        job_count > BATCH_MAXSLOT ||
        (dispatch && (!strcmp(dispatch, "icb") ||
                      !strcmp(dispatch, "metal-icb"))))
        return -1;
    total_rows = (uint64_t)(uint32_t)R * (uint32_t)job_count;
    output_span = (uint64_t)(uint32_t)B * (uint32_t)R;
    if (total_rows > UINT32_MAX || output_span > UINT32_MAX ||
        (uint64_t)y_float_offset > (uint64_t)UINT32_MAX - output_span ||
        output_span > (uint64_t)SIZE_MAX ||
        y_float_offset > SIZE_MAX - (size_t)output_span ||
        _direct_pending_n >= DIRECT_PENDING_MAX ||
        batch_pipeline_prepare() != 0)
        return -1;
    output_end = y_float_offset + (size_t)output_span;
    if (output_end > output->nbytes / sizeof(float)) return -1;
    ybuffer = (id<MTLBuffer>)output->backend;
    if ([ybuffer contents] != output->contents ||
        (size_t)[ybuffer length] < output->nbytes)
        return -1;
    /* The staged path owns one pending descriptor. Complete it before a
     * direct-output submission; direct submissions never force each other. */
    if ((_pend_cb || _direct_pending_n > 0) && salt_gpu_sync() != 0) return -1;


    {
        tensor_slot = batch_slot(vals, scales, biases, key, R, C);
        if (tensor_slot < 0) goto done;
        for (int j = 0; j < job_count; j++) {
            slots[j] = tensor_slot;
            rows[j] = R;
        }
    }
    if (batch_resource_for_slots(slots, job_count, &resource) != 0)
        goto done;

    @autoreleasepool {
        size_t xcount, xbytes, desc_nbytes;
        id<MTLBuffer> xbuffer = nil, desc_buffer = nil;
        id<MTLBuffer> operation_view = nil;
        id<MTLBuffer> value_view = nil, scale_view = nil, bias_view = nil;
        uint64_t value_base = 0, scale_base = 0, bias_base = 0;
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;
        BatchDesc4 *desc;
        if ((size_t)job_count > SIZE_MAX / (size_t)C ||
            (xcount = (size_t)job_count * (size_t)C) >
                SIZE_MAX / sizeof(float) ||
            (size_t)job_count > SIZE_MAX / (2u * sizeof(BatchDesc4)))
            goto direct_done;
        xbytes = xcount * sizeof(float);
        desc_nbytes = (size_t)job_count * 2u * sizeof(BatchDesc4);
        if (!_bx2[0] || !_desc || _bx2_len[0] < xbytes ||
            _desc_len < desc_nbytes) goto direct_done;
        xbuffer = _bx2[0];
        desc_buffer = _desc;
        memcpy([xbuffer contents], xs + (size_t)first_job * C, xbytes);
        desc = (BatchDesc4 *)[desc_buffer contents];
        uint64_t operation_base = 0;
        if (exact_views) {
            size_t value_bytes, scale_bytes;
            MetalWeightResource *generic = metal_weight_resource(
                resource->kind, resource->resource_id);
            const SaltGpuResourceRef *ref = &_tslot[tensor_slot].resource;
            if (!generic || batch_weight_sizes(R, C,
                    &value_bytes, &scale_bytes) != 0 ||
                metal_exact_component_view(generic, ref->value_offset,
                    value_bytes, &value_view, &value_base) != 0 ||
                metal_exact_component_view(generic, ref->scale_offset,
                    scale_bytes, &scale_view, &scale_base) != 0 ||
                metal_exact_component_view(generic, ref->bias_offset,
                    scale_bytes, &bias_view, &bias_base) != 0 ||
                batch_build_component_desc(desc, tensor_slot, R, C, job_count,
                    value_base, scale_base, bias_base) != 0)
                goto direct_done;
        } else if (metal_q4_operation_view(
                resource, slots, rows, C, job_count,
                &operation_view, &operation_base) != 0 ||
            batch_build_desc(desc, slots, rows, C, job_count, 0,
                operation_base) != 0) {
            goto direct_done;
        }
        for (int j = 0; j < job_count; j++) {
            uint64_t canonical = (uint64_t)y_float_offset +
                (uint64_t)(uint32_t)(first_job + j) * (uint32_t)R;

            /* desc[2*j+1].w remains the dispatch row prefix used only to
             * identify the job. desc[2*j].w is its preallocated final
             * destination and is never recomputed at completion. */
            desc[2 * j].yoff = canonical;
        }
        int use_weight_stationary = _weight_stationary_min_b > 0 &&
            (uint32_t)job_count >= _weight_stationary_min_b &&
            (C & 31) == 0;
        if (use_weight_stationary) {
            desc[0].xoff = 0;
            desc[1].soff = (uint64_t)(uint32_t)R;
            desc[1].xoff = (uint64_t)(uint32_t)C;
            desc[1].yoff = (uint64_t)(uint32_t)job_count;
        }
        cb = [_bqueue2 commandBuffer];
        if (!cb) goto direct_done;
        enc = [cb computeCommandEncoder];
        if (!enc) goto direct_done;
        [enc setComputePipelineState:
            use_weight_stationary ? _bweight_pso : _bpso2];
        if (exact_views) {
            [enc setBuffer:value_view offset:0 atIndex:0];
            [enc setBuffer:scale_view offset:0 atIndex:1];
            [enc setBuffer:bias_view offset:0 atIndex:2];
        } else if (batch_bind_weight_buffers(
                enc, resource, operation_view) != 0) {
            [enc endEncoding];
            goto direct_done;
        }
        [enc setBuffer:xbuffer offset:0 atIndex:3];
        [enc setBuffer:ybuffer offset:0 atIndex:4];
        [enc setBuffer:desc_buffer offset:0 atIndex:5];
        if (use_weight_stationary) {
            [enc setThreadgroupMemoryLength:512u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:
                MTLSizeMake((NSUInteger)(R + 7) / 8u,
                            (NSUInteger)(job_count + 7) / 8u, 1)
                 threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        } else {
            uint rv = (uint)total_rows, cv = (uint)C, nj = (uint)job_count;
            [enc setBytes:&rv length:sizeof(rv) atIndex:6];
            [enc setBytes:&cv length:sizeof(cv) atIndex:7];
            [enc setBytes:&nj length:sizeof(nj) atIndex:8];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)total_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        if (operation_view || exact_views) {
            [cb waitUntilCompleted];
            if (metal_command_succeeded(cb) != 0) goto direct_done;
        } else {
            _direct_pending[_direct_pending_n++] = [cb retain];
        }
        switch (resource->kind) {
        case SALT_GPU_RESOURCE_ARENA:
            _batch_stats.arena_batches++;
            break;
        case SALT_GPU_RESOURCE_TRUNK:
            _batch_stats.trunk_batches++;
            break;
        case SALT_GPU_RESOURCE_EXPERT_LAYER:
            _batch_stats.expert_layer_batches++;
            break;
        default:
            break;
        }
        _batch_stats.direct_output_batches++;
        _batch_stats.direct_output_jobs += (uint64_t)(uint32_t)job_count;
        if (use_weight_stationary)
            _batch_stats.q4_weight_stationary_batches++;
        rc = 0;
direct_done:
        metal_release(operation_view);
        metal_release(value_view);
        metal_release(scale_view);
        metal_release(bias_view);
    }

done:
    return rc;
}

int salt_gpu_proj_batch_indexed(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *key) {
    return salt_gpu_proj_batch_indexed_impl(vals, scales, biases,
        R, C, B, first_job, job_count, xs, output, y_float_offset, key, 0);
}

int salt_gpu_proj_batch_indexed_exact_views(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *key) {
    return salt_gpu_proj_batch_indexed_impl(vals, scales, biases,
        R, C, B, first_job, job_count, xs, output, y_float_offset, key, 1);
}

int salt_gpu_q4_batch(const uint32_t *const *vals,
                        const uint16_t *const *scales,
                        const uint16_t *const *biases,
                        const float *const *xs, float *const *ys,
                        const void *const *ids,
                        const int *Rj, int C, int njobs) {
    uint64_t row_total = 0;
    uint32_t Rtot;
    int detail = getenv("SALT_GPU_MS_DETAIL") != NULL;
    struct timespec detail_start, detail_sync, detail_view, detail_desc;
    struct timespec detail_commit, detail_wait, detail_scatter;
    static unsigned long long detail_sequence = 0;
    unsigned long long detail_id = 0;
    unsigned long long detail_view_bytes = 0;
    int detail_resource_kind = -1;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-dbg] batch entry njobs=%d C=%d Rj0=%d\n",
                njobs, C, Rj ? Rj[0] : -1);
    if (!_dev || !vals || !scales || !xs || !ys || !ids || !Rj ||
        njobs < 1 || njobs > BATCH_MAXSLOT || C < 1) return -1;
    if (detail) {
        detail_id = ++detail_sequence;
        clock_gettime(CLOCK_MONOTONIC, &detail_start);
    }
    const char *dispatch = getenv("SALT_GPU_DISPATCH");
    if (dispatch && (!strcmp(dispatch, "icb") ||
                     !strcmp(dispatch, "metal-icb"))) {
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "[gpu-fail] metal-icb unsupported\n");
        return -1;
    }
    /* There is exactly one pending result descriptor. Never overwrite it:
     * complete and scatter prior work before accepting another batch. */
    if (_pend_cb && salt_gpu_sync() != 0) return -1;
    if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_sync);
    for (int j = 0; j < njobs; j++) {
        if (Rj[j] < 1 || row_total > UINT32_MAX - (uint32_t)Rj[j])
            return -1;
        row_total += (uint32_t)Rj[j];
    }
    Rtot = (uint32_t)row_total;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-dbg] Rtot=%u (last Rj=%d)\n", Rtot,
                njobs > 0 ? Rj[njobs - 1] : -1);
    if (getenv("SALT_GPU_MS"))
        fprintf(stderr, "[gpu-ms] ENTER njobs=%d Rtot=%u C=%d\n",
                njobs, Rtot, C);
    if (Rtot < 1) {
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "[gpu-fail] guards _dev=%d njobs=%d Rtot=%u "
                    "C=%d\n", _dev != nil, njobs, Rtot, C);
        return -1;
    }
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-dbg] guards ok\n");
    if (batch_pipeline_prepare() != 0) {
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "[gpu-fail] batch pipeline\n");
        return -1;
    }
    /* cache all tensors by stable id (first sight copies into the arena) */
    int slot[BATCH_MAXSLOT];
    const SaltGpuResourceRef *resource = NULL;
    if ((size_t)C > SIZE_MAX / (size_t)njobs / sizeof(float)) return -1;
    size_t xbytes = (size_t)C * (size_t)njobs * sizeof(float);
    size_t ybytes = (size_t)Rtot * sizeof(float);
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-dbg] xbytes=%zu ybytes=%zu\n",
                xbytes, ybytes);
    @autoreleasepool {
        static int _buf_idx = 0;
        int bi = _buf_idx;
        int pooled_components = 0;
        id<MTLBuffer> operation_view = nil;
        uint64_t operation_base = 0;
        for (int j = 0; j < njobs; j++) {
            slot[j] = batch_slot(vals[j], scales[j],
                                 biases ? biases[j] : NULL, ids[j],
                                 Rj[j], C);
            if (getenv("SALT_GPU_DIAG") && j < 2)
                fprintf(stderr, "[gpu-dbg] slot j=%d = %d (Rj=%d)\n",
                        j, slot[j], Rj[j]);
            if (slot[j] < 0) {
                if (getenv("SALT_GPU_DIAG"))
                    fprintf(stderr, "[gpu-fail] batch_slot<0 j=%d "
                            "tslot_n=%d njobs=%d Rj=%d C=%d id=%p "
                            "vals=%p res_base=%p res_len=%zu\n",
                            j, _tslot_n, njobs, Rj[j], C, ids[j],
                            (const void *)vals[j],
                            (const void *)_res_base, _res_len);
                return -1;
            }
        }
        if (batch_resource_for_slots(slot, njobs, &resource) != 0) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr, "[gpu-fail] mixed or stale weight resources\n");
            return -1;
        }
        pooled_components = 1;
        for (int j = 0; j < njobs; j++)
            if (!_tslot[slot[j]].value_map || !_tslot[slot[j]].scale_map ||
                !_tslot[slot[j]].bias_map) {
                pooled_components = 0;
                break;
            }
        if (!pooled_components && metal_q4_operation_view(
                resource, slot, Rj, C, njobs,
                &operation_view, &operation_base) != 0)
            return -1;
        if (operation_view) {
            if (detail) detail_view_bytes = (unsigned long long)[operation_view length];
            [operation_view autorelease];
            if (_defer) return -1;
        }
        if (pooled_components && detail)
            for (int j = 0; j < njobs; j++)
                detail_view_bytes +=
                    (unsigned long long)_tslot[slot[j]].pooled_nbytes;
        if (detail) detail_resource_kind = (int)resource->kind;
        if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_view);
        if (!_bx2[bi] || !_by2[bi] || !_desc ||
            _bx2_len[bi] < xbytes || _by2_len[bi] < ybytes ||
            _desc_len < (size_t)njobs * 2u * sizeof(BatchDesc4)) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "[gpu-fail] session buffer capacity x=%zu/%zu "
                    "y=%zu/%zu desc=%zu/%zu\n",
                    xbytes, _bx2_len[bi], ybytes, _by2_len[bi],
                    (size_t)njobs * 2u * sizeof(BatchDesc4), _desc_len);
            return -1;
        }
        BatchDesc4 *desc = (BatchDesc4 *)_desc.contents;
        int descriptor_count = 0, max_rows = 0, max_batch = 0;
        int tiled_rc = pooled_components ? 1 : batch_build_tiled_desc(
            desc, slot, Rj, C, njobs, operation_base,
            &descriptor_count, &max_rows, &max_batch);
        if (tiled_rc < 0) return -1;
        int use_tiled = tiled_rc == 0;
        /* pack per-job x at stride C -- but if every job shares the
         * same x pointer (gate/up all take the latent), copy ONCE and
         * give every job xoff=0: avoids the redundant per-job copy. */
        int xs_same = 1;
        for (int j = 1; j < njobs; j++)
            if (xs[j] != xs[0]) { xs_same = 0; break; }
        if (xs_same && !use_tiled) {
            memcpy(_bx2[bi].contents, xs[0], (size_t)C * sizeof(float));
        } else {
            for (int j = 0; j < njobs; j++)
                memcpy((char *)_bx2[bi].contents + (size_t)j * C * sizeof(float),
                       xs[j], (size_t)C * sizeof(float));
        }
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "[gpu-dbg] x packed (same=%d)\n", xs_same);

        /* desc: 2 uint4 per job -- {voff, soff, xoff, yoff} in
         * ELEMENT units + {boff, row-start, 0, 0} (the bias offset;
         * the arena stores biases at soff, the pool map at rel_b;
         * the row-start is the job's prefix into the y buffer -- the
         * kernel's mixed-row job find). */
        if (pooled_components) {
            uint64_t output_prefix = 0;
            for (int j = 0; j < njobs; j++) {
                TensorSlot *pooled = &_tslot[slot[j]];
                const SaltGpuResourceRef *ref = &pooled->resource;
                uint64_t xoff = xs_same ? 0 :
                    (uint64_t)(uint32_t)j * (uint32_t)C;
                if (ref->value_offset < pooled->value_base ||
                    ref->scale_offset < pooled->scale_base ||
                    ref->bias_offset < pooled->bias_base ||
                    output_prefix > UINT64_MAX - (uint32_t)Rj[j])
                    return -1;
                desc[2 * j] = (BatchDesc4) {
                    ref->value_offset - pooled->value_base,
                    ref->scale_offset - pooled->scale_base,
                    xoff, output_prefix,
                };
                desc[2 * j + 1] = (BatchDesc4) {
                    ref->bias_offset - pooled->bias_base,
                    0, 0, 0,
                };
                output_prefix += (uint32_t)Rj[j];
            }
        } else if (!use_tiled && batch_build_desc(desc, slot, Rj, C, njobs,
                xs_same, operation_base) != 0) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr, "[gpu-fail] descriptor offset overflow\n");
            return -1;
        }
        if (getenv("SALT_GPU_TILED"))
            fprintf(stderr,
                "[gpu-tiled] active=%d logical_jobs=%d descriptors=%d "
                "max_batch=%d max_rows=%d C=%d\n",
                use_tiled, njobs, use_tiled ? descriptor_count : njobs,
                use_tiled ? max_batch : 1, use_tiled ? max_rows : 0, C);
        if (use_tiled && getenv("SALT_GPU_DIAG")) {
            memset(_by2[bi].contents, 0xa5, ybytes);
            fprintf(stderr,
                "[gpu-tiled-desc] v=%llu s=%llu b=%llu x=%llu y=%llu "
                "R=%llu C=%llu B=%llu\n",
                (unsigned long long)desc[0].voff,
                (unsigned long long)desc[0].soff,
                (unsigned long long)desc[1].voff,
                (unsigned long long)desc[0].xoff,
                (unsigned long long)desc[0].yoff,
                (unsigned long long)desc[1].soff,
                (unsigned long long)desc[1].xoff,
                (unsigned long long)desc[1].yoff);
        }
        if (!pooled_components && _trunk_address_audit_active &&
            !_trunk_address_desc_reported &&
            resource->kind == SALT_GPU_RESOURCE_TRUNK) {
            const SaltGpuResourceRef *ref = &_tslot[slot[0]].resource;
            uint64_t gpu_address = 0;
            int gpu_address_available =
                metal_buffer_gpu_address(_trunk_map, &gpu_address);
            unsigned long long gpu_base =
                (unsigned long long)gpu_address;
            unsigned long long gpu_v = gpu_base +
                (unsigned long long)desc[0].voff;
            unsigned long long gpu_s = gpu_base +
                (unsigned long long)desc[0].soff;
            unsigned long long gpu_b = gpu_base +
                (unsigned long long)desc[1].voff;
            fprintf(stderr,
                    "[gpu-address] descriptor sample=1 key=%p "
                    "resource=trunk resource_id=%u "
                    "voff_bytes=%llu soff_bytes=%llu boff_bytes=%llu "
                    "gpu_address_available=%d "
                    "gpu_base=0x%llx gpu_v=0x%llx gpu_s=0x%llx "
                    "gpu_b=0x%llx host_v=%p host_s=%p host_b=%p\n",
                    _tslot[slot[0]].key, ref->resource_id,
                    (unsigned long long)desc[0].voff,
                    (unsigned long long)desc[0].soff,
                    (unsigned long long)desc[1].voff,
                    gpu_address_available,
                    gpu_base, gpu_v, gpu_s, gpu_b,
                    (const void *)(_trunk_base + ref->value_offset),
                    (const void *)(_trunk_base + ref->scale_offset),
                    (const void *)(_trunk_base + ref->bias_offset));
            _trunk_address_desc_reported = 1;
        }
        if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_desc);
        if (getenv("SALT_GPU_DIAG") && Rtot == 512) {
            fprintf(stderr, "[gpu-desc] njobs=%d Rtot=%u C=%d: "
                    "j0 key-hit voff=%llu soff=%llu xoff=%llu yoff=%llu | "
                    "j1 voff=%llu soff=%llu yoff=%llu\n",
                    njobs, Rtot, C,
                    (unsigned long long)desc[0].voff,
                    (unsigned long long)desc[0].soff,
                    (unsigned long long)desc[0].xoff,
                    (unsigned long long)desc[0].yoff,
                    (unsigned long long)desc[1].voff,
                    (unsigned long long)desc[1].soff,
                    (unsigned long long)desc[1].yoff);
        }

        id<MTLCommandBuffer> cb = [_bqueue2 commandBuffer];
        if (!cb) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr, "[gpu-fail] command buffer creation\n");
            return -1;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr, "[gpu-fail] compute encoder creation\n");
            return -1;
        }
        if (pooled_components) {
            uint cv = (uint)C, nj = 1;
            [enc setComputePipelineState:_bpso2];
            [enc setBuffer:_bx2[bi] offset:0 atIndex:3];
            [enc setBuffer:_by2[bi] offset:0 atIndex:4];
            for (int j = 0; j < njobs; j++) {
                TensorSlot *pooled = &_tslot[slot[j]];
                uint rv = (uint)Rj[j];
                [enc setBuffer:pooled->value_map offset:0 atIndex:0];
                [enc setBuffer:pooled->scale_map offset:0 atIndex:1];
                [enc setBuffer:pooled->bias_map offset:0 atIndex:2];
                [enc setBuffer:_desc
                        offset:(NSUInteger)(2 * j) * sizeof(BatchDesc4)
                       atIndex:5];
                [enc setBytes:&rv length:sizeof(rv) atIndex:6];
                [enc setBytes:&cv length:sizeof(cv) atIndex:7];
                [enc setBytes:&nj length:sizeof(nj) atIndex:8];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)Rj[j], 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        } else if (batch_bind_weight_buffers(
                enc, resource, operation_view) != 0) {
            return -1;
        } else if (use_tiled) {
            uint nd = (uint)descriptor_count;
            [enc setComputePipelineState:
                _tiled_simdgroups > 0 ? _btiled_pso : _bcoalesced_pso];
            [enc setBuffer:_bx2[bi] offset:0 atIndex:3];
            [enc setBuffer:_by2[bi] offset:0 atIndex:4];
            [enc setBuffer:_desc offset:0 atIndex:5];
            [enc setBytes:&nd length:sizeof(nd) atIndex:6];
            if (_tiled_simdgroups > 0) {
                [enc setBytes:&_tiled_simdgroups
                       length:sizeof(_tiled_simdgroups) atIndex:7];
                [enc setBytes:&_tiled_token_span
                       length:sizeof(_tiled_token_span) atIndex:8];
                [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)max_rows,
                                (NSUInteger)((max_batch +
                                    (int)_tiled_token_span - 1) /
                                    (int)_tiled_token_span),
                                (NSUInteger)descriptor_count)
                    threadsPerThreadgroup:
                        MTLSizeMake((NSUInteger)_tiled_simdgroups * 32u, 1, 1)];
            } else {
                [enc dispatchThreads:
                    MTLSizeMake((NSUInteger)max_rows,
                                (NSUInteger)max_batch,
                                (NSUInteger)descriptor_count)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            }
        } else {
            uint rv = (uint)Rtot, cv = (uint)C, nj = (uint)njobs;
            [enc setComputePipelineState:_bpso2];
            [enc setBuffer:_bx2[bi] offset:0 atIndex:3];
            [enc setBuffer:_by2[bi] offset:0 atIndex:4];
            [enc setBuffer:_desc offset:0 atIndex:5];
            [enc setBytes:&rv length:sizeof(rv) atIndex:6];
            [enc setBytes:&cv length:sizeof(cv) atIndex:7];
            [enc setBytes:&nj length:sizeof(nj) atIndex:8];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)Rtot, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        struct timespec _ta, _tb;
        clock_gettime(CLOCK_MONOTONIC, &_ta);
        [cb commit];
        if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_commit);
        switch (resource->kind) {
        case SALT_GPU_RESOURCE_ARENA:
            _batch_stats.arena_batches++;
            break;
        case SALT_GPU_RESOURCE_TRUNK:
            _batch_stats.trunk_batches++;
            break;
        case SALT_GPU_RESOURCE_EXPERT_LAYER:
            _batch_stats.expert_layer_batches++;
            break;
        default:
            break;
        }
        if (_defer) {
            /* the deferred-sync mode: the engine commits the group
             * (a layer's projections) without waiting, syncs once at
             * the consumer boundary -- the per-call wait amortized. */
            _pend_cb = [cb retain];
            for (int j = 0; j < njobs; j++) {
                _pend_ys[j] = ys[j];
                _pend_Rj[j] = Rj[j];
            }
            _pend_n = njobs;
            _pend_bi = bi;
        } else {
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return -1;
        if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_wait);
        if (use_tiled && getenv("SALT_GPU_DIAG")) {
            uint32_t first[4];
            memcpy(first, _by2[bi].contents, sizeof first);
            fprintf(stderr,
                "[gpu-tiled-write] first=0x%08x/%08x/%08x/%08x\n",
                first[0], first[1], first[2], first[3]);
        }
        _buf_idx = 1 - bi;
        clock_gettime(CLOCK_MONOTONIC, &_tb);
        if (getenv("SALT_GPU_MS")) {
            double dt = (double)(_tb.tv_sec - _ta.tv_sec) +
                        (double)(_tb.tv_nsec - _ta.tv_nsec) * 1e-9;
            static double _tacc = 0; static long _tcnt = 0;
            _tcnt++; _tacc += dt;
            if (_tcnt <= 4 || _tcnt % 100 == 0)
                fprintf(stderr, "[gpu-ms] njobs=%d Rtot=%u C=%d: %.3f ms (avg %.3f)\n",
                        njobs, Rtot, C, dt * 1e3, _tacc / _tcnt * 1e3);
        }
        /* scatter: job j's rows live at [prefix, prefix+Rj) in
         * _by2[bi] -- the same prefix the desc carried. */
        {
            size_t ypref2 = 0;
            for (int j = 0; j < njobs; j++) {
                memcpy(ys[j],
                       (char *)_by2[bi].contents + ypref2 * sizeof(float),
                       (size_t)Rj[j] * sizeof(float));
                ypref2 += (size_t)Rj[j];
            }
        }
        if (detail) {
            double admit_sync_ms, view_ms, desc_ms, encode_ms, wait_ms;
            double scatter_ms, total_ms;
            clock_gettime(CLOCK_MONOTONIC, &detail_scatter);
#define G4_METAL_MS(a, b) \
            ((double)((b).tv_sec - (a).tv_sec) * 1e3 + \
             (double)((b).tv_nsec - (a).tv_nsec) * 1e-6)
            admit_sync_ms = G4_METAL_MS(detail_start, detail_sync);
            view_ms = G4_METAL_MS(detail_sync, detail_view);
            desc_ms = G4_METAL_MS(detail_view, detail_desc);
            encode_ms = G4_METAL_MS(detail_desc, detail_commit);
            wait_ms = G4_METAL_MS(detail_commit, detail_wait);
            scatter_ms = G4_METAL_MS(detail_wait, detail_scatter);
            total_ms = G4_METAL_MS(detail_start, detail_scatter);
#undef G4_METAL_MS
            fprintf(stderr,
                "[gpu-ms-detail] seq=%llu njobs=%d Rtot=%u C=%d "
                "resource_kind=%d view_bytes=%llu "
                "admit_sync_ms=%.3f view_ms=%.3f desc_ms=%.3f "
                "encode_ms=%.3f wait_ms=%.3f scatter_ms=%.3f total_ms=%.3f\n",
                detail_id, njobs, Rtot, C, detail_resource_kind,
                detail_view_bytes, admit_sync_ms, view_ms, desc_ms,
                encode_ms, wait_ms, scatter_ms, total_ms);
        }
        }
    }
    return 0;
}


int salt_gpu_set_defer(int on) {
    int rc = 0;
    if (!on && (_pend_cb || _direct_pending_n > 0)) rc = salt_gpu_sync();
    _defer = on;
    return rc;
}

int salt_gpu_defer(void) { return _defer; }

int salt_gpu_sync(void) {
    int failed = 0;
    for (int i = 0; i < _direct_pending_n; i++) {
        id<MTLCommandBuffer> cb = _direct_pending[i];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) failed = 1;
        _direct_pending[i] = nil;
        [cb release];
    }
    _direct_pending_n = 0;
    if (_pend_cb) {
        id<MTLCommandBuffer> cb = _pend_cb;
        [cb waitUntilCompleted];
        int rc = metal_command_succeeded(cb);
        if (rc == 0 && _pend_n > 0) {
            size_t ypref2 = 0;
            for (int j = 0; j < _pend_n; j++) {
                memcpy(_pend_ys[j],
                       (char *)_by2[_pend_bi].contents +
                           ypref2 * sizeof(float),
                       (size_t)_pend_Rj[j] * sizeof(float));
                ypref2 += (size_t)_pend_Rj[j];
            }
        }
        _pend_cb = nil;
        _pend_n = 0;
        [cb release];
        if (rc != 0) failed = 1;
    }
    return failed ? -1 : 0;
}

int salt_gpu_shared_buffer_free(SaltGpuSharedBuffer *buffer) {
    id<MTLBuffer> storage;
    if (!buffer) return -1;
    if (!buffer->contents && !buffer->backend && buffer->nbytes == 0) return 0;
    if (!buffer->contents || !buffer->backend || buffer->nbytes < 1 ||
        salt_gpu_sync() != 0)
        return -1;
    storage = (id<MTLBuffer>)buffer->backend;
    if ([storage contents] != buffer->contents ||
        (size_t)[storage length] < buffer->nbytes)
        return -1;
    for (int i = 0; i < METAL_SHARED_SLOTS; i++)
        if (_shared_slots[i].buffer == storage)
            memset(&_shared_slots[i], 0, sizeof _shared_slots[i]);
    memset(buffer, 0, sizeof *buffer);
    [storage release];
    return 0;
}

int salt_gpu_attention_batch(
    int full_attention, int n_heads, int n_kv_heads, int head_dim, int window,
    const SaltGpuSharedBuffer *queries, size_t query_float_offset,
    const SaltGpuSharedBuffer *kv, size_t key_float_offset,
    size_t value_float_offset, int start_position, int batch,
    int query_stride, int kv_stride, SaltGpuSharedBuffer *outputs,
    size_t output_float_offset) {
    id<MTLBuffer> qbuffer, kvbuffer, obuffer;
    uint64_t qcount, kvcount, tasks;
    size_t score_bytes;
    struct MetalAttentionArgs {
        uint32_t full, n_heads, n_kv_heads, head_dim, window;
        uint32_t start, batch, qstride, kvstride;
    } args;
    if ((full_attention != 0 && full_attention != 1) || n_heads < 1 ||
        n_kv_heads < 1 || n_heads % n_kv_heads != 0 || head_dim < 1 ||
        head_dim > 512 || (!full_attention && window < 1) ||
        start_position < 0 || batch < 1 || start_position > INT_MAX - batch ||
        query_stride != n_heads * head_dim ||
        kv_stride != n_kv_heads * head_dim || metal_ops_prepare() != 0)
        return -1;
    qcount = (uint64_t)(uint32_t)batch * (uint32_t)query_stride;
    kvcount = (uint64_t)(uint32_t)(start_position + batch) *
        (uint32_t)kv_stride;
    tasks = (uint64_t)(uint32_t)batch * (uint32_t)n_heads;
    score_bytes = (size_t)(start_position + batch) * sizeof(float);
    if (qcount > SIZE_MAX || kvcount > SIZE_MAX || tasks > NSUIntegerMax ||
        score_bytes > 32u * 1024u ||
        metal_shared_range(queries, query_float_offset, (size_t)qcount,
            &qbuffer) != 0 ||
        metal_shared_range(kv, key_float_offset, (size_t)kvcount,
            &kvbuffer) != 0 ||
        value_float_offset > kv->nbytes / sizeof(float) ||
        (size_t)kvcount > kv->nbytes / sizeof(float) - value_float_offset ||
        metal_shared_range(outputs, output_float_offset, (size_t)qcount,
            &obuffer) != 0)
        return -1;
    *(uint32_t *)[_op_status contents] = 0;
    args = (struct MetalAttentionArgs) {
        (uint32_t)full_attention, (uint32_t)n_heads,
        (uint32_t)n_kv_heads, (uint32_t)head_dim, (uint32_t)window,
        (uint32_t)start_position, (uint32_t)batch,
        (uint32_t)query_stride, (uint32_t)kv_stride,
    };
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1;
        [enc setComputePipelineState:_attention_pso];
        [enc setBuffer:qbuffer offset:query_float_offset * sizeof(float)
                atIndex:0];
        [enc setBuffer:kvbuffer offset:key_float_offset * sizeof(float)
                atIndex:1];
        [enc setBuffer:kvbuffer offset:value_float_offset * sizeof(float)
                atIndex:2];
        [enc setBuffer:obuffer offset:output_float_offset * sizeof(float)
                atIndex:3];
        [enc setBuffer:_op_status offset:0 atIndex:4];
        [enc setBytes:&args length:sizeof args atIndex:5];
        [enc setThreadgroupMemoryLength:score_bytes atIndex:0];
        [enc setThreadgroupMemoryLength:2u * sizeof(float) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return -1;
    }
    return *(uint32_t *)[_op_status contents] ? -1 : 0;
}

int salt_gpu_attention_prepare(const SaltGpuSharedBuffer *kv) {
    id<MTLBuffer> storage;
    return metal_ops_prepare() == 0 && metal_session_buffers_prepare() == 0 &&
        kv && kv->nbytes >= sizeof(float) &&
        metal_shared_range(kv, 0, kv->nbytes / sizeof(float), &storage) == 0
        ? 0 : -1;
}

int salt_gpu_attention_bind_kv(size_t key_float_offset,
                               size_t value_float_offset, int kv_stride) {
    return _dev && kv_stride > 0 && key_float_offset < value_float_offset
        ? 0 : -1;
}

int salt_gpu_attention_materialize(SaltGpuSharedBuffer *kv, int position,
                                   uint64_t *copied_bytes) {
    if (!kv || !kv->contents || !kv->backend || !copied_bytes || position < 0 ||
        salt_gpu_sync() != 0)
        return -1;
    *copied_bytes = 0;
    return 0;
}

int salt_gpu_attention_import(SaltGpuSharedBuffer *kv, int position,
                              uint64_t *copied_bytes) {
    return salt_gpu_attention_materialize(kv, position, copied_bytes);
}

int salt_gpu_attention_rewind(int position) {
    return position >= 0 ? salt_gpu_sync() : -1;
}

int salt_gpu_attention_transform(
    int n_heads, int n_kv_heads, int head_dim, int rope_dim,
    int start_position, int batch, int query_stride, int kv_stride, float eps,
    SaltGpuSharedBuffer *queries, size_t query_float_offset,
    SaltGpuSharedBuffer *keys, size_t key_float_offset,
    SaltGpuSharedBuffer *values, size_t value_float_offset,
    const float *q_weight, const float *k_weight,
    const float *cosines, const float *sines, int rope_pairs,
    SaltGpuSharedBuffer *kv, size_t key_cache_float_offset,
    size_t value_cache_float_offset) {
    id<MTLBuffer> qbuffer, kbuffer, vbuffer, kvbuffer;
    uint64_t qcount, kvcount, cache_count, factor_count, tasks, meta_count;
    struct MetalTransformArgs {
        uint32_t n_heads, n_kv_heads, head_dim, rope_dim, start;
        uint32_t batch, qstride, kvstride, rope_pairs;
        float eps;
    } args;
    if (n_heads < 1 || n_kv_heads < 1 || n_heads % n_kv_heads != 0 ||
        head_dim < 2 || head_dim > 512 || head_dim % 2 != 0 ||
        rope_dim < 2 || rope_dim > head_dim || rope_dim % 2 != 0 ||
        rope_pairs != rope_dim / 2 || start_position < 0 || batch < 1 ||
        start_position > INT_MAX - batch ||
        query_stride != n_heads * head_dim ||
        kv_stride != n_kv_heads * head_dim || !(eps >= 0.0f) ||
        !q_weight || !k_weight || !cosines || !sines ||
        metal_ops_prepare() != 0)
        return -1;
    qcount = (uint64_t)(uint32_t)batch * (uint32_t)query_stride;
    kvcount = (uint64_t)(uint32_t)batch * (uint32_t)kv_stride;
    cache_count = (uint64_t)(uint32_t)(start_position + batch) *
        (uint32_t)kv_stride;
    factor_count = (uint64_t)(uint32_t)batch * (uint32_t)rope_pairs;
    tasks = (uint64_t)(uint32_t)batch *
        (uint32_t)(n_heads + 2 * n_kv_heads);
    meta_count = 2u * (uint32_t)head_dim + 2u * factor_count;
    if (qcount > SIZE_MAX || kvcount > SIZE_MAX || cache_count > SIZE_MAX ||
        meta_count > SIZE_MAX / sizeof(float) || tasks > NSUIntegerMax ||
        metal_shared_range(queries, query_float_offset, (size_t)qcount,
            &qbuffer) != 0 ||
        metal_shared_range(keys, key_float_offset, (size_t)kvcount,
            &kbuffer) != 0 ||
        metal_shared_range(values, value_float_offset, (size_t)kvcount,
            &vbuffer) != 0 ||
        metal_shared_range(kv, key_cache_float_offset, (size_t)cache_count,
            &kvbuffer) != 0 ||
        value_cache_float_offset > kv->nbytes / sizeof(float) ||
        (size_t)cache_count > kv->nbytes / sizeof(float) -
            value_cache_float_offset || !_op_meta ||
        (size_t)meta_count > (size_t)[_op_meta length] / sizeof(float))
        return -1;
    float *meta = (float *)[_op_meta contents];
    memcpy(meta, q_weight, (size_t)head_dim * sizeof(float));
    memcpy(meta + head_dim, k_weight, (size_t)head_dim * sizeof(float));
    memcpy(meta + 2u * head_dim, cosines,
           (size_t)factor_count * sizeof(float));
    memcpy(meta + 2u * head_dim + factor_count, sines,
           (size_t)factor_count * sizeof(float));
    *(uint32_t *)[_op_status contents] = 0;
    args = (struct MetalTransformArgs) {
        (uint32_t)n_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim,
        (uint32_t)rope_dim, (uint32_t)start_position, (uint32_t)batch,
        (uint32_t)query_stride, (uint32_t)kv_stride, (uint32_t)rope_pairs,
        eps,
    };
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1;
        [enc setComputePipelineState:_transform_pso];
        [enc setBuffer:qbuffer offset:query_float_offset * sizeof(float)
                atIndex:0];
        [enc setBuffer:kbuffer offset:key_float_offset * sizeof(float)
                atIndex:1];
        [enc setBuffer:vbuffer offset:value_float_offset * sizeof(float)
                atIndex:2];
        [enc setBuffer:_op_meta offset:0 atIndex:3];
        [enc setBuffer:kvbuffer offset:key_cache_float_offset * sizeof(float)
                atIndex:4];
        [enc setBuffer:kvbuffer offset:value_cache_float_offset * sizeof(float)
                atIndex:5];
        [enc setBuffer:_op_status offset:0 atIndex:6];
        [enc setBytes:&args length:sizeof args atIndex:7];
        [enc setThreadgroupMemoryLength:sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return -1;
    }
    return *(uint32_t *)[_op_status contents] ? -1 : 0;
}

int salt_gpu_q4_moe_chain(const SaltGpuMoeExpert *experts, int count,
                          int hidden, int routed,
                          const float *inputs, float *outputs,
                          float *gate_scratch, float *up_scratch,
                          size_t scratch_float_capacity) {
    uint64_t rows64 = 0, elements64;
    int total_rows, rc = -1;
    if (!experts || count < 1 || count > 128 || hidden < 1 || routed < 1 ||
        !inputs || !outputs || !gate_scratch || !up_scratch ||
        metal_ops_prepare() != 0)
        return -1;
    for (int i = 0; i < count; i++) {
        if (experts[i].group < 1) return -1;
        rows64 += (uint32_t)experts[i].group;
    }
    elements64 = rows64 * (uint32_t)routed;
    if (rows64 > BATCH_MAXSLOT || elements64 > SIZE_MAX / sizeof(float) ||
        elements64 > scratch_float_capacity)
        return -1;
    total_rows = (int)rows64;
    {
        int row = 0;
        for (int i = 0; i < count; i++)
            for (int g = 0; g < experts[i].group; g++, row++) {
                _moe_vals[row] = experts[i].gate_vals;
                _moe_scales[row] = experts[i].gate_scales;
                _moe_biases[row] = experts[i].gate_biases;
                _moe_ids[row] = experts[i].gate_id;
                _moe_xs[row] = inputs + (size_t)row * (size_t)hidden;
                _moe_ys[row] = gate_scratch + (size_t)row * (size_t)routed;
                _moe_rows[row] = routed;
            }
    }
    if (salt_gpu_q4_batch(_moe_vals, _moe_scales, _moe_biases,
            _moe_xs, _moe_ys, _moe_ids, _moe_rows,
            hidden, total_rows) != 0 || salt_gpu_sync() != 0)
        goto done;
    {
        int row = 0;
        for (int i = 0; i < count; i++)
            for (int g = 0; g < experts[i].group; g++, row++) {
                _moe_vals[row] = experts[i].up_vals;
                _moe_scales[row] = experts[i].up_scales;
                _moe_biases[row] = experts[i].up_biases;
                _moe_ids[row] = experts[i].up_id;
                _moe_ys[row] = up_scratch + (size_t)row * (size_t)routed;
            }
    }
    if (salt_gpu_q4_batch(_moe_vals, _moe_scales, _moe_biases,
            _moe_xs, _moe_ys, _moe_ids, _moe_rows,
            hidden, total_rows) != 0 || salt_gpu_sync() != 0)
        goto done;
    @autoreleasepool {
        id<MTLBuffer> gate_buffer, up_buffer;
        size_t gate_offset, up_offset;
        id<MTLCommandBuffer> cb = [_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        uint32_t elements = (uint32_t)elements64;
        if (!cb || !enc ||
            metal_shared_pointer(gate_scratch, (size_t)elements64,
                &gate_buffer, &gate_offset) != 0 ||
            metal_shared_pointer(up_scratch, (size_t)elements64,
                &up_buffer, &up_offset) != 0)
            goto done;
        [enc setComputePipelineState:_activation_pso];
        [enc setBuffer:gate_buffer offset:gate_offset atIndex:0];
        [enc setBuffer:up_buffer offset:up_offset atIndex:1];
        [enc setBuffer:gate_buffer offset:gate_offset atIndex:2];
        [enc setBytes:&elements length:sizeof elements atIndex:3];
        [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) goto done;
    }
    {
        int row = 0;
        for (int i = 0; i < count; i++)
            for (int g = 0; g < experts[i].group; g++, row++) {
                _moe_vals[row] = experts[i].down_vals;
                _moe_scales[row] = experts[i].down_scales;
                _moe_biases[row] = experts[i].down_biases;
                _moe_ids[row] = experts[i].down_id;
                _moe_xs[row] = gate_scratch + (size_t)row * (size_t)routed;
                _moe_ys[row] = outputs + (size_t)row * (size_t)hidden;
                _moe_rows[row] = hidden;
            }
    }
    if (salt_gpu_q4_batch(_moe_vals, _moe_scales, _moe_biases,
            _moe_xs, _moe_ys, _moe_ids, _moe_rows,
            routed, total_rows) != 0 || salt_gpu_sync() != 0)
        goto done;
    rc = 0;
done:
    return rc;
}

static int metal_selected_encode(id<MTLComputeCommandEncoder> encoder,
                                 id<MTLBuffer> resource_arguments,
                                 id<MTLBuffer> input_buffer,
                                 id<MTLBuffer> output_buffer,
                                 NSUInteger descriptor_offset,
                                 int descriptor_count,
                                 int max_rows, int max_batch) {
    if (!encoder || !resource_arguments || !input_buffer || !output_buffer ||
        descriptor_count < 1 || descriptor_count > 256 ||
        max_rows < 1 || max_batch < 1 || !_bselected_pso ||
        !_selected_desc || !_selected_logical_slots ||
        !_selected_logical_payload_offsets || !_op_status ||
        descriptor_offset > [_selected_desc length] ||
        (NSUInteger)descriptor_count * sizeof(SelectedDesc) >
            [_selected_desc length] - descriptor_offset)
        return -1;
    uint nd = (uint)descriptor_count;
    [encoder setComputePipelineState:_bselected_pso];
    [encoder setBuffer:resource_arguments offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:_selected_desc offset:descriptor_offset atIndex:3];
    [encoder setBytes:&nd length:sizeof(nd) atIndex:4];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:_selected_logical_payload_offsets offset:0 atIndex:6];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder dispatchThreads:
        MTLSizeMake((NSUInteger)max_rows, (NSUInteger)max_batch,
                    (NSUInteger)descriptor_count)
         threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    return 0;
}

static int metal_selected_encode_ragged(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> resource_arguments, id<MTLBuffer> input_buffer,
        id<MTLBuffer> output_buffer, NSUInteger descriptor_offset,
        int descriptor_count, int max_rows, NSUInteger task_offset,
        uint32_t task_count) {
    uint32_t counts[2];
    if (!encoder || !resource_arguments || !input_buffer || !output_buffer ||
        descriptor_count < 1 || descriptor_count > 256 || max_rows < 1 ||
        task_count < 1 || !_bselected_ragged_pso || !_selected_desc ||
        !_selected_task_map || !_selected_logical_slots ||
        !_selected_logical_payload_offsets || !_op_status ||
        descriptor_offset > [_selected_desc length] ||
        (NSUInteger)descriptor_count * sizeof(SelectedDesc) >
            [_selected_desc length] - descriptor_offset ||
        task_offset > [_selected_task_map length] ||
        (NSUInteger)task_count * sizeof(MetalSelectedTask) >
            [_selected_task_map length] - task_offset)
        return -1;
    counts[0] = (uint32_t)descriptor_count;
    counts[1] = task_count;
    [encoder setComputePipelineState:_bselected_ragged_pso];
    [encoder setBuffer:resource_arguments offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:_selected_desc offset:descriptor_offset atIndex:3];
    [encoder setBytes:counts length:sizeof counts atIndex:4];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:_selected_logical_payload_offsets offset:0 atIndex:6];
    [encoder setBuffer:_selected_task_map offset:task_offset atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder dispatchThreads:
        MTLSizeMake((NSUInteger)max_rows, (NSUInteger)task_count, 1)
         threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    return 0;
}

int salt_gpu_q4_selected_shared(
        const SaltGpuSelectedProjectionJob *jobs, int job_count,
        SaltGpuSharedBuffer *arena) {
    id<MTLBuffer> storage;
    SelectedDesc *desc;
    uintptr_t arena_base;
    int max_rows = 0, max_batch = 0;
    uint64_t logical_jobs = 0;
    if (!jobs || job_count < 1 || job_count > 256 || !arena ||
        !arena->contents || !arena->backend || arena->nbytes < 1 ||
        !_bselected_pso || !_bqueue2 || !_wave_selected_args ||
        !_selected_desc || !_op_status ||
        !(desc = (SelectedDesc *)[_selected_desc contents]))
        return -1;
    storage = (id<MTLBuffer>)arena->backend;
    if ([storage contents] != arena->contents ||
        (size_t)[storage length] < arena->nbytes)
        return -1;
    arena_base = (uintptr_t)arena->contents;
    for (int job = 0; job < job_count; job++) {
        const SaltGpuSelectedProjectionJob *entry = &jobs[job];
        MetalSelectedCacheResource *resource;
        size_t value_bytes, scale_bytes;
        uint64_t value_offset, scale_offset, bias_offset;
        uintptr_t input, output;
        size_t input_bytes, output_bytes, input_offset, output_offset;
        if (!entry->vals || !entry->scales || !entry->biases ||
            !entry->inputs || !entry->outputs || entry->rows < 1 ||
            entry->cols < 1 || entry->batch < 1 ||
            entry->resource_slot < 0 ||
            entry->resource_slot >= _selected_cache_capacity ||
            entry->logical_resource_id >=
                (uint64_t)(uint32_t)_selected_logical_capacity ||
            logical_jobs > UINT64_MAX - (uint32_t)entry->batch ||
            batch_weight_sizes(entry->rows, entry->cols,
                &value_bytes, &scale_bytes) != 0)
            return -1;
        resource = &_selected_cache_resources[entry->resource_slot];
        if (!resource->map || !resource->base || resource->nbytes < 1 ||
            resource->logical_resource_id != entry->logical_resource_id ||
            !_selected_logical_slot_host ||
            _selected_logical_slot_host[entry->logical_resource_id] !=
                entry->resource_slot ||
            mapped_range_offset(resource->base, resource->nbytes,
                entry->vals, value_bytes, &value_offset) != 0 ||
            mapped_range_offset(resource->base, resource->nbytes,
                entry->scales, scale_bytes, &scale_offset) != 0 ||
            mapped_range_offset(resource->base, resource->nbytes,
                entry->biases, scale_bytes, &bias_offset) != 0 ||
            (size_t)entry->cols > SIZE_MAX / (size_t)entry->batch /
                sizeof(float) ||
            (size_t)entry->rows > SIZE_MAX / (size_t)entry->batch /
                sizeof(float))
            return -1;
        input_bytes = (size_t)entry->cols * (size_t)entry->batch *
            sizeof(float);
        output_bytes = (size_t)entry->rows * (size_t)entry->batch *
            sizeof(float);
        input = (uintptr_t)(const void *)entry->inputs;
        output = (uintptr_t)(void *)entry->outputs;
        if (input < arena_base || output < arena_base ||
            input - arena_base > SIZE_MAX || output - arena_base > SIZE_MAX)
            return -1;
        input_offset = (size_t)(input - arena_base);
        output_offset = (size_t)(output - arena_base);
        if (input_offset > arena->nbytes ||
            input_bytes > arena->nbytes - input_offset ||
            output_offset > arena->nbytes ||
            output_bytes > arena->nbytes - output_offset ||
            (input_offset < output_offset + output_bytes &&
             output_offset < input_offset + input_bytes) ||
            (input_offset | output_offset) % sizeof(float) != 0)
            return -1;
        for (int prior = 0; prior < job; prior++) {
            uintptr_t prior_output = (uintptr_t)(void *)jobs[prior].outputs;
            size_t prior_bytes = (size_t)jobs[prior].rows *
                (size_t)jobs[prior].batch * sizeof(float);
            if (output < prior_output + prior_bytes &&
                prior_output < output + output_bytes)
                return -1;
        }
        desc[job] = (SelectedDesc) {
            {value_offset, scale_offset, input_offset / sizeof(float),
             output_offset / sizeof(float)},
            {bias_offset, (uint64_t)(uint32_t)entry->rows,
             (uint64_t)(uint32_t)entry->cols,
             (uint64_t)(uint32_t)entry->batch},
            {(uint64_t)(uint32_t)entry->resource_slot, 0u, 0u, 0u},
        };
        if (entry->rows > max_rows) max_rows = entry->rows;
        if (entry->batch > max_batch) max_batch = entry->batch;
        logical_jobs += (uint32_t)entry->batch;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [_bqueue2 commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1;
        *(uint32_t *)[_op_status contents] = 0u;
        for (int job = 0; job < job_count; job++)
            [enc useResource:_selected_cache_resources[
                jobs[job].resource_slot].map usage:MTLResourceUsageRead];
        if (metal_selected_encode(enc, _wave_selected_args, storage, storage,
                0, job_count, max_rows, max_batch) != 0) {
            [enc endEncoding];
            return -1;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0 ||
            *(uint32_t *)[_op_status contents] != 0u)
            return -1;
    }
    _batch_stats.expert_layer_batches++;
    _batch_stats.direct_output_batches++;
    _batch_stats.direct_output_jobs += logical_jobs;
    return 0;
}

int salt_gpu_q4_moe_chain_selected(const SaltGpuMoeExpert *experts, int count,
                                   int hidden, int routed,
                                   const float *inputs, float *outputs,
                                   float *gate_scratch, float *up_scratch,
                                   size_t scratch_float_capacity) {
    id<MTLBuffer> views[128] = {nil};
    uint8_t transient_views[128] = {0};
    id<MTLBuffer> input_buffer, output_buffer, gate_buffer, up_buffer;
    size_t input_offset, output_offset, gate_offset, up_offset;
    uint64_t total_rows = 0;
    int max_group = 0, rc = -1;
    long page_l;
    int detail = getenv("SALT_GPU_MOE_MS_DETAIL") != NULL;
    struct timespec detail_start, detail_prepared, detail_encoded;
    struct timespec detail_committed, detail_waited;
    static unsigned long long detail_sequence = 0;
    unsigned long long detail_id = 0;
    if (!experts || count < 1 || count > 128 || hidden < 1 || routed < 1 ||
        (hidden & 31) != 0 || (routed & 31) != 0 || !inputs || !outputs ||
        !gate_scratch || !up_scratch || batch_pipeline_prepare() != 0 ||
        metal_ops_prepare() != 0 || (page_l = sysconf(_SC_PAGESIZE)) <= 0)
        return -1;
    if (detail) {
        detail_id = ++detail_sequence;
        clock_gettime(CLOCK_MONOTONIC, &detail_start);
    }
    for (int i = 0; i < count; i++) {
        if (experts[i].group < 1 ||
            total_rows > UINT64_MAX - (uint32_t)experts[i].group)
            return -1;
        total_rows += (uint32_t)experts[i].group;
        if (experts[i].group > max_group) max_group = experts[i].group;
    }
    if (total_rows > SIZE_MAX / (size_t)hidden ||
        total_rows > SIZE_MAX / (size_t)routed ||
        total_rows * (size_t)routed > scratch_float_capacity ||
        metal_shared_pointer(inputs, (size_t)total_rows * hidden,
            &input_buffer, &input_offset) != 0 ||
        metal_shared_pointer(outputs, (size_t)total_rows * hidden,
            &output_buffer, &output_offset) != 0 ||
        metal_shared_pointer(gate_scratch, (size_t)total_rows * routed,
            &gate_buffer, &gate_offset) != 0 ||
        metal_shared_pointer(up_scratch, (size_t)total_rows * routed,
            &up_buffer, &up_offset) != 0 || gate_buffer != up_buffer ||
        input_offset % sizeof(float) != 0 ||
        output_offset % sizeof(float) != 0 ||
        gate_offset % sizeof(float) != 0 || up_offset % sizeof(float) != 0)
        return -1;
    @autoreleasepool {
        size_t gate_vbytes, gate_sbytes, down_vbytes, down_sbytes;
        SelectedDesc *desc = (SelectedDesc *)[_selected_desc contents];
        uint64_t row_prefix = 0;
        if (!desc || batch_weight_sizes(routed, hidden,
                &gate_vbytes, &gate_sbytes) != 0 ||
            batch_weight_sizes(hidden, routed,
                &down_vbytes, &down_sbytes) != 0)
            goto done;
        for (int i = 0; i < count; i++) {
            const SaltGpuMoeExpert *entry = &experts[i];
            uintptr_t lo = (uintptr_t)(const void *)entry->gate_vals;
            uintptr_t hi;
            const uintptr_t starts[8] = {
                (uintptr_t)(const void *)entry->gate_scales,
                (uintptr_t)(const void *)entry->gate_biases,
                (uintptr_t)(const void *)entry->up_vals,
                (uintptr_t)(const void *)entry->up_scales,
                (uintptr_t)(const void *)entry->up_biases,
                (uintptr_t)(const void *)entry->down_vals,
                (uintptr_t)(const void *)entry->down_scales,
                (uintptr_t)(const void *)entry->down_biases,
            };
            const size_t sizes[8] = {
                gate_sbytes, gate_sbytes, gate_vbytes, gate_sbytes,
                gate_sbytes, down_vbytes, down_sbytes, down_sbytes,
            };
            if (!entry->gate_vals || !entry->gate_scales ||
                !entry->gate_biases || !entry->up_vals || !entry->up_scales ||
                !entry->up_biases || !entry->down_vals || !entry->down_scales ||
                !entry->down_biases)
                goto done;
            if (lo > UINTPTR_MAX - gate_vbytes) goto done;
            hi = lo + gate_vbytes;
            for (int part = 0; part < 8; part++) {
                if (starts[part] < lo) lo = starts[part];
                if (starts[part] > UINTPTR_MAX - sizes[part]) goto done;
                if (starts[part] + sizes[part] > hi)
                    hi = starts[part] + sizes[part];
            }
            uintptr_t aligned = lo - lo % (uintptr_t)page_l;
            if (hi <= aligned || hi - aligned > NSUIntegerMax) goto done;
            if (entry->resource_slot >= 0) {
                if (!_selected_cache_resources ||
                    entry->resource_slot >= _selected_cache_capacity ||
                    entry->logical_resource_id >=
                        (uint64_t)(uint32_t)_selected_logical_capacity ||
                    !_selected_logical_slot_host)
                    goto done;
                MetalSelectedCacheResource *resource =
                    &_selected_cache_resources[entry->resource_slot];
                uintptr_t resource_lo = (uintptr_t)(const void *)resource->base;
                if (!resource->map || !resource->base || resource->nbytes < 1 ||
                    resource->logical_resource_id !=
                        entry->logical_resource_id ||
                    _selected_logical_slot_host[entry->logical_resource_id] !=
                        entry->resource_slot ||
                    resource->nbytes > UINTPTR_MAX - resource_lo ||
                    lo < resource_lo || hi > resource_lo + resource->nbytes)
                    goto done;
                views[i] = resource->map;
                aligned = resource_lo;
            } else {
                views[i] = [_dev newBufferWithBytesNoCopy:(void *)aligned
                    length:(NSUInteger)(hi - aligned)
                    options:MTLResourceStorageModeShared deallocator:nil];
                if (!views[i] || [views[i] contents] != (void *)aligned)
                    goto done;
                transient_views[i] = 1;
            }
            [_selected_arg_encoder setBuffer:views[i] offset:0 atIndex:(NSUInteger)i];
            uint64_t xoff = input_offset / sizeof(float) + row_prefix * hidden;
            uint64_t goff = gate_offset / sizeof(float) + row_prefix * routed;
            uint64_t uoff = up_offset / sizeof(float) + row_prefix * routed;
            desc[2 * i] = (SelectedDesc) {
                {(uintptr_t)entry->gate_vals - aligned,
                 (uintptr_t)entry->gate_scales - aligned, xoff, goff},
                {(uintptr_t)entry->gate_biases - aligned,
                 (uint64_t)routed, (uint64_t)hidden,
                 (uint64_t)(uint32_t)entry->group},
                {(uint64_t)(uint32_t)i, 0, 0, 0},
            };
            desc[2 * i + 1] = (SelectedDesc) {
                {(uintptr_t)entry->up_vals - aligned,
                 (uintptr_t)entry->up_scales - aligned, xoff, uoff},
                {(uintptr_t)entry->up_biases - aligned,
                 (uint64_t)routed, (uint64_t)hidden,
                 (uint64_t)(uint32_t)entry->group},
                {(uint64_t)(uint32_t)i, 0, 0, 0},
            };
            row_prefix += (uint32_t)entry->group;
        }
        SelectedDesc *down_desc = (SelectedDesc *)((unsigned char *)desc +
            METAL_SELECTED_DESC_DOWN_OFFSET);
        row_prefix = 0;
        for (int i = 0; i < count; i++) {
            const SaltGpuMoeExpert *entry = &experts[i];
            uintptr_t aligned = (uintptr_t)[views[i] contents];
            down_desc[i] = (SelectedDesc) {
                {(uintptr_t)entry->down_vals - aligned,
                 (uintptr_t)entry->down_scales - aligned,
                 gate_offset / sizeof(float) + row_prefix * routed,
                 output_offset / sizeof(float) + row_prefix * hidden},
                {(uintptr_t)entry->down_biases - aligned,
                 (uint64_t)hidden, (uint64_t)routed,
                 (uint64_t)(uint32_t)entry->group},
                {(uint64_t)(uint32_t)i, 0, 0, 0},
            };
            row_prefix += (uint32_t)entry->group;
        }
        if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_prepared);
        {
            id<MTLCommandBuffer> cb = [_bqueue2 commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            uint32_t elements = (uint32_t)(total_rows * (uint32_t)routed);
            if (!cb || !enc) goto done;
            for (int i = 0; i < count; i++)
                [enc useResource:views[i] usage:MTLResourceUsageRead];
            if (metal_selected_encode(enc, _selected_args,
                    input_buffer, gate_buffer, 0,
                    2 * count, routed, max_group) != 0) {
                [enc endEncoding];
                goto done;
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:_activation_pso];
            [enc setBuffer:gate_buffer offset:gate_offset atIndex:0];
            [enc setBuffer:up_buffer offset:up_offset atIndex:1];
            [enc setBuffer:gate_buffer offset:gate_offset atIndex:2];
            [enc setBytes:&elements length:sizeof(elements) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (metal_selected_encode(enc, _selected_args,
                    gate_buffer, output_buffer,
                    METAL_SELECTED_DESC_DOWN_OFFSET,
                    count, hidden, max_group) != 0) {
                [enc endEncoding];
                goto done;
            }
            [enc endEncoding];
            if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_encoded);
            [cb commit];
            if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_committed);
            [cb waitUntilCompleted];
            if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_waited);
            if (metal_command_succeeded(cb) != 0) goto done;
            if (detail) {
                double gpu_start = [cb GPUStartTime];
                double gpu_end = [cb GPUEndTime];
                double gpu_ms = gpu_end >= gpu_start
                    ? (gpu_end - gpu_start) * 1e3 : 0.0;
#define G4_MOE_MS(a, b) \
                ((double)((b).tv_sec - (a).tv_sec) * 1e3 + \
                 (double)((b).tv_nsec - (a).tv_nsec) * 1e-6)
                fprintf(stderr,
                    "[gpu-moe-ms-detail] seq=%llu experts=%d rows=%llu "
                    "prepare_ms=%.3f encode_ms=%.3f commit_ms=%.3f "
                    "wait_ms=%.3f gpu_ms=%.3f total_ms=%.3f\n",
                    detail_id, count, (unsigned long long)total_rows,
                    G4_MOE_MS(detail_start, detail_prepared),
                    G4_MOE_MS(detail_prepared, detail_encoded),
                    G4_MOE_MS(detail_encoded, detail_committed),
                    G4_MOE_MS(detail_committed, detail_waited),
                    gpu_ms,
                    G4_MOE_MS(detail_start, detail_waited));
#undef G4_MOE_MS
            }
        }
        if (getenv("SALT_GPU_DIAG")) {
            fprintf(stderr, "[gpu-selected] chain=%a up=%a output=%a\n",
                    gate_scratch[0], up_scratch[0], outputs[0]);
        }
        _batch_stats.expert_layer_batches += 2;
        _batch_stats.direct_output_batches += 2;
        _batch_stats.direct_output_jobs += 3u * total_rows;
        rc = 0;
done:
        for (int i = 0; i < count; i++) {
            [_selected_arg_encoder setBuffer:nil offset:0 atIndex:(NSUInteger)i];
            if (transient_views[i]) metal_release(views[i]);
        }
    }
    return rc;
}

static int salt_gpu_free_impl(void) {
    if (salt_gpu_sync() != 0) {
        fprintf(stderr, "gpu: refusing teardown with pending work\n");
        return -1;
    }
    if (((_trunk_map == nil) != (_trunk_base == NULL)) ||
        ((_trunk_map == nil) != (_trunk_len == 0)) ||
        ((_res_map == nil) != (_res_base == NULL)) ||
        ((_res_map == nil) != (_res_len == 0))) {
        fprintf(stderr, "gpu: refusing teardown with inconsistent mapping state\n");
        return -1;
    }
    for (int i = 0; i < _selected_cache_capacity; i++) {
        if (_selected_cache_resources[i].map) {
            fprintf(stderr,
                    "gpu: refusing teardown with bound selected cache slot\n");
            return -1;
        }
    }
    for (int logical = 0; logical < _selected_logical_capacity; logical++)
        if (!_selected_logical_slot_host ||
            _selected_logical_slot_host[logical] != -1) {
            fprintf(stderr,
                    "gpu: refusing teardown with stale logical expert slot\n");
            return -1;
        }

    if (_trunk_map) {
        tensor_slots_forget(SALT_GPU_RESOURCE_TRUNK, 0);
        metal_release(_trunk_map);
        _trunk_map = nil;
        _trunk_base = NULL;
        _trunk_len = 0;
    }
    if (_res_map) {
        tensor_slots_forget(SALT_GPU_RESOURCE_EXPERT_LAYER, _res_id);
        metal_release(_res_map);
        _res_map = nil;
        _res_base = NULL;
        _res_len = 0;
        _res_id = 0;
    }
    for (int i = 0; i < METAL_WEIGHT_RESOURCES; i++) {
        if (!_weight_resources[i].used) continue;
        tensor_slots_forget(_weight_resources[i].kind,
                            _weight_resources[i].resource_id);
        metal_release(_weight_resources[i].map);
        memset(&_weight_resources[i], 0, sizeof _weight_resources[i]);
    }
    _weight_peak_window_bytes = 0;
    _component_pool_bytes = 0;
    _component_pool_peak_bytes = 0;

    metal_release(_bv); _bv = nil;
    metal_release(_bs); _bs = nil;
    metal_release(_bb); _bb = nil;
    metal_release(_bx); _bx = nil;
    metal_release(_by); _by = nil;
    for (int i = 0; i < 2; i++) {
        metal_release(_bx2[i]); _bx2[i] = nil;
        metal_release(_by2[i]); _by2[i] = nil;
        _bx2_len[i] = 0;
        _by2_len[i] = 0;
    }
    metal_release(_arena_v); _arena_v = nil;
    metal_release(_arena_s); _arena_s = nil;
    metal_release(_arena_b); _arena_b = nil;
    metal_release(_desc); _desc = nil;
    metal_release(_bpso2); _bpso2 = nil;
    metal_release(_bweight_pso); _bweight_pso = nil;
    metal_release(_bcoalesced_pso); _bcoalesced_pso = nil;
    metal_release(_bselected_pso); _bselected_pso = nil;
    metal_release(_bselected_ragged_pso); _bselected_ragged_pso = nil;
    metal_release(_selected_arg_encoder); _selected_arg_encoder = nil;
    metal_release(_selected_args); _selected_args = nil;
    metal_release(_selected_desc); _selected_desc = nil;
    metal_release(_selected_task_map); _selected_task_map = nil;
    metal_release(_wave_selected_arg_encoder);
    _wave_selected_arg_encoder = nil;
    metal_release(_wave_selected_args); _wave_selected_args = nil;
    metal_release(_selected_logical_slots); _selected_logical_slots = nil;
    metal_release(_selected_logical_payload_offsets);
    _selected_logical_payload_offsets = nil;
    _selected_logical_slot_host = NULL;
    _selected_logical_payload_host = NULL;
    _selected_logical_capacity = 0;
    metal_release(_btiled_pso); _btiled_pso = nil;
    metal_release(_bqueue2); _bqueue2 = nil;
    metal_release(_transform_pso); _transform_pso = nil;
    metal_release(_attention_pso); _attention_pso = nil;
    metal_release(_activation_pso); _activation_pso = nil;
    metal_release(_q8_pso); _q8_pso = nil;
    metal_release(_text_embedding_pso); _text_embedding_pso = nil;
    metal_release(_text_route_maps_pso); _text_route_maps_pso = nil;
    metal_release(_text_routed_gather_pso); _text_routed_gather_pso = nil;
    metal_release(_text_expert_reduce_pso); _text_expert_reduce_pso = nil;
    metal_release(_text_combine_pso); _text_combine_pso = nil;
    metal_release(_text_rms_rows_pso); _text_rms_rows_pso = nil;
    metal_release(_text_residual_rows_pso); _text_residual_rows_pso = nil;
    metal_release(_text_router_rows_pso); _text_router_rows_pso = nil;
    metal_release(_text_topk_rows_pso); _text_topk_rows_pso = nil;
    metal_release(_text_softcap_pso); _text_softcap_pso = nil;
    metal_release(_text_transform_view_pso); _text_transform_view_pso = nil;
    metal_release(_text_attention_view_pso); _text_attention_view_pso = nil;
    metal_release(_exact_rms_pso); _exact_rms_pso = nil;
    metal_release(_exact_topk_pso); _exact_topk_pso = nil;
    metal_release(_exact_residual_pso); _exact_residual_pso = nil;
    metal_release(_exact_combine_pso); _exact_combine_pso = nil;
    metal_release(_exact_softcap_pso); _exact_softcap_pso = nil;
    metal_release(_exact_router_pso); _exact_router_pso = nil;
    metal_release(_op_meta); _op_meta = nil;
    metal_release(_op_status); _op_status = nil;
    metal_release(_q8_x); _q8_x = nil;
    metal_release(_q8_y); _q8_y = nil;
    free(_selected_cache_resources);
    _selected_cache_resources = NULL;
    _selected_cache_capacity = 0;
    _selected_retirement_fenced = 0;

    metal_release(_pso); _pso = nil;
    metal_release(_queue); _queue = nil;
    metal_release(_offline_library); _offline_library = nil;
    metal_release(_dev); _dev = nil;

    memset(_tslot, 0, sizeof _tslot);
    _tslot_n = 0;
    _arena_v_used = 0;
    _arena_s_used = 0;
    _desc_len = 0;
    _mapped_only = 0;
    _defer = 0;
    _pend_cb = nil;
    _pend_n = 0;
    _pend_bi = 0;
    _direct_pending_n = 0;
    memset(_direct_pending, 0, sizeof _direct_pending);
    memset(_pend_ys, 0, sizeof _pend_ys);
    memset(_pend_Rj, 0, sizeof _pend_Rj);
    memset(&_batch_stats, 0, sizeof _batch_stats);
    memset(_shared_slots, 0, sizeof _shared_slots);
    _tiled_simdgroups = 0;
    _tiled_token_span = 0;
    _tiled_max_descriptors = 0;
    _weight_stationary_min_b = 0;
    _tiled_configured = 0;
    _R = _C = -1;
    _has_bias = 0;
    return 0;
}

/* ---- the dispatch managers -------------------------------------- */
/* The direct manager: the per-call encode (the baseline). */
static int dispatch_direct_init(void **state) { (void)state; return 0; }
static int dispatch_direct_replay(void *state, const SaltGpuBatchDesc *d) {
    (void)state;
    return salt_gpu_q4_batch(d->vals, d->scales, d->biases, d->xs, d->ys,
                             d->ids, d->Rj, d->C, d->njobs);
}
static void dispatch_direct_destroy(void *state) { (void)state; }

/* The old ConcurrentDispatch ICB record path crashes on this Metal.
 * Keep explicit selection fail-closed until a qualified replacement lands. */
static int dispatch_icb_replay(void *state, const SaltGpuBatchDesc *d) {
    (void)state;
    (void)d;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-fail] metal-icb unsupported\n");
    return -1;
}
static void dispatch_icb_destroy(void *state) { (void)state; }

static const SaltGpuDispatch salt_dispatch_direct = {
    "direct", dispatch_direct_init, dispatch_direct_replay,
    dispatch_direct_destroy };
static const SaltGpuDispatch salt_dispatch_icb = {
    "metal-icb", dispatch_direct_init, dispatch_icb_replay,
    dispatch_icb_destroy };

int salt_gpu_q8_matvec(const uint32_t *vals, const uint16_t *scales,
                       const uint16_t *biases, int R, int C,
                       const float *x, float *y) {
    const uint32_t *vj[1] = {vals};
    const uint16_t *sj[1] = {scales}, *bj[1] = {biases};
    const float *xj[1] = {x};
    float *yj[1] = {y};
    const void *ids[1] = {vals};
    int rows[1] = {R};
    return salt_gpu_q8_batch(vj, sj, bj, xj, yj, ids, rows, C, 1);
}

int salt_gpu_nvfp4_qdq(const float *input, int C, float input_global_scale,
                       uint8_t *packed, uint8_t *scales, float *dequant) {
    (void)input; (void)C; (void)input_global_scale;
    (void)packed; (void)scales; (void)dequant;
    return -1;
}

int salt_gpu_nvfp4_matvec(const void *key, int R, int C,
                          const float *input, float *output) {
    (void)key; (void)R; (void)C; (void)input; (void)output;
    return -1;
}

int salt_gpu_nvfp4_matvec_batch(const void *key, int R, int C, int B,
                                const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return -1;
}

int salt_gpu_nvfp4_batch(const void *const *keys,
                         const float *const *inputs, float *const *outputs,
                         const int *batches, int R, int C, int njobs) {
    (void)keys; (void)inputs; (void)outputs; (void)batches;
    (void)R; (void)C; (void)njobs;
    return -1;
}

int salt_gpu_nvfp4_mixed_batch(const void *const *keys,
                               const float *const *inputs,
                               float *const *outputs, const int *batches,
                               const int *rows, int C, int njobs) {
    (void)keys; (void)inputs; (void)outputs; (void)batches; (void)rows;
    (void)C; (void)njobs;
    return -1;
}

int salt_gpu_bf16_resource_slot(uint32_t kind, uint32_t resource_id,
                                const void *key, const uint16_t *weights,
                                int R, int C) {
    (void)kind; (void)resource_id; (void)key; (void)weights; (void)R; (void)C;
    return -1;
}
int salt_gpu_bf16_matvec(const void *key, int R, int C,
                         const float *input, float *output) {
    (void)key; (void)R; (void)C; (void)input; (void)output;
    return -1;
}

int salt_gpu_bf16_matvec_batch(const void *key, int R, int C, int B,
                               const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return -1;
}

int salt_gpu_q8_batch(const uint32_t *const *vals,
                      const uint16_t *const *scales,
                      const uint16_t *const *biases,
                      const float *const *xs, float *const *ys,
                      const void *const *ids,
                      const int *Rj, int C, int njobs) {
    id<MTLBuffer> xbuffer = _q8_x, ybuffer = _q8_y;
    const SaltGpuResourceRef *resource = NULL;
    MetalWeightResource *mapped;
    id<MTLBuffer> operation_view = nil;
    uint64_t operation_base = 0;
    uint64_t value_base = 0, scale_base = 0, bias_base = 0;
    int R, rc = -1, pooled_components = 0;
    int *slots = _q8_slots;
    if (!vals || !scales || !biases || !xs || !ys || !ids || !Rj ||
        njobs < 1 || njobs > BATCH_MAXSLOT || C < 1 ||
        metal_ops_prepare() != 0)
        return -1;
    R = Rj[0];
    if (R < 1) return -1;

    for (int j = 0; j < njobs; j++) {
        if (Rj[j] != R || !vals[j] || !scales[j] || !biases[j] ||
            !xs[j] || !ys[j] || !ids[j]) goto q8_done;
        slots[j] = -1;
        for (int i = 0; i < _tslot_n; i++)
            if (_tslot[i].key == ids[j]) { slots[j] = i; break; }
        if (slots[j] < 0) goto q8_done;
    }
    if (batch_resource_for_slots(slots, njobs, &resource) != 0 ||
        !(mapped = metal_weight_resource(resource->kind,
                                         resource->resource_id)))
        goto q8_done;
    pooled_components = njobs == 1 && _tslot[slots[0]].value_map &&
        _tslot[slots[0]].scale_map && _tslot[slots[0]].bias_map;
    for (int j = 1; j < njobs; j++) {
        const SaltGpuResourceRef *a = &_tslot[slots[0]].resource;
        const SaltGpuResourceRef *b = &_tslot[slots[j]].resource;
        if (a->value_offset != b->value_offset ||
            a->scale_offset != b->scale_offset ||
            a->bias_offset != b->bias_offset)
            goto q8_done;
    }
    {
        const SaltGpuResourceRef *ref = &_tslot[slots[0]].resource;
        uint64_t elements = (uint64_t)(uint32_t)R * (uint32_t)C;
        uint64_t groups = (elements + 63u) / 64u;
        uint64_t lo = ref->value_offset;
        uint64_t hi = ref->value_offset + elements;
        uint64_t send = ref->scale_offset + groups * 2u;
        uint64_t bend = ref->bias_offset + groups * 2u;
        long page_l = sysconf(_SC_PAGESIZE);
        if (ref->scale_offset < lo) lo = ref->scale_offset;
        if (ref->bias_offset < lo) lo = ref->bias_offset;
        if (send > hi) hi = send;
        if (bend > hi) hi = bend;
        if (page_l <= 0 || hi <= lo || hi > mapped->nbytes) goto q8_done;
        if (pooled_components) {
            value_base = _tslot[slots[0]].value_base;
            scale_base = _tslot[slots[0]].scale_base;
            bias_base = _tslot[slots[0]].bias_base;
        } else if (mapped->map) {
            if (lo < mapped->map_offset ||
                mapped->map_nbytes > UINT64_MAX - mapped->map_offset ||
                hi > mapped->map_offset + mapped->map_nbytes)
                goto q8_done;
            operation_view = [mapped->map retain];
            operation_base = mapped->map_offset;
            value_base = scale_base = bias_base = operation_base;
        } else {
            uint64_t page = (uint64_t)page_l;
            operation_base = lo - lo % page;
            uint64_t length = hi - operation_base;
            operation_view = [_dev newBufferWithBytesNoCopy:
                (void *)(mapped->base + operation_base) length:(NSUInteger)length
                options:MTLResourceStorageModeShared deallocator:nil];
            if (!operation_view)
                operation_view = [_dev newBufferWithBytesNoCopy:
                    (void *)(mapped->base + operation_base)
                    length:(NSUInteger)length
                    options:MTLResourceStorageModeManaged deallocator:nil];
            if (!operation_view) goto q8_done;
            value_base = scale_base = bias_base = operation_base;
        }
    }
    {
        size_t xcount = (size_t)njobs * (size_t)C;
        size_t ycount = (size_t)njobs * (size_t)R;
        if (xcount > SIZE_MAX / sizeof(float) ||
            ycount > SIZE_MAX / sizeof(float) || !xbuffer || !ybuffer ||
            xcount * sizeof(float) > (size_t)[xbuffer length] ||
            ycount * sizeof(float) > (size_t)[ybuffer length]) goto q8_done;
        @autoreleasepool {
            for (int j = 0; j < njobs; j++) {
                memcpy((float *)[xbuffer contents] + (size_t)j * C,
                       xs[j], (size_t)C * sizeof(float));
            }
            const SaltGpuResourceRef *ref = &_tslot[slots[0]].resource;
            uint64_t voff = ref->value_offset - value_base;
            uint64_t soff = ref->scale_offset - scale_base;
            uint64_t boff = ref->bias_offset - bias_base;
            struct { uint32_t rows, cols, jobs; } args = {
                (uint32_t)R, (uint32_t)C, (uint32_t)njobs,
            };
            id<MTLCommandBuffer> cb = [_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!cb || !enc) goto q8_done;
            [enc setComputePipelineState:_q8_pso];
            if (pooled_components) {
                TensorSlot *pooled = &_tslot[slots[0]];
                [enc setBuffer:pooled->value_map offset:0 atIndex:0];
                [enc setBuffer:pooled->scale_map offset:0 atIndex:1];
                [enc setBuffer:pooled->bias_map offset:0 atIndex:2];
            } else {
                [enc setBuffer:operation_view offset:0 atIndex:0];
                [enc setBuffer:operation_view offset:0 atIndex:1];
                [enc setBuffer:operation_view offset:0 atIndex:2];
            }
            [enc setBuffer:xbuffer offset:0 atIndex:3];
            [enc setBuffer:ybuffer offset:0 atIndex:4];
            [enc setBytes:&voff length:sizeof voff atIndex:5];
            [enc setBytes:&soff length:sizeof soff atIndex:6];
            [enc setBytes:&boff length:sizeof boff atIndex:7];
            [enc setBytes:&args length:sizeof args atIndex:8];
            [enc dispatchThreads:MTLSizeMake(ycount, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (metal_command_succeeded(cb) != 0) goto q8_done;
            for (int j = 0; j < njobs; j++)
                memcpy(ys[j], (float *)[ybuffer contents] + (size_t)j * R,
                       (size_t)R * sizeof(float));
            _batch_stats.trunk_batches++;
            rc = 0;
        }
    }
q8_done:
    metal_release(operation_view);
    return rc;
}

int salt_gpu_weight_resource_describe(uint32_t kind, uint32_t resource_id,
                                      const void *base, size_t nbytes) {
    MetalWeightResource *slot = NULL;
    if (!_dev || !base || nbytes < 1 ||
        kind > SALT_GPU_RESOURCE_EXPERT_LAYER ||
        metal_weight_resource(kind, resource_id))
        return -1;
    for (int i = 0; i < METAL_WEIGHT_RESOURCES; i++)
        if (!_weight_resources[i].used) { slot = &_weight_resources[i]; break; }
    if (!slot || nbytes > (size_t)NSUIntegerMax) return -1;
    slot->used = 1;
    slot->active = 0;
    slot->kind = kind;
    slot->resource_id = resource_id;
    slot->policy = SALT_GPU_WEIGHT_ADDRESS_AUTO;
    slot->map = nil;
    slot->base = (const uint8_t *)base;
    slot->nbytes = nbytes;
    return 0;
}

int salt_gpu_weight_resource_activate(uint32_t kind, uint32_t resource_id,
                                      SaltGpuWeightAddressability policy) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource || resource->active || resource->map ||
        policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW)
        return -1;
    resource->policy = policy;
    resource->active = 1;
    return 0;
}

int salt_gpu_weight_resource_pool_bind(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource || !resource->active || resource->map ||
        resource->policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW ||
        resource->component_pool || !resource->base || resource->nbytes < 1)
        return -1;
    resource->component_pool = 1;
    return 0;
}

int salt_gpu_weight_resource_deactivate(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource || !resource->active || salt_gpu_sync() != 0) return -1;
    metal_release(resource->map);
    resource->map = nil;
    resource->map_offset = 0;
    resource->map_nbytes = 0;
    resource->component_pool = 0;
    resource->active = 0;
    resource->policy = SALT_GPU_WEIGHT_ADDRESS_AUTO;
    return 0;
}

int salt_gpu_weight_resource_usage(SaltGpuWeightResourceUsage *usage,
                                   size_t usage_size) {
    if (!usage || usage_size != sizeof *usage) return -1;
    memset(usage, 0, sizeof *usage);
    for (int i = 0; i < METAL_WEIGHT_RESOURCES; i++) {
        const MetalWeightResource *resource = &_weight_resources[i];
        if (!resource->used) continue;
        usage->described_resources++;
        usage->described_bytes += resource->nbytes;
        if (resource->active) usage->active_resources++;
        if (resource->map) {
            usage->active_windows++;
            usage->active_window_bytes += resource->map_nbytes;
        }
        if (resource->component_pool) usage->active_windows++;
    }
    if (usage->active_window_bytes > UINT64_MAX - _component_pool_bytes)
        return -1;
    usage->active_window_bytes += _component_pool_bytes;
    usage->peak_window_bytes = _weight_peak_window_bytes;
    if (usage->peak_window_bytes < _component_pool_peak_bytes)
        usage->peak_window_bytes = _component_pool_peak_bytes;
    return 0;
}

int salt_gpu_weight_resource_bind(uint32_t kind, uint32_t resource_id,
                                  const void *base, size_t nbytes) {
    if (salt_gpu_weight_resource_describe(kind, resource_id, base, nbytes) != 0)
        return -1;
    if (salt_gpu_weight_resource_activate(kind, resource_id,
            SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW) != 0) {
        MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
        if (resource) memset(resource, 0, sizeof *resource);
        return -1;
    }
    return 0;
}

int salt_gpu_weight_window_bind(uint32_t kind, uint32_t resource_id,
                                uint64_t offset, size_t nbytes) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    long page_l;
    if (!resource || !resource->active ||
        resource->policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW ||
        resource->component_pool || resource->map || nbytes < 1 ||
        offset > resource->nbytes || nbytes > resource->nbytes - (size_t)offset ||
        (page_l = sysconf(_SC_PAGESIZE)) <= 0 ||
        ((uintptr_t)(const void *)(resource->base + offset) %
            (uintptr_t)page_l) != 0)
        return -1;
    @autoreleasepool {
        id<MTLBuffer> map = [_dev newBufferWithBytesNoCopy:
            (void *)(resource->base + offset) length:nbytes
            options:MTLResourceStorageModeShared deallocator:nil];
        if (!map)
            map = [_dev newBufferWithBytesNoCopy:
                (void *)(resource->base + offset) length:nbytes
                options:MTLResourceStorageModeManaged deallocator:nil];
        if (!map || [map contents] != resource->base + offset) {
            metal_release(map);
            return -1;
        }
        resource->map = map;
    }
    resource->map_offset = offset;
    resource->map_nbytes = nbytes;
    if ((uint64_t)nbytes > _weight_peak_window_bytes)
        _weight_peak_window_bytes = nbytes;
    return 0;
}

int salt_gpu_weight_window_unbind(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource || !resource->active || !resource->map ||
        salt_gpu_sync() != 0) return -1;
    metal_release(resource->map);
    resource->map = nil;
    resource->map_offset = 0;
    resource->map_nbytes = 0;
    return 0;
}

int salt_gpu_weight_resource_slot(uint32_t kind, uint32_t resource_id,
                                  const void *key, const uint32_t *vals,
                                  const uint16_t *scales,
                                  const uint16_t *biases,
                                  int bits, int R, int C) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    size_t vbytes, sbytes;
    uint64_t voff, soff, boff;
    int slot_index;
    if (!resource || !resource->active || !key || !vals || !scales || !biases ||
        (bits != 4 && bits != 8) || R < 1 || C < 1)
        return -1;
    if (batch_weight_sizes(R, C, &vbytes, &sbytes) != 0) return -1;
    if (bits == 8) {
        uint64_t elements = (uint64_t)(uint32_t)R * (uint32_t)C;
        if (elements > SIZE_MAX) return -1;
        vbytes = (size_t)elements;
    }
    if (mapped_range_offset(resource->base, resource->nbytes,
            vals, vbytes, &voff) != 0 ||
        mapped_range_offset(resource->base, resource->nbytes,
            scales, sbytes, &soff) != 0 ||
        mapped_range_offset(resource->base, resource->nbytes,
            biases, sbytes, &boff) != 0)
        return -1;
    for (int i = 0; i < _tslot_n; i++) {
        const SaltGpuResourceRef *ref;
        if (_tslot[i].key != key) continue;
        ref = &_tslot[i].resource;
        return ref->kind == kind && ref->resource_id == resource_id &&
            ref->value_offset == voff && ref->scale_offset == soff &&
            ref->bias_offset == boff ? 0 : -1;
    }
    slot_index = tensor_slot_store(key, kind, resource_id, voff, soff, boff);
    if (slot_index < 0) return -1;
    if (resource->component_pool) {
        TensorSlot *slot = &_tslot[slot_index];
        uint64_t pooled;
        if (metal_exact_component_view(resource, voff, vbytes,
                &slot->value_map, &slot->value_base) != 0 ||
            metal_exact_component_view(resource, soff, sbytes,
                &slot->scale_map, &slot->scale_base) != 0 ||
            metal_exact_component_view(resource, boff, sbytes,
                &slot->bias_map, &slot->bias_base) != 0 ||
            (uint64_t)[slot->value_map length] >
                UINT64_MAX - (uint64_t)[slot->scale_map length] ||
            (pooled = (uint64_t)[slot->value_map length] +
                (uint64_t)[slot->scale_map length]) >
                UINT64_MAX - (uint64_t)[slot->bias_map length] ||
            (pooled += (uint64_t)[slot->bias_map length]) > SIZE_MAX ||
            pooled > UINT64_MAX - _component_pool_bytes) {
            metal_release(slot->value_map);
            metal_release(slot->scale_map);
            metal_release(slot->bias_map);
            memset(slot, 0, sizeof *slot);
            _tslot_n--;
            return -1;
        }
        slot->pooled_nbytes = (size_t)pooled;
        _component_pool_bytes += pooled;
        if (_component_pool_bytes > _component_pool_peak_bytes)
            _component_pool_peak_bytes = _component_pool_bytes;
    }
    return 0;
}

int salt_gpu_nvfp4_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    const float *weight_global_scale, const float *input_global_scale,
    int R, int C) {
    (void)kind; (void)resource_id; (void)key;
    (void)weight_packed; (void)weight_scales;
    (void)weight_global_scale; (void)input_global_scale;
    (void)R; (void)C;
    return -1;
}

int salt_gpu_weight_resource_unbind(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource) return -1;
    if (resource->active &&
        salt_gpu_weight_resource_deactivate(kind, resource_id) != 0)
        return -1;
    if (resource->map) return -1;
    tensor_slots_forget(kind, resource_id);
    memset(resource, 0, sizeof *resource);
    return 0;
}

/* SALT_GPU_DISPATCH may request the fail-closed ICB placeholder;
 * automatic selection uses direct Metal until ICB is qualified. */
int salt_gpu_cuda_present(void) {
    /* the CUDA runtime probe: dlopen the loader; the cuda dispatch
     * module (future) owns the full binding. */
    void *h = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libcudart.dylib", RTLD_NOW | RTLD_LOCAL);
    if (h) { dlclose(h); return 1; }
    return 0;
}
int salt_gpu_rocm_present(void) { return 0; }
int salt_gpu_pageable_mmap_active(void) { return 0; }
const SaltGpuDispatch *salt_dispatch_get(void) {
    const char *ov = getenv("SALT_GPU_DISPATCH");
    if (ov && *ov) {
        if (!strcmp(ov, "icb") || !strcmp(ov, "metal-icb"))
            return &salt_dispatch_icb;
        if (!strcmp(ov, "direct")) return &salt_dispatch_direct;
        /* "cuda" / "auto" fall through to the auto order */
    }
    if (salt_gpu_cuda_present()) return NULL;  /* the cuda module (future) */
    /* the auto order: the metal-icb's record path SEGVs on this
     * Metal (the old-style ConcurrentDispatch form), so the direct
     * encode is the safe default until the ICB's runtime lands. */
    return &salt_dispatch_direct;
}

/* Projection adapter: B jobs, one shared weight + stable id, per-job
 * x (stride C in caller's xs[]) and y (stride R). Falls back -1
 * (CPU path) on any failure. This is the S0/S1 prefill-offload
 * primitive: the chunked qkv/z/o projections become B salt_gpu_q4_batch
 * jobs over the SAME weight, so the arena caches the weight once and
 * the kernel re-reads it per job (L2-served at chunk shapes -- the
 * exact thing the microbenchmark measures). */
int salt_gpu_proj_batch(const uint32_t *vals, const uint16_t *scales,
                        const uint16_t *biases, int R, int C, int B,
                        const float *const *xs, float *const *ys,
                        const void *id) {
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-dbg] proj_batch R=%d C=%d B=%d\n", R, C, B);
    if (B < 1 || R < 1 || C < 1 || !vals || !id) return -1;
    const uint32_t *v[BATCH_MAXSLOT];
    const uint16_t *s[BATCH_MAXSLOT], *b[BATCH_MAXSLOT];
    const void *ids[BATCH_MAXSLOT];
    if (B > BATCH_MAXSLOT) return -1;
    int Rj[BATCH_MAXSLOT];
    for (int j = 0; j < B; j++) {
        v[j] = vals; s[j] = scales; b[j] = biases;
        ids[j] = id;
        Rj[j] = R;
    }
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "[gpu-dbg] proj_batch filled, calling batch\n");
    return salt_gpu_q4_batch(v, s, b, xs, ys, ids, Rj, C, B);
}

#define METAL_TEXT_MAGIC UINT64_C(0x4d544c5458505247)
#define METAL_TEXT_MAX_LAYERS 64u
#define METAL_TEXT_MAX_EXPERTS 256u
#define METAL_TEXT_MAX_CANDIDATES 64u

typedef struct MetalTextTensorRef {
    id<MTLBuffer> value;
    id<MTLBuffer> scale;
    id<MTLBuffer> bias;
    NSUInteger value_offset;
    NSUInteger scale_offset;
    NSUInteger bias_offset;
    SaltGpuSharedBuffer value_alias;
    uint32_t rows;
    uint32_t cols;
    uint32_t encoding;
} MetalTextTensorRef;

typedef struct MetalTextLayerRefs {
    MetalTextTensorRef q, k, v, o;
    MetalTextTensorRef dense_gate, dense_up, dense_down, router;
    MetalTextTensorRef pre_attention_norm, q_norm, k_norm;
    MetalTextTensorRef post_attention_norm;
    MetalTextTensorRef pre_ffn_norm_1, pre_ffn_norm_2;
    MetalTextTensorRef post_ffn_norm_1, post_ffn_norm_2;
    MetalTextTensorRef post_ffn_norm, router_scale, per_expert_scale;
    MetalTextTensorRef layer_scalar;
    const SaltTextKvLayerDesc *kv;
    size_t meta_float_offset;
} MetalTextLayerRefs;

typedef struct MetalTextProgramState {
    uint64_t magic;
    const SaltTextVerifyProgram *program;
    SaltGpuSharedBuffer canonical;
    size_t canonical_bytes;
    MetalTextTensorRef embedding, final_norm, output_head;
    MetalTextLayerRefs layers[METAL_TEXT_MAX_LAYERS];
    uint32_t layer_count;
    uint32_t ready;
} MetalTextProgramState;

typedef enum MetalTextCommandPhase {
    METAL_TEXT_IDLE = 0,
    METAL_TEXT_RECORDING_PREFIX = 1,
    METAL_TEXT_RESOURCE_PENDING = 2,
    METAL_TEXT_RECORDING_EXPERT = 3,
    METAL_TEXT_RECORDING_SUFFIX = 4,
    METAL_TEXT_SUBMITTED = 5,
    METAL_TEXT_FINISHED = 6
} MetalTextCommandPhase;

typedef struct MetalTextProgramCommand {
    uint64_t generation;
    uint32_t source_position;
    uint32_t input_count;
    uint32_t authoritative;
    uint32_t authoritative_output_rows;
    uint32_t maximum_depth;
    uint32_t attention_score_rows;
    uint32_t next_cell;
    uint32_t highest_completion;
    uint32_t encoded_cells;
    uint32_t active_layer;
    uint32_t request_count;
    uint32_t expert_chain_mask;
    uint32_t phase;
    uint32_t profile_enabled;
    uint32_t profile_prefix_commands;
    uint32_t profile_continuation_commands;
    uint32_t profile_command_contains_expert;
    uint32_t profile_expert_layers;
    uint32_t profile_expert_min_group;
    uint32_t profile_expert_max_group;
    uint64_t profile_expert_unique_total;
    uint64_t profile_expert_rows_total;
    uint32_t profile_layer_unique[METAL_TEXT_MAX_LAYERS];
    uint32_t profile_layer_rows[METAL_TEXT_MAX_LAYERS];
    uint32_t profile_layer_max_group[METAL_TEXT_MAX_LAYERS];
    uint64_t profile_prefix_gpu_ns;
    uint64_t profile_continuation_gpu_ns;
    SaltTextTouchedSpan touched_spans[SALT_TEXT_MAX_TOUCHED_SPANS];
    uint32_t touched_span_count;
    uint64_t touched_span_bytes;
    int32_t request_experts[METAL_TEXT_MAX_EXPERTS];
    int32_t request_slots[METAL_TEXT_MAX_EXPERTS];
    uint32_t occupancies[METAL_TEXT_MAX_EXPERTS];
    id<MTLCommandBuffer> command;
    SaltTextVerifyBackendStats stats;
} MetalTextProgramCommand;

static void metal_text_alias(id<MTLBuffer> buffer, SaltGpuSharedBuffer *alias) {
    memset(alias, 0, sizeof *alias);
    if (!buffer) return;
    alias->contents = [buffer contents];
    alias->nbytes = (size_t)[buffer length];
    alias->backend = (void *)buffer;
}

static const SaltTensorResourceSpec *metal_text_resource_spec(
        const SaltTextVerifyProgram *program,
        const SaltTensorStorageSpec *storage) {
    return program && program->descriptor && storage
        ? salt_tensor_resource_find(program->descriptor->tensor_resources,
            program->descriptor->tensor_resource_count,
            storage->resource_kind, storage->resource_id)
        : NULL;
}

static int metal_text_shared_resource(
        const SaltTensorResourceSpec *spec, id<MTLBuffer> *buffer_out,
        size_t *base_offset_out) {
    uintptr_t base;
    if (!spec || !spec->base || !buffer_out || !base_offset_out)
        return -1;
    base = (uintptr_t)spec->base;
    for (int index = 0; index < METAL_SHARED_SLOTS; index++) {
        uintptr_t shared;
        if (!_shared_slots[index].buffer || !_shared_slots[index].contents)
            continue;
        shared = (uintptr_t)_shared_slots[index].contents;
        if (base < shared || base - shared > SIZE_MAX) continue;
        size_t delta = (size_t)(base - shared);
        if (delta <= _shared_slots[index].nbytes &&
            spec->nbytes <= _shared_slots[index].nbytes - delta) {
            *buffer_out = _shared_slots[index].buffer;
            *base_offset_out = delta;
            return 0;
        }
    }
    return -1;
}

static int metal_text_realize_tensor(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        MetalTextTensorRef *ref) {
    const SaltTensorStorageSpec *storage;
    const SaltTensorResourceSpec *spec;
    MetalWeightResource *resource;
    size_t base_offset = 0;
    if (!program || !tensor || !ref ||
        salt_tensor_desc_validate(tensor) != 0 ||
        !(storage = &tensor->storage) ||
        !(spec = metal_text_resource_spec(program, storage)))
        return -1;
    memset(ref, 0, sizeof *ref);
    ref->rows = tensor->rows;
    ref->cols = tensor->cols;
    ref->encoding = (uint32_t)storage->encoding;
    if (storage->source_class == SALT_TENSOR_SOURCE_SHARED) {
        if (metal_text_shared_resource(spec, &ref->value, &base_offset) != 0 ||
            storage->value_offset > NSUIntegerMax - base_offset)
            return -1;
        ref->scale = ref->bias = ref->value;
        ref->value_offset = (NSUInteger)(base_offset + storage->value_offset);
        ref->scale_offset = (NSUInteger)(base_offset + storage->scale_offset);
        ref->bias_offset = (NSUInteger)(base_offset + storage->bias_offset);
        metal_text_alias(ref->value, &ref->value_alias);
        return 0;
    }
    if (storage->source_class == SALT_TENSOR_SOURCE_SELECTED) return -1;
    resource = metal_weight_resource(storage->resource_kind,
                                     storage->resource_id);
    if (!resource || !resource->active) return -1;
    for (int index = 0; index < _tslot_n; index++) {
        const TensorSlot *slot = &_tslot[index];
        const SaltGpuResourceRef *source = &slot->resource;
        if (source->kind != storage->resource_kind ||
            source->resource_id != storage->resource_id ||
            source->value_offset != storage->value_offset ||
            source->scale_offset != storage->scale_offset ||
            source->bias_offset != storage->bias_offset)
            continue;
        if (slot->value_map && slot->scale_map && slot->bias_map) {
            if (storage->value_offset < slot->value_base ||
                storage->scale_offset < slot->scale_base ||
                storage->bias_offset < slot->bias_base)
                return -1;
            ref->value = slot->value_map;
            ref->scale = slot->scale_map;
            ref->bias = slot->bias_map;
            ref->value_offset =
                (NSUInteger)(storage->value_offset - slot->value_base);
            ref->scale_offset =
                (NSUInteger)(storage->scale_offset - slot->scale_base);
            ref->bias_offset =
                (NSUInteger)(storage->bias_offset - slot->bias_base);
            metal_text_alias(ref->value, &ref->value_alias);
            return 0;
        }
    }
    if (!resource->map || storage->value_offset < resource->map_offset ||
        storage->scale_offset < resource->map_offset ||
        storage->bias_offset < resource->map_offset)
        return -1;
    ref->value = ref->scale = ref->bias = resource->map;
    ref->value_offset =
        (NSUInteger)(storage->value_offset - resource->map_offset);
    ref->scale_offset =
        (NSUInteger)(storage->scale_offset - resource->map_offset);
    ref->bias_offset =
        (NSUInteger)(storage->bias_offset - resource->map_offset);
    metal_text_alias(ref->value, &ref->value_alias);
    return 0;
}

static int metal_text_realize_optional(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        MetalTextTensorRef *ref) {
    if (salt_tensor_desc_absent(tensor)) {
        memset(ref, 0, sizeof *ref);
        return 0;
    }
    return metal_text_realize_tensor(program, tensor, ref);
}

static int metal_text_requirements(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        SaltTextGpuProgramRequirements *requirements,
        size_t requirements_size) {
    if (!program || !program->ready || !program->descriptor || !policy ||
        !requirements || requirements_size != sizeof *requirements || !_dev ||
        policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        program->layer_count == 0 ||
        program->layer_count > METAL_TEXT_MAX_LAYERS)
        return -1;
    memset(requirements, 0, sizeof *requirements);
    requirements->backend_state_bytes = sizeof(MetalTextProgramState);
    requirements->command_bytes = sizeof(MetalTextProgramCommand);
    requirements->maximum_commands = program->dispatch.cell_count;
    requirements->flags = 0;
    return 0;
}

static int metal_text_prepare(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltGpuSharedBuffer **canonical_out,
        void *backend_state, size_t backend_state_bytes,
        void *command, size_t command_bytes) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    size_t meta_cursor = 0, required_meta_floats = 0;
    id<MTLBuffer> resized_meta = nil;
    if (!program || !plan || plan->program != program ||
        plan->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !canonical_out || !state || backend_state_bytes < sizeof *state ||
        !command || command_bytes < sizeof(MetalTextProgramCommand) ||
        program->layer_count > METAL_TEXT_MAX_LAYERS)
        return -1;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextLayerExecDesc *source =
            program->layers[layer].descriptor;
        const SaltAttentionDesc *attention;
        size_t layer_meta;
        if (!source || !source->plan) return -1;
        attention = &source->plan->attention;
        if (attention->head_dim < 2 || attention->rope_dim < 2 ||
            attention->rope_dim > attention->head_dim ||
            (size_t)program->maximum_candidates >
                (SIZE_MAX - 2u * (size_t)attention->head_dim) /
                    attention->rope_dim)
            return -1;
        layer_meta = 2u * (size_t)attention->head_dim +
            (size_t)program->maximum_candidates * attention->rope_dim;
        if (layer_meta > SIZE_MAX - required_meta_floats)
            return -1;
        required_meta_floats += layer_meta;
    }
    if (required_meta_floats > SIZE_MAX / sizeof(float) ||
        metal_ops_prepare() != 0 || metal_exact_prepare() != 0 ||
        batch_pipeline_prepare() != 0 || !_op_meta || !_op_status ||
        !_selected_cache_resources || !_selected_logical_slot_host ||
        !_selected_logical_payload_host)
        return -1;
    if (required_meta_floats * sizeof(float) > (size_t)[_op_meta length]) {
        resized_meta = [_dev newBufferWithLength:
            required_meta_floats * sizeof(float)
            options:MTLResourceStorageModeShared];
        if (!resized_meta) return -1;
        [_op_meta release];
        _op_meta = resized_meta;
    }
    memset(state, 0, sizeof *state);
    memset(command, 0, sizeof(MetalTextProgramCommand));
    if (salt_gpu_shared_buffer_alloc(&state->canonical,
            program->layout.total_bytes) != 0)
        return -1;
    state->program = program;
    state->canonical_bytes = program->layout.total_bytes;
    state->layer_count = program->layer_count;
    if (metal_text_realize_tensor(program, &program->descriptor->embedding,
            &state->embedding) != 0 ||
        metal_text_realize_tensor(program, &program->descriptor->final_norm,
            &state->final_norm) != 0 ||
        metal_text_realize_tensor(program, &program->descriptor->output_head,
            &state->output_head) != 0)
        goto fail;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer];
        const SaltTextLayerExecDesc *source = compiled->descriptor;
        MetalTextLayerRefs *target = &state->layers[layer];
        const SaltAttentionDesc *attention = &source->plan->attention;
        size_t meta_count;
#define MT_REALIZE(field) \
        if (metal_text_realize_tensor(program, &source->field, \
                &target->field) != 0) goto fail
#define MT_OPTIONAL(field) \
        if (metal_text_realize_optional(program, &source->field, \
                &target->field) != 0) goto fail
        MT_REALIZE(q); MT_REALIZE(k); MT_OPTIONAL(v); MT_REALIZE(o);
        MT_REALIZE(dense_gate); MT_REALIZE(dense_up); MT_REALIZE(dense_down);
        MT_REALIZE(router); MT_REALIZE(pre_attention_norm);
        MT_REALIZE(q_norm); MT_REALIZE(k_norm);
        MT_REALIZE(post_attention_norm);
        MT_REALIZE(pre_ffn_norm_1); MT_REALIZE(pre_ffn_norm_2);
        MT_REALIZE(post_ffn_norm_1); MT_REALIZE(post_ffn_norm_2);
        MT_REALIZE(post_ffn_norm); MT_REALIZE(router_scale);
        MT_REALIZE(per_expert_scale); MT_OPTIONAL(layer_scalar);
#undef MT_REALIZE
#undef MT_OPTIONAL
        if (!source->kv || attention->head_dim < 2 ||
            attention->rope_dim < 2 || attention->rope_dim > attention->head_dim)
            goto fail;
        target->kv = source->kv;
        meta_count = 2u * (size_t)attention->head_dim +
            (size_t)program->maximum_candidates * attention->rope_dim;
        if (meta_count > SIZE_MAX - meta_cursor ||
            meta_cursor + meta_count > (size_t)[_op_meta length] / sizeof(float))
            goto fail;
        target->meta_float_offset = meta_cursor;
        meta_cursor += meta_count;
    }
    state->magic = METAL_TEXT_MAGIC;
    state->ready = 1u;
    *canonical_out = &state->canonical;
    return 0;
fail:
    (void)salt_gpu_shared_buffer_free(&state->canonical);
    memset(state, 0, sizeof *state);
    return -1;
}

static int metal_text_new_command(MetalTextProgramCommand *command) {
    if (!command || command->command) return -1;
    command->profile_command_contains_expert = 0u;
    @autoreleasepool {
        command->command = [[_queue commandBuffer] retain];
    }
    return command->command ? 0 : -1;
}

static void metal_text_record_gpu_time(
        MetalTextProgramCommand *command, id<MTLCommandBuffer> current) {
    double start, end, elapsed;
    uint64_t elapsed_ns;
    if (!command || !current || !command->profile_enabled) return;
    start = [current GPUStartTime];
    end = [current GPUEndTime];
    if (!isfinite(start) || !isfinite(end) || end < start) return;
    elapsed = (end - start) * 1e9;
    if (elapsed < 0.0 || elapsed > (double)UINT64_MAX) return;
    elapsed_ns = (uint64_t)(elapsed + 0.5);
    if (command->profile_command_contains_expert) {
        command->profile_continuation_gpu_ns += elapsed_ns;
        command->profile_continuation_commands++;
    } else {
        command->profile_prefix_gpu_ns += elapsed_ns;
        command->profile_prefix_commands++;
    }
}

static int metal_text_complete_command(MetalTextProgramCommand *command) {
    id<MTLCommandBuffer> current;
    int status;
    if (!command || !(current = command->command)) return -1;
    command->command = nil;
    [current commit];
    [current waitUntilCompleted];
    status = metal_command_succeeded(current) == 0 &&
        *(uint32_t *)[_op_status contents] == 0u ? 0 : -1;
    metal_text_record_gpu_time(command, current);
    [current release];
    command->stats.backend_host_kernel_launch_calls++;
    return status;
}

static float metal_text_ref_value(const MetalTextTensorRef *ref,
                                  uint32_t index) {
    const unsigned char *base;
    float value = NAN;
    if (!ref || ref->encoding != SALT_TENSOR_ENCODING_F32 ||
        !ref->value_alias.contents || index >= ref->rows * ref->cols ||
        ref->value_offset > ref->value_alias.nbytes ||
        (size_t)(index + 1u) * sizeof(float) >
            ref->value_alias.nbytes - ref->value_offset)
        return NAN;
    base = (const unsigned char *)ref->value_alias.contents +
        ref->value_offset;
    memcpy(&value, base + (size_t)index * sizeof(float), sizeof value);
    return value;
}

static int metal_text_begin_depth(void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t maximum_depth) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !state->ready ||
        !command || command->phase != METAL_TEXT_IDLE || generation == 0 ||
        !input_token_ids || input_count == 0 ||
        input_count > state->program->maximum_candidates ||
        source_position >= state->program->maximum_context ||
        maximum_depth >= state->program->maximum_context - source_position)
        return -1;
    memset(command, 0, sizeof *command);
    if (salt_text_attention_score_rows(state->program, source_position,
            maximum_depth, &command->attention_score_rows) != 0 ||
        salt_text_touched_span_plan(state->program, input_count,
            command->attention_score_rows,
            command->touched_spans, SALT_TEXT_MAX_TOUCHED_SPANS,
            &command->touched_span_count, &command->touched_span_bytes) != 0 ||
        salt_text_touched_span_clear(state->canonical.contents,
            state->canonical_bytes, command->touched_spans,
            command->touched_span_count,
            &command->stats.canonical_clear_bytes) != 0)
        return -1;
    memcpy((unsigned char *)state->canonical.contents +
            state->program->layout.candidate_token_ids,
        input_token_ids, (size_t)input_count * sizeof(int32_t));
    {
        uint32_t *parents = (uint32_t *)((unsigned char *)
            state->canonical.contents +
            state->program->layout.target_parent_rows);
        uint32_t *depths = (uint32_t *)((unsigned char *)
            state->canonical.contents + state->program->layout.target_depths);
        for (uint32_t row = 0; row < input_count; row++) {
            parents[row] = row == 0u ? UINT32_MAX : row - 1u;
            depths[row] = row;
        }
    }
    *(uint32_t *)[_op_status contents] = 0u;
    for (uint32_t layer = 0; layer < state->layer_count; layer++) {
        const SaltTextLayerExecDesc *source =
            state->program->layers[layer].descriptor;
        const SaltAttentionDesc *attention = &source->plan->attention;
        const MetalTextLayerRefs *refs = &state->layers[layer];
        uint32_t pairs = (uint32_t)attention->rope_dim / 2u;
        float *meta = (float *)[_op_meta contents] + refs->meta_float_offset;
        if (refs->q_norm.encoding != SALT_TENSOR_ENCODING_F32 ||
            refs->k_norm.encoding != SALT_TENSOR_ENCODING_F32 ||
            refs->q_norm.cols != (uint32_t)attention->head_dim ||
            refs->k_norm.cols != (uint32_t)attention->head_dim)
            return -1;
        for (uint32_t index = 0; index < (uint32_t)attention->head_dim; index++) {
            meta[index] = metal_text_ref_value(&refs->q_norm, index);
            meta[attention->head_dim + index] =
                metal_text_ref_value(&refs->k_norm, index);
            if (!isfinite(meta[index]) ||
                !isfinite(meta[attention->head_dim + index]))
                return -1;
        }
        float *cosines = meta + 2u * attention->head_dim;
        float *sines = cosines + (size_t)input_count * pairs;
        for (uint32_t row = 0; row < input_count; row++)
            for (uint32_t pair = 0; pair < pairs; pair++) {
                float exponent = (float)(2u * pair) /
                    (float)attention->rope_base_dim;
                float angle = (float)(source_position + row) /
                    salt_powf((float)attention->rope_theta, exponent);
                size_t index = (size_t)row * pairs + pair;
                cosines[index] = salt_cosf(angle);
                sines[index] = salt_sinf(angle);
                if (!isfinite(cosines[index]) || !isfinite(sines[index]))
                    return -1;
            }
    }
    command->generation = generation;
    command->source_position = source_position;
    command->input_count = input_count;
    command->authoritative_output_rows = input_count;
    command->maximum_depth = maximum_depth;
    command->active_layer = UINT32_MAX;
    command->phase = METAL_TEXT_RECORDING_PREFIX;
    {
        const char *profile = getenv("SALT_TARGET_WATERFALL");
        command->profile_enabled = profile && strcmp(profile, "1") == 0;
    }
    command->stats.backend_template_count =
        state->program->dispatch.cell_count;
    command->stats.backend_dynamic_patches = 4u;
    command->stats.backend_selected_job_capacity =
        state->program->maximum_candidates *
        state->program->layout.maximum_topk;
    if (metal_text_new_command(command) != 0) {
        memset(command, 0, sizeof *command);
        return -1;
    }
    return 0;
}

static int metal_text_begin(void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    if (input_count == 0) return -1;
    return metal_text_begin_depth(backend_state, command_state, generation,
        source_position, input_token_ids, input_count, input_count - 1u);
}

static int metal_text_begin_authoritative(
        void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (metal_text_begin(backend_state, command_state, generation,
            source_position, input_token_ids, input_count) != 0 || !command)
        return -1;
    command->authoritative = 1u;
    return 0;
}

static int metal_text_begin_authoritative_output(
        void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows) {
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (output_rows > 1u || input_count == 0u ||
        metal_text_begin_depth(backend_state, command_state, generation,
            source_position, input_token_ids, input_count,
            input_count - 1u) != 0 || !command)
        return -1;
    command->authoritative = 1u;
    command->authoritative_output_rows = output_rows;
    return 0;
}

static int metal_text_begin_frontier(
        void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids,
        const uint32_t *parent_rows, const uint32_t *depths,
        uint32_t input_count) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    uint32_t maximum_depth = 0u;
    if (!state || !state->program || !parent_rows || !depths ||
        !command || input_count == 0 ||
        input_count > state->program->maximum_candidates)
        return -1;
    for (uint32_t row = 0; row < input_count; row++)
        if (depths[row] > maximum_depth) maximum_depth = depths[row];
    if (maximum_depth == UINT32_MAX ||
        source_position > state->program->maximum_context -
            (maximum_depth + 1u) ||
        metal_text_begin_depth(backend_state, command_state,
            generation, source_position, input_token_ids, input_count,
            maximum_depth) != 0)
        return -1;
    memcpy((unsigned char *)state->canonical.contents +
            state->program->layout.target_parent_rows,
        parent_rows, (size_t)input_count * sizeof(uint32_t));
    memcpy((unsigned char *)state->canonical.contents +
            state->program->layout.target_depths,
        depths, (size_t)input_count * sizeof(uint32_t));
    for (uint32_t layer = 0; layer < state->layer_count; layer++) {
        const SaltAttentionDesc *attention =
            &state->program->layers[layer].descriptor->plan->attention;
        const MetalTextLayerRefs *refs = &state->layers[layer];
        uint32_t pairs = (uint32_t)attention->rope_dim / 2u;
        float *meta = (float *)[_op_meta contents] + refs->meta_float_offset;
        float *cosines = meta + 2u * attention->head_dim;
        float *sines = cosines + (size_t)input_count * pairs;
        for (uint32_t row = 0; row < input_count; row++)
            for (uint32_t pair = 0; pair < pairs; pair++) {
                float exponent = (float)(2u * pair) /
                    (float)attention->rope_base_dim;
                float angle = (float)(source_position + depths[row]) /
                    salt_powf((float)attention->rope_theta, exponent);
                size_t index = (size_t)row * pairs + pair;
                cosines[index] = salt_cosf(angle);
                sines[index] = salt_sinf(angle);
                if (!isfinite(cosines[index]) || !isfinite(sines[index]))
                    return -1;
            }
    }
    return 0;
}

static int metal_text_encode_projection(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        const MetalTextTensorRef *ref, size_t input_offset,
        size_t output_offset, uint32_t batch) {
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> canonical;
    uint32_t physical_dispatches = 0u;
    if (!state || !command || !command->command || !ref || !ref->value ||
        !ref->scale || !ref->bias || batch == 0 || ref->rows == 0 ||
        ref->cols == 0 || input_offset % sizeof(float) != 0 ||
        output_offset % sizeof(float) != 0)
        return -1;
    canonical = (id<MTLBuffer>)state->canonical.backend;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
        BatchDesc4 descriptors[2u * METAL_TEXT_MAX_CANDIDATES];
        uint32_t first = 0u;
        if (ref->rows > UINT32_MAX / batch) {
            [encoder endEncoding];
            return -1;
        }
        [encoder setComputePipelineState:_bpso2];
        [encoder setBuffer:ref->value offset:0 atIndex:0];
        [encoder setBuffer:ref->scale offset:0 atIndex:1];
        [encoder setBuffer:ref->bias offset:0 atIndex:2];
        [encoder setBuffer:canonical offset:0 atIndex:3];
        [encoder setBuffer:canonical offset:0 atIndex:4];
        [encoder setBytes:&ref->cols length:sizeof ref->cols atIndex:7];
        while (first < batch) {
            uint32_t active = batch - first;
            uint32_t total_rows;
            if (active > METAL_TEXT_MAX_CANDIDATES)
                active = METAL_TEXT_MAX_CANDIDATES;
            total_rows = ref->rows * active;
            for (uint32_t row = 0; row < active; row++) {
                uint32_t logical = first + row;
                descriptors[2u * row] = (BatchDesc4) {
                    ref->value_offset, ref->scale_offset,
                    input_offset / sizeof(float) +
                        (uint64_t)logical * ref->cols,
                    output_offset / sizeof(float) +
                        (uint64_t)logical * ref->rows,
                };
                descriptors[2u * row + 1u] = (BatchDesc4) {
                    ref->bias_offset, 0u, 0u,
                    (uint64_t)row * ref->rows,
                };
            }
            [encoder setBytes:descriptors
                       length:(NSUInteger)active * 2u * sizeof(BatchDesc4)
                      atIndex:5];
            [encoder setBytes:&total_rows length:sizeof total_rows atIndex:6];
            [encoder setBytes:&active length:sizeof active atIndex:8];
            [encoder dispatchThreads:MTLSizeMake(total_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            first += active;
            physical_dispatches++;
        }
    } else if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q8) {
        struct { uint32_t rows, cols, jobs; } args = {
            ref->rows, ref->cols, batch,
        };
        uint64_t voff = ref->value_offset;
        uint64_t soff = ref->scale_offset;
        uint64_t boff = ref->bias_offset;
        [encoder setComputePipelineState:_q8_pso];
        [encoder setBuffer:ref->value offset:0 atIndex:0];
        [encoder setBuffer:ref->scale offset:0 atIndex:1];
        [encoder setBuffer:ref->bias offset:0 atIndex:2];
        [encoder setBuffer:canonical offset:input_offset atIndex:3];
        [encoder setBuffer:canonical offset:output_offset atIndex:4];
        [encoder setBytes:&voff length:sizeof voff atIndex:5];
        [encoder setBytes:&soff length:sizeof soff atIndex:6];
        [encoder setBytes:&boff length:sizeof boff atIndex:7];
        [encoder setBytes:&args length:sizeof args atIndex:8];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)batch * ref->rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        physical_dispatches = 1u;
    } else {
        [encoder endEncoding];
        return -1;
    }
    [encoder endEncoding];
    command->stats.projection_dispatches++;
    command->stats.backend_physical_kernel_nodes += physical_dispatches;
    command->stats.area_output_row_tiles += ref->rows;
    command->stats.area_candidate_output_tiles +=
        (uint64_t)ref->rows * batch;
    if (batch > 1u) {
        command->stats.area_mn_dispatches++;
        command->stats.area_matrix_parallel_dispatches++;
    } else {
        command->stats.area_m1_dispatches++;
    }
    return 0;
}

static int metal_text_encode_embedding(
        MetalTextProgramState *state, MetalTextProgramCommand *command) {
    const MetalTextTensorRef *ref = &state->embedding;
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint64_t voff, soff, boff;
        uint32_t vocab, width, tokens;
        float scale;
    } args;
    uint64_t elements;
    if (!command->command ||
        ref->encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        !ref->value || !ref->scale || !ref->bias ||
        ref->rows != state->program->vocabulary ||
        ref->cols != state->program->hidden)
        return -1;
    elements = (uint64_t)command->input_count * ref->cols;
    if (elements > NSUIntegerMax) return -1;
    args.voff = ref->value_offset;
    args.soff = ref->scale_offset;
    args.boff = ref->bias_offset;
    args.vocab = ref->rows;
    args.width = ref->cols;
    args.tokens = command->input_count;
    args.scale = state->program->descriptor->embedding_scale;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_embedding_pso];
    [encoder setBuffer:ref->value offset:0 atIndex:0];
    [encoder setBuffer:ref->scale offset:0 atIndex:1];
    [encoder setBuffer:ref->bias offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:state->program->layout.candidate_token_ids atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:state->program->layout.state_a atIndex:4];
    [encoder setBuffer:_op_status offset:0 atIndex:5];
    [encoder setBytes:&args length:sizeof args atIndex:6];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_activation(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        size_t gate_offset, size_t up_offset, size_t output_offset,
        size_t elements) {
    id<MTLComputeCommandEncoder> encoder;
    uint32_t count;
    if (!state || !command || !command->command || elements == 0 ||
        elements > UINT32_MAX) return -1;
    count = (uint32_t)elements;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_activation_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:gate_offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:up_offset atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:output_offset atIndex:2];
    [encoder setBytes:&count length:sizeof count atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_route_complete(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t experts, uint32_t topk) {
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLBuffer> canonical = (id<MTLBuffer>)state->canonical.backend;
    id<MTLComputeCommandEncoder> encoder;
    struct { uint32_t rows, experts, topk, hidden; } args = {
        command->input_count, experts, topk, state->program->hidden,
    };
    uint64_t jobs = (uint64_t)command->input_count * topk;
    uint64_t elements = jobs * state->program->hidden;
    if (jobs == 0 || jobs > INT32_MAX || elements > NSUIntegerMax)
        return -1;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_route_maps_pso];
    [encoder setBuffer:canonical offset:layout->selected_experts atIndex:0];
    [encoder setBuffer:canonical offset:layout->selected_weights atIndex:1];
    [encoder setBuffer:canonical offset:layout->grouped_to_canonical atIndex:2];
    [encoder setBuffer:canonical offset:layout->canonical_to_grouped atIndex:3];
    [encoder setBuffer:canonical offset:layout->tensor_row atIndex:4];
    [encoder setBuffer:_op_status offset:0 atIndex:5];
    [encoder setBytes:&args length:sizeof args atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_routed_gather_pso];
    [encoder setBuffer:canonical offset:layout->grouped_to_canonical atIndex:0];
    [encoder setBuffer:canonical offset:layout->combine_a atIndex:1];
    [encoder setBuffer:canonical offset:layout->routed_input atIndex:2];
    [encoder setBuffer:_op_status offset:0 atIndex:3];
    [encoder setBytes:&args length:sizeof args atIndex:4];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_expert_reduce(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t topk) {
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLBuffer> canonical = (id<MTLBuffer>)state->canonical.backend;
    id<MTLComputeCommandEncoder> encoder;
    struct { uint32_t rows, experts, topk, hidden; } args = {
        command->input_count, 0u, topk, state->program->hidden,
    };
    uint64_t elements =
        (uint64_t)command->input_count * state->program->hidden;
    if (elements == 0 || elements > NSUIntegerMax) return -1;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_expert_reduce_pso];
    [encoder setBuffer:canonical offset:layout->selected_weights atIndex:0];
    [encoder setBuffer:canonical offset:layout->canonical_to_grouped atIndex:1];
    [encoder setBuffer:canonical offset:layout->routed_output_jobs atIndex:2];
    [encoder setBuffer:canonical offset:layout->routed_output atIndex:3];
    [encoder setBuffer:_op_status offset:0 atIndex:4];
    [encoder setBytes:&args length:sizeof args atIndex:5];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_kv_rows(
        const SaltTextKvRowsDesc *rows, uint32_t width,
        id<MTLBuffer> *private_buffer, size_t *private_offset,
        id<MTLBuffer> *shared_buffer, size_t *shared_offset,
        uint32_t *shared_rows, uint32_t *shared_stride) {
    const float *shared = NULL;
    size_t shared_capacity = 0, shared_row_stride = 0;
    uint32_t shared_row_count = 0;
    if (!rows || !private_buffer || !private_offset || !shared_buffer ||
        !shared_offset || !shared_rows || !shared_stride ||
        rows->private_row_stride != width || rows->private_row_capacity == 0 ||
        rows->private_float_capacity <
            (size_t)rows->private_row_capacity * width ||
        metal_shared_pointer(rows->private_rows,
            rows->private_float_capacity, private_buffer, private_offset) != 0)
        return -1;
    if (rows->shared_prefix_state) {
        shared = rows->shared_prefix_state->rows;
        shared_capacity = rows->shared_prefix_state->float_capacity;
        shared_row_stride = rows->shared_prefix_state->row_stride;
        shared_row_count = rows->shared_prefix_state->row_count;
    } else {
        shared = rows->shared_prefix;
        shared_capacity = rows->shared_prefix_float_capacity;
        shared_row_stride = rows->shared_prefix_row_stride;
        shared_row_count = rows->shared_prefix_rows;
    }
    if (shared_row_count > 0) {
        if (!shared || shared_row_stride != width ||
            shared_capacity < (size_t)shared_row_count * width ||
            metal_shared_pointer(shared, shared_capacity,
                shared_buffer, shared_offset) != 0)
            return -1;
    } else {
        *shared_buffer = *private_buffer;
        *shared_offset = *private_offset;
        shared_row_stride = width;
    }
    if (rows->private_mode != SALT_TEXT_KV_PRIVATE_ABSOLUTE &&
        rows->private_mode != SALT_TEXT_KV_PRIVATE_RING)
        return -1;
    *shared_rows = shared_row_count;
    *shared_stride = (uint32_t)shared_row_stride;
    return 0;
}

static int metal_text_encode_attention_transform(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t layer_index) {
    const SaltTextCompiledLayer *compiled = &state->program->layers[layer_index];
    const SaltTextLayerExecDesc *source = compiled->descriptor;
    const SaltAttentionDesc *attention = &source->plan->attention;
    const MetalTextLayerRefs *refs = &state->layers[layer_index];
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLBuffer> canonical = (id<MTLBuffer>)state->canonical.backend;
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint32_t n_heads, n_kv_heads, head_dim, rope_dim, batch;
        uint32_t qstride, kvstride, rope_pairs, shared_kv;
        float eps;
    } args = {
        (uint32_t)attention->n_heads, (uint32_t)attention->n_kv_heads,
        (uint32_t)attention->head_dim, (uint32_t)attention->rope_dim,
        command->input_count, compiled->query_width, compiled->kv_width,
        (uint32_t)attention->rope_dim / 2u,
        0u,
        state->program->descriptor->norm_epsilon,
    };
    uint64_t tasks = (uint64_t)command->input_count *
        (uint32_t)(attention->n_heads + 2 * attention->n_kv_heads);
    if (tasks == 0 || tasks > NSUIntegerMax) return -1;
    if (attention->shared_kv_projection) {
        size_t elements, bytes;
        id<MTLBlitCommandEncoder> blit;
        if (compiled->kv_width == 0u ||
            (size_t)command->input_count > SIZE_MAX / compiled->kv_width)
            return -1;
        elements = (size_t)command->input_count * compiled->kv_width;
        if (elements > SIZE_MAX / sizeof(float)) return -1;
        bytes = elements * sizeof(float);
        if (layout->keys > state->canonical_bytes ||
            bytes > state->canonical_bytes - layout->keys ||
            layout->values > state->canonical_bytes ||
            bytes > state->canonical_bytes - layout->values)
            return -1;
        blit = [command->command blitCommandEncoder];
        if (!blit) return -1;
        [blit copyFromBuffer:canonical sourceOffset:layout->keys
                    toBuffer:canonical destinationOffset:layout->values
                        size:bytes];
        [blit endEncoding];
        command->stats.backend_physical_kernel_nodes++;
    }
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_transform_view_pso];
    [encoder setBuffer:canonical offset:layout->queries atIndex:0];
    [encoder setBuffer:canonical offset:layout->keys atIndex:1];
    [encoder setBuffer:canonical offset:layout->values atIndex:2];
    [encoder setBuffer:_op_meta
                offset:refs->meta_float_offset * sizeof(float) atIndex:3];
    [encoder setBuffer:canonical offset:compiled->tentative_key_offset atIndex:4];
    [encoder setBuffer:canonical offset:compiled->tentative_value_offset atIndex:5];
    [encoder setBuffer:_op_status offset:0 atIndex:6];
    [encoder setBytes:&args length:sizeof args atIndex:7];
    [encoder setThreadgroupMemoryLength:sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_attention_body(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t layer_index) {
    const SaltTextCompiledLayer *compiled = &state->program->layers[layer_index];
    const SaltTextLayerExecDesc *source = compiled->descriptor;
    const SaltAttentionDesc *attention = &source->plan->attention;
    const SaltTextKvLayerDesc *kv = state->layers[layer_index].kv;
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLBuffer> private_k, private_v, shared_k, shared_v;
    size_t private_ko, private_vo, shared_ko, shared_vo;
    uint32_t shared_kr, shared_vr, shared_ks, shared_vs;
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint32_t full, n_heads, n_kv_heads, head_dim, window;
        uint32_t source, batch, qstride, kvstride;
        uint32_t private_mode, private_capacity, private_base;
        uint32_t shared_rows, shared_stride;
    } args;
    uint64_t tasks;
    size_t score_bytes;
    if (!kv || metal_text_kv_rows(&kv->keys, compiled->kv_width,
            &private_k, &private_ko, &shared_k, &shared_ko,
            &shared_kr, &shared_ks) != 0 ||
        metal_text_kv_rows(&kv->values, compiled->kv_width,
            &private_v, &private_vo, &shared_v, &shared_vo,
            &shared_vr, &shared_vs) != 0 ||
        shared_kr != shared_vr || shared_ks != shared_vs ||
        kv->keys.private_mode != kv->values.private_mode ||
        kv->keys.private_row_capacity != kv->values.private_row_capacity ||
        kv->keys.private_position_base != kv->values.private_position_base)
        return -1;
    args.full = attention->kind == SALT_ATTN_FULL;
    args.n_heads = (uint32_t)attention->n_heads;
    args.n_kv_heads = (uint32_t)attention->n_kv_heads;
    args.head_dim = (uint32_t)attention->head_dim;
    args.window = (uint32_t)attention->window;
    args.source = command->source_position;
    args.batch = command->input_count;
    args.qstride = compiled->query_width;
    args.kvstride = compiled->kv_width;
    args.private_mode = (uint32_t)kv->keys.private_mode;
    args.private_capacity = kv->keys.private_row_capacity;
    args.private_base = kv->keys.private_position_base;
    args.shared_rows = shared_kr;
    args.shared_stride = shared_ks;
    tasks = (uint64_t)command->input_count * attention->n_heads;
    score_bytes = (size_t)command->attention_score_rows * sizeof(float);
    if (tasks == 0 || tasks > NSUIntegerMax || score_bytes > 32u * 1024u)
        return -1;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_attention_view_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->queries atIndex:0];
    [encoder setBuffer:private_k offset:private_ko atIndex:1];
    [encoder setBuffer:private_v offset:private_vo atIndex:2];
    [encoder setBuffer:shared_k offset:shared_ko atIndex:3];
    [encoder setBuffer:shared_v offset:shared_vo atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:compiled->tentative_key_offset atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:compiled->tentative_value_offset atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->attention_output atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder setBytes:&args length:sizeof args atIndex:9];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->target_parent_rows atIndex:10];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->target_depths atIndex:11];
    [encoder setThreadgroupMemoryLength:score_bytes atIndex:0];
    [encoder setThreadgroupMemoryLength:2u * sizeof(float) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_prepare_resource_request(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t layer) {
    const SaltTextLayerPlan *plan;
    const int32_t *selected;
    const int32_t *grouped;
    uint32_t jobs;
    if (!state || !command || layer >= state->layer_count ||
        (command->phase != METAL_TEXT_RECORDING_PREFIX &&
         command->phase != METAL_TEXT_RECORDING_SUFFIX))
        return -1;
    plan = state->program->layers[layer].descriptor->plan;
    if (!plan || plan->n_experts < 1 ||
        plan->n_experts > (int)METAL_TEXT_MAX_EXPERTS ||
        plan->top_k_experts < 1 ||
        command->input_count > UINT32_MAX / (uint32_t)plan->top_k_experts)
        return -1;
    jobs = command->input_count * (uint32_t)plan->top_k_experts;
    selected = (const int32_t *)((const unsigned char *)
        state->canonical.contents + state->program->layout.selected_experts);
    grouped = (const int32_t *)((const unsigned char *)
        state->canonical.contents + state->program->layout.grouped_to_canonical);
    memset(command->occupancies, 0, sizeof command->occupancies);
    for (uint32_t row = 0; row < command->input_count; row++)
        for (uint32_t rank = 0; rank < (uint32_t)plan->top_k_experts; rank++) {
            uint32_t canonical = row * (uint32_t)plan->top_k_experts + rank;
            int32_t expert = selected[canonical];
            if (expert < 0 || expert >= plan->n_experts ||
                (rank > 0 && expert <= selected[canonical - 1u]))
                return -1;
            command->occupancies[(uint32_t)expert]++;
        }
    command->request_count = 0;
    uint32_t grouped_cursor = 0;
    for (uint32_t expert = 0; expert < (uint32_t)plan->n_experts; expert++) {
        uint32_t population = command->occupancies[expert];
        if (population == 0) continue;
        if (command->request_count >= METAL_TEXT_MAX_EXPERTS)
            return -1;
        command->request_experts[command->request_count++] = (int32_t)expert;
        for (uint32_t item = 0; item < population; item++) {
            uint32_t canonical = (uint32_t)grouped[grouped_cursor++];
            if (canonical >= jobs || selected[canonical] != (int32_t)expert)
                return -1;
        }
    }
    if (grouped_cursor != jobs || command->request_count == 0)
        return -1;
    command->active_layer = layer;
    command->phase = METAL_TEXT_RESOURCE_PENDING;
    return 0;
}

static int metal_text_resource_request(
        void *backend_state, void *command_state, uint32_t *layer,
        int32_t *experts, uint32_t expert_capacity, uint32_t *expert_count) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        command->phase != METAL_TEXT_RESOURCE_PENDING || !layer || !experts ||
        !expert_count || command->request_count == 0 ||
        command->request_count > expert_capacity)
        return -1;
    *layer = command->active_layer;
    *expert_count = command->request_count;
    memcpy(experts, command->request_experts,
        (size_t)command->request_count * sizeof *experts);
    return 0;
}

static int metal_text_resource_resume(
        void *backend_state, void *command_state, uint32_t layer,
        const int32_t *experts, const int32_t *slots, uint32_t expert_count) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        command->phase != METAL_TEXT_RESOURCE_PENDING ||
        layer != command->active_layer || !experts || !slots ||
        expert_count != command->request_count)
        return -1;
    for (uint32_t index = 0; index < expert_count; index++) {
        int32_t expert = experts[index];
        int32_t slot = slots[index];
        uint64_t logical = (uint64_t)layer *
            (uint32_t)state->program->layers[layer].descriptor->plan->n_experts +
            (uint32_t)expert;
        if (expert != command->request_experts[index] || slot < 0 ||
            slot >= _selected_cache_capacity ||
            logical >= (uint64_t)_selected_logical_capacity ||
            _selected_logical_slot_host[logical] != slot ||
            !_selected_cache_resources[slot].map ||
            _selected_cache_resources[slot].logical_resource_id != logical)
            return -1;
        command->request_slots[index] = slot;
    }
    if (metal_text_new_command(command) != 0) return -1;
    command->phase = METAL_TEXT_RECORDING_EXPERT;
    command->expert_chain_mask = 0u;
    return 0;
}

static int metal_text_dependency_barrier(
        void *backend_state, void *command_state,
        uint32_t dependency_epoch, uint32_t completion_epoch);
static int metal_text_encode_cell(void *backend_state, void *command_state,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        const SaltTextExecutionAssignment *assignment,
        uint32_t input_count);

static int metal_text_encode_extent(void *backend_state, void *command_state,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        uint32_t first_cell, uint32_t cell_count, uint32_t input_count,
        uint32_t *encoded_cells) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || !command || !program || !plan || !encoded_cells ||
        plan->program != program ||
        plan->assignment_count != program->dispatch.cell_count ||
        first_cell != command->next_cell || cell_count == 0 ||
        cell_count > program->dispatch.cell_count - first_cell)
        return -1;
    *encoded_cells = 0;
    for (uint32_t offset = 0; offset < cell_count; offset++) {
        uint32_t index = first_cell + offset;
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        if (offset > 0 && cell->dependency_epoch > 0) {
            int barrier = metal_text_dependency_barrier(
                backend_state, command_state,
                cell->dependency_epoch, cell->completion_epoch);
            if (barrier == SALT_TEXT_GPU_NEED_RESOURCE) {
                *encoded_cells = offset;
                return barrier;
            }
            if (barrier != SALT_TEXT_GPU_DEPENDENCY_READY) return -1;
        }
        if (metal_text_encode_cell(backend_state, command_state, program, cell,
                &plan->assignments[index], input_count) != 0)
            return -1;
        *encoded_cells = offset + 1u;
    }
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int metal_text_encode_expert_chain(
        MetalTextProgramState *state, MetalTextProgramCommand *command) {
    const SaltTextLayerExecDesc *layer;
    const SaltTextLayerPlan *plan;
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    SelectedDesc *descriptors;
    SelectedDesc *down;
    MetalSelectedTask *tasks;
    id<MTLComputeCommandEncoder> encoder;
    uint64_t row_prefix = 0;
    uint32_t gate_up_tasks = 0, down_tasks = 0;
    uint32_t max_group = 0;
    if (!state || !command || !command->command ||
        command->phase != METAL_TEXT_RECORDING_EXPERT ||
        command->active_layer >= state->layer_count ||
        command->request_count == 0 ||
        !(descriptors = (SelectedDesc *)[_selected_desc contents]) ||
        !(tasks = (MetalSelectedTask *)[_selected_task_map contents]))
        return -1;
    layer = state->program->layers[command->active_layer].descriptor;
    plan = layer->plan;
    down = (SelectedDesc *)((unsigned char *)descriptors +
        METAL_SELECTED_DESC_DOWN_OFFSET);
    for (uint32_t index = 0; index < command->request_count; index++) {
        uint32_t expert = (uint32_t)command->request_experts[index];
        uint32_t slot = (uint32_t)command->request_slots[index];
        uint32_t group = command->occupancies[expert];
        const SaltTextExpertDesc *entry = &layer->experts[expert];
        const SaltTensorStorageSpec *gate = &entry->gate.storage;
        const SaltTensorStorageSpec *up = &entry->up.storage;
        const SaltTensorStorageSpec *down_storage = &entry->down.storage;
        uint64_t logical = gate->logical_resource_id;
        uint64_t payload, base;
        if (group == 0 || logical >= (uint64_t)_selected_logical_capacity ||
            _selected_logical_slot_host[logical] != (int32_t)slot ||
            up->logical_resource_id != logical ||
            down_storage->logical_resource_id != logical ||
            up->value_offset < gate->value_offset ||
            down_storage->value_offset < gate->value_offset)
            return -1;
        payload = _selected_logical_payload_host[logical];
        base = gate->value_offset;
        descriptors[2u * index] = (SelectedDesc) {
            {payload + gate->value_offset - base,
             payload + gate->scale_offset - base,
             layout->routed_input / sizeof(float) +
                row_prefix * state->program->hidden,
             layout->routed_gate / sizeof(float) +
                row_prefix * (uint32_t)plan->expert_intermediate},
            {payload + gate->bias_offset - base,
             (uint64_t)(uint32_t)plan->expert_intermediate,
             state->program->hidden, group},
            {slot, 0u, 0u, 0u},
        };
        descriptors[2u * index + 1u] = (SelectedDesc) {
            {payload + up->value_offset - base,
             payload + up->scale_offset - base,
             layout->routed_input / sizeof(float) +
                row_prefix * state->program->hidden,
             layout->routed_up / sizeof(float) +
                row_prefix * (uint32_t)plan->expert_intermediate},
            {payload + up->bias_offset - base,
             (uint64_t)(uint32_t)plan->expert_intermediate,
             state->program->hidden, group},
            {slot, 0u, 0u, 0u},
        };
        down[index] = (SelectedDesc) {
            {payload + down_storage->value_offset - base,
             payload + down_storage->scale_offset - base,
             layout->routed_chain / sizeof(float) +
                row_prefix * (uint32_t)plan->expert_intermediate,
             layout->routed_output_jobs / sizeof(float) +
                row_prefix * state->program->hidden},
            {payload + down_storage->bias_offset - base,
             state->program->hidden,
             (uint64_t)(uint32_t)plan->expert_intermediate, group},
            {slot, 0u, 0u, 0u},
        };
        for (uint32_t token = 0; token < group; token++) {
            if (gate_up_tasks > METAL_SELECTED_GATE_UP_TASK_CAPACITY - 2u ||
                down_tasks >= METAL_SELECTED_DOWN_TASK_CAPACITY)
                return -1;
            tasks[gate_up_tasks++] = (MetalSelectedTask) {2u * index, token};
            tasks[gate_up_tasks++] =
                (MetalSelectedTask) {2u * index + 1u, token};
            tasks[METAL_SELECTED_GATE_UP_TASK_CAPACITY + down_tasks++] =
                (MetalSelectedTask) {index, token};
        }
        row_prefix += group;
        if (group > max_group) max_group = group;
    }
    if (row_prefix != (uint64_t)command->input_count *
            (uint32_t)plan->top_k_experts ||
        gate_up_tasks != 2u * row_prefix || down_tasks != row_prefix)
        return -1;
    if (command->profile_enabled) {
        uint32_t layer_index = command->active_layer;
        command->profile_expert_layers++;
        command->profile_expert_unique_total += command->request_count;
        command->profile_expert_rows_total += row_prefix;
        command->profile_layer_unique[layer_index] = command->request_count;
        command->profile_layer_rows[layer_index] = (uint32_t)row_prefix;
        command->profile_layer_max_group[layer_index] = max_group;
        for (uint32_t index = 0; index < command->request_count; index++) {
            uint32_t group = command->occupancies[
                (uint32_t)command->request_experts[index]];
            if (command->profile_expert_min_group == 0u ||
                group < command->profile_expert_min_group)
                command->profile_expert_min_group = group;
            if (group > command->profile_expert_max_group)
                command->profile_expert_max_group = group;
        }
    }
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    for (uint32_t index = 0; index < command->request_count; index++)
        [encoder useResource:_selected_cache_resources[
            command->request_slots[index]].map usage:MTLResourceUsageRead];
    if (metal_selected_encode_ragged(encoder, _wave_selected_args,
            (id<MTLBuffer>)state->canonical.backend,
            (id<MTLBuffer>)state->canonical.backend, 0,
            (int)(2u * command->request_count), plan->expert_intermediate,
            0, gate_up_tasks) != 0) {
        [encoder endEncoding];
        return -1;
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:_activation_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->routed_gate atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->routed_up atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->routed_chain atIndex:2];
    uint32_t elements = (uint32_t)(row_prefix *
        (uint32_t)plan->expert_intermediate);
    [encoder setBytes:&elements length:sizeof elements atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    if (metal_selected_encode_ragged(encoder, _wave_selected_args,
            (id<MTLBuffer>)state->canonical.backend,
            (id<MTLBuffer>)state->canonical.backend,
            METAL_SELECTED_DESC_DOWN_OFFSET,
            (int)command->request_count, state->program->hidden,
            (NSUInteger)METAL_SELECTED_GATE_UP_TASK_CAPACITY *
                sizeof(MetalSelectedTask), down_tasks) != 0) {
        [encoder endEncoding];
        return -1;
    }
    [encoder endEncoding];
    command->stats.expert_gate_up_dispatches += 2u;
    command->stats.expert_down_dispatches++;
    command->stats.backend_selected_jobs = (uint32_t)row_prefix;
    command->stats.backend_physical_kernel_nodes += 4u;
    command->expert_chain_mask = 15u;
    command->profile_command_contains_expert = 1u;
    return 0;
}

static int metal_text_dependency_barrier(
        void *backend_state, void *command_state,
        uint32_t dependency_epoch, uint32_t completion_epoch) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    const SaltTextExecutionCell *upcoming;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        command->phase == METAL_TEXT_IDLE ||
        command->phase == METAL_TEXT_SUBMITTED ||
        command->phase == METAL_TEXT_FINISHED ||
        command->next_cell >= state->program->dispatch.cell_count ||
        dependency_epoch == 0 || completion_epoch <= dependency_epoch ||
        dependency_epoch > command->highest_completion)
        return -1;
    upcoming = &state->program->dispatch.cells[command->next_cell];
    if (upcoming->kind == SALT_TEXT_CELL_EXPERT_GATE &&
        (command->phase == METAL_TEXT_RECORDING_PREFIX ||
         command->phase == METAL_TEXT_RECORDING_SUFFIX)) {
        if (metal_text_complete_command(command) != 0 ||
            metal_text_prepare_resource_request(
                state, command, upcoming->layer) != 0)
            return -1;
        return SALT_TEXT_GPU_NEED_RESOURCE;
    }
    if (upcoming->kind == SALT_TEXT_CELL_EXPERT_REDUCTION &&
        command->phase == METAL_TEXT_RECORDING_EXPERT) {
        if (command->expert_chain_mask != 15u ||
            metal_text_complete_command(command) != 0 ||
            metal_text_new_command(command) != 0)
            return -1;
        command->phase = METAL_TEXT_RECORDING_SUFFIX;
        command->active_layer = UINT32_MAX;
        command->request_count = 0;
        command->expert_chain_mask = 0;
    }
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int metal_text_encode_rows(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        id<MTLComputePipelineState> pipeline,
        const MetalTextTensorRef *weight,
        size_t x_offset, size_t y_offset, size_t z_offset,
        uint32_t rows, uint32_t width, uint32_t aux,
        float eps, float scalar) {
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint64_t x, y, z;
        uint32_t rows, width, aux;
        float eps, scalar;
    } args;
    if (!state || !command || !command->command || !pipeline || rows == 0 ||
        width == 0 || x_offset % sizeof(float) != 0 ||
        y_offset % sizeof(float) != 0 ||
        (z_offset != SIZE_MAX && z_offset % sizeof(float) != 0) ||
        (weight && (weight->encoding != SALT_TENSOR_ENCODING_F32 ||
                    !weight->value)))
        return -1;
    args.x = x_offset / sizeof(float);
    args.y = y_offset / sizeof(float);
    args.z = z_offset == SIZE_MAX ? 0u : z_offset / sizeof(float);
    args.rows = rows;
    args.width = width;
    args.aux = aux;
    args.eps = eps;
    args.scalar = scalar;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:0 atIndex:0];
    [encoder setBuffer:weight ? weight->value :
                (id<MTLBuffer>)state->canonical.backend
                offset:weight ? weight->value_offset : 0 atIndex:1];
    [encoder setBuffer:_op_status offset:0 atIndex:2];
    [encoder setBytes:&args length:sizeof args atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_topk_rows(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        const MetalTextTensorRef *scales, size_t logits_offset,
        size_t selected_offset, size_t weights_offset,
        uint32_t rows, uint32_t experts, uint32_t topk) {
    return metal_text_encode_rows(state, command, _text_topk_rows_pso,
        scales, logits_offset, selected_offset, weights_offset,
        rows, experts, topk, 0.0f, 0.0f);
}

static int metal_text_encode_softcap(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        size_t logits_offset, uint32_t rows, uint32_t vocabulary, float cap) {
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint64_t x, y, z;
        uint32_t rows, width, aux;
        float eps, scalar;
    } args;
    uint64_t elements = (uint64_t)rows * vocabulary;
    if (!state || !command || !command->command || !(cap > 0.0f) ||
        !isfinite(cap) || elements > UINT32_MAX ||
        logits_offset % sizeof(float) != 0)
        return -1;
    memset(&args, 0, sizeof args);
    args.x = logits_offset / sizeof(float);
    args.rows = rows;
    args.width = vocabulary;
    args.scalar = cap;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_softcap_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:0 atIndex:0];
    [encoder setBuffer:_op_status offset:0 atIndex:1];
    [encoder setBytes:&args length:sizeof args atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_combine(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        const MetalTextLayerRefs *refs, const SaltTextLayerPlan *plan) {
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint64_t residual, dense, routed, scratch_a, scratch_b, output;
        uint32_t rows, width;
        float eps, scalar;
    } args;
    float scalar = 1.0f;
    if (!refs || !plan ||
        refs->post_ffn_norm_1.encoding != SALT_TENSOR_ENCODING_F32 ||
        refs->post_ffn_norm_2.encoding != SALT_TENSOR_ENCODING_F32 ||
        refs->post_ffn_norm.encoding != SALT_TENSOR_ENCODING_F32)
        return -1;
    if (plan->final_layer_scale) {
        scalar = metal_text_ref_value(&refs->layer_scalar, 0);
        if (!isfinite(scalar)) return -1;
    }
    args.residual = layout->state_b / sizeof(float);
    args.dense = layout->dense_output / sizeof(float);
    args.routed = layout->routed_output / sizeof(float);
    args.scratch_a = layout->combine_a / sizeof(float);
    args.scratch_b = layout->combine_b / sizeof(float);
    args.output = layout->state_a / sizeof(float);
    args.rows = command->input_count;
    args.width = state->program->hidden;
    args.eps = state->program->descriptor->norm_epsilon;
    args.scalar = scalar;
    encoder = [command->command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:_text_combine_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:0 atIndex:0];
    [encoder setBuffer:refs->post_ffn_norm_1.value
                offset:refs->post_ffn_norm_1.value_offset atIndex:1];
    [encoder setBuffer:refs->post_ffn_norm_2.value
                offset:refs->post_ffn_norm_2.value_offset atIndex:2];
    [encoder setBuffer:refs->post_ffn_norm.value
                offset:refs->post_ffn_norm.value_offset atIndex:3];
    [encoder setBuffer:_op_status offset:0 atIndex:4];
    [encoder setBytes:&args length:sizeof args atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(command->input_count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_cell(
        void *backend_state, void *command_state,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        const SaltTextExecutionAssignment *assignment,
        uint32_t input_count) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    const SaltTextCanonicalLayout *layout;
    const SaltTextCompiledLayer *compiled = NULL;
    const SaltTextLayerExecDesc *source = NULL;
    const SaltTextLayerPlan *plan = NULL;
    MetalTextLayerRefs *refs = NULL;
    uint32_t actual, output_first = 0u;
    int terminal_output;
    int rc = -1;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        !command->command || command->phase == METAL_TEXT_IDLE ||
        command->phase == METAL_TEXT_RESOURCE_PENDING ||
        command->phase == METAL_TEXT_SUBMITTED ||
        command->phase == METAL_TEXT_FINISHED || state->program != program ||
        command->next_cell >= program->dispatch.cell_count ||
        cell != &program->dispatch.cells[command->next_cell] || !assignment ||
        assignment->cpu.count != 0 || assignment->gpu.first != 0 ||
        input_count != command->input_count)
        return -1;
    terminal_output = cell->kind == SALT_TEXT_CELL_FINAL_NORM ||
        cell->kind == SALT_TEXT_CELL_FINAL_HEAD ||
        cell->kind == SALT_TEXT_CELL_LOGIT_SOFTCAP;
    actual = cell->unit == SALT_TEXT_EXECUTION_ROWS ? input_count :
        cell->unit == SALT_TEXT_EXECUTION_JOBS && cell->layer < program->layer_count
        ? input_count * (uint32_t)program->layers[cell->layer].descriptor->
            plan->top_k_experts : 0u;
    if (terminal_output && command->authoritative_output_rows != input_count) {
        actual = command->authoritative_output_rows;
        output_first = input_count - actual;
    }
    if ((!terminal_output && actual == 0u) || assignment->gpu.count < actual)
        return -1;
    layout = &program->layout;
    if (cell->layer < program->layer_count) {
        compiled = &program->layers[cell->layer];
        source = compiled->descriptor;
        plan = source->plan;
        refs = &state->layers[cell->layer];
    }
    switch (cell->kind) {
    case SALT_TEXT_CELL_EMBEDDING:
        rc = metal_text_encode_embedding(state, command);
        break;
    case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
        rc = metal_text_encode_rows(state, command, _text_rms_rows_pso,
            &refs->pre_attention_norm, layout->state_a, layout->normalized,
            SIZE_MAX, input_count, program->hidden, 1u,
            program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_QUERY_PROJECTION:
        rc = metal_text_encode_projection(state, command, &refs->q,
            layout->normalized, layout->queries, input_count);
        break;
    case SALT_TEXT_CELL_KEY_PROJECTION:
        rc = metal_text_encode_projection(state, command, &refs->k,
            layout->normalized, layout->keys, input_count);
        break;
    case SALT_TEXT_CELL_VALUE_PROJECTION:
        rc = plan->attention.shared_kv_projection ? 0 :
            metal_text_encode_projection(state, command, &refs->v,
                layout->normalized, layout->values, input_count);
        break;
    case SALT_TEXT_CELL_ATTENTION_TRANSFORM:
        rc = metal_text_encode_attention_transform(state, command, cell->layer);
        break;
    case SALT_TEXT_CELL_ATTENTION_BODY:
        rc = metal_text_encode_attention_body(state, command, cell->layer);
        break;
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
        rc = metal_text_encode_projection(state, command, &refs->o,
            layout->attention_output, layout->branch, input_count);
        break;
    case SALT_TEXT_CELL_ATTENTION_COMBINE:
        rc = metal_text_encode_rows(state, command, _text_residual_rows_pso,
            &refs->post_attention_norm, layout->state_a, layout->branch,
            layout->state_b, input_count, program->hidden, 0u,
            program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_DENSE_NORM:
        rc = metal_text_encode_rows(state, command, _text_rms_rows_pso,
            &refs->pre_ffn_norm_1, layout->state_b, layout->normalized,
            SIZE_MAX, input_count, program->hidden, 1u,
            program->descriptor->norm_epsilon, 1.0f);
        if (rc == 0)
            rc = metal_text_encode_rows(state, command, _text_rms_rows_pso,
                &refs->pre_ffn_norm_2, layout->state_b, layout->combine_a,
                SIZE_MAX, input_count, program->hidden, 1u,
                program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_DENSE_GATE:
        rc = metal_text_encode_projection(state, command, &refs->dense_gate,
            layout->normalized, layout->dense_gate, input_count);
        break;
    case SALT_TEXT_CELL_DENSE_UP:
        rc = metal_text_encode_projection(state, command, &refs->dense_up,
            layout->normalized, layout->dense_up, input_count);
        break;
    case SALT_TEXT_CELL_DENSE_ACTIVATION:
        rc = metal_text_encode_activation(state, command,
            layout->dense_gate, layout->dense_up, layout->dense_chain,
            (size_t)input_count * (uint32_t)plan->dense_intermediate);
        break;
    case SALT_TEXT_CELL_DENSE_DOWN:
        rc = metal_text_encode_projection(state, command, &refs->dense_down,
            layout->dense_chain, layout->dense_output, input_count);
        break;
    case SALT_TEXT_CELL_ROUTER_INPUT:
        rc = metal_text_encode_rows(state, command, _text_router_rows_pso,
            &refs->router_scale, layout->state_b, layout->router_input,
            SIZE_MAX, input_count, program->hidden, 0u,
            program->descriptor->norm_epsilon,
            1.0f / salt_sqrtf((float)program->hidden));
        break;
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
        rc = metal_text_encode_projection(state, command, &refs->router,
            layout->router_input, layout->router_logits, input_count);
        break;
    case SALT_TEXT_CELL_ROUTED_NORM:
        rc = 0;
        break;
    case SALT_TEXT_CELL_ROUTER_TOPK:
        rc = metal_text_encode_topk_rows(state, command,
            &refs->per_expert_scale, layout->router_logits,
            layout->selected_experts, layout->selected_weights,
            input_count, (uint32_t)plan->n_experts,
            (uint32_t)plan->top_k_experts);
        if (rc == 0)
            rc = metal_text_encode_route_complete(state, command,
                (uint32_t)plan->n_experts, (uint32_t)plan->top_k_experts);
        break;
    case SALT_TEXT_CELL_EXPERT_GATE:
        command->expert_chain_mask |= 1u; rc = 0; break;
    case SALT_TEXT_CELL_EXPERT_UP:
        command->expert_chain_mask |= 2u; rc = 0; break;
    case SALT_TEXT_CELL_EXPERT_ACTIVATION:
        command->expert_chain_mask |= 4u; rc = 0; break;
    case SALT_TEXT_CELL_EXPERT_DOWN:
        if (command->expert_chain_mask != 7u) return -1;
        rc = metal_text_encode_expert_chain(state, command);
        break;
    case SALT_TEXT_CELL_EXPERT_REDUCTION:
        rc = metal_text_encode_expert_reduce(state, command,
            (uint32_t)plan->top_k_experts);
        break;
    case SALT_TEXT_CELL_FFN_COMBINE:
        rc = metal_text_encode_combine(state, command, refs, plan);
        break;
    case SALT_TEXT_CELL_FINAL_NORM:
        rc = actual == 0u ? 0 :
            metal_text_encode_rows(state, command, _text_rms_rows_pso,
                &state->final_norm,
                layout->state_a + (size_t)output_first * program->hidden *
                    sizeof(float),
                layout->final_state + (size_t)output_first * program->hidden *
                    sizeof(float),
                SIZE_MAX, actual, program->hidden, 1u,
                program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_FINAL_HEAD:
        rc = actual == 0u ? 0 :
            metal_text_encode_projection(state, command, &state->output_head,
                layout->final_state +
                    (size_t)output_first * program->hidden * sizeof(float),
                layout->position_logits +
                    (size_t)output_first * program->vocabulary * sizeof(float),
                actual);
        break;
    case SALT_TEXT_CELL_LOGIT_SOFTCAP: {
        float cap = state->program->layers[state->layer_count - 1u].descriptor->
            plan->logit_softcap;
        rc = actual == 0u ? 0 :
            metal_text_encode_softcap(state, command,
                layout->position_logits +
                    (size_t)output_first * program->vocabulary * sizeof(float),
                actual, program->vocabulary, cap);
        break;
    }
    default:
        return -1;
    }
    if (rc != 0) return -1;
    command->next_cell++;
    command->encoded_cells++;
    command->highest_completion = cell->completion_epoch;
    command->stats.backend_template_reuses++;
    return 0;
}

static int metal_text_submit(void *backend_state, void *command_state) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        !command->command ||
        (command->phase != METAL_TEXT_RECORDING_PREFIX &&
         command->phase != METAL_TEXT_RECORDING_SUFFIX) ||
        command->next_cell != state->program->dispatch.cell_count ||
        command->encoded_cells != state->program->dispatch.cell_count)
        return -1;
    [command->command commit];
    command->stats.backend_host_kernel_launch_calls++;
    command->phase = METAL_TEXT_SUBMITTED;
    return 0;
}

static int metal_text_finish(
        void *backend_state, void *command_state,
        SaltTextExecutionView *view, SaltTextVerifyBackendStats *stats) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    id<MTLCommandBuffer> current;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        command->phase != METAL_TEXT_SUBMITTED ||
        !(current = command->command) || !view || !stats)
        return -1;
    [current waitUntilCompleted];
    command->command = nil;
    if (metal_command_succeeded(current) != 0 ||
        *(uint32_t *)[_op_status contents] != 0u) {
        const char *diag = getenv("SALT_GPU_DIAG");
        if (diag && strcmp(diag, "1") == 0) {
            NSError *error = [current error];
            fprintf(stderr,
                "metal-text-traverse: finish status=%ld op_status=%u "
                "cells=%u/%u epoch=%u/%u error=%s\n",
                (long)[current status], *(uint32_t *)[_op_status contents],
                command->encoded_cells, state->program->dispatch.cell_count,
                command->highest_completion,
                state->program->dispatch.final_dependency_epoch,
                error ? [[error localizedDescription] UTF8String] : "none");
        }
        [current release];
        return -1;
    }
    metal_text_record_gpu_time(command, current);
    [current release];
    command->phase = METAL_TEXT_FINISHED;
    command->stats.final_logits_transfer_bytes =
        (uint64_t)(command->authoritative
            ? command->authoritative_output_rows : command->input_count) *
        state->program->vocabulary *
        sizeof(float);
    view->canonical_base = (unsigned char *)state->canonical.contents;
    view->canonical_bytes = state->canonical_bytes;
    *stats = command->stats;
    if (command->profile_enabled) {
        fprintf(stderr,
            "GEMMA4_TARGET_METAL_WATERFALL gpu_total_ns=%llu "
            "prefix_gpu_ns=%llu continuation_gpu_ns=%llu "
            "prefix_commands=%u continuation_commands=%u "
            "host_launches=%u kernel_nodes=%u "
            "expert_layers=%u expert_unique_total=%llu "
            "expert_rows_total=%llu expert_min_group=%u "
            "expert_max_group=%u\n",
            (unsigned long long)(command->profile_prefix_gpu_ns +
                command->profile_continuation_gpu_ns),
            (unsigned long long)command->profile_prefix_gpu_ns,
            (unsigned long long)command->profile_continuation_gpu_ns,
            command->profile_prefix_commands,
            command->profile_continuation_commands,
            command->stats.backend_host_kernel_launch_calls,
            command->stats.backend_physical_kernel_nodes,
            command->profile_expert_layers,
            (unsigned long long)command->profile_expert_unique_total,
            (unsigned long long)command->profile_expert_rows_total,
            command->profile_expert_min_group,
            command->profile_expert_max_group);
        for (uint32_t layer = 0; layer < state->layer_count; layer++) {
            uint64_t padded = (uint64_t)command->profile_layer_unique[layer] *
                command->profile_layer_max_group[layer];
            uint64_t useful = command->profile_layer_rows[layer];
            uint64_t ratio_ppm = padded ? useful * 1000000u / padded : 0u;
            fprintf(stderr,
                "GEMMA4_TARGET_METAL_OCCUPANCY layer=%u experts=%u "
                "rows=%u max_group=%u padded_rows=%llu "
                "useful_ratio_ppm=%llu\n",
                layer, command->profile_layer_unique[layer],
                command->profile_layer_rows[layer],
                command->profile_layer_max_group[layer],
                (unsigned long long)padded,
                (unsigned long long)ratio_ppm);
        }
    }
    return 0;
}

static int metal_text_resolve(
        void *backend_state, void *command_state,
        uint32_t committed_count, uint32_t input_count,
        uint64_t *scrubbed_bytes) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        command->phase != METAL_TEXT_FINISHED || !scrubbed_bytes ||
        input_count != command->input_count || committed_count > input_count)
        return -1;
    *scrubbed_bytes = 0;
    memset(command, 0, sizeof *command);
    return 0;
}

static int metal_text_scrub(
        void *backend_state, void *command_state, uint64_t *scrubbed_bytes) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        !scrubbed_bytes) return -1;
    if (command->command) {
        id<MTLCommandBuffer> current = command->command;
        command->command = nil;
        if (command->phase == METAL_TEXT_SUBMITTED) {
            [current waitUntilCompleted];
            if (metal_command_succeeded(current) != 0) {
                [current release];
                return -1;
            }
        }
        [current release];
    }
    {
        uint64_t cleared = 0;
        if (command->touched_span_count == 0 ||
            salt_text_touched_span_clear(state->canonical.contents,
                state->canonical_bytes, command->touched_spans,
                command->touched_span_count, &cleared) != 0)
            return -1;
    }
    *scrubbed_bytes = state->program->tentative_kv_bytes;
    memset(command, 0, sizeof *command);
    return 0;
}

static int metal_text_destroy(void *backend_state, void *command_state) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (!state || state->magic != METAL_TEXT_MAGIC || !command ||
        command->phase != METAL_TEXT_IDLE || command->command)
        return -1;
    if (salt_gpu_shared_buffer_free(&state->canonical) != 0) return -1;
    memset(state, 0, sizeof *state);
    memset(command, 0, sizeof *command);
    return 0;
}

static const SaltTextGpuProgramOps metal_text_program_ops = {
    metal_text_requirements,
    metal_text_prepare,
    metal_text_begin,
    metal_text_encode_cell,
    metal_text_encode_extent,
    metal_text_dependency_barrier,
    metal_text_resource_request,
    metal_text_resource_resume,
    metal_text_submit,
    metal_text_finish,
    metal_text_resolve,
    metal_text_scrub,
    metal_text_destroy,
    metal_text_begin_frontier,
    metal_text_begin_authoritative,
    metal_text_begin_authoritative_output,
};

const struct SaltTextGpuProgramOps *salt_gpu_tensor_program_ops(void) {
    return _dev ? &metal_text_program_ops : NULL;
}

const SaltGpuResidencyBackendOps *salt_gpu_residency_backend_ops(void) {
    return NULL;
}
