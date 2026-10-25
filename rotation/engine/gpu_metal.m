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
#include <salt/rotation/fail.h>
#include "salt/rotation/inference_rotation.h"
#include "rotation_arena.h"
#include "../runtime.h"
#include "salt/gpu.h"
#include "include/salt/ledger.h"
#include "salt/trace.h"
#include "salt/gpu_residency.h"
#include "salt/gpu_resource.h"
#include "include/salt/text_verify.h"
#include "salt/bitmath.h"
#include "gpu_metal_ops_source.h"
#include "attention_product_tile.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <dispatch/dispatch.h>

static id<MTLResidencySet> _rotation_trunk_residency;
static uint64_t _rotation_trunk_bytes, _rotation_trunk_members;
static id<MTLBuffer> _rotation_bank_view[SALT_LR_MAX_BANKS], _rotation_arena_backing;
static NSUInteger _rotation_bank_bytes, _rotation_stride;
static int _rotation_arena_slots, _rotation_experts_per_bank;
static id<MTLBuffer> _sel_args_scratch;
static NSUInteger _sel_args_cursor;
#define SEL_ARGS_SCRATCH_BYTES (256u * 1024u)

static NSUInteger sel_args_put(const void *bytes, size_t n) {
    NSUInteger off;
    unsigned char *base;
    if (!_sel_args_scratch || !bytes || n == 0 ||
        n > SEL_ARGS_SCRATCH_BYTES) return (NSUInteger)-1;
    if (_sel_args_cursor + n > SEL_ARGS_SCRATCH_BYTES) _sel_args_cursor = 0;
    off = _sel_args_cursor;
    base = (unsigned char *)[_sel_args_scratch contents];
    if (!base) return (NSUInteger)-1;
    memcpy(base + off, bytes, n);
    _sel_args_cursor = (off + n + 15u) & ~(NSUInteger)15u;
    return off;
}

static void metal_release(id object) {
    if (object) [object release];
}

static int metal_command_succeeded(id<MTLCommandBuffer> command) {
    if ([command status] == MTLCommandBufferStatusCompleted) return 0;
    NSError *error = [command error];
    fprintf(stderr, "gpu: Metal command failed%s%s\n",
            error ? ": " : "",
            error ? [[error localizedDescription] UTF8String] : "");
    return SALT_TFAIL(-1);
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
static id<MTLSharedEvent> _rotation_done_event, _rotation_ready_event;
static id<MTLSharedEvent> _command_done_event;
static uint64_t _command_done_value;
static uint64_t _command_done_waited, _command_done_harvested;
static uint64_t _command_done_forced;
static double _mt_probe_ns(void);
static void metal_text_arrival_release(void);
static void metal_manifest_gpu_window_record(
    id<MTLCommandBuffer> command_buffer, uint32_t layer,
    uint32_t stage, uint32_t experts, uint64_t rows,
    uint64_t commit_ns, uint64_t wait_return_ns);
static uint64_t metal_command_event_encode(id<MTLCommandBuffer> command);
static int metal_command_event_wait(uint64_t value);
static int metal_command_event_record(id<MTLCommandBuffer> command,
    uint32_t layer, uint32_t stage, uint32_t experts, uint64_t rows,
    uint64_t commit_ns, uint64_t wait_return_ns);
static int metal_command_event_flush(void);
static uint64_t _rotation_epoch;
static int _rotation_stream_error;
static MTLSharedEventListener *_rotation_done_listener;
static uint64_t _rotation_done_armed, _rotation_done_notifications;
static void metal_rotation_arm_done(uint64_t base, uint32_t layers) {
    /* Register CPU observability before the GPU stream is submitted, not
     * after the loader reaches a wait inside an already running stream.
     * Capture the registered layer threshold, not the callback's current event
     * value: several actual DONE signals may arrive before a callback runs. */
    for (uint32_t layer = 0; layer < layers; layer++) {
        const uint64_t completed_value = base + layer + 1u;
        [_rotation_done_event notifyListener:_rotation_done_listener
            atValue:completed_value
            block:^(id<MTLSharedEvent> event, uint64_t value) {
                const uint64_t callback_enter_ns = _mt_probe_ns();
                (void)event;
                (void)value;
                __atomic_fetch_add(&_rotation_done_notifications, 1u, __ATOMIC_RELAXED);
                salt_lr_compute_done_tick(completed_value, callback_enter_ns);
            }];
        _rotation_done_armed++;
    }
}
static uint64_t _rotation_done_value_reads, _rotation_done_value_hits;
static uint64_t _rotation_done_value_yields;
static int metal_rotation_wait_done(void *context, uint64_t value);
static void metal_rotation_publish_ready(void *context, uint64_t value);
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
static id<MTLComputePipelineState> _text_combine_multi_pso;
static id<MTLComputePipelineState> _text_rms_multi_pso;
static id<MTLComputePipelineState> _text_residual_multi_pso;
static id<MTLComputePipelineState> _text_router_multi_pso;
static id<MTLComputePipelineState> _text_topk_rows_pso;
static id<MTLComputePipelineState> _text_softcap_pso;
static id<MTLComputePipelineState> _text_transform_view_pso;
static id<MTLComputePipelineState> _text_attention_view_pso;
static id<MTLComputePipelineState> _text_attention_product_view_pso;
static uint64_t _attention_decode_product_calls, _attention_decode_control_calls;
static uint64_t _attention_prefill_control_calls;
static id<MTLComputePipelineState> _exact_rms_pso, _exact_topk_pso;
static id<MTLComputePipelineState> _exact_residual_pso, _exact_combine_pso;
static id<MTLComputePipelineState> _exact_softcap_pso, _exact_router_pso;
static id<MTLBuffer> _op_meta, _op_status;

/* The selected-expert kernels raise status 2 when an expert is absent from
 * the device-visible slot table.  The model polls this to decide whether to
 * resolve and retry rather than treat it as a hard failure. */
int salt_gpu_selected_missed(void) {
    if (!_op_status || ![_op_status contents]) return 0;
    return *(uint32_t *)[_op_status contents] == 2u ? 1 : 0;
}
typedef struct {
    id<MTLBuffer> buffer;
    void *contents;
    size_t nbytes;
} MetalSharedSlot;
#define METAL_SHARED_SLOTS 11
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
        _text_combine_multi_pso && _text_rms_multi_pso &&
        _text_residual_multi_pso && _text_router_multi_pso &&
        _text_rms_rows_pso && _text_residual_rows_pso &&
        _text_router_rows_pso && _text_topk_rows_pso && _text_softcap_pso &&
        _text_transform_view_pso && _text_attention_view_pso &&
        _text_attention_product_view_pso)
        return 0;
    @autoreleasepool {
        NSError *err = nil;
        NSString *product_source = rotation_attention_product_source();
        if (!product_source) return SALT_TFAIL(-1);
        id<MTLLibrary> lib = metal_library_for_source_mode(
            [product_source UTF8String], &err, 0, 0);
        if (!lib) {
            fprintf(stderr, "gpu: Metal operator compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return SALT_TFAIL(-1);
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
            {"text_parallel_combine_multirows", &_text_combine_multi_pso},
            {"text_rmsnorm_multirows", &_text_rms_multi_pso},
            {"text_residual_multirows", &_text_residual_multi_pso},
            {"text_router_input_multirows", &_text_router_multi_pso},
            {"text_topk_rows", &_text_topk_rows_pso},
            {"text_softcap", &_text_softcap_pso},
            {"attention_transform_view", &_text_transform_view_pso},
            {"attention_exact_view", &_text_attention_view_pso},
            {"attention_exact_product_view", &_text_attention_product_view_pso},
        };
        for (size_t i = 0; i < sizeof specs / sizeof specs[0]; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:specs[i].name]];
            if (!fn) { [lib release]; return SALT_TFAIL(-1); }
            *specs[i].slot = [_dev newComputePipelineStateWithFunction:fn
                                                                  error:&err];
            [fn release];
            if (!*specs[i].slot) { [lib release]; return SALT_TFAIL(-1); }
        }
        [lib release];
        _op_status = [_dev newBufferWithLength:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
        _op_meta = [_dev newBufferWithLength:2u * 1024u * 1024u
                                      options:MTLResourceStorageModeShared];
        if (!_op_status || !_op_meta) return SALT_TFAIL(-1);
    }
    return 0;
}

static int metal_exact_prepare(void) {
    id<MTLComputePipelineState> created[6] = {nil, nil, nil, nil, nil, nil};
    if (_exact_rms_pso && _exact_topk_pso && _exact_residual_pso &&
        _exact_combine_pso && _exact_softcap_pso && _exact_router_pso)
        return 0;
    if (metal_ops_prepare() != 0) return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    }
}

static int metal_shared_range(const SaltGpuSharedBuffer *buffer,
                              size_t float_offset, size_t float_count,
                              id<MTLBuffer> *storage_out) {
    id<MTLBuffer> storage;
    size_t capacity;
    if (!buffer || !buffer->contents || !buffer->backend || !storage_out ||
        buffer->nbytes < sizeof(float)) return SALT_TFAIL(-1);
    storage = (id<MTLBuffer>)buffer->backend;
    capacity = buffer->nbytes / sizeof(float);
    if ([storage contents] != buffer->contents ||
        (size_t)[storage length] < buffer->nbytes ||
        float_offset > capacity || float_count > capacity - float_offset)
        return SALT_TFAIL(-1);
    *storage_out = storage;
    return 0;
}

static int metal_shared_pointer(const float *pointer, size_t count,
                                id<MTLBuffer> *buffer_out,
                                size_t *byte_offset_out) {
    uintptr_t p = (uintptr_t)(const void *)pointer;
    size_t bytes;
    if (!pointer || !buffer_out || !byte_offset_out ||
        count > SIZE_MAX / sizeof(float)) return SALT_TFAIL(-1);
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
    return SALT_TFAIL(-1);
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
    if (!requirements || requirements_size != sizeof *requirements) return SALT_TFAIL(-1);
    memset(requirements, 0, sizeof *requirements);
    requirements->shared_buffer_slots = METAL_SHARED_SLOTS;
    return 0;
}

/* THE FIXED CONFIGURATION (invariant: configuration is a fixed input to the
 * state machine, not something consulted during operation).  Every switch below
 * is read EXACTLY ONCE, here, at the earliest initialization, and is never
 * consulted again on the operation path.  Each is a non-NULL-and-not-"0" test so
 * that a set-to-"0" variable means OFF, which getenv() truthiness gets wrong. */
static int _cfg_diag, _cfg_ms, _cfg_tiled, _cfg_ms_detail, _cfg_moe_ms_detail;
static int _cfg_address_audit, _cfg_expert_prefetch, _cfg_cb_empty, _cfg_no_useresource;

/* Failure origins use the engine-owned SALT_TFAIL recorder. */
/* THE ENGINE'S OWN RESIDENT CEILING, RESOLVED ONCE.  The engine drives the walk, so the engine is
 * what has to bound it; the model supplies weights, not control flow.  Decimal GB, as the recipe
 * states it. */
static uint64_t _cfg_mem_limit_bytes;
static const char *_cfg_dispatch, *_cfg_target_waterfall;
static int _cfg_on(const char *name) {
    const char *v = getenv(name);
    return v != NULL && strcmp(v, "0") != 0;
}
static void metal_cfg_resolve(void) {
    _cfg_diag = _cfg_on("SALT_GPU_DIAG");
    _cfg_ms = _cfg_on("SALT_GPU_MS");
    _cfg_tiled = _cfg_on("SALT_GPU_TILED");
    _cfg_ms_detail = _cfg_on("SALT_GPU_MS_DETAIL");
    _cfg_moe_ms_detail = _cfg_on("SALT_GPU_MOE_MS_DETAIL");
    _cfg_address_audit = _cfg_on("SALT_GPU_ADDRESS_AUDIT");
    _cfg_expert_prefetch = _cfg_on("SALT_EXPERT_PREFETCH");
    {
        const char *ml = getenv("SALT_GEMMA_MEMORY_LIMIT_GB");
        if (ml && *ml) {
            double gb = strtod(ml, (char **)0);
            if (gb > 0.0 && gb < 1.0e6)
                _cfg_mem_limit_bytes = (uint64_t)(gb * 1.0e9);
        }
    }
    _cfg_cb_empty = _cfg_on("SALT_CB_EMPTY_PROBE");
    _cfg_no_useresource = _cfg_on("SALT_MOE_NO_USERESOURCE");
    _cfg_dispatch = _cfg_dispatch;
    _cfg_target_waterfall = _cfg_target_waterfall;
}

static void metal_text_manifest_init(void);   /* the storage, allocated at init */

int salt_gpu_init(void) {
    metal_text_manifest_init();
    if (_dev) return 0;
    metal_cfg_resolve();
    @autoreleasepool {
        _dev = MTLCreateSystemDefaultDevice();
        if (!_dev) return SALT_TFAIL(-1);
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
                return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q4mv"];
        [lib release];
        if (!fn) {
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return SALT_TFAIL(-1);
        }
        _pso = [_dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        if (!_pso) {
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return SALT_TFAIL(-1);
        }
        _queue = [_dev newCommandQueue];
        if (!_queue) {
            [_pso release];
            _pso = nil;
            metal_release(_offline_library);
            _offline_library = nil;
            [_dev release];
            _dev = nil;
            return SALT_TFAIL(-1);
        }
        if (metal_ops_prepare() != 0 || metal_session_buffers_prepare() != 0) {
            (void)salt_gpu_free_impl();
            return SALT_TFAIL(-1);
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
    if (!buffer || !_dev || nbytes < 1) return SALT_TFAIL(-1);
    memset(buffer, 0, sizeof *buffer);
    if (nbytes > (size_t)NSUIntegerMax) return SALT_TFAIL(-1);
    @autoreleasepool {
        storage = [_dev newBufferWithLength:nbytes
                                    options:MTLResourceStorageModeShared];
        if (!storage || ![storage contents] || [storage length] < nbytes) {
            metal_release(storage);
            return SALT_TFAIL(-1);
        }
        buffer->contents = [storage contents];
        buffer->nbytes = (size_t)[storage length];
        buffer->backend = (void *)storage;
        for (int i = 0; i < METAL_SHARED_SLOTS; i++)
            if (!_shared_slots[i].buffer) { shared_slot = i; break; }
        if (shared_slot < 0) {
            memset(buffer, 0, sizeof *buffer);
            [storage release];
            return SALT_TFAIL(-1);
        }
        _shared_slots[shared_slot] = (MetalSharedSlot) {
            storage, [storage contents], (size_t)[storage length],
        };
    }
    return 0;
}

int salt_gpu_shared_buffer_map_readonly(SaltGpuSharedBuffer *buffer,
                                        const void *contents, size_t nbytes) {
    id<MTLBuffer> storage = nil;
    long page = sysconf(_SC_PAGESIZE);
    int shared_slot = -1;
    if (!buffer || !_dev || !contents || nbytes < 1 || page < 1 ||
        (uintptr_t)contents % (uintptr_t)page != 0 ||
        nbytes % (size_t)page != 0 || nbytes > (size_t)NSUIntegerMax)
        return SALT_TFAIL(-1);
    memset(buffer, 0, sizeof *buffer);
    @autoreleasepool {
        storage = [_dev newBufferWithBytesNoCopy:(void *)contents length:nbytes
            options:MTLResourceStorageModeShared deallocator:nil];
        if (!storage)
            storage = [_dev newBufferWithBytesNoCopy:(void *)contents
                length:nbytes options:MTLResourceStorageModeManaged
                deallocator:nil];
        if (!storage || [storage contents] != contents ||
            (size_t)[storage length] < nbytes) {
            metal_release(storage);
            return SALT_TFAIL(-1);
        }
        for (int i = 0; i < METAL_SHARED_SLOTS; i++)
            if (!_shared_slots[i].buffer) { shared_slot = i; break; }
        if (shared_slot < 0) {
            [storage release];
            return SALT_TFAIL(-1);
        }
        buffer->contents = (void *)contents;
        buffer->nbytes = nbytes;
        buffer->backend = (void *)storage;
        _shared_slots[shared_slot] = (MetalSharedSlot) {
            storage, (void *)contents, nbytes,
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
    if (!cell || view < 0 || view >= 8 || !cell->views[view]) return SALT_TFAIL(-1);
    return metal_shared_range(cell->views[view], cell->offsets[view],
                              count, storage);
}

static int metal_exact_encode(id<MTLCommandBuffer> command,
                              const SaltGpuExactCell *cell) {
    MetalExactArgs args;
    id<MTLBuffer> views[8] = {nil, nil, nil, nil, nil, nil, nil, nil};
    id<MTLComputePipelineState> pipeline = nil;
    int used = 0;
    if (!command || !cell || cell->n < 1) return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
        pipeline = _exact_rms_pso; used = 3;
        break;
    case SALT_GPU_EXACT_ROUTER_INPUT:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) ||
            !(cell->scalar > 0.0f) || !isfinite(cell->scalar) ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->n, &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->n, &views[2]) != 0)
            return SALT_TFAIL(-1);
        pipeline = _exact_router_pso; used = 3;
        break;
    case SALT_GPU_EXACT_TOPK:
        if (cell->aux < 1u || cell->aux > cell->n ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->n, &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->aux, &views[2]) != 0 ||
            metal_exact_view(cell, 3, cell->aux, &views[3]) != 0)
            return SALT_TFAIL(-1);
        pipeline = _exact_topk_pso; used = 4;
        break;
    case SALT_GPU_EXACT_RESIDUAL_POSTNORM:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0 ||
            metal_exact_view(cell, 1, cell->n, &views[1]) != 0 ||
            metal_exact_view(cell, 2, cell->n, &views[2]) != 0 ||
            metal_exact_view(cell, 3, cell->n, &views[3]) != 0)
            return SALT_TFAIL(-1);
        pipeline = _exact_residual_pso; used = 4;
        break;
    case SALT_GPU_EXACT_PARALLEL_COMBINE:
        if (!(cell->eps > 0.0f) || !isfinite(cell->eps) ||
            !isfinite(cell->scalar)) return SALT_TFAIL(-1);
        for (int i = 0; i < 8; i++)
            if (metal_exact_view(cell, i,
                    i == 6 ? (size_t)cell->n * 2u : cell->n,
                    &views[i]) != 0)
                return SALT_TFAIL(-1);
        pipeline = _exact_combine_pso; used = 8;
        break;
    case SALT_GPU_EXACT_SOFTCAP:
        if (!(cell->scalar > 0.0f) || !isfinite(cell->scalar) ||
            metal_exact_view(cell, 0, cell->n, &views[0]) != 0)
            return SALT_TFAIL(-1);
        pipeline = _exact_softcap_pso; used = 1;
        break;
    default:
        return SALT_TFAIL(-1);
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder || !pipeline) return SALT_TFAIL(-1);
        [encoder setComputePipelineState:pipeline];
        for (int i = 0; i < used; i++)
            [encoder setBuffer:views[i] offset:0 atIndex:(NSUInteger)i];
        [encoder setBuffer:_op_status offset:0 atIndex:(NSUInteger)used];
        {
            NSUInteger _off = sel_args_put(&args, sizeof args);
            if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
            [encoder setBuffer:_sel_args_scratch offset:_off
                       atIndex:(NSUInteger)(used + 1)];
        }
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
        return SALT_TFAIL(-1);
    *(uint32_t *)[_op_status contents] = 0;
    command = [_queue commandBuffer];
    if (!command) return SALT_TFAIL(-1);
    for (int index = 0; index < count; index++)
        if (metal_exact_encode(command, &cells[index]) != 0)
            return SALT_TFAIL(-1);
    [command commit];
    [command waitUntilCompleted];
    return metal_command_succeeded(command) == 0 &&
        *(uint32_t *)[_op_status contents] == 0 ? 0 : -1;
}

int salt_gpu_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y) {
    if (!_dev || !_pso || R < 1 || C < 1) return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
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
                return SALT_TFAIL(-1);
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
            if (!bx) return SALT_TFAIL(-1);
            metal_release(_bx);
            _bx = bx;
        }
        if (!_by || (size_t)R * sizeof(float) > _by.length) {
            id<MTLBuffer> by = [_dev newBufferWithLength:
                (size_t)R * sizeof(float) options:MTLResourceStorageModeShared];
            if (!by) return SALT_TFAIL(-1);
            metal_release(_by);
            _by = by;
        }
        memcpy(_bx.contents, x, (size_t)C * sizeof(float));

        id<MTLCommandBuffer> cb = [_queue commandBuffer];
        if (!cb) return SALT_TFAIL(-1);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return SALT_TFAIL(-1);
        [enc setComputePipelineState:_pso];
        [enc setBuffer:_bv offset:0 atIndex:0];
        [enc setBuffer:_bs offset:0 atIndex:1];
        [enc setBuffer:_bb offset:0 atIndex:2];
        [enc setBuffer:_bx offset:0 atIndex:3];
        [enc setBuffer:_by offset:0 atIndex:4];
        uint rv = (uint)R, cv = (uint)C;
        {
        NSUInteger _off = sel_args_put(&rv, sizeof rv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
        {
        NSUInteger _off = sel_args_put(&cv, sizeof cv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake((NSUInteger)R, 1, 1);
        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return SALT_TFAIL(-1);
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
    "    uint linear=ti.y*8u+ti.x;\n"
    "    bool active=r<R&&token<B;float4 a0=float4(0.0f),a1=float4(0.0f);\n"
    "    float4 a2=float4(0.0f),a3=float4(0.0f),a4=float4(0.0f),a5=float4(0.0f);\n"
    "    float4 a6=float4(0.0f),a7=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){\n"
    "        /* DOUBLE BUFFER: alternate tile halves so the NEXT iteration's load cannot\n"
    "         * overwrite the half this iteration is still reading.  That is exactly what\n"
    "         * the second threadgroup_barrier was for, so removing it halves the barrier\n"
    "         * count (176 -> 88 per threadgroup at C=2816) with no arithmetic change. */\n"
    "        uint ph=((c>>5u)&1u)*512u;threadgroup float *wt=tile+ph,*xt=tile+ph+256u;\n"
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
    "#ifdef SALT_METAL_SELECTED_CAPACITY\n"
    "struct SelectedResources { array<device const uchar *,SALT_METAL_SELECTED_CAPACITY> resource [[id(0)]]; };\n"
    "kernel void q4selected(\n"
    "    constant SelectedResources &resources [[buffer(0)]],device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]],device const ulong4 *desc [[buffer(3)]],\n"
    "    constant uint &ndesc [[buffer(4)]],device const int *logical_slots [[buffer(5)]],\n"
    "    device const uint *payload_offsets [[buffer(6)]],device const int *selected [[buffer(7)]],\n"
    "    device atomic_uint *status [[buffer(8)]],uint3 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    uint r=pos.x,token=pos.y,di=pos.z;if(di>=ndesc)return;ulong4 d=desc[di*3u],db=desc[di*3u+1u],dc=desc[di*3u+2u];\n"
    "    uint R=uint(db.y),C=uint(db.z),B=uint(db.w);if(r>=R||token>=B)return;uint slot=uint(dc.x);ulong payload=0u;if(dc.y!=0u){uint logical=uint(dc.z);int mapped=logical_slots[logical];if(mapped<0){atomic_store_explicit(status,2u,memory_order_relaxed);return;}slot=uint(mapped);payload=0u;}\n"
    "    if(slot>=SALT_METAL_SELECTED_CAPACITY){atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
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
    "struct RotationExpertArgs { uint jobs,R,C,ne,layer,projection,projections,bank_origin,arena_slots; ulong stride,input,output,second_output; };\n"
    "kernel void q4selectedragged(\n"
    "    device const uchar *arena [[buffer(0)]],device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]],device const ulong *manifest [[buffer(3)]],\n"
    "    constant RotationExpertArgs &args [[buffer(4)]],device const int *logical_slots [[buffer(5)]],\n"
    "    device const int *grouped [[buffer(7)]],device atomic_uint *status [[buffer(8)]],\n"
    "    device const int *selected [[buffer(9)]],uint3 pos [[thread_position_in_grid]])\n"
    "{\n"
    "    uint r=pos.x,token=pos.y,R=args.R,C=args.C;if(r>=R||token>=args.jobs||pos.z>=args.projections)return;\n"
    "    int canonical=grouped[token];if(canonical<0||uint(canonical)>=args.jobs){atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    int expert=selected[canonical];if(expert<0||uint(expert)>=args.ne){atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    uint logical=args.layer*args.ne+uint(expert);int slot=logical_slots[logical];\n"
    "    if(slot<0||uint(slot)>=args.arena_slots){atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    device const ulong *mf=manifest+ulong(logical)*9u;uint projection=args.projection+pos.z;ulong origin=mf[0];\n"
    "    device const uchar *base=arena+ulong(uint(slot)-args.bank_origin)*args.stride;\n"
    "    device const uchar *vr=base+mf[projection*3u]-origin,*sr=base+mf[projection*3u+1u]-origin,*br=base+mf[projection*3u+2u]-origin;\n"
    "    device const float *xj=x+args.input+ulong(token)*C;device float *yj=ys+(pos.z==0u?args.output:args.second_output)+ulong(token)*R;\n"
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
    "kernel void q4rotationrows4(\n"
    "    device const uchar *arena [[buffer(0)]],device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]],device const ulong *manifest [[buffer(3)]],\n"
    "    constant RotationExpertArgs &args [[buffer(4)]],device const int *logical_slots [[buffer(5)]],\n"
    "    device const int *grouped [[buffer(7)]],device atomic_uint *status [[buffer(8)]],\n"
    "    device const int *selected [[buffer(9)]],uint3 group [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_index_in_simdgroup]],uint warp [[simdgroup_index_in_threadgroup]])\n"
    "{\n"
    "    uint task=group.y;if(task>=args.jobs||group.z>=args.projections)return;\n"
    "    int canonical=grouped[task];if(canonical<0||uint(canonical)>=args.jobs){if(lane==0u&&warp==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    int expert=selected[canonical];if(expert<0||uint(expert)>=args.ne){if(lane==0u&&warp==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    uint logical=args.layer*args.ne+uint(expert);int slot=logical_slots[logical];\n"
    "    if(slot<0||uint(slot)>=args.arena_slots){if(lane==0u&&warp==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    uint projection=args.projection+group.z,R=args.R,C=args.C;\n"
    "    uint sub=lane>>3u,k8=lane&7u,r0=group.x*128u+warp*16u+sub*4u;\n"
    "    device const ulong *mf=manifest+ulong(logical)*9u;ulong origin=mf[0];\n"
    "    device const uchar *base=arena+ulong(uint(slot)-args.bank_origin)*args.stride;\n"
    "    device const uchar *vr=base+mf[projection*3u]-origin,*sr=base+mf[projection*3u+1u]-origin,*br=base+mf[projection*3u+2u]-origin;\n"
    "    device const float *xj=x+args.input+ulong(task)*C;\n"
    "    float4 acc0=float4(0.0f),acc1=float4(0.0f),acc2=float4(0.0f),acc3=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){float4 xv=float4(xj[c+k8*4u],xj[c+k8*4u+1u],xj[c+k8*4u+2u],xj[c+k8*4u+3u]);\n"
    "        for(uint rr=0u;rr<4u;rr++){uint r=r0+rr;if(r<R){ulong kb=ulong(r)*C+c,k=kb+k8*4u,g=kb/64u;\n"
    "            uint packed=load32(vr+(k>>3u)*4u),sh=(k8&1u)*16u;\n"
    "            float4 q=float4((packed>>sh)&15u,(packed>>(sh+4u))&15u,(packed>>(sh+8u))&15u,(packed>>(sh+12u))&15u);\n"
    "            float s=as_type<float>(uint(load16(sr+g*2u))<<16),b=as_type<float>(uint(load16(br+g*2u))<<16);\n"
    "            float4 w=q*s;w+=b;float4 p=w*xv;if(rr==0u)acc0+=p;else if(rr==1u)acc1+=p;else if(rr==2u)acc2+=p;else acc3+=p;}}}\n"
    "    uint base_lane=sub*8u;float4 sum0=float4(0.0f),sum1=float4(0.0f),sum2=float4(0.0f),sum3=float4(0.0f);\n"
    "    for(uint a=0u;a<8u;a++){ushort src=ushort(base_lane+a);sum0+=simd_shuffle(acc0,src);\n"
    "        sum1+=simd_shuffle(acc1,src);sum2+=simd_shuffle(acc2,src);sum3+=simd_shuffle(acc3,src);}\n"
    "    if(k8==0u){float4 sums[4]={sum0,sum1,sum2,sum3};ulong out=group.z==0u?args.output:args.second_output;\n"
    "        for(uint rr=0u;rr<4u&&r0+rr<R;rr++){float2 tt=sums[rr].xy+sums[rr].zw;ys[out+ulong(task)*R+r0+rr]=tt.x+tt.y;}}\n"
    "}\n"
    "kernel void q4rotationgrouped(\n"
    "    device const uchar *arena [[buffer(0)]], device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]], device const ulong *manifest [[buffer(3)]],\n"
    "    constant RotationExpertArgs &args [[buffer(4)]], device const int *logical_slots [[buffer(5)]],\n"
    "    device const int *group_ends [[buffer(6)]], device const int *grouped [[buffer(7)]],\n"
    "    device atomic_uint *status [[buffer(8)]], device const int *selected [[buffer(9)]],\n"
    "    threadgroup float *tile [[threadgroup(0)]],\n"
    "    uint3 grid [[threadgroup_position_in_grid]], uint3 ti [[thread_position_in_threadgroup]])\n"
    "{\n"
    "    int raw_count=group_ends[args.ne];\n"
    "    if(raw_count<0||uint(raw_count)>args.ne+(args.jobs+7u)/8u){\n"
    "        if(ti.x==0u&&ti.y==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    if(grid.y>=uint(raw_count))return;\n"
    "    int raw_ex=group_ends[args.ne+1u+2u*grid.y];\n"
    "    int raw_tile=group_ends[args.ne+2u+2u*grid.y];\n"
    "    if(raw_ex<0||uint(raw_ex)>=args.ne||raw_tile<0){\n"
    "        if(ti.x==0u&&ti.y==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    uint ex=uint(raw_ex),local_tile=uint(raw_tile);\n"
    "    int raw_begin=ex?group_ends[ex-1u]:0,raw_end=group_ends[ex];\n"
    "    if(raw_begin<0||raw_end<=raw_begin||uint(raw_end)>args.jobs||\n"
    "       local_tile>=(uint(raw_end-raw_begin)+7u)/8u){\n"
    "        if(ti.x==0u&&ti.y==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    uint start=uint(raw_begin),end=uint(raw_end);\n"
    "    uint logical=args.layer*args.ne+ex;\n"
    "    int slot=logical_slots[logical];\n"
    "    if(slot<0||uint(slot)>=args.arena_slots){\n"
    "        if(ti.x==0u&&ti.y==0u)atomic_store_explicit(status,2u,memory_order_relaxed);\n"
    "        return;\n"
    "    }\n"
    "    uint projection=args.projection+grid.z;\n"
    "    device const ulong *mf=manifest+ulong(logical)*9u;\n"
    "    ulong origin=mf[0],vo=mf[projection*3u],so=mf[projection*3u+1u],bo=mf[projection*3u+2u];\n"
    "    if(vo<origin||so<origin||bo<origin){\n"
    "        if(ti.x==0u&&ti.y==0u)atomic_store_explicit(status,2u,memory_order_relaxed);\n"
    "        return;\n"
    "    }\n"
    "    uint check=start+local_tile*8u+ti.y;\n"
    "    if(check<end){int canonical=grouped[check];\n"
    "        if(canonical<0||uint(canonical)>=args.jobs||selected[canonical]!=int(ex))\n"
    "            atomic_store_explicit(status,2u,memory_order_relaxed);\n"
    "    }\n"
    "    device const uchar *vals=arena+ulong(uint(slot)-args.bank_origin)*args.stride;\n"
    "    device const uchar *scales=vals,*biases=vals;\n"
    "    ulong out=grid.z==0u?args.output:args.second_output;\n"
    "    ulong4 d=ulong4(vo-origin,so-origin,args.input+ulong(start)*args.C,\n"
    "                   out+ulong(start)*args.R);\n"
    "    ulong4 db=ulong4(bo-origin,args.R,args.C,end-start);\n"
    "    uint2 tg=uint2(grid.x,local_tile);\n"
    "    uint rbase=tg.x*8u,tbase=tg.y*8u,r=rbase+ti.x,token=tbase+ti.y;\n"
    "    uint linear=ti.y*8u+ti.x;\n"
    "    uint R=args.R,C=args.C,B=end-start;\n"
    "    bool active=r<R&&token<B;float4 a0=float4(0.0f),a1=float4(0.0f);\n"
    "    float4 a2=float4(0.0f),a3=float4(0.0f),a4=float4(0.0f),a5=float4(0.0f);\n"
    "    float4 a6=float4(0.0f),a7=float4(0.0f);\n"
    "    for(uint c=0u;c<C;c+=32u){\n"
    "        /* DOUBLE BUFFER: alternate tile halves so the NEXT iteration's load cannot\n"
    "         * overwrite the half this iteration is still reading.  That is exactly what\n"
    "         * the second threadgroup_barrier was for, so removing it halves the barrier\n"
    "         * count (176 -> 88 per threadgroup at C=2816) with no arithmetic change. */\n"
    "        uint ph=((c>>5u)&1u)*512u;threadgroup float *wt=tile+ph,*xt=tile+ph+256u;\n"
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
    "    }\n"
    "    if(active){float4 s4=float4(0.0f);s4+=a0;s4+=a1;s4+=a2;s4+=a3;\n"
    "        s4+=a4;s4+=a5;s4+=a6;s4+=a7;float2 tt=s4.xy+s4.zw;\n"
    "        ys[d.w+ulong(token)*R+r]=tt.x+tt.y;}\n"
    "}\n"
    "kernel void q4selectedwarp(\n"
    "    constant SelectedResources &resources [[buffer(0)]],device const float *x [[buffer(1)]],\n"
    "    device float *ys [[buffer(2)]],device const ulong4 *desc [[buffer(3)]],\n"
    "    constant uint2 &counts [[buffer(4)]],device const int *logical_slots [[buffer(5)]],\n"
    "    device const uint *payload_offsets [[buffer(6)]],device const uint2 *tasks [[buffer(7)]],\n"
    "    device atomic_uint *status [[buffer(8)]],\n"
    "    uint3 group [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_index_in_simdgroup]],\n"
    "    uint warp [[simdgroup_index_in_threadgroup]])\n"
    "{\n"
    "    uint sub=lane>>3u;uint k8=lane&7u;\n"
    "    uint r=group.x*32u+warp*4u+sub,task=group.y,ndesc=counts.x,ntasks=counts.y;if(task>=ntasks)return;\n"
    "    uint2 work=tasks[task];uint di=work.x,token=work.y;if(di>=ndesc)return;\n"
    "    ulong4 d=desc[di*3u],db=desc[di*3u+1u],dc=desc[di*3u+2u];\n"
    "    uint R=uint(db.y),C=uint(db.z),B=uint(db.w);if(r>=R||token>=B)return;\n"
    "    /* DEVICE RESOLVE, not a synthetic miss.  The gate/up sibling of this kernel has\n"
    "     * always resolved logical->slot through _selected_logical_slots; the down pass\n"
    "     * rejected dc.y instead, which is why the HOST had to resolve every layer and\n"
    "     * why the per-layer resource_request yield exists.  The encode already writes\n"
    "     * dc={slot,1u,logical,0u} for the down descriptor, so the logical id is in dc.z\n"
    "     * and the table is already bound here as buffer 5.  Same table, same order, same\n"
    "     * addresses as the host's resolution - bit-identical by construction. */\n"
    "    uint slot=uint(dc.x);\n"
    "    if(dc.y!=0u){uint logical=uint(dc.z);int mapped=logical_slots[logical];\n"
    "        if(mapped<0){if(lane==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "        slot=uint(mapped);}\n"
    "    if(slot>=SALT_METAL_SELECTED_CAPACITY){if(lane==0u)atomic_store_explicit(status,2u,memory_order_relaxed);return;}\n"
    "    device const uchar *base=resources.resource[slot];device const uchar *vr=base+d.x,*sr=base+d.y,*br=base+db.x;\n"
    "    device const float *xj=x+d.z+ulong(token)*ulong(C);\n"
    "    float4 acc=float4(0.0f);\n"
    "    for(uint cb=0u;cb<C;cb+=32u){\n"
    "        uint kb=r*C+cb,g=kb/64u,k=kb+k8*4u;\n"
    "        uint u=load32(vr+(k>>3u)*4u);uint sh=(k8&1u)*16u;\n"
    "        float4 q=float4((u>>sh)&15u,(u>>(sh+4u))&15u,(u>>(sh+8u))&15u,(u>>(sh+12u))&15u);\n"
    "        float s=as_type<float>(uint(load16(sr+g*2u))<<16);\n"
    "        float b=as_type<float>(uint(load16(br+g*2u))<<16);\n"
    "        float4 w=q*s;w+=b;\n"
    "        acc+=w*float4(xj[cb+k8*4u],xj[cb+k8*4u+1u],xj[cb+k8*4u+2u],xj[cb+k8*4u+3u]);}\n"
    "    uint base_lane=(lane&~7u);\n"
    "    float4 s4=float4(0.0f);\n"
    "    for(uint a=0u;a<8u;a++)s4+=simd_shuffle(acc,ushort(base_lane+a));\n"
    "    float tt0=s4.x+s4.z,tt1=s4.y+s4.w;\n"
    "    if(k8==0u)ys[d.w+ulong(token)*ulong(R)+r]=tt0+tt1;\n"
    "}\n"
    "#endif\n"
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
    "}\n"
    "struct MtpQ4Args { ulong weight_off,input_off,output_off; uint rows,cols; };\n"
    "kernel void mtp_q4_0(device const uchar *source [[buffer(0)]],\n"
    "    device uchar *scratch [[buffer(1)]],constant MtpQ4Args &args [[buffer(2)]],\n"
    "    uint row [[thread_position_in_grid]]){\n"
    "    if(row>=args.rows)return;device const float *x=(device const float *)(scratch+args.input_off);\n"
    "    device float *y=(device float *)(scratch+args.output_off);\n"
    "    ulong row_bytes=ulong(args.cols/32u)*18u;device const uchar *weights=source+args.weight_off+ulong(row)*row_bytes;\n"
    "    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f,a4=0.0f,a5=0.0f,a6=0.0f,a7=0.0f;\n"
    "    for(uint column=0u;column<args.cols;column++){device const uchar *block=weights+ulong(column/32u)*18u;\n"
    "        ushort encoded=load16(block);float scale=float(as_type<half>(encoded));uint packed=uint(block[2u+(column&15u)]);\n"
    "        int code=column<((column/32u)*32u+16u)?int(packed&15u):int(packed>>4u);float product=scale*float(code-8)*x[column];\n"
    "        switch(column&7u){case 0u:a0+=product;break;case 1u:a1+=product;break;case 2u:a2+=product;break;case 3u:a3+=product;break;\n"
    "        case 4u:a4+=product;break;case 5u:a5+=product;break;case 6u:a6+=product;break;default:a7+=product;break;}\n"
    "    }\n"
    "    y[row]=((a0+a2)+(a4+a6))+((a1+a3)+(a5+a7));\n"
    "}\n";

static id<MTLComputePipelineState> _bpso2;
static id<MTLComputePipelineState> _bweight_pso;
static id<MTLComputePipelineState> _bcoalesced_pso;
static id<MTLComputePipelineState> _mtp_q4_0_pso;
static id<MTLComputePipelineState> _bselected_pso;
static id<MTLComputePipelineState> _bselected_ragged_pso;
static id<MTLComputePipelineState> _bselected_rows4_pso;
static id<MTLComputePipelineState> _bselected_grouped_pso;
static id<MTLComputePipelineState> _bselected_warp_pso;
static id<MTLArgumentEncoder> _selected_arg_encoder;
static id<MTLBuffer> _selected_args, _selected_desc;
static int _selected_argument_capacity;
static id<MTLBuffer> _selected_task_map;
static uint64_t _q4_simd_trunk_dispatches;
static uint64_t _q4_simd_selected_dispatches;
static uint64_t _q4_non_simd_fallback_dispatches;
/* MEASURED: a per-SITE constant buffer is NOT a valid setBytes replacement.
 * setBytes copies the value INTO the recorded command; setBuffer references a
 * buffer whose contents can change before execution.  These sites are called
 * once per layer within one encoder, so every dispatch read the LAST value
 * written and the gate went semantic-RED (7f24bdfe... vs f52b5d47...).
 * The ICB path needs PER-DISPATCH argument slots - a scratch buffer with a
 * unique offset per dispatch - not one buffer per site.  Buffers retained,
 * unused, for that work. */
/* ICB argument model: setBytes COPIES into the recorded command, setBuffer
 * REFERENCES a buffer whose contents can change before execution.  These sites
 * run once per layer inside one encoder, so a per-SITE buffer made every
 * dispatch read the last value written (measured: semantic RED 7f24bdfe...).
 * The valid replacement is PER-DISPATCH slots: one scratch, offset per put. */
static id<MTLArgumentEncoder> _wave_selected_arg_encoder;
static id<MTLBuffer> _wave_selected_args, _selected_logical_slots;



/* THE LEDGER BOOK - weight-geometry sub-ledger (book 125).  One entry per
 * LOGICAL RESOURCE (logical = layer*n_experts + expert), holding the static
 * per-(layer,expert) storage geometry that the chain descriptor needs:
 *   [0..2] gate value/scale/bias offsets
 *   [3..5] up   value/scale/bias offsets
 *   [6..8] down value/scale/bias offsets
 * Built ONCE at initialization from the program's layer expert descriptors -
 * the same metadata and structure a host realization would hold, differing
 * only in the addresses.  Device-resident and indexed by logical id, exactly
 * like _selected_logical_slots, so a kernel can resolve geometry the same way
 * it already resolves slots (the dc.y branch). */
#define METAL_TEXT_MANIFEST_FIELDS 9u
/* The format's own bounds, fixed, so the manifest's storage can be allocated at init - before any
 * program exists and before any failure can make a record impossible. */
#define METAL_TEXT_MANIFEST_MAX_LAYERS 64u
#define METAL_TEXT_MANIFEST_MAX_EXPERTS 256u
static id<MTLBuffer> _text_manifest;
static uint64_t *_text_manifest_host;
/* MANIFEST PLAN HEADER: the phase batch shapes.  Prefill can leverage a HIGH B
 * (the whole chunk at once); decode is shape-limited (one authoritative row, plus
 * whatever the candidate frontier supplies).  Two values, not one - a single B
 * cannot serve both phases, which is why one of them was always mis-shaped.
 * Model owns the shape (its weight structure); the device bounds it (its physical
 * capability).  Both are resolved at INITIALIZATION into the book. */
#define METAL_TEXT_PLAN_PREFILL_B 0u
#define METAL_TEXT_PLAN_DECODE_B  1u
#define METAL_TEXT_PLAN_ROWS_MAX  2u
#define METAL_TEXT_PLAN_FIELDS    4u
static uint64_t _text_manifest_plan[METAL_TEXT_PLAN_FIELDS];
static int _text_manifest_capacity;
static uint32_t _text_manifest_rows, _text_manifest_experts;
#define METAL_MANIFEST_GPU_WINDOWS 32768u
enum { METAL_MANIFEST_GPU_TRUNK = 0u, METAL_MANIFEST_GPU_EXPERT = 1u };
typedef struct MetalManifestGpuWindow {
    uint32_t phase, layer, stage, experts;
    uint64_t rows, begin_ns, end_ns, commit_ns, wait_return_ns;
} MetalManifestGpuWindow;
static MetalManifestGpuWindow _manifest_gpu_windows[METAL_MANIFEST_GPU_WINDOWS];
static uint32_t _manifest_gpu_window_count, _manifest_gpu_window_dropped;
#define METAL_COMMAND_EVENT_PENDING 4u
typedef struct MetalCommandEventPending {
    id<MTLCommandBuffer> command;
    uint32_t phase, layer, stage, experts;
    uint64_t rows, commit_ns, wait_return_ns;
} MetalCommandEventPending;
static MetalCommandEventPending
    _command_event_pending[METAL_COMMAND_EVENT_PENDING];
static uint32_t _command_event_pending_head, _command_event_pending_count;
static int _command_event_error;
static uint32_t _text_grouped_prefill_encoded_layers;
static uint64_t _text_grouped_prefill_encoded_jobs;
static uint32_t _text_rows4_decode_encoded_layers;
static uint64_t _text_rows4_decode_dispatches;
/* LEDGER (117): per-logical resolved slot + validity generation, allocated AT
 * STARTUP from the same capacity as the selected-resource tables - the pool, not
 * a backend-private array.  Indexed by the canonical logical id, which already
 * encodes (layer, expert), so no per-layer array is needed. */
static id<MTLBuffer> _text_ledger_slots;
static int32_t *_text_ledger_slot_host;
static uint64_t *_text_ledger_gen_host;
static int32_t *_text_ledger_expert_host;
static int _text_ledger_logical_capacity;
/* The enable switch is part of the fixed configuration: resolved ONCE at
 * startup, never consulted per expert boundary. */
static int _text_ledger_skip_enabled;

/* PROBE LEDGER: per-commit observations recorded IN MEMORY ONLY.  Nothing is
 * emitted here - the dump rides an existing emission point so the measurement
 * cannot pollute the run.  Flag resolved ONCE at startup. */
#define METAL_TEXT_PROBE_MAX 8192
static struct {
    float wall, gpu, commit;
    float completion_tail;
    int completion_tail_valid;
    uint32_t rows, layer, phase;
    uint32_t operation_phase, operation_kind, phase_io_record;
    uint64_t read_bytes, write_bytes, major_faults;
    uint64_t io_begin_ns, io_end_ns, io_sample_ns;
    int read_bytes_valid, write_bytes_valid, major_faults_valid;
    uint32_t asynchronous;
    uint64_t async_submit_ns, async_observed_done_ns;
} _text_probe[METAL_TEXT_PROBE_MAX];
static uint64_t _decode_overlap_issued, _decode_overlap_completed, _decode_overlap_deferred_cells;
static int _text_probe_n;
static int _text_probe_enabled;
static double _mt_probe_ns(void) {
    struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t);
    return (double)_t.tv_sec * 1e9 + (double)_t.tv_nsec;
}
/* THE NODE LEDGER.  One record per boundary/function call on the operation path, so the
 * manifest carries WHERE the wall goes, not only how much of it there was.  Pure memory
 * writes in the hot path; the flag is resolved ONCE at initialization; emitted at the
 * designated dump point.  Each record holds the wall SINCE THE PREVIOUS boundary, which is
 * the work that boundary closed. */
#define MT_NODE_SLOTS 8192
typedef struct { const char *name; const char *caller; uint32_t layer; uint32_t kind; double ms; } MtNodeRec;
static MtNodeRec _mt_node[MT_NODE_SLOTS];
static int _mt_node_n;
static int _text_last_rot_layer = -1;
/* the last layer the walk told the rotation about - one call per layer, not per cell */
static int _mt_node_on;
static int _mt_node_enabled_resolved;
/* THE RESOLVED CONFIG SNAPSHOT.  The record must state what the ENGINE resolved at
 * initialization - the fixed input the operation runs on - not what the environment happens
 * to say when the dump runs.  Captured once, here, and read from here. */
#define MT_CFG_KEYS 24
static const char *_mt_cfg_key[MT_CFG_KEYS];
static char _mt_cfg_val[MT_CFG_KEYS][64];
static int _mt_cfg_n;
static void mt_cfg_snapshot(void) {
    static const char *keys[] = {
        "SALT_GEMMA_PLATFORM_RECIPE", "SALT_GEMMA_DECODE_MODE", "SALT_TEXT_FINE_TOKEN",
        "SALT_EXPERT_PREFETCH", "SALT_EXPERT_PRELOAD", "SALT_EXPERT_PRELOAD_LAYERS",
        "SALT_EXPERT_BUDGET_GB", "SALT_FETCH_TOUCH_BYTES", "SALT_PREFILL_B",
        "SALT_PREFILL_DECODE_LOOKAHEAD", "SALT_TEXT_LEDGER_SKIP", "SALT_TEXT_NODE_LEDGER",
        "SALT_METAL_Q4_WEIGHT_STATIONARY_MIN_B", "SALT_METAL_Q4_SIMDGROUPS",
        "SALT_TARGET_GPU_PROGRAM", "SALT_CACHE_ACCOUNTING", "SALT_GEMMA_MEMORY_LIMIT_GB",
        "SALT_TARGET_ROUTE_N", "SALT_TARGET_ROUTE_F", "SALT_TARGET_ROUTE_Q",
        "SALT_TARGET_X", "SALT_TARGET_AREA_WORKERS", "SALT_ATTN_THREADS",
        "SALT_GEMMA_COMPUTE_NODE",
    };
    unsigned i;
    _mt_cfg_n = 0;
    for (i = 0; i < sizeof keys / sizeof keys[0] && i < MT_CFG_KEYS; i++) {
        const char *v = getenv(keys[i]);
        if (!strcmp(keys[i], "SALT_GEMMA_DECODE_MODE") &&
                salt_lr_decode_mixed())
            v = "mixed";
        _mt_cfg_key[_mt_cfg_n] = keys[i];
        snprintf(_mt_cfg_val[_mt_cfg_n], sizeof _mt_cfg_val[0], "%s",
                 (v && v[0]) ? v : "(unset)");
        _mt_cfg_n++;
    }
}
static double _mt_node_last;
static void mt_node_tick(const char *name, const char *caller, uint32_t layer, uint32_t kind) {
    double now;
    if (!_mt_node_on) return;
    now = _mt_probe_ns();
    if (_mt_node_n < MT_NODE_SLOTS) {
        _mt_node[_mt_node_n].name = name;
        _mt_node[_mt_node_n].caller = caller;
        _mt_node[_mt_node_n].layer = layer;
        _mt_node[_mt_node_n].kind = kind;
        _mt_node[_mt_node_n].ms = (now - _mt_node_last) / 1e6;
        _mt_node_n++;
    }
    _mt_node_last = now;
}

/* THE TRACE'S EMIT: the records are in memory; this only formats them. */
static void _mt_trace_emit(const char *line, void *ctx) {
    FILE *f = (FILE *)ctx;
    if (!f) return;
    fputs(line, f);
    fputc('\n', f);
}
/* THE MODEL'S OWN CHECKPOINT: one call from anywhere in the inference, recorded in memory. */
void salt_gpu_text_note(const char *name, uint32_t a, uint32_t b) {
    if (!_mt_node_enabled_resolved) {
        _mt_node_enabled_resolved = 1;
        _mt_node_on = getenv("SALT_TEXT_NODE_LEDGER") != NULL;
        _mt_node_last = _mt_probe_ns();
        mt_cfg_snapshot();
    }
    mt_node_tick(name ? name : "note", "model", a, b);
}


/* The ledger is PER TOKEN: an entry is only usable by the token AFTER the one
 * that registered it.  Without this the generations accumulate across tokens and
 * the re-arm encodes the chain for every expert ever demanded - fewer commits,
 * more work, which is exactly the 36% regression measured at 119. */
static uint32_t _text_ledger_cur_epoch = 1u;
static uint32_t _text_ledger_prev_epoch = 0u;
static id<MTLBuffer> _selected_logical_payload_offsets;
static int32_t *_selected_logical_slot_host;

/* THE MANIFEST'S STORAGE, ALLOCATED AT INIT WITH EVERY OTHER FIXED RESOURCE.  It was built at program
 * prepare, which is after an initialization failure - so a run that failed at init had no record, and
 * every such failure in this work was invisible.  The geometry is fixed and known here: the format's own
 * maximum, which every program's layer/expert count fits inside. */
static void metal_text_manifest_init(void) {
    NSUInteger cap;
    if (_text_manifest || !_dev) return;
    cap = (NSUInteger)METAL_TEXT_MANIFEST_MAX_LAYERS * (NSUInteger)METAL_TEXT_MANIFEST_MAX_EXPERTS;
    _text_manifest = [_dev newBufferWithLength:
        cap * METAL_TEXT_MANIFEST_FIELDS * sizeof(uint64_t)
        options:MTLResourceStorageModeShared];
    if (_text_manifest && [_text_manifest contents]) {
        _text_manifest_host = (uint64_t *)[_text_manifest contents];
        _text_manifest_capacity = (int)cap;
    } else {
        metal_release(_text_manifest);
        _text_manifest = nil;
        _text_manifest_host = NULL;
        _text_manifest_capacity = 0;
    }
}
/* WHICH guard refused a bind: 1 args/capacity, 2 payload range, 3 geometry, 4 logical already bound */
static uint32_t _bind_refuse[5];
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
    return _cfg_address_audit;   /* fixed configuration, resolved once */
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

static int metal_selected_capacity_valid(int capacity, int logical_capacity,
                                         size_t max_buffer_length) {
    if (capacity < 1 || logical_capacity < 1 || logical_capacity > INT32_MAX ||
        (size_t)capacity > SIZE_MAX / sizeof(MetalSelectedCacheResource) ||
        (size_t)capacity > max_buffer_length / sizeof(uint64_t) ||
        (size_t)logical_capacity > NSUIntegerMax / sizeof(int32_t) ||
        (size_t)logical_capacity > max_buffer_length / sizeof(int32_t) ||
        (size_t)logical_capacity > max_buffer_length / sizeof(uint32_t))
        return 0;
    return 1;
}

/* Only the startup cache count may fix this layout. Generic batch setup must
 * not create it first, and replay must never replace its buffers/pipelines. */
static int metal_selected_resources_prepare(int capacity) {
    if (_selected_argument_capacity)
        return capacity == _selected_argument_capacity ? 0 : -1;
    if (!_dev || capacity < 1 ||
        [_dev argumentBuffersSupport] != MTLArgumentBuffersTier2)
        return SALT_TFAIL(-1);
    @autoreleasepool {
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        metal_exact_math(options);
        options.preprocessorMacros = @{
            @"SALT_METAL_SELECTED_CAPACITY": @(capacity),
        };
        /* An offline library has no authenticated capacity specialization. */
        id<MTLLibrary> library = [_dev newLibraryWithSource:
            [NSString stringWithUTF8String:_batch_src2]
            options:options error:&error];
        [options release];
        if (!library) return SALT_TFAIL(-1);
        id<MTLFunction> function =
            [library newFunctionWithName:@"q4selected"];
        id<MTLFunction> ragged_function =
            [library newFunctionWithName:@"q4selectedragged"];
        id<MTLFunction> rows4_function =
            [library newFunctionWithName:@"q4rotationrows4"];
        id<MTLFunction> grouped_function =
            [library newFunctionWithName:@"q4rotationgrouped"];
        id<MTLFunction> warp_function =
            [library newFunctionWithName:@"q4selectedwarp"];
        [library release];
        if (!function || !ragged_function || !rows4_function ||
            !grouped_function || !warp_function) {
            metal_release(function); metal_release(ragged_function);
            metal_release(rows4_function);
            metal_release(grouped_function); metal_release(warp_function);
            return SALT_TFAIL(-1);
        }
        id<MTLComputePipelineState> pipeline =
            [_dev newComputePipelineStateWithFunction:function error:&error];
        id<MTLComputePipelineState> ragged_pipeline =
            [_dev newComputePipelineStateWithFunction:ragged_function
                                               error:&error];
        id<MTLComputePipelineState> rows4_pipeline =
            [_dev newComputePipelineStateWithFunction:rows4_function
                                               error:&error];
        id<MTLComputePipelineState> grouped_pipeline =
            [_dev newComputePipelineStateWithFunction:grouped_function
                                               error:&error];
        id<MTLComputePipelineState> warp_pipeline =
            [_dev newComputePipelineStateWithFunction:warp_function
                                               error:&error];
        id<MTLArgumentEncoder> encoder =
            [function newArgumentEncoderWithBufferIndex:0];
        id<MTLArgumentEncoder> wave_encoder =
            [function newArgumentEncoderWithBufferIndex:0];
        [function release];
        [ragged_function release];
        [rows4_function release];
        [grouped_function release];
        [warp_function release];
        NSUInteger argument_bytes = [encoder encodedLength];
        if (!pipeline || !ragged_pipeline || !rows4_pipeline ||
            !grouped_pipeline || !warp_pipeline ||
            !encoder || !wave_encoder ||
            pipeline.threadExecutionWidth != 32 ||
            pipeline.maxTotalThreadsPerThreadgroup < 32 ||
            argument_bytes == 0 || argument_bytes > [_dev maxBufferLength] ||
            argument_bytes != [wave_encoder encodedLength]) {
            metal_release(pipeline); metal_release(ragged_pipeline);
            metal_release(rows4_pipeline);
            metal_release(grouped_pipeline); metal_release(warp_pipeline);
            metal_release(encoder); metal_release(wave_encoder);
            return SALT_TFAIL(-1);
        }
        id<MTLBuffer> arguments = [_dev newBufferWithLength:argument_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> wave_arguments = [_dev newBufferWithLength:argument_bytes
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> descriptors = [_dev newBufferWithLength:
            METAL_SELECTED_DESC_BYTES options:MTLResourceStorageModeShared];
        id<MTLBuffer> tasks = [_dev newBufferWithLength:
            (NSUInteger)METAL_SELECTED_TASK_CAPACITY *
                sizeof(MetalSelectedTask)
            options:MTLResourceStorageModeShared];
        if (!arguments || !wave_arguments || !descriptors || !tasks) {
            metal_release(pipeline);
            metal_release(ragged_pipeline);
            metal_release(rows4_pipeline);
            metal_release(grouped_pipeline);
            metal_release(warp_pipeline);
            metal_release(encoder);
            metal_release(wave_encoder);
            metal_release(arguments);
            metal_release(wave_arguments);
            metal_release(descriptors);
            metal_release(tasks);
            return SALT_TFAIL(-1);
        }
        _bselected_pso = pipeline;
        _bselected_ragged_pso = ragged_pipeline;
        _bselected_rows4_pso = rows4_pipeline;
        _bselected_grouped_pso = grouped_pipeline;
        _bselected_warp_pso = warp_pipeline;
        _selected_arg_encoder = encoder;
        _selected_args = arguments;
        _selected_desc = descriptors;
        _selected_task_map = tasks;
        _sel_args_scratch = [_dev newBufferWithLength:SEL_ARGS_SCRATCH_BYTES
                                              options:MTLResourceStorageModeShared];
        _sel_args_cursor = 0u;
        _wave_selected_arg_encoder = wave_encoder;
        _wave_selected_args = wave_arguments;
        [_selected_arg_encoder setArgumentBuffer:_selected_args offset:0];
        [_wave_selected_arg_encoder setArgumentBuffer:
            _wave_selected_args offset:0];
        _selected_argument_capacity = capacity;
    }
    return 0;
}

int salt_gpu_selected_resources_prepare(int capacity, int logical_capacity) {
    if (!_dev || !metal_selected_capacity_valid(capacity, logical_capacity,
            (size_t)[_dev maxBufferLength]) ||
        [_dev argumentBuffersSupport] != MTLArgumentBuffersTier2)
        return SALT_TFAIL(-1);
    if (_selected_cache_resources)
        return capacity == _selected_cache_capacity &&
            capacity == _selected_argument_capacity &&
            logical_capacity == _selected_logical_capacity ? 0 : -1;
    if (batch_pipeline_prepare() != 0) return SALT_TFAIL(-1);
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
        ![_selected_logical_payload_offsets contents] ||
        metal_selected_resources_prepare(capacity) != 0) {
        free(_selected_cache_resources);
        _selected_cache_resources = NULL;
        metal_release(_selected_logical_slots);
        _selected_logical_slots = nil;
        metal_release(_selected_logical_payload_offsets);
        _selected_logical_payload_offsets = nil;
        return SALT_TFAIL(-1);
    }
    _selected_logical_slot_host =
        (int32_t *)[_selected_logical_slots contents];
    _selected_logical_payload_host =
        (uint32_t *)[_selected_logical_payload_offsets contents];
    _text_ledger_slots = [_dev newBufferWithLength:
        (NSUInteger)logical_capacity * sizeof(int32_t)
        options:MTLResourceStorageModeShared];
    _text_ledger_gen_host = (uint64_t *)calloc(
        (size_t)logical_capacity, sizeof(uint64_t));
    _text_ledger_expert_host = (int32_t *)calloc(
        (size_t)logical_capacity, sizeof(int32_t));
    if (!_text_ledger_slots || ![_text_ledger_slots contents] ||
        !_text_ledger_gen_host || !_text_ledger_expert_host) {
        metal_release(_text_ledger_slots);
        _text_ledger_slots = nil;
        free(_text_ledger_gen_host);
        _text_ledger_gen_host = NULL;
        free(_text_ledger_expert_host);
        _text_ledger_expert_host = NULL;
        return SALT_TFAIL(-1);
    }
    _text_ledger_slot_host = (int32_t *)[_text_ledger_slots contents];
    for (int _l = 0; _l < logical_capacity; _l++)
        _text_ledger_expert_host[_l] = -1;
    _text_ledger_logical_capacity = logical_capacity;
    {
        const char *_v = getenv("SALT_TEXT_LEDGER_SKIP");
        _text_ledger_skip_enabled = _v && strcmp(_v, "1") == 0;
        const char *_p = getenv("SALT_GPU_PROBE_LEDGER");
        _text_probe_enabled = _p && strcmp(_p, "1") == 0;
        _text_probe_n = 0;
    }
    for (int logical = 0; logical < logical_capacity; logical++) {
        _selected_logical_slot_host[logical] = -1;
        _selected_logical_payload_host[logical] = 0;
        _text_ledger_slot_host[logical] = -1;
    }
    _selected_cache_capacity = capacity;
    _selected_logical_capacity = logical_capacity;
    _selected_retirement_fenced = 0;
    _rotation_done_event = [_dev newSharedEvent];
    _rotation_ready_event = [_dev newSharedEvent];
    _command_done_event = [_dev newSharedEvent];
    if (!_rotation_done_event || !_rotation_ready_event || !_command_done_event)
        return SALT_TFAIL(-1);
    _command_done_value = 0;
    _command_done_waited = 0;
    _command_done_harvested = 0;
    _command_done_forced = 0;
    _command_event_pending_head = 0;
    _command_event_pending_count = 0;
    _command_event_error = 0;
    _rotation_done_listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0)];
    if (!_rotation_done_listener)
        return SALT_TFAIL(-1);
    /* Borrow the engine allocation once. Argument entries are addresses
     * within that ONE resource, not per-expert MTLBuffer registrations. */
    {
        const SaltGpuSharedBuffer *storage = salt_rotation_expert_arena();
        const uint8_t *arena = NULL;
        uint64_t stride = 0, slot_bytes = 0;
        int slots = 0, ne;
        if (salt_rot_inference_arena(&arena, &stride, &slot_bytes, &slots) != 0 ||
            !storage->backend || storage->contents != arena ||
            slots < 2 || slots > capacity ||
            stride != slot_bytes || stride > SIZE_MAX / (size_t)slots ||
            storage->nbytes != (size_t)slots * stride ||
            [(id<MTLBuffer>)storage->backend contents] != arena)
            return SALT_TFAIL(-1);
        ne = (int)salt_lr_layer_experts();
        if (ne < 1 || slots < SALT_LR_BANK_COUNT * ne)
            return SALT_TFAIL(-1);
        _rotation_arena_slots = slots;
        _rotation_experts_per_bank = ne;
        _rotation_stride = (NSUInteger)stride;
        _rotation_bank_bytes = (NSUInteger)ne * stride;
        _rotation_arena_backing = [(id<MTLBuffer>)storage->backend retain];
        for (NSUInteger bank = 0; bank < SALT_LR_BANK_COUNT; bank++) {
            void *base = (uint8_t *)storage->contents + bank * _rotation_bank_bytes;
            _rotation_bank_view[bank] = [_dev newBufferWithBytesNoCopy:base
                length:_rotation_bank_bytes options:MTLResourceStorageModeShared |
                    MTLResourceHazardTrackingModeUntracked deallocator:nil];
            if (!_rotation_bank_view[bank] || [_rotation_bank_view[bank] contents] != base)
                return SALT_TFAIL(-1);
        }
        /* The argument table names physical BANK resources, not expert seats.
         * The immutable descriptor carries the expert's byte offset within its
         * bank. This keeps Metal argument count bounded by bank count while the
         * engine ledger retains the complete logical expert -> physical seat map. */
        for (int bank = 0; bank < SALT_LR_BANK_COUNT; bank++)
            [_wave_selected_arg_encoder setBuffer:_rotation_bank_view[bank]
                offset:0 atIndex:(NSUInteger)bank];
        for (int logical = 0; logical < logical_capacity; logical++) {
            int slot = salt_rot_inference_slot(logical / ne, logical % ne);
            if (slot < 0 || slot >= slots) return SALT_TFAIL(-1);
            _selected_logical_slot_host[logical] = slot;
            _text_ledger_slot_host[logical] = slot;
        }
        salt_trace_register(254, 1);
        salt_trace_value(254, (int)storage->nbytes);
        salt_trace_register(255, 1);
        salt_trace_value(255, 1);
        salt_trace_register(251, 1);
        salt_trace_value(251, 0); /* no per-expert resource objects */
    }
    return 0;
}

uint32_t salt_gpu_selected_bind_refuse(int which) {
    return (which >= 1 && which <= 4) ? _bind_refuse[which] : 0u;
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
        !_selected_logical_slot_host || !_selected_logical_payload_host) {
        _bind_refuse[1]++;
        return SALT_TFAIL(-1);
    }
    base_value = (uintptr_t)base;
    payload_value = (uintptr_t)payload;
    if (payload_value < base_value || payload_value - base_value > nbytes ||
        payload_value - base_value > UINT32_MAX) {
        _bind_refuse[2]++;
        return SALT_TFAIL(-1);
    }
    MetalSelectedCacheResource *resource = &_selected_cache_resources[slot];
    /* A SLOT IS AN ADDRESS, NOT AN OWNER.  In the rotation one slot holds layer i's expert e and, one
     * step later, layer i+2's expert e - the same address, a different logical id.  The geometry is what
     * must match; the logical id is what the logical->slot table is for, and it is indexed by logical,
     * not by slot. */
    if (resource->map) {
        if (resource->base == base && resource->nbytes == nbytes)
            return 0;
        /* A SLOT IS AN INDEX, THE ADDRESS BELONGS TO THE EXPERT.  On the zero-copy mode one slot holds
         * layer i's expert e at one address and layer i+2's expert e at another, so the resource has to
         * be re-bound to the new address rather than refused.  Release the old view and fall through. */
        metal_release(resource->map);
        resource->map = nil;
        resource->base = NULL;
        resource->nbytes = 0;
        resource->logical_resource_id = 0;
    }
    if (!salt_lr_decode_zero_copy_active() &&
        _selected_logical_slot_host[logical_resource_id] != -1) {
        _bind_refuse[4]++;
        return SALT_TFAIL(-1);
    }
    @autoreleasepool {
        map = [_dev newBufferWithBytesNoCopy:(void *)base length:nbytes
            options:MTLResourceStorageModeShared deallocator:nil];
        if (!map)
            map = [_dev newBufferWithBytesNoCopy:(void *)base length:nbytes
                options:MTLResourceStorageModeManaged deallocator:nil];
        if (!map || [map contents] != base) {
            metal_release(map);
            return SALT_TFAIL(-1);
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
    return SALT_TFAIL(-1);
}

int salt_gpu_selected_resources_fence(void) {
    if (salt_gpu_sync() != 0) return SALT_TFAIL(-1);
    _selected_retirement_fenced = 1;
    return 0;
}

int salt_gpu_selected_resource_unbind(int slot,
                                      uint64_t logical_resource_id) {
    int restore_rotation = salt_lr_decode_zero_copy_active();
    if (!_selected_cache_resources || slot < 0 ||
        slot >= _selected_cache_capacity ||
        logical_resource_id >= (uint64_t)(uint32_t)_selected_logical_capacity ||
        !_selected_logical_slot_host || !_selected_logical_payload_host)
        return SALT_TFAIL(-1);
    MetalSelectedCacheResource *resource = &_selected_cache_resources[slot];
    if (!resource->map) return 0;
    if (resource->logical_resource_id != logical_resource_id ||
        _selected_logical_slot_host[logical_resource_id] != slot ||
        (!_selected_retirement_fenced && salt_gpu_sync() != 0))
        return SALT_TFAIL(-1);
    if (restore_rotation) {
        int layer, expert, restored;
        if (_rotation_experts_per_bank < 1) return SALT_TFAIL(-1);
        layer = (int)(logical_resource_id /
            (uint64_t)(uint32_t)_rotation_experts_per_bank);
        expert = (int)(logical_resource_id %
            (uint64_t)(uint32_t)_rotation_experts_per_bank);
        restored = salt_lr_slot(layer, expert);
        if (_rotation_experts_per_bank < 1 || restored < 0 ||
            restored >= _rotation_arena_slots || !_rotation_stride)
            return SALT_TFAIL(-1);
        _selected_logical_slot_host[logical_resource_id] = restored;
        _selected_logical_payload_host[logical_resource_id] = 0;
        if (slot < _rotation_arena_slots) {
            int bank = slot / _rotation_experts_per_bank;
            if (bank < 0 || bank >= SALT_LR_BANK_COUNT ||
                !_rotation_bank_view[bank])
                return SALT_TFAIL(-1);
            [_wave_selected_arg_encoder setBuffer:_rotation_bank_view[bank]
                offset:(NSUInteger)(slot % _rotation_experts_per_bank) *
                    _rotation_stride atIndex:(NSUInteger)slot];
        } else {
            [_wave_selected_arg_encoder setBuffer:nil offset:0
                atIndex:(NSUInteger)slot];
        }
    } else {
        _selected_logical_slot_host[logical_resource_id] = -1;
        _selected_logical_payload_host[logical_resource_id] = 0;
        [_wave_selected_arg_encoder setBuffer:nil offset:0
            atIndex:(NSUInteger)slot];
    }
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
        if (!_bx2[i] || !_by2[i]) return SALT_TFAIL(-1);
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
    if (!_desc || !_q8_x || !_q8_y)
        return SALT_TFAIL(-1);
    _desc_len = METAL_SESSION_DESC_BYTES;
    return 0;
}

static int mapped_range_offset(const uint8_t *base, size_t map_len,
                               const void *ptr, size_t nbytes,
                               uint64_t *offset) {
    uintptr_t b, p;
    uint64_t delta;
    if (!base || !ptr || !offset) return SALT_TFAIL(-1);
    b = (uintptr_t)(const void *)base;
    p = (uintptr_t)ptr;
    if (p < b) return SALT_TFAIL(-1);
    delta = (uint64_t)(p - b);
    if (delta > map_len || nbytes > map_len - (size_t)delta) return SALT_TFAIL(-1);
    *offset = delta;
    return 0;
}

static int batch_weight_sizes(int R, int C, size_t *vbytes,
                              size_t *sbytes) {
    uint64_t elements, packed_words, groups;
    if (R < 1 || C < 1 || !vbytes || !sbytes) return SALT_TFAIL(-1);
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    packed_words = (elements + 7u) / 8u;
    groups = (elements + 63u) / 64u;
    if (packed_words > SIZE_MAX / sizeof(uint32_t) ||
        groups > SIZE_MAX / sizeof(uint16_t)) return SALT_TFAIL(-1);
    *vbytes = (size_t)packed_words * sizeof(uint32_t);
    *sbytes = (size_t)groups * sizeof(uint16_t);
    return 0;
}

static int tensor_slot_store(const void *key, uint32_t kind,
                             uint32_t resource_id, uint64_t voff,
                             uint64_t soff, uint64_t boff) {
    TensorSlot *slot;
    if (!key || _tslot_n >= BATCH_MAXSLOT) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    vtotal = _arena_v_used + vneed;
    stotal = _arena_s_used + sneed;
    return SALT_TFAIL(-1);   /* the backend owns no storage: nothing is created here */
    /* grow on demand: new buffer, copy, swap (Metal buffers are
     * immutable-sized; reallocate like realloc). */
    if (vtotal > _arena_v.length) {
        size_t nlen = _arena_v.length;
        while (nlen < vtotal) {
            if (nlen > SIZE_MAX / 2) return SALT_TFAIL(-1);
            nlen *= 2;
        }
        id<MTLBuffer> nv = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        if (!nv) return SALT_TFAIL(-1);
        memcpy(nv.contents, _arena_v.contents, _arena_v_used);
        metal_release(_arena_v);
        _arena_v = nv;
    }
    if (stotal > _arena_s.length) {
        size_t nlen = _arena_s.length;
        while (nlen < stotal) {
            if (nlen > SIZE_MAX / 2) return SALT_TFAIL(-1);
            nlen *= 2;
        }
        id<MTLBuffer> ns = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> nb = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        if (!ns || !nb) {
            metal_release(ns); metal_release(nb);
            return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
        }
        return i;
    }
    if (_tslot_n >= BATCH_MAXSLOT) return SALT_TFAIL(-1);
    if (batch_weight_sizes(R, C, &vbytes, &sbytes) != 0) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    }
    /* NO STAGING.  The backend copied values, scales and biases into three buffers it owned and grew
     * them at runtime.  Weights are read where they already are - the mapped source - so a miss here is a
     * failure, not a copy into backend-owned storage. */
    _batch_stats.mapped_only_misses++;
    return SALT_TFAIL(-1);
    int stored = tensor_slot_store(id, SALT_GPU_RESOURCE_ARENA, 0,
                                   _arena_v_used, _arena_s_used,
                                   _arena_s_used);
    if (stored < 0) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        if (!_trunk_map) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        if (!map) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    if (salt_gpu_sync() != 0) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    if (_tslot_n >= BATCH_MAXSLOT) return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
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
    return SALT_TFAIL(-1);
}

int salt_gpu_resident_slot(const void *key, const uint32_t *vals,
                           const uint16_t *scales, const uint16_t *biases,
                           int R, int C) {
    if (_res_map && _res_base)
        return salt_gpu_expert_resource_slot(_res_id, key, vals, scales,
                                             biases, R, C);
    return SALT_TFAIL(-1);
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
        slots[0] >= _tslot_n) return SALT_TFAIL(-1);
    first = &_tslot[slots[0]].resource;
    if (!salt_gpu_resource_ref_valid(first)) return SALT_TFAIL(-1);
    for (int j = 1; j < nslots; j++) {
        if (slots[j] < 0 || slots[j] >= _tslot_n ||
            !salt_gpu_resource_same(first, &_tslot[slots[j]].resource))
            return SALT_TFAIL(-1);
    }
    *resource = first;
    return 0;
}

static int batch_build_desc(BatchDesc4 *desc, const int *slots,
                            const int *rows, int C, int njobs,
                            int shared_x, uint64_t resource_base) {
    uint64_t row_prefix = 0;
    if (!desc || !slots || !rows || C < 1 || njobs < 1) return SALT_TFAIL(-1);
    for (int j = 0; j < njobs; j++) {
        const SaltGpuResourceRef *ref;
        uint64_t xoff;
        uint64_t voff, soff, boff;
        if (slots[j] < 0 || slots[j] >= _tslot_n || rows[j] < 1)
            return SALT_TFAIL(-1);
        ref = &_tslot[slots[j]].resource;
        xoff = shared_x ? 0 : (uint64_t)(uint32_t)j * (uint32_t)C;
        if ((uint64_t)(uint32_t)rows[j] > UINT64_MAX - row_prefix) {
            if (_cfg_diag)
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
            return SALT_TFAIL(-1);
        }
        if (ref->value_offset < resource_base ||
            ref->scale_offset < resource_base ||
            ref->bias_offset < resource_base) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    ref = &_tslot[slot].resource;
    if (ref->value_offset < value_base || ref->scale_offset < scale_base ||
        ref->bias_offset < bias_base)
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    if (_tiled_max_descriptors == 0 || njobs < 2 || (C & 31) != 0) return 1;
    for (int job = 0; job < njobs;) {
        const SaltGpuResourceRef *ref;
        int run = 1;
        if (slots[job] < 0 || slots[job] >= _tslot_n || rows[job] < 1)
            return SALT_TFAIL(-1);
        ref = &_tslot[slots[job]].resource;
        while (job + run < njobs && slots[job + run] == slots[job] &&
               rows[job + run] == rows[job]) run++;
        if (descriptor_count >= BATCH_MAXSLOT ||
            ref->value_offset < resource_base ||
            ref->scale_offset < resource_base ||
            ref->bias_offset < resource_base ||
            (uint64_t)(uint32_t)run * (uint32_t)rows[job] >
                UINT64_MAX - output_prefix)
            return SALT_TFAIL(-1);
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
        !view_out || !base_offset_out) return SALT_TFAIL(-1);
    *view_out = nil;
    *base_offset_out = 0;
    generic = metal_weight_resource(resource->kind, resource->resource_id);
    if (!generic) return 0;
    if (!generic->active) return SALT_TFAIL(-1);
    for (int j = 0; j < nslots; j++) {
        size_t vbytes, sbytes;
        const SaltGpuResourceRef *ref;
        uint64_t ends[3];
        if (slots[j] < 0 || slots[j] >= _tslot_n || rows[j] < 1 ||
            batch_weight_sizes(rows[j], C, &vbytes, &sbytes) != 0)
            return SALT_TFAIL(-1);
        ref = &_tslot[slots[j]].resource;
        if (!salt_gpu_resource_same(resource, ref) ||
            ref->value_offset > UINT64_MAX - vbytes ||
            ref->scale_offset > UINT64_MAX - sbytes ||
            ref->bias_offset > UINT64_MAX - sbytes)
            return SALT_TFAIL(-1);
        if (ref->value_offset < lo) lo = ref->value_offset;
        if (ref->scale_offset < lo) lo = ref->scale_offset;
        if (ref->bias_offset < lo) lo = ref->bias_offset;
        ends[0] = ref->value_offset + vbytes;
        ends[1] = ref->scale_offset + sbytes;
        ends[2] = ref->bias_offset + sbytes;
        for (int i = 0; i < 3; i++) if (ends[i] > hi) hi = ends[i];
    }
    if (lo == UINT64_MAX || hi <= lo || hi > generic->nbytes)
        return SALT_TFAIL(-1);
    if (generic->map) {
        if (lo < generic->map_offset ||
            generic->map_nbytes > UINT64_MAX - generic->map_offset ||
            hi > generic->map_offset + generic->map_nbytes)
            return SALT_TFAIL(-1);
        *view_out = [generic->map retain];
        *base_offset_out = generic->map_offset;
        return 0;
    }
    if ((page_l = sysconf(_SC_PAGESIZE)) <= 0) return SALT_TFAIL(-1);
    uint64_t page = (uint64_t)page_l;
    uint64_t aligned = lo - lo % page;
    uint64_t length = hi - aligned;
    if (length > (uint64_t)NSUIntegerMax) return SALT_TFAIL(-1);
    id<MTLBuffer> view = [_dev newBufferWithBytesNoCopy:
        (void *)(generic->base + aligned) length:(NSUInteger)length
        options:MTLResourceStorageModeShared deallocator:nil];
    if (!view)
        view = [_dev newBufferWithBytesNoCopy:
            (void *)(generic->base + aligned) length:(NSUInteger)length
            options:MTLResourceStorageModeManaged deallocator:nil];
    if (!view || [view contents] != generic->base + aligned) {
        metal_release(view);
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    uint64_t page = (uint64_t)page_l;
    uint64_t aligned = offset - offset % page;
    uint64_t length = offset - aligned + nbytes;
    if (length > (uint64_t)NSUIntegerMax) return SALT_TFAIL(-1);
    id<MTLBuffer> view = [_dev newBufferWithBytesNoCopy:
        (void *)(resource->base + aligned) length:(NSUInteger)length
        options:MTLResourceStorageModeShared deallocator:nil];
    if (!view)
        view = [_dev newBufferWithBytesNoCopy:
            (void *)(resource->base + aligned) length:(NSUInteger)length
            options:MTLResourceStorageModeManaged deallocator:nil];
    if (!view || [view contents] != resource->base + aligned) {
        metal_release(view);
        return SALT_TFAIL(-1);
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
    if (!enc || !salt_gpu_resource_ref_valid(resource)) return SALT_TFAIL(-1);
    generic = metal_weight_resource(resource->kind, resource->resource_id);
    if (generic && !generic->active) return SALT_TFAIL(-1);
    if (operation_view) {
        values = scales = biases = operation_view;
    } else if (generic) {
        values = scales = biases = generic->map;
    } else {
    switch (resource->kind) {
    case SALT_GPU_RESOURCE_ARENA:
        if (resource->resource_id != 0) return SALT_TFAIL(-1);
        values = _arena_v; scales = _arena_s; biases = _arena_b;
        break;
    case SALT_GPU_RESOURCE_TRUNK:
        if (resource->resource_id != 0) return SALT_TFAIL(-1);
        values = scales = biases = _trunk_map;
        break;
    case SALT_GPU_RESOURCE_EXPERT_LAYER:
        if (resource->resource_id != _res_id) return SALT_TFAIL(-1);
        values = scales = biases = _res_map;
        break;
    default:
        return SALT_TFAIL(-1);
    }
    }
    if (!values || !scales || !biases) return SALT_TFAIL(-1);
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
    if (!value_out) return SALT_TFAIL(-1);
    if (!text || !*text) {
        *value_out = fallback;
        return 0;
    }
    value = strtoul(text, &end, 10);
    if (!end || *end || value < minimum || value > maximum) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    _tiled_token_span = _tiled_simdgroups * 16u;
    _tiled_configured = 1;
    return 0;
}

static int batch_pipeline_prepare(void) {
    if (metal_tiled_configure() != 0) return SALT_TFAIL(-1);
    if (_bpso2 && _bweight_pso && _bcoalesced_pso && _btiled_pso &&
        _mtp_q4_0_pso && _bqueue2)
        return 0;
    metal_release(_bpso2); _bpso2 = nil;
    metal_release(_bweight_pso); _bweight_pso = nil;
    metal_release(_bcoalesced_pso); _bcoalesced_pso = nil;
    metal_release(_btiled_pso); _btiled_pso = nil;
    metal_release(_mtp_q4_0_pso); _mtp_q4_0_pso = nil;
    metal_release(_bqueue2); _bqueue2 = nil;
    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> lib = metal_library_for_source_mode(
            _batch_src2, &err, 0, 0);
        if (!lib) {
            fprintf(stderr, "gpu-metal: batch library failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown");
            return SALT_TFAIL(-1);
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q4batch2"];
        id<MTLFunction> weight_fn =
            [lib newFunctionWithName:@"q4weightstationary"];
        id<MTLFunction> coalesced_fn =
            [lib newFunctionWithName:@"q4coalesced"];
        id<MTLFunction> tiled_fn = [lib newFunctionWithName:@"q4warpbatch"];
        id<MTLFunction> mtp_fn = [lib newFunctionWithName:@"mtp_q4_0"];
        if (!fn || !weight_fn || !coalesced_fn || !tiled_fn || !mtp_fn) {
            fprintf(stderr,
                "gpu-metal: batch function lookup failed base=%d weight=%d "
                "coalesced=%d tiled=%d mtp=%d names=%s\n",
                fn != nil, weight_fn != nil, coalesced_fn != nil,
                tiled_fn != nil, mtp_fn != nil,
                [[[lib functionNames] componentsJoinedByString:@","] UTF8String]);
            [lib release];
            metal_release(fn); metal_release(weight_fn);
            metal_release(coalesced_fn);
            metal_release(tiled_fn);
            metal_release(mtp_fn);
            return SALT_TFAIL(-1);
        }
        [lib release];
        id<MTLComputePipelineState> pso =
            [_dev newComputePipelineStateWithFunction:fn error:&err];
        id<MTLComputePipelineState> weight_pso =
            [_dev newComputePipelineStateWithFunction:weight_fn error:&err];
        id<MTLComputePipelineState> coalesced_pso =
            [_dev newComputePipelineStateWithFunction:coalesced_fn error:&err];
        id<MTLComputePipelineState> tiled_pso =
            [_dev newComputePipelineStateWithFunction:tiled_fn error:&err];
        id<MTLComputePipelineState> mtp_pso =
            [_dev newComputePipelineStateWithFunction:mtp_fn error:&err];
        [fn release];
        [weight_fn release];
        [coalesced_fn release];
        [tiled_fn release];
        [mtp_fn release];
        id<MTLCommandQueue> queue = [_dev newCommandQueue];
        if (!pso || !weight_pso || !coalesced_pso || !tiled_pso || !mtp_pso || !queue ||
            pso.threadExecutionWidth != 32 ||
            tiled_pso.threadExecutionWidth != 32 ||
            (_tiled_simdgroups > 0 &&
             tiled_pso.maxTotalThreadsPerThreadgroup <
                (NSUInteger)_tiled_simdgroups * 32u)) {
            fprintf(stderr, "gpu-metal: batch pipeline failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown");
            metal_release(pso);
            metal_release(weight_pso);
            metal_release(coalesced_pso);
            metal_release(tiled_pso);
            metal_release(mtp_pso);
            metal_release(queue);
            return SALT_TFAIL(-1);
        }
        _bpso2 = pso;
        _bweight_pso = weight_pso;
        _bcoalesced_pso = coalesced_pso;
        _btiled_pso = tiled_pso;
        _mtp_q4_0_pso = mtp_pso;
        _bqueue2 = queue;
    }
    return 0;
}

typedef struct MetalMtpQ4Args {
    uint64_t weight_off, input_off, output_off;
    uint32_t rows, cols;
} MetalMtpQ4Args;

int salt_gpu_mtp_q4_0_prepare(void) {
    return _dev && batch_pipeline_prepare() == 0 ? 0 : SALT_TFAIL(-1);
}

int salt_gpu_mtp_q4_0_matvec(
        const SaltGpuSharedBuffer *source, size_t weight_byte_offset,
        SaltGpuSharedBuffer *scratch, size_t input_float_offset,
        size_t output_float_offset, size_t argument_byte_offset,
        int rows, int columns) {
    id<MTLBuffer> source_buffer, scratch_buffer;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    MetalMtpQ4Args args;
    size_t row_bytes, weight_bytes, input_bytes, output_bytes;
    NSUInteger threads;
    if (!_dev || !source || !source->contents || !source->backend ||
        !scratch || !scratch->contents || !scratch->backend ||
        rows < 1 || columns < 1 || (columns & 31) != 0 ||
        (size_t)rows > SIZE_MAX / ((size_t)columns / 32u * 18u) ||
        input_float_offset > SIZE_MAX / sizeof(float) ||
        output_float_offset > SIZE_MAX / sizeof(float) ||
        (size_t)columns > SIZE_MAX / sizeof(float) ||
        (size_t)rows > SIZE_MAX / sizeof(float) ||
        batch_pipeline_prepare() != 0)
        return SALT_TFAIL(-1);
    row_bytes = (size_t)columns / 32u * 18u;
    weight_bytes = (size_t)rows * row_bytes;
    input_bytes = (size_t)columns * sizeof(float);
    output_bytes = (size_t)rows * sizeof(float);
    if (weight_byte_offset > source->nbytes ||
        weight_bytes > source->nbytes - weight_byte_offset ||
        input_float_offset * sizeof(float) > scratch->nbytes ||
        input_bytes > scratch->nbytes - input_float_offset * sizeof(float) ||
        output_float_offset * sizeof(float) > scratch->nbytes ||
        output_bytes > scratch->nbytes - output_float_offset * sizeof(float) ||
        argument_byte_offset > scratch->nbytes ||
        sizeof args > scratch->nbytes - argument_byte_offset ||
        (argument_byte_offset & 7u) != 0)
        return SALT_TFAIL(-1);
    source_buffer = (id<MTLBuffer>)source->backend;
    scratch_buffer = (id<MTLBuffer>)scratch->backend;
    if ([source_buffer contents] != source->contents ||
        [scratch_buffer contents] != scratch->contents ||
        (size_t)[source_buffer length] < source->nbytes ||
        (size_t)[scratch_buffer length] < scratch->nbytes)
        return SALT_TFAIL(-1);
    args = (MetalMtpQ4Args) {
        weight_byte_offset,
        input_float_offset * sizeof(float),
        output_float_offset * sizeof(float),
        (uint32_t)rows, (uint32_t)columns,
    };
    memcpy((unsigned char *)scratch->contents + argument_byte_offset,
           &args, sizeof args);
    command = [_bqueue2 commandBuffer];
    if (!command) return SALT_TFAIL(-1);
    encoder = [command computeCommandEncoder];
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_mtp_q4_0_pso];
    [encoder setBuffer:source_buffer offset:0 atIndex:0];
    [encoder setBuffer:scratch_buffer offset:0 atIndex:1];
    [encoder setBuffer:scratch_buffer offset:argument_byte_offset atIndex:2];
    threads = _mtp_q4_0_pso.maxTotalThreadsPerThreadgroup < 64u
        ? _mtp_q4_0_pso.maxTotalThreadsPerThreadgroup : 64u;
    if (threads < 1u) {
        [encoder endEncoding];
        return SALT_TFAIL(-1);
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return metal_command_succeeded(command) == 0 ? 0 : SALT_TFAIL(-1);
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
    if (!stats || stats_size != sizeof *stats) return SALT_TFAIL(-1);
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
    const char *dispatch = _cfg_dispatch;

    if (!_dev || !vals || !scales || !biases || !xs || !output ||
        !output->contents ||
        !output->backend || !key || R < 1 || C < 1 || B < 2 ||
        first_job < 0 || job_count < 1 || first_job > B - job_count ||
        job_count > BATCH_MAXSLOT ||
        (dispatch && (!strcmp(dispatch, "icb") ||
                      !strcmp(dispatch, "metal-icb"))))
        return SALT_TFAIL(-1);
    total_rows = (uint64_t)(uint32_t)R * (uint32_t)job_count;
    output_span = (uint64_t)(uint32_t)B * (uint32_t)R;
    if (total_rows > UINT32_MAX || output_span > UINT32_MAX ||
        (uint64_t)y_float_offset > (uint64_t)UINT32_MAX - output_span ||
        output_span > (uint64_t)SIZE_MAX ||
        y_float_offset > SIZE_MAX - (size_t)output_span ||
        _direct_pending_n >= DIRECT_PENDING_MAX ||
        batch_pipeline_prepare() != 0)
        return SALT_TFAIL(-1);
    output_end = y_float_offset + (size_t)output_span;
    if (output_end > output->nbytes / sizeof(float)) return SALT_TFAIL(-1);
    ybuffer = (id<MTLBuffer>)output->backend;
    if ([ybuffer contents] != output->contents ||
        (size_t)[ybuffer length] < output->nbytes)
        return SALT_TFAIL(-1);
    /* The staged path owns one pending descriptor. Complete it before a
     * direct-output submission; direct submissions never force each other. */
    if ((_pend_cb || _direct_pending_n > 0) && salt_gpu_sync() != 0) return SALT_TFAIL(-1);


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
            {
        NSUInteger _off = sel_args_put(&rv, sizeof rv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
            {
        NSUInteger _off = sel_args_put(&cv, sizeof cv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
            {
        NSUInteger _off = sel_args_put(&nj, sizeof nj);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:8];
    }
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
    int detail = _cfg_ms_detail;
    struct timespec detail_start, detail_sync, detail_view, detail_desc;
    struct timespec detail_commit, detail_wait, detail_scatter;
    double detail_gpu_ms = 0.0;
    static unsigned long long detail_sequence = 0;
    unsigned long long detail_id = 0;
    unsigned long long detail_view_bytes = 0;
    int detail_resource_kind = -1;
    if (_cfg_diag)
        fprintf(stderr, "[gpu-dbg] batch entry njobs=%d C=%d Rj0=%d\n",
                njobs, C, Rj ? Rj[0] : -1);
    if (!_dev || !vals || !scales || !xs || !ys || !ids || !Rj ||
        njobs < 1 || njobs > BATCH_MAXSLOT || C < 1) return SALT_TFAIL(-1);
    if (detail) {
        detail_id = ++detail_sequence;
        clock_gettime(CLOCK_MONOTONIC, &detail_start);
    }
    const char *dispatch = _cfg_dispatch;
    if (dispatch && (!strcmp(dispatch, "icb") ||
                     !strcmp(dispatch, "metal-icb"))) {
        if (_cfg_diag)
            fprintf(stderr, "[gpu-fail] metal-icb unsupported\n");
        return SALT_TFAIL(-1);
    }
    /* There is exactly one pending result descriptor. Never overwrite it:
     * complete and scatter prior work before accepting another batch. */
    if (_pend_cb && salt_gpu_sync() != 0) return SALT_TFAIL(-1);
    if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_sync);
    for (int j = 0; j < njobs; j++) {
        if (Rj[j] < 1 || row_total > UINT32_MAX - (uint32_t)Rj[j])
            return SALT_TFAIL(-1);
        row_total += (uint32_t)Rj[j];
    }
    Rtot = (uint32_t)row_total;
    if (_cfg_diag)
        fprintf(stderr, "[gpu-dbg] Rtot=%u (last Rj=%d)\n", Rtot,
                njobs > 0 ? Rj[njobs - 1] : -1);
    if (_cfg_ms)
        fprintf(stderr, "[gpu-ms] ENTER njobs=%d Rtot=%u C=%d\n",
                njobs, Rtot, C);
    if (Rtot < 1) {
        if (_cfg_diag)
            fprintf(stderr, "[gpu-fail] guards _dev=%d njobs=%d Rtot=%u "
                    "C=%d\n", _dev != nil, njobs, Rtot, C);
        return SALT_TFAIL(-1);
    }
    if (_cfg_diag)
        fprintf(stderr, "[gpu-dbg] guards ok\n");
    if (batch_pipeline_prepare() != 0) {
        if (_cfg_diag)
            fprintf(stderr, "[gpu-fail] batch pipeline\n");
        return SALT_TFAIL(-1);
    }
    /* cache all tensors by stable id (first sight copies into the arena) */
    int slot[BATCH_MAXSLOT];
    const SaltGpuResourceRef *resource = NULL;
    if ((size_t)C > SIZE_MAX / (size_t)njobs / sizeof(float)) return SALT_TFAIL(-1);
    size_t xbytes = (size_t)C * (size_t)njobs * sizeof(float);
    size_t ybytes = (size_t)Rtot * sizeof(float);
    if (_cfg_diag)
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
            if (_cfg_diag && j < 2)
                fprintf(stderr, "[gpu-dbg] slot j=%d = %d (Rj=%d)\n",
                        j, slot[j], Rj[j]);
            if (slot[j] < 0) {
                if (_cfg_diag)
                    fprintf(stderr, "[gpu-fail] batch_slot<0 j=%d "
                            "tslot_n=%d njobs=%d Rj=%d C=%d id=%p "
                            "vals=%p res_base=%p res_len=%zu\n",
                            j, _tslot_n, njobs, Rj[j], C, ids[j],
                            (const void *)vals[j],
                            (const void *)_res_base, _res_len);
                return SALT_TFAIL(-1);
            }
        }
        if (batch_resource_for_slots(slot, njobs, &resource) != 0) {
            if (_cfg_diag)
                fprintf(stderr, "[gpu-fail] mixed or stale weight resources\n");
            return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
        if (operation_view) {
            if (detail) detail_view_bytes = (unsigned long long)[operation_view length];
            [operation_view autorelease];
            if (_defer) return SALT_TFAIL(-1);
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
            if (_cfg_diag)
                fprintf(stderr,
                    "[gpu-fail] session buffer capacity x=%zu/%zu "
                    "y=%zu/%zu desc=%zu/%zu\n",
                    xbytes, _bx2_len[bi], ybytes, _by2_len[bi],
                    (size_t)njobs * 2u * sizeof(BatchDesc4), _desc_len);
            return SALT_TFAIL(-1);
        }
        BatchDesc4 *desc = (BatchDesc4 *)_desc.contents;
        int descriptor_count = 0, max_rows = 0, max_batch = 0;
        int tiled_rc = pooled_components ? 1 : batch_build_tiled_desc(
            desc, slot, Rj, C, njobs, operation_base,
            &descriptor_count, &max_rows, &max_batch);
        if (tiled_rc < 0) return SALT_TFAIL(-1);
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
        if (_cfg_diag)
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
                    return SALT_TFAIL(-1);
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
            if (_cfg_diag)
                fprintf(stderr, "[gpu-fail] descriptor offset overflow\n");
            return SALT_TFAIL(-1);
        }
        if (_cfg_tiled)
            fprintf(stderr,
                "[gpu-tiled] active=%d logical_jobs=%d descriptors=%d "
                "max_batch=%d max_rows=%d C=%d\n",
                use_tiled, njobs, use_tiled ? descriptor_count : njobs,
                use_tiled ? max_batch : 1, use_tiled ? max_rows : 0, C);
        if (use_tiled && _cfg_diag) {
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
        if (_cfg_diag && Rtot == 512) {
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
            if (_cfg_diag)
                fprintf(stderr, "[gpu-fail] command buffer creation\n");
            return SALT_TFAIL(-1);
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) {
            if (_cfg_diag)
                fprintf(stderr, "[gpu-fail] compute encoder creation\n");
            return SALT_TFAIL(-1);
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
                {
        NSUInteger _off = sel_args_put(&rv, sizeof rv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
                {
        NSUInteger _off = sel_args_put(&cv, sizeof cv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
                {
        NSUInteger _off = sel_args_put(&nj, sizeof nj);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:8];
    }
                [enc dispatchThreads:MTLSizeMake((NSUInteger)Rj[j], 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                _q4_simd_trunk_dispatches++;
            }
        } else if (batch_bind_weight_buffers(
                enc, resource, operation_view) != 0) {
            return SALT_TFAIL(-1);
        } else if (use_tiled) {
            uint nd = (uint)descriptor_count;
            [enc setComputePipelineState:
                _tiled_simdgroups > 0 ? _btiled_pso : _bcoalesced_pso];
            [enc setBuffer:_bx2[bi] offset:0 atIndex:3];
            [enc setBuffer:_by2[bi] offset:0 atIndex:4];
            [enc setBuffer:_desc offset:0 atIndex:5];
            {
        NSUInteger _off = sel_args_put(&nd, sizeof nd);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
            if (_tiled_simdgroups > 0) {
                {
                NSUInteger _off = sel_args_put(&_tiled_simdgroups, sizeof _tiled_simdgroups);
                if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
                [enc setBuffer:_sel_args_scratch offset:_off atIndex:7];
            }
                {
                NSUInteger _off = sel_args_put(&_tiled_token_span, sizeof _tiled_token_span);
                if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
                [enc setBuffer:_sel_args_scratch offset:_off atIndex:8];
            }
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
            {
        NSUInteger _off = sel_args_put(&rv, sizeof rv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
            {
        NSUInteger _off = sel_args_put(&cv, sizeof cv);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
            {
        NSUInteger _off = sel_args_put(&nj, sizeof nj);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:8];
    }
            [enc dispatchThreads:MTLSizeMake((NSUInteger)Rtot, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            _q4_simd_trunk_dispatches++;
        }
        [enc endEncoding];
        struct timespec _ta, _tb, _tw;
        uint64_t command_done_value = metal_command_event_encode(cb);
        if (command_done_value == 0u) return SALT_TFAIL(-1);
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
        if (metal_command_event_wait(command_done_value) != 0)
            return SALT_TFAIL(-1);
        clock_gettime(CLOCK_MONOTONIC, &_tw);
        if (detail) {
            [cb waitUntilCompleted];
            if (metal_command_succeeded(cb) != 0) return SALT_TFAIL(-1);
        }
        if (metal_command_event_record(cb, UINT32_MAX,
            METAL_MANIFEST_GPU_TRUNK, 0u, Rtot,
            (uint64_t)_ta.tv_sec * 1000000000u + (uint64_t)_ta.tv_nsec,
            (uint64_t)_tw.tv_sec * 1000000000u + (uint64_t)_tw.tv_nsec) != 0)
            return SALT_TFAIL(-1);
        if (detail) {
            clock_gettime(CLOCK_MONOTONIC, &detail_wait);
            /* This command buffer's OWN device execution window. Summing this
             * across every submission gives the device's total busy time, which
             * is what separates "host re-arming the device" from "device idle". */
            detail_gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;
        }
        if (use_tiled && _cfg_diag) {
            uint32_t first[4];
            memcpy(first, _by2[bi].contents, sizeof first);
            fprintf(stderr,
                "[gpu-tiled-write] first=0x%08x/%08x/%08x/%08x\n",
                first[0], first[1], first[2], first[3]);
        }
        _buf_idx = 1 - bi;
        clock_gettime(CLOCK_MONOTONIC, &_tb);
        if (_cfg_ms) {
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
                "encode_ms=%.3f wait_ms=%.3f scatter_ms=%.3f total_ms=%.3f "
                "gpu_ms=%.3f\n",
                detail_id, njobs, Rtot, C, detail_resource_kind,
                detail_view_bytes, admit_sync_ms, view_ms, desc_ms,
                encode_ms, wait_ms, scatter_ms, total_ms, detail_gpu_ms);
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

/* Phase-scoped text-row normalization on the device.  The same kernels the
 * target program uses, dispatched into the serving queue instead of the
 * program encoder, so PREFILL can keep its router/routed inputs on the device.
 * Host-visible shared buffers are used so existing consumers keep working. */
int salt_gpu_text_row_norm(const SaltGpuSharedBuffer *canonical,
                           size_t x_float_offset, size_t y_float_offset,
                           const SaltGpuSharedBuffer *weight,
                           int rows, int width, int use_weight,
                           float eps, float scalar) {
    id<MTLComputePipelineState> pipeline;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint64_t x, y, z;
        uint32_t rows, width, aux;
        float eps, scalar;
    } args;
    if (!canonical || !canonical->backend || !_bqueue2 || !_op_status ||
        rows < 1 || width < 1 ||
        x_float_offset % sizeof(float) != 0 ||
        y_float_offset % sizeof(float) != 0)
        return SALT_TFAIL(-1);
    if (use_weight && (!weight || !weight->backend)) return SALT_TFAIL(-1);
    /* the same pending discipline the batch path uses */
    if (_pend_cb && salt_gpu_sync() != 0) return SALT_TFAIL(-1);
    pipeline = use_weight ? _text_router_multi_pso : _text_rms_multi_pso;
    if (!pipeline) return SALT_TFAIL(-1);
    args.x = (uint64_t)(x_float_offset / sizeof(float));
    args.y = (uint64_t)(y_float_offset / sizeof(float));
    args.z = 0u;
    args.rows = (uint32_t)rows;
    args.width = (uint32_t)width;
    args.aux = use_weight ? 1u : 0u;
    args.eps = eps;
    args.scalar = scalar;
    cb = [_bqueue2 commandBuffer];
    if (!cb) return SALT_TFAIL(-1);
    encoder = [cb computeCommandEncoder];
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)canonical->backend offset:0 atIndex:0];
    [encoder setBuffer:use_weight ? (id<MTLBuffer>)weight->backend
                                 : (id<MTLBuffer>)canonical->backend
                offset:0 atIndex:1];
    [encoder setBuffer:_op_status offset:0 atIndex:2];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:3];
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return metal_command_succeeded(cb) != 0 ? -1 : 0;
}

int salt_gpu_shared_buffer_free(SaltGpuSharedBuffer *buffer) {
    id<MTLBuffer> storage;
    if (!buffer) return SALT_TFAIL(-1);
    if (!buffer->contents && !buffer->backend && buffer->nbytes == 0) return 0;
    if (!buffer->contents || !buffer->backend || buffer->nbytes < 1 ||
        salt_gpu_sync() != 0)
        return SALT_TFAIL(-1);
    storage = (id<MTLBuffer>)buffer->backend;
    if ([storage contents] != buffer->contents ||
        (size_t)[storage length] < buffer->nbytes)
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        if (!cb || !enc) return SALT_TFAIL(-1);
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
        {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
        [enc setThreadgroupMemoryLength:score_bytes atIndex:0];
        [enc setThreadgroupMemoryLength:2u * sizeof(float) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        if (!cb || !enc) return SALT_TFAIL(-1);
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
        {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
        [enc setThreadgroupMemoryLength:sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    for (int i = 0; i < count; i++) {
        if (experts[i].group < 1) return SALT_TFAIL(-1);
        rows64 += (uint32_t)experts[i].group;
    }
    elements64 = rows64 * (uint32_t)routed;
    if (rows64 > BATCH_MAXSLOT || elements64 > SIZE_MAX / sizeof(float) ||
        elements64 > scratch_float_capacity)
        return SALT_TFAIL(-1);
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
        {
        NSUInteger _off = sel_args_put(&elements, sizeof elements);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:3];
    }
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
        return SALT_TFAIL(-1);
    uint nd = (uint)descriptor_count;
    [encoder setComputePipelineState:_bselected_pso];
    [encoder setBuffer:resource_arguments offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:_selected_desc offset:descriptor_offset atIndex:3];
    {
        NSUInteger _off = sel_args_put(&nd, sizeof nd);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:4];
    }
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:_selected_logical_payload_offsets offset:0 atIndex:6];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder dispatchThreads:
        MTLSizeMake((NSUInteger)max_rows, (NSUInteger)max_batch,
                    (NSUInteger)descriptor_count)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    _q4_simd_selected_dispatches++;
    return 0;
}

typedef struct {
    uint32_t jobs, rows, cols, experts, layer, projection, projections, bank_origin,
        arena_slots;
    uint64_t stride, input, output, second_output;
} MetalRotationExpertArgs;

static int __attribute__((unused)) metal_selected_encode_ragged(
        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> canonical,
        NSUInteger selected_offset, NSUInteger grouped_offset,
        const MetalRotationExpertArgs *args) {
    id<MTLBuffer> bank = salt_lr_decode_active() ? _rotation_arena_backing :
        (args ? _rotation_bank_view[args->bank_origin / args->experts] : nil);
    if (!encoder || !canonical || !args || !args->jobs || !bank ||
        !_bselected_ragged_pso || !_text_manifest || !_selected_logical_slots)
        return SALT_TFAIL(-1);
    NSUInteger offset = sel_args_put(args, sizeof *args);
    if (offset == (NSUInteger)-1) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_bselected_ragged_pso];
    [encoder setBuffer:bank offset:0 atIndex:0];
    [encoder setBuffer:canonical offset:0 atIndex:1];
    [encoder setBuffer:canonical offset:0 atIndex:2];
    [encoder setBuffer:_text_manifest offset:0 atIndex:3];
    [encoder setBuffer:_sel_args_scratch offset:offset atIndex:4];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:canonical offset:grouped_offset atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder setBuffer:canonical offset:selected_offset atIndex:9];
    [encoder dispatchThreads:MTLSizeMake(args->rows, args->jobs, args->projections)
         threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    return 0;
}

static int metal_selected_encode_rows4(
        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> canonical,
        NSUInteger selected_offset, NSUInteger grouped_offset,
        const MetalRotationExpertArgs *args) {
    id<MTLBuffer> bank = salt_lr_decode_active() ? _rotation_arena_backing :
        (args ? _rotation_bank_view[args->bank_origin / args->experts] : nil);
    if (!encoder || !canonical || !args || !args->jobs || !args->rows ||
        !args->projections || !bank || !_bselected_rows4_pso ||
        !_text_manifest || !_selected_logical_slots || !_op_status)
        return SALT_TFAIL(-1);
    NSUInteger offset = sel_args_put(args, sizeof *args);
    if (offset == (NSUInteger)-1) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_bselected_rows4_pso];
    [encoder setBuffer:bank offset:0 atIndex:0];
    [encoder setBuffer:canonical offset:0 atIndex:1];
    [encoder setBuffer:canonical offset:0 atIndex:2];
    [encoder setBuffer:_text_manifest offset:0 atIndex:3];
    [encoder setBuffer:_sel_args_scratch offset:offset atIndex:4];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:canonical offset:grouped_offset atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder setBuffer:canonical offset:selected_offset atIndex:9];
    [encoder dispatchThreadgroups:
        MTLSizeMake(((NSUInteger)args->rows + 127u) / 128u,
                    (NSUInteger)args->jobs, (NSUInteger)args->projections)
         threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    return 0;
}

static int metal_selected_encode_grouped(
        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> canonical,
        const SaltTextCanonicalLayout *layout,
        const MetalRotationExpertArgs *args) {
    id<MTLBuffer> bank = salt_lr_decode_active() ? _rotation_arena_backing :
        (args ? _rotation_bank_view[args->bank_origin / args->experts] : nil);
    if (!encoder || !canonical || !layout || !args || !args->jobs ||
        !args->rows || !args->cols || !args->experts || !args->projections ||
        !bank || !_bselected_grouped_pso ||
        !_text_manifest || !_selected_logical_slots || !_op_status)
        return SALT_TFAIL(-1);
    uint64_t maximum_tiles = (uint64_t)args->experts + (args->jobs + 7u) / 8u;
    uint64_t indirect_index = (uint64_t)args->experts + 1u + 2u * maximum_tiles +
        (args->projection == 2u ? 3u : 0u);
    if (args->projection != 0u && args->projection != 2u)
        return SALT_TFAIL(-1);
    if (indirect_index + 3u > layout->maximum_tensor_row ||
        indirect_index > (NSUIntegerMax - layout->tensor_row) / sizeof(uint32_t))
        return SALT_TFAIL(-1);
    NSUInteger offset = sel_args_put(args, sizeof *args);
    if (offset == (NSUInteger)-1) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_bselected_grouped_pso];
    [encoder setBuffer:bank offset:0 atIndex:0];
    [encoder setBuffer:canonical offset:0 atIndex:1];
    [encoder setBuffer:canonical offset:0 atIndex:2];
    [encoder setBuffer:_text_manifest offset:0 atIndex:3];
    [encoder setBuffer:_sel_args_scratch offset:offset atIndex:4];
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:canonical offset:layout->tensor_row atIndex:6];
    [encoder setBuffer:canonical offset:layout->grouped_to_canonical atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    [encoder setBuffer:canonical offset:layout->selected_experts atIndex:9];
    [encoder setThreadgroupMemoryLength:1024u * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroupsWithIndirectBuffer:canonical
        indirectBufferOffset:layout->tensor_row +
            (NSUInteger)indirect_index * sizeof(uint32_t)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    return 0;
}

/* SIMD-group shape of the selected/ragged expert GEMV: one threadgroup (one
 * SIMD group of 32 lanes) per (output row r, task), with the K dimension split
 * across the lanes and a fixed-order simd_shuffle reduction.  The reduction
 * tree is term-for-term identical to the serial kernel's
 *     s4 = a0+a1+..+a7; tt = s4.xy+s4.zw; y = tt.x+tt.y
 * so the arithmetic - and therefore the committed state - is unchanged.
 * Grid is by THREADGROUP (the kernel reads threadgroup_position_in_grid), so
 * this uses dispatchThreadgroups, not dispatchThreads.  No threadgroup memory
 * is needed: each lane owns element (cb+lane) of every 32-element chunk. */
static int metal_selected_encode_warp(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> resource_arguments, id<MTLBuffer> input_buffer,
        id<MTLBuffer> output_buffer, NSUInteger descriptor_offset,
        int descriptor_count, int max_rows, NSUInteger task_offset,
        uint32_t task_count) {
    uint32_t counts[2];
    if (!encoder || !resource_arguments || !input_buffer || !output_buffer ||
        descriptor_count < 1 || descriptor_count > 256 || max_rows < 1 ||
        task_count < 1 || !_bselected_warp_pso || !_selected_desc ||
        !_selected_task_map || !_selected_logical_slots ||
        !_selected_logical_payload_offsets || !_op_status ||
        descriptor_offset > [_selected_desc length] ||
        (NSUInteger)descriptor_count * sizeof(SelectedDesc) >
            [_selected_desc length] - descriptor_offset ||
        task_offset > [_selected_task_map length] ||
        (NSUInteger)task_count * sizeof(MetalSelectedTask) >
            [_selected_task_map length] - task_offset)
        return SALT_TFAIL(-1);
    counts[0] = (uint32_t)descriptor_count;
    counts[1] = task_count;
    [encoder setComputePipelineState:_bselected_warp_pso];
    [encoder setBuffer:resource_arguments offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:_selected_desc offset:descriptor_offset atIndex:3];
    {
        NSUInteger _off = sel_args_put(&counts, sizeof counts);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:4];
    }
    [encoder setBuffer:_selected_logical_slots offset:0 atIndex:5];
    [encoder setBuffer:_selected_logical_payload_offsets offset:0 atIndex:6];
    [encoder setBuffer:_selected_task_map offset:task_offset atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    /* 8 SIMD groups per threadgroup, each owning a different output row, so
     * the threadgroup is 256 lanes and the grid is rows/8.  The reduction
     * tree is per-SIMD-group, so exactness is unaffected by the packing. */
    /* 8 SIMD groups per threadgroup; each SIMD group owns 4 output rows
     * (8 lanes per row), each lane decoding one float4 of the chunk -
     * the same vectorized decode the serial kernel uses. */
    [encoder dispatchThreadgroups:
        MTLSizeMake(((NSUInteger)max_rows + 31u) / 32u,
                    (NSUInteger)task_count, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
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
        return SALT_TFAIL(-1);
    storage = (id<MTLBuffer>)arena->backend;
    if ([storage contents] != arena->contents ||
        (size_t)[storage length] < arena->nbytes)
        return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
        input_bytes = (size_t)entry->cols * (size_t)entry->batch *
            sizeof(float);
        output_bytes = (size_t)entry->rows * (size_t)entry->batch *
            sizeof(float);
        input = (uintptr_t)(const void *)entry->inputs;
        output = (uintptr_t)(void *)entry->outputs;
        if (input < arena_base || output < arena_base ||
            input - arena_base > SIZE_MAX || output - arena_base > SIZE_MAX)
            return SALT_TFAIL(-1);
        input_offset = (size_t)(input - arena_base);
        output_offset = (size_t)(output - arena_base);
        if (input_offset > arena->nbytes ||
            input_bytes > arena->nbytes - input_offset ||
            output_offset > arena->nbytes ||
            output_bytes > arena->nbytes - output_offset ||
            (input_offset < output_offset + output_bytes &&
             output_offset < input_offset + input_bytes) ||
            (input_offset | output_offset) % sizeof(float) != 0)
            return SALT_TFAIL(-1);
        for (int prior = 0; prior < job; prior++) {
            uintptr_t prior_output = (uintptr_t)(void *)jobs[prior].outputs;
            size_t prior_bytes = (size_t)jobs[prior].rows *
                (size_t)jobs[prior].batch * sizeof(float);
            if (output < prior_output + prior_bytes &&
                prior_output < output + output_bytes)
                return SALT_TFAIL(-1);
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
        if (!cb || !enc) return SALT_TFAIL(-1);
        *(uint32_t *)[_op_status contents] = 0u;
        for (int job = 0; job < job_count; job++)
            [enc useResource:_selected_cache_resources[
                jobs[job].resource_slot].map usage:MTLResourceUsageRead];
        if (metal_selected_encode(enc, _wave_selected_args, storage, storage,
                0, job_count, max_rows, max_batch) != 0) {
            [enc endEncoding];
            return SALT_TFAIL(-1);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (metal_command_succeeded(cb) != 0 ||
            *(uint32_t *)[_op_status contents] != 0u)
            return SALT_TFAIL(-1);
    }
    _batch_stats.expert_layer_batches++;
    _batch_stats.direct_output_batches++;
    _batch_stats.direct_output_jobs += logical_jobs;
    return 0;
}


/* One-shot empty-command-buffer latency probe (SALT_CB_EMPTY_PROBE=1).
 * A CB with no dispatches costs nothing on the GPU, so its commit->completion
 * time is pure driver/CB lifecycle. Compare with the ~5 ms a real 0.5 ms expert
 * CB pays: if they match, the cost is the CB lifecycle, not execution. */
static void metal_probe_empty_cb_latency(void) {
    static int done = 0;
    struct timespec a, b;
    id<MTLCommandBuffer> cb;
    if (done || !_cfg_cb_empty) return;
    done = 1;
    for (int i = 0; i < 5; i++) {
        @autoreleasepool {
            cb = [_queue commandBuffer];
            if (!cb) return;
            clock_gettime(CLOCK_MONOTONIC, &a);
            [cb commit];
            [cb waitUntilCompleted];
            clock_gettime(CLOCK_MONOTONIC, &b);
            fprintf(stderr, "[cb-empty-ms] i=%d commit_to_complete_ms=%.3f"
                    " gpu_ms=%.3f\n", i,
                    (double)(b.tv_sec - a.tv_sec) * 1e3 +
                        (double)(b.tv_nsec - a.tv_nsec) * 1e-6,
                    ([cb GPUEndTime] >= [cb GPUStartTime])
                        ? ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 : 0.0);
        }
    }
}
static void metal_manifest_gpu_window_record(
        id<MTLCommandBuffer> command_buffer, uint32_t layer,
        uint32_t stage, uint32_t experts, uint64_t rows,
        uint64_t commit_ns, uint64_t wait_return_ns) {
    uint32_t phases;
    double begin, end;
    MetalManifestGpuWindow *record;
    SaltRotationTimelinePhase phase;
    if (!command_buffer ||
            (phases = salt_lr_timeline_phase_count()) < 2u ||
            !salt_lr_timeline_phase(phases - 1u, &phase) || phase.kind != 2u)
        return;
    begin = [command_buffer GPUStartTime];
    end = [command_buffer GPUEndTime];
    if (!isfinite(begin) || !isfinite(end) || begin <= 0.0 || end < begin ||
        !commit_ns || wait_return_ns < commit_ns ||
        begin > (double)UINT64_MAX * 1e-9 ||
        end > (double)UINT64_MAX * 1e-9)
        return;
    if (_manifest_gpu_window_count >= METAL_MANIFEST_GPU_WINDOWS) {
        _manifest_gpu_window_dropped++;
        return;
    }
    record = &_manifest_gpu_windows[_manifest_gpu_window_count++];
    *record = (MetalManifestGpuWindow) {
        .phase = phases - 1u,
        .layer = layer,
        .stage = stage,
        .experts = experts,
        .rows = rows,
        .begin_ns = (uint64_t)(begin * 1e9 + 0.5),
        .end_ns = (uint64_t)(end * 1e9 + 0.5),
        .commit_ns = commit_ns,
        .wait_return_ns = wait_return_ns,
    };
}

static void metal_command_event_harvest_one(
        const MetalCommandEventPending *pending) {
    double begin, end;
    MetalManifestGpuWindow *record;
    if (!pending || !pending->command || pending->phase == UINT32_MAX) return;
    begin = [pending->command GPUStartTime];
    end = [pending->command GPUEndTime];
    if (!isfinite(begin) || !isfinite(end) || begin <= 0.0 || end < begin ||
            !pending->commit_ns || pending->wait_return_ns < pending->commit_ns ||
            begin > (double)UINT64_MAX * 1e-9 ||
            end > (double)UINT64_MAX * 1e-9)
        return;
    if (_manifest_gpu_window_count >= METAL_MANIFEST_GPU_WINDOWS) {
        _manifest_gpu_window_dropped++;
        return;
    }
    record = &_manifest_gpu_windows[_manifest_gpu_window_count++];
    *record = (MetalManifestGpuWindow) {
        .phase = pending->phase,
        .layer = pending->layer,
        .stage = pending->stage,
        .experts = pending->experts,
        .rows = pending->rows,
        .begin_ns = (uint64_t)(begin * 1e9 + 0.5),
        .end_ns = (uint64_t)(end * 1e9 + 0.5),
        .commit_ns = pending->commit_ns,
        .wait_return_ns = pending->wait_return_ns,
    };
}

static int metal_command_event_harvest(int force) {
    while (_command_event_pending_count > 0u) {
        MetalCommandEventPending *pending =
            &_command_event_pending[_command_event_pending_head];
        MTLCommandBufferStatus status = [pending->command status];
        if (status != MTLCommandBufferStatusCompleted &&
                status != MTLCommandBufferStatusError) {
            if (!force) break;
            _command_done_forced++;
            [pending->command waitUntilCompleted];
            status = [pending->command status];
        }
        if (status == MTLCommandBufferStatusCompleted)
            metal_command_event_harvest_one(pending);
        else
            _command_event_error = 1;
        [pending->command release];
        memset(pending, 0, sizeof *pending);
        _command_event_pending_head =
            (_command_event_pending_head + 1u) % METAL_COMMAND_EVENT_PENDING;
        _command_event_pending_count--;
        _command_done_harvested++;
    }
    return _command_event_error ? SALT_TFAIL(-1) : 0;
}

static uint64_t metal_command_event_encode(id<MTLCommandBuffer> command) {
    uint64_t value;
    if (!command || !_command_done_event || _command_done_value == UINT64_MAX)
        return 0;
    value = ++_command_done_value;
    [command encodeSignalEvent:_command_done_event value:value];
    return value;
}

static int metal_command_event_wait(uint64_t value) {
    if (!_command_done_event || value == 0u ||
            ![_command_done_event waitUntilSignaledValue:value timeoutMS:600000])
        return SALT_TFAIL(-1);
    _command_done_waited++;
    return _command_event_error ? SALT_TFAIL(-1) : 0;
}

static int metal_command_event_record(id<MTLCommandBuffer> command,
        uint32_t layer, uint32_t stage, uint32_t experts, uint64_t rows,
        uint64_t commit_ns, uint64_t wait_return_ns) {
    uint32_t phase_index = UINT32_MAX;
    uint32_t phases;
    SaltRotationTimelinePhase phase;
    MetalCommandEventPending *pending;
    if (!command || metal_command_event_harvest(0) != 0)
        return SALT_TFAIL(-1);
    if (_command_event_pending_count >= METAL_COMMAND_EVENT_PENDING &&
            metal_command_event_harvest(1) != 0)
        return SALT_TFAIL(-1);
    if (_command_event_pending_count >= METAL_COMMAND_EVENT_PENDING)
        return SALT_TFAIL(-1);
    phases = salt_lr_timeline_phase_count();
    if (phases >= 2u && salt_lr_timeline_phase(phases - 1u, &phase) &&
            phase.kind == 2u)
        phase_index = phases - 1u;
    pending = &_command_event_pending[
        (_command_event_pending_head + _command_event_pending_count) %
            METAL_COMMAND_EVENT_PENDING];
    *pending = (MetalCommandEventPending) {
        .command = [command retain],
        .phase = phase_index,
        .layer = layer,
        .stage = stage,
        .experts = experts,
        .rows = rows,
        .commit_ns = commit_ns,
        .wait_return_ns = wait_return_ns,
    };
    _command_event_pending_count++;
    return 0;
}

static int metal_command_event_flush(void) {
    return metal_command_event_harvest(1);
}

int salt_gpu_q4_moe_chain_selected(const SaltGpuMoeExpert *experts, int count,
                                   int hidden, int routed,
                                   const float *inputs, float *outputs,
                                   float *gate_scratch, float *up_scratch,
                                   size_t scratch_float_capacity) {
    metal_probe_empty_cb_latency();
    id<MTLBuffer> views[128] = {nil};
    uintptr_t aligned_bases[128] = {0};
    uint32_t resource_indices[128] = {0};
    uint32_t resolve_on_device[128] = {0};
    const uint8_t *rotation_arena = NULL;
    uint64_t rotation_stride = 0, rotation_slot_bytes = 0;
    int rotation_slots = 0;
    int rotation_direct = salt_rot_inference_active() &&
        !salt_lr_decode_zero_copy_active();
    id<MTLBuffer> input_buffer, output_buffer, gate_buffer, up_buffer;
    size_t input_offset, output_offset, gate_offset, up_offset;
    uint64_t total_rows = 0;
    int max_group = 0, rc = -1;
    int detail = _cfg_moe_ms_detail;
    struct timespec detail_start, detail_prepared, detail_encoded;
    struct timespec detail_committed, detail_waited;
    struct timespec manifest_commit, manifest_wait;
    static unsigned long long detail_sequence = 0;
    unsigned long long detail_id = 0;
    if (!experts || count < 1 || count > 128 || hidden < 1 || routed < 1 ||
        (hidden & 31) != 0 || (routed & 31) != 0 || !inputs || !outputs ||
        !gate_scratch || !up_scratch || !_selected_cache_resources ||
        !_bselected_pso || !_wave_selected_args ||
        batch_pipeline_prepare() != 0 || metal_ops_prepare() != 0)
        return SALT_TFAIL(-1);
    if (rotation_direct) {
        if (salt_rot_inference_arena(&rotation_arena, &rotation_stride,
                &rotation_slot_bytes, &rotation_slots) != 0 ||
            !rotation_arena || !_rotation_arena_backing ||
            rotation_slots < 1 || rotation_stride != rotation_slot_bytes ||
            _rotation_experts_per_bank < 1)
            return SALT_TFAIL(-1);
    }
    if (detail) {
        detail_id = ++detail_sequence;
        clock_gettime(CLOCK_MONOTONIC, &detail_start);
    }
    for (int i = 0; i < count; i++) {
        if (experts[i].group < 1 ||
            total_rows > UINT64_MAX - (uint32_t)experts[i].group)
            return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
            uintptr_t aligned;
            /* Rotation seats are fixed argument-buffer entries created once at
             * initialization.  Consume that existing entry directly: no resource
             * object, bind, map, allocation or logical-slot rewrite occurs here. */
            if (entry->resource_slot >= 0 && rotation_direct) {
                int bank;
                uintptr_t resource_lo;
                if (entry->resource_slot >= rotation_slots ||
                    entry->logical_resource_id >=
                        (uint64_t)(uint32_t)_selected_logical_capacity ||
                    rotation_stride > UINTPTR_MAX -
                        (uintptr_t)(const void *)rotation_arena)
                    goto done;
                bank = entry->resource_slot / _rotation_experts_per_bank;
                if (bank < SALT_LR_BANK_COUNT) {
                    if (bank < 0 || !_rotation_bank_view[bank]) goto done;
                    views[i] = _rotation_bank_view[bank];
                } else {
                    views[i] = _rotation_arena_backing;
                }
                resource_lo = (uintptr_t)(const void *)rotation_arena +
                    (uint64_t)(uint32_t)bank * _rotation_experts_per_bank *
                        rotation_stride;
                if (_rotation_bank_bytes > UINTPTR_MAX - resource_lo ||
                    lo < resource_lo || hi > resource_lo + _rotation_bank_bytes)
                    goto done;
                aligned = resource_lo;
                resource_indices[i] = (uint32_t)bank;
                resolve_on_device[i] = 0u;
            } else if (entry->resource_slot >= 0) {
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
                resource_indices[i] = (uint32_t)entry->resource_slot;
                resolve_on_device[i] = 1u;
            } else {
                goto done;
            }
            aligned_bases[i] = aligned;
            uint64_t xoff = input_offset / sizeof(float) + row_prefix * hidden;
            uint64_t goff = gate_offset / sizeof(float) + row_prefix * routed;
            uint64_t uoff = up_offset / sizeof(float) + row_prefix * routed;
            desc[2 * i] = (SelectedDesc) {
                {(uintptr_t)entry->gate_vals - aligned,
                 (uintptr_t)entry->gate_scales - aligned, xoff, goff},
                {(uintptr_t)entry->gate_biases - aligned,
                 (uint64_t)routed, (uint64_t)hidden,
                 (uint64_t)(uint32_t)entry->group},
                /* dc = {bank_resource_index, resolve_on_device,
                 * expert_logical_id, unused}. The descriptor offsets address
                 * the expert seat within that startup-bound bank resource. */
                {resource_indices[i], resolve_on_device[i],
                 entry->logical_resource_id, 0u},
            };
            desc[2 * i + 1] = (SelectedDesc) {
                {(uintptr_t)entry->up_vals - aligned,
                 (uintptr_t)entry->up_scales - aligned, xoff, uoff},
                {(uintptr_t)entry->up_biases - aligned,
                 (uint64_t)routed, (uint64_t)hidden,
                 (uint64_t)(uint32_t)entry->group},
                /* dc = {bank_resource_index, resolve_on_device,
                 * expert_logical_id, unused}. The descriptor offsets address
                 * the expert seat within that startup-bound bank resource. */
                {resource_indices[i], resolve_on_device[i],
                 entry->logical_resource_id, 0u},
            };
            row_prefix += (uint32_t)entry->group;
        }
        SelectedDesc *down_desc = (SelectedDesc *)((unsigned char *)desc +
            METAL_SELECTED_DESC_DOWN_OFFSET);
        row_prefix = 0;
        for (int i = 0; i < count; i++) {
            const SaltGpuMoeExpert *entry = &experts[i];
            uintptr_t aligned = aligned_bases[i];
            down_desc[i] = (SelectedDesc) {
                {(uintptr_t)entry->down_vals - aligned,
                 (uintptr_t)entry->down_scales - aligned,
                 gate_offset / sizeof(float) + row_prefix * routed,
                 output_offset / sizeof(float) + row_prefix * hidden},
                {(uintptr_t)entry->down_biases - aligned,
                 (uint64_t)hidden, (uint64_t)routed,
                 (uint64_t)(uint32_t)entry->group},
                /* dc = {bank_resource_index, resolve_on_device,
                 * expert_logical_id, unused}. The descriptor offsets address
                 * the expert seat within that startup-bound bank resource. */
                {resource_indices[i], resolve_on_device[i],
                 entry->logical_resource_id, 0u},
            };
            row_prefix += (uint32_t)entry->group;
        }
        if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_prepared);
        {
            id<MTLCommandBuffer> cb = [_bqueue2 commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            uint32_t elements = (uint32_t)(total_rows * (uint32_t)routed);
            if (!cb || !enc) goto done;
            /* useResource is REQUIRED here: removing it (SALT_MOE_USERESOURCE
             * unset) made decode 215.67 ms vs 145.54 ms with it on. The ~5 ms
             * commit->completion is a platform round trip, not this call. */
            if (!_cfg_no_useresource)
                for (int i = 0; i < count; i++)
                    [enc useResource:views[i] usage:MTLResourceUsageRead];
            if (metal_selected_encode(enc, _wave_selected_args,
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
            {
        NSUInteger _off = sel_args_put(&elements, sizeof elements);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:3];
    }
            [enc dispatchThreads:MTLSizeMake(elements, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (metal_selected_encode(enc, _wave_selected_args,
                    gate_buffer, output_buffer,
                    METAL_SELECTED_DESC_DOWN_OFFSET,
                    count, hidden, max_group) != 0) {
                [enc endEncoding];
                goto done;
            }
            [enc endEncoding];
            if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_encoded);
            uint64_t command_done_value = metal_command_event_encode(cb);
            if (command_done_value == 0u) goto done;
            clock_gettime(CLOCK_MONOTONIC, &manifest_commit);
            [cb commit];
            if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_committed);
            if (metal_command_event_wait(command_done_value) != 0) goto done;
            clock_gettime(CLOCK_MONOTONIC, &manifest_wait);
            if (detail) clock_gettime(CLOCK_MONOTONIC, &detail_waited);
            if (detail) {
                [cb waitUntilCompleted];
                if (metal_command_succeeded(cb) != 0) goto done;
            }
            if (metal_command_event_record(cb,
                _text_manifest_experts
                    ? (uint32_t)(experts[0].logical_resource_id /
                        _text_manifest_experts) : UINT32_MAX,
                METAL_MANIFEST_GPU_EXPERT, (uint32_t)count, total_rows,
                (uint64_t)manifest_commit.tv_sec * 1000000000u +
                    (uint64_t)manifest_commit.tv_nsec,
                (uint64_t)manifest_wait.tv_sec * 1000000000u +
                    (uint64_t)manifest_wait.tv_nsec) != 0)
                goto done;
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
        if (_cfg_diag) {
            fprintf(stderr, "[gpu-selected] chain=%a up=%a output=%a\n",
                    gate_scratch[0], up_scratch[0], outputs[0]);
        }
        _batch_stats.expert_layer_batches += 2;
        _batch_stats.direct_output_batches += 2;
        _batch_stats.direct_output_jobs += 3u * total_rows;
        rc = 0;
done:
        ; /* Borrowed seats remain bound until cache-owned retirement. */
    }
    return rc;
}

static int salt_gpu_free_impl(void) {
    if (salt_gpu_sync() != 0) {
        fprintf(stderr, "gpu: refusing teardown with pending work\n");
        return SALT_TFAIL(-1);
    }
    if (((_trunk_map == nil) != (_trunk_base == NULL)) ||
        ((_trunk_map == nil) != (_trunk_len == 0)) ||
        ((_res_map == nil) != (_res_base == NULL)) ||
        ((_res_map == nil) != (_res_len == 0))) {
        fprintf(stderr, "gpu: refusing teardown with inconsistent mapping state\n");
        return SALT_TFAIL(-1);
    }
    for (int i = 0; i < _selected_cache_capacity; i++) {
        if (_selected_cache_resources[i].map) {
            fprintf(stderr,
                    "gpu: refusing teardown with bound selected cache slot\n");
            return SALT_TFAIL(-1);
        }
    }
    /* The engine releases the single arena before backend teardown. The
     * logical slot table is immutable address metadata, not a live lease. */
    if (salt_rotation_expert_arena()->backend)
        return SALT_TFAIL(-1);

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
    metal_release(_mtp_q4_0_pso); _mtp_q4_0_pso = nil;
    metal_release(_bselected_pso); _bselected_pso = nil;
    metal_release(_bselected_ragged_pso); _bselected_ragged_pso = nil;
    metal_release(_bselected_rows4_pso); _bselected_rows4_pso = nil;
    metal_release(_bselected_grouped_pso); _bselected_grouped_pso = nil;
    metal_release(_bselected_warp_pso); _bselected_warp_pso = nil;
    metal_release(_selected_arg_encoder); _selected_arg_encoder = nil;
    metal_release(_selected_args); _selected_args = nil;
    metal_release(_selected_desc); _selected_desc = nil;
    metal_release(_selected_task_map); _selected_task_map = nil;
    if (@available(macOS 15.0, *)) {
        if (_rotation_trunk_residency) {
            [_queue removeResidencySet:_rotation_trunk_residency];
            [_rotation_trunk_residency endResidency];
            metal_release(_rotation_trunk_residency);
            _rotation_trunk_residency = nil;
        }
    }
    _rotation_trunk_bytes = _rotation_trunk_members = 0;
    for (NSUInteger bank = 0; bank < SALT_LR_BANK_COUNT; bank++) {
        metal_release(_rotation_bank_view[bank]); _rotation_bank_view[bank] = nil;
    }
    metal_release(_rotation_arena_backing); _rotation_arena_backing = nil;
    _rotation_bank_bytes = 0;
    (void)metal_command_event_flush();
    metal_release(_command_done_event); _command_done_event = nil;
    _command_done_value = 0;
    _command_done_waited = _command_done_harvested = _command_done_forced = 0;
    metal_release(_rotation_done_listener); _rotation_done_listener = nil;
    metal_release(_rotation_done_event); _rotation_done_event = nil;
    metal_release(_rotation_ready_event); _rotation_ready_event = nil;
    metal_text_arrival_release();
    _rotation_epoch = 0;
    metal_release(_wave_selected_arg_encoder);
    _wave_selected_arg_encoder = nil;
    metal_release(_wave_selected_args); _wave_selected_args = nil;
    _selected_argument_capacity = 0;
    metal_release(_selected_logical_slots); _selected_logical_slots = nil;
    metal_release(_text_ledger_slots); _text_ledger_slots = nil;
    free(_text_ledger_gen_host); _text_ledger_gen_host = NULL;
    _text_ledger_slot_host = NULL; _text_ledger_logical_capacity = 0;
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
    metal_release(_text_combine_multi_pso); _text_combine_multi_pso = nil;
    metal_release(_text_rms_multi_pso); _text_rms_multi_pso = nil;
    metal_release(_text_residual_multi_pso); _text_residual_multi_pso = nil;
    metal_release(_text_router_multi_pso); _text_router_multi_pso = nil;
    metal_release(_text_topk_rows_pso); _text_topk_rows_pso = nil;
    metal_release(_text_softcap_pso); _text_softcap_pso = nil;
    metal_release(_text_transform_view_pso); _text_transform_view_pso = nil;
    metal_release(_text_attention_view_pso); _text_attention_view_pso = nil;
    metal_release(_text_attention_product_view_pso); _text_attention_product_view_pso = nil;
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
    if (_cfg_diag)
        fprintf(stderr, "[gpu-fail] metal-icb unsupported\n");
    return SALT_TFAIL(-1);
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
    return SALT_TFAIL(-1);
}

int salt_gpu_nvfp4_matvec(const void *key, int R, int C,
                          const float *input, float *output) {
    (void)key; (void)R; (void)C; (void)input; (void)output;
    return SALT_TFAIL(-1);
}

int salt_gpu_nvfp4_matvec_batch(const void *key, int R, int C, int B,
                                const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return SALT_TFAIL(-1);
}

int salt_gpu_nvfp4_batch(const void *const *keys,
                         const float *const *inputs, float *const *outputs,
                         const int *batches, int R, int C, int njobs) {
    (void)keys; (void)inputs; (void)outputs; (void)batches;
    (void)R; (void)C; (void)njobs;
    return SALT_TFAIL(-1);
}

int salt_gpu_nvfp4_mixed_batch(const void *const *keys,
                               const float *const *inputs,
                               float *const *outputs, const int *batches,
                               const int *rows, int C, int njobs) {
    (void)keys; (void)inputs; (void)outputs; (void)batches; (void)rows;
    (void)C; (void)njobs;
    return SALT_TFAIL(-1);
}

int salt_gpu_bf16_resource_slot(uint32_t kind, uint32_t resource_id,
                                const void *key, const uint16_t *weights,
                                int R, int C) {
    (void)kind; (void)resource_id; (void)key; (void)weights; (void)R; (void)C;
    return SALT_TFAIL(-1);
}
int salt_gpu_bf16_matvec(const void *key, int R, int C,
                         const float *input, float *output) {
    (void)key; (void)R; (void)C; (void)input; (void)output;
    return SALT_TFAIL(-1);
}

int salt_gpu_bf16_matvec_batch(const void *key, int R, int C, int B,
                               const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    R = Rj[0];
    if (R < 1) return SALT_TFAIL(-1);

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
            {
        NSUInteger _off = sel_args_put(&voff, sizeof voff);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
            {
        NSUInteger _off = sel_args_put(&soff, sizeof soff);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
            {
        NSUInteger _off = sel_args_put(&boff, sizeof boff);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
            {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [enc setBuffer:_sel_args_scratch offset:_off atIndex:8];
    }
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
        return SALT_TFAIL(-1);
    for (int i = 0; i < METAL_WEIGHT_RESOURCES; i++)
        if (!_weight_resources[i].used) { slot = &_weight_resources[i]; break; }
    if (!slot || nbytes > (size_t)NSUIntegerMax) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    resource->policy = policy;
    resource->active = 1;
    return 0;
}

int salt_gpu_weight_resource_pool_bind(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource || !resource->active || resource->map ||
        resource->policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW ||
        resource->component_pool || !resource->base || resource->nbytes < 1)
        return SALT_TFAIL(-1);
    resource->component_pool = 1;
    return 0;
}

int salt_gpu_weight_resource_deactivate(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource || !resource->active || salt_gpu_sync() != 0) return SALT_TFAIL(-1);
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
    if (!usage || usage_size != sizeof *usage) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    usage->active_window_bytes += _component_pool_bytes;
    usage->peak_window_bytes = _weight_peak_window_bytes;
    if (usage->peak_window_bytes < _component_pool_peak_bytes)
        usage->peak_window_bytes = _component_pool_peak_bytes;
    return 0;
}

int salt_gpu_weight_resource_bind(uint32_t kind, uint32_t resource_id,
                                  const void *base, size_t nbytes) {
    if (salt_gpu_weight_resource_describe(kind, resource_id, base, nbytes) != 0)
        return SALT_TFAIL(-1);
    if (salt_gpu_weight_resource_activate(kind, resource_id,
            SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW) != 0) {
        MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
        if (resource) memset(resource, 0, sizeof *resource);
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
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
        salt_gpu_sync() != 0) return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    if (batch_weight_sizes(R, C, &vbytes, &sbytes) != 0) return SALT_TFAIL(-1);
    if (bits == 8) {
        uint64_t elements = (uint64_t)(uint32_t)R * (uint32_t)C;
        if (elements > SIZE_MAX) return SALT_TFAIL(-1);
        vbytes = (size_t)elements;
    }
    if (mapped_range_offset(resource->base, resource->nbytes,
            vals, vbytes, &voff) != 0 ||
        mapped_range_offset(resource->base, resource->nbytes,
            scales, sbytes, &soff) != 0 ||
        mapped_range_offset(resource->base, resource->nbytes,
            biases, sbytes, &boff) != 0)
        return SALT_TFAIL(-1);
    for (int i = 0; i < _tslot_n; i++) {
        const SaltGpuResourceRef *ref;
        if (_tslot[i].key != key) continue;
        ref = &_tslot[i].resource;
        return ref->kind == kind && ref->resource_id == resource_id &&
            ref->value_offset == voff && ref->scale_offset == soff &&
            ref->bias_offset == boff ? 0 : -1;
    }
    slot_index = tensor_slot_store(key, kind, resource_id, voff, soff, boff);
    if (slot_index < 0) return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
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
    return SALT_TFAIL(-1);
}

int salt_gpu_weight_resource_unbind(uint32_t kind, uint32_t resource_id) {
    MetalWeightResource *resource = metal_weight_resource(kind, resource_id);
    if (!resource) return SALT_TFAIL(-1);
    if (resource->active &&
        salt_gpu_weight_resource_deactivate(kind, resource_id) != 0)
        return SALT_TFAIL(-1);
    if (resource->map) return SALT_TFAIL(-1);
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
int salt_gpu_metal_present(void) { return 1; }
int salt_gpu_pageable_mmap_active(void) { return 0; }
const SaltGpuDispatch *salt_dispatch_get(void) {
    const char *ov = _cfg_dispatch;
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
    if (_cfg_diag)
        fprintf(stderr, "[gpu-dbg] proj_batch R=%d C=%d B=%d\n", R, C, B);
    if (B < 1 || R < 1 || C < 1 || !vals || !id) return SALT_TFAIL(-1);
    const uint32_t *v[BATCH_MAXSLOT];
    const uint16_t *s[BATCH_MAXSLOT], *b[BATCH_MAXSLOT];
    const void *ids[BATCH_MAXSLOT];
    if (B > BATCH_MAXSLOT) return SALT_TFAIL(-1);
    int Rj[BATCH_MAXSLOT];
    for (int j = 0; j < B; j++) {
        v[j] = vals; s[j] = scales; b[j] = biases;
        ids[j] = id;
        Rj[j] = R;
    }
    if (_cfg_diag)
        fprintf(stderr, "[gpu-dbg] proj_batch filled, calling batch\n");
    return salt_gpu_q4_batch(v, s, b, xs, ys, ids, Rj, C, B);
}

#define METAL_TEXT_MAGIC UINT64_C(0x4d544c5458505247)
#define METAL_TEXT_MAX_LAYERS 64u
#define METAL_ARRIVAL_SAMPLES_PER_BUFFER 4096u
#define METAL_ARRIVAL_ENCODERS_PER_STAGE 4u
/* Metal feature-set limit: 32 counter buffers, each at most 32 KiB.
 * Capture capacity is explicit; it never limits inference itself. */
#define METAL_ARRIVAL_MAX_BUFFERS 32u
static id<MTLCounterSampleBuffer> _arrival_samples[METAL_ARRIVAL_MAX_BUFFERS];
static uint32_t _arrival_buffer_count, _arrival_layer_count, _arrival_available;
static uint32_t _arrival_phase_capacity;
static MTLComputePassDescriptor *_arrival_pass;
static uint32_t _arrival_encoder_uses[SALT_LR_TIMELINE_PHASES][2u * METAL_TEXT_MAX_LAYERS + 1u];
static uint64_t metal_text_counter_stride(void) {
    return (4u * _arrival_layer_count + 2u) * METAL_ARRIVAL_ENCODERS_PER_STAGE;
}
static uint64_t metal_text_counter_index(uint32_t phase_index,
                                        uint32_t segment, uint32_t encoder) {
    return (uint64_t)phase_index * metal_text_counter_stride() +
        2u * ((uint64_t)segment * METAL_ARRIVAL_ENCODERS_PER_STAGE + encoder);
}
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
    id<MTLFence> dependency_fence;
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
    uint64_t rotation_base;
    uint32_t rotation_block_layers, rotation_ready_waits, rotation_done_signals;
    uint32_t decode_experts;
    uint32_t deferred_dense_mask, deferred_dense_layer, overlap_layer;
    id<MTLCommandBuffer> overlap_command;
    uint64_t overlap_submit_ns;
    float overlap_commit_ms;
    id<MTLFence> dependency_fence; /* Borrowed from initialization-owned state. */
    uint32_t dependency_fence_updated;
    double submit_t0, submit_tc;
    SaltRotationIoSnapshot phase_io_begin;
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
    uint32_t arrival_target_layer;
    uint32_t timing_segment;
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
    /* One compute encoder per command buffer. Dependent nodes are separated
     * by buffer-scope memory barriers instead of encoder boundaries; kernels,
     * dispatch geometry, and arithmetic order are unchanged. */
    id<MTLComputeCommandEncoder> encoder;
    uint32_t encoder_nodes;
    SaltTextVerifyBackendStats stats;
} MetalTextProgramCommand;

static int metal_text_decode_demand(MetalTextProgramState *state,
    MetalTextProgramCommand *command, uint32_t layer);
static void metal_text_arm_decode_demands(MetalTextProgramState *state,
    MetalTextProgramCommand *command);
static uint64_t _decode_demand_armed, _decode_demand_received;
static uint64_t _decode_ready_published, _decode_demand_failed;

static int metal_rotation_wait_done(void *context, uint64_t value) {
    (void)context;
    _rotation_done_value_reads++;
    const uint64_t begin_ns = _mt_probe_ns();
    const uint64_t observed = [_rotation_done_event signaledValue];
    const uint64_t end_ns = _mt_probe_ns();
    const uint32_t direct = salt_lr_done_event_proves(value, observed);
    salt_lr_done_event_receive_tick(value, observed, begin_ns, end_ns, direct);
    int rc = direct ? 1 : salt_lr_wait_compute_done(value);
    if (rc == 1) _rotation_done_value_hits++;
    return rc;
}

static void metal_rotation_publish_ready(void *context, uint64_t value) {
    MetalTextProgramCommand *command = (MetalTextProgramCommand *)context;
    SaltMemSnapshot memory;
    if (value != UINT64_MAX && _cfg_mem_limit_bytes &&
        salt_mem_snapshot(&memory) == 0) {
        if (value > command->rotation_base &&
            value - command->rotation_base - 1u < salt_lr_timeline_layer_count())
            salt_lr_timeline_memory(command->rotation_base,
                (uint32_t)(value - command->rotation_base - 1u),
                memory.resident_b > 0 ? (uint64_t)memory.resident_b : 0,
                memory.resident_os_peak_b > 0 ?
                    (uint64_t)memory.resident_os_peak_b : 0);
        salt_trace_register(181, 1);
        salt_trace_value(181, (int)((uint64_t)memory.resident_b / 1000000u));
        salt_trace_register(182, 1);
        salt_trace_value(182, (int)(value - command->rotation_base - 1u));
        if ((uint64_t)memory.resident_b > _cfg_mem_limit_bytes) {
            _rotation_stream_error = 1;
            salt_rot_fail_at(__FILE__, __LINE__, -1);
            salt_lr_stream_cancel();
            value = UINT64_MAX;
        }
    }
    [_rotation_ready_event setSignaledValue:value];
}

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
        return SALT_TFAIL(-1);
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
    return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    memset(ref, 0, sizeof *ref);
    ref->rows = tensor->rows;
    ref->cols = tensor->cols;
    ref->encoding = (uint32_t)storage->encoding;
    if (storage->source_class == SALT_TENSOR_SOURCE_SHARED) {
        if (metal_text_shared_resource(spec, &ref->value, &base_offset) != 0 ||
            storage->value_offset > NSUIntegerMax - base_offset)
            return SALT_TFAIL(-1);
        ref->scale = ref->bias = ref->value;
        ref->value_offset = (NSUInteger)(base_offset + storage->value_offset);
        ref->scale_offset = (NSUInteger)(base_offset + storage->scale_offset);
        ref->bias_offset = (NSUInteger)(base_offset + storage->bias_offset);
        metal_text_alias(ref->value, &ref->value_alias);
        return 0;
    }
    if (storage->source_class == SALT_TENSOR_SOURCE_SELECTED) return SALT_TFAIL(-1);
    resource = metal_weight_resource(storage->resource_kind,
                                     storage->resource_id);
    if (!resource || !resource->active) return SALT_TFAIL(-1);
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
                return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    memset(requirements, 0, sizeof *requirements);
    requirements->backend_state_bytes = sizeof(MetalTextProgramState);
    requirements->command_bytes = sizeof(MetalTextProgramCommand);
    requirements->maximum_commands = program->dispatch.cell_count;
    requirements->flags = 0;
    return 0;
}

static void metal_text_arrival_release(void) {
    for (uint32_t i = 0; i < METAL_ARRIVAL_MAX_BUFFERS; i++) {
        [_arrival_samples[i] release];
        _arrival_samples[i] = nil;
    }
    [_arrival_pass release];
    _arrival_pass = nil;
    _arrival_buffer_count = _arrival_layer_count = _arrival_available = 0u;
    _arrival_phase_capacity = 0u;
    memset(_arrival_encoder_uses, 0, sizeof _arrival_encoder_uses);
}

/* Fixed engine-lifetime start/end samples on EXISTING prefix, expert and tail
 * encoders. No new encoder boundary, submission, event or host readback. */
static int metal_text_arrival_prepare(uint32_t layer_count) {
    id<MTLCounterSet> timestamp_set = nil;
    MTLCounterSampleBufferDescriptor *descriptor;
    uint64_t capacity;
    if (_arrival_available)
        return _arrival_layer_count == layer_count ? 0 : SALT_TFAIL(-1);
    if (![_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
        return 0;
    for (id<MTLCounterSet> set in [_dev counterSets])
        if ([[set name] isEqualToString:MTLCommonCounterSetTimestamp])
            timestamp_set = set;
    if (!timestamp_set || !layer_count || layer_count > METAL_TEXT_MAX_LAYERS)
        return timestamp_set ? SALT_TFAIL(-1) : 0;
    uint64_t phase_samples = (4u * layer_count + 2u) * METAL_ARRIVAL_ENCODERS_PER_STAGE;
    uint64_t maximum_phases = (uint64_t)METAL_ARRIVAL_MAX_BUFFERS *
        METAL_ARRIVAL_SAMPLES_PER_BUFFER / phase_samples;
    _arrival_phase_capacity = maximum_phases < SALT_LR_TIMELINE_PHASES
        ? (uint32_t)maximum_phases : SALT_LR_TIMELINE_PHASES;
    if (!_arrival_phase_capacity) return SALT_TFAIL(-1);
    capacity = (uint64_t)_arrival_phase_capacity * phase_samples;
    _arrival_buffer_count = (uint32_t)((capacity +
        METAL_ARRIVAL_SAMPLES_PER_BUFFER - 1u) / METAL_ARRIVAL_SAMPLES_PER_BUFFER);
    if (_arrival_buffer_count > METAL_ARRIVAL_MAX_BUFFERS)
        return SALT_TFAIL(-1);
    descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
    if (!descriptor) return SALT_TFAIL(-1);
    descriptor.counterSet = timestamp_set;
    descriptor.storageMode = MTLStorageModeShared;
    for (uint32_t i = 0; i < _arrival_buffer_count; i++) {
        uint64_t remaining = capacity - (uint64_t)i * METAL_ARRIVAL_SAMPLES_PER_BUFFER;
        NSError *error = nil;
        descriptor.sampleCount = (NSUInteger)(remaining < METAL_ARRIVAL_SAMPLES_PER_BUFFER
            ? remaining : METAL_ARRIVAL_SAMPLES_PER_BUFFER);
        _arrival_samples[i] = [_dev newCounterSampleBufferWithDescriptor:descriptor
                                                                 error:&error];
        if (!_arrival_samples[i]) {
            [descriptor release];
            metal_text_arrival_release();
            return SALT_TFAIL(-1);
        }
    }
    [descriptor release];
    _arrival_pass = [[MTLComputePassDescriptor computePassDescriptor] retain];
    if (!_arrival_pass) {
        metal_text_arrival_release();
        return SALT_TFAIL(-1);
    }
    _arrival_pass.sampleBufferAttachments[0].startOfEncoderSampleIndex =
        MTLCounterDontSample;
    _arrival_layer_count = layer_count;
    _arrival_available = 1u;
    return 0;
}

static int metal_text_prepare(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltGpuSharedBuffer **canonical_out,
        void *backend_state, size_t backend_state_bytes,
        void *command, size_t command_bytes) {
    /* RSS AT THE POINT THE MANIFEST EXISTS AND THE PROGRAM IS NOT YET PREPARED.  Everything the model
     * allocated during its load is already in the process here, so this is the startup footprint the
     * layer-boundary gate later reports - recorded before anything else in this path runs. */
    {
        SaltMemSnapshot _ps;
        if (salt_mem_snapshot(&_ps) == 0 && _ps.resident_b > 0) {
            salt_trace_register(185, 1);
            salt_trace_value(185, (int)((uint64_t)_ps.resident_b / 1000000u));
        }
    }
    if (!_mt_node_enabled_resolved) {
        _mt_node_enabled_resolved = 1;
        _mt_node_on = getenv("SALT_TEXT_NODE_LEDGER") != NULL;
        _mt_node_last = _mt_probe_ns();
        mt_cfg_snapshot();
    }

    if (!_mt_node_enabled_resolved) {
        _mt_node_enabled_resolved = 1;
        _mt_node_on = getenv("SALT_TEXT_NODE_LEDGER") != NULL;
        _mt_node_last = _mt_probe_ns();
    }

    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    size_t meta_cursor = 0, required_meta_floats = 0;
    size_t required_task_jobs = 0;
    id<MTLBuffer> resized_meta = nil;
    if (!program || !plan || plan->program != program ||
        plan->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !canonical_out || !state || backend_state_bytes < sizeof *state ||
        !command || command_bytes < sizeof(MetalTextProgramCommand) ||
        program->layer_count > METAL_TEXT_MAX_LAYERS)
        return SALT_TFAIL(-1);
    /* THE LEDGER BOOK: build the weight-geometry sub-ledger once, here, from
     * the program's own layer expert descriptors.  Pure static model data -
     * nothing runtime, nothing per layer at execution time. */
    /* Resolve the two phase batch shapes ONCE, into the book. */
    {
        uint32_t _pb = 0u, _db = 0u;
        (void)metal_env_u32("SALT_PREFILL_B", 1, 4096, 0, &_pb);
        if (!_pb) _pb = 512u;                    /* model default */
        (void)metal_env_u32("SALT_DECODE_B", 1, 4096, 0, &_db);
        if (!_db) _db = 1u;                      /* one authoritative row */
        /* Device bound: the prefill chunk is capped by what the program can hold. */
        if (_pb > program->maximum_candidates && program->maximum_candidates)
            _pb = program->maximum_candidates;
        if (_db > _pb) _db = _pb;
        _text_manifest_plan[METAL_TEXT_PLAN_PREFILL_B] = _pb;
        _text_manifest_plan[METAL_TEXT_PLAN_DECODE_B] = _db;
        _text_manifest_plan[METAL_TEXT_PLAN_ROWS_MAX] = _pb;
    }
    if (_text_manifest_capacity < (int)(program->layer_count *
            (uint32_t)program->layers[0].descriptor->plan->n_experts)) {
        uint32_t _max_exp = 0u;
        for (uint32_t _l = 0; _l < program->layer_count; _l++) {
            const SaltTextLayerExecDesc *_s = program->layers[_l].descriptor;
            if (_s && _s->plan &&
                (uint32_t)_s->plan->n_experts > _max_exp)
                _max_exp = (uint32_t)_s->plan->n_experts;
        }
        NSUInteger _cap = (NSUInteger)program->layer_count * _max_exp;
        /* THE BUFFER ALREADY EXISTS, allocated at init.  Reallocating here is what made a failed init
         * invisible, so this only checks that it is large enough. */
        if (_text_manifest && _text_manifest_host &&
            (NSUInteger)_text_manifest_capacity >= _cap)
            goto _manifest_ready;
        metal_release(_text_manifest);
        free(_text_manifest_host);
        _text_manifest = [_dev newBufferWithLength:
            _cap * METAL_TEXT_MANIFEST_FIELDS * sizeof(uint64_t)
            options:MTLResourceStorageModeShared];
        _text_manifest_host = (uint64_t *)calloc(
            (size_t)_cap * METAL_TEXT_MANIFEST_FIELDS, sizeof(uint64_t));
        if (!_text_manifest || ![_text_manifest contents] ||
            !_text_manifest_host) {
            metal_release(_text_manifest); _text_manifest = nil;
            free(_text_manifest_host); _text_manifest_host = NULL;
            _text_manifest_capacity = 0;
            return SALT_TFAIL(-1);
        }
        _text_manifest_host = (uint64_t *)[_text_manifest contents];
        _text_manifest_capacity = (int)_cap;
    }
_manifest_ready:
    _text_manifest_experts = (uint32_t)program->layers[0].descriptor->plan->n_experts;
    _text_manifest_rows = program->layer_count * _text_manifest_experts;
    for (uint32_t _ml = 0; _ml < program->layer_count; _ml++) {
        const SaltTextLayerExecDesc *_ms = program->layers[_ml].descriptor;
        uint32_t _ne = (uint32_t)_ms->plan->n_experts;
        for (uint32_t _me = 0; _me < _ne; _me++) {
            uint64_t _lg = (uint64_t)_ml * _ne + _me;
            uint64_t *_row;
            if (_lg >= (uint64_t)_text_manifest_capacity) break;
            _row = &_text_manifest_host[_lg * METAL_TEXT_MANIFEST_FIELDS];
            _row[0] = _ms->experts[_me].gate.storage.value_offset;
            _row[1] = _ms->experts[_me].gate.storage.scale_offset;
            _row[2] = _ms->experts[_me].gate.storage.bias_offset;
            _row[3] = _ms->experts[_me].up.storage.value_offset;
            _row[4] = _ms->experts[_me].up.storage.scale_offset;
            _row[5] = _ms->experts[_me].up.storage.bias_offset;
            _row[6] = _ms->experts[_me].down.storage.value_offset;
            _row[7] = _ms->experts[_me].down.storage.scale_offset;
            _row[8] = _ms->experts[_me].down.storage.bias_offset;
        }
    }
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextLayerExecDesc *source =
            program->layers[layer].descriptor;
        const SaltAttentionDesc *attention;
        size_t layer_meta;
        if (!source || !source->plan) return SALT_TFAIL(-1);
        if (source->plan->top_k_experts < 1 ||
            program->maximum_candidates > UINT32_MAX /
                (uint32_t)source->plan->top_k_experts)
            return SALT_TFAIL(-1);
        /* Sized for the FULL EXPERT SET, not the routed subset.  A full-set dispatch
         * covers every expert of the layer and lets the kernel decide per (expert, token)
         * from the router - which is what removes the host-side occupancy readback and
         * with it the per-layer commit and wait.  The routed path needs only
         * candidates x top_k; the full-set path needs candidates x n_experts, and the
         * larger of the two is what the map must hold.  Same startup pool, same
         * allocator, no second structure. */
        size_t task_jobs = (size_t)program->maximum_candidates *
            (uint32_t)(source->plan->n_experts > source->plan->top_k_experts
                ? source->plan->n_experts : source->plan->top_k_experts);
        if (task_jobs > required_task_jobs) required_task_jobs = task_jobs;
        attention = &source->plan->attention;
        if (attention->head_dim < 2 || attention->rope_dim < 2 ||
            attention->rope_dim > attention->head_dim ||
            (size_t)program->maximum_candidates >
                (SIZE_MAX - 2u * (size_t)attention->head_dim) /
                    attention->rope_dim)
            return SALT_TFAIL(-1);
        layer_meta = 2u * (size_t)attention->head_dim +
            (size_t)program->maximum_candidates * attention->rope_dim;
        if (layer_meta > SIZE_MAX - required_meta_floats)
            return SALT_TFAIL(-1);
        required_meta_floats += layer_meta;
    }
    if (required_meta_floats > SIZE_MAX / sizeof(float) ||
        metal_ops_prepare() != 0 || metal_exact_prepare() != 0 ||
        batch_pipeline_prepare() != 0 || !_op_meta || !_op_status ||
        !_selected_cache_resources || !_selected_logical_slot_host ||
        !_selected_logical_payload_host)
        return SALT_TFAIL(-1);
    if (required_meta_floats * sizeof(float) > (size_t)[_op_meta length]) {
        resized_meta = [_dev newBufferWithLength:
            required_meta_floats * sizeof(float)
            options:MTLResourceStorageModeShared];
        if (!resized_meta) return SALT_TFAIL(-1);
        [_op_meta release];
        _op_meta = resized_meta;
    }
    /* The selected task metadata belongs to startup, just like _op_meta.
     * Its old fixed B128 extent must not cap the independent PREFILL phase. */
    if (!_selected_task_map || required_task_jobs > UINT32_MAX / 2u ||
        required_task_jobs > SIZE_MAX / (3u * sizeof(MetalSelectedTask)))
        return SALT_TFAIL(-1);
    size_t task_bytes = required_task_jobs * 3u * sizeof(MetalSelectedTask);
    if (task_bytes > (size_t)[_selected_task_map length]) {
        id<MTLBuffer> resized_tasks = [_dev newBufferWithLength:task_bytes
            options:MTLResourceStorageModeShared];
        if (!resized_tasks) return SALT_TFAIL(-1);
        [_selected_task_map release];
        _selected_task_map = resized_tasks;
    }
    memset(state, 0, sizeof *state);
    memset(command, 0, sizeof(MetalTextProgramCommand));
    if (salt_gpu_shared_buffer_alloc(&state->canonical,
            program->layout.total_bytes) != 0)
        return SALT_TFAIL(-1);
    state->dependency_fence = [_dev newFence];
    if (!state->dependency_fence) goto fail;
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
    /* The fixed dense trunk is lifecycle-resident; only experts rotate.
     * Consume the EXISTING component-view book, never the source-shard maps
     * or expert arena. No payload allocation, copying, or expert preload. */
    if (@available(macOS 15.0, *)) {
        if (!_rotation_trunk_residency) {
            MTLResidencySetDescriptor *rd = [[MTLResidencySetDescriptor alloc] init];
            rd.initialCapacity = (NSUInteger)_tslot_n * 3u;
            rd.label = @"SALT canonical fixed trunk";
            NSError *error = nil;
            _rotation_trunk_residency = [_dev newResidencySetWithDescriptor:rd error:&error];
            [rd release];
            if (!_rotation_trunk_residency || !_cfg_mem_limit_bytes ||
                _cfg_mem_limit_bytes <= [_rotation_arena_backing length]) goto fail;
            uint64_t budget = _cfg_mem_limit_bytes - [_rotation_arena_backing length];
            for (int ti = 0; ti < _tslot_n; ti++) {
                const TensorSlot *t = &_tslot[ti];
                if (t->resource.kind != SALT_GPU_RESOURCE_TRUNK) continue;
                id<MTLBuffer> parts[3] = {t->value_map, t->scale_map, t->bias_map};
                for (int pi = 0; pi < 3; pi++) {
                    id<MTLBuffer> part = parts[pi];
                    if (!part || [_rotation_trunk_residency containsAllocation:part]) continue;
                    uint64_t bytes = [part length];
                    /* Check lengths BEFORE requesting residency: a whole-source
                     * binding can never cause a full expert-pool materialization. */
                    if (bytes > budget - _rotation_trunk_bytes) goto fail;
                    _rotation_trunk_bytes += bytes;
                    [_rotation_trunk_residency addAllocation:part];
                    _rotation_trunk_members++;
                }
            }
            if (!_rotation_trunk_members) goto fail;
            [_rotation_trunk_residency commit];
            [_rotation_trunk_residency requestResidency];
            [_queue addResidencySet:_rotation_trunk_residency];
        }
    } else goto fail;
    _manifest_gpu_window_count = 0u;
    _manifest_gpu_window_dropped = 0u;
    if (metal_text_arrival_prepare(program->layer_count) != 0) goto fail;
    state->magic = METAL_TEXT_MAGIC;
    state->ready = 1u;
    *canonical_out = &state->canonical;
        {
            SaltMemSnapshot _pe;
            if (salt_mem_snapshot(&_pe) == 0 && _pe.resident_b > 0) {
                salt_trace_register(186, 1);
                salt_trace_value(186, (int)((uint64_t)_pe.resident_b / 1000000u));
            }
        }
    return 0;
fail:
    metal_text_arrival_release();
    metal_release(state->dependency_fence);
    (void)salt_gpu_shared_buffer_free(&state->canonical);
    memset(state, 0, sizeof *state);
    return SALT_TFAIL(-1);
}

/* Resolve the fixed GPU counter seats only at the manifest emission point.
 * Metal GPU counter ticks are calibrated against the host CLOCK_MONOTONIC
 * timeline with one paired CPU/GPU sample. No hot-path readback or allocation. */
static uint64_t metal_text_counter_value(NSData **resolved, uint64_t sample) {
    uint32_t bi = (uint32_t)(sample / METAL_ARRIVAL_SAMPLES_PER_BUFFER);
    NSUInteger index = (NSUInteger)(sample % METAL_ARRIVAL_SAMPLES_PER_BUFFER);
    if (bi >= _arrival_buffer_count || !resolved[bi] ||
        [resolved[bi] length] < (index + 1u) * sizeof(MTLCounterResultTimestamp))
        return 0;
    const MTLCounterResultTimestamp *samples =
        (const MTLCounterResultTimestamp *)[resolved[bi] bytes];
    return samples[index].timestamp;
}
static uint64_t metal_text_counter_host_ns(uint64_t raw, uint64_t gpu,
        uint64_t before, uint64_t after) {
    if (!raw || !gpu || after < before) return 0;
    int64_t offset = (int64_t)(before / 2u + after / 2u) - (int64_t)gpu;
    int64_t aligned = (int64_t)raw + offset;
    return aligned > 0 ? (uint64_t)aligned : 0;
}
static void metal_text_arrival_dump(FILE *mf) {
    uint32_t phases = salt_lr_timeline_phase_count();
    uint64_t stride = metal_text_counter_stride();
    uint64_t used = phases > 1u ? (uint64_t)(phases - 1u) * stride : 0u;
    uint64_t cal_cpu = 0, cal_gpu = 0;
    uint64_t host_before = 0, host_after = 0, uncertainty = 0;
    uint32_t observed = 0, recorded = 0;
    NSData *resolved[METAL_ARRIVAL_MAX_BUFFERS] = { nil };
    if (!mf) return;
    @autoreleasepool {
        if (_arrival_available && used) {
            /* Post-operation only: a cold calibration call can be longer than
             * the short shared-FFN encoder being observed. Keep the tightest
             * measured bracket; never manufacture a zero uncertainty. */
            uint64_t best_span = UINT64_MAX;
            for (unsigned attempt = 0; attempt < 8u; attempt++) {
                uint64_t cpu = 0, gpu = 0;
                uint64_t before = (uint64_t)_mt_probe_ns();
                [_dev sampleTimestamps:&cpu gpuTimestamp:&gpu];
                uint64_t after = (uint64_t)_mt_probe_ns();
                if (after >= before && after - before < best_span) {
                    best_span = after - before;
                    host_before = before; host_after = after;
                    cal_cpu = cpu; cal_gpu = gpu;
                }
            }
            uncertainty = (host_after - host_before + 1u) / 2u;
            /* The host clock is quantized: an equal before/after reading does
             * not establish zero uncertainty. Include its reported resolution. */
            struct timespec resolution;
            if (clock_getres(CLOCK_MONOTONIC, &resolution) == 0 &&
                    (resolution.tv_sec > 0 || resolution.tv_nsec > 0))
                uncertainty += (uint64_t)resolution.tv_sec * 1000000000u +
                    (uint64_t)resolution.tv_nsec;
            else cal_gpu = 0; /* Host alignment explicitly unavailable. */
            for (uint32_t i = 0; i < _arrival_buffer_count; i++) {
                uint64_t first = (uint64_t)i * METAL_ARRIVAL_SAMPLES_PER_BUFFER;
                if (first >= used) break;
                NSUInteger count = (NSUInteger)(used - first);
                if (count > [_arrival_samples[i] sampleCount])
                    count = [_arrival_samples[i] sampleCount];
                resolved[i] = [_arrival_samples[i]
                    resolveCounterRange:NSMakeRange(0, count)];
            }
        }
        fprintf(mf, "# A before LOAD_DONE receive: previous layer finish encoder-end. Initial layer has no preceding encoder sample; no marker dispatch is added.\n");
        fprintf(mf, "# A enabled=%u cpu_tick=%llu gpu_tick=%llu host_before_ns=%llu host_after_ns=%llu uncertainty_ns=%llu drift=unmeasured\n",
            _arrival_available, (unsigned long long)cal_cpu,
            (unsigned long long)cal_gpu, (unsigned long long)host_before,
            (unsigned long long)host_after, (unsigned long long)uncertainty);
        salt_ledger_emit(NULL, "# G capture phase_capacity=%u buffers=%u max_buffers=%u samples_per_buffer=%u; phases beyond capacity retain invalid records, never guessed timestamps\n",
            _arrival_phase_capacity, _arrival_buffer_count,
            METAL_ARRIVAL_MAX_BUFFERS, METAL_ARRIVAL_SAMPLES_PER_BUFFER);
        salt_ledger_emit(NULL, "# G one row per PHYSICAL encoder; unique init-owned timestamp pair. Spans include stalls/barriers, NOT pure arithmetic; negative gap means overlap.\n");
        salt_ledger_emit(NULL, "# G finish belongs to its completed layer, prefix to its current layer, expert to its current layer, tail to the final head. Host calibration validity is separate from raw GPU validity.\n");
        salt_ledger_emit(NULL, "# G ordinary model-token DECODE contributes its retained selected-expert encoder per layer; prefix/finish/tail belong to compiled program phases only.\n");
        salt_ledger_emit(NULL, "# G phase layer stage gpu_begin_tick gpu_end_tick begin_host_ns end_host_ns span_ms gap_before_ms encoder_count valid encoder_ordinal stage_encoder_count clock_valid clock_uncertainty_ns\n");
        for (uint32_t phase = 1u; phase < phases; phase++) {
            SaltRotationTimelinePhase ph;
            uint64_t previous = 0u;
            if (!salt_lr_timeline_phase(phase, &ph)) continue;
            for (uint32_t segment = 0; segment <= 2u * _arrival_layer_count; segment++) {
                uint32_t count = _arrival_encoder_uses[phase - 1u][segment];
                uint32_t bounded = count < METAL_ARRIVAL_ENCODERS_PER_STAGE
                    ? count : METAL_ARRIVAL_ENCODERS_PER_STAGE;
                observed += count;
                for (uint32_t ordinal = 0; ordinal < bounded; ordinal++) {
                    uint64_t sample = metal_text_counter_index(phase - 1u, segment, ordinal);
                    uint64_t lo = metal_text_counter_value(resolved, sample);
                    uint64_t hi = metal_text_counter_value(resolved, sample + 1u);
                    uint64_t begin = metal_text_counter_host_ns(lo, cal_gpu, host_before, host_after);
                    uint64_t end = metal_text_counter_host_ns(hi, cal_gpu, host_before, host_after);
                    uint32_t layer = segment / 2u;
                    const char *stage;
                    if (segment & 1u) stage = "expert";
                    else if (segment > 0u && ordinal == 0u) {
                        stage = "finish"; layer--;
                    } else if (segment == 2u * _arrival_layer_count) {
                        stage = "tail"; layer = UINT32_MAX;
                    } else stage = "prefix";
                    int valid = _arrival_available && lo && hi >= lo && hi != UINT64_MAX;
                    int clock_valid = valid && ph.submit_ns && ph.complete_ns &&
                        begin && begin + uncertainty >= ph.submit_ns && end >= begin &&
                        end <= ph.complete_ns + uncertainty;
                    salt_ledger_emit(NULL, "G %u %u %s %llu %llu %llu %llu %.6f %.6f 1 %d %u %u %d %llu\n",
                        phase, layer, stage, (unsigned long long)lo, (unsigned long long)hi,
                        (unsigned long long)begin, (unsigned long long)end,
                        valid ? (double)(hi - lo) * 1e-6 : 0.0,
                        valid && previous ? (double)((int64_t)lo - (int64_t)previous) * 1e-6 : 0.0,
                        valid, ordinal, count, clock_valid, (unsigned long long)uncertainty);
                    previous = valid ? hi : 0u;
                    recorded++;
                }
            }
        }
        {
            uint32_t previous_phase = UINT32_MAX;
            uint32_t trunk_count = 0u, expert_count = 0u;
            uint64_t previous_end = 0u;
            salt_ledger_emit(NULL,
                "# Y phase ordinal layer stage commit_ns gpu_begin_host_ns gpu_end_host_ns wait_return_ns experts rows valid\n");
            for (uint32_t i = 0; i < _manifest_gpu_window_count; i++) {
                const MetalManifestGpuWindow *window =
                    &_manifest_gpu_windows[i];
                uint64_t begin_host = metal_text_counter_host_ns(
                    window->begin_ns, cal_gpu, host_before, host_after);
                uint64_t end_host = metal_text_counter_host_ns(
                    window->end_ns, cal_gpu, host_before, host_after);
                int clock_valid = begin_host && end_host >= begin_host;
                double span_ms = (double)(window->end_ns -
                    window->begin_ns) * 1e-6;
                double gap_ms = previous_phase == window->phase &&
                        window->begin_ns >= previous_end
                    ? (double)(window->begin_ns - previous_end) * 1e-6 : 0.0;
                const char *stage = window->stage == METAL_MANIFEST_GPU_EXPERT
                    ? "expert" : "trunk";
                if (window->stage == METAL_MANIFEST_GPU_EXPERT) expert_count++;
                else trunk_count++;
                salt_ledger_emit(NULL,
                    "G %u %u %s %llu %llu %llu %llu %.6f %.6f 1 1 0 1 %d %llu\n",
                    window->phase, window->layer, stage,
                    (unsigned long long)window->begin_ns,
                    (unsigned long long)window->end_ns,
                    (unsigned long long)begin_host,
                    (unsigned long long)end_host,
                    span_ms, gap_ms, clock_valid,
                    (unsigned long long)uncertainty);
                {
                    int sequence_valid = clock_valid &&
                        window->commit_ns <= begin_host + uncertainty &&
                        end_host <= window->wait_return_ns + uncertainty;
                    salt_ledger_emit(NULL,
                        "Y %u %u %u %s %llu %llu %llu %llu %u %llu %d\n",
                        window->phase, i, window->layer, stage,
                        (unsigned long long)window->commit_ns,
                        (unsigned long long)begin_host,
                        (unsigned long long)end_host,
                        (unsigned long long)window->wait_return_ns,
                        window->experts, (unsigned long long)window->rows,
                        sequence_valid);
                }
                previous_phase = window->phase;
                previous_end = window->end_ns;
            }
            salt_ledger_emit(NULL,
                "# G ordinary_command_encoders recorded=%u trunk=%u expert=%u dropped=%u source=MTLCommandBuffer.GPUStartTime/GPUEndTime raw_timebase=system-mach-time host_columns=post-operation-calibrated-CLOCK_MONOTONIC no-extra-encoder-or-submission\n",
                _manifest_gpu_window_count, trunk_count, expert_count,
                _manifest_gpu_window_dropped);
            salt_ledger_emit(NULL,
                "# shared_event_wait encoded=%llu waited=%llu harvested=%llu forced_terminal_waits=%llu pending=%u errors=%d\n",
                (unsigned long long)_command_done_value,
                (unsigned long long)_command_done_waited,
                (unsigned long long)_command_done_harvested,
                (unsigned long long)_command_done_forced,
                _command_event_pending_count, _command_event_error);
        }
        salt_ledger_emit(NULL, "# G capacity encoders_per_stage=%u observed=%u recorded=%u dropped=%u\n",
            METAL_ARRIVAL_ENCODERS_PER_STAGE, observed, recorded, observed - recorded);
        fprintf(mf, "# A phase layer gpu_tick arrival_host_ns ready_begin_ns ready_end_ns wait_min_ms wait_max_ms valid reason\n");
        salt_ledger_emit(NULL, "# Q same-GPU-clock gap: previous layer finish to next prefix; PREFILL includes DONE/READY signaling, resident DECODE has no such events. NOT isolated wait or arithmetic. Initial receive unobserved.\n");
        salt_ledger_emit(NULL, "# Q phase layer completed_layer gpu_before_tick gpu_after_tick gap_ms before_host_ns after_host_ns clock_valid valid\n");
        for (uint32_t phase = 1u; phase < phases; phase++) {
            SaltRotationTimelinePhase ph;
            if (!salt_lr_timeline_phase(phase, &ph)) continue;
            for (uint32_t layer = 0; layer < _arrival_layer_count; layer++) {
                SaltRotationTimelineRow h = {0};
                uint64_t raw = 0, after = 0;
                uint32_t count = _arrival_encoder_uses[phase - 1u][2u * layer];
                if (layer > 0u && count >= 2u && count <= METAL_ARRIVAL_ENCODERS_PER_STAGE) {
                    raw = metal_text_counter_value(resolved,
                        metal_text_counter_index(phase - 1u, 2u * layer, 0u) + 1u);
                    after = metal_text_counter_value(resolved,
                        metal_text_counter_index(phase - 1u, 2u * layer, 1u));
                }
                uint64_t before_ns = metal_text_counter_host_ns(raw, cal_gpu, host_before, host_after);
                uint64_t after_ns = metal_text_counter_host_ns(after, cal_gpu, host_before, host_after);
                int raw_valid = raw && raw != UINT64_MAX && after >= raw && after != UINT64_MAX;
                int clock_valid = raw_valid && ph.submit_ns && ph.complete_ns && before_ns &&
                    before_ns + uncertainty >= ph.submit_ns && after_ns <= ph.complete_ns + uncertainty;
                int have_row = salt_lr_timeline_row(phase, layer, &h);
                int valid = clock_valid && have_row &&
                    h.ready_begin_ns && h.ready_end_ns >= h.ready_begin_ns;
                uint64_t latest_before = before_ns + uncertainty;
                double minimum = valid && h.ready_begin_ns > latest_before
                    ? (double)(h.ready_begin_ns - latest_before) * 1e-6 : 0.0;
                double maximum = valid && h.ready_end_ns + uncertainty > before_ns
                    ? (double)(h.ready_end_ns + uncertainty - before_ns) * 1e-6 : 0.0;
                fprintf(mf, "A %u %u %llu %llu %llu %llu %.6f %.6f %d %s\n",
                    phase, layer, (unsigned long long)raw, (unsigned long long)before_ns,
                    (unsigned long long)h.ready_begin_ns, (unsigned long long)h.ready_end_ns,
                    minimum, maximum, valid && ph.kind != 2u,
                    ph.kind == 2u ? "resident_decode_no_READY_event" : layer == 0u ? "initial_unobserved" : valid ? "ok" : "invalid");
                if (layer > 0u)
                    salt_ledger_emit(NULL, "Q %u %u %u %llu %llu %.6f %llu %llu %d %d\n",
                        phase, layer, layer - 1u, (unsigned long long)raw,
                        (unsigned long long)after,
                        raw_valid ? (double)(after - raw) * 1e-6 : 0.0,
                        (unsigned long long)before_ns, (unsigned long long)after_ns,
                        clock_valid, raw_valid);
            }
        }
    }
}

/* The native-process cleanup is the sole manifest emission point. */
int salt_gpu_text_dump_ledger(void) {
    /* THE DUMP HAPPENS ONCE.  It has more than one call site on purpose - a failing run must still emit
     * - but only the first one in a run writes the record; the rest return. */
    static int _dumped;
    (void)metal_command_event_flush();
    if (_dumped) return 0;
    _dumped = 1;
    if (!salt_lr_manifest_emit()) return 0;
    int n = _text_probe_n;
    double sw = 0.0, sg = 0.0, sc = 0.0;
    uint32_t dec = 0, pre = 0;
    /* THE TRACE, EMITTED WITH THE MANIFEST.  It is in memory already; this only formats it, and it is
     * written BEFORE the empty check so a run that failed before its first commit still gets one. */
    for (int i = 0; i < n; i++) {
        if (_text_probe[i].asynchronous) continue;
        sw += _text_probe[i].wall; sg += _text_probe[i].gpu;
        if (_text_probe[i].rows <= 100u) dec++; else pre++;
    }
    for (int i = 0; i < n; i++)
        if (!_text_probe[i].asynchronous) sc += _text_probe[i].commit;
    /* THE BOOK, EMITTED WITH ITS SHAPE AND ITS TIMERS, at the designated dump point so the
     * artifact carries a real request's numbers.  Three record families: the storage badges
     * (S), the boundary/function calls (N), and the per-commit timers (T). */
    {
        /* THE MANIFEST IS A CORE FEATURE, NOT A DIAGNOSTIC.  It is the engine's own record
         * of the shape it ran on (storage badges), where the wall went (node calls) and
         * what each commit cost (timers) - emitted at the designated dump point on every
         * request, unconditionally.  SALT_TEXT_MANIFEST_DUMP only chooses WHERE; absence
         * no longer means "no record".  A run that produces no record is a run that cannot
         * be audited, and this arc lost two measurements to exactly that. */
        const char *dump = getenv("SALT_TEXT_MANIFEST_DUMP");
        if (!dump || !dump[0]) {
            /* Timestamped by default: a record's NAME must say when it was made, or a
             * directory of them is undateable.  UTC, second resolution, with the pid so
             * concurrent engines cannot collide. */
            static char _mf_name[96];
            time_t _mf_t = time(NULL);
            struct tm _mf_tm;
            gmtime_r(&_mf_t, &_mf_tm);
            strftime(_mf_name, sizeof _mf_name, "salt-manifest-%Y%m%dT%H%M%SZ", &_mf_tm);
            snprintf(_mf_name + strlen(_mf_name),
                     sizeof _mf_name - strlen(_mf_name), "-pid%d.txt", (int)getpid());
            dump = _mf_name;
        }
        /* THE RECORD IS WRITTEN EVEN WHEN THE BUFFER IS NOT THERE.  The buffer is built at program
         * prepare, so an initialization failure has none - and that is exactly the run that most needs
         * a record.  The registers and the config do not need the buffer; the storage rows do, and they
         * are skipped when it is absent. */
        {
            /* ONE WRITER, ONE EMISSION (88f12f2): the init-time writer is gone, so this is the only
             * writer and "w" is correct. */
            FILE *mf = fopen(dump, "w");
            if (mf) {
            /* THE LAST FAILURE ORIGIN, in the record.  A bare -1 cannot be told from any
             * other -1 in the tree; this names the line that produced the one that stopped
             * the run.  The name field carries the file. */
            mt_node_tick(salt_rot_fail_file(), "fail",
                         (uint32_t)salt_rot_fail_line(), (uint32_t)salt_rot_fail_status());
                /* THE ROTATION'S OWN REGISTERS, recorded here and emitted once, with the rest. */
                {
                    extern uint32_t salt_rot_inference_bind_fail(int);
                    extern uint32_t salt_rot_inference_loads(void);
                    extern uint32_t salt_rot_inference_served(void);
                    salt_trace_register(40, 1); salt_trace_value(40, (int)salt_rot_inference_bind_fail(1));
                    salt_trace_register(41, 1); salt_trace_value(41, (int)salt_rot_inference_bind_fail(2));
                    salt_trace_register(42, 1); salt_trace_value(42, (int)salt_rot_inference_loads());
                    salt_trace_register(43, 1); salt_trace_value(43, (int)salt_rot_inference_served());
                    {
                        /* WHAT THE DEVICE TABLE HOLDS, read back at the dump: the first entries and how
                         * many are filled.  A table of -1 is why a kernel resolves to a miss. */
                        int _filled = 0, _first = -1, _last = -1, _i;
                        if (_selected_logical_slot_host && _selected_logical_capacity > 0) {
                            for (_i = 0; _i < _selected_logical_capacity; _i++)
                                if (_selected_logical_slot_host[_i] != -1) {
                                    if (_first < 0) _first = _i;
                                    _last = _i;
                                    _filled++;
                                }
                        }
                        salt_trace_register(102, 1); salt_trace_value(102, _filled);
                        salt_trace_register(103, 1); salt_trace_value(103, _first);
                        salt_trace_register(104, 1); salt_trace_value(104, _last);
                        salt_trace_register(105, 1);
                        salt_trace_value(105, _selected_logical_capacity);
                    }
                    {
                        extern uint32_t salt_gpu_selected_bind_refuse(int);
                        for (int _r = 1; _r <= 4; _r++) {
                            salt_trace_register(43 + _r, 1);
                            salt_trace_value(43 + _r, (int)salt_gpu_selected_bind_refuse(_r));
                        }
                    }
                }
                /* THE REGISTERS, IN THE MANIFEST - the whole operation, once, at the end. */
                salt_trace_dump(_mt_trace_emit, mf);
                fprintf(mf, "# ==================================================================\n");
                fprintf(mf, "# SALT text manifest - storage badges, node calls, and timers\n");
                fprintf(mf, "# attention_product_tile configured=%u decode_tiled=%llu "
                    "decode_control=%llu prefill_control=%llu positions=64 columns=64\n",
                    salt_lr_decode_attention_product_tile(),
                    (unsigned long long)_attention_decode_product_calls,
                    (unsigned long long)_attention_decode_control_calls,
                    (unsigned long long)_attention_prefill_control_calls);
                salt_ledger_emit(NULL, "# handoff DONE_observer=engine-condition channel=separate-DONE-READY receive=block-boundary reads=%llu releases=%llu yields=%llu armed=%llu notifications=%llu\n",
                    (unsigned long long)_rotation_done_value_reads,
                    (unsigned long long)_rotation_done_value_hits,
                    (unsigned long long)_rotation_done_value_yields,
                    (unsigned long long)_rotation_done_armed,
                    (unsigned long long)__atomic_load_n(&_rotation_done_notifications, __ATOMIC_RELAXED));
                salt_ledger_emit(NULL, "# handoff v=phase_base+layer+1; independent READY and DONE timelines; H keeps semantic layer identities\n");
                salt_ledger_emit(NULL, "# rotation_config prefill_layer_block=%u decode_mode=layer-local-b1/global-hot-floor-wide-target decode_dense_chain=%u manifest_emit=%u banks=%u source=rotation/engine.config resolved_at=initialization\n",
                    salt_lr_prefill_layers(), salt_lr_decode_dense_chain(),
                    salt_lr_manifest_emit(), (unsigned)SALT_LR_BANK_COUNT);
                salt_ledger_emit(NULL, "# decode_residency total_slots=%u min_layer_slots=%u max_layer_slots=%u insertion=bottom hit=move-front records=%u dropped=%u\n",
                    salt_lr_decode_seats(), salt_lr_decode_min_layer_seats(),
                    salt_lr_decode_max_layer_seats(), salt_lr_decode_record_count(),
                    salt_lr_decode_record_dropped());
                salt_ledger_emit(NULL, "# C step layer requests hits misses evictions resident bytes\n");
                for (uint32_t ci = 0; ci < salt_lr_decode_record_count(); ci++) {
                    SaltDecodeExpertRecord cr;
                    if (salt_lr_decode_record(ci, &cr))
                        salt_ledger_emit(NULL, "C %u %u %u %u %u %u %u %llu\n",
                            cr.step, cr.layer, cr.requests, cr.hits, cr.misses, cr.evictions,
                            cr.resident, (unsigned long long)cr.bytes);
                }
                salt_ledger_emit(NULL,
                    "# Z exact first-DECODE-step read replay plan; startup-resolved file offsets and destination seats, in request order; records=%u dropped=%u\n",
                    salt_lr_decode_replay_plan_count(),
                    salt_lr_decode_replay_plan_dropped());
                salt_ledger_emit(NULL,
                    "# Z step layer ordinal expert seat file_offset bytes\n");
                for (uint32_t zi = 0; zi < salt_lr_decode_replay_plan_count(); zi++) {
                    uint32_t step, layer, ordinal, expert, seat;
                    uint64_t file_offset, bytes;
                    if (salt_lr_decode_replay_plan(zi, &step, &layer, &ordinal,
                            &expert, &seat, &file_offset, &bytes))
                        salt_ledger_emit(NULL,
                            "Z %u %u %u %u %u %llu %llu\n",
                            step, layer, ordinal, expert, seat,
                            (unsigned long long)file_offset,
                            (unsigned long long)bytes);
                }
                salt_ledger_emit(NULL, "# K layer rank expert slot; rank 0 is front\n");
                for (uint32_t l = 0; l < _text_manifest_rows / _text_manifest_experts; l++)
                    for (unsigned rank = 0; rank < salt_lr_decode_layer_seats(l); rank++) {
                        int expert, slot;
                        if (salt_lr_decode_layer_entry(l, rank, &expert, &slot))
                            salt_ledger_emit(NULL, "K %u %u %d %d\n", l, rank, expert, slot);
                    }
                salt_ledger_emit(NULL, "# B phase rows block_layers banks ready_waits done_signals\n");
                for (uint32_t pi = 1; pi < salt_lr_timeline_phase_count(); pi++) {
                    SaltRotationTimelinePhase bp;
                    if (salt_lr_timeline_phase(pi, &bp))
                        salt_ledger_emit(NULL, "B %u %u %u %u %u %u\n", pi, bp.rows,
                            bp.block_layers, (unsigned)SALT_LR_BANK_COUNT,
                            bp.ready_waits, bp.done_signals);
                }
                salt_ledger_emit(NULL, "# fixed_trunk_residency members=%llu view_bytes=%llu copied_payload_bytes=0 selected_experts=0\n",
                    (unsigned long long)_rotation_trunk_members,
                    (unsigned long long)_rotation_trunk_bytes);
                salt_ledger_emit(NULL, "# bank_resources views=%u payload_allocations=1 payload_bytes=%llu bank_bytes=%llu copy_bytes=0 lookahead=%u\n",
                    (unsigned)SALT_LR_BANK_COUNT,
                    (unsigned long long)[_rotation_arena_backing length],
                    (unsigned long long)_rotation_bank_bytes,
                    (unsigned)(SALT_LR_BANK_COUNT - 1));
                fprintf(mf, "# RUN CONFIG - what this record was produced under\n");
                fprintf(mf, "# ==================================================================\n");
                {
                    /* THE RESOLVED CONFIG, not a fresh getenv.  These are the values the
                     * engine captured at initialization - the fixed input the operation was
                     * built on - so the record states what actually ran, and stays true even
                     * if the environment is not what it was. */
                    int ki;
                    for (ki = 0; ki < _mt_cfg_n; ki++)
                        fprintf(mf, "# cfg %-38s %s\n", _mt_cfg_key[ki],
                                _mt_cfg_val[ki]);
                    fprintf(mf, "# cfg %-38s %u\n", "manifest_rows",
                        _text_manifest_rows);
                    fprintf(mf, "# cfg %-38s %u\n", "manifest_fields",
                        (unsigned)METAL_TEXT_MANIFEST_FIELDS);
                }
                fprintf(mf, "# ==================================================================\n");
                fprintf(mf, "# grouped_gemm prefill_encoded_layers=%u routed_jobs=%llu "
                            "decode_path=rows4_reuse decode_layers=%u decode_dispatches=%llu "
                            "rows_per_simd=4\n",
                    _text_grouped_prefill_encoded_layers,
                    (unsigned long long)_text_grouped_prefill_encoded_jobs,
                    _text_rows4_decode_encoded_layers,
                    (unsigned long long)_text_rows4_decode_dispatches);
                fprintf(mf, "# q4_simd trunk_dispatches=%llu selected_dispatches=%llu "
                            "non_simd_fallback_dispatches=%llu trunk_lanes_per_simd=32 "
                            "trunk_outputs_per_simd=32 selected_lanes_per_simd=32 "
                            "selected_outputs_per_simd=32\n",
                    (unsigned long long)_q4_simd_trunk_dispatches,
                    (unsigned long long)_q4_simd_selected_dispatches,
                    (unsigned long long)_q4_non_simd_fallback_dispatches);
                fprintf(mf, "# probe n=%d decode=%u prefill=%u wall_ms=%.3f gpu_ms=%.3f "
                            "overhead_ms=%.3f commit_call_ms=%.3f wait_nongpu_ms=%.3f\n",
                        n, dec, pre, sw, sg, sw - sg, sc, (sw - sg) - sc);
                fprintf(mf, "# S: layer expert logical_id slot <gate_v gate_s gate_b up_v up_s up_b down_v down_s down_b>\n");
                for (uint32_t li = 0; li < _text_manifest_rows; li++) {
                    const uint64_t *row = &_text_manifest_host[li * METAL_TEXT_MANIFEST_FIELDS];
                    fprintf(mf, "S %u %u %u %d", li / _text_manifest_experts,
                        li % _text_manifest_experts, li,
                        (_selected_logical_slot_host &&
                         (uint64_t)li < (uint64_t)_selected_logical_capacity)
                            ? (int)_selected_logical_slot_host[li] : -1);
                    for (uint32_t f = 0; f < METAL_TEXT_MANIFEST_FIELDS; f++)
                        fprintf(mf, " %llu", (unsigned long long)row[f]);
                    fprintf(mf, "\n");
                }
                fprintf(mf, "# N: boundary/function call, wall SINCE the previous boundary\n");
                fprintf(mf, "# N name layer kind wall_since_prev_ms\n");
                for (int ni = 0; ni < _mt_node_n; ni++)
                    fprintf(mf, "N %s <- %s %u %u %.4f\n", _mt_node[ni].name,
                        _mt_node[ni].caller ? _mt_node[ni].caller : "?",
                        _mt_node[ni].layer, _mt_node[ni].kind, _mt_node[ni].ms);
                /* the ring accumulates for the whole run - the dump does not reset it */
                fprintf(mf, "# R source_policy=%s\n", salt_lr_source_policy());
                fprintf(mf, "# R buckets: 0=unstreamed 1=first_stream 2=later_streams; sums across copies\n");
                fprintf(mf, "# R bucket layer copies bytes done_wait_ms copy_span_ms worker_copy_sum_ms copy_join_ms source_discard_ms ready_callback_ms discard_failures worker_source_prepare_sum_ms prepare_calls prepare_failures load_span_ms critical_start_ms critical_prepare_ms critical_memcpy_ms\n");
                {
                    SaltRotationTiming r;
                    for (uint32_t ri = 0; salt_lr_timing(ri, &r); ri++) {
                        if (!r.copies) continue;
                        fprintf(mf, "R %u %u %u %llu %.6f %.6f %.6f %.6f %.6f %.6f %u %.6f %u %u %.6f %.6f %.6f %.6f\n",
                            r.bucket, r.layer, r.copies, (unsigned long long)r.bytes,
                            (double)r.done_wait_ns * 1e-6, (double)r.copy_span_ns * 1e-6,
                            (double)r.worker_copy_sum_ns * 1e-6, (double)r.copy_join_ns * 1e-6,
                            (double)r.source_discard_ns * 1e-6,
                            (double)r.ready_callback_ns * 1e-6, r.discard_failures,
                            (double)r.source_prepare_ns * 1e-6,
                            r.prepare_calls, r.prepare_failures,
                            (double)r.load_span_ns * 1e-6,
                            (double)r.critical_start_ns * 1e-6,
                            (double)r.critical_prepare_ns * 1e-6,
                            (double)r.critical_copy_ns * 1e-6);
                    }
                }
                fprintf(mf, "# H host CLOCK_MONOTONIC nanoseconds; 0=not observed. GPU event thresholds are host-observed, not GPU execution timestamps.\n");
                fprintf(mf, "# P phase rows event_base submit_ns complete_ns kind\n");
                fprintf(mf, "# H phase layer bank primed copies critical_worker expected_done expected_ready wait_begin_ns done_seen_ns first_prepare_ns critical_prepare_ns critical_copy_ns first_copy_ns last_copy_ns joined_ns cleanup_begin_ns cleanup_end_ns ready_begin_ns ready_end_ns resident_at_ready_bytes peak_at_ready_bytes done_layer wait_worker\n");
                fprintf(mf, "# P kind: 0=startup 1=full-layer-PREFILL 2=resident-DECODE; window is first GPU submission to final completion\n");
                fprintf(mf, "# H resident-DECODE: copies is expert misses; READY is CPU cache-tag publication, not a fabricated GPU event. Hit-only rows have zero payload and unavailable I/O counters.\n");
                fprintf(mf, "# H phases=%u layers=%u dropped=%u\n",
                    salt_lr_timeline_phase_count(), salt_lr_timeline_layer_count(),
                    salt_lr_timeline_dropped());
                for (uint32_t pi = 0; pi < salt_lr_timeline_phase_count(); pi++) {
                    SaltRotationTimelinePhase phase;
                    if (!salt_lr_timeline_phase(pi, &phase)) continue;
                    fprintf(mf, "P %u %u %llu %llu %llu %u\n", pi, phase.rows,
                        (unsigned long long)phase.event_base,
                        (unsigned long long)phase.submit_ns,
                        (unsigned long long)phase.complete_ns, phase.kind);
                    for (uint32_t li = 0; li < salt_lr_timeline_layer_count(); li++) {
                        SaltRotationTimelineRow h;
                        if (!salt_lr_timeline_row(pi, li, &h)) continue;
                        fprintf(mf, "H %u %u %u %u %u %u %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %d %u\n",
                            h.phase, h.layer, h.bank, h.primed, h.copies,
                            h.critical_worker,
                            (unsigned long long)h.expected_done,
                            (unsigned long long)h.expected_ready,
                            (unsigned long long)h.wait_begin_ns,
                            (unsigned long long)h.done_seen_ns,
                            (unsigned long long)h.first_prepare_ns,
                            (unsigned long long)h.critical_prepare_ns,
                            (unsigned long long)h.critical_copy_ns,
                            (unsigned long long)h.first_copy_ns,
                            (unsigned long long)h.last_copy_ns,
                            (unsigned long long)h.joined_ns,
                            (unsigned long long)h.cleanup_begin_ns,
                            (unsigned long long)h.cleanup_end_ns,
                            (unsigned long long)h.ready_begin_ns,
                            (unsigned long long)h.ready_end_ns,
                            (unsigned long long)h.resident_at_ready_bytes,
                            (unsigned long long)h.peak_at_ready_bytes,
                            h.done_layer, h.wait_worker);
                    }
                }
                fprintf(mf, "# V phase layer required observed event_read_begin_ns event_read_end_ns direct\n");
                fprintf(mf, "# E phase layer callback_enter_ns callback_lock_ns done_notify_begin_ns done_notify_end_ns permit_begin_ns permit_end_ns\n");
                fprintf(mf, "# U phase layer worker task_begin_ns lock_acquired_ns wait_begin_ns wait_end_ns gate_pass_ns read_begin_ns read_end_ns task_end_ns read_bytes waits read_calls read_interrupts short_reads finalizer read_status read_cpu_begin_ns read_cpu_end_ns read_cpu_valid\n");
                {
                    unsigned emitted_workers = 0, emitted_signals = 0;
                    for (uint32_t pi = 0; pi < salt_lr_timeline_phase_count(); pi++) {
                        for (uint32_t li = 0; li < salt_lr_timeline_layer_count(); li++) {
                            SaltRotationSignalTick e;
                            if (salt_lr_signal_tick(pi, li, &e)) {
                                fprintf(mf, "E %u %u %llu %llu %llu %llu %llu %llu\n", pi, li,
                                    (unsigned long long)e.callback_enter_ns,
                                    (unsigned long long)e.callback_lock_ns,
                                    (unsigned long long)e.done_notify_begin_ns,
                                    (unsigned long long)e.done_notify_end_ns,
                                    (unsigned long long)e.permit_begin_ns,
                                    (unsigned long long)e.permit_end_ns);
                                if (e.event_read_begin_ns)
                                    fprintf(mf, "V %u %u %llu %llu %llu %llu %u\n", pi, li,
                                        (unsigned long long)e.receive_required,
                                        (unsigned long long)e.receive_observed,
                                        (unsigned long long)e.event_read_begin_ns,
                                        (unsigned long long)e.event_read_end_ns, e.receive_direct);
                                emitted_signals++;
                            }
                            for (uint32_t wi = 0; wi < salt_lr_worker_width(); wi++) {
                                SaltRotationWorkerTick u;
                                if (!salt_lr_worker_tick(pi, li, wi, &u)) continue;
                                fprintf(mf, "U %u %u %u %llu %llu %llu %llu %llu %llu %llu %llu %llu %u %u %u %u %u %d %llu %llu %u\n",
                                    pi, li, wi,
                                    (unsigned long long)u.task_begin_ns,
                                    (unsigned long long)u.lock_acquired_ns,
                                    (unsigned long long)u.wait_begin_ns,
                                    (unsigned long long)u.wait_end_ns,
                                    (unsigned long long)u.gate_pass_ns,
                                    (unsigned long long)u.read_begin_ns,
                                    (unsigned long long)u.read_end_ns,
                                    (unsigned long long)u.task_end_ns,
                                    (unsigned long long)u.read_bytes,
                                    u.waits, u.read_calls, u.read_interrupts, u.short_reads,
                                    u.finalizer, u.read_status,
                                    (unsigned long long)u.read_cpu_begin_ns,
                                    (unsigned long long)u.read_cpu_end_ns, u.read_cpu_valid);
                                emitted_workers++;
                            }
                        }
                    }
                    fprintf(mf, "# U capacity=%u storage_bytes=%llu observed=%u worker_dropped=%llu signal_records=%u signal_dropped=%llu workers=%u\n",
                        SALT_LR_TICK_CAPACITY,
                        (unsigned long long)salt_lr_tick_storage_bytes(), emitted_workers,
                        (unsigned long long)salt_lr_worker_ticks_dropped(), emitted_signals,
                        (unsigned long long)salt_lr_signal_ticks_dropped(), salt_lr_worker_width());
                }
                fprintf(mf, "# L exact first-DECODE-step pread syscall boundaries; begin/end use host CLOCK_MONOTONIC. These timestamps are observational and do not identify device service.\n");
                fprintf(mf, "# L phase step layer ordinal worker attempt expert seat begin_ns end_ns file_offset requested_bytes returned_bytes errno\n");
                {
                    unsigned emitted_reads = 0;
                    for (uint32_t li = 0;
                            li < salt_lr_decode_read_call_capacity(); li++) {
                        SaltDecodeReadCallRecord call;
                        if (!salt_lr_decode_read_call(li, &call)) continue;
                        fprintf(mf,
                            "L %u %u %u %u %u %u %u %u %llu %llu %llu %llu %lld %d\n",
                            call.phase, call.step, call.layer, call.ordinal,
                            call.worker, call.attempt, call.expert, call.seat,
                            (unsigned long long)call.begin_ns,
                            (unsigned long long)call.end_ns,
                            (unsigned long long)call.file_offset,
                            (unsigned long long)call.requested_bytes,
                            (long long)call.returned_bytes, call.error);
                        emitted_reads++;
                    }
                    fprintf(mf, "# L capacity=%u observed=%u dropped=%u\n",
                        salt_lr_decode_read_call_capacity(), emitted_reads,
                        salt_lr_decode_read_call_dropped());
                }
                salt_ledger_emit(NULL, "# J process I/O overlapping each loader window, not per-file or per-worker attribution. copy_bytes is payload read into the shared arena, NOT disk writes. primed=1 reuses startup. Resident hit-only rows have no load or sampled I/O. sample_ns is observer overhead.\n");
                salt_ledger_emit(NULL, "# J phase layer primed begin_ns end_ns read_bytes read_valid write_bytes write_valid major_faults faults_valid sample_ns copy_bytes\n");
                for (uint32_t pi = 0; pi < salt_lr_timeline_phase_count(); pi++) {
                    for (uint32_t li = 0; li < salt_lr_timeline_layer_count(); li++) {
                        SaltRotationTimelineRow h;
                        if (!salt_lr_timeline_row(pi, li, &h)) continue;
                        salt_ledger_emit(NULL, "J %u %u %u %llu %llu %llu %u %llu %u %llu %u %llu %llu\n",
                            pi, li, h.primed,
                            (unsigned long long)h.io_begin_ns, (unsigned long long)h.io_end_ns,
                            (unsigned long long)h.io_read_bytes, !!(h.io_valid & 1u),
                            (unsigned long long)h.io_write_bytes, !!(h.io_valid & 2u),
                            (unsigned long long)h.io_major_faults, !!(h.io_valid & 4u),
                            (unsigned long long)h.io_sample_ns, (unsigned long long)h.copy_bytes);
                    }
                }
                metal_text_arrival_dump(mf);
                fprintf(mf, "# T: synchronous submission completions; O records asynchronous submissions separately; gpu_ms includes encoded event waits, not active kernel time\n");
                fprintf(mf, "# T i rows layer phase wall_ms gpu_ms commit_ms handoff_ms operation_phase operation_kind\n");
                fprintf(mf, "# decode_read_overlap configured=%u deferred_cells=%llu issued=%llu completed=%llu\n",
                    salt_lr_decode_read_overlap(), (unsigned long long)_decode_overlap_deferred_cells,
                    (unsigned long long)_decode_overlap_issued, (unsigned long long)_decode_overlap_completed);
                fprintf(mf, "# decode_demand armed=%llu received=%llu ready=%llu failed=%llu\n",
                    (unsigned long long)_decode_demand_armed,
                    (unsigned long long)_decode_demand_received,
                    (unsigned long long)_decode_ready_published,
                    (unsigned long long)_decode_demand_failed);
                fprintf(mf, "# O asynchronous shared-FFN submissions; probe n counts T+O, probe timing totals cover synchronous T only. observed_done is later harvest, NOT completion time or CPU wait. G rows carry actual encoder windows.\n");
                fprintf(mf, "# O phase layer submit_ns observed_done_ns gpu_ms commit_ms\n");
                for (int i = 0; i < n; i++) {
                    if (!_text_probe[i].asynchronous) continue;
                    fprintf(mf, "O %u %u %llu %llu %.6f %.6f\n",
                        _text_probe[i].operation_phase, _text_probe[i].layer,
                        (unsigned long long)_text_probe[i].async_submit_ns,
                        (unsigned long long)_text_probe[i].async_observed_done_ns,
                        _text_probe[i].gpu, _text_probe[i].commit);
                }
                for (int i = 0; i < n; i++)
                    if (!_text_probe[i].asynchronous)
                    fprintf(mf, "T %d %u %u %u %.3f %.3f %.3f %.3f %u %u\n", i,
                        _text_probe[i].rows, _text_probe[i].layer, _text_probe[i].phase,
                        _text_probe[i].wall, _text_probe[i].gpu, _text_probe[i].commit,
                        (_text_probe[i].wall - _text_probe[i].gpu) - _text_probe[i].commit,
                        _text_probe[i].operation_phase, _text_probe[i].operation_kind);
                fprintf(mf, "# completion_tail: CPU final-layer READY publication to phase finish; includes remaining device work and completion signaling\n");
                for (int i = 0; i < n; i++)
                    fprintf(mf, "# completion_tail index=%d rows=%u valid=%d ms=%.6f\n",
                        i, _text_probe[i].rows, _text_probe[i].completion_tail_valid,
                        _text_probe[i].completion_tail);
                fprintf(mf, "# I: process I/O during each submitted phase, not source-specific attribution\n");
                fprintf(mf, "# I index rows read_bytes read_valid major_faults faults_valid\n");
                for (int i = 0; i < n; i++) {
                    if (!_text_probe[i].phase_io_record) continue;
                    fprintf(mf, "I %d %u %llu %d %llu %d\n", (int)_text_probe[i].operation_phase - 1, _text_probe[i].rows,
                        (unsigned long long)_text_probe[i].read_bytes,
                        _text_probe[i].read_bytes_valid,
                        (unsigned long long)_text_probe[i].major_faults,
                        _text_probe[i].major_faults_valid);
                }
                salt_ledger_emit(NULL, "# D process disk I/O across submitted phase; ends BEFORE manifest emission. Counters are not source-specific. write_bytes excludes arena memcpy and includes any OS-attributed process disk writes.\n");
                salt_ledger_emit(NULL, "# D phase rows begin_ns end_ns read_bytes read_valid write_bytes write_valid major_faults faults_valid sample_ns\n");
                for (int i = 0; i < n; i++) {
                    if (!_text_probe[i].phase_io_record) continue;
                    salt_ledger_emit(NULL, "D %u %u %llu %llu %llu %d %llu %d %llu %d %llu\n",
                        _text_probe[i].operation_phase, _text_probe[i].rows,
                        (unsigned long long)_text_probe[i].io_begin_ns,
                        (unsigned long long)_text_probe[i].io_end_ns,
                        (unsigned long long)_text_probe[i].read_bytes, _text_probe[i].read_bytes_valid,
                        (unsigned long long)_text_probe[i].write_bytes, _text_probe[i].write_bytes_valid,
                        (unsigned long long)_text_probe[i].major_faults, _text_probe[i].major_faults_valid,
                        (unsigned long long)_text_probe[i].io_sample_ns);
                }
                /* The existing manifest writer is the ONLY consumer of the
                 * book. No stderr flush, atexit writer, or private formatter. */
                for (unsigned ri = 0; ri < salt_ledger_book.performance.head &&
                        ri < SALT_LEDGER_SLOTS; ri++)
                    _mt_trace_emit(salt_ledger_book.performance.ring[ri], mf);
                if (salt_ledger_book.performance.dropped)
                    _mt_trace_emit("# manifest_performance_overflow=1", mf);
                salt_ledger_book.performance.head = 0;
                fclose(mf);
            }
        }
    }
    _text_probe_n = 0;
    /* The book has already been consumed by the single manifest emitter. */
    return n;
}

static int metal_text_new_command(MetalTextProgramCommand *command) {
    if (!command || command->command || command->encoder) return SALT_TFAIL(-1);
    command->profile_command_contains_expert = 0u;
    command->encoder_nodes = 0u;
    @autoreleasepool {
        command->command = [[_queue commandBuffer] retain];
        /* THE STATUS WORD BELONGS TO THIS COMMAND BUFFER.  Every selected-expert kernel writes 2 into it for
         * a synthetic miss and nothing clears it, so a miss written by one buffer's kernels surfaced as -1
         * at a LATER boundary's metal_text_complete_command - which is why the driver stopped with no miss
         * in sight.  The exact-cell harness clears it per run (salt_gpu_exact_cells); the program path never
         * did.  Clear it here, where the buffer is taken. */
        if (_op_status && [_op_status contents] && !command->overlap_command)
            *(uint32_t *)[_op_status contents] = 0u;
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

static void metal_text_close_encoder(MetalTextProgramCommand *command) {
    if (!command || !command->encoder) return;
    [command->encoder endEncoding];
    [command->encoder release];
    command->encoder = nil;
    command->encoder_nodes = 0u;
}

/* Lazily open the command buffer's single compute encoder. Every node after
 * the first is ordered behind the previous one by a buffer-scope barrier,
 * which is the dependency the per-node encoder boundaries used to provide. */
static id<MTLComputeCommandEncoder> metal_text_encoder(
        MetalTextProgramCommand *command) {
    if (!command || !command->command) return nil;
    if (!command->encoder) {
        if (_arrival_available && command->timing_segment <= 2u * _arrival_layer_count) {
            uint64_t phase_index = command->rotation_base /
                ((uint64_t)_arrival_layer_count + 2u);
            uint32_t ordinal = phase_index < SALT_LR_TIMELINE_PHASES
                ? _arrival_encoder_uses[phase_index][command->timing_segment]++ :
                  METAL_ARRIVAL_ENCODERS_PER_STAGE;
            uint64_t sample_index = metal_text_counter_index((uint32_t)phase_index,
                command->timing_segment, ordinal);
            if (phase_index < SALT_LR_TIMELINE_PHASES &&
                phase_index < _arrival_phase_capacity &&
                ordinal < METAL_ARRIVAL_ENCODERS_PER_STAGE &&
                sample_index / METAL_ARRIVAL_SAMPLES_PER_BUFFER < _arrival_buffer_count) {
                uint32_t buffer_index =
                    (uint32_t)(sample_index / METAL_ARRIVAL_SAMPLES_PER_BUFFER);
                _arrival_pass.sampleBufferAttachments[0].sampleBuffer =
                    _arrival_samples[buffer_index];
                _arrival_pass.sampleBufferAttachments[0].startOfEncoderSampleIndex =
                    (NSUInteger)(sample_index % METAL_ARRIVAL_SAMPLES_PER_BUFFER);
                _arrival_pass.sampleBufferAttachments[0].endOfEncoderSampleIndex =
                    (NSUInteger)(sample_index % METAL_ARRIVAL_SAMPLES_PER_BUFFER) + 1u;
                command->encoder = [[command->command
                    computeCommandEncoderWithDescriptor:_arrival_pass] retain];
            }
        }
        if (!command->encoder)
            command->encoder = [[command->command computeCommandEncoder] retain];
        command->encoder_nodes = 0u;
        if (!command->encoder) return nil;
        if (command->dependency_fence_updated) {
            [command->encoder waitForFence:command->dependency_fence];
            command->dependency_fence_updated = 0u;
        }
    } else {
        [command->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    command->encoder_nodes++;
    return command->encoder;
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
        return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
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
            return SALT_TFAIL(-1);
        for (uint32_t index = 0; index < (uint32_t)attention->head_dim; index++) {
            meta[index] = metal_text_ref_value(&refs->q_norm, index);
            meta[attention->head_dim + index] =
                metal_text_ref_value(&refs->k_norm, index);
            if (!isfinite(meta[index]) ||
                !isfinite(meta[attention->head_dim + index]))
                return SALT_TFAIL(-1);
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
                    return SALT_TFAIL(-1);
            }
    }
    command->generation = generation;
    command->source_position = source_position;
    command->input_count = input_count;
    command->authoritative_output_rows = input_count;
    command->maximum_depth = maximum_depth;
    command->active_layer = UINT32_MAX;
    command->arrival_target_layer = 0u;
    command->timing_segment = 0u;
    command->dependency_fence = state->dependency_fence;
    command->dependency_fence_updated = 0u;
    command->phase = METAL_TEXT_RECORDING_PREFIX;
    {
        const char *profile = _cfg_target_waterfall;
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
        return SALT_TFAIL(-1);
    }
    _sel_args_cursor = 0u;
    if (_rotation_epoch > UINT64_MAX / 2u - state->layer_count - 2u)
        return SALT_TFAIL(-1);
    command->rotation_base = _rotation_epoch;
    _rotation_epoch += state->layer_count + 2u;
    command->decode_experts = salt_rotation_decode_phase();
    if (command->decode_experts) {
        if (salt_lr_decode_begin() != 0) return SALT_TFAIL(-1);
        salt_lr_timeline_open(command->rotation_base, 2u);
        salt_lr_timeline_block(command->rotation_base, 1u, 0u, 0u);
        _rotation_stream_error = 0;
        metal_text_arm_decode_demands(state, command);
        if (_rotation_stream_error) return SALT_TFAIL(-1);
        if (_text_probe_enabled) salt_lr_io_snapshot(&command->phase_io_begin);
        return 0;
    }
    salt_lr_decode_invalidate();
    for (uint32_t l = 0; l < state->layer_count; l++) {
        uint32_t ne = (uint32_t)state->program->layers[l].descriptor->plan->n_experts;
        for (uint32_t e = 0; e < ne; e++) {
            uint32_t logical = l * ne + e;
            _selected_logical_slot_host[logical] = (int32_t)((l % SALT_LR_BANK_COUNT) * ne + e);
            _text_ledger_slot_host[logical] = _selected_logical_slot_host[logical];
        }
    }
    metal_rotation_arm_done(command->rotation_base, state->layer_count);
    /* No layer, including the initial embedding/prefix, starts before its
     * own LOAD_DONE. This is a receive, never a fabricated COMPUTE_DONE. */
    command->rotation_block_layers = input_count > 1u ? salt_lr_prefill_layers() : 1u;
    uint32_t initial_end = command->rotation_block_layers < state->layer_count
        ? command->rotation_block_layers : state->layer_count;
    [command->command encodeWaitForEvent:_rotation_ready_event
        value:command->rotation_base + initial_end];
    command->rotation_ready_waits++;
    return 0;
}

static int metal_text_begin(void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    if (input_count == 0) return SALT_TFAIL(-1);
    return metal_text_begin_depth(backend_state, command_state, generation,
        source_position, input_token_ids, input_count, input_count - 1u);
}

static int metal_text_live_b1_validate(MetalTextProgramState *state,
                                      MetalTextProgramCommand *command);

static int metal_text_begin_authoritative(
        void *backend_state, void *command_state,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    MetalTextProgramCommand *command =
        (MetalTextProgramCommand *)command_state;
    if (metal_text_begin(backend_state, command_state, generation,
            source_position, input_token_ids, input_count) != 0 || !command)
        return SALT_TFAIL(-1);
    command->authoritative = 1u;
    return metal_text_live_b1_validate(backend_state, command);
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
        return SALT_TFAIL(-1);
    command->authoritative = 1u;
    command->authoritative_output_rows = output_rows;
    return metal_text_live_b1_validate(backend_state, command);
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
        return SALT_TFAIL(-1);
    for (uint32_t row = 0; row < input_count; row++)
        if (depths[row] > maximum_depth) maximum_depth = depths[row];
    if (maximum_depth == UINT32_MAX ||
        source_position > state->program->maximum_context -
            (maximum_depth + 1u) ||
        metal_text_begin_depth(backend_state, command_state,
            generation, source_position, input_token_ids, input_count,
            maximum_depth) != 0)
        return SALT_TFAIL(-1);
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
                    return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    canonical = (id<MTLBuffer>)state->canonical.backend;
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
        _weight_stationary_min_b > 0u && batch >= _weight_stationary_min_b &&
        (ref->cols & 31u) == 0u) {
        /* Reuse the qualified PREFILL kernel on the current canonical views.
         * TARGET may use this same lowering; it never sets PREFILL capacity. */
        BatchDesc4 descriptors[2] = {
            {ref->value_offset, ref->scale_offset,
             input_offset / sizeof(float), output_offset / sizeof(float)},
            {ref->bias_offset, ref->rows, ref->cols, batch},
        };
        [encoder setComputePipelineState:_bweight_pso];
        [encoder setBuffer:ref->value offset:0 atIndex:0];
        [encoder setBuffer:ref->scale offset:0 atIndex:1];
        [encoder setBuffer:ref->bias offset:0 atIndex:2];
        [encoder setBuffer:canonical offset:0 atIndex:3];
        [encoder setBuffer:canonical offset:0 atIndex:4];
        {
        NSUInteger _off = sel_args_put(&descriptors, sizeof descriptors);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
        /* TWO tile halves: the dense kernel ping-pongs between them, so it needs ONE
         * threadgroup barrier per iteration instead of two. */
        [encoder setThreadgroupMemoryLength:1024u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)ref->rows + 7u) / 8u,
                        ((NSUInteger)batch + 7u) / 8u, 1)
             threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        physical_dispatches = 1u;
    } else if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
        BatchDesc4 descriptors[2u * METAL_TEXT_MAX_CANDIDATES];
        uint32_t total_rows;
        uint32_t first = 0u;
        if (batch > METAL_TEXT_MAX_CANDIDATES ||
            ref->rows > UINT32_MAX / batch) {
            metal_text_close_encoder(command);
            return SALT_TFAIL(-1);
        }
        [encoder setComputePipelineState:_bpso2];
        [encoder setBuffer:ref->value offset:0 atIndex:0];
        [encoder setBuffer:ref->scale offset:0 atIndex:1];
        [encoder setBuffer:ref->bias offset:0 atIndex:2];
        [encoder setBuffer:canonical offset:0 atIndex:3];
        [encoder setBuffer:canonical offset:0 atIndex:4];
        {
            NSUInteger _off = sel_args_put(&ref->cols, sizeof ref->cols);
            if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
            [encoder setBuffer:_sel_args_scratch offset:_off atIndex:7];
        }
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
            {
                NSUInteger _len = (NSUInteger)active * 2u * sizeof(BatchDesc4);
                NSUInteger _off = sel_args_put(descriptors, _len);
                if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
                [encoder setBuffer:_sel_args_scratch offset:_off atIndex:5];
            }
            {
        NSUInteger _off = sel_args_put(&total_rows, sizeof total_rows);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
            {
        NSUInteger _off = sel_args_put(&active, sizeof active);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:8];
    }
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
        {
        NSUInteger _off = sel_args_put(&voff, sizeof voff);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
        {
        NSUInteger _off = sel_args_put(&soff, sizeof soff);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
        {
        NSUInteger _off = sel_args_put(&boff, sizeof boff);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
        {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:8];
    }
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)batch * ref->rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        physical_dispatches = 1u;
    } else {
        metal_text_close_encoder(command);
        return SALT_TFAIL(-1);
    }
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
        return SALT_TFAIL(-1);
    elements = (uint64_t)command->input_count * ref->cols;
    if (elements > NSUIntegerMax) return SALT_TFAIL(-1);
    args.voff = ref->value_offset;
    args.soff = ref->scale_offset;
    args.boff = ref->bias_offset;
    args.vocab = ref->rows;
    args.width = ref->cols;
    args.tokens = command->input_count;
    args.scale = state->program->descriptor->embedding_scale;
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_text_embedding_pso];
    [encoder setBuffer:ref->value offset:0 atIndex:0];
    [encoder setBuffer:ref->scale offset:0 atIndex:1];
    [encoder setBuffer:ref->bias offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:state->program->layout.candidate_token_ids atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:state->program->layout.state_a atIndex:4];
    [encoder setBuffer:_op_status offset:0 atIndex:5];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
        elements > UINT32_MAX) return SALT_TFAIL(-1);
    count = (uint32_t)elements;
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_activation_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:gate_offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:up_offset atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:output_offset atIndex:2];
    {
        NSUInteger _off = sel_args_put(&count, sizeof count);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:3];
    }
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_encode_route_complete(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t experts, uint32_t topk, uint32_t expert_width) {

    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLBuffer> canonical = (id<MTLBuffer>)state->canonical.backend;
    id<MTLComputeCommandEncoder> encoder;
    struct { uint32_t rows, experts, topk, hidden, expert_width; } args = {
        command->input_count, experts, topk, state->program->hidden,
        expert_width,
    };
    uint64_t jobs = (uint64_t)command->input_count * topk;
    uint64_t elements = jobs * state->program->hidden;
    uint64_t maximum_tiles = (uint64_t)experts + (jobs + 7u) / 8u;
    if (jobs == 0 || jobs > INT32_MAX || elements > NSUIntegerMax ||
        expert_width == 0u ||
        (uint64_t)experts + 1u + 2u * maximum_tiles + 6u >
            layout->maximum_tensor_row)
        return SALT_TFAIL(-1);
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_text_route_maps_pso];
    [encoder setBuffer:canonical offset:layout->selected_experts atIndex:0];
    [encoder setBuffer:canonical offset:layout->selected_weights atIndex:1];
    [encoder setBuffer:canonical offset:layout->grouped_to_canonical atIndex:2];
    [encoder setBuffer:canonical offset:layout->canonical_to_grouped atIndex:3];
    [encoder setBuffer:canonical offset:layout->tensor_row atIndex:4];
    [encoder setBuffer:_op_status offset:0 atIndex:5];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:6];
    }
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    command->stats.backend_physical_kernel_nodes++;
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_text_routed_gather_pso];
    [encoder setBuffer:canonical offset:layout->grouped_to_canonical atIndex:0];
    [encoder setBuffer:canonical offset:layout->combine_a atIndex:1];
    [encoder setBuffer:canonical offset:layout->routed_input atIndex:2];
    [encoder setBuffer:_op_status offset:0 atIndex:3];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:4];
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
    if (elements == 0 || elements > NSUIntegerMax) return SALT_TFAIL(-1);
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_text_expert_reduce_pso];
    [encoder setBuffer:canonical offset:layout->selected_weights atIndex:0];
    [encoder setBuffer:canonical offset:layout->canonical_to_grouped atIndex:1];
    [encoder setBuffer:canonical offset:layout->routed_output_jobs atIndex:2];
    [encoder setBuffer:canonical offset:layout->routed_output atIndex:3];
    [encoder setBuffer:_op_status offset:0 atIndex:4];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

static int metal_text_kv_rows(
        const SaltTextKvRowsDesc *rows, uint32_t width,
        id<MTLBuffer> *private_buffer, size_t *private_offset,
        id<MTLBuffer> *shared_buffer, size_t *shared_offset,
        uint32_t *shared_base, uint32_t *shared_rows,
        uint32_t *shared_stride) {
    const float *shared = NULL;
    size_t shared_capacity = 0, shared_row_stride = 0;
    uint32_t shared_row_count = 0;
    if (!rows || !private_buffer || !private_offset || !shared_buffer ||
        !shared_offset || !shared_base || !shared_rows || !shared_stride ||
        rows->private_row_stride != width || rows->private_row_capacity == 0 ||
        rows->private_float_capacity <
            (size_t)rows->private_row_capacity * width ||
        metal_shared_pointer(rows->private_rows,
            rows->private_float_capacity, private_buffer, private_offset) != 0)
        return SALT_TFAIL(-1);
    if (rows->shared_prefix_state) {
        shared = rows->shared_prefix_state->rows;
        shared_capacity = rows->shared_prefix_state->float_capacity;
        shared_row_stride = rows->shared_prefix_state->row_stride;
        *shared_base = rows->shared_prefix_state->row_base;
        shared_row_count = rows->shared_prefix_state->row_count;
    } else {
        shared = rows->shared_prefix;
        shared_capacity = rows->shared_prefix_float_capacity;
        shared_row_stride = rows->shared_prefix_row_stride;
        *shared_base = 0;
        shared_row_count = rows->shared_prefix_rows;
    }
    if (shared_row_count > 0) {
        if (!shared || shared_row_stride != width ||
            shared_capacity < (size_t)shared_row_count * width ||
            metal_shared_pointer(shared, shared_capacity,
                shared_buffer, shared_offset) != 0)
            return SALT_TFAIL(-1);
    } else {
        *shared_buffer = *private_buffer;
        *shared_offset = *private_offset;
        shared_row_stride = width;
    }
    if (rows->private_mode != SALT_TEXT_KV_PRIVATE_ABSOLUTE &&
        rows->private_mode != SALT_TEXT_KV_PRIVATE_RING)
        return SALT_TFAIL(-1);
    *shared_rows = shared_row_count;
    *shared_stride = (uint32_t)shared_row_stride;
    return 0;
}

/* Resolve the authoritative next seat through the existing shared allocation.
 * No allocation, alternate address table, or copy/publish pass is involved. */
static int metal_text_live_b1_row(const SaltTextKvRowsDesc *rows,
        const SaltAttentionDesc *attention, uint32_t width, uint32_t position,
        id<MTLBuffer> *buffer, size_t *offset) {
    id<MTLBuffer> shared;
    size_t shared_offset;
    uint32_t shared_rows, shared_stride, row;
    uint32_t shared_base;
    if (metal_text_kv_rows(rows, width, buffer, offset, &shared,
            &shared_offset, &shared_base, &shared_rows, &shared_stride) != 0 ||
        shared_base > UINT32_MAX - shared_rows ||
        (position >= shared_base && position < shared_base + shared_rows) ||
        position < rows->private_position_base)
        return SALT_TFAIL(-1);
    row = position - rows->private_position_base;
    if (rows->private_mode == SALT_TEXT_KV_PRIVATE_RING) {
        /* An overwritten row must be outside this token's causal window. */
        if (row >= rows->private_row_capacity &&
            (attention->kind == SALT_ATTN_FULL || attention->window <= 0 ||
             rows->private_row_capacity < (uint32_t)attention->window))
            return SALT_TFAIL(-1);
        row %= rows->private_row_capacity;
    } else if (row >= rows->private_row_capacity) {
        return SALT_TFAIL(-1);
    }
    *offset += (size_t)row * width * sizeof(float);
    return 0;
}

static int metal_text_live_b1_validate(MetalTextProgramState *state,
                                      MetalTextProgramCommand *command) {
    if (command->input_count != 1u) return 0;
    for (uint32_t layer = 0; layer < state->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &state->program->layers[layer];
        const SaltTextKvLayerDesc *kv = state->layers[layer].kv;
        const SaltAttentionDesc *attention = &compiled->descriptor->plan->attention;
        id<MTLBuffer> buffer;
        size_t offset;
        if (!kv || metal_text_live_b1_row(&kv->keys, attention,
                compiled->kv_width, command->source_position,
                &buffer, &offset) != 0 ||
            metal_text_live_b1_row(&kv->values, attention,
                compiled->kv_width, command->source_position,
                &buffer, &offset) != 0)
            return SALT_TFAIL(-1);
    }
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
    if (tasks == 0 || tasks > NSUIntegerMax) return SALT_TFAIL(-1);
    if (attention->shared_kv_projection) {
        size_t elements, bytes;
        id<MTLBlitCommandEncoder> blit;
        if (compiled->kv_width == 0u ||
            (size_t)command->input_count > SIZE_MAX / compiled->kv_width)
            return SALT_TFAIL(-1);
        elements = (size_t)command->input_count * compiled->kv_width;
        if (elements > SIZE_MAX / sizeof(float)) return SALT_TFAIL(-1);
        bytes = elements * sizeof(float);
        if (layout->keys > state->canonical_bytes ||
            bytes > state->canonical_bytes - layout->keys ||
            layout->values > state->canonical_bytes ||
            bytes > state->canonical_bytes - layout->values)
            return SALT_TFAIL(-1);
        metal_text_close_encoder(command);
        blit = [command->command blitCommandEncoder];
        if (!blit) return SALT_TFAIL(-1);
        [blit copyFromBuffer:canonical sourceOffset:layout->keys
                    toBuffer:canonical destinationOffset:layout->values
                        size:bytes];
        [blit endEncoding];
        command->stats.backend_physical_kernel_nodes++;
    }
    id<MTLBuffer> key_output = canonical, value_output = canonical;
    size_t key_offset = compiled->tentative_key_offset;
    size_t value_offset = compiled->tentative_value_offset;
    if (command->authoritative && command->input_count == 1u &&
        (metal_text_live_b1_row(&refs->kv->keys, attention,
             compiled->kv_width, command->source_position,
             &key_output, &key_offset) != 0 ||
         metal_text_live_b1_row(&refs->kv->values, attention,
             compiled->kv_width, command->source_position,
             &value_output, &value_offset) != 0))
        return SALT_TFAIL(-1);
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_text_transform_view_pso];
    [encoder setBuffer:canonical offset:layout->queries atIndex:0];
    [encoder setBuffer:canonical offset:layout->keys atIndex:1];
    [encoder setBuffer:canonical offset:layout->values atIndex:2];
    [encoder setBuffer:_op_meta
                offset:refs->meta_float_offset * sizeof(float) atIndex:3];
    [encoder setBuffer:key_output offset:key_offset atIndex:4];
    [encoder setBuffer:value_output offset:value_offset atIndex:5];
    [encoder setBuffer:_op_status offset:0 atIndex:6];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:7];
    }
    [encoder setThreadgroupMemoryLength:sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
    uint32_t shared_kb, shared_vb, shared_kr, shared_vr, shared_ks, shared_vs;
    id<MTLComputeCommandEncoder> encoder;
    struct {
        uint32_t full, n_heads, n_kv_heads, head_dim, window;
        uint32_t source, batch, qstride, kvstride;
        uint32_t private_mode, private_capacity, private_base;
        uint32_t shared_base, shared_rows, shared_stride;
    } args;
    uint64_t tasks;
    size_t score_bytes;
    if (!kv || metal_text_kv_rows(&kv->keys, compiled->kv_width,
            &private_k, &private_ko, &shared_k, &shared_ko,
            &shared_kb, &shared_kr, &shared_ks) != 0 ||
        metal_text_kv_rows(&kv->values, compiled->kv_width,
            &private_v, &private_vo, &shared_v, &shared_vo,
            &shared_vb, &shared_vr, &shared_vs) != 0 ||
        shared_kb != shared_vb || shared_kr != shared_vr ||
        shared_ks != shared_vs ||
        kv->keys.private_mode != kv->values.private_mode ||
        kv->keys.private_row_capacity != kv->values.private_row_capacity ||
        kv->keys.private_position_base != kv->values.private_position_base)
        return SALT_TFAIL(-1);
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
    args.shared_base = shared_kb;
    args.shared_rows = shared_kr;
    args.shared_stride = shared_ks;
    tasks = (uint64_t)command->input_count * attention->n_heads;
    score_bytes = (size_t)command->attention_score_rows * sizeof(float);
    if (tasks == 0 || tasks > NSUIntegerMax || score_bytes > 32u * 1024u)
        return SALT_TFAIL(-1);
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    const int product_tiled = command->decode_experts &&
        salt_lr_decode_attention_product_tile();
    if (product_tiled && (score_bytes + 16384u + 2u * sizeof(float) >
            [_dev maxThreadgroupMemoryLength] ||
            [_text_attention_product_view_pso maxTotalThreadsPerThreadgroup] < 256u))
        return SALT_TFAIL(-1);
    [encoder setComputePipelineState:product_tiled
        ? _text_attention_product_view_pso : _text_attention_view_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->queries atIndex:0];
    [encoder setBuffer:private_k offset:private_ko atIndex:1];
    [encoder setBuffer:private_v offset:private_vo atIndex:2];
    [encoder setBuffer:shared_k offset:shared_ko atIndex:3];
    [encoder setBuffer:shared_v offset:shared_vo atIndex:4];
    id<MTLBuffer> current_k = (id<MTLBuffer>)state->canonical.backend;
    id<MTLBuffer> current_v = current_k;
    size_t current_ko = compiled->tentative_key_offset;
    size_t current_vo = compiled->tentative_value_offset;
    if (command->authoritative && command->input_count == 1u &&
        (metal_text_live_b1_row(&kv->keys, attention, compiled->kv_width,
             command->source_position, &current_k, &current_ko) != 0 ||
         metal_text_live_b1_row(&kv->values, attention, compiled->kv_width,
             command->source_position, &current_v, &current_vo) != 0)) {
        return SALT_TFAIL(-1);
    }
    [encoder setBuffer:current_k offset:current_ko atIndex:5];
    [encoder setBuffer:current_v offset:current_vo atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->attention_output atIndex:7];
    [encoder setBuffer:_op_status offset:0 atIndex:8];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:9];
    }
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->target_parent_rows atIndex:10];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:layout->target_depths atIndex:11];
    [encoder setThreadgroupMemoryLength:score_bytes atIndex:0];
    [encoder setThreadgroupMemoryLength:2u * sizeof(float) atIndex:1];
    if (product_tiled)
        [encoder setThreadgroupMemoryLength:16384u atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)tasks, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(product_tiled ? 256u : 64u, 1, 1)];
    if (product_tiled) _attention_decode_product_calls++;
    else if (command->decode_experts) _attention_decode_control_calls++;
    else _attention_prefill_control_calls++;
    command->stats.backend_physical_kernel_nodes++;
    return 0;
}

/* PREFETCH: the per-layer expert selection the host already reads here is also the
 * best predictor of the NEXT token's selection for the same layer.  Remember it and
 * issue the next layer's fetch NOW, so the I/O overlaps the current layer's device
 * compute instead of starting on the critical path at the next gate. */
#define METAL_TEXT_SEL_HISTORY 64
static void *_text_expert_cache;   /* published by the model; used for leases and prefetch */
static int32_t _sel_history[METAL_TEXT_SEL_HISTORY][METAL_TEXT_MAX_EXPERTS];
static int32_t _sel_prefetch_slots[METAL_TEXT_MAX_EXPERTS];
static uint32_t _sel_history_count[METAL_TEXT_SEL_HISTORY];
#include "decode_overlap.inc"

static int __attribute__((unused)) metal_text_decode_complete(MetalTextProgramCommand *command, uint32_t layer) {
    id<MTLCommandBuffer> current = command->command;
    if (!current) return SALT_TFAIL(-1);
    metal_text_close_encoder(command);
    command->command = nil;
    double start = _mt_probe_ns();
    [current commit];
    double committed = _mt_probe_ns();
    salt_lr_timeline_submit(command->rotation_base, command->input_count, (uint64_t)committed);
    [current waitUntilCompleted];
    double end = _mt_probe_ns();
    if (_text_probe_enabled && _text_probe_n < METAL_TEXT_PROBE_MAX) {
        _text_probe[_text_probe_n].wall = (float)((end - start) * 1e-6);
        _text_probe[_text_probe_n].commit = (float)((committed - start) * 1e-6);
        _text_probe[_text_probe_n].gpu = (float)(([current GPUEndTime] - [current GPUStartTime]) * 1e3);
        _text_probe[_text_probe_n].rows = command->input_count;
        _text_probe[_text_probe_n].layer = layer;
        _text_probe[_text_probe_n].phase = command->phase;
        _text_probe[_text_probe_n].operation_phase = (uint32_t)(command->rotation_base / ((uint64_t)_arrival_layer_count + 2u)) + 1u;
        _text_probe[_text_probe_n].operation_kind = 2u;
        _text_probe_n++;
    }
    int bad = metal_command_succeeded(current) != 0 || *(uint32_t *)[_op_status contents] != 0u;
    if (metal_text_overlap_collect(command) != 0) bad = 1;
    metal_text_record_gpu_time(command, current);
    [current release];
    command->stats.backend_host_kernel_launch_calls++;
    return bad ? SALT_TFAIL(-1) : 0;
}
static int metal_text_prepare_resource_request(
        MetalTextProgramState *state, MetalTextProgramCommand *command,
        uint32_t layer) {
    const SaltTextLayerPlan *plan;
    const int32_t *selected;
    const int32_t *grouped;
    uint32_t jobs;
    /* ONE REGISTER PER CONDITION, so the request says which one refused instead of being reasoned about. */
    if (!state) { salt_trace_register(200, 1); return SALT_TFAIL(-1); }
    if (!command) { salt_trace_register(201, 1); return SALT_TFAIL(-1); }
    if (layer >= state->layer_count) {
        salt_trace_register(202, 1);
        salt_trace_value(202, (int)layer);
        return SALT_TFAIL(-1);
    }
    if (command->phase != METAL_TEXT_RECORDING_PREFIX &&
        command->phase != METAL_TEXT_RECORDING_SUFFIX) {
        salt_trace_register(203, 1);
        salt_trace_value(203, (int)command->phase);
        return SALT_TFAIL(-1);
    }
    plan = state->program->layers[layer].descriptor->plan;
    if (!plan) { salt_trace_register(204, 1); return SALT_TFAIL(-1); }
    if (plan->n_experts < 1 || plan->n_experts > (int)METAL_TEXT_MAX_EXPERTS) {
        salt_trace_register(205, 1);
        salt_trace_value(205, plan->n_experts);
        return SALT_TFAIL(-1);
    }
    if (plan->top_k_experts < 1 ||
        command->input_count > UINT32_MAX / (uint32_t)plan->top_k_experts) {
        salt_trace_register(206, 1);
        salt_trace_value(206, plan->top_k_experts);
        return SALT_TFAIL(-1);
    }
    /* THE ROUTING IS THE ROUTER'S.  Fully resident removes the FETCH, not the routing: which experts
     * hold which rows is the model's decision, and it is read here from the canonical buffers the device
     * wrote.  The commit that precedes this call serves exactly this read and nothing else.  Assuming
     * every expert holds every row gives row_prefix = 128 x input_count instead of input_count x top_k,
     * and the chain's own guard refuses that. */
    /* THE CANONICAL BINDINGS, WHICH MY REWRITE DROPPED.  selected[] and grouped[] are the device's own
     * routing output inside the canonical buffer - the commit above serves exactly this read.  Without
     * these two lines the pointers were uninitialised, so the loop read arbitrary memory: the first entry
     * happened to be 0, which passed the guard at 0 >= 128 and printed a value the code could not
     * produce. */
    selected = (const int32_t *)((const uint8_t *)state->canonical.contents +
        state->program->layout.selected_experts);
    grouped = (const int32_t *)((const uint8_t *)state->canonical.contents +
        state->program->layout.grouped_to_canonical);
    jobs = command->input_count * (uint32_t)plan->top_k_experts;
    memset(command->occupancies, 0, sizeof command->occupancies);
    /* OCCUPANCY IS THE NUMBER OF ROWS THIS EXPERT WAS ACTUALLY ROUTED. */
    for (uint32_t item = 0; item < jobs; item++) {
        /* THE RAW ENTRY, BEFORE ANY CAST.  selected[] is int32 and the loop works in uint32, so a negative
         * entry becomes a huge value and a cast back to int loses it - the record then shows a number that
         * is not what the router wrote.  Register 210 carries the value as it is stored. */
        salt_trace_register(210, 1);
        salt_trace_value(210, (int)selected[item]);
        uint32_t expert = (uint32_t)selected[item];
        if (expert >= (uint32_t)plan->n_experts) {
            /* THE ROUTER'S OWN VALUE, RECORDED.  It named an expert the plan does not allow, which means
             * either the buffer is not what the commit should have made visible, or n_experts is wrong
             * for this layer. */
            salt_trace_register(207, 1);
            salt_trace_value(207, (int)expert);
            salt_trace_register(208, 1);
            salt_trace_value(208, plan->n_experts);
            return SALT_TFAIL(-1);
        }
        if (command->occupancies[expert] == UINT32_MAX) {
            salt_trace_register(209, 1);
            salt_trace_value(209, (int)expert);
            return SALT_TFAIL(-1);
        }
        command->occupancies[expert]++;
    }
    /* THE REQUEST, IN EXPERT ORDER, AND THE GROUPED ORDER MUST AGREE. */
    command->request_count = 0;
    uint32_t grouped_cursor = 0;
    for (uint32_t expert = 0; expert < (uint32_t)plan->n_experts; expert++) {
        uint32_t population = command->occupancies[expert];
        if (population == 0) continue;
        if (command->request_count >= METAL_TEXT_MAX_EXPERTS) {
            salt_trace_register(195, 1);
            salt_trace_value(195, (int)command->request_count);
            return SALT_TFAIL(-1);
        }
        command->request_experts[command->request_count++] = (int32_t)expert;
        for (uint32_t item = 0; item < population; item++) {
            uint32_t canonical = (uint32_t)grouped[grouped_cursor++];
            if (canonical >= jobs || selected[canonical] != (int32_t)expert) {
                salt_trace_register(196, 1);
                salt_trace_value(196, (int)canonical);
                salt_trace_value(197, (int)expert);
                return SALT_TFAIL(-1);
            }
        }
    }
    if (grouped_cursor != jobs || command->request_count == 0) {
        salt_trace_register(198, 1);
        salt_trace_value(198, (int)grouped_cursor);
        salt_trace_value(199, (int)jobs);
        return SALT_TFAIL(-1);
    }
    command->active_layer = layer;
    command->phase = METAL_TEXT_RESOURCE_PENDING;
    /* PREFETCH (see the note above the function): record this layer's selection,
     * then issue the NEXT layer's fetch from the remembered selection. */
    if (layer < METAL_TEXT_SEL_HISTORY && command->request_count > 0u) {
        uint32_t _rc = command->request_count;
        if (_rc > METAL_TEXT_MAX_EXPERTS) _rc = METAL_TEXT_MAX_EXPERTS;
        memcpy(_sel_history[layer], command->request_experts, sizeof(int32_t) * _rc);
        _sel_history_count[layer] = _rc;
    }
    /* LAYER ROTATION (routing-INDEPENDENT): while layer i computes, bring in layer
     * i+1's FULL expert set.  It needs no routing knowledge, so it fires on the FIRST
     * pass and on every pass - prefill and decode alike.  The history-based variant
     * that used to sit here could only fire from the second pass onward, which is why
     * the prefill never got it and the fetch stayed on the critical path.  Bytes per
     * layer are higher (full set vs demand) but the read hides inside the layer's
     * compute, and the fetch pool is a ring, so the batch does not barrier.
     */
    /* THE ROTATION OWNS THIS.  The load is rotation/layer_rotation.c's: it walks the trunk one layer
     * ahead of the compute, from the source handed over at initiation, into the pair of fixed ranges.
     * The backend has no window, no lead and no fetch - it realizes the compute and nothing else. */
    (void)_text_expert_cache;
    /* THE FUNCTION MUST RETURN A VALUE.  My rewrite removed the closing return, so it fell off the end -
     * undefined for a non-void function, and in practice non-zero, which is why every guard above passed
     * and the caller still saw a failure. */
    return 0;
}

static int metal_text_decode_demand(MetalTextProgramState *state,
        MetalTextProgramCommand *command, uint32_t layer) {
    MetalTextProgramCommand request;
    int32_t slots[METAL_TEXT_MAX_EXPERTS];
    uint32_t ne;
    if (!state || !command || !command->decode_experts ||
        layer >= state->layer_count)
        return SALT_TFAIL(-1);
    memset(&request, 0, sizeof request);
    request.phase = METAL_TEXT_RECORDING_PREFIX;
    request.input_count = command->input_count;
    if (metal_text_prepare_resource_request(state, &request, layer) != 0 ||
        request.request_count == 0u ||
        salt_lr_decode_fetch((int)layer, request.request_experts,
            (int)request.request_count, slots) != 0)
        return SALT_TFAIL(-1);

    /* Rebuild the complete logical table from the conservation catalog. An
     * eviction can remove an expert from any earlier layer; clearing only the
     * current layer leaves a stale device-visible hit. */
    for (int logical = 0; logical < _selected_logical_capacity; logical++) {
        _selected_logical_slot_host[logical] = -1;
        _selected_logical_payload_host[logical] = 0u;
        if (logical < _text_ledger_logical_capacity)
            _text_ledger_slot_host[logical] = -1;
    }
    ne = (uint32_t)state->program->layers[layer].descriptor->plan->n_experts;
    for (uint32_t il = 0; il < state->layer_count; il++) {
        for (unsigned rank = 0; rank < salt_lr_decode_layer_seats(il); rank++) {
            int expert = -1, seat = -1;
            uint64_t logical;
            if (!salt_lr_decode_seat(il, rank, &expert, &seat)) break;
            logical = (uint64_t)il * ne + (uint32_t)expert;
            if (logical >= (uint64_t)_selected_logical_capacity || seat < 0 ||
                seat >= _selected_cache_capacity)
                return SALT_TFAIL(-1);
            _selected_logical_slot_host[logical] = seat;
            if (logical < (uint64_t)_text_ledger_logical_capacity) {
                _text_ledger_slot_host[logical] = seat;
                _text_ledger_expert_host[logical] = expert;
                _text_ledger_gen_host[logical] =
                    (uint64_t)_text_ledger_cur_epoch;
            }
        }
    }
    for (uint32_t i = 0; i < request.request_count; i++) {
        uint64_t logical = (uint64_t)layer * ne +
            (uint32_t)request.request_experts[i];
        if (logical >= (uint64_t)_selected_logical_capacity ||
            _selected_logical_slot_host[logical] != slots[i])
            return SALT_TFAIL(-1);
    }
    return 0;
}

static void metal_text_arm_decode_demands(MetalTextProgramState *state,
        MetalTextProgramCommand *command) {
    if (!state || !command || !_rotation_done_event ||
        !_rotation_ready_event || !_rotation_done_listener) {
        _rotation_stream_error = 1;
        return;
    }
    for (uint32_t layer = 0; layer < state->layer_count; layer++) {
        const uint64_t demand = command->rotation_base + layer + 1u;
        [_rotation_done_event notifyListener:_rotation_done_listener
            atValue:demand
            block:^(id<MTLSharedEvent> event, uint64_t value) {
                (void)event;
                if (value != demand || !command->decode_experts ||
                    command->rotation_base + layer + 1u != demand)
                    return;
                __atomic_fetch_add(&_decode_demand_received, 1u,
                    __ATOMIC_RELAXED);
                if (metal_text_decode_demand(state, command, layer) != 0) {
                    _rotation_stream_error = 1;
                    __atomic_fetch_add(&_decode_demand_failed, 1u,
                        __ATOMIC_RELAXED);
                    salt_rot_fail_at(__FILE__, __LINE__, -1);
                    [_rotation_ready_event setSignaledValue:UINT64_MAX];
                    return;
                }
                metal_rotation_publish_ready(command, demand);
                __atomic_fetch_add(&_decode_ready_published, 1u,
                    __ATOMIC_RELAXED);
            }];
        _decode_demand_armed++;
    }
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
        return SALT_TFAIL(-1);
    *layer = command->active_layer;
    *expert_count = command->request_count;
    memcpy(experts, command->request_experts,
        (size_t)command->request_count * sizeof *experts);
    return 0;
}

static void metal_text_register_resolution(uint32_t layer, uint32_t count,
                                           const int32_t *experts,
                                           const int32_t *slots,
                                           uint32_t n_experts);

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
        return SALT_TFAIL(-1);
    if (command->decode_experts) {
        uint32_t ne = (uint32_t)state->program->layers[layer].descriptor->plan->n_experts;
        for (uint32_t e = 0; e < ne; e++) {
            int seat = -1;
            (void)salt_lr_decode_find((int)layer, (int)e, &seat);
            _selected_logical_slot_host[layer * ne + e] = seat;
            _text_ledger_slot_host[layer * ne + e] = seat;
        }
    }
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
            (!command->decode_experts && (!_selected_cache_resources[slot].map ||
             _selected_cache_resources[slot].logical_resource_id != logical)))
            return SALT_TFAIL(-1);
        command->request_slots[index] = slot;
    }
    metal_text_register_resolution(layer, expert_count, command->request_experts,
        command->request_slots,
        (uint32_t)state->program->layers[layer].descriptor->plan->n_experts);
    if (_text_ledger_slot_host && _text_ledger_gen_host &&
        _text_ledger_expert_host) {
        uint32_t _ne = (uint32_t)state->program->layers[layer]
            .descriptor->plan->n_experts;
        for (uint32_t _i = 0; _i < expert_count; _i++) {
            uint64_t _lg = (uint64_t)layer * _ne + (uint32_t)experts[_i];
            if (_lg >= (uint64_t)_text_ledger_logical_capacity) break;
            _text_ledger_expert_host[_lg] = experts[_i];
            _text_ledger_slot_host[_lg] = slots[_i];
            _text_ledger_gen_host[_lg] = (uint64_t)_text_ledger_cur_epoch;
        }
    }
    if (metal_text_new_command(command) != 0) return SALT_TFAIL(-1);
    command->phase = METAL_TEXT_RECORDING_EXPERT;
    command->expert_chain_mask = 0u;
    return 0;
}

/* Layer residency mask published by the model (see gpu.h). Bit i set means
 * layer i's expert set is resident, so the expert-boundary yield can be
 * skipped for that layer and recording continues in the same command
 * buffer instead of draining the pipeline. */
/* LIVE per-layer load-epoch array owned by the cache (the state machine).
 * The address book is a dynamic ledger populated with each layer load, so the
 * layer load IS the invalidation event: if a layer's epoch is unchanged since
 * this backend resolved it, its table is still current and the expert-boundary
 * yield can be skipped without draining the pipeline.  The array is
 * startup-owned, so the pointer is stable for the engine's lifetime. */
static const uint64_t *_text_load_epochs = NULL;
static int _text_load_epoch_count = 0;

/* Per-layer resolution state, owned by this backend.  A layer's
 * resolved table is current iff the cache's load epoch for that layer
 * is unchanged since this backend last resolved it. */
#define METAL_TEXT_RESOLVE_LAYERS 64
#define METAL_TEXT_RESOLVE_EXPERTS METAL_TEXT_MAX_EXPERTS
static int _text_resolved_valid[METAL_TEXT_RESOLVE_LAYERS];
static int _text_resolved_count[METAL_TEXT_RESOLVE_LAYERS];
static uint64_t _text_resolved_epoch[METAL_TEXT_RESOLVE_LAYERS];
/* The expert IDs the table was resolved FOR.  The epoch alone is not enough: the
 * routing changes per token and is not a cache event, so a table resolved during
 * PREFILL looks "valid" to a DECODE token whose routing differs.  Set-equality
 * alone is not enough either (a retire can reassign slots at the same set).  The
 * predicate needs BOTH. */
static int32_t _text_resolved_experts[METAL_TEXT_RESOLVE_LAYERS][METAL_TEXT_MAX_EXPERTS];
/* The selected-resource table is SHARED: resource_resume writes the
 * CURRENT layer's refs into it, so an unchanged epoch for layer i
 * does not mean the table holds layer i's refs - it holds whatever
 * layer was resolved last.  Validity therefore requires that the
 * table currently reflects THIS layer, at an unchanged epoch. */
static uint32_t _text_table_layer = UINT32_MAX;

/* Ledger registration: records that THIS layer's table is populated for the
 * epoch current at registration.  Keyed on the layer itself - never on a
 * last-resolved global, which is what made the earlier predicates impossible.
 */
static void metal_text_register_resolution(uint32_t layer, uint32_t count,
                                           const int32_t *experts,
                                           const int32_t *slots,
                                           uint32_t n_experts) {
    uint64_t epoch;
    if (layer >= METAL_TEXT_RESOLVE_LAYERS) return;
    _text_resolved_valid[layer] = 1;
    _text_resolved_count[layer] = (int)count;
    for (uint32_t i = 0; i < count && i < METAL_TEXT_MAX_EXPERTS; i++)
        _text_resolved_experts[layer][i] = experts ? experts[i] : -1;
    epoch = (_text_load_epochs && (int)layer < _text_load_epoch_count)
        ? _text_load_epochs[layer] : 0u;
    _text_resolved_epoch[layer] = epoch;
    _text_table_layer = layer;
    /* POPULATE THE LEDGER - the table the dependency barrier's skip path reads to rebuild
     * request_experts/request_slots WITHOUT the host reading the device's router, which is
     * the only readback that forces the per-layer commit and wait.  Written HERE because
     * this is the one place the resolution is known complete: resource_resume has validated
     * every slot for THIS layer.  Indexed per (layer, expert), and the epoch recorded is
     * the cache's own layer epoch, so a later fetch invalidates the entry. */
    if (!_text_ledger_gen_host || !_text_ledger_expert_host || !_text_ledger_slot_host ||
        n_experts == 0u)
        return;
    for (uint32_t i = 0; i < count && i < METAL_TEXT_MAX_EXPERTS; i++) {
        int32_t e = experts ? experts[i] : -1;
        uint64_t logical;
        if (e < 0 || (uint32_t)e >= n_experts) continue;
        logical = (uint64_t)layer * n_experts + (uint64_t)e;
        if (logical >= (uint64_t)_text_ledger_logical_capacity) continue;
        _text_ledger_gen_host[logical] = epoch;
        _text_ledger_expert_host[logical] = e;
        _text_ledger_slot_host[logical] = slots ? slots[i] : -1;
        /* THE TABLE THE KERNEL AND THE RECORD READ.  Two tables hold the same mapping: the ledger one, and
         * _selected_logical_slot_host, which is uploaded to the device as _selected_logical_slots and is
         * what the manifest's S rows are emitted from.  Writing only the ledger leaves the device table at
         * -1, so the kernel resolves every expert to a miss and the record reports nothing registered. */
        if (_selected_logical_slot_host && slots &&
            logical < (uint64_t)_selected_logical_capacity)
            _selected_logical_slot_host[logical] = slots[i];
    }
    _text_ledger_prev_epoch = epoch;
}

/* The expert cache handle published by the model (see gpu.h).  Held so the
 * backend can LEASE every slot an encoded dispatch references: a leased slot
 * cannot be evicted, so a deferred dispatch reads a table that cannot move.
 * Without this the cache retires the earliest layer's experts before their
 * dispatches execute, which is what forces the per-layer commit. */
static void *_text_expert_cache;





int salt_gpu_text_resident_layers(const uint64_t *load_epochs, int count) {
    if (!load_epochs || count < 0 || count > 64) return SALT_TFAIL(-1);
    _text_load_epochs = load_epochs;
    _text_load_epoch_count = count;
    return 0;
}


/* Returns 1 when this layer's requested expert set is identical to the
 * one last resolved (table still valid), 0 when it differs.  Records the
 * requested set either way so the next call can compare. */
static int __attribute__((unused)) metal_text_resolution_valid(MetalTextProgramCommand *command,
                                       uint32_t layer) {
    uint64_t epoch;
    if (!command || layer >= METAL_TEXT_RESOLVE_LAYERS) return 0;
    /* No epoch source published: never skip, always resolve. */
    if (!_text_load_epochs || (int)layer >= _text_load_epoch_count) return 0;
    epoch = _text_load_epochs[layer];
    /* PER-LAYER, keyed on the layer's own book entry. */
    if (_text_resolved_valid[layer] &&
        _text_resolved_epoch[layer] == epoch)
        return 1;
    _text_resolved_epoch[layer] = epoch;
    _text_resolved_count[layer] = (int)command->request_count;
    _text_resolved_valid[layer] = 1;
    _text_table_layer = layer;
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
        return SALT_TFAIL(-1);
    *encoded_cells = 0;
    for (uint32_t offset = 0; offset < cell_count; offset++) {
        uint32_t index = first_cell + offset;
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        mt_node_tick("extent_cell", __func__, cell->layer, (uint32_t)cell->kind);
        if ((int)cell->layer != _text_last_rot_layer) {
            _text_last_rot_layer = (int)cell->layer;
            /* THE ROTATOR IS NOT STEPPED HERE.  This is a layer's FIRST cell, and the range its
             * successor will occupy is still being read by the layer BEFORE it: that layer's chain
             * dispatches are in a command buffer the expert-gate barrier has not committed yet, so a load
             * issued here overwrites weights that have been RECORDED but not EXECUTED.  The retire point
             * is the gate commit - see metal_text_dependency_barrier. */
            /* THE RESIDENT CEILING, CHECKED WHERE THE WALK ADVANCES.  The program-driven prefill runs the
             * whole program in one call, so the only gate it had was before the program - nothing bounded
             * RSS during the walk, and that is where the peak happened.  This is the engine's own walk
             * boundary, and the engine owns the limit. */
            if (_cfg_mem_limit_bytes != 0u) {
                SaltMemSnapshot _ms;
                if (salt_mem_snapshot(&_ms) == 0 && _ms.resident_b > 0) {
                    /* RECORDED AT EVERY BOUNDARY, not only on failure.  first/last then show where RSS
                     * grows - the layer it starts at and the layer it reaches the ceiling on. */
                    salt_trace_register(181, 1);
                    salt_trace_value(181, (int)((uint64_t)_ms.resident_b / 1000000u));
                    salt_trace_value(182, (int)cell->layer);
                    if ((uint64_t)_ms.resident_b > _cfg_mem_limit_bytes)
                        return SALT_TFAIL(-1);
                }
            }
        }
        /* THE DRIVER ALREADY CALLED THE BARRIER FOR THIS CELL.  text_verify.c:4496 calls
         * backend_ops->dependency_barrier for every cell with dependency_epoch > 0 and accepts its answer
         * (register 100 is silent), and then calls this extent encoder - which called the barrier a SECOND
         * time for the same cells.  The barrier is stateful: the first call advances the phase, so the
         * second cannot yield and returns -1 (register 98).  The driver owns the decision; the extent only
         * records. */
        (void)cell->completion_epoch;
        {
            int erc = metal_text_encode_cell(backend_state, command_state, program, cell,
                &plan->assignments[index], input_count);
            if (erc != 0) {
                salt_trace_register(99, 1);   /* the cell's encode refused */
                salt_trace_value(99, (int)cell->kind);
                return SALT_TFAIL(-1);
            }
        }
        *encoded_cells = offset + 1u;
    }
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int metal_text_encode_expert_chain(
        MetalTextProgramState *state, MetalTextProgramCommand *command) {
    if (!state || !command || !command->command || !_rotation_arena_backing ||
        !_rotation_stride || _rotation_arena_slots < 1 ||
        command->phase != METAL_TEXT_RECORDING_EXPERT ||
        command->active_layer >= state->layer_count)
        return SALT_TFAIL(-1);
    const SaltTextLayerPlan *plan = state->program->layers[
        command->active_layer].descriptor->plan;
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    id<MTLBuffer> canonical = (id<MTLBuffer>)state->canonical.backend;
    uint32_t jobs = command->input_count * (uint32_t)plan->top_k_experts;
    uint64_t stride = _rotation_stride;
    MetalRotationExpertArgs args = {
        jobs, (uint32_t)plan->expert_intermediate, state->program->hidden,
        (uint32_t)plan->n_experts, command->active_layer, 0u, 2u,
        command->decode_experts ? salt_lr_decode_bank_origin(command->active_layer)
            : command->active_layer % SALT_LR_BANK_COUNT * (uint32_t)plan->n_experts,
        (uint32_t)_rotation_arena_slots,
        stride, layout->routed_input / sizeof(float),
        layout->routed_gate / sizeof(float), layout->routed_up / sizeof(float)
    };
    id<MTLComputeCommandEncoder> encoder = metal_text_encoder(command);
    int grouped = command->input_count > 1u;
    if (!encoder || (grouped
            ? metal_selected_encode_grouped(encoder, canonical, layout, &args)
            : metal_selected_encode_rows4(encoder, canonical,
                layout->selected_experts, layout->grouped_to_canonical,
                &args)) != 0)
        return SALT_TFAIL(-1);
    if (metal_text_encode_activation(state, command,
            layout->routed_gate, layout->routed_up, layout->routed_chain,
            (size_t)jobs * (uint32_t)plan->expert_intermediate) != 0)
        return SALT_TFAIL(-1);
    args.rows = state->program->hidden;
    args.cols = (uint32_t)plan->expert_intermediate;
    args.projection = 2u; args.projections = 1u;
    args.input = layout->routed_chain / sizeof(float);
    args.output = layout->routed_output_jobs / sizeof(float);
    args.second_output = args.output;
    encoder = metal_text_encoder(command);
    if (!encoder || (grouped
            ? metal_selected_encode_grouped(encoder, canonical, layout, &args)
            : metal_selected_encode_rows4(encoder, canonical,
                layout->selected_experts, layout->grouped_to_canonical,
                &args)) != 0)
        return SALT_TFAIL(-1);
    if (grouped) {
        _text_grouped_prefill_encoded_layers++;
        _text_grouped_prefill_encoded_jobs += jobs;
    } else {
        _text_rows4_decode_encoded_layers++;
        _text_rows4_decode_dispatches += 2u;
    }
    command->expert_chain_mask = 15u;
    command->stats.expert_gate_up_dispatches += 2u;
    command->stats.expert_down_dispatches++;
    command->stats.backend_selected_jobs = jobs;
    /* Two projection kernels; the activation helper accounts for its own. */
    command->stats.backend_physical_kernel_nodes += 2u;
    command->profile_command_contains_expert = 1u;
    command->profile_expert_layers++;
    command->profile_expert_rows_total += jobs;
    command->profile_layer_rows[command->active_layer] = jobs;
    return 0;
}

/* 1 when this layer's table is populated AND the cache's epoch for it is
 * unchanged: nothing to fetch, so the expert-boundary commit may be skipped.
 * Default off via SALT_TEXT_LEDGER_SKIP while it is qualified. */
static int metal_text_layer_ready(const MetalTextProgramCommand *command,
                                  uint32_t layer) {
    if (layer >= METAL_TEXT_RESOLVE_LAYERS || !command) return 0;
    if (!_text_load_epochs || (int)layer >= _text_load_epoch_count) return 0;
    if (!_text_resolved_valid[layer]) return 0;
    /* HALF ONE: the cache's epoch for this layer is unchanged (no retire/eviction has
     * reassigned its slots since the table was populated). */
    if (_text_resolved_epoch[layer] != _text_load_epochs[layer]) return 0;
    /* HALF TWO: the table was resolved for THIS command's routing.  Without this the
     * PREFILL's resolution satisfies a DECODE token whose routing differs. */
    if ((uint32_t)_text_resolved_count[layer] != command->request_count) return 0;
    for (uint32_t i = 0; i < command->request_count; i++)
        if (_text_resolved_experts[layer][i] != command->request_experts[i]) return 0;
    return 1;
}

static int metal_text_dependency_barrier(
        void *backend_state, void *command_state,
        uint32_t dependency_epoch, uint32_t completion_epoch) {
    MetalTextProgramState *state = (MetalTextProgramState *)backend_state;
    MetalTextProgramCommand *command = (MetalTextProgramCommand *)command_state;
    if (!state || !command || !command->command ||
        command->next_cell >= state->program->dispatch.cell_count ||
        dependency_epoch == 0u || completion_epoch <= dependency_epoch ||
        dependency_epoch > command->highest_completion)
        return SALT_TFAIL(-1);
    const SaltTextExecutionCell *cell = &state->program->dispatch.cells[command->next_cell];
    if (command->decode_experts && cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
        if (command->phase != METAL_TEXT_RECORDING_EXPERT) {
            if (metal_text_decode_complete(command, cell->layer) != 0 ||
                metal_text_overlap_launch(state, command, cell->layer) != 0 ||
                metal_text_prepare_resource_request(state, command, cell->layer) != 0)
                return SALT_TFAIL(-1);
            return SALT_TEXT_GPU_NEED_RESOURCE;
        }
        command->arrival_target_layer = UINT32_MAX;
        command->timing_segment = 2u * cell->layer + 1u;
        return SALT_TEXT_GPU_DEPENDENCY_READY;
    }
    if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
        /* Router-generated indirect dimensions must be available before the
         * expert pass begins. This is GPU ordering, not another LOAD_DONE. */
        if (!command->encoder) return SALT_TFAIL(-1);
        [command->encoder updateFence:command->dependency_fence];
        command->dependency_fence_updated = 1u;
        metal_text_close_encoder(command);
        command->arrival_target_layer = UINT32_MAX;
        command->timing_segment = 2u * cell->layer + 1u;
        command->active_layer = cell->layer;
        command->phase = METAL_TEXT_RECORDING_EXPERT;
        command->expert_chain_mask = 0u;
    } else if (cell->kind == SALT_TEXT_CELL_EXPERT_REDUCTION) {
        if (command->expert_chain_mask != 15u) return SALT_TFAIL(-1);
        metal_text_close_encoder(command);
        command->arrival_target_layer = cell->layer + 1u;
        command->timing_segment = 2u * (cell->layer + 1u);
        command->phase = METAL_TEXT_RECORDING_SUFFIX;
        command->active_layer = UINT32_MAX;
        command->expert_chain_mask = 0u;
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
        return SALT_TFAIL(-1);
    args.x = x_offset / sizeof(float);
    args.y = y_offset / sizeof(float);
    args.z = z_offset == SIZE_MAX ? 0u : z_offset / sizeof(float);
    args.rows = rows;
    args.width = width;
    args.aux = aux;
    args.eps = eps;
    args.scalar = scalar;
    int parallel_b1 = command->authoritative && command->input_count == 1u &&
        rows == 1u && (pipeline == _text_rms_rows_pso ||
                      pipeline == _text_residual_rows_pso ||
                      pipeline == _text_router_rows_pso);
    if (!parallel_b1) {
        if (pipeline == _text_rms_rows_pso) pipeline = _text_rms_multi_pso;
        else if (pipeline == _text_residual_rows_pso)
            pipeline = _text_residual_multi_pso;
        else if (pipeline == _text_router_rows_pso)
            pipeline = _text_router_multi_pso;
    }
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:0 atIndex:0];
    [encoder setBuffer:weight ? weight->value :
                (id<MTLBuffer>)state->canonical.backend
                offset:weight ? weight->value_offset : 0 atIndex:1];
    [encoder setBuffer:_op_status offset:0 atIndex:2];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:3];
    }
    if (parallel_b1) {
        NSUInteger threads = MIN((NSUInteger)128,
                                 pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else {
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
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
        return SALT_TFAIL(-1);
    memset(&args, 0, sizeof args);
    args.x = logits_offset / sizeof(float);
    args.rows = rows;
    args.width = vocabulary;
    args.scalar = cap;
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    [encoder setComputePipelineState:_text_softcap_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:0 atIndex:0];
    [encoder setBuffer:_op_status offset:0 atIndex:1];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:2];
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
        return SALT_TFAIL(-1);
    if (plan->final_layer_scale) {
        scalar = metal_text_ref_value(&refs->layer_scalar, 0);
        if (!isfinite(scalar)) return SALT_TFAIL(-1);
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
    encoder = metal_text_encoder(command);
    if (!encoder) return SALT_TFAIL(-1);
    int parallel_b1 = command->authoritative && command->input_count == 1u;
    [encoder setComputePipelineState:parallel_b1 ? _text_combine_pso
                                                : _text_combine_multi_pso];
    [encoder setBuffer:(id<MTLBuffer>)state->canonical.backend
                offset:0 atIndex:0];
    [encoder setBuffer:refs->post_ffn_norm_1.value
                offset:refs->post_ffn_norm_1.value_offset atIndex:1];
    [encoder setBuffer:refs->post_ffn_norm_2.value
                offset:refs->post_ffn_norm_2.value_offset atIndex:2];
    [encoder setBuffer:refs->post_ffn_norm.value
                offset:refs->post_ffn_norm.value_offset atIndex:3];
    [encoder setBuffer:_op_status offset:0 atIndex:4];
    {
        NSUInteger _off = sel_args_put(&args, sizeof args);
        if (_off == (NSUInteger)-1) return SALT_TFAIL(-1);
        [encoder setBuffer:_sel_args_scratch offset:_off atIndex:5];
    }
    if (parallel_b1) {
        NSUInteger threads = MIN((NSUInteger)128,
                                 _text_combine_pso.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:MTLSizeMake(command->input_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else {
        [encoder dispatchThreads:MTLSizeMake(command->input_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
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
    if (!state || state->magic != METAL_TEXT_MAGIC) {
        salt_trace_register(233, 1);
        return SALT_TFAIL(-1);
    }
    if (!command) { salt_trace_register(234, 1); return SALT_TFAIL(-1); }
    if (!command->command) { salt_trace_register(235, 1); return SALT_TFAIL(-1); }
    if (command->phase == METAL_TEXT_IDLE ||
        command->phase == METAL_TEXT_RESOURCE_PENDING ||
        command->phase == METAL_TEXT_SUBMITTED ||
        command->phase == METAL_TEXT_FINISHED) {
        salt_trace_register(229, 1);
        salt_trace_value(229, (int)command->phase);
        return SALT_TFAIL(-1);
    }
    if (state->program != program) { salt_trace_register(236, 1); return SALT_TFAIL(-1); }
    if (command->next_cell >= program->dispatch.cell_count) {
        salt_trace_register(237, 1);
        salt_trace_value(237, (int)command->next_cell);
        return SALT_TFAIL(-1);
    }
    if (cell != &program->dispatch.cells[command->next_cell]) {
        salt_trace_register(238, 1);
        salt_trace_value(238, (int)command->next_cell);
        return SALT_TFAIL(-1);
    }
    if (!assignment) { salt_trace_register(239, 1); return SALT_TFAIL(-1); }
    if (assignment->cpu.count != 0) {
        salt_trace_register(240, 1);
        salt_trace_value(240, (int)assignment->cpu.count);
        return SALT_TFAIL(-1);
    }
    if (assignment->gpu.first != 0) {
        salt_trace_register(241, 1);
        salt_trace_value(241, (int)assignment->gpu.first);
        return SALT_TFAIL(-1);
    }
    if (input_count != command->input_count) {
        salt_trace_register(242, 1);
        salt_trace_value(242, (int)input_count);
        return SALT_TFAIL(-1);
    }
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
    if ((!terminal_output && actual == 0u) || assignment->gpu.count < actual) {
        salt_trace_register(231, 1);
        salt_trace_value(231, (int)actual);
        salt_trace_value(232, (int)assignment->gpu.count);
        return SALT_TFAIL(-1);
    }
    layout = &program->layout;
    if (cell->layer < program->layer_count) {
        compiled = &program->layers[cell->layer];
        source = compiled->descriptor;
        plan = source->plan;
        refs = &state->layers[cell->layer];
    }
    switch (cell->kind) {
    case SALT_TEXT_CELL_EMBEDDING:
        mt_node_tick("metal_text_encode_embedding", __func__, 0u, 1u);
        rc = metal_text_encode_embedding(state, command);
        break;
    case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
        mt_node_tick("metal_text_encode_rows", __func__, 0u, 1u);
        rc = metal_text_encode_rows(state, command, _text_rms_rows_pso,
            &refs->pre_attention_norm, layout->state_a, layout->normalized,
            SIZE_MAX, input_count, program->hidden, 1u,
            program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_QUERY_PROJECTION:
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
        rc = metal_text_encode_projection(state, command, &refs->q,
            layout->normalized, layout->queries, input_count);
        break;
    case SALT_TEXT_CELL_KEY_PROJECTION:
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
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
        mt_node_tick("metal_text_encode_attention_body", __func__, 0u, 1u);
        rc = metal_text_encode_attention_body(state, command, cell->layer);
        break;
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
        rc = metal_text_encode_projection(state, command, &refs->o,
            layout->attention_output, layout->branch, input_count);
        break;
    case SALT_TEXT_CELL_ATTENTION_COMBINE:
        mt_node_tick("metal_text_encode_rows", __func__, 0u, 1u);
        rc = metal_text_encode_rows(state, command, _text_residual_rows_pso,
            &refs->post_attention_norm, layout->state_a, layout->branch,
            layout->state_b, input_count, program->hidden, 0u,
            program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_DENSE_NORM:
        mt_node_tick("metal_text_encode_rows", __func__, 0u, 1u);
        rc = metal_text_encode_rows(state, command, _text_rms_rows_pso,
            &refs->pre_ffn_norm_1, layout->state_b, layout->normalized,
            SIZE_MAX, input_count, program->hidden, 1u,
            program->descriptor->norm_epsilon, 1.0f);
        if (rc == 0)
            mt_node_tick("metal_text_encode_rows", __func__, 0u, 1u);
            rc = metal_text_encode_rows(state, command, _text_rms_rows_pso,
                &refs->pre_ffn_norm_2, layout->state_b, layout->combine_a,
                SIZE_MAX, input_count, program->hidden, 1u,
                program->descriptor->norm_epsilon, 1.0f);
        break;
    case SALT_TEXT_CELL_DENSE_GATE:
        if (command->decode_experts && salt_lr_decode_read_overlap()) {
            command->deferred_dense_layer = cell->layer;
            command->deferred_dense_mask |= 1u;
            _decode_overlap_deferred_cells++;
            rc = 0;
            break;
        }
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
        rc = metal_text_encode_projection(state, command, &refs->dense_gate,
            layout->normalized, layout->dense_gate, input_count);
        break;
    case SALT_TEXT_CELL_DENSE_UP:
        if (command->decode_experts && salt_lr_decode_read_overlap()) {
            command->deferred_dense_layer = cell->layer;
            command->deferred_dense_mask |= 2u;
            _decode_overlap_deferred_cells++;
            rc = 0;
            break;
        }
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
        rc = metal_text_encode_projection(state, command, &refs->dense_up,
            layout->normalized, layout->dense_up, input_count);
        break;
    case SALT_TEXT_CELL_DENSE_ACTIVATION:
        if (command->decode_experts && salt_lr_decode_read_overlap()) {
            command->deferred_dense_layer = cell->layer;
            command->deferred_dense_mask |= 4u;
            _decode_overlap_deferred_cells++;
            rc = 0;
            break;
        }
        mt_node_tick("metal_text_encode_activation", __func__, 0u, 1u);
        rc = metal_text_encode_activation(state, command,
            layout->dense_gate, layout->dense_up, layout->dense_chain,
            (size_t)input_count * (uint32_t)plan->dense_intermediate);
        break;
    case SALT_TEXT_CELL_DENSE_DOWN:
        if (command->decode_experts && salt_lr_decode_read_overlap()) {
            command->deferred_dense_layer = cell->layer;
            command->deferred_dense_mask |= 8u;
            _decode_overlap_deferred_cells++;
            rc = 0;
            break;
        }
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
        rc = metal_text_encode_projection(state, command, &refs->dense_down,
            layout->dense_chain, layout->dense_output, input_count);
        break;
    case SALT_TEXT_CELL_ROUTER_INPUT:
        mt_node_tick("metal_text_encode_rows", __func__, 0u, 1u);
        rc = metal_text_encode_rows(state, command, _text_router_rows_pso,
            &refs->router_scale, layout->state_b, layout->router_input,
            SIZE_MAX, input_count, program->hidden, 0u,
            program->descriptor->norm_epsilon,
            1.0f / salt_sqrtf((float)program->hidden));
        break;
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
        mt_node_tick("metal_text_encode_projection", __func__, 0u, 1u);
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
                (uint32_t)plan->n_experts, (uint32_t)plan->top_k_experts,
                (uint32_t)plan->expert_intermediate);
        break;
    case SALT_TEXT_CELL_EXPERT_GATE:
        command->expert_chain_mask |= 1u; rc = 0; break;
    case SALT_TEXT_CELL_EXPERT_UP:
        command->expert_chain_mask |= 2u; rc = 0; break;
    case SALT_TEXT_CELL_EXPERT_ACTIVATION:
        command->expert_chain_mask |= 4u; rc = 0; break;
    case SALT_TEXT_CELL_EXPERT_DOWN:
        if (command->expert_chain_mask != 7u) {
            salt_trace_register(227, 1);
            salt_trace_value(227, (int)command->expert_chain_mask);
            salt_trace_value(228, (int)command->active_layer);
            return SALT_TFAIL(-1);
        }
        mt_node_tick("metal_text_encode_expert_chain", __func__, 0u, 1u);
        rc = metal_text_encode_expert_chain(state, command);
        break;
    case SALT_TEXT_CELL_EXPERT_REDUCTION:
        mt_node_tick("metal_text_encode_expert_reduce", __func__, 0u, 1u);
        rc = metal_text_encode_expert_reduce(state, command,
            (uint32_t)plan->top_k_experts);
        break;
    case SALT_TEXT_CELL_FFN_COMBINE:
        mt_node_tick("metal_text_encode_combine", __func__, 0u, 1u);
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
        return SALT_TFAIL(-1);
    }
    if (rc != 0) return SALT_TFAIL(-1);
    if (command->decode_experts && cell->kind == SALT_TEXT_CELL_FFN_COMBINE)
        metal_text_close_encoder(command);
    if (!command->decode_experts && cell->kind == SALT_TEXT_CELL_FFN_COMBINE) {
        /* The GPU reports only its own completed layer. Which bank/load that
         * enables belongs to the engine-owned rotation schedule. */
        const uint64_t done_value =
            command->rotation_base + cell->layer + 1u;
        metal_text_close_encoder(command);
        [command->command encodeSignalEvent:_rotation_done_event
            value:done_value];
        /* On this Metal implementation, several signal-only values in one
         * command buffer become host-visible as one later wave. Consuming the
         * value immediately after its signal preserves one submission while
         * making DONE(i) visible before any layer i+1 pass can begin. */
        [command->command encodeWaitForEvent:_rotation_done_event
            value:done_value];
        _rotation_done_value_yields++;
        command->rotation_done_signals++;
        /* Dependencies and arithmetic stay layer-causal. Only pair boundaries
         * receive the next complete block; there is no load wait inside it. */
        if (cell->layer + 1u < state->layer_count &&
            (cell->layer + 1u) % command->rotation_block_layers == 0u) {
            uint32_t next_end = cell->layer + 1u + command->rotation_block_layers;
            if (next_end > state->layer_count) next_end = state->layer_count;
            [command->command encodeWaitForEvent:_rotation_ready_event
                value:command->rotation_base + next_end];
            command->rotation_ready_waits++;
        }
    }
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
        command->encoded_cells != state->program->dispatch.cell_count) {
        /* THE RECORD NAMES WHERE THE RECORDING STOPPED.  The submit requires every cell; when it refuses,
         * these two numbers and the phase say which cell the walk stopped at. */
        salt_trace_register(153, 1);
        salt_trace_value(153, (int)command->next_cell);
        salt_trace_value(154, (int)command->encoded_cells);
        salt_trace_value(155, (int)command->phase);
        return SALT_TFAIL(-1);
    }
    metal_text_close_encoder(command);
    _rotation_stream_error = 0;
    if (_text_probe_enabled && !command->decode_experts) salt_lr_io_snapshot(&command->phase_io_begin);
    if (!command->decode_experts) {
        if (salt_lr_stream_begin(command->rotation_base, metal_rotation_wait_done,
                metal_rotation_publish_ready, command) != 0)
            return SALT_TFAIL(-1);
    }
    salt_lr_timeline_block(command->rotation_base,
        command->rotation_block_layers, command->rotation_ready_waits,
        command->rotation_done_signals);
    command->submit_t0 = _mt_probe_ns();
    [command->command commit];
    command->submit_tc = _mt_probe_ns();
    salt_lr_timeline_submit(command->rotation_base, command->input_count,
        (uint64_t)command->submit_tc);
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
        return SALT_TFAIL(-1);
    [current waitUntilCompleted];
    int overlap_rc = metal_text_overlap_collect(command);
    if (metal_command_succeeded(current) != 0) salt_lr_stream_cancel();
    int rotation_rc = command->decode_experts ? 0 : salt_lr_stream_finish();
    salt_lr_timeline_complete(command->rotation_base, (uint64_t)_mt_probe_ns());
    if (_text_probe_enabled && _text_probe_n < METAL_TEXT_PROBE_MAX) {
        double end = _mt_probe_ns();
        uint64_t ready_ns = salt_lr_last_ready_ns();
        _text_probe[_text_probe_n].completion_tail_valid = ready_ns && end >= (double)ready_ns;
        _text_probe[_text_probe_n].completion_tail = _text_probe[_text_probe_n].completion_tail_valid
            ? (float)((end - (double)ready_ns) * 1e-6) : 0.0f;
        _text_probe[_text_probe_n].wall = (float)((end - command->submit_t0) * 1e-6);
        _text_probe[_text_probe_n].commit = (float)((command->submit_tc - command->submit_t0) * 1e-6);
        _text_probe[_text_probe_n].gpu = (float)(([current GPUEndTime] - [current GPUStartTime]) * 1e3);
        _text_probe[_text_probe_n].rows = command->input_count;
        _text_probe[_text_probe_n].layer = UINT32_MAX;
        _text_probe[_text_probe_n].phase = command->phase;
        _text_probe[_text_probe_n].operation_phase = (uint32_t)(command->rotation_base / ((uint64_t)_arrival_layer_count + 2u)) + 1u;
        _text_probe[_text_probe_n].operation_kind = command->decode_experts ? 2u : 1u;
        _text_probe[_text_probe_n].phase_io_record = 1u;
        SaltRotationIoSnapshot io_end;
        salt_lr_io_snapshot(&io_end);
        const SaltRotationIoSnapshot *io_start = &command->phase_io_begin;
        uint32_t io_valid = io_start->valid & io_end.valid;
        _text_probe[_text_probe_n].io_begin_ns = io_start->sample_end_ns;
        _text_probe[_text_probe_n].io_end_ns = io_end.sample_begin_ns;
        _text_probe[_text_probe_n].io_sample_ns =
            io_start->sample_end_ns - io_start->sample_begin_ns +
            io_end.sample_end_ns - io_end.sample_begin_ns;
        _text_probe[_text_probe_n].read_bytes_valid =
            (io_valid & 1u) && io_end.read_bytes >= io_start->read_bytes;
        _text_probe[_text_probe_n].write_bytes_valid =
            (io_valid & 2u) && io_end.write_bytes >= io_start->write_bytes;
        _text_probe[_text_probe_n].major_faults_valid =
            (io_valid & 4u) && io_end.major_faults >= io_start->major_faults;
        _text_probe[_text_probe_n].read_bytes = _text_probe[_text_probe_n].read_bytes_valid
            ? io_end.read_bytes - io_start->read_bytes : 0;
        _text_probe[_text_probe_n].write_bytes = _text_probe[_text_probe_n].write_bytes_valid
            ? io_end.write_bytes - io_start->write_bytes : 0;
        _text_probe[_text_probe_n].major_faults = _text_probe[_text_probe_n].major_faults_valid
            ? io_end.major_faults - io_start->major_faults : 0;
        _text_probe_n++;
    }
    command->command = nil;
    if (overlap_rc != 0 || rotation_rc != 0 || _rotation_stream_error || metal_command_succeeded(current) != 0 ||
        *(uint32_t *)[_op_status contents] != 0u) {
        NSError *command_error = [current error];
        salt_trace_register(88, 1);
        salt_trace_value(88, (int)[current status]);
        salt_trace_register(89, 1);
        salt_trace_value(89,
            (overlap_rc != 0 ? 1 : 0) |
            (rotation_rc != 0 ? 2 : 0) |
            (_rotation_stream_error ? 4 : 0) |
            (metal_command_succeeded(current) != 0 ? 8 : 0) |
            ((int)(*(uint32_t *)[_op_status contents]) << 8));
        mt_node_tick("FINISH_FAIL", "metal_text_finish",
            (uint32_t)[current status], *(uint32_t *)[_op_status contents]);
        if (command_error)
            mt_node_tick("FINISH_ERROR", "metal_text_finish",
                (uint32_t)[command_error code], 0u);
        if (_cfg_diag) {   /* fixed configuration, resolved once */
            NSError *error = command_error;
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
        return SALT_TFAIL(-1);
    }
    metal_text_record_gpu_time(command, current);
    [current release];
    command->phase = METAL_TEXT_FINISHED;
    /* TOKEN BOUNDARY: rotate the ledger epoch.  Entries registered during this
     * token become usable by the NEXT token only. */
    _text_ledger_prev_epoch = _text_ledger_cur_epoch;
    _text_ledger_cur_epoch++;
    command->stats.final_logits_transfer_bytes =
        (uint64_t)(command->authoritative
            ? command->authoritative_output_rows : command->input_count) *
        state->program->vocabulary *
        sizeof(float);
    view->canonical_base = (unsigned char *)state->canonical.contents;
    view->canonical_bytes = state->canonical_bytes;
    view->backend_published_kv =
        command->authoritative && command->input_count == 1u;
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
        return SALT_TFAIL(-1);
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
        !scrubbed_bytes) return SALT_TFAIL(-1);
    if (command->command) {
        id<MTLCommandBuffer> current = command->command;
        metal_text_close_encoder(command);
        command->command = nil;
        if (command->phase == METAL_TEXT_SUBMITTED) {
            salt_lr_stream_cancel();
            [_rotation_ready_event setSignaledValue:UINT64_MAX];
            [current waitUntilCompleted];
            (void)salt_lr_stream_finish();
            if (metal_command_succeeded(current) != 0) {
                [current release];
                return SALT_TFAIL(-1);
            }
        }
        [current release];
    }
    if (metal_text_overlap_collect(command) != 0) return SALT_TFAIL(-1);
    {
        uint64_t cleared = 0;
        if (command->touched_span_count == 0 ||
            salt_text_touched_span_clear(state->canonical.contents,
                state->canonical_bytes, command->touched_spans,
                command->touched_span_count, &cleared) != 0)
            return SALT_TFAIL(-1);
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
        return SALT_TFAIL(-1);
    if (salt_gpu_shared_buffer_free(&state->canonical) != 0) return SALT_TFAIL(-1);
    metal_release(state->dependency_fence);
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

/* THE SLOT TABLE THE KERNELS RESOLVE THROUGH IS THE ROTATION'S TO FILL.
 *
 * Every selected-expert kernel resolves its expert with `logical_slots[logical]` and treats a negative
 * entry as a miss, writing op_status = 2 (gpu_metal.m:904/945/999).  Nothing populated that table under
 * the rotation, so the first expert the prefill reached resolved to a miss, the command buffer was
 * poisoned, metal_text_complete_command returned -1, dependency_barrier returned -1 and the driver
 * stopped - which is the whole chain the ccb record shows.  The rotator calls this as each expert's bytes
 * land, so the table carries the slot the map fixed at initiation. */
void salt_gpu_text_register_expert(int layer, int expert, int n_experts, int slot) {
    int32_t logical;
    if (!_text_ledger_slot_host || n_experts <= 0) return;
    logical = layer * n_experts + expert;
    if (logical < 0 || logical >= (int32_t)_text_ledger_logical_capacity) return;
    /* THE LEDGER, NOT THE BIND'S TABLE.  metal_text_dependency_barrier reads
     * _text_ledger_slot_host to decide residency (:7015/:7016) and asks for the whole layer with
     * NEED_RESOURCE when any entry is negative - that is the table the rotation has to fill.  The bind's
     * table (_selected_logical_slot_host) is the bind's own and it refuses any entry that is not -1
     * (:1550). */
    _text_ledger_slot_host[logical] = (int32_t)slot;
}
