#ifndef SALT_GPU_H
#define SALT_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SaltTextGpuProgramOps;

/* Optional GPU acceleration (macOS Metal; stub elsewhere).
 *
 * The engine's default path is CPU-only and byte-deterministic; GPU
 * mode is opt-in (--gpu / SALT_GPU=1) and accelerates the output-head
 * MLX-4bit matvec (the single largest per-token matvec: 248320 x 2048).
 * GPU logits are float32-accumulated like the CPU SIMD path but the
 * reduction order differs, so logits may differ by float32 rounding --
 * acceptable for sampling, NOT for byte-identical determinism checks
 * (keep those on the CPU path).
 *
 * Functions return 0 on success and -1 on unavailable/failed. Ordinary
 * optional operations may fall back to CPU; explicit mapped residency and
 * cleanup callers must propagate failure instead. */
int  salt_gpu_init(void);
int  salt_gpu_free(void);

/* y = MLX-4bit_matvec(vals, scales, biases, R, C, x) -- identical
 * layout to salt_q4_matvec: U32 packed nibbles (8 per word),
 * per-64-element BF16 scale (+ optional BF16 bias). R rows, C cols;
 * y must hold R floats. Returns 0 if computed, -1 if not (caller
 * should run the CPU matvec). */
int salt_gpu_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y);
int salt_gpu_q8_matvec(const uint32_t *vals, const uint16_t *scales,
                       const uint16_t *biases, int R, int C,
                       const float *x, float *y);

/* Compressed-tensors NVFP4 primitives. Q/DQ uses dynamic group-16 input
 * scaling with correctly rounded operations. Projection requires a registered
 * source-backed NVFP4 tensor key. */
int salt_gpu_nvfp4_qdq(const float *input, int C, float input_global_scale,
                       uint8_t *packed, uint8_t *scales, float *dequant);
int salt_gpu_nvfp4_matvec(const void *key, int R, int C,
                          const float *input, float *output);
/* One source tensor, B contiguous input rows [B,C], B contiguous outputs
 * [B,R]. C retains task/dependency ownership; the backend performs one exact
 * Q/DQ launch and one exact projection launch for the complete prompt batch. */
int salt_gpu_nvfp4_matvec_batch(const void *key, int R, int C, int B,
                                const float *inputs, float *outputs);
int salt_gpu_nvfp4_batch(const void *const *keys,
                         const float *const *inputs, float *const *outputs,
                         const int *batches, int R, int C, int njobs);
int salt_gpu_nvfp4_mixed_batch(const void *const *keys,
                               const float *const *inputs,
                               float *const *outputs, const int *batches,
                               const int *rows, int C, int njobs);
int salt_gpu_bf16_resource_slot(uint32_t kind, uint32_t resource_id,
                                const void *key, const uint16_t *weights,
                                int R, int C);
int salt_gpu_bf16_matvec(const void *key, int R, int C,
                         const float *input, float *output);
/* One immutable BF16 matrix over B contiguous FP32 input rows. Each output
 * row preserves the portable-C serial column accumulation order. */
int salt_gpu_bf16_matvec_batch(const void *key, int R, int C, int B,
                               const float *inputs, float *outputs);

/* Batched expert matvec: y[j] = MLX4_matvec(vals[j], scales[j],
 * biases[j], R, C, xs[j]) for j in [0, njobs). All jobs share R, C;
 * each job has its OWN x (the shared latent for gate/up, the per-
 * expert chain for down).
 *
 * ids[j] is a STABLE identity for job j's tensor (e.g. the expert
 * layout pointer). Weight buffers are cached per id in a growing
 * arena (first sight = one copy; thereafter zero per-token copies) --
 * the vals POINTER alone is NOT a stable key: the engine's cache
 * slots rotate, so the same expert sits at a different address every
 * token. The whole batch is ONE dispatch -- the design the
 * microbenchmark measured at 13x CPU. Returns 0 if computed, -1 to
 * fall back to the CPU path (Metal unavailable, arena full, OOM). */
int salt_gpu_q4_batch(const uint32_t *const *vals,
                        const uint16_t *const *scales,
                        const uint16_t *const *biases,
                        const float *const *xs, float *const *ys,
                        const void *const *ids,
                        const int *Rj, int C, int njobs);
int salt_gpu_q8_batch(const uint32_t *const *vals,
                      const uint16_t *const *scales,
                      const uint16_t *const *biases,
                      const float *const *xs, float *const *ys,
                      const void *const *ids,
                      const int *Rj, int C, int njobs);

/* Require every batch key to resolve to an explicitly registered
 * source-backed resource. While enabled, misses return -1 and must be
 * propagated; they never copy weights into the legacy arena. */
void salt_gpu_set_mapped_only(int on);
int salt_gpu_mapped_only(void);

typedef struct SaltGpuBatchStats {
    uint64_t arena_batches;
    uint64_t trunk_batches;
    uint64_t expert_layer_batches;
    uint64_t mapped_only_misses;
    uint64_t direct_output_batches;
    uint64_t direct_output_jobs;
} SaltGpuBatchStats;

typedef struct SaltGpuBatchStatsV2 {
    uint64_t arena_batches;
    uint64_t trunk_batches;
    uint64_t expert_layer_batches;
    uint64_t mapped_only_misses;
    uint64_t direct_output_batches;
    uint64_t direct_output_jobs;
    /* Successful completed NVFP4 operations.  Byte fields are logical source
     * and host-transfer extents, not physical memory traffic or residency. */
    uint64_t nvfp4_logical_packed_weight_bytes;
    uint64_t nvfp4_logical_weight_scale_bytes;
    uint64_t nvfp4_activation_input_bytes;
    uint64_t nvfp4_activation_qdq_bytes;
    uint64_t nvfp4_projection_output_bytes;
    uint64_t nvfp4_kernel_launches;
    uint64_t nvfp4_completion_fences;
    uint64_t nvfp4_h2d_ns;
    uint64_t nvfp4_qdq_ns;
    uint64_t nvfp4_projection_ns;
    uint64_t nvfp4_d2h_ns;
    uint64_t q4_weight_stationary_batches;
} SaltGpuBatchStatsV2;

void salt_gpu_batch_stats_get(SaltGpuBatchStats *stats);
int salt_gpu_batch_stats_get_v2(SaltGpuBatchStatsV2 *stats, size_t stats_size);

/* The pre-recorded dispatch interface: the engine's GPU batch is a
 * MANAGED dispatch unit, never an inline encode. Each backend's
 * allocator differs (MTLIndirectCommandBuffer vs cudaGraphCreate vs
 * hipGraph) -- the module owns it; the engine calls replay(). */
typedef struct SaltGpuBatchDesc {
    const uint32_t *const *vals;
    const uint16_t *const *scales;
    const uint16_t *const *biases;
    const float *const *xs;
    float *const *ys;
    const void *const *ids;
    const int *Rj;
    int C, njobs;
} SaltGpuBatchDesc;

typedef struct SaltGpuDispatch {
    const char *name;                  /* "direct", "metal-icb", ... */
    int (*init)(void **state);         /* the backend's allocator */
    int (*replay)(void *state, const SaltGpuBatchDesc *d);
    void (*destroy)(void *state);
} SaltGpuDispatch;

/* The self-aware selector: SALT_GPU_DISPATCH may select direct Metal or
 * the fail-closed metal-icb placeholder. Auto returns no backend when a
 * CUDA runtime is present (the CUDA module is pending); otherwise it uses
 * direct Metal until ICB command recording is qualified. */
const SaltGpuDispatch *salt_dispatch_get(void);

/* the CUDA-runtime probe: 1 when the CUDA runtime is dlopen-able on
 * this host (the cuda dispatch module's gate); the non-Darwin stub
 * returns 0. */
int salt_gpu_cuda_present(void);
/* Nonzero only for the HIP/ROCm backend. */
int salt_gpu_rocm_present(void);
/* Nonzero only when the active backend uses pageable host-page-table weight
 * mappings. Model recipes may choose a different activation-seat policy. */
int salt_gpu_pageable_mmap_active(void);

/* the trunk's zero-copy residency: wrap the trunk's mmap -- the
 * proj-batch's first sight registers tensor OFFSETS into the wrapped
 * map instead of copying into the arena (the vLLM-style residency:
 * the weights live in the unified memory once, the GPU references
 * them). The pool's wrap has the same API. */
int salt_gpu_resident_trunk_map(const void *base, size_t nbytes);
/* Synchronize all readers, forget trunk slots, and release the no-copy
 * device view before the caller drops the host mapping. */
int salt_gpu_resident_trunk_unmap(void);

/* the deferred-sync batch mode: the engine commits several batches
 * (a layer's projections) without waiting, then syncs ONCE at the
 * consumer boundary -- the per-call wait amortized over the group
 * (the 4 projections of a GQA layer: 4 commits, 1 wait). Disabling defer
 * synchronizes pending work and returns its completion status. */
int salt_gpu_set_defer(int on);
int salt_gpu_defer(void);
int salt_gpu_sync(void);

/* CPU-visible storage that is also a backend buffer. The engine allocates
 * the canonical output once, derives every job's final float offset from
 * the existing index, and gives that exact allocation to both executors.
 * No Objective-C type crosses this ABI. */
typedef struct SaltGpuSharedBuffer {
    void *contents;
    size_t nbytes;
    void *backend;
} SaltGpuSharedBuffer;

int salt_gpu_shared_buffer_alloc(SaltGpuSharedBuffer *buffer, size_t nbytes);
int salt_gpu_shared_buffer_free(SaltGpuSharedBuffer *buffer);

/* Backend realization of the engine-owned immutable TensorOps program. NULL is
 * a fail-closed unsupported backend; the model never installs this interface. */
const struct SaltTextGpuProgramOps *salt_gpu_tensor_program_ops(void);

typedef enum SaltGpuExactCellKind {
    SALT_GPU_EXACT_RMSNORM = 1,
    SALT_GPU_EXACT_TOPK = 2,
    SALT_GPU_EXACT_RESIDUAL_POSTNORM = 3,
    SALT_GPU_EXACT_PARALLEL_COMBINE = 4,
    SALT_GPU_EXACT_SOFTCAP = 5,
    SALT_GPU_EXACT_ROUTER_INPUT = 6
} SaltGpuExactCellKind;

/* Model-neutral exact FP32 cells over startup-owned shared buffers. Offsets are
 * element offsets: float elements for every view except TOPK view 2, whose
 * selected indices are int elements. The backend encodes the whole sequence in
 * order into one command buffer and performs one status-bearing completion. */
typedef struct SaltGpuExactCell {
    SaltGpuExactCellKind kind;
    SaltGpuSharedBuffer *views[8];
    size_t offsets[8];
    uint32_t n;
    uint32_t aux;
    float eps;
    float scalar;
} SaltGpuExactCell;

int salt_gpu_exact_cells(const SaltGpuExactCell *cells, int count);

/* Exact prefill attention body over startup-owned shared buffers. Q/K/V are
 * already normalized and rotated. Each (token,head) preserves the portable-C
 * serial dot, softmax, and value-reduction order. */
int salt_gpu_attention_batch(
    int full_attention, int n_heads, int n_kv_heads, int head_dim, int window,
    const SaltGpuSharedBuffer *queries, size_t query_float_offset,
    const SaltGpuSharedBuffer *kv, size_t key_float_offset,
    size_t value_float_offset, int start_position, int batch,
    int query_stride, int kv_stride, SaltGpuSharedBuffer *outputs,
    size_t output_float_offset);
/* Allocate the backend-owned device mirror for one fixed canonical KV arena.
 * Must run at model initialization before any attention submission. */
int salt_gpu_attention_prepare(const SaltGpuSharedBuffer *kv);
/* Bind one fixed layer's key/value ranges inside the startup-owned KV arena. */
int salt_gpu_attention_bind_kv(size_t key_float_offset,
                               size_t value_float_offset, int kv_stride);
/* Explicit state boundaries only. Materialize copies committed device-live rows
 * into the existing host KV; import populates device-live rows from it. */
int salt_gpu_attention_materialize(SaltGpuSharedBuffer *kv, int position,
                                   uint64_t *copied_bytes);
int salt_gpu_attention_import(SaltGpuSharedBuffer *kv, int position,
                              uint64_t *copied_bytes);
int salt_gpu_attention_rewind(int position);

/* Exact Q/K/V post-projection transform. Q remains staged for the immediately
 * following attention call; transformed K/V remain in backend-live KV until an
 * explicit engine materialization transaction. */
int salt_gpu_attention_transform(
    int n_heads, int n_kv_heads, int head_dim, int rope_dim,
    int start_position, int batch, int query_stride, int kv_stride, float eps,
    SaltGpuSharedBuffer *queries, size_t query_float_offset,
    SaltGpuSharedBuffer *keys, size_t key_float_offset,
    SaltGpuSharedBuffer *values, size_t value_float_offset,
    const float *q_weight, const float *k_weight,
    const float *cosines, const float *sines, int rope_pairs,
    SaltGpuSharedBuffer *kv, size_t key_cache_float_offset,
    size_t value_cache_float_offset);

typedef struct SaltGpuMoeExpert {
    const uint32_t *gate_vals;
    const uint16_t *gate_scales;
    const uint16_t *gate_biases;
    const void *gate_id;
    const uint32_t *up_vals;
    const uint16_t *up_scales;
    const uint16_t *up_biases;
    const void *up_id;
    const uint32_t *down_vals;
    const uint16_t *down_scales;
    const uint16_t *down_biases;
    const void *down_id;
    int group;
    int resource_slot; /* fixed expert-cache slot, or -1 for command view */
    /* Canonical model-supplied selected-resource identity. Backends must
     * compare this exact value with the resource currently bound to
     * resource_slot before constructing a command descriptor. */
    uint64_t logical_resource_id;
} SaltGpuMoeExpert;

typedef struct SaltGpuSelectedProjectionJob {
    const uint32_t *vals;
    const uint16_t *scales;
    const uint16_t *biases;
    const void *id;
    const float *inputs;
    float *outputs;
    int rows;
    int cols;
    int batch;
    int resource_slot;
    uint64_t logical_resource_id;
} SaltGpuSelectedProjectionJob;

/* Execute heterogeneous selected-expert projections directly inside one
 * engine-owned shared arena. Inputs and outputs must resolve to disjoint,
 * preassigned ranges of arena; the backend performs no output scatter. */
int salt_gpu_q4_selected_shared(
    const SaltGpuSelectedProjectionJob *jobs, int job_count,
    SaltGpuSharedBuffer *arena);

/* Complete routed expert projection chain. Inputs and outputs are grouped by
 * expert in the same canonical order as experts[]. */
int salt_gpu_q4_moe_chain(const SaltGpuMoeExpert *experts, int count,
                          int hidden, int routed,
                          const float *inputs, float *outputs,
                          float *gate_scratch, float *up_scratch,
                          size_t scratch_float_capacity);
int salt_gpu_q4_moe_chain_selected(const SaltGpuMoeExpert *experts, int count,
                                   int hidden, int routed,
                                   const float *inputs, float *outputs,
                                   float *gate_scratch, float *up_scratch,
                                   size_t scratch_float_capacity);
/* Fixed selected-expert resource table. prepare() allocates only bounded
 * backend metadata at engine start. The ordinary bind retains existing
 * host/unified addressability. Device-local residency first populates a fixed
 * seat through gpu_residency, then publishes that resolved address through the
 * resident bind. Both follow the authoritative cache READY/EMPTY lifecycle. */
int salt_gpu_selected_resources_prepare(int capacity, int logical_capacity);
int salt_gpu_selected_resource_bind(int slot, uint64_t logical_resource_id,
                                    const void *base, size_t nbytes,
                                    const void *payload);
int salt_gpu_selected_resource_bind_resident(
    int slot, uint64_t logical_resource_id,
    const void *payload, size_t nbytes,
    uintptr_t device_address, uint64_t generation);
/* Complete all physical work that may consume selected resources. Cache
 * retirement may call this once before unbinding a complete victim batch. */
int salt_gpu_selected_resources_fence(void);
int salt_gpu_selected_resource_unbind(int slot,
                                      uint64_t logical_resource_id);

/* Submit a contiguous range of complete projection jobs asynchronously.
 * xs is the canonical [B][C] input allocation. Job b writes directly to
 * output[y_float_offset + b*R .. + (b+1)*R); first_job/job_count merely
 * choose the disjoint jobs assigned to the GPU. salt_gpu_sync() is the
 * publication fence; it never looks up or scatters output destinations. */
int salt_gpu_proj_batch_indexed(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *id);
/* Same indexed destination contract, with exact value/scale/bias no-copy
 * views for one homogeneous tensor. The backend holds those views through
 * completion; canonical source bytes and output ownership are unchanged. */
int salt_gpu_proj_batch_indexed_exact_views(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *id);

/* Heterogeneous complete-job scheduler. The GPU owns [gpu_first_job,
 * gpu_first_job+gpu_jobs); CPU owns the contiguous complement. Both write
 * the canonical shared output directly, and the caller fences at its first
 * consumer boundary. */
int salt_gpu_mixed_proj_batch(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int gpu_first_job, int gpu_jobs, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *id);

/* Projection adapter (S0/S1 of the prefill-offload scope): one
 * weight x B tokens. Equivalent to B salt_gpu_q4_batch jobs that
 * share the SAME weight (and stable id) but distinct per-token
 * x/y buffers. Used by the chunked-prefill paths to offload the
 * batched qkv/z/o projections to the GPU. Falls back to -1 exactly
 * like the batch API. id must be a stable tensor identity (the
 * trunk layout entry). */
int salt_gpu_proj_batch(const uint32_t *vals, const uint16_t *scales,
                        const uint16_t *biases, int R, int C, int B,
                        const float *const *xs, float *const *ys,
                        const void *id);

/* Migration compatibility registration: register a tensor only within the
 * currently source-backed expert map. It never creates an arena copy. */
int salt_gpu_resident_slot(const void *key, const uint32_t *vals,
                           const uint16_t *scales, const uint16_t *biases,
                           int R, int C);

/* Command-scoped expert resource API. The portable engine assigns an
 * immutable resource_id, registers only local tensor offsets within that
 * resource, executes, then unbinds before releasing the host mapping.
 * unbind must not return until all backend reads of base have completed. */
int salt_gpu_expert_resource_bind(uint32_t resource_id,
                                  const void *base, size_t nbytes);
int salt_gpu_expert_resource_slot(uint32_t resource_id, const void *key,
                                  const uint32_t *vals,
                                  const uint16_t *scales,
                                  const uint16_t *biases, int R, int C);
int salt_gpu_expert_resource_unbind(uint32_t resource_id);

typedef enum SaltGpuWeightAddressability {
    SALT_GPU_WEIGHT_ADDRESS_AUTO = 0,
    SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW = 1,
    SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT = 2,
    SALT_GPU_WEIGHT_ADDRESS_PAGEABLE = 3
} SaltGpuWeightAddressability;

typedef struct SaltGpuWeightResourceUsage {
    uint64_t described_bytes;
    uint64_t registered_bytes;
    uint64_t pageable_bytes;
    uint64_t active_window_bytes;
    uint64_t peak_window_bytes;
    uint64_t device_copied_weight_bytes;
    uint32_t described_resources;
    uint32_t active_resources;
    uint32_t active_windows;
    uint32_t reserved;
} SaltGpuWeightResourceUsage;

typedef struct SaltGpuStartupRequirements {
    uint64_t fixed_device_bytes;
    uint64_t fixed_pinned_host_bytes;
    uint64_t descriptor_bytes;
    uint64_t device_copied_weight_bytes;
    uint32_t max_output_floats;
    uint32_t max_batch_jobs;
    uint32_t shared_buffer_slots;
    uint32_t reserved;
} SaltGpuStartupRequirements;

/* Pure capacity query: performs no backend initialization or allocation. */
int salt_gpu_startup_requirements(SaltGpuStartupRequirements *requirements,
                                  size_t requirements_size);

/* Resource identity and device addressability are separate lifecycle steps.
 * describe() records canonical host bytes only. activate() applies an explicit
 * backend policy. Bounded windows are independent leases held through the
 * completion fence. Learned-weight copy bytes must remain zero. */
int salt_gpu_weight_resource_describe(uint32_t kind, uint32_t resource_id,
                                      const void *base, size_t nbytes);
int salt_gpu_weight_resource_activate(uint32_t kind, uint32_t resource_id,
                                      SaltGpuWeightAddressability policy);
/* Bind one already-described canonical source as a process-lifetime no-copy
 * resource in the backend's fixed shared resource table. This creates no
 * learned-weight payload copy and does not force source pages resident. */
int salt_gpu_weight_resource_pool_bind(uint32_t kind, uint32_t resource_id);
int salt_gpu_weight_resource_deactivate(uint32_t kind, uint32_t resource_id);
int salt_gpu_weight_resource_usage(SaltGpuWeightResourceUsage *usage,
                                   size_t usage_size);

/* Compatibility wrapper: describe + backend-default activation. New
 * production wiring uses the explicit lifecycle above. bits is 4 or 8;
 * backends retain 64-bit byte offsets internally. */
int salt_gpu_weight_resource_bind(uint32_t kind, uint32_t resource_id,
                                  const void *base, size_t nbytes);
int salt_gpu_weight_window_bind(uint32_t kind, uint32_t resource_id,
                                uint64_t offset, size_t nbytes);
int salt_gpu_weight_window_unbind(uint32_t kind, uint32_t resource_id);
int salt_gpu_weight_resource_slot(uint32_t kind, uint32_t resource_id,
                                  const void *key, const uint32_t *vals,
                                  const uint16_t *scales,
                                  const uint16_t *biases,
                                  int bits, int R, int C);
int salt_gpu_nvfp4_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    const float *weight_global_scale, const float *input_global_scale,
    int R, int C);
int salt_gpu_weight_resource_unbind(uint32_t kind, uint32_t resource_id);

/* Migration-only whole-pool wrapper. Production uses bounded expert-layer
 * resources derived from the canonical CPU index. No arena fallback. */
int salt_gpu_resident_pool_map(const void *base, size_t nbytes);

#ifdef __cplusplus
}
#endif

#endif /* SALT_GPU_H */
