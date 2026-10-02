#define _POSIX_C_SOURCE 200809L

#include "gemma4_text.h"
#include "gemma4.h"
#include "salt/area_scan.h"
#include "salt/gpu.h"
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#include "gemma4_operation.h"
#include "salt/gpu_residency.h"
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
#include "salt/gpu_resource.h"
#include "salt/attn.h"
#include "salt/attn_batch.h"
#include "salt/bitmath.h"
#include "salt/kernels.h"
#include "salt/model.h"
#include "salt/moe_group.h"
#include "salt/nvfp4.h"
#include "salt/salt.h"
#include "salt/text_exec.h"
#include "salt/text_token.h"
#include "salt/text_verify.h"
#include "salt/tokenizer.h"
#include "sha256.h"
#include "thread-lifecycle.h"

#include <errno.h>
#include <fcntl.h>

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define G4_LAYERS 30
#define G4_MAX_CONTEXT 262144
#define G4_HIDDEN 2816
#define G4_VOCAB 262144
#define G4_EXPERTS 128
#define G4_TOPK 8
#define G4_DENSE 2112
#define G4_ROUTED 704
#define G4_MAX_Q 8192
#define G4_MAX_KV 2048
#define G4_MAX_MAPS 4
#define G4_TOKEN_PROGRAM_CELLS (1 + 2 * G4_LAYERS + 4)
#define G4_BF16_VECTOR_BINDINGS (1 + 12 * G4_LAYERS)
typedef char G4IntFloatSizeMustMatch[
    sizeof(int) == sizeof(float) ? 1 : -1
];
#define G4_CUDA_BATCH_MAX 512
#define G4_CUDA_HETERO_JOBS 4096

#define G4_EXPECTED_BINDINGS 597
#define G4_TEXT_BINDINGS \
    (G4_EXPECTED_BINDINGS + G4_LAYERS * G4_EXPERTS * 3)
#define G4_TEXT_RESOURCES \
    (G4_MAX_MAPS + G4_BF16_VECTOR_BINDINGS + 1)
#define G4_EPS 0.000001f
#define G4_LOGIT_CAP 30.0f
#define G4_EMBED_SCALE 53.0f
#define G4_IMAGE_TOKEN 258880
#define G4_BOI_TOKEN 255999
#define G4_EOI_TOKEN 258882
#define G4_SLIDING_WINDOW 1024
#define G4_POOL_HEADER 24u
#define G4_WEIGHT_BYTES ((size_t)G4_HIDDEN * G4_ROUTED / 2u)
#define G4_SCALE_BYTES \
    ((size_t)G4_HIDDEN * G4_ROUTED / 64u * sizeof(uint16_t))
#define G4_BIAS_OFFSET (G4_WEIGHT_BYTES + G4_SCALE_BYTES)
#define G4_PROJ_BYTES (G4_WEIGHT_BYTES + 2u * G4_SCALE_BYTES)
#define G4_SLOT_BYTES (3u * G4_PROJ_BYTES)
#define G4_EXPERT_BUDGET_UNIT 1000000000ULL
#define G4_KV_SNAPSHOT_VERSION 3u
#define G4_KV_TRANSFER_VERSION 6u
#define G4_KV_TRANSFER_HEADER 256u

#ifndef SALT_GEMMA4_KV_COMPAT_SHA256
#error "SALT_GEMMA4_KV_COMPAT_SHA256 is required"
#endif
#ifndef SALT_GEMMA4_KV_LEGACY_V5_SHA256
#define SALT_GEMMA4_KV_LEGACY_V5_SHA256 ""
#endif
#ifndef SALT_GEMMA4_KV_LEGACY_V4_SHA256
#define SALT_GEMMA4_KV_LEGACY_V4_SHA256 ""
#endif

typedef char G4ProjectionBytesMustMatchContract[
    G4_PROJ_BYTES == 1115136u ? 1 : -1];
typedef char G4SlotBytesMustMatchContract[
    G4_SLOT_BYTES == 3345408u ? 1 : -1];
typedef char G4TextBindingCountMustMatchContract[
    G4_TEXT_BINDINGS == 12117 ? 1 : -1];

static const char G4_SOURCE_SHA[] =
    "312e73b836f39d06f5c982f4b5a83575d7c4bb23539b787293a247cd2bc71949";
static const char G4_ROLE_SHA[] =
    "fb4980b498cbe3985220560eca5ab0636fd1298d2f05c94f6367c93fb129c46c";
static const char G4_MAP_SHA[] =
    "51fa40aca8cb910896328026fcfb3f07c36a7519d0bd76e6fce574ba3c178a3e";
static const char G4_NV_SOURCE_SHA[] =
    "abf0073037606dfecc1b90d0b51aaa83be712d4d44b914d3cd6d48f74c51ba6b";
static const char G4_NV_PAYLOAD_SHA[] =
    "62fd36b05775b9462b7914af3f911998d372d8d5a200a34b9c5ae88666dce56f";
static const char G4_KV_COMPAT_SHA[] = SALT_GEMMA4_KV_COMPAT_SHA256;
typedef char G4KVCompatShaMustBe64[
    sizeof G4_KV_COMPAT_SHA == 65u ? 1 : -1];

typedef struct {
    int set;
    int bits, rows, cols;
    const uint32_t *weight;
    const uint16_t *scales;
    const uint16_t *biases;
    const float *weight_global_scale;
    const float *input_global_scale;
    uint8_t *nv_packed_scratch;
    uint8_t *nv_scale_scratch;
    float *nv_dequant_scratch;
} G4Q;

typedef struct {
    int set;
    int count;
    float *values;
} G4B;

typedef struct G4DeferredBf16 {
    G4B *target;
    const unsigned char *source;
    int count;
} G4DeferredBf16;

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t layers;
    uint32_t max_context;
    uint32_t position;
    uint64_t bytes;
    uint32_t kv_dims[G4_LAYERS];
    uint64_t instance_tag;
    unsigned char payload_sha256[32];
} G4KVSnapshotHeader;

typedef char G4KVSnapshotHeaderMustBe192[
    sizeof(G4KVSnapshotHeader) == 192 ? 1 : -1
];

static const unsigned char G4_KV_SNAPSHOT_MAGIC[8] = {
    'G', '4', 'K', 'V', 'Q', 'A', '0', '3'
};
static const unsigned char G4_KV_TRANSFER_MAGIC[8] = {
    'G', '4', 'K', 'V', 'C', '0', '0', '6'
};
static const unsigned char G4_KV_TRANSFER_MAGIC_V5[8] = {
    'G', '4', 'K', 'V', 'C', '0', '0', '5'
};
static const unsigned char G4_KV_TRANSFER_MAGIC_V4[8] = {
    'G', '4', 'K', 'V', 'C', '0', '0', '4'
};

typedef struct {
    int fd;
    char name[64];
    unsigned long long dev, ino;
    long long mtime_ns, ctime_ns;
    size_t size, map_len;
    unsigned char *base;
} G4Map;

typedef struct {
    G4Q q_proj, k_proj, v_proj, kv_proj, o_proj;
    G4Q dense_gate, dense_up, dense_down, router;
    G4B q_norm, k_norm;
    G4B input_norm, post_attention_norm;
    G4B pre_ffn_norm, pre_ffn_norm_2;
    G4B post_ffn_norm, post_ffn_norm_1, post_ffn_norm_2;
    G4B router_scale, per_expert_scale, layer_scalar;
    int full_attention;
    int head_dim, kv_heads, kv_dim, q_dim, rope_dim;
    float rope_theta;
    float *key_cache;
    float *value_cache;
    int kv_capacity;
    int kv_ring;
    const float *shared_key_cache;
    const float *shared_value_cache;
    int shared_row_base;
    int shared_position;
    int shared_key_private;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    float *proposal_key_cache;
    float *proposal_value_cache;
    int proposal_source_position;
    int proposal_capacity;
    int proposal_active;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    const SaltTextLayerPlan *plan;
} G4Layer;

typedef enum {
    G4_TEXT_BINDING_Q = 1,
    G4_TEXT_BINDING_B = 2,
    G4_TEXT_BINDING_MLX4_EXPERT = 3,
    G4_TEXT_BINDING_NVFP4_EXPERT = 4
} G4TextBindingKind;

typedef struct {
    G4TextBindingKind kind;
    struct SaltGemma4Text *owner;
    const G4Q *q;
    const G4B *b;
    uint64_t canonical_pool_offset;
    int layer;
    int expert;
    int projection;
} G4TextTensorBinding;

typedef enum {
    G4_TEXT_CPU_NODE_NONE = 0,
    G4_TEXT_CPU_NODE_Q4 = 1,
    G4_TEXT_CPU_NODE_Q8 = 2,
    G4_TEXT_CPU_NODE_MIXED = 3
} G4TextCpuNodeKind;

#define G4_TEXT_CPU_NODE_MAX_JOBS (2 * G4_EXPERTS + 1)

typedef struct {
    const uint32_t *values;
    const uint16_t *scales;
    const uint16_t *biases;
    const float *inputs;
    float *outputs;
    int rows;
    int columns;
    int batch;
} G4TextCpuQ8Job;

typedef struct {
    G4TextCpuNodeKind kind;
    SaltBatchJob q4_jobs[G4_TEXT_CPU_NODE_MAX_JOBS];
    G4TextCpuQ8Job q8_jobs[G4_TEXT_CPU_NODE_MAX_JOBS];
    const G4TextTensorBinding *bindings[G4_TEXT_CPU_NODE_MAX_JOBS];
    const unsigned char *projections[G4_TEXT_CPU_NODE_MAX_JOBS];
    int slots[G4_TEXT_CPU_NODE_MAX_JOBS];
    int64_t boundaries[33];
    uint32_t job_count;
    uint32_t q4_job_count;
    uint32_t q8_job_count;
    uint32_t workers;
    int pair_mode;
    int failed[32];
    int prepared;
} G4TextCpuNode;

typedef struct {
    int enabled;
    double attention_alloc_s;
    double attention_norm_s;
    double attention_qkv_s;
    double attention_transform_s;
    double attention_body_s;
    double attention_o_s;
    double attention_residual_s;
    double attention_cleanup_s;
    double ffn_alloc_s;
    double ffn_dense_s;
    double ffn_route_s;
    double ffn_group_s;
    double ffn_fetch_s;
    double ffn_gate_up_s;
    double ffn_activation_s;
    double ffn_down_s;
    double ffn_combine_s;
    double ffn_cleanup_s;
} G4PrefillDetail;

typedef struct {
    int enabled;
    int ffn_detail_enabled;
    uint64_t ffn_detail_tasks;
    uint64_t ffn_detail_waves;
    double attention_qkv_s;
    double attention_transform_s;
    double attention_body_s;
    double attention_o_s;
    double attention_residual_s;
    double ffn_dense_s;
    double ffn_route_s;
    double ffn_experts_s;
    double ffn_expert_fetch_s;
    double ffn_expert_compute_s;
    double ffn_expert_release_s;
    double ffn_combine_s;
    double ffn_expert_gate_work_s;
    double ffn_expert_up_work_s;
    double ffn_expert_activation_work_s;
    double ffn_expert_down_work_s;
    double ffn_expert_worker_critical_s;
    double ffn_expert_worker_skew_s;
    double ffn_expert_pool_overhead_s;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    uint64_t ffn_matrix_flows;
    double ffn_matrix_gate_up_setup_s;
    double ffn_matrix_gate_up_s;
    double ffn_matrix_activation_s;
    double ffn_matrix_down_setup_s;
    double ffn_matrix_down_s;
    double ffn_matrix_total_s;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
} G4DecodeDetail;

struct SaltGemma4Text {
    uint64_t proof_validation_calls;
    int full_gpu_intent;
    int metal_exact_cells;
    int fine_token_enabled;
    int decode_gpu_only;
    int gpu_bounded_weights;
    int gpu_expert_layer_view;
    int gpu_trunk_layer_view;
    int gpu_trunk_shared_pool;
    int gpu_trunk_exact_views;
    SaltGpuWeightAddressability gpu_weight_addressability;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    int gpu_device_residency;
    uint64_t gpu_device_expert_bytes;
    uint64_t gpu_residency_staging_bytes;
    uint32_t gpu_residency_staging_slots;
    SaltGpuResidency gpu_residency;
    void *gpu_residency_arena;
    size_t gpu_residency_arena_bytes;
    uint64_t gpu_residency_generation;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    uint32_t gpu_trunk_window_mask;
    uint64_t memory_limit_bytes;
    uint64_t startup_forecast_bytes;
    uint64_t startup_source_bytes;
    uint64_t startup_registered_bytes;
    uint64_t startup_pageable_bytes;
    uint64_t startup_expert_residency_bytes;
    uint64_t startup_host_runtime_bytes;
    uint64_t startup_host_kv_bytes;
    uint64_t startup_shared_arena_bytes;
    uint64_t startup_backend_pinned_bytes;
    uint64_t startup_backend_device_bytes;
    uint64_t startup_device_kv_bytes;
    uint64_t startup_cache_metadata_bytes;
    uint64_t startup_rope_bytes;
    uint64_t startup_descriptor_bytes;
    uint64_t startup_text_program_bytes;
    uint64_t startup_text_cpu_bytes;
    uint64_t startup_text_verify_bytes;
    int startup_admitted;
    int gpu_initialized;
    uint32_t gpu_registered_source_mask;
    int gpu_pool_registered;
    uint64_t gpu_dense_submissions;
    uint64_t gpu_router_submissions;
    uint64_t gpu_expert_gate_up_submissions;
    uint64_t gpu_expert_down_submissions;
    uint64_t gpu_expert_gate_up_commands;
    uint64_t gpu_expert_down_commands;
    uint64_t gpu_decode_expert_gate_up_commands;
    uint64_t gpu_decode_expert_down_commands;
    uint64_t gpu_head_submissions;
    uint64_t gpu_decode_submissions;
    uint64_t gpu_failures;
    SaltGpuSharedBuffer operation_shared;
    SaltGpuSharedBuffer compute_scratch_shared;
    SaltGpuSharedBuffer decode_shared;
    int shared_arenas_ready;
    uint64_t gpu_shared_output_batches;
    uint64_t gpu_shared_output_jobs;
    SaltGpuSharedBuffer kv_shared;
    int kv_pageable_mapping;
    int gpu_attention_enabled;
    int gpu_decode_attention_enabled;
    int gpu_decode_dense_chain_enabled;
    uint64_t gpu_attention_batches;
    uint64_t gpu_attention_tasks;
    G4Map maps[G4_MAX_MAPS];
    int map_count;
    G4Map pool;
    int pool_set;
    SaltExpertPool cache_pool;
    SaltCache expert_cache;
    int expert_cache_set;
    uint64_t expert_budget_bytes;
    int expert_cache_slots;
    int expert_cache_peak_slots;
    int expert_preload_enabled;
    int expert_preloaded_slots;
    int expert_preload_layer_count;
    int expert_preload_protected_slots;
    uint64_t expert_preload_layer_mask;
    uint64_t expert_preload_fetch_jobs;
    int expert_inflight_slots;
    int expert_peak_inflight_slots;
    SaltGemma4RouteObserver *route_observer;
    SaltGemma4AttentionPlanLease *attention_plan_lease;
    uint64_t expert_union_fetch_calls;
    uint64_t expert_union_fetch_experts;
    uint64_t expert_gate_up_epochs;
    uint64_t expert_down_epochs;
    int prefill_retain_layers;
    int prefill_decode_lookahead_enabled;
    SaltTextPhaseLookahead prefill_decode_lookahead;
    int tokenizer_fd;
    SaltTokenizer tokenizer;
    int tokenizer_set;

    G4Q embedding;
    int nvfp4_mode;
    int nvfp4_decode_aggregate;
    int nvfp4_cuda_head;
    int nvfp4_qkv_aggregate;
    int nvfp4_dense_aggregate;
    G4Q *nvfp4_experts;
    uint8_t *nvfp4_packed_scratch;
    uint8_t *nvfp4_scale_scratch;
    float *nvfp4_dequant_scratch;
    int nvfp4_expert_bindings;
    G4B final_norm;
    G4Layer layers[G4_LAYERS];
    int binding_count;

    int max_context;
    int position;
    float *kv_arena;
    size_t kv_arena_floats;
    int expert_workers;
    SaltTextTargetPolicy target_policy;
    int target_area_active_workers;
    int expert_matrix_waves;
    int prefill_operation_flow;
    int prefill_expert_matrix_flow;
    int target_gpu_experts;
    SaltKvCache compute_pool;
    int compute_pool_ready;
    uint64_t q4_pool_submissions;
    uint64_t q4_pool_fallbacks;
    uint64_t q4_multi_pool_submissions;
    uint64_t q4_multi_pool_jobs;
    uint64_t qkv_multi_pool_submissions;
    uint64_t q4_multi_pool_fallbacks;
    uint64_t q8_pool_submissions;
    uint64_t q8_pool_fallbacks;
    uint64_t expert_pool_submissions;
    uint64_t expert_pool_fallbacks;
    uint64_t expert_matrix_gate_up_waves;
    uint64_t expert_matrix_down_waves;
    uint64_t expert_matrix_projection_jobs;
    uint64_t prefill_expert_matrix_flow_sessions;
    uint64_t target_matrix_flow_sessions;
    uint64_t target_gpu_expert_batches;
    uint64_t target_gpu_expert_jobs;
    float *sliding_rope_cos;
    float *sliding_rope_sin;
    float *full_rope_cos;
    float *full_rope_sin;
    int sliding_rope_pairs;
    int full_rope_pairs;
    int use_ondemand_rope_proof;
    const SaltModelDesc *model_desc;
    SaltStateControl state_control;
    SaltTextLayerPlan layer_plan[G4_LAYERS];
    SaltKvLayerPlan kv_plan[G4_LAYERS];
    size_t planned_kv_floats;
    unsigned char kv_compat_sha256[32];
    unsigned char build_identity_sha256[32];
    G4PrefillDetail prefill_detail;
    G4DecodeDetail decode_detail;

    SaltTextLayerExecDesc text_layers[G4_LAYERS];
    SaltTextExpertDesc text_experts[G4_LAYERS][G4_EXPERTS];
    SaltTextKvLayerDesc text_kv[G4_LAYERS];
    SaltTextKvSharedPrefixState text_shared_keys[G4_LAYERS];
    SaltTextKvSharedPrefixState text_shared_values[G4_LAYERS];
    G4TextTensorBinding text_bindings[G4_TEXT_BINDINGS];
    size_t text_binding_count;
    SaltTensorResourceSpec text_resources[G4_TEXT_RESOURCES];
    uint32_t text_resource_count;
    SaltTextModelExecDesc text_descriptor;
    SaltTextKvState text_kv_state;
    SaltTextVerifyProgram text_program;
    SaltTextVerifyCpuContext text_cpu;
    SaltTextVerifyHeterogeneousContext text_gpu;
    SaltTextVerifyExecutor text_executor;
    SaltAreaWfqRuntime target_wfq;
    SaltAreaWfqBranch target_wfq_branches[
        SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    SaltAreaWfqItem target_wfq_items[
        SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    SaltAreaWfqItemResult target_wfq_results[
        SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    SaltAreaNfqMatrix target_nfq;
    SaltAreaNfqRoute target_nfq_routes[
        SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    SaltAreaNfqItem target_nfq_items[
        SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES * 8u];
    SaltTextTargetNode target_nfq_nodes[
        SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    uint32_t target_nfq_ready[32];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    SaltTextProjectionWindow target_projection;
    SaltTextTokenEpochController scheduler_controller;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    void *text_program_arena;
    size_t text_program_arena_bytes;
    void *text_cpu_arena;
    size_t text_cpu_arena_bytes;
    SaltGpuSharedBuffer text_cpu_shared;
    void *text_gpu_arena;
    size_t text_gpu_arena_bytes;
    SaltTextTokenProgram token_program;
    SaltTextTokenCell token_program_cells[G4_TOKEN_PROGRAM_CELLS];
    uint64_t token_transition_generation;
    SaltTextTokenBackendStats token_backend_last;

    float **owned_bf16;
    size_t owned_count, owned_capacity;
    uint64_t owned_bf16_bytes;
    G4DeferredBf16 deferred_bf16[G4_BF16_VECTOR_BINDINGS];
    size_t deferred_bf16_count;
    SaltGpuSharedBuffer bf16_shared;

    float *state;
    float *after_attention;
    float *layer_output;
    float *norm_a;
    float *norm_b;
    float *branch;
    float *q;
    float *k;
    float *v;
    float *attention_output;
    float *dense_gate;
    float *dense_up;
    float *dense_output;
    float *routed_output;
    float *parallel_scratch;
    float *router_logits;
    float *route_storage;
    int *route_selected;
    float *route_weights;
    float *expert_outputs;
    float *expert_gate;
    float *expert_up;
    float *scores;
    float *final_state;
    float *head_logits;

    void *operation_arena;
    size_t operation_arena_bytes;
    float *operation_states_a;
    float *operation_states_b;
    float *operation_q_all;
    unsigned char *operation_full_mask;
    unsigned char *operation_sliding_mask;
    unsigned char *operation_valid;
    int *operation_block_ids;
    int operation_capacity;
    SaltTextPrefillScratch text_prefill_scratch;
    float *prefill_attention_norm;
    float *prefill_attention_queries;
    float *prefill_attention_keys;
    float *prefill_attention_values;
    float *prefill_attention_outputs;
    float *prefill_attention_branches;
    int prefill_attention_capacity;

    int prefill_capacity;
    uint64_t prefill_ffn_arena_bytes;
    uint64_t prefill_ffn_arena_calls;
    float *prefill_dense_inputs;
    float *prefill_dense_gate;
    float *prefill_dense_up;
    float *prefill_dense_chain;
    float *prefill_dense_outputs;
    float *prefill_router_inputs;
    float *prefill_router_logits;
    float *prefill_routed_inputs;
    float *prefill_routed_outputs;
    float *prefill_weights;
    float *prefill_group_gate;
    float *prefill_group_up;
    float *prefill_group_inputs;
    float *prefill_group_outputs;
    float *prefill_selection_outputs;
    int *prefill_selected;
    SaltMoEGroupScratch prefill_moe_scratch;
    unsigned char prefill_moe_used[G4_EXPERTS];
    unsigned char prefill_moe_acquired[G4_EXPERTS];
    void *prefill_moe_leases[G4_EXPERTS];
    void *prefill_moe_batch_leases[G4_EXPERTS];
    void *prefill_moe_run_leases[G4_EXPERTS];
    int prefill_moe_expert_ids[G4_EXPERTS];
    int prefill_moe_run_groups[G4_EXPERTS];
    SaltBatchJob prefill_gate_up_jobs[2 * G4_EXPERTS];
    SaltBatchJob prefill_down_jobs[G4_EXPERTS];
    SaltGpuMoeExpert prefill_gpu_moe[G4_EXPERTS];
    const uint32_t *prefill_gpu_vals[G4_CUDA_HETERO_JOBS];
    const uint16_t *prefill_gpu_scales[G4_CUDA_HETERO_JOBS];
    const uint16_t *prefill_gpu_biases[G4_CUDA_HETERO_JOBS];
    const float *prefill_gpu_xs[G4_CUDA_HETERO_JOBS];
    float *prefill_gpu_ys[G4_CUDA_HETERO_JOBS];
    const void *prefill_gpu_ids[G4_CUDA_HETERO_JOBS];
    int prefill_gpu_rows[G4_CUDA_HETERO_JOBS];
    const unsigned char *prefill_job_slots[G4_EXPERTS];
    int prefill_job_offsets[G4_EXPERTS + 1];
};

static int g4_target_gpu_program(const SaltGemma4Text *model) {
    return model && model->target_policy.ready &&
        model->target_policy.execution_class == SALT_TEXT_EXECUTION_GPU_ONLY;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
static unsigned char *g4_target_canonical_base(
        SaltGemma4Text *model, size_t *bytes) {
    if (bytes) *bytes = 0;
    if (!model || !bytes || !model->text_program.ready)
        return NULL;
    if (g4_target_gpu_program(model)) {
        if (!model->text_gpu.ready || !model->text_gpu.canonical ||
            !model->text_gpu.canonical->contents)
            return NULL;
        *bytes = model->text_gpu.canonical->nbytes;
        return (unsigned char *)model->text_gpu.canonical->contents;
    }
    if (!model->text_cpu.ready || !model->text_cpu.arena)
        return NULL;
    *bytes = model->text_cpu.arena_bytes;
    return model->text_cpu.arena;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_recipe_int(const char *name, int minimum, int maximum,
                         int *value);
static int g4_recipe_decimal_gb_bytes(
    const char *name, uint32_t minimum, uint32_t maximum, uint64_t *bytes);
static int q_matvec(const G4Q *matrix, const float *input, float *output);
static int q_matvec_batch(SaltGemma4Text *model,
                          const G4Q *matrix, int batch,
                          const float *inputs, float *outputs);
static int g4_q4_pool_batch(SaltGemma4Text *model,
                            const uint32_t *values,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            int rows, int columns, int batch,
                            const float *inputs, float *outputs);
static int g4_q4_pool_multi_batch(SaltGemma4Text *model,
                                  SaltBatchJob *jobs, int job_count);
static int g4_q4_pool_multi_batch_workers(SaltGemma4Text *model,
                                          SaltBatchJob *jobs, int job_count,
                                          int worker_limit);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
static int g4_q4_pool_multi_batch_tiles(SaltGemma4Text *model,
                                        SaltBatchJob *jobs, int job_count,
                                        int worker_limit, int tile_rounds);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_q8_pool_batch(SaltGemma4Text *model, const G4Q *matrix,
                            int batch, const float *inputs, float *outputs);
static int gpu_q8_matvec_batch(SaltGemma4Text *model,
                               const G4Q *matrix, int batch,
                               const float *inputs, float *outputs);
static int init_text_verify(SaltGemma4Text *model);
static int init_text_token_program(SaltGemma4Text *model);
static int g4_text_state_aligned(const SaltGemma4Text *model);
static int g4_state_materialize(
    void *context, SaltStateTransactionKind transaction,
    SaltStateArtifactKind artifact, SaltStateTransactionReason reason);
static int g4_state_finalize(
    void *context, SaltStateTransactionKind transaction,
    SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
    const SaltStateView *committed_view);
static int g4_gpu_multi_batch(SaltGemma4Text *model,
                              const SaltBatchJob *jobs, int job_count,
                              int *expanded_jobs);
static int g4_reconstruct_shared_key_from_value(
    SaltGemma4Text *model, const G4Layer *layer, int position,
    float *key, const float *value);

static const float *g4_key_row(const G4Layer *layer, int position) {
    size_t index;
    if (!layer || position < 0 || layer->kv_dim < 1 || layer->kv_capacity < 1)
        return NULL;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (layer->proposal_active &&
        position >= layer->proposal_source_position &&
        position - layer->proposal_source_position < layer->proposal_capacity) {
        index = (size_t)(position - layer->proposal_source_position);
        return layer->proposal_key_cache + index * (size_t)layer->kv_dim;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (position < layer->shared_position) {
        if (!layer->shared_key_cache || position < layer->shared_row_base)
            return NULL;
        index = (size_t)(position - layer->shared_row_base);
        return layer->shared_key_cache + index * (size_t)layer->kv_dim;
    }
    if (layer->kv_ring) {
        index = (size_t)(position - layer->shared_position);
        index %= (size_t)layer->kv_capacity;
    } else {
        index = (size_t)position;
    }
    if (index >= (size_t)layer->kv_capacity) return NULL;
    return layer->key_cache + index * (size_t)layer->kv_dim;
}

static const float *g4_value_row(const G4Layer *layer, int position) {
    size_t index;
    if (!layer || position < 0 || layer->kv_dim < 1 || layer->kv_capacity < 1)
        return NULL;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (layer->proposal_active &&
        position >= layer->proposal_source_position &&
        position - layer->proposal_source_position < layer->proposal_capacity) {
        index = (size_t)(position - layer->proposal_source_position);
        return layer->proposal_value_cache + index * (size_t)layer->kv_dim;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (position < layer->shared_position) {
        if (!layer->shared_value_cache || position < layer->shared_row_base)
            return NULL;
        index = (size_t)(position - layer->shared_row_base);
        return layer->shared_value_cache + index * (size_t)layer->kv_dim;
    }
    if (layer->kv_ring) {
        index = (size_t)(position - layer->shared_position);
        index %= (size_t)layer->kv_capacity;
    } else {
        index = (size_t)position;
    }
    if (index >= (size_t)layer->kv_capacity) return NULL;
    return layer->value_cache + index * (size_t)layer->kv_dim;
}

static float *g4_private_key_row(G4Layer *layer, int position) {
    return (float *)(uintptr_t)g4_key_row(layer, position);
}

static float *g4_private_value_row(G4Layer *layer, int position) {
    return (float *)(uintptr_t)g4_value_row(layer, position);
}

static int g4_attention_heads(
        SaltGemma4Text *model, const G4Layer *layer, int first, int count) {
    SaltAttnHeadOperation operation;
    int end, position;
    if (!model || !layer || !model->compute_pool_ready || first < 0 ||
        count < 1 || first > INT_MAX - count)
        return -1;
    memset(&operation, 0, sizeof operation);
    operation.queries = model->q;
    operation.outputs = model->attention_output;
    operation.head_count = 16;
    operation.head_dim = layer->head_dim;
    operation.kv_heads = layer->kv_heads;
    operation.query_stride = layer->head_dim;
    operation.output_stride = layer->head_dim;
    operation.kv_row_stride = layer->kv_dim;
    end = first + count;
    position = first;
    while (position < end) {
        SaltAttnKvSpan *span;
        int span_end = end;
        if (operation.span_count >= SALT_ATTN_HEAD_MAX_SPANS)
            return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        if (layer->proposal_active &&
            position >= layer->proposal_source_position) {
            int proposal_end = layer->proposal_source_position +
                layer->proposal_capacity;
            if (span_end > proposal_end) span_end = proposal_end;
        } else {
            if (layer->proposal_active &&
                span_end > layer->proposal_source_position)
                span_end = layer->proposal_source_position;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        if (position < layer->shared_position) {
            if (position < layer->shared_row_base) return -1;
            if (span_end > layer->shared_position)
                span_end = layer->shared_position;
        } else if (layer->kv_ring) {
            size_t logical = (size_t)(position - layer->shared_position);
            size_t index = logical % (size_t)layer->kv_capacity;
            size_t contiguous = (size_t)layer->kv_capacity - index;
            if ((size_t)(span_end - position) > contiguous)
                span_end = position + (int)contiguous;
        }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        span = &operation.spans[operation.span_count++];
        span->keys = g4_key_row(layer, position);
        span->values = g4_value_row(layer, position);
        span->count = span_end - position;
        if (!span->keys || !span->values || span->count < 1)
            return -1;
        position = span_end;
    }
    return salt_attn_heads_execute(&model->compute_pool, &operation);
}

static void set_error(char *error, size_t size, const char *message) {
    if (!error || size == 0) return;
    snprintf(error, size, "%s", message ? message : "unknown Gemma 4 error");
}

static int step_fail(const char *stage, int layer, int position) {
    fprintf(stderr, "GEMMA4_STEP_FAIL stage=%s layer=%d position=%d\n",
        stage, layer, position);
    fflush(stderr);
    return -1;
}

static int g4_sha256_from_hex(const char *hex, unsigned char digest[32]) {
    if (!hex || !digest || strlen(hex) != 64u) return -1;
    for (size_t i = 0; i < 32u; i++) {
        unsigned int high, low;
        char a = hex[i * 2u], b = hex[i * 2u + 1u];
        if (a >= '0' && a <= '9') high = (unsigned int)(a - '0');
        else if (a >= 'a' && a <= 'f') high = (unsigned int)(a - 'a' + 10);
        else return -1;
        if (b >= '0' && b <= '9') low = (unsigned int)(b - '0');
        else if (b >= 'a' && b <= 'f') low = (unsigned int)(b - 'a' + 10);
        else return -1;
        digest[i] = (unsigned char)((high << 4u) | low);
    }
    return 0;
}

static void set_role_error(char *error, size_t size,
                           const char *prefix, const char *role) {
    if (!error || size == 0) return;
    snprintf(error, size, "%s%s", prefix, role ? role : "<null>");
}

static long long timespec_ns(time_t sec, long nsec) {
    return (long long)sec * 1000000000LL + (long long)nsec;
}

static double g4_now_s(void) {
    struct timespec time_value;
    if (clock_gettime(CLOCK_MONOTONIC, &time_value) != 0) return 0.0;
    return (double)time_value.tv_sec + (double)time_value.tv_nsec * 1e-9;
}

static int g4_memory_budget_ok(const SaltGemma4Text *model,
                               const char *phase, int layer) {
    SaltMemSnapshot snapshot;
    if (!model || model->memory_limit_bytes == 0) return 0;
    if (salt_mem_snapshot(&snapshot) != 0 || snapshot.resident_b < 0)
        return -1;
    if ((uint64_t)snapshot.resident_b > model->memory_limit_bytes) {
        fprintf(stderr,
            "GEMMA4_MEMORY_LIMIT phase=%s layer=%d rss_bytes=%lld "
            "limit_bytes=%llu\n", phase ? phase : "unknown", layer,
            (long long)snapshot.resident_b,
            (unsigned long long)model->memory_limit_bytes);
        return -1;
    }
    return 0;
}

static int g4_compact_hmm_gpu_program(const SaltGemma4Text *model) {
    return model && model->full_gpu_intent && model->fine_token_enabled &&
        model->gpu_bounded_weights && !model->nvfp4_mode &&
        !model->gpu_expert_layer_view &&
        model->gpu_weight_addressability == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE;
}

/* The explicit full-GPU registered CUDA cell uses the existing C99 text
 * program; the phase-mixed cell retains its established callback path. */
static int g4_registered_cuda_gpu_program(const SaltGemma4Text *model) {
    return model && model->full_gpu_intent && model->decode_gpu_only &&
        !model->fine_token_enabled && !model->gpu_bounded_weights &&
        !model->gpu_expert_layer_view && g4_target_gpu_program(model) &&
        model->gpu_weight_addressability ==
            SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT &&
        salt_gpu_cuda_present();
}

/* The registered plan counts retained history; ordinary attention also needs
 * the current token's seat before consuming that history. Keep the established
 * live ring geometry independently of CPU/GPU execution policy. */
static size_t g4_live_kv_capacity(const SaltGemma4Text *model, int layer) {
    size_t capacity = model->kv_plan[layer].token_capacity;
    if (model->layer_plan[layer].attention.kind == SALT_ATTN_SLIDING &&
        capacity < (size_t)model->max_context)
        capacity++;
    return capacity;
}

static int g4_apply_kv_forecast_policy(SaltGemma4Text *model) {
    size_t total = 0;
    if (!model || model->max_context < 1) return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        const SaltKvLayerPlan *plan = &model->kv_plan[i];
        size_t capacity = g4_live_kv_capacity(model, i);
        size_t width, count;
        if (plan->k_width < 1 || plan->v_width < 1 ||
            capacity == 0 || capacity > (size_t)model->max_context)
            return -1;
        width = (size_t)plan->k_width + (size_t)plan->v_width;
        if (capacity > SIZE_MAX / width) return -1;
        count = capacity * width;
        if (total > SIZE_MAX - count) return -1;
        total += count;
    }
    model->planned_kv_floats = total;
    return 0;
}

static void stat_times(const struct stat *st, long long *mtime, long long *ctime) {
#if defined(__APPLE__)
    *mtime = timespec_ns(st->st_mtime, st->st_mtimensec);
    *ctime = timespec_ns(st->st_ctime, st->st_ctimensec);
#else
    *mtime = timespec_ns(st->st_mtim.tv_sec, st->st_mtim.tv_nsec);
    *ctime = timespec_ns(st->st_ctim.tv_sec, st->st_ctim.tv_nsec);
#endif
}

static int map_identity_matches(const G4Map *map) {
    struct stat st;
    long long mtime, ctime;
    if (!map || map->fd < 0 || fstat(map->fd, &st) != 0 ||
        !S_ISREG(st.st_mode)) return 0;
    stat_times(&st, &mtime, &ctime);
    return (unsigned long long)st.st_dev == map->dev &&
           (unsigned long long)st.st_ino == map->ino &&
           (size_t)st.st_size == map->size &&
           mtime == map->mtime_ns && ctime == map->ctime_ns;
}

static int all_identities_match(const SaltGemma4Text *model) {
    if (!model || (!model->nvfp4_mode &&
            (!model->pool_set || !map_identity_matches(&model->pool))))
        return 0;
    for (int i = 0; i < model->map_count; i++)
        if (!map_identity_matches(&model->maps[i])) return 0;
    return 1;
}

static G4Map *find_map(SaltGemma4Text *model, int fd) {
    for (int i = 0; i < model->map_count; i++)
        if (model->maps[i].fd == fd) return &model->maps[i];
    return NULL;
}

static int checked_extent(const G4Map *map, unsigned long long offset,
                          unsigned long long bytes, const void **pointer) {
    if (!map || !pointer || bytes == 0 || offset > map->size ||
        bytes > map->size - (size_t)offset) return -1;
    *pointer = map->base + (size_t)offset;
    return 0;
}

static float bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}

static int own_bf16(SaltGemma4Text *model, float *values) {
    if (model->owned_count == model->owned_capacity) {
        size_t capacity = model->owned_capacity ? model->owned_capacity * 2 : 64;
        float **grown = (float **)realloc(model->owned_bf16,
                                         capacity * sizeof *grown);
        if (!grown) return -1;
        model->owned_bf16 = grown;
        model->owned_capacity = capacity;
    }
    model->owned_bf16[model->owned_count++] = values;
    return 0;
}

static int load_bf16(SaltGemma4Text *model, int count, int fd,
                     unsigned long long offset, unsigned long long bytes,
                     G4B *out) {
    const void *pointer = NULL;
    G4Map *map = find_map(model, fd);
    if (!out || out->set || count < 1 || bytes != (unsigned long long)count * 2 ||
        checked_extent(map, offset, bytes, &pointer) != 0) return -1;
    float *values = (float *)malloc((size_t)count * sizeof *values);
    if (!values || own_bf16(model, values) != 0) {
        free(values);
        return -1;
    }
    if ((uint64_t)(uint32_t)count * sizeof *values >
            UINT64_MAX - model->owned_bf16_bytes)
        return -1;
    model->owned_bf16_bytes += (uint64_t)(uint32_t)count * sizeof *values;
    const unsigned char *source = (const unsigned char *)pointer;
    for (int i = 0; i < count; i++) {
        uint16_t encoded;
        memcpy(&encoded, source + (size_t)i * 2, sizeof encoded);
        values[i] = bf16_to_float(encoded);
        if (!isfinite(values[i])) return -1;
    }
    out->set = 1;
    out->count = count;
    out->values = values;
    return 0;
}

static int defer_bf16(SaltGemma4Text *model, int count, int fd,
                      unsigned long long offset, unsigned long long bytes,
                      G4B *out) {
    const void *pointer = NULL;
    G4Map *map = find_map(model, fd);
    uint64_t decoded_bytes;
    G4DeferredBf16 *descriptor;
    if (!model || !model->full_gpu_intent || !out || out->set || count < 1 ||
        bytes != (unsigned long long)(uint32_t)count * 2u ||
        checked_extent(map, offset, bytes, &pointer) != 0 ||
        model->deferred_bf16_count >= G4_BF16_VECTOR_BINDINGS)
        return -1;
    decoded_bytes = (uint64_t)(uint32_t)count * sizeof(float);
    if (decoded_bytes > UINT64_MAX - model->owned_bf16_bytes) return -1;
    descriptor = &model->deferred_bf16[model->deferred_bf16_count++];
    descriptor->target = out;
    descriptor->source = (const unsigned char *)pointer;
    descriptor->count = count;
    model->owned_bf16_bytes += decoded_bytes;
    out->set = 1;
    out->count = count;
    out->values = NULL;
    return 0;
}

static int materialize_bf16_shared(SaltGemma4Text *model) {
    float *cursor, *end;
    if (!model || !model->full_gpu_intent || !model->gpu_initialized ||
        !model->startup_admitted || model->deferred_bf16_count !=
            G4_BF16_VECTOR_BINDINGS || model->owned_bf16_bytes < sizeof(float) ||
        model->owned_bf16_bytes > SIZE_MAX)
        return -1;
    if (salt_gpu_shared_buffer_alloc(&model->bf16_shared,
            (size_t)model->owned_bf16_bytes) != 0)
        return -1;
    cursor = (float *)model->bf16_shared.contents;
    end = cursor + model->owned_bf16_bytes / sizeof(float);
    for (size_t binding = 0; binding < model->deferred_bf16_count; binding++) {
        G4DeferredBf16 *descriptor = &model->deferred_bf16[binding];
        size_t count = (size_t)descriptor->count;
        if (!descriptor->target || descriptor->target->values ||
            !descriptor->source || cursor > end ||
            count > (size_t)(end - cursor))
            return -1;
        descriptor->target->values = cursor;
        for (size_t i = 0; i < count; i++) {
            uint16_t encoded;
            memcpy(&encoded, descriptor->source + i * 2u, sizeof encoded);
            cursor[i] = bf16_to_float(encoded);
            if (!isfinite(cursor[i])) return -1;
        }
        cursor += count;
    }
    return cursor == end ? 0 : -1;
}

static int q_expected_bytes(int bits, int rows, int cols,
                            unsigned long long *weight,
                            unsigned long long *scales) {
    if ((bits != 4 && bits != 8) || rows < 1 || cols < 1 || cols % 64 != 0)
        return -1;
    *weight = (unsigned long long)rows * (unsigned long long)cols *
              (unsigned long long)bits / 8u;
    *scales = (unsigned long long)rows * (unsigned long long)(cols / 64) * 2u;
    return 0;
}

static int add_q_payload_bytes(const G4Q *matrix, uint64_t *total) {
    unsigned long long weight, scale, bytes;
    if (!matrix || !total) return -1;
    if (!matrix->set) return 0;
    if (matrix->bits == 16) {
        uint64_t elements = (uint64_t)(uint32_t)matrix->rows *
                            (uint64_t)(uint32_t)matrix->cols;
        if (elements > (UINT64_MAX - *total) / 2u) return -1;
        *total += elements * 2u;
        return 0;
    }
    if (matrix->bits == 40) {
        uint64_t elements = (uint64_t)(uint32_t)matrix->rows *
                            (uint64_t)(uint32_t)matrix->cols;
        bytes = elements / 2u + elements / 16u + 8u;
        if (bytes > UINT64_MAX - *total) return -1;
        *total += bytes;
        return 0;
    }
    if (q_expected_bytes(matrix->bits, matrix->rows, matrix->cols,
            &weight, &scale) != 0 || scale > (ULLONG_MAX - weight) / 2u)
        return -1;
    bytes = weight + 2u * scale;
    if (bytes > UINT64_MAX - *total) return -1;
    *total += (uint64_t)bytes;
    return 0;
}

static int dense_payload_bytes(const SaltGemma4Text *model, uint64_t *bytes) {
    uint64_t total = 0;
    if (!model || !bytes || add_q_payload_bytes(&model->embedding, &total) != 0)
        return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        const G4Layer *entry = &model->layers[layer];
        if (add_q_payload_bytes(&entry->q_proj, &total) != 0 ||
            add_q_payload_bytes(&entry->k_proj, &total) != 0 ||
            add_q_payload_bytes(&entry->v_proj, &total) != 0 ||
            add_q_payload_bytes(&entry->kv_proj, &total) != 0 ||
            add_q_payload_bytes(&entry->o_proj, &total) != 0 ||
            add_q_payload_bytes(&entry->dense_gate, &total) != 0 ||
            add_q_payload_bytes(&entry->dense_up, &total) != 0 ||
            add_q_payload_bytes(&entry->dense_down, &total) != 0 ||
            add_q_payload_bytes(&entry->router, &total) != 0)
            return -1;
    }
    *bytes = total;
    return total > 0 ? 0 : -1;
}

static void release_mapped_range(const void *pointer, size_t bytes) {
    size_t released = 0;
    if (!pointer || bytes == 0) return;
    (void)salt_mmap_dontneed_contained(pointer, bytes, &released);
}

static void release_q_residency(const G4Q *matrix) {
    unsigned long long weight = 0, scale = 0;
    if (!matrix || !matrix->set ||
        q_expected_bytes(matrix->bits, matrix->rows, matrix->cols,
                         &weight, &scale) != 0 ||
        weight > SIZE_MAX || scale > SIZE_MAX)
        return;
    release_mapped_range(matrix->weight, (size_t)weight);
    release_mapped_range(matrix->scales, (size_t)scale);
    release_mapped_range(matrix->biases, (size_t)scale);
}

static int gpu_map_for_q(const SaltGemma4Text *model, const G4Q *q);

static int q_weight_bytes(const G4Q *matrix, size_t *bytes_out) {
    unsigned long long weight = 0, scale = 0;
    uint64_t elements;
    if (!matrix || !matrix->set || !bytes_out ||
            matrix->rows < 1 || matrix->cols < 1 ||
            (uint64_t)(uint32_t)matrix->rows >
                UINT64_MAX / (uint64_t)(uint32_t)matrix->cols)
        return -1;
    elements = (uint64_t)(uint32_t)matrix->rows *
               (uint64_t)(uint32_t)matrix->cols;
    if (matrix->bits == 16) {
        if (elements > SIZE_MAX / 2u) return -1;
        weight = elements * 2u;
    } else if (matrix->bits == 40) {
        if (elements % 2u != 0 || elements / 2u > SIZE_MAX) return -1;
        weight = elements / 2u;
    } else if (q_expected_bytes(
            matrix->bits, matrix->rows, matrix->cols,
            &weight, &scale) != 0 || weight > SIZE_MAX) {
        return -1;
    }
    *bytes_out = (size_t)weight;
    return 0;
}

static int prepare_q_weight_residency(
        const SaltGemma4Text *model, const G4Q *matrix) {
    uintptr_t base, weight_pointer;
    size_t weight = 0;
    int map_id;
    if (!model || q_weight_bytes(matrix, &weight) != 0 ||
            (map_id = gpu_map_for_q(model, matrix)) < 0 ||
            map_id >= model->map_count)
        return -1;
    base = (uintptr_t)(const void *)model->maps[map_id].base;
    weight_pointer = (uintptr_t)(const void *)matrix->weight;
    if (weight_pointer < base || weight_pointer - base > INT64_MAX ||
            weight_pointer - base > model->maps[map_id].map_len ||
            weight > model->maps[map_id].map_len - (weight_pointer - base))
        return -1;
    return salt_file_willneed(
        model->maps[map_id].fd, (int64_t)(weight_pointer - base), weight);
}

static int prepare_layer_weight_residency(
        const SaltGemma4Text *model, G4Layer *layer) {
    G4Q *matrices[9];
    if (!model || !layer) return -1;
    matrices[0] = &layer->q_proj;
    matrices[1] = &layer->k_proj;
    matrices[2] = &layer->v_proj;
    matrices[3] = &layer->kv_proj;
    matrices[4] = &layer->o_proj;
    matrices[5] = &layer->dense_gate;
    matrices[6] = &layer->dense_up;
    matrices[7] = &layer->dense_down;
    matrices[8] = &layer->router;
    for (size_t index = 0;
            index < sizeof matrices / sizeof matrices[0]; index++)
        if (matrices[index]->set &&
                prepare_q_weight_residency(model, matrices[index]) != 0)
            return -1;
    return 0;
}

static void release_layer_residency(G4Layer *layer) {
    if (!layer) return;
    release_q_residency(&layer->q_proj);
    release_q_residency(&layer->k_proj);
    release_q_residency(&layer->v_proj);
    release_q_residency(&layer->kv_proj);
    release_q_residency(&layer->o_proj);
    release_q_residency(&layer->dense_gate);
    release_q_residency(&layer->dense_up);
    release_q_residency(&layer->dense_down);
    release_q_residency(&layer->router);
}

static int load_q(SaltGemma4Text *model, int bits, int rows, int cols,
                  int wfd, unsigned long long woff, unsigned long long wn,
                  int sfd, unsigned long long soff, unsigned long long sn,
                  int bfd, unsigned long long boff, unsigned long long bn,
                  G4Q *out) {
    unsigned long long expected_weight, expected_scale;
    const void *weight = NULL, *scales = NULL, *biases = NULL;
    if (!out || out->set ||
        q_expected_bytes(bits, rows, cols, &expected_weight, &expected_scale) != 0 ||
        wn != expected_weight || sn != expected_scale || bn != expected_scale ||
        checked_extent(find_map(model, wfd), woff, wn, &weight) != 0 ||
        checked_extent(find_map(model, sfd), soff, sn, &scales) != 0 ||
        checked_extent(find_map(model, bfd), boff, bn, &biases) != 0 ||
        !weight || !scales || !biases) return -1;
    out->set = 1;
    out->bits = bits;
    out->rows = rows;
    out->cols = cols;
    out->weight = (const uint32_t *)weight;
    out->scales = (const uint16_t *)scales;
    out->biases = (const uint16_t *)biases;
    return 0;
}

static int enable_nvfp4(SaltGemma4Text *model) {
    size_t experts = (size_t)G4_LAYERS * G4_EXPERTS * 3u;
    const char *simd = getenv("SALT_NVFP4_SIMD");
    const char *decode_aggregate = getenv("SALT_NVFP4_DECODE_AGGREGATE");
    const char *cuda_head = getenv("SALT_NVFP4_CUDA_HEAD");
    const char *qkv_aggregate = getenv("SALT_NVFP4_QKV_AGGREGATE");
    const char *dense_aggregate = getenv("SALT_NVFP4_DENSE_AGGREGATE");
    if (!model || model->nvfp4_mode || model->expert_workers != 1)
        return -1;
    if (simd && strcmp(simd, "0") != 0 && strcmp(simd, "1") != 0)
        return -1;
    if (decode_aggregate && strcmp(decode_aggregate, "0") != 0 &&
            strcmp(decode_aggregate, "1") != 0)
        return -1;
    if (cuda_head && strcmp(cuda_head, "0") != 0 &&
            strcmp(cuda_head, "1") != 0)
        return -1;
    if (qkv_aggregate && strcmp(qkv_aggregate, "0") != 0 &&
            strcmp(qkv_aggregate, "1") != 0)
        return -1;
    if (dense_aggregate && strcmp(dense_aggregate, "0") != 0 &&
            strcmp(dense_aggregate, "1") != 0)
        return -1;
    salt_nvfp4_set_simd(simd && !strcmp(simd, "1"));
    model->nvfp4_decode_aggregate =
        !decode_aggregate || strcmp(decode_aggregate, "0") != 0;
    model->nvfp4_cuda_head = !cuda_head || strcmp(cuda_head, "0") != 0;
    model->nvfp4_qkv_aggregate =
        !qkv_aggregate || strcmp(qkv_aggregate, "0") != 0;
    model->nvfp4_dense_aggregate =
        !dense_aggregate || strcmp(dense_aggregate, "0") != 0;
    model->nvfp4_experts = (G4Q *)calloc(experts, sizeof(G4Q));
    model->nvfp4_packed_scratch = (uint8_t *)malloc(G4_MAX_Q / 2u);
    model->nvfp4_scale_scratch = (uint8_t *)malloc(G4_MAX_Q / 16u);
    model->nvfp4_dequant_scratch =
        (float *)malloc((size_t)G4_MAX_Q * sizeof(float));
    if (!model->nvfp4_experts || !model->nvfp4_packed_scratch ||
        !model->nvfp4_scale_scratch || !model->nvfp4_dequant_scratch)
        return -1;
    model->nvfp4_mode = 1;
    return 0;
}

static int load_bf16_matrix(SaltGemma4Text *model, int rows, int cols,
                            int fd, unsigned long long offset,
                            unsigned long long bytes, G4Q *out) {
    const void *pointer = NULL;
    uint64_t count;
    if (!model || !out || out->set || rows < 1 || cols < 1)
        return -1;
    count = (uint64_t)(uint32_t)rows * (uint64_t)(uint32_t)cols;
    if (count > ULLONG_MAX / 2u || bytes != count * 2u ||
        checked_extent(find_map(model, fd), offset, bytes, &pointer) != 0)
        return -1;
    out->set = 1;
    out->bits = 16;
    out->rows = rows;
    out->cols = cols;
    out->weight = (const uint32_t *)pointer;
    return 0;
}

static int load_nvfp4_matrix(
    SaltGemma4Text *model, int rows, int cols,
    int wfd, unsigned long long woff, unsigned long long wn,
    int sfd, unsigned long long soff, unsigned long long sn,
    int wgfd, unsigned long long wgoff, unsigned long long wgn,
    int igfd, unsigned long long igoff, unsigned long long ign,
    G4Q *out) {
    const void *weight = NULL, *scales = NULL, *weight_global = NULL;
    const void *input_global = NULL;
    uint64_t elements;
    if (!model || !model->nvfp4_mode || !out || out->set ||
        rows < 1 || cols < 16 || cols % 16 != 0)
        return -1;
    elements = (uint64_t)(uint32_t)rows * (uint64_t)(uint32_t)cols;
    if (wn != elements / 2u || sn != elements / 16u || wgn != 4u || ign != 4u ||
        checked_extent(find_map(model, wfd), woff, wn, &weight) != 0 ||
        checked_extent(find_map(model, sfd), soff, sn, &scales) != 0 ||
        checked_extent(find_map(model, wgfd), wgoff, wgn, &weight_global) != 0 ||
        checked_extent(find_map(model, igfd), igoff, ign, &input_global) != 0)
        return -1;
    out->set = 1;
    out->bits = 40;
    out->rows = rows;
    out->cols = cols;
    out->weight = (const uint32_t *)weight;
    out->scales = (const uint16_t *)scales;
    out->weight_global_scale = (const float *)weight_global;
    out->input_global_scale = (const float *)input_global;
    out->nv_packed_scratch = model->nvfp4_packed_scratch;
    out->nv_scale_scratch = model->nvfp4_scale_scratch;
    out->nv_dequant_scratch = model->nvfp4_dequant_scratch;
    return 0;
}

static G4Q *nvfp4_expert_role(SaltGemma4Text *model, const char *role) {
    int layer, expert, used = -1, component = -1;
    const char *suffix;
    if (!model || !model->nvfp4_experts || !role ||
        sscanf(role, "text.layers.%d.experts.%d.%n", &layer, &expert, &used) != 2 ||
        used < 0 || layer < 0 || layer >= G4_LAYERS ||
        expert < 0 || expert >= G4_EXPERTS)
        return NULL;
    suffix = role + used;
    if (!strcmp(suffix, "gate_proj")) component = 0;
    else if (!strcmp(suffix, "up_proj")) component = 1;
    else if (!strcmp(suffix, "down_proj")) component = 2;
    if (component < 0) return NULL;
    return &model->nvfp4_experts[
        ((size_t)layer * G4_EXPERTS + (size_t)expert) * 3u +
        (size_t)component];
}

static int q_shape(const G4Q *q, int bits, int rows, int cols) {
    return q && q->set && q->bits == bits && q->rows == rows && q->cols == cols;
}

static int b_shape(const G4B *b, int count) {
    return b && b->set && b->count == count;
}

static int layer_suffix(const char *role, int *layer, const char **suffix) {
    int used = -1;
    if (sscanf(role, "text.layers.%d.%n", layer, &used) != 1 ||
        used < 0 || *layer < 0 || *layer >= G4_LAYERS || role[used] == 0)
        return -1;
    *suffix = role + used;
    return 0;
}

static G4Q *q_role(SaltGemma4Text *model, const char *role) {
    int layer;
    const char *suffix;
    if (!strcmp(role, "text.embedding")) return &model->embedding;
    if (layer_suffix(role, &layer, &suffix) != 0) return NULL;
    G4Layer *l = &model->layers[layer];
    if (!strcmp(suffix, "attention.q_proj")) return &l->q_proj;
    if (!strcmp(suffix, "attention.k_proj")) return &l->k_proj;
    if (!strcmp(suffix, "attention.v_proj")) return &l->v_proj;
    if (!strcmp(suffix, "attention.kv_proj")) return &l->kv_proj;
    if (!strcmp(suffix, "attention.o_proj")) return &l->o_proj;
    if (!strcmp(suffix, "dense.gate_proj")) return &l->dense_gate;
    if (!strcmp(suffix, "dense.up_proj")) return &l->dense_up;
    if (!strcmp(suffix, "dense.down_proj")) return &l->dense_down;
    if (!strcmp(suffix, "router.proj")) return &l->router;
    return NULL;
}

static G4B *b_role(SaltGemma4Text *model, const char *role) {
    int layer;
    const char *suffix;
    if (!strcmp(role, "text.final_norm")) return &model->final_norm;
    if (layer_suffix(role, &layer, &suffix) != 0) return NULL;
    G4Layer *l = &model->layers[layer];
    if (!strcmp(suffix, "attention.q_norm")) return &l->q_norm;
    if (!strcmp(suffix, "attention.k_norm")) return &l->k_norm;
    if (!strcmp(suffix, "input_layernorm")) return &l->input_norm;
    if (!strcmp(suffix, "post_attention_layernorm")) return &l->post_attention_norm;
    if (!strcmp(suffix, "pre_feedforward_layernorm")) return &l->pre_ffn_norm;
    if (!strcmp(suffix, "pre_feedforward_layernorm_2")) return &l->pre_ffn_norm_2;
    if (!strcmp(suffix, "post_feedforward_layernorm")) return &l->post_ffn_norm;
    if (!strcmp(suffix, "post_feedforward_layernorm_1")) return &l->post_ffn_norm_1;
    if (!strcmp(suffix, "post_feedforward_layernorm_2")) return &l->post_ffn_norm_2;
    if (!strcmp(suffix, "router.scale")) return &l->router_scale;
    if (!strcmp(suffix, "router.per_expert_scale")) return &l->per_expert_scale;
    if (!strcmp(suffix, "layer_scalar")) return &l->layer_scalar;
    return NULL;
}

static int registered_layer_matches(const SaltGemma4Text *model, int layer,
                                    const G4Layer *loaded) {
    const SaltTextLayerPlan *plan;
    const SaltAttentionDesc *attention;
    const SaltKvLayerPlan *kv;
    int full;
    if (!model || !loaded || layer < 0 || layer >= G4_LAYERS)
        return 0;
    plan = &model->layer_plan[layer];
    attention = &plan->attention;
    kv = &model->kv_plan[layer];
    full = attention->kind == SALT_ATTN_FULL;
    return plan->layer == layer &&
        plan->runtime_ready == model->model_desc->runtime_ready &&
        plan->hidden == G4_HIDDEN &&
        plan->n_experts == G4_EXPERTS &&
        plan->top_k_experts == G4_TOPK &&
        plan->expert_intermediate == G4_ROUTED &&
        plan->dense_intermediate == G4_DENSE &&
        plan->activation == SALT_ACT_GELU_TANH &&
        plan->router_rmsnorm == 1 &&
        plan->parallel_dense_routed == 1 &&
        plan->residual_postnorm == 1 &&
        plan->final_layer_scale == 1 &&
        plan->tied_embeddings == 1 &&
        plan->vision_bidirectional_local == 1 &&
        plan->logit_softcap == G4_LOGIT_CAP &&
        attention->causal == 1 &&
        attention->n_heads == 16 &&
        attention->n_kv_heads == loaded->kv_heads &&
        attention->head_dim == loaded->head_dim &&
        attention->rope_dim == loaded->rope_dim &&
        attention->rope_base_dim == loaded->head_dim &&
        attention->window == (full ? 0 : G4_SLIDING_WINDOW) &&
        attention->shared_kv_projection == full &&
        (float)attention->rope_theta == loaded->rope_theta &&
        attention->score_scale == 1.0f &&
        kv->token_capacity > 0 &&
        kv->k_width == loaded->kv_dim &&
        kv->v_width == loaded->kv_dim;
}

static int validate_closure(SaltGemma4Text *model) {
    int matrix_bits = model && model->nvfp4_mode ? 40 : 4;
    int router_bits = model && model->nvfp4_mode ? 16 : 8;
    int embedding_bits = model && model->nvfp4_mode ? 16 : 4;
    if (model->binding_count != G4_EXPECTED_BINDINGS ||
        !q_shape(&model->embedding, embedding_bits, G4_VOCAB, G4_HIDDEN) ||
        !b_shape(&model->final_norm, G4_HIDDEN)) return -1;
    if (model->nvfp4_mode && model->nvfp4_expert_bindings !=
            G4_LAYERS * G4_EXPERTS * 3)
        return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *l = &model->layers[i];
        l->full_attention = (i % 6) == 5;
        l->head_dim = l->full_attention ? 512 : 256;
        l->kv_heads = l->full_attention ? 2 : 8;
        l->kv_dim = l->head_dim * l->kv_heads;
        l->q_dim = l->head_dim * 16;
        l->rope_dim = l->full_attention ? l->head_dim / 4 : l->head_dim;
        l->rope_theta = l->full_attention ? 1000000.0f : 10000.0f;
        l->plan = &model->layer_plan[i];
        if (!registered_layer_matches(model, i, l)) return -1;
        if (!q_shape(&l->q_proj, matrix_bits, l->q_dim, G4_HIDDEN) ||
            !q_shape(&l->o_proj, matrix_bits, G4_HIDDEN, l->q_dim) ||
            !q_shape(&l->dense_gate, matrix_bits, G4_DENSE, G4_HIDDEN) ||
            !q_shape(&l->dense_up, matrix_bits, G4_DENSE, G4_HIDDEN) ||
            !q_shape(&l->dense_down, matrix_bits, G4_HIDDEN, G4_DENSE) ||
            !q_shape(&l->router, router_bits, G4_EXPERTS, G4_HIDDEN) ||
            !b_shape(&l->q_norm, l->head_dim) ||
            !b_shape(&l->k_norm, l->head_dim) ||
            !b_shape(&l->input_norm, G4_HIDDEN) ||
            !b_shape(&l->post_attention_norm, G4_HIDDEN) ||
            !b_shape(&l->pre_ffn_norm, G4_HIDDEN) ||
            !b_shape(&l->pre_ffn_norm_2, G4_HIDDEN) ||
            !b_shape(&l->post_ffn_norm, G4_HIDDEN) ||
            !b_shape(&l->post_ffn_norm_1, G4_HIDDEN) ||
            !b_shape(&l->post_ffn_norm_2, G4_HIDDEN) ||
            !b_shape(&l->router_scale, G4_HIDDEN) ||
            !b_shape(&l->per_expert_scale, G4_EXPERTS) ||
            !b_shape(&l->layer_scalar, 1)) return -1;
        if (l->full_attention) {
            if (!q_shape(&l->kv_proj, matrix_bits, l->kv_dim, G4_HIDDEN) ||
                l->k_proj.set || l->v_proj.set) return -1;
        } else {
            if (!q_shape(&l->k_proj, matrix_bits, l->kv_dim, G4_HIDDEN) ||
                !q_shape(&l->v_proj, matrix_bits, l->kv_dim, G4_HIDDEN) ||
                l->kv_proj.set) return -1;
        }
    }
    return 0;
}

int salt_gemma4_text_validate_proof(SaltGemma4Text *model) {
    if (!model || model->proof_validation_calls == UINT64_MAX) return -1;
    model->proof_validation_calls++;
    return all_identities_match(model) && validate_closure(model) == 0 ? 0 : -1;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
int salt_gemma4_text_set_proof_state(SaltGemma4Text *model, int enabled) {
    if (!model || !model->expert_cache_set) return -1;
    if (salt_cache_set_proof_state(&model->expert_cache, enabled) != 0)
        return -1;
    model->text_program.proof_state = enabled;
    return 0;
}

const char *salt_gemma4_text_expert_cache_mode(const SaltGemma4Text *model) {
    if (!model || !model->expert_cache_set) return NULL;
    switch (model->expert_cache.mode) {
    case 0: return "arena";
    case 1: return "flat-mmap-zero-copy";
    case 2: return "mixture";
    case 3: return "bounded-mmap-zero-copy";
    default: return NULL;
    }
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int add_map(SaltGemma4Text *model, const char *name, int fd,
                   unsigned long long dev, unsigned long long ino,
                   unsigned long long bytes, long long mtime, long long ctime) {
    struct stat st;
    long long actual_mtime, actual_ctime;
    if (model->map_count >= G4_MAX_MAPS || fd < 0 || !name || strlen(name) >= 64 ||
        bytes == 0 || bytes > SIZE_MAX || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode))
        return -1;
    stat_times(&st, &actual_mtime, &actual_ctime);
    if ((unsigned long long)st.st_dev != dev ||
        (unsigned long long)st.st_ino != ino ||
        (unsigned long long)st.st_size != bytes ||
        actual_mtime != mtime || actual_ctime != ctime || find_map(model, fd))
        return -1;
    /* Source authority is immutable and retained by fd. A read-only shared
     * view remains zero-copy while avoiding compressed private backing. */
    void *base = mmap(NULL, (size_t)bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) return -1;
    G4Map *map = &model->maps[model->map_count++];
    memset(map, 0, sizeof *map);
    map->fd = fd;
    snprintf(map->name, sizeof map->name, "%s", name);
    map->dev = dev;
    map->ino = ino;
    map->size = (size_t)bytes;
    map->map_len = (size_t)bytes;
    map->mtime_ns = mtime;
    map->ctime_ns = ctime;
    map->base = (unsigned char *)base;
    return 0;
}

static int g4_stable_pageable_pool(const SaltGemma4Text *model) {
    return model && model->full_gpu_intent && model->gpu_bounded_weights &&
        !model->nvfp4_mode && !model->gpu_expert_layer_view &&
        model->gpu_weight_addressability == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE;
}

static int add_pool(SaltGemma4Text *model, int fd,
                    unsigned long long dev, unsigned long long ino,
                    unsigned long long bytes, long long mtime, long long ctime,
                    unsigned long long slot, unsigned long long layers,
                    unsigned long long experts) {
    struct stat st;
    long long actual_mtime, actual_ctime;
    if (model->pool_set || fd < 0 || bytes != 12846366744ULL ||
        slot != G4_SLOT_BYTES || layers != G4_LAYERS || experts != G4_EXPERTS ||
        fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return -1;
    stat_times(&st, &actual_mtime, &actual_ctime);
    if ((unsigned long long)st.st_dev != dev ||
        (unsigned long long)st.st_ino != ino ||
        (unsigned long long)st.st_size != bytes ||
        actual_mtime != mtime || actual_ctime != ctime) return -1;
    /* Pageable CUDA requires one canonical process-lifetime virtual mapping.
     * The cache still bounds READY/resident expert ranges and releases pages;
     * mapping the file does not preload, register, or copy its payload. */
    int stable_pageable = g4_stable_pageable_pool(model);
    size_t map_len = model->gpu_bounded_weights && !stable_pageable
        ? G4_POOL_HEADER : (model->full_gpu_intent ? (size_t)bytes
                                                   : G4_POOL_HEADER);
    void *base = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) return -1;
    uint64_t header[3];
    memcpy(header, base, sizeof header);
    if (header[0] != slot || header[1] != layers || header[2] != experts) {
        munmap(base, map_len);
        return -1;
    }
    memset(&model->pool, 0, sizeof model->pool);
    model->pool.fd = fd;
    model->pool.dev = dev;
    model->pool.ino = ino;
    model->pool.size = (size_t)bytes;
    model->pool.map_len = map_len;
    model->pool.mtime_ns = mtime;
    model->pool.ctime_ns = ctime;
    model->pool.base = (unsigned char *)base;
    model->pool_set = 1;
    return 0;
}

static int parse_preload_layers(uint64_t *mask_out, int *count_out) {
    const char *text = getenv("SALT_EXPERT_PRELOAD_LAYERS");
    const char *cursor;
    uint64_t mask = 0;
    int count = 0;
    if (!mask_out || !count_out || !text || !*text) return -1;
    if (!strcmp(text, "none")) {
        *mask_out = 0;
        *count_out = 0;
        return 0;
    }
    cursor = text;
    while (*cursor) {
        int layer = 0;
        if (*cursor < '0' || *cursor > '9' ||
            (*cursor == '0' && cursor[1] >= '0' && cursor[1] <= '9'))
            return -1;
        while (*cursor >= '0' && *cursor <= '9') {
            layer = layer * 10 + (*cursor - '0');
            if (layer >= G4_LAYERS) return -1;
            cursor++;
        }
        if (mask & (UINT64_C(1) << layer)) return -1;
        mask |= UINT64_C(1) << layer;
        count++;
        if (!*cursor) break;
        if (*cursor != ',' || !cursor[1]) return -1;
        cursor++;
    }
    *mask_out = mask;
    *count_out = count;
    return count > 0 ? 0 : -1;
}

static int preload_expert_layers(SaltGemma4Text *model, uint64_t layer_mask,
                                 int layer_count, int protect) {
    int experts[G4_EXPERTS], slots[G4_EXPERTS];
    int64_t prior_touch;
    uint64_t jobs = 0;
    int expected, seen_layers = 0;
    if (!model || !model->expert_cache_set ||
        layer_count < 1 || layer_count > G4_LAYERS)
        return -1;
    expected = layer_count * G4_EXPERTS;
    for (int expert = 0; expert < G4_EXPERTS; expert++)
        experts[expert] = expert;
    prior_touch = model->expert_cache.fetch_touch_bytes;
    /* A preload widens only an enabled operational touch. Do not turn a
     * registered all-expert preload into an implicit whole-pool page walk. */
    if (prior_touch > 0)
        model->expert_cache.fetch_touch_bytes = G4_SLOT_BYTES;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        int fetched;
        if (!(layer_mask & (UINT64_C(1) << layer))) continue;
        seen_layers++;
        fetched = salt_cache_getmany(&model->expert_cache, layer, experts,
                                     G4_EXPERTS, slots);
        if (fetched < 0) {
            model->expert_cache.fetch_touch_bytes = prior_touch;
            return -1;
        }
        jobs += (uint64_t)fetched;
        for (int expert = 0; expert < G4_EXPERTS; expert++) {
            if (!salt_cache_slot_for(&model->expert_cache, slots[expert],
                                     layer, expert) ||
                (protect && !salt_cache_pin_expert(
                    &model->expert_cache, layer, expert))) {
                model->expert_cache.fetch_touch_bytes = prior_touch;
                return -1;
            }
        }
    }
    model->expert_cache.fetch_touch_bytes = prior_touch;
    if (seen_layers != layer_count || jobs != (uint64_t)expected ||
        (protect && salt_cache_npinned(&model->expert_cache) != expected))
        return -1;
    model->expert_preload_enabled = 1;
    model->expert_preloaded_slots = expected;
    model->expert_preload_layer_count = layer_count;
    model->expert_preload_protected_slots = protect ? expected : 0;
    model->expert_preload_layer_mask = layer_mask;
    model->expert_preload_fetch_jobs = jobs;
    model->expert_cache_peak_slots = expected;
    return 0;
}

static int init_expert_cache(SaltGemma4Text *model) {
    const size_t count = (size_t)G4_LAYERS * G4_EXPERTS;
    SaltExpertPool *pool;
    uint64_t budget_bytes, slots, preload_mask;
    int budget_gb, preload, preload_count;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    const char *cache_mode = getenv("SALT_CACHE_MODE");
    int cpu_zerocopy = cache_mode && strcmp(cache_mode, "zerocopy") == 0;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (!model || !model->pool_set || !model->pool.base ||
        g4_recipe_int("SALT_EXPERT_BUDGET_GB", 1, 13, &budget_gb) != 0 ||
        g4_recipe_int("SALT_EXPERT_PRELOAD", 0, 1, &preload) != 0 ||
        parse_preload_layers(&preload_mask, &preload_count) != 0 ||
        (preload && (budget_gb < 13 || preload_count != 0)))
        return -1;
    budget_bytes = (uint64_t)budget_gb * G4_EXPERT_BUDGET_UNIT;
    slots = budget_bytes / (uint64_t)G4_SLOT_BYTES;
    if (slots > count) slots = count;
    if (slots < G4_TOPK || slots > INT_MAX ||
        (preload_count > 0 &&
         (uint64_t)preload_count * G4_EXPERTS + G4_EXPERTS > slots))
        return -1;
    model->expert_budget_bytes = budget_bytes;
    model->expert_cache_slots = (int)slots;
    pool = &model->cache_pool;
    memset(pool, 0, sizeof *pool);
    pool->fd = model->pool.fd;
    pool->owns_fd = 0;
    int stable_pageable = g4_stable_pageable_pool(model);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    pool->map = model->full_gpu_intent &&
        (!model->gpu_bounded_weights || stable_pageable)
        ? model->pool.base : NULL;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    pool->map = (model->full_gpu_intent &&
        (!model->gpu_bounded_weights || stable_pageable)) || cpu_zerocopy
        ? model->pool.base : NULL;
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    pool->map_len = model->pool.size;
    pool->n_layers = G4_LAYERS;
    pool->n_experts = G4_EXPERTS;
    pool->nbytes = G4_SLOT_BYTES;
    pool->ref = (SaltExpertRef *)calloc(count, sizeof *pool->ref);
    if (!pool->ref) return -1;
    for (size_t i = 0; i < count; i++) {
        size_t off = G4_POOL_HEADER + i * (size_t)G4_SLOT_BYTES;
        if (off > pool->map_len || G4_SLOT_BYTES > pool->map_len - off) {
            free(pool->ref);
            pool->ref = NULL;
            return -1;
        }
        pool->ref[i].off = (int64_t)off;
        pool->ref[i].nbytes = G4_SLOT_BYTES;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    int cache_rc = model->full_gpu_intent &&
            (!model->gpu_bounded_weights || stable_pageable)
        ? salt_cache_init_zerocopy(&model->expert_cache, pool,
                                   model->expert_cache_slots,
                                   model->expert_workers)
        : salt_cache_init_bounded_mmap(&model->expert_cache, pool,
                                       model->expert_cache_slots,
                                       model->expert_workers);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    int cache_rc = (model->full_gpu_intent &&
            (!model->gpu_bounded_weights || stable_pageable)) || cpu_zerocopy
        ? salt_cache_init_zerocopy(&model->expert_cache, pool,
                                   model->expert_cache_slots,
                                   model->expert_workers)
        : salt_cache_init_bounded_mmap(&model->expert_cache, pool,
                                       model->expert_cache_slots,
                                       model->expert_workers);
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (cache_rc != 0) {
        free(pool->ref);
        pool->ref = NULL;
        return -1;
    }
    model->expert_cache_set = 1;
    if (preload_count > 0) {
        if (!model->expert_cache.quota_enabled ||
            model->expert_cache.shared_slots < G4_EXPERTS)
            return -1;
        for (int layer = 0; layer < G4_LAYERS; layer++) {
            int expected_quota =
                (preload_mask & (UINT64_C(1) << layer)) ? G4_EXPERTS : 0;
            if (model->expert_cache.layer_quota[layer] != expected_quota)
                return -1;
        }
        if (preload_expert_layers(model, preload_mask,
                                  preload_count, 1) != 0)
            return -1;
    } else if (preload) {
        uint64_t all_layers = (UINT64_C(1) << G4_LAYERS) - UINT64_C(1);
        if (model->expert_cache.nslot != G4_LAYERS * G4_EXPERTS ||
            preload_expert_layers(model, all_layers, G4_LAYERS, 0) != 0)
            return -1;
    }
    return 0;
}

static int gpu_map_for_q(const SaltGemma4Text *model, const G4Q *q) {
    unsigned long long weight = 0, scale = 0;
    if (!model || !q || !q->set)
        return -1;
    if (q->bits == 16) {
        uint64_t elements = (uint64_t)(uint32_t)q->rows *
                            (uint64_t)(uint32_t)q->cols;
        if (elements > ULLONG_MAX / 2u) return -1;
        weight = elements * 2u;
    } else if (q->bits == 40) {
        uint64_t elements = (uint64_t)(uint32_t)q->rows *
                            (uint64_t)(uint32_t)q->cols;
        weight = elements / 2u;
        scale = elements / 16u;
    } else if (q_expected_bytes(q->bits, q->rows, q->cols,
                               &weight, &scale) != 0)
        return -1;
    for (int i = 0; i < model->map_count; i++) {
        const G4Map *map = &model->maps[i];
        uintptr_t lo = (uintptr_t)map->base;
        uintptr_t hi = lo + map->map_len;
        uintptr_t w = (uintptr_t)q->weight;
        uintptr_t s = (uintptr_t)q->scales;
        uintptr_t b = (uintptr_t)q->biases;
        if (q->bits == 16) {
            if (hi >= lo && w >= lo && w <= hi && weight <= hi - w)
                return i;
        } else if (q->bits == 40) {
            uintptr_t wg = (uintptr_t)q->weight_global_scale;
            uintptr_t ig = (uintptr_t)q->input_global_scale;
            if (hi >= lo && w >= lo && w <= hi && s >= lo && s <= hi &&
                wg >= lo && wg <= hi && ig >= lo && ig <= hi &&
                weight <= hi - w && scale <= hi - s &&
                sizeof(float) <= hi - wg && sizeof(float) <= hi - ig)
                return i;
        } else if (hi >= lo && w >= lo && w <= hi && s >= lo && s <= hi &&
                   b >= lo && b <= hi &&
                   weight <= hi - w && scale <= hi - s && scale <= hi - b)
            return i;
    }
    return -1;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
static int g4_residency_payload_read(
        void *context, uint64_t source_offset,
        void *destination, size_t nbytes) {
    const unsigned char *source = (const unsigned char *)context;
    if (!source || !destination || nbytes == 0 || source_offset != 0)
        return -1;
    memcpy(destination, source + (size_t)source_offset, nbytes);
    return 0;
}

static int g4_selected_cache_bind_resident(
        SaltGemma4Text *model, int slot, int layer, int expert,
        const void *base, size_t nbytes, const void *payload) {
    SaltGpuResidencySource source;
    SaltGpuResidencyBinding binding;
    uintptr_t base_address, payload_address, device_address = 0;
    uint64_t logical, generation, ticket = 0;
    uint32_t staging_slot;
    int poll_result, acquired = 0, published = 0;
    double population_started;
    SaltGpuResidencyStats residency_stats;
    if (!model || !model->gpu_residency.ready || !base || !payload ||
        slot < 0 || (uint32_t)slot >= model->gpu_residency.plan.slot_count ||
        layer < 0 || layer >= G4_LAYERS || expert < 0 || expert >= G4_EXPERTS ||
        nbytes < G4_SLOT_BYTES || model->gpu_residency_generation == UINT64_MAX)
        return -1;
    base_address = (uintptr_t)base;
    payload_address = (uintptr_t)payload;
    if (nbytes > UINTPTR_MAX - base_address || payload_address < base_address ||
        payload_address - base_address > nbytes ||
        G4_SLOT_BYTES > nbytes - (size_t)(payload_address - base_address))
        return -1;
    logical = (uint64_t)(uint32_t)layer * G4_EXPERTS + (uint32_t)expert;
    generation = ++model->gpu_residency_generation;
    staging_slot = (uint32_t)((generation - 1u) %
        model->gpu_residency.plan.staging_count);
    memset(&source, 0, sizeof source);
    source.resource.kind = SALT_GPU_RESOURCE_EXPERT_LAYER;
    source.resource.resource_id = (uint32_t)logical;
    source.logical_resource_id = logical;
    source.nbytes = G4_SLOT_BYTES;
    source.read = g4_residency_payload_read;
    source.read_context = (void *)payload;
    population_started = g4_now_s();
    if (salt_gpu_residency_population_begin(
            &model->gpu_residency, (uint32_t)slot, generation,
            &source, staging_slot, &ticket) != 0)
        return -1;
    do {
        poll_result = salt_gpu_residency_population_poll(
            &model->gpu_residency, (uint32_t)slot,
            logical, generation, ticket);
    } while (poll_result == 0);
    if (poll_result != 1 ||
        salt_gpu_residency_acquire(
            &model->gpu_residency, (uint32_t)slot,
            logical, generation, &binding) != 1)
        goto fail;
    acquired = 1;
    if (salt_gpu_residency_resolve(
            &model->gpu_residency, &binding, &device_address) != 0 ||
        salt_gpu_selected_resource_bind_resident(
            slot, logical, payload, G4_SLOT_BYTES,
            device_address, generation) != 0)
        goto fail;
    published = 1;
    if (salt_gpu_residency_release(&model->gpu_residency, &binding) != 0)
        return -1;
    if (getenv("SALT_GPU_DIAG") &&
        salt_gpu_residency_stats(
            &model->gpu_residency, &residency_stats,
            sizeof residency_stats) == 0 &&
        (residency_stats.population_submissions == 1u ||
         residency_stats.population_submissions % 64u == 0u))
        fprintf(stderr,
            "GEMMA4_GPU_EXPERT_RESIDENCY event=populate slot=%d "
            "logical=%llu generation=%llu bytes=%u wall_ns=%llu "
            "submissions=%llu completions=%llu aborts=%llu "
            "completed_bytes=%llu retirements=%llu ready_slots=%u "
            "peak_ready_slots=%u active_leases=%u\n",
            slot, (unsigned long long)logical,
            (unsigned long long)generation, (unsigned int)G4_SLOT_BYTES,
            (unsigned long long)((g4_now_s() - population_started) * 1e9),
            (unsigned long long)residency_stats.population_submissions,
            (unsigned long long)residency_stats.population_completions,
            (unsigned long long)residency_stats.population_aborts,
            (unsigned long long)residency_stats.population_bytes_completed,
            (unsigned long long)residency_stats.retirements,
            residency_stats.ready_slots, residency_stats.peak_ready_slots,
            residency_stats.active_leases);
    return 0;
fail:
    if (published)
        (void)salt_gpu_selected_resource_unbind(slot, logical);
    if (acquired)
        (void)salt_gpu_residency_release(&model->gpu_residency, &binding);
    if (model->gpu_residency.slots[slot].state == SALT_GPU_RESIDENCY_POPULATING)
        (void)salt_gpu_residency_population_abort(
            &model->gpu_residency, (uint32_t)slot,
            logical, generation, ticket);
    else if (model->gpu_residency.slots[slot].state == SALT_GPU_RESIDENCY_READY)
        (void)salt_gpu_residency_retire(
            &model->gpu_residency, (uint32_t)slot, logical, generation);
    return -1;
}

static int g4_selected_cache_unbind_resident(
        SaltGemma4Text *model, int slot, int layer, int expert) {
    SaltGpuResidencySlot *resident;
    uint64_t logical;
    if (!model || !model->gpu_residency.ready || slot < 0 ||
        (uint32_t)slot >= model->gpu_residency.plan.slot_count ||
        layer < 0 || layer >= G4_LAYERS || expert < 0 || expert >= G4_EXPERTS)
        return -1;
    logical = (uint64_t)(uint32_t)layer * G4_EXPERTS + (uint32_t)expert;
    resident = &model->gpu_residency.slots[slot];
    if (resident->state != SALT_GPU_RESIDENCY_READY ||
        resident->logical_resource_id != logical || resident->generation == 0 ||
        resident->lease_count != 0 ||
        salt_gpu_selected_resource_unbind(slot, logical) != 0)
        return -1;
    return salt_gpu_residency_retire(
        &model->gpu_residency, (uint32_t)slot,
        logical, resident->generation);
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_selected_cache_bind(void *opaque, int slot,
                                  int layer, int expert,
                                  const void *base, size_t nbytes,
                                  const void *payload) {
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    uint64_t logical;
    if (!model || layer < 0 || layer >= G4_LAYERS ||
        expert < 0 || expert >= G4_EXPERTS)
        return -1;
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    uint64_t logical;
    (void)opaque;
    if (layer < 0 || layer >= G4_LAYERS ||
        expert < 0 || expert >= G4_EXPERTS)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (model->gpu_device_residency)
        return g4_selected_cache_bind_resident(
            model, slot, layer, expert, base, nbytes, payload);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    logical = (uint64_t)(uint32_t)layer * G4_EXPERTS + (uint32_t)expert;
    return salt_gpu_selected_resource_bind(
        slot, logical, base, nbytes, payload);
}

static int g4_selected_cache_fence(void *opaque) {
    return opaque ? salt_gpu_selected_resources_fence() : -1;
}

static int g4_selected_cache_unbind(void *opaque, int slot,
                                    int layer, int expert) {
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    uint64_t logical;
    if (!model || layer < 0 || layer >= G4_LAYERS ||
        expert < 0 || expert >= G4_EXPERTS)
        return -1;
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    uint64_t logical;
    (void)opaque;
    if (layer < 0 || layer >= G4_LAYERS ||
        expert < 0 || expert >= G4_EXPERTS)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (model->gpu_device_residency)
        return g4_selected_cache_unbind_resident(
            model, slot, layer, expert);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    logical = (uint64_t)(uint32_t)layer * G4_EXPERTS + (uint32_t)expert;
    return salt_gpu_selected_resource_unbind(slot, logical);
}

static int g4_text_gpu_resource_acquire(
        void *opaque, uint32_t layer, const int32_t *experts,
        uint32_t count, int32_t *slots) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    int requested[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    int acquired[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    const uint8_t *payloads[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (!model || !model->expert_cache_set || !experts || !slots ||
        layer >= G4_LAYERS || count == 0 ||
        count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
        return -1;
    for (uint32_t index = 0; index < count; index++) {
        if (experts[index] < 0 || experts[index] >= G4_EXPERTS)
            return -1;
        requested[index] = experts[index];
        acquired[index] = -1;
    }
    if (salt_cache_getmany(&model->expert_cache, (int)layer,
            requested, (int)count, acquired) < 0)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    for (uint32_t index = 0; index < count; index++) {
        if (acquired[index] < 0 || !salt_cache_acquire(
                &model->expert_cache, acquired[index], (int)layer,
                requested[index])) {
            while (index > 0) {
                index--;
                (void)salt_cache_release(&model->expert_cache,
                    acquired[index], (int)layer, requested[index]);
            }
            return -1;
        }
        slots[index] = acquired[index];
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (salt_cache_acquire_many(&model->expert_cache, acquired, (int)layer,
            requested, (int)count, payloads) != 0)
        return -1;
    for (uint32_t index = 0; index < count; index++) {
        if (acquired[index] < 0 || !payloads[index]) {
            (void)salt_cache_release_many(&model->expert_cache, acquired,
                (int)layer, requested, (int)count);
            return -1;
        }
        slots[index] = acquired[index];
    }
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    return 0;
}

static int g4_text_gpu_resource_release(
        void *opaque, uint32_t layer, const int32_t *experts,
        const int32_t *slots, uint32_t count) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    int rc = 0;
    if (!model || !experts || !slots || layer >= G4_LAYERS || count == 0 ||
        count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
        return -1;
    for (uint32_t index = count; index > 0; index--)
        if (salt_cache_release(&model->expert_cache, slots[index - 1u],
                (int)layer, experts[index - 1u]) != 0)
            rc = -1;
    return rc;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (!model || !experts || !slots || layer >= G4_LAYERS || count == 0 ||
        count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
        return -1;
    return salt_cache_release_many(&model->expert_cache, slots, (int)layer,
        experts, (int)count);
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
}

static const SaltTextExpertResourceOps g4_text_gpu_resource_ops = {
    g4_text_gpu_resource_acquire,
    g4_text_gpu_resource_release,
};

static int gpu_register_q(SaltGemma4Text *model, const G4Q *q) {
    int map_id;
    if (!q || !q->set) return 0;
    map_id = gpu_map_for_q(model, q);
    if (map_id < 0) return -1;
    if (q->bits == 16)
        return salt_gpu_bf16_resource_slot(SALT_GPU_RESOURCE_TRUNK,
            (uint32_t)map_id, q->weight,
            (const uint16_t *)(const void *)q->weight, q->rows, q->cols);
    if (q->bits == 40)
        return salt_gpu_nvfp4_resource_slot(SALT_GPU_RESOURCE_TRUNK,
            (uint32_t)map_id, q->weight,
            (const uint8_t *)(const void *)q->weight,
            (const uint8_t *)(const void *)q->scales,
            q->weight_global_scale, q->input_global_scale,
            q->rows, q->cols);
    return salt_gpu_weight_resource_slot(SALT_GPU_RESOURCE_TRUNK,
        (uint32_t)map_id, q->weight, q->weight, q->scales, q->biases,
        q->bits, q->rows, q->cols);
}

static int g4_trunk_layer_window_end(SaltGemma4Text *model) {
    int rc = 0;
    if (!model) return -1;
    for (int map_id = model->map_count - 1; map_id >= 0; map_id--) {
        uint32_t bit = (uint32_t)1u << map_id;
        if (!(model->gpu_trunk_window_mask & bit)) continue;
        if (salt_gpu_weight_window_unbind(
                SALT_GPU_RESOURCE_TRUNK, (uint32_t)map_id) != 0) {
            rc = -1;
            continue;
        }
        model->gpu_trunk_window_mask &= ~bit;
    }
    return rc;
}

static int g4_trunk_layer_window_begin(SaltGemma4Text *model, int layer) {
    uintptr_t lo[G4_MAX_MAPS], hi[G4_MAX_MAPS];
    long page_l;
    if (!model || layer < 0 || layer >= G4_LAYERS) return -1;
    if (!model->gpu_trunk_layer_view) return 0;
    if (!model->full_gpu_intent || model->gpu_trunk_window_mask != 0 ||
        model->map_count < 1 || model->map_count > G4_MAX_MAPS ||
        (page_l = sysconf(_SC_PAGESIZE)) <= 0)
        return -1;
    for (int i = 0; i < G4_MAX_MAPS; i++) {
        lo[i] = UINTPTR_MAX;
        hi[i] = 0;
    }
    G4Layer *entry = &model->layers[layer];
    G4Q *matrices[] = {
        &entry->q_proj, &entry->k_proj, &entry->v_proj,
        &entry->kv_proj, &entry->o_proj, &entry->dense_gate,
        &entry->dense_up, &entry->dense_down, &entry->router,
    };
    for (size_t i = 0; i < sizeof matrices / sizeof matrices[0]; i++) {
        G4Q *q = matrices[i];
        unsigned long long weight_bytes = 0, scale_bytes = 0;
        if (!q->set) continue;
        if (q_expected_bytes(q->bits, q->rows, q->cols,
                &weight_bytes, &scale_bytes) != 0)
            goto fail;
        int map_id = gpu_map_for_q(model, q);
        if (map_id < 0 || map_id >= model->map_count) goto fail;
        const uintptr_t starts[3] = {
            (uintptr_t)(const void *)q->weight,
            (uintptr_t)(const void *)q->scales,
            (uintptr_t)(const void *)q->biases,
        };
        const uint64_t sizes[3] = {
            (uint64_t)weight_bytes,
            (uint64_t)scale_bytes,
            (uint64_t)scale_bytes,
        };
        for (int part = 0; part < 3; part++) {
            if (starts[part] > UINTPTR_MAX - sizes[part]) goto fail;
            uintptr_t end = starts[part] + (uintptr_t)sizes[part];
            if (starts[part] < lo[map_id]) lo[map_id] = starts[part];
            if (end > hi[map_id]) hi[map_id] = end;
        }
    }
    for (int map_id = 0; map_id < model->map_count; map_id++) {
        if (lo[map_id] == UINTPTR_MAX) continue;
        uintptr_t map_base = (uintptr_t)(const void *)model->maps[map_id].base;
        uintptr_t aligned = lo[map_id] - lo[map_id] % (uintptr_t)page_l;
        if (aligned < map_base || hi[map_id] <= aligned ||
            aligned - map_base > UINT64_MAX ||
            hi[map_id] - aligned > SIZE_MAX ||
            salt_gpu_weight_window_bind(SALT_GPU_RESOURCE_TRUNK,
                (uint32_t)map_id, (uint64_t)(aligned - map_base),
                (size_t)(hi[map_id] - aligned)) != 0)
            goto fail;
        model->gpu_trunk_window_mask |= (uint32_t)1u << map_id;
    }
    return 0;
fail:
    (void)g4_trunk_layer_window_end(model);
    return -1;
}

static int g4_prefill_ffn_scratch_bytes(int capacity, size_t *bytes_out);
static int g4_prefill_attention_scratch_bytes(int capacity,
                                               size_t *bytes_out);
static int g4_operation_arena_bytes(int capacity, size_t *bytes_out);

static int g4_u64_add(uint64_t *total, uint64_t value) {
    if (!total || value > UINT64_MAX - *total) return -1;
    *total += value;
    return 0;
}

static int g4_resolve_weight_addressability(
    SaltGemma4Text *model, SaltGpuWeightAddressability *policy_out) {
    const char *pageable = getenv("SALT_CUDA_PAGEABLE_MMAP");
    SaltGpuWeightAddressability policy;
    int pageable_requested;
    if (!model || !policy_out ||
        (pageable && strcmp(pageable, "0") && strcmp(pageable, "1")))
        return -1;
    pageable_requested = pageable && !strcmp(pageable, "1");
    policy = model->gpu_weight_addressability;
    if (policy == SALT_GPU_WEIGHT_ADDRESS_AUTO) {
        if (pageable_requested)
            policy = SALT_GPU_WEIGHT_ADDRESS_PAGEABLE;
        else if (model->gpu_bounded_weights)
            policy = SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW;
        else
            policy = SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT;
    }
    if ((policy == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE && !pageable_requested) ||
        (policy != SALT_GPU_WEIGHT_ADDRESS_PAGEABLE && pageable_requested) ||
        policy == SALT_GPU_WEIGHT_ADDRESS_AUTO)
        return -1;
    model->gpu_weight_addressability = policy;
    *policy_out = policy;
    return 0;
}

static int g4_decode_arena_floats(const SaltGemma4Text *model,
                                  uint64_t *floats_out) {
    uint64_t floats = 0;
    if (!model || !floats_out || model->max_context < 1) return -1;
#define ADD_FLOATS(count) do { \
    uint64_t add = (uint64_t)(count); \
    if (add > UINT64_MAX - floats) return -1; \
    floats += add; \
} while (0)
    ADD_FLOATS(6u * G4_HIDDEN);
    ADD_FLOATS(G4_MAX_Q + 2u * G4_MAX_KV + G4_MAX_Q);
    ADD_FLOATS(2u * G4_DENSE + 5u * G4_HIDDEN);
    ADD_FLOATS(G4_EXPERTS + G4_TOPK * G4_HIDDEN);
    ADD_FLOATS(2u * G4_TOPK * G4_ROUTED);
    ADD_FLOATS((uint32_t)model->max_context);
    ADD_FLOATS(2u * G4_TOPK + G4_VOCAB);
#undef ADD_FLOATS
    *floats_out = floats;
    return 0;
}

static int g4_runtime_host_bytes(const SaltGemma4Text *model,
                                 uint64_t *bytes_out) {
    uint64_t floats, bytes = sizeof *model;
    if (!model || !bytes_out ||
        g4_decode_arena_floats(model, &floats) != 0 ||
        floats > UINT64_MAX / sizeof(float) ||
        (!model->full_gpu_intent &&
         g4_u64_add(&bytes, floats * sizeof(float)) != 0) ||
        (!model->full_gpu_intent &&
         g4_u64_add(&bytes, model->owned_bf16_bytes) != 0) ||
        (uint64_t)model->owned_capacity > UINT64_MAX / sizeof(float *) ||
        g4_u64_add(&bytes,
            (uint64_t)model->owned_capacity * sizeof(float *)) != 0)
        return -1;
    *bytes_out = bytes;
    return 0;
}

static int g4_startup_admit_full_gpu(SaltGemma4Text *model) {
    SaltGpuStartupRequirements backend;
    SaltGpuWeightAddressability policy;
    const SaltTextGraphDesc *graph;
    uint64_t source_bytes = 0, addressability_bytes = 0;
    uint64_t runtime_bytes, kv_bytes, resident_kv_bytes, rope_bytes = 0;
    uint64_t cache_bytes;
    uint64_t decode_floats, decode_bytes;
    uint64_t pool_ref_bytes, expert_residency_bytes, dense_bytes = 0;
    uint64_t total = 0, required_outputs, required_jobs;
    uint64_t attention_pool_bytes, attention_per;
    size_t ffn_bytes, attention_bytes, operation_bytes, scratch_bytes;
    int budget_gb, prefill_capacity, shared_arenas;
    int gpu_attention, compact_hmm_gpu_program;
    int attn_threads = 8, q4_pool_enabled = 1;
    int cache_slots, attn_workers;
    const char *attn_env, *q4_env;
    if (!model || !model->full_gpu_intent || !model->model_desc ||
        !(graph = model->model_desc->text_graph) ||
        salt_gpu_startup_requirements(&backend, sizeof backend) != 0 ||
        g4_resolve_weight_addressability(model, &policy) != 0 ||
        g4_recipe_int("SALT_EXPERT_BUDGET_GB", 1, 13, &budget_gb) != 0 ||
        g4_recipe_int("SALT_PREFILL_B", 1, 4096, &prefill_capacity) != 0 ||
        g4_recipe_int("SALT_PREFILL_SHARED_ARENAS", 0, 1,
            &shared_arenas) != 0 ||
        g4_recipe_int("SALT_PREFILL_GPU_ATTENTION", 0, 1,
            &gpu_attention) != 0)
        return -1;
    if (prefill_capacity > model->max_context)
        prefill_capacity = model->max_context;

    for (int i = 0; i < model->map_count; i++)
        if (g4_u64_add(&source_bytes, model->maps[i].map_len) != 0)
            return -1;
    if (!model->nvfp4_mode &&
        (!model->gpu_bounded_weights || g4_stable_pageable_pool(model)) &&
        g4_u64_add(&source_bytes, model->pool.map_len) != 0)
        return -1;
    expert_residency_bytes =
        (uint64_t)(uint32_t)budget_gb * G4_EXPERT_BUDGET_UNIT;
    cache_slots = (int)(expert_residency_bytes / G4_SLOT_BYTES);
    if (cache_slots > G4_LAYERS * G4_EXPERTS)
        cache_slots = G4_LAYERS * G4_EXPERTS;
    expert_residency_bytes = (uint64_t)(uint32_t)cache_slots * G4_SLOT_BYTES;
    if (policy == SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT) {
        model->startup_registered_bytes = source_bytes;
        addressability_bytes = source_bytes;
    } else if (policy == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE) {
        model->startup_pageable_bytes = source_bytes;
        if (model->gpu_bounded_weights) {
            if (dense_payload_bytes(model, &dense_bytes) != 0 ||
                g4_u64_add(&addressability_bytes, dense_bytes) != 0 ||
                g4_u64_add(&addressability_bytes,
                    expert_residency_bytes) != 0)
                return -1;
        } else {
            addressability_bytes = source_bytes;
        }
    } else if (policy == SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW) {
        if (dense_payload_bytes(model, &dense_bytes) != 0 ||
            g4_u64_add(&addressability_bytes, dense_bytes) != 0 ||
            g4_u64_add(&addressability_bytes, expert_residency_bytes) != 0)
            return -1;
    } else {
        return -1;
    }
    if (model->planned_kv_floats == 0 ||
        model->planned_kv_floats > UINT64_MAX / sizeof(float))
        return -1;
    kv_bytes = (uint64_t)model->planned_kv_floats * sizeof(float);
    compact_hmm_gpu_program = g4_compact_hmm_gpu_program(model);
    resident_kv_bytes = compact_hmm_gpu_program &&
            policy == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE ? 0 : kv_bytes;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* Device-live attention owns the operational KV allocation. Keep the
     * logical host KV geometry for diagnostics, but do not charge the same
     * bytes again to the host-resident startup forecast. */
    if (gpu_attention) resident_kv_bytes = 0;
    /* CUDA attention borrows mapped host KV, not a second device allocation. */
    if (salt_gpu_cuda_present()) resident_kv_bytes = kv_bytes;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (g4_prefill_ffn_scratch_bytes(prefill_capacity, &ffn_bytes) != 0 ||
        g4_prefill_attention_scratch_bytes(
            prefill_capacity, &attention_bytes) != 0 ||
        g4_operation_arena_bytes(prefill_capacity, &operation_bytes) != 0 ||
        g4_runtime_host_bytes(model, &runtime_bytes) != 0 ||
        g4_decode_arena_floats(model, &decode_floats) != 0 ||
        decode_floats > UINT64_MAX / sizeof(float))
        return -1;
    decode_bytes = decode_floats * sizeof(float);
    scratch_bytes = ffn_bytes > attention_bytes ? ffn_bytes : attention_bytes;
    if ((uint64_t)(uint32_t)model->max_context > UINT64_MAX /
            (uint64_t)(uint32_t)(graph->sliding.rope_dim / 2) ||
        (uint64_t)(uint32_t)model->max_context *
            (uint64_t)(uint32_t)(graph->sliding.rope_dim / 2) >
            UINT64_MAX / (2u * sizeof(float)))
        return -1;
    rope_bytes = (uint64_t)(uint32_t)model->max_context *
        (uint32_t)(graph->sliding.rope_dim / 2) * 2u * sizeof(float);
    if ((uint64_t)(uint32_t)model->max_context > UINT64_MAX /
            (uint64_t)(uint32_t)(graph->full.rope_dim / 2) ||
        (uint64_t)(uint32_t)model->max_context *
            (uint64_t)(uint32_t)(graph->full.rope_dim / 2) >
            UINT64_MAX / (2u * sizeof(float)) ||
        g4_u64_add(&rope_bytes,
            (uint64_t)(uint32_t)model->max_context *
            (uint32_t)(graph->full.rope_dim / 2) * 2u * sizeof(float)) != 0)
        return -1;
    if (salt_cache_startup_metadata_bytes(cache_slots, G4_LAYERS,
            G4_EXPERTS, model->expert_workers,
            model->gpu_bounded_weights && !g4_stable_pageable_pool(model),
            &cache_bytes) != 0)
        return -1;
    pool_ref_bytes = (uint64_t)G4_LAYERS * G4_EXPERTS *
        sizeof(SaltExpertRef);
    if (g4_u64_add(&cache_bytes, pool_ref_bytes) != 0) return -1;
    attn_env = getenv("SALT_ATTN_THREADS");
    if (attn_env) {
        int parsed = atoi(attn_env);
        if (parsed >= 1 && parsed <= 32) attn_threads = parsed;
    }
    q4_env = getenv("SALT_Q4_MATVEC2_POOL");
    q4_pool_enabled = !q4_env || *q4_env != '0';
    attn_workers = model->expert_workers;
    if (q4_pool_enabled && attn_threads > attn_workers)
        attn_workers = attn_threads;
    if ((int)model->target_policy.worker_budget > attn_workers)
        attn_workers = (int)model->target_policy.worker_budget;
    attention_per = (uint64_t)8u * (uint32_t)model->max_context;
    if ((uint64_t)(uint32_t)attn_workers > UINT64_MAX / attention_per /
            sizeof(float) / 2u)
        return -1;
    attention_pool_bytes = (uint64_t)(uint32_t)attn_workers *
        attention_per * sizeof(float) * 2u;
    if (g4_u64_add(&attention_pool_bytes,
            (uint64_t)(uint32_t)attn_workers * sizeof(pthread_t)) != 0 ||
        g4_u64_add(&runtime_bytes, attention_pool_bytes) != 0)
        return -1;
    model->startup_shared_arena_bytes = decode_bytes;
    if (g4_u64_add(&model->startup_shared_arena_bytes,
            model->owned_bf16_bytes) != 0)
        return -1;
    if (shared_arenas) {
        if (g4_u64_add(&model->startup_shared_arena_bytes,
                (uint64_t)operation_bytes) != 0 ||
            g4_u64_add(&model->startup_shared_arena_bytes,
                (uint64_t)scratch_bytes) != 0)
            return -1;
    } else if (g4_u64_add(&runtime_bytes, operation_bytes) != 0 ||
               g4_u64_add(&runtime_bytes, scratch_bytes) != 0) {
        return -1;
    }
    required_outputs = (uint64_t)(uint32_t)prefill_capacity *
        G4_TOPK * G4_HIDDEN;
    if ((uint64_t)(uint32_t)prefill_capacity * G4_DENSE > required_outputs)
        required_outputs = (uint64_t)(uint32_t)prefill_capacity * G4_DENSE;
    if ((uint64_t)(uint32_t)prefill_capacity * G4_MAX_Q > required_outputs)
        required_outputs = (uint64_t)(uint32_t)prefill_capacity * G4_MAX_Q;
    required_jobs = (uint64_t)(uint32_t)prefill_capacity * G4_TOPK * 2u;
    if ((backend.max_output_floats &&
         required_outputs > backend.max_output_floats) ||
        (backend.max_batch_jobs && required_jobs > backend.max_batch_jobs) ||
        (backend.shared_buffer_slots &&
         (uint32_t)(4 + (gpu_attention ? 1 : 0)) >
            backend.shared_buffer_slots) ||
        backend.device_copied_weight_bytes != 0)
        return -1;
    model->startup_source_bytes = source_bytes;
    model->startup_expert_residency_bytes = expert_residency_bytes;
    model->startup_host_runtime_bytes = runtime_bytes;
    model->startup_host_kv_bytes = kv_bytes;
    model->startup_backend_pinned_bytes = backend.fixed_pinned_host_bytes;
    model->startup_backend_device_bytes = backend.fixed_device_bytes;
    model->startup_device_kv_bytes =
        gpu_attention && !compact_hmm_gpu_program && !salt_gpu_cuda_present()
            ? kv_bytes : 0;
    model->startup_cache_metadata_bytes = cache_bytes;
    model->startup_rope_bytes = rope_bytes;
    model->startup_descriptor_bytes = backend.descriptor_bytes;
    if (g4_u64_add(&total, addressability_bytes) != 0 ||
        g4_u64_add(&total, runtime_bytes) != 0 ||
        g4_u64_add(&total, resident_kv_bytes) != 0 ||
        g4_u64_add(&total, model->startup_shared_arena_bytes) != 0 ||
        g4_u64_add(&total, backend.fixed_pinned_host_bytes) != 0 ||
        g4_u64_add(&total, backend.fixed_device_bytes) != 0 ||
        g4_u64_add(&total, model->startup_device_kv_bytes) != 0 ||
        g4_u64_add(&total, cache_bytes) != 0 ||
        g4_u64_add(&total, rope_bytes) != 0)
        return -1;
    model->startup_forecast_bytes = total;
    if (total > model->memory_limit_bytes) {
        fprintf(stderr,
            "GEMMA4_STARTUP_ADMISSION admitted=0 policy=%u forecast_bytes=%llu "
            "limit_bytes=%llu source_bytes=%llu registered_bytes=%llu "
            "pageable_bytes=%llu host_runtime_bytes=%llu host_kv_bytes=%llu "
            "host_kv_resident_forecast_bytes=%llu "
            "shared_arena_bytes=%llu backend_pinned_bytes=%llu "
            "backend_device_bytes=%llu device_kv_bytes=%llu "
            "cache_metadata_bytes=%llu rope_bytes=%llu descriptor_bytes=%llu\n",
            (unsigned int)policy, (unsigned long long)total,
            (unsigned long long)model->memory_limit_bytes,
            (unsigned long long)source_bytes,
            (unsigned long long)model->startup_registered_bytes,
            (unsigned long long)model->startup_pageable_bytes,
            (unsigned long long)runtime_bytes,
            (unsigned long long)kv_bytes,
            (unsigned long long)resident_kv_bytes,
            (unsigned long long)model->startup_shared_arena_bytes,
            (unsigned long long)backend.fixed_pinned_host_bytes,
            (unsigned long long)backend.fixed_device_bytes,
            (unsigned long long)model->startup_device_kv_bytes,
            (unsigned long long)cache_bytes,
            (unsigned long long)rope_bytes,
            (unsigned long long)backend.descriptor_bytes);
        return -1;
    }
    model->startup_admitted = 1;
    fprintf(stderr,
        "GEMMA4_STARTUP_ADMISSION admitted=1 policy=%u forecast_bytes=%llu "
        "limit_bytes=%llu source_bytes=%llu registered_bytes=%llu "
        "pageable_bytes=%llu expert_residency_bytes=%llu "
        "host_runtime_bytes=%llu host_kv_bytes=%llu "
        "host_kv_resident_forecast_bytes=%llu shared_arena_bytes=%llu "
        "backend_pinned_bytes=%llu backend_device_bytes=%llu "
        "device_kv_bytes=%llu cache_metadata_bytes=%llu rope_bytes=%llu "
        "descriptor_bytes=%llu required_output_floats=%llu "
        "required_batch_jobs=%llu\n",
        (unsigned int)policy, (unsigned long long)total,
        (unsigned long long)model->memory_limit_bytes,
        (unsigned long long)source_bytes,
        (unsigned long long)model->startup_registered_bytes,
        (unsigned long long)model->startup_pageable_bytes,
        (unsigned long long)expert_residency_bytes,
        (unsigned long long)runtime_bytes, (unsigned long long)kv_bytes,
        (unsigned long long)resident_kv_bytes,
        (unsigned long long)model->startup_shared_arena_bytes,
        (unsigned long long)backend.fixed_pinned_host_bytes,
        (unsigned long long)backend.fixed_device_bytes,
        (unsigned long long)model->startup_device_kv_bytes,
        (unsigned long long)cache_bytes, (unsigned long long)rope_bytes,
        (unsigned long long)backend.descriptor_bytes,
        (unsigned long long)required_outputs,
        (unsigned long long)required_jobs);
    return 0;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
static int g4_init_device_residency(SaltGemma4Text *model) {
    const SaltGpuResidencyBackendOps *ops;
    SaltGpuResidencyPermanentLayout layout;
    SaltGpuResidencyPlan plan;
    SaltGpuResidencyStats stats;
    uint64_t slots, slot_bytes, slot_base, device_bytes;
    size_t arena_bytes;
    if (!model || !model->gpu_device_residency) return model ? 0 : -1;
    ops = salt_gpu_residency_backend_ops();
    if (!ops || salt_gpu_residency_permanent_layout(
            ops, NULL, &layout, sizeof layout) != 0)
        return -1;
    slot_bytes = G4_SLOT_BYTES;
    slots = model->gpu_device_expert_bytes / slot_bytes;
    if (slots < G4_TOPK || slots > UINT32_MAX ||
        model->gpu_residency_staging_slots == 0 ||
        model->gpu_residency_staging_bytes < slot_bytes ||
        layout.alignment == 0 ||
        layout.permanent_bytes > UINT64_MAX - (layout.alignment - 1u))
        return -1;
    slot_base = (layout.permanent_bytes + layout.alignment - 1u) /
        layout.alignment * layout.alignment;
    if (slots > (UINT64_MAX - slot_base) / slot_bytes)
        return -1;
    device_bytes = slot_base + slots * slot_bytes;
    memset(&plan, 0, sizeof plan);
    plan.device_budget_bytes = device_bytes;
    plan.permanent_bytes = layout.permanent_bytes;
    plan.slot_base_offset = slot_base;
    plan.slot_bytes = slot_bytes;
    plan.staging_bytes = model->gpu_residency_staging_bytes;
    plan.slot_count = (uint32_t)slots;
    plan.staging_count = model->gpu_residency_staging_slots;
    plan.device_alignment = layout.alignment;
    arena_bytes = salt_gpu_residency_arena_requirement(&plan);
    if (arena_bytes == 0 ||
        !(model->gpu_residency_arena = calloc(1u, arena_bytes)))
        return -1;
    model->gpu_residency_arena_bytes = arena_bytes;
    if (salt_gpu_residency_init(
            &model->gpu_residency, &plan, ops, NULL,
            model->gpu_residency_arena, arena_bytes) != 0 ||
        salt_gpu_residency_populate_permanent(&model->gpu_residency) != 0 ||
        salt_gpu_residency_stats(
            &model->gpu_residency, &stats, sizeof stats) != 0) {
        if (model->gpu_residency.ready)
            (void)salt_gpu_residency_destroy(&model->gpu_residency);
        free(model->gpu_residency_arena);
        model->gpu_residency_arena = NULL;
        model->gpu_residency_arena_bytes = 0;
        return -1;
    }
    fprintf(stderr,
        "GEMMA4_GPU_RESIDENCY mode=device-local permanent_arena_bytes=%llu "
        "permanent_payload_bytes=%llu permanent_spans=%llu "
        "permanent_chunks=%llu slot_bytes=%llu expert_slots=%u "
        "expert_capacity_bytes=%llu staging_bytes=%llu staging_slots=%u\n",
        (unsigned long long)layout.permanent_bytes,
        (unsigned long long)stats.permanent_bytes_populated,
        (unsigned long long)stats.permanent_spans,
        (unsigned long long)stats.permanent_chunks,
        (unsigned long long)slot_bytes, plan.slot_count,
        (unsigned long long)(slots * slot_bytes),
        (unsigned long long)(plan.staging_bytes * plan.staging_count),
        plan.staging_count);
    return 0;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int init_full_gpu(SaltGemma4Text *model) {
    const char *timing = getenv("SALT_NVFP4_TIMING");
    int profile = timing && *timing && *timing != '0';
    double start = 0.0, backend_done = 0.0, described_done = 0.0;
    double resources_done = 0.0;
    double tensors_done = 0.0, sync_done = 0.0;
    SaltGpuWeightResourceUsage usage;
    SaltGpuWeightAddressability policy;
    int status;
    if (!model) return -1;
    if (!model->full_gpu_intent) return 0;
    if (!model->startup_admitted) return -1;
    if (profile) start = g4_now_s();
    if (salt_gpu_init() != 0) return -1;
    if (profile) backend_done = g4_now_s();
    model->gpu_initialized = 1;
    salt_gpu_set_mapped_only(1);
    policy = model->gpu_weight_addressability;
    if (policy == SALT_GPU_WEIGHT_ADDRESS_AUTO) {
        if (salt_gpu_pageable_mmap_active())
            policy = SALT_GPU_WEIGHT_ADDRESS_PAGEABLE;
        else if (model->gpu_bounded_weights)
            policy = SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW;
        else
            policy = SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT;
    }
    model->gpu_weight_addressability = policy;
    if (model->gpu_bounded_weights && !model->gpu_expert_layer_view &&
        model->expert_cache_set &&
        (salt_gpu_selected_resources_prepare(model->expert_cache.nslot,
             G4_LAYERS * G4_EXPERTS) != 0 ||
         salt_cache_set_resource_hooks(&model->expert_cache,
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
             model->gpu_device_residency ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
             g4_stable_pageable_pool(model) ? g4_selected_cache_fence : NULL,
             g4_selected_cache_bind, g4_selected_cache_unbind, model) != 0))
        return -1;
    for (int i = 0; i < model->map_count; i++) {
        if (salt_gpu_weight_resource_describe(SALT_GPU_RESOURCE_TRUNK,
                (uint32_t)i, model->maps[i].base,
                model->maps[i].map_len) != 0)
            return -1;
        model->gpu_registered_source_mask |= (uint32_t)1u << i;
    }
    if (!model->nvfp4_mode && !model->gpu_bounded_weights) {
        if (salt_gpu_weight_resource_describe(
                SALT_GPU_RESOURCE_EXPERT_LAYER, 15,
                model->pool.base, model->pool.map_len) != 0)
            return -1;
        model->gpu_pool_registered = 1;
    }
    if (profile) described_done = g4_now_s();
    for (int i = 0; i < model->map_count; i++)
        if (salt_gpu_weight_resource_activate(SALT_GPU_RESOURCE_TRUNK,
                (uint32_t)i, policy) != 0)
            return -1;
    if (model->gpu_trunk_shared_pool) {
        if (policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW)
            return -1;
        for (int i = 0; i < model->map_count; i++)
            if (salt_gpu_weight_resource_pool_bind(SALT_GPU_RESOURCE_TRUNK,
                    (uint32_t)i) != 0)
                return -1;
    }
    if (model->gpu_pool_registered &&
        salt_gpu_weight_resource_activate(SALT_GPU_RESOURCE_EXPERT_LAYER,
            15, policy) != 0)
        return -1;
    /* The registered Spark pool is already activated and every expert is
     * preloaded in the existing zerocopy cache. Bind those READY slots to the
     * same selected-resource ledger before compiling the target program. */
    if (g4_target_gpu_program(model) && !model->gpu_bounded_weights &&
        !model->gpu_expert_layer_view && !model->nvfp4_mode &&
        (policy != SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT ||
         !model->gpu_pool_registered || !model->expert_cache_set ||
         !model->expert_preload_enabled ||
         model->expert_preloaded_slots != G4_LAYERS * G4_EXPERTS ||
         model->expert_cache.nslot != G4_LAYERS * G4_EXPERTS ||
         salt_gpu_selected_resources_prepare(model->expert_cache.nslot,
             G4_LAYERS * G4_EXPERTS) != 0 ||
         salt_cache_set_resource_hooks(&model->expert_cache,
             g4_selected_cache_fence, g4_selected_cache_bind,
             g4_selected_cache_unbind, model) != 0))
        return -1;
    if (profile) resources_done = g4_now_s();
    if (gpu_register_q(model, &model->embedding) != 0) return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        G4Layer *entry = &model->layers[layer];
        G4Q *matrices[] = {
            &entry->q_proj, &entry->k_proj, &entry->v_proj,
            &entry->kv_proj, &entry->o_proj, &entry->dense_gate,
            &entry->dense_up, &entry->dense_down, &entry->router,
        };
        for (size_t i = 0; i < sizeof matrices / sizeof matrices[0]; i++)
            if (gpu_register_q(model, matrices[i]) != 0) return -1;
        if (model->nvfp4_mode) {
            for (int expert = 0; expert < G4_EXPERTS; expert++)
                for (int projection = 0; projection < 3; projection++)
                    if (gpu_register_q(model,
                            &model->nvfp4_experts[
                                ((size_t)layer * G4_EXPERTS + expert) * 3u +
                                (size_t)projection]) != 0)
                        return -1;
            continue;
        }
        if (model->gpu_bounded_weights)
            continue;
        for (int expert = 0; expert < G4_EXPERTS; expert++) {
            size_t expert_index = (size_t)layer * G4_EXPERTS + expert;
            unsigned char *slot = model->pool.base + G4_POOL_HEADER +
                expert_index * G4_SLOT_BYTES;
            unsigned char *projections[3] = {
                slot, slot + G4_PROJ_BYTES, slot + 2u * G4_PROJ_BYTES,
            };
            int rows[3] = {G4_ROUTED, G4_ROUTED, G4_HIDDEN};
            int cols[3] = {G4_HIDDEN, G4_HIDDEN, G4_ROUTED};
            for (int projection = 0; projection < 3; projection++) {
                unsigned char *base = projections[projection];
                if (salt_gpu_weight_resource_slot(
                        SALT_GPU_RESOURCE_EXPERT_LAYER, 15, base,
                        (const uint32_t *)(const void *)base,
                        (const uint16_t *)(const void *)(base + G4_WEIGHT_BYTES),
                        (const uint16_t *)(const void *)(base + G4_BIAS_OFFSET),
                        4, rows[projection], cols[projection]) != 0)
                    return -1;
            }
        }
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (g4_init_device_residency(model) != 0) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (profile) tensors_done = g4_now_s();
    status = salt_gpu_sync();
    if (profile) {
        sync_done = g4_now_s();
        if (salt_gpu_weight_resource_usage(&usage, sizeof usage) != 0)
            memset(&usage, 0, sizeof usage);
        fprintf(stderr,
            "GEMMA4_GPU_INIT backend_s=%.6f resource_describe_s=%.6f "
            "resource_activate_s=%.6f resource_bind_s=%.6f "
            "tensor_register_s=%.6f sync_s=%.6f total_s=%.6f "
            "resources=%d address_policy=%u described_bytes=%llu "
            "registered_bytes=%llu pageable_bytes=%llu "
            "active_window_bytes=%llu peak_window_bytes=%llu "
            "device_copied_weight_bytes=%llu "
            "nvfp4_expert_tensors=%d status=%d\n",
            backend_done - start, described_done - backend_done,
            resources_done - described_done, resources_done - backend_done,
            tensors_done - resources_done, sync_done - tensors_done,
            sync_done - start, model->map_count,
            (unsigned int)policy,
            (unsigned long long)usage.described_bytes,
            (unsigned long long)usage.registered_bytes,
            (unsigned long long)usage.pageable_bytes,
            (unsigned long long)usage.active_window_bytes,
            (unsigned long long)usage.peak_window_bytes,
            (unsigned long long)usage.device_copied_weight_bytes,
            model->nvfp4_mode ? model->nvfp4_expert_bindings : 0, status);
    }
    return status;
}

static int expert_cache_occupancy(const SaltGemma4Text *model) {
    int occupied = 0;
    if (!model || !model->expert_cache_set) return 0;
    for (int slot = 0; slot < model->expert_cache.nslot; slot++)
        if (model->expert_cache.state[slot] != SALT_SLOT_EMPTY) occupied++;
    return occupied;
}

static void update_expert_cache_peak(SaltGemma4Text *model) {
    int occupied = expert_cache_occupancy(model);
    if (occupied > model->expert_cache_peak_slots)
        model->expert_cache_peak_slots = occupied;
}

static int allocate_runtime(SaltGemma4Text *model) {
    size_t kv_floats = 0;
    uint64_t decode_floats = 0;
    float *cursor, *decode_cursor = NULL, *decode_end = NULL;
    int gpu_attention, gpu_decode_attention, gpu_decode_dense_chain;
    int gpu_kv_ring;
    int compact_hmm_gpu_program;
    if (!model || g4_decode_arena_floats(model, &decode_floats) != 0 ||
        decode_floats > SIZE_MAX / sizeof(float))
        return -1;
    if (model->full_gpu_intent) {
        size_t decode_bytes = (size_t)decode_floats * sizeof(float);
        if (!model->gpu_initialized ||
            salt_gpu_shared_buffer_alloc(
                &model->decode_shared, decode_bytes) != 0)
            return -1;
        memset(model->decode_shared.contents, 0, decode_bytes);
        decode_cursor = (float *)model->decode_shared.contents;
        decode_end = decode_cursor + decode_floats;
    }
#define ALLOC(field, count) do { \
    size_t field_count = (size_t)(count); \
    if (model->full_gpu_intent) { \
        if (!decode_cursor || decode_cursor > decode_end || \
            field_count > (size_t)(decode_end - decode_cursor)) return -1; \
        model->field = decode_cursor; \
        decode_cursor += field_count; \
    } else { \
        model->field = (float *)calloc(field_count, sizeof(float)); \
        if (!model->field) return -1; \
    } \
} while (0)
    ALLOC(state, G4_HIDDEN);
    ALLOC(after_attention, G4_HIDDEN);
    ALLOC(layer_output, G4_HIDDEN);
    ALLOC(norm_a, G4_HIDDEN);
    ALLOC(norm_b, G4_HIDDEN);
    ALLOC(branch, G4_HIDDEN);
    ALLOC(q, G4_MAX_Q);
    ALLOC(k, G4_MAX_KV);
    ALLOC(v, G4_MAX_KV);
    ALLOC(attention_output, G4_MAX_Q);
    ALLOC(dense_gate, G4_DENSE);
    ALLOC(dense_up, G4_DENSE);
    ALLOC(dense_output, G4_HIDDEN);
    ALLOC(routed_output, G4_HIDDEN);
    ALLOC(parallel_scratch, G4_HIDDEN * 2);
    ALLOC(router_logits, G4_EXPERTS);
    ALLOC(route_storage, 2u * G4_TOPK);
    model->route_selected = (int *)(void *)model->route_storage;
    model->route_weights = model->route_storage + G4_TOPK;
    ALLOC(expert_outputs, G4_TOPK * G4_HIDDEN);
    ALLOC(expert_gate, G4_TOPK * G4_ROUTED);
    ALLOC(expert_up, G4_TOPK * G4_ROUTED);
    ALLOC(scores, model->max_context);
    ALLOC(final_state, G4_HIDDEN);
    ALLOC(head_logits, G4_VOCAB);
#undef ALLOC
    if (model->full_gpu_intent && decode_cursor != decode_end) return -1;
    if (g4_recipe_int("SALT_PREFILL_GPU_ATTENTION", 0, 1,
                      &gpu_attention) != 0 ||
        g4_recipe_int("SALT_GEMMA_GPU_KV_RING", 0, 1,
                      &gpu_kv_ring) != 0 ||
        g4_recipe_int("SALT_DECODE_GPU_ATTENTION", 0, 1,
                      &gpu_decode_attention) != 0 ||
        g4_recipe_int("SALT_DECODE_GPU_DENSE_CHAIN", 0, 1,
                      &gpu_decode_dense_chain) != 0 ||
        (gpu_decode_attention && !gpu_attention) ||
        (gpu_kv_ring && (!model->full_gpu_intent || !gpu_attention)))
        return -1;
    compact_hmm_gpu_program = g4_compact_hmm_gpu_program(model);
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *l = &model->layers[i];
        const SaltKvLayerPlan *plan = &model->kv_plan[i];
        size_t count, capacity = g4_live_kv_capacity(model, i);
        /* KV geometry belongs to the registered model, not the backend or
         * GPU attention policy. All executors bind these same bounded seats. */
        if (l->kv_dim < 1 || plan->k_width != l->kv_dim ||
            plan->v_width != l->kv_dim ||
            capacity == 0 || capacity > (size_t)model->max_context ||
            capacity > INT_MAX ||
            capacity > SIZE_MAX / (size_t)l->kv_dim)
            return -1;
        l->kv_capacity = (int)capacity;
        l->kv_ring = !l->full_attention &&
            capacity < (size_t)model->max_context;
        count = capacity * (size_t)l->kv_dim;
        if (count > (SIZE_MAX - kv_floats) / 2u) return -1;
        kv_floats += 2u * count;
    }
    if (kv_floats == 0 || kv_floats != model->planned_kv_floats ||
        kv_floats > SIZE_MAX / sizeof(float))
        return -1;
    if (gpu_attention || compact_hmm_gpu_program) {
        size_t kv_bytes = kv_floats * sizeof(float);
        if (!model->full_gpu_intent || !model->gpu_initialized)
            return -1;
        /* Weight addressability never selects live KV backing. Both ordinary
         * and compiled execution consume the initialized shared seat. */
        if (salt_gpu_shared_buffer_alloc(&model->kv_shared, kv_bytes) != 0)
            return -1;
        memset(model->kv_shared.contents, 0, kv_bytes);
        model->kv_arena = (float *)model->kv_shared.contents;
        if (!compact_hmm_gpu_program) {
            if (salt_gpu_attention_prepare(&model->kv_shared) != 0)
                return -1;
            model->gpu_attention_enabled = 1;
            model->gpu_decode_attention_enabled = gpu_decode_attention;
        }
    } else {
        model->kv_arena = (float *)calloc(kv_floats, sizeof(float));
        if (!model->kv_arena) return -1;
    }
    model->gpu_decode_dense_chain_enabled = gpu_decode_dense_chain;
    model->kv_arena_floats = kv_floats;
    model->planned_kv_floats = kv_floats;
    cursor = model->kv_arena;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *l = &model->layers[i];
        size_t count = (size_t)l->kv_capacity * (size_t)l->kv_dim;
        l->key_cache = cursor;
        cursor += count;
        l->value_cache = cursor;
        cursor += count;
        if (gpu_attention && !compact_hmm_gpu_program &&
            salt_gpu_attention_bind_kv(
                (size_t)(l->key_cache - model->kv_arena),
                (size_t)(l->value_cache - model->kv_arena),
                l->kv_dim) != 0)
            return -1;
    }
    if (cursor != model->kv_arena + model->kv_arena_floats) return -1;
    return 0;
}

static int build_rope_table(int max_context, const SaltAttentionDesc *attention,
                            float **cosines_out, float **sines_out,
                            int *pairs_out) {
    float *cosines = NULL, *sines = NULL;
    size_t count;
    int pairs;
    if (!attention || !cosines_out || !sines_out || !pairs_out ||
        max_context < 1 || attention->rope_dim < 2 ||
        attention->rope_dim % 2 != 0 ||
        attention->rope_base_dim < attention->rope_dim ||
        !(attention->rope_theta > 0.0))
        return -1;
    pairs = attention->rope_dim / 2;
    if ((size_t)max_context > SIZE_MAX / (size_t)pairs)
        return -1;
    count = (size_t)max_context * (size_t)pairs;
    if (count > SIZE_MAX / sizeof(float)) return -1;
    cosines = (float *)malloc(count * sizeof(float));
    sines = (float *)malloc(count * sizeof(float));
    if (!cosines || !sines) {
        free(sines);
        free(cosines);
        return -1;
    }
    for (int position = 0; position < max_context; position++)
        for (int pair = 0; pair < pairs; pair++) {
            float exponent = (float)(2 * pair) /
                             (float)attention->rope_base_dim;
            float angle = (float)position /
                          salt_powf((float)attention->rope_theta, exponent);
            size_t index = (size_t)position * (size_t)pairs + (size_t)pair;
            cosines[index] = salt_cosf(angle);
            sines[index] = salt_sinf(angle);
            if (!isfinite(cosines[index]) || !isfinite(sines[index])) {
                free(sines);
                free(cosines);
                return -1;
            }
        }
    *cosines_out = cosines;
    *sines_out = sines;
    *pairs_out = pairs;
    return 0;
}

static int init_rope_tables(SaltGemma4Text *model) {
    const SaltTextGraphDesc *graph;
    if (!model || !model->model_desc ||
        !(graph = model->model_desc->text_graph))
        return -1;
    if (build_rope_table(model->max_context, &graph->sliding,
            &model->sliding_rope_cos, &model->sliding_rope_sin,
            &model->sliding_rope_pairs) != 0)
        return -1;
    if (build_rope_table(model->max_context, &graph->full,
            &model->full_rope_cos, &model->full_rope_sin,
            &model->full_rope_pairs) != 0) {
        free(model->sliding_rope_sin);
        free(model->sliding_rope_cos);
        model->sliding_rope_sin = NULL;
        model->sliding_rope_cos = NULL;
        model->sliding_rope_pairs = 0;
        return -1;
    }
    return 0;
}

static int g4_prefill_ffn_scratch_bytes(int capacity, size_t *bytes_out) {
    struct G4IntAlignment { char byte; int value; };
    size_t float_elements_per_token =
        5u * G4_HIDDEN + 3u * G4_DENSE + G4_EXPERTS + G4_TOPK +
        2u * G4_TOPK * G4_ROUTED + 3u * G4_TOPK * G4_HIDDEN;
    size_t float_elements, int_elements, float_bytes, int_bytes, padding;
    size_t int_alignment = offsetof(struct G4IntAlignment, value);
    if (!bytes_out || capacity < 1 ||
        (size_t)capacity > SIZE_MAX / float_elements_per_token)
        return -1;
    float_elements = (size_t)capacity * float_elements_per_token;
    if (float_elements > SIZE_MAX / sizeof(float) ||
        (size_t)capacity > SIZE_MAX / G4_TOPK)
        return -1;
    int_elements = (size_t)capacity * G4_TOPK;
    if (int_elements > SIZE_MAX / sizeof(int)) return -1;
    float_bytes = float_elements * sizeof(float);
    int_bytes = int_elements * sizeof(int);
    if (int_alignment < 1) return -1;
    padding = (int_alignment - float_bytes % int_alignment) % int_alignment;
    if (float_bytes > SIZE_MAX - padding ||
        float_bytes + padding > SIZE_MAX - int_bytes)
        return -1;
    *bytes_out = float_bytes + padding + int_bytes;
    return 0;
}

static int g4_bind_prefill_ffn_scratch(SaltGemma4Text *model, int capacity) {
    unsigned char *cursor, *end;
    size_t bytes, hidden_count, dense_count, selection_count;
    if (!model || !model->compute_pool.scratch ||
        model->compute_pool.scratch_n < 1 || capacity < 1 ||
        capacity > model->max_context ||
        g4_prefill_ffn_scratch_bytes(capacity, &bytes) != 0 ||
        (size_t)model->compute_pool.scratch_n > SIZE_MAX / sizeof(float) ||
        bytes > (size_t)model->compute_pool.scratch_n * sizeof(float))
        return -1;
    cursor = (unsigned char *)(void *)model->compute_pool.scratch;
    end = cursor + bytes;
    hidden_count = (size_t)capacity * G4_HIDDEN;
    dense_count = (size_t)capacity * G4_DENSE;
    selection_count = (size_t)capacity * G4_TOPK;
#define G4_BIND_FLOAT(field, count) do { \
    size_t bind_bytes = (size_t)(count) * sizeof(float); \
    if (cursor > end || bind_bytes > (size_t)(end - cursor)) return -1; \
    model->field = (float *)(void *)cursor; \
    cursor += bind_bytes; \
} while (0)
    G4_BIND_FLOAT(prefill_dense_inputs, hidden_count);
    G4_BIND_FLOAT(prefill_dense_gate, dense_count);
    G4_BIND_FLOAT(prefill_dense_up, dense_count);
    G4_BIND_FLOAT(prefill_dense_chain, dense_count);
    G4_BIND_FLOAT(prefill_dense_outputs, hidden_count);
    G4_BIND_FLOAT(prefill_router_inputs, hidden_count);
    G4_BIND_FLOAT(prefill_router_logits,
                  (size_t)capacity * G4_EXPERTS);
    G4_BIND_FLOAT(prefill_routed_inputs, hidden_count);
    G4_BIND_FLOAT(prefill_routed_outputs, hidden_count);
    G4_BIND_FLOAT(prefill_weights, selection_count);
    G4_BIND_FLOAT(prefill_group_gate, selection_count * G4_ROUTED);
    G4_BIND_FLOAT(prefill_group_up, selection_count * G4_ROUTED);
    G4_BIND_FLOAT(prefill_group_inputs, selection_count * G4_HIDDEN);
    G4_BIND_FLOAT(prefill_group_outputs, selection_count * G4_HIDDEN);
    G4_BIND_FLOAT(prefill_selection_outputs, selection_count * G4_HIDDEN);
#undef G4_BIND_FLOAT
    {
        struct G4IntAlignment { char byte; int value; };
        size_t int_alignment = offsetof(struct G4IntAlignment, value);
        size_t offset = (size_t)(cursor -
            (unsigned char *)(void *)model->compute_pool.scratch);
        size_t padding = (int_alignment - offset % int_alignment) % int_alignment;
        if (cursor > end || padding > (size_t)(end - cursor)) return -1;
        cursor += padding;
    }
    if (cursor > end || selection_count >
            (size_t)(end - cursor) / sizeof(int))
        return -1;
    model->prefill_selected = (int *)(void *)cursor;
    cursor += selection_count * sizeof(int);
    if (cursor != end) return -1;
    model->prefill_capacity = capacity;
    model->prefill_ffn_arena_bytes = bytes;
    model->prefill_moe_scratch = (SaltMoEGroupScratch) {
        .used = model->prefill_moe_used,
        .acquired = model->prefill_moe_acquired,
        .leases = model->prefill_moe_leases,
        .batch_leases = model->prefill_moe_batch_leases,
        .run_leases = model->prefill_moe_run_leases,
        .expert_ids = model->prefill_moe_expert_ids,
        .run_groups = model->prefill_moe_run_groups,
        .group_inputs = model->prefill_group_inputs,
        .group_outputs = model->prefill_group_outputs,
        .selection_outputs = model->prefill_selection_outputs,
        .expert_capacity = G4_EXPERTS,
        .acquire_capacity = G4_EXPERTS,
        .run_capacity = G4_EXPERTS,
        .group_element_capacity = selection_count * G4_HIDDEN,
        .selection_element_capacity = selection_count * G4_HIDDEN,
    };
    return 0;
}

static int g4_prefill_attention_scratch_bytes(int capacity,
                                               size_t *bytes_out) {
    size_t row_elements, history_capacity, elements;
    if (!bytes_out || capacity < 1 ||
        (size_t)capacity > SIZE_MAX - (G4_SLIDING_WINDOW - 1u))
        return -1;
    history_capacity = (size_t)capacity + G4_SLIDING_WINDOW - 1u;
    if ((size_t)capacity > SIZE_MAX / (2u * G4_HIDDEN + 2u * G4_MAX_Q) ||
        history_capacity > SIZE_MAX / (2u * G4_MAX_KV))
        return -1;
    row_elements = (size_t)capacity *
        (2u * G4_HIDDEN + 2u * G4_MAX_Q);
    elements = history_capacity * (2u * G4_MAX_KV);
    if (row_elements > SIZE_MAX - elements) return -1;
    elements += row_elements;
    if (elements > SIZE_MAX / sizeof(float)) return -1;
    *bytes_out = elements * sizeof(float);
    return 0;
}

static int g4_bind_prefill_attention_scratch(SaltGemma4Text *model,
                                              int capacity) {
    unsigned char *cursor, *end;
    size_t bytes, hidden_count, query_count, kv_count, history_capacity;
    if (!model || !model->compute_pool.scratch || capacity < 1 ||
        capacity > model->max_context ||
        g4_prefill_attention_scratch_bytes(capacity, &bytes) != 0 ||
        (size_t)model->compute_pool.scratch_n > SIZE_MAX / sizeof(float) ||
        bytes > (size_t)model->compute_pool.scratch_n * sizeof(float))
        return -1;
    cursor = (unsigned char *)(void *)model->compute_pool.scratch;
    end = cursor + bytes;
    hidden_count = (size_t)capacity * G4_HIDDEN;
    query_count = (size_t)capacity * G4_MAX_Q;
    history_capacity = (size_t)capacity + G4_SLIDING_WINDOW - 1u;
    kv_count = history_capacity * G4_MAX_KV;
#define G4_BIND_ATTN(field, count) do { \
    size_t bind_bytes = (size_t)(count) * sizeof(float); \
    if (cursor > end || bind_bytes > (size_t)(end - cursor)) return -1; \
    model->field = (float *)(void *)cursor; \
    cursor += bind_bytes; \
} while (0)
    G4_BIND_ATTN(prefill_attention_norm, hidden_count);
    G4_BIND_ATTN(prefill_attention_queries, query_count);
    G4_BIND_ATTN(prefill_attention_keys, kv_count);
    G4_BIND_ATTN(prefill_attention_values, kv_count);
    G4_BIND_ATTN(prefill_attention_outputs, query_count);
    G4_BIND_ATTN(prefill_attention_branches, hidden_count);
#undef G4_BIND_ATTN
    if (cursor != end) return -1;
    model->prefill_attention_capacity = capacity;
    return 0;
}

static int g4_operation_arena_bytes(int capacity, size_t *bytes_out) {
    const size_t float_elements_per_token = 2u * G4_HIDDEN + G4_MAX_Q;
    size_t float_elements, float_bytes, mask_elements, mask_bytes;
    size_t valid_bytes, int_bytes, padding, total;
    struct G4IntAlignment { char byte; int value; };
    size_t int_alignment = offsetof(struct G4IntAlignment, value);
    if (!bytes_out || capacity < 1 || int_alignment < 1 ||
        (size_t)capacity > SIZE_MAX / float_elements_per_token)
        return -1;
    float_elements = (size_t)capacity * float_elements_per_token;
    if (float_elements > SIZE_MAX / sizeof(float) ||
        (size_t)capacity > SIZE_MAX / (size_t)capacity)
        return -1;
    float_bytes = float_elements * sizeof(float);
    mask_elements = (size_t)capacity * (size_t)capacity;
    if (mask_elements > SIZE_MAX / 2u ||
        (size_t)capacity > SIZE_MAX / sizeof(int))
        return -1;
    mask_bytes = 2u * mask_elements;
    valid_bytes = (size_t)capacity;
    int_bytes = (size_t)capacity * sizeof(int);
    if (float_bytes > SIZE_MAX - mask_bytes ||
        float_bytes + mask_bytes > SIZE_MAX - valid_bytes)
        return -1;
    total = float_bytes + mask_bytes + valid_bytes;
    padding = (int_alignment - total % int_alignment) % int_alignment;
    if (total > SIZE_MAX - padding || total + padding > SIZE_MAX - int_bytes)
        return -1;
    *bytes_out = total + padding + int_bytes;
    return 0;
}

static int g4_bind_operation_arena(SaltGemma4Text *model) {
    unsigned char *cursor, *end;
    size_t bytes, state_count, q_count, mask_count;
    struct G4IntAlignment { char byte; int value; };
    size_t int_alignment = offsetof(struct G4IntAlignment, value);
    if (!model || !model->operation_arena || model->operation_capacity < 1 ||
        model->operation_capacity > model->max_context ||
        g4_operation_arena_bytes(model->operation_capacity, &bytes) != 0 ||
        bytes != model->operation_arena_bytes)
        return -1;
    cursor = (unsigned char *)model->operation_arena;
    end = cursor + bytes;
    state_count = (size_t)model->operation_capacity * G4_HIDDEN;
    q_count = (size_t)model->operation_capacity * G4_MAX_Q;
    mask_count = (size_t)model->operation_capacity *
        (size_t)model->operation_capacity;
#define G4_BIND_OPERATION_FLOAT(field, count) do { \
    size_t bind_bytes = (size_t)(count) * sizeof(float); \
    if (cursor > end || bind_bytes > (size_t)(end - cursor)) return -1; \
    model->field = (float *)(void *)cursor; \
    cursor += bind_bytes; \
} while (0)
    G4_BIND_OPERATION_FLOAT(operation_states_a, state_count);
    G4_BIND_OPERATION_FLOAT(operation_states_b, state_count);
    G4_BIND_OPERATION_FLOAT(operation_q_all, q_count);
#undef G4_BIND_OPERATION_FLOAT
#define G4_BIND_OPERATION_BYTES(field, count) do { \
    size_t bind_bytes = (size_t)(count); \
    if (cursor > end || bind_bytes > (size_t)(end - cursor)) return -1; \
    model->field = cursor; \
    cursor += bind_bytes; \
} while (0)
    G4_BIND_OPERATION_BYTES(operation_full_mask, mask_count);
    G4_BIND_OPERATION_BYTES(operation_sliding_mask, mask_count);
    G4_BIND_OPERATION_BYTES(operation_valid,
                            (size_t)model->operation_capacity);
#undef G4_BIND_OPERATION_BYTES
    {
        size_t offset = (size_t)(cursor -
            (unsigned char *)model->operation_arena);
        size_t padding = (int_alignment - offset % int_alignment) % int_alignment;
        if (cursor > end || padding > (size_t)(end - cursor)) return -1;
        cursor += padding;
    }
    if (cursor > end || (size_t)model->operation_capacity >
            (size_t)(end - cursor) / sizeof(int))
        return -1;
    model->operation_block_ids = (int *)(void *)cursor;
    cursor += (size_t)model->operation_capacity * sizeof(int);
    if (cursor != end) return -1;
    model->text_prefill_scratch = (SaltTextPrefillScratch) {
        model->operation_states_a,
        model->operation_states_b,
        state_count,
    };
    return 0;
}

static int init_compute_pool(SaltGemma4Text *model) {
    size_t ffn_bytes, attention_bytes, scratch_bytes, scratch_floats;
    size_t operation_bytes;
    int prefill_capacity, shared_arenas;
    if (!model || model->compute_pool_ready || model->expert_workers < 1 ||
        model->max_context < 1)
        return -1;
    model->compute_pool.nthreads = model->expert_workers;
    model->compute_pool.apool_threads = (int)model->target_policy.worker_budget;
    model->compute_pool.max_tokens = model->max_context;
    if (g4_recipe_int("SALT_PREFILL_B", 1, 4096,
            &prefill_capacity) != 0 ||
        g4_recipe_int("SALT_PREFILL_SHARED_ARENAS", 0, 1,
            &shared_arenas) != 0)
        return -1;
    if (shared_arenas && (!model->full_gpu_intent || !model->gpu_initialized))
        return -1;
    if (prefill_capacity > model->max_context)
        prefill_capacity = model->max_context;
    if (g4_prefill_ffn_scratch_bytes(prefill_capacity, &ffn_bytes) != 0 ||
        g4_prefill_attention_scratch_bytes(
            prefill_capacity, &attention_bytes) != 0 ||
        g4_operation_arena_bytes(prefill_capacity, &operation_bytes) != 0)
        return -1;
    scratch_bytes = ffn_bytes > attention_bytes ? ffn_bytes : attention_bytes;
    if (scratch_bytes > SIZE_MAX - (sizeof(float) - 1u) ||
        operation_bytes < 1)
        return -1;
    if (shared_arenas) {
        if (salt_gpu_shared_buffer_alloc(
                &model->operation_shared, operation_bytes) != 0)
            return -1;
        memset(model->operation_shared.contents, 0, operation_bytes);
        model->operation_arena = model->operation_shared.contents;
    } else {
        model->operation_arena = calloc(1, operation_bytes);
        if (!model->operation_arena) return -1;
    }
    model->operation_arena_bytes = operation_bytes;
    model->operation_capacity = prefill_capacity;
    if (g4_bind_operation_arena(model) != 0) return -1;
    scratch_floats = (scratch_bytes + sizeof(float) - 1u) / sizeof(float);
    if (scratch_floats > LONG_MAX)
        return -1;
    if (shared_arenas) {
        if (scratch_floats > SIZE_MAX / sizeof(float) ||
            salt_gpu_shared_buffer_alloc(&model->compute_scratch_shared,
                scratch_floats * sizeof(float)) != 0)
            return -1;
        memset(model->compute_scratch_shared.contents, 0,
               scratch_floats * sizeof(float));
        if (salt_kv_scratch_bind(&model->compute_pool,
                (float *)model->compute_scratch_shared.contents,
                (long)scratch_floats) != 0)
            return -1;
    } else if (salt_kv_scratch_init(
            &model->compute_pool, (long)scratch_floats) != 0) {
        return -1;
    }
    if (
        g4_bind_prefill_ffn_scratch(model, prefill_capacity) != 0 ||
        g4_bind_prefill_attention_scratch(model, prefill_capacity) != 0)
        return -1;
    if (salt_attn_pool_init(&model->compute_pool) != 0)
        return -1;
    model->compute_pool_ready = 1;
    model->shared_arenas_ready = shared_arenas;
    return 0;
}

SaltGemma4Text *salt_gemma4_text_load(FILE *binding, int max_context,
                                      int expert_workers,
                                      char *error, size_t error_size) {
    char line[640];
    const char *compute_node = getenv("SALT_GEMMA_COMPUTE_NODE");
    const char *platform_recipe = getenv("SALT_GEMMA_PLATFORM_RECIPE");
    SaltTextExecutionClass requested_target_execution;
    int auth_seen = 0, model_seen = 0, end_seen = 0;
    SaltGemma4Text *model = NULL;
    uint16_t endian = 1;
    if (!compute_node ||
        (strcmp(compute_node, "cpu") != 0 &&
         strcmp(compute_node, "mixed") != 0 &&
         strcmp(compute_node, "full") != 0)) {
        set_error(error, error_size, "invalid Gemma compute node");
        return NULL;
    }
    if (!strcmp(compute_node, "mixed")) {
        set_error(error, error_size, "mixed GPU node unavailable");
        return NULL;
    }
    if (!binding || max_context < 1 || max_context > G4_MAX_CONTEXT ||
        expert_workers < 1 || expert_workers > G4_TOPK ||
        *(unsigned char *)&endian != 1) {
        set_error(error, error_size, "invalid Gemma 4 loader arguments");
        return NULL;
    }
    model = (SaltGemma4Text *)calloc(1, sizeof *model);
    if (!model) {
        set_error(error, error_size, "Gemma 4 loader allocation failed");
        return NULL;
    }
    model->max_context = max_context;
    model->expert_workers = expert_workers;
    model->full_gpu_intent = !strcmp(compute_node, "full");
    requested_target_execution = model->full_gpu_intent
        ? SALT_TEXT_EXECUTION_GPU_ONLY : SALT_TEXT_EXECUTION_CPU_ONLY;
    if (salt_text_target_policy_compile_environment(
            &model->target_policy, requested_target_execution,
            SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES) != 0 ||
        model->target_policy.candidate_count >
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES ||
        (model->target_policy.target_rows >
             SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES &&
         (!platform_recipe || strcmp(platform_recipe, "rocm") != 0))) {
        set_error(error, error_size, "invalid engine TARGET policy");
        goto fail;
    }
    model->target_area_active_workers =
        (int)model->target_policy.worker_budget;
    if (g4_recipe_int("SALT_EXPERT_MATRIX_WAVES", 0, 1,
            &model->expert_matrix_waves) != 0) {
        set_error(error, error_size, "invalid expert matrix-wave policy");
        goto fail;
    }
    if (g4_recipe_int("SALT_PREFILL_OPERATION_FLOW", 0, 1,
            &model->prefill_operation_flow) != 0 ||
        g4_recipe_int("SALT_PREFILL_EXPERT_MATRIX_FLOW", 0, 1,
            &model->prefill_expert_matrix_flow) != 0) {
        set_error(error, error_size, "invalid matrix-flow policy");
        goto fail;
    }
    if (g4_recipe_int("SALT_TARGET_GPU_EXPERTS", 0, 1,
            &model->target_gpu_experts) != 0) {
        set_error(error, error_size, "invalid target GPU expert policy");
        goto fail;
    }
    if (model->target_gpu_experts && !model->full_gpu_intent) {
        set_error(error, error_size,
            "target GPU experts require full GPU compute intent");
        goto fail;
    }
    if (g4_target_gpu_program(model) &&
        (!model->full_gpu_intent || model->target_gpu_experts)) {
        set_error(error, error_size,
            "target GPU program requires exclusive full GPU intent");
        goto fail;
    }
    {
        const char *exact_cells = getenv("SALT_GEMMA_METAL_EXACT_CELLS");
        const char *fine_token = getenv("SALT_TEXT_FINE_TOKEN");
        if (exact_cells && strcmp(exact_cells, "0") != 0 &&
            strcmp(exact_cells, "1") != 0) {
            set_error(error, error_size, "invalid exact Metal cell policy");
            goto fail;
        }
        if (fine_token && strcmp(fine_token, "0") != 0 &&
            strcmp(fine_token, "1") != 0) {
            set_error(error, error_size, "invalid fine-token policy");
            goto fail;
        }
        model->metal_exact_cells = exact_cells && !strcmp(exact_cells, "1");
        model->fine_token_enabled = fine_token && !strcmp(fine_token, "1");
        model->decode_gpu_only = model->fine_token_enabled;
        {
            const char *decode_mode = getenv("SALT_GEMMA_DECODE_MODE");
            if (decode_mode && strcmp(decode_mode, "recipe") != 0) {
                if ((strcmp(decode_mode, "mixed") != 0 &&
                     strcmp(decode_mode, "full-gpu") != 0) ||
                    !model->full_gpu_intent || salt_gpu_rocm_present() ||
                    model->metal_exact_cells || !g4_target_gpu_program(model)) {
                    set_error(error, error_size,
                              "Gemma decode mode requires Metal/CUDA mixed or full-gpu");
                    goto fail;
                }
                model->decode_gpu_only = !strcmp(decode_mode, "full-gpu");
            }
        }
        if (model->metal_exact_cells && !model->full_gpu_intent) {
            set_error(error, error_size,
                      "exact Metal cells require full GPU intent");
            goto fail;
        }

    }
    {
        if (g4_recipe_decimal_gb_bytes(
                "SALT_GEMMA_MEMORY_LIMIT_GB", 1u, 128u,
                &model->memory_limit_bytes) != 0) {
            set_error(error, error_size, "invalid Gemma memory limit");
            goto fail;
        }
    }
    if (g4_recipe_int("SALT_GPU_BOUNDED_WEIGHTS", 0, 1,
            &model->gpu_bounded_weights) != 0) {
        set_error(error, error_size, "invalid bounded GPU weight policy");
        goto fail;
    }
    if (g4_recipe_int("SALT_GPU_TRUNK_LAYER_VIEW", 0, 1,
            &model->gpu_trunk_layer_view) != 0) {
        set_error(error, error_size, "invalid GPU trunk layer-view policy");
        goto fail;
    }
    if (g4_recipe_int("SALT_GPU_TRUNK_SHARED_POOL", 0, 1,
            &model->gpu_trunk_shared_pool) != 0 ||
        (model->gpu_trunk_shared_pool && model->gpu_trunk_layer_view)) {
        set_error(error, error_size, "invalid GPU trunk shared-pool policy");
        goto fail;
    }
    if (g4_recipe_int("SALT_GPU_TRUNK_EXACT_VIEWS", 0, 1,
            &model->gpu_trunk_exact_views) != 0) {
        set_error(error, error_size, "invalid GPU trunk schedule policy");
        goto fail;
    }
    {
        const char *policy = getenv("SALT_GPU_WEIGHT_ADDRESSABILITY");
        if (!policy || !strcmp(policy, "auto"))
            model->gpu_weight_addressability = SALT_GPU_WEIGHT_ADDRESS_AUTO;
        else if (!strcmp(policy, "bounded-window"))
            model->gpu_weight_addressability =
                SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW;
        else if (!strcmp(policy, "registered-persistent"))
            model->gpu_weight_addressability =
                SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT;
        else if (!strcmp(policy, "pageable"))
            model->gpu_weight_addressability = SALT_GPU_WEIGHT_ADDRESS_PAGEABLE;
        else {
            set_error(error, error_size,
                      "invalid GPU weight addressability policy");
            goto fail;
        }
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    {
        const char *residency = getenv("SALT_GPU_RESIDENCY");
        int staging_mb = 0, staging_slots = 0;
        if (!residency || !*residency || !strcmp(residency, "disabled"))
            model->gpu_device_residency = 0;
        else if (!strcmp(residency, "device-local"))
            model->gpu_device_residency = 1;
        else {
            set_error(error, error_size, "invalid GPU residency policy");
            goto fail;
        }
        if (model->gpu_device_residency &&
            (g4_recipe_decimal_gb_bytes(
                 "SALT_GPU_RESIDENCY_EXPERT_GB", 1u, 7u,
                 &model->gpu_device_expert_bytes) != 0 ||
             g4_recipe_int("SALT_GPU_RESIDENCY_STAGING_MB", 4, 256,
                 &staging_mb) != 0 ||
             g4_recipe_int("SALT_GPU_RESIDENCY_STAGING_SLOTS", 1, 8,
                 &staging_slots) != 0)) {
            set_error(error, error_size, "invalid GPU residency capacity");
            goto fail;
        }
        if (model->gpu_device_residency) {
            model->gpu_residency_staging_bytes =
                (uint64_t)(uint32_t)staging_mb * UINT64_C(1000000);
            model->gpu_residency_staging_slots = (uint32_t)staging_slots;
            if (!model->full_gpu_intent || !salt_gpu_rocm_present() ||
                !model->gpu_bounded_weights || !model->gpu_trunk_shared_pool ||
                model->gpu_trunk_layer_view ||
                model->gpu_weight_addressability !=
                    SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW) {
                set_error(error, error_size,
                    "device-local residency requires ROCm shared-pool intent");
                goto fail;
            }
        }
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    {
        const char *view = getenv("SALT_GPU_EXPERT_VIEW");
        if (!view || !*view || !strcmp(view, "selected"))
            model->gpu_expert_layer_view = 0;
        else if (!strcmp(view, "layer"))
            model->gpu_expert_layer_view = 1;
        else {
            set_error(error, error_size, "invalid GPU expert view policy");
            goto fail;
        }
    }
    model->model_desc = salt_model_get("gemma4-26b-a4b");
    if (!model->model_desc ||
        salt_state_control_init(&model->state_control,
            salt_model_state(model->model_desc),
            g4_state_materialize, g4_state_finalize, model) != 0 ||
        salt_model_text_plan(model->model_desc, model->layer_plan,
                             G4_LAYERS) != G4_LAYERS ||
        salt_model_kv_plan(model->model_desc, (size_t)max_context,
                           model->kv_plan, G4_LAYERS,
                           &model->planned_kv_floats) != 0 ||
        g4_apply_kv_forecast_policy(model) != 0) {
        set_error(error, error_size, "Gemma 4 registered execution plan failed");
        goto fail;
    }
    model->tokenizer_fd = -1;
    if (!fgets(line, sizeof line, binding) ||
        strcmp(line, "SALT_GEMMA4_TEXT_BINDING_V3\n")) {
        set_error(error, error_size, "invalid Gemma 4 binding header");
        goto fail;
    }
    while (fgets(line, sizeof line, binding)) {
        size_t length = strlen(line);
        if (length == 0 || line[length - 1] != '\n') {
            set_error(error, error_size, "unterminated Gemma 4 binding line");
            goto fail;
        }
        line[length - 1] = 0;
        if (!strncmp(line, "FORMAT NVFP4 ", 13)) {
            char source[65], payload[65], extra;
            if (model->nvfp4_mode ||
                sscanf(line, "FORMAT NVFP4 %64s %64s %c",
                       source, payload, &extra) != 2 ||
                strcmp(source, G4_NV_SOURCE_SHA) ||
                strcmp(payload, G4_NV_PAYLOAD_SHA) ||
                enable_nvfp4(model) != 0) {
                set_error(error, error_size, "NVFP4 source authority drift");
                goto fail;
            }
        } else if (!strncmp(line, "AUTHNV ", 7)) {
            char compat[65], build[65], extra;
            if (auth_seen || !model->nvfp4_mode ||
                sscanf(line, "AUTHNV %64s %64s %c", compat, build, &extra) != 2 ||
                strcmp(compat, G4_KV_COMPAT_SHA) ||
                g4_sha256_from_hex(compat, model->kv_compat_sha256) != 0 ||
                g4_sha256_from_hex(build, model->build_identity_sha256) != 0 ||
                !memcmp(model->build_identity_sha256,
                        (unsigned char[32]){0}, 32)) {
                set_error(error, error_size, "NVFP4 binding authority drift");
                goto fail;
            }
            auth_seen = 1;
        } else if (!strncmp(line, "AUTH ", 5)) {
            char source[65], role[65], map[65], compat[65], build[65], extra;
            if (auth_seen || sscanf(line,
                    "AUTH %64s %64s %64s %64s %64s %c",
                    source, role, map, compat, build, &extra) != 5 ||
                strcmp(source, G4_SOURCE_SHA) || strcmp(role, G4_ROLE_SHA) ||
                strcmp(map, G4_MAP_SHA) || strcmp(compat, G4_KV_COMPAT_SHA) ||
                g4_sha256_from_hex(compat, model->kv_compat_sha256) != 0 ||
                g4_sha256_from_hex(build, model->build_identity_sha256) != 0 ||
                !memcmp(model->build_identity_sha256, (unsigned char[32]){0}, 32)) {
                set_error(error, error_size, "Gemma 4 binding authority drift");
                goto fail;
            }
            auth_seen = 1;
        } else if (!strncmp(line, "MODEL ", 6)) {
            int layers, hidden, vocab, experts, topk, window;
            float eps, cap, embed;
            char extra;
            if (model_seen || sscanf(line, "MODEL %d %d %d %d %d %d %f %f %f %c",
                &layers, &hidden, &vocab, &experts, &topk, &window,
                &eps, &cap, &embed, &extra) != 9 ||
                layers != G4_LAYERS || hidden != G4_HIDDEN || vocab != G4_VOCAB ||
                experts != G4_EXPERTS || topk != G4_TOPK || window != 1024 ||
                eps != G4_EPS || cap != G4_LOGIT_CAP || embed != G4_EMBED_SCALE) {
                set_error(error, error_size, "Gemma 4 binding model drift");
                goto fail;
            }
            model_seen = 1;
        } else if (!strncmp(line, "FD ", 3)) {
            char name[64], extra;
            int fd;
            unsigned long long dev, ino, bytes;
            long long mtime, ctime;
            if (sscanf(line, "FD %63s %d %llu %llu %llu %lld %lld %c",
                       name, &fd, &dev, &ino, &bytes, &mtime, &ctime, &extra) != 7 ||
                add_map(model, name, fd, dev, ino, bytes, mtime, ctime) != 0) {
                set_error(error, error_size, "invalid retained source descriptor");
                goto fail;
            }
        } else if (!strncmp(line, "POOL ", 5)) {
            char extra;
            int fd;
            unsigned long long dev, ino, bytes, slot, layers, experts;
            long long mtime, ctime;
            if (sscanf(line, "POOL %d %llu %llu %llu %lld %lld %llu %llu %llu %c",
                       &fd, &dev, &ino, &bytes, &mtime, &ctime,
                       &slot, &layers, &experts, &extra) != 9 ||
                add_pool(model, fd, dev, ino, bytes, mtime, ctime,
                         slot, layers, experts) != 0) {
                set_error(error, error_size, "invalid retained expert pool descriptor");
                goto fail;
            }
        } else if (!strncmp(line, "Q ", 2)) {
            char role[128], extra;
            int bits, rows, cols, wfd, sfd, bfd;
            unsigned long long woff, wn, soff, sn, boff, bn;
            if (sscanf(line,
                "Q %127s %d %d %d %d %llu %llu %d %llu %llu %d %llu %llu %c",
                role, &bits, &rows, &cols, &wfd, &woff, &wn,
                &sfd, &soff, &sn, &bfd, &boff, &bn, &extra) != 13) {
                set_error(error, error_size, "invalid affine binding record");
                goto fail;
            }
            G4Q *target = q_role(model, role);
            if (!target || load_q(model, bits, rows, cols,
                    wfd, woff, wn, sfd, soff, sn, bfd, boff, bn, target) != 0) {
                set_role_error(error, error_size, "invalid affine role: ", role);
                goto fail;
            }
            model->binding_count++;
        } else if (!strncmp(line, "F ", 2)) {
            char role[128], extra;
            int rows, cols, fd;
            unsigned long long offset, bytes;
            if (!model->nvfp4_mode ||
                sscanf(line, "F %127s %d %d %d %llu %llu %c",
                       role, &rows, &cols, &fd, &offset, &bytes, &extra) != 6) {
                set_error(error, error_size, "invalid BF16 matrix binding");
                goto fail;
            }
            G4Q *target = q_role(model, role);
            if (!target || load_bf16_matrix(
                    model, rows, cols, fd, offset, bytes, target) != 0) {
                set_role_error(error, error_size,
                               "invalid BF16 matrix role: ", role);
                goto fail;
            }
            model->binding_count++;
        } else if (!strncmp(line, "N ", 2)) {
            char role[128], extra;
            int rows, cols, wfd, sfd, wgfd, igfd;
            unsigned long long woff, wn, soff, sn, wgoff, wgn, igoff, ign;
            if (!model->nvfp4_mode || sscanf(line,
                    "N %127s %d %d %d %llu %llu %d %llu %llu %d %llu %llu %d %llu %llu %c",
                    role, &rows, &cols, &wfd, &woff, &wn,
                    &sfd, &soff, &sn, &wgfd, &wgoff, &wgn,
                    &igfd, &igoff, &ign, &extra) != 15) {
                set_error(error, error_size, "invalid NVFP4 matrix binding");
                goto fail;
            }
            G4Q *target = q_role(model, role);
            int expert = 0;
            if (!target) {
                target = nvfp4_expert_role(model, role);
                expert = target != NULL;
            }
            if (!target || load_nvfp4_matrix(model, rows, cols,
                    wfd, woff, wn, sfd, soff, sn, wgfd, wgoff, wgn,
                    igfd, igoff, ign, target) != 0) {
                set_role_error(error, error_size,
                               "invalid NVFP4 matrix role: ", role);
                goto fail;
            }
            if (expert) model->nvfp4_expert_bindings++;
            else model->binding_count++;
        } else if (!strncmp(line, "B ", 2)) {
            char role[128], extra;
            int count, fd;
            unsigned long long offset, bytes;
            if (sscanf(line, "B %127s %d %d %llu %llu %c",
                       role, &count, &fd, &offset, &bytes, &extra) != 5) {
                set_error(error, error_size, "invalid BF16 binding record");
                goto fail;
            }
            G4B *target = b_role(model, role);
            int bind_rc = model->full_gpu_intent
                ? defer_bf16(model, count, fd, offset, bytes, target)
                : load_bf16(model, count, fd, offset, bytes, target);
            if (!target || bind_rc != 0) {
                set_role_error(error, error_size, "invalid BF16 role: ", role);
                goto fail;
            }
            model->binding_count++;
        } else if (!strncmp(line, "TOKENIZER ", 10)) {
            int fd;
            char extra;
            if (model->tokenizer_fd >= 0 ||
                sscanf(line, "TOKENIZER %d %c", &fd, &extra) != 1 ||
                !find_map(model, fd)) {
                set_error(error, error_size, "invalid tokenizer descriptor binding");
                goto fail;
            }
            model->tokenizer_fd = fd;
        } else if (!strcmp(line, "END")) {
            end_seen = 1;
            break;
        } else {
            set_error(error, error_size, "unknown Gemma 4 binding record");
            goto fail;
        }
    }
    if (!auth_seen || !model_seen || !end_seen ||
        model->map_count != (model->nvfp4_mode ? 2 : 4) ||
        (!model->nvfp4_mode && !model->pool_set) || model->tokenizer_fd < 0 ||
        validate_closure(model) != 0 || !all_identities_match(model)) {
        set_error(error, error_size, "incomplete Gemma 4 binding closure");
        goto fail;
    }
    if (model->full_gpu_intent && g4_startup_admit_full_gpu(model) != 0) {
        set_error(error, error_size, "Gemma 4 startup memory admission failed");
        goto fail;
    }
    if (salt_tokenizer_load_fd(&model->tokenizer, model->tokenizer_fd) != 0) {
        set_error(error, error_size, "authenticated Gemma 4 tokenizer load failed");
        goto fail;
    }
    model->tokenizer_set = 1;
    if (!model->nvfp4_mode && init_expert_cache(model) != 0) {
        set_error(error, error_size, "Gemma 4 zero-copy expert cache failed");
        goto fail;
    }
    if (init_full_gpu(model) != 0) {
        set_error(error, error_size, "Gemma 4 full CUDA initialization failed");
        goto fail;
    }
    if (model->full_gpu_intent && materialize_bf16_shared(model) != 0) {
        set_error(error, error_size,
                  "Gemma 4 shared BF16 vector materialization failed");
        goto fail;
    }
    if (allocate_runtime(model) != 0) {
        set_error(error, error_size, "Gemma 4 runtime allocation failed");
        goto fail;
    }
    if (init_rope_tables(model) != 0) {
        set_error(error, error_size, "Gemma 4 deterministic RoPE table failed");
        goto fail;
    }
    if (init_compute_pool(model) != 0) {
        set_error(error, error_size, "Gemma 4 compute pool initialization failed");
        goto fail;
    }
    if (init_text_verify(model) != 0) {
        set_error(error, error_size,
                  "Gemma 4 target-block program initialization failed");
        goto fail;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    fputs("GEMMA4_TEXT_TOKEN_STAGE compile_begin\n", stderr);
    fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (init_text_token_program(model) != 0) {
        set_error(error, error_size,
                  "Gemma 4 token program initialization failed");
        goto fail;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    fputs("GEMMA4_TEXT_TOKEN_STAGE ready\n", stderr);
    fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    return model;

fail:
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    fprintf(stderr, "GEMMA4_LOAD_FAIL error=%s\n",
        error && *error ? error : "unknown");
    fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    salt_gemma4_text_free(model);
    return NULL;
}

void salt_gemma4_text_free(SaltGemma4Text *model) {
    int shared_failed = 0;
    if (!model) return;
    if (model->prefill_decode_lookahead.status ==
            SALT_TEXT_LOOKAHEAD_ELIGIBLE ||
        model->prefill_decode_lookahead.status ==
            SALT_TEXT_LOOKAHEAD_PREPARED ||
        model->prefill_decode_lookahead.status == SALT_TEXT_LOOKAHEAD_READY)
        (void)salt_text_phase_lookahead_cancel(
            &model->prefill_decode_lookahead);
    if (!model->text_cpu_shared.contents)
        free(model->text_cpu_arena);
    model->text_cpu_arena = NULL;
    if (model->compute_scratch_shared.contents &&
        salt_kv_scratch_unbind(&model->compute_pool,
            (float *)model->compute_scratch_shared.contents) != 0)
        shared_failed = 1;
    salt_kv_free(&model->compute_pool);
    model->compute_pool_ready = 0;
    if (model->gpu_initialized) {
        int failed = shared_failed || salt_gpu_sync() != 0;
        if (model->text_gpu.ready &&
            salt_text_verify_heterogeneous_destroy(&model->text_gpu) != 0)
            failed = 1;
        if (!model->text_gpu.ready) {
            free(model->text_gpu_arena);
            model->text_gpu_arena = NULL;
            model->text_gpu_arena_bytes = 0;
        }
        if (model->text_cpu_shared.contents &&
            salt_gpu_shared_buffer_free(&model->text_cpu_shared) != 0)
            failed = 1;
        if (model->expert_cache_set) {
            salt_cache_free(&model->expert_cache);
            model->expert_cache_set = 0;
        }
        if (model->gpu_trunk_window_mask &&
            g4_trunk_layer_window_end(model) != 0)
            failed = 1;
        if (model->compute_scratch_shared.contents &&
            salt_gpu_shared_buffer_free(
                &model->compute_scratch_shared) != 0)
            failed = 1;
        if (model->operation_shared.contents) {
            if (salt_gpu_shared_buffer_free(&model->operation_shared) != 0)
                failed = 1;
            else
                model->operation_arena = NULL;
        }
        if (model->decode_shared.contents) {
            if (salt_gpu_shared_buffer_free(&model->decode_shared) != 0) {
                failed = 1;
            } else {
                model->state = NULL;
                model->after_attention = NULL;
                model->layer_output = NULL;
                model->norm_a = NULL;
                model->norm_b = NULL;
                model->branch = NULL;
                model->q = NULL;
                model->k = NULL;
                model->v = NULL;
                model->attention_output = NULL;
                model->dense_gate = NULL;
                model->dense_up = NULL;
                model->dense_output = NULL;
                model->routed_output = NULL;
                model->parallel_scratch = NULL;
                model->router_logits = NULL;
                model->route_storage = NULL;
                model->route_selected = NULL;
                model->route_weights = NULL;
                model->expert_outputs = NULL;
                model->expert_gate = NULL;
                model->expert_up = NULL;
                model->scores = NULL;
                model->final_state = NULL;
                model->head_logits = NULL;
            }
        }
        if (model->bf16_shared.contents) {
            if (salt_gpu_shared_buffer_free(&model->bf16_shared) != 0) {
                failed = 1;
            } else {
                for (size_t binding = 0;
                     binding < model->deferred_bf16_count; binding++)
                    if (model->deferred_bf16[binding].target)
                        model->deferred_bf16[binding].target->values = NULL;
            }
        }
        if (model->kv_shared.contents) {
            if (model->kv_pageable_mapping) {
                if (salt_gpu_sync() != 0 ||
                        munmap(model->kv_shared.contents,
                               model->kv_shared.nbytes) != 0) {
                    failed = 1;
                } else {
                    memset(&model->kv_shared, 0, sizeof model->kv_shared);
                    model->kv_pageable_mapping = 0;
                    model->kv_arena = NULL;
                }
            } else if (salt_gpu_shared_buffer_free(&model->kv_shared) != 0) {
                failed = 1;
            } else {
                model->kv_arena = NULL;
            }
        }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        if (model->gpu_residency.ready) {
            SaltGpuResidencyStats residency_stats;
            if (salt_gpu_residency_stats(
                    &model->gpu_residency, &residency_stats,
                    sizeof residency_stats) != 0 ||
                salt_gpu_residency_destroy(&model->gpu_residency) != 0) {
                failed = 1;
            } else if (getenv("SALT_GPU_DIAG")) {
                fprintf(stderr,
                    "GEMMA4_GPU_RESIDENCY_DESTROY permanent_bytes=%llu "
                    "permanent_spans=%llu permanent_chunks=%llu "
                    "expert_submissions=%llu expert_completions=%llu "
                    "expert_aborts=%llu expert_bytes=%llu "
                    "expert_retirements=%llu ready_slots=%u "
                    "peak_ready_slots=%u active_leases=%u\n",
                    (unsigned long long)
                        residency_stats.permanent_bytes_populated,
                    (unsigned long long)residency_stats.permanent_spans,
                    (unsigned long long)residency_stats.permanent_chunks,
                    (unsigned long long)residency_stats.population_submissions,
                    (unsigned long long)residency_stats.population_completions,
                    (unsigned long long)residency_stats.population_aborts,
                    (unsigned long long)
                        residency_stats.population_bytes_completed,
                    (unsigned long long)residency_stats.retirements,
                    residency_stats.ready_slots,
                    residency_stats.peak_ready_slots,
                    residency_stats.active_leases);
            }
            if (!model->gpu_residency.ready) {
                free(model->gpu_residency_arena);
                model->gpu_residency_arena = NULL;
                model->gpu_residency_arena_bytes = 0;
            }
        }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        if (model->gpu_pool_registered &&
            salt_gpu_weight_resource_unbind(
                SALT_GPU_RESOURCE_EXPERT_LAYER, 15) != 0)
            failed = 1;
        for (int i = model->map_count - 1; i >= 0; i--)
            if ((model->gpu_registered_source_mask & ((uint32_t)1u << i)) &&
                salt_gpu_weight_resource_unbind(
                    SALT_GPU_RESOURCE_TRUNK, (uint32_t)i) != 0)
                failed = 1;
        if (salt_gpu_free() != 0) failed = 1;
        model->gpu_initialized = 0;
        model->gpu_pool_registered = 0;
        model->gpu_registered_source_mask = 0;
        if (failed) {
            fprintf(stderr, "Gemma 4 full CUDA teardown failed\n");
            fflush(stderr);
            _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
        }
    }
    free(model->text_program_arena);
    model->text_program_arena = NULL;
    if (model->tokenizer_set) salt_tokenizer_free(&model->tokenizer);
    if (model->expert_cache_set) salt_cache_free(&model->expert_cache);
    free(model->cache_pool.ref);
    model->cache_pool.ref = NULL;
    for (int i = 0; i < model->map_count; i++)
        if (model->maps[i].base)
            munmap(model->maps[i].base, model->maps[i].map_len);
    if (model->pool_set && model->pool.base)
        munmap(model->pool.base, model->pool.map_len);
    for (size_t i = 0; i < model->owned_count; i++)
        free(model->owned_bf16[i]);
    free(model->owned_bf16);
    free(model->kv_arena);
    free(model->state);
    free(model->after_attention);
    free(model->layer_output);
    free(model->norm_a);
    free(model->norm_b);
    free(model->branch);
    free(model->q);
    free(model->k);
    free(model->v);
    free(model->attention_output);
    free(model->dense_gate);
    free(model->dense_up);
    free(model->dense_output);
    free(model->routed_output);
    free(model->parallel_scratch);
    free(model->router_logits);
    free(model->route_storage);
    free(model->expert_outputs);
    free(model->expert_gate);
    free(model->expert_up);
    free(model->scores);
    free(model->final_state);
    free(model->head_logits);
    free(model->operation_arena);
    free(model->sliding_rope_cos);
    free(model->sliding_rope_sin);
    free(model->full_rope_cos);
    free(model->full_rope_sin);
    free(model->nvfp4_experts);
    free(model->nvfp4_packed_scratch);
    free(model->nvfp4_scale_scratch);
    free(model->nvfp4_dequant_scratch);
    free(model);
}

int salt_gemma4_text_encode(const SaltGemma4Text *model, const char *text,
                            int *tokens, int token_capacity) {
    if (!model || !model->tokenizer_set) return -1;
    return salt_tokenizer_encode(&model->tokenizer, text, tokens, token_capacity);
}

int salt_gemma4_text_decode(const SaltGemma4Text *model, const int *tokens,
                            int token_count, char *text, int text_capacity) {
    if (!model || !model->tokenizer_set) return -1;
    return salt_tokenizer_decode(&model->tokenizer, tokens, token_count,
                                 text, text_capacity);
}

static int q_matvec(const G4Q *matrix, const float *input, float *output) {
    if (!matrix || !input || !output) return -1;
    if (matrix->bits == 4) {
        salt_q4_matvec(matrix->weight, matrix->scales, matrix->biases,
                       matrix->rows, matrix->cols, input, output);
        return 0;
    }
    if (matrix->bits == 8) {
        salt_q8_matvec(matrix->weight, matrix->scales, matrix->biases,
                       matrix->rows, matrix->cols, input, output);
        return 0;
    }
    if (matrix->bits == 16) {
        salt_bf16_matvec((const uint16_t *)(const void *)matrix->weight,
                         matrix->rows, matrix->cols, input, NULL, output);
        return 0;
    }
    if (matrix->bits == 40) {
        float weight_global, input_global;
        if (!matrix->weight_global_scale || !matrix->input_global_scale ||
            !matrix->nv_packed_scratch || !matrix->nv_scale_scratch ||
            !matrix->nv_dequant_scratch)
            return -1;
        memcpy(&weight_global, matrix->weight_global_scale, sizeof weight_global);
        memcpy(&input_global, matrix->input_global_scale, sizeof input_global);
        return salt_nvfp4_matvec_reference(
            (const uint8_t *)(const void *)matrix->weight,
            (const uint8_t *)(const void *)matrix->scales,
            weight_global, input_global, matrix->rows, matrix->cols,
            input, output, matrix->nv_packed_scratch,
            matrix->nv_scale_scratch, matrix->nv_dequant_scratch);
    }
    return -1;
}

static const G4TextTensorBinding *g4_text_binding(
        const SaltTextTensorDesc *tensor) {
    const G4TextTensorBinding *binding;
    if (!tensor || !tensor->binding || tensor->stable_handle == 0)
        return NULL;
    binding = (const G4TextTensorBinding *)tensor->binding;
    if (!binding->owner || binding < binding->owner->text_bindings ||
        binding >= binding->owner->text_bindings + G4_TEXT_BINDINGS ||
        tensor->stable_handle !=
            (uint64_t)(binding - binding->owner->text_bindings) + 1u)
        return NULL;
    return binding;
}

static size_t g4_text_fixed_scratch(
        const SaltTextTensorDesc *tensor) {
    return g4_text_binding(tensor) ? sizeof(G4TextCpuNode) : SIZE_MAX;
}

static int g4_text_gather_rows(
        const SaltTextTensorDesc *tensor, const int32_t *row_ids,
        uint32_t row_count, float *output, void *scratch,
        size_t scratch_bytes) {
    const G4TextTensorBinding *binding = g4_text_binding(tensor);
    (void)scratch;
    (void)scratch_bytes;
    if (!binding || !row_ids || row_count == 0 || !output)
        return -1;
    if (binding->kind == G4_TEXT_BINDING_B) {
        if (!binding->b || !binding->b->set || tensor->rows != 1u ||
            tensor->cols != (uint32_t)binding->b->count)
            return -1;
        for (uint32_t row = 0; row < row_count; row++) {
            if (row_ids[row] != 0) return -1;
            memcpy(output + (size_t)row * tensor->cols,
                   binding->b->values,
                   (size_t)tensor->cols * sizeof(float));
        }
        return 0;
    }
    if (binding->kind != G4_TEXT_BINDING_Q || !binding->q ||
        !binding->q->set || tensor->rows != (uint32_t)binding->q->rows ||
        tensor->cols != (uint32_t)binding->q->cols)
        return -1;
    for (uint32_t row = 0; row < row_count; row++) {
        const G4Q *q = binding->q;
        size_t base;
        float *destination = output + (size_t)row * tensor->cols;
        if (row_ids[row] < 0 || (uint32_t)row_ids[row] >= tensor->rows)
            return -1;
        base = (size_t)(uint32_t)row_ids[row] * tensor->cols;
        if (q->bits == 16) {
            const unsigned char *source =
                (const unsigned char *)(const void *)q->weight + base * 2u;
            for (uint32_t column = 0; column < tensor->cols; column++) {
                uint16_t encoded;
                memcpy(&encoded, source + (size_t)column * 2u,
                       sizeof encoded);
                destination[column] = bf16_to_float(encoded);
            }
        } else if (q->bits == 4) {
            const uint32_t *weight = (const uint32_t *)(const void *)(
                (const unsigned char *)(const void *)q->weight + base / 2u);
            const uint16_t *scales = (const uint16_t *)(const void *)(
                (const unsigned char *)(const void *)q->scales +
                base / 64u * 2u);
            const uint16_t *biases = (const uint16_t *)(const void *)(
                (const unsigned char *)(const void *)q->biases +
                base / 64u * 2u);
            salt_q4_decode(weight, scales, biases, q->cols, destination);
        } else if (q->bits == 8) {
            const unsigned char *weight =
                (const unsigned char *)(const void *)q->weight;
            for (uint32_t column = 0; column < tensor->cols; column++) {
                size_t index = base + column;
                float scale = bf16_to_float(q->scales[index / 64u]);
                float bias = bf16_to_float(q->biases[index / 64u]);
                destination[column] =
                    (float)weight[index] * scale + bias;
            }
        } else if (q->bits == 40) {
            const uint8_t *weight = (const uint8_t *)(const void *)q->weight;
            const uint8_t *scales = (const uint8_t *)(const void *)q->scales;
            float stored_global;
            if (!q->weight_global_scale) return -1;
            memcpy(&stored_global, q->weight_global_scale,
                   sizeof stored_global);
            if (!(stored_global > 0.0f) || !isfinite(stored_global)) return -1;
            for (uint32_t column = 0; column < tensor->cols; column++) {
                size_t index = base + column;
                uint8_t packed = weight[index / 2u];
                uint8_t code = (index & 1u)
                    ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 15u);
                float block = salt_nvfp4_e4m3fn(scales[index / 16u]);
                destination[column] = salt_nvfp4_e2m1(code) *
                    (block * (1.0f / stored_global));
            }
        } else {
            return -1;
        }
    }
    /* Deliberately raw: the portable executor applies embedding_scale once. */
    return 0;
}

static int g4_text_mlx4_projection_acquire(
        const G4TextTensorBinding *binding,
        const unsigned char **projection_out, int *slot_out) {
    SaltGemma4Text *model;
    const unsigned char *slot_base;
    size_t expert_index, slot_offset, projection_offset;
    int selected, slot = -1;
    if (!binding || binding->kind != G4_TEXT_BINDING_MLX4_EXPERT ||
        !(model = binding->owner) || !projection_out || !slot_out ||
        !model->expert_cache_set || binding->layer < 0 ||
        binding->layer >= G4_LAYERS || binding->expert < 0 ||
        binding->expert >= G4_EXPERTS || binding->projection < 0 ||
        binding->projection > 2)
        return -1;
    expert_index = (size_t)binding->layer * G4_EXPERTS +
        (size_t)binding->expert;
    slot_offset = G4_POOL_HEADER + expert_index * G4_SLOT_BYTES;
    projection_offset = slot_offset +
        (size_t)binding->projection * G4_PROJ_BYTES;
    if (binding->canonical_pool_offset != (uint64_t)projection_offset ||
        !model->cache_pool.ref ||
        model->cache_pool.ref[expert_index].off != (int64_t)slot_offset)
        return -1;
    selected = binding->expert;
    if (salt_cache_getmany(&model->expert_cache, binding->layer,
            &selected, 1, &slot) < 0)
        return -1;
    slot_base = salt_cache_acquire(
        &model->expert_cache, slot, binding->layer, binding->expert);
    if (!slot_base) return -1;
    *projection_out = slot_base +
        (size_t)binding->projection * G4_PROJ_BYTES;
    *slot_out = slot;
    return 0;
}

static int g4_text_mlx4_projection_release(
        const G4TextTensorBinding *binding, int slot) {
    return binding && binding->owner && slot >= 0
        ? salt_cache_release(&binding->owner->expert_cache, slot,
              binding->layer, binding->expert)
        : -1;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#define G4_TEXT_CPU_FFN_LEASED 2
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_text_cpu_node_release(G4TextCpuNode *node, int result) {
    if (!node) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (node->prepared == G4_TEXT_CPU_FFN_LEASED) {
        const G4TextTensorBinding *binding = node->bindings[0];
        SaltGemma4Text *owner = binding ? binding->owner : NULL;
        if (!owner || node->job_count == 0 || node->job_count > G4_EXPERTS)
            return -1;
        if (salt_cache_release_many(&owner->expert_cache, node->slots,
                binding->layer, node->slots + G4_EXPERTS,
                (int)node->job_count) != 0)
            result = -1;
        if (owner->text_cpu.graph_state.node_seat == node) {
            owner->text_cpu.graph_state.node_ops = NULL;
            owner->text_cpu.graph_state.node_seat = NULL;
        }
        node->prepared = 0;
        node->job_count = 0;
        return result;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    for (uint32_t job = node->job_count; job > 0; job--)
        if (node->slots[job - 1u] >= 0 &&
            g4_text_mlx4_projection_release(
                node->bindings[job - 1u], node->slots[job - 1u]) != 0)
            result = -1;
    node->prepared = 0;
    return result;
}

static int g4_text_cpu_node_prepare(
        const SaltTensorHostBatch *jobs, uint32_t job_count,
        uint32_t active_rows, uint32_t worker_limit,
        void *seat, size_t seat_bytes, SaltTensorHostNodePlan *plan) {
    G4TextCpuNode *node = (G4TextCpuNode *)seat;
    SaltGemma4Text *owner = NULL;
    SaltAreaScanRequest request;
    uint64_t output_tiles = 0, candidate_tiles = 0;
    if (!jobs || job_count == 0 ||
        job_count > G4_TEXT_CPU_NODE_MAX_JOBS || active_rows == 0 ||
        worker_limit == 0 || !node || seat_bytes < sizeof(*node) || !plan)
        return -1;
    memset(node, 0, sizeof *node);
    memset(plan, 0, sizeof *plan);
    for (uint32_t job = 0; job < G4_TEXT_CPU_NODE_MAX_JOBS; job++)
        node->slots[job] = -1;
    node->job_count = job_count;
    for (uint32_t job = 0; job < job_count; job++) {
        const SaltTensorHostBatch *source = &jobs[job];
        const G4TextTensorBinding *binding = g4_text_binding(source->tensor);
        const uint32_t *values;
        const uint16_t *scales, *biases;
        uint64_t candidate_job_tiles;
        if (!binding || !source->inputs || !source->outputs ||
            source->row_count == 0 || source->row_count > INT_MAX ||
            source->tensor->rows == 0 || source->tensor->rows > INT_MAX ||
            source->tensor->cols == 0 || source->tensor->cols > INT_MAX ||
            (owner && owner != binding->owner) ||
            output_tiles > UINT64_MAX - source->tensor->rows ||
            source->row_count > UINT64_MAX / source->tensor->rows)
            goto fail;
        candidate_job_tiles =
            (uint64_t)source->row_count * source->tensor->rows;
        if (candidate_tiles > UINT64_MAX - candidate_job_tiles) goto fail;
        owner = binding->owner;
        output_tiles += source->tensor->rows;
        candidate_tiles += candidate_job_tiles;
        node->bindings[job] = binding;
        if (binding->kind == G4_TEXT_BINDING_Q &&
            binding->q && binding->q->set && binding->q->bits == 8) {
            G4TextCpuQ8Job *target;
            if (node->q8_job_count >= G4_TEXT_CPU_NODE_MAX_JOBS)
                goto fail;
            target = &node->q8_jobs[node->q8_job_count++];
            *target = (G4TextCpuQ8Job) {
                binding->q->weight, binding->q->scales, binding->q->biases,
                source->inputs, source->outputs,
                (int)source->tensor->rows, (int)source->tensor->cols,
                (int)source->row_count,
            };
            continue;
        }
        if (binding->kind == G4_TEXT_BINDING_Q && binding->q &&
            binding->q->set && binding->q->bits == 4) {
            values = binding->q->weight;
            scales = binding->q->scales;
            biases = binding->q->biases;
        } else if (binding->kind == G4_TEXT_BINDING_MLX4_EXPERT) {
            if (g4_text_mlx4_projection_acquire(binding,
                    &node->projections[job], &node->slots[job]) != 0)
                goto fail;
            values = (const uint32_t *)(const void *)node->projections[job];
            scales = (const uint16_t *)(const void *)(
                node->projections[job] + G4_WEIGHT_BYTES);
            biases = (const uint16_t *)(const void *)(
                node->projections[job] + G4_BIAS_OFFSET);
        } else {
            goto fail;
        }
        if (node->q4_job_count >= G4_TEXT_CPU_NODE_MAX_JOBS) goto fail;
        node->q4_jobs[node->q4_job_count++] = (SaltBatchJob) {
            values, scales, biases,
            (int)source->tensor->rows, (int)source->tensor->cols,
            (int)source->row_count, source->inputs, source->outputs,
            0, (int)source->tensor->rows,
        };

    }
    if (!owner || owner->full_gpu_intent || !owner->compute_pool_ready ||
        owner->target_area_active_workers < 1 ||
        owner->target_area_active_workers > owner->compute_pool.apool_threads)
        goto fail;
    memset(&request, 0, sizeof request);
    request.requested_rows = active_rows;
    request.admitted_rows = active_rows;
    request.resident_workers = (uint32_t)owner->compute_pool.apool_threads;
    request.active_worker_limit = (uint32_t)owner->target_area_active_workers;
    if (request.active_worker_limit > worker_limit)
        request.active_worker_limit = worker_limit;
    request.narrow_workers = (uint32_t)owner->compute_pool.aq4_threads;
    if (request.narrow_workers > request.active_worker_limit)
        request.narrow_workers = request.active_worker_limit;
    request.wide_allowed = 1u;
    request.output_row_tiles = output_tiles;
    request.candidate_output_tiles = candidate_tiles;
    if (salt_area_scan_plan(&request, &plan->area) != 0) goto fail;
    node->workers = plan->area.active_workers;
    if (node->q4_job_count && node->q8_job_count)
        node->kind = G4_TEXT_CPU_NODE_MIXED;
    else if (node->q4_job_count)
        node->kind = G4_TEXT_CPU_NODE_Q4;
    else if (node->q8_job_count)
        node->kind = G4_TEXT_CPU_NODE_Q8;
    else
        goto fail;
    if (node->workers == 0 || node->workers > 32u) goto fail;
    plan->active_workers = node->workers;
    plan->area.active_workers = node->workers;
    node->prepared = 1;
    return 0;
fail:
    return g4_text_cpu_node_release(node, -1);
}

static int g4_text_cpu_node_worker(void *seat, uint32_t worker) {
    G4TextCpuNode *node = (G4TextCpuNode *)seat;
    if (!node || !node->prepared || worker >= node->workers) return -1;
    /* Independent matrices share one completion boundary, not one concatenated
     * row partition. Every retained worker keeps its disjoint row slice for
     * each matrix in canonical job order, preserving the accepted full-width
     * weight-stationary ownership without changing any row arithmetic. */
    for (uint32_t job = 0; job < node->q4_job_count; job++) {
        SaltBatchJob *q4 = &node->q4_jobs[job];
        int64_t row_start =
            ((int64_t)q4->R * (int64_t)worker) / (int64_t)node->workers;
        int64_t row_end =
            ((int64_t)q4->R * (int64_t)(worker + 1u)) /
            (int64_t)node->workers;
        if (salt_q4_multi_batch_rows(
                q4, 1, row_start, row_end) != 0) {
            node->failed[worker] = 1;
            break;
        }
    }
    for (uint32_t job = 0; job < node->q8_job_count; job++) {
        const G4TextCpuQ8Job *q8 = &node->q8_jobs[job];
        int chunk = q8->rows / (int)node->workers +
                    (q8->rows % (int)node->workers != 0);
        int row_start = (int)worker * chunk;
        int row_end = row_start + chunk;
        if (row_start > q8->rows) row_start = q8->rows;
        if (row_end > q8->rows) row_end = q8->rows;
        if (row_start < row_end && salt_q8_matvec_batch_rows(
                q8->values, q8->scales, q8->biases,
                q8->rows, q8->columns, q8->batch,
                q8->inputs, q8->outputs,
                row_start, row_end) != 0)
            node->failed[worker] = 1;
    }
    return node->failed[worker] ? -1 : 0;
}

static int g4_text_cpu_node_finish(void *seat) {
    G4TextCpuNode *node = (G4TextCpuNode *)seat;
    SaltGemma4Text *owner;
    int result = 0;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (node && node->prepared == G4_TEXT_CPU_FFN_LEASED)
        return g4_text_cpu_node_release(node, 0);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (!node || !node->prepared || node->job_count == 0 ||
        !node->bindings[0] || !(owner = node->bindings[0]->owner))
        return -1;
    for (uint32_t worker = 0; worker < node->workers; worker++)
        if (node->failed[worker]) result = -1;
    if (node->q4_job_count) {
        owner->q4_pool_submissions++;
        owner->q4_multi_pool_submissions++;
        owner->q4_multi_pool_jobs += node->q4_job_count;
    }
    owner->q8_pool_submissions += node->q8_job_count;
    return g4_text_cpu_node_release(node, result);
}

static const SaltTensorHostNodeOps g4_text_cpu_node_ops = {
    g4_text_fixed_scratch,
    g4_text_cpu_node_prepare,
    g4_text_cpu_node_worker,
    g4_text_cpu_node_finish,
};

static int g4_text_cpu_matvec(
        const SaltTextTensorDesc *tensor, const float *input, float *output,
        void *scratch, size_t scratch_bytes) {
    const G4TextTensorBinding *binding = g4_text_binding(tensor);
    const unsigned char *projection = NULL;
    int slot = -1, result = -1;
    (void)scratch;
    (void)scratch_bytes;
    if (!binding || !input || !output) return -1;
    if (binding->kind == G4_TEXT_BINDING_B) {
        float sum = 0.0f;
        if (!binding->b || tensor->rows != 1u ||
            tensor->cols != (uint32_t)binding->b->count)
            return -1;
        for (uint32_t column = 0; column < tensor->cols; column++)
            sum += binding->b->values[column] * input[column];
        output[0] = sum;
        return 0;
    }
    if (binding->kind == G4_TEXT_BINDING_Q ||
        binding->kind == G4_TEXT_BINDING_NVFP4_EXPERT)
        return binding->q ? q_matvec(binding->q, input, output) : -1;
    if (g4_text_mlx4_projection_acquire(
            binding, &projection, &slot) != 0)
        return -1;
    {
        int prior = salt_kernels_in_expert();
        salt_kernels_set_in_expert(1);
        salt_q4_matvec((const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
            (int)tensor->rows, (int)tensor->cols, input, output);
        salt_kernels_set_in_expert(prior);
        result = 0;
    }
    if (g4_text_mlx4_projection_release(binding, slot) != 0)
        result = -1;
    return result;
}

static int g4_text_cpu_matvec_batch(
        const SaltTextTensorDesc *tensor, const float *inputs,
        uint32_t row_count, float *outputs, void *scratch,
        size_t scratch_bytes) {
    const G4TextTensorBinding *binding = g4_text_binding(tensor);
    const unsigned char *projection = NULL;
    int slot = -1, result = -1;
    (void)scratch;
    (void)scratch_bytes;
    if (!binding || !inputs || row_count == 0 || !outputs ||
        row_count > INT_MAX)
        return -1;
    if (binding->kind == G4_TEXT_BINDING_Q ||
        binding->kind == G4_TEXT_BINDING_NVFP4_EXPERT) {
        if (!binding->q) return -1;
        if (binding->q->bits == 8) {
            if (binding->owner->full_gpu_intent)
                return gpu_q8_matvec_batch(binding->owner, binding->q,
                    (int)row_count, inputs, outputs);
            if (g4_q8_pool_batch(binding->owner, binding->q,
                    (int)row_count, inputs, outputs) == 0)
                return 0;
            binding->owner->q8_pool_fallbacks++;
            for (uint32_t row = 0; row < row_count; row++)
                if (q_matvec(binding->q,
                        inputs + (size_t)row * tensor->cols,
                        outputs + (size_t)row * tensor->rows) != 0)
                    return -1;
            return 0;
        }
        return q_matvec_batch(binding->owner, binding->q, (int)row_count,
            inputs, outputs);
    }
    if (binding->kind == G4_TEXT_BINDING_B) {
        for (uint32_t row = 0; row < row_count; row++)
            if (g4_text_cpu_matvec(tensor,
                    inputs + (size_t)row * tensor->cols,
                    outputs + (size_t)row * tensor->rows,
                    scratch, scratch_bytes) != 0)
                return -1;
        return 0;
    }
    if (g4_text_mlx4_projection_acquire(
            binding, &projection, &slot) != 0)
        return -1;
    {
        int prior = salt_kernels_in_expert();
        salt_kernels_set_in_expert(1);
        result = g4_q4_pool_batch(binding->owner,
            (const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
            (int)tensor->rows, (int)tensor->cols, (int)row_count,
            inputs, outputs);
        salt_kernels_set_in_expert(prior);
    }
    if (g4_text_mlx4_projection_release(binding, slot) != 0)
        result = -1;
    return result;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
/* The portable FFN already supplies expert-major gate/up pairs followed by
 * the same expert-major down jobs. Reuse its startup-owned node seat: first
 * half of slots stores acquired slots, second half stores selected expert IDs.
 * Bindings and payloads survive activation; down consumes them without another
 * cache lookup, acquisition, expert-indexed map, or canonical-offset rebuild. */
static int g4_text_cpu_ffn_batches(
        const SaltTextTensorHostBatch *jobs, uint32_t job_count,
        void *scratch, size_t scratch_bytes, int worker_limit) {
    G4TextCpuNode *node = (G4TextCpuNode *)scratch;
    const G4TextTensorBinding *first;
    SaltGemma4Text *owner;
    uint32_t experts;
    int down, result;
    if (!node || scratch_bytes < sizeof *node || !jobs || job_count == 0 ||
        job_count > 2u * G4_EXPERTS ||
        !(first = g4_text_binding(jobs[0].tensor)) ||
        !(owner = first->owner) || owner->full_gpu_intent ||
        !owner->expert_cache_set)
        return -1;
    down = node->prepared == G4_TEXT_CPU_FFN_LEASED;
    if (down) {
        experts = node->job_count;
        if (experts == 0 || experts > G4_EXPERTS || job_count != experts ||
            owner->text_cpu.graph_state.node_ops != &g4_text_cpu_node_ops ||
            owner->text_cpu.graph_state.node_seat != node)
            return g4_text_cpu_node_release(node, -1);
    } else {
        if (node->prepared || job_count % 2u != 0u ||
            owner->text_cpu.graph_state.node_ops ||
            owner->text_cpu.graph_state.node_seat)
            return -1;
        experts = job_count / 2u;
        node->job_count = 0;
        for (uint32_t index = 0; index < experts; index++) {
            uint32_t gate = 2u * index;
            const SaltTextTensorHostBatch *g = &jobs[gate];
            const SaltTextTensorHostBatch *u = &jobs[gate + 1u];
            const G4TextTensorBinding *gb = g4_text_binding(g->tensor);
            const G4TextTensorBinding *ub = g4_text_binding(u->tensor);
            if (!gb || !ub || gb->owner != owner || ub->owner != owner ||
                gb->kind != G4_TEXT_BINDING_MLX4_EXPERT ||
                ub->kind != G4_TEXT_BINDING_MLX4_EXPERT ||
                gb->layer != first->layer || ub->layer != gb->layer ||
                gb->expert < 0 || gb->expert >= G4_EXPERTS ||
                ub->expert != gb->expert || gb->projection != 0 ||
                ub->projection != 1 ||
                (index && gb->expert <= node->slots[G4_EXPERTS + index - 1u]) ||
                !g->inputs || !g->outputs || !u->outputs ||
                g->inputs != u->inputs || g->row_count == 0 ||
                g->row_count > INT_MAX || g->row_count != u->row_count ||
                g->tensor->rows != u->tensor->rows ||
                g->tensor->cols != u->tensor->cols ||
                ub->canonical_pool_offset < gb->canonical_pool_offset ||
                ub->canonical_pool_offset - gb->canonical_pool_offset !=
                    G4_PROJ_BYTES)
                return -1;
            node->bindings[gate] = gb;
            node->bindings[gate + 1u] = ub;
            node->slots[G4_EXPERTS + index] = gb->expert;
        }
        if (salt_cache_getmany(&owner->expert_cache, first->layer,
                node->slots + G4_EXPERTS, (int)experts, node->slots) < 0 ||
            salt_cache_acquire_many(&owner->expert_cache, node->slots,
                first->layer, node->slots + G4_EXPERTS, (int)experts,
                node->projections) != 0)
            return -1;
        node->job_count = experts;
        node->prepared = G4_TEXT_CPU_FFN_LEASED;
        owner->text_cpu.graph_state.node_ops = &g4_text_cpu_node_ops;
        owner->text_cpu.graph_state.node_seat = node;
    }
    for (uint32_t job = 0; job < job_count; job++) {
        const SaltTextTensorHostBatch *source = &jobs[job];
        uint32_t index = down ? job : job / 2u;
        const G4TextTensorBinding *gate = node->bindings[2u * index];
        const G4TextTensorBinding *binding = down
            ? g4_text_binding(source->tensor) : node->bindings[job];
        const unsigned char *projection;
        uint64_t offset;
        if (!binding || !gate || binding->owner != owner ||
            binding->kind != G4_TEXT_BINDING_MLX4_EXPERT ||
            binding->layer != gate->layer || binding->expert != gate->expert ||
            binding->projection != (down ? 2 : (int)(job % 2u)) ||
            binding->canonical_pool_offset < gate->canonical_pool_offset ||
            !node->projections[index] || !source->inputs || !source->outputs ||
            source->row_count == 0 || source->row_count > INT_MAX ||
            source->tensor->rows == 0 || source->tensor->rows > INT_MAX ||
            source->tensor->cols == 0 || source->tensor->cols > INT_MAX ||
            (down && source->row_count !=
                (uint32_t)node->q4_jobs[2u * index].B))
            return g4_text_cpu_node_release(node, -1);
        offset = binding->canonical_pool_offset - gate->canonical_pool_offset;
        if (offset != (uint64_t)(down ? 2u : job % 2u) * G4_PROJ_BYTES)
            return g4_text_cpu_node_release(node, -1);
        projection = node->projections[index] + (size_t)offset;
        node->q4_jobs[job] = (SaltBatchJob) {
            (const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
            (int)source->tensor->rows, (int)source->tensor->cols,
            (int)source->row_count, source->inputs, source->outputs,
            0, (int)source->tensor->rows,
        };
    }
    result = g4_q4_pool_multi_batch_workers(
        owner, node->q4_jobs, (int)job_count, worker_limit);
    if (down || result != 0)
        return g4_text_cpu_node_release(node, result);
    return 0;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_text_cpu_matvec_batches(
        const SaltTextTensorHostBatch *jobs, uint32_t job_count,
        void *scratch, size_t scratch_bytes, int worker_limit) {
    SaltBatchJob native_jobs[256];
    SaltGpuSelectedProjectionJob direct_jobs[256];
    const G4TextTensorBinding *bindings[256];
    const unsigned char *projections[256];
    int slots[256], has_expert = 0, result = -1;
    SaltGemma4Text *owner = NULL;
    (void)scratch;
    (void)scratch_bytes;
    if (!jobs || job_count < 1 || job_count > 256) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    {
        const G4TextTensorBinding *first = g4_text_binding(jobs[0].tensor);
        if (first && first->owner && !first->owner->full_gpu_intent &&
            first->owner->text_cpu.submitted &&
            first->kind == G4_TEXT_BINDING_MLX4_EXPERT)
            return g4_text_cpu_ffn_batches(
                jobs, job_count, scratch, scratch_bytes, worker_limit);
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    memset(native_jobs, 0, sizeof native_jobs);
    memset(direct_jobs, 0, sizeof direct_jobs);
    memset(bindings, 0, sizeof bindings);
    memset(projections, 0, sizeof projections);
    for (uint32_t job = 0; job < job_count; job++) slots[job] = -1;
    for (uint32_t job = 0; job < job_count; job++) {
        const SaltTextTensorHostBatch *source = &jobs[job];
        const G4TextTensorBinding *binding = g4_text_binding(source->tensor);
        const uint32_t *values;
        const uint16_t *scales, *biases;
        if (!binding || !source->inputs || !source->outputs ||
            source->row_count == 0 || source->row_count > INT_MAX ||
            !source->tensor || source->tensor->rows > INT_MAX ||
            source->tensor->cols > INT_MAX ||
            (owner && owner != binding->owner))
            goto done;
        owner = binding->owner;
        bindings[job] = binding;
        if (binding->kind == G4_TEXT_BINDING_Q && binding->q &&
            binding->q->set && binding->q->bits == 4) {
            values = binding->q->weight;
            scales = binding->q->scales;
            biases = binding->q->biases;
        } else if (binding->kind == G4_TEXT_BINDING_MLX4_EXPERT) {
            if (g4_text_mlx4_projection_acquire(
                    binding, &projections[job], &slots[job]) != 0)
                goto done;
            values = (const uint32_t *)(const void *)projections[job];
            scales = (const uint16_t *)(const void *)(
                projections[job] + G4_WEIGHT_BYTES);
            biases = (const uint16_t *)(const void *)(
                projections[job] + G4_BIAS_OFFSET);
            has_expert = 1;
        } else {
            goto done;
        }
        native_jobs[job] = (SaltBatchJob) {
            values, scales, biases,
            (int)source->tensor->rows, (int)source->tensor->cols,
            (int)source->row_count, source->inputs, source->outputs,
            0, (int)source->tensor->rows,
        };
    }
    if (owner->full_gpu_intent && has_expert && owner->target_gpu_experts) {
        int expanded = 0;
        for (uint32_t job = 0; job < job_count; job++) {
            if (bindings[job]->kind != G4_TEXT_BINDING_MLX4_EXPERT ||
                slots[job] < 0 || native_jobs[job].B > INT_MAX - expanded)
                goto done;
            direct_jobs[job] = (SaltGpuSelectedProjectionJob) {
                native_jobs[job].vals, native_jobs[job].scales,
                native_jobs[job].biases, native_jobs[job].vals,
                native_jobs[job].xs, native_jobs[job].ys,
                native_jobs[job].R, native_jobs[job].C,
                native_jobs[job].B, slots[job],
                (uint64_t)(uint32_t)bindings[job]->layer * G4_EXPERTS +
                    (uint32_t)bindings[job]->expert,
            };
            expanded += native_jobs[job].B;
        }
        result = salt_gpu_q4_selected_shared(
            direct_jobs, (int)job_count, &owner->text_cpu_shared);
        if (result == 0) {
            owner->target_gpu_expert_batches++;
            owner->target_gpu_expert_jobs += (uint64_t)(uint32_t)expanded;
        }
    } else if (owner->full_gpu_intent && !has_expert) {
        int expanded = 0;
        result = g4_gpu_multi_batch(
            owner, native_jobs, (int)job_count, &expanded);
        if (result == 0) {
            owner->gpu_dense_submissions += (uint64_t)(uint32_t)expanded;
            if (jobs[0].row_count == 1u)
                owner->gpu_decode_submissions += (uint64_t)(uint32_t)expanded;
        }
    } else {
        result = g4_q4_pool_multi_batch_workers(
            owner, native_jobs, (int)job_count, worker_limit);
    }
done:
    for (uint32_t job = job_count; job > 0; job--)
        if (slots[job - 1u] >= 0 &&
            g4_text_mlx4_projection_release(
                bindings[job - 1u], slots[job - 1u]) != 0)
            result = -1;
    return result;
}

static int g4_text_cpu_matvec_multi_batch(
        const SaltTextTensorHostBatch *jobs, uint32_t job_count,
        void *scratch, size_t scratch_bytes) {
    if (job_count < 2u) return -1;
    return g4_text_cpu_matvec_batches(
        jobs, job_count, scratch, scratch_bytes, 0);
}

static int g4_text_cpu_matvec_area_batch(
        const SaltTextTensorHostBatch *jobs, uint32_t job_count,
        uint32_t active_rows, SaltAreaScanPlan *plan,
        void *scratch, size_t scratch_bytes) {
    SaltGemma4Text *owner = NULL;
    SaltAreaScanRequest request;
    uint64_t output_tiles = 0, candidate_tiles = 0;
    int result;
    if (!jobs || job_count < 1 || job_count > 256 || active_rows < 1 || !plan)
        return -1;
    for (uint32_t job = 0; job < job_count; job++) {
        const G4TextTensorBinding *binding = g4_text_binding(jobs[job].tensor);
        if (!binding || !binding->owner || !jobs[job].tensor ||
            jobs[job].tensor->rows < 1 || jobs[job].row_count < 1 ||
            UINT64_MAX - output_tiles < jobs[job].tensor->rows ||
            jobs[job].tensor->rows >
                UINT64_MAX / jobs[job].row_count ||
            UINT64_MAX - candidate_tiles <
                (uint64_t)jobs[job].tensor->rows * jobs[job].row_count ||
            (owner && owner != binding->owner))
            return -1;
        if (binding->kind == G4_TEXT_BINDING_Q) {
            if (!binding->q || !binding->q->set || binding->q->bits != 4)
                return 1;
        } else if (binding->kind != G4_TEXT_BINDING_MLX4_EXPERT) {
            return 1;
        }
        owner = binding->owner;
        output_tiles += jobs[job].tensor->rows;
        candidate_tiles +=
            (uint64_t)jobs[job].tensor->rows * jobs[job].row_count;
    }
    if (!owner || !owner->compute_pool_ready ||
        owner->target_area_active_workers < 1 ||
        owner->target_area_active_workers > owner->compute_pool.apool_threads)
        return -1;
    memset(&request, 0, sizeof request);
    request.requested_rows = active_rows;
    request.admitted_rows = active_rows;
    request.resident_workers = (uint32_t)owner->compute_pool.apool_threads;
    request.active_worker_limit =
        (uint32_t)owner->target_area_active_workers;
    request.narrow_workers = (uint32_t)owner->compute_pool.aq4_threads;
    request.wide_allowed = 1u;
    request.output_row_tiles = output_tiles;
    request.candidate_output_tiles = candidate_tiles;
    if (salt_area_scan_plan(&request, plan) != 0)
        return -1;
    result = g4_text_cpu_matvec_batches(jobs, job_count,
        scratch, scratch_bytes, (int)plan->active_workers);
    return result;
}

static const SaltTextTensorBackendOps g4_text_cpu_ops = {
    g4_text_fixed_scratch,
    g4_text_gather_rows,
    g4_text_cpu_matvec,
    g4_text_cpu_matvec_batch,
    g4_text_cpu_matvec_multi_batch,
    g4_text_cpu_matvec_area_batch,
    &g4_text_cpu_node_ops,
};


static const SaltTextTensorOps g4_text_q_ops = {
    &g4_text_cpu_ops,
    NULL
};

static const SaltTextTensorOps g4_text_b_ops = {
    &g4_text_cpu_ops,
    NULL
};

/* Tensor storage identity is compiled once from authenticated model bindings.
 * Backends resolve these generic specs; they never inspect G4Q/G4B. */
static int g4_text_resource_table_init(SaltGemma4Text *model) {
    uint32_t count = 0;
    if (!model || model->map_count < 1 || model->map_count > G4_MAX_MAPS ||
        !model->pool.base || model->pool.map_len < G4_POOL_HEADER)
        return -1;
    memset(model->text_resources, 0, sizeof model->text_resources);
    for (int map_id = 0; map_id < model->map_count; map_id++) {
        const G4Map *map = &model->maps[map_id];
        SaltTensorResourceSpec *resource = &model->text_resources[count++];
        if (!map->base || map->map_len < 1) return -1;
        resource->kind = SALT_TENSOR_RESOURCE_TRUNK;
        resource->resource_id = (uint32_t)map_id;
        resource->source_class = SALT_TENSOR_SOURCE_STATIC;
        resource->base = map->base;
        resource->nbytes = map->map_len;
        resource->mapped_nbytes = map->map_len;
    }
    if (model->bf16_shared.contents && model->bf16_shared.nbytes > 0)
        model->text_resources[count++] = (SaltTensorResourceSpec) {
            SALT_TENSOR_RESOURCE_SHARED, 0u, SALT_TENSOR_SOURCE_SHARED, 0u,
            model->bf16_shared.contents, model->bf16_shared.nbytes,
            model->bf16_shared.nbytes,
        };
    model->text_resources[count++] = (SaltTensorResourceSpec) {
        SALT_TENSOR_RESOURCE_EXPERT, 0u, SALT_TENSOR_SOURCE_SELECTED, 0u,
        model->pool.base, model->pool.size, model->pool.map_len,
    };
    if (count > G4_TEXT_RESOURCES) return -1;
    model->text_resource_count = count;
    return 0;
}

static int g4_text_pointer_offset(const SaltTensorResourceSpec *resource,
                                  const void *pointer, uint64_t bytes,
                                  uint64_t *offset_out) {
    uintptr_t base, value;
    if (!resource || !pointer || bytes == 0 || !offset_out ||
        !resource->base || resource->nbytes == 0)
        return -1;
    base = (uintptr_t)resource->base;
    value = (uintptr_t)pointer;
    if (value < base || value - base > resource->nbytes ||
        bytes > resource->mapped_nbytes - (size_t)(value - base))
        return -1;
    *offset_out = (uint64_t)(value - base);
    return 0;
}

static int g4_text_storage_q(SaltGemma4Text *model, const G4Q *q,
                             SaltTensorStorageSpec *storage) {
    const SaltTensorResourceSpec *resource;
    unsigned long long weight = 0, scale = 0;
    uint64_t elements;
    int map_id;
    if (!model || !q || !q->set || !storage ||
        (map_id = gpu_map_for_q(model, q)) < 0 ||
        !(resource = salt_tensor_resource_find(model->text_resources,
            model->text_resource_count, SALT_TENSOR_RESOURCE_TRUNK,
            (uint32_t)map_id)))
        return -1;
    memset(storage, 0, sizeof *storage);
    storage->source_class = SALT_TENSOR_SOURCE_STATIC;
    storage->resource_kind = SALT_TENSOR_RESOURCE_TRUNK;
    storage->resource_id = (uint32_t)map_id;
    elements = (uint64_t)(uint32_t)q->rows * (uint32_t)q->cols;
    if (q->bits == 16) {
        if (elements > UINT64_MAX / 2u) return -1;
        storage->encoding = SALT_TENSOR_ENCODING_BF16;
        storage->value_bytes = elements * 2u;
    } else if (q->bits == 40) {
        storage->encoding = SALT_TENSOR_ENCODING_NVFP4;
        storage->group_size = 16u;
        storage->value_bytes = elements / 2u;
        storage->scale_bytes = elements / 16u;
        storage->auxiliary_bytes = sizeof(float);
        storage->auxiliary_2_bytes = sizeof(float);
    } else {
        if (q_expected_bytes(q->bits, q->rows, q->cols,
                &weight, &scale) != 0)
            return -1;
        storage->encoding = q->bits == 4
            ? SALT_TENSOR_ENCODING_AFFINE_Q4
            : SALT_TENSOR_ENCODING_AFFINE_Q8;
        storage->group_size = 64u;
        storage->value_bytes = weight;
        storage->scale_bytes = scale;
        storage->bias_bytes = scale;
    }
    if (g4_text_pointer_offset(resource, q->weight, storage->value_bytes,
            &storage->value_offset) != 0)
        return -1;
    if (storage->scale_bytes > 0 &&
        g4_text_pointer_offset(resource, q->scales, storage->scale_bytes,
            &storage->scale_offset) != 0)
        return -1;
    if (storage->bias_bytes > 0 &&
        g4_text_pointer_offset(resource, q->biases, storage->bias_bytes,
            &storage->bias_offset) != 0)
        return -1;
    if (storage->encoding == SALT_TENSOR_ENCODING_NVFP4 &&
        (g4_text_pointer_offset(resource, q->weight_global_scale,
             storage->auxiliary_bytes, &storage->auxiliary_offset) != 0 ||
         g4_text_pointer_offset(resource, q->input_global_scale,
             storage->auxiliary_2_bytes,
             &storage->auxiliary_2_offset) != 0))
        return -1;
    return salt_tensor_storage_validate(storage,
        (uint32_t)q->rows, (uint32_t)q->cols);
}

static int g4_text_storage_b(SaltGemma4Text *model, const G4B *b,
                             SaltTensorStorageSpec *storage) {
    const SaltTensorResourceSpec *resource;
    uint64_t value_bytes, value_offset;
    uint32_t resource_id = 0u;
    if (!model || !b || !b->set || b->count < 1 || !b->values || !storage)
        return -1;
    value_bytes = (uint64_t)(uint32_t)b->count * sizeof(float);
    resource = salt_tensor_resource_find(model->text_resources,
        model->text_resource_count, SALT_TENSOR_RESOURCE_SHARED, 0u);
    if (!resource || g4_text_pointer_offset(resource, b->values, value_bytes,
            &value_offset) != 0) {
        SaltTensorResourceSpec *created;
        if (model->text_resource_count >= G4_TEXT_RESOURCES) return -1;
        resource_id = (uint32_t)model->text_binding_count + 1u;
        created = &model->text_resources[model->text_resource_count++];
        *created = (SaltTensorResourceSpec) {
            SALT_TENSOR_RESOURCE_SHARED, resource_id,
            SALT_TENSOR_SOURCE_SHARED, 0u, b->values, (size_t)value_bytes,
            (size_t)value_bytes,
        };
        resource = created;
        value_offset = 0;
    }
    memset(storage, 0, sizeof *storage);
    storage->encoding = SALT_TENSOR_ENCODING_F32;
    storage->source_class = SALT_TENSOR_SOURCE_SHARED;
    storage->resource_kind = SALT_TENSOR_RESOURCE_SHARED;
    storage->resource_id = resource_id;
    storage->flags = SALT_TENSOR_STORAGE_DECODED_F32;
    storage->value_offset = value_offset;
    storage->value_bytes = value_bytes;
    return salt_tensor_storage_validate(storage, 1u, (uint32_t)b->count);
}

static int g4_text_storage_selected(const G4TextTensorBinding *binding,
                                    uint32_t rows, uint32_t cols,
                                    SaltTensorStorageSpec *storage) {
    uint64_t logical;
    if (!binding || binding->kind != G4_TEXT_BINDING_MLX4_EXPERT ||
        binding->layer < 0 || binding->layer >= G4_LAYERS ||
        binding->expert < 0 || binding->expert >= G4_EXPERTS ||
        binding->projection < 0 || binding->projection > 2 || !storage)
        return -1;
    logical = (uint64_t)(uint32_t)binding->layer * G4_EXPERTS +
              (uint32_t)binding->expert;
    memset(storage, 0, sizeof *storage);
    storage->encoding = SALT_TENSOR_ENCODING_AFFINE_Q4;
    storage->source_class = SALT_TENSOR_SOURCE_SELECTED;
    storage->resource_kind = SALT_TENSOR_RESOURCE_EXPERT;
    storage->resource_id = 0u;
    storage->logical_resource_id = logical;
    storage->group_size = 64u;
    storage->value_offset = binding->canonical_pool_offset;
    storage->scale_offset = binding->canonical_pool_offset + G4_WEIGHT_BYTES;
    storage->bias_offset = binding->canonical_pool_offset + G4_BIAS_OFFSET;
    storage->value_bytes = (uint64_t)rows * cols / 2u;
    storage->scale_bytes = (uint64_t)rows * (cols / 64u) * sizeof(uint16_t);
    storage->bias_bytes = storage->scale_bytes;
    return salt_tensor_storage_validate(storage, rows, cols);
}

static int g4_text_bind_tensor(
        SaltGemma4Text *model, G4TextBindingKind kind,
        const G4Q *q, const G4B *b, int layer, int expert,
        int projection, uint64_t canonical_pool_offset,
        uint32_t rows, uint32_t cols, const SaltTextTensorOps *ops,
        SaltTextTensorDesc *out) {
    G4TextTensorBinding *binding;
    SaltTextTensorDesc descriptor;
    int storage_rc = -1;
    if (!model || !out || !ops || rows == 0 || cols == 0 ||
        model->text_binding_count >= G4_TEXT_BINDINGS)
        return -1;
    binding = &model->text_bindings[model->text_binding_count];
    memset(binding, 0, sizeof *binding);
    binding->kind = kind;
    binding->owner = model;
    binding->q = q;
    binding->b = b;
    binding->canonical_pool_offset = canonical_pool_offset;
    binding->layer = layer;
    binding->expert = expert;
    binding->projection = projection;
    memset(&descriptor, 0, sizeof descriptor);
    descriptor.rows = rows;
    descriptor.cols = cols;
    descriptor.stable_handle = (uint64_t)model->text_binding_count + 1u;
    descriptor.binding = binding;
    descriptor.host_ops = ops;
    if (kind == G4_TEXT_BINDING_B) {
        storage_rc = g4_text_storage_b(model, b, &descriptor.storage);
    } else if (kind == G4_TEXT_BINDING_MLX4_EXPERT) {
        storage_rc = g4_text_storage_selected(binding, rows, cols,
            &descriptor.storage);
    } else if (q) {
        storage_rc = g4_text_storage_q(model, q, &descriptor.storage);
    }
    if (storage_rc != 0) {
        if (getenv("SALT_TENSOROPS_DIAG"))
            fprintf(stderr,
                "GEMMA4_TENSOR_SPEC_FAIL binding=%zu kind=%d layer=%d "
                "expert=%d projection=%d rows=%u cols=%u resources=%u\n",
                model->text_binding_count, (int)kind, layer, expert,
                projection, rows, cols, model->text_resource_count);
        return -1;
    }
    model->text_binding_count++;
    *out = descriptor;
    return 0;
}

static int g4_text_bind_q(
        SaltGemma4Text *model, const G4Q *q, SaltTextTensorDesc *out) {
    if (!q || !q->set || q->rows < 1 || q->cols < 1) return -1;
    return g4_text_bind_tensor(model, G4_TEXT_BINDING_Q,
        q, NULL, -1, -1, -1, 0,
        (uint32_t)q->rows, (uint32_t)q->cols, &g4_text_q_ops, out);
}

static int g4_text_bind_b(
        SaltGemma4Text *model, const G4B *b, SaltTextTensorDesc *out) {
    if (!b || !b->set || b->count < 1) return -1;
    return g4_text_bind_tensor(model, G4_TEXT_BINDING_B,
        NULL, b, -1, -1, -1, 0,
        1u, (uint32_t)b->count, &g4_text_b_ops, out);
}

static int g4_text_bind_expert(
        SaltGemma4Text *model, int layer, int expert, int projection,
        uint32_t rows, uint32_t cols, SaltTextTensorDesc *out) {
    if (!model || layer < 0 || layer >= G4_LAYERS || expert < 0 ||
        expert >= G4_EXPERTS || projection < 0 || projection > 2)
        return -1;
    if (model->nvfp4_mode) {
        const G4Q *q = &model->nvfp4_experts[
            ((size_t)layer * G4_EXPERTS + (size_t)expert) * 3u +
            (size_t)projection];
        if (!q_shape(q, 40, (int)rows, (int)cols)) return -1;
        return g4_text_bind_tensor(model,
            G4_TEXT_BINDING_NVFP4_EXPERT, q, NULL,
            layer, expert, projection, 0, rows, cols,
            &g4_text_q_ops, out);
    }
    {
        size_t expert_index = (size_t)layer * G4_EXPERTS + (size_t)expert;
        size_t offset = G4_POOL_HEADER + expert_index * G4_SLOT_BYTES +
            (size_t)projection * G4_PROJ_BYTES;
        if (!model->pool_set || offset >= model->pool.size ||
            G4_PROJ_BYTES > model->pool.size - offset)
            return -1;
        return g4_text_bind_tensor(model,
            G4_TEXT_BINDING_MLX4_EXPERT, NULL, NULL,
            layer, expert, projection, (uint64_t)offset,
            rows, cols, &g4_text_q_ops, out);
    }
}

static int g4_text_shared_prefix_refresh(SaltGemma4Text *model) {
    if (!model) return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        G4Layer *item = &model->layers[layer];
        SaltTextKvSharedPrefixState keys, values;
        if (item->shared_position < 0 ||
            item->shared_position > model->max_context || item->kv_dim < 1 ||
            item->shared_row_base < 0 ||
            item->shared_row_base > item->shared_position ||
            (item->shared_position == 0 &&
             (item->shared_row_base != 0 || item->shared_key_cache ||
              item->shared_value_cache || item->shared_key_private)) ||
            (item->shared_position > 0 &&
             (!item->shared_key_cache || !item->shared_value_cache)) ||
            (item->shared_key_private &&
             (item->shared_key_cache != item->key_cache ||
              item->shared_row_base != 0 || !item->plan ||
              !item->plan->attention.shared_kv_projection)))
            return -1;
        memset(&keys, 0, sizeof keys);
        memset(&values, 0, sizeof values);
        if (item->shared_position > 0 && item->shared_row_base == 0) {
            size_t capacity = (size_t)item->shared_position *
                (size_t)item->kv_dim;
            keys.rows = item->shared_key_cache;
            keys.float_capacity = capacity;
            keys.row_stride = (size_t)item->kv_dim;
            keys.row_count = (uint32_t)item->shared_position;
            values.rows = item->shared_value_cache;
            values.float_capacity = capacity;
            values.row_stride = (size_t)item->kv_dim;
            values.row_count = (uint32_t)item->shared_position;
        }
        model->text_shared_keys[layer] = keys;
        model->text_shared_values[layer] = values;
    }
    return 0;
}

static int g4_text_state_aligned(const SaltGemma4Text *model) {
    return model && model->position >= 0 &&
        (!model->text_program.ready ||
         model->text_kv_state.position == (uint32_t)model->position);
}

static int g4_gpu_kv_rewind(SaltGemma4Text *model, int position) {
    if (!model || position < 0) return -1;
    return model->full_gpu_intent && model->gpu_attention_enabled
        ? salt_gpu_attention_rewind(position) : 0;
}

static int g4_gpu_kv_materialize(
        SaltGemma4Text *model, int position, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!model || !bytes || position < 0) return -1;
    return model->full_gpu_intent && model->gpu_attention_enabled
        ? salt_gpu_attention_materialize(&model->kv_shared, position, bytes) : 0;
}

static int g4_gpu_kv_import(
        SaltGemma4Text *model, int position, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!model || !bytes || position < 0) return -1;
    return model->full_gpu_intent && model->gpu_attention_enabled
        ? salt_gpu_attention_import(&model->kv_shared, position, bytes) : 0;
}

static int g4_text_state_publish(SaltGemma4Text *model, int position) {
    if (!model || position < 0 || position > model->max_context ||
        (model->text_program.ready &&
         model->text_kv_state.transition_generation == UINT64_MAX))
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    memset(&model->target_projection, 0, sizeof model->target_projection);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->position = position;
    model->text_kv_state.position = (uint32_t)position;
    if (model->text_program.ready)
        model->text_kv_state.transition_generation++;
    /* The retained ordinary executor consumes the same committed generation
     * after PREFILL/import as the compiled executor. Never start a second
     * token-generation sequence when changing physical phase callers. */
    model->token_transition_generation =
        model->text_kv_state.transition_generation;
    return 0;
}

static int g4_text_state_sync_from_kv(SaltGemma4Text *model) {
    if (!model || model->text_kv_state.position > (uint32_t)model->max_context ||
        model->text_kv_state.position > (uint32_t)INT_MAX)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    memset(&model->target_projection, 0, sizeof model->target_projection);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->position = (int)model->text_kv_state.position;
    model->token_transition_generation =
        model->text_kv_state.transition_generation;
    return g4_text_state_aligned(model) ? 0 : -1;
}


static int g4_text_verify_parallel_run(
        void *opaque, int active_workers,
        void (*worker)(int worker, void *task), void *task) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !model->compute_pool_ready || !worker || !task ||
        active_workers < 1 ||
        active_workers > model->target_area_active_workers ||
        active_workers > model->compute_pool.apool_threads)
        return -1;
    return salt_attn_pool_run_n(
        &model->compute_pool, active_workers, worker, task);
}

static int init_text_verify(SaltGemma4Text *model) {
    const SaltTextGpuProgramOps *gpu_ops = NULL;
    SaltTextDispatchPolicy gpu_policy;
    uint32_t maximum_candidates;
    size_t program_bytes = 0, executor_bytes = 0;
    uint64_t verify_bytes = 0, forecast_bytes = 0, canonical_bytes = 0;
    void *program_arena = NULL, *executor_arena = NULL;
    const char *target_waterfall = getenv("SALT_TARGET_WATERFALL");
    int target_profile = 0;
    if (target_waterfall) {
        if (strcmp(target_waterfall, "0") != 0 &&
            strcmp(target_waterfall, "1") != 0)
            return -1;
        target_profile = strcmp(target_waterfall, "1") == 0;
    }

    if (!model || !model->compute_pool_ready || !model->model_desc ||
        model->position != 0 || model->text_program.ready ||
        model->text_program_arena || model->text_cpu_arena ||
        model->text_cpu_shared.contents || model->text_gpu_arena ||
        model->text_gpu.ready)
        return -1;
    memset(&gpu_policy, 0, sizeof gpu_policy);
    memset(model->text_layers, 0, sizeof model->text_layers);
    memset(model->text_experts, 0, sizeof model->text_experts);
    memset(model->text_kv, 0, sizeof model->text_kv);
    memset(model->text_shared_keys, 0, sizeof model->text_shared_keys);
    memset(model->text_shared_values, 0, sizeof model->text_shared_values);
    memset(model->text_bindings, 0, sizeof model->text_bindings);
    memset(model->text_resources, 0, sizeof model->text_resources);
    memset(&model->text_descriptor, 0, sizeof model->text_descriptor);
    memset(&model->text_kv_state, 0, sizeof model->text_kv_state);
    memset(&model->text_program, 0, sizeof model->text_program);
    memset(&model->text_cpu, 0, sizeof model->text_cpu);
    memset(&model->text_executor, 0, sizeof model->text_executor);
    model->text_binding_count = 0;
    model->text_resource_count = 0;
    if (g4_text_resource_table_init(model) != 0 ||
        g4_text_shared_prefix_refresh(model) != 0 ||
        g4_text_bind_q(model, &model->embedding,
                       &model->text_descriptor.embedding) != 0 ||
        g4_text_bind_b(model, &model->final_norm,
                       &model->text_descriptor.final_norm) != 0)
        return -1;
    model->text_descriptor.output_head = model->text_descriptor.embedding;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        G4Layer *source = &model->layers[layer];
        SaltTextLayerExecDesc *destination = &model->text_layers[layer];
        size_t kv_capacity;
        destination->plan = &model->layer_plan[layer];
        destination->kv = &model->text_kv[layer];
        if (g4_text_bind_q(model, &source->q_proj, &destination->q) != 0 ||
            g4_text_bind_q(model,
                source->full_attention ? &source->kv_proj : &source->k_proj,
                &destination->k) != 0 ||
            (!source->full_attention &&
             g4_text_bind_q(model, &source->v_proj, &destination->v) != 0) ||
            g4_text_bind_q(model, &source->o_proj, &destination->o) != 0 ||
            g4_text_bind_q(model, &source->dense_gate,
                           &destination->dense_gate) != 0 ||
            g4_text_bind_q(model, &source->dense_up,
                           &destination->dense_up) != 0 ||
            g4_text_bind_q(model, &source->dense_down,
                           &destination->dense_down) != 0 ||
            g4_text_bind_q(model, &source->router,
                           &destination->router) != 0 ||
            g4_text_bind_b(model, &source->input_norm,
                           &destination->pre_attention_norm) != 0 ||
            g4_text_bind_b(model, &source->q_norm,
                           &destination->q_norm) != 0 ||
            g4_text_bind_b(model, &source->k_norm,
                           &destination->k_norm) != 0 ||
            g4_text_bind_b(model, &source->post_attention_norm,
                           &destination->post_attention_norm) != 0 ||
            g4_text_bind_b(model, &source->pre_ffn_norm,
                           &destination->pre_ffn_norm_1) != 0 ||
            g4_text_bind_b(model, &source->pre_ffn_norm_2,
                           &destination->pre_ffn_norm_2) != 0 ||
            g4_text_bind_b(model, &source->post_ffn_norm_1,
                           &destination->post_ffn_norm_1) != 0 ||
            g4_text_bind_b(model, &source->post_ffn_norm_2,
                           &destination->post_ffn_norm_2) != 0 ||
            g4_text_bind_b(model, &source->post_ffn_norm,
                           &destination->post_ffn_norm) != 0 ||
            g4_text_bind_b(model, &source->router_scale,
                           &destination->router_scale) != 0 ||
            g4_text_bind_b(model, &source->per_expert_scale,
                           &destination->per_expert_scale) != 0 ||
            g4_text_bind_b(model, &source->layer_scalar,
                           &destination->layer_scalar) != 0)
            return -1;
        destination->experts = model->text_experts[layer];
        destination->expert_count = G4_EXPERTS;
        for (int expert = 0; expert < G4_EXPERTS; expert++) {
            SaltTextExpertDesc *entry = &model->text_experts[layer][expert];
            entry->expert_id = (uint32_t)expert;
            if (g4_text_bind_expert(model, layer, expert, 0,
                    G4_ROUTED, G4_HIDDEN, &entry->gate) != 0 ||
                g4_text_bind_expert(model, layer, expert, 1,
                    G4_ROUTED, G4_HIDDEN, &entry->up) != 0 ||
                g4_text_bind_expert(model, layer, expert, 2,
                    G4_HIDDEN, G4_ROUTED, &entry->down) != 0)
                return -1;
        }
        kv_capacity = (size_t)source->kv_capacity *
            (size_t)source->kv_dim;
        model->text_kv[layer].keys.private_mode =
            source->kv_ring ? SALT_TEXT_KV_PRIVATE_RING :
                SALT_TEXT_KV_PRIVATE_ABSOLUTE;
        model->text_kv[layer].keys.private_rows = source->key_cache;
        model->text_kv[layer].keys.private_float_capacity = kv_capacity;
        model->text_kv[layer].keys.private_row_stride = (size_t)source->kv_dim;
        model->text_kv[layer].keys.private_row_capacity =
            (uint32_t)source->kv_capacity;
        model->text_kv[layer].keys.private_position_base = 0;
        model->text_kv[layer].keys.shared_prefix_state =
            &model->text_shared_keys[layer];
        model->text_kv[layer].values.private_mode =
            source->kv_ring ? SALT_TEXT_KV_PRIVATE_RING :
                SALT_TEXT_KV_PRIVATE_ABSOLUTE;
        model->text_kv[layer].values.private_rows = source->value_cache;
        model->text_kv[layer].values.private_float_capacity = kv_capacity;
        model->text_kv[layer].values.private_row_stride =
            (size_t)source->kv_dim;
        model->text_kv[layer].values.private_row_capacity =
            (uint32_t)source->kv_capacity;
        model->text_kv[layer].values.private_position_base = 0;
        model->text_kv[layer].values.shared_prefix_state =
            &model->text_shared_values[layer];
    }
    if (model->text_binding_count != G4_TEXT_BINDINGS)
        return -1;
    model->text_descriptor.model = model->model_desc;
    model->text_descriptor.layers = model->text_layers;
    model->text_descriptor.layer_count = G4_LAYERS;
    model->text_descriptor.tensor_resources = model->text_resources;
    model->text_descriptor.tensor_resource_count = model->text_resource_count;
    model->text_descriptor.vocabulary = G4_VOCAB;
    model->text_descriptor.maximum_context = (uint32_t)model->max_context;
    model->text_descriptor.norm_epsilon = G4_EPS;
    model->text_descriptor.embedding_scale = G4_EMBED_SCALE;
    model->text_kv_state.position = (uint32_t)model->position;
    maximum_candidates = model->target_policy.target_rows;
    if (maximum_candidates < SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES)
        maximum_candidates = SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES;
    if ((g4_compact_hmm_gpu_program(model) ||
         g4_registered_cuda_gpu_program(model)) &&
        model->prefill_capacity > 0 &&
        (uint32_t)model->prefill_capacity > maximum_candidates)
        maximum_candidates = (uint32_t)model->prefill_capacity;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* Metal PREFILL uses the model-owned phase scratch and KV ring path,
     * not this generic TARGET executor. Keep its established TARGET capacity
     * independent of PREFILL B; the compact GPU PREFILL path above still
     * sizes its executor for the phase that actually consumes it. */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (maximum_candidates > (uint32_t)model->max_context)
        maximum_candidates = (uint32_t)model->max_context;
    if (salt_text_verify_program_arena_requirement(
            &model->text_descriptor, maximum_candidates,
            &program_bytes) != 0 || program_bytes == 0 ||
        !(program_arena = malloc(program_bytes)) ||
        salt_text_verify_program_compile(
            &model->text_program, &model->text_descriptor,
            &model->text_kv_state, maximum_candidates,
            program_arena, program_bytes) != 0 ||
        salt_text_verify_program_bind_target_policy(
            &model->text_program, &model->target_policy) != 0) {
        free(program_arena);
        memset(&model->text_program, 0, sizeof model->text_program);
        return -1;
    }
    if (g4_target_gpu_program(model)) {
        gpu_ops = salt_gpu_tensor_program_ops();
        gpu_policy.execution_class = model->target_policy.execution_class;
        if (!gpu_ops || salt_text_verify_heterogeneous_arena_requirement(
                &model->text_program, &gpu_policy, gpu_ops,
                &executor_bytes) != 0 || executor_bytes == 0) {
            free(program_arena);
            memset(&model->text_program, 0, sizeof model->text_program);
            return -1;
        }
        canonical_bytes = (uint64_t)model->text_program.layout.total_bytes;
    } else if (salt_text_verify_cpu_arena_requirement(
            &model->text_program, &executor_bytes) != 0 ||
            executor_bytes == 0) {
        free(program_arena);
        memset(&model->text_program, 0, sizeof model->text_program);
        return -1;
    }
    if (g4_u64_add(&verify_bytes, (uint64_t)program_bytes) != 0 ||
        g4_u64_add(&verify_bytes, (uint64_t)executor_bytes) != 0 ||
        g4_u64_add(&verify_bytes, canonical_bytes) != 0) {
        free(program_arena);
        memset(&model->text_program, 0, sizeof model->text_program);
        return -1;
    }
    forecast_bytes = model->startup_forecast_bytes;
    if (g4_u64_add(&forecast_bytes, verify_bytes) != 0 ||
        model->memory_limit_bytes == 0 ||
        verify_bytes > model->memory_limit_bytes ||
        (model->full_gpu_intent &&
         forecast_bytes > model->memory_limit_bytes)) {
        fprintf(stderr,
            "GEMMA4_TEXT_VERIFY_ADMISSION admitted=0 base_complete=%d "
            "base_forecast_bytes=%llu program_arena_bytes=%llu "
            "executor_arena_bytes=%llu verify_bytes=%llu forecast_bytes=%llu "
            "canonical_bytes=%llu limit_bytes=%llu gpu_program=%d\n",
            model->full_gpu_intent ? 1 : 0,
            (unsigned long long)model->startup_forecast_bytes,
            (unsigned long long)program_bytes,
            (unsigned long long)executor_bytes,
            (unsigned long long)verify_bytes,
            (unsigned long long)forecast_bytes,
            (unsigned long long)canonical_bytes,
            (unsigned long long)model->memory_limit_bytes,
            g4_target_gpu_program(model));
        free(program_arena);
        memset(&model->text_program, 0, sizeof model->text_program);
        return -1;
    }
    model->startup_text_program_bytes = (uint64_t)program_bytes;
    model->startup_text_cpu_bytes = g4_target_gpu_program(model)
        ? 0u : (uint64_t)executor_bytes;
    model->startup_text_verify_bytes = verify_bytes;
    if (model->full_gpu_intent) {
        model->startup_forecast_bytes = forecast_bytes;
        if (g4_u64_add(&model->startup_host_runtime_bytes,
                (uint64_t)program_bytes) != 0 ||
            g4_u64_add(&model->startup_host_runtime_bytes,
                (uint64_t)executor_bytes) != 0 ||
            g4_u64_add(&model->startup_shared_arena_bytes,
                canonical_bytes) != 0) {
            free(program_arena);
            memset(&model->text_program, 0, sizeof model->text_program);
            return -1;
        }
    }
    fprintf(stderr,
        "GEMMA4_TEXT_VERIFY_ADMISSION admitted=1 base_complete=%d "
        "base_forecast_bytes=%llu program_arena_bytes=%llu "
        "executor_arena_bytes=%llu verify_bytes=%llu forecast_bytes=%llu "
        "canonical_bytes=%llu limit_bytes=%llu gpu_program=%d\n",
        model->full_gpu_intent ? 1 : 0,
        (unsigned long long)(forecast_bytes - verify_bytes),
        (unsigned long long)program_bytes,
        (unsigned long long)executor_bytes,
        (unsigned long long)verify_bytes,
        (unsigned long long)forecast_bytes,
        (unsigned long long)canonical_bytes,
        (unsigned long long)model->memory_limit_bytes,
        g4_target_gpu_program(model));
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    fputs("GEMMA4_TEXT_VERIFY_STAGE executor_alloc_begin\n", stderr);
    fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    executor_arena = malloc(executor_bytes);
    if (!executor_arena) {
        free(program_arena);
        memset(&model->text_program, 0, sizeof model->text_program);
        return -1;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    fputs("GEMMA4_TEXT_VERIFY_STAGE executor_alloc_done\n", stderr);
    fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (g4_target_gpu_program(model)) {
        if (salt_text_verify_heterogeneous_compile(
                &model->text_gpu, &model->text_program, &gpu_policy, gpu_ops,
                executor_arena, executor_bytes) != 0 ||
            salt_text_verify_heterogeneous_resource_bind(
                &model->text_gpu, &g4_text_gpu_resource_ops, model) != 0 ||
            salt_text_verify_heterogeneous_executor_init(
                &model->text_executor, &model->text_gpu) != 0) {
            if (model->text_gpu.ready)
                (void)salt_text_verify_heterogeneous_destroy(&model->text_gpu);
            free(executor_arena);
            free(program_arena);
            memset(&model->text_program, 0, sizeof model->text_program);
            memset(&model->text_gpu, 0, sizeof model->text_gpu);
            memset(&model->text_executor, 0, sizeof model->text_executor);
            return -1;
        }
    } else {
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        fputs("GEMMA4_TEXT_VERIFY_STAGE cpu_compile_begin\n", stderr);
        fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        if (model->target_gpu_experts) {
            if (salt_gpu_shared_buffer_alloc(
                    &model->text_cpu_shared, executor_bytes) != 0) {
                free(executor_arena);
                free(program_arena);
                memset(&model->text_program, 0, sizeof model->text_program);
                return -1;
            }
            free(executor_arena);
            executor_arena = model->text_cpu_shared.contents;
        }
        if (salt_text_verify_cpu_compile(
                &model->text_cpu, &model->text_program,
                executor_arena, executor_bytes) != 0 ||
            salt_text_verify_cpu_parallel_bind(
                &model->text_cpu, g4_text_verify_parallel_run, model,
                model->compute_pool.asc8,
                (size_t)8u * (size_t)model->max_context,
                (uint32_t)model->target_area_active_workers) != 0 ||
            salt_text_verify_cpu_profile_set(
                &model->text_cpu, target_profile) != 0 ||
            salt_text_verify_cpu_executor_init(
                &model->text_executor, &model->text_cpu) != 0) {
            if (model->text_cpu_shared.contents)
                (void)salt_gpu_shared_buffer_free(&model->text_cpu_shared);
            else
                free(executor_arena);
            free(program_arena);
            memset(&model->text_program, 0, sizeof model->text_program);
            memset(&model->text_cpu, 0, sizeof model->text_cpu);
            memset(&model->text_executor, 0, sizeof model->text_executor);
            return -1;
        }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        fputs("GEMMA4_TEXT_VERIFY_STAGE cpu_compile_done\n", stderr);
        fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    }
    model->text_program_arena = program_arena;
    model->text_program_arena_bytes = program_bytes;
    if (g4_target_gpu_program(model)) {
        model->text_gpu_arena = executor_arena;
        model->text_gpu_arena_bytes = executor_bytes;
    } else {
        model->text_cpu_arena = executor_arena;
        model->text_cpu_arena_bytes = executor_bytes;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    fputs("GEMMA4_TEXT_VERIFY_STAGE ready\n", stderr);
    fflush(stderr);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    return 0;
}

static int init_text_token_program(SaltGemma4Text *model) {
    SaltTextTokenCell cells[G4_TOKEN_PROGRAM_CELLS];
    SaltTextTokenProgramDesc descriptor;
    uint32_t index = 0, completion = 1;
    size_t required = 0;
    if (!model || model->token_program.ready || model->position != 0)
        return -1;
    memset(cells, 0, sizeof cells);
    cells[index++] = (SaltTextTokenCell) {
        SALT_TEXT_TOKEN_EMBED, -1, 0, -1, completion++,
        SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
        0, 0, G4_HIDDEN * sizeof(float),
    };
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        cells[index] = (SaltTextTokenCell) {
            SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT, layer, 0,
            (int32_t)index - 1, completion++,
            SALT_TEXT_TOKEN_CELL_BARRIER_AFTER |
                SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
            0, 0, G4_HIDDEN * sizeof(float),
        };
        index++;
        cells[index] = (SaltTextTokenCell) {
            SALT_TEXT_TOKEN_LAYER_FEED_FORWARD_PARENT, layer, 0,
            (int32_t)index - 1, completion++,
            SALT_TEXT_TOKEN_CELL_BARRIER_AFTER |
                SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
            0, 0, G4_HIDDEN * sizeof(float),
        };
        index++;
    }
    cells[index] = (SaltTextTokenCell) {
        SALT_TEXT_TOKEN_FINAL_NORM, -1, 0, (int32_t)index - 1,
        completion++, SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
        0, 0, G4_HIDDEN * sizeof(float),
    };
    index++;
    cells[index] = (SaltTextTokenCell) {
        SALT_TEXT_TOKEN_HEAD, -1, 0, (int32_t)index - 1,
        completion++, SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
        0, 0, G4_VOCAB * sizeof(float),
    };
    index++;
    cells[index] = (SaltTextTokenCell) {
        SALT_TEXT_TOKEN_SOFTCAP, -1, 0, (int32_t)index - 1,
        completion++, SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
        0, 0, G4_VOCAB * sizeof(float),
    };
    index++;
    cells[index] = (SaltTextTokenCell) {
        SALT_TEXT_TOKEN_FINALIZE, -1, 0, (int32_t)index - 1,
        completion++, 0, 0, 0, 0,
    };
    index++;
    if (index != G4_TOKEN_PROGRAM_CELLS) return -1;
    descriptor = (SaltTextTokenProgramDesc) {
        cells, index, G4_LAYERS, 2u * G4_LAYERS,
    };
    if (salt_text_token_program_arena_requirement(
            &descriptor, &required) != 0 ||
        required != sizeof model->token_program_cells ||
        salt_text_token_program_compile(&model->token_program, &descriptor,
            model->token_program_cells,
            sizeof model->token_program_cells) != 0)
        return -1;
    return 0;
}

typedef struct {
    const uint32_t *values;
    const uint16_t *scales;
    const uint16_t *biases;
    int rows;
    int columns;
    int batch;
    int workers;
    const float *inputs;
    float *outputs;
    int failed[32];
} G4Q4PoolTask;

static void g4_q4_pool_worker(int worker, void *opaque) {
    G4Q4PoolTask *task = (G4Q4PoolTask *)opaque;
    int chunk = task->rows / task->workers +
                (task->rows % task->workers != 0);
    int64_t wide_start = (int64_t)worker * chunk;
    int64_t wide_end = wide_start + chunk;
    int row_start = wide_start < task->rows ? (int)wide_start : task->rows;
    int row_end = wide_end < task->rows ? (int)wide_end : task->rows;
    if (row_start >= row_end) return;
    if (salt_q4_matvec_batch_rows(
            task->values, task->scales, task->biases,
            task->rows, task->columns, task->batch,
            task->inputs, task->outputs, row_start, row_end) != 0)
        task->failed[worker] = 1;
}

static int g4_q4_pool_batch_workers(SaltGemma4Text *model,
                                    const uint32_t *values,
                                    const uint16_t *scales,
                                    const uint16_t *biases,
                                    int rows, int columns, int batch,
                                    const float *inputs, float *outputs,
                                    int worker_limit) {
    G4Q4PoolTask task;
    int workers;
    if (!model || !model->compute_pool_ready ||
        !model->compute_pool.aq4_pool_enabled ||
        model->compute_pool.aq4_threads < 1 ||
        model->compute_pool.aq4_threads > model->compute_pool.apool_threads)
        return -1;
    workers = worker_limit > 0
        ? worker_limit : model->compute_pool.aq4_threads;
    if (workers > model->compute_pool.apool_threads) return -1;
    if (workers > rows) workers = rows;
    memset(&task, 0, sizeof task);
    task.values = values;
    task.scales = scales;
    task.biases = biases;
    task.rows = rows;
    task.columns = columns;
    task.batch = batch;
    task.workers = workers;
    task.inputs = inputs;
    task.outputs = outputs;
    if (salt_attn_pool_run_n(
            &model->compute_pool, workers, g4_q4_pool_worker, &task) != 0)
        return -1;
    for (int worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return -1;
    model->q4_pool_submissions++;
    return 0;
}

static int g4_q4_pool_batch(SaltGemma4Text *model,
                            const uint32_t *values,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            int rows, int columns, int batch,
                            const float *inputs, float *outputs) {
    return g4_q4_pool_batch_workers(model, values, scales, biases,
        rows, columns, batch, inputs, outputs, 0);
}

typedef struct {
    SaltBatchJob *jobs;
    int job_count;
    int64_t units;
    int workers;
    int pair_mode;
    int64_t boundaries[33];
    int failed[32];
} G4Q4MultiPoolTask;

static void g4_q4_multi_pool_worker(int worker, void *opaque) {
    G4Q4MultiPoolTask *task = (G4Q4MultiPoolTask *)opaque;
    int64_t unit_start = task->boundaries[worker];
    int64_t unit_end = task->boundaries[worker + 1];
    if (unit_start >= unit_end) return;
    if (!task->pair_mode) {
        if (salt_q4_multi_batch_rows(
                task->jobs, task->job_count, unit_start, unit_end) != 0)
            task->failed[worker] = 1;
        return;
    }
    {
        int64_t base = 0;
        for (int job = 0; job < task->job_count && base < unit_end; job++) {
            SaltBatchJob *current = &task->jobs[job];
            int64_t pairs = (int64_t)current->R * current->B;
            int64_t lo = unit_start > base ? unit_start - base : 0;
            int64_t hi = unit_end < base + pairs
                ? unit_end - base : pairs;
            while (lo < hi) {
                int batch = (int)(lo / current->R);
                int row_first = (int)(lo % current->R);
                int64_t batch_end = (int64_t)(batch + 1) * current->R;
                int64_t segment_end = hi < batch_end ? hi : batch_end;
                int row_end = (int)(segment_end -
                    (int64_t)batch * current->R);
                if (salt_q4_matvec_batch_tile(
                        current->vals, current->scales, current->biases,
                        current->R, current->C, current->B,
                        current->xs, current->ys,
                        batch, batch + 1, row_first, row_end) != 0) {
                    task->failed[worker] = 1;
                    return;
                }
                lo = segment_end;
            }
            base += pairs;
        }
    }
}

static int g4_q4_pool_multi_batch_workers(SaltGemma4Text *model,
                                          SaltBatchJob *jobs, int job_count,
                                          int worker_limit) {
    G4Q4MultiPoolTask task;
    int64_t units = 0, work = 0;
    int workers, pair_mode;
    if (!model || !jobs || job_count < 1 || !model->compute_pool_ready ||
        !model->compute_pool.aq4_pool_enabled ||
        model->compute_pool.aq4_threads < 1 ||
        model->compute_pool.aq4_threads > model->compute_pool.apool_threads)
        return -1;
    for (int job = 0; job < job_count; job++) {
        int64_t job_units, per_unit;
        if (jobs[job].R < 1 || jobs[job].B < 1 || jobs[job].C < 1 ||
            jobs[job].B > INT64_MAX / jobs[job].R ||
            jobs[job].B > INT64_MAX / jobs[job].C)
            return -1;
        job_units = jobs[job].R;
        per_unit = (int64_t)jobs[job].B * jobs[job].C;
        if (per_unit < 1 || job_units > INT64_MAX - units ||
            job_units > (INT64_MAX - work) / per_unit)
            return -1;
        units += job_units;
        work += job_units * per_unit;
    }
    workers = worker_limit > 0
        ? worker_limit : model->compute_pool.aq4_threads;
    if (workers > model->compute_pool.apool_threads) return -1;
    /* Preserve the exact N-row batch microtile whenever output rows can fill
     * the admitted worker width. Candidate/output pair tiling is a bounded
     * fallback only for genuinely output-row-narrow work. */
    pair_mode = worker_limit > 0 && model->target_policy.n_parallel &&
        units < workers;
    if (pair_mode) {
        units = 0;
        work = 0;
        for (int job = 0; job < job_count; job++) {
            int64_t job_units = (int64_t)jobs[job].B * jobs[job].R;
            int64_t per_unit = jobs[job].C;
            if (job_units < 1 || job_units > INT64_MAX - units ||
                job_units > (INT64_MAX - work) / per_unit)
                return -1;
            units += job_units;
            work += job_units * per_unit;
        }
    }
    if ((int64_t)workers > units) workers = (int)units;
    memset(&task, 0, sizeof task);
    task.jobs = jobs;
    task.job_count = job_count;
    task.units = units;
    task.workers = workers;
    task.pair_mode = pair_mode;
    task.boundaries[0] = 0;
    task.boundaries[workers] = units;
    for (int worker = 1; worker < workers; worker++) {
        int64_t quotient = work / workers;
        int64_t remainder = work % workers;
        int64_t target = quotient * worker +
            (remainder * worker + workers - 1) / workers;
        int64_t prior_work = 0, prior_units = 0;
        for (int job = 0; job < job_count; job++) {
            int64_t job_units = pair_mode
                ? (int64_t)jobs[job].B * jobs[job].R
                : jobs[job].R;
            int64_t per_unit = pair_mode
                ? jobs[job].C
                : (int64_t)jobs[job].B * jobs[job].C;
            int64_t job_work = job_units * per_unit;
            if (target <= prior_work + job_work) {
                int64_t within = target - prior_work;
                int64_t assigned = within / per_unit +
                    (within % per_unit != 0);
                if (assigned > job_units) assigned = job_units;
                task.boundaries[worker] = prior_units + assigned;
                break;
            }
            prior_work += job_work;
            prior_units += job_units;
        }
    }
    for (int worker = 0; worker < workers; worker++)
        if (task.boundaries[worker] > task.boundaries[worker + 1] ||
            task.boundaries[worker + 1] > units)
            return -1;
    if (salt_attn_pool_run_n(&model->compute_pool, workers,
            g4_q4_multi_pool_worker, &task) != 0)
        return -1;
    for (int worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return -1;
    model->q4_pool_submissions++;
    model->q4_multi_pool_submissions++;
    model->q4_multi_pool_jobs += (uint64_t)job_count;
    return 0;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
typedef struct {
    SaltBatchJob *jobs;
    int job_count;
    int workers;
    int tile_rounds;
    int64_t tile_count;
    int failed[32];
} G4Q4TilePoolTask;

static void g4_q4_tile_pool_worker(int worker, void *opaque) {
    G4Q4TilePoolTask *task = (G4Q4TilePoolTask *)opaque;
    int64_t tile;
    if (!task || worker < 0 || worker >= task->workers) return;
    /* A fixed permutation gives every worker the same row work while rotating
     * it across expert/projection mappings. Completion order cannot select or
     * publish results, and no claim mutex sits on the compute path. */
    for (tile = worker; tile < task->tile_count; tile += task->workers) {
        int lane, job, round, row_first, row_end;
        SaltBatchJob *current;
        lane = (int)(tile % task->job_count);
        round = (int)(tile / task->job_count);
        job = (lane + round) % task->job_count;
        current = &task->jobs[job];
        row_first = (int)(((int64_t)current->R * round) /
                          task->tile_rounds);
        row_end = (int)(((int64_t)current->R * (round + 1)) /
                        task->tile_rounds);
        if (row_first >= row_end) continue;
        if (salt_q4_matvec_batch_rows(
                current->vals, current->scales, current->biases,
                current->R, current->C, current->B,
                current->xs, current->ys, row_first, row_end) != 0) {
            task->failed[worker] = 1;
            return;
        }
    }
}

static int g4_q4_pool_multi_batch_tiles(SaltGemma4Text *model,
                                        SaltBatchJob *jobs, int job_count,
                                        int worker_limit, int tile_rounds) {
    G4Q4TilePoolTask task;
    int workers;
    if (!model || !jobs || job_count < 1 || worker_limit < 1 ||
        tile_rounds < 1 || tile_rounds > 64 ||
        !model->compute_pool_ready || !model->compute_pool.aq4_pool_enabled ||
        model->compute_pool.aq4_threads < 1 ||
        model->compute_pool.aq4_threads > model->compute_pool.apool_threads ||
        job_count > INT64_MAX / tile_rounds)
        return -1;
    workers = worker_limit;
    if (workers > model->compute_pool.apool_threads || workers > 32)
        return -1;
    memset(&task, 0, sizeof task);
    task.jobs = jobs;
    task.job_count = job_count;
    task.workers = workers;
    task.tile_rounds = tile_rounds;
    task.tile_count = (int64_t)job_count * tile_rounds;
    for (int job = 0; job < job_count; job++)
        if (jobs[job].R < 1 || jobs[job].B != 1 || jobs[job].C < 1 ||
            !jobs[job].vals || !jobs[job].scales || !jobs[job].biases ||
            !jobs[job].xs || !jobs[job].ys)
            return -1;
    if (salt_attn_pool_run_n(&model->compute_pool, workers,
            g4_q4_tile_pool_worker, &task) != 0)
        return -1;
    for (int worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return -1;
    model->q4_pool_submissions++;
    model->q4_multi_pool_submissions++;
    model->q4_multi_pool_jobs += (uint64_t)job_count;
    return 0;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_q4_pool_multi_batch(SaltGemma4Text *model,
                                  SaltBatchJob *jobs, int job_count) {
    return g4_q4_pool_multi_batch_workers(model, jobs, job_count, 0);
}

typedef struct {
    const uint32_t *values;
    const uint16_t *scales;
    const uint16_t *biases;
    int rows;
    int columns;
    int batch;
    int workers;
    const float *inputs;
    float *outputs;
    int failed[32];
} G4Q8PoolTask;

static void g4_q8_pool_worker(int worker, void *opaque) {
    G4Q8PoolTask *task = (G4Q8PoolTask *)opaque;
    int chunk = task->rows / task->workers +
                (task->rows % task->workers != 0);
    int64_t wide_start = (int64_t)worker * chunk;
    int64_t wide_end = wide_start + chunk;
    int row_start = wide_start < task->rows ? (int)wide_start : task->rows;
    int row_end = wide_end < task->rows ? (int)wide_end : task->rows;
    if (row_start >= row_end) return;
    if (salt_q8_matvec_batch_rows(
            task->values, task->scales, task->biases,
            task->rows, task->columns, task->batch,
            task->inputs, task->outputs, row_start, row_end) != 0)
        task->failed[worker] = 1;
}

static int g4_q8_pool_batch(SaltGemma4Text *model, const G4Q *matrix,
                            int batch, const float *inputs, float *outputs) {
    G4Q8PoolTask task;
    int workers;
    if (!model || !matrix || !matrix->set || matrix->bits != 8 || batch < 1 ||
        !inputs || !outputs || !model->compute_pool_ready ||
        model->compute_pool.aq4_threads < 1 ||
        model->compute_pool.aq4_threads > model->compute_pool.apool_threads)
        return -1;
    workers = model->compute_pool.aq4_threads;
    if (workers > matrix->rows) workers = matrix->rows;
    memset(&task, 0, sizeof task);
    task.values = matrix->weight;
    task.scales = matrix->scales;
    task.biases = matrix->biases;
    task.rows = matrix->rows;
    task.columns = matrix->cols;
    task.batch = batch;
    task.workers = workers;
    task.inputs = inputs;
    task.outputs = outputs;
    if (salt_attn_pool_run_n(
            &model->compute_pool, workers, g4_q8_pool_worker, &task) != 0)
        return -1;
    for (int worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return -1;
    model->q8_pool_submissions++;
    return 0;
}

static int g4_shared_output_ref(SaltGemma4Text *model, const float *output,
                                size_t float_count,
                                SaltGpuSharedBuffer **buffer_out,
                                size_t *float_offset_out) {
    SaltGpuSharedBuffer *buffers[3];
    uintptr_t pointer;
    if (!model || !output || !buffer_out || !float_offset_out ||
        !model->shared_arenas_ready ||
        float_count > SIZE_MAX / sizeof(float))
        return 0;
    buffers[0] = &model->operation_shared;
    buffers[1] = &model->compute_scratch_shared;
    buffers[2] = &model->decode_shared;
    pointer = (uintptr_t)(const void *)output;
    for (size_t i = 0; i < sizeof buffers / sizeof buffers[0]; i++) {
        uintptr_t base;
        size_t delta;
        if (!buffers[i]->contents || !buffers[i]->backend ||
            buffers[i]->nbytes < sizeof(float))
            continue;
        base = (uintptr_t)buffers[i]->contents;
        if (pointer < base || pointer - base > SIZE_MAX) continue;
        delta = (size_t)(pointer - base);
        if (delta % sizeof(float) != 0 || delta > buffers[i]->nbytes ||
            float_count > (buffers[i]->nbytes - delta) / sizeof(float))
            continue;
        *buffer_out = buffers[i];
        *float_offset_out = delta / sizeof(float);
        return 1;
    }
    return 0;
}

static int q_matvec_batch(SaltGemma4Text *model,
                          const G4Q *matrix, int batch,
                          const float *inputs, float *outputs) {
    if (!matrix || !matrix->set || batch < 1 ||
        !inputs || !outputs)
        return -1;
    if (matrix->bits == 40 && model && model->full_gpu_intent) {
        if (salt_gpu_nvfp4_matvec_batch(matrix->weight,
                matrix->rows, matrix->cols, batch, inputs, outputs) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_dense_submissions += (uint64_t)batch;
        if (batch == 1) model->gpu_decode_submissions++;
        return 0;
    }
    if (matrix->bits == 16) {
        for (int token = 0; token < batch; token++)
            if (q_matvec(matrix,
                    inputs + (size_t)token * (size_t)matrix->cols,
                    outputs + (size_t)token * (size_t)matrix->rows) != 0)
                return -1;
        return 0;
    }
    if (matrix->bits != 4) {
        const char *mode = getenv("SALT_PREFILL_CHUNK");
        if (!mode || strcmp(mode, "0") != 0) return -1;
        for (int token = 0; token < batch; token++)
            if (q_matvec(matrix,
                    inputs + (size_t)token * (size_t)matrix->cols,
                    outputs + (size_t)token * (size_t)matrix->rows) != 0)
                return -1;
        return 0;
    }
    if (model && model->full_gpu_intent) {
        const float *xs[G4_CUDA_BATCH_MAX];
        float *ys[G4_CUDA_BATCH_MAX];
        SaltGpuSharedBuffer *shared_output = NULL;
        size_t shared_offset = 0;
        int shared = batch > 1 && g4_shared_output_ref(
            model, outputs, (size_t)batch * (size_t)matrix->rows,
            &shared_output, &shared_offset);
        for (int first = 0; first < batch; first += G4_CUDA_BATCH_MAX) {
            int count = batch - first;
            if (count > G4_CUDA_BATCH_MAX) count = G4_CUDA_BATCH_MAX;
            if (shared && count > 1) {
                if (salt_gpu_proj_batch_indexed(
                        matrix->weight, matrix->scales, matrix->biases,
                        matrix->rows, matrix->cols, count, 0, count,
                        inputs + (size_t)first * (size_t)matrix->cols,
                        shared_output,
                        shared_offset + (size_t)first * (size_t)matrix->rows,
                        matrix->weight) != 0 || salt_gpu_sync() != 0) {
                    model->gpu_failures++;
                    return -1;
                }
                model->gpu_shared_output_batches++;
                model->gpu_shared_output_jobs += (uint64_t)count;
                continue;
            }
            for (int token = 0; token < count; token++) {
                xs[token] = inputs +
                    (size_t)(first + token) * (size_t)matrix->cols;
                ys[token] = outputs +
                    (size_t)(first + token) * (size_t)matrix->rows;
            }
            if (salt_gpu_proj_batch(matrix->weight, matrix->scales,
                    matrix->biases, matrix->rows, matrix->cols, count,
                    xs, ys, matrix->weight) != 0) {
                model->gpu_failures++;
                return -1;
            }
        }
        if (matrix == &model->embedding)
            model->gpu_head_submissions += (uint64_t)batch;
        else
            model->gpu_dense_submissions += (uint64_t)batch;
        if (batch == 1) model->gpu_decode_submissions++;
        return 0;
    }
    if (batch == 1) {
        const char *mode = getenv("SALT_PREFILL_CHUNK");
        int serial_proof = mode && strcmp(mode, "0") == 0;
        if (!serial_proof)
            goto persistent;
        return q_matvec(matrix, inputs, outputs);
    }
persistent:
    if (g4_q4_pool_batch(model,
            matrix->weight, matrix->scales, matrix->biases,
            matrix->rows, matrix->cols, batch, inputs, outputs) == 0)
        return 0;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* A selected persistent-pool execution failed; never retry its work. */
    return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (model) model->q4_pool_fallbacks++;
    return salt_q4_matvec_batch(
        matrix->weight, matrix->scales, matrix->biases,
        matrix->rows, matrix->cols, batch, inputs, outputs);
}

static int gpu_q8_matvec_batch(SaltGemma4Text *model,
                               const G4Q *matrix, int batch,
                               const float *inputs, float *outputs) {
    const uint32_t *vals[G4_CUDA_BATCH_MAX];
    const uint16_t *scales[G4_CUDA_BATCH_MAX];
    const uint16_t *biases[G4_CUDA_BATCH_MAX];
    const float *xs[G4_CUDA_BATCH_MAX];
    float *ys[G4_CUDA_BATCH_MAX];
    const void *ids[G4_CUDA_BATCH_MAX];
    int rows[G4_CUDA_BATCH_MAX];
    if (!model || !matrix || !matrix->set || matrix->bits != 8 ||
        batch < 1 || !inputs || !outputs)
        return -1;
    for (int first = 0; first < batch; first += G4_CUDA_BATCH_MAX) {
        int count = batch - first;
        if (count > G4_CUDA_BATCH_MAX) count = G4_CUDA_BATCH_MAX;
        for (int token = 0; token < count; token++) {
            vals[token] = matrix->weight;
            scales[token] = matrix->scales;
            biases[token] = matrix->biases;
            xs[token] = inputs +
                (size_t)(first + token) * (size_t)matrix->cols;
            ys[token] = outputs +
                (size_t)(first + token) * (size_t)matrix->rows;
            ids[token] = matrix->weight;
            rows[token] = matrix->rows;
        }
        if (salt_gpu_q8_batch(vals, scales, biases, xs, ys, ids,
                rows, matrix->cols, count) != 0) {
            model->gpu_failures++;
            return -1;
        }
    }
    model->gpu_router_submissions += (uint64_t)batch;
    if (batch == 1) model->gpu_decode_submissions++;
    return 0;
}

static int q_matvec_operation(SaltGemma4Text *model, const G4Q *matrix,
                              const float *input, float *output) {
    const char *mode;
    int serial_proof;
    if (!model || !matrix || !matrix->set || !input || !output)
        return -1;
    if (matrix->bits == 16 && model->full_gpu_intent &&
            model->nvfp4_cuda_head && matrix == &model->embedding) {
        if (salt_gpu_bf16_matvec(matrix->weight, matrix->rows, matrix->cols,
                                 input, output) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_head_submissions++;
        model->gpu_decode_submissions++;
        return 0;
    }
    if (matrix->bits == 40 && model->full_gpu_intent) {
        if (salt_gpu_nvfp4_matvec(matrix->weight, matrix->rows, matrix->cols,
                                  input, output) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_dense_submissions++;
        model->gpu_decode_submissions++;
        return 0;
    }
    mode = getenv("SALT_PREFILL_CHUNK");
    serial_proof = mode && strcmp(mode, "0") == 0;
    if (serial_proof) {
        return q_matvec(matrix, input, output);
    }
    if (matrix->bits == 4)
        return q_matvec_batch(model, matrix, 1, input, output);
    if (matrix->bits == 8) {
        if (model->full_gpu_intent)
            return gpu_q8_matvec_batch(model, matrix, 1, input, output);
        if (g4_q8_pool_batch(model, matrix, 1, input, output) == 0)
            return 0;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        return -1; /* Failed execution is not an unsupported-method result. */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        model->q8_pool_fallbacks++;
        return q_matvec(matrix, input, output);
    }
    return q_matvec(matrix, input, output);
}

static int q_matvec_batch_exact_views(SaltGemma4Text *model,
                                      const G4Q *matrix, int batch,
                                      const float *inputs, float *outputs) {
    SaltGpuSharedBuffer *shared_output = NULL;
    size_t shared_offset = 0;
    if (!model || !model->full_gpu_intent || !matrix || !matrix->set ||
        matrix->bits != 4 || batch < 2 || !inputs || !outputs ||
        !g4_shared_output_ref(model, outputs,
            (size_t)batch * (size_t)matrix->rows,
            &shared_output, &shared_offset))
        return -1;
    for (int first = 0; first < batch; first += G4_CUDA_BATCH_MAX) {
        int count = batch - first;
        if (count > G4_CUDA_BATCH_MAX) count = G4_CUDA_BATCH_MAX;
        if (salt_gpu_proj_batch_indexed_exact_views(
                matrix->weight, matrix->scales, matrix->biases,
                matrix->rows, matrix->cols, batch, first, count, inputs,
                shared_output, shared_offset, matrix->weight) != 0 ||
            salt_gpu_sync() != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_shared_output_batches++;
        model->gpu_shared_output_jobs += (uint64_t)(uint32_t)count;
    }
    model->gpu_dense_submissions += (uint64_t)(uint32_t)batch;
    return 0;
}

static int qkv_projection_batch(SaltGemma4Text *model, G4Layer *layer,
                                int batch, const float *input,
                                float *query, float *key, float *value) {
    G4Q *matrices[3];
    float *outputs[3];
    SaltBatchJob jobs[3];
    const char *enabled, *mode;
    int job_count;
    if (!model || !layer || batch < 1 || !input || !query || !key ||
        (!layer->full_attention && !value))
        return -1;
    if (batch == 1 && model->full_gpu_intent && model->nvfp4_mode &&
            model->nvfp4_qkv_aggregate) {
        int count = layer->full_attention ? 2 : 3;
        G4Q *nv_matrices[3] = {
            &layer->q_proj,
            layer->full_attention ? &layer->kv_proj : &layer->k_proj,
            &layer->v_proj,
        };
        float *nv_outputs[3] = {query, key, value};
        int nv_batches[3] = {batch, batch, batch};
        int nv_rows[3];
        for (int job = 0; job < count; job++) {
            if (!nv_matrices[job]->set || nv_matrices[job]->bits != 40)
                return -1;
            model->prefill_gpu_ids[job] = nv_matrices[job]->weight;
            model->prefill_gpu_xs[job] = input;
            model->prefill_gpu_ys[job] = nv_outputs[job];
            nv_rows[job] = nv_matrices[job]->rows;
        }
        if (salt_gpu_nvfp4_mixed_batch(model->prefill_gpu_ids,
                model->prefill_gpu_xs, model->prefill_gpu_ys,
                nv_batches, nv_rows, G4_HIDDEN, count) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_dense_submissions += (uint64_t)batch * (uint64_t)count;
        if (batch == 1)
            model->gpu_decode_submissions += (uint64_t)count;
        return 0;
    }
    if (!model->full_gpu_intent) {
        enabled = getenv("SALT_QKV_BATCH");
        if (!enabled || !*enabled || *enabled == '0') goto sequential;
        mode = getenv("SALT_PREFILL_CHUNK");
        if (mode && strcmp(mode, "0") == 0) goto sequential;
    }
    matrices[0] = &layer->q_proj;
    outputs[0] = query;
    if (layer->full_attention) {
        matrices[1] = &layer->kv_proj;
        outputs[1] = key;
        job_count = 2;
    } else {
        matrices[1] = &layer->k_proj;
        matrices[2] = &layer->v_proj;
        outputs[1] = key;
        outputs[2] = value;
        job_count = 3;
    }
    for (int job = 0; job < job_count; job++) {
        G4Q *matrix = matrices[job];
        if (!matrix->set || matrix->bits != 4) goto sequential;
        jobs[job] = (SaltBatchJob) {
            matrix->weight, matrix->scales, matrix->biases,
            matrix->rows, matrix->cols, batch, input, outputs[job],
            0, matrix->rows,
        };
    }
    if (model->full_gpu_intent && model->gpu_trunk_exact_views &&
            batch > 1) {
        for (int job = 0; job < job_count; job++)
            if (q_matvec_batch_exact_views(model, matrices[job], batch,
                    input, outputs[job]) != 0)
                return -1;
        return 0;
    }
    if (model->full_gpu_intent) {
        int expanded = 0;
        if (g4_gpu_multi_batch(model, jobs, job_count, &expanded) != 0)
            return -1;
        model->gpu_dense_submissions += (uint64_t)expanded;
        if (batch == 1) model->gpu_decode_submissions += (uint64_t)expanded;
        return 0;
    }
    if (g4_q4_pool_multi_batch(model, jobs, job_count) == 0) {
        model->qkv_multi_pool_submissions++;
        return 0;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    return -1; /* Preserve explicit pre-dispatch sequential selection only. */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->q4_multi_pool_fallbacks++;

sequential:
    if (q_matvec_batch(model, &layer->q_proj, batch, input, query) != 0)
        return -1;
    if (layer->full_attention)
        return q_matvec_batch(model, &layer->kv_proj, batch, input, key);
    if (q_matvec_batch(model, &layer->k_proj, batch, input, key) != 0)
        return -1;
    return q_matvec_batch(model, &layer->v_proj, batch, input, value);
}

static int qkv_projection_operation(SaltGemma4Text *model, G4Layer *layer,
                                    const float *input) {
    return qkv_projection_batch(model, layer, 1, input,
                                model->q, model->k, model->v);
}

static int dense_gate_up_batch(SaltGemma4Text *model, G4Layer *layer,
                               int batch, const float *input,
                               float *gate, float *up) {
    SaltBatchJob jobs[2];
    const char *mode;
    if (!model || !layer || batch < 1 || !input || !gate || !up)
        return -1;
    if (batch == 1 && model->full_gpu_intent && model->nvfp4_mode &&
            model->nvfp4_dense_aggregate) {
        const void *keys[2] = {
            layer->dense_gate.weight, layer->dense_up.weight,
        };
        const float *inputs[2] = {input, input};
        float *outputs[2] = {gate, up};
        int batches[2] = {1, 1};
        if (!layer->dense_gate.set || layer->dense_gate.bits != 40 ||
            !layer->dense_up.set || layer->dense_up.bits != 40 ||
            layer->dense_gate.rows != layer->dense_up.rows ||
            layer->dense_gate.cols != layer->dense_up.cols ||
            salt_gpu_nvfp4_batch(keys, inputs, outputs, batches,
                layer->dense_gate.rows, layer->dense_gate.cols, 2) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_dense_submissions += 2;
        model->gpu_decode_submissions += 2;
        return 0;
    }
    if (!model->full_gpu_intent) {
        mode = getenv("SALT_PREFILL_CHUNK");
        if (mode && strcmp(mode, "0") == 0) goto sequential;
    }
    if (!layer->dense_gate.set || layer->dense_gate.bits != 4 ||
        !layer->dense_up.set || layer->dense_up.bits != 4)
        goto sequential;
    jobs[0] = (SaltBatchJob) {
        layer->dense_gate.weight, layer->dense_gate.scales,
        layer->dense_gate.biases, layer->dense_gate.rows,
        layer->dense_gate.cols, batch, input, gate,
        0, layer->dense_gate.rows,
    };
    jobs[1] = (SaltBatchJob) {
        layer->dense_up.weight, layer->dense_up.scales,
        layer->dense_up.biases, layer->dense_up.rows,
        layer->dense_up.cols, batch, input, up,
        0, layer->dense_up.rows,
    };
    if (model->full_gpu_intent) {
        int expanded = 0;
        if (g4_gpu_multi_batch(model, jobs, 2, &expanded) != 0)
            return -1;
        model->gpu_dense_submissions += (uint64_t)expanded;
        if (batch == 1) model->gpu_decode_submissions += (uint64_t)expanded;
        return 0;
    }
    if (g4_q4_pool_multi_batch(model, jobs, 2) == 0) return 0;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->q4_multi_pool_fallbacks++;

sequential:
    if (q_matvec_batch(model, &layer->dense_gate,
            batch, input, gate) != 0)
        return -1;
    return q_matvec_batch(model, &layer->dense_up,
                          batch, input, up);
}

/* One exact dense gate/up -> GeGLU -> down operation over the existing
 * backend resource/address map. PREFILL already uses this chain on the
 * registered/pageable CUDA path; ordinary B1 DECODE now uses the same
 * model-neutral backend operation on Metal, CUDA, and HIP. */
static int g4_gpu_dense_q4_chain(SaltGemma4Text *model, G4Layer *layer,
                                int batch, const float *inputs,
                                float *outputs, float *gate_scratch,
                                float *up_scratch) {
    SaltGpuMoeExpert dense;
    size_t scratch_count;
    if (!model || !layer || batch < 1 || !inputs || !outputs ||
        !gate_scratch || !up_scratch)
        return -1;
    if (!model->full_gpu_intent || model->nvfp4_mode) return 1;
    if (!layer->dense_gate.set || layer->dense_gate.bits != 4 ||
        !layer->dense_up.set || layer->dense_up.bits != 4 ||
        !layer->dense_down.set || layer->dense_down.bits != 4 ||
        (size_t)batch > SIZE_MAX / G4_DENSE)
        return -1;
    scratch_count = (size_t)batch * G4_DENSE;
    memset(&dense, 0, sizeof dense);
    dense.gate_vals = layer->dense_gate.weight;
    dense.gate_scales = layer->dense_gate.scales;
    dense.gate_biases = layer->dense_gate.biases;
    dense.gate_id = layer->dense_gate.weight;
    dense.up_vals = layer->dense_up.weight;
    dense.up_scales = layer->dense_up.scales;
    dense.up_biases = layer->dense_up.biases;
    dense.up_id = layer->dense_up.weight;
    dense.down_vals = layer->dense_down.weight;
    dense.down_scales = layer->dense_down.scales;
    dense.down_biases = layer->dense_down.biases;
    dense.down_id = layer->dense_down.weight;
    dense.group = batch;
    dense.resource_slot = -1;
    dense.logical_resource_id = UINT64_MAX;
    if (salt_gpu_q4_moe_chain(&dense, 1, G4_HIDDEN, G4_DENSE,
            inputs, outputs, gate_scratch, up_scratch,
            scratch_count) != 0) {
        model->gpu_failures++;
        return -1;
    }
    model->gpu_dense_submissions += (uint64_t)(uint32_t)batch * 3u;
    if (batch == 1) model->gpu_decode_submissions += 3u;
    return 0;
}

static int g4_apply_text_rope(SaltGemma4Text *model, const G4Layer *layer,
                              float *head, int position) {
    const SaltAttentionDesc *attention;
    const float *cosines, *sines;
    int pairs;
    size_t offset;
    if (!model || !layer || !layer->plan || !head || position < 0 ||
        position >= model->max_context)
        return -1;
    attention = &layer->plan->attention;
    if (model->use_ondemand_rope_proof)
        return salt_gemma4_text_rope(
            head, attention->head_dim, attention->rope_dim,
            attention->rope_base_dim, position,
            (float)attention->rope_theta);
    if (layer->full_attention) {
        cosines = model->full_rope_cos;
        sines = model->full_rope_sin;
        pairs = model->full_rope_pairs;
    } else {
        cosines = model->sliding_rope_cos;
        sines = model->sliding_rope_sin;
        pairs = model->sliding_rope_pairs;
    }
    if (!cosines || !sines || pairs != attention->rope_dim / 2)
        return -1;
    offset = (size_t)position * (size_t)pairs;
    return salt_gemma4_text_rope_factors(
        head, attention->head_dim, attention->rope_dim,
        cosines + offset, sines + offset);
}

/* Full-attention Gemma layers share one source K/V projection, but their live
 * cache rows diverge after K scaling/RoPE. G4KVC006 stores the normalized V
 * base once and reconstructs the exact K row during import. This reproduces the
 * original sequence: y = projected * rms_inverse; y *= k_norm; RoPE(y). */
static int g4_reconstruct_shared_key_from_value(
        SaltGemma4Text *model, const G4Layer *layer, int position,
        float *key, const float *value) {
    if (!model || !layer || !layer->plan || !key || !value || key == value ||
        !layer->plan->attention.shared_kv_projection ||
        layer->kv_heads < 1 || layer->head_dim < 1 ||
        layer->kv_dim != layer->kv_heads * layer->head_dim ||
        !layer->k_norm.values || layer->k_norm.count != layer->head_dim)
        return -1;
    for (int head = 0; head < layer->kv_heads; head++) {
        float *destination = key + (size_t)head * layer->head_dim;
        const float *source = value + (size_t)head * layer->head_dim;
        for (int component = 0; component < layer->head_dim; component++) {
            float reconstructed = source[component] *
                layer->k_norm.values[component];
            if (!isfinite(reconstructed)) return -1;
            destination[component] = reconstructed;
        }
        if (g4_apply_text_rope(model, layer, destination, position) != 0)
            return -1;
    }
    return 0;
}

static int embedding_row(const G4Q *embedding, int token, float *output) {
    if (!embedding || !output || token < 0 || token >= embedding->rows)
        return -1;
    size_t base = (size_t)token * (size_t)embedding->cols;
    if (embedding->bits == 16) {
        const unsigned char *source =
            (const unsigned char *)(const void *)embedding->weight + base * 2u;
        for (int c = 0; c < embedding->cols; c++) {
            uint16_t encoded;
            memcpy(&encoded, source + (size_t)c * 2u, sizeof encoded);
            output[c] = bf16_to_float(encoded) * G4_EMBED_SCALE;
        }
        return 0;
    }
    if (embedding->bits != 4) return -1;
    const uint32_t *weight = (const uint32_t *)(const void *)(
        (const unsigned char *)(const void *)embedding->weight + base / 2);
    const uint16_t *scales = (const uint16_t *)(const void *)(
        (const unsigned char *)(const void *)embedding->scales + base / 64 * 2);
    const uint16_t *biases = (const uint16_t *)(const void *)(
        (const unsigned char *)(const void *)embedding->biases + base / 64 * 2);
    salt_q4_decode(weight, scales, biases, embedding->cols, output);
    for (int c = 0; c < embedding->cols; c++)
        output[c] *= G4_EMBED_SCALE;
    return 0;
}

static int g4_exact_view(SaltGemma4Text *model, const void *pointer,
                         size_t count, SaltGpuSharedBuffer **buffer,
                         size_t *offset) {
    const uintptr_t value = (uintptr_t)pointer;
    SaltGpuSharedBuffer *candidates[2];
    if (!model || !pointer || count < 1 || !buffer || !offset ||
        count > SIZE_MAX / sizeof(float))
        return -1;
    candidates[0] = &model->decode_shared;
    candidates[1] = &model->bf16_shared;
    for (int i = 0; i < 2; i++) {
        const uintptr_t base = (uintptr_t)candidates[i]->contents;
        size_t bytes = count * sizeof(float);
        if (!base || candidates[i]->nbytes < bytes || value < base ||
            value - base > candidates[i]->nbytes - bytes ||
            (value - base) % sizeof(float) != 0)
            continue;
        *buffer = candidates[i];
        *offset = (size_t)(value - base) / sizeof(float);
        return 0;
    }
    return -1;
}

static int g4_layer_digest(const SaltGemma4Text *model, const char *stage,
                           int layer, const void *data, size_t bytes) {
    static const char digits[] = "0123456789abcdef";
    const char *enabled = getenv("SALT_GEMMA_LAYER_DIGESTS");
    unsigned char digest[32];
    if (!enabled || strcmp(enabled, "1") != 0) return 0;
    if (!model || !stage || !data || bytes < 1 ||
        salt_sha256_bytes(data, bytes, digest) != 0)
        return -1;
    fprintf(stderr, "GEMMA4_LAYER_DIGEST position=%d layer=%d stage=%s "
            "bytes=%zu sha256=", model->position, layer, stage, bytes);
    for (int i = 0; i < 32; i++) {
        fputc(digits[digest[i] >> 4], stderr);
        fputc(digits[digest[i] & 15u], stderr);
    }
    fputc('\n', stderr);
    return 0;
}

static int g4_exact_rmsnorm(SaltGemma4Text *model, float *out,
                            const float *input, const float *weight,
                            int count, float eps, int with_scale) {
    SaltGpuExactCell cell;
    if (!model || count < 1 || (with_scale && !weight)) return -1;
    memset(&cell, 0, sizeof cell);
    cell.kind = SALT_GPU_EXACT_RMSNORM;
    cell.n = (uint32_t)count;
    cell.aux = (uint32_t)(with_scale != 0);
    cell.eps = eps;
    if (g4_exact_view(model, input, (size_t)count,
            &cell.views[0], &cell.offsets[0]) != 0 ||
        g4_exact_view(model, weight ? weight : input,
            weight ? (size_t)count : 1u,
            &cell.views[1], &cell.offsets[1]) != 0 ||
        g4_exact_view(model, out, (size_t)count,
            &cell.views[2], &cell.offsets[2]) != 0)
        return -1;
    return salt_gpu_exact_cells(&cell, 1);
}

static int g4_exact_router_input(SaltGemma4Text *model, float *out,
                                 const float *input, const float *scale,
                                 int count, float eps) {
    SaltGpuExactCell cell;
    memset(&cell, 0, sizeof cell);
    cell.kind = SALT_GPU_EXACT_ROUTER_INPUT;
    cell.n = (uint32_t)count;
    cell.eps = eps;
    cell.scalar = 1.0f / salt_sqrtf((float)count);
    if (g4_exact_view(model, input, (size_t)count,
            &cell.views[0], &cell.offsets[0]) != 0 ||
        g4_exact_view(model, scale, (size_t)count,
            &cell.views[1], &cell.offsets[1]) != 0 ||
        g4_exact_view(model, out, (size_t)count,
            &cell.views[2], &cell.offsets[2]) != 0)
        return -1;
    return salt_gpu_exact_cells(&cell, 1);
}

static int g4_exact_topk(SaltGemma4Text *model, const float *logits,
                         const float *scale, int *selected, float *weights) {
    SaltGpuExactCell cell;
    memset(&cell, 0, sizeof cell);
    cell.kind = SALT_GPU_EXACT_TOPK;
    cell.n = G4_EXPERTS;
    cell.aux = G4_TOPK;
    if (g4_exact_view(model, logits, G4_EXPERTS,
            &cell.views[0], &cell.offsets[0]) != 0 ||
        g4_exact_view(model, scale, G4_EXPERTS,
            &cell.views[1], &cell.offsets[1]) != 0 ||
        g4_exact_view(model, selected, G4_TOPK,
            &cell.views[2], &cell.offsets[2]) != 0 ||
        g4_exact_view(model, weights, G4_TOPK,
            &cell.views[3], &cell.offsets[3]) != 0)
        return -1;
    return salt_gpu_exact_cells(&cell, 1);
}

static int g4_exact_residual(SaltGemma4Text *model, float *out,
                             const float *residual, const float *branch,
                             const float *weight, int count, float eps) {
    SaltGpuExactCell cell;
    memset(&cell, 0, sizeof cell);
    cell.kind = SALT_GPU_EXACT_RESIDUAL_POSTNORM;
    cell.n = (uint32_t)count;
    cell.eps = eps;
    if (g4_exact_view(model, residual, (size_t)count,
            &cell.views[0], &cell.offsets[0]) != 0 ||
        g4_exact_view(model, branch, (size_t)count,
            &cell.views[1], &cell.offsets[1]) != 0 ||
        g4_exact_view(model, weight, (size_t)count,
            &cell.views[2], &cell.offsets[2]) != 0 ||
        g4_exact_view(model, out, (size_t)count,
            &cell.views[3], &cell.offsets[3]) != 0)
        return -1;
    return salt_gpu_exact_cells(&cell, 1);
}

static int g4_exact_combine(SaltGemma4Text *model, float *out,
                            const float *residual, const float *dense,
                            const float *routed, const float *dense_weight,
                            const float *routed_weight, const float *final_weight,
                            float scalar, int count, float eps, float *scratch) {
    const void *pointers[8] = {residual, dense, routed, dense_weight,
        routed_weight, final_weight, scratch, out};
    size_t counts[8] = {(size_t)count, (size_t)count, (size_t)count,
        (size_t)count, (size_t)count, (size_t)count,
        (size_t)count * 2u, (size_t)count};
    SaltGpuExactCell cell;
    memset(&cell, 0, sizeof cell);
    cell.kind = SALT_GPU_EXACT_PARALLEL_COMBINE;
    cell.n = (uint32_t)count;
    cell.eps = eps;
    cell.scalar = scalar;
    for (int i = 0; i < 8; i++)
        if (g4_exact_view(model, pointers[i], counts[i],
                &cell.views[i], &cell.offsets[i]) != 0)
            return -1;
    return salt_gpu_exact_cells(&cell, 1);
}

static int g4_exact_softcap(SaltGemma4Text *model, float *logits,
                            int count, float cap) {
    SaltGpuExactCell cell;
    memset(&cell, 0, sizeof cell);
    cell.kind = SALT_GPU_EXACT_SOFTCAP;
    cell.n = (uint32_t)count;
    cell.scalar = cap;
    if (g4_exact_view(model, logits, (size_t)count,
            &cell.views[0], &cell.offsets[0]) != 0)
        return -1;
    return salt_gpu_exact_cells(&cell, 1);
}

/* Shared PREFILL/DECODE lowering for the existing exact backend attention
 * transform. Return 0 when GPU work completed, 1 when the current state shape
 * is not admitted by this GPU path, and -1 on a full-intent failure. */
static int g4_gpu_attention_transform_rows(
        SaltGemma4Text *model, G4Layer *layer,
        const SaltAttentionDesc *plan, int start, int batch,
        float *queries, size_t query_count,
        float *keys, float *values, size_t kv_count) {
    SaltGpuSharedBuffer *query_buffer = NULL, *key_buffer = NULL;
    SaltGpuSharedBuffer *value_buffer = NULL;
    size_t query_offset = 0, key_offset = 0, value_offset = 0;
    size_t key_cache_offset, value_cache_offset;
    const float *cosines, *sines;
    int pairs;
    if (!model || !layer || !plan || start < 0 || batch < 1 ||
        start > model->max_context - batch || layer->q_dim < 1 ||
        layer->kv_dim < 1 ||
        (size_t)batch > SIZE_MAX / (size_t)layer->q_dim ||
        (size_t)batch > SIZE_MAX / (size_t)layer->kv_dim ||
        query_count != (size_t)batch * (size_t)layer->q_dim ||
        kv_count != (size_t)batch * (size_t)layer->kv_dim ||
        !queries || !keys || !values)
        return -1;
    if (!model->gpu_attention_enabled || layer->shared_position != 0 ||
        (layer->kv_ring && !salt_gpu_cuda_present() && !salt_gpu_rocm_present()) ||
        model->use_ondemand_rope_proof)
        return 1;
    cosines = layer->full_attention
        ? model->full_rope_cos : model->sliding_rope_cos;
    sines = layer->full_attention
        ? model->full_rope_sin : model->sliding_rope_sin;
    pairs = layer->full_attention
        ? model->full_rope_pairs : model->sliding_rope_pairs;
    if (!cosines || !sines || pairs != plan->rope_dim / 2 || pairs < 1 ||
        (size_t)start > SIZE_MAX / (size_t)pairs || !model->kv_arena ||
        !model->kv_shared.contents ||
        !g4_shared_output_ref(model, queries, query_count,
            &query_buffer, &query_offset) ||
        !g4_shared_output_ref(model, keys, kv_count,
            &key_buffer, &key_offset) ||
        !g4_shared_output_ref(model, values, kv_count,
            &value_buffer, &value_offset))
        return -1;
    key_cache_offset = (size_t)(layer->key_cache - model->kv_arena);
    value_cache_offset = (size_t)(layer->value_cache - model->kv_arena);
    return salt_gpu_attention_transform(
        plan->n_heads, plan->n_kv_heads, plan->head_dim,
        plan->rope_dim, start, batch, layer->q_dim, layer->kv_dim, G4_EPS,
        query_buffer, query_offset, key_buffer, key_offset,
        value_buffer, value_offset, layer->q_norm.values,
        layer->k_norm.values,
        cosines + (size_t)start * (size_t)pairs,
        sines + (size_t)start * (size_t)pairs, pairs,
        &model->kv_shared, key_cache_offset, value_cache_offset);
}

static int g4_gpu_attention_body_rows(
        SaltGemma4Text *model, G4Layer *layer,
        const SaltAttentionDesc *plan, int start, int batch,
        float *queries, float *outputs, size_t query_count) {
    SaltGpuSharedBuffer *query_buffer = NULL, *output_buffer = NULL;
    size_t query_offset = 0, output_offset = 0;
    size_t key_offset, value_offset;
    if (!model || !layer || !plan || start < 0 || batch < 1 ||
        start > model->max_context - batch || layer->q_dim < 1 ||
        (size_t)batch > SIZE_MAX / (size_t)layer->q_dim ||
        query_count != (size_t)batch * (size_t)layer->q_dim ||
        !queries || !outputs)
        return -1;
    if (!model->gpu_attention_enabled || layer->shared_position != 0 ||
        (layer->kv_ring && !salt_gpu_cuda_present() && !salt_gpu_rocm_present()))
        return 1;
    if (!model->kv_arena || !model->kv_shared.contents ||
        !g4_shared_output_ref(model, queries, query_count,
            &query_buffer, &query_offset) ||
        !g4_shared_output_ref(model, outputs, query_count,
            &output_buffer, &output_offset))
        return -1;
    key_offset = (size_t)(layer->key_cache - model->kv_arena);
    value_offset = (size_t)(layer->value_cache - model->kv_arena);
    if (salt_gpu_attention_batch(
            plan->kind == SALT_ATTN_FULL,
            plan->n_heads, plan->n_kv_heads, plan->head_dim,
            plan->window, query_buffer, query_offset,
            &model->kv_shared, key_offset, value_offset,
            start, batch, layer->q_dim, layer->kv_dim,
            output_buffer, output_offset) != 0)
        return -1;
    model->gpu_attention_batches++;
    model->gpu_attention_tasks +=
        (uint64_t)(uint32_t)batch * (uint32_t)plan->n_heads;
    return 0;
}

static int attention(SaltGemma4Text *model, G4Layer *layer,
                     const float *residual, int position, float *output) {
    int profile = model && model->decode_detail.enabled;
    double mark = profile ? g4_now_s() : 0.0;
    int rc, gpu_attention = 0;
    const SaltAttentionDesc *plan;
    if (!model || !layer || !layer->plan) return -1;
    plan = &layer->plan->attention;
    if (salt_gemma4_rmsnorm(model->norm_a, residual,
            layer->input_norm.values, G4_HIDDEN, G4_EPS, 1) != 0) return -1;
    if (qkv_projection_operation(model, layer, model->norm_a) != 0)
        return -1;
    if (layer->full_attention) {
        memcpy(model->v, model->k, (size_t)layer->kv_dim * sizeof(float));
    }
    if (profile) {
        model->decode_detail.attention_qkv_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    rc = model->gpu_decode_attention_enabled
        ? g4_gpu_attention_transform_rows(
            model, layer, plan, position, 1, model->q,
            (size_t)layer->q_dim, model->k, model->v,
            (size_t)layer->kv_dim)
        : 1;
    if (rc < 0) {
        model->gpu_failures++;
        return -1;
    }
    gpu_attention = rc == 0;
    if (!gpu_attention) {
        for (int head = 0; head < 16; head++) {
            float *q = model->q + (size_t)head * layer->head_dim;
            if (salt_gemma4_rmsnorm(q, q, layer->q_norm.values,
                    layer->head_dim, G4_EPS, 1) != 0 ||
                g4_apply_text_rope(model, layer, q, position) != 0)
                return -1;
        }
        for (int head = 0; head < layer->kv_heads; head++) {
            float *k = model->k + (size_t)head * layer->head_dim;
            float *v = model->v + (size_t)head * layer->head_dim;
            if (salt_gemma4_rmsnorm(k, k, layer->k_norm.values,
                    layer->head_dim, G4_EPS, 1) != 0 ||
                g4_apply_text_rope(model, layer, k, position) != 0 ||
                salt_gemma4_rmsnorm(v, v, NULL,
                    layer->head_dim, G4_EPS, 0) != 0)
                return -1;
        }
        float *key_row = g4_private_key_row(layer, position);
        float *value_row = g4_private_value_row(layer, position);
        if (!key_row || !value_row) return -1;
        memcpy(key_row, model->k, (size_t)layer->kv_dim * sizeof(float));
        memcpy(value_row, model->v, (size_t)layer->kv_dim * sizeof(float));
    }
    if (profile) {
        model->decode_detail.attention_transform_s += g4_now_s() - mark;
        mark = g4_now_s();
    }

    if (gpu_attention) {
        rc = g4_gpu_attention_body_rows(
            model, layer, plan, position, 1, model->q,
            model->attention_output, (size_t)layer->q_dim);
        if (rc != 0) {
            model->gpu_failures++;
            return -1;
        }
    } else {
        int first = layer->full_attention ? 0 : position - 1024 + 1;
        if (first < 0) first = 0;
        int count = position - first + 1;
        if (g4_attention_heads(model, layer, first, count) != 0)
            return -1;
    }
    if (profile) {
        model->decode_detail.attention_body_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (q_matvec_operation(model, &layer->o_proj,
            model->attention_output, model->branch) != 0) return -1;
    if (profile) {
        model->decode_detail.attention_o_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    rc = salt_gemma4_residual_postnorm(output, residual, model->branch,
        layer->post_attention_norm.values, G4_HIDDEN, G4_EPS);
    if (profile)
        model->decode_detail.attention_residual_s += g4_now_s() - mark;
    return rc;
}

static int attention_t4_proof(SaltGemma4Text *model, G4Layer *layer,
                     const float *residual, int position, float *output) {
    int profile = model && model->decode_detail.enabled;
    double mark = profile ? g4_now_s() : 0.0;
    int rc, layer_index;
    if (!model || !model->metal_exact_cells || !layer || layer < model->layers ||
        layer >= model->layers + G4_LAYERS)
        return -1;
    layer_index = (int)(layer - model->layers);
    if (g4_exact_rmsnorm(model, model->norm_a, residual,
            layer->input_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
        return -1;
    if (g4_layer_digest(model, "attention-input-norm", layer_index,
            model->norm_a, G4_HIDDEN * sizeof(float)) != 0)
        return -1;
    if (qkv_projection_operation(model, layer, model->norm_a) != 0)
        return -1;
    if (layer->full_attention) {
        memcpy(model->v, model->k, (size_t)layer->kv_dim * sizeof(float));
    }
    if (profile) {
        model->decode_detail.attention_qkv_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    for (int head = 0; head < 16; head++) {
        float *q = model->q + (size_t)head * layer->head_dim;
        if (salt_gemma4_rmsnorm(q, q, layer->q_norm.values,
                layer->head_dim, G4_EPS, 1) != 0 ||
            g4_apply_text_rope(model, layer, q, position) != 0) return -1;
    }
    for (int head = 0; head < layer->kv_heads; head++) {
        float *k = model->k + (size_t)head * layer->head_dim;
        float *v = model->v + (size_t)head * layer->head_dim;
        if (salt_gemma4_rmsnorm(k, k, layer->k_norm.values,
                layer->head_dim, G4_EPS, 1) != 0 ||
            g4_apply_text_rope(model, layer, k, position) != 0 ||
            salt_gemma4_rmsnorm(v, v, NULL,
                layer->head_dim, G4_EPS, 0) != 0) return -1;
    }
    {
        float *key_row = g4_private_key_row(layer, position);
        float *value_row = g4_private_value_row(layer, position);
        if (!key_row || !value_row) return -1;
        memcpy(key_row, model->k, (size_t)layer->kv_dim * sizeof(float));
        memcpy(value_row, model->v, (size_t)layer->kv_dim * sizeof(float));
    }
    if (profile) {
        model->decode_detail.attention_transform_s += g4_now_s() - mark;
        mark = g4_now_s();
    }

    int first = layer->full_attention ? 0 : position - 1024 + 1;
    if (first < 0) first = 0;
    int count = position - first + 1;
    if (g4_attention_heads(model, layer, first, count) != 0)
        return -1;
    if (profile) {
        model->decode_detail.attention_body_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (q_matvec_operation(model, &layer->o_proj,
            model->attention_output, model->branch) != 0) return -1;
    if (profile) {
        model->decode_detail.attention_o_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    rc = g4_exact_residual(model, output, residual, model->branch,
        layer->post_attention_norm.values, G4_HIDDEN, G4_EPS);
    if (rc == 0 && g4_layer_digest(model, "attention-residual", layer_index,
            output, G4_HIDDEN * sizeof(float)) != 0)
        rc = -1;
    if (profile)
        model->decode_detail.attention_residual_s += g4_now_s() - mark;
    return rc;
}

static int prefill_attention_layer(SaltGemma4Text *model, G4Layer *layer,
                                   const float *states, int token_count,
                                   const unsigned char *allow,
                                   float *q_all, float *outputs,
                                   int serial_proof) {
    float *norm = NULL, *keys = NULL, *values = NULL;
    float *attention_outputs = NULL, *branches = NULL;
    size_t hidden_count, query_count, kv_count, q_stride;
    int groups, rc = -1;
    if (!model || !layer || !states || token_count < 1 || !allow ||
        !q_all || !outputs || (serial_proof != 0 && serial_proof != 1) ||
        layer->q_dim < 1 || layer->kv_dim < 1 ||
        token_count > model->prefill_attention_capacity ||
        (layer->kv_ring && token_count > layer->kv_capacity))
        return -1;
    q_stride = (size_t)layer->q_dim;
    groups = 16 / layer->kv_heads;
    hidden_count = (size_t)token_count * G4_HIDDEN;
    query_count = (size_t)token_count * q_stride;
    kv_count = (size_t)token_count * (size_t)layer->kv_dim;
    if (hidden_count > SIZE_MAX / sizeof(float) ||
        query_count > SIZE_MAX / sizeof(float) ||
        kv_count > SIZE_MAX / sizeof(float))
        return -1;
    norm = model->prefill_attention_norm;
    keys = model->prefill_attention_keys;
    values = model->prefill_attention_values;
    attention_outputs = model->prefill_attention_outputs;
    branches = model->prefill_attention_branches;
    if (!norm || !keys || !values || !attention_outputs || !branches)
        return -1;

    for (int position = 0; position < token_count; position++)
        if (salt_gemma4_rmsnorm(
                norm + (size_t)position * G4_HIDDEN,
                states + (size_t)position * G4_HIDDEN,
                layer->input_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
            goto done;
    if (serial_proof) {
        for (int position = 0; position < token_count; position++) {
            const float *norm_row = norm + (size_t)position * G4_HIDDEN;
            if (q_matvec_operation(model, &layer->q_proj, norm_row,
                    q_all + (size_t)position * q_stride) != 0)
                goto done;
            if (layer->full_attention) {
                if (q_matvec_operation(model, &layer->kv_proj, norm_row,
                        keys + (size_t)position * layer->kv_dim) != 0)
                    goto done;
                memcpy(values + (size_t)position * layer->kv_dim,
                       keys + (size_t)position * layer->kv_dim,
                       (size_t)layer->kv_dim * sizeof(float));
            } else {
                if (q_matvec_operation(model, &layer->k_proj, norm_row,
                        keys + (size_t)position * layer->kv_dim) != 0 ||
                    q_matvec_operation(model, &layer->v_proj, norm_row,
                        values + (size_t)position * layer->kv_dim) != 0)
                    goto done;
            }
        }
    } else if (qkv_projection_batch(model, layer, token_count, norm,
                   q_all, keys, values) != 0) {
        goto done;
    } else if (layer->full_attention) {
        memcpy(values, keys, kv_count * sizeof(float));
    }

    /* Produce the complete layer K/V set before attention so image queries may
     * read later positions in their own contiguous image block. */
    for (int position = 0; position < token_count; position++) {
        float *q_row = q_all + (size_t)position * q_stride;
        float *key_row = keys + (size_t)position * layer->kv_dim;
        float *value_row = values + (size_t)position * layer->kv_dim;
        for (int head = 0; head < 16; head++) {
            float *q = q_row + (size_t)head * layer->head_dim;
            if (salt_gemma4_rmsnorm(q, q, layer->q_norm.values,
                    layer->head_dim, G4_EPS, 1) != 0 ||
                g4_apply_text_rope(model, layer, q, position) != 0)
                goto done;
        }
        for (int head = 0; head < layer->kv_heads; head++) {
            float *k = key_row + (size_t)head * layer->head_dim;
            float *v = value_row + (size_t)head * layer->head_dim;
            if (salt_gemma4_rmsnorm(k, k, layer->k_norm.values,
                    layer->head_dim, G4_EPS, 1) != 0 ||
                g4_apply_text_rope(model, layer, k, position) != 0 ||
                salt_gemma4_rmsnorm(v, v, NULL,
                    layer->head_dim, G4_EPS, 0) != 0)
                goto done;
        }
        float *key_destination = g4_private_key_row(layer, position);
        float *value_destination = g4_private_value_row(layer, position);
        if (!key_destination || !value_destination) goto done;
        memcpy(key_destination, key_row,
               (size_t)layer->kv_dim * sizeof(float));
        memcpy(value_destination, value_row,
               (size_t)layer->kv_dim * sizeof(float));
    }

    for (int position = 0; position < token_count; position++) {
        const float *q_row = q_all + (size_t)position * q_stride;
        const unsigned char *row = allow + (size_t)position * token_count;
        for (int head = 0; head < 16; head++) {
            int kv_head = head / groups;
            const float *q = q_row + (size_t)head * layer->head_dim;
            float maximum = -INFINITY;
            int allowed_count = 0;
            for (int key_position = 0; key_position < token_count;
                 key_position++) {
                const float *k;
                float score = 0.0f;
                if (!row[key_position]) continue;
                k = g4_key_row(layer, key_position) +
                    (size_t)kv_head * layer->head_dim;
                for (int d = 0; d < layer->head_dim; d++)
                    score += q[d] * k[d];
                model->scores[allowed_count++] = score;
                if (score > maximum) maximum = score;
            }
            if (allowed_count < 1) goto done;
            float sum = 0.0f;
            for (int i = 0; i < allowed_count; i++) {
                model->scores[i] = salt_expf(model->scores[i] - maximum);
                sum += model->scores[i];
            }
            if (!(sum > 0.0f) || !isfinite(sum)) goto done;
            float *head_out = attention_outputs +
                (size_t)position * q_stride +
                (size_t)head * layer->head_dim;
            memset(head_out, 0, (size_t)layer->head_dim * sizeof(float));
            int score_index = 0;
            for (int key_position = 0; key_position < token_count;
                 key_position++) {
                const float *v;
                float weight;
                if (!row[key_position]) continue;
                v = g4_value_row(layer, key_position) +
                    (size_t)kv_head * layer->head_dim;
                weight = model->scores[score_index++] / sum;
                for (int d = 0; d < layer->head_dim; d++)
                    head_out[d] += weight * v[d];
            }
        }
    }
    if (serial_proof) {
        for (int position = 0; position < token_count; position++)
            if (q_matvec_operation(model, &layer->o_proj,
                    attention_outputs + (size_t)position * q_stride,
                    branches + (size_t)position * G4_HIDDEN) != 0)
                goto done;
    } else if (q_matvec_batch(model, &layer->o_proj,
                   token_count, attention_outputs, branches) != 0) {
        goto done;
    }
    for (int position = 0; position < token_count; position++)
        if (salt_gemma4_residual_postnorm(
                outputs + (size_t)position * G4_HIDDEN,
                states + (size_t)position * G4_HIDDEN,
                branches + (size_t)position * G4_HIDDEN,
                layer->post_attention_norm.values,
                G4_HIDDEN, G4_EPS) != 0)
            goto done;
    rc = 0;

done:
    return rc;
}

typedef struct {
    SaltGemma4Text *model;
    G4Layer *layer;
    const SaltAttentionDesc *plan;
    const float *states;
    float *outputs;
    int start;
    int batch;
    int history_rows;
    size_t hidden_count;
    size_t query_count;
    size_t kv_count;
    float *norm;
    float *queries;
    float *history_keys;
    float *history_values;
    float *keys;
    float *values;
    float *attention_outputs;
    float *branches;
    int gpu_transformed;
} G4PrefillAttentionBatch;

static int g4_prefill_attention_batch_prepare(
        G4PrefillAttentionBatch *operation) {
    operation->plan = &operation->layer->plan->attention;
    operation->hidden_count = (size_t)operation->batch * G4_HIDDEN;
    operation->query_count =
        (size_t)operation->batch * (size_t)operation->layer->q_dim;
    operation->kv_count =
        (size_t)operation->batch * (size_t)operation->layer->kv_dim;
    if (operation->hidden_count > SIZE_MAX / sizeof(float) ||
        operation->query_count > SIZE_MAX / sizeof(float) ||
        operation->kv_count > SIZE_MAX / sizeof(float))
        return -1;
    operation->history_rows = operation->layer->kv_ring
        ? operation->start : 0;
    if (operation->history_rows > G4_SLIDING_WINDOW - 1)
        operation->history_rows = G4_SLIDING_WINDOW - 1;
    operation->norm = operation->model->prefill_attention_norm;
    operation->queries = operation->model->prefill_attention_queries;
    operation->history_keys = operation->model->prefill_attention_keys;
    operation->history_values = operation->model->prefill_attention_values;
    operation->keys = operation->history_keys +
        (size_t)operation->history_rows * (size_t)operation->layer->kv_dim;
    operation->values = operation->history_values +
        (size_t)operation->history_rows * (size_t)operation->layer->kv_dim;
    operation->attention_outputs =
        operation->model->prefill_attention_outputs;
    operation->branches = operation->model->prefill_attention_branches;
    if (!operation->norm || !operation->queries || !operation->keys ||
        !operation->values || !operation->attention_outputs ||
        !operation->branches)
        return -1;
    memset(operation->attention_outputs, 0,
           operation->query_count * sizeof(float));
    return 0;
}

static int g4_prefill_attention_batch_history(
        const G4PrefillAttentionBatch *operation) {
    G4Layer *layer = operation->layer;
    size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
    int first = operation->start - operation->history_rows;
    if (!layer->kv_ring) return operation->history_rows == 0 ? 0 : -1;
    for (int row = 0; row < operation->history_rows; row++) {
        const float *key = g4_key_row(layer, first + row);
        const float *value = g4_value_row(layer, first + row);
        if (!key || !value) return -1;
        memcpy(operation->history_keys + (size_t)row * layer->kv_dim,
               key, row_bytes);
        memcpy(operation->history_values + (size_t)row * layer->kv_dim,
               value, row_bytes);
    }
    return 0;
}

static int g4_prefill_attention_batch_norm(
        const G4PrefillAttentionBatch *operation) {
    for (int token = 0; token < operation->batch; token++)
        if (salt_gemma4_rmsnorm(
                operation->norm + (size_t)token * G4_HIDDEN,
                operation->states + (size_t)token * G4_HIDDEN,
                operation->layer->input_norm.values,
                G4_HIDDEN, G4_EPS, 1) != 0)
            return -1;
    return 0;
}

static int g4_prefill_attention_batch_project(
        const G4PrefillAttentionBatch *operation) {
    if (qkv_projection_batch(operation->model, operation->layer,
            operation->batch, operation->norm, operation->queries,
            operation->keys, operation->values) != 0)
        return -1;
    if (operation->plan->shared_kv_projection)
        memcpy(operation->values, operation->keys,
               operation->kv_count * sizeof(float));
    return 0;
}

static int g4_prefill_attention_batch_transform(
        G4PrefillAttentionBatch *operation) {
    SaltGemma4Text *model = operation->model;
    G4Layer *layer = operation->layer;
    const SaltAttentionDesc *plan = operation->plan;
    int transform_rc = g4_gpu_attention_transform_rows(
        model, layer, plan, operation->start, operation->batch,
        operation->queries, operation->query_count,
        operation->keys, operation->values, operation->kv_count);
    int transformed;
    if (transform_rc < 0) {
        model->gpu_failures++;
        return -1;
    }
    transformed = transform_rc == 0;
    operation->gpu_transformed = transformed;
    if (!transformed)
        for (int token = 0; token < operation->batch; token++) {
            int position = operation->start + token;
            float *query_row = operation->queries +
                (size_t)token * layer->q_dim;
            float *key_row = operation->keys +
                (size_t)token * layer->kv_dim;
            float *value_row = operation->values +
                (size_t)token * layer->kv_dim;
            for (int head = 0; head < plan->n_heads; head++) {
                float *query = query_row + (size_t)head * plan->head_dim;
                if (salt_gemma4_rmsnorm(query, query,
                        layer->q_norm.values, plan->head_dim,
                        G4_EPS, 1) != 0 ||
                    g4_apply_text_rope(model, layer, query, position) != 0)
                    return -1;
            }
            for (int head = 0; head < plan->n_kv_heads; head++) {
                float *key = key_row + (size_t)head * plan->head_dim;
                float *value = value_row + (size_t)head * plan->head_dim;
                if (salt_gemma4_rmsnorm(key, key,
                        layer->k_norm.values, plan->head_dim,
                        G4_EPS, 1) != 0 ||
                    g4_apply_text_rope(model, layer, key, position) != 0 ||
                    salt_gemma4_rmsnorm(value, value, NULL,
                        plan->head_dim, G4_EPS, 0) != 0)
                    return -1;
            }
            if (!layer->kv_ring) {
                float *key_destination = g4_private_key_row(layer, position);
                float *value_destination =
                    g4_private_value_row(layer, position);
                if (!key_destination || !value_destination) return -1;
                memcpy(key_destination, key_row,
                       (size_t)layer->kv_dim * sizeof(float));
                memcpy(value_destination, value_row,
                       (size_t)layer->kv_dim * sizeof(float));
            }
        }
    return 0;
}

static int g4_prefill_attention_batch_body(
        const G4PrefillAttentionBatch *operation) {
    SaltGemma4Text *model = operation->model;
    G4Layer *layer = operation->layer;
    const SaltAttentionDesc *plan = operation->plan;
    SaltAttentionBatchJob attention_job;
    memset(&attention_job, 0, sizeof attention_job);
    attention_job.attention = *plan;
    attention_job.queries = operation->queries;
    if (layer->kv_ring) {
        attention_job.private_keys = operation->history_keys;
        attention_job.private_values = operation->history_values;
        attention_job.start_position = operation->history_rows;
    } else {
        attention_job.private_keys = layer->key_cache;
        attention_job.private_values = layer->value_cache;
        attention_job.shared_keys = layer->shared_key_cache;
        attention_job.shared_values = layer->shared_value_cache;
        attention_job.shared_tokens = layer->shared_position;
        attention_job.start_position = operation->start;
    }
    attention_job.batch = operation->batch;
    attention_job.query_stride = layer->q_dim;
    attention_job.kv_stride = layer->kv_dim;
    attention_job.outputs = operation->attention_outputs;
    attention_job.pool = &model->compute_pool;
    int gpu_rc = g4_gpu_attention_body_rows(
        model, layer, plan, operation->start, operation->batch,
        operation->queries, operation->attention_outputs,
        operation->query_count);
    if (gpu_rc < 0) {
        model->gpu_failures++;
        return -1;
    }
    if (gpu_rc == 0) return 0;
    return salt_attention_batch_run(&attention_job, model->expert_workers);
}

static int g4_prefill_attention_batch_commit(
        const G4PrefillAttentionBatch *operation) {
    G4Layer *layer = operation->layer;
    size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
    if (!layer->kv_ring) return 0;
    /* GPU transform has already published the normalized/RoPE K/V into the
     * canonical shared ring. Staging keys/values still hold raw projections. */
    if (operation->gpu_transformed && salt_gpu_cuda_present()) return 0;
    for (int token = 0; token < operation->batch; token++) {
        int position = operation->start + token;
        float *key = g4_private_key_row(layer, position);
        float *value = g4_private_value_row(layer, position);
        if (!key || !value) return -1;
        memcpy(key, operation->keys + (size_t)token * layer->kv_dim,
               row_bytes);
        memcpy(value, operation->values + (size_t)token * layer->kv_dim,
               row_bytes);
    }
    return 0;
}

static int g4_prefill_attention_batch_output(
        const G4PrefillAttentionBatch *operation) {
    if (operation->model->full_gpu_intent &&
        operation->model->gpu_trunk_exact_views && operation->batch > 1)
        return q_matvec_batch_exact_views(operation->model,
            &operation->layer->o_proj, operation->batch,
            operation->attention_outputs, operation->branches);
    return q_matvec_batch(operation->model, &operation->layer->o_proj,
        operation->batch, operation->attention_outputs, operation->branches);
}

static int g4_prefill_attention_batch_residual(
        const G4PrefillAttentionBatch *operation) {
    for (int token = 0; token < operation->batch; token++)
        if (salt_gemma4_residual_postnorm(
                operation->outputs + (size_t)token * G4_HIDDEN,
                operation->states + (size_t)token * G4_HIDDEN,
                operation->branches + (size_t)token * G4_HIDDEN,
                operation->layer->post_attention_norm.values,
                G4_HIDDEN, G4_EPS) != 0)
            return -1;
    return 0;
}

static int prefill_attention_layer_batch(
        SaltGemma4Text *model, G4Layer *layer,
        const float *states, int start, int batch, float *outputs) {
    G4PrefillAttentionBatch operation;
    int rc = -1;
    int profile;
    double mark, cleanup_mark;
    if (!model || !layer || !layer->plan || !states || !outputs ||
        start < 0 || batch < 1 || start > model->max_context - batch ||
        batch > model->prefill_attention_capacity)
        return -1;
    profile = model->prefill_detail.enabled;
    mark = profile ? g4_now_s() : 0.0;
    memset(&operation, 0, sizeof operation);
    operation.model = model;
    operation.layer = layer;
    operation.states = states;
    operation.outputs = outputs;
    operation.start = start;
    operation.batch = batch;
    if (g4_prefill_attention_batch_prepare(&operation) != 0)
        return -1;
    if (profile) {
        model->prefill_detail.attention_alloc_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_attention_batch_history(&operation) != 0 ||
        g4_prefill_attention_batch_norm(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.attention_norm_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_attention_batch_project(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.attention_qkv_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_attention_batch_transform(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.attention_transform_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_attention_batch_body(&operation) != 0 ||
        g4_prefill_attention_batch_commit(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.attention_body_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_attention_batch_output(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.attention_o_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_attention_batch_residual(&operation) != 0)
        goto done;
    if (profile)
        model->prefill_detail.attention_residual_s += g4_now_s() - mark;
    rc = 0;

done:
    cleanup_mark = profile ? g4_now_s() : 0.0;
    if (profile)
        model->prefill_detail.attention_cleanup_s +=
            g4_now_s() - cleanup_mark;
    return rc;
}

typedef struct {
    const unsigned char *slot;
    const float *input;
    float *gate;
    float *up;
    float *output;
    int detail_enabled;
    double gate_s;
    double up_s;
    double activation_s;
    double down_s;
} ExpertTask;

static void *expert_worker(void *opaque) {
    ExpertTask *task = (ExpertTask *)opaque;
    const unsigned char *gate = task->slot;
    const unsigned char *up = task->slot + G4_PROJ_BYTES;
    const unsigned char *down = task->slot + 2 * G4_PROJ_BYTES;
    double mark = task->detail_enabled ? g4_now_s() : 0.0;
    int prior_expert = salt_kernels_in_expert();
    salt_kernels_set_in_expert(1);
    salt_q4_matvec((const uint32_t *)gate,
        (const uint16_t *)(gate + G4_WEIGHT_BYTES),
        (const uint16_t *)(gate + G4_BIAS_OFFSET),
        G4_ROUTED, G4_HIDDEN, task->input, task->gate);
    if (task->detail_enabled) {
        double now = g4_now_s();
        task->gate_s = now - mark;
        mark = now;
    }
    salt_q4_matvec((const uint32_t *)up,
        (const uint16_t *)(up + G4_WEIGHT_BYTES),
        (const uint16_t *)(up + G4_BIAS_OFFSET),
        G4_ROUTED, G4_HIDDEN, task->input, task->up);
    if (task->detail_enabled) {
        double now = g4_now_s();
        task->up_s = now - mark;
        mark = now;
    }
    for (int i = 0; i < G4_ROUTED; i++)
        task->gate[i] = salt_gemma4_gelu_tanh(task->gate[i]) * task->up[i];
    if (task->detail_enabled) {
        double now = g4_now_s();
        task->activation_s = now - mark;
        mark = now;
    }
    salt_q4_matvec((const uint32_t *)down,
        (const uint16_t *)(down + G4_WEIGHT_BYTES),
        (const uint16_t *)(down + G4_BIAS_OFFSET),
        G4_HIDDEN, G4_ROUTED, task->gate, task->output);
    if (task->detail_enabled)
        task->down_s = g4_now_s() - mark;
    salt_kernels_set_in_expert(prior_expert);
    return NULL;
}

typedef struct G4ExpertMatrixFlow {
    SaltGemma4Text *model;
    ExpertTask *tasks;
    int workers;
} G4ExpertMatrixFlow;

static int g4_cpu_expert_matrix_flow(void *opaque) {
    G4ExpertMatrixFlow *flow = (G4ExpertMatrixFlow *)opaque;
    SaltGemma4Text *model;
    ExpertTask *tasks;
    SaltBatchJob gate_up[2 * G4_TOPK];
    SaltBatchJob down[G4_TOPK];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    int tile_rounds;
    int detail_enabled;
    double detail_total_mark;
    double detail_mark;
    double detail_now;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (!flow || !(model = flow->model) || !(tasks = flow->tasks) ||
        flow->workers < 1)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    detail_enabled = tasks[0].detail_enabled;
    detail_total_mark = detail_enabled ? g4_now_s() : 0.0;
    detail_mark = detail_total_mark;
    /* One claim covers at most about 64 KiB of a compressed projection.
     * This is model geometry, not a second scheduler or arithmetic path. */
    tile_rounds = (int)((G4_PROJ_BYTES + 65535u) / 65536u);
    if (tile_rounds < flow->workers) tile_rounds = flow->workers;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    for (int i = 0; i < G4_TOPK; i++) {
        const unsigned char *gate = tasks[i].slot;
        const unsigned char *up;
        if (!gate || !tasks[i].input || !tasks[i].gate ||
            !tasks[i].up || !tasks[i].output)
            return -1;
        up = gate + G4_PROJ_BYTES;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        gate_up[2 * i] = (SaltBatchJob) {
            (const uint32_t *)(const void *)gate,
            (const uint16_t *)(const void *)(gate + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(gate + G4_BIAS_OFFSET),
            G4_ROUTED, G4_HIDDEN, 1, tasks[i].input, tasks[i].gate,
            0, G4_ROUTED,
        };
        gate_up[2 * i + 1] = (SaltBatchJob) {
            (const uint32_t *)(const void *)up,
            (const uint16_t *)(const void *)(up + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(up + G4_BIAS_OFFSET),
            G4_ROUTED, G4_HIDDEN, 1, tasks[i].input, tasks[i].up,
            0, G4_ROUTED,
        };
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
        /* Gates lead the round-robin frontier so the first worker wave spans
         * every selected mapping instead of pinning two projections of one
         * expert to each worker. Per-row arithmetic and destinations do not
         * change. */
        gate_up[i] = (SaltBatchJob) {
            (const uint32_t *)(const void *)gate,
            (const uint16_t *)(const void *)(gate + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(gate + G4_BIAS_OFFSET),
            G4_ROUTED, G4_HIDDEN, 1, tasks[i].input, tasks[i].gate,
            0, G4_ROUTED,
        };
        gate_up[G4_TOPK + i] = (SaltBatchJob) {
            (const uint32_t *)(const void *)up,
            (const uint16_t *)(const void *)(up + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(up + G4_BIAS_OFFSET),
            G4_ROUTED, G4_HIDDEN, 1, tasks[i].input, tasks[i].up,
            0, G4_ROUTED,
        };
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (detail_enabled) {
        detail_now = g4_now_s();
        model->decode_detail.ffn_matrix_gate_up_setup_s +=
            detail_now - detail_mark;
        detail_mark = detail_now;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (g4_q4_pool_multi_batch_workers(
            model, gate_up, 2 * G4_TOPK, flow->workers) != 0)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (g4_q4_pool_multi_batch_tiles(
            model, gate_up, 2 * G4_TOPK,
            flow->workers, tile_rounds) != 0)
        return -1;
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (detail_enabled) {
        detail_now = g4_now_s();
        model->decode_detail.ffn_matrix_gate_up_s +=
            detail_now - detail_mark;
        detail_mark = detail_now;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->expert_matrix_gate_up_waves++;
    model->expert_matrix_projection_jobs += 2u * G4_TOPK;
    for (int i = 0; i < G4_TOPK; i++)
        for (int d = 0; d < G4_ROUTED; d++)
            tasks[i].gate[d] =
                salt_gemma4_gelu_tanh(tasks[i].gate[d]) * tasks[i].up[d];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (detail_enabled) {
        detail_now = g4_now_s();
        model->decode_detail.ffn_matrix_activation_s +=
            detail_now - detail_mark;
        detail_mark = detail_now;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    for (int i = 0; i < G4_TOPK; i++) {
        const unsigned char *projection =
            tasks[i].slot + 2 * G4_PROJ_BYTES;
        down[i] = (SaltBatchJob) {
            (const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(
                projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(
                projection + G4_BIAS_OFFSET),
            G4_HIDDEN, G4_ROUTED, 1, tasks[i].gate, tasks[i].output,
            0, G4_HIDDEN,
        };
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (detail_enabled) {
        detail_now = g4_now_s();
        model->decode_detail.ffn_matrix_down_setup_s +=
            detail_now - detail_mark;
        detail_mark = detail_now;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (g4_q4_pool_multi_batch_workers(
            model, down, G4_TOPK, flow->workers) != 0)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (g4_q4_pool_multi_batch_tiles(
            model, down, G4_TOPK,
            flow->workers, tile_rounds) != 0)
        return -1;
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (detail_enabled) {
        detail_now = g4_now_s();
        model->decode_detail.ffn_matrix_down_s += detail_now - detail_mark;
        model->decode_detail.ffn_matrix_total_s +=
            detail_now - detail_total_mark;
        model->decode_detail.ffn_matrix_flows++;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->expert_matrix_down_waves++;
    model->expert_matrix_projection_jobs += G4_TOPK;
    return 0;
}

static int g4_cpu_expert_matrix_waves(
        SaltGemma4Text *model, ExpertTask tasks[G4_TOPK]) {
    G4ExpertMatrixFlow flow;
    if (!model || !tasks || !model->compute_pool_ready ||
        model->compute_pool.aq4_threads < 1)
        return -1;
    flow.model = model;
    flow.tasks = tasks;
    flow.workers = model->compute_pool.aq4_threads;
    if (flow.workers > model->compute_pool.apool_threads) return -1;
    return model->compute_pool.aflow_running
        ? g4_cpu_expert_matrix_flow(&flow)
        : salt_attn_pool_flow_run(
            &model->compute_pool, g4_cpu_expert_matrix_flow, &flow);
}

typedef struct {
    ExpertTask *tasks;
    int first;
    int count;
    int workers;
} G4ExpertPoolWave;

static void g4_expert_detail_wave(
        G4DecodeDetail *detail, ExpertTask *tasks,
        int first, int last, int pooled, double wall_s) {
    double critical_s = 0.0, minimum_s = 0.0, work_s = 0.0;
    if (!detail || !detail->ffn_detail_enabled || !tasks ||
        first < 0 || last <= first)
        return;
    for (int i = first; i < last; i++) {
        double task_s = tasks[i].gate_s + tasks[i].up_s +
            tasks[i].activation_s + tasks[i].down_s;
        detail->ffn_expert_gate_work_s += tasks[i].gate_s;
        detail->ffn_expert_up_work_s += tasks[i].up_s;
        detail->ffn_expert_activation_work_s += tasks[i].activation_s;
        detail->ffn_expert_down_work_s += tasks[i].down_s;
        work_s += task_s;
        if (i == first || task_s < minimum_s) minimum_s = task_s;
        if (task_s > critical_s) critical_s = task_s;
    }
    detail->ffn_detail_tasks += (uint64_t)(last - first);
    detail->ffn_detail_waves++;
    if (!pooled) critical_s = work_s;
    detail->ffn_expert_worker_critical_s += critical_s;
    if (pooled)
        detail->ffn_expert_worker_skew_s += critical_s - minimum_s;
    if (wall_s > critical_s)
        detail->ffn_expert_pool_overhead_s += wall_s - critical_s;
}

static void route_observe(SaltGemma4Text *model, uint32_t kind,
                          uint64_t position, int layer,
                          const int selected[G4_TOPK]) {
    SaltGemma4RouteObserver *observer;
    if (!model || !(observer = model->route_observer) || !selected) return;
    if (observer->route_kind != 0 && observer->route_kind != kind) return;
    for (int i = 0; i < G4_TOPK; i++) {
        SaltGemma4RouteObservation *item;
        if (observer->count >= observer->capacity) {
            observer->overflow = 1;
            return;
        }
        item = &observer->items[observer->count++];
        item->kind = kind;
        item->position = position;
        item->layer = (uint32_t)layer;
        item->expert = (uint32_t)selected[i];
    }
}

static void g4_expert_pool_worker(int worker, void *opaque) {
    G4ExpertPoolWave *wave = (G4ExpertPoolWave *)opaque;
    if (!wave || !wave->tasks || wave->workers < 1 ||
        worker < 0 || worker >= wave->workers)
        return;
    for (int item = worker; item < wave->count; item += wave->workers)
        (void)expert_worker(&wave->tasks[wave->first + item]);
}

static int g4_gpu_bounded_layer_moe(SaltGemma4Text *model, int layer,
                                    const int *experts, const int *groups,
                                    int count, const float *inputs,
                                    float *outputs, float *gate_scratch,
                                    float *up_scratch);

static int routed_experts(SaltGemma4Text *model, int layer_index,
                          const float *input, int *selected, float *weights,
                          float *output) {
    ExpertTask tasks[G4_TOPK];
    int cache_slots[G4_TOPK];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    const uint8_t *payloads[G4_TOPK];
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    int acquired = 0;
    int release_failed = 0;
    int compute_failed = 0;
    int profile = model && model->decode_detail.enabled;
    int detail = model && model->decode_detail.ffn_detail_enabled &&
        !model->full_gpu_intent && !model->nvfp4_mode;
    double mark = profile ? g4_now_s() : 0.0;
    for (int i = 0; i < G4_TOPK; i++) {
        for (int j = i + 1; j < G4_TOPK; j++) {
            if (selected[j] < selected[i]) {
                int si = selected[i]; selected[i] = selected[j]; selected[j] = si;
                float sw = weights[i]; weights[i] = weights[j]; weights[j] = sw;
            }
        }
    }
    route_observe(model, SALT_GEMMA4_ROUTE_DECODE,
                  (uint64_t)model->position, layer_index, selected);
    if (model->attention_plan_lease &&
            model->attention_plan_lease->active) {
        SaltGemma4AttentionPlanLease *lease = model->attention_plan_lease;
        if (lease->exact_routes <= UINT32_MAX - G4_TOPK)
            lease->exact_routes += G4_TOPK;
        else
            lease->exact_routes = UINT32_MAX;
        for (int route = 0; route < G4_TOPK; route++)
            for (uint32_t item = 0; item < lease->count; item++)
                if (lease->entries[item].layer == (uint32_t)layer_index &&
                    lease->entries[item].expert == (uint32_t)selected[route]) {
                    uint64_t bit = UINT64_C(1) << item;
                    if (!(lease->used_mask & bit)) {
                        lease->used_mask |= bit;
                        if (lease->ready_matches != UINT32_MAX)
                            lease->ready_matches++;
                    }
                    break;
                }
    }
    if (model->nvfp4_mode) {
        memset(output, 0, (size_t)G4_HIDDEN * sizeof(float));
        if (model->full_gpu_intent && model->nvfp4_decode_aggregate) {
            for (int i = 0; i < G4_TOPK; i++) {
                G4Q *base = &model->nvfp4_experts[
                    ((size_t)layer_index * G4_EXPERTS +
                     (size_t)selected[i]) * 3u];
                model->prefill_gpu_ids[2 * i] = base[0].weight;
                model->prefill_gpu_ids[2 * i + 1] = base[1].weight;
                model->prefill_gpu_xs[2 * i] = input;
                model->prefill_gpu_xs[2 * i + 1] = input;
                model->prefill_gpu_ys[2 * i] =
                    model->expert_gate + (size_t)i * G4_ROUTED;
                model->prefill_gpu_ys[2 * i + 1] =
                    model->expert_up + (size_t)i * G4_ROUTED;
                model->prefill_gpu_rows[2 * i] = 1;
                model->prefill_gpu_rows[2 * i + 1] = 1;
            }
            if (salt_gpu_nvfp4_batch(model->prefill_gpu_ids,
                    model->prefill_gpu_xs, model->prefill_gpu_ys,
                    model->prefill_gpu_rows, G4_ROUTED, G4_HIDDEN,
                    2 * G4_TOPK) != 0) {
                model->gpu_failures++;
                return -1;
            }
            model->gpu_dense_submissions += 2u * G4_TOPK;
            model->gpu_expert_gate_up_submissions += 2u * G4_TOPK;
            model->gpu_expert_gate_up_commands++;
            model->gpu_decode_expert_gate_up_commands++;
            model->gpu_decode_submissions += 2u * G4_TOPK;
            for (int i = 0; i < G4_TOPK; i++) {
                float *gate = model->expert_gate + (size_t)i * G4_ROUTED;
                float *up = model->expert_up + (size_t)i * G4_ROUTED;
                for (int row = 0; row < G4_ROUTED; row++)
                    gate[row] = salt_gemma4_gelu_tanh(gate[row]) * up[row];
                G4Q *base = &model->nvfp4_experts[
                    ((size_t)layer_index * G4_EXPERTS +
                     (size_t)selected[i]) * 3u];
                model->prefill_gpu_ids[i] = base[2].weight;
                model->prefill_gpu_xs[i] = gate;
                model->prefill_gpu_ys[i] =
                    model->expert_outputs + (size_t)i * G4_HIDDEN;
                model->prefill_gpu_rows[i] = 1;
            }
            if (salt_gpu_nvfp4_batch(model->prefill_gpu_ids,
                    model->prefill_gpu_xs, model->prefill_gpu_ys,
                    model->prefill_gpu_rows, G4_HIDDEN, G4_ROUTED,
                    G4_TOPK) != 0) {
                model->gpu_failures++;
                return -1;
            }
            model->gpu_dense_submissions += G4_TOPK;
            model->gpu_expert_down_submissions += G4_TOPK;
            model->gpu_expert_down_commands++;
            model->gpu_decode_expert_down_commands++;
            model->gpu_decode_submissions += G4_TOPK;
            for (int i = 0; i < G4_TOPK; i++) {
                const float *expert_output =
                    model->expert_outputs + (size_t)i * G4_HIDDEN;
                for (int row = 0; row < G4_HIDDEN; row++)
                    output[row] += weights[i] * expert_output[row];
            }
            return 0;
        }
        for (int i = 0; i < G4_TOPK; i++) {
            G4Q *base = &model->nvfp4_experts[
                ((size_t)layer_index * G4_EXPERTS +
                 (size_t)selected[i]) * 3u];
            float *gate = model->expert_gate + (size_t)i * G4_ROUTED;
            float *up = model->expert_up + (size_t)i * G4_ROUTED;
            float *expert_output =
                model->expert_outputs + (size_t)i * G4_HIDDEN;
            if (q_matvec_operation(model, &base[0], input, gate) != 0 ||
                q_matvec_operation(model, &base[1], input, up) != 0)
                return -1;
            for (int row = 0; row < G4_ROUTED; row++)
                gate[row] = salt_gemma4_gelu_tanh(gate[row]) * up[row];
            if (q_matvec_operation(model, &base[2], gate, expert_output) != 0)
                return -1;
            for (int row = 0; row < G4_HIDDEN; row++)
                output[row] += weights[i] * expert_output[row];
        }
        return 0;
    }
    if (model->full_gpu_intent && model->gpu_bounded_weights &&
        model->gpu_expert_layer_view) {
        int groups[G4_TOPK];
        for (int i = 0; i < G4_TOPK; i++) {
            groups[i] = 1;
            memcpy(model->prefill_routed_inputs + (size_t)i * G4_HIDDEN,
                   input, G4_HIDDEN * sizeof(float));
        }
        if (profile) mark = g4_now_s();
        if (g4_gpu_bounded_layer_moe(model, layer_index,
                selected, groups, G4_TOPK, model->prefill_routed_inputs,
                model->expert_outputs, model->prefill_group_gate,
                model->prefill_group_up) != 0)
            return -1;
        model->gpu_expert_gate_up_submissions += 2u * G4_TOPK;
        model->gpu_expert_down_submissions += G4_TOPK;
        model->gpu_expert_gate_up_commands++;
        model->gpu_expert_down_commands++;
        model->gpu_decode_expert_gate_up_commands++;
        model->gpu_decode_expert_down_commands++;
        model->gpu_decode_submissions += 3u * G4_TOPK;
        if (profile)
            model->decode_detail.ffn_expert_compute_s += g4_now_s() - mark;
        memset(output, 0, G4_HIDDEN * sizeof(float));
        for (int i = 0; i < G4_TOPK; i++)
            for (int d = 0; d < G4_HIDDEN; d++)
                output[d] += model->expert_outputs[
                    (size_t)i * G4_HIDDEN + d] * weights[i];
        return 0;
    }
    if (!model->expert_cache_set ||
        salt_cache_getmany(&model->expert_cache, layer_index, selected,
                           G4_TOPK, cache_slots) < 0)
        return -1;
    update_expert_cache_peak(model);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    for (int i = 0; i < G4_TOPK; i++) {
        tasks[i].slot = salt_cache_acquire(&model->expert_cache,
            cache_slots[i], layer_index, selected[i]);
        if (!tasks[i].slot) {
            while (acquired > 0) {
                acquired--;
                salt_cache_release(&model->expert_cache,
                    cache_slots[acquired], layer_index, selected[acquired]);
            }
            return -1;
        }
        acquired++;
        tasks[i].input = input;
        tasks[i].gate = model->expert_gate + (size_t)i * G4_ROUTED;
        tasks[i].up = model->expert_up + (size_t)i * G4_ROUTED;
        tasks[i].output = model->expert_outputs + (size_t)i * G4_HIDDEN;
        tasks[i].detail_enabled = detail;
        if (detail) {
            tasks[i].gate_s = 0.0;
            tasks[i].up_s = 0.0;
            tasks[i].activation_s = 0.0;
            tasks[i].down_s = 0.0;
        }
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (salt_cache_acquire_many(&model->expert_cache, cache_slots,
            layer_index, selected, G4_TOPK, payloads) != 0)
        return -1;
    acquired = G4_TOPK;
    for (int i = 0; i < G4_TOPK; i++) {
        tasks[i].slot = payloads[i];
        if (!tasks[i].slot) {
            (void)salt_cache_release_many(&model->expert_cache, cache_slots,
                layer_index, selected, G4_TOPK);
            return -1;
        }
        tasks[i].input = input;
        tasks[i].gate = model->expert_gate + (size_t)i * G4_ROUTED;
        tasks[i].up = model->expert_up + (size_t)i * G4_ROUTED;
        tasks[i].output = model->expert_outputs + (size_t)i * G4_HIDDEN;
        tasks[i].detail_enabled = detail;
        if (detail) {
            tasks[i].gate_s = 0.0;
            tasks[i].up_s = 0.0;
            tasks[i].activation_s = 0.0;
            tasks[i].down_s = 0.0;
        }
    }
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    /* Each selected mapping is now leased independently of the sequential
     * model API. Victim selection cannot recycle it until every persistent
     * worker wave completes and the matching lease is released below. */
    model->expert_inflight_slots = G4_TOPK;
    if (model->expert_inflight_slots > model->expert_peak_inflight_slots)
        model->expert_peak_inflight_slots = model->expert_inflight_slots;
    if (profile) {
        model->decode_detail.ffn_expert_fetch_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (model->full_gpu_intent) {
        if (model->gpu_bounded_weights) {
            for (int i = 0; i < G4_TOPK; i++) {
                const unsigned char *slot = tasks[i].slot;
                model->prefill_gpu_moe[i] = (SaltGpuMoeExpert) {
                    (const uint32_t *)(const void *)slot,
                    (const uint16_t *)(const void *)(slot + G4_WEIGHT_BYTES),
                    (const uint16_t *)(const void *)(slot + G4_BIAS_OFFSET),
                    slot,
                    (const uint32_t *)(const void *)(slot + G4_PROJ_BYTES),
                    (const uint16_t *)(const void *)(
                        slot + G4_PROJ_BYTES + G4_WEIGHT_BYTES),
                    (const uint16_t *)(const void *)(
                        slot + G4_PROJ_BYTES + G4_BIAS_OFFSET),
                    slot + G4_PROJ_BYTES,
                    (const uint32_t *)(const void *)(slot + 2 * G4_PROJ_BYTES),
                    (const uint16_t *)(const void *)(
                        slot + 2 * G4_PROJ_BYTES + G4_WEIGHT_BYTES),
                    (const uint16_t *)(const void *)(
                        slot + 2 * G4_PROJ_BYTES + G4_BIAS_OFFSET),
                    slot + 2 * G4_PROJ_BYTES,
                    1,
                    cache_slots[i],
                    (uint64_t)(uint32_t)layer_index * G4_EXPERTS +
                        (uint32_t)selected[i],
                };
                memcpy(model->prefill_routed_inputs + (size_t)i * G4_HIDDEN,
                       input, G4_HIDDEN * sizeof(float));
                tasks[i].output = model->prefill_routed_outputs +
                    (size_t)i * G4_HIDDEN;
            }
            int gpu_rc = salt_gpu_q4_moe_chain_selected(
                model->prefill_gpu_moe, G4_TOPK, G4_HIDDEN, G4_ROUTED,
                model->prefill_routed_inputs, model->prefill_routed_outputs,
                model->prefill_group_gate, model->prefill_group_up,
                (size_t)G4_TOPK * G4_ROUTED);
            if (gpu_rc != 0)
                compute_failed = 1;
            if (!compute_failed) {
                model->gpu_expert_gate_up_submissions += 2u * G4_TOPK;
                model->gpu_expert_down_submissions += G4_TOPK;
                model->gpu_expert_gate_up_commands++;
                model->gpu_expert_down_commands++;
                model->gpu_decode_expert_gate_up_commands++;
                model->gpu_decode_expert_down_commands++;
                model->gpu_decode_submissions += 3u * G4_TOPK;
            }
        } else {
        SaltBatchJob gate_up[2 * G4_TOPK];
        SaltBatchJob down[G4_TOPK];
        int expanded = 0;
        for (int i = 0; i < G4_TOPK; i++) {
            const unsigned char *gate = tasks[i].slot;
            const unsigned char *up = gate + G4_PROJ_BYTES;
            gate_up[2 * i] = (SaltBatchJob) {
                (const uint32_t *)(const void *)gate,
                (const uint16_t *)(const void *)(gate + G4_WEIGHT_BYTES),
                (const uint16_t *)(const void *)(gate + G4_BIAS_OFFSET),
                G4_ROUTED, G4_HIDDEN, 1, input, tasks[i].gate,
                0, G4_ROUTED,
            };
            gate_up[2 * i + 1] = (SaltBatchJob) {
                (const uint32_t *)(const void *)up,
                (const uint16_t *)(const void *)(up + G4_WEIGHT_BYTES),
                (const uint16_t *)(const void *)(up + G4_BIAS_OFFSET),
                G4_ROUTED, G4_HIDDEN, 1, input, tasks[i].up,
                0, G4_ROUTED,
            };
        }
        if (g4_gpu_multi_batch(model, gate_up, 2 * G4_TOPK,
                               &expanded) != 0) {
            compute_failed = 1;
        } else {
            model->gpu_expert_gate_up_submissions += (uint64_t)expanded;
            model->gpu_expert_gate_up_commands++;
            model->gpu_decode_expert_gate_up_commands++;
            model->gpu_decode_submissions += (uint64_t)expanded;
        }
        if (!compute_failed) {
            for (int i = 0; i < G4_TOPK; i++)
                for (int d = 0; d < G4_ROUTED; d++)
                    tasks[i].gate[d] =
                        salt_gemma4_gelu_tanh(tasks[i].gate[d]) *
                        tasks[i].up[d];
            for (int i = 0; i < G4_TOPK; i++) {
                const unsigned char *projection =
                    tasks[i].slot + 2 * G4_PROJ_BYTES;
                down[i] = (SaltBatchJob) {
                    (const uint32_t *)(const void *)projection,
                    (const uint16_t *)(const void *)(
                        projection + G4_WEIGHT_BYTES),
                    (const uint16_t *)(const void *)(
                        projection + G4_BIAS_OFFSET),
                    G4_HIDDEN, G4_ROUTED, 1, tasks[i].gate,
                    tasks[i].output, 0, G4_HIDDEN,
                };
            }
            if (g4_gpu_multi_batch(model, down, G4_TOPK, &expanded) != 0) {
                compute_failed = 1;
            } else {
                model->gpu_expert_down_submissions += (uint64_t)expanded;
                model->gpu_expert_down_commands++;
                model->gpu_decode_expert_down_commands++;
                model->gpu_decode_submissions += (uint64_t)expanded;
            }
        }
        }
    } else if (model->expert_matrix_waves) {
        if (g4_cpu_expert_matrix_waves(model, tasks) != 0)
            compute_failed = 1;
    } else {
        int wave_width = model->compute_pool_ready
            ? model->compute_pool.aq4_threads : 0;
        int action_width;
        if (wave_width > G4_TOPK) wave_width = G4_TOPK;
        if (wave_width < 1) wave_width = 1;
        action_width = model->compute_pool.aflow_running
            ? G4_TOPK : wave_width;
        for (int first = 0; first < G4_TOPK; first += action_width) {
            G4ExpertPoolWave wave;
            int pooled = 0;
            double wave_start = detail ? g4_now_s() : 0.0;
            int last = first + action_width;
            if (last > G4_TOPK) last = G4_TOPK;
            wave.tasks = tasks;
            wave.first = first;
            wave.count = last - first;
            wave.workers = wave.count < wave_width ? wave.count : wave_width;
            if (!model->compute_pool_ready ||
                salt_attn_pool_run_n(&model->compute_pool, wave.workers,
                                     g4_expert_pool_worker, &wave) != 0) {
                model->expert_pool_fallbacks++;
                for (int i = first; i < last; i++)
                    (void)expert_worker(&tasks[i]);
            } else {
                model->expert_pool_submissions++;
                pooled = 1;
            }
            if (detail)
                g4_expert_detail_wave(&model->decode_detail, tasks,
                    first, last, pooled, g4_now_s() - wave_start);
        }
    }
    if (profile) {
        model->decode_detail.ffn_expert_compute_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    for (int i = 0; i < acquired; i++)
        if (salt_cache_release(&model->expert_cache, cache_slots[i],
                               layer_index, selected[i]) != 0)
            release_failed = 1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (acquired > 0 && salt_cache_release_many(&model->expert_cache,
            cache_slots, layer_index, selected, acquired) != 0)
        release_failed = 1;
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->expert_inflight_slots = 0;
    if (profile)
        model->decode_detail.ffn_expert_release_s += g4_now_s() - mark;
    if (release_failed || compute_failed) return -1;
    memset(output, 0, G4_HIDDEN * sizeof(float));
    for (int i = 0; i < G4_TOPK; i++)
        for (int d = 0; d < G4_HIDDEN; d++)
            output[d] += tasks[i].output[d] * weights[i];
    return 0;
}

static int feed_forward(SaltGemma4Text *model, G4Layer *layer,
                        int layer_index, const float *residual, float *output,
                        int ordinary_decode) {
    int profile = model && model->decode_detail.enabled;
    double mark = profile ? g4_now_s() : 0.0;
    int rc;
    if (salt_gemma4_rmsnorm(model->norm_a, residual,
            layer->pre_ffn_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
        return step_fail("pre-ffn-norm", layer_index, model->position);
    {
        int chain = model->gpu_decode_dense_chain_enabled
            ? g4_gpu_dense_q4_chain(model, layer, 1, model->norm_a,
                model->dense_output, model->dense_gate, model->dense_up)
            : 1;
        if (chain < 0)
            return step_fail("dense-gpu-chain", layer_index, model->position);
        if (chain > 0) {
            if (dense_gate_up_batch(model, layer, 1, model->norm_a,
                    model->dense_gate, model->dense_up) != 0)
                return step_fail("dense-gate-up", layer_index,
                                 model->position);
            for (int i = 0; i < G4_DENSE; i++)
                model->dense_gate[i] =
                    salt_gemma4_gelu_tanh(model->dense_gate[i]) *
                    model->dense_up[i];
            if (q_matvec_operation(model, &layer->dense_down,
                    model->dense_gate, model->dense_output) != 0)
                return step_fail("dense-down", layer_index,
                                 model->position);
        }
    }
    if (profile) {
        model->decode_detail.ffn_dense_s += g4_now_s() - mark;
        mark = g4_now_s();
    }

    if (salt_gemma4_router_input(model->norm_a, residual,
            layer->router_scale.values, G4_HIDDEN, G4_EPS) != 0)
        return step_fail("router-input", layer_index, model->position);
    /* The small ordinary Metal router uses the existing exact CPU workers.
     * Batch/PREFILL, proof callers and other backends retain their dispatch. */
    rc = ordinary_decode && model->full_gpu_intent &&
            !salt_gpu_cuda_present() && !salt_gpu_rocm_present() &&
            layer->router.bits == 8
        ? g4_q8_pool_batch(model, &layer->router, 1,
            model->norm_a, model->router_logits)
        : q_matvec_operation(model, &layer->router,
            model->norm_a, model->router_logits);
    if (rc != 0)
        return step_fail("router-projection", layer_index, model->position);
    int selected[G4_TOPK];
    float weights[G4_TOPK];
    if (salt_gemma4_router_topk(model->router_logits,
            layer->per_expert_scale.values, G4_EXPERTS, G4_TOPK,
            selected, weights) != 0)
        return step_fail("router-topk", layer_index, model->position);
    if (salt_gemma4_rmsnorm(model->norm_b, residual,
            layer->pre_ffn_norm_2.values, G4_HIDDEN, G4_EPS, 1) != 0)
        return step_fail("routed-norm", layer_index, model->position);
    if (profile) {
        model->decode_detail.ffn_route_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (routed_experts(model, layer_index, model->norm_b,
            selected, weights, model->routed_output) != 0)
        return step_fail("routed-experts", layer_index, model->position);
    if (profile) {
        model->decode_detail.ffn_experts_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    rc = salt_gemma4_parallel_ffn_combine(output, residual,
        model->dense_output, model->routed_output,
        layer->post_ffn_norm_1.values, layer->post_ffn_norm_2.values,
        layer->post_ffn_norm.values, layer->layer_scalar.values[0],
        G4_HIDDEN, G4_EPS, model->parallel_scratch);
    if (profile)
        model->decode_detail.ffn_combine_s += g4_now_s() - mark;
    if (rc != 0)
        return step_fail("parallel-combine", layer_index, model->position);
    if (g4_memory_budget_ok(model, "decode-layer", layer_index) != 0)
        return step_fail("decode-memory-limit", layer_index, model->position);
    return 0;
}

static int feed_forward_t4_proof(SaltGemma4Text *model, G4Layer *layer,
                        int layer_index, const float *residual, float *output) {
    int profile = model && model->decode_detail.enabled;
    double mark = profile ? g4_now_s() : 0.0;
    int *selected;
    float *weights;
    int rc;
    if (!model || !model->metal_exact_cells || !model->route_selected ||
        !model->route_weights)
        return -1;
    selected = model->route_selected;
    weights = model->route_weights;
    if (g4_exact_rmsnorm(model, model->norm_a, residual,
            layer->pre_ffn_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
        return step_fail("pre-ffn-norm", layer_index, model->position);
    if (g4_layer_digest(model, "ffn-pre-norm", layer_index,
            model->norm_a, G4_HIDDEN * sizeof(float)) != 0)
        return step_fail("ffn-pre-norm-digest", layer_index, model->position);
    if (dense_gate_up_batch(model, layer, 1, model->norm_a,
            model->dense_gate, model->dense_up) != 0)
        return step_fail("dense-gate-up", layer_index, model->position);
    for (int i = 0; i < G4_DENSE; i++)
        model->dense_gate[i] = salt_gemma4_gelu_tanh(model->dense_gate[i]) *
                               model->dense_up[i];
    if (q_matvec_operation(model, &layer->dense_down,
            model->dense_gate, model->dense_output) != 0)
        return step_fail("dense-down", layer_index, model->position);
    if (profile) {
        model->decode_detail.ffn_dense_s += g4_now_s() - mark;
        mark = g4_now_s();
    }

    if (g4_exact_router_input(model, model->norm_a, residual,
            layer->router_scale.values, G4_HIDDEN, G4_EPS) != 0)
        return step_fail("router-input", layer_index, model->position);
    if (g4_layer_digest(model, "router-input", layer_index,
            model->norm_a, G4_HIDDEN * sizeof(float)) != 0)
        return step_fail("router-input-digest", layer_index, model->position);
    if (q_matvec_operation(model, &layer->router,
            model->norm_a, model->router_logits) != 0)
        return step_fail("router-projection", layer_index, model->position);
    if (g4_exact_topk(model, model->router_logits,
            layer->per_expert_scale.values, selected, weights) != 0)
        return step_fail("router-topk", layer_index, model->position);
    if (g4_layer_digest(model, "router-selected", layer_index,
            selected, G4_TOPK * sizeof(int)) != 0 ||
        g4_layer_digest(model, "router-weights", layer_index,
            weights, G4_TOPK * sizeof(float)) != 0)
        return step_fail("router-topk-digest", layer_index, model->position);
    if (g4_exact_rmsnorm(model, model->norm_b, residual,
            layer->pre_ffn_norm_2.values, G4_HIDDEN, G4_EPS, 1) != 0)
        return step_fail("routed-norm", layer_index, model->position);
    if (g4_layer_digest(model, "routed-norm", layer_index,
            model->norm_b, G4_HIDDEN * sizeof(float)) != 0)
        return step_fail("routed-norm-digest", layer_index, model->position);
    if (profile) {
        model->decode_detail.ffn_route_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (routed_experts(model, layer_index, model->norm_b,
            selected, weights, model->routed_output) != 0)
        return step_fail("routed-experts", layer_index, model->position);
    if (profile) {
        model->decode_detail.ffn_experts_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    rc = g4_exact_combine(model, output, residual,
        model->dense_output, model->routed_output,
        layer->post_ffn_norm_1.values, layer->post_ffn_norm_2.values,
        layer->post_ffn_norm.values, layer->layer_scalar.values[0],
        G4_HIDDEN, G4_EPS, model->parallel_scratch);
    if (profile)
        model->decode_detail.ffn_combine_s += g4_now_s() - mark;
    if (rc != 0)
        return step_fail("parallel-combine", layer_index, model->position);
    if (g4_layer_digest(model, "parallel-combine", layer_index,
            output, G4_HIDDEN * sizeof(float)) != 0)
        return step_fail("parallel-combine-digest", layer_index,
                         model->position);
    if (g4_memory_budget_ok(model, "decode-layer", layer_index) != 0)
        return step_fail("decode-memory-limit", layer_index, model->position);
    return 0;
}

static int batch_expert_projection(SaltGemma4Text *model,
                                   const unsigned char *projection,
                                   int rows, int cols, int batch,
                                   const float *inputs, float *outputs) {
    int prior_scope, rc;
    if (!projection || batch < 1 || !inputs || !outputs) return -1;
    if (model && model->full_gpu_intent) {
        const float *xs[G4_CUDA_BATCH_MAX];
        float *ys[G4_CUDA_BATCH_MAX];
        for (int first = 0; first < batch; first += G4_CUDA_BATCH_MAX) {
            int count = batch - first;
            if (count > G4_CUDA_BATCH_MAX) count = G4_CUDA_BATCH_MAX;
            for (int token = 0; token < count; token++) {
                xs[token] = inputs +
                    (size_t)(first + token) * (size_t)cols;
                ys[token] = outputs +
                    (size_t)(first + token) * (size_t)rows;
            }
            if (salt_gpu_proj_batch(
                    (const uint32_t *)(const void *)projection,
                    (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
                    (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
                    rows, cols, count, xs, ys, projection) != 0) {
                model->gpu_failures++;
                return -1;
            }
        }
        if (rows == G4_ROUTED)
            model->gpu_expert_gate_up_submissions += (uint64_t)batch;
        else
            model->gpu_expert_down_submissions += (uint64_t)batch;
        if (batch == 1) model->gpu_decode_submissions++;
        return 0;
    }
    if (batch == 1) {
        int prior_expert = salt_kernels_in_expert();
        salt_kernels_set_in_expert(1);
        salt_q4_matvec((const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
            rows, cols, inputs, outputs);
        salt_kernels_set_in_expert(prior_expert);
        return 0;
    }
    if (g4_q4_pool_batch(model,
            (const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
            rows, cols, batch, inputs, outputs) == 0)
        return 0;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (model) model->q4_pool_fallbacks++;
    prior_scope = salt_kernels_cpu_expert_scope();
    salt_kernels_set_cpu_expert_scope(1);
    rc = salt_q4_matvec_batch(
        (const uint32_t *)(const void *)projection,
        (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
        (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
        rows, cols, batch, inputs, outputs);
    salt_kernels_set_cpu_expert_scope(prior_scope);
    return rc;
}

static void sort_selection(int *selected, float *weights) {
    for (int i = 0; i < G4_TOPK; i++)
        for (int j = i + 1; j < G4_TOPK; j++)
            if (selected[j] < selected[i]) {
                int id = selected[i]; selected[i] = selected[j];
                selected[j] = id;
                float weight = weights[i]; weights[i] = weights[j];
                weights[j] = weight;
            }
}

typedef struct {
    const unsigned char *bytes;
    int cache_slot;
    uint64_t logical_resource_id;
    int active;
} G4ExpertLease;

typedef struct {
    SaltGemma4Text *model;
    int layer;
    float *gate;
    float *up;
    G4ExpertLease leases[G4_EXPERTS];
} G4GroupContext;

static int g4_gpu_bounded_layer_moe(SaltGemma4Text *model, int layer,
                                    const int *experts, const int *groups,
                                    int count, const float *inputs,
                                    float *outputs, float *gate_scratch,
                                    float *up_scratch) {
    uint64_t layer_bytes = (uint64_t)G4_EXPERTS * G4_SLOT_BYTES;
    uint64_t payload_offset = G4_POOL_HEADER +
        (uint64_t)(uint32_t)layer * layer_bytes;
    uint64_t map_offset, delta, map_bytes, total_rows = 0;
    long page_l;
    void *mapping = MAP_FAILED;
    unsigned char *payload;
    int bound = 0, rc = -1;
    if (!model || !experts || !groups || !inputs || !outputs ||
        !gate_scratch || !up_scratch || layer < 0 || layer >= G4_LAYERS ||
        count < 1 || count > G4_EXPERTS || model->pool.fd < 0 ||
        (page_l = sysconf(_SC_PAGESIZE)) <= 0)
        return -1;
    for (int i = 0; i < count; i++) {
        if (experts[i] < 0 || experts[i] >= G4_EXPERTS || groups[i] < 1 ||
            total_rows > UINT64_MAX - (uint32_t)groups[i])
            return -1;
        total_rows += (uint32_t)groups[i];
    }
    if (total_rows > SIZE_MAX / G4_ROUTED) return -1;
    map_offset = payload_offset - payload_offset % (uint64_t)page_l;
    delta = payload_offset - map_offset;
    if (layer_bytes > UINT64_MAX - delta) return -1;
    map_bytes = delta + layer_bytes;
    if (map_offset > (uint64_t)INT64_MAX || map_bytes > SIZE_MAX ||
        map_offset > model->pool.size ||
        map_bytes > model->pool.size - (size_t)map_offset)
        return -1;
    mapping = mmap(NULL, (size_t)map_bytes, PROT_READ, MAP_SHARED,
                   model->pool.fd, (off_t)map_offset);
    if (mapping == MAP_FAILED) return -1;
    payload = (unsigned char *)mapping + (size_t)delta;
    if (salt_gpu_weight_resource_bind(SALT_GPU_RESOURCE_EXPERT_LAYER, 15,
            mapping, (size_t)map_bytes) != 0)
        goto done;
    bound = 1;
    for (int i = 0; i < count; i++) {
        unsigned char *slot = payload + (size_t)experts[i] * G4_SLOT_BYTES;
        SaltGpuMoeExpert *entry = &model->prefill_gpu_moe[i];
        *entry = (SaltGpuMoeExpert) {
            .gate_vals = (const uint32_t *)(const void *)slot,
            .gate_scales = (const uint16_t *)(const void *)(
                slot + G4_WEIGHT_BYTES),
            .gate_biases = (const uint16_t *)(const void *)(
                slot + G4_BIAS_OFFSET),
            .gate_id = slot,
            .up_vals = (const uint32_t *)(const void *)(slot + G4_PROJ_BYTES),
            .up_scales = (const uint16_t *)(const void *)(
                slot + G4_PROJ_BYTES + G4_WEIGHT_BYTES),
            .up_biases = (const uint16_t *)(const void *)(
                slot + G4_PROJ_BYTES + G4_BIAS_OFFSET),
            .up_id = slot + G4_PROJ_BYTES,
            .down_vals = (const uint32_t *)(const void *)(
                slot + 2 * G4_PROJ_BYTES),
            .down_scales = (const uint16_t *)(const void *)(
                slot + 2 * G4_PROJ_BYTES + G4_WEIGHT_BYTES),
            .down_biases = (const uint16_t *)(const void *)(
                slot + 2 * G4_PROJ_BYTES + G4_BIAS_OFFSET),
            .down_id = slot + 2 * G4_PROJ_BYTES,
            .group = groups[i],
            .resource_slot = -1,
            .logical_resource_id =
                (uint64_t)(uint32_t)layer * G4_EXPERTS +
                (uint32_t)experts[i],
        };
        if (salt_gpu_weight_resource_slot(SALT_GPU_RESOURCE_EXPERT_LAYER, 15,
                entry->gate_id, entry->gate_vals, entry->gate_scales,
                entry->gate_biases, 4, G4_ROUTED, G4_HIDDEN) != 0 ||
            salt_gpu_weight_resource_slot(SALT_GPU_RESOURCE_EXPERT_LAYER, 15,
                entry->up_id, entry->up_vals, entry->up_scales,
                entry->up_biases, 4, G4_ROUTED, G4_HIDDEN) != 0 ||
            salt_gpu_weight_resource_slot(SALT_GPU_RESOURCE_EXPERT_LAYER, 15,
                entry->down_id, entry->down_vals, entry->down_scales,
                entry->down_biases, 4, G4_HIDDEN, G4_ROUTED) != 0)
            goto done;
    }
    if (salt_gpu_q4_moe_chain(model->prefill_gpu_moe, count,
            G4_HIDDEN, G4_ROUTED, inputs, outputs,
            gate_scratch, up_scratch, (size_t)total_rows * G4_ROUTED) != 0)
        goto done;
    rc = 0;
done:
    if (bound && salt_gpu_weight_resource_unbind(
            SALT_GPU_RESOURCE_EXPERT_LAYER, 15) != 0)
        rc = -1;
    if (mapping != MAP_FAILED) {
        size_t released = 0;
        (void)salt_mmap_dontneed_contained(
            mapping, (size_t)map_bytes, &released);
        if (munmap(mapping, (size_t)map_bytes) != 0) rc = -1;
    }
    return rc;
}

static int g4_group_acquire_many(void *opaque, const int *experts, int count,
                                 void **lease_out) {
    G4GroupContext *context = (G4GroupContext *)opaque;
    int cache_slots[G4_EXPERTS];
    double fetch_start;
    if (!context || !context->model || !experts || !lease_out ||
        count < 1 || count > G4_EXPERTS ||
        (!context->model->expert_cache_set && !context->model->nvfp4_mode))
        return -1;
    fetch_start = context->model->prefill_detail.enabled ? g4_now_s() : 0.0;
    for (int i = 0; i < count; i++) {
        if (experts[i] < 0 || experts[i] >= G4_EXPERTS ||
            context->leases[experts[i]].active)
            return -1;
        lease_out[i] = NULL;
    }
    if (context->model->nvfp4_mode) {
        for (int i = 0; i < count; i++) {
            int expert = experts[i];
            G4Q *base = &context->model->nvfp4_experts[
                ((size_t)context->layer * G4_EXPERTS + expert) * 3u];
            if (!base[0].set || !base[1].set || !base[2].set)
                return -1;
        }
        for (int i = 0; i < count; i++) {
            int expert = experts[i];
            G4ExpertLease *lease = &context->leases[expert];
            G4Q *base = &context->model->nvfp4_experts[
                ((size_t)context->layer * G4_EXPERTS + expert) * 3u];
            lease->bytes = (const unsigned char *)(const void *)base[0].weight;
            lease->cache_slot = -1;
            lease->logical_resource_id =
                (uint64_t)(uint32_t)context->layer * G4_EXPERTS +
                (uint32_t)expert;
            lease->active = 1;
            context->model->expert_inflight_slots++;
            if (context->model->expert_inflight_slots >
                    context->model->expert_peak_inflight_slots)
                context->model->expert_peak_inflight_slots =
                    context->model->expert_inflight_slots;
            lease_out[i] = lease;
        }
        return 0;
    }
    if (salt_cache_getmany(&context->model->expert_cache, context->layer,
            experts, count, cache_slots) < 0) {
        if (context->model->prefill_detail.enabled)
            context->model->prefill_detail.ffn_fetch_s +=
                g4_now_s() - fetch_start;
        return -1;
    }
    context->model->expert_union_fetch_calls++;
    context->model->expert_union_fetch_experts += (uint64_t)count;
    update_expert_cache_peak(context->model);
    for (int i = 0; i < count; i++) {
        int expert = experts[i];
        G4ExpertLease *lease = &context->leases[expert];
        lease->bytes = salt_cache_acquire(&context->model->expert_cache,
            cache_slots[i], context->layer, expert);
        lease->cache_slot = cache_slots[i];
        if (!lease->bytes) {
            if (context->model->prefill_detail.enabled)
                context->model->prefill_detail.ffn_fetch_s +=
                    g4_now_s() - fetch_start;
            return -1;
        }
        lease->logical_resource_id =
            (uint64_t)(uint32_t)context->layer * G4_EXPERTS +
            (uint32_t)expert;
        lease->active = 1;
        context->model->expert_inflight_slots++;
        if (context->model->expert_inflight_slots >
                context->model->expert_peak_inflight_slots)
            context->model->expert_peak_inflight_slots =
                context->model->expert_inflight_slots;
        lease_out[i] = lease;
    }
    if (context->model->prefill_detail.enabled)
        context->model->prefill_detail.ffn_fetch_s +=
            g4_now_s() - fetch_start;
    return 0;
}

static int g4_group_run(void *opaque, int expert, void *lease_opaque,
                        int group, const float *inputs, float *outputs) {
    G4GroupContext *context = (G4GroupContext *)opaque;
    G4ExpertLease *lease = (G4ExpertLease *)lease_opaque;
    const unsigned char *slot;
    int profile, rc;
    double mark;
    if (!context || expert < 0 || expert >= G4_EXPERTS || !lease ||
        !(slot = lease->bytes) || group < 1 ||
        lease->logical_resource_id !=
            (uint64_t)(uint32_t)context->layer * G4_EXPERTS +
            (uint32_t)expert)
        return -1;
    profile = context->model->prefill_detail.enabled;
    mark = profile ? g4_now_s() : 0.0;
    if (batch_expert_projection(context->model,
            slot, G4_ROUTED, G4_HIDDEN,
            group, inputs, context->gate) != 0 ||
        batch_expert_projection(context->model,
            slot + G4_PROJ_BYTES,
            G4_ROUTED, G4_HIDDEN, group, inputs, context->up) != 0)
        return -1;
    if (profile) {
        context->model->prefill_detail.ffn_gate_up_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    for (size_t i = 0; i < (size_t)group * G4_ROUTED; i++)
        context->gate[i] = salt_gemma4_gelu_tanh(context->gate[i]) *
                           context->up[i];
    if (profile) {
        context->model->prefill_detail.ffn_activation_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    rc = batch_expert_projection(context->model,
        slot + 2 * G4_PROJ_BYTES,
        G4_HIDDEN, G4_ROUTED, group, context->gate, outputs);
    if (profile)
        context->model->prefill_detail.ffn_down_s += g4_now_s() - mark;
    return rc;
}

static int g4_gpu_multi_batch(SaltGemma4Text *model,
                              const SaltBatchJob *jobs, int job_count,
                              int *expanded_jobs) {
    int expanded = 0, cols;
    if (!model || !model->full_gpu_intent || !jobs || job_count < 1 ||
        !expanded_jobs)
        return -1;
    cols = jobs[0].C;
    if (cols < 1) return -1;
    for (int job = 0; job < job_count; job++) {
        const SaltBatchJob *entry = &jobs[job];
        if (!entry->vals || !entry->scales || !entry->biases ||
            !entry->xs || !entry->ys || entry->R < 1 ||
            entry->C != cols || entry->B < 1 || entry->r0 != 0 ||
            entry->r1 != entry->R ||
            entry->B > G4_CUDA_HETERO_JOBS - expanded) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gemma4-gpu-hetero: admission job=%d R=%d C=%d B=%d expanded=%d\n",
                    job, entry->R, entry->C, entry->B, expanded);
            return -1;
        }
        for (int token = 0; token < entry->B; token++) {
            model->prefill_gpu_vals[expanded] = entry->vals;
            model->prefill_gpu_scales[expanded] = entry->scales;
            model->prefill_gpu_biases[expanded] = entry->biases;
            model->prefill_gpu_xs[expanded] = entry->xs +
                (size_t)token * (size_t)entry->C;
            model->prefill_gpu_ys[expanded] = entry->ys +
                (size_t)token * (size_t)entry->R;
            model->prefill_gpu_ids[expanded] = entry->vals;
            model->prefill_gpu_rows[expanded] = entry->R;
            expanded++;
        }
    }
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "gemma4-gpu-hetero: submit source_jobs=%d expanded=%d C=%d\n",
            job_count, expanded, cols);
    if (salt_gpu_q4_batch(model->prefill_gpu_vals,
            model->prefill_gpu_scales, model->prefill_gpu_biases,
            model->prefill_gpu_xs, model->prefill_gpu_ys,
            model->prefill_gpu_ids, model->prefill_gpu_rows,
            cols, expanded) != 0) {
        model->gpu_failures++;
        return -1;
    }
    *expanded_jobs = expanded;
    return 0;
}

static int g4_group_run_many(void *opaque, const int *experts,
                             void *const *lease_opaque, const int *groups,
                             int count, const float *inputs, float *outputs) {
    G4GroupContext *context = (G4GroupContext *)opaque;
    SaltGemma4Text *model;
    SaltBatchJob *gate_up, *down;
    const unsigned char **slots;
    int *offsets;
    double mark = 0.0, gate_up_s = 0.0, activation_s = 0.0, down_s = 0.0;
    int profile, rows = 0;
    if (!context || !(model = context->model) || !experts || !lease_opaque ||
        !groups || !inputs || !outputs || count < 1 || count > G4_EXPERTS)
        return -1;
    if (model->nvfp4_mode) {
        int offsets[G4_EXPERTS + 1];
        offsets[0] = 0;
        profile = model->prefill_detail.enabled;
        mark = profile ? g4_now_s() : 0.0;
        for (int i = 0; i < count; i++) {
            G4ExpertLease *lease = (G4ExpertLease *)lease_opaque[i];
            G4Q *base;
            if (!lease || !lease->active || groups[i] < 1 ||
                experts[i] < 0 || experts[i] >= G4_EXPERTS ||
                lease->logical_resource_id !=
                    (uint64_t)(uint32_t)context->layer * G4_EXPERTS +
                    (uint32_t)experts[i])
                return -1;
            base = &model->nvfp4_experts[
                ((size_t)context->layer * G4_EXPERTS + experts[i]) * 3u];
            offsets[i + 1] = offsets[i] + groups[i];
            model->prefill_gpu_ids[2 * i] = base[0].weight;
            model->prefill_gpu_ids[2 * i + 1] = base[1].weight;
            model->prefill_gpu_xs[2 * i] =
                inputs + (size_t)offsets[i] * G4_HIDDEN;
            model->prefill_gpu_xs[2 * i + 1] =
                inputs + (size_t)offsets[i] * G4_HIDDEN;
            model->prefill_gpu_ys[2 * i] =
                context->gate + (size_t)offsets[i] * G4_ROUTED;
            model->prefill_gpu_ys[2 * i + 1] =
                context->up + (size_t)offsets[i] * G4_ROUTED;
            model->prefill_gpu_rows[2 * i] = groups[i];
            model->prefill_gpu_rows[2 * i + 1] = groups[i];
        }
        rows = offsets[count];
        if (salt_gpu_nvfp4_batch(model->prefill_gpu_ids,
                model->prefill_gpu_xs, model->prefill_gpu_ys,
                model->prefill_gpu_rows, G4_ROUTED, G4_HIDDEN,
                2 * count) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_dense_submissions += (uint64_t)rows * 2u;
        model->gpu_expert_gate_up_submissions += (uint64_t)rows * 2u;
        model->gpu_expert_gate_up_commands++;
        model->expert_gate_up_epochs++;
        if (profile) {
            gate_up_s = g4_now_s() - mark;
            mark = g4_now_s();
        }
        for (size_t element = 0; element < (size_t)rows * G4_ROUTED; element++)
            context->gate[element] =
                salt_gemma4_gelu_tanh(context->gate[element]) *
                context->up[element];
        if (profile) {
            activation_s = g4_now_s() - mark;
            mark = g4_now_s();
        }
        for (int i = 0; i < count; i++) {
            G4Q *base = &model->nvfp4_experts[
                ((size_t)context->layer * G4_EXPERTS + experts[i]) * 3u];
            model->prefill_gpu_ids[i] = base[2].weight;
            model->prefill_gpu_xs[i] =
                context->gate + (size_t)offsets[i] * G4_ROUTED;
            model->prefill_gpu_ys[i] =
                outputs + (size_t)offsets[i] * G4_HIDDEN;
            model->prefill_gpu_rows[i] = groups[i];
        }
        if (salt_gpu_nvfp4_batch(model->prefill_gpu_ids,
                model->prefill_gpu_xs, model->prefill_gpu_ys,
                model->prefill_gpu_rows, G4_HIDDEN, G4_ROUTED,
                count) != 0) {
            model->gpu_failures++;
            return -1;
        }
        model->gpu_dense_submissions += (uint64_t)rows;
        model->gpu_expert_down_submissions += (uint64_t)rows;
        model->gpu_expert_down_commands++;
        model->expert_down_epochs++;
        if (profile) {
            down_s = g4_now_s() - mark;
            model->prefill_detail.ffn_gate_up_s += gate_up_s;
            model->prefill_detail.ffn_activation_s += activation_s;
            model->prefill_detail.ffn_down_s += down_s;
        }
        return 0;
    }
    if (model->full_gpu_intent && model->gpu_bounded_weights &&
        model->gpu_expert_layer_view) {
        profile = model->prefill_detail.enabled;
        for (int i = 0; i < count; i++) {
            if (groups[i] < 1 || rows > INT_MAX - groups[i]) return -1;
            rows += groups[i];
        }
        mark = profile ? g4_now_s() : 0.0;
        if (g4_gpu_bounded_layer_moe(model, context->layer,
                experts, groups, count, inputs, outputs,
                context->gate, context->up) != 0)
            return -1;
        model->gpu_expert_gate_up_submissions += (uint64_t)rows * 2u;
        model->gpu_expert_down_submissions += (uint64_t)rows;
        model->gpu_expert_gate_up_commands++;
        model->gpu_expert_down_commands++;
        model->expert_gate_up_epochs++;
        model->expert_down_epochs++;
        if (profile)
            model->prefill_detail.ffn_gate_up_s += g4_now_s() - mark;
        return 0;
    }
    gate_up = model->prefill_gate_up_jobs;
    down = model->prefill_down_jobs;
    slots = model->prefill_job_slots;
    offsets = model->prefill_job_offsets;
    profile = model->prefill_detail.enabled;
    offsets[0] = 0;
    for (int i = 0; i < count; i++) {
        G4ExpertLease *lease = (G4ExpertLease *)lease_opaque[i];
        const unsigned char *gate, *up;
        if (!lease || !lease->active || !lease->bytes || groups[i] < 1 ||
            experts[i] < 0 || experts[i] >= G4_EXPERTS ||
            lease->logical_resource_id !=
                (uint64_t)(uint32_t)context->layer * G4_EXPERTS +
                (uint32_t)experts[i])
            goto fallback;
        slots[i] = lease->bytes;
        gate = slots[i];
        up = slots[i] + G4_PROJ_BYTES;
        offsets[i + 1] = offsets[i] + groups[i];
        rows += groups[i];
        gate_up[2 * i] = (SaltBatchJob) {
            (const uint32_t *)(const void *)gate,
            (const uint16_t *)(const void *)(gate + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(gate + G4_BIAS_OFFSET),
            G4_ROUTED, G4_HIDDEN, groups[i],
            inputs + (size_t)offsets[i] * G4_HIDDEN,
            context->gate + (size_t)offsets[i] * G4_ROUTED,
            0, G4_ROUTED,
        };
        gate_up[2 * i + 1] = (SaltBatchJob) {
            (const uint32_t *)(const void *)up,
            (const uint16_t *)(const void *)(up + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(up + G4_BIAS_OFFSET),
            G4_ROUTED, G4_HIDDEN, groups[i],
            inputs + (size_t)offsets[i] * G4_HIDDEN,
            context->up + (size_t)offsets[i] * G4_ROUTED,
            0, G4_ROUTED,
        };
        model->prefill_gpu_moe[i] = (SaltGpuMoeExpert) {
            .gate_vals = gate_up[2 * i].vals,
            .gate_scales = gate_up[2 * i].scales,
            .gate_biases = gate_up[2 * i].biases,
            .gate_id = gate_up[2 * i].vals,
            .up_vals = gate_up[2 * i + 1].vals,
            .up_scales = gate_up[2 * i + 1].scales,
            .up_biases = gate_up[2 * i + 1].biases,
            .up_id = gate_up[2 * i + 1].vals,
            .down_vals = (const uint32_t *)(const void *)(
                slots[i] + 2 * G4_PROJ_BYTES),
            .down_scales = (const uint16_t *)(const void *)(
                slots[i] + 2 * G4_PROJ_BYTES + G4_WEIGHT_BYTES),
            .down_biases = (const uint16_t *)(const void *)(
                slots[i] + 2 * G4_PROJ_BYTES + G4_BIAS_OFFSET),
            .down_id = slots[i] + 2 * G4_PROJ_BYTES,
            .group = groups[i],
            .resource_slot = lease->cache_slot,
            .logical_resource_id = lease->logical_resource_id,
        };
    }
    mark = profile ? g4_now_s() : 0.0;
    if (model->full_gpu_intent) {
        if (model->gpu_bounded_weights && !model->gpu_expert_layer_view) {
            if (salt_gpu_q4_moe_chain_selected(model->prefill_gpu_moe, count,
                    G4_HIDDEN, G4_ROUTED, inputs, outputs,
                    context->gate, context->up,
                    (size_t)rows * G4_ROUTED) != 0)
                return -1;
            model->gpu_expert_gate_up_submissions += (uint64_t)rows * 2u;
            model->gpu_expert_down_submissions += (uint64_t)rows;
            model->gpu_expert_gate_up_commands++;
            model->gpu_expert_down_commands++;
            model->expert_gate_up_epochs++;
            model->expert_down_epochs++;
            if (profile)
                model->prefill_detail.ffn_gate_up_s += g4_now_s() - mark;
            return 0;
        }
        if (salt_gpu_q4_moe_chain(model->prefill_gpu_moe, count,
                G4_HIDDEN, G4_ROUTED, inputs, outputs,
                context->gate, context->up,
                (size_t)rows * G4_ROUTED) != 0)
            return -1;
        model->gpu_expert_gate_up_submissions += (uint64_t)rows * 2u;
        model->gpu_expert_down_submissions += (uint64_t)rows;
        model->gpu_expert_gate_up_commands++;
        model->gpu_expert_down_commands++;
        model->expert_gate_up_epochs++;
        model->expert_down_epochs++;
        if (profile)
            context->model->prefill_detail.ffn_gate_up_s += g4_now_s() - mark;
        return 0;
    }
    if (g4_q4_pool_multi_batch(model, gate_up, 2 * count) != 0) {
        goto fallback;
    }
    model->expert_gate_up_epochs++;
    if (profile) {
        gate_up_s = g4_now_s() - mark;
        mark = g4_now_s();
    }
    for (size_t element = 0; element < (size_t)rows * G4_ROUTED; element++)
        context->gate[element] =
            salt_gemma4_gelu_tanh(context->gate[element]) * context->up[element];
    if (profile) {
        activation_s = g4_now_s() - mark;
        mark = g4_now_s();
    }
    for (int i = 0; i < count; i++) {
        const unsigned char *projection = slots[i] + 2 * G4_PROJ_BYTES;
        down[i] = (SaltBatchJob) {
            (const uint32_t *)(const void *)projection,
            (const uint16_t *)(const void *)(projection + G4_WEIGHT_BYTES),
            (const uint16_t *)(const void *)(projection + G4_BIAS_OFFSET),
            G4_HIDDEN, G4_ROUTED, groups[i],
            context->gate + (size_t)offsets[i] * G4_ROUTED,
            outputs + (size_t)offsets[i] * G4_HIDDEN,
            0, G4_HIDDEN,
        };
    }
    if (g4_q4_pool_multi_batch(model, down, count) != 0) {
        goto fallback;
    }
    model->expert_down_epochs++;
    if (profile) {
        down_s = g4_now_s() - mark;
        context->model->prefill_detail.ffn_gate_up_s += gate_up_s;
        context->model->prefill_detail.ffn_activation_s += activation_s;
        context->model->prefill_detail.ffn_down_s += down_s;
    }
    return 0;

fallback:
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* Includes down failure after gate/up and activation: do not replay them.
     * Frozen legacy code below is unreachable, retained for V6 projection. */
    return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    model->q4_pool_fallbacks++;
    model->q4_multi_pool_fallbacks++;
    rows = 0;
    for (int i = 0; i < count; i++) {
        if (g4_group_run(context, experts[i], lease_opaque[i], groups[i],
                inputs + (size_t)rows * G4_HIDDEN,
                outputs + (size_t)rows * G4_HIDDEN) != 0)
            return -1;
        rows += groups[i];
    }
    return 0;
}

static int g4_group_release(void *opaque, int expert, void *lease_opaque) {
    G4GroupContext *context = (G4GroupContext *)opaque;
    G4ExpertLease *lease = (G4ExpertLease *)lease_opaque;
    int rc;
    if (!context || !context->model || !lease || !lease->active) return -1;
    if (context->model->nvfp4_mode) {
        int expected_slot = -1;
        if (expert < 0 || expert >= G4_EXPERTS ||
            lease->cache_slot != expected_slot)
            return -1;
        rc = 0;
    } else {
        rc = salt_cache_release(&context->model->expert_cache,
            lease->cache_slot, context->layer, expert);
    }
    if (rc == 0) {
        lease->active = 0;
        lease->bytes = NULL;
        lease->cache_slot = -1;
        lease->logical_resource_id = UINT64_MAX;
        if (context->model->expert_inflight_slots > 0)
            context->model->expert_inflight_slots--;
    }
    return rc;
}

typedef struct {
    SaltGemma4Text *model;
    G4Layer *layer;
    int layer_index;
    const float *residuals;
    int batch;
    float *outputs;
    size_t dense_count;
    size_t selection_count;
    float *dense_inputs;
    float *dense_gate;
    float *dense_up;
    float *dense_chain;
    float *dense_outputs;
    float *router_inputs;
    float *router_logits;
    float *routed_inputs;
    float *routed_outputs;
    float *group_gate;
    float *group_up;
    int *selected;
    float *weights;
    SaltMoEGroupJob group_job;
    SaltMoEGroupOps group_ops;
    G4GroupContext group_context;
} G4PrefillFfnBatch;

static int g4_prefill_ffn_batch_prepare(G4PrefillFfnBatch *operation) {
    operation->dense_count = (size_t)operation->batch * G4_DENSE;
    operation->selection_count = (size_t)operation->batch * G4_TOPK;
    if (operation->selection_count > SIZE_MAX / G4_ROUTED ||
        operation->selection_count * G4_ROUTED > SIZE_MAX / sizeof(float))
        return 1;
    operation->dense_inputs = operation->model->prefill_dense_inputs;
    operation->dense_gate = operation->model->prefill_dense_gate;
    operation->dense_up = operation->model->prefill_dense_up;
    operation->dense_chain = operation->model->prefill_dense_chain;
    operation->dense_outputs = operation->model->prefill_dense_outputs;
    operation->router_inputs = operation->model->prefill_router_inputs;
    operation->router_logits = operation->model->prefill_router_logits;
    operation->routed_inputs = operation->model->prefill_routed_inputs;
    operation->routed_outputs = operation->model->prefill_routed_outputs;
    operation->selected = operation->model->prefill_selected;
    operation->weights = operation->model->prefill_weights;
    operation->group_gate = operation->model->prefill_group_gate;
    operation->group_up = operation->model->prefill_group_up;
    if (!operation->dense_inputs || !operation->dense_gate ||
        !operation->dense_up || !operation->dense_chain ||
        !operation->dense_outputs || !operation->router_inputs ||
        !operation->router_logits || !operation->routed_inputs ||
        !operation->routed_outputs || !operation->selected ||
        !operation->weights || !operation->group_gate ||
        !operation->group_up)
        return -1;
    operation->model->prefill_ffn_arena_calls++;
    return 0;
}

static int g4_prefill_ffn_batch_dense(
        const G4PrefillFfnBatch *operation) {
    SaltGemma4Text *model = operation->model;
    G4Layer *layer = operation->layer;
    for (int token = 0; token < operation->batch; token++)
        if (salt_gemma4_rmsnorm(
                operation->dense_inputs + (size_t)token * G4_HIDDEN,
                operation->residuals + (size_t)token * G4_HIDDEN,
                layer->pre_ffn_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
            return -1;
    if (model->full_gpu_intent && model->gpu_trunk_exact_views) {
        if (q_matvec_batch_exact_views(model, &layer->dense_gate,
                operation->batch, operation->dense_inputs,
                operation->dense_gate) != 0 ||
            q_matvec_batch_exact_views(model, &layer->dense_up,
                operation->batch, operation->dense_inputs,
                operation->dense_up) != 0)
            return -1;
        for (size_t i = 0; i < operation->dense_count; i++)
            operation->dense_chain[i] =
                salt_gemma4_gelu_tanh(operation->dense_gate[i]) *
                operation->dense_up[i];
        return q_matvec_batch_exact_views(model, &layer->dense_down,
            operation->batch, operation->dense_chain,
            operation->dense_outputs);
    }
    if (model->full_gpu_intent)
        return g4_gpu_dense_q4_chain(model, layer, operation->batch,
            operation->dense_inputs, operation->dense_outputs,
            operation->dense_gate, operation->dense_up) == 0 ? 0 : -1;
    if (dense_gate_up_batch(model, layer, operation->batch,
            operation->dense_inputs, operation->dense_gate,
            operation->dense_up) != 0)
        return -1;
    for (size_t i = 0; i < operation->dense_count; i++)
        operation->dense_chain[i] =
            salt_gemma4_gelu_tanh(operation->dense_gate[i]) *
            operation->dense_up[i];
    return q_matvec_batch(model, &layer->dense_down, operation->batch,
        operation->dense_chain, operation->dense_outputs);
}

static int g4_prefill_ffn_batch_route(
        const G4PrefillFfnBatch *operation) {
    SaltGemma4Text *model = operation->model;
    G4Layer *layer = operation->layer;
    for (int token = 0; token < operation->batch; token++) {
        const float *residual =
            operation->residuals + (size_t)token * G4_HIDDEN;
        float *router_input =
            operation->router_inputs + (size_t)token * G4_HIDDEN;
        if (salt_gemma4_router_input(router_input, residual,
                layer->router_scale.values, G4_HIDDEN, G4_EPS) != 0 ||
            salt_gemma4_rmsnorm(
                operation->routed_inputs + (size_t)token * G4_HIDDEN,
                residual, layer->pre_ffn_norm_2.values,
                G4_HIDDEN, G4_EPS, 1) != 0)
            return -1;
    }
    if (model->full_gpu_intent && !model->gpu_trunk_exact_views &&
            layer->router.bits == 8) {
        if (gpu_q8_matvec_batch(model, &layer->router, operation->batch,
                operation->router_inputs, operation->router_logits) != 0) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr, "gemma4-prefill-ffn: router layer=%d\n",
                        operation->layer_index);
            return -1;
        }
    } else if (layer->router.bits == 16) {
        if (q_matvec_batch(model, &layer->router, operation->batch,
                operation->router_inputs, operation->router_logits) != 0)
            return -1;
    } else if (g4_q8_pool_batch(model, &layer->router, operation->batch,
            operation->router_inputs, operation->router_logits) != 0) {
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        model->q8_pool_fallbacks++;
        for (int token = 0; token < operation->batch; token++)
            if (q_matvec(&layer->router,
                    operation->router_inputs + (size_t)token * G4_HIDDEN,
                    operation->router_logits +
                        (size_t)token * G4_EXPERTS) != 0)
                return -1;
    }
    for (int token = 0; token < operation->batch; token++) {
        float *logits =
            operation->router_logits + (size_t)token * G4_EXPERTS;
        int *token_selected =
            operation->selected + (size_t)token * G4_TOPK;
        float *token_weights =
            operation->weights + (size_t)token * G4_TOPK;
        if (salt_gemma4_router_topk(logits,
                layer->per_expert_scale.values, G4_EXPERTS, G4_TOPK,
                token_selected, token_weights) != 0)
            return -1;
        sort_selection(token_selected, token_weights);
        route_observe(model, SALT_GEMMA4_ROUTE_PREFILL,
                      (uint64_t)(model->position + token),
                      operation->layer_index, token_selected);
    }
    return 0;
}

typedef struct G4PrefillExpertFlowCall {
    SaltMoEGroupJob *job;
    SaltMoEGroupOps *ops;
    G4GroupContext *context;
} G4PrefillExpertFlowCall;

static int g4_prefill_expert_flow_call(void *opaque) {
    G4PrefillExpertFlowCall *call = (G4PrefillExpertFlowCall *)opaque;
    return call && call->job && call->ops && call->context
        ? salt_moe_group_execute(call->job, call->ops, call->context)
        : -1;
}

static int g4_prefill_ffn_batch_experts(G4PrefillFfnBatch *operation) {
    G4PrefillExpertFlowCall flow;
    int rc, opened_flow = 0;
    memset(&operation->group_context, 0, sizeof operation->group_context);
    operation->group_context.model = operation->model;
    operation->group_context.layer = operation->layer_index;
    operation->group_context.gate = operation->group_gate;
    operation->group_context.up = operation->group_up;
    memset(&operation->group_job, 0, sizeof operation->group_job);
    operation->group_job.batch = operation->batch;
    operation->group_job.topk = G4_TOPK;
    operation->group_job.n_experts = G4_EXPERTS;
    operation->group_job.hidden = G4_HIDDEN;
    operation->group_job.selected = operation->selected;
    operation->group_job.weights = operation->weights;
    operation->group_job.inputs = operation->routed_inputs;
    operation->group_job.outputs = operation->routed_outputs;
    operation->group_job.scratch = &operation->model->prefill_moe_scratch;
    memset(&operation->group_ops, 0, sizeof operation->group_ops);
    operation->group_ops.acquire_many = g4_group_acquire_many;
    operation->group_ops.acquire_batch = G4_EXPERTS;
    operation->group_ops.run_many = g4_group_run_many;
    operation->group_ops.run_batch = G4_EXPERTS;
    operation->group_ops.release = g4_group_release;
    flow = (G4PrefillExpertFlowCall) {
        &operation->group_job, &operation->group_ops,
        &operation->group_context,
    };
    if (operation->model->prefill_expert_matrix_flow &&
        !operation->model->compute_pool.aflow_running) {
        opened_flow = 1;
        rc = salt_attn_pool_flow_run(&operation->model->compute_pool,
            g4_prefill_expert_flow_call, &flow);
    } else {
        rc = g4_prefill_expert_flow_call(&flow);
    }
    if (rc != 0) {
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "gemma4-prefill-ffn: routed-group layer=%d "
                    "inflight=%d\n", operation->layer_index,
                    operation->model->expert_inflight_slots);
        return -1;
    }
    if (opened_flow)
        operation->model->prefill_expert_matrix_flow_sessions++;
    return 0;
}

static int g4_prefill_ffn_batch_combine(
        const G4PrefillFfnBatch *operation) {
    G4Layer *layer = operation->layer;
    for (int token = 0; token < operation->batch; token++) {
        float *routed =
            operation->routed_outputs + (size_t)token * G4_HIDDEN;
        if (salt_gemma4_parallel_ffn_combine(
                operation->outputs + (size_t)token * G4_HIDDEN,
                operation->residuals + (size_t)token * G4_HIDDEN,
                operation->dense_outputs + (size_t)token * G4_HIDDEN,
                routed, layer->post_ffn_norm_1.values,
                layer->post_ffn_norm_2.values,
                layer->post_ffn_norm.values,
                layer->layer_scalar.values[0], G4_HIDDEN, G4_EPS,
                operation->model->parallel_scratch) != 0)
            return -1;
    }
    return 0;
}

static int feed_forward_batch(SaltGemma4Text *model, G4Layer *layer,
                              int layer_index, const float *residuals,
                              int batch, float *outputs) {
    G4PrefillFfnBatch operation;
    int prepare_rc;
    int rc = -1;
    int profile;
    double mark, cleanup_mark;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (!model || !layer || !layer->plan || !residuals || !outputs ||
        batch < 2 || layer->plan->activation != SALT_ACT_GELU_TANH ||
        !layer->plan->parallel_dense_routed ||
        batch > model->prefill_capacity)
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if (!model || !layer || !layer->plan || !residuals || !outputs ||
        batch < 1 || layer->plan->activation != SALT_ACT_GELU_TANH ||
        !layer->plan->parallel_dense_routed ||
        batch > model->prefill_capacity)
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        return -1;
    profile = model->prefill_detail.enabled;
    mark = profile ? g4_now_s() : 0.0;
    memset(&operation, 0, sizeof operation);
    operation.model = model;
    operation.layer = layer;
    operation.layer_index = layer_index;
    operation.residuals = residuals;
    operation.batch = batch;
    operation.outputs = outputs;
    prepare_rc = g4_prefill_ffn_batch_prepare(&operation);
    if (prepare_rc > 0)
        return -1;
    if (prepare_rc < 0)
        goto done;
    if (profile) {
        model->prefill_detail.ffn_alloc_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_ffn_batch_dense(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.ffn_dense_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_ffn_batch_route(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.ffn_route_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_ffn_batch_experts(&operation) != 0)
        goto done;
    if (profile) {
        model->prefill_detail.ffn_group_s += g4_now_s() - mark;
        mark = g4_now_s();
    }
    if (g4_prefill_ffn_batch_combine(&operation) != 0)
        goto done;
    if (profile)
        model->prefill_detail.ffn_combine_s += g4_now_s() - mark;
    rc = 0;

done:
    cleanup_mark = profile ? g4_now_s() : 0.0;
    if (model && model->expert_inflight_slots != 0) rc = -1;
    if (profile)
        model->prefill_detail.ffn_cleanup_s += g4_now_s() - cleanup_mark;
    return rc;
}

static int g4_prefill_embed(void *opaque, const int *tokens, int batch,
                            float *states) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    for (int token = 0; token < batch; token++)
        if (tokens[token] < 0 || tokens[token] >= G4_VOCAB ||
            tokens[token] == G4_IMAGE_TOKEN ||
            embedding_row(&model->embedding, tokens[token],
                states + (size_t)token * G4_HIDDEN) != 0)
            return step_fail("prefill-embedding", -1, model->position);
    return 0;
}

static int g4_prefill_attention(void *opaque, int layer, int start,
                                int batch, const float *states,
                                float *outputs) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (g4_trunk_layer_window_begin(model, layer) != 0)
        return step_fail("prefill-layer-window-bind", layer, start);
    if (prefill_attention_layer_batch(model, &model->layers[layer],
            states, start, batch, outputs) != 0) {
        (void)g4_trunk_layer_window_end(model);
        return step_fail("prefill-attention", layer, start);
    }
    return 0;
}

static int g4_prefill_ffn(void *opaque, int layer, int start, int batch,
                          const float *states, float *outputs) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    G4Layer *gemma_layer = &model->layers[layer];
    model->position = start + batch - 1;
    int compute_rc = feed_forward_batch(model, gemma_layer, layer,
        states, batch, outputs);
    int window_rc = g4_trunk_layer_window_end(model);
    if (compute_rc != 0)
        return step_fail("prefill-feed-forward-batch", layer,
                         model->position);
    if (window_rc != 0)
        return step_fail("prefill-layer-window-release", layer,
                         model->position);
    if (g4_memory_budget_ok(model, "prefill-layer", layer) != 0)
        return step_fail("prefill-memory-limit", layer, model->position);
    return 0;
}

static void g4_prefill_release(void *opaque, int layer) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    release_layer_residency(&model->layers[layer]);
}

static int g4_prefill_decode_lookahead_prepare(
        void *opaque, const SaltTextPhaseLookaheadRequest *request) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !request ||
        request->source_phase != SALT_TEXT_PHASE_PREFILL ||
        request->target_phase != SALT_TEXT_PHASE_DECODE ||
        request->source_generation !=
            model->text_kv_state.transition_generation ||
        request->result_generation != request->source_generation + 1u ||
        request->source_position != model->text_kv_state.position ||
        request->result_position != (uint32_t)model->position + 1u)
        return -1;
    if (!model->prefill_decode_lookahead_enabled)
        return -1;
    if (request->result_position >= (uint32_t)model->max_context)
        return 1;
    if (model->prefill_retain_layers) return 0;
    return prepare_layer_weight_residency(
        model, &model->layers[0]) == 0 ? 0 : 1;
}

static int g4_prefill_decode_lookahead_claim(
        void *opaque, const SaltTextPhaseLookaheadRequest *request) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !request ||
        request->target_phase != SALT_TEXT_PHASE_DECODE ||
        request->result_generation !=
            model->text_kv_state.transition_generation ||
        request->result_position != (uint32_t)model->position)
        return -1;
    return 0;
}

static int g4_prefill_decode_lookahead_cancel(
        void *opaque, const SaltTextPhaseLookaheadRequest *request) {
    return opaque && request ? 0 : -1;
}

static const SaltTextPhaseLookaheadOps g4_prefill_decode_lookahead_ops = {
    g4_prefill_decode_lookahead_prepare,
    g4_prefill_decode_lookahead_claim,
    g4_prefill_decode_lookahead_cancel,
};

static int g4_prefill_finish(void *opaque, const float *last_state,
                             float *logits) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (salt_gemma4_rmsnorm(model->final_state, last_state,
            model->final_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
        return step_fail("prefill-final-norm", G4_LAYERS,
                         model->position);
    if (q_matvec_operation(model, &model->embedding,
            model->final_state, logits) != 0) {
        release_q_residency(&model->embedding);
        return step_fail("prefill-head", G4_LAYERS, model->position);
    }
    if (!model->prefill_retain_layers)
        release_q_residency(&model->embedding);
    if (salt_gemma4_softcap(logits, G4_VOCAB, G4_LOGIT_CAP) != 0) {
        if (model->prefill_retain_layers)
            release_q_residency(&model->embedding);
        return step_fail("prefill-final", G4_LAYERS, model->position);
    }
    return 0;
}

static int g4_prefill_finish_batch(void *opaque, const float *states,
                                   int batch, float *position_logits,
                                   size_t logits_stride) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    float *normalized;
    if (!model || !states || batch < 1 || !position_logits ||
        logits_stride != G4_VOCAB || batch > model->max_context ||
        !model->operation_states_a)
        return -1;
    normalized = model->operation_states_a;
    for (int row = 0; row < batch; row++)
        if (salt_gemma4_rmsnorm(
                normalized + (size_t)row * G4_HIDDEN,
                states + (size_t)row * G4_HIDDEN,
                model->final_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
            return step_fail("dpr-final-norm", G4_LAYERS, model->position);
    if (q_matvec_batch(model, &model->embedding, batch,
                       normalized, position_logits) != 0) {
        release_q_residency(&model->embedding);
        return step_fail("dpr-head-batch", G4_LAYERS, model->position);
    }
    if (!model->prefill_retain_layers)
        release_q_residency(&model->embedding);
    for (int row = 0; row < batch; row++)
        if (salt_gemma4_softcap(
                position_logits + (size_t)row * logits_stride,
                G4_VOCAB, G4_LOGIT_CAP) != 0) {
            if (model->prefill_retain_layers)
                release_q_residency(&model->embedding);
            return step_fail("dpr-final", G4_LAYERS, model->position);
        }
    return 0;
}

static int g4_prefill_publish(void *opaque, int position) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    uint64_t bytes = 0;
    double started;
    if (!model || g4_text_state_publish(model, position) != 0)
        return -1;
    if (!model->full_gpu_intent || !model->gpu_attention_enabled)
        return 0;
    started = g4_now_s();
    if (g4_gpu_kv_materialize(model, position, &bytes) != 0)
        return -1;
    fprintf(stderr,
        "GEMMA4_KV_MATERIALIZE action=consumer-handoff artifact=0 reason=0 "
        "position=%d bytes=%llu wall_s=%.6f\n",
        position, (unsigned long long)bytes, g4_now_s() - started);
    return 0;
}

static int g4_recipe_int(const char *name, int minimum, int maximum,
                         int *value) {
    const char *text = getenv(name);
    char *end = NULL;
    long parsed;
    if (!name || !value || !text || !*text) return -1;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' ||
        parsed < minimum || parsed > maximum)
        return -1;
    *value = (int)parsed;
    return 0;
}

static int g4_recipe_decimal_gb_bytes(
        const char *name, uint32_t minimum, uint32_t maximum,
        uint64_t *bytes) {
    const char *text = name ? getenv(name) : NULL;
    const unsigned char *cursor = (const unsigned char *)text;
    uint64_t whole = 0, fraction = 0;
    uint32_t digits = 0;
    if (!bytes || !text || !*text || minimum > maximum) return -1;
    while (*cursor >= (unsigned char)'0' && *cursor <= (unsigned char)'9') {
        if (whole > UINT64_C(1000000000)) return -1;
        whole = whole * 10u + (uint64_t)(*cursor - (unsigned char)'0');
        cursor++;
    }
    if (cursor == (const unsigned char *)text) return -1;
    if (*cursor == (unsigned char)'.') {
        cursor++;
        while (*cursor >= (unsigned char)'0' &&
               *cursor <= (unsigned char)'9') {
            if (digits == 9u) return -1;
            fraction = fraction * 10u +
                (uint64_t)(*cursor - (unsigned char)'0');
            digits++;
            cursor++;
        }
        if (digits == 0u) return -1;
    }
    if (*cursor != (unsigned char)'\0' ||
        whole < minimum || whole > maximum ||
        (whole == maximum && fraction != 0u))
        return -1;
    while (digits++ < 9u) fraction *= 10u;
    if (whole > UINT64_MAX / UINT64_C(1000000000) ||
        whole * UINT64_C(1000000000) > UINT64_MAX - fraction)
        return -1;
    *bytes = whole * UINT64_C(1000000000) + fraction;
    return 0;
}

static int g4_prefill_recipe(SaltGemma4Text *model,
                             SaltTextPrefillPlan *plan) {
    const char *chunked = getenv("SALT_PREFILL_CHUNK");
    const char *lookahead = getenv("SALT_PREFILL_DECODE_LOOKAHEAD");
    int m2_enabled, retain_layers;
    if (!model || !plan || !chunked || strcmp(chunked, "1") != 0 ||
        g4_recipe_int("SALT_PREFILL_B", 1, 4096,
                      &plan->max_batch) != 0 ||
        g4_recipe_int("SALT_M2_BATCH", 0, 1, &m2_enabled) != 0 ||
        g4_recipe_int("SALT_PREFILL_RETAIN_LAYERS", 0, 1,
                      &retain_layers) != 0)
        return -1;
    if (lookahead && strcmp(lookahead, "0") != 0 &&
            strcmp(lookahead, "1") != 0)
        return -1;
    plan->n_layers = G4_LAYERS;
    plan->hidden = G4_HIDDEN;
    plan->resource_policy = retain_layers
        ? SALT_TEXT_RETAIN_OPERATION : SALT_TEXT_RELEASE_PER_LAYER;
    model->prefill_retain_layers = retain_layers;
    model->prefill_decode_lookahead_enabled =
        lookahead && strcmp(lookahead, "1") == 0;
    if (model->prefill_decode_lookahead_enabled && model->fine_token_enabled)
        return -1;
    return m2_enabled == 1 ? 0 : -1;
}

static int g4_gpu_phase_begin(
        void *opaque, SaltTextRuntimePhase phase) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !model->full_gpu_intent ||
        (phase != SALT_TEXT_PHASE_PREFILL &&
         phase != SALT_TEXT_PHASE_DECODE))
        return -1;
    if (!salt_gpu_cuda_present()) return 0;
    if (salt_gpu_defer()) return -1;
    return salt_gpu_set_defer(1);
}

static int g4_gpu_phase_end(
        void *opaque, SaltTextRuntimePhase phase, int operation_status) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    int rc;
    if (!model || !model->full_gpu_intent ||
        (phase != SALT_TEXT_PHASE_PREFILL &&
         phase != SALT_TEXT_PHASE_DECODE))
        return -1;
    if (!salt_gpu_cuda_present()) return operation_status == 0 ? 0 : -1;
    if (!salt_gpu_defer()) return -1;
    rc = salt_gpu_set_defer(0);
    return operation_status == 0 && rc == 0 ? 0 : -1;
}

static int g4_prefill_transaction_run(
        void *opaque, int (*operation)(void *), void *operation_context) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !operation) return -1;
    if (model->full_gpu_intent) {
        const SaltTextPhaseScopeOps phase_ops = {
            g4_gpu_phase_begin, g4_gpu_phase_end,
        };
        const SaltTextPhaseScope phase_scope = { &phase_ops, model };
        return salt_text_phase_scope_run(
            &phase_scope, SALT_TEXT_PHASE_PREFILL,
            operation, operation_context);
    }
    if (!model->prefill_operation_flow || model->compute_pool.aflow_running)
        return operation(operation_context);
    if (!model->compute_pool_ready) return -1;
    return salt_attn_pool_flow_run(
        &model->compute_pool, operation, operation_context);
}

static void g4_emit_prefill_waterfall(
        const char *marker, const SaltGemma4Text *model, int token_count,
        const SaltTextPrefillProfile *profile,
        SaltTextResourcePolicy resource_policy) {
    if (!marker || !model || !profile) return;
    fprintf(stderr,
        "%s tokens=%d chunks=%d grouped_chunks=%d "
        "total_s=%.6f setup_s=%.6f embed_s=%.6f attention_s=%.6f "
        "feed_forward_s=%.6f release_s=%.6f finish_s=%.6f "
        "validate_s=%.6f publish_s=%.6f cleanup_s=%.6f "
        "released_layers=%d retained_layers=%d resource_policy=%s\n",
        marker, token_count, profile->chunks, profile->grouped_chunks,
        profile->total_s, profile->setup_s, profile->embed_s,
        profile->attention_s, profile->feed_forward_s, profile->release_s,
        profile->finish_s, profile->validate_s, profile->publish_s,
        profile->cleanup_s, profile->released_layers,
        profile->retained_layers,
        resource_policy == SALT_TEXT_RETAIN_OPERATION ? "retain" : "release");
    fprintf(stderr,
        "%s_ATTENTION_DETAIL alloc_s=%.6f norm_s=%.6f "
        "qkv_s=%.6f transform_s=%.6f body_s=%.6f o_s=%.6f "
        "residual_s=%.6f cleanup_s=%.6f\n",
        marker,
        model->prefill_detail.attention_alloc_s,
        model->prefill_detail.attention_norm_s,
        model->prefill_detail.attention_qkv_s,
        model->prefill_detail.attention_transform_s,
        model->prefill_detail.attention_body_s,
        model->prefill_detail.attention_o_s,
        model->prefill_detail.attention_residual_s,
        model->prefill_detail.attention_cleanup_s);
    fprintf(stderr,
        "%s_FFN_DETAIL alloc_s=%.6f dense_s=%.6f "
        "route_s=%.6f group_s=%.6f fetch_s=%.6f combine_s=%.6f "
        "gate_up_s=%.6f activation_s=%.6f down_s=%.6f cleanup_s=%.6f\n",
        marker,
        model->prefill_detail.ffn_alloc_s,
        model->prefill_detail.ffn_dense_s,
        model->prefill_detail.ffn_route_s,
        model->prefill_detail.ffn_group_s,
        model->prefill_detail.ffn_fetch_s,
        model->prefill_detail.ffn_combine_s,
        model->prefill_detail.ffn_gate_up_s,
        model->prefill_detail.ffn_activation_s,
        model->prefill_detail.ffn_down_s,
        model->prefill_detail.ffn_cleanup_s);
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
static int g4_text_executor_ready_with_shared_prefix(
    const SaltGemma4Text *model);
#define g4_text_executor_ready g4_text_executor_ready_with_shared_prefix
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_text_executor_ready(const SaltGemma4Text *model);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */

static int g4_pageable_gpu_prefill_selected(const SaltGemma4Text *model) {
    if (g4_registered_cuda_gpu_program(model) && model->text_gpu.ready)
        return 1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* Metal PREFILL uses the retained phase-specific batch operators.
     * The compiled TARGET program remains available for TARGET, not as the
     * ordinary PREFILL realization. Keep later ring/shared-state handling. */
    if (model && model->full_gpu_intent && g4_target_gpu_program(model) &&
        model->text_gpu.ready && salt_gpu_rocm_present() &&
        model->gpu_device_residency && model->gpu_residency.ready &&
        model->gpu_weight_addressability == SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW)
        return 1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    return model && model->full_gpu_intent && model->fine_token_enabled &&
        g4_target_gpu_program(model) && model->text_gpu.ready &&
        model->gpu_weight_addressability == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE &&
        salt_gpu_cuda_present() && salt_gpu_pageable_mmap_active();
}

/* The pageable full-GPU recipe already compiles the immutable text TensorOps
 * program and its GPU-only executor at startup.  Use that same engine phase
 * entry for known PREFILL rows instead of invoking the legacy model callbacks,
 * which leave normalization, routing, residual, activation, and reduction on
 * the host.  The executor owns traversal, selected-resource leases, completion,
 * and canonical destinations; salt_text_execute_all() owns KV COMMIT. */
static int g4_pageable_gpu_prefill(
        SaltGemma4Text *model, const SaltTextPrefillPlan *plan,
        const int *tokens, int token_count, float *position_logits,
        size_t logits_stride, float *final_logits) {
    uint64_t submissions = 0, fences = 0, barriers = 0;
    uint64_t projections = 0, expert_gate_up = 0, expert_down = 0;
    uint64_t templates = 0, template_reuses = 0, dynamic_patches = 0;
    uint64_t selected_jobs = 0, physical_kernels = 0, host_launches = 0;
    uint64_t transfer_bytes = 0, logits_bytes = 0, kv_publish_bytes = 0;
    uint64_t clear_bytes = 0;
    int start, completed = 0, chunks = 0;
    if (!g4_pageable_gpu_prefill_selected(model) || !plan || !tokens ||
        token_count < 1 || !final_logits || sizeof(int) != sizeof(int32_t) ||
        plan->max_batch < 1 ||
        model->position > model->max_context - token_count ||
        (position_logits &&
         (logits_stride < G4_VOCAB ||
          (size_t)token_count > SIZE_MAX / logits_stride)))
        return step_fail("pageable-prefill-arguments", -1,
                         model ? model->position : -1);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (plan->resource_policy != SALT_TEXT_RELEASE_PER_LAYER)
        return step_fail("pageable-prefill-resource-policy", -1,
                         model->position);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (!model->text_program.ready)
        return step_fail("pageable-prefill-program", -1, model->position);
    if (!g4_text_executor_ready(model))
        return step_fail("pageable-prefill-executor", -1, model->position);
    if (!g4_text_state_aligned(model))
        return step_fail("pageable-prefill-state-alignment", -1,
                         model->position);
    if (model->text_program.maximum_candidates < 1u)
        return step_fail("pageable-prefill-candidate-capacity", -1,
                         model->position);
    start = model->position;
    while (completed < token_count) {
        SaltTextExecuteAllResult result;
        const SaltTextVerifyBackendStats *backend;
        const float *chunk_logits;
        uint64_t generation;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        uint64_t expected_logits_bytes;
        uint32_t output_rows;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        int execute_status;
        int batch = token_count - completed;
        if (batch > plan->max_batch) batch = plan->max_batch;
        if ((uint32_t)batch > model->text_program.maximum_candidates)
            batch = (int)model->text_program.maximum_candidates;
        if (batch < 1 ||
            model->text_kv_state.transition_generation == UINT64_MAX)
            return -1;
        generation = model->text_kv_state.transition_generation + 1u;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        output_rows = position_logits
            ? (uint32_t)batch
            : (completed + batch == token_count ? 1u : 0u);
        expected_logits_bytes =
            (uint64_t)output_rows * G4_VOCAB * sizeof(float);
        memset(&result, 0, sizeof result);
        execute_status = position_logits
            ? salt_text_execute_all(&model->text_program,
                &model->text_executor, generation,
                (uint32_t)(start + completed),
                (const int32_t *)(const void *)(tokens + completed),
                (uint32_t)batch, &result)
            : salt_text_execute_prefill(&model->text_program,
                &model->text_executor, generation,
                (uint32_t)(start + completed),
                (const int32_t *)(const void *)(tokens + completed),
                (uint32_t)batch, output_rows, &result);
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        memset(&result, 0, sizeof result);
        execute_status = salt_text_execute_all(&model->text_program,
                &model->text_executor, generation,
                (uint32_t)(start + completed),
                (const int32_t *)(const void *)(tokens + completed),
                (uint32_t)batch, &result);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        if (execute_status != 0) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "GEMMA4_PREFILL_GPU_EXECUTE_FAIL status=%d committed=%u "
                    "position=%u generation=%llu submissions=%u fences=%u "
                    "templates=%u reuses=%u patches=%u selected=%u graphs=%u "
                    "kernels=%u host_launches=%u input_bytes=%llu "
                    "logits_bytes=%llu kv_bytes=%llu\n",
                    (int)result.status, result.committed_count,
                    result.result_position,
                    (unsigned long long)result.transition_generation,
                    result.backend.engine_submissions,
                    result.backend.completion_fences,
                    result.backend.backend_template_count,
                    result.backend.backend_template_reuses,
                    result.backend.backend_dynamic_patches,
                    result.backend.backend_selected_jobs,
                    result.backend.backend_graph_count,
                    result.backend.backend_physical_kernel_nodes,
                    result.backend.backend_host_kernel_launch_calls,
                    (unsigned long long)result.backend.initial_transfer_bytes,
                    (unsigned long long)result.backend.final_logits_transfer_bytes,
                    (unsigned long long)result.backend.final_kv_publish_bytes);
            return -1;
        }
        backend = &result.backend;
        if (result.status != SALT_TEXT_VERIFY_COMMITTED ||
            result.committed_count != (uint32_t)batch ||
            result.result_position != (uint32_t)(start + completed + batch) ||
            result.transition_generation != generation ||
            !result.view.canonical_base ||
            result.view.canonical_bytes < model->text_program.layout.total_bytes ||
            backend->engine_submissions != 1u ||
            backend->completion_fences != 1u ||
            backend->intermediate_host_publications != 0u ||
            backend->projection_dispatches == 0u ||
            backend->expert_gate_up_dispatches == 0u ||
            backend->expert_down_dispatches == 0u ||
            backend->cpu_matrix_pool_phases != 0u ||
            backend->cpu_qkv_waves != 0u ||
            backend->cpu_dense_gate_up_waves != 0u ||
            backend->cpu_expert_gate_up_waves != 0u ||
            backend->cpu_expert_down_waves != 0u ||
            backend->cpu_graph_sessions != 0u ||
            backend->cpu_graph_nodes != 0u ||
            backend->cpu_graph_spans != 0u ||
            backend->cpu_graph_serial_spans != 0u ||
            backend->cpu_graph_parallel_spans != 0u ||
            backend->cpu_graph_barriers != 0u ||
            backend->backend_template_count == 0u ||
            backend->backend_template_reuses == 0u ||
            backend->backend_dynamic_patches == 0u ||
            backend->backend_selected_jobs != (uint32_t)batch * G4_TOPK ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
            (backend->backend_graph_count == 0u
                ? (backend->backend_graph_launches != 0u ||
                   backend->backend_graph_parameter_patches != 0u)
                : backend->backend_graph_launches == 0u) ||
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
            backend->backend_graph_count != 0u ||
            backend->backend_graph_launches != 0u ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
            backend->backend_physical_kernel_nodes == 0u ||
            backend->backend_host_kernel_launch_calls == 0u ||
            backend->initial_transfer_bytes !=
                (uint64_t)(uint32_t)batch * sizeof(int32_t) ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
            backend->final_logits_transfer_bytes != expected_logits_bytes ||
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
            backend->final_logits_transfer_bytes !=
                (uint64_t)(uint32_t)batch * G4_VOCAB * sizeof(float) ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
            backend->final_kv_publish_bytes == 0u) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "GEMMA4_PREFILL_GPU_RESULT_FAIL status=%d committed=%u/%d "
                    "position=%u/%d generation=%llu/%llu submissions=%u "
                    "fences=%u cpu_cells=%u templates=%u reuses=%u patches=%u "
                    "selected=%u/%u graphs=%u kernels=%u host_launches=%u "
                    "input_bytes=%llu/%llu logits_bytes=%llu/%llu kv_bytes=%llu\n",
                    (int)result.status, result.committed_count, batch,
                    result.result_position, start + completed + batch,
                    (unsigned long long)result.transition_generation,
                    (unsigned long long)generation,
                    backend->engine_submissions, backend->completion_fences,
                    backend->cpu_graph_nodes, backend->backend_template_count,
                    backend->backend_template_reuses,
                    backend->backend_dynamic_patches,
                    backend->backend_selected_jobs,
                    (uint32_t)batch * G4_TOPK,
                    backend->backend_graph_count,
                    backend->backend_physical_kernel_nodes,
                    backend->backend_host_kernel_launch_calls,
                    (unsigned long long)backend->initial_transfer_bytes,
                    (unsigned long long)((uint64_t)(uint32_t)batch *
                        sizeof(int32_t)),
                    (unsigned long long)backend->final_logits_transfer_bytes,
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
                    (unsigned long long)expected_logits_bytes,
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
                    (unsigned long long)((uint64_t)(uint32_t)batch *
                        G4_VOCAB * sizeof(float)),
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
                    (unsigned long long)backend->final_kv_publish_bytes);
            return -1;
        }
        chunk_logits = (const float *)(const void *)(
            result.view.canonical_base +
            model->text_program.layout.position_logits);
        if (position_logits)
            for (int row = 0; row < batch; row++)
                memcpy(position_logits +
                           (size_t)(completed + row) * logits_stride,
                       chunk_logits + (size_t)row * G4_VOCAB,
                       (size_t)G4_VOCAB * sizeof(float));
        if (completed + batch == token_count)
            memcpy(final_logits,
                   chunk_logits + (size_t)(batch - 1) * G4_VOCAB,
                   (size_t)G4_VOCAB * sizeof(float));
        submissions += backend->engine_submissions;
        fences += backend->completion_fences;
        barriers += backend->internal_dependency_barriers;
        projections += backend->projection_dispatches;
        expert_gate_up += backend->expert_gate_up_dispatches;
        expert_down += backend->expert_down_dispatches;
        templates += backend->backend_template_count;
        template_reuses += backend->backend_template_reuses;
        dynamic_patches += backend->backend_dynamic_patches;
        selected_jobs += backend->backend_selected_jobs;
        physical_kernels += backend->backend_physical_kernel_nodes;
        host_launches += backend->backend_host_kernel_launch_calls;
        transfer_bytes += backend->initial_transfer_bytes;
        logits_bytes += backend->final_logits_transfer_bytes;
        kv_publish_bytes += backend->final_kv_publish_bytes;
        clear_bytes += backend->canonical_clear_bytes;
        completed += batch;
        chunks++;
        if (g4_memory_budget_ok(model, "prefill-gpu-program", -1) != 0)
            return -1;
    }
    if (completed != token_count || chunks < 1 ||
        submissions != (uint64_t)(uint32_t)chunks ||
        fences != submissions || g4_text_state_sync_from_kv(model) != 0 ||
        model->position != start + token_count)
        return -1;
    model->token_transition_generation =
        model->text_kv_state.transition_generation;
    fprintf(stderr,
        "GEMMA4_PREFILL_GPU_PROGRAM tokens=%d chunks=%d max_rows=%u "
        "engine_submissions=%llu completion_fences=%llu "
        "intermediate_publications=0 cpu_model_cells=0 legacy_callbacks=0 "
        "dependency_barriers=%llu projections=%llu "
        "expert_gate_up=%llu expert_down=%llu templates=%llu "
        "template_reuses=%llu dynamic_patches=%llu selected_jobs=%llu "
        "physical_kernels=%llu host_launches=%llu "
        "input_transfer_bytes=%llu logits_transfer_bytes=%llu "
        "kv_publish_bytes=%llu canonical_clear_bytes=%llu\n",
        token_count, chunks, model->text_program.maximum_candidates,
        (unsigned long long)submissions, (unsigned long long)fences,
        (unsigned long long)barriers, (unsigned long long)projections,
        (unsigned long long)expert_gate_up, (unsigned long long)expert_down,
        (unsigned long long)templates, (unsigned long long)template_reuses,
        (unsigned long long)dynamic_patches,
        (unsigned long long)selected_jobs,
        (unsigned long long)physical_kernels,
        (unsigned long long)host_launches,
        (unsigned long long)transfer_bytes,
        (unsigned long long)logits_bytes,
        (unsigned long long)kv_publish_bytes,
        (unsigned long long)clear_bytes);
    return 0;
}

int salt_gemma4_text_prefill(SaltGemma4Text *model,
                             const int *tokens, int token_count,
                             float *logits) {
    SaltTextPrefillPlan plan;
    SaltTextPrefillProfile profile;
    SaltTextPrefillOps ops;
    int start, rc;
    if (!model || !tokens || !logits || token_count < 1 ||
        model->position < 0 ||
        model->position > model->max_context - token_count)
        return step_fail("prefill-arguments", -1,
                         model ? model->position : -1);
    start = model->position;
    memset(&model->prefill_detail, 0, sizeof model->prefill_detail);
    model->prefill_detail.enabled = getenv("SALT_WATERFALL") != NULL;
    memset(&plan, 0, sizeof plan);
    if (g4_prefill_recipe(model, &plan) != 0)
        return step_fail("prefill-engine-config", -1, start);
    if (plan.max_batch > model->prefill_capacity)
        plan.max_batch = model->prefill_capacity;
    plan.scratch = &model->text_prefill_scratch;
    if (model->prefill_decode_lookahead_enabled) {
        plan.lookahead = &model->prefill_decode_lookahead;
        plan.lookahead_ops = &g4_prefill_decode_lookahead_ops;
        plan.lookahead_context = model;
        plan.transition_generation =
            model->text_kv_state.transition_generation;
        plan.next_phase = SALT_TEXT_PHASE_DECODE;
    }
    if (model->prefill_operation_flow || model->full_gpu_intent) {
        plan.transaction_run = g4_prefill_transaction_run;
        plan.transaction_context = model;
    }
    if (g4_pageable_gpu_prefill_selected(model)) {
        rc = g4_pageable_gpu_prefill(
            model, &plan, tokens, token_count, NULL, 0, logits);
        if (rc != 0) {
            model->gpu_failures++;
            if (g4_gpu_kv_rewind(model, start) != 0 ||
                g4_text_state_publish(model, start) != 0)
                return step_fail("prefill-state-restore", -1, start);
            return step_fail("prefill-gpu-program", -1, start);
        }
        return 0;
    }
    if (model->prefill_detail.enabled) plan.profile = &profile;
    memset(&ops, 0, sizeof ops);
    ops.embed = g4_prefill_embed;
    ops.attention = g4_prefill_attention;
    ops.feed_forward = g4_prefill_ffn;
    ops.release_layer = g4_prefill_release;
    ops.finish = g4_prefill_finish;
    ops.publish_position = g4_prefill_publish;
    rc = salt_text_prefill_execute(
        &plan, &ops, model, start, tokens, token_count, logits);
    if (rc != 0) {
        if (g4_gpu_kv_rewind(model, start) != 0 ||
            g4_text_state_publish(model, start) != 0)
            return step_fail("prefill-state-restore", -1, start);
        return step_fail("prefill-execute", -1, start);
    }
    if (plan.profile)
        g4_emit_prefill_waterfall(
            "GEMMA4_PREFILL_WATERFALL", model, token_count,
            &profile, plan.resource_policy);
    return 0;
}

int salt_gemma4_text_prefill_verify(SaltGemma4Text *model,
                                    const int *tokens, int token_count,
                                    float *position_logits,
                                    size_t logits_stride,
                                    float *final_logits) {
    SaltTextPrefillPlan plan;
    SaltTextPrefillProfile profile;
    SaltTextPrefillOps ops;
    int start, rc;
    if (!model || !tokens || token_count < 1 || !position_logits ||
        logits_stride != G4_VOCAB || !final_logits ||
        model->position < 0 ||
        model->position > model->max_context - token_count)
        return -1;
    start = model->position;
    memset(&model->prefill_detail, 0, sizeof model->prefill_detail);
    model->prefill_detail.enabled = getenv("SALT_WATERFALL") != NULL;
    memset(&plan, 0, sizeof plan);
    if (g4_prefill_recipe(model, &plan) != 0 ||
            token_count > plan.max_batch)
        return -1;
    if (plan.max_batch > model->prefill_capacity)
        plan.max_batch = model->prefill_capacity;
    if (token_count > plan.max_batch) return -1;
    plan.scratch = &model->text_prefill_scratch;
    if (model->prefill_operation_flow || model->full_gpu_intent) {
        plan.transaction_run = g4_prefill_transaction_run;
        plan.transaction_context = model;
    }
    plan.position_logits = position_logits;
    plan.position_logits_stride = logits_stride;
    plan.position_logits_width = G4_VOCAB;
    if (g4_pageable_gpu_prefill_selected(model)) {
        rc = g4_pageable_gpu_prefill(model, &plan, tokens, token_count,
            position_logits, logits_stride, final_logits);
        if (rc != 0) {
            model->gpu_failures++;
            if (g4_gpu_kv_rewind(model, start) != 0 ||
                g4_text_state_publish(model, start) != 0)
                return -1;
        }
        return rc;
    }
    if (model->prefill_detail.enabled) plan.profile = &profile;
    memset(&ops, 0, sizeof ops);
    ops.embed = g4_prefill_embed;
    ops.attention = g4_prefill_attention;
    ops.feed_forward = g4_prefill_ffn;
    ops.release_layer = g4_prefill_release;
    ops.finish = g4_prefill_finish;
    ops.finish_batch = g4_prefill_finish_batch;
    ops.publish_position = g4_prefill_publish;
    rc = salt_text_prefill_execute(
        &plan, &ops, model, start, tokens, token_count, final_logits);
    if (rc != 0 &&
        (g4_gpu_kv_rewind(model, start) != 0 ||
         g4_text_state_publish(model, start) != 0))
        return -1;
    if (rc == 0 && plan.profile)
        g4_emit_prefill_waterfall(
            "GEMMA4_DPR_VERIFY_WATERFALL", model, token_count,
            &profile, plan.resource_policy);
    return rc;
}

int salt_gemma4_text_rollback_position(SaltGemma4Text *model,
                                       int new_position) {
    int old_position;
    if (!model || new_position < 0 || new_position > model->position)
        return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++)
        if (model->layers[layer].shared_position > new_position)
            return -1;
    old_position = model->position;
    if (g4_gpu_kv_rewind(model, new_position) != 0) return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        G4Layer *value = &model->layers[layer];
        size_t start = (size_t)new_position * (size_t)value->kv_dim;
        size_t count = (size_t)(old_position - new_position) *
                       (size_t)value->kv_dim;
        memset(value->key_cache + start, 0, count * sizeof(float));
        memset(value->value_cache + start, 0, count * sizeof(float));
    }
    return g4_text_state_publish(model, new_position);
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
static int g4_text_executor_ready_with_shared_prefix(
        const SaltGemma4Text *model) {
    return model && model->text_program.ready && model->text_executor.program ==
        &model->text_program && model->text_executor.plan &&
        model->text_executor.ops && model->text_executor.context;
}
#if 0
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_text_executor_ready(const SaltGemma4Text *model) {
    return model && model->layers[0].shared_position == 0 &&
        model->text_program.ready && model->text_executor.program ==
        &model->text_program && model->text_executor.plan &&
        model->text_executor.ops && model->text_executor.context;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */

int salt_gemma4_text_target_block(SaltGemma4Text *model,
                                  const int *candidate_tokens,
                                  int candidate_count,
                                  const float *parent_logits,
                                  SaltTextVerifyResult *result) {
    SaltTextTargetBlock block;
    SaltTextVerifyResult committed;
    uint64_t generation;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !candidate_tokens || candidate_count < 1 ||
        candidate_count > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES ||
        !parent_logits || !result || sizeof(int) != sizeof(int32_t) ||
        !model->text_program.ready || !g4_text_executor_ready(model) ||
        !g4_text_state_aligned(model) ||
        model->position > model->max_context - candidate_count ||
        model->text_kv_state.transition_generation == UINT64_MAX)
        return -1;
    generation = model->text_kv_state.transition_generation + 1u;
    memset(&block, 0, sizeof block);
    block.transition_generation = generation;
    block.source_position = (uint32_t)model->position;
    block.candidate_token_ids = (const int32_t *)(const void *)candidate_tokens;
    block.candidate_count = (uint32_t)candidate_count;
    block.parent_logits = parent_logits;
    memset(&committed, 0, sizeof committed);
    if (salt_text_verify_execute(
            &model->text_program, &model->text_executor,
            &block, &committed) != 0) {
        if (g4_text_state_sync_from_kv(model) != 0)
            return -1;
        *result = committed;
        return -1;
    }
    if (g4_text_state_sync_from_kv(model) != 0) {
        *result = committed;
        return -1;
    }
    if (committed.status != SALT_TEXT_VERIFY_COMMITTED ||
        committed.committed_count != committed.accepted_count ||
        committed.produced_count != committed.accepted_count + 1u ||
        committed.result_position != (uint32_t)model->position ||
        committed.transition_generation != generation) {
        *result = committed;
        return -1;
    }
    *result = committed;
    return 0;
}

static int g4_text_target_frontier_impl(
        SaltGemma4Text *model,
        const SaltTextTargetNode *nodes, int node_count,
        const float *parent_logits, SaltTextVerifyResult *result,
        int force_full_frontier) {
    SaltTextTargetFrontierBlock block;
    SaltTextVerifyResult committed;
    uint64_t generation;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !nodes || node_count < 1 ||
        node_count > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES ||
        !parent_logits || !result || !model->text_program.ready ||
        !g4_text_executor_ready(model) || !g4_text_state_aligned(model) ||
        model->text_kv_state.transition_generation == UINT64_MAX)
        return -1;
    generation = model->text_kv_state.transition_generation + 1u;
    memset(&block, 0, sizeof block);
    block.source_position = (uint32_t)model->position;
    block.frontier.nodes = nodes;
    block.frontier.node_count = (uint32_t)node_count;
    block.frontier.generation = generation;
    block.parent_logits = parent_logits;
    block.force_full_frontier = force_full_frontier ? 1u : 0u;
    memset(&committed, 0, sizeof committed);
    if (salt_text_verify_frontier_execute(
            &model->text_program, &model->text_executor,
            &block, &committed) != 0) {
        if (g4_text_state_sync_from_kv(model) != 0)
            return -1;
        *result = committed;
        return -1;
    }
    if (g4_text_state_sync_from_kv(model) != 0) {
        *result = committed;
        return -1;
    }
    if (committed.status != SALT_TEXT_VERIFY_COMMITTED ||
        committed.committed_count != committed.accepted_count ||
        committed.produced_count != committed.accepted_count + 1u ||
        committed.result_position != (uint32_t)model->position ||
        committed.transition_generation != generation ||
        (committed.accepted_count != 0u &&
         (committed.winning_node_index >= (uint32_t)node_count ||
          committed.winning_node_id !=
              nodes[committed.winning_node_index].node_id))) {
        *result = committed;
        return -1;
    }
    *result = committed;
    return 0;
}

int salt_gemma4_text_target_frontier(
        SaltGemma4Text *model,
        const SaltTextTargetNode *nodes, int node_count,
        const float *parent_logits, SaltTextVerifyResult *result) {
    return g4_text_target_frontier_impl(
        model, nodes, node_count, parent_logits, result, 0);
}

int salt_gemma4_text_target_cascade(
        SaltGemma4Text *model,
        const int32_t *seed_token_ids, int tile_count, int frontier_width,
        const float *parent_logits,
        SaltTextTargetNode *node_scratch, int node_scratch_capacity,
        SaltTextVerifyResult *result) {
    SaltTextTargetCascadeBlock block;
    SaltTextVerifyResult committed;
    uint64_t generation;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !seed_token_ids || tile_count < 1 ||
        frontier_width < 1 ||
        frontier_width > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES ||
        !parent_logits || !node_scratch ||
        node_scratch_capacity < frontier_width || !result ||
        !model->text_program.ready || !g4_text_executor_ready(model) ||
        !g4_text_state_aligned(model) ||
        model->position > model->max_context - tile_count ||
        model->text_kv_state.transition_generation == UINT64_MAX ||
        (uint64_t)(uint32_t)(tile_count - 1) >
            UINT64_MAX - model->text_kv_state.transition_generation - 1u)
        return -1;
    generation = model->text_kv_state.transition_generation + 1u;
    memset(&block, 0, sizeof block);
    block.first_transition_generation = generation;
    block.source_position = (uint32_t)model->position;
    block.seed_token_ids = seed_token_ids;
    block.tile_count = (uint32_t)tile_count;
    block.frontier_width = (uint32_t)frontier_width;
    block.parent_logits = parent_logits;
    block.node_scratch = node_scratch;
    block.node_scratch_capacity = (uint32_t)node_scratch_capacity;
    memset(&committed, 0, sizeof committed);
    if (salt_text_verify_cascade_execute(
            &model->text_program, &model->text_executor,
            &block, &committed) != 0) {
        if (g4_text_state_sync_from_kv(model) != 0)
            return -1;
        *result = committed;
        return -1;
    }
    if (g4_text_state_sync_from_kv(model) != 0) {
        *result = committed;
        return -1;
    }
    if (committed.status != SALT_TEXT_VERIFY_COMMITTED ||
        committed.committed_count != committed.accepted_count ||
        committed.produced_count != committed.accepted_count + 1u ||
        committed.result_position != (uint32_t)model->position) {
        *result = committed;
        return -1;
    }
    *result = committed;
    return 0;
}

/* Return 0 after executing one valid B=N×F=nW frontier, 1 when the shape is
 * not an active NFQ block for this worker budget, and -1 on failure. */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 1
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
static int g4_text_target_nfq_execute(
        SaltGemma4Text *model, const int32_t *route_token_ids,
        uint32_t candidate_count,
        const float *parent_logits, SaltTextVerifyResult *result) {
    const SaltTextTargetPolicy *policy;
    SaltAreaNfqPlan plan;
    uint32_t route_ids[SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    uint64_t generation, width;
    uint32_t processed = 0, retired = 0, scheduler_workers;
    uint32_t sequence_tiles, route_count;
    int execute_rc;
    if (!model || !route_token_ids || !parent_logits || !result)
        return -1;
    policy = &model->target_policy;
    if (!policy->ready || policy->target_rows == 0 ||
        policy->route_count == 0 ||
        candidate_count != policy->target_rows * policy->route_count ||
        model->target_area_active_workers < 1 ||
        model->target_area_active_workers > 32)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* HIP realizes only the verified causal TARGET trajectory. Proposal-level
     * NFQ remains the zero-model-work check in salt_text_verify_route_execute. */
    if (salt_gpu_rocm_present()) return 1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    sequence_tiles = policy->target_rows;
    route_count = policy->route_count;
    width = (uint64_t)sequence_tiles * route_count;
    scheduler_workers = (uint32_t)model->target_area_active_workers;
    if (!model->full_gpu_intent && width < scheduler_workers)
        scheduler_workers = 1u;
    if ((model->full_gpu_intent &&
            (!g4_target_gpu_program(model) || !model->text_gpu.backend_ops ||
             !model->text_gpu.backend_ops->begin_frontier)) ||
        width < scheduler_workers || width % scheduler_workers != 0u ||
        width > SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES)
        return 1;
    generation = model->text_kv_state.transition_generation + 1u;
    for (uint32_t route = 0; route < route_count; route++) {
        route_ids[route] = route;
        for (uint32_t sequence = 0; sequence < sequence_tiles; sequence++) {
            uint32_t index = route * sequence_tiles + sequence;
            model->target_nfq_nodes[index] = (SaltTextTargetNode) {
                route_token_ids[index], index,
                sequence == 0u ? SALT_TEXT_TARGET_NO_PARENT : index - 1u,
                sequence, index, 0u,
            };
        }
    }
    plan = (SaltAreaNfqPlan) {
        scheduler_workers,
        sequence_tiles, route_count, policy->queue_length,
    };
    if (salt_area_nfq_bind(&model->target_nfq,
            model->target_nfq_routes,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES,
            model->target_nfq_items,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES * 8u) != 0 ||
        salt_area_nfq_start(&model->target_nfq, &plan,
            generation, generation - 1u, (uint32_t)model->position,
            route_ids, route_count) != 0)
        return -1;
    execute_rc = g4_text_target_frontier_impl(
        model, model->target_nfq_nodes, (int)width,
        parent_logits, result, 1);
    if (execute_rc != 0) {
        (void)salt_area_nfq_cancel(
            &model->target_nfq, generation, generation - 1u, &retired);
        return -1;
    }
    while (processed < (uint32_t)width) {
        uint32_t ready_count = 0;
        if (salt_area_nfq_take_ready(&model->target_nfq,
                generation, generation - 1u, model->target_nfq_ready, 32u,
                &ready_count) != 0 || ready_count == 0u)
            return -1;
        for (uint32_t ready = 0; ready < ready_count; ready++) {
            uint32_t item_index = model->target_nfq_ready[ready];
            if (item_index >= model->target_nfq.item_count ||
                model->target_nfq.items[item_index].queue_index != 0u ||
                salt_area_nfq_complete(&model->target_nfq,
                    generation, generation - 1u, item_index,
                    SALT_AREA_WFQ_CONTINUE, &retired) != 0)
                return -1;
            processed++;
        }
    }
    if (result->accepted_count == 0u) {
        if (salt_area_nfq_cancel(&model->target_nfq,
                generation, generation - 1u, &retired) != 0)
            return -1;
    } else {
        uint32_t winner = result->winning_node_index;
        if (winner >= width || salt_area_nfq_resolve(&model->target_nfq,
                generation, generation - 1u,
                winner / sequence_tiles, winner % sequence_tiles, 0u,
                &retired) != 0)
            return -1;
    }
    result->backend.production_frontier_sessions = 1u;
    result->backend.production_frontier_tasks_queued =
        (uint32_t)width * plan.queue_length;
    result->backend.production_frontier_tasks_executed = (uint32_t)width;
    result->backend.production_frontier_helper_executions = (uint32_t)width;
    result->backend.production_frontier_queued_cancellations = retired;
    result->backend.production_frontier_peak_ready_depth =
        scheduler_workers;
    if (model->target_policy.matrix_flow)
        model->target_matrix_flow_sessions++;
    return 0;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
static int g4_text_target_nfq_execute(
        SaltGemma4Text *model, const int32_t *route_token_ids,
        uint32_t candidate_count,
        const float *parent_logits, SaltTextVerifyResult *result) {
    const SaltTextTargetPolicy *policy;
    SaltTextTargetNfqBlock block;
    uint32_t route_ids[SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    uint32_t scheduler_workers, route_width;
    int rc;
    if (!model || !route_token_ids || !parent_logits || !result)
        return -1;
    policy = &model->target_policy;
    route_width = policy->sequence_tiles * policy->route_count;
    if (!policy->ready || candidate_count != policy->candidate_count ||
        policy->candidate_count != route_width * policy->queue_length ||
        policy->sequence_tiles == 0 || policy->route_count == 0 ||
        model->target_area_active_workers < 1 ||
        model->target_area_active_workers > 32)
        return -1;
    scheduler_workers = (uint32_t)model->target_area_active_workers;

    if ((model->full_gpu_intent &&
            (!g4_target_gpu_program(model) || !model->text_gpu.backend_ops ||
             !model->text_gpu.backend_ops->begin_frontier)) ||
        route_width < scheduler_workers ||
        route_width % scheduler_workers != 0u)
        return -1;
    if (salt_area_nfq_bind(&model->target_nfq,
            model->target_nfq_routes,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES,
            model->target_nfq_items,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES * 8u) != 0)
        return -1;
    memset(&block, 0, sizeof block);
    block.route.transition_generation =
        model->text_kv_state.transition_generation + 1u;
    block.route.source_position = (uint32_t)model->position;
    block.route.route_token_ids = route_token_ids;
    block.route.sequence_tiles = policy->sequence_tiles;
    block.route.route_count = policy->route_count;
    block.route.parent_logits = parent_logits;

    block.plan = (SaltAreaNfqPlan) {
        scheduler_workers, policy->sequence_tiles,
        policy->route_count, policy->queue_length,
    };
    block.matrix = &model->target_nfq;
    block.nodes = model->target_nfq_nodes;
    block.node_capacity = SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES;
    block.route_ids = route_ids;
    block.route_capacity = SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES;
    block.ready_items = model->target_nfq_ready;
    block.ready_capacity = 32u;
    rc = salt_text_verify_nfq_execute(
        &model->text_program, &model->text_executor, &block, result);
    if (g4_text_state_sync_from_kv(model) != 0 || rc != 0)
        return -1;
    if (result->status != SALT_TEXT_VERIFY_COMMITTED ||
        result->committed_count != result->accepted_count ||
        result->produced_count != result->accepted_count + 1u ||
        result->result_position != (uint32_t)model->position ||
        result->transition_generation != block.route.transition_generation ||
        (result->accepted_count != 0u &&
         (result->winning_node_index >= route_width ||
          result->winning_node_id >= candidate_count)))
        return -1;
    if (policy->matrix_flow) model->target_matrix_flow_sessions++;
    return 0;
}
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */

typedef struct G4TextTargetRouteFlowCall {
    SaltGemma4Text *model;
    const SaltTextTargetRouteBlock *block;
    SaltTextVerifyResult *result;
} G4TextTargetRouteFlowCall;

static SaltAreaFrontier *g4_text_target_area_frontier(
        SaltGemma4Text *model) {
    if (!model) return NULL;
    if (g4_target_gpu_program(model))
        return model->text_gpu.ready ? &model->text_gpu.area_frontier : NULL;
    return model->text_cpu.ready ? &model->text_cpu.area_frontier : NULL;
}

static int g4_text_target_wfq_prepare(
        SaltGemma4Text *model, uint64_t generation, uint32_t route_count) {
    SaltAreaFrontier *frontier = g4_text_target_area_frontier(model);
    SaltAreaWfqPlan plan;
    uint32_t seeds[SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    if (!model || !frontier || !frontier->tasks || generation == 0 ||
        route_count == 0 ||
        route_count > (uint32_t)model->target_area_active_workers ||
        route_count > SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES ||
        salt_area_frontier_reset(frontier, generation) != 0)
        return -1;
    for (uint32_t branch = 0; branch < route_count; branch++) {
        uint32_t route = route_count - 1u - branch;
        seeds[branch] = route;
        for (uint32_t queue = 0; queue < 2u; queue++) {
            SaltAreaTask task;
            memset(&task, 0, sizeof task);
            task.generation = generation;
            task.node_id = route;
            task.parent_id = UINT32_MAX;
            task.phase = SALT_TEXT_CELL_ROUTER_TOPK;
            task.expert_id = -1;
            task.job_index = route;
            task.candidate_first = route;
            task.candidate_count = 1u;
            task.output_first = queue;
            task.output_count = 1u;
            task.state = SALT_AREA_TASK_QUEUED;
            if (salt_area_frontier_append(frontier, &task) != 0) return -1;
        }
    }
    memset(&plan, 0, sizeof plan);
    plan.workers = (uint32_t)model->target_area_active_workers;
    plan.frontier_width = route_count;
    plan.queue_length = 2u;
    return salt_area_wfq_bind(
            &model->target_wfq, model->target_wfq_branches,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES,
            model->target_wfq_items,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES,
            model->target_wfq_results,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES,
            frontier->tasks, frontier->capacity) != 0
        ? -1 : salt_area_wfq_start(
            &model->target_wfq, &plan, generation, seeds, route_count);
}

static int g4_text_target_wfq_pool_run(
        void *opaque, int active_workers,
        void (*worker)(int worker, void *task), void *task) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    return model && model->compute_pool_ready
        ? salt_attn_pool_run_n(
            &model->compute_pool, active_workers, worker, task)
        : -1;
}

static int g4_text_target_wfq_execute(
        void *opaque, SaltAreaWfqRuntime *runtime,
        SaltAreaWfqItemExecute item_execute, void *item_context,
        SaltAreaWfqResult *result) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !runtime || !item_execute || !item_context || !result)
        return -1;
    if (model->compute_pool.aflow_running)
        return salt_area_wfq_execute(
            runtime, g4_text_target_wfq_pool_run, model,
            item_execute, item_context, result);
    return salt_attn_pool_wfq_execute(
        &model->compute_pool, runtime, item_execute, item_context, result);
}

static int g4_text_target_route_flow_call(void *opaque) {
    G4TextTargetRouteFlowCall *call =
        (G4TextTargetRouteFlowCall *)opaque;
    return call && call->model && call->block && call->result
        ? salt_text_verify_route_execute(
            &call->model->text_program, &call->model->text_executor,
            call->block, call->result)
        : -1;
}

int salt_gemma4_text_target_route(
        SaltGemma4Text *model,
        const int32_t *route_token_ids, int candidate_count,
        const float *parent_logits, SaltTextVerifyResult *result) {
    const SaltTextTargetPolicy *policy;
    SaltTextTargetRouteBlock block;
    SaltTextVerifyResult committed;
    G4TextTargetRouteFlowCall flow;
    uint64_t generation;
    int execute_rc, opened_flow = 0, wfq_engaged = 0, nfq_rc;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !route_token_ids || candidate_count < 1 ||
        !parent_logits || !result || !model->text_program.ready ||
        !g4_text_executor_ready(model) || !g4_text_state_aligned(model) ||
        !model->target_policy.ready ||
        model->text_kv_state.transition_generation == UINT64_MAX)
        return -1;
    policy = &model->target_policy;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 1
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if ((uint32_t)candidate_count !=
            policy->target_rows * policy->route_count ||
        policy->target_rows >
            (uint32_t)(model->max_context - model->position))
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
    if ((uint32_t)candidate_count != policy->candidate_count ||
        policy->candidate_count !=
            policy->sequence_tiles * policy->route_count * policy->queue_length ||
        model->position >= model->max_context)
        return -1;
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    /* A selected causal trajectory belongs to the portable linear verifier.
     * Worker width must not redirect X rows into model-owned NFQ bookkeeping.
     * Keep the explicit multi-route frontier control separate. */
    if (policy->route_count > 1u) {
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    nfq_rc = g4_text_target_nfq_execute(
        model, route_token_ids, (uint32_t)candidate_count,
        parent_logits, result);
    if (nfq_rc <= 0) return nfq_rc;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    generation = model->text_kv_state.transition_generation + 1u;
    if (policy->route_count <=
            (uint32_t)model->target_area_active_workers) {
        if (g4_text_target_wfq_prepare(
                model, generation, policy->route_count) != 0)
            return -1;
        wfq_engaged = 1;
    }
    memset(&block, 0, sizeof block);
    block.transition_generation = generation;
    block.source_position = (uint32_t)model->position;
    block.route_token_ids = route_token_ids;
    block.sequence_tiles = policy->target_rows;
    block.route_count = policy->route_count;
    block.parent_logits = parent_logits;
    if (wfq_engaged) {
        block.wfq = &model->target_wfq;
        block.wfq_execute = g4_text_target_wfq_execute;
        block.wfq_context = model;
    }
    memset(&committed, 0, sizeof committed);
    flow = (G4TextTargetRouteFlowCall) { model, &block, &committed };
    if (model->target_policy.matrix_flow &&
        !model->compute_pool.aflow_running) {
        opened_flow = 1;
        execute_rc = salt_attn_pool_flow_run(
            &model->compute_pool, g4_text_target_route_flow_call, &flow);
    } else {
        execute_rc = g4_text_target_route_flow_call(&flow);
    }
    if (execute_rc != 0) {
        if (g4_text_state_sync_from_kv(model) != 0)
            return -1;
        *result = committed;
        return -1;
    }
    if (opened_flow) model->target_matrix_flow_sessions++;
    if (g4_text_state_sync_from_kv(model) != 0) {
        *result = committed;
        return -1;
    }
    if (committed.status != SALT_TEXT_VERIFY_COMMITTED ||
        committed.committed_count != committed.accepted_count ||
        committed.produced_count != committed.accepted_count + 1u ||
        committed.result_position != (uint32_t)model->position ||
        committed.transition_generation != generation) {
        *result = committed;
        return -1;
    }
    *result = committed;
    return 0;
}

typedef struct G4TextTargetEpochColdContext {
    SaltGemma4Text *model;
} G4TextTargetEpochColdContext;

static int g4_text_target_epoch_cold_search(
        void *opaque, const SaltTextTargetRouteBlock *route,
        SaltTextVerifyResult *result) {
    G4TextTargetEpochColdContext *context =
        (G4TextTargetEpochColdContext *)opaque;
    if (!context || !context->model || !route || !result ||
        route->sequence_tiles > INT_MAX || route->route_count > INT_MAX)
        return -1;
    return salt_gemma4_text_target_route(
        context->model, route->route_token_ids,
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 1
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        (int)(route->sequence_tiles * route->route_count),
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
        (int)context->model->target_policy.candidate_count,
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        route->parent_logits, result);
}

int salt_gemma4_text_target_epoch(
        SaltGemma4Text *model, SaltTextTokenEpochController *controller,
        const int32_t *route_token_ids, int candidate_count,
        const float *parent_logits,
        SaltTextTokenExactLookupCommit exact_lookup_commit,
        void *exact_context, SaltTextTokenEpochResult *result) {
    SaltTextTargetRouteBlock route;
    SaltTextTokenEpochRequest request;
    G4TextTargetEpochColdContext cold;
    uint64_t generation, parent_generation;
    int rc;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !controller || !route_token_ids || candidate_count < 1 ||
        !parent_logits || !exact_lookup_commit || !result ||
        !model->text_program.ready || !g4_text_executor_ready(model) ||
        !g4_text_state_aligned(model) ||
        !model->target_policy.ready ||
        (uint32_t)candidate_count != model->target_policy.target_rows *
            model->target_policy.route_count ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#if 1
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        model->target_policy.target_rows >
            (uint32_t)(model->max_context - model->position) ||
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
#else
        model->position >= model->max_context ||
#endif
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
        model->text_kv_state.transition_generation == UINT64_MAX)
        return -1;
    parent_generation = model->text_kv_state.transition_generation;
    generation = parent_generation + 1u;
    memset(&route, 0, sizeof route);
    route.transition_generation = generation;
    route.source_position = (uint32_t)model->position;
    route.route_token_ids = route_token_ids;
    route.sequence_tiles = model->target_policy.target_rows;
    route.route_count = model->target_policy.route_count;
    route.parent_logits = parent_logits;
    memset(&cold, 0, sizeof cold);
    cold.model = model;
    memset(&request, 0, sizeof request);
    request.epoch_generation = generation;
    request.parent_generation = parent_generation;
    request.parent_position = (uint32_t)model->position;
    request.route = &route;
    request.exact_lookup_commit = exact_lookup_commit;
    request.exact_context = exact_context;
    request.cold_search = g4_text_target_epoch_cold_search;
    request.cold_context = &cold;
    rc = salt_text_token_epoch_execute(controller, &request, result);
    if (rc != 0) {
        if (g4_text_state_sync_from_kv(model) != 0)
            return -1;
        return rc;
    }
    if (g4_text_state_sync_from_kv(model) != 0 ||
        result->target.status != SALT_TEXT_VERIFY_COMMITTED ||
        result->target.transition_generation != generation ||
        result->target.result_position != (uint32_t)model->position)
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (result->target.accepted_count && result->target.projection_logits &&
        result->target.projection_rows) {
        SaltTextProjectionWindow *window = &model->target_projection;
        size_t bytes = 0u;
        unsigned char *base = g4_target_canonical_base(model, &bytes);
        if (!base || bytes < model->text_program.layout.total_bytes ||
            result->target.projection_logits != (const float *)(base +
                model->text_program.layout.position_logits))
            return -1;
        window->logits = result->target.projection_logits;
        window->generation = result->target.transition_generation;
        window->position = result->target.result_position;
        window->sequence_tiles = model->target_policy.target_rows;
        window->route_count = model->target_policy.route_count;
        window->accepted_count = result->target.accepted_count;
        window->winning_route = result->target.winning_node_index /
            window->sequence_tiles;
        if (window->winning_route >= window->route_count ||
            result->target.projection_rows !=
                window->sequence_tiles * window->route_count)
            return -1;
    }
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    return 0;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
int salt_gemma4_text_target_epoch_active(
        SaltGemma4Text *model, SaltTextTokenEpochController *controller,
        const int32_t *route_token_ids, int candidate_count,
        const float *parent_logits,
        SaltTextTokenExactLookupCommit exact_lookup_commit,
        void *exact_context, SaltTextTokenEpochResult *result) {
    SaltTextTargetPolicy compiled, active;
    uint32_t sequence_tiles, route_count;
    int rc;
    if (!model || !model->target_policy.ready || candidate_count < 1 ||
        salt_text_target_policy_active_shape(
            &model->target_policy, (uint32_t)candidate_count,
            &sequence_tiles, &route_count) != 0)
        return -1;
    compiled = model->target_policy;
    active = compiled;
    active.target_rows = sequence_tiles;
    active.route_count = route_count;
    active.candidate_count = (uint32_t)candidate_count;
    model->target_policy = active;
    rc = salt_gemma4_text_target_epoch(
        model, controller, route_token_ids, candidate_count,
        parent_logits, exact_lookup_commit, exact_context, result);
    model->target_policy = compiled;
    if (rc == 0 && result->target.accepted_count &&
        result->target.projection_logits &&
        result->target.projection_rows >= sequence_tiles) {
        SaltTextProjectionWindow *window = &model->target_projection;
        size_t bytes = 0u;
        unsigned char *base = g4_target_canonical_base(model, &bytes);
        if (!base || bytes < model->text_program.layout.total_bytes ||
            result->target.projection_logits != (const float *)(base +
                model->text_program.layout.position_logits))
            return -1;
        window->logits = result->target.projection_logits;
        window->generation = result->target.transition_generation;
        window->position = result->target.result_position;
        window->sequence_tiles = sequence_tiles;
        window->accepted_count = result->target.accepted_count;
        window->route_count = result->target.projection_rows / sequence_tiles;
        window->winning_route = window->route_count > 1u ?
            result->target.winning_node_index / sequence_tiles : 0u;
        if (window->route_count > route_count ||
            window->winning_route >= window->route_count ||
            result->target.projection_rows % sequence_tiles != 0u)
            return -1;
    }
    return rc;
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
int salt_gemma4_text_target_policy_get(
        const SaltGemma4Text *model, SaltTextTargetPolicy *policy) {
    if (!model || !policy || !model->target_policy.ready)
        return -1;
    *policy = model->target_policy;
    return 0;
}

/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
int salt_gemma4_text_is_stop_token(int token) {
    return token == 1 || token == 50 || token == 106;
}

static int g4_materialized_argmax(const float *logits) {
    int best = -1;
    float best_value = -INFINITY;
    if (!logits) return -1;
    for (int token = 0; token < G4_VOCAB; token++) {
        float value = logits[token];
        if (!isfinite(value)) return -1;
        if (best < 0 || value > best_value) {
            best = token;
            best_value = value;
        }
    }
    return best;
}

/* One authoritative known-row materializer for B=1 and B=j.  The common
 * startup-compiled text program owns compute, canonical destinations, resource
 * leases, KV publication, and the single completion boundary. */
static int g4_materialize_known_span(
        SaltGemma4Text *model, const int32_t *tokens, uint32_t count,
        SaltTextVerifyResult *result) {
    SaltTextExecuteAllResult executed;
    const float *boundary;
    uint64_t generation;
    int pending, execute_status;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !tokens || count == 0u || count > INT_MAX || !result ||
        !model->text_program.ready || !g4_text_executor_ready(model) ||
        !g4_text_state_aligned(model) ||
        count > model->text_program.maximum_candidates ||
        (uint32_t)model->position >
            model->text_program.maximum_context - count ||
        model->text_kv_state.transition_generation == UINT64_MAX ||
        model->prefill_decode_lookahead.status == SALT_TEXT_LOOKAHEAD_READY)
        return -1;
    for (uint32_t row = 0u; row < count; row++)
        if (tokens[row] < 0 || tokens[row] >= G4_VOCAB)
            return -1;
    /* Ordinary CPU and recipe-selected Metal B1 use the retained token program,
     * not an X=1 target
     * program.  NFQ has already selected the root; this changes only its
     * physical materialization.  head_logits is startup-owned storage. */
    if (count == 1u && (!model->full_gpu_intent ||
            (!salt_gpu_cuda_present() && !salt_gpu_rocm_present() &&
             !model->fine_token_enabled))) {
        uint32_t source = (uint32_t)model->position;
        generation = model->text_kv_state.transition_generation + 1u;
        if (salt_gemma4_text_step(model, tokens[0], model->head_logits) != 0 ||
            model->text_kv_state.position != source + 1u ||
            model->text_kv_state.transition_generation != generation ||
            model->token_transition_generation != generation)
            return -1;
        pending = g4_materialized_argmax(model->head_logits);
        if (pending < 0) return -1;
        result->status = SALT_TEXT_VERIFY_COMMITTED;
        result->accepted_count = 1u;
        result->committed_count = 1u;
        result->produced_count = 2u;
        result->pending_token_id = pending;
        result->result_position = model->text_kv_state.position;
        result->transition_generation = generation;
        result->pending_logits = model->head_logits;
        result->pending_logits_count = G4_VOCAB;
        result->backend.engine_submissions =
            model->token_backend_last.caller_submissions;
        result->backend.completion_fences =
            model->token_backend_last.completion_fences;
        result->backend.internal_dependency_barriers =
            model->token_backend_last.internal_dependency_barriers;
        result->backend.intermediate_host_publications =
            model->token_backend_last.intermediate_host_publications;
        return 0;
    }
    memset(&model->target_projection, 0, sizeof model->target_projection);
    generation = model->text_kv_state.transition_generation + 1u;
    memset(&executed, 0, sizeof executed);
    execute_status = count == 1u ||
            model->text_executor.ops->submit_authoritative_output
        ? salt_text_execute_prefill(&model->text_program,
            &model->text_executor, generation,
            (uint32_t)model->position, tokens, count, 1u, &executed)
        : salt_text_execute_all(&model->text_program,
            &model->text_executor, generation,
            (uint32_t)model->position, tokens, count, &executed);
    if (execute_status != 0 ||
        executed.status != SALT_TEXT_VERIFY_COMMITTED ||
        executed.committed_count != count ||
        executed.result_position != (uint32_t)model->position + count ||
        executed.transition_generation != generation ||
        !executed.view.canonical_base ||
        executed.view.canonical_bytes < model->text_program.layout.total_bytes ||
        executed.backend.engine_submissions != 1u ||
        executed.backend.completion_fences != 1u ||
        executed.backend.intermediate_host_publications != 0u)
        return -1;
    boundary = (const float *)(const void *)(executed.view.canonical_base +
        model->text_program.layout.position_logits) +
        (size_t)(count - 1u) * G4_VOCAB;
    pending = g4_materialized_argmax(boundary);
    if (pending < 0 || g4_text_state_sync_from_kv(model) != 0 ||
        model->position != (int)executed.result_position)
        return -1;
    model->token_transition_generation = executed.transition_generation;
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = count;
    result->committed_count = count;
    result->produced_count = count + 1u;
    result->pending_token_id = pending;
    result->result_position = executed.result_position;
    result->transition_generation = executed.transition_generation;
    result->winning_node_index = count - 1u;
    result->winning_node_id = count - 1u;
    result->pending_logits = boundary;
    result->pending_logits_count = G4_VOCAB;
    result->backend = executed.backend;
    memset(&model->token_backend_last, 0, sizeof model->token_backend_last);
    model->token_backend_last.caller_submissions =
        executed.backend.engine_submissions;
    model->token_backend_last.cells_completed =
        model->text_program.dispatch.cell_count;
    model->token_backend_last.internal_dependency_barriers =
        executed.backend.internal_dependency_barriers;
    model->token_backend_last.completion_fences =
        executed.backend.completion_fences;
    model->token_backend_last.intermediate_host_publications =
        executed.backend.intermediate_host_publications;
    model->token_backend_last.final_publications = 1u;
    model->token_backend_last.tentative_write_bytes =
        executed.backend.final_kv_publish_bytes;
    if (getenv("SALT_WATERFALL"))
        fprintf(stderr,
            "GEMMA4_KNOWN_MATERIALIZER source=%u B=%u result=%u "
            "submissions=%u fences=%u intermediate_publish=%u "
            "logits_rows=%u kv_publish_bytes=%llu\n",
            executed.result_position - count, count, executed.result_position,
            executed.backend.engine_submissions,
            executed.backend.completion_fences,
            executed.backend.intermediate_host_publications,
            count == 1u || model->text_executor.ops->submit_authoritative_output
                ? 1u : count,
            (unsigned long long)executed.backend.final_kv_publish_bytes);
    return 0;
}

int salt_gemma4_text_consume_known(
        SaltGemma4Text *model, int token, float *logits) {
    SaltTextVerifyResult materialized;
    int32_t known = token;
    if (model && (!model->full_gpu_intent ||
            (!salt_gpu_cuda_present() && !salt_gpu_rocm_present() &&
             !model->fine_token_enabled)))
        return salt_gemma4_text_step(model, token, logits);
    if (g4_materialize_known_span(model, &known, 1u, &materialized) != 0)
        return -1;
    if (logits)
        memcpy(logits, materialized.pending_logits,
               (size_t)G4_VOCAB * sizeof(float));
    return 0;
}

#if 0 /* Reference-only rollout; production never proposes with model steps. */
static int g4_target_generate_argmax(const float *logits) {
    int best = -1;
    float best_value = -INFINITY;
    if (!logits) return -1;
    for (int token = 0; token < G4_VOCAB; token++) {
        if (!isfinite(logits[token])) return -1;
        if (best < 0 || logits[token] > best_value) {
            best = token;
            best_value = logits[token];
        }
    }
    return best;
}
#endif

static uint64_t g4_target_generate_now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * 1000000000u + (uint64_t)value.tv_nsec;
}

static int g4_target_generate_exact_miss(
        void *opaque, uint64_t epoch_generation,
        uint64_t parent_generation, uint32_t parent_position,
        SaltTextVerifyResult *result) {
    (void)opaque;
    (void)epoch_generation;
    (void)parent_generation;
    (void)parent_position;
    if (!result) return -1;
    memset(result, 0, sizeof *result);
    return 0;
}

#if 0 /* Reference-only full-model proposal scratch. */
static int g4_target_proposal_rows_begin(
        SaltGemma4Text *model, int source_position, int capacity) {
    unsigned char *base;
    size_t bytes;
    if (!model || source_position < 0 || capacity < 1 ||
        capacity > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES ||
        !(base = g4_target_canonical_base(model, &bytes)) ||
        bytes < model->text_program.layout.total_bytes)
        return -1;
    for (uint32_t index = 0; index < model->text_program.layer_count; index++) {
        const SaltTextCompiledLayer *compiled = &model->text_program.layers[index];
        G4Layer *layer = &model->layers[index];
        size_t row_bytes = (size_t)compiled->kv_width * sizeof(float);
        size_t span;
        if (compiled->kv_width != (uint32_t)layer->kv_dim ||
            (size_t)capacity > SIZE_MAX / row_bytes)
            return -1;
        span = (size_t)capacity * row_bytes;
        if (compiled->tentative_key_offset > bytes ||
            span > bytes - compiled->tentative_key_offset ||
            compiled->tentative_value_offset > bytes ||
            span > bytes - compiled->tentative_value_offset)
            return -1;
        layer->proposal_key_cache =
            (float *)(void *)(base + compiled->tentative_key_offset);
        layer->proposal_value_cache =
            (float *)(void *)(base + compiled->tentative_value_offset);
        layer->proposal_source_position = source_position;
        layer->proposal_capacity = capacity;
        layer->proposal_active = 1;
        memset(layer->proposal_key_cache, 0, span);
        memset(layer->proposal_value_cache, 0, span);
    }
    return 0;
}

static void g4_target_proposal_rows_end(SaltGemma4Text *model) {
    if (!model) return;
    for (int index = 0; index < G4_LAYERS; index++) {
        G4Layer *layer = &model->layers[index];
        layer->proposal_active = 0;
        layer->proposal_source_position = 0;
        layer->proposal_capacity = 0;
        layer->proposal_key_cache = NULL;
        layer->proposal_value_cache = NULL;
    }
}
#endif


static int g4_scheduler_stop(void *opaque, int32_t token) {
    (void)opaque;
    return salt_gemma4_text_is_stop_token((int)token);
}

static int g4_scheduler_nfq_search(
        void *opaque, const SaltTextTargetRouteBlock *route,
        SaltTextVerifyResult *result) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    SaltTextTargetNfqBlock block;
    uint32_t route_ids[SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES];
    if (!model || !route || !result ||
        salt_area_nfq_bind(&model->target_nfq,
            model->target_nfq_routes,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES,
            model->target_nfq_items,
            SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES * 8u) != 0)
        return -1;
    memset(&block, 0, sizeof block);
    block.route = *route;
    block.plan = (SaltAreaNfqPlan) {
        (uint32_t)model->target_area_active_workers,
        route->sequence_tiles, route->route_count,
        model->target_policy.queue_length,
    };
    block.matrix = &model->target_nfq;
    block.route_ids = route_ids;
    block.route_capacity = SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES;
    block.ready_items = model->target_nfq_ready;
    block.ready_capacity = 32u;
    return salt_text_verify_nfq_execute(
        &model->text_program, &model->text_executor, &block, result);
}

static int g4_scheduler_select_proposal(
        void *opaque, SaltTextTokenEpochController *controller,
        const int32_t *tokens, uint32_t count, const float *parent,
        SaltTextTokenEpochResult *result) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    SaltTextTargetRouteBlock route;
    SaltTextTokenEpochRequest request;
    int rc;
    if (!model || !controller || !tokens || !parent || !result ||
        !model->target_policy.ready || !g4_text_state_aligned(model) ||
        count != model->target_policy.candidate_count *
            model->target_policy.queue_length ||
        model->text_kv_state.transition_generation == UINT64_MAX)
        return -1;
    memset(&route, 0, sizeof route);
    route.transition_generation = model->text_kv_state.transition_generation + 1u;
    route.source_position = (uint32_t)model->position;
    route.route_token_ids = tokens;
    route.sequence_tiles = model->target_policy.sequence_tiles;
    route.route_count = model->target_policy.route_count;
    route.parent_logits = parent;
    memset(&request, 0, sizeof request);
    request.epoch_generation = route.transition_generation;
    request.parent_generation = model->text_kv_state.transition_generation;
    request.parent_position = route.source_position;
    request.route = &route;
    request.exact_lookup_commit = g4_target_generate_exact_miss;
    request.cold_search = g4_scheduler_nfq_search;
    request.cold_context = model;
    rc = salt_text_token_epoch_execute(controller, &request, result);
    if (rc == 0 && getenv("SALT_WATERFALL"))
        fprintf(stderr, "GEMMA4_NFQ_SELECTION source=%u n=%u f=%u q=%u "
            "queued=%u checked=%u cancelled=%u winner=%u token=%d "
            "model_submissions=%u committed=%u\n",
            route.source_position, route.sequence_tiles, route.route_count,
            model->target_policy.queue_length,
            result->target.backend.production_frontier_tasks_queued,
            result->target.backend.production_frontier_tasks_executed,
            result->target.backend.production_frontier_queued_cancellations,
            result->target.winning_node_index, result->target.pending_token_id,
            result->target.backend.engine_submissions,
            result->target.committed_count);
    return rc;
}

static int g4_scheduler_target(
        void *opaque, SaltTextTokenEpochController *controller,
        const int32_t *tokens, uint32_t count, const float *parent,
        SaltTextTokenEpochResult *result) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (!model || !controller || !tokens || !parent || !result ||
        count == 0u || count > INT_MAX)
        return -1;
    return salt_gemma4_text_target_epoch_active(
        model, controller, tokens, (int)count,
        parent, g4_target_generate_exact_miss, NULL, result);
}

#if 0 /* Isolated control; retain for comparison, not normal dispatch. */
/* Normal history/projection candidates are proposals, not exact transition
 * evidence.  Horizontal NFQ has proved only the parent-selected root, so this
 * provider fails closed to B1.  A future authenticated edge provider may widen
 * B without changing the materializer. */
static int g4_scheduler_prove_prefix(
        void *opaque, const int32_t *tokens, uint32_t count,
        uint64_t parent_generation, uint32_t parent_position,
        uint32_t *checked_count, uint32_t *proven_count) {
    SaltGemma4Text *model = (SaltGemma4Text *)opaque;
    if (checked_count) *checked_count = 0u;
    if (proven_count) *proven_count = 0u;
    if (!model || !tokens || count == 0u || !checked_count || !proven_count ||
        tokens[0] < 0 || tokens[0] >= G4_VOCAB ||
        model->text_kv_state.transition_generation != parent_generation ||
        model->text_kv_state.position != parent_position ||
        model->position < 0 || (uint32_t)model->position != parent_position)
        return -1;
    *proven_count = 1u;
    *checked_count = count == 1u ? 1u : 2u;
    return 0;
}

static int g4_scheduler_materialize_known(
        void *opaque, const int32_t *tokens, uint32_t count,
        SaltTextVerifyResult *result) {
    return g4_materialize_known_span(
        (SaltGemma4Text *)opaque, tokens, count, result);
}
#endif

static uint64_t g4_scheduler_clock(void *opaque) {
    (void)opaque;
    return g4_target_generate_now_ns();
}

int salt_gemma4_text_scheduler_binding(
        SaltGemma4Text *model, SaltTextGenerationBinding *binding,
        SaltTextTokenEpochController **controller) {
    if (!model || !binding || !controller || !model->text_program.ready ||
        !model->target_policy.ready || !g4_text_state_aligned(model)) return -1;
    memset(binding, 0, sizeof *binding);
    binding->program = &model->text_program;
    binding->policy = &model->target_policy;
    binding->projection = &model->target_projection;
    binding->context = model;
    binding->is_stop = g4_scheduler_stop;

    binding->select_proposal = g4_scheduler_select_proposal;
    binding->target = g4_scheduler_target;
    /* Isolation candidate: preserve the known-prefix implementation above,
     * but let the existing target verifier evaluate tentative proposals. */
    binding->prove_prefix = NULL;
    binding->materialize_known = NULL;
    binding->target_sublane_matrix = &model->target_nfq;
    binding->target_sublane_ready_items = model->target_nfq_ready;
    binding->target_sublane_ready_capacity = 32u;
    *controller = &model->scheduler_controller;
    return 0;
}

int salt_gemma4_text_target_generate(
        SaltGemma4Text *model, float *logits, int maximum_tokens,
        float *proposal_logits, SaltGemma4TargetGenerateResult *result) {
    SaltTextGenerationBinding binding;
    SaltTextTokenEpochController *controller;
    if (maximum_tokens < 1 || salt_gemma4_text_scheduler_binding(
            model, &binding, &controller) != 0 || controller->schedule_active)
        return -1;
    if (!controller->ready && salt_text_token_epoch_init(controller) != 0)
        return -1;
    return salt_text_generate_candidates(controller, &binding, logits,
        (uint32_t)maximum_tokens, proposal_logits, g4_scheduler_clock, model, result);
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
int salt_gemma4_text_target_progressive(
        SaltGemma4Text *model,
        const int *candidate_tokens,
        int candidate_count,
        const float *parent_logits,
        int seed_candidates,
        int maximum_tile_candidates,
        SaltTextProgressiveResult *result) {
    SaltTextProgressivePlan plan;
    SaltTextProgressiveResult committed;
    uint64_t generation;
    if (result) memset(result, 0, sizeof *result);
    if (!model || !candidate_tokens || candidate_count < 1 ||
        candidate_count > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES ||
        !parent_logits || seed_candidates < 1 ||
        maximum_tile_candidates < seed_candidates ||
        maximum_tile_candidates > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES ||
        !result || sizeof(int) != sizeof(int32_t) ||
        !model->text_program.ready || !g4_text_executor_ready(model) ||
        !g4_text_state_aligned(model) ||
        model->position > model->max_context - candidate_count ||
        model->text_kv_state.transition_generation == UINT64_MAX)
        return -1;
    generation = model->text_kv_state.transition_generation + 1u;
    plan.seed_candidates = (uint32_t)seed_candidates;
    plan.maximum_tile_candidates = (uint32_t)maximum_tile_candidates;
    memset(&committed, 0, sizeof committed);
    if (salt_text_verify_progressive_execute(
            &model->text_program, &model->text_executor,
            generation, (uint32_t)model->position,
            (const int32_t *)(const void *)candidate_tokens,
            (uint32_t)candidate_count, parent_logits, &plan, &committed) != 0) {
        if (g4_text_state_sync_from_kv(model) != 0)
            return -1;
        *result = committed;
        return -1;
    }
    if (g4_text_state_sync_from_kv(model) != 0) {
        *result = committed;
        return -1;
    }
    *result = committed;
    return 0;
}

typedef struct G4ImagePrefillRun {
    SaltGemma4Text *model;
    const int *tokens;
    const unsigned char *mm_token_type;
    int token_count;
    const float *image_features;
    int image_feature_tokens;
    int image_first;
    int serial_proof;
    float *logits;
    float *states_a;
    float *states_b;
    float *q_all;
    unsigned char *full_mask;
    unsigned char *sliding_mask;
    unsigned char *valid;
    int *block_ids;
    float *current;
    float *next;
} G4ImagePrefillRun;

typedef struct G4ImagePrefillTransaction {
    const SaltTensorProgramCallbacks *program;
    G4ImagePrefillRun *run;
    uint32_t *executed;
} G4ImagePrefillTransaction;

static int g4_image_prefill_transaction(void *opaque) {
    G4ImagePrefillTransaction *call = (G4ImagePrefillTransaction *)opaque;
    return call && call->program && call->run && call->executed
        ? salt_tensor_program_execute(
              call->program, call->run, call->executed)
        : -1;
}

static int g4_image_prefill_node(void *opaque, uint32_t node) {
    G4ImagePrefillRun *run = (G4ImagePrefillRun *)opaque;
    SaltGemma4Text *model;
    if (!run || !(model = run->model)) return -1;
    if (node == 0u) {
        size_t mask_count = (size_t)run->token_count * run->token_count;
        memset(run->valid, 1, (size_t)run->token_count);
        if (salt_gemma4_attention_masks(run->full_mask, run->sliding_mask,
                mask_count, run->mm_token_type, run->valid,
                run->token_count, G4_SLIDING_WINDOW,
                run->block_ids, (size_t)run->token_count) != 0)
            return step_fail("image-prefill-masks", -1, run->token_count);
        for (int token = 0; token < run->token_count; token++)
            if (embedding_row(&model->embedding, run->tokens[token],
                    run->states_a + (size_t)token * G4_HIDDEN) != 0)
                return step_fail("image-prefill-embedding", -1, token);
        if (salt_gemma4_scatter_image_features(run->states_a, run->tokens,
                run->token_count, G4_HIDDEN, G4_IMAGE_TOKEN,
                run->image_features, run->image_feature_tokens) != 0)
            return step_fail("image-prefill-scatter", -1, run->image_first);
        run->current = run->states_a;
        run->next = run->states_b;
        return 0;
    }
    if (node <= G4_LAYERS) {
        int layer_index = (int)node - 1;
        G4Layer *layer = &model->layers[layer_index];
        const unsigned char *allow = layer->full_attention
            ? run->full_mask : run->sliding_mask;
        if (prefill_attention_layer(model, layer, run->current,
                run->token_count, allow, run->q_all, run->next,
                run->serial_proof) != 0) {
            release_layer_residency(layer);
            return step_fail("image-prefill-attention", layer_index,
                             model->position);
        }
        if (run->serial_proof) {
            for (int token = 0; token < run->token_count; token++) {
                model->position = token;
                if (feed_forward(model, layer, layer_index,
                        run->next + (size_t)token * G4_HIDDEN,
                        run->current + (size_t)token * G4_HIDDEN, 0) != 0) {
                    release_layer_residency(layer);
                    return step_fail("image-prefill-feed-forward-proof",
                                     layer_index, token);
                }
            }
        } else {
            model->position = run->token_count - 1;
            if (feed_forward_batch(model, layer, layer_index,
                    run->next, run->token_count, run->current) != 0) {
                release_layer_residency(layer);
                return step_fail("image-prefill-feed-forward-batch",
                                 layer_index, model->position);
            }
        }
        if (!model->prefill_retain_layers) release_layer_residency(layer);
        return 0;
    }
    if (node == G4_LAYERS + 1u) {
        if (salt_gemma4_rmsnorm(model->final_state,
                run->current + (size_t)(run->token_count - 1) * G4_HIDDEN,
                model->final_norm.values, G4_HIDDEN, G4_EPS, 1) != 0)
            return step_fail("image-prefill-final-norm", G4_LAYERS,
                             run->token_count - 1);
        if (q_matvec_operation(model, &model->embedding,
                model->final_state, run->logits) != 0) {
            release_q_residency(&model->embedding);
            return step_fail("image-prefill-head", G4_LAYERS,
                             run->token_count - 1);
        }
        if (!model->prefill_retain_layers)
            release_q_residency(&model->embedding);
        if (salt_gemma4_softcap(run->logits, G4_VOCAB, G4_LOGIT_CAP) != 0) {
            if (model->prefill_retain_layers)
                release_q_residency(&model->embedding);
            return step_fail("image-prefill-final", G4_LAYERS,
                             run->token_count - 1);
        }
        return 0;
    }
    if (node == G4_LAYERS + 2u)
        return g4_prefill_publish(model, run->token_count) == 0 ? 0 :
            step_fail("image-prefill-state-publish", G4_LAYERS,
                      run->token_count);
    return -1;
}

int salt_gemma4_text_prefill_image(SaltGemma4Text *model,
                                   const int *tokens,
                                   const unsigned char *mm_token_type,
                                   int token_count,
                                   const float *image_features,
                                   int image_feature_tokens,
                                   float *logits) {
    float *states_a = NULL, *states_b = NULL, *q_all = NULL;
    unsigned char *full_mask = NULL, *sliding_mask = NULL, *valid = NULL;
    int *block_ids = NULL;
    size_t state_count, q_count;
    int image_first = -1, image_last = -1, image_count = 0;
    int rc = -1, serial_proof, retain_layers;
    const char *prefill_mode;
    G4ImagePrefillRun run;
    G4ImagePrefillTransaction transaction;
    SaltTensorProgramCallbacks program;
    uint32_t executed = 0;

    if (!model || !tokens || !mm_token_type || !image_features || !logits ||
        token_count < 1 || token_count > model->max_context ||
        token_count > model->operation_capacity ||
        image_feature_tokens < 1 || image_feature_tokens > token_count ||
        model->position != 0)
        return step_fail("image-prefill-arguments", -1,
                         model ? model->position : -1);
    prefill_mode = getenv("SALT_PREFILL_CHUNK");
    if (!prefill_mode ||
        (strcmp(prefill_mode, "0") != 0 && strcmp(prefill_mode, "1") != 0))
        return step_fail("image-prefill-engine-config", -1, model->position);
    if (g4_recipe_int("SALT_PREFILL_RETAIN_LAYERS", 0, 1,
            &retain_layers) != 0)
        return step_fail("image-prefill-engine-config", -1, model->position);
    model->prefill_retain_layers = retain_layers;
    serial_proof = strcmp(prefill_mode, "0") == 0;
    for (int t = 0; t < token_count; t++) {
        if (tokens[t] < 0 || tokens[t] >= G4_VOCAB ||
            mm_token_type[t] > 1)
            return step_fail("image-prefill-token", -1, t);
        if (mm_token_type[t] == 1) {
            if (tokens[t] != G4_IMAGE_TOKEN)
                return step_fail("image-prefill-placeholder", -1, t);
            if (image_first < 0) image_first = t;
            image_last = t;
            image_count++;
        } else if (tokens[t] == G4_IMAGE_TOKEN) {
            return step_fail("image-prefill-type", -1, t);
        }
    }
    if (image_count != image_feature_tokens || image_first < 1 ||
        image_last + 1 >= token_count ||
        image_last - image_first + 1 != image_count ||
        tokens[image_first - 1] != G4_BOI_TOKEN ||
        tokens[image_last + 1] != G4_EOI_TOKEN)
        return step_fail("image-prefill-layout", -1, image_first);
    for (size_t i = 0;
         i < (size_t)image_feature_tokens * G4_HIDDEN; i++)
        if (!isfinite(image_features[i]))
            return step_fail("image-prefill-feature", -1, image_first);

    if ((size_t)token_count > SIZE_MAX / G4_HIDDEN ||
        (size_t)token_count > SIZE_MAX / (size_t)token_count ||
        (size_t)token_count > SIZE_MAX / G4_MAX_Q)
        return step_fail("image-prefill-size", -1, token_count);
    state_count = (size_t)token_count * G4_HIDDEN;
    q_count = (size_t)token_count * G4_MAX_Q;
    if (state_count > SIZE_MAX / sizeof(float) ||
        q_count > SIZE_MAX / sizeof(float))
        return step_fail("image-prefill-size", -1, token_count);

    states_a = model->operation_states_a;
    states_b = model->operation_states_b;
    q_all = model->operation_q_all;
    full_mask = model->operation_full_mask;
    sliding_mask = model->operation_sliding_mask;
    valid = model->operation_valid;
    block_ids = model->operation_block_ids;
    if (!states_a || !states_b || !q_all || !full_mask || !sliding_mask ||
        !valid || !block_ids) {
        step_fail("image-prefill-allocation", -1, token_count);
        goto done;
    }
    run = (G4ImagePrefillRun) {
        model, tokens, mm_token_type, token_count,
        image_features, image_feature_tokens, image_first,
        serial_proof, logits, states_a, states_b, q_all,
        full_mask, sliding_mask, valid, block_ids,
        NULL, NULL,
    };
    program = (SaltTensorProgramCallbacks) {
        G4_LAYERS + 3u, g4_image_prefill_node,
    };
    transaction = (G4ImagePrefillTransaction) {
        &program, &run, &executed,
    };
    if ((model->prefill_operation_flow && !model->full_gpu_intent
            ? (model->compute_pool_ready
                ? salt_attn_pool_flow_run(&model->compute_pool,
                      g4_image_prefill_transaction, &transaction)
                : -1)
            : g4_image_prefill_transaction(&transaction)) == 0 &&
        executed == program.node_count)
        rc = 0;

done:
    if (rc != 0 &&
        (g4_gpu_kv_rewind(model, 0) != 0 ||
         g4_text_state_publish(model, 0) != 0))
        return step_fail("image-prefill-state-restore", -1, 0);
    return rc;
}

typedef struct G4TokenExecutionContext {
    SaltGemma4Text *model;
    float *logits;
    float *current;
    float *next;
    double total_start;
    double embedding_s;
    double attention_s;
    double ffn_s;
    double release_s;
    double head_s;
    double publish_s;
    double head_start;
    const char *failure_stage;
    int failure_layer;
    int active_window;
    int layer_lookahead;
    int cpu_graph_active;
    int cpu_flow_active;
    int cpu_soa_profile;
    int cpu_soa_ready;
    SaltAttnPoolStats pool_before;
    SaltAttnPoolStats pool_after;
    uint64_t q4_pool_before;
    uint64_t q4_multi_pool_before;
    uint64_t q4_multi_jobs_before;
    uint64_t qkv_multi_pool_before;
    uint64_t q8_pool_before;
    uint64_t expert_pool_before;
    uint64_t expert_fallback_before;
    int profile;
    int ffn_detail;
} G4TokenExecutionContext;

static uint64_t g4_soa_delta_u64(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0;
}

static double g4_soa_delta_s(double after, double before) {
    return after >= before ? after - before : 0.0;
}

typedef struct {
    const SaltTextTokenProgram *program;
    SaltTextTokenExecutor *executor;
    const SaltTextTokenExecution *execution;
    SaltTextTokenResult *result;
} G4TokenFlowCall;

static int g4_token_flow_call(void *opaque) {
    G4TokenFlowCall *call = (G4TokenFlowCall *)opaque;
    return call ? salt_text_token_execute(call->program, call->executor,
        call->execution, call->result) : -1;
}

static int g4_token_begin(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    if (!context || !context->model || !program || !execution ||
        context->model->position != (int)execution->source_position ||
        execution->input_token_id < 0 || execution->input_token_id >= G4_VOCAB)
        return -1;
    memset(&context->model->decode_detail, 0,
           sizeof context->model->decode_detail);
    context->model->decode_detail.enabled =
        context->profile || context->ffn_detail;
    context->model->decode_detail.ffn_detail_enabled = context->ffn_detail;
    context->cpu_soa_profile = !context->model->full_gpu_intent &&
        context->model->compute_pool_ready &&
        context->model->compute_pool.asoa_profile;
    if (context->cpu_soa_profile && !context->cpu_soa_ready) {
        if (salt_attn_pool_stats(&context->model->compute_pool,
                &context->pool_before) != 0)
            return -1;
        context->q4_pool_before = context->model->q4_pool_submissions;
        context->q4_multi_pool_before =
            context->model->q4_multi_pool_submissions;
        context->q4_multi_jobs_before = context->model->q4_multi_pool_jobs;
        context->qkv_multi_pool_before =
            context->model->qkv_multi_pool_submissions;
        context->q8_pool_before = context->model->q8_pool_submissions;
        context->expert_pool_before = context->model->expert_pool_submissions;
        context->expert_fallback_before = context->model->expert_pool_fallbacks;
        context->cpu_soa_ready = 1;
    }
    /* Ordinary CPU B1 has a fixed physical path.  Experimental retained
     * graphs belong to the separate batch executor, not token admission. */
    if (context->profile) context->total_start = g4_now_s();
    return 0;
}

static SaltTextTokenCellResult g4_token_cell(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    SaltGemma4Text *model;
    double mark = context && context->profile ? g4_now_s() : 0.0;
    int rc = 0;
    (void)program;
    (void)cell_index;
    if (!context || !(model = context->model) || !execution || !cell)
        return SALT_TEXT_TOKEN_CELL_FAILED;
    switch (cell->kind) {
    case SALT_TEXT_TOKEN_EMBED:
        rc = embedding_row(&model->embedding, execution->input_token_id,
                           model->state);
        if (rc == 0) {
            context->current = model->state;
            context->next = model->layer_output;
        }
        context->failure_stage = "embedding";
        context->failure_layer = -1;
        if (context->profile) context->embedding_s += g4_now_s() - mark;
        break;
    case SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT:
        context->failure_stage = "attention";
        context->failure_layer = cell->layer;
        if (g4_trunk_layer_window_begin(model, cell->layer) != 0)
            rc = -1;
        else {
            context->active_window = 1;
            rc = attention(model, &model->layers[cell->layer],
                context->current, model->position, model->after_attention);
        }
        if (context->profile) context->attention_s += g4_now_s() - mark;
        break;
    case SALT_TEXT_TOKEN_LAYER_FEED_FORWARD_PARENT:
        {
        int window_rc;
        context->failure_stage = "feed-forward";
        context->failure_layer = cell->layer;
        rc = feed_forward(model, &model->layers[cell->layer], cell->layer,
            model->after_attention, context->next, 1);
        window_rc = g4_trunk_layer_window_end(model);
        if (window_rc != 0 && rc == 0) {
            context->failure_stage = "layer-window-release";
            rc = -1;
        }
        if (window_rc == 0) context->active_window = 0;
        if (context->profile) context->ffn_s += g4_now_s() - mark;
        if (rc == 0) {
            float *swap = context->current;
            context->current = context->next;
            context->next = swap;
        }
        break;
        }
    case SALT_TEXT_TOKEN_FINAL_NORM:
        context->failure_stage = "final-norm";
        context->failure_layer = G4_LAYERS;
        context->head_start = mark;
        if (context->logits)
            rc = salt_gemma4_rmsnorm(model->final_state, context->current,
                model->final_norm.values, G4_HIDDEN, G4_EPS, 1);
        break;
    case SALT_TEXT_TOKEN_HEAD:
        context->failure_stage = "head";
        context->failure_layer = G4_LAYERS;
        if (context->logits)
            rc = q_matvec_operation(model, &model->embedding,
                model->final_state, context->logits);
        break;
    case SALT_TEXT_TOKEN_SOFTCAP:
        context->failure_stage = "softcap";
        context->failure_layer = G4_LAYERS;
        if (context->logits)
            rc = salt_gemma4_softcap(
                context->logits, G4_VOCAB, G4_LOGIT_CAP);
        if (context->profile && context->head_start > 0.0)
            context->head_s += g4_now_s() - context->head_start;
        break;
    case SALT_TEXT_TOKEN_FINALIZE:
        context->failure_stage = "finalize";
        context->failure_layer = G4_LAYERS;
        break;
    default:
        context->failure_stage = "token-cell-kind";
        context->failure_layer = cell->layer;
        rc = -1;
        break;
    }
    return rc == 0 ? SALT_TEXT_TOKEN_CELL_OK : SALT_TEXT_TOKEN_CELL_FAILED;
}

static SaltTextTokenCellResult g4_token_cell_t4_proof(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    SaltGemma4Text *model;
    double mark = context && context->profile ? g4_now_s() : 0.0;
    int rc = 0;
    (void)program;
    (void)cell_index;
    if (!context || !(model = context->model) || !model->metal_exact_cells ||
        !model->head_logits || !execution || !cell)
        return SALT_TEXT_TOKEN_CELL_FAILED;
    switch (cell->kind) {
    case SALT_TEXT_TOKEN_EMBED:
        rc = embedding_row(&model->embedding, execution->input_token_id,
                           model->state);
        if (rc == 0) {
            context->current = model->state;
            context->next = model->layer_output;
        }
        context->failure_stage = "embedding";
        context->failure_layer = -1;
        if (context->profile) context->embedding_s += g4_now_s() - mark;
        break;
    case SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT:
        context->failure_stage = "attention";
        context->failure_layer = cell->layer;
        if (g4_trunk_layer_window_begin(model, cell->layer) != 0)
            rc = -1;
        else {
            context->active_window = 1;
            rc = attention_t4_proof(model, &model->layers[cell->layer],
                context->current, model->position, model->after_attention);
        }
        if (context->profile) context->attention_s += g4_now_s() - mark;
        break;
    case SALT_TEXT_TOKEN_LAYER_FEED_FORWARD_PARENT:
        {
        int window_rc;
        context->failure_stage = "feed-forward";
        context->failure_layer = cell->layer;
        rc = feed_forward_t4_proof(model, &model->layers[cell->layer],
            cell->layer, model->after_attention, context->next);
        window_rc = g4_trunk_layer_window_end(model);
        if (window_rc != 0 && rc == 0) {
            context->failure_stage = "layer-window-release";
            rc = -1;
        }
        if (window_rc == 0) context->active_window = 0;
        if (context->profile) context->ffn_s += g4_now_s() - mark;
        if (rc == 0) {
            float *swap = context->current;
            context->current = context->next;
            context->next = swap;
        }
        break;
        }
    case SALT_TEXT_TOKEN_FINAL_NORM:
        context->failure_stage = "final-norm";
        context->failure_layer = G4_LAYERS;
        context->head_start = mark;
        if (context->logits)
            rc = g4_exact_rmsnorm(model, model->final_state,
                context->current, model->final_norm.values,
                G4_HIDDEN, G4_EPS, 1);
        if (rc == 0 && context->logits && g4_layer_digest(model, "final-norm",
                G4_LAYERS, model->final_state,
                G4_HIDDEN * sizeof(float)) != 0)
            rc = -1;
        break;
    case SALT_TEXT_TOKEN_HEAD:
        context->failure_stage = "head";
        context->failure_layer = G4_LAYERS;
        if (context->logits)
            rc = q_matvec_operation(model, &model->embedding,
                model->final_state, model->head_logits);
        if (rc == 0 && context->logits && g4_layer_digest(model, "head-logits",
                G4_LAYERS, model->head_logits,
                G4_VOCAB * sizeof(float)) != 0)
            rc = -1;
        break;
    case SALT_TEXT_TOKEN_SOFTCAP:
        context->failure_stage = "softcap";
        context->failure_layer = G4_LAYERS;
        if (context->logits)
            rc = g4_exact_softcap(model, model->head_logits,
                G4_VOCAB, G4_LOGIT_CAP);
        if (rc == 0 && context->logits && g4_layer_digest(model, "softcap",
                G4_LAYERS, model->head_logits,
                G4_VOCAB * sizeof(float)) != 0)
            rc = -1;
        if (context->profile && context->head_start > 0.0)
            context->head_s += g4_now_s() - context->head_start;
        break;
    case SALT_TEXT_TOKEN_FINALIZE:
        context->failure_stage = "finalize";
        context->failure_layer = G4_LAYERS;
        if (context->logits)
            memcpy(context->logits, model->head_logits,
                   (size_t)G4_VOCAB * sizeof(float));
        break;
    default:
        context->failure_stage = "token-cell-kind";
        context->failure_layer = cell->layer;
        rc = -1;
        break;
    }
    return rc == 0 ? SALT_TEXT_TOKEN_CELL_OK : SALT_TEXT_TOKEN_CELL_FAILED;
}

static int g4_token_barrier(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index) {
    (void)opaque;
    (void)program;
    (void)execution;
    (void)cell;
    (void)cell_index;
    return 0;
}

static int g4_token_finish(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    if (!context || !context->model || !program || !execution ||
        context->active_window)
        return -1;
    if (context->cpu_graph_active) {
        if (salt_attn_pool_graph_end(&context->model->compute_pool) != 0)
            return -1;
        context->cpu_graph_active = 0;
    }
    if (context->cpu_soa_ready && !context->cpu_flow_active &&
        salt_attn_pool_stats(&context->model->compute_pool,
            &context->pool_after) != 0)
        return -1;
    return 0;
}

static int g4_token_abort(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        SaltTextTokenStatus status, uint32_t completed_cells) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    (void)program;
    (void)execution;
    (void)status;
    (void)completed_cells;
    if (!context || !context->model) return -1;
    if (context->active_window) {
        context->active_window = 0;
        if (g4_trunk_layer_window_end(context->model) != 0)
            return -1;
    }
    if (context->cpu_graph_active) {
        if (salt_attn_pool_graph_end(&context->model->compute_pool) != 0)
            return -1;
        context->cpu_graph_active = 0;
    }
    return 0;
}

static int g4_token_publish(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution, uint32_t result_position) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    double mark = context && context->profile ? g4_now_s() : 0.0;
    (void)program;
    if (!context || !context->model || !execution ||
        result_position != execution->source_position + 1u ||
        execution->transition_generation == 0)
        return -1;
    if (g4_text_state_publish(context->model, (int)result_position) != 0)
        return -1;
    context->model->token_transition_generation =
        execution->transition_generation;
    if (context->profile) context->publish_s += g4_now_s() - mark;
    return 0;
}

static int g4_token_prepare_next_layer(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        uint32_t current_layer, uint32_t next_layer) {
    G4TokenExecutionContext *context = (G4TokenExecutionContext *)opaque;
    SaltGemma4Text *model;
    if (!context || !(model = context->model) || !program || !execution ||
        program != &model->token_program || program->layer_count != G4_LAYERS ||
        execution->source_position != (uint32_t)model->position ||
        current_layer >= G4_LAYERS || next_layer >= G4_LAYERS ||
        next_layer != current_layer + 1u)
        return -1;
    if (!context->layer_lookahead) return 1;
    if (model->prefill_retain_layers) return 0;
    return prepare_layer_weight_residency(
        model, &model->layers[next_layer]) == 0 ? 0 : 1;
}

static const SaltTextTokenExecutorOps g4_token_ops = {
    g4_token_begin,
    g4_token_cell,
    g4_token_barrier,
    g4_token_finish,
    g4_token_abort,
    g4_token_publish,
    g4_token_prepare_next_layer,
};

static const SaltTextTokenExecutorOps g4_token_t4_proof_ops = {
    g4_token_begin,
    g4_token_cell_t4_proof,
    g4_token_barrier,
    g4_token_finish,
    g4_token_abort,
    g4_token_publish,
    g4_token_prepare_next_layer,
};

int salt_gemma4_text_step(SaltGemma4Text *model, int token, float *logits) {
    G4TokenExecutionContext context;
    G4TokenFlowCall flow_call;
    SaltTextTokenExecution execution;
    SaltTextTokenResult result;
    SaltTextTokenExecutor executor;
    const char *flow_env, *ffn_detail_env;
    int flow_enabled, execute_rc, lookahead_claim;
    if (!model) return step_fail("model", -1, -1);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    memset(&model->target_projection, 0, sizeof model->target_projection);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    if (token < 0 || token >= G4_VOCAB)
        return step_fail("token", -1, model->position);
    if (model->position < 0 || model->position >= model->max_context ||
        !model->token_program.ready ||
        model->token_transition_generation == UINT64_MAX)
        return step_fail("position", -1, model->position);
    if (model->full_gpu_intent && model->decode_gpu_only) {
        SaltTextExecuteAllResult fine;
        int32_t input_token = token;
        uint64_t generation;
        double start = g4_now_s();
        if (model->prefill_decode_lookahead.status ==
                SALT_TEXT_LOOKAHEAD_READY)
            return step_fail("lookahead-fine-token", -1, model->position);
        if (!model->text_program.ready || !model->text_executor.ops ||
            model->text_kv_state.position != (uint32_t)model->position ||
            model->text_kv_state.transition_generation == UINT64_MAX)
            return step_fail("fine-token-state", -1, model->position);
        generation = model->text_kv_state.transition_generation + 1u;
        if (salt_text_execute_all(&model->text_program,
                &model->text_executor, generation,
                (uint32_t)model->position, &input_token, 1u, &fine) != 0 ||
            fine.status != SALT_TEXT_VERIFY_COMMITTED ||
            fine.committed_count != 1u ||
            fine.result_position != (uint32_t)model->position + 1u ||
            fine.transition_generation != generation ||
            !fine.view.canonical_base ||
            fine.view.canonical_bytes < model->text_program.layout.total_bytes ||
            fine.backend.engine_submissions != 1u ||
            fine.backend.completion_fences != 1u ||
            fine.backend.intermediate_host_publications != 0u)
            return step_fail("fine-token-execute", -1, model->position);
        if (logits)
            memcpy(logits, fine.view.canonical_base +
                       model->text_program.layout.position_logits,
                   (size_t)G4_VOCAB * sizeof(float));
        model->position = (int)fine.result_position;
        model->token_transition_generation = fine.transition_generation;
        memset(&model->token_backend_last, 0,
               sizeof model->token_backend_last);
        model->token_backend_last.caller_submissions =
            fine.backend.engine_submissions;
        model->token_backend_last.cells_completed =
            model->text_program.dispatch.cell_count;
        model->token_backend_last.internal_dependency_barriers =
            fine.backend.internal_dependency_barriers;
        model->token_backend_last.completion_fences =
            fine.backend.completion_fences;
        model->token_backend_last.intermediate_host_publications =
            fine.backend.intermediate_host_publications;
        model->token_backend_last.final_publications = 1u;
        model->token_backend_last.tentative_write_bytes =
            fine.backend.final_kv_publish_bytes;
        if (getenv("SALT_WATERFALL"))
            fprintf(stderr,
                "GEMMA4_FINE_TOKEN_WATERFALL position=%d total_s=%.9f "
                "cells=%u submissions=%u barriers=%u fences=%u "
                "graph_sessions=%u graph_nodes=%u graph_barriers=%u "
                "intermediate_publications=%u final_publications=1 "
                "kv_publish_bytes=%llu\n",
                model->position, g4_now_s() - start,
                model->token_backend_last.cells_completed,
                model->token_backend_last.caller_submissions,
                model->token_backend_last.internal_dependency_barriers,
                model->token_backend_last.completion_fences,
                fine.backend.cpu_graph_sessions,
                fine.backend.cpu_graph_nodes,
                fine.backend.cpu_graph_barriers,
                model->token_backend_last.intermediate_host_publications,
                (unsigned long long)fine.backend.final_kv_publish_bytes);
        return 0;
    }
    /* CPU B1 cannot be redirected by optimization flags.  Preserve the
     * existing GPU admission behavior without enabling CPU fallback. */
    flow_env = model->full_gpu_intent ? getenv("SALT_TOKEN_CPU_FLOW") : NULL;
    if (flow_env && strcmp(flow_env, "0") != 0 &&
        strcmp(flow_env, "1") != 0)
        return step_fail("cpu-flow-config", -1, model->position);
    flow_enabled = flow_env && strcmp(flow_env, "1") == 0;
    ffn_detail_env = getenv("SALT_FFN_DETAIL");
    if (ffn_detail_env && strcmp(ffn_detail_env, "0") != 0 &&
        strcmp(ffn_detail_env, "1") != 0)
        return step_fail("ffn-detail-config", -1, model->position);
    if (flow_enabled && (model->full_gpu_intent || !model->compute_pool_ready))
        return step_fail("cpu-flow-admission", -1, model->position);
    lookahead_claim = salt_text_phase_lookahead_claim(
        &model->prefill_decode_lookahead, SALT_TEXT_PHASE_DECODE,
        model->text_kv_state.transition_generation,
        (uint32_t)model->position);
    if (lookahead_claim < 0)
        return step_fail("lookahead-claim", -1, model->position);
    memset(&context, 0, sizeof context);
    context.model = model;
    context.logits = logits;
    context.failure_stage = "token-program";
    context.failure_layer = -1;
    context.profile = getenv("SALT_WATERFALL") != NULL;
    context.ffn_detail = ffn_detail_env && strcmp(ffn_detail_env, "1") == 0;
    context.cpu_flow_active = flow_enabled;
    context.layer_lookahead = lookahead_claim == 1;
    if (flow_enabled && model->compute_pool.asoa_profile) {
        context.cpu_soa_profile = 1;
        if (salt_attn_pool_stats(&model->compute_pool,
                &context.pool_before) != 0)
            return step_fail("cpu-flow-stats", -1, model->position);
        context.q4_pool_before = model->q4_pool_submissions;
        context.q4_multi_pool_before = model->q4_multi_pool_submissions;
        context.q4_multi_jobs_before = model->q4_multi_pool_jobs;
        context.qkv_multi_pool_before = model->qkv_multi_pool_submissions;
        context.q8_pool_before = model->q8_pool_submissions;
        context.expert_pool_before = model->expert_pool_submissions;
        context.expert_fallback_before = model->expert_pool_fallbacks;
        context.cpu_soa_ready = 1;
    }
    execution = (SaltTextTokenExecution) {
        model->token_transition_generation + 1u,
        (uint32_t)model->position,
        token,
    };
    executor = (SaltTextTokenExecutor) {
        model->metal_exact_cells ? &g4_token_t4_proof_ops : &g4_token_ops,
        &context,
        context.layer_lookahead,
    };
    flow_call = (G4TokenFlowCall) {
        &model->token_program, &executor, &execution, &result,
    };
    if (model->full_gpu_intent) {
        const SaltTextPhaseScopeOps phase_ops = {
            g4_gpu_phase_begin, g4_gpu_phase_end,
        };
        const SaltTextPhaseScope phase_scope = { &phase_ops, model };
        execute_rc = salt_text_phase_scope_run(
            &phase_scope, SALT_TEXT_PHASE_DECODE,
            g4_token_flow_call, &flow_call);
    } else {
        execute_rc = flow_enabled
            ? salt_attn_pool_flow_run(&model->compute_pool,
                g4_token_flow_call, &flow_call)
            : salt_text_token_execute(&model->token_program, &executor,
                &execution, &result);
    }
    if (flow_enabled && context.cpu_soa_ready &&
        salt_attn_pool_stats(&model->compute_pool, &context.pool_after) != 0)
        execute_rc = -1;
    if (execute_rc != 0 ||
        result.status != SALT_TEXT_TOKEN_COMMITTED ||
        result.result_position != (uint32_t)model->position ||
        result.backend.caller_submissions != 1 ||
        result.backend.completion_fences != 1 ||
        result.backend.intermediate_host_publications != 0 ||
        result.backend.final_publications != 1)
        return step_fail(context.failure_stage,
            context.failure_layer, model->position);
    model->token_backend_last = result.backend;
    if (context.profile) {
        double total_s, control_s, stage_s;
        total_s = g4_now_s() - context.total_start;
        stage_s = context.embedding_s + context.attention_s + context.ffn_s +
                  context.release_s + context.head_s + context.publish_s;
        control_s = total_s - stage_s;
        if (control_s < 0.0 && control_s > -1e-6) control_s = 0.0;
        fprintf(stderr,
            "GEMMA4_DECODE_WATERFALL position=%d total_s=%.9f "
            "embedding_s=%.9f attention_s=%.9f "
            "feed_forward_s=%.9f release_s=%.9f head_s=%.9f "
            "publish_s=%.9f control_s=%.9f "
            "attention_qkv_s=%.9f attention_transform_s=%.9f "
            "attention_body_s=%.9f attention_o_s=%.9f "
            "attention_residual_s=%.9f ffn_dense_s=%.9f "
            "ffn_route_s=%.9f ffn_experts_s=%.9f ffn_combine_s=%.9f "
            "ffn_expert_fetch_s=%.9f ffn_expert_compute_s=%.9f "
            "ffn_expert_release_s=%.9f "
            "expert_pool_submissions=%llu expert_pool_fallbacks=%llu "
            "token_cells=%u token_submissions=%u token_barriers=%u "
            "token_fences=%u token_intermediate_publications=%u "
            "token_publications=%u\n",
            model->position, total_s, context.embedding_s,
            context.attention_s, context.ffn_s, context.release_s,
            context.head_s, context.publish_s, control_s,
            model->decode_detail.attention_qkv_s,
            model->decode_detail.attention_transform_s,
            model->decode_detail.attention_body_s,
            model->decode_detail.attention_o_s,
            model->decode_detail.attention_residual_s,
            model->decode_detail.ffn_dense_s,
            model->decode_detail.ffn_route_s,
            model->decode_detail.ffn_experts_s,
            model->decode_detail.ffn_combine_s,
            model->decode_detail.ffn_expert_fetch_s,
            model->decode_detail.ffn_expert_compute_s,
            model->decode_detail.ffn_expert_release_s,
            (unsigned long long)model->expert_pool_submissions,
            (unsigned long long)model->expert_pool_fallbacks,
            result.backend.cells_completed,
            result.backend.caller_submissions,
            result.backend.internal_dependency_barriers,
            result.backend.completion_fences,
            result.backend.intermediate_host_publications,
            result.backend.final_publications);
    }
    if (context.ffn_detail) {
        double worker_work_s =
            model->decode_detail.ffn_expert_gate_work_s +
            model->decode_detail.ffn_expert_up_work_s +
            model->decode_detail.ffn_expert_activation_work_s +
            model->decode_detail.ffn_expert_down_work_s;
        fprintf(stderr,
            "GEMMA4_FFN_DETAIL position=%d tasks=%llu waves=%llu "
            "routed_expert_parent_s=%.9f expert_fetch_s=%.9f "
            "expert_compute_wall_s=%.9f expert_release_s=%.9f "
            "expert_gate_work_s=%.9f expert_up_work_s=%.9f "
            "expert_activation_work_s=%.9f expert_down_work_s=%.9f "
            "expert_worker_work_s=%.9f expert_worker_critical_s=%.9f "
            "expert_worker_skew_s=%.9f expert_pool_overhead_s=%.9f\n",
            model->position,
            (unsigned long long)model->decode_detail.ffn_detail_tasks,
            (unsigned long long)model->decode_detail.ffn_detail_waves,
            model->decode_detail.ffn_experts_s,
            model->decode_detail.ffn_expert_fetch_s,
            model->decode_detail.ffn_expert_compute_s,
            model->decode_detail.ffn_expert_release_s,
            model->decode_detail.ffn_expert_gate_work_s,
            model->decode_detail.ffn_expert_up_work_s,
            model->decode_detail.ffn_expert_activation_work_s,
            model->decode_detail.ffn_expert_down_work_s,
            worker_work_s,
            model->decode_detail.ffn_expert_worker_critical_s,
            model->decode_detail.ffn_expert_worker_skew_s,
            model->decode_detail.ffn_expert_pool_overhead_s);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
        fprintf(stderr,
            "GEMMA4_FFN_MATRIX_DETAIL position=%d flows=%llu "
            "gate_up_setup_s=%.9f gate_up_s=%.9f activation_s=%.9f "
            "down_setup_s=%.9f down_s=%.9f total_s=%.9f\n",
            model->position,
            (unsigned long long)model->decode_detail.ffn_matrix_flows,
            model->decode_detail.ffn_matrix_gate_up_setup_s,
            model->decode_detail.ffn_matrix_gate_up_s,
            model->decode_detail.ffn_matrix_activation_s,
            model->decode_detail.ffn_matrix_down_setup_s,
            model->decode_detail.ffn_matrix_down_s,
            model->decode_detail.ffn_matrix_total_s);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    }
    if (context.cpu_soa_ready) {
        uint64_t pool_actions = g4_soa_delta_u64(
            context.pool_after.phase_actions,
            context.pool_before.phase_actions);
        uint64_t q4_actions = g4_soa_delta_u64(
            model->q4_pool_submissions, context.q4_pool_before);
        uint64_t q4_multi_actions = g4_soa_delta_u64(
            model->q4_multi_pool_submissions,
            context.q4_multi_pool_before);
        uint64_t q4_multi_jobs = g4_soa_delta_u64(
            model->q4_multi_pool_jobs, context.q4_multi_jobs_before);
        uint64_t qkv_multi_actions = g4_soa_delta_u64(
            model->qkv_multi_pool_submissions,
            context.qkv_multi_pool_before);
        uint64_t q8_actions = g4_soa_delta_u64(
            model->q8_pool_submissions, context.q8_pool_before);
        uint64_t expert_actions = g4_soa_delta_u64(
            model->expert_pool_submissions, context.expert_pool_before);
        uint64_t expert_fallbacks = g4_soa_delta_u64(
            model->expert_pool_fallbacks, context.expert_fallback_before);
        uint64_t classified = q4_actions + q8_actions + expert_actions;
        uint64_t other_actions = pool_actions >= classified
            ? pool_actions - classified : 0;
        uint32_t program_headroom = G4_TOKEN_PROGRAM_CELLS >=
                result.backend.cells_completed
            ? G4_TOKEN_PROGRAM_CELLS - result.backend.cells_completed : 0;
        uint32_t barrier_headroom =
            model->token_program.maximum_internal_barriers >=
                result.backend.internal_dependency_barriers
            ? model->token_program.maximum_internal_barriers -
                result.backend.internal_dependency_barriers : 0;
        double flow_wait_s = g4_soa_delta_s(
            context.pool_after.flow_wait_s,
            context.pool_before.flow_wait_s);
        double worker_critical_s = g4_soa_delta_s(
            context.pool_after.worker_critical_s,
            context.pool_before.worker_critical_s);
        double wait_overhead_s = flow_wait_s > worker_critical_s
            ? flow_wait_s - worker_critical_s : 0.0;
        fprintf(stderr,
            "GEMMA4_CPU_SOA position=%d graph=%d flow=%d "
            "program_capacity=%u program_actions=%u program_headroom=%u "
            "barrier_capacity=%u barrier_callbacks=%u barrier_headroom=%u "
            "pool_resident_workers=%u pool_base_workers=%u "
            "pool_wide_workers=%u pool_actions=%llu "
            "pool_worker_actions=%llu pool_worker_slot_capacity=%llu "
            "pool_worker_slot_headroom=%llu pool_peak_active_global=%u "
            "flow_lock_acquires=%llu flow_signal_calls=%llu "
            "flow_wait_calls=%llu graph_begin_wait_calls=%llu "
            "graph_end_wait_calls=%llu graph_sessions=%llu flow_sessions=%llu "
            "internal_lock_acquires=%llu internal_signal_calls=%llu "
            "internal_wait_calls=%llu internal_lock_wait_s=%.9f "
            "internal_wait_s=%.9f "
            "rejected_actions=%llu flow_lock_wait_s=%.9f "
            "flow_wait_s=%.9f worker_action_sum_s=%.9f "
            "worker_critical_s=%.9f wait_overhead_s=%.9f "
            "q4_actions=%llu q4_multi_actions=%llu q4_multi_jobs=%llu "
            "qkv_multi_actions=%llu q8_actions=%llu "
            "expert_actions=%llu expert_fallbacks=%llu "
            "other_actions=%llu\n",
            model->position, context.cpu_graph_active ? 1 :
                (g4_soa_delta_u64(context.pool_after.graph_sessions,
                    context.pool_before.graph_sessions) ? 1 : 0),
            context.cpu_flow_active,
            (unsigned int)G4_TOKEN_PROGRAM_CELLS,
            result.backend.cells_completed, program_headroom,
            model->token_program.maximum_internal_barriers,
            result.backend.internal_dependency_barriers, barrier_headroom,
            context.pool_after.resident_workers,
            context.pool_after.base_workers,
            context.pool_after.wide_workers,
            (unsigned long long)pool_actions,
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.worker_actions,
                context.pool_before.worker_actions),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.worker_slot_capacity,
                context.pool_before.worker_slot_capacity),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.worker_slot_headroom,
                context.pool_before.worker_slot_headroom),
            context.pool_after.peak_active_workers,
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.flow_lock_acquires,
                context.pool_before.flow_lock_acquires),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.flow_signal_calls,
                context.pool_before.flow_signal_calls),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.flow_wait_calls,
                context.pool_before.flow_wait_calls),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.graph_begin_wait_calls,
                context.pool_before.graph_begin_wait_calls),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.graph_end_wait_calls,
                context.pool_before.graph_end_wait_calls),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.graph_sessions,
                context.pool_before.graph_sessions),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.flow_sessions,
                context.pool_before.flow_sessions),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.internal_lock_acquires,
                context.pool_before.internal_lock_acquires),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.internal_signal_calls,
                context.pool_before.internal_signal_calls),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.internal_wait_calls,
                context.pool_before.internal_wait_calls),
            g4_soa_delta_s(context.pool_after.internal_lock_wait_s,
                context.pool_before.internal_lock_wait_s),
            g4_soa_delta_s(context.pool_after.internal_wait_s,
                context.pool_before.internal_wait_s),
            (unsigned long long)g4_soa_delta_u64(
                context.pool_after.rejected_actions,
                context.pool_before.rejected_actions),
            g4_soa_delta_s(context.pool_after.flow_lock_wait_s,
                context.pool_before.flow_lock_wait_s),
            flow_wait_s,
            g4_soa_delta_s(context.pool_after.worker_action_sum_s,
                context.pool_before.worker_action_sum_s),
            worker_critical_s, wait_overhead_s,
            (unsigned long long)q4_actions,
            (unsigned long long)q4_multi_actions,
            (unsigned long long)q4_multi_jobs,
            (unsigned long long)qkv_multi_actions,
            (unsigned long long)q8_actions,
            (unsigned long long)expert_actions,
            (unsigned long long)expert_fallbacks,
            (unsigned long long)other_actions);
    }
    return 0;
}

static int g4_state_row_base(const G4Layer *layer, int position) {
    const int retained = G4_SLIDING_WINDOW - 1;
    if (!layer || position < 0) return -1;
    if (!layer->full_attention && position > retained)
        return position - retained;
    return 0;
}

typedef enum {
    G4_TRANSFER_FORMAT_V4 = 4,
    G4_TRANSFER_FORMAT_V5 = 5,
    G4_TRANSFER_FORMAT_V6 = 6
} G4TransferFormat;

static int g4_transfer_segment_count(
        const G4Layer *layer, G4TransferFormat format) {
    if (!layer || !layer->plan) return -1;
    return format == G4_TRANSFER_FORMAT_V6 &&
            layer->plan->attention.shared_kv_projection
        ? 1 : 2;
}

static int g4_transfer_bytes_at_version(
        const SaltGemma4Text *model, int position,
        G4TransferFormat format, size_t *bytes) {
    size_t total = G4_KV_TRANSFER_HEADER;
    if (!model || !bytes || position < 1 || position > model->max_context ||
        (format != G4_TRANSFER_FORMAT_V4 &&
         format != G4_TRANSFER_FORMAT_V5 &&
         format != G4_TRANSFER_FORMAT_V6))
        return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        const G4Layer *layer = &model->layers[i];
        int base = format == G4_TRANSFER_FORMAT_V4
            ? 0 : g4_state_row_base(layer, position);
        int segments = g4_transfer_segment_count(layer, format);
        size_t rows, dim, count;
        if (base < 0 || base > position || segments < 1) return -1;
        rows = (size_t)(position - base);
        dim = (size_t)layer->kv_dim;
        if (!dim || rows > SIZE_MAX / dim) return -1;
        count = rows * dim;
        if (count > (SIZE_MAX - total) /
                ((size_t)segments * sizeof(float)))
            return -1;
        total += count * (size_t)segments * sizeof(float);
    }
    *bytes = total;
    return 0;
}

static int kv_snapshot_bytes_at(const SaltGemma4Text *model, int position,
                                size_t *bytes) {
    size_t total = sizeof(G4KVSnapshotHeader);
    if (!model || !bytes || position < 1 || position > model->max_context)
        return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        int dim = model->layers[i].kv_dim;
        int base = g4_state_row_base(&model->layers[i], position);
        size_t rows, count;
        if (dim < 1 || base < 0 || base > position)
            return -1;
        rows = (size_t)(position - base);
        if (rows > SIZE_MAX / (size_t)dim) return -1;
        count = rows * (size_t)dim;
        if (count > (SIZE_MAX - total) / (2u * sizeof(float)))
            return -1;
        total += count * 2u * sizeof(float);
    }
    *bytes = total;
    return 0;
}

int salt_gemma4_text_state_view(const SaltGemma4Text *model,
                                SaltStateView *view) {
    const SaltStateModelDesc *state_model;
    uint64_t bytes_per_row = 0;
    if (!model || !view || !model->model_desc ||
        !(state_model = salt_model_state(model->model_desc)) ||
        model->position < 0)
        return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        uint64_t add;
        if (model->layers[layer].kv_dim < 1)
            return -1;
        add = (uint64_t)(uint32_t)model->layers[layer].kv_dim *
              2u * sizeof(float);
        if (bytes_per_row > UINT64_MAX - add) return -1;
        bytes_per_row += add;
    }
    memset(view, 0, sizeof *view);
    view->schema_version = SALT_STATE_SCHEMA_VERSION;
    view->session_epoch = 1u;
    view->lifetime_turn = (uint64_t)(uint32_t)model->position;
    view->position = (uint64_t)(uint32_t)model->position;
    view->mindset_end = state_model->default_mindset_end;
    view->bytes_per_row = bytes_per_row;
    view->mindset_kind = state_model->mindset_kind;
    view->facts_kind = state_model->facts_kind;
    view->facts_optional = state_model->facts_optional;
    return salt_state_view_validate(state_model, view);
}

static int g4_state_materialize(
        void *context, SaltStateTransactionKind transaction,
        SaltStateArtifactKind artifact, SaltStateTransactionReason reason) {
    SaltGemma4Text *model = (SaltGemma4Text *)context;
    uint64_t bytes = 0;
    double started;
    if (!model) return -1;
    if (!model->full_gpu_intent || !model->gpu_initialized) return 0;
    if (transaction == SALT_STATE_TRANSACTION_EXPORT) {
        started = g4_now_s();
        if ((model->gpu_attention_enabled &&
             salt_gpu_attention_materialize(
                 &model->kv_shared, model->position, &bytes) != 0) ||
            (!model->gpu_attention_enabled && salt_gpu_sync() != 0))
            return -1;
        fprintf(stderr,
            "GEMMA4_KV_MATERIALIZE action=export artifact=%d reason=%d "
            "position=%d bytes=%llu wall_s=%.6f\n",
            (int)artifact, (int)reason, model->position,
            (unsigned long long)bytes, g4_now_s() - started);
        return 0;
    }
    return salt_gpu_sync();
}

static int g4_state_finalize(
        void *context, SaltStateTransactionKind transaction,
        SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
        const SaltStateView *committed_view) {
    SaltGemma4Text *model = (SaltGemma4Text *)context;
    uint64_t bytes = 0;
    double started;
    if (!model || !committed_view || committed_view->position > INT_MAX)
        return -1;
    if (!model->full_gpu_intent || !model->gpu_initialized ||
        transaction != SALT_STATE_TRANSACTION_IMPORT)
        return 0;
    started = g4_now_s();
    if ((model->gpu_attention_enabled &&
         salt_gpu_attention_import(&model->kv_shared,
             (int)committed_view->position, &bytes) != 0) ||
        (!model->gpu_attention_enabled && salt_gpu_sync() != 0))
        return -1;
    fprintf(stderr,
        "GEMMA4_KV_MATERIALIZE action=import artifact=%d reason=%d "
        "position=%llu bytes=%llu wall_s=%.6f\n",
        (int)artifact, (int)reason,
        (unsigned long long)committed_view->position,
        (unsigned long long)bytes, g4_now_s() - started);
    return 0;
}

int salt_gemma4_text_state_transaction_begin(
        SaltGemma4Text *model, SaltStateTransactionKind kind,
        SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
        SaltStateTransaction *transaction) {
    SaltStateView before;
    if (!model || salt_gemma4_text_state_view(model, &before) != 0)
        return -1;
    return salt_state_transaction_begin(&model->state_control, &before,
        kind, artifact, reason, transaction);
}

int salt_gemma4_text_state_transaction_finish(
        SaltGemma4Text *model, SaltStateTransaction *transaction) {
    SaltStateView after;
    if (!model || !transaction ||
        salt_gemma4_text_state_view(model, &after) != 0)
        return -1;
    return salt_state_transaction_finish(transaction, &after);
}

size_t salt_gemma4_text_kv_snapshot_size(const SaltGemma4Text *model) {
    size_t bytes = 0;
    if (!model || !all_identities_match(model) ||
        kv_snapshot_bytes_at(model, model->position, &bytes) != 0)
        return 0;
    return bytes;
}

size_t salt_gemma4_text_kv_snapshot_header_size(void) {
    return sizeof(G4KVSnapshotHeader);
}

int salt_gemma4_text_kv_snapshot(const SaltGemma4Text *model,
                                 void *buffer, size_t buffer_size) {
    G4KVSnapshotHeader header;
    unsigned char *cursor = (unsigned char *)buffer;
    unsigned char *payload;
    size_t expected = 0;
    uint64_t materialized = 0;
    if (!model || !buffer || !all_identities_match(model) ||
        g4_gpu_kv_materialize((SaltGemma4Text *)(uintptr_t)model,
            model->position, &materialized) != 0 ||
        kv_snapshot_bytes_at(model, model->position, &expected) != 0 ||
        buffer_size != expected || (uint64_t)buffer_size != buffer_size)
        return -1;
    memset(&header, 0, sizeof header);
    memcpy(header.magic, G4_KV_SNAPSHOT_MAGIC, sizeof header.magic);
    header.version = G4_KV_SNAPSHOT_VERSION;
    header.layers = G4_LAYERS;
    header.max_context = (uint32_t)model->max_context;
    header.position = (uint32_t)model->position;
    header.bytes = (uint64_t)buffer_size;
    header.instance_tag = (uint64_t)(uintptr_t)model;
    for (int i = 0; i < G4_LAYERS; i++)
        header.kv_dims[i] = (uint32_t)model->layers[i].kv_dim;
    cursor += sizeof header;
    payload = cursor;
    for (int i = 0; i < G4_LAYERS; i++) {
        const G4Layer *layer = &model->layers[i];
        int base = g4_state_row_base(layer, model->position);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        for (int value_row = 0; value_row < 2; value_row++)
            for (int position = base; position < model->position; position++) {
                const float *row = value_row
                    ? g4_value_row(layer, position)
                    : g4_key_row(layer, position);
                if (!row) return -1;
                for (int j = 0; j < layer->kv_dim; j++)
                    if (!isfinite(row[j])) return -1;
                memcpy(cursor, row, row_bytes);
                cursor += row_bytes;
            }
    }
    if (cursor != (unsigned char *)buffer + buffer_size ||
        !all_identities_match(model) ||
        salt_sha256_bytes(payload, buffer_size - sizeof header,
                          header.payload_sha256) != 0)
        return -1;
    memcpy(buffer, &header, sizeof header);
    return 0;
}

int salt_gemma4_text_kv_poison_prefix(SaltGemma4Text *model, int position,
                                      size_t *poisoned_bytes) {
    uint32_t poison_bits = 0x7fc00001u;
    float poison;
    size_t snapshot_bytes = 0;
    uint64_t transferred = 0;
    if (poisoned_bytes) *poisoned_bytes = 0;
    if (!model || !poisoned_bytes || position < 1 ||
        position > model->position || !all_identities_match(model) ||
        g4_gpu_kv_materialize(model, position, &transferred) != 0 ||
        kv_snapshot_bytes_at(model, position, &snapshot_bytes) != 0)
        return -1;
    memcpy(&poison, &poison_bits, sizeof poison);
    for (size_t j = 0; j < model->kv_arena_floats; j++)
        model->kv_arena[j] = poison;
    for (size_t j = 0; j < model->kv_arena_floats; j++)
        if (!isnan(model->kv_arena[j]))
                return -1;
    if (!all_identities_match(model) ||
        g4_gpu_kv_import(model, position, &transferred) != 0)
        return -1;
    *poisoned_bytes = snapshot_bytes - sizeof(G4KVSnapshotHeader);
    return 0;
}

int salt_gemma4_text_kv_restore(SaltGemma4Text *model,
                                const void *buffer, size_t buffer_size) {
    G4KVSnapshotHeader header;
    const unsigned char *cursor = (const unsigned char *)buffer;
    const unsigned char *end;
    unsigned char payload_sha256[32];
    size_t expected = 0;
    uint64_t transferred = 0;
    if (!model || !buffer || buffer_size < sizeof header ||
        !all_identities_match(model))
        return -1;
    memcpy(&header, cursor, sizeof header);
    if (memcmp(header.magic, G4_KV_SNAPSHOT_MAGIC, sizeof header.magic) ||
        header.version != G4_KV_SNAPSHOT_VERSION ||
        header.layers != G4_LAYERS ||
        header.max_context != (uint32_t)model->max_context ||
        header.position < 1 || header.position > (uint32_t)model->max_context ||
        header.bytes != (uint64_t)buffer_size ||
        header.instance_tag != (uint64_t)(uintptr_t)model ||
        kv_snapshot_bytes_at(model, (int)header.position, &expected) != 0 ||
        expected != buffer_size)
        return -1;
    for (int i = 0; i < G4_LAYERS; i++)
        if (header.kv_dims[i] != (uint32_t)model->layers[i].kv_dim)
            return -1;

    cursor += sizeof header;
    end = (const unsigned char *)buffer + buffer_size;
    if (salt_sha256_bytes(cursor, (size_t)(end - cursor), payload_sha256) != 0 ||
        memcmp(payload_sha256, header.payload_sha256,
               sizeof header.payload_sha256) != 0)
        return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *layer = &model->layers[i];
        int base = g4_state_row_base(layer, (int)header.position);
        size_t count = (size_t)((int)header.position - base) *
                       (size_t)layer->kv_dim * 2u;
        for (size_t j = 0; j < count; j++) {
            float value;
            memcpy(&value, cursor + j * sizeof value, sizeof value);
            if (!isfinite(value)) return -1;
        }
        cursor += count * sizeof(float);
    }
    if (cursor != end || !all_identities_match(model)) return -1;

    memset(model->kv_arena, 0,
           model->kv_arena_floats * sizeof *model->kv_arena);
    cursor = (const unsigned char *)buffer + sizeof header;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *layer = &model->layers[i];
        int base = g4_state_row_base(layer, (int)header.position);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        for (int value_row = 0; value_row < 2; value_row++)
            for (int position = base; position < (int)header.position; position++) {
                float *row = value_row
                    ? g4_private_value_row(layer, position)
                    : g4_private_key_row(layer, position);
                if (!row) return -1;
                memcpy(row, cursor, row_bytes);
                if (memcmp(row, cursor, row_bytes) != 0) return -1;
                cursor += row_bytes;
            }
    }
    if (g4_gpu_kv_import(model, (int)header.position, &transferred) != 0 ||
        g4_text_state_publish(model, (int)header.position) != 0)
        return -1;
    if (!all_identities_match(model)) {
        (void)g4_text_state_publish(model, 0);
        return -1;
    }
    return 0;
}

static void g4_le32_store(unsigned char *dst, uint32_t value) {
    dst[0] = (unsigned char)(value & 0xffu);
    dst[1] = (unsigned char)((value >> 8) & 0xffu);
    dst[2] = (unsigned char)((value >> 16) & 0xffu);
    dst[3] = (unsigned char)((value >> 24) & 0xffu);
}

static void g4_le64_store(unsigned char *dst, uint64_t value) {
    for (unsigned int i = 0; i < 8; i++)
        dst[i] = (unsigned char)((value >> (i * 8u)) & 0xffu);
}

static uint32_t g4_le32_load(const unsigned char *src) {
    return (uint32_t)src[0] |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static uint64_t g4_le64_load(const unsigned char *src) {
    uint64_t value = 0;
    for (unsigned int i = 0; i < 8; i++)
        value |= (uint64_t)src[i] << (i * 8u);
    return value;
}

static int g4_binary32_little_endian(void) {
    float one = 1.0f;
    unsigned char bytes[sizeof one];
    if (sizeof one != 4u) return 0;
    memcpy(bytes, &one, sizeof bytes);
    return bytes[0] == 0x00u && bytes[1] == 0x00u &&
           bytes[2] == 0x80u && bytes[3] == 0x3fu;
}

static int kv_transfer_bytes_at(const SaltGemma4Text *model, int position,
                                size_t *bytes) {
    return g4_transfer_bytes_at_version(
        model, position, G4_TRANSFER_FORMAT_V6, bytes);
}

static int state_sha256_at(const SaltGemma4Text *model, int position,
                              uint8_t digest[32]);
static void state_hash_u32(SaltSha256 *state, uint32_t value);
static int g4_transfer_logical_sha256_live(
    const SaltGemma4Text *model, int position, G4TransferFormat format,
    const unsigned char compatibility[32], uint8_t out[32]);

static uint32_t g4_transfer_sliding_base(uint32_t position) {
    const uint32_t retained = G4_SLIDING_WINDOW - 1u;
    return position > retained ? position - retained : 0u;
}

static int g4_transfer_v6_geometry_valid(
        const SaltGemma4Text *model, const unsigned char *header,
        size_t encoded_size, uint32_t *position_out) {
    uint32_t context, position;
    uint64_t encoded_bytes;
    size_t expected = 0;
    if (!model || !header || !position_out ||
        memcmp(header, G4_KV_TRANSFER_MAGIC,
               sizeof G4_KV_TRANSFER_MAGIC) != 0 ||
        g4_le32_load(header + 8) != G4_KV_TRANSFER_VERSION ||
        g4_le32_load(header + 12) != G4_LAYERS)
        return 0;
    context = g4_le32_load(header + 16);
    position = g4_le32_load(header + 20);
    encoded_bytes = g4_le64_load(header + 24);
    if (context != (uint32_t)model->max_context || position < 1 ||
        position > context || encoded_bytes != (uint64_t)encoded_size ||
        kv_transfer_bytes_at(model, (int)position, &expected) != 0 ||
        expected != encoded_size ||
        memcmp(header + 152, model->kv_compat_sha256,
               sizeof model->kv_compat_sha256) != 0 ||
        g4_le32_load(header + 248) != g4_transfer_sliding_base(position) ||
        g4_le32_load(header + 252) != G4_SLIDING_WINDOW)
        return 0;
    for (int i = 0; i < G4_LAYERS; i++)
        if (g4_le32_load(header + 32 + (size_t)i * 4u) !=
                (uint32_t)model->layers[i].kv_dim)
            return 0;
    *position_out = position;
    return 1;
}

static int g4_transfer_v5_geometry_valid(
        const SaltGemma4Text *model, const unsigned char *header,
        size_t encoded_size, uint32_t *position_out) {
    unsigned char compatibility[32];
    uint32_t context, position;
    uint64_t encoded_bytes;
    size_t expected = 0;
    if (!model || !header || !position_out ||
        sizeof(SALT_GEMMA4_KV_LEGACY_V5_SHA256) != 65u ||
        g4_sha256_from_hex(SALT_GEMMA4_KV_LEGACY_V5_SHA256,
                           compatibility) != 0 ||
        memcmp(header, G4_KV_TRANSFER_MAGIC_V5,
               sizeof G4_KV_TRANSFER_MAGIC_V5) != 0 ||
        g4_le32_load(header + 8) != 5u ||
        g4_le32_load(header + 12) != G4_LAYERS)
        return 0;
    context = g4_le32_load(header + 16);
    position = g4_le32_load(header + 20);
    encoded_bytes = g4_le64_load(header + 24);
    if (context != (uint32_t)model->max_context || position < 1 ||
        position > context || encoded_bytes != (uint64_t)encoded_size ||
        g4_transfer_bytes_at_version(model, (int)position,
            G4_TRANSFER_FORMAT_V5, &expected) != 0 ||
        expected != encoded_size ||
        memcmp(header + 152, compatibility, sizeof compatibility) != 0 ||
        g4_le32_load(header + 248) != g4_transfer_sliding_base(position) ||
        g4_le32_load(header + 252) != G4_SLIDING_WINDOW)
        return 0;
    for (int i = 0; i < G4_LAYERS; i++)
        if (g4_le32_load(header + 32 + (size_t)i * 4u) !=
                (uint32_t)model->layers[i].kv_dim)
            return 0;
    *position_out = position;
    return 1;
}

static int g4_transfer_v4_bytes_at(const SaltGemma4Text *model, int position,
                                   size_t *bytes) {
    return g4_transfer_bytes_at_version(
        model, position, G4_TRANSFER_FORMAT_V4, bytes);
}

static int g4_transfer_v4_geometry_valid(
        const SaltGemma4Text *model, const unsigned char *header,
        size_t encoded_size, uint32_t *position_out) {
    unsigned char compatibility[32];
    uint32_t context, position;
    uint64_t encoded_bytes;
    size_t expected = 0;
    if (!model || !header || !position_out ||
        sizeof(SALT_GEMMA4_KV_LEGACY_V4_SHA256) != 65u ||
        g4_sha256_from_hex(SALT_GEMMA4_KV_LEGACY_V4_SHA256,
                           compatibility) != 0 ||
        memcmp(header, G4_KV_TRANSFER_MAGIC_V4,
               sizeof G4_KV_TRANSFER_MAGIC_V4) != 0 ||
        g4_le32_load(header + 8) != 4u ||
        g4_le32_load(header + 12) != G4_LAYERS)
        return 0;
    context = g4_le32_load(header + 16);
    position = g4_le32_load(header + 20);
    encoded_bytes = g4_le64_load(header + 24);
    if (context != (uint32_t)model->max_context || position < 1 ||
        position > context || encoded_bytes != (uint64_t)encoded_size ||
        g4_transfer_v4_bytes_at(model, (int)position, &expected) != 0 ||
        expected != encoded_size ||
        memcmp(header + 152, compatibility, sizeof compatibility) != 0)
        return 0;
    for (int i = 0; i < G4_LAYERS; i++)
        if (g4_le32_load(header + 32 + (size_t)i * 4u) !=
                (uint32_t)model->layers[i].kv_dim)
            return 0;
    for (size_t i = 184; i < 216; i++)
        if (header[i] != 0) return 0;
    for (size_t i = 248; i < G4_KV_TRANSFER_HEADER; i++)
        if (header[i] != 0) return 0;
    *position_out = position;
    return 1;
}

size_t salt_gemma4_text_kv_export_size(const SaltGemma4Text *model) {
    size_t bytes = 0;
    if (!model || !all_identities_match(model) ||
        kv_transfer_bytes_at(model, model->position, &bytes) != 0)
        return 0;
    return bytes;
}

int salt_gemma4_text_kv_export(const SaltGemma4Text *model,
                               void *buffer, size_t buffer_size) {
    unsigned char *bytes = (unsigned char *)buffer;
    unsigned char *cursor;
    unsigned char payload_sha256[32];
    unsigned char logical_state_sha256[32];
    size_t expected = 0;
    if (!model || !buffer || !g4_binary32_little_endian() ||
        !all_identities_match(model) ||
        g4_transfer_logical_sha256_live(
            model, model->position, G4_TRANSFER_FORMAT_V6,
            model->kv_compat_sha256, logical_state_sha256) != 0 ||
        kv_transfer_bytes_at(model, model->position, &expected) != 0 ||
        expected != buffer_size || (uint64_t)buffer_size != buffer_size)
        return -1;

    cursor = bytes + G4_KV_TRANSFER_HEADER;
    for (int i = 0; i < G4_LAYERS; i++) {
        const G4Layer *layer = &model->layers[i];
        int base = g4_state_row_base(layer, model->position);
        int segments = g4_transfer_segment_count(
            layer, G4_TRANSFER_FORMAT_V6);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        if (segments < 1) return -1;
        for (int segment = 0; segment < segments; segment++)
            for (int position = base; position < model->position; position++) {
                const float *row = segments == 1 || segment == 1
                    ? g4_value_row(layer, position)
                    : g4_key_row(layer, position);
                if (!row) return -1;
                for (int j = 0; j < layer->kv_dim; j++)
                    if (!isfinite(row[j])) return -1;
                memcpy(cursor, row, row_bytes);
                cursor += row_bytes;
            }
    }
    if (cursor != bytes + buffer_size ||
        salt_sha256_bytes(bytes + G4_KV_TRANSFER_HEADER,
                          buffer_size - G4_KV_TRANSFER_HEADER,
                          payload_sha256) != 0 ||
        !all_identities_match(model))
        return -1;

    memset(bytes, 0, G4_KV_TRANSFER_HEADER);
    memcpy(bytes, G4_KV_TRANSFER_MAGIC, sizeof G4_KV_TRANSFER_MAGIC);
    g4_le32_store(bytes + 8, G4_KV_TRANSFER_VERSION);
    g4_le32_store(bytes + 12, G4_LAYERS);
    g4_le32_store(bytes + 16, (uint32_t)model->max_context);
    g4_le32_store(bytes + 20, (uint32_t)model->position);
    g4_le64_store(bytes + 24, (uint64_t)buffer_size);
    for (int i = 0; i < G4_LAYERS; i++)
        g4_le32_store(bytes + 32 + (size_t)i * 4u,
                      (uint32_t)model->layers[i].kv_dim);
    memcpy(bytes + 152, model->kv_compat_sha256,
           sizeof model->kv_compat_sha256);
    memcpy(bytes + 184, logical_state_sha256,
           sizeof logical_state_sha256);
    memcpy(bytes + 216, payload_sha256, sizeof payload_sha256);
    g4_le32_store(bytes + 248,
                  g4_transfer_sliding_base((uint32_t)model->position));
    g4_le32_store(bytes + 252, G4_SLIDING_WINDOW);
    return 0;
}

int salt_gemma4_text_kv_import(SaltGemma4Text *model,
                               const void *buffer, size_t buffer_size) {
    const unsigned char *bytes = (const unsigned char *)buffer;
    const unsigned char *cursor;
    unsigned char payload_sha256[32];
    unsigned char logical_state_sha256[32];
    uint32_t position;
    uint64_t transferred = 0;
    G4TransferFormat format;
    if (!model || !buffer || buffer_size < G4_KV_TRANSFER_HEADER ||
        !g4_binary32_little_endian() || !all_identities_match(model))
        return -1;
    if (g4_transfer_v6_geometry_valid(
            model, bytes, buffer_size, &position))
        format = G4_TRANSFER_FORMAT_V6;
    else if (g4_transfer_v5_geometry_valid(
            model, bytes, buffer_size, &position))
        format = G4_TRANSFER_FORMAT_V5;
    else if (g4_transfer_v4_geometry_valid(
            model, bytes, buffer_size, &position))
        format = G4_TRANSFER_FORMAT_V4;
    else
        return -1;
    if (salt_sha256_bytes(bytes + G4_KV_TRANSFER_HEADER,
                          buffer_size - G4_KV_TRANSFER_HEADER,
                          payload_sha256) != 0 ||
        memcmp(bytes + 216, payload_sha256, sizeof payload_sha256) != 0)
        return -1;

    cursor = bytes + G4_KV_TRANSFER_HEADER;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *layer = &model->layers[i];
        int input_base = format == G4_TRANSFER_FORMAT_V4 ? 0 :
            g4_state_row_base(layer, (int)position);
        int segments = g4_transfer_segment_count(layer, format);
        size_t count = (size_t)((int)position - input_base) *
                       (size_t)layer->kv_dim * (size_t)segments;
        if (segments < 1) return -1;
        for (size_t j = 0; j < count; j++) {
            float value;
            memcpy(&value, cursor + j * sizeof value, sizeof value);
            if (!isfinite(value)) return -1;
        }
        cursor += count * sizeof(float);
    }
    if (cursor != bytes + buffer_size || !all_identities_match(model))
        return -1;

    memset(model->kv_arena, 0,
           model->kv_arena_floats * sizeof *model->kv_arena);
    cursor = bytes + G4_KV_TRANSFER_HEADER;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *layer = &model->layers[i];
        int input_base = format == G4_TRANSFER_FORMAT_V4 ? 0 :
            g4_state_row_base(layer, (int)position);
        int retained_base = g4_state_row_base(layer, (int)position);
        int segments = g4_transfer_segment_count(layer, format);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        if (segments < 1) goto copy_fail;
        for (int segment = 0; segment < segments; segment++)
            for (int logical = input_base; logical < (int)position; logical++) {
                if (logical >= retained_base) {
                    float *row = segments == 1 || segment == 1
                        ? g4_private_value_row(layer, logical)
                        : g4_private_key_row(layer, logical);
                    if (!row) goto copy_fail;
                    memcpy(row, cursor, row_bytes);
                    if (memcmp(row, cursor, row_bytes) != 0) goto copy_fail;
                }
                cursor += row_bytes;
            }
        if (segments == 1)
            for (int logical = retained_base;
                    logical < (int)position; logical++)
                if (g4_reconstruct_shared_key_from_value(
                        model, layer, logical,
                        g4_private_key_row(layer, logical),
                        g4_private_value_row(layer, logical)) != 0)
                    goto copy_fail;
    }
    if (g4_gpu_kv_import(model, (int)position, &transferred) != 0 ||
        g4_text_state_publish(model, (int)position) != 0)
        goto copy_fail;
    if (!all_identities_match(model) ||
        (format != G4_TRANSFER_FORMAT_V4 &&
         (g4_transfer_logical_sha256_live(
              model, (int)position, format, bytes + 152,
              logical_state_sha256) != 0 ||
          memcmp(bytes + 184, logical_state_sha256,
                 sizeof logical_state_sha256) != 0)))
        goto copy_fail;
    return 0;

copy_fail:
    if (model->kv_arena && model->kv_arena_floats)
        memset(model->kv_arena, 0,
               model->kv_arena_floats * sizeof(float));
    (void)g4_gpu_kv_rewind(model, 0);
    (void)g4_text_state_publish(model, 0);
    return -1;
}

static int g4_kv_stream_write(FILE *stream, const void *data, size_t bytes) {
    return stream && data && bytes > 0 && fwrite(data, 1, bytes, stream) == bytes
        ? 0 : -1;
}

int salt_gemma4_text_kv_export_stream(const SaltGemma4Text *model,
                                      FILE *stream) {
    unsigned char header[G4_KV_TRANSFER_HEADER];
    unsigned char payload_sha256[32];
    unsigned char logical_state_sha256[32];
    SaltSha256 hasher;
    size_t expected = 0;
    if (!model || !stream || !g4_binary32_little_endian() ||
        !all_identities_match(model) ||
        g4_transfer_logical_sha256_live(
            model, model->position, G4_TRANSFER_FORMAT_V6,
            model->kv_compat_sha256, logical_state_sha256) != 0 ||
        kv_transfer_bytes_at(model, model->position, &expected) != 0)
        return -1;

    salt_sha256_init(&hasher);
    for (int i = 0; i < G4_LAYERS; i++) {
        const G4Layer *layer = &model->layers[i];
        int base = g4_state_row_base(layer, model->position);
        int segments = g4_transfer_segment_count(
            layer, G4_TRANSFER_FORMAT_V6);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        if (segments < 1) return -1;
        for (int segment = 0; segment < segments; segment++)
            for (int position = base; position < model->position; position++) {
                const float *row = segments == 1 || segment == 1
                    ? g4_value_row(layer, position)
                    : g4_key_row(layer, position);
                if (!row) return -1;
                for (int j = 0; j < layer->kv_dim; j++)
                    if (!isfinite(row[j])) return -1;
                salt_sha256_update(&hasher, row, row_bytes);
            }
    }
    salt_sha256_final(&hasher, payload_sha256);

    memset(header, 0, sizeof header);
    memcpy(header, G4_KV_TRANSFER_MAGIC, sizeof G4_KV_TRANSFER_MAGIC);
    g4_le32_store(header + 8, G4_KV_TRANSFER_VERSION);
    g4_le32_store(header + 12, G4_LAYERS);
    g4_le32_store(header + 16, (uint32_t)model->max_context);
    g4_le32_store(header + 20, (uint32_t)model->position);
    g4_le64_store(header + 24, (uint64_t)expected);
    for (int i = 0; i < G4_LAYERS; i++)
        g4_le32_store(header + 32 + (size_t)i * 4u,
                      (uint32_t)model->layers[i].kv_dim);
    memcpy(header + 152, model->kv_compat_sha256,
           sizeof model->kv_compat_sha256);
    memcpy(header + 184, logical_state_sha256,
           sizeof logical_state_sha256);
    memcpy(header + 216, payload_sha256, sizeof payload_sha256);
    g4_le32_store(header + 248,
                  g4_transfer_sliding_base((uint32_t)model->position));
    g4_le32_store(header + 252, G4_SLIDING_WINDOW);
    if (g4_kv_stream_write(stream, header, sizeof header) != 0)
        return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        const G4Layer *layer = &model->layers[i];
        int base = g4_state_row_base(layer, model->position);
        int segments = g4_transfer_segment_count(
            layer, G4_TRANSFER_FORMAT_V6);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        if (segments < 1) return -1;
        for (int segment = 0; segment < segments; segment++)
            for (int position = base; position < model->position; position++) {
                const float *row = segments == 1 || segment == 1
                    ? g4_value_row(layer, position)
                    : g4_key_row(layer, position);
                if (!row || g4_kv_stream_write(
                        stream, row, row_bytes) != 0)
                    return -1;
            }
    }
    return fflush(stream) == 0 && !ferror(stream) &&
           all_identities_match(model) ? 0 : -1;
}

int salt_gemma4_text_kv_import_stream_sha256(
        SaltGemma4Text *model, FILE *stream, uint8_t file_sha256[32]) {
    unsigned char header[G4_KV_TRANSFER_HEADER];
    unsigned char payload_sha256[32];
    unsigned char logical_state_sha256[32];
    SaltSha256 hasher, file_hasher;
    float scratch[G4_MAX_KV];
    uint32_t position;
    uint64_t encoded_bytes;
    G4TransferFormat format;
    if (!model || !stream || !g4_binary32_little_endian() ||
        !all_identities_match(model) ||
        fread(header, 1, sizeof header, stream) != sizeof header)
        goto fail;
    encoded_bytes = g4_le64_load(header + 24);
    if (encoded_bytes > SIZE_MAX) goto fail;
    if (g4_transfer_v6_geometry_valid(
            model, header, (size_t)encoded_bytes, &position))
        format = G4_TRANSFER_FORMAT_V6;
    else if (g4_transfer_v5_geometry_valid(
            model, header, (size_t)encoded_bytes, &position))
        format = G4_TRANSFER_FORMAT_V5;
    else if (g4_transfer_v4_geometry_valid(
            model, header, (size_t)encoded_bytes, &position))
        format = G4_TRANSFER_FORMAT_V4;
    else
        goto fail;
    for (int i = 0; i < G4_LAYERS; i++)
        if (model->layers[i].shared_position != 0)
            goto fail;

    salt_sha256_init(&hasher);
    salt_sha256_init(&file_hasher);
    salt_sha256_update(&file_hasher, header, sizeof header);
    memset(model->kv_arena, 0,
           model->kv_arena_floats * sizeof *model->kv_arena);
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *layer = &model->layers[i];
        int input_base = format == G4_TRANSFER_FORMAT_V4 ? 0 :
            g4_state_row_base(layer, (int)position);
        int retained_base = g4_state_row_base(layer, (int)position);
        int segments = g4_transfer_segment_count(layer, format);
        size_t row_bytes = (size_t)layer->kv_dim * sizeof(float);
        if (segments < 1) goto fail;
        for (int segment = 0; segment < segments; segment++) {
            for (int logical = input_base; logical < (int)position; logical++) {
                float *row = scratch;
                if (logical >= retained_base)
                    row = segments == 1 || segment == 1
                        ? g4_private_value_row(layer, logical)
                        : g4_private_key_row(layer, logical);
                if (!row || fread(row, 1, row_bytes, stream) != row_bytes)
                    goto fail;
                for (int j = 0; j < layer->kv_dim; j++)
                    if (!isfinite(row[j])) goto fail;
                salt_sha256_update(&hasher, row, row_bytes);
                salt_sha256_update(&file_hasher, row, row_bytes);
            }
        }
        if (segments == 1)
            for (int logical = retained_base;
                    logical < (int)position; logical++)
                if (g4_reconstruct_shared_key_from_value(
                        model, layer, logical,
                        g4_private_key_row(layer, logical),
                        g4_private_value_row(layer, logical)) != 0)
                    goto fail;
    }
    if (fgetc(stream) != EOF || ferror(stream)) goto fail;
    salt_sha256_final(&hasher, payload_sha256);
    if (memcmp(header + 216, payload_sha256, sizeof payload_sha256) != 0 ||
        !all_identities_match(model))
        goto fail;
    if (g4_text_state_publish(model, (int)position) != 0 ||
        (format != G4_TRANSFER_FORMAT_V4 &&
         (g4_transfer_logical_sha256_live(
              model, (int)position, format, header + 152,
              logical_state_sha256) != 0 ||
          memcmp(header + 184, logical_state_sha256,
                 sizeof logical_state_sha256) != 0)))
        goto fail;
    if (file_sha256) salt_sha256_final(&file_hasher, file_sha256);
    return 0;

fail:
    if (model && model->kv_arena && model->kv_arena_floats > 0)
        memset(model->kv_arena, 0,
               model->kv_arena_floats * sizeof *model->kv_arena);
    if (model) (void)g4_text_state_publish(model, 0);
    return -1;
}

int salt_gemma4_text_kv_import_stream(SaltGemma4Text *model, FILE *stream) {
    return salt_gemma4_text_kv_import_stream_sha256(model, stream, NULL);
}

static int g4_kv_stream_read_at(FILE *stream, uint64_t offset,
                                void *destination, size_t bytes) {
    off_t file_offset = (off_t)offset;
    if (!stream || !destination || bytes == 0 || file_offset < 0 ||
        (uint64_t)file_offset != offset ||
        fseeko(stream, file_offset, SEEK_SET) != 0)
        return -1;
    return fread(destination, 1, bytes, stream) == bytes ? 0 : -1;
}

static int g4_kv_stream_compare_at(FILE *stream, uint64_t offset,
                                   const void *expected, size_t bytes) {
    unsigned char scratch[4096];
    const unsigned char *reference = (const unsigned char *)expected;
    off_t file_offset = (off_t)offset;
    size_t compared = 0;
    if (!stream || !expected || bytes == 0 || file_offset < 0 ||
        (uint64_t)file_offset != offset ||
        fseeko(stream, file_offset, SEEK_SET) != 0)
        return -1;
    while (compared < bytes) {
        size_t count = bytes - compared;
        if (count > sizeof scratch) count = sizeof scratch;
        if (fread(scratch, 1, count, stream) != count ||
            memcmp(scratch, reference + compared, count) != 0)
            return -1;
        compared += count;
    }
    return 0;
}

int salt_gemma4_text_kv_import_delta_stream(
        SaltGemma4Text *model, FILE *stream,
        int source_position, int token_count) {
    unsigned char header[G4_KV_TRANSFER_HEADER];
    unsigned char logical_state_sha256[32];
    uint32_t reference_position;
    uint64_t cursor;
    uint64_t transferred = 0;
    size_t expected = 0;
    int result_position, shared_position, compare_prefix;
    G4TransferFormat format;
    if (!model || !stream || !g4_binary32_little_endian() ||
        !all_identities_match(model) || source_position < 0 || token_count < 1 ||
        source_position != model->position ||
        source_position > model->max_context - token_count ||
        g4_gpu_kv_materialize(model, source_position, &transferred) != 0)
        return -1;
    result_position = source_position + token_count;
    shared_position = model->layers[0].shared_position;
    if (shared_position != 0 && shared_position != source_position)
        return -1;
    compare_prefix = source_position > 0 && shared_position == source_position;
    if (fseeko(stream, 0, SEEK_SET) != 0 ||
        fread(header, 1, sizeof header, stream) != sizeof header)
        goto fail;
    if (g4_le64_load(header + 24) > SIZE_MAX) goto fail;
    if (g4_transfer_v6_geometry_valid(
            model, header, (size_t)g4_le64_load(header + 24),
            &reference_position))
        format = G4_TRANSFER_FORMAT_V6;
    else if (g4_transfer_v5_geometry_valid(
            model, header, (size_t)g4_le64_load(header + 24),
            &reference_position))
        format = G4_TRANSFER_FORMAT_V5;
    else if (g4_transfer_v4_geometry_valid(
            model, header, (size_t)g4_le64_load(header + 24),
            &reference_position))
        format = G4_TRANSFER_FORMAT_V4;
    else
        goto fail;
    if (reference_position != (uint32_t)result_position ||
        g4_transfer_bytes_at_version(
            model, result_position, format, &expected) != 0 ||
        g4_le64_load(header + 24) != (uint64_t)expected)
        goto fail;
    for (int layer = 0; layer < G4_LAYERS; layer++)
        if (model->layers[layer].shared_position != shared_position)
            goto fail;

    cursor = G4_KV_TRANSFER_HEADER;
    for (int layer_index = 0; layer_index < G4_LAYERS; layer_index++) {
        G4Layer *layer = &model->layers[layer_index];
        int base = format == G4_TRANSFER_FORMAT_V4 ? 0 :
            g4_state_row_base(layer, (int)reference_position);
        int segments = g4_transfer_segment_count(layer, format);
        uint64_t row_bytes = (uint64_t)(uint32_t)layer->kv_dim * sizeof(float);
        uint64_t rows = (uint64_t)(reference_position - (uint32_t)base);
        uint64_t segment_bytes = rows * row_bytes;
        if (base < 0 || source_position < base || segments < 1 ||
            row_bytes == 0 ||
            rows > UINT64_MAX / row_bytes ||
            segment_bytes > (UINT64_MAX - cursor) / (uint64_t)segments)
            goto fail;
        if (compare_prefix)
            for (int logical = base; logical < source_position; logical++) {
                uint64_t offset =
                    (uint64_t)(uint32_t)(logical - base) * row_bytes;
                const float *key = g4_key_row(layer, logical);
                const float *value = g4_value_row(layer, logical);
                if (!key || !value || row_bytes > SIZE_MAX ||
                    (segments == 1
                        ? g4_kv_stream_compare_at(
                            stream, cursor + offset, value,
                            (size_t)row_bytes) != 0
                        : (g4_kv_stream_compare_at(
                               stream, cursor + offset, key,
                               (size_t)row_bytes) != 0 ||
                           g4_kv_stream_compare_at(
                               stream, cursor + segment_bytes + offset, value,
                               (size_t)row_bytes) != 0)))
                    goto fail;
            }
        for (int logical = source_position; logical < result_position; logical++) {
            uint64_t offset =
                (uint64_t)(uint32_t)(logical - base) * row_bytes;
            float *key = g4_private_key_row(layer, logical);
            float *value = g4_private_value_row(layer, logical);
            if (!key || !value || row_bytes > SIZE_MAX ||
                (segments == 1
                    ? (g4_kv_stream_read_at(
                           stream, cursor + offset, value,
                           (size_t)row_bytes) != 0 ||
                       g4_reconstruct_shared_key_from_value(
                           model, layer, logical, key, value) != 0)
                    : (g4_kv_stream_read_at(
                           stream, cursor + offset, key,
                           (size_t)row_bytes) != 0 ||
                       g4_kv_stream_read_at(
                           stream, cursor + segment_bytes + offset,
                           value, (size_t)row_bytes) != 0)))
                goto fail;
            for (int component = 0; component < layer->kv_dim; component++)
                if (!isfinite(key[component]) || !isfinite(value[component]))
                    goto fail;
        }
        cursor += (uint64_t)segments * segment_bytes;
    }
    if (cursor != (uint64_t)expected || !all_identities_match(model)) goto fail;
    if (g4_gpu_kv_import(model, result_position, &transferred) != 0 ||
        g4_text_state_publish(model, result_position) != 0 ||
        (format != G4_TRANSFER_FORMAT_V4 &&
         (g4_transfer_logical_sha256_live(
              model, result_position, format, header + 152,
              logical_state_sha256) != 0 ||
          memcmp(header + 184, logical_state_sha256,
                 sizeof logical_state_sha256) != 0)))
        goto fail;
    return 0;

fail:
    if (model) {
        int safe_position = shared_position > 0 &&
            shared_position == source_position ? shared_position : 0;
        if (model->kv_arena && model->kv_arena_floats)
            for (int layer = 0; layer < G4_LAYERS; layer++) {
                G4Layer *item = &model->layers[layer];
                size_t count = (size_t)item->kv_capacity *
                    (size_t)item->kv_dim;
                size_t key_start = item->shared_key_private
                    ? (size_t)item->shared_position * (size_t)item->kv_dim
                    : 0;
                if (key_start <= count) {
                    memset(item->key_cache + key_start, 0,
                           (count - key_start) * sizeof(float));
                    memset(item->value_cache, 0, count * sizeof(float));
                }
            }
        (void)g4_gpu_kv_rewind(model, safe_position);
        (void)g4_text_state_publish(model, safe_position);
    }
    return -1;
}

int salt_gemma4_text_kv_import_facts_stream(
        SaltGemma4Text *model, FILE *stream,
        int mindset_end, int token_count) {
    unsigned char before[32], after[32];
    if (!model || !stream || mindset_end < 1 || token_count < 1 ||
        model->position != mindset_end ||
        salt_gemma4_text_shared_position(model) != mindset_end ||
        salt_gemma4_text_prefix_sha256(model, mindset_end, before) != 0)
        return -1;
    if (salt_gemma4_text_kv_import_delta_stream(
            model, stream, mindset_end, token_count) != 0)
        return -1;
    if (salt_gemma4_text_prefix_sha256(model, mindset_end, after) != 0 ||
        memcmp(before, after, sizeof before) != 0) {
        (void)salt_gemma4_text_clear_facts(model, mindset_end);
        return -1;
    }
    return 0;
}

int salt_gemma4_text_kv_attach_shared(SaltGemma4Text *model,
                                      const void *buffer, size_t buffer_size) {
    const unsigned char *bytes = (const unsigned char *)buffer;
    const unsigned char *cursor;
    const float *keys[G4_LAYERS], *values[G4_LAYERS];
    int bases[G4_LAYERS], private_keys[G4_LAYERS];
    unsigned char payload_sha256[32];
    unsigned char logical_state_sha256[32];
    uint32_t position;
    G4TransferFormat format;
    if (!model || !buffer || model->position != 0 ||
        buffer_size < G4_KV_TRANSFER_HEADER ||
        !g4_binary32_little_endian() || !all_identities_match(model))
        return -1;
    if (g4_transfer_v6_geometry_valid(
            model, bytes, buffer_size, &position))
        format = G4_TRANSFER_FORMAT_V6;
    else if (g4_transfer_v5_geometry_valid(
            model, bytes, buffer_size, &position))
        format = G4_TRANSFER_FORMAT_V5;
    else if (g4_transfer_v4_geometry_valid(
            model, bytes, buffer_size, &position))
        format = G4_TRANSFER_FORMAT_V4;
    else
        return -1;
    for (int i = 0; i < G4_LAYERS; i++)
        if (model->layers[i].shared_position != 0 ||
            model->layers[i].shared_key_private)
            return -1;
    if (salt_sha256_bytes(bytes + G4_KV_TRANSFER_HEADER,
                          buffer_size - G4_KV_TRANSFER_HEADER,
                          payload_sha256) != 0 ||
        memcmp(bytes + 216, payload_sha256, sizeof payload_sha256) != 0)
        return -1;
    cursor = bytes + G4_KV_TRANSFER_HEADER;
    for (int i = 0; i < G4_LAYERS; i++) {
        G4Layer *layer = &model->layers[i];
        int base = format == G4_TRANSFER_FORMAT_V4 ? 0 :
            g4_state_row_base(layer, (int)position);
        int segments = g4_transfer_segment_count(layer, format);
        size_t count = (size_t)((int)position - base) *
            (size_t)layer->kv_dim;
        size_t byte_count = count * sizeof(float);
        bases[i] = base;
        if (segments == 1) {
            keys[i] = layer->key_cache;
            values[i] = (const float *)(const void *)cursor;
            private_keys[i] = 1;
            for (size_t j = 0; j < count; j++) {
                float value;
                memcpy(&value, cursor + j * sizeof value, sizeof value);
                if (!isfinite(value)) return -1;
            }
            cursor += byte_count;
            for (int logical = base; logical < (int)position; logical++)
                if (g4_reconstruct_shared_key_from_value(
                        model, layer, logical,
                        layer->key_cache + (size_t)logical * layer->kv_dim,
                        values[i] + (size_t)(logical - base) * layer->kv_dim) != 0)
                    return -1;
        } else if (segments == 2) {
            keys[i] = (const float *)(const void *)cursor;
            private_keys[i] = 0;
            for (size_t j = 0; j < count; j++) {
                float value;
                memcpy(&value, cursor + j * sizeof value, sizeof value);
                if (!isfinite(value)) return -1;
            }
            cursor += byte_count;
            values[i] = (const float *)(const void *)cursor;
            for (size_t j = 0; j < count; j++) {
                float value;
                memcpy(&value, cursor + j * sizeof value, sizeof value);
                if (!isfinite(value)) return -1;
            }
            cursor += byte_count;
        } else {
            return -1;
        }
    }
    if (cursor != bytes + buffer_size || !all_identities_match(model)) return -1;
    for (int i = 0; i < G4_LAYERS; i++) {
        model->layers[i].shared_key_cache = keys[i];
        model->layers[i].shared_value_cache = values[i];
        model->layers[i].shared_row_base = bases[i];
        model->layers[i].shared_position = (int)position;
        model->layers[i].shared_key_private = private_keys[i];
    }
    if (g4_text_shared_prefix_refresh(model) != 0 ||
        g4_text_state_publish(model, (int)position) != 0 ||
        (format != G4_TRANSFER_FORMAT_V4 &&
         (g4_transfer_logical_sha256_live(
              model, (int)position, format, bytes + 152,
              logical_state_sha256) != 0 ||
          memcmp(bytes + 184, logical_state_sha256,
                 sizeof logical_state_sha256) != 0))) {
        for (int i = 0; i < G4_LAYERS; i++) {
            if (model->layers[i].shared_key_private)
                memset(model->layers[i].key_cache, 0,
                    (size_t)model->layers[i].shared_position *
                    (size_t)model->layers[i].kv_dim * sizeof(float));
            model->layers[i].shared_key_cache = NULL;
            model->layers[i].shared_value_cache = NULL;
            model->layers[i].shared_row_base = 0;
            model->layers[i].shared_position = 0;
            model->layers[i].shared_key_private = 0;
        }
        (void)g4_text_shared_prefix_refresh(model);
        (void)g4_text_state_publish(model, 0);
        return -1;
    }
    return 0;
}

int salt_gemma4_text_shared_position(const SaltGemma4Text *model) {
    if (!g4_text_state_aligned(model)) return -1;
    return model->layers[0].shared_position;
}

int salt_gemma4_text_position(const SaltGemma4Text *model) {
    return g4_text_state_aligned(model) ? model->position : -1;
}

int salt_gemma4_text_context_capacity(const SaltGemma4Text *model) {
    return model && model->max_context > 0 ? model->max_context : -1;
}

int salt_gemma4_text_prefill_chunk_capacity(const SaltGemma4Text *model) {
    return model && model->prefill_capacity > 0 ? model->prefill_capacity : -1;
}

size_t salt_gemma4_text_kv_capacity_bytes(const SaltGemma4Text *model) {
    if (!model || !model->kv_arena ||
        model->kv_arena_floats > SIZE_MAX / sizeof(float))
        return 0;
    return model->kv_arena_floats * sizeof(float);
}

size_t salt_gemma4_text_kv_required_bytes(const SaltGemma4Text *model,
                                          int position) {
    size_t total = 0;
    if (!model || position < 0 || position > model->max_context)
        return 0;
    for (int i = 0; i < G4_LAYERS; i++) {
        const G4Layer *layer = &model->layers[i];
        size_t rows = (size_t)position;
        size_t dim = (size_t)layer->kv_dim;
        size_t add;
        if (layer->kv_ring && rows > (size_t)layer->kv_capacity)
            rows = (size_t)layer->kv_capacity;
        if (!dim || rows > SIZE_MAX / dim ||
            rows * dim > SIZE_MAX / (2u * sizeof(float)))
            return 0;
        add = rows * dim * 2u * sizeof(float);
        if (total > SIZE_MAX - add) return 0;
        total += add;
    }
    return total;
}

int salt_gemma4_text_vocab_size(const SaltGemma4Text *model) {
    return model ? G4_VOCAB : -1;
}

static void state_hash_u32(SaltSha256 *state, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)value,
        (unsigned char)(value >> 8),
        (unsigned char)(value >> 16),
        (unsigned char)(value >> 24),
    };
    salt_sha256_update(state, bytes, sizeof bytes);
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
/* State audits are explicit materialization boundaries. Salt owns the proof
 * request; this model handle makes its device-live KV representation visible
 * to the canonical format-independent hash without changing logical state. */
static int g4_state_audit_materialize(const SaltGemma4Text *model) {
    uint64_t bytes = 0;
    if (!model || model->position < 0 || model->position > model->max_context)
        return -1;
    return g4_gpu_kv_materialize((SaltGemma4Text *)(void *)model,
        model->position, &bytes);
}
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */

static int g4_transfer_logical_sha256_live(
        const SaltGemma4Text *model, int position, G4TransferFormat format,
        const unsigned char compatibility[32], uint8_t out[32]) {
    static const unsigned char domain_v5[8] = {
        'G', '4', 'S', 'T', 'A', 'T', 'E', '4'
    };
    SaltSha256 state;
    if (!model || !compatibility || !out || position < 0 ||
        position > model->position || position > model->max_context ||
        (format != G4_TRANSFER_FORMAT_V5 &&
         format != G4_TRANSFER_FORMAT_V6) || !all_identities_match(model))
        return -1;
    if (format == G4_TRANSFER_FORMAT_V6)
        return state_sha256_at(model, position, out);
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (g4_state_audit_materialize(model) != 0) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    salt_sha256_init(&state);
    salt_sha256_update(&state, domain_v5, sizeof domain_v5);
    salt_sha256_update(&state, compatibility, 32u);
    state_hash_u32(&state, G4_LAYERS);
    state_hash_u32(&state, (uint32_t)model->max_context);
    state_hash_u32(&state, (uint32_t)position);
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        const G4Layer *item = &model->layers[layer];
        int base = g4_state_row_base(item, position);
        int segments = g4_transfer_segment_count(item, format);
        size_t row_bytes = (size_t)item->kv_dim * sizeof(float);
        if (item->kv_dim < 1 || base < 0 || base > position ||
            segments < 1 ||
            row_bytes / sizeof(float) != (size_t)item->kv_dim)
            return -1;
        state_hash_u32(&state, (uint32_t)item->kv_dim);
        state_hash_u32(&state, (uint32_t)base);
        for (int segment = 0; segment < segments; segment++)
            for (int row = base; row < position; row++) {
                const float *value = segments == 1 || segment == 1
                    ? g4_value_row(item, row)
                    : g4_key_row(item, row);
                if (!value) return -1;
                salt_sha256_update(&state, value, row_bytes);
            }
    }
    salt_sha256_final(&state, out);
    return 0;
}

static int state_sha256_at(const SaltGemma4Text *model, int position,
                           uint8_t out[32]) {
    static const unsigned char domain[8] = {
        'G', '4', 'S', 'T', 'A', 'T', 'E', '4'
    };
    SaltSha256 state;
    if (!model || !out || position < 0 || position > model->position ||
        position > model->max_context || !all_identities_match(model))
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (g4_state_audit_materialize(model) != 0) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    salt_sha256_init(&state);
    salt_sha256_update(&state, domain, sizeof domain);
    salt_sha256_update(&state, model->kv_compat_sha256,
                       sizeof model->kv_compat_sha256);
    state_hash_u32(&state, G4_LAYERS);
    state_hash_u32(&state, (uint32_t)model->max_context);
    state_hash_u32(&state, (uint32_t)position);
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        const G4Layer *item = &model->layers[layer];
        int base = g4_state_row_base(item, position);
        size_t row_bytes = (size_t)item->kv_dim * sizeof(float);
        if (item->kv_dim < 1 || base < 0 || base > position ||
            row_bytes / sizeof(float) != (size_t)item->kv_dim)
            return -1;
        state_hash_u32(&state, (uint32_t)item->kv_dim);
        state_hash_u32(&state, (uint32_t)base);
        for (int row = base; row < position; row++) {
            const float *value = g4_key_row(item, row);
            if (!value) return -1;
            salt_sha256_update(&state, value, row_bytes);
        }
        for (int row = base; row < position; row++) {
            const float *value = g4_value_row(item, row);
            if (!value) return -1;
            salt_sha256_update(&state, value, row_bytes);
        }
    }
    salt_sha256_final(&state, out);
    return 0;
}

int salt_gemma4_text_state_sha256(const SaltGemma4Text *model,
                                  uint8_t out[32]) {
    return model ? state_sha256_at(model, model->position, out) : -1;
}

int salt_gemma4_text_transition_sha256(const SaltGemma4Text *model,
                                       uint8_t out[32]) {
    static const unsigned char domain[8] = {
        'G', '4', 'D', 'P', 'R', 'T', 'R', '2'
    };
    SaltSha256 state;
    if (!model || !out || model->position < 0 ||
        model->position > model->max_context || !all_identities_match(model))
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (g4_state_audit_materialize(model) != 0) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    salt_sha256_init(&state);
    salt_sha256_update(&state, domain, sizeof domain);
    salt_sha256_update(&state, model->kv_compat_sha256,
                       sizeof model->kv_compat_sha256);
    state_hash_u32(&state, G4_LAYERS);
    state_hash_u32(&state, (uint32_t)model->max_context);
    state_hash_u32(&state, (uint32_t)model->position);
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        const G4Layer *item = &model->layers[layer];
        int base = g4_state_row_base(item, model->position);
        size_t row_bytes = (size_t)item->kv_dim * sizeof(float);
        if (item->kv_dim < 1 || base < 0 || base > model->position ||
            row_bytes / sizeof(float) != (size_t)item->kv_dim)
            return -1;
        state_hash_u32(&state, (uint32_t)item->kv_dim);
        state_hash_u32(&state, (uint32_t)base);
        for (int row = base; row < model->position; row++) {
            const float *value = g4_key_row(item, row);
            if (!value) return -1;
            salt_sha256_update(&state, value, row_bytes);
        }
        for (int row = base; row < model->position; row++) {
            const float *value = g4_value_row(item, row);
            if (!value) return -1;
            salt_sha256_update(&state, value, row_bytes);
        }
    }
    salt_sha256_final(&state, out);
    return 0;
}

int salt_gemma4_text_build_identity_sha256(const SaltGemma4Text *model,
                                           uint8_t out[32]) {
    static const unsigned char zero[32] = {0};
    if (!model || !out ||
        memcmp(model->build_identity_sha256, zero, sizeof zero) == 0)
        return -1;
    memcpy(out, model->build_identity_sha256,
           sizeof model->build_identity_sha256);
    return 0;
}

int salt_gemma4_text_facts_sha256(const SaltGemma4Text *model,
                                  int mindset_end, uint8_t out[32]) {
    static const unsigned char domain[8] = {
        'G', '4', 'F', 'A', 'C', 'T', 'S', '3'
    };
    SaltSha256 state;
    if (!model || !out || mindset_end < 0 ||
        mindset_end > model->position || !all_identities_match(model))
        return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    if (g4_state_audit_materialize(model) != 0) return -1;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    salt_sha256_init(&state);
    salt_sha256_update(&state, domain, sizeof domain);
    salt_sha256_update(&state, model->kv_compat_sha256,
                       sizeof model->kv_compat_sha256);
    state_hash_u32(&state, G4_LAYERS);
    state_hash_u32(&state, (uint32_t)model->max_context);
    state_hash_u32(&state, (uint32_t)mindset_end);
    state_hash_u32(&state, (uint32_t)model->position);
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        const G4Layer *item = &model->layers[layer];
        int base = g4_state_row_base(item, model->position);
        size_t row_bytes = (size_t)item->kv_dim * sizeof(float);
        if (base < mindset_end) base = mindset_end;
        if (item->kv_dim < 1 || base < 0 || base > model->position ||
            row_bytes / sizeof(float) != (size_t)item->kv_dim)
            return -1;
        state_hash_u32(&state, (uint32_t)item->kv_dim);
        state_hash_u32(&state, (uint32_t)base);
        for (int row = base; row < model->position; row++) {
            const float *value = g4_key_row(item, row);
            if (!value) return -1;
            salt_sha256_update(&state, value, row_bytes);
        }
        for (int row = base; row < model->position; row++) {
            const float *value = g4_value_row(item, row);
            if (!value) return -1;
            salt_sha256_update(&state, value, row_bytes);
        }
    }
    salt_sha256_final(&state, out);
    return 0;
}

int salt_gemma4_text_prefix_sha256(const SaltGemma4Text *model,
                                   int position, uint8_t out[32]) {
    return state_sha256_at(model, position, out);
}

int salt_gemma4_text_clear_facts(SaltGemma4Text *model, int mindset_end) {
    int old_position;
    uint64_t materialized = 0;
    if (!model || mindset_end < 0 || mindset_end > model->position)
        return -1;
    for (int layer = 0; layer < G4_LAYERS; layer++)
        if (model->layers[layer].shared_position > mindset_end)
            return -1;
    old_position = model->position;
    if (g4_gpu_kv_materialize(model, old_position, &materialized) != 0)
        return -1;
    if (model->kv_pageable_mapping && mindset_end == 0) {
        int zero_fd;
        void *contents;
        if (salt_gpu_sync() != 0 ||
                (zero_fd = open("/dev/zero", O_RDWR)) < 0)
            return -1;
        contents = mmap(model->kv_shared.contents, model->kv_shared.nbytes,
                        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED,
                        zero_fd, 0);
        if (close(zero_fd) != 0 || contents == MAP_FAILED ||
                contents != model->kv_shared.contents)
            return -1;
        return g4_text_state_publish(model, 0);
    }
    for (int layer = 0; layer < G4_LAYERS; layer++) {
        G4Layer *item = &model->layers[layer];
        size_t count, key_start = 0;
        if (item->kv_dim < 1 || item->kv_capacity < 1 ||
            (size_t)item->kv_capacity >
                SIZE_MAX / (size_t)item->kv_dim)
            return -1;
        count = (size_t)item->kv_capacity * (size_t)item->kv_dim;
        if (item->shared_key_private) {
            if (item->shared_position < 0 ||
                item->shared_position > item->kv_capacity)
                return -1;
            key_start = (size_t)item->shared_position *
                (size_t)item->kv_dim;
        }
        memset(item->key_cache + key_start, 0,
               (count - key_start) * sizeof(float));
        memset(item->value_cache, 0, count * sizeof(float));
    }
    return g4_gpu_kv_rewind(model, mindset_end) == 0
        ? g4_text_state_publish(model, mindset_end) : -1;
}

int salt_gemma4_text_use_ondemand_rope_for_proof(SaltGemma4Text *model,
                                                  int enabled) {
    if (!model || !g4_text_state_aligned(model) ||
        (enabled != 0 && enabled != 1) || model->position != 0)
        return -1;
    model->use_ondemand_rope_proof = enabled;
    return 0;
}

int salt_gemma4_text_prepare_attention_plan_leased(
        SaltGemma4Text *model, const SaltDprComputeIntent *intent,
        const SaltDprAttentionPlan *plan,
        SaltGemma4AttentionPlanLease *lease,
        SaltGemma4AttentionPrepareStats *stats) {
    int position;
    if (!model || !intent || !plan || !lease || !stats ||
        !g4_text_state_aligned(model) || model->attention_plan_lease ||
        lease->active) return -1;
    memset(lease, 0, sizeof *lease);
    memset(stats, 0, sizeof *stats);
    if (salt_dpr_attention_plan_validate(
            plan, intent, G4_LAYERS, G4_EXPERTS,
            (uint64_t)model->max_context) != 1)
        return -1;
    stats->requested_items = plan->item_count;
    if (!model->expert_cache_set || model->nvfp4_mode) return 1;
    for (uint32_t i = 0; i < plan->item_count; i++)
        if (plan->items[i].kind != SALT_DPR_RELEVANCE_EXPERT_PREFETCH)
            return 1;
    position = model->position;
    for (uint32_t i = 0; i < plan->item_count; i++) {
        int expert = (int)plan->items[i].index;
        int slot = -1;
        const void *payload;
        int fetched = salt_cache_getmany(
            &model->expert_cache, (int)plan->items[i].layer,
            &expert, 1, &slot);
        if (fetched < 0 || !salt_cache_slot_for(
                &model->expert_cache, slot,
                (int)plan->items[i].layer, expert) ||
            !(payload = salt_cache_acquire(
                &model->expert_cache, slot,
                (int)plan->items[i].layer, expert))) {
            while (lease->count > 0) {
                SaltGemma4AttentionLeaseEntry *entry =
                    &lease->entries[--lease->count];
                (void)salt_cache_release(&model->expert_cache, entry->slot,
                    (int)entry->layer, (int)entry->expert);
            }
            return -1;
        }
        (void)payload;
        lease->entries[lease->count++] = (SaltGemma4AttentionLeaseEntry) {
            plan->items[i].layer, plan->items[i].index, slot
        };
        stats->prepared_items++;
        if (fetched == 0) stats->resident_hits++;
        stats->fetch_jobs += (uint32_t)fetched;
    }
    if (model->position != position) {
        while (lease->count > 0) {
            SaltGemma4AttentionLeaseEntry *entry =
                &lease->entries[--lease->count];
            (void)salt_cache_release(&model->expert_cache, entry->slot,
                (int)entry->layer, (int)entry->expert);
        }
        return -1;
    }
    lease->active = 1;
    model->attention_plan_lease = lease;
    return 0;
}

int salt_gemma4_text_release_attention_plan_lease(
        SaltGemma4Text *model, SaltGemma4AttentionPlanLease *lease) {
    int rc = 0;
    if (!model || !lease || !lease->active ||
        model->attention_plan_lease != lease)
        return -1;
    for (uint32_t i = lease->count; i > 0; i--) {
        SaltGemma4AttentionLeaseEntry *entry = &lease->entries[i - 1u];
        if (salt_cache_release(&model->expert_cache, entry->slot,
                (int)entry->layer, (int)entry->expert) != 0)
            rc = -1;
    }
    model->attention_plan_lease = NULL;
    lease->active = 0;
    return rc;
}

int salt_gemma4_text_prepare_attention_plan(
        SaltGemma4Text *model, const SaltDprComputeIntent *intent,
        const SaltDprAttentionPlan *plan,
        SaltGemma4AttentionPrepareStats *stats) {
    SaltGemma4AttentionPlanLease lease = {0};
    int rc = salt_gemma4_text_prepare_attention_plan_leased(
        model, intent, plan, &lease, stats);
    if (rc != 0) return rc;
    return salt_gemma4_text_release_attention_plan_lease(model, &lease);
}

int salt_gemma4_text_route_observer_begin(
        SaltGemma4Text *model, SaltGemma4RouteObserver *observer) {
    if (!model || !observer || !observer->items || observer->capacity == 0 ||
        model->route_observer)
        return -1;
    observer->count = 0;
    observer->overflow = 0;
    model->route_observer = observer;
    return 0;
}

int salt_gemma4_text_route_observer_end(SaltGemma4Text *model) {
    if (!model || !model->route_observer) return -1;
    model->route_observer = NULL;
    return 0;
}

static int route_observer_counts(
        const SaltGemma4RouteObserver *observer, uint32_t route_kind,
        uint64_t counts[G4_LAYERS][G4_EXPERTS]) {
    int matched = 0;
    if (!observer || !observer->items || observer->overflow ||
        observer->count > observer->capacity || !counts ||
        (route_kind != SALT_GEMMA4_ROUTE_DECODE &&
         route_kind != SALT_GEMMA4_ROUTE_PREFILL))
        return -1;
    memset(counts, 0,
           (size_t)G4_LAYERS * G4_EXPERTS * sizeof counts[0][0]);
    for (size_t index = 0; index < observer->count; index++) {
        const SaltGemma4RouteObservation *item = &observer->items[index];
        if (item->kind != route_kind) continue;
        if (item->layer >= G4_LAYERS || item->expert >= G4_EXPERTS ||
            counts[item->layer][item->expert] == UINT64_MAX)
            return -1;
        counts[item->layer][item->expert]++;
        matched = 1;
    }
    return matched ? 0 : 1;
}

int salt_gemma4_text_route_observer_compile_attention_plan(
        const SaltGemma4RouteObserver *observer, uint32_t route_kind,
        uint32_t maximum_per_layer, uint32_t maximum_items,
        SaltDprAttentionPlan *plan) {
    uint64_t counts[G4_LAYERS][G4_EXPERTS];
    int counted;
    if (!plan) return -1;
    counted = route_observer_counts(observer, route_kind, counts);
    if (counted < 0) return -1;
    if (counted > 0) {
        memset(plan->items, 0, sizeof plan->items);
        plan->item_count = 0;
        plan->expected_saved_numerator = 0;
        plan->expected_saved_denominator = 0;
        return 1;
    }
    return salt_dpr_attention_plan_compile_expert_coverage(
        plan, &counts[0][0], G4_LAYERS, G4_EXPERTS,
        maximum_per_layer, maximum_items);
}

int salt_gemma4_text_route_observer_update_coverage_ledger(
        const SaltGemma4RouteObserver *observer, uint32_t route_kind,
        SaltDprExpertCoverageLedger *ledger) {
    uint64_t counts[G4_LAYERS][G4_EXPERTS];
    int counted;
    if (!ledger || ledger->layer_count != G4_LAYERS ||
        ledger->expert_count != G4_EXPERTS)
        return -1;
    counted = route_observer_counts(observer, route_kind, counts);
    if (counted != 0) return counted;
    return salt_dpr_expert_coverage_ledger_merge(ledger, &counts[0][0]);
}

int salt_gemma4_text_memory_stats(const SaltGemma4Text *model,
                                  SaltGemma4MemoryStats *stats) {
    int resident;
    SaltGpuWeightResourceUsage gpu_weight_usage;
    if (!model || !stats ||
        (!model->expert_cache_set && !model->nvfp4_mode)) return -1;
    memset(stats, 0, sizeof *stats);
    stats->proof_validation_calls = model->proof_validation_calls;
    stats->kv_arena_bytes = salt_gemma4_text_kv_capacity_bytes(model);
    stats->startup_limit_bytes = model->memory_limit_bytes;
    stats->startup_forecast_bytes = model->startup_forecast_bytes;
    stats->startup_source_bytes = model->startup_source_bytes;
    stats->startup_registered_bytes = model->startup_registered_bytes;
    stats->startup_pageable_bytes = model->startup_pageable_bytes;
    stats->startup_expert_residency_bytes =
        model->startup_expert_residency_bytes;
    stats->startup_host_runtime_bytes = model->startup_host_runtime_bytes;
    stats->startup_host_kv_bytes = model->startup_host_kv_bytes;
    stats->startup_shared_arena_bytes = model->startup_shared_arena_bytes;
    stats->startup_backend_pinned_bytes =
        model->startup_backend_pinned_bytes;
    stats->startup_backend_device_bytes =
        model->startup_backend_device_bytes;
    stats->startup_device_kv_bytes = model->startup_device_kv_bytes;
    stats->startup_cache_metadata_bytes = model->startup_cache_metadata_bytes;
    stats->startup_rope_bytes = model->startup_rope_bytes;
    stats->startup_descriptor_bytes = model->startup_descriptor_bytes;
    stats->startup_admitted = model->startup_admitted;
    resident = expert_cache_occupancy(model);
    stats->expert_budget_bytes = model->expert_budget_bytes;
    stats->expert_slot_bytes = G4_SLOT_BYTES;
    stats->expert_capacity_slots = model->expert_cache.nslot;
    stats->expert_capacity_bytes =
        (uint64_t)model->expert_cache.nslot * G4_SLOT_BYTES;
    stats->expert_resident_slots = resident;
    stats->expert_peak_slots = model->expert_cache_peak_slots;
    stats->expert_preload_enabled = model->expert_preload_enabled;
    stats->expert_preloaded_slots = model->expert_preloaded_slots;
    stats->expert_preload_layer_count = model->expert_preload_layer_count;
    stats->expert_preload_protected_slots =
        model->expert_preload_protected_slots;
    stats->expert_preload_layer_mask = model->expert_preload_layer_mask;
    stats->expert_preload_fetch_jobs = model->expert_preload_fetch_jobs;
    stats->expert_resident_logical_bytes = (uint64_t)resident * G4_SLOT_BYTES;
    stats->expert_peak_logical_bytes =
        (uint64_t)model->expert_cache_peak_slots * G4_SLOT_BYTES;
    stats->expert_requests = (uint64_t)model->expert_cache.nreq;
    stats->expert_hits = (uint64_t)model->expert_cache.nhit;
    stats->expert_misses = (uint64_t)model->expert_cache.nmiss;
    stats->expert_evictions = (uint64_t)model->expert_cache.nevict;
    stats->expert_release_calls =
        (uint64_t)model->expert_cache.madvise_calls;
    stats->expert_release_failures =
        (uint64_t)model->expert_cache.madvise_failures;
    stats->expert_released_page_bytes =
        (uint64_t)model->expert_cache.madvise_bytes;
    stats->expert_mapped_bytes =
        (uint64_t)model->expert_cache.mapped_current_bytes;
    stats->expert_peak_mapped_bytes =
        (uint64_t)model->expert_cache.mapped_peak_bytes;
    stats->expert_unmap_calls =
        (uint64_t)model->expert_cache.munmap_calls;
    stats->expert_unmap_failures =
        (uint64_t)model->expert_cache.munmap_failures;
    stats->expert_unmapped_bytes =
        (uint64_t)model->expert_cache.munmap_bytes;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */
    stats->expert_prepare_calls =
        (uint64_t)model->expert_cache.nprepare_calls;
    stats->expert_prepare_failures =
        (uint64_t)model->expert_cache.nprepare_failures;
    stats->expert_prepared_bytes =
        (uint64_t)model->expert_cache.prepared_bytes;
/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */
    stats->expert_union_fetch_calls = model->expert_union_fetch_calls;
    stats->expert_union_fetch_experts = model->expert_union_fetch_experts;
    stats->expert_gate_up_epochs = model->expert_gate_up_epochs;
    stats->expert_down_epochs = model->expert_down_epochs;
    stats->expert_matrix_gate_up_waves =
        model->expert_matrix_gate_up_waves;
    stats->expert_matrix_down_waves = model->expert_matrix_down_waves;
    stats->expert_matrix_projection_jobs =
        model->expert_matrix_projection_jobs;
    stats->prefill_expert_matrix_flow_sessions =
        model->prefill_expert_matrix_flow_sessions;
    stats->target_matrix_flow_sessions = model->target_matrix_flow_sessions;
    stats->target_gpu_expert_batches = model->target_gpu_expert_batches;
    stats->target_gpu_expert_jobs = model->target_gpu_expert_jobs;
    stats->q4_pool_submissions = model->q4_pool_submissions;
    stats->q4_pool_fallbacks = model->q4_pool_fallbacks;
    stats->q4_multi_pool_submissions = model->q4_multi_pool_submissions;
    stats->q4_multi_pool_jobs = model->q4_multi_pool_jobs;
    stats->qkv_multi_pool_submissions = model->qkv_multi_pool_submissions;
    stats->q4_multi_pool_fallbacks = model->q4_multi_pool_fallbacks;
    stats->q8_pool_submissions = model->q8_pool_submissions;
    stats->q8_pool_fallbacks = model->q8_pool_fallbacks;
    stats->expert_pool_submissions = model->expert_pool_submissions;
    stats->expert_pool_fallbacks = model->expert_pool_fallbacks;
    stats->gpu_dense_submissions = model->gpu_dense_submissions;
    stats->gpu_router_submissions = model->gpu_router_submissions;
    stats->gpu_expert_gate_up_submissions =
        model->gpu_expert_gate_up_submissions;
    stats->gpu_expert_down_submissions = model->gpu_expert_down_submissions;
    stats->gpu_expert_gate_up_commands = model->gpu_expert_gate_up_commands;
    stats->gpu_expert_down_commands = model->gpu_expert_down_commands;
    stats->gpu_decode_expert_gate_up_commands =
        model->gpu_decode_expert_gate_up_commands;
    stats->gpu_decode_expert_down_commands =
        model->gpu_decode_expert_down_commands;
    stats->gpu_head_submissions = model->gpu_head_submissions;
    stats->gpu_decode_submissions = model->gpu_decode_submissions;
    stats->gpu_failures = model->gpu_failures;
    stats->gpu_shared_output_batches = model->gpu_shared_output_batches;
    stats->gpu_shared_output_jobs = model->gpu_shared_output_jobs;
    stats->gpu_shared_arena_bytes =
        (uint64_t)model->operation_shared.nbytes +
        (uint64_t)model->compute_scratch_shared.nbytes +
        (uint64_t)model->decode_shared.nbytes +
        (uint64_t)model->bf16_shared.nbytes;
    memset(&gpu_weight_usage, 0, sizeof gpu_weight_usage);
    if (model->gpu_initialized && salt_gpu_weight_resource_usage(
            &gpu_weight_usage, sizeof gpu_weight_usage) != 0)
        return -1;
    stats->gpu_weight_described_bytes = gpu_weight_usage.described_bytes;
    stats->gpu_weight_registered_bytes = gpu_weight_usage.registered_bytes;
    stats->gpu_weight_pageable_bytes = gpu_weight_usage.pageable_bytes;
    stats->gpu_weight_active_window_bytes =
        gpu_weight_usage.active_window_bytes;
    stats->gpu_weight_peak_window_bytes = gpu_weight_usage.peak_window_bytes;
    stats->gpu_weight_copied_bytes =
        gpu_weight_usage.device_copied_weight_bytes;
    stats->gpu_weight_addressability =
        (uint32_t)model->gpu_weight_addressability;
    stats->gpu_weight_described_resources =
        gpu_weight_usage.described_resources;
    stats->gpu_weight_active_resources = gpu_weight_usage.active_resources;
    stats->gpu_weight_active_windows = gpu_weight_usage.active_windows;
    stats->token_program_cells = model->token_backend_last.cells_completed;
    stats->token_caller_submissions =
        model->token_backend_last.caller_submissions;
    stats->token_internal_barriers =
        model->token_backend_last.internal_dependency_barriers;
    stats->token_completion_fences =
        model->token_backend_last.completion_fences;
    stats->token_intermediate_publications =
        model->token_backend_last.intermediate_host_publications;
    stats->token_final_publications =
        model->token_backend_last.final_publications;
    stats->token_cpu_pool_sessions = model->compute_pool.agraph_sessions;
    stats->token_cpu_pool_phases = model->compute_pool.agraph_phases;
    stats->gpu_attention_batches = model->gpu_attention_batches;
    stats->gpu_attention_tasks = model->gpu_attention_tasks;
    stats->gpu_attention_kv_bytes = (uint64_t)model->kv_shared.nbytes;
    stats->prefill_ffn_arena_bytes = model->prefill_ffn_arena_bytes;
    stats->prefill_ffn_arena_calls = model->prefill_ffn_arena_calls;
    stats->q4_pool_workers = model->compute_pool_ready
        ? model->compute_pool.aq4_threads : 0;
    stats->expert_inflight_slots = model->expert_inflight_slots;
    stats->expert_peak_inflight_slots = model->expert_peak_inflight_slots;
    if (dense_payload_bytes(model, &stats->dense_payload_bytes) != 0)
        return -1;
    stats->dense_retained_immutable = 1;
    return 0;
}
