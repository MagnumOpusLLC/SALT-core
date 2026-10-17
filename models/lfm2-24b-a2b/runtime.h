/* LFM2.5-24B-A2B runtime.
 *
 * Each model owns its own runtime and its own server entry: maple-preview keeps
 * both in server.c, qwen38-27b splits them into runtime.c/runtime.h. LFM follows
 * the split, because its admission gate (server.c) verifies the container and
 * has no business holding a live engine.
 *
 * The engine supplies the seam -- model_layer_ops on the execution descriptor,
 * and SALT_ATTN_MODEL as a layer kind -- and this fills it. Nothing here goes
 * into src/ or include/salt/.
 *
 * Model-owned. */

#ifndef SALT_LFM_RUNTIME_H
#define SALT_LFM_RUNTIME_H

#include <stdint.h>

typedef struct LfmRuntime LfmRuntime;

typedef struct LfmRuntimeStateDigests {
    char state_sha256[65];
    char kv_sha256[65];
    char model_state_sha256[65];
    char logits_sha256[65];
    uint64_t kv_bytes;
    uint64_t model_state_bytes;
    uint64_t transition_generation;
    uint32_t position;
} LfmRuntimeStateDigests;

typedef struct LfmRuntimeMetrics {
    uint64_t request_ns;
    uint64_t prefill_ns;
    uint64_t decode_ns;
    uint64_t backend_graph_launches;
    uint64_t backend_physical_kernel_nodes;
    uint64_t backend_host_kernel_launch_calls;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
} LfmRuntimeMetrics;

/* Open the container, build the execution descriptor, compile the program and
 * the CPU executor, and bind the scheduler. All allocation happens here, once;
 * nothing is allocated on the generation path.
 *
 * Returns 0, or -1 with a named reason on stderr. */
int lfm_runtime_open(LfmRuntime **out, int trunk_fd, int pool_fd, int index_fd,
                     int tokenizer_fd, uint32_t context, uint32_t batch,
                     uint32_t workers, int slots, uint64_t rss_limit);

/* One token at a time over the live state. The prompt is consumed by the first
 * call; later calls continue from the committed position. Returns the number of
 * tokens written to out_ids, or -1. */
int lfm_runtime_generate(LfmRuntime *m, const int32_t *prompt_ids,
                         uint32_t prompt_count, uint32_t max_output,
                         int32_t *out_ids);

/* The runtime owns the authenticated tokenizer loaded from TOKENIZER_FD. */
int lfm_runtime_encode(LfmRuntime *m, const char *text, int32_t *ids,
                       uint32_t capacity);
int lfm_runtime_decode(LfmRuntime *m, const int32_t *ids, uint32_t count,
                       char *text, uint32_t capacity);
int lfm_runtime_last_metrics(const LfmRuntime *m, LfmRuntimeMetrics *metrics);
int lfm_runtime_state_digests(const LfmRuntime *m,
                              LfmRuntimeStateDigests *digests);

void lfm_runtime_close(LfmRuntime *m);

#endif /* SALT_LFM_RUNTIME_H */
