#ifndef SALT_ATTN_BATCH_H
#define SALT_ATTN_BATCH_H

#include "salt/model.h"

struct SaltKvCache;

/* Package-independent M7 attention-body descriptor. Q/K/V rows are already
 * projected, normalized, and rotated. Workers own disjoint (token, head)
 * outputs; each dot/softmax/value reduction retains serial order. */
typedef struct {
    SaltAttentionDesc attention;
    const float *queries;       /* [batch][query_stride] */
    const float *private_keys;  /* absolute [key_capacity][kv_stride] */
    const float *private_values;
    const float *shared_keys;   /* optional absolute prefix */
    const float *shared_values;
    int shared_tokens;
    int start_position;
    int batch;
    int query_stride;
    int kv_stride;
    float *outputs;             /* [batch][query_stride] */
    struct SaltKvCache *pool;   /* optional existing persistent worker pool */
} SaltAttentionBatchJob;

int salt_attention_batch_run(const SaltAttentionBatchJob *job, int threads);

#endif
