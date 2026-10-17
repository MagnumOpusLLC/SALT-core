#ifndef SALT_LFM_ATTENTION_H
#define SALT_LFM_ATTENTION_H

#include "salt/text_verify.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmAttentionIoLedger {
    uint32_t maximum_context;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t heads_per_kv;
    uint32_t rows_per_wave;
    uint32_t score_threads;
    uint32_t decode_score_workers;
    uint32_t value_column_tile;
    uint32_t score_task_capacity;
    uint32_t ready;
} LfmAttentionIoLedger;

int lfm_attention_io_ledger_prepare(LfmAttentionIoLedger *ledger,
                                    uint32_t maximum_context);
int lfm_attention_cuda_launch(const LfmAttentionIoLedger *ledger,
                              const SaltTextModelGpuAttentionArgs *args);

#ifdef __cplusplus
}
#endif

#endif /* SALT_LFM_ATTENTION_H */
