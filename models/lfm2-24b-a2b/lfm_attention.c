#include "lfm_attention.h"
#include "lfm.h"

#include <string.h>

int lfm_attention_io_ledger_prepare(LfmAttentionIoLedger *ledger,
                                    uint32_t maximum_context) {
    if (!ledger || maximum_context == 0u)
        return -1;
    memset(ledger, 0, sizeof *ledger);
    ledger->maximum_context = maximum_context;
    ledger->query_heads = LFM_HEADS;
    ledger->kv_heads = LFM_KV_HEADS;
    ledger->head_dim = LFM_HEAD_DIM;
    ledger->heads_per_kv = LFM_HEADS / LFM_KV_HEADS;
    ledger->rows_per_wave = 8u;
    ledger->score_threads = 256u;
    ledger->decode_score_workers = 2u;
    ledger->value_column_tile = 64u;
    ledger->score_task_capacity = 8u * LFM_HEADS;
    ledger->ready = 1u;
    return 0;
}
