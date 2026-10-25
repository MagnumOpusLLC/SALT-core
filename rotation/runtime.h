#ifndef SALT_ROTATION_RUNTIME_H
#define SALT_ROTATION_RUNTIME_H
#include <stdint.h>
#define SALT_LR_TIMELINE_PHASES 1024u
/* Maximum init-planned rotating expert banks. */
#define SALT_LR_MAX_BANKS 6
/* Process startup configuration; no environment or file reads in execution. */
int salt_lr_configure(unsigned prefill_layer_block);
int salt_lr_configure_file(const char *path);
unsigned salt_lr_prefill_layers(void);
int salt_lr_bank_count(void);
unsigned salt_lr_decode_attention_product_tile(void);
unsigned salt_lr_decode_read_overlap(void);
unsigned salt_lr_decode_mixed(void);
unsigned salt_lr_decode_dense_chain(void);
unsigned salt_lr_manifest_emit(void);
unsigned salt_lr_mtp_enabled(void);
unsigned salt_lr_mtp_draft_n(void);
const char *salt_lr_mtp_source(void);
const char *salt_lr_mtp_receipt(void);
unsigned salt_lr_native_target_warm_all(void);
unsigned salt_lr_native_target_warmup_rows(void);
unsigned salt_lr_native_target_x(void);
#define SALT_LR_BANK_COUNT (salt_lr_bank_count())

/* Process counters are observations, never control inputs. Bits: read=1,
 * write=2, faults=4. Unavailable counters remain explicitly invalid. */
typedef struct {
    uint64_t sample_begin_ns, sample_end_ns;
    uint64_t read_bytes, write_bytes, major_faults;
    uint32_t valid;
} SaltRotationIoSnapshot;
void salt_lr_io_snapshot(SaltRotationIoSnapshot *out);

/* Backend realization of the engine's layer-DONE and layer-READY signals.
 * wait_done returns 1 when signaled, 0 on a bounded wait timeout, -1 on error.
 * No callback allocates storage or changes the schedule. */
typedef int (*SaltRotationWaitDone)(void *, uint64_t);
typedef void (*SaltRotationPublishReady)(void *, uint64_t);
int salt_lr_stream_begin(uint64_t base, SaltRotationWaitDone wait_done,
                         SaltRotationPublishReady publish_ready, void *context);
/* Actual GPU notifications enter the existing rotator state machine. The
 * values identify completed layers, never permission for another layer. */
void salt_lr_compute_done(uint64_t value);
void salt_lr_compute_done_tick(uint64_t value, uint64_t callback_enter_ns);
int salt_lr_wait_compute_done(uint64_t value);
/* Required carries the current phase epoch; an older phase cannot satisfy it. */
static inline int salt_lr_done_event_proves(uint64_t required, uint64_t observed) {
    return required != 0u && observed >= required;
}
void salt_lr_done_event_receive_tick(uint64_t required, uint64_t observed,
    uint64_t begin_ns, uint64_t end_ns, uint32_t direct);
int salt_lr_stream_finish(void);
void salt_lr_stream_cancel(void);
void salt_lr_shutdown(void);

/* Fixed in-memory observations, read only after stream_finish. Buckets are
 * unstreamed loads, first streamed walk, and later streamed walks. They are
 * deliberately not named PREFILL/DECODE: the operation supplies that meaning. */
typedef struct {
    uint32_t bucket, layer, copies, discard_failures;
    uint64_t bytes, done_wait_ns, copy_span_ns, worker_copy_sum_ns;
    uint64_t copy_join_ns, source_discard_ns, ready_callback_ns;
    uint64_t source_prepare_ns; /* sum of W worker wall intervals, not additive */
    uint64_t load_span_ns;      /* earliest preparation to final copied byte */
    uint64_t critical_start_ns, critical_prepare_ns, critical_copy_ns;
    uint32_t prepare_calls, prepare_failures;
} SaltRotationTiming;
int salt_lr_timing(uint32_t index, SaltRotationTiming *out);
const char *salt_lr_source_policy(void);
uint64_t salt_lr_last_ready_ns(void);

/* Manifest handoff sub-ledger. Host CLOCK_MONOTONIC timestamps only: no GPU
 * clock mixing. Fixed process-lifetime capacity, no operation-time allocation.
 * Phase 0 is initialization's layer-0 prime; subsequent phases carry actual
 * command rows and monotonically increasing event bases. */
typedef struct {
    uint64_t event_base, submit_ns, complete_ns;
    uint32_t rows, kind; /* 0 startup, 1 full-layer phase, 2 resident DECODE. */
    uint32_t block_layers, ready_waits, done_signals;
} SaltRotationTimelinePhase;
typedef struct {
    uint64_t expected_done, expected_ready;
    uint64_t wait_begin_ns, done_seen_ns;
    uint64_t first_prepare_ns, critical_prepare_ns, critical_copy_ns;
    uint64_t first_copy_ns, last_copy_ns, joined_ns;
    uint64_t cleanup_begin_ns, cleanup_end_ns;
    uint64_t ready_begin_ns, ready_end_ns;
    uint64_t resident_at_ready_bytes, peak_at_ready_bytes;
    uint64_t copy_bytes, io_begin_ns, io_end_ns;
    uint64_t io_read_bytes, io_write_bytes, io_major_faults, io_sample_ns;
    uint32_t io_valid;
    uint32_t phase, layer, bank, critical_worker, copies, primed;
    int32_t done_layer; /* Actual GPU producer; -1 means an initially free bank. */
    uint32_t wait_worker;
} SaltRotationTimelineRow;
void salt_lr_timeline_open(uint64_t event_base, uint32_t kind);
void salt_lr_timeline_submit(uint64_t event_base, uint32_t rows, uint64_t host_ns);
void salt_lr_timeline_complete(uint64_t event_base, uint64_t host_ns);
void salt_lr_timeline_block(uint64_t event_base, uint32_t block_layers,
                             uint32_t ready_waits, uint32_t done_signals);
void salt_lr_timeline_memory(uint64_t event_base, uint32_t layer,
                             uint64_t resident_bytes, uint64_t peak_bytes);
uint32_t salt_lr_timeline_phase_count(void);
uint32_t salt_lr_timeline_layer_count(void);
uint32_t salt_lr_timeline_dropped(void);
int salt_lr_timeline_phase(uint32_t phase, SaltRotationTimelinePhase *out);
int salt_lr_timeline_row(uint32_t phase, uint32_t layer, SaltRotationTimelineRow *out);
/* Bounded observation seats; initialized once, never consulted by scheduling. */
#define SALT_LR_TICK_CAPACITY 8192u
typedef struct {
    uint64_t callback_enter_ns, callback_lock_ns;
    uint64_t done_notify_begin_ns, done_notify_end_ns;
    uint64_t permit_begin_ns, permit_end_ns;
    uint64_t receive_required, receive_observed, event_read_begin_ns, event_read_end_ns;
    uint32_t receive_direct;
} SaltRotationSignalTick;
typedef struct {
    uint64_t task_begin_ns, lock_acquired_ns, wait_begin_ns, wait_end_ns;
    uint64_t gate_pass_ns, read_begin_ns, read_end_ns, task_end_ns, read_bytes;
    uint32_t waits, read_calls, read_interrupts, short_reads, finalizer;
    int32_t read_status;
    uint64_t read_cpu_begin_ns, read_cpu_end_ns;
    uint32_t read_cpu_valid; /* begin=1, end=2; unavailable is not zero CPU work. */
} SaltRotationWorkerTick;
uint32_t salt_lr_worker_width(void);
uint64_t salt_lr_tick_storage_bytes(void);
uint64_t salt_lr_worker_ticks_dropped(void);
uint64_t salt_lr_signal_ticks_dropped(void);
int salt_lr_signal_tick(uint32_t phase, uint32_t layer, SaltRotationSignalTick *out);
int salt_lr_worker_tick(uint32_t phase, uint32_t layer, uint32_t worker,
                         SaltRotationWorkerTick *out);
/* Decode-only resident seats share the existing arena and W loader threads. */
int salt_rotation_decode_phase(void);
int salt_rotation_phase_exchange(int phase);
int salt_rotation_target_selected_exchange(int selected);
int salt_rotation_target_selected(void);
void salt_lr_decode_invalidate(void);
int salt_lr_decode_begin(void);
int salt_lr_decode_active(void);
int salt_lr_decode_global_exchange(int enabled);
int salt_lr_decode_global_active(void);
int salt_lr_decode_zero_copy_active(void);
const uint8_t *salt_lr_decode_payload(int layer, int expert);
unsigned salt_lr_decode_seats(void);
unsigned salt_lr_decode_layer_seats(unsigned layer);
unsigned salt_lr_decode_min_layer_seats(void);
unsigned salt_lr_decode_max_layer_seats(void);
unsigned salt_lr_decode_capacity(uint64_t slot_bytes, unsigned logical_experts);
unsigned salt_lr_layer_experts(void);
int salt_lr_decode_find(int layer, int expert, int *slot);
int salt_lr_slot(int layer, int expert);
unsigned salt_lr_decode_bank_origin(unsigned layer);
int salt_lr_decode_fetch(int layer, const int *requested, int count, int *out);
typedef struct {
    uint32_t step, layer, requests, hits, misses, evictions, resident;
    uint64_t bytes;
} SaltDecodeExpertRecord;
uint32_t salt_lr_decode_record_count(void);
uint32_t salt_lr_decode_record_dropped(void);
int salt_lr_decode_record(uint32_t index, SaltDecodeExpertRecord *out);
void salt_lr_decode_stats(uint64_t *requests, uint64_t *hits,
                          uint64_t *misses, uint64_t *bytes);
uint32_t salt_lr_decode_replay_plan_count(void);
uint32_t salt_lr_decode_replay_plan_dropped(void);
int salt_lr_decode_replay_plan(uint32_t index, uint32_t *step,
    uint32_t *layer, uint32_t *ordinal, uint32_t *expert, uint32_t *seat,
    uint64_t *file_offset, uint64_t *bytes);
typedef struct {
    uint64_t begin_ns, end_ns, file_offset, requested_bytes;
    int64_t returned_bytes;
    uint32_t phase, step, layer, ordinal, worker, attempt, expert, seat;
    int32_t error;
} SaltDecodeReadCallRecord;
uint32_t salt_lr_decode_read_call_capacity(void);
uint32_t salt_lr_decode_read_call_dropped(void);
int salt_lr_decode_read_call(uint32_t index, SaltDecodeReadCallRecord *out);
int salt_lr_decode_seat(unsigned layer, unsigned rank, int *expert, int *slot);
int salt_lr_decode_layer_entry(unsigned layer, unsigned rank,
    int *expert, int *slot);
#endif
