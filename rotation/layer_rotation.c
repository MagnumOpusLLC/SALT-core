/* Engine-owned two-range rotation. Fixed addresses and W persistent workers.
 * DONE authorizes overwrite of the other range; READY follows all W copies.
 * No worker allocates storage or submits device work. */
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#include "salt/rotation/layer_rotation.h"
#include "salt/rotation/fail.h"
#include "salt/salt.h"
#include "salt/gpu.h"
#include "salt/trace.h"
#include "runtime.h"
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <errno.h>
#include <stdio.h>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <libproc.h>
#endif

static const char *fail_file = "(none)";
static int fail_line, fail_status;
static pthread_mutex_t fail_mu = PTHREAD_MUTEX_INITIALIZER;
void salt_rot_fail_at(const char *file, int line, int status) {
    pthread_mutex_lock(&fail_mu);
    if (!fail_line) { fail_file = file; fail_line = line; fail_status = status; }
    pthread_mutex_unlock(&fail_mu);
}
const char *salt_rot_fail_file(void) { return fail_file; }
int salt_rot_fail_line(void) { return fail_line; }
int salt_rot_fail_status(void) { return fail_status; }

typedef struct {
    uint64_t file_offset;
    uint8_t *dst;
    size_t bytes;
} RotationSlice;
static RotationSlice slices[SALT_LR_MAX_LAYERS][SALT_LR_MAX_THREADS];
typedef struct {
    int done_layer; /* Init-compiled previous owner of this destination bank. */
} RotationSource;
static RotationSource source_layers[SALT_LR_MAX_LAYERS];
static int source_fd;
static uint8_t *arena;
static uint64_t slot_bytes, layer_bytes;
static int layers, experts, width, initialized, primed, arena_slot_count;
static unsigned prefill_block_layers = 1;
static unsigned decode_seats;
static unsigned decode_conservation_gb;
static unsigned decode_attention_product_tile;
static unsigned decode_read_overlap;
static unsigned decode_read_advice;
static unsigned decode_read_mmap;
static unsigned decode_zero_copy;
static unsigned decode_mixed;
static unsigned decode_cpu;
static unsigned decode_dense_chain;
static unsigned manifest_emit = 1u;
static unsigned mtp_enabled, mtp_draft_n;
static unsigned native_target_warm_all, native_target_warmup_rows, native_target_x;
static char mtp_source[PATH_MAX], mtp_receipt[PATH_MAX];
static size_t source_page_bytes;
static uint64_t decode_map_calls[SALT_LR_MAX_THREADS];
static uint64_t decode_map_errors[SALT_LR_MAX_THREADS];
static uint64_t decode_zero_maps, decode_zero_unmaps, decode_zero_advice_errors;
static uint64_t decode_advice_calls[SALT_LR_MAX_THREADS];
static uint64_t decode_advice_errors[SALT_LR_MAX_THREADS];
unsigned salt_lr_decode_read_overlap(void) { return decode_read_overlap; }
unsigned salt_lr_decode_mixed(void) { return decode_mixed; }
unsigned salt_lr_cpu_mode(void) { return decode_cpu; }
unsigned salt_lr_decode_dense_chain(void) { return decode_dense_chain; }
unsigned salt_lr_manifest_emit(void) { return manifest_emit; }
unsigned salt_lr_mtp_enabled(void) { return mtp_enabled; }
unsigned salt_lr_mtp_draft_n(void) { return mtp_draft_n; }
const char *salt_lr_mtp_source(void) { return mtp_source; }
const char *salt_lr_mtp_receipt(void) { return mtp_receipt; }
unsigned salt_lr_native_target_warm_all(void) { return native_target_warm_all; }
unsigned salt_lr_native_target_warmup_rows(void) { return native_target_warmup_rows; }
unsigned salt_lr_native_target_x(void) { return native_target_x; }
unsigned salt_lr_decode_capacity(uint64_t bytes_per_slot,
        unsigned logical_experts) {
    uint64_t capacity;
    if (!bytes_per_slot || !logical_experts) return 0u;
    if (!decode_conservation_gb)
        return decode_seats <= logical_experts ? decode_seats : logical_experts;
    capacity = (uint64_t)decode_conservation_gb * 1000000000u /
        bytes_per_slot;
    if (capacity > logical_experts) capacity = logical_experts;
    return capacity <= UINT_MAX ? (unsigned)capacity : 0u;
}
unsigned salt_lr_layer_experts(void) {
    return experts > 0 ? (unsigned)experts : 0u;
}
unsigned salt_lr_decode_attention_product_tile(void) { return decode_attention_product_tile; }
unsigned salt_lr_prefill_layers(void) { return prefill_block_layers; }
int salt_lr_bank_count(void) { return prefill_block_layers == 2u ? 6 : 3; }
int salt_lr_configure(unsigned block) {
    if (initialized || (block != 1u && block != 2u)) return SALT_TFAIL(-1);
    prefill_block_layers = block;
    return 0;
}
int salt_lr_configure_file(const char *path) {
    char line[160];
    unsigned block = 1, seats = 0, conservation_gb = 0, products = 0, overlap = 0, advice = 0, mapped = 0, zero_copy = 0, mixed = 0, cpu = 0, dense_chain = 0, emit = 1, mtp = 0, draft_n = 0, warm_all = 0, warmup_rows = 0, target_x = 0, found = 0;
    FILE *f;
    if (initialized || !path || !(f = fopen(path, "r"))) return SALT_TFAIL(-1);
    while (fgets(line, sizeof line, f)) {
        char *p = line, tail;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        if (sscanf(p, "prefill_layer_block=%u %c", &block, &tail) == 1 &&
                !(found & 1u) && (block == 1u || block == 2u)) found |= 1u;
        else if (sscanf(p, "decode_expert_seats=%u %c", &seats, &tail) == 1 &&
                !(found & 2u) && seats >= 1u &&
                seats <= SALT_LR_MAX_LAYERS * SALT_LR_MAX_EXPERTS) found |= 2u;
        else if (sscanf(p, "decode_conservation_gb=%u %c", &conservation_gb, &tail) == 1 &&
                !(found & 128u) && conservation_gb >= 1u &&
                conservation_gb <= 128u) found |= 128u;
        else if (sscanf(p, "decode_attention_product_tile=%u %c", &products, &tail) == 1 &&
                !(found & 4u) && products <= 1u) found |= 4u;
        else if (sscanf(p, "decode_read_overlap=%u %c", &overlap, &tail) == 1 &&
                !(found & 8u) && overlap <= 1u) found |= 8u;
        else if (sscanf(p, "decode_read_advice=%u %c", &advice, &tail) == 1 &&
                !(found & 16u) && advice <= 1u) found |= 16u;
        else if (sscanf(p, "decode_read_mmap=%u %c", &mapped, &tail) == 1 &&
                !(found & 32u) && mapped <= 1u) found |= 32u;
        else if (sscanf(p, "decode_zero_copy=%u %c", &zero_copy, &tail) == 1 &&
                !(found & 64u) && zero_copy <= 1u) found |= 64u;
        else if ((!strcmp(p, "decode_mode=mixed\n") ||
                  !strcmp(p, "decode_mode=mixed") ||
                  !strcmp(p, "decode_mode=full-gpu\n") ||
                  !strcmp(p, "decode_mode=full-gpu") ||
                  !strcmp(p, "decode_mode=cpu\n") ||
                  !strcmp(p, "decode_mode=cpu")) && !(found & 256u)) {
            mixed = !strncmp(p, "decode_mode=mixed", 17);
            cpu = !strncmp(p, "decode_mode=cpu", 15);
            found |= 256u;
        }
        else if (sscanf(p, "decode_dense_chain=%u %c", &dense_chain, &tail) == 1 &&
                !(found & 65536u) && dense_chain <= 1u) found |= 65536u;
        else if (sscanf(p, "manifest_emit=%u %c", &emit, &tail) == 1 &&
                !(found & 131072u) && emit <= 1u) found |= 131072u;
        else if (sscanf(p, "mtp=%u %c", &mtp, &tail) == 1 &&
                !(found & 512u) && mtp <= 1u) found |= 512u;
        else if (sscanf(p, "mtp_draft_n=%u %c", &draft_n, &tail) == 1 &&
                !(found & 1024u) && draft_n >= 2u && draft_n <= 8u)
            found |= 1024u;
        else if (!strncmp(p, "mtp_source=", 11) && !(found & 2048u)) {
            char *value = p + 11, *end = strchr(value, '\n');
            if (end) *end = 0;
            if (!*value || strlen(value) >= sizeof mtp_source) {
                fclose(f); return SALT_TFAIL(-1);
            }
            memcpy(mtp_source, value, strlen(value) + 1u);
            found |= 2048u;
        }
        else if (!strncmp(p, "mtp_receipt=", 12) && !(found & 4096u)) {
            char *value = p + 12, *end = strchr(value, '\n');
            if (end) *end = 0;
            if (!*value || strlen(value) >= sizeof mtp_receipt) {
                fclose(f); return SALT_TFAIL(-1);
            }
            memcpy(mtp_receipt, value, strlen(value) + 1u);
            found |= 4096u;
        }
        else if (sscanf(p, "native_target_warm_all=%u %c", &warm_all, &tail) == 1 &&
                !(found & 8192u) && warm_all <= 1u) found |= 8192u;
        else if (sscanf(p, "native_target_warmup_rows=%u %c", &warmup_rows, &tail) == 1 &&
                !(found & 16384u) && warmup_rows >= 1u) found |= 16384u;
        else if (sscanf(p, "native_target_x=%u %c", &target_x, &tail) == 1 &&
                !(found & 32768u) && target_x >= 1u && target_x <= 356u)
            found |= 32768u;
        else { fclose(f); return SALT_TFAIL(-1); }
    }
    int bad = ferror(f);
    fclose(f);
    if (bad || !(found & 1u) || !(found & 256u) ||
        !(found & 512u) || !(found & 1024u) ||
        !(found & 2048u) || !(found & 4096u) ||
        !(found & 8192u) || !(found & 16384u) ||
        !(found & 32768u) ||
        ((found & 2u) && (found & 128u)) ||
        (!(found & 2u) && !(found & 128u)) ||
        (!mtp && draft_n != 2u) ||
        salt_lr_configure(block) != 0)
        return SALT_TFAIL(-1);
    decode_seats = seats;
    decode_conservation_gb = conservation_gb;
    decode_attention_product_tile = products;
    decode_read_overlap = overlap;
    decode_read_advice = advice;
    decode_read_mmap = mapped;
    decode_zero_copy = zero_copy;
    decode_mixed = mixed;
    decode_cpu = cpu;
    decode_dense_chain = dense_chain;
    manifest_emit = emit;
    mtp_enabled = mtp;
    mtp_draft_n = draft_n;
    native_target_warm_all = warm_all;
    native_target_warmup_rows = warmup_rows;
    native_target_x = target_x;
    return 0;
}
static int range_layer[SALT_LR_MAX_BANKS];
static uint32_t inits, loads, announced, ready, done, bind_noarg, bind_refused;
static pthread_t workers[SALT_LR_MAX_THREADS];
static int worker_ids[SALT_LR_MAX_THREADS], created;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t work_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;
static uint64_t sequence, event_base;
static uint64_t compute_done_values[SALT_LR_MAX_BANKS];
static uint64_t ready_delivered_values[SALT_LR_MAX_BANKS];
static uint64_t last_ready_ns; /* observed after stream_finish; not control state */
uint64_t salt_lr_last_ready_ns(void) { return last_ready_ns; }
static int stopping, busy, cancelled, failed, allowed_layer;
static int first_layer, end_layer, finished_workers;
static int waiting_layer, next_publish_layer;
typedef struct {
    uint64_t wait_begin_ns, done_seen_ns;
    SaltRotationIoSnapshot io_begin;
    uint32_t worker;
} RotationLoadGate;
static RotationLoadGate load_gates[SALT_LR_MAX_LAYERS];
static unsigned copied[SALT_LR_MAX_LAYERS];
static uint64_t load_begin_ns[SALT_LR_MAX_LAYERS][SALT_LR_MAX_THREADS];
static unsigned prepare_attempts[SALT_LR_MAX_LAYERS][SALT_LR_MAX_THREADS];
static unsigned prepare_errors[SALT_LR_MAX_LAYERS][SALT_LR_MAX_THREADS];
static uint64_t copy_begin_ns[SALT_LR_MAX_LAYERS][SALT_LR_MAX_THREADS];
static uint64_t copy_end_ns[SALT_LR_MAX_LAYERS][SALT_LR_MAX_THREADS];
static SaltRotationTiming timings[3][SALT_LR_MAX_LAYERS];
static unsigned stream_walks, timing_bucket;
static SaltRotationTimelinePhase timeline_phases[SALT_LR_TIMELINE_PHASES];
static SaltRotationTimelineRow timeline_rows[SALT_LR_TIMELINE_PHASES][SALT_LR_MAX_LAYERS];
static uint32_t timeline_count, timeline_current, timeline_dropped;
static SaltRotationSignalTick signal_ticks[SALT_LR_TICK_CAPACITY];
static SaltRotationWorkerTick worker_ticks[SALT_LR_TICK_CAPACITY];
static uint64_t worker_ticks_dropped[SALT_LR_MAX_THREADS], signal_ticks_dropped, receive_ticks_dropped;
static SaltRotationSignalTick *signal_tick_phase(uint32_t phase, int layer) {
    uint64_t index = (uint64_t)phase * (uint32_t)layers + (uint32_t)layer;
    if (index >= SALT_LR_TICK_CAPACITY) { signal_ticks_dropped++; return NULL; }
    return &signal_ticks[index];
}
static SaltRotationSignalTick *signal_tick_current(int layer) {
    return signal_tick_phase(timeline_current, layer);
}
static SaltRotationWorkerTick *worker_tick_current(int layer, int worker) {
    uint64_t index = ((uint64_t)timeline_current * (uint32_t)layers +
        (uint32_t)layer) * (uint32_t)width + (uint32_t)worker;
    if (index >= SALT_LR_TICK_CAPACITY) { worker_ticks_dropped[worker]++; return NULL; }
    return &worker_ticks[index];
}
uint32_t salt_lr_worker_width(void) { return (uint32_t)width; }
uint64_t salt_lr_tick_storage_bytes(void) { return sizeof signal_ticks + sizeof worker_ticks; }
uint64_t salt_lr_worker_ticks_dropped(void) {
    uint64_t n = 0;
    for (int w = 0; w < width; w++) n += worker_ticks_dropped[w];
    return n;
}
uint64_t salt_lr_signal_ticks_dropped(void) { return signal_ticks_dropped + receive_ticks_dropped; }
void salt_lr_done_event_receive_tick(uint64_t required, uint64_t observed,
        uint64_t begin_ns, uint64_t end_ns, uint32_t direct) {
    if (!initialized || required <= event_base ||
        required - event_base > (uint64_t)layers) return;
    uint64_t layer = required - event_base - 1u + (uint32_t)SALT_LR_BANK_COUNT;
    uint64_t index = (uint64_t)timeline_current * (uint32_t)layers + layer;
    if (layer >= (uint32_t)layers) return;
    if (index >= SALT_LR_TICK_CAPACITY) { receive_ticks_dropped++; return; }
    /* One existing bank-gate owner writes these observation-only fields.
     * No callback clock/state is fabricated, and no new lock is taken. */
    SaltRotationSignalTick *tick = &signal_ticks[index];
    tick->receive_required = required; tick->receive_observed = observed;
    tick->event_read_begin_ns = begin_ns; tick->event_read_end_ns = end_ns;
    tick->receive_direct = direct;
}
int salt_lr_signal_tick(uint32_t phase, uint32_t layer, SaltRotationSignalTick *out) {
    uint64_t index = (uint64_t)phase * (uint32_t)layers + layer;
    if (!out || phase >= timeline_count || layer >= (uint32_t)layers ||
        index >= SALT_LR_TICK_CAPACITY) return 0;
    /* Observation at cleanup, never a new operation-path lock. */
    pthread_mutex_lock(&mu);
    *out = signal_ticks[index];
    pthread_mutex_unlock(&mu);
    return out->callback_enter_ns || out->permit_begin_ns || out->event_read_begin_ns;
}
int salt_lr_worker_tick(uint32_t phase, uint32_t layer, uint32_t worker,
                         SaltRotationWorkerTick *out) {
    uint64_t index = ((uint64_t)phase * (uint32_t)layers + layer) * (uint32_t)width + worker;
    if (!out || phase >= timeline_count || layer >= (uint32_t)layers ||
        worker >= (uint32_t)width || index >= SALT_LR_TICK_CAPACITY) return 0;
    *out = worker_ticks[index];
    return out->task_begin_ns != 0;
}
static SaltRotationTimelineRow *timeline_current_row(int layer) {
    if (timeline_current >= SALT_LR_TIMELINE_PHASES ||
        layer < 0 || layer >= layers) return NULL;
    return &timeline_rows[timeline_current][layer];
}
void salt_lr_timeline_open(uint64_t base, uint32_t kind) {
    uint64_t index = kind ? base / ((uint32_t)layers + 2u) + 1u : 0u;
    timeline_current = index < SALT_LR_TIMELINE_PHASES ? (uint32_t)index : SALT_LR_TIMELINE_PHASES;
    if (timeline_current >= SALT_LR_TIMELINE_PHASES) { timeline_dropped++; return; }
    if (timeline_count <= timeline_current) timeline_count = timeline_current + 1u;
    timeline_phases[timeline_current] = (SaltRotationTimelinePhase){.event_base = base, .kind = kind};
}
void salt_lr_timeline_submit(uint64_t base, uint32_t rows, uint64_t host_ns) {
    if (timeline_current < timeline_count &&
        timeline_phases[timeline_current].event_base == base) {
        timeline_phases[timeline_current].rows = rows;
        if (!timeline_phases[timeline_current].submit_ns)
            timeline_phases[timeline_current].submit_ns = host_ns;
    }
}
void salt_lr_timeline_complete(uint64_t base, uint64_t host_ns) {
    if (timeline_current < timeline_count &&
        timeline_phases[timeline_current].event_base == base)
        timeline_phases[timeline_current].complete_ns = host_ns;
}
void salt_lr_timeline_memory(uint64_t base, uint32_t layer,
                             uint64_t resident_bytes, uint64_t peak_bytes) {
    if (timeline_current >= timeline_count ||
        timeline_phases[timeline_current].event_base != base ||
        layer >= (uint32_t)layers) return;
    SaltRotationTimelineRow *row = &timeline_rows[timeline_current][layer];
    row->resident_at_ready_bytes = resident_bytes;
    row->peak_at_ready_bytes = peak_bytes;
}
void salt_lr_timeline_block(uint64_t base, uint32_t block,
                             uint32_t waits, uint32_t signals) {
    if (timeline_current < timeline_count &&
        timeline_phases[timeline_current].event_base == base) {
        timeline_phases[timeline_current].block_layers = block;
        timeline_phases[timeline_current].ready_waits = waits;
        timeline_phases[timeline_current].done_signals = signals;
    }
}
uint32_t salt_lr_timeline_phase_count(void) { return timeline_count; }
uint32_t salt_lr_timeline_layer_count(void) { return layers > 0 ? (uint32_t)layers : 0u; }
uint32_t salt_lr_timeline_dropped(void) { return timeline_dropped; }
int salt_lr_timeline_phase(uint32_t phase, SaltRotationTimelinePhase *out) {
    if (!out || phase >= timeline_count) return 0;
    *out = timeline_phases[phase];
    return 1;
}
int salt_lr_timeline_row(uint32_t phase, uint32_t layer,
                         SaltRotationTimelineRow *out) {
    if (!out || phase >= timeline_count || layer >= (uint32_t)layers ||
        (!timeline_rows[phase][layer].copies && !timeline_rows[phase][layer].ready_end_ns)) return 0;
    *out = timeline_rows[phase][layer];
    return 1;
}
static uint64_t rotation_now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return 0;
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static uint32_t rotation_thread_cpu_ns(uint64_t *out) {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec t;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0) {
        *out = (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
        return 1u;
    }
#endif
    *out = 0;
    return 0u;
}
static uint64_t rotation_elapsed(uint64_t start, uint64_t end) {
    return start && end >= start ? end - start : 0;
}
void salt_lr_io_snapshot(SaltRotationIoSnapshot *out) {
    struct rusage usage;
    if (!out) return;
    memset(out, 0, sizeof *out);
    out->sample_begin_ns = rotation_now_ns();
    memset(&usage, 0, sizeof usage);
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_majflt >= 0) {
        out->major_faults = (uint64_t)usage.ru_majflt;
        out->valid |= 4u;
    }
#if defined(__APPLE__)
    struct rusage_info_v2 info;
    memset(&info, 0, sizeof info);
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
            (rusage_info_t *)&info) == 0) {
        out->read_bytes = info.ri_diskio_bytesread;
        out->write_bytes = info.ri_diskio_byteswritten;
        out->valid |= 3u;
    }
#endif
    out->sample_end_ns = rotation_now_ns();
}
const char *salt_lr_source_policy(void) {
    static char policy[256];
    uint64_t calls = 0u, errors = 0u, maps = 0u, map_errors = 0u;
    for (int w = 0; w < width; w++) {
        calls += decode_advice_calls[w];
        errors += decode_advice_errors[w];
        maps += decode_map_calls[w];
        map_errors += decode_map_errors[w];
    }
    if (decode_zero_copy) {
        snprintf(policy, sizeof policy,
                 "prefill-buffered-pread/DECODE-layer-local-mmap-zero-copy/F_RDADVISE/no-touch "
                 "total_slots=%u min_layer_slots=%u max_layer_slots=%u maps=%llu unmaps=%llu advice_errors=%llu",
                 decode_seats, salt_lr_decode_min_layer_seats(),
                 salt_lr_decode_max_layer_seats(),
                 (unsigned long long)decode_zero_maps,
                 (unsigned long long)decode_zero_unmaps,
                 (unsigned long long)decode_zero_advice_errors);
        return policy;
    }
    if (decode_read_mmap) {
        snprintf(policy, sizeof policy,
                 "prefill-buffered-pread/DECODE-bounded-selected-view/unchanged-canonical-source-map "
                 "decode_read_mmap=1 mmap_calls=%llu mmap_errors=%llu "
                 "decode_read_advice=%u advice_calls=%llu advice_errors=%llu",
                 (unsigned long long)maps, (unsigned long long)map_errors,
                 decode_read_advice, (unsigned long long)calls, (unsigned long long)errors);
        return policy;
    }
    snprintf(policy, sizeof policy,
             "buffered-bulk-pread/W-parallel/no-source-map-touch/no-source-map-mutation "
             "layer_local_total_slots=%u min_layer_slots=%u max_layer_slots=%u "
             "decode_read_advice=%u advice_calls=%llu advice_errors=%llu "
             "decode_read_mmap=0 mmap_calls=0 mmap_errors=0",
             decode_seats, salt_lr_decode_min_layer_seats(),
             salt_lr_decode_max_layer_seats(), decode_read_advice,
             (unsigned long long)calls, (unsigned long long)errors);
    return policy;
}
int salt_lr_timing(uint32_t index, SaltRotationTiming *out) {
    if (!out || layers < 1 || index >= 3u * (uint32_t)layers) return 0;
    *out = timings[index / (uint32_t)layers][index % (uint32_t)layers];
    return 1;
}
static SaltRotationWaitDone wait_done;
static SaltRotationPublishReady publish_ready;
static void *signal_context;

int salt_lr_slot(int layer, int expert) {
    if (!initialized || layer < 0 || layer >= layers || expert < 0 || expert >= experts)
        return SALT_TFAIL(-1);
    return (layer % SALT_LR_BANK_COUNT) * experts + expert;
}
const uint8_t *salt_lr_payload(int layer, int expert) {
    int slot = salt_lr_slot(layer, expert);
    return slot < 0 ? NULL : arena + (uint64_t)slot * slot_bytes;
}
const uint8_t *salt_lr_layer_range(int layer, uint64_t *bytes_out) {
    if (!initialized || layer < 0 || layer >= layers) return NULL;
    if (bytes_out) *bytes_out = layer_bytes;
    return arena + (uint64_t)(layer % SALT_LR_BANK_COUNT) * layer_bytes;
}
int salt_lr_arena(const uint8_t **out, uint64_t *stride, uint64_t *bytes, int *slots) {
    if (!initialized) return SALT_TFAIL(-1);
    if (out) *out = arena;
    if (stride) *stride = slot_bytes;
    if (bytes) *bytes = slot_bytes;
    if (slots) *slots = arena_slot_count;
    return 0;
}
int salt_lr_inited(void) { return initialized; }
int salt_lr_lookup(int layer, int expert) { return salt_lr_slot(layer, expert); }
int salt_lr_ready(int layer) {
    int result;
    pthread_mutex_lock(&mu);
    result = initialized && layer >= 0 && layer < layers && range_layer[layer % SALT_LR_BANK_COUNT] == layer;
    pthread_mutex_unlock(&mu);
    return result;
}
int salt_lr_state(void) {
    int result;
    pthread_mutex_lock(&mu);
    result = 0;
    if (initialized)
        for (int bank = 0; bank < SALT_LR_BANK_COUNT; bank++)
            result += range_layer[bank] >= 0;
    pthread_mutex_unlock(&mu);
    return result;
}
static void deliver_ready(int layer) {
    SaltRotationPublishReady fn = NULL;
    void *context = NULL;
    SaltRotationTimelineRow *handoff = NULL;
    SaltRotationTiming *timing = NULL;
    uint64_t value = 0, base = 0;
    int final_layer = 0;
    pthread_mutex_lock(&mu);
    if (!cancelled && !stopping && publish_ready && layer >= 0 && layer < layers &&
        range_layer[layer % SALT_LR_BANK_COUNT] == layer) {
        uint32_t bank = (uint32_t)(layer % SALT_LR_BANK_COUNT);
        value = event_base + (uint64_t)layer + 1u;
        /* LOAD_DONE records only landed bytes. Its independent timeline may
         * run ahead of GPU DONE without inventing a compute completion. */
        if (ready_delivered_values[bank] < value) {
            ready_delivered_values[bank] = value;
            fn = publish_ready; context = signal_context; base = event_base;
            handoff = timeline_current_row(layer);
            timing = &timings[timing_bucket][layer];
            final_layer = layer + 1 == end_layer;
        }
    }
    pthread_mutex_unlock(&mu);
    if (fn) {
        uint64_t begin = rotation_now_ns();
        fn(context, value);
        uint64_t end = rotation_now_ns();
        timing->ready_callback_ns += rotation_elapsed(begin, end);
        if (handoff) {
            handoff->ready_begin_ns = begin;
            handoff->ready_end_ns = end;
        }
        if (final_layer) {
            pthread_mutex_lock(&mu);
            if (event_base == base) last_ready_ns = end;
            pthread_mutex_unlock(&mu);
        }
    }
}
static int is_cancelled(void) {
    int result;
    pthread_mutex_lock(&mu);
    result = cancelled || stopping;
    pthread_mutex_unlock(&mu);
    return result;
}
static int read_fixed_slice(const RotationSlice *slice, SaltRotationWorkerTick *tick) {
    size_t landed = 0;
    while (landed < slice->bytes) {
        if (tick) tick->read_calls++;
        ssize_t n = pread(source_fd, slice->dst + landed,
            slice->bytes - landed, (off_t)(slice->file_offset + landed));
        if (n < 0 && errno == EINTR) {
            if (tick) tick->read_interrupts++;
            continue;
        }
        if (n <= 0) return SALT_TFAIL(-1);
        if (tick) {
            tick->short_reads += (size_t)n < slice->bytes - landed;
            tick->read_bytes += (uint64_t)n;
        }
        landed += (size_t)n;
    }
    return 0;
}
#include "decode_residency.inc"

static int read_decode_pread_slice(const RotationSlice *slice,
        SaltRotationWorkerTick *tick, int worker, int ordinal, int layer) {
    size_t landed = 0;
    uint32_t attempt = 0;
    while (landed < slice->bytes) {
        SaltDecodeReadCallRecord *record = NULL;
        if (decode_step == 1u) {
            if (attempt < DECODE_READ_CALL_ATTEMPTS) {
                uint32_t index = ((uint32_t)layer * SALT_LR_MAX_EXPERTS +
                    (uint32_t)ordinal) * DECODE_READ_CALL_ATTEMPTS + attempt;
                record = &decode_read_calls[index];
                *record = (SaltDecodeReadCallRecord) {
                    .file_offset = slice->file_offset + landed,
                    .requested_bytes = slice->bytes - landed,
                    .phase = timeline_current,
                    .step = decode_step,
                    .layer = (uint32_t)layer,
                    .ordinal = (uint32_t)ordinal,
                    .worker = (uint32_t)worker,
                    .attempt = attempt,
                    .expert = (uint32_t)decode_job_tags[ordinal],
                    .seat = (uint32_t)decode_job_seats[ordinal],
                };
                record->begin_ns = rotation_now_ns();
            } else if (worker >= 0 && worker < SALT_LR_MAX_THREADS) {
                decode_read_call_dropped[worker]++;
            }
        }
        if (tick) tick->read_calls++;
        ssize_t n = pread(source_fd, slice->dst + landed,
            slice->bytes - landed, (off_t)(slice->file_offset + landed));
        int read_error = n < 0 ? errno : 0;
        if (record) {
            record->returned_bytes = (int64_t)n;
            record->error = read_error;
            record->end_ns = rotation_now_ns();
        }
        attempt++;
        if (n < 0 && read_error == EINTR) {
            if (tick) tick->read_interrupts++;
            continue;
        }
        if (n <= 0) return SALT_TFAIL(-1);
        if (tick) {
            tick->short_reads += (size_t)n < slice->bytes - landed;
            tick->read_bytes += (uint64_t)n;
        }
        landed += (size_t)n;
    }
    return 0;
}

static int read_decode_slice(const RotationSlice *slice,
        SaltRotationWorkerTick *tick, int worker, int ordinal, int layer) {
    if (!decode_read_mmap)
        return read_decode_pread_slice(slice, tick, worker, ordinal, layer);
    size_t prefix = (size_t)(slice->file_offset % source_page_bytes);
    off_t offset = (off_t)(slice->file_offset - prefix);
    size_t bytes = prefix + slice->bytes;
    decode_map_calls[worker]++;
    void *view = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, source_fd, offset);
    if (view == MAP_FAILED) {
        decode_map_errors[worker]++;
        return SALT_TFAIL(-1);
    }
    memcpy(slice->dst, (const uint8_t *)view + prefix, slice->bytes);
    if (tick) tick->read_bytes += slice->bytes;
    /* Only this transient selected extent is unmapped, never the init address map. */
    if (munmap(view, bytes) != 0) {
        decode_map_errors[worker]++;
        return SALT_TFAIL(-1);
    }
    return 0;
}
static void *rotation_worker(void *opaque) {
    int id = *(const int *)opaque;
    uint64_t seen = 0;
    pthread_mutex_lock(&mu);
    for (;;) {
        int first, end;
        while (!stopping && seen == sequence) pthread_cond_wait(&work_cv, &mu);
        if (stopping) break;
        seen = sequence; first = first_layer; end = end_layer;
        int expert_reads = decode_io_mode ? decode_io_count : 0;
        pthread_mutex_unlock(&mu);
        if (expert_reads) {
            SaltRotationWorkerTick *tick = id < expert_reads ? worker_tick_current(decode_io_layer, id) : NULL;
            if (tick) {
                tick->task_begin_ns = rotation_now_ns();
                tick->gate_pass_ns = tick->task_begin_ns;
                tick->read_begin_ns = rotation_now_ns();
                tick->read_cpu_valid = rotation_thread_cpu_ns(&tick->read_cpu_begin_ns);
            }
            int rc = 0;
            for (int j = id; j < expert_reads; j += width) {
#if defined(__APPLE__)
                if (decode_read_advice && decode_jobs[j].bytes <= INT_MAX) {
                    /* Exact selected extent only; hint and read are both timed. */
                    struct radvisory advice = {0};
                    advice.ra_offset = (off_t)decode_jobs[j].file_offset;
                    advice.ra_count = (int)decode_jobs[j].bytes;
                    decode_advice_calls[id]++;
                    if (fcntl(source_fd, F_RDADVISE, &advice) != 0)
                        decode_advice_errors[id]++;
                }
#endif
                rc = read_decode_slice(&decode_jobs[j], tick, id, j,
                    decode_io_layer);
                if (rc != 0) {
                    pthread_mutex_lock(&mu); failed = 1; pthread_mutex_unlock(&mu);
                    break;
                }
            }
            if (tick) {
                tick->read_cpu_valid |= rotation_thread_cpu_ns(&tick->read_cpu_end_ns) << 1u;
                tick->read_end_ns = rotation_now_ns();
                tick->read_status = rc;
                tick->task_end_ns = rotation_now_ns();
            }
        } else for (int layer = first; layer < end; layer++) {
            SaltRotationWorkerTick *tick = worker_tick_current(layer, id);
            if (tick) tick->task_begin_ns = rotation_now_ns();
            pthread_mutex_lock(&mu);
            if (tick) tick->lock_acquired_ns = rotation_now_ns();
            while (!cancelled && !stopping && allowed_layer < layer) {
                if (waiting_layer < layer) {
                    /* Any free worker may observe the previous bank owner's
                     * DONE. Do not wait for all slices of the preceding load. */
                    waiting_layer = layer;
                    pthread_mutex_unlock(&mu);
                    RotationLoadGate *gate = &load_gates[layer];
                    gate->worker = (uint32_t)id;
                    int rc = 1;
                    if (wait_done && source_layers[layer].done_layer >= 0) {
                        gate->wait_begin_ns = rotation_now_ns();
                        do {
                            if (is_cancelled()) { rc = -1; break; }
                            rc = wait_done(signal_context,
                                event_base + (uint64_t)source_layers[layer].done_layer + 1u);
                        } while (rc == 0);
                        gate->done_seen_ns = rotation_now_ns();
                    }
                    salt_lr_io_snapshot(&gate->io_begin);
                    pthread_mutex_lock(&mu);
                    if (rc < 0) { failed = 1; cancelled = 1; }
                    allowed_layer = layer;
                    SaltRotationSignalTick *signal = signal_tick_current(layer);
                    if (signal) signal->permit_begin_ns = rotation_now_ns();
                    pthread_cond_broadcast(&work_cv);
                    pthread_cond_broadcast(&done_cv);
                    if (signal) signal->permit_end_ns = rotation_now_ns();
                } else {
                    if (tick && !tick->wait_begin_ns) tick->wait_begin_ns = rotation_now_ns();
                    pthread_cond_wait(&work_cv, &mu);
                    if (tick) { tick->waits++; tick->wait_end_ns = rotation_now_ns(); }
                }
            }
            if (cancelled || stopping) { pthread_mutex_unlock(&mu); break; }
            pthread_mutex_unlock(&mu);
            const RotationSlice *slice = &slices[layer][id];
            load_begin_ns[layer][id] = rotation_now_ns();
            prepare_attempts[layer][id] = prepare_errors[layer][id] = 0;
            copy_begin_ns[layer][id] = rotation_now_ns();
            if (tick) {
                tick->gate_pass_ns = load_begin_ns[layer][id];
                tick->read_begin_ns = copy_begin_ns[layer][id];
            }
            if (tick) tick->read_cpu_valid = rotation_thread_cpu_ns(&tick->read_cpu_begin_ns);
            int read_rc = read_fixed_slice(slice, tick);
            if (tick) tick->read_cpu_valid |= rotation_thread_cpu_ns(&tick->read_cpu_end_ns) << 1u;
            copy_end_ns[layer][id] = rotation_now_ns();
            if (tick) { tick->read_end_ns = copy_end_ns[layer][id]; tick->read_status = read_rc; }
            if (read_rc != 0) {
                pthread_mutex_lock(&mu);
                failed = cancelled = 1;
                pthread_cond_broadcast(&work_cv);
                pthread_cond_broadcast(&done_cv);
                pthread_mutex_unlock(&mu);
                if (publish_ready) publish_ready(signal_context, UINT64_MAX);
                break;
            }
            pthread_mutex_lock(&mu);
            copied[layer]++;
            if (copied[layer] == (unsigned)width) {
                if (tick) tick->finalizer = 1;
                /* Only the last copy finalizes this layer. Other workers can
                 * immediately take their next fixed slice when its bank is free. */
                pthread_mutex_unlock(&mu);
                uint64_t joined = rotation_now_ns();
                SaltRotationTiming *timing = &timings[timing_bucket][layer];
                uint64_t earliest = UINT64_MAX, load_earliest = UINT64_MAX;
                uint64_t latest = 0, worker_sum = 0, prepare_sum = 0;
                unsigned calls = 0, errors = 0;
                int critical_worker = 0;
                for (int w = 0; w < width; w++) {
                    uint64_t a = copy_begin_ns[layer][w], b = copy_end_ns[layer][w];
                    uint64_t load_start = load_begin_ns[layer][w];
                    if (a < earliest) earliest = a;
                    if (load_start < load_earliest) load_earliest = load_start;
                    if (b > latest) { latest = b; critical_worker = w; }
                    worker_sum += rotation_elapsed(a, b);
                    prepare_sum += rotation_elapsed(load_start, a);
                    calls += prepare_attempts[layer][w];
                    errors += prepare_errors[layer][w];
                }
                timing->bucket = timing_bucket; timing->layer = (uint32_t)layer;
                timing->copies++; timing->bytes += layer_bytes;
                timing->done_wait_ns += rotation_elapsed(load_gates[layer].wait_begin_ns,
                    load_gates[layer].done_seen_ns);
                timing->copy_span_ns += rotation_elapsed(earliest, latest);
                timing->worker_copy_sum_ns += worker_sum;
                timing->copy_join_ns += rotation_elapsed(latest, joined);
                timing->source_prepare_ns += prepare_sum;
                timing->prepare_calls += calls;
                timing->prepare_failures += errors;
                timing->load_span_ns += rotation_elapsed(load_earliest, latest);
                timing->critical_start_ns += rotation_elapsed(load_earliest,
                    load_begin_ns[layer][critical_worker]);
                timing->critical_prepare_ns += rotation_elapsed(load_begin_ns[layer][critical_worker],
                    copy_begin_ns[layer][critical_worker]);
                timing->critical_copy_ns += rotation_elapsed(copy_begin_ns[layer][critical_worker], latest);
                SaltRotationTimelineRow *handoff = timeline_current_row(layer);
                if (handoff) {
                    handoff->copies = 1;
                    handoff->wait_begin_ns = load_gates[layer].wait_begin_ns;
                    handoff->done_seen_ns = load_gates[layer].done_seen_ns;
                    handoff->wait_worker = load_gates[layer].worker;
                    handoff->first_prepare_ns = load_earliest;
                    handoff->critical_prepare_ns = load_begin_ns[layer][critical_worker];
                    handoff->critical_copy_ns = copy_begin_ns[layer][critical_worker];
                    handoff->first_copy_ns = earliest;
                    handoff->last_copy_ns = latest;
                    handoff->joined_ns = joined;
                    handoff->critical_worker = (uint32_t)critical_worker;
                }
                uint64_t discard_start = rotation_now_ns();
                if (handoff) handoff->cleanup_begin_ns = discard_start;
                /* The initialization mapping stays untouched across walks.
                 * Only the fixed destination bank is overwritten. */
                uint64_t discard_end = rotation_now_ns();
                timing->source_discard_ns += rotation_elapsed(discard_start, discard_end);
                if (handoff) {
                    SaltRotationIoSnapshot io_end;
                    SaltRotationIoSnapshot io_begin = load_gates[layer].io_begin;
                    salt_lr_io_snapshot(&io_end);
                    handoff->cleanup_end_ns = discard_end;
                    handoff->copy_bytes = layer_bytes;
                    handoff->io_begin_ns = io_begin.sample_end_ns;
                    handoff->io_end_ns = io_end.sample_begin_ns;
                    handoff->io_sample_ns =
                        rotation_elapsed(io_begin.sample_begin_ns, io_begin.sample_end_ns) +
                        rotation_elapsed(io_end.sample_begin_ns, io_end.sample_end_ns);
                    handoff->io_valid = io_begin.valid & io_end.valid;
                    if (io_end.read_bytes < io_begin.read_bytes) handoff->io_valid &= ~1u;
                    if (io_end.write_bytes < io_begin.write_bytes) handoff->io_valid &= ~2u;
                    if (io_end.major_faults < io_begin.major_faults) handoff->io_valid &= ~4u;
                    handoff->io_read_bytes = handoff->io_valid & 1u
                        ? io_end.read_bytes - io_begin.read_bytes : 0u;
                    handoff->io_write_bytes = handoff->io_valid & 2u
                        ? io_end.write_bytes - io_begin.write_bytes : 0u;
                    handoff->io_major_faults = handoff->io_valid & 4u
                        ? io_end.major_faults - io_begin.major_faults : 0u;
                }
                for (int e = 0; e < experts; e++)
                    salt_gpu_text_register_expert(layer, e, experts, (layer % SALT_LR_BANK_COUNT) * experts + e);
                pthread_mutex_lock(&mu);
                /* Shared-event values are monotonic. Only final publication is
                 * ordered; workers do not rendezvous here after every copy. */
                while (!cancelled && !stopping && next_publish_layer != layer)
                    pthread_cond_wait(&done_cv, &mu);
                if (cancelled || stopping) { pthread_mutex_unlock(&mu); break; }
                range_layer[layer % SALT_LR_BANK_COUNT] = layer; loads++; ready++;
                pthread_mutex_unlock(&mu);
                deliver_ready(layer);
                pthread_mutex_lock(&mu);
                next_publish_layer++;
                pthread_cond_broadcast(&done_cv);
                pthread_mutex_unlock(&mu);
            } else pthread_mutex_unlock(&mu);
            if (tick) tick->task_end_ns = rotation_now_ns();
        }
        pthread_mutex_lock(&mu);
        finished_workers++;
        if (finished_workers == width) { busy = 0; pthread_cond_broadcast(&done_cv); }
    }
    pthread_mutex_unlock(&mu);
    return NULL;
}
static int decode_read_many(int count, int layer) {
    pthread_mutex_lock(&mu);
    if (busy || !initialized || count < 1 || (unsigned)count > decode_seats) {
        pthread_mutex_unlock(&mu); return SALT_TFAIL(-1);
    }
    decode_io_mode = 1; decode_io_count = count; decode_io_layer = layer;
    failed = finished_workers = 0; busy = 1;
    sequence++; pthread_cond_broadcast(&work_cv);
    while (busy) pthread_cond_wait(&done_cv, &mu);
    int bad = failed;
    decode_io_mode = decode_io_count = 0;
    pthread_mutex_unlock(&mu);
    return bad ? SALT_TFAIL(-1) : 0;
}
void salt_lr_compute_done(uint64_t value) {
    salt_lr_compute_done_tick(value, rotation_now_ns());
}
void salt_lr_compute_done_tick(uint64_t value, uint64_t callback_enter_ns) {
    pthread_mutex_lock(&mu);
    if (!cancelled && !stopping && wait_done && value > event_base &&
        value - event_base <= (uint64_t)layers) {
        uint32_t bank = (uint32_t)((value - event_base - 1u) % SALT_LR_BANK_COUNT);
        if (compute_done_values[bank] < value) {
            SaltRotationSignalTick *signal = signal_tick_phase((uint32_t)(event_base / ((uint32_t)layers + 2u)) + 1u, (int)(value - event_base - 1u));
            if (signal) {
                signal->callback_enter_ns = callback_enter_ns;
                signal->callback_lock_ns = rotation_now_ns();
            }
            compute_done_values[bank] = value;
            if (signal) signal->done_notify_begin_ns = rotation_now_ns();
            pthread_cond_broadcast(&done_cv);
            if (signal) signal->done_notify_end_ns = rotation_now_ns();
        }
    }
    pthread_mutex_unlock(&mu);
}
int salt_lr_wait_compute_done(uint64_t value) {
    int aborted;
    pthread_mutex_lock(&mu);
    uint32_t bank = (uint32_t)((value - event_base - 1u) % SALT_LR_BANK_COUNT);
    while (!cancelled && !stopping && compute_done_values[bank] < value)
        pthread_cond_wait(&done_cv, &mu);
    aborted = cancelled || stopping;
    pthread_mutex_unlock(&mu);
    return aborted ? SALT_TFAIL(-1) : 1;
}
static int start_walk(int first, int end, uint64_t base,
                      SaltRotationWaitDone wait_fn,
                      SaltRotationPublishReady ready_fn, void *context) {
    pthread_mutex_lock(&mu);
    if (!initialized || busy || first < 0 || end > layers || first >= end) {
        pthread_mutex_unlock(&mu);
        return SALT_TFAIL(-1);
    }
    decode_io_mode = decode_io_count = 0;
    first_layer = first; end_layer = end; allowed_layer = first - 1;
    waiting_layer = first - 1; next_publish_layer = first;
    if (wait_fn)
        for (int bank = 0; bank < SALT_LR_BANK_COUNT; bank++)
            if (!(first == 1 && bank == 0 && primed && range_layer[0] == 0))
                range_layer[bank] = -1;
    memset(load_gates, 0, sizeof load_gates);
    event_base = base; wait_done = wait_fn; publish_ready = ready_fn; signal_context = context;
    for (int bank = 0; bank < SALT_LR_BANK_COUNT; bank++) {
        compute_done_values[bank] = base;
        ready_delivered_values[bank] = base;
    }
    salt_lr_timeline_open(base, wait_fn ? 1u : 0u);
    if (timeline_current < SALT_LR_TIMELINE_PHASES) {
        for (int layer = first; layer < end; layer++)
            timeline_rows[timeline_current][layer] = (SaltRotationTimelineRow) {
                .phase = timeline_current, .layer = (uint32_t)layer,
                .bank = (uint32_t)(layer % SALT_LR_BANK_COUNT),
                .expected_done = wait_fn && source_layers[layer].done_layer >= 0
                    ? base + (uint64_t)source_layers[layer].done_layer + 1u : 0,
                .expected_ready = wait_fn ? base + (uint64_t)layer + 1u : 0,
                .done_layer = wait_fn ? source_layers[layer].done_layer : -1
            };
    }
    timing_bucket = wait_fn ? (stream_walks++ == 0u ? 1u : 2u) : 0u;
    memset(copied, 0, sizeof copied);
    cancelled = failed = finished_workers = 0;
    last_ready_ns = 0;
    busy = 1; announced += (uint32_t)(end - first);
    pthread_mutex_unlock(&mu);
    if (wait_fn && first == 1) {
        /* Publish the already loaded layer 0 before waking the first streamed
         * load. An initially free bank is not a fabricated GPU DONE event. */
        SaltRotationTimelineRow *prime = timeline_current_row(0);
        if (prime) {
            prime->phase = timeline_current; prime->layer = 0; prime->bank = 0;
            prime->expected_ready = base + 1u; prime->done_layer = -1;
            prime->primed = 1; prime->copies = 1;
            prime->ready_begin_ns = rotation_now_ns();
        }
        deliver_ready(0);
        if (prime) prime->ready_end_ns = rotation_now_ns();
    }
    pthread_mutex_lock(&mu);
    sequence++;
    pthread_cond_broadcast(&work_cv);
    pthread_mutex_unlock(&mu);
    return 0;
}
int salt_lr_stream_begin(uint64_t base, SaltRotationWaitDone wait_fn,
                         SaltRotationPublishReady ready_fn, void *context) {
    if (!wait_fn || !ready_fn) return SALT_TFAIL(-1);
    /* Initialization already loaded layer 0. Consume that initial READY once;
     * subsequent walks reload 0 because the range now holds a later layer. */
    if (primed && layers > 1) {
        int rc = start_walk(1, layers, base, wait_fn, ready_fn, context);
        if (rc != 0) return rc;
        primed = 0;
        return 0;
    }
    return start_walk(0, layers, base, wait_fn, ready_fn, context);
}
int salt_lr_stream_finish(void) {
    int rc;
    pthread_mutex_lock(&mu);
    while (busy) pthread_cond_wait(&done_cv, &mu);
    rc = failed || cancelled;
    pthread_mutex_unlock(&mu);
    return rc ? SALT_TFAIL(-1) : 0;
}
void salt_lr_stream_cancel(void) {
    pthread_mutex_lock(&mu);
    cancelled = 1;
    pthread_cond_broadcast(&work_cv); pthread_cond_broadcast(&done_cv);
    pthread_mutex_unlock(&mu);
}
void salt_lr_shutdown(void) {
    salt_lr_stream_cancel();
    pthread_mutex_lock(&mu);
    stopping = 1;
    pthread_cond_broadcast(&work_cv); pthread_cond_broadcast(&done_cv);
    pthread_mutex_unlock(&mu);
    for (int i = 0; i < created; i++) pthread_join(workers[i], NULL);
    created = 0; initialized = 0; arena = NULL; arena_slot_count = 0;
}
int salt_lr_prepare(int layer) {
    int active;
    if (!initialized || layer < 0 || layer >= layers) return SALT_TFAIL(-1);
    if (salt_lr_ready(layer)) return 0;
    pthread_mutex_lock(&mu);
    active = busy;
    pthread_mutex_unlock(&mu);
    if (active && salt_lr_stream_finish() != 0) return SALT_TFAIL(-1);
    if (salt_lr_ready(layer)) return 0;
    if (start_walk(layer, layer + 1, 0, NULL, NULL, NULL) != 0 ||
        salt_lr_stream_finish() != 0) return SALT_TFAIL(-1);
    return salt_lr_ready(layer) ? 0 : SALT_TFAIL(-1);
}
int salt_lr_gate_next(int compute) {
    if (!initialized || compute < 0 || compute >= layers) return SALT_TFAIL(-1);
    int next = (compute + 1) % layers;
    if (salt_lr_prepare(next) != 0) return SALT_TFAIL(-1);
    done++;
    return 0;
}
int salt_lr_step(int compute) {
    int next;
    if (!decode_cpu) return salt_lr_gate_next(compute);
    if (salt_lr_prepare(compute) != 0) return SALT_TFAIL(-1);
    if (compute + 1 == layers) {
        done++;
        return 0;
    }
    next = (compute + 1) % layers;
    if (!salt_lr_ready(next) &&
            start_walk(next, next + 1, 0, NULL, NULL, NULL) != 0)
        return SALT_TFAIL(-1);
    done++;
    return 0;
}
void salt_lr_stats(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d, uint32_t *e) {
    if (a) *a = inits;
    if (b) *b = announced;
    if (c) *c = loads;
    if (d) *d = ready;
    if (e) *e = 0;
}
uint32_t salt_lr_gate_stats(uint32_t *a, uint32_t *b, uint32_t *c) {
    if (a) *a = announced;
    if (b) *b = ready;
    if (c) *c = done;
    return done;
}
void salt_lr_note_bind_fail(int which) { if (which == 1) bind_noarg++; else bind_refused++; }
uint32_t salt_lr_bind_fail(int which) { return which == 1 ? bind_noarg : bind_refused; }
void salt_lr_note_mem(int which) {
    if (which == 4) salt_trace_value(74, (int)((double)clock() * 1000.0 / CLOCKS_PER_SEC));
}
int salt_lr_init(const SaltLayerRotationConfig *cfg) {
    if (!cfg || initialized || !cfg->arena || !cfg->src || cfg->fd < 0 ||
        cfg->src_len > INT64_MAX || !cfg->slot_bytes ||
        cfg->n_layers < 1 || cfg->n_layers > SALT_LR_MAX_LAYERS ||
        cfg->n_experts < 1 || cfg->n_experts > SALT_LR_MAX_EXPERTS ||
        cfg->arena_slots < SALT_LR_BANK_COUNT * cfg->n_experts || cfg->threads < 1 ||
        cfg->threads > SALT_LR_MAX_THREADS ||
        cfg->slot_bytes > UINT64_MAX / (uint64_t)cfg->n_experts)
        return SALT_TFAIL(-1);
    layers = cfg->n_layers; experts = cfg->n_experts;
    width = cfg->threads < experts ? cfg->threads : experts;
    slot_bytes = cfg->slot_bytes; layer_bytes = slot_bytes * (uint64_t)experts;
    decode_seats = salt_lr_decode_capacity(slot_bytes,
        (unsigned)layers * (unsigned)experts);
    if (decode_seats > (unsigned)cfg->arena_slots)
        return SALT_TFAIL(-1);
    arena_slot_count = cfg->arena_slots;
    if (layer_bytes > SIZE_MAX || cfg->payload_offset > cfg->src_len ||
        (uint64_t)layers > (cfg->src_len - cfg->payload_offset) / layer_bytes)
        return SALT_TFAIL(-1);
    arena = cfg->arena;
    source_fd = cfg->fd;
    long page_bytes = sysconf(_SC_PAGESIZE);
    if (page_bytes <= 0) return SALT_TFAIL(-1);
    source_page_bytes = (size_t)page_bytes;
    memset(decode_map_calls, 0, sizeof decode_map_calls);
    memset(decode_map_errors, 0, sizeof decode_map_errors);
    for (int layer = 0; layer < layers; layer++) {
        source_layers[layer] = (RotationSource){
            .done_layer = layer >= SALT_LR_BANK_COUNT ? layer - SALT_LR_BANK_COUNT : -1
        };
        for (int w = 0; w < width; w++) {
            uint64_t first = (uint64_t)w * (uint64_t)experts / (uint64_t)width;
            uint64_t end = (uint64_t)(w + 1) * (uint64_t)experts / (uint64_t)width;
            slices[layer][w] = (RotationSlice) {
                .file_offset = cfg->payload_offset + (uint64_t)layer * layer_bytes + first * slot_bytes,
                .dst = arena + (uint64_t)(layer % SALT_LR_BANK_COUNT) * layer_bytes + first * slot_bytes,
                .bytes = (size_t)((end - first) * slot_bytes)
            };
        }
    }
    decode_residency_init();
    memset(decode_advice_calls, 0, sizeof decode_advice_calls);
    memset(decode_advice_errors, 0, sizeof decode_advice_errors);
    stopping = busy = cancelled = failed = 0; sequence = 0;
    stream_walks = 0;
    timeline_count = timeline_current = timeline_dropped = 0;
    memset(timings, 0, sizeof timings);
    memset(signal_ticks, 0, sizeof signal_ticks);
    memset(worker_ticks, 0, sizeof worker_ticks);
    memset(worker_ticks_dropped, 0, sizeof worker_ticks_dropped);
    signal_ticks_dropped = receive_ticks_dropped = 0;
    for (int bank = 0; bank < SALT_LR_BANK_COUNT; bank++) range_layer[bank] = -1;
    initialized = 1;
    for (int w = 0; w < width; w++) {
        worker_ids[w] = w;
        if (pthread_create(&workers[w], NULL, rotation_worker, &worker_ids[w]) != 0) {
            salt_lr_shutdown();
            return SALT_TFAIL(-1);
        }
        created++;
    }
    inits++; salt_trace_register(253, 1); salt_trace_value(253, width);
    if (start_walk(0, 1, 0, NULL, NULL, NULL) != 0 || salt_lr_stream_finish() != 0) {
        salt_lr_shutdown();
        return SALT_TFAIL(-1);
    }
    primed = 1;
    return 0;
}
