/* Included by layer_rotation.c: one arena, partitioned into layer-local seats. */
#define DECODE_RECORD_CAPACITY 8192u
#define DECODE_REPLAY_PLAN_CAPACITY (SALT_LR_MAX_LAYERS * SALT_LR_MAX_EXPERTS)
#define DECODE_READ_CALL_ATTEMPTS 4u
#define DECODE_READ_CALL_CAPACITY \
    (SALT_LR_MAX_LAYERS * SALT_LR_MAX_EXPERTS * DECODE_READ_CALL_ATTEMPTS)
#define DECODE_MAX_SLOTS (SALT_LR_MAX_LAYERS * SALT_LR_MAX_EXPERTS)
static int decode_active, decode_eligible, decode_cleanup_failed;
static int decode_global;
static uint32_t decode_step, decode_record_count, decode_record_dropped;
static uint64_t decode_clock;
static uint64_t decode_offsets[SALT_LR_MAX_LAYERS][SALT_LR_MAX_EXPERTS];
typedef struct {
    int layer, expert;
    uint64_t used_at, hits;
} DecodeGlobalSeat;
static DecodeGlobalSeat decode_catalog[DECODE_MAX_SLOTS];
static int decode_order[DECODE_MAX_SLOTS];
static uint32_t decode_layer_begin[SALT_LR_MAX_LAYERS];
static uint32_t decode_layer_count[SALT_LR_MAX_LAYERS];
static SaltDecodeExpertRecord decode_records[DECODE_RECORD_CAPACITY];
typedef struct {
    uint32_t step, layer, ordinal, expert, seat;
    uint64_t file_offset, bytes;
} DecodeReplayPlanRecord;
static DecodeReplayPlanRecord decode_replay_plan[DECODE_REPLAY_PLAN_CAPACITY];
static uint32_t decode_replay_plan_count, decode_replay_plan_dropped;
static SaltDecodeReadCallRecord decode_read_calls[DECODE_READ_CALL_CAPACITY];
static uint32_t decode_read_call_dropped[SALT_LR_MAX_THREADS];
static RotationSlice decode_jobs[SALT_LR_MAX_EXPERTS];
static int decode_job_tags[SALT_LR_MAX_EXPERTS];
static int decode_job_layers[SALT_LR_MAX_EXPERTS];
static int decode_job_seats[SALT_LR_MAX_EXPERTS];
static int decode_job_old_tags[SALT_LR_MAX_EXPERTS];
static int decode_job_old_layers[SALT_LR_MAX_EXPERTS];
static void *decode_map_base[DECODE_MAX_SLOTS];
static size_t decode_map_len[DECODE_MAX_SLOTS];
static const uint8_t *decode_map_payload[DECODE_MAX_SLOTS];
static int decode_io_count, decode_io_mode, decode_io_layer;
static int decode_read_many(int count, int layer);

static int decode_zero_release(int seat, int layer, int expert) {
    size_t released = 0;
    void *base;
    size_t map_len;
    const uint8_t *payload;
    if (seat < 0 || (unsigned)seat >= decode_seats ||
        layer < 0 || layer >= layers || expert < 0 || expert >= experts)
        return SALT_TFAIL(-1);
    base = decode_map_base[seat];
    map_len = decode_map_len[seat];
    payload = decode_map_payload[seat];
    if (!base && !map_len && !payload) return 0;
    if (!base || !map_len || !payload)
        return SALT_TFAIL(-1);
    if (salt_gpu_selected_resource_unbind(seat,
            (uint64_t)(uint32_t)layer * (uint32_t)experts +
                (uint32_t)expert) != 0 ||
        salt_mmap_dontneed_contained(payload, (size_t)slot_bytes,
            &released) != 0 ||
        munmap(base, map_len) != 0)
        return SALT_TFAIL(-1);
    decode_map_base[seat] = NULL;
    decode_map_len[seat] = 0;
    decode_map_payload[seat] = NULL;
    decode_zero_unmaps++;
    return 0;
}

static int decode_zero_map(int seat, int layer, int expert) {
    uint64_t file_offset, map_offset;
    size_t delta, map_len;
    void *base;
    const uint8_t *payload;
    if (seat < 0 || (unsigned)seat >= decode_seats ||
        layer < 0 || layer >= layers || expert < 0 || expert >= experts ||
        !source_page_bytes || slot_bytes > SIZE_MAX ||
        decode_map_base[seat] || decode_map_len[seat] ||
        decode_map_payload[seat])
        return SALT_TFAIL(-1);
    file_offset = decode_offsets[layer][expert];
    map_offset = file_offset - file_offset % source_page_bytes;
    delta = (size_t)(file_offset - map_offset);
    if (slot_bytes > SIZE_MAX - delta || map_offset > INT64_MAX)
        return SALT_TFAIL(-1);
    map_len = delta + (size_t)slot_bytes;
#if defined(__APPLE__)
    if (slot_bytes <= INT_MAX) {
        struct radvisory advice = {
            .ra_offset = (off_t)file_offset,
            .ra_count = (int)slot_bytes,
        };
        if (fcntl(source_fd, F_RDADVISE, &advice) != 0)
            decode_zero_advice_errors++;
    }
#endif
    base = mmap(NULL, map_len, PROT_READ, MAP_SHARED,
                source_fd, (off_t)map_offset);
    if (base == MAP_FAILED) return SALT_TFAIL(-1);
    payload = (const uint8_t *)base + delta;
    if (salt_gpu_selected_resource_bind(seat,
            (uint64_t)(uint32_t)layer * (uint32_t)experts +
                (uint32_t)expert,
            base, map_len, payload) != 0) {
        (void)munmap(base, map_len);
        return SALT_TFAIL(-1);
    }
    decode_map_base[seat] = base;
    decode_map_len[seat] = map_len;
    decode_map_payload[seat] = payload;
    decode_zero_maps++;
    return 0;
}

static void decode_move(int layer, int seat, int front) {
    uint32_t begin = decode_layer_begin[layer];
    uint32_t end = begin + decode_layer_count[layer];
    uint32_t at = begin;
    while (at < end && decode_order[at] != seat) at++;
    if (at == end) return;
    if (front) {
        for (uint32_t i = at; i > begin; i--)
            decode_order[i] = decode_order[i - 1u];
        decode_order[begin] = seat;
    } else {
        for (uint32_t i = at; i + 1u < end; i++)
            decode_order[i] = decode_order[i + 1u];
        decode_order[end - 1u] = seat;
    }
}

static int decode_lookup(int layer, int expert) {
    for (uint32_t s = 0; s < decode_seats; s++)
        if (decode_catalog[s].layer == layer &&
            decode_catalog[s].expert == expert)
            return (int)s;
    return -1;
}

static void decode_residency_init(void) {
    decode_active = 0;
    decode_global = 0;
    decode_eligible = decode_seats > 0u && decode_seats <= DECODE_MAX_SLOTS;
    if (!decode_zero_copy && decode_seats > (unsigned)arena_slot_count)
        decode_eligible = 0;
    decode_step = decode_record_count = decode_record_dropped = 0;
    decode_replay_plan_count = decode_replay_plan_dropped = 0;
    decode_clock = 0;
    decode_io_count = decode_io_mode = 0;
    decode_cleanup_failed = 0;
    decode_zero_maps = decode_zero_unmaps = decode_zero_advice_errors = 0;
    memset(decode_records, 0, sizeof decode_records);
    memset(decode_read_calls, 0, sizeof decode_read_calls);
    memset(decode_read_call_dropped, 0, sizeof decode_read_call_dropped);
    memset(decode_map_base, 0, sizeof decode_map_base);
    memset(decode_map_len, 0, sizeof decode_map_len);
    memset(decode_map_payload, 0, sizeof decode_map_payload);
    for (int l = 0; l < layers; l++) {
        uint32_t begin = (uint32_t)(((uint64_t)(uint32_t)l * decode_seats) /
            (uint32_t)layers);
        uint32_t end = (uint32_t)(((uint64_t)(uint32_t)(l + 1) * decode_seats) /
            (uint32_t)layers);
        decode_layer_begin[l] = begin;
        decode_layer_count[l] = end - begin;
        if (decode_layer_count[l] < 1u || decode_layer_count[l] > SALT_LR_MAX_EXPERTS)
            decode_eligible = 0;
    }
    for (unsigned s = 0; s < decode_seats; s++) {
        decode_catalog[s] = (DecodeGlobalSeat){-1, -1, 0, 0};
        decode_order[s] = (int)s;
    }
    for (int l = 0; l < layers; l++)
        for (int e = 0; e < experts; e++)
            decode_offsets[l][e] = slices[l][0].file_offset +
                (uint64_t)e * slot_bytes;
}

void salt_lr_decode_invalidate(void) {
    if (decode_zero_copy) {
        int have_maps = 0;
        for (unsigned s = 0; s < decode_seats; s++)
            have_maps |= decode_map_base[s] != NULL;
        if (have_maps && salt_gpu_selected_resources_fence() != 0)
            decode_cleanup_failed = 1;
        if (!decode_cleanup_failed)
            for (unsigned s = 0; s < decode_seats; s++)
                if (decode_map_base[s] && decode_zero_release((int)s,
                        decode_catalog[s].layer,
                        decode_catalog[s].expert) != 0)
                    decode_cleanup_failed = 1;
    }
    decode_active = 0;
    for (unsigned s = 0; s < decode_seats; s++) {
        decode_catalog[s] = (DecodeGlobalSeat){-1, -1, 0, 0};
        decode_order[s] = (int)s;
    }
}

int salt_lr_decode_begin(void) {
    if (!initialized || !decode_eligible || busy || decode_cleanup_failed)
        return SALT_TFAIL(-1);
    decode_active = 1;
    decode_step++;
    if (decode_cpu) {
        salt_trace_register(158, 1);
        salt_trace_value(158, (int)decode_step);
    }
    return 0;
}
int salt_lr_decode_active(void) { return decode_active; }
int salt_lr_decode_global_exchange(int enabled) {
    return __atomic_exchange_n(&decode_global, enabled != 0, __ATOMIC_ACQ_REL);
}
int salt_lr_decode_global_active(void) {
    return __atomic_load_n(&decode_global, __ATOMIC_ACQUIRE);
}
int salt_lr_decode_zero_copy_active(void) {
    return decode_active && decode_zero_copy;
}
const uint8_t *salt_lr_decode_payload(int layer, int expert) {
    int seat;
    if (!decode_active || !decode_zero_copy ||
        (seat = decode_lookup(layer, expert)) < 0)
        return NULL;
    return decode_map_payload[seat];
}
unsigned salt_lr_decode_seats(void) { return decode_seats; }
int salt_lr_decode_find(int layer, int expert, int *slot) {
    int seat;
    if (!decode_active || !slot || layer < 0 || layer >= layers ||
        expert < 0 || expert >= experts ||
        (seat = decode_lookup(layer, expert)) < 0)
        return 0;
    *slot = seat;
    return 1;
}
unsigned salt_lr_decode_bank_origin(unsigned layer) {
    (void)layer;
    return 0u;
}

int salt_lr_decode_fetch(int layer, const int *requested, int count, int *out) {
    unsigned char protected_seats[DECODE_MAX_SLOTS] = {0};
    uint32_t resident_by_layer[SALT_LR_MAX_LAYERS] = {0};
    int misses = 0;
    SaltDecodeExpertRecord record;
    uint32_t layer_begin = layer >= 0 && layer < layers
        ? decode_layer_begin[layer] : 0u;
    uint32_t layer_count = layer >= 0 && layer < layers
        ? decode_layer_count[layer] : 0u;
    uint32_t layer_end = layer_begin + layer_count;
    int global = salt_lr_decode_global_active();
    uint32_t victim_begin = global ? 0u : layer_begin;
    uint32_t victim_end = global ? decode_seats : layer_end;
    uint32_t capacity = victim_end - victim_begin;
    uint32_t hot_floor = global ? salt_lr_decode_min_layer_seats() : 0u;
    if (global) {
        salt_trace_register(107, 1);
        salt_trace_value(107, count);
        salt_trace_register(108, 1);
        salt_trace_value(108, (int)hot_floor);
        for (uint32_t seat = 0; seat < decode_seats; seat++)
            if (decode_catalog[seat].layer >= 0 &&
                decode_catalog[seat].layer < layers &&
                decode_catalog[seat].expert >= 0)
                resident_by_layer[decode_catalog[seat].layer]++;
    }
    if (!decode_active || !requested || !out || layer < 0 || layer >= layers ||
        count < 1 || (uint32_t)count > capacity || !decode_eligible || busy)
        return SALT_TFAIL(-1);
    memset(&record, 0, sizeof record);
    record.step = decode_step;
    record.layer = (uint32_t)layer;
    record.requests = (uint32_t)count;
    /* Protect every hit before selecting any victim. */
    for (int i = 0; i < count; i++) {
        int seat;
        if (requested[i] < 0 || requested[i] >= experts)
            return SALT_TFAIL(-1);
        for (int j = 0; j < i; j++)
            if (requested[j] == requested[i]) return SALT_TFAIL(-1);
        seat = decode_lookup(layer, requested[i]);
        out[i] = seat;
        if (seat >= 0) {
            protected_seats[seat] = 1u;
            decode_catalog[seat].used_at = ++decode_clock;
            decode_catalog[seat].hits++;
            record.hits++;
            if (!global) decode_move(layer, seat, 1);
        }
    }
    for (int i = 0; i < count; i++) {
        int seat = -1;
        if (out[i] >= 0) continue;
        for (uint32_t rank = victim_end; rank > victim_begin; rank--) {
            int candidate = decode_order[rank - 1u];
            if (!protected_seats[candidate] &&
                decode_catalog[candidate].expert < 0) {
                seat = candidate;
                break;
            }
        }
        if (seat < 0)
            for (uint32_t rank = victim_end; rank > victim_begin; rank--) {
                int candidate = decode_order[rank - 1u];
                int owner = decode_catalog[candidate].layer;
                if (!protected_seats[candidate] &&
                    (!global || owner < 0 || owner >= layers ||
                        resident_by_layer[owner] > hot_floor) &&
                    (seat < 0 || decode_catalog[candidate].used_at <
                        decode_catalog[seat].used_at)) {
                    seat = candidate;
                }
            }
        if (seat < 0 && global)
            for (uint32_t rank = victim_end; rank > victim_begin; rank--) {
                int candidate = decode_order[rank - 1u];
                if (!protected_seats[candidate] &&
                    (seat < 0 || decode_catalog[candidate].used_at <
                        decode_catalog[seat].used_at))
                    seat = candidate;
            }
        if (seat < 0) return SALT_TFAIL(-1);
        if (decode_catalog[seat].expert >= 0) record.evictions++;
        protected_seats[seat] = 1u;
        decode_job_old_layers[misses] = decode_catalog[seat].layer;
        decode_job_old_tags[misses] = decode_catalog[seat].expert;
        if (global && decode_catalog[seat].layer >= 0 &&
            decode_catalog[seat].layer < layers &&
            resident_by_layer[decode_catalog[seat].layer] > 0u)
            resident_by_layer[decode_catalog[seat].layer]--;
        if (global) resident_by_layer[layer]++;
        decode_catalog[seat].layer = -1;
        decode_catalog[seat].expert = -1;
        out[i] = seat;
        decode_jobs[misses] = (RotationSlice){
            decode_offsets[layer][requested[i]],
            arena + (uint64_t)seat * slot_bytes,
            (size_t)slot_bytes,
        };
        decode_job_layers[misses] = layer;
        decode_job_tags[misses] = requested[i];
        decode_job_seats[misses] = seat;
        misses++;
        if (!global) decode_move(layer, seat, 0);
    }
    if (decode_step == 1u) {
        for (int i = 0; i < count; i++) {
            if (decode_replay_plan_count < DECODE_REPLAY_PLAN_CAPACITY) {
                DecodeReplayPlanRecord *p =
                    &decode_replay_plan[decode_replay_plan_count++];
                *p = (DecodeReplayPlanRecord) {
                    .step = decode_step,
                    .layer = (uint32_t)layer,
                    .ordinal = (uint32_t)i,
                    .expert = (uint32_t)requested[i],
                    .seat = (uint32_t)out[i],
                    .file_offset = decode_offsets[layer][requested[i]],
                    .bytes = slot_bytes,
                };
            } else {
                decode_replay_plan_dropped++;
            }
        }
    }
    SaltRotationIoSnapshot io_begin = {0}, io_end = {0};
    uint64_t transport_begin = rotation_now_ns();
    if (misses) salt_lr_io_snapshot(&io_begin);
    if (misses && decode_zero_copy) {
        int retire = 0, mapped = 0;
        for (int i = 0; i < misses; i++)
            retire |= decode_job_old_tags[i] >= 0;
        if (retire && salt_gpu_selected_resources_fence() != 0)
            return SALT_TFAIL(-1);
        for (int i = 0; i < misses; i++)
            if (decode_job_old_tags[i] >= 0 &&
                decode_zero_release(decode_job_seats[i],
                    decode_job_old_layers[i], decode_job_old_tags[i]) != 0)
                return SALT_TFAIL(-1);
        for (; mapped < misses; mapped++)
            if (decode_zero_map(decode_job_seats[mapped],
                    decode_job_layers[mapped], decode_job_tags[mapped]) != 0)
                break;
        if (mapped != misses) {
            if (mapped && salt_gpu_selected_resources_fence() == 0)
                while (mapped > 0) {
                    mapped--;
                    (void)decode_zero_release(decode_job_seats[mapped],
                        decode_job_layers[mapped], decode_job_tags[mapped]);
                }
            return SALT_TFAIL(-1);
        }
    } else if (misses && decode_read_many(misses, layer) != 0) {
        return SALT_TFAIL(-1);
    }
    uint64_t joined = rotation_now_ns();
    if (misses) salt_lr_io_snapshot(&io_end);
    SaltRotationTimelineRow *h = timeline_current_row(layer);
    if (h) {
        *h = (SaltRotationTimelineRow){
            .phase = timeline_current,
            .layer = (uint32_t)layer,
            .bank = 0,
            .copies = (uint32_t)misses,
            .copy_bytes = decode_zero_copy ? 0u :
                (uint64_t)misses * slot_bytes,
            .done_layer = -1,
            .joined_ns = joined,
        };
        uint64_t earliest = UINT64_MAX, first_read = UINT64_MAX;
        if (decode_zero_copy && misses) {
            h->first_prepare_ns = transport_begin;
            h->critical_prepare_ns = transport_begin;
            h->critical_copy_ns = transport_begin;
            h->first_copy_ns = transport_begin;
            h->last_copy_ns = joined;
        }
        if (!decode_zero_copy)
            for (int w = 0; w < width && w < misses; w++) {
                SaltRotationWorkerTick *t = worker_tick_current(layer, w);
                if (!t) continue;
                if (t->task_begin_ns < earliest) earliest = t->task_begin_ns;
                if (t->read_begin_ns < first_read) first_read = t->read_begin_ns;
                if (t->read_end_ns > h->last_copy_ns) {
                    h->last_copy_ns = t->read_end_ns;
                    h->critical_worker = (uint32_t)w;
                    h->critical_prepare_ns = t->gate_pass_ns;
                    h->critical_copy_ns = t->read_begin_ns;
                }
            }
        if (misses) {
            if (!decode_zero_copy) {
                h->first_prepare_ns = earliest == UINT64_MAX ? 0 : earliest;
                h->first_copy_ns = first_read == UINT64_MAX ? 0 : first_read;
            }
            h->cleanup_begin_ns = h->cleanup_end_ns = joined;
            h->io_begin_ns = io_begin.sample_end_ns;
            h->io_end_ns = io_end.sample_begin_ns;
            h->io_sample_ns = rotation_elapsed(io_begin.sample_begin_ns,
                io_begin.sample_end_ns) + rotation_elapsed(io_end.sample_begin_ns,
                io_end.sample_end_ns);
            h->io_valid = io_begin.valid & io_end.valid;
            if (io_end.read_bytes < io_begin.read_bytes) h->io_valid &= ~1u;
            if (io_end.write_bytes < io_begin.write_bytes) h->io_valid &= ~2u;
            if (io_end.major_faults < io_begin.major_faults) h->io_valid &= ~4u;
            h->io_read_bytes = h->io_valid & 1u ?
                io_end.read_bytes - io_begin.read_bytes : 0;
            h->io_write_bytes = h->io_valid & 2u ?
                io_end.write_bytes - io_begin.write_bytes : 0;
            h->io_major_faults = h->io_valid & 4u ?
                io_end.major_faults - io_begin.major_faults : 0;
        }
        h->ready_begin_ns = rotation_now_ns();
    }
    for (int i = 0; i < misses; i++) {
        int seat = decode_job_seats[i];
        decode_catalog[seat].layer = decode_job_layers[i];
        decode_catalog[seat].expert = decode_job_tags[i];
        decode_catalog[seat].used_at = ++decode_clock;
    }
    if (h) h->ready_end_ns = rotation_now_ns();
    if (h && layer + 1 == layers) last_ready_ns = h->ready_end_ns;
    record.misses = (uint32_t)misses;
    record.bytes = (uint64_t)misses * slot_bytes;
    if (decode_cpu) {
        salt_trace_register(159, misses ? 2 : 1);
        salt_trace_value(159, layer);
        salt_trace_register(160, 1);
        salt_trace_value(160, misses);
    }
    for (uint32_t s = 0; s < decode_seats; s++)
        record.resident += decode_catalog[s].layer == layer &&
            decode_catalog[s].expert >= 0;
    if (decode_record_count < DECODE_RECORD_CAPACITY)
        decode_records[decode_record_count++] = record;
    else
        decode_record_dropped++;
    return 0;
}

uint32_t salt_lr_decode_record_count(void) { return decode_record_count; }
uint32_t salt_lr_decode_record_dropped(void) { return decode_record_dropped; }
void salt_lr_decode_stats(uint64_t *requests, uint64_t *hits,
                          uint64_t *misses, uint64_t *bytes) {
    uint64_t req = 0, hit = 0, miss = 0, nbytes = 0;
    for (uint32_t index = 0; index < decode_record_count; index++) {
        req += decode_records[index].requests;
        hit += decode_records[index].hits;
        miss += decode_records[index].misses;
        nbytes += decode_records[index].bytes;
    }
    if (requests) *requests = req;
    if (hits) *hits = hit;
    if (misses) *misses = miss;
    if (bytes) *bytes = nbytes;
}
uint32_t salt_lr_decode_replay_plan_count(void) { return decode_replay_plan_count; }
uint32_t salt_lr_decode_replay_plan_dropped(void) { return decode_replay_plan_dropped; }
uint32_t salt_lr_decode_read_call_capacity(void) {
    return DECODE_READ_CALL_CAPACITY;
}
uint32_t salt_lr_decode_read_call_dropped(void) {
    uint32_t total = 0;
    for (int w = 0; w < SALT_LR_MAX_THREADS; w++)
        total += decode_read_call_dropped[w];
    return total;
}
int salt_lr_decode_read_call(uint32_t index, SaltDecodeReadCallRecord *out) {
    if (!out || index >= DECODE_READ_CALL_CAPACITY ||
            !decode_read_calls[index].begin_ns)
        return 0;
    *out = decode_read_calls[index];
    return 1;
}
int salt_lr_decode_replay_plan(uint32_t index, uint32_t *step,
        uint32_t *layer, uint32_t *ordinal, uint32_t *expert,
        uint32_t *seat, uint64_t *file_offset, uint64_t *bytes) {
    if (index >= decode_replay_plan_count || !step || !layer || !ordinal ||
            !expert || !seat || !file_offset || !bytes)
        return 0;
    DecodeReplayPlanRecord *p = &decode_replay_plan[index];
    *step = p->step;
    *layer = p->layer;
    *ordinal = p->ordinal;
    *expert = p->expert;
    *seat = p->seat;
    *file_offset = p->file_offset;
    *bytes = p->bytes;
    return 1;
}
int salt_lr_decode_record(uint32_t index, SaltDecodeExpertRecord *out) {
    if (!out || index >= decode_record_count) return 0;
    *out = decode_records[index];
    return 1;
}
int salt_lr_decode_seat(unsigned layer, unsigned rank, int *expert, int *slot) {
    unsigned found = 0;
    if (!decode_eligible || layer >= (unsigned)layers ||
        !expert || !slot)
        return 0;
    for (uint32_t at = 0; at < decode_seats; at++) {
        int seat = decode_order[at];
        if (decode_catalog[seat].layer != (int)layer ||
            decode_catalog[seat].expert < 0)
            continue;
        if (found++ == rank) {
            *expert = decode_catalog[seat].expert;
            *slot = seat;
            return 1;
        }
    }
    return 0;
}
unsigned salt_lr_decode_layer_seats(unsigned layer) {
    unsigned count = 0;
    if (layer >= (unsigned)layers) return 0u;
    for (uint32_t seat = 0; seat < decode_seats; seat++)
        count += decode_catalog[seat].layer == (int)layer &&
            decode_catalog[seat].expert >= 0;
    return count;
}
unsigned salt_lr_decode_min_layer_seats(void) {
    unsigned result = UINT_MAX;
    for (int layer = 0; layer < layers; layer++)
        if (decode_layer_count[layer] < result) result = decode_layer_count[layer];
    return result == UINT_MAX ? 0u : result;
}
unsigned salt_lr_decode_max_layer_seats(void) {
    unsigned result = 0;
    for (int layer = 0; layer < layers; layer++)
        if (decode_layer_count[layer] > result) result = decode_layer_count[layer];
    return result;
}
int salt_lr_decode_layer_entry(unsigned layer, unsigned rank,
        int *expert, int *slot) {
    return salt_lr_decode_seat(layer, rank, expert, slot);
}
