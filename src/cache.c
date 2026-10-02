/* cache.c -- the routed-expert cache. Three explicit slot states:
 * EMPTY, LOADING (being read into right now), or READY.
 * Fetch is three phases: reserve serially (no double-claim), read in
 * parallel in disk-offset order (keeps the device queue deep), publish
 * only what arrived.
 *
 * Phase 2 uses a PERSISTENT fetch pool: worker threads wait on a
 * queue instead of spawning/joining 8 threads per layer. The spawn
 * churn was the bulk of the measured fetch phase (22-38 ms for a
 * ~3.5 ms disk read); the pool removes it. */
#include "salt/salt.h"
#include "thread-lifecycle.h"

/* NOTE: _GNU_SOURCE comes from the Makefile on non-Darwin (glibc gates
 * MADV_DONTNEED / POSIX_FADV_* behind it). Defining it here would be
 * too late -- salt/salt.h pulls in libc headers first. */
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>     /* mmap / madvise */
#include <fcntl.h>        /* posix_fadvise / POSIX_FADV_* (Linux);
                           * F_RDADVISE (macOS, via <sys/fcntl.h>) */
#ifdef __APPLE__
#include <sys/fcntl.h>
#endif
#include <string.h>
#include <unistd.h>
#include <time.h>

/* SALT_FETCH_MS diagnostic timer (same monotonic clock as now_s). */
static double salt_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifndef SALT_MAX_EXPERT_BATCH
#define SALT_MAX_EXPERT_BATCH 256
#endif


/* mixture mode: a slot whose evicted victim had at least this many
 * hits stays a resident arena copy (the memo set); colder victims
 * return to the mmap (zerocopy fetch). */
#ifndef SALT_MIX_HITS
#define SALT_MIX_HITS 2
#endif

/* Keys are offset by +1 on both fields so zero remains an invalid key. */
static uint64_t expert_key(int layer, int expert) {
    return (((uint64_t)(uint32_t)(layer + 1)) << 32) | (uint32_t)(expert + 1);
}

static int key_layer(uint64_t key) {
    return (int)(uint32_t)(key >> 32) - 1;
}

static int key_expert(uint64_t key) {
    return (int)(uint32_t)key - 1;
}

static size_t resident_index(const SaltCache *c, int layer, int expert) {
    return (size_t)layer * (size_t)c->n_experts + (size_t)expert;
}

/* O(1) authoritative residency lookup. -1 means absent; -2 means the
 * direct table and slot metadata disagree and the caller must fail closed. */
static int cache_lookup(const SaltCache *c, int layer, int expert) {
    if (!c || layer < 0 || layer >= c->n_layers ||
        expert < 0 || expert >= c->n_experts)
        return -2;
    uint64_t key = expert_key(layer, expert);
    int slot = c->resident_slot[resident_index(c, layer, expert)];
    if (slot == -1) return -1;
    if (slot < -1 || slot >= c->nslot ||
        c->state[slot] == SALT_SLOT_EMPTY ||
        c->key[slot] != key)
        return -2;
    return slot;
}

static int env_enabled(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

/* Diagnostic only: mincore reports whether the mmap-backed canonical source
 * pages are resident without faulting them in. The result is page-granular
 * and immediately stale, so it is accounting evidence, never a fetch gate. */
static int source_pages_probe(const SaltCache *c, int64_t off,
                              int64_t *resident, int64_t *total,
                              double *elapsed) {
    if (!c || !c->pool || !c->pool->map || off < 0 ||
        c->slot_bytes < 1 || (size_t)off > c->pool->map_len ||
        (size_t)c->slot_bytes > c->pool->map_len - (size_t)off)
        return -1;
    long page_l = sysconf(_SC_PAGESIZE);
    if (page_l <= 0) return -1;
    size_t page = (size_t)page_l;
    size_t uoff = (size_t)off;
    size_t start = (uoff / page) * page;
    size_t span = (uoff - start) + (size_t)c->slot_bytes;
    size_t npages = (span + page - 1) / page;
    unsigned char *vec = (unsigned char *)malloc(npages);
    if (!vec) return -1;
    double t0 = salt_now_s();
#ifdef __APPLE__
    int rc = mincore(c->pool->map + start, span, (char *)vec);
#else
    int rc = mincore(c->pool->map + start, span, vec);
#endif
    double dt = salt_now_s() - t0;
    int64_t hot = 0;
    if (rc == 0)
        for (size_t i = 0; i < npages; i++) hot += (vec[i] & 1) != 0;
    free(vec);
    if (rc != 0) return -1;
    *resident = hot;
    *total = (int64_t)npages;
    *elapsed = dt;
    return 0;
}

/* Mechanical quota plan. Arena mode defaults to auto: retain one full
 * layer's possible expert universe in a shared working set, then divide
 * every remaining physical slot deterministically among layers. A custom
 * comma-separated list is accepted only if its sum fits and every layer can
 * still reach the active working-set requirement through shared borrowing. */
static int quota_plan_init(SaltCache *c) {
    const char *spec = getenv("SALT_CACHE_LAYER_QUOTAS");
    int automatic = 0;
    if (!spec || !*spec) {
        c->quota_enabled = c->mode == 0;
        automatic = c->quota_enabled;
    } else if (strcmp(spec, "0") == 0 || strcmp(spec, "off") == 0) {
        c->quota_enabled = 0;
    } else if (strcmp(spec, "1") == 0 || strcmp(spec, "auto") == 0) {
        c->quota_enabled = 1;
        automatic = 1;
    } else {
        c->quota_enabled = 1;
        size_t spec_n = strlen(spec);
        if (spec[0] == ',' || spec[spec_n - 1] == ',' || strstr(spec, ",,")) {
            fprintf(stderr, "cache quota: invalid SALT_CACHE_LAYER_QUOTAS\n");
            return -1;
        }
        char *copy = strdup(spec);
        if (!copy) return -1;
        char *save = NULL;
        int n = 0;
        int64_t sum = 0;
        for (char *tok = strtok_r(copy, ",", &save); tok;
             tok = strtok_r(NULL, ",", &save)) {
            while (*tok == ' ' || *tok == '\t') tok++;
            char *end = NULL;
            long q = strtol(tok, &end, 10);
            while (end && (*end == ' ' || *end == '\t')) end++;
            if (n >= c->n_layers || q < 0 || q > c->nslot ||
                !end || end == tok || *end != '\0') {
                free(copy);
                fprintf(stderr, "cache quota: invalid SALT_CACHE_LAYER_QUOTAS\n");
                return -1;
            }
            c->layer_quota[n++] = (int)q;
            sum += q;
        }
        free(copy);
        if (n != c->n_layers || sum > c->nslot) {
            fprintf(stderr,
                    "cache quota: expected %d entries with sum <= %d\n",
                    c->n_layers, c->nslot);
            return -1;
        }
        c->reserved_slots = (int)sum;
        c->shared_slots = c->nslot - c->reserved_slots;
    }

    if (!c->quota_enabled) {
        c->reserved_slots = 0;
        c->shared_slots = c->nslot;
    } else if (automatic) {
        c->shared_slots = c->n_experts < c->nslot
            ? c->n_experts : c->nslot;
        c->reserved_slots = c->nslot - c->shared_slots;
        int base = c->reserved_slots / c->n_layers;
        int extra = c->reserved_slots % c->n_layers;
        for (int layer = 0; layer < c->n_layers; layer++)
            c->layer_quota[layer] = base + (layer < extra);
    }

    int sum = 0;
    for (int layer = 0; layer < c->n_layers; layer++) {
        c->layer_stats[layer].quota = c->layer_quota[layer];
        sum += c->layer_quota[layer];
    }
    if (sum != c->reserved_slots || sum + c->shared_slots != c->nslot) {
        fprintf(stderr, "cache quota: internal slot-sum mismatch\n");
        return -1;
    }
    if (c->quota_enabled) {
        int active_need = c->n_experts < c->nslot
            ? c->n_experts : c->nslot;
        if (c->shared_slots < active_need) {
            fprintf(stderr,
                    "cache quota: shared pool has %d slots, needs at least %d\n",
                    c->shared_slots, active_need);
            return -1;
        }
        for (int layer = 0; layer < c->n_layers; layer++) {
            if (c->layer_quota[layer] + c->shared_slots < active_need) {
                fprintf(stderr,
                        "cache quota: layer %d can reach %d slots, needs %d\n",
                        layer, c->layer_quota[layer] + c->shared_slots,
                        active_need);
                return -1;
            }
        }
    }

    int slot = 0;
    for (int layer = 0; layer < c->n_layers; layer++)
        for (int j = 0; j < c->layer_quota[layer]; j++)
            c->slot_home[slot++] = layer;
    while (slot < c->nslot) c->slot_home[slot++] = -1;
    return 0;
}

/* Frequency first, LRU second, slot index last: deterministic and cheap.
 * LOADING and pinned slots are never victims. */
static int better_victim(const SaltCache *c, int slot, int best) {
    if (best < 0) return 1;
    if (c->hits[slot] != c->hits[best])
        return c->hits[slot] < c->hits[best];
    if (c->used_at[slot] != c->used_at[best])
        return c->used_at[slot] < c->used_at[best];
    return slot < best;
}

static int slot_is_leased(const SaltCache *c, int slot,
                          const int *lease, int nlease) {
    if (c->lease_count && c->lease_count[slot] != 0) return 1;
    for (int i = 0; i < nlease; i++)
        if (lease[i] == slot) return 1;
    return 0;
}

static int pick_victim(const SaltCache *c, int layer,
                       const int *lease, int nlease) {
    int best = -1;
    if (!c->quota_enabled) {
        for (int slot = 0; slot < c->nslot; slot++) {
            if (slot_is_leased(c, slot, lease, nlease)) continue;
            if (c->state[slot] == SALT_SLOT_EMPTY) return slot;
            if (c->state[slot] == SALT_SLOT_LOADING || c->pinned[slot]) continue;
            if (better_victim(c, slot, best)) best = slot;
        }
        return best;
    }

    /* Fill the requesting layer's protected reservation first. */
    for (int slot = 0; slot < c->nslot; slot++)
        if (!slot_is_leased(c, slot, lease, nlease) &&
            c->slot_home[slot] == layer &&
            c->state[slot] == SALT_SLOT_EMPTY)
            return slot;
    /* Then consume free shared working-set slots. */
    for (int slot = 0; slot < c->nslot; slot++)
        if (!slot_is_leased(c, slot, lease, nlease) &&
            c->slot_home[slot] < 0 &&
            c->state[slot] == SALT_SLOT_EMPTY)
            return slot;
    /* Reclaim another layer's borrowed shared slot before disturbing this
     * layer's current working set. */
    for (int slot = 0; slot < c->nslot; slot++) {
        if (slot_is_leased(c, slot, lease, nlease) ||
            c->slot_home[slot] >= 0 ||
            c->state[slot] != SALT_SLOT_READY || c->pinned[slot])
            continue;
        if (key_layer(c->key[slot]) == layer) continue;
        if (better_victim(c, slot, best)) best = slot;
    }
    if (best >= 0) return best;
    /* Same-layer shared entries: oldest/coldest loses, so entries fetched or
     * hit in the current pass survive ahead of stale ones. */
    for (int slot = 0; slot < c->nslot; slot++) {
        if (slot_is_leased(c, slot, lease, nlease) ||
            c->slot_home[slot] >= 0 ||
            c->state[slot] != SALT_SLOT_READY || c->pinned[slot])
            continue;
        if (better_victim(c, slot, best)) best = slot;
    }
    if (best >= 0) return best;
    /* Last bounded fallback: recycle this layer's own reservation. */
    for (int slot = 0; slot < c->nslot; slot++) {
        if (slot_is_leased(c, slot, lease, nlease) ||
            c->slot_home[slot] != layer ||
            c->state[slot] != SALT_SLOT_READY || c->pinned[slot])
            continue;
        if (better_victim(c, slot, best)) best = slot;
    }
    return best;
}

static int mmap_advise_contained(const void *pointer, size_t bytes,
                                 int advice, size_t *advised_bytes) {
    uintptr_t lo, hi, begin, end, page, rem;
    long page_l;
    if (advised_bytes) *advised_bytes = 0;
    if (!pointer || bytes == 0 || !advised_bytes) return -1;
    lo = (uintptr_t)pointer;
    if (bytes > UINTPTR_MAX - lo) return -1;
    hi = lo + bytes;
    page_l = sysconf(_SC_PAGESIZE);
    if (page_l <= 0) return -1;
    page = (uintptr_t)page_l;
    begin = lo;
    rem = begin % page;
    if (rem) {
        if (page - rem > UINTPTR_MAX - begin) return -1;
        begin += page - rem;
    }
    end = hi - hi % page;
    if (begin >= end) return 0;
    if (madvise((void *)begin, (size_t)(end - begin), advice) != 0)
        return -1;
    *advised_bytes = (size_t)(end - begin);
    return 0;
}

int salt_mmap_dontneed_contained(const void *pointer, size_t bytes,
                                 size_t *released_bytes) {
    return mmap_advise_contained(
        pointer, bytes, MADV_DONTNEED, released_bytes);
}

int salt_mmap_willneed_contained(const void *pointer, size_t bytes,
                                 size_t *advised_bytes) {
    return mmap_advise_contained(
        pointer, bytes, MADV_WILLNEED, advised_bytes);
}

int salt_file_willneed(int fd, int64_t offset, size_t bytes) {
    if (fd < 0 || offset < 0 || bytes == 0 ||
            (uint64_t)bytes > UINT64_MAX - (uint64_t)offset)
        return -1;
#ifdef __APPLE__
    while (bytes > 0) {
        int chunk = bytes > (size_t)INT_MAX ? INT_MAX : (int)bytes;
        struct radvisory advice;
        advice.ra_offset = (off_t)offset;
        advice.ra_count = chunk;
        if ((int64_t)advice.ra_offset != offset ||
                fcntl(fd, F_RDADVISE, &advice) != 0)
            return -1;
        offset += chunk;
        bytes -= (size_t)chunk;
    }
    return 0;
#else
    {
        off_t start = (off_t)offset;
        off_t length = (off_t)bytes;
        if ((int64_t)start != offset || length <= 0 ||
                (uint64_t)length != (uint64_t)bytes)
            return -1;
        return posix_fadvise(fd, start, length, POSIX_FADV_WILLNEED) == 0
            ? 0 : -1;
    }
#endif
}

/* release the pages of an evicted slot back to the OS. The zero-copy
 * slot pointed into the pool mmap; when it is evicted we must flush
 * those pages (madvise DONTNEED) -- otherwise the touched-but-evicted
 * pages stay resident forever and the run's footprint grows unbounded
 * (the measured attn-decay anomaly: 229->65->55ms as the OS reclaimed
 * our pages for us, late and unevenly). */
static void slot_release(SaltCache *c, int slot) {
    if (slot < 0 || slot >= c->nslot) return;
    const uint8_t *p = c->payload[slot];
    if (!p || c->state[slot] != SALT_SLOT_READY) return;
    /* Only mmap-backed payloads get flushed (zerocopy + mixture's
     * cold slots). Arena copies are freed with the arena itself. */
    if (c->slot_bytes < 1) return;
    const void *map_base = c->pool->map;
    size_t map_len = c->pool->map_len;
    if (c->mode == 3) {
        if (!c->slot_map_base || !c->slot_map_len ||
            !c->slot_map_base[slot] || c->slot_map_len[slot] == 0)
            return;
        map_base = c->slot_map_base[slot];
        map_len = c->slot_map_len[slot];
    }
    if (!map_base) return;
    uintptr_t map_lo = (uintptr_t)map_base;
    uintptr_t p_lo = (uintptr_t)p;
    if (map_len > UINTPTR_MAX - map_lo) return;
    uintptr_t map_hi = map_lo + map_len;
    if (p_lo < map_lo || p_lo >= map_hi ||
        (uint64_t)c->slot_bytes > (uint64_t)(map_hi - p_lo))
        return;

    size_t released = 0;
    c->madvise_calls++;
    if (salt_mmap_dontneed_contained(p, (size_t)c->slot_bytes,
                                     &released) != 0)
        c->madvise_failures++;
    else
        c->madvise_bytes += (int64_t)released;
}

static int uncharge_slot(SaltCache *c, int slot, int eviction) {
    if (slot < 0 || slot >= c->nslot ||
        c->state[slot] == SALT_SLOT_EMPTY)
        return 0;
    if (c->lease_count && c->lease_count[slot] != 0) return -1;
    int layer = key_layer(c->key[slot]);
    int expert = key_expert(c->key[slot]);
    uint8_t prior_state = c->state[slot];
    if (c->resource_bound && c->resource_bound[slot]) {
        if (!c->resource_unbind_fn ||
            c->resource_unbind_fn(c->resource_ctx, slot,
                layer, expert) != 0)
            return -1;
        c->resource_bound[slot] = 0;
    }
    if (prior_state == SALT_SLOT_READY) slot_release(c, slot);
    if (c->mode == 3 && c->slot_map_base && c->slot_map_len &&
        c->slot_map_base[slot]) {
        size_t map_len = c->slot_map_len[slot];
        c->munmap_calls++;
        if (!c->munmap_fn || c->munmap_fn(c->slot_map_base[slot], map_len) != 0) {
            c->munmap_failures++;
            return -1;
        }
        c->munmap_bytes += (int64_t)map_len;
        c->mapped_current_bytes -= (int64_t)map_len;
        c->slot_map_base[slot] = NULL;
        c->slot_map_len[slot] = 0;
    }
    if (layer >= 0 && layer < c->n_layers &&
        expert >= 0 && expert < c->n_experts) {
        size_t ri = resident_index(c, layer, expert);
        if (c->resident_slot[ri] == slot) c->resident_slot[ri] = -1;
        SaltCacheLayerStats *st = &c->layer_stats[layer];
        if (c->slot_home[slot] == layer) {
            if (st->home_current > 0) st->home_current--;
        } else {
            if (st->shared_current > 0) st->shared_current--;
            st->borrow_returns++;
        }
        if (eviction && prior_state == SALT_SLOT_READY) {
            st->evictions++;
            c->nevict++;
        }
    }
    c->state[slot] = SALT_SLOT_EMPTY;
    c->key[slot] = 0;
    c->payload[slot] = NULL;
    c->used_at[slot] = 0;
    c->hits[slot] = 0;
    c->pinned[slot] = 0;
    return 0;
}

static void charge_loading(SaltCache *c, int slot, int layer, int expert) {
    c->state[slot] = SALT_SLOT_LOADING;
    c->key[slot] = expert_key(layer, expert);
    c->payload[slot] = NULL;
    c->resident_slot[resident_index(c, layer, expert)] = slot;
    SaltCacheLayerStats *st = &c->layer_stats[layer];
    if (c->slot_home[slot] == layer) {
        st->home_current++;
        if (st->home_current > st->home_peak)
            st->home_peak = st->home_current;
    } else {
        st->shared_current++;
        st->borrow_acquires++;
        if (st->shared_current > st->shared_peak)
            st->shared_peak = st->shared_current;
    }
}

/* A hot shared resident can take over a colder protected slot by swapping
 * only reservation labels. Payload bytes and slot pointers do not move. */
static void promote_shared_hit(SaltCache *c, int slot, int layer, int force) {
    if (!c->quota_enabled || slot < 0 || slot >= c->nslot ||
        c->slot_home[slot] >= 0 || c->state[slot] != SALT_SLOT_READY)
        return;
    int home = -1;
    for (int s = 0; s < c->nslot; s++) {
        if (c->slot_home[s] != layer || c->state[s] == SALT_SLOT_LOADING ||
            c->pinned[s])
            continue;
        if (c->state[s] == SALT_SLOT_EMPTY) { home = s; break; }
        if (key_layer(c->key[s]) != layer) continue;
        if (home < 0) {
            home = s;
        } else {
            int se = key_expert(c->key[s]);
            int he = key_expert(c->key[home]);
            uint64_t sd = c->demand[resident_index(c, layer, se)];
            uint64_t hd = c->demand[resident_index(c, layer, he)];
            if (sd < hd || (sd == hd && better_victim(c, s, home))) home = s;
        }
    }
    if (home < 0) return;
    if (c->state[home] == SALT_SLOT_READY && !force) {
        int shared_expert = key_expert(c->key[slot]);
        int home_expert = key_expert(c->key[home]);
        uint64_t shared_demand =
            c->demand[resident_index(c, layer, shared_expert)];
        uint64_t home_demand =
            c->demand[resident_index(c, layer, home_expert)];
        if (shared_demand <= home_demand) return;
    }

    SaltCacheLayerStats *st = &c->layer_stats[layer];
    c->slot_home[slot] = layer;
    c->slot_home[home] = -1;
    st->borrow_returns++;
    if (c->state[home] == SALT_SLOT_EMPTY) {
        if (st->shared_current > 0) st->shared_current--;
        st->home_current++;
        if (st->home_current > st->home_peak)
            st->home_peak = st->home_current;
    } else {
        /* The displaced colder resident becomes the same layer's borrower. */
        st->borrow_acquires++;
    }
    st->promotions++;
}

static int64_t touch_mapped_prefix(const uint8_t *payload, int64_t bytes) {
    const volatile uint8_t *p8 = (const volatile uint8_t *)payload;
    if (!payload || bytes <= 0) return 0;
    for (int64_t i = 0; i < bytes; i += 4096) (void)p8[i];
    (void)p8[bytes - 1];
    return bytes;
}

static void perform_fetch(FetchJob *j) {
    if (j->slotmap) {
        if (j->off < 0 || (uint64_t)j->off > (uint64_t)j->pool->map_len ||
            j->pool->nbytes < 1 ||
            (uint64_t)j->pool->nbytes >
                (uint64_t)j->pool->map_len - (uint64_t)j->off) {
            j->ok = 0;
            return;
        }
        long page_l = sysconf(_SC_PAGESIZE);
        if (page_l <= 0) {
            j->ok = 0;
            return;
        }
        int64_t page = (int64_t)page_l;
        int64_t map_off_i = j->off - j->off % page;
        size_t delta = (size_t)(j->off - map_off_i);
        if ((uint64_t)j->pool->nbytes > (uint64_t)SIZE_MAX - delta) {
            j->ok = 0;
            return;
        }
        size_t map_len = delta + (size_t)j->pool->nbytes;
        double t0 = j->queued_s > 0 ? salt_now_s() : 0;
#ifdef __APPLE__
        struct radvisory r = { .ra_offset = (off_t)j->off,
                              .ra_count = (int)j->pool->nbytes };
        fcntl(j->pool->fd, F_RDADVISE, &r);
#else
        posix_fadvise(j->pool->fd, (off_t)j->off,
                      (off_t)j->pool->nbytes, POSIX_FADV_WILLNEED);
#endif
        /* Borrow immutable authenticated bytes as a shared read-only view.
         * On macOS a private file mapping can become compressed process-owned
         * backing after repeated DONTNEED/refault cycles; MAP_SHARED keeps
         * these clean pages discardable without permitting writes. */
        void *base = mmap(NULL, map_len, PROT_READ, MAP_SHARED,
                          j->pool->fd, (off_t)map_off_i);
        if (t0 > 0) j->readahead_s = salt_now_s() - t0;
        if (base == MAP_FAILED) {
            j->ok = 0;
            return;
        }
        j->map_base = base;
        j->map_len = map_len;
        j->dst = (uint8_t *)base + delta;
        /* Consume only preparation explicitly requested by the fetch job. */
        if (j->touch_bytes > 0) {
            t0 = j->queued_s > 0 ? salt_now_s() : 0;
            (void)touch_mapped_prefix(j->dst, j->touch_bytes);
            if (t0 > 0) j->touch_s = salt_now_s() - t0;
        }
        j->ok = 1;
    } else if (j->zc && j->pool->map) {
        if (j->off < 0 || (size_t)j->off > j->pool->map_len ||
            (size_t)j->pool->nbytes > j->pool->map_len - (size_t)j->off) {
            j->ok = 0;
            return;
        }
        j->dst = j->pool->map + (size_t)j->off;
        double t0 = j->queued_s > 0 ? salt_now_s() : 0;
#ifdef __APPLE__
        struct radvisory r = { .ra_offset = (off_t)j->off,
                              .ra_count = (int)j->pool->nbytes };
        fcntl(j->pool->fd, F_RDADVISE, &r);
#else
        posix_fadvise(j->pool->fd, (off_t)j->off,
                      (off_t)j->pool->nbytes, POSIX_FADV_WILLNEED);
#endif
        if (t0 > 0) j->readahead_s = salt_now_s() - t0;
        if (j->touch_bytes > 0) {
            t0 = j->queued_s > 0 ? salt_now_s() : 0;
            (void)touch_mapped_prefix(j->dst, j->touch_bytes);
            if (t0 > 0) j->touch_s = salt_now_s() - t0;
        }
        j->ok = 1;
    } else {
        /* pread includes the syscall, kernel-to-userspace copy, and any
         * destination-page faults; there is no second userspace copy.
         * SALT_CACHE_PRETOUCH=1 moves destination faulting into an explicit
         * measured stage for diagnostics, changing timing but not bytes. */
        if (j->pretouch) {
            double touch0 = j->queued_s > 0 ? salt_now_s() : 0;
            long page_l = sysconf(_SC_PAGESIZE);
            size_t page = page_l > 0 ? (size_t)page_l : 4096;
            volatile uint8_t *p8 = (volatile uint8_t *)j->dst;
            for (size_t i = 0; i < (size_t)j->pool->nbytes; i += page)
                p8[i] = 0;
            p8[(size_t)j->pool->nbytes - 1] = 0;
            if (touch0 > 0) j->touch_s = salt_now_s() - touch0;
        }
        double t0 = j->queued_s > 0 ? salt_now_s() : 0;
        j->ok = pread(j->pool->fd, j->dst, (size_t)j->pool->nbytes,
                      j->off) == (ssize_t)j->pool->nbytes;
        if (t0 > 0) j->pread_s = salt_now_s() - t0;
    }
}

/* fallback worker for the no-pool path (one-shot spawn per batch) */
static void *fetch_worker_fb(void *p) {
    FetchJob *j = (FetchJob *)p;
    if (j->queued_s > 0) j->queue_wait_s = salt_now_s() - j->queued_s;
    perform_fetch(j);
    return NULL;
}

/* ---- persistent fetch pool ------------------------------------- */

static void *fetch_worker(void *p) {
    SaltCache *c = (SaltCache *)p;
    for (;;) {
        double lock0 = c->accounting ? salt_now_s() : 0;
        pthread_mutex_lock(&c->fmu);
        double lock_wait = lock0 > 0 ? salt_now_s() - lock0 : 0;
        while (!c->fshutdown && (!c->fbusy || c->fclaim >= c->fnjobs))
            pthread_cond_wait(&c->fcv_work, &c->fmu);
        if (c->fshutdown) { pthread_mutex_unlock(&c->fmu); return NULL; }
        int j = c->fclaim++;
        pthread_mutex_unlock(&c->fmu);

        FetchJob *job = &c->fjobs[j];
        job->lock_wait_s += lock_wait;
        if (job->queued_s > 0)
            job->queue_wait_s = salt_now_s() - job->queued_s;
        perform_fetch(job);

        lock0 = c->accounting ? salt_now_s() : 0;
        pthread_mutex_lock(&c->fmu);
        if (lock0 > 0)
            job->lock_wait_s += salt_now_s() - lock0;
        c->fdone++;
        if (c->fdone >= c->fnjobs) {
            /* Fetch completion ends worker ownership. The transaction still
             * holds call_mu through publication and descriptor reuse. */
            c->fbusy = 0;
            pthread_cond_broadcast(&c->fcv_done);
        }
        pthread_mutex_unlock(&c->fmu);
    }
}

static void fetch_pool_free(SaltCache *c);

static int fetch_pool_init(SaltCache *c) {
    c->fth = NULL;
    c->fcreated = 0;
    c->fsync_init = 0;
    c->fshutdown = 0;
    c->fbusy = 0;
    c->fclaim = 0;
    c->fdone = 0;
    c->fnjobs = 0;

    c->fcapacity = SALT_MAX_EXPERT_BATCH;
    c->fjobs = (FetchJob *)calloc(
        (size_t)c->fcapacity, sizeof *c->fjobs);
    if (!c->fjobs) return -1;
    if (c->fetch_device_fault_only) return 0;
    if (pthread_mutex_init(&c->fmu, NULL) != 0) {
        fetch_pool_free(c);
        return -1;
    }
    c->fsync_init = 1;
    if (pthread_cond_init(&c->fcv_work, NULL) != 0) {
        fetch_pool_free(c);
        return -1;
    }
    c->fsync_init = 2;
    if (pthread_cond_init(&c->fcv_done, NULL) != 0) {
        fetch_pool_free(c);
        return -1;
    }
    c->fsync_init = 3;
    c->fth = (pthread_t *)calloc((size_t)c->nthreads, sizeof(pthread_t));
    if (!c->fth) {
        fetch_pool_free(c);
        return -1;
    }
    for (int i = 0; i < c->nthreads; i++) {
        if (pthread_create(&c->fth[i], NULL, fetch_worker, c) != 0) {
            fetch_pool_free(c);
            return -1;
        }
        c->fcreated++;
    }
    return 0;
}

static void fetch_pool_free(SaltCache *c) {
    if (!c) return;
    if (c->fsync_init >= 1) {
        pthread_mutex_lock(&c->fmu);
        c->fshutdown = 1;
        if (c->fsync_init >= 2) pthread_cond_broadcast(&c->fcv_work);
        pthread_mutex_unlock(&c->fmu);
    }
    for (int i = 0; i < c->fcreated; i++) {
        int started = 1;
        salt_join_started_or_exit(&c->fth[i], &started, 1, pthread_join,
                                  "cache-fetch-pool");
    }
    free(c->fth);
    free(c->fjobs);
    c->fth = NULL;

    c->fjobs = NULL;
    c->fcapacity = 0;
    c->fcreated = 0;
    if (c->fsync_init >= 3) pthread_cond_destroy(&c->fcv_done);
    if (c->fsync_init >= 2) pthread_cond_destroy(&c->fcv_work);
    if (c->fsync_init >= 1) pthread_mutex_destroy(&c->fmu);
    c->fsync_init = 0;
}

static int cache_init_mode(SaltCache *c, SaltExpertPool *pool, int nslot,
                           int nthreads, int requested_mode) {
    if (!c) return -1;
    memset(c, 0, sizeof *c);
    c->munmap_fn = munmap;
    if (!pool || nslot < 1 || pool->nbytes < 1 || pool->n_layers < 1 ||
        pool->n_experts < 1 || !pool->ref)
        return -1;
    c->pool = pool;
    c->nslot = nslot;
    c->slot_bytes = pool->nbytes;
    c->nthreads = nthreads > 0 ? nthreads : 1;
    c->n_layers = pool->n_layers;
    c->n_experts = pool->n_experts;
    c->accounting = env_enabled("SALT_CACHE_ACCOUNTING") ||
                    env_enabled("SALT_ROOFLINE");
    c->check_enabled = env_enabled("SALT_CACHE_CHECK");
    c->page_probe_enabled = env_enabled("SALT_CACHE_PAGE_RESIDENCY");
    c->pretouch_enabled = env_enabled("SALT_CACHE_PRETOUCH");
    {
        const char *touch = getenv("SALT_FETCH_TOUCH_BYTES");
        int touch_on_miss = env_enabled("SALT_FETCH_TOUCH");
        if (touch && *touch) {
            char *end = NULL;
            unsigned long long value;
            if (*touch == '-') return -1;
            value = strtoull(touch, &end, 10);
            if (!end || end == touch || *end != '\0' ||
                value > (unsigned long long)c->slot_bytes || value > INT64_MAX)
                return -1;
            if (value == 0u)
                c->fetch_device_fault_only = 1;
            else if (touch_on_miss)
                c->fetch_touch_bytes = (int64_t)value;
        } else if (touch_on_miss) {
            c->fetch_touch_bytes = c->slot_bytes;
        }
    }
    if ((size_t)c->n_layers > (size_t)-1 / (size_t)c->n_experts)
        return -1;
    size_t nresident = (size_t)c->n_layers * (size_t)c->n_experts;
    if (nresident > (size_t)-1 / sizeof(int) ||
        nresident > (size_t)-1 / sizeof(uint64_t))
        return -1;
    /* TWO MODES (SALT_CACHE_MODE, default "arena" = backwards
     * compatible):
     *   arena    -- always allocate the slot arena; workers pread into
     *               it (the original path). Larger footprint (~cache-gb)
     *               but the fastest fetch (one batched read/expert).
     *   zerocopy -- pool->map valid: skip the arena entirely, payloads
     *               point into the mmap; fill = readahead+touch;
     *               eviction flushes pages (DONTNEED). Footprint drops
     *               by the full cache size (1 GB at cache-gb 1,
     *               ~1.9 GB at cache-gb 2); fetch trades the copy for
     *               the fault/touch path. */
    {
        /* THREE MODES (SALT_CACHE_MODE, default "arena"):
         *   arena    -- pread into arena copies everywhere (default;
         *               the memo cache: resident + fast decode)
         *   zerocopy -- payloads point into the pool mmap; fill =
         *               readahead+touch (faster fetch, reclaimable)
         *   mixture  -- arena for hot/keep experts + zerocopy for
         *               fresh fetches (SALT_MIX_HITS threshold) */
        const char *cm = requested_mode < 0 ? getenv("SALT_CACHE_MODE") : NULL;
        int mode = requested_mode < 0 ? 0 : requested_mode;
        if (requested_mode < 0 && cm && strcmp(cm, "zerocopy") == 0) mode = 1;
        else if (requested_mode < 0 && cm && strcmp(cm, "mixture") == 0)
            mode = 2;
        if (mode < 0 || mode > 3) return -1;
        if ((mode == 1 || mode == 2) && !pool->map) {
            fprintf(stderr,
                    "cache: %s mode requires a mapped expert pool\n",
                    mode == 1 ? "zerocopy" : "mixture");
            return -1;
        }
        if (mode == 3 && pool->fd < 0) {
            fprintf(stderr,
                    "cache: bounded-mmap mode requires an expert-pool fd\n");
            return -1;
        }
        c->mode = mode;
        /* arena is needed in arena (all copies) and mixture (keep
         * set); zerocopy needs the pool mmap. */
        if (mode == 0 || mode == 2)
            c->arena = (uint8_t *)calloc((size_t)nslot, (size_t)pool->nbytes);
    }
    c->payload = (const uint8_t **)calloc((size_t)nslot,
                                          sizeof(uint8_t *));
    c->state  = (uint8_t *)calloc((size_t)nslot, 1);
    c->key    = (uint64_t *)calloc((size_t)nslot, sizeof(uint64_t));
    c->slot_home = (int *)malloc((size_t)nslot * sizeof(int));
    c->layer_quota = (int *)calloc((size_t)c->n_layers, sizeof(int));
    c->layer_stats = (SaltCacheLayerStats *)calloc(
        (size_t)c->n_layers, sizeof(SaltCacheLayerStats));
    c->resident_slot = (int *)malloc(nresident * sizeof(int));
    c->demand = (uint64_t *)calloc(nresident, sizeof(uint64_t));
    c->used_at = (uint64_t *)calloc((size_t)nslot, sizeof(uint64_t));
    c->hits   = (uint64_t *)calloc((size_t)nslot, sizeof(uint64_t));
    c->pinned = (uint8_t *)calloc((size_t)nslot, 1);
    c->lease_count = (uint32_t *)calloc((size_t)nslot, sizeof(uint32_t));
    c->resource_bound = (uint8_t *)calloc((size_t)nslot, 1);
    if (c->mode == 3) {
        c->slot_map_base = (void **)calloc((size_t)nslot, sizeof(void *));
        c->slot_map_len = (size_t *)calloc((size_t)nslot, sizeof(size_t));
    }
    if (((c->mode == 0 || c->mode == 2) && !c->arena) ||
        (c->mode == 3 && (!c->slot_map_base || !c->slot_map_len)) ||
        !c->payload || !c->state || !c->key ||
        !c->slot_home || !c->layer_quota || !c->layer_stats ||
        !c->resident_slot || !c->demand || !c->used_at || !c->hits ||
        !c->pinned || !c->lease_count || !c->resource_bound) {
        salt_cache_free(c);
        return -1;
    }
    for (int slot = 0; slot < nslot; slot++) c->slot_home[slot] = -1;
    for (size_t i = 0; i < nresident; i++) c->resident_slot[i] = -1;
    if (pthread_mutex_init(&c->call_mu, NULL) != 0) {
        salt_cache_free(c);
        return -1;
    }
    c->call_mu_init = 1;
    if (quota_plan_init(c) != 0 ||
        (c->check_enabled && salt_cache_check(c, stderr) != 0) ||
        fetch_pool_init(c) != 0) {
        salt_cache_free(c);
        return -1;
    }
    return 0;
}

int salt_cache_startup_metadata_bytes(int nslot, int n_layers, int n_experts,
                                      int nthreads, int bounded_mmap,
                                      uint64_t *bytes_out) {
    uint64_t total = 0, nresident;
    if (!bytes_out || nslot < 1 || n_layers < 1 || n_experts < 1 ||
        nthreads < 1 || (bounded_mmap != 0 && bounded_mmap != 1) ||
        (uint64_t)(uint32_t)n_layers >
            UINT64_MAX / (uint64_t)(uint32_t)n_experts)
        return -1;
    nresident = (uint64_t)(uint32_t)n_layers * (uint32_t)n_experts;
#define ADD_ARRAY(count, type) do { \
    uint64_t add_count = (uint64_t)(count); \
    uint64_t add_size = (uint64_t)sizeof(type); \
    if (add_count > UINT64_MAX / add_size || \
        add_count * add_size > UINT64_MAX - total) return -1; \
    total += add_count * add_size; \
} while (0)
    ADD_ARRAY((uint32_t)nslot, const uint8_t *);
    ADD_ARRAY((uint32_t)nslot, uint8_t);
    ADD_ARRAY((uint32_t)nslot, uint64_t);
    ADD_ARRAY((uint32_t)nslot, int);
    ADD_ARRAY((uint32_t)nslot, uint64_t);
    ADD_ARRAY((uint32_t)nslot, uint64_t);
    ADD_ARRAY((uint32_t)nslot, uint8_t);
    ADD_ARRAY((uint32_t)nslot, uint32_t);
    ADD_ARRAY((uint32_t)nslot, uint8_t);
    if (bounded_mmap) {
        ADD_ARRAY((uint32_t)nslot, void *);
        ADD_ARRAY((uint32_t)nslot, size_t);
    }
    ADD_ARRAY((uint32_t)n_layers, int);
    ADD_ARRAY((uint32_t)n_layers, SaltCacheLayerStats);
    ADD_ARRAY(nresident, int);
    ADD_ARRAY(nresident, uint64_t);
    ADD_ARRAY((uint32_t)nthreads, pthread_t);
    ADD_ARRAY(SALT_MAX_EXPERT_BATCH, FetchJob);
#undef ADD_ARRAY
    *bytes_out = total;
    return 0;
}

int salt_cache_init(SaltCache *c, SaltExpertPool *pool, int nslot,
                    int nthreads) {
    return cache_init_mode(c, pool, nslot, nthreads, -1);
}

int salt_cache_init_zerocopy(SaltCache *c, SaltExpertPool *pool, int nslot,
                             int nthreads) {
    return cache_init_mode(c, pool, nslot, nthreads, 1);
}

int salt_cache_init_bounded_mmap(SaltCache *c, SaltExpertPool *pool, int nslot,
                                 int nthreads) {
    return cache_init_mode(c, pool, nslot, nthreads, 3);
}

int salt_cache_set_proof_state(SaltCache *c, int enabled) {
    if (!c || !c->call_mu_init || (enabled != 0 && enabled != 1)) return -1;
    if (pthread_mutex_lock(&c->call_mu) != 0) return -1;
    c->proof_state = enabled;
    pthread_mutex_unlock(&c->call_mu);
    return 0;
}

int salt_cache_set_resource_hooks(SaltCache *c,
                                  SaltCacheResourceFenceFn fence_fn,
                                  SaltCacheResourceBindFn bind_fn,
                                  SaltCacheResourceUnbindFn unbind_fn,
                                  void *ctx) {
    int rc = 0;
    if (!c || !c->call_mu_init ||
        (c->mode != 1 && c->mode != 3) ||
        !bind_fn || !unbind_fn || !c->resource_bound)
        return -1;
    if (pthread_mutex_lock(&c->call_mu) != 0) return -1;
    if (c->resource_bind_fn || c->resource_unbind_fn || c->resource_ctx) {
        pthread_mutex_unlock(&c->call_mu);
        return -1;
    }
    c->resource_bind_fn = bind_fn;
    c->resource_fence_fn = fence_fn;
    c->resource_unbind_fn = unbind_fn;
    c->resource_ctx = ctx;
    for (int slot = 0; slot < c->nslot; slot++) {
        const void *base;
        const void *payload;
        size_t nbytes;
        if (c->state[slot] != SALT_SLOT_READY) continue;
        payload = c->payload[slot];
        if (c->mode == 3) {
            if (!c->slot_map_base || !c->slot_map_len ||
                !c->slot_map_base[slot] || c->slot_map_len[slot] == 0) {
                rc = -1;
                break;
            }
            base = c->slot_map_base[slot];
            nbytes = c->slot_map_len[slot];
        } else {
            if (!payload || c->slot_bytes < 1) {
                rc = -1;
                break;
            }
            base = payload;
            nbytes = (size_t)c->slot_bytes;
        }
        if (bind_fn(ctx, slot, key_layer(c->key[slot]),
                    key_expert(c->key[slot]), base, nbytes, payload) != 0) {
            rc = -1;
            break;
        }
        c->resource_bound[slot] = 1;
    }
    if (rc != 0) {
        for (int slot = 0; slot < c->nslot; slot++) {
            if (!c->resource_bound[slot]) continue;
            if (unbind_fn(ctx, slot, key_layer(c->key[slot]),
                    key_expert(c->key[slot])) != 0) break;
            c->resource_bound[slot] = 0;
        }
        int any_bound = 0;
        for (int slot = 0; slot < c->nslot; slot++)
            any_bound |= c->resource_bound[slot] != 0;
        if (!any_bound) {
            c->resource_bind_fn = NULL;
            c->resource_fence_fn = NULL;
            c->resource_unbind_fn = NULL;
            c->resource_ctx = NULL;
        }
    }
    pthread_mutex_unlock(&c->call_mu);
    return rc;
}

void salt_cache_free(SaltCache *c) {
    if (!c) return;
    fetch_pool_free(c);
    if (c->call_mu_init) {
        pthread_mutex_destroy(&c->call_mu);
        c->call_mu_init = 0;
    }
    if (c->resource_bound) {
        int any_bound = 0;
        for (int slot = 0; slot < c->nslot; slot++)
            any_bound |= c->resource_bound[slot] != 0;
        if (any_bound && c->resource_fence_fn &&
            c->resource_fence_fn(c->resource_ctx) != 0) {
            fprintf(stderr,
                    "cache: backend resource fence failed during final ownership release\n");
            fflush(stderr);
            _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
        }
        for (int slot = 0; slot < c->nslot; slot++)
            if (c->resource_bound[slot] &&
                (!c->resource_unbind_fn ||
                 c->resource_unbind_fn(c->resource_ctx, slot,
                     key_layer(c->key[slot]),
                     key_expert(c->key[slot])) != 0)) {
                fprintf(stderr,
                        "cache: backend resource unbind failed during final ownership release\n");
                fflush(stderr);
                _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
            }
    }
    if (c->slot_map_base && c->slot_map_len) {
        for (int slot = 0; slot < c->nslot; slot++)
            if (c->slot_map_base[slot] &&
                (!c->munmap_fn ||
                 c->munmap_fn(c->slot_map_base[slot],
                              c->slot_map_len[slot]) != 0)) {
                fprintf(stderr,
                        "cache: munmap failed during final ownership release\n");
                fflush(stderr);
                _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
            }
    }
    free(c->arena);
    free(c->payload);
    free(c->slot_map_base);
    free(c->slot_map_len);
    free(c->state);
    free(c->key);
    free(c->resident_slot);
    free(c->demand);
    free(c->slot_home);
    free(c->layer_quota);
    free(c->layer_stats);
    free(c->used_at);
    free(c->hits);
    free(c->pinned);
    free(c->lease_count);
    free(c->resource_bound);
    memset(c, 0, sizeof *c);
}

static int cache_getmany_locked(SaltCache *c, int layer,
                                const int *experts, int n, int *out_slot) {
    if (!c || !experts || !out_slot || layer < 0 || layer >= c->n_layers ||
        n < 0)
        return -1;
    if (n > SALT_MAX_EXPERT_BATCH || !c->fjobs || c->fcapacity < n)
        return -1;
    int fetch_profile = env_enabled("SALT_FETCH_MS");
    int profile = fetch_profile || c->accounting ||
                  c->page_probe_enabled || c->pretouch_enabled;
    double _g0 = profile ? salt_now_s() : 0;
    static double _ga = 0, _pa = 0, _la = 0;
    static long _gc = 0, _pc = 0, _lc = 0;
    double _p0 = 0, _pt = 0;

    /* phase 1: reserve serially, so no two experts take the same slot */
    FetchJob *jobs = c->fjobs;
    int njob = 0;
    int failed = 0;
    int resources_fenced = 0;
    for (int j = 0; j < n; j++) out_slot[j] = -1;
    c->nreq += n;
    SaltCacheLayerStats *lst = &c->layer_stats[layer];
    lst->requests += n;
    double lookup0 = profile ? salt_now_s() : 0;

    for (int j = 0; j < n; j++) {
        int expert = experts[j];
        if (expert < 0 || expert >= c->n_experts) {
            c->ndrop++;
            failed = 1;
            continue;
        }
        size_t demand_i = resident_index(c, layer, expert);
        if (c->demand[demand_i] != UINT64_MAX) c->demand[demand_i]++;
        int s = cache_lookup(c, layer, expert);
        if (s == -2) {
            c->ndrop++;
            failed = 1;
            continue;
        }
        if (s >= 0) {
            out_slot[j] = s;
            if (c->state[s] == SALT_SLOT_LOADING) {
                c->ninflight_join++;
                lst->inflight_joins++;
                continue;
            }
            c->nhit++;
            lst->ready_hits++;
            c->hits[s]++;
            c->used_at[s] = ++c->clock;
            promote_shared_hit(c, s, layer, 0);
            /* MIXTURE: promote a reused expert to the arena (memo).
             * Once an mmap-backed slot crosses SALT_MIX_HITS, copy it
             * into the resident arena so repeated decode hits never
             * depend on the OS page cache (fewer reads, faster). */
            if (c->mode == 2 && c->hits[s] >= SALT_MIX_HITS &&
                c->payload[s] >= c->pool->map &&
                c->payload[s] + (size_t)c->slot_bytes <=
                    c->pool->map + c->pool->map_len) {
                uint8_t *dst = c->arena + (int64_t)s * c->slot_bytes;
                double copy0 = profile ? salt_now_s() : 0;
                memcpy(dst, c->payload[s], (size_t)c->slot_bytes);
                if (copy0 > 0) {
                    double copy_s = salt_now_s() - copy0;
                    c->touch_s += copy_s;
                    lst->touch_s += copy_s;
                }
                c->payload[s] = dst;
            }
            continue;
        }
        c->nmiss++;
        lst->misses++;
        /* Every slot returned for an earlier element is transaction-leased.
         * A later miss must not recycle it and make two successful outputs
         * silently name the later expert. LOADING reservations are protected
         * independently by pick_victim(). */
        s = pick_victim(c, layer, out_slot, j);
        if (s < 0) {
            c->ndrop++;
            if (c->quota_enabled) {
                c->nquota_denial++;
                lst->quota_denials++;
            } else {
                c->nadmission_denial++;
            }
            failed = 1;
            continue;
        }
        /* transport for THIS slot: pure modes fix it; mixture starts
         * every fill zerocopy (fast fetch), promotion to arena happens
         * on reuse (above). */
        int zc;
        if (c->mode == 1) zc = 1;                       /* flat zerocopy */
        else if (c->mode == 2) zc = 1;                  /* mixture: fresh */
        else if (c->mode == 3) zc = 1;                  /* bounded mmap */
        else zc = 0;                                    /* arena */
        if (c->state[s] != SALT_SLOT_EMPTY && c->resource_bound &&
            c->resource_bound[s] && c->resource_fence_fn &&
            !resources_fenced) {
            if (c->resource_fence_fn(c->resource_ctx) != 0) {
                c->ndrop++;
                failed = 1;
                continue;
            }
            resources_fenced = 1;
        }
        if (c->state[s] != SALT_SLOT_EMPTY && uncharge_slot(c, s, 1) != 0) {
            c->ndrop++;
            failed = 1;
            continue;
        }
        charge_loading(c, s, layer, expert);
        memset(&jobs[njob], 0, sizeof jobs[njob]);
        jobs[njob].slot = s;
        jobs[njob].layer = layer;
        jobs[njob].expert = expert;
        jobs[njob].off = c->pool->ref[(size_t)layer * c->pool->n_experts +
                                      expert].off;
        /* Prepare the configured mapped prefix on true misses, independent
         * of request proof; READY hits never create fetch jobs. */
        jobs[njob].touch_bytes = zc ? c->fetch_touch_bytes : 0;
        jobs[njob].zc = zc;
        jobs[njob].slotmap = c->mode == 3;
        jobs[njob].pretouch = !zc && c->pretouch_enabled;
        jobs[njob].dst = zc ? NULL
            : c->arena + (int64_t)s * c->slot_bytes;
        jobs[njob].pool = c->pool;
        if (c->page_probe_enabled) {
            int64_t resident = 0, total = 0;
            double probe_s = 0;
            if (source_pages_probe(c, jobs[njob].off, &resident, &total,
                                   &probe_s) == 0) {
                lst->source_probe_calls++;
                lst->source_pages += total;
                lst->source_resident_pages += resident;
                lst->source_fully_resident += resident == total;
                lst->source_probe_s += probe_s;
            }
        }
        out_slot[j] = s;
        njob++;
    }
    if (lookup0 > 0) {
        double t = salt_now_s() - lookup0;
        c->lookup_s += t;
        lst->lookup_s += t;
    }

    if (njob > 0) {
        int device_fault_only = c->mode == 1 &&
            c->fetch_device_fault_only && c->pool->map;
        /* phase 2: drain the fixed startup-owned descriptor array in
         * disk-offset order through the persistent workers. */
        if (device_fault_only) {
            for (int i = 0; i < njob; i++) {
                FetchJob *job = &jobs[i];
                if (job->off < 0 ||
                    (size_t)job->off > c->pool->map_len ||
                    (size_t)c->slot_bytes >
                        c->pool->map_len - (size_t)job->off) {
                    job->ok = 0;
                } else {
                    job->dst = c->pool->map + (size_t)job->off;
                    job->ok = 1;
                }
            }
        } else {
            for (int i = 1; i < njob; i++) {
                FetchJob t = jobs[i];
                int k = i - 1;
                while (k >= 0 && jobs[k].off > t.off) {
                    jobs[k + 1] = jobs[k];
                    k--;
                }
                jobs[k + 1] = t;
            }
            _p0 = profile ? salt_now_s() : 0;
            double queued = profile ? salt_now_s() : 0;
            for (int i = 0; i < njob; i++) jobs[i].queued_s = queued;
            c->nfetch_jobs += njob;
            if (c->fth && c->fcreated > 0) {
                double lock0 = profile ? salt_now_s() : 0;
                pthread_mutex_lock(&c->fmu);
                double caller_lock = lock0 > 0 ? salt_now_s() - lock0 : 0;
                c->fnjobs = njob;
                c->fclaim = 0;
                c->fdone = 0;
                c->fbusy = 1;
                pthread_cond_broadcast(&c->fcv_work);
                while (c->fdone < c->fnjobs)
                    pthread_cond_wait(&c->fcv_done, &c->fmu);
                pthread_mutex_unlock(&c->fmu);
                c->lock_wait_s += caller_lock;
                lst->lock_wait_s += caller_lock;
            } else {
                /* One-worker configuration remains allocation- and spawn-free. */
                for (int i = 0; i < njob; i++) fetch_worker_fb(&jobs[i]);
            }
            if (_p0 > 0) {
                _pt = salt_now_s() - _p0;
                c->fetch_wait_s += _pt;
                lst->barrier_s += _pt;
            }
        }
        for (int i = 0; i < njob; i++) {
            FetchJob *job = &jobs[i];
            c->queue_wait_s += job->queue_wait_s;
            c->lock_wait_s += job->lock_wait_s;
            c->pread_s += job->pread_s;
            c->readahead_s += job->readahead_s;
            c->touch_s += job->touch_s;
            if (job->pretouch) {
                c->npretouch_calls++;
                lst->pretouch_calls++;
            }
            lst->queue_wait_s += job->queue_wait_s;
            lst->lock_wait_s += job->lock_wait_s;
            lst->pread_s += job->pread_s;
            lst->readahead_s += job->readahead_s;
            lst->touch_s += job->touch_s;
            if (job->zc) {
                if (!device_fault_only) lst->readahead_calls++;
            } else {
                lst->pread_calls++;
                if (job->ok) lst->pread_bytes += c->slot_bytes;
            }
        }
    }

    /* phase 3: publish only what arrived */
    double publish0 = profile ? salt_now_s() : 0;
    for (int i = 0; i < njob; i++) {
        FetchJob *j = &jobs[i];
        uint64_t expected = expert_key(j->layer, j->expert);
        int publish_ok = j->ok &&
            c->state[j->slot] == SALT_SLOT_LOADING &&
            c->key[j->slot] == expected;
        if (publish_ok && j->zc && c->resource_bind_fn) {
            const void *bind_base = j->map_base;
            const void *bind_payload = j->dst;
            size_t bind_bytes = j->map_len;
            if (!j->slotmap) {
                if (!c->pool->map || j->off < 0 ||
                    (size_t)j->off > c->pool->map_len ||
                    (size_t)c->slot_bytes >
                        c->pool->map_len - (size_t)j->off) {
                    publish_ok = 0;
                } else {
                    bind_payload = c->pool->map + (size_t)j->off;
                    bind_base = bind_payload;
                    bind_bytes = (size_t)c->slot_bytes;
                }
            }
            if (publish_ok)
                publish_ok = c->resource_bind_fn(c->resource_ctx, j->slot,
                    j->layer, j->expert, bind_base, bind_bytes,
                    bind_payload) == 0;
            if (publish_ok) c->resource_bound[j->slot] = 1;
        }
        if (publish_ok) {
            c->state[j->slot] = SALT_SLOT_READY;
            c->used_at[j->slot] = ++c->clock;
            if (j->slotmap) {
                c->slot_map_base[j->slot] = j->map_base;
                c->slot_map_len[j->slot] = j->map_len;
                c->payload[j->slot] = j->dst;
                c->mapped_current_bytes += (int64_t)j->map_len;
                if (c->mapped_current_bytes > c->mapped_peak_bytes)
                    c->mapped_peak_bytes = c->mapped_current_bytes;
                j->map_base = NULL;
                j->map_len = 0;
            } else if (j->zc)
                c->payload[j->slot] = c->pool->map + (size_t)j->off;
            else
                c->payload[j->slot] = c->arena +
                    (int64_t)j->slot * c->slot_bytes;
            c->nread += c->slot_bytes;
            c->ninsert++;
            c->layer_stats[j->layer].inserts++;
            promote_shared_hit(c, j->slot, j->layer, 0);
        } else {
            if (j->map_base) {
                if (!c->munmap_fn ||
                    c->munmap_fn(j->map_base, j->map_len) != 0) {
                    fprintf(stderr,
                            "cache: munmap failed for unpublished mapping\n");
                    fflush(stderr);
                    _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
                }
                j->map_base = NULL;
                j->map_len = 0;
            }
            if (uncharge_slot(c, j->slot, 0) != 0)
                c->ncheck_fail++;
            c->ndrop++;
            failed = 1;
        }
    }
    if (publish0 > 0) {
        double t = salt_now_s() - publish0;
        c->publish_s += t;
        lst->publish_s += t;
    }
    /* Enforce the public success contract at the transaction boundary: every
     * output must still be READY for the exact requested key. */
    if (!failed) {
        for (int j = 0; j < n; j++) {
            int s = cache_lookup(c, layer, experts[j]);
            if (s != out_slot[j] || s < 0 ||
                c->state[s] != SALT_SLOT_READY) {
                c->ndrop++;
                failed = 1;
                break;
            }
        }
    }
    if (c->check_enabled && salt_cache_check(c, stderr) != 0) {
        c->ncheck_fail++;
        failed = 1;
    }
    if (fetch_profile) {
        double _gt = salt_now_s() - _g0;
        _ga += _gt; _pa += _pt; _la += _gt - _pt;
        _gc++; _pc += _pt > 0; _lc += _pt > 0;
        if (_gc <= 5 || _gc % 80 == 0)
            fprintf(stderr, "[fetch] n=%d total %.3f ms (avg %.3f) "
                    "read %.3f ms (avg %.3f) lookup+flush %.3f ms "
                    "(avg %.3f) -- %ld calls\n",
                    n, _gt * 1e3, _ga / _gc * 1e3,
                    _pt * 1e3, (_pc ? _pa / _pc * 1e3 : 0),
                    (_gt - _pt) * 1e3,
                    (_lc ? _la / _lc * 1e3 : 0), _gc);
    }
    return failed ? -1 : njob;
}

int salt_cache_getmany(SaltCache *c, int layer, const int *experts, int n,
                       int *out_slot) {
    if (!c || !c->call_mu_init) return -1;
    /* No asynchronous preparation survives this transaction. Its existing
     * owner lock covers lookup, fetch completion, publication, and reuse. */
    if (pthread_mutex_lock(&c->call_mu) != 0) return -1;
    int rc = cache_getmany_locked(c, layer, experts, n, out_slot);
    pthread_mutex_unlock(&c->call_mu);
    return rc;
}

const uint8_t *salt_cache_slot(const SaltCache *c, int slot) {
    if (!c || slot < 0 || slot >= c->nslot ||
        c->state[slot] != SALT_SLOT_READY)
        return NULL;
    return c->payload[slot];
}

const uint8_t *salt_cache_slot_for(const SaltCache *c, int slot,
                                   int layer, int expert) {
    if (!c || slot < 0 || slot >= c->nslot) return NULL;
    int resident = cache_lookup(c, layer, expert);
    if (resident != slot || c->state[slot] != SALT_SLOT_READY) return NULL;
    return c->payload[slot];
}

const uint8_t *salt_cache_acquire(SaltCache *c, int slot,
                                  int layer, int expert) {
    const uint8_t *payload = NULL;
    if (!c || !c->call_mu_init || slot < 0 || slot >= c->nslot)
        return NULL;
    if (pthread_mutex_lock(&c->call_mu) != 0) return NULL;
    if (c->lease_count && c->lease_count[slot] != UINT32_MAX &&
        cache_lookup(c, layer, expert) == slot &&
        c->state[slot] == SALT_SLOT_READY) {
        c->lease_count[slot]++;
        payload = c->payload[slot];
    }
    pthread_mutex_unlock(&c->call_mu);
    return payload;
}

int salt_cache_release(SaltCache *c, int slot, int layer, int expert) {
    int rc = -1;
    if (!c || !c->call_mu_init || slot < 0 || slot >= c->nslot)
        return -1;
    if (pthread_mutex_lock(&c->call_mu) != 0) return -1;
    if (c->lease_count && c->lease_count[slot] != 0 &&
        cache_lookup(c, layer, expert) == slot &&
        c->state[slot] == SALT_SLOT_READY) {
        c->lease_count[slot]--;
        rc = 0;
    }
    pthread_mutex_unlock(&c->call_mu);
    return rc;
}

int salt_cache_acquire_many(SaltCache *c, const int *slots, int layer,
                            const int *experts, int n,
                            const uint8_t **payloads) {
    int rc = -1;
    if (!c || !c->call_mu_init || !slots || !experts || !payloads ||
        n < 1 || n > c->nslot)
        return -1;
    for (int index = 0; index < n; index++) payloads[index] = NULL;
    if (pthread_mutex_lock(&c->call_mu) != 0) return -1;
    for (int index = 0; index < n; index++) {
        int slot = slots[index];
        if (slot < 0 || slot >= c->nslot || !c->lease_count ||
            c->lease_count[slot] == UINT32_MAX ||
            cache_lookup(c, layer, experts[index]) != slot ||
            c->state[slot] != SALT_SLOT_READY || !c->payload[slot])
            goto done;
        for (int prior = 0; prior < index; prior++)
            if (slots[prior] == slot) goto done;
    }
    for (int index = 0; index < n; index++) {
        c->lease_count[slots[index]]++;
        payloads[index] = c->payload[slots[index]];
    }
    rc = 0;
done:
    pthread_mutex_unlock(&c->call_mu);
    return rc;
}


int salt_cache_release_many(SaltCache *c, const int *slots, int layer,
                            const int *experts, int n) {
    int rc = -1;
    if (!c || !c->call_mu_init || !slots || !experts ||
        n < 1 || n > c->nslot)
        return -1;
    if (pthread_mutex_lock(&c->call_mu) != 0) return -1;
    for (int index = 0; index < n; index++) {
        int slot = slots[index];
        if (slot < 0 || slot >= c->nslot || !c->lease_count ||
            c->lease_count[slot] == 0u ||
            cache_lookup(c, layer, experts[index]) != slot ||
            c->state[slot] != SALT_SLOT_READY)
            goto done;
        for (int prior = 0; prior < index; prior++)
            if (slots[prior] == slot) goto done;
    }
    for (int index = 0; index < n; index++)
        c->lease_count[slots[index]]--;
    rc = 0;
done:
    pthread_mutex_unlock(&c->call_mu);
    return rc;
}

void salt_cache_pin(SaltCache *c, int slot, int pin) {
    if (!c || slot < 0 || slot >= c->nslot) return;
    if (!pin) { c->pinned[slot] = 0; return; }
    if (c->state[slot] != SALT_SLOT_READY) return;
    if (c->pinned[slot]) return;
    int layer = key_layer(c->key[slot]);
    if (c->quota_enabled && c->slot_home[slot] < 0) {
        promote_shared_hit(c, slot, layer, 1);
        if (c->slot_home[slot] < 0) {
            c->npin_refusal++;
            return;
        }
    } else if (!c->quota_enabled) {
        int reserve = c->n_experts < c->nslot ? c->n_experts : c->nslot;
        if (salt_cache_npinned(c) >= c->nslot - reserve) {
            c->npin_refusal++;
            return;
        }
    }
    c->pinned[slot] = 1;
}

/* L2 (read-head): pin the slot currently holding (layer, expert) so it
 * is never evicted. Returns 1 if a resident slot was pinned, 0 if the
 * expert is not cached (caller may fetch-then-pin). */
int salt_cache_pin_expert(SaltCache *c, int layer, int expert) {
    int s = cache_lookup(c, layer, expert);
    if (s < 0 || c->state[s] != SALT_SLOT_READY) return 0;
    salt_cache_pin(c, s, 1);
    return c->pinned[s] ? 1 : 0;
}

/* return the number of currently pinned slots (diagnostics) */
int salt_cache_npinned(const SaltCache *c) {
    if (!c) return 0;
    int n = 0;
    for (int i = 0; i < c->nslot; i++)
        if (c->pinned[i]) n++;
    return n;
}

int salt_cache_check(const SaltCache *c, FILE *out) {
    int errors = 0;
#define CACHE_BAD(...) do { errors++; if (out) fprintf(out, __VA_ARGS__); } while (0)
    if (!c || c->nslot < 1 || c->n_layers < 1 || c->n_experts < 1 ||
        !c->state || !c->key || !c->payload || !c->resident_slot || !c->demand ||
        !c->slot_home || !c->layer_quota || !c->layer_stats ||
        !c->lease_count || !c->munmap_fn) {
        CACHE_BAD("cache-check: missing cache metadata\n");
        return -1;
    }
    size_t nresident = (size_t)c->n_layers * (size_t)c->n_experts;
    uint8_t *seen = (uint8_t *)calloc(nresident, 1);
    int *home = (int *)calloc((size_t)c->n_layers, sizeof(int));
    int *shared = (int *)calloc((size_t)c->n_layers, sizeof(int));
    int *home_slots = (int *)calloc((size_t)c->n_layers, sizeof(int));
    if (!seen || !home || !shared || !home_slots) {
        free(seen); free(home); free(shared); free(home_slots);
        CACHE_BAD("cache-check: allocation failed\n");
        return -1;
    }

    int quota_sum = 0;
    for (int layer = 0; layer < c->n_layers; layer++) {
        if (c->layer_quota[layer] < 0)
            CACHE_BAD("cache-check: negative quota at layer %d\n", layer);
        quota_sum += c->layer_quota[layer];
    }
    if (quota_sum != c->reserved_slots ||
        quota_sum + c->shared_slots != c->nslot)
        CACHE_BAD("cache-check: quota sum %d + shared %d != slots %d\n",
                  quota_sum, c->shared_slots, c->nslot);
    if (c->quota_enabled) {
        int active_need = c->n_experts < c->nslot
            ? c->n_experts : c->nslot;
        if (c->shared_slots < active_need)
            CACHE_BAD("cache-check: shared pool %d smaller than required %d\n",
                      c->shared_slots, active_need);
    }

    int occupied = 0, shared_total = 0;
    int64_t mapped_bytes = 0;
    for (int slot = 0; slot < c->nslot; slot++) {
        int owner = c->slot_home[slot];
        if (owner >= c->n_layers || owner < -1) {
            CACHE_BAD("cache-check: slot %d has invalid home %d\n", slot, owner);
            continue;
        }
        if (owner >= 0) home_slots[owner]++;
        if (c->state[slot] == SALT_SLOT_EMPTY) {
            if (c->key[slot] != 0 || c->payload[slot] != NULL ||
                c->pinned[slot] ||
                c->lease_count[slot] != 0 ||
                (c->slot_map_base && c->slot_map_base[slot]))
                CACHE_BAD("cache-check: dirty empty slot %d\n", slot);
            continue;
        }
        if (c->state[slot] != SALT_SLOT_LOADING &&
            c->state[slot] != SALT_SLOT_READY) {
            CACHE_BAD("cache-check: slot %d has invalid state %u\n",
                      slot, (unsigned)c->state[slot]);
            continue;
        }
        occupied++;
        int layer = key_layer(c->key[slot]);
        int expert = key_expert(c->key[slot]);
        if (layer < 0 || layer >= c->n_layers ||
            expert < 0 || expert >= c->n_experts) {
            CACHE_BAD("cache-check: slot %d has invalid key\n", slot);
            continue;
        }
        size_t ri = resident_index(c, layer, expert);
        if (seen[ri])
            CACHE_BAD("cache-check: duplicate key L%d E%d\n", layer, expert);
        seen[ri] = 1;
        if (c->resident_slot[ri] != slot)
            CACHE_BAD("cache-check: stale index L%d E%d\n", layer, expert);
        if (owner >= 0 && owner != layer)
            CACHE_BAD("cache-check: slot %d home L%d holds L%d\n",
                      slot, owner, layer);
        if (owner == layer) home[layer]++;
        else { shared[layer]++; shared_total++; }
        if (c->state[slot] == SALT_SLOT_LOADING) {
            if (c->payload[slot] != NULL || c->pinned[slot] ||
                c->lease_count[slot] != 0)
                CACHE_BAD("cache-check: exposed loading slot %d\n", slot);
        } else if (!c->payload[slot]) {
            CACHE_BAD("cache-check: ready slot %d has no payload\n", slot);
        }
        if (c->mode == 3 && c->state[slot] == SALT_SLOT_READY &&
            (!c->slot_map_base || !c->slot_map_len ||
             !c->slot_map_base[slot] || c->slot_map_len[slot] == 0))
            CACHE_BAD("cache-check: bounded-mmap slot %d has no mapping\n", slot);
        if (c->mode == 3 && c->slot_map_len)
            mapped_bytes += (int64_t)c->slot_map_len[slot];
    }
    for (size_t ri = 0; ri < nresident; ri++) {
        int slot = c->resident_slot[ri];
        if (slot < -1 || slot >= c->nslot)
            CACHE_BAD("cache-check: invalid resident slot %d at %zu\n",
                      slot, ri);
        if ((slot < 0) != (seen[ri] == 0))
            CACHE_BAD("cache-check: residency presence mismatch at %zu\n", ri);
    }
    for (int layer = 0; layer < c->n_layers; layer++) {
        const SaltCacheLayerStats *st = &c->layer_stats[layer];
        if (home_slots[layer] != c->layer_quota[layer])
            CACHE_BAD("cache-check: layer %d owns %d quota slots, expected %d\n",
                      layer, home_slots[layer], c->layer_quota[layer]);
        if (home[layer] != st->home_current ||
            shared[layer] != st->shared_current)
            CACHE_BAD("cache-check: layer %d occupancy mismatch\n", layer);
        if (home[layer] > c->layer_quota[layer] ||
            st->home_peak > c->layer_quota[layer])
            CACHE_BAD("cache-check: layer %d exceeded quota\n", layer);
        if (st->home_current < 0 || st->shared_current < 0 ||
            st->borrow_returns > st->borrow_acquires)
            CACHE_BAD("cache-check: layer %d counter underflow\n", layer);
    }
    if (occupied > c->nslot || shared_total > c->shared_slots)
        CACHE_BAD("cache-check: occupancy %d shared %d exceeds %d/%d\n",
                  occupied, shared_total, c->nslot, c->shared_slots);
    if (mapped_bytes != c->mapped_current_bytes ||
        c->mapped_current_bytes < 0 ||
        c->mapped_peak_bytes < c->mapped_current_bytes ||
        c->munmap_failures < 0 || c->munmap_failures > c->munmap_calls ||
        c->nprepare_failures < 0 ||
        c->nprepare_failures > c->nprepare_calls ||
        c->prepared_bytes < 0)
        CACHE_BAD("cache-check: bounded-mmap accounting drift\n");

    free(seen); free(home); free(shared); free(home_slots);
    if (errors && out) fprintf(out, "cache-check: FAIL (%d errors)\n", errors);
#undef CACHE_BAD
    return errors ? -1 : 0;
}

void salt_cache_report(const SaltCache *c, FILE *out) {
    if (!c || !out) return;
    const char *mode = c->mode == 0 ? "arena" :
                       c->mode == 1 ? "zerocopy" :
                       c->mode == 2 ? "mixture" : "bounded-mmap";
    int ok = salt_cache_check(c, NULL) == 0;
    int64_t probes = 0, pages = 0, resident_pages = 0, fully = 0;
    double probe_s = 0;
    for (int layer = 0; layer < c->n_layers; layer++) {
        const SaltCacheLayerStats *st = &c->layer_stats[layer];
        probes += st->source_probe_calls;
        pages += st->source_pages;
        resident_pages += st->source_resident_pages;
        fully += st->source_fully_resident;
        probe_s += st->source_probe_s;
    }

    fprintf(out,
            "[cache-accounting] mode=%s quota=%s slots=%d reserved=%d "
            "shared=%d invariant=%s\n",
            mode, c->quota_enabled ? "on" : "off", c->nslot,
            c->reserved_slots, c->shared_slots, ok ? "ok" : "FAIL");
    fprintf(out,
            "[cache-accounting] req=%lld ready_hit=%lld miss=%lld "
            "inflight_join=%lld insert=%lld evict=%lld denial=%lld "
            "drop=%lld jobs=%lld bytes=%lld\n",
            (long long)c->nreq, (long long)c->nhit, (long long)c->nmiss,
            (long long)c->ninflight_join, (long long)c->ninsert,
            (long long)c->nevict, (long long)c->nquota_denial,
            (long long)c->ndrop, (long long)c->nfetch_jobs,
            (long long)c->nread);
    fprintf(out,
            "[cache-accounting] wall_ms lookup=%.3f barrier=%.3f "
            "publish=%.3f worker_sum_ms queue=%.3f lock=%.3f "
            "pread_transport=%.3f readahead=%.3f touch=%.3f "
            "prepare=%.3f pretouch_calls=%lld prepare_calls=%lld "
            "prepare_failures=%lld prepared_bytes=%lld\n",
            c->lookup_s * 1e3, c->fetch_wait_s * 1e3,
            c->publish_s * 1e3, c->queue_wait_s * 1e3,
            c->lock_wait_s * 1e3, c->pread_s * 1e3,
            c->readahead_s * 1e3, c->touch_s * 1e3,
            c->prepare_s * 1e3, (long long)c->npretouch_calls,
            (long long)c->nprepare_calls,
            (long long)c->nprepare_failures,
            (long long)c->prepared_bytes);
    fprintf(out,
            "[cache-admission] nonquota_denial=%lld pin_refusal=%lld "
            "madvise_calls=%lld madvise_failures=%lld madvise_bytes=%lld "
            "mapped_current=%lld mapped_peak=%lld munmap_calls=%lld "
            "munmap_failures=%lld munmap_bytes=%lld\n",
            (long long)c->nadmission_denial, (long long)c->npin_refusal,
            (long long)c->madvise_calls, (long long)c->madvise_failures,
            (long long)c->madvise_bytes,
            (long long)c->mapped_current_bytes,
            (long long)c->mapped_peak_bytes,
            (long long)c->munmap_calls,
            (long long)c->munmap_failures,
            (long long)c->munmap_bytes);
    if (probes > 0)
        fprintf(out,
                "[cache-page-residency] probes=%lld fully_resident=%lld "
                "resident_pages=%lld total_pages=%lld probe_ms=%.3f "
                "diagnostic_only=1\n",
                (long long)probes, (long long)fully,
                (long long)resident_pages, (long long)pages,
                probe_s * 1e3);
    if (!c->accounting) return;
    for (int layer = 0; layer < c->n_layers; layer++) {
        const SaltCacheLayerStats *st = &c->layer_stats[layer];
        fprintf(out,
                "[cache-layer] L=%d quota=%d home=%d peak=%d shared=%d "
                "peak_shared=%d req=%lld hit=%lld miss=%lld join=%lld "
                "insert=%lld evict=%lld promote=%lld deny=%lld borrow=%lld return=%lld "
                "pread=%lld bytes=%lld pretouch=%lld barrier_ms=%.3f\n",
                layer, st->quota, st->home_current, st->home_peak,
                st->shared_current, st->shared_peak,
                (long long)st->requests, (long long)st->ready_hits,
                (long long)st->misses, (long long)st->inflight_joins,
                (long long)st->inserts, (long long)st->evictions,
                (long long)st->promotions,
                (long long)st->quota_denials,
                (long long)st->borrow_acquires,
                (long long)st->borrow_returns,
                (long long)st->pread_calls, (long long)st->pread_bytes,
                (long long)st->pretouch_calls,
                st->barrier_s * 1e3);
        fprintf(out,
                "[cache-layer-time] L=%d lookup_ms=%.3f queue_ms=%.3f "
                "queue_lock_ms=%.3f pread_transport_ms=%.3f "
                "readahead_ms=%.3f touch_ms=%.3f publish_ms=%.3f "
                "page_probe_ms=%.3f pages=%lld/%lld\n",
                layer, st->lookup_s * 1e3, st->queue_wait_s * 1e3,
                st->lock_wait_s * 1e3, st->pread_s * 1e3,
                st->readahead_s * 1e3, st->touch_s * 1e3,
                st->publish_s * 1e3, st->source_probe_s * 1e3,
                (long long)st->source_resident_pages,
                (long long)st->source_pages);
    }
}
