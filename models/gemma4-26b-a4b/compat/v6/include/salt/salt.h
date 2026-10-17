/*
 * ds4f-disk: disk-streaming MoE inference structure for
 * DeepSeek-V4-Flash-class models. Portable C99, zero dependencies
 * beyond libc + pthreads.
 *
 * The design mirrors the family proven by kimi-k3-in-c and Colibrì:
 * tiny in-RAM pointer map, contiguous packed trunk, policy-controlled
 * expert streaming. This is the STRUCTURE, validated on fixtures; real
 * weights drop in behind the same interfaces.
 *
 * Five invariants that must hold -- each one a place where a plausible
 * implementation runs, emits fluent text, and is wrong:
 *
 *  1. The pointer map is loaded BEFORE any weight bytes. Lookup is
 *     (layer, expert) -> file offset, O(1), never a scan. Fixed-rate
 *     payloads only: expert N is at base + N*nbytes. Variable-rate
 *     (entropy) coding breaks random access and is banned in the hot path.
 *
 *  2. Routing does not depend on the cache. The same prompt picks the
 *     same experts in the same order no matter what the cache does.
 *     This is what makes one trace replayable at any capacity under
 *     any policy (tools/trace_replay.py).
 *
 *  3. The biased score SELECTS, the unbiased score WEIGHTS. Collapsing
 *     them into one variable is a two-character edit that changes the model.
 *
 *  4. Precision mirrors access: full precision where resident (trunk
 *     pin, router, shared experts), 4-bit where streamed (routed
 *     experts). The trunk is deliberately NOT quantised; the experts
 *     ride MoE aggregation.
 *
 *  5. The memory plan is a forecast, not a result: sum everything
 *     before allocating anything, refuse to start past 95% of
 *     available memory, and measure peak RSS afterwards. Quote the
 *     measurement, not the forecast.
 */
#ifndef SALT_H
#define SALT_H

/* glibc hides POSIX declarations under -std=c99 unless asked; macOS
 * BSD headers break if asked. Ask only on Linux, before any system
 * header is included. */
#ifdef __linux__
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* config                                                             */
/* ------------------------------------------------------------------ */

typedef struct SaltCfg {
    int      n_layers;       /* trunk layers */
    int      n_experts;      /* routed experts per layer */
    int      topk;           /* experts selected per layer per token */
    int      n_shared;       /* shared experts per layer (resident) */
    int      hidden;         /* hidden width */
    int      latent;         /* latent width (2D factor: shared proj) */
    int      moe_inter;      /* expert MLP intermediate width */
    int      n_heads;        /* attention heads (0 = the kvhalf fallback) */
    int      n_kv_heads;     /* GQA kv heads (0 = fallback to n_heads) */
    int      qk_rope;        /* qk_rope_head_dim (0 = no rope term) */
    int      linear_num_key_heads;
    int      linear_num_value_heads;
    int      linear_key_head_dim;
    int      linear_value_head_dim;
    double   rope_factor;    /* tyrope scaling factor (0 = plain rotary) */
    double   rope_beta_fast; /* tyrope beta_fast (rotations) */
    double   rope_beta_slow; /* tyrope beta_slow (rotations) */
    double   rope_max_pos;   /* original_max_position_embeddings */
    double   rope_theta;     /* tyrope base theta (0 = 10000) */
    int64_t  expert_nbytes;  /* fixed-rate payload, one routed expert */
    int      n_shards;
    uint64_t seed;
} SaltCfg;

/* Parse config.json. Collects every absent required key and refuses the
 * load -- a defaulting reader produces a model that runs and speaks
 * English from the wrong architecture, with nothing to indicate it.
 * Returns 0 on success, -1 on failure (missing keys printed to stderr). */
int salt_cfg_load(SaltCfg *cfg, const char *model_dir);

/* ------------------------------------------------------------------ */
/* safetensors index -> pointer map                                   */
/* ------------------------------------------------------------------ */

typedef struct SaltTensor {
    const char *name;
    int         name_len;    /* safetensors keys are not NUL-terminated */
    int64_t     off;         /* absolute file offset */
    int64_t     nbytes;
} SaltTensor;

typedef struct SaltSt {
    int         fd;
    char       *hdr;         /* header buffer (names point into it) */
    int         n;
    SaltTensor *t;
    int64_t     hdr_len;     /* 8 + json header length = payload begin */
} SaltSt;

/* Open model.safetensors, parse the index, build the tensor map. */
int  salt_st_open(SaltSt *st, const char *path);
void salt_st_close(SaltSt *st);
/* O(1) name lookup over the pointer map. */
const SaltTensor *salt_st_find(const SaltSt *st, const char *name);

/* ------------------------------------------------------------------ */
/* expert pool: (layer, expert) -> file offset                        */
/* ------------------------------------------------------------------ */

typedef struct SaltExpertRef { int64_t off, nbytes; } SaltExpertRef;

typedef struct SaltExpertPool {
    int              fd;       /* pool-owned source descriptor */
    int              owns_fd;  /* 1 after build/open succeeds */
    uint8_t         *map;      /* mmap of the whole pool (flat map) */
    size_t           map_len;  /* 24 + nbytes*n_layers*n_experts */
    SaltExpertRef   *ref;      /* [layer*n_experts + expert] */
    int              n_layers, n_experts;
    int64_t          nbytes;   /* per-expert fixed payload */
} SaltExpertPool;

/* Release the pool's OWNED resources: the packed-path fd and the
 * mmap (the conservation pairing for salt_pool_open_packed's
 * ACQUIRE -- closes the fd + unmaps). The build path owns a duplicated
 * source fd and no mapping; this closes that duplicate as well. Returns
 * nonzero without discarding still-owned state on cleanup failure. */
int salt_pool_close(SaltExpertPool *pool);

/* Build the (layer, expert) -> offset table from the safetensors index.
 * Names are expected as "e.{layer}.{expert}". */
int salt_pool_build(SaltExpertPool *pool, const SaltSt *st, const SaltCfg *cfg);

/* Open a packed expert pool written by tools/convert-salt.py:
 *   [u64 expert_nbytes][u64 n_layers][u64 n_experts] then payload,
 *   layer-major expert-minor, fixed-rate (expert N at base + N*nbytes).
 * No index: the offsets are arithmetic. */
int salt_pool_open_packed(SaltExpertPool *pool, const char *path,
                          const SaltCfg *cfg);

/* Return a checked pointer to one expert payload in the flat mapping.
 * Refuses missing metadata, out-of-range ordinals, short payloads, and
 * any offset/range outside map_len. */
const uint8_t *salt_pool_expert_ptr(const SaltExpertPool *pool,
                                    size_t ordinal, int64_t min_nbytes);

/* ------------------------------------------------------------------ */
/* trunk: packed dense layers, pinned prefix + ring, prefetch         */
/* ------------------------------------------------------------------ */

typedef struct SaltTrunkLayer { int64_t off, nbytes; } SaltTrunkLayer;

typedef struct SaltTrunk {
    int             fd;
    int             n_layers;
    SaltTrunkLayer *lay;
    uint64_t        source_nbytes;

    /* Optional canonical layer lease. GPU-trunk mode maps at most one
     * layer-aligned source range at a time; map_base is the page-aligned
     * mapping while bind() returns map_base + map_data_offset. */
    int             mapped_mode;
    int             mapped_layer;
    uint8_t        *map_base;
    size_t          map_nbytes;
    size_t          map_data_offset;

    int             npin;        /* first npin layers resident */
    uint8_t        *pin;         /* pin arena, layers back to back */
    int64_t        *pin_off;     /* [npin] offsets within pin arena */

    int             nring;       /* ring slots (>= 2) */
    int64_t         slot;        /* ring slot size */
    uint8_t        *ring;        /* nring * slot */

    /* async reader: window = nring layers in flight */
    pthread_t       th;
    int             th_started;
    int             stop, failed;
    int             next_req, consumed, ready;
    pthread_mutex_t mu;
    pthread_cond_t  cv;

    int64_t         nread;       /* streamed bytes actually read */
} SaltTrunk;

/* Open trunk.bin + trunk.offsets. The offsets file is written by
 * tools/pack-trunk.c: [u64 n][u64 off x n][u64 size x n]. */
int  salt_trunk_open(SaltTrunk *tr, const char *bin_path, const char *off_path);
/* Plan the pin prefix for a byte budget (fixed point: pinning and slot
 * size are mutually dependent). */
void salt_trunk_plan(SaltTrunk *tr, int64_t budget, int *npin_out,
                     int64_t *slot_out, int nring);
/* Allocate arenas per the plan, load the pinned prefix, start reader. */
int  salt_trunk_start(SaltTrunk *tr, int npin, int64_t slot, int nring);
/* Start bounded canonical-source mode instead of allocating pin/ring copies. */
int  salt_trunk_start_mapped(SaltTrunk *tr);
void salt_trunk_rewind(SaltTrunk *tr);
/* Bind layer L: pinned -> resident pointer; streamed -> waits for the
 * reader, returns the ring slot. Blocks. */
const uint8_t *salt_trunk_bind(SaltTrunk *tr, int L);
/* Return/release the current page-aligned source mapping. A different layer
 * cannot be bound until the caller has unregistered every device view and
 * released this host lease. */
const void *salt_trunk_mapping(const SaltTrunk *tr, size_t *nbytes);
int salt_trunk_unbind(SaltTrunk *tr);
int salt_trunk_close(SaltTrunk *tr);

/* ------------------------------------------------------------------ */
/* expert cache: three-phase fetch                                    */
/* ------------------------------------------------------------------ */

#define SALT_SLOT_EMPTY    0
#define SALT_SLOT_LOADING  1
#define SALT_SLOT_READY    2

/* one disk read: pool->ref[layer*n_experts+expert].off -> dst */
typedef struct FetchJob {
    int              slot, layer, expert, ok;
    int              zc;        /* zerocopy mode: readahead+touch */
    int              slotmap;   /* borrowed per-slot mmap; munmap on eviction */
    int              pretouch;  /* diagnostic: fault arena dst before pread */
    int64_t          off;
    int64_t          touch_bytes; /* strict-proof synchronous mapped prefix */
    uint8_t         *dst;
    void            *map_base;
    size_t           map_len;
    SaltExpertPool  *pool;
    double           queued_s;
    double           queue_wait_s;
    double           lock_wait_s;
    double           pread_s;
    double           readahead_s;
    double           touch_s;
} FetchJob;

typedef struct SaltCacheLayerStats {
    int             quota;
    int             home_current, home_peak;
    int             shared_current, shared_peak;
    int64_t         requests, ready_hits, inflight_joins, misses;
    int64_t         inserts, evictions, promotions, quota_denials;
    int64_t         borrow_acquires, borrow_returns;
    int64_t         pread_calls, pread_bytes, readahead_calls;
    int64_t         prepare_calls, prepare_failures, prepared_bytes;
    int64_t         pretouch_calls;
    int64_t         source_probe_calls, source_pages;
    int64_t         source_resident_pages, source_fully_resident;
    double          lookup_s, queue_wait_s, lock_wait_s;
    /* pread syscall wall: storage/transport + kernel copy + faults not
     * removed by optional destination pretouch; never pure SSD latency. */
    double          pread_s, readahead_s, touch_s, prepare_s;
    double          source_probe_s, barrier_s, publish_s;
} SaltCacheLayerStats;

typedef int (*SaltCacheResourceBindFn)(void *ctx, int slot,
                                       int layer, int expert,
                                       const void *base, size_t nbytes,
                                       const void *payload);
typedef int (*SaltCacheResourceFenceFn)(void *ctx);
typedef int (*SaltCacheResourceUnbindFn)(void *ctx, int slot,
                                         int layer, int expert);

typedef struct SaltCache {
    SaltExpertPool *pool;
    int             nslot;
    int64_t         slot_bytes;
    /* FOUR MODES (SALT_CACHE_MODE plus explicit bounded-mmap policy):
     *   arena    -- all slots: pread into arena copies (default;
     *               the memo cache: resident, fast decode, +cache-gb)
     *   zerocopy -- all slots: payloads point into the pool mmap;
     *               fill = readahead+touch (faster fetch, reclaimable)
     *   mixture  -- arena for the keep/memo set (hot experts: resident
     *               copies, fewer reads, faster decode) + zerocopy for
     *               fresh fetches (readahead+touch, faster fetch).
     *               A slot whose victim had >= SALT_MIX_HITS stays a
     *               resident copy; cold victims return to the mmap.
     *   bounded-mmap -- one borrowed file mapping per occupied slot;
     *               eviction advises and unmaps the complete borrowed view. */
    uint8_t        *arena;      /* nslot * slot_bytes (arena / mixture) */
    const uint8_t **payload;    /* [nslot] effective expert pointers */
    void          **slot_map_base; /* [nslot], bounded-mmap borrowed views */
    size_t         *slot_map_len;  /* [nslot], exact mmap lengths */
    int           (*munmap_fn)(void *, size_t); /* injectable release seam */
    SaltCacheResourceBindFn resource_bind_fn;
    SaltCacheResourceFenceFn resource_fence_fn;
    SaltCacheResourceUnbindFn resource_unbind_fn;
    void           *resource_ctx;
    uint8_t        *resource_bound; /* [nslot], backend view owns mapping */
    int             mode;       /* 0 arena, 1 zerocopy, 2 mixture, 3 slotmap */
    uint8_t        *state;      /* EMPTY, LOADING, or READY */
    uint64_t       *key;        /* actual (layer,expert) key when non-empty */
    int            *resident_slot; /* [n_layers*n_experts], -1 if absent */
    uint64_t       *demand;        /* per-key request history; victim authority */
    int            *slot_home;  /* [nslot], layer reservation or -1 shared */
    int            *layer_quota;
    SaltCacheLayerStats *layer_stats;
    int             n_layers, n_experts;
    int             quota_enabled;
    int             reserved_slots, shared_slots;
    int             accounting, check_enabled;
    int             page_probe_enabled, pretouch_enabled;
    int             fetch_device_fault_only;
    int64_t         fetch_touch_bytes;
    int             proof_state; /* strict proof may synchronously first-touch */
    uint64_t       *used_at;    /* LRU clock */
    uint64_t       *hits;       /* slot frequency; logical demand ranks victims */
    uint8_t        *pinned;
    uint32_t       *lease_count; /* active pointer users; eviction forbidden */
    uint64_t        clock;
    int             nthreads;

    /* Serialize the complete lookup/reserve/read/publish transaction. The
     * worker mutex below protects only queue execution; it is not a cache
     * metadata lock. */
    pthread_mutex_t call_mu;
    int             call_mu_init;

    /* persistent fetch pool: workers wait on the queue instead of
     * spawning/joining 8 threads per layer (the spawn churn was
     * ~20-30ms of the 22-38ms fetch phase -- the disk read itself
     * is ~3.5ms). */
    pthread_t       *fth;       /* [nthreads] worker threads */
    pthread_mutex_t fmu;
    pthread_cond_t  fcv_work, fcv_done;
    int             fshutdown;
    int             fbusy;      /* a batch is in flight */
    int             fclaim;     /* next fixed descriptor to claim */
    int             fdone;      /* mappings completed in current batch */
    int             fnjobs;
    FetchJob       *fjobs;      /* fixed startup-owned descriptor array */
    int             fcapacity;
    int             fcreated;   /* successfully created worker threads */
    int             fsync_init; /* 0..3: mutex, work cond, done cond */

    int64_t         nreq, nhit, ndrop;   /* stats */
    int64_t         nread;               /* logical miss payload bytes */
    int64_t         nfetch_jobs;         /* physical fetch jobs issued */
    int64_t         nmiss, ninflight_join, ninsert, nevict;
    int64_t         nquota_denial, nadmission_denial, npin_refusal;
    int64_t         npretouch_calls;
    int64_t         nprepare_calls, nprepare_failures, prepared_bytes;
    int64_t         ncheck_fail;
    int64_t         madvise_calls, madvise_failures, madvise_bytes;
    int64_t         mapped_current_bytes, mapped_peak_bytes;
    int64_t         munmap_calls, munmap_failures, munmap_bytes;
    double          fetch_wait_s;        /* map/read worker barrier */
    double          lookup_s, queue_wait_s, lock_wait_s;
    double          pread_s, readahead_s, touch_s, prepare_s, publish_s;
} SaltCache;

/* Allocate a cache sized in slots (caller computes slots from bytes).
 * A successfully initialized cache must be freed before reuse. */
int salt_cache_init(SaltCache *c, SaltExpertPool *pool, int nslot,
                    int nthreads);
/* Explicit policy entry point for model runtimes whose memory contract
 * requires mapped payloads and must not inherit the arena default or a
 * process-global SALT_CACHE_MODE value. */
int salt_cache_init_zerocopy(SaltCache *c, SaltExpertPool *pool, int nslot,
                             int nthreads);
/* Strong mapped-residency policy: map only occupied expert slots and munmap
 * every victim. Unlike flat zero-copy, task RSS cannot retain evicted slots. */
int salt_cache_init_bounded_mmap(SaltCache *c, SaltExpertPool *pool, int nslot,
                                 int nthreads);
/* Select request-scoped strict proof behavior. Normal operation publishes a
 * correctly mapped/bound resource READY without synchronously proving page
 * residency. Strict proof additionally performs the configured first-touch. */
int salt_cache_set_proof_state(SaltCache *c, int enabled);
/* Exact persistent metadata forecast for zero-copy/bounded-mmap cache modes.
 * Performs no allocation and excludes the canonical expert payload itself. */
int salt_cache_startup_metadata_bytes(int nslot, int n_layers, int n_experts,
                                      int nthreads, int bounded_mmap,
                                      uint64_t *bytes_out);
/* Install fixed-slot backend resource hooks for direct flat zero-copy or
 * bounded-mmap payloads. Each READY slot is bound as its exact expert range.
 * An optional fence runs once before all retirements in one cache transaction;
 * NULL preserves a backend unbind's ordinary per-slot completion behavior.
 * READY publication binds before exposure; eviction and teardown unbind after
 * the final lease and before madvise/munmap. */
int salt_cache_set_resource_hooks(SaltCache *c,
                                  SaltCacheResourceFenceFn fence_fn,
                                  SaltCacheResourceBindFn bind_fn,
                                  SaltCacheResourceUnbindFn unbind_fn,
                                  void *ctx);
void salt_cache_free(SaltCache *c);
/* Release only page-aligned pages wholly contained by [pointer, pointer+bytes).
 * Partial boundary pages are deliberately retained because adjacent live
 * tensors or expert slots may share them. */
int salt_mmap_dontneed_contained(const void *pointer, size_t bytes,
                                 size_t *released_bytes);
/* Request asynchronous page-cache preparation only for complete pages wholly
 * contained by the canonical mapped range. This creates no mapping or payload. */
int salt_mmap_willneed_contained(const void *pointer, size_t bytes,
                                 size_t *advised_bytes);
/* Issue bounded asynchronous read-ahead for an existing canonical file range.
 * No mapping, allocation, payload copy, or residency claim is created. */
int salt_file_willneed(int fd, int64_t offset, size_t bytes);
/* Calls on one cache are serialized. Phase 1 reserves serially (no
 * double-claim, LOADING skipped, EMPTY
 * preferred, pinned skipped last). Phase 2: read in parallel, in
 * disk-offset order. Phase 3: publish only what arrived. Returns the number
 * of physical fetch jobs, or -1; every out_slot is READY on success. */
int  salt_cache_getmany(SaltCache *c, int layer, const int *experts, int n,
                        int *out_slot);
/* Direct pointer to a READY slot's payload, otherwise NULL. */
const uint8_t *salt_cache_slot(const SaltCache *c, int slot);
/* Key-validated handle lookup: NULL unless slot is READY and still contains
 * exactly (layer, expert). Retained slot indices must use this accessor. */
const uint8_t *salt_cache_slot_for(const SaltCache *c, int slot,
                                   int layer, int expert);
/* Acquire/release a key-validated pointer lease. Victim selection skips a
 * slot until its final lease is released. Every successful acquire requires
 * exactly one matching release after the final pointer user has completed. */
const uint8_t *salt_cache_acquire(SaltCache *c, int slot,
                                  int layer, int expert);
int salt_cache_release(SaltCache *c, int slot, int layer, int expert);
/* Atomic multi-slot lease operations under one existing cache mutex. Every
 * slot/key is validated before any count changes; duplicate slots fail. */
int salt_cache_acquire_many(SaltCache *c, const int *slots, int layer,
                            const int *experts, int n,
                            const uint8_t **payloads);
int salt_cache_release_many(SaltCache *c, const int *slots, int layer,
                            const int *experts, int n);
/* Pinning is an optimization. It may be refused to preserve bounded
 * admission capacity; salt_cache_pin_expert() reports whether it stuck. */
void salt_cache_pin(SaltCache *c, int slot, int pin);
/* L2: pin the resident slot for (layer, expert); 1 if pinned, 0 if absent or
 * refused to preserve bounded admission. */
int  salt_cache_pin_expert(SaltCache *c, int layer, int expert);
/* L2: count of pinned slots (diagnostics). */
int  salt_cache_npinned(const SaltCache *c);
/* Debug/accounting surfaces. The checker never repairs state: any mismatch
 * is a hard failure for the caller to handle. */
int  salt_cache_check(const SaltCache *c, FILE *out);
void salt_cache_report(const SaltCache *c, FILE *out);

/* ------------------------------------------------------------------ */
/* router: biased score selects, unbiased score weights               */
/* ------------------------------------------------------------------ */

/* In fixture mode the score is a deterministic hash of (state, layer,
 * expert); with real weights this becomes a dot product against the
 * resident router matrix. The invariant is the split, not the source.
 * locality > 0 boosts SELECTION toward a fixed popular set (experts
 * 0..63 per layer), which creates temporal reuse in the fixture so the
 * cache sweep shows policy-vs-capacity separation. The boost touches
 * ch only -- sc stays the unbiased score. */
void salt_router(int *idx, float *w, const SaltCfg *cfg,
                 uint64_t state, int layer, double locality);

/* ------------------------------------------------------------------ */
/* memory planner                                                     */
/* ------------------------------------------------------------------ */

typedef struct SaltMemPlan {
    double trunk_pin_b;   /* pinned trunk prefix */
    double trunk_ring_b;  /* ring slots */
    double cache_b;       /* expert cache arenas */
    double shared_b;      /* resident shared experts */
    double state_b;       /* activations / KV / scratch */
    double index_b;       /* safetensors index estimate */
    double need_b, have_b;
} SaltMemPlan;

int64_t salt_mem_available(void);        /* bytes; SALT_TEST_MEM overrides */
void    salt_mem_plan(SaltMemPlan *p, const SaltCfg *cfg,
                      int64_t trunk_budget, int64_t cache_bytes,
                      int64_t shared_bytes, int npin, int64_t slot, int nring);
int     salt_mem_refuses(const SaltMemPlan *p);   /* need > have * 0.95 */
void    salt_mem_print(const SaltMemPlan *p);

typedef struct SaltMemSnapshot {
    int64_t application_b;          /* current hard-limit metric */
    int64_t application_os_peak_b;  /* kernel lifetime peak; 0 if absent */
    int64_t resident_b;             /* current touched working set */
    int64_t resident_os_peak_b;     /* kernel resident high-water */
    int64_t anonymous_b;            /* internal/RssAnon attribution */
    int64_t compressed_b;           /* private compressed attribution */
    int64_t file_backed_b;          /* external/RssFile comparison */
    int64_t shared_b;               /* RssShmem on Linux */
    int64_t page_table_b;           /* VmPTE on Linux */
    int64_t swap_b;                 /* VmSwap comparison on Linux */
    int application_is_approximate; /* Linux process-local approximation */
} SaltMemSnapshot;

typedef struct SaltIoSnapshot {
    uint64_t process_read_bytes;       /* OS-attributed cumulative reads */
    uint64_t major_faults;             /* process cumulative major faults */
    int process_read_bytes_valid;
    int major_faults_valid;
} SaltIoSnapshot;

int     salt_mem_snapshot(SaltMemSnapshot *snapshot);
int     salt_io_snapshot(SaltIoSnapshot *snapshot);
uint64_t salt_disk_read_bytes(void);      /* cumulative physical reads */

/* ------------------------------------------------------------------ */
/* misc                                                               */
/* ------------------------------------------------------------------ */

uint64_t salt_mix64(uint64_t x);         /* splitmix64 finalizer */
uint64_t salt_checksum(const uint8_t *p, int64_t n);   /* compute sink */

#endif /* SALT_H */
