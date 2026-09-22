/* trunk.c -- packed dense layers. Pinned prefix + rotating ring with an
 * async reader so the next layer's read overlaps the current layer's
 * compute. The pin/ring split exists because a cyclic scan is the
 * pathological case for LRU eviction: pinning the first N layers gives
 * a deterministic N/n_layers hit rate where any LRU gives zero. */
#include "salt/salt.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* offsets file layout (little-endian u64s, written by pack-trunk):
 *   [0]   n
 *   [1..] n x (off, nbytes)   -- offsets into trunk.bin */
static int rd_u64s(int fd, uint64_t *buf, int n, int64_t at) {
    return pread(fd, buf, (size_t)n * 8, at) == (ssize_t)n * 8 ? 0 : -1;
}

int salt_trunk_open(SaltTrunk *tr, const char *bin_path, const char *off_path) {
    int ofd = -1;
    uint64_t *pair = NULL;
    struct stat offsets_st, source_st;
    int mutex_ok = 0;
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    tr->mapped_layer = -1;

    ofd = open(off_path, O_RDONLY);
    if (ofd < 0) {
        fprintf(stderr, "trunk: cannot open %s: %s\n", off_path, strerror(errno));
        return -1;
    }
    uint64_t n = 0;
    if (fstat(ofd, &offsets_st) != 0 || offsets_st.st_size < 0 ||
        rd_u64s(ofd, &n, 1, 0) != 0 || n == 0 || n > 65536 ||
        n > ((uint64_t)INT64_MAX - 8) / 16 ||
        (uint64_t)offsets_st.st_size != 8 + n * 16) {
        fprintf(stderr, "trunk: bad offsets header in %s\n", off_path);
        goto fail;
    }
    tr->n_layers = (int)n;
    tr->lay = (SaltTrunkLayer *)calloc((size_t)n, sizeof(SaltTrunkLayer));
    if (!tr->lay) goto fail;
    pair = (uint64_t *)malloc(16);
    if (!pair) goto fail;
    for (int i = 0; i < tr->n_layers; i++) {
        if (rd_u64s(ofd, pair, 2, 8 + (int64_t)i * 16) != 0 ||
            pair[0] > INT64_MAX || pair[1] == 0 || pair[1] > INT64_MAX ||
            pair[0] > UINT64_MAX - pair[1]) {
            fprintf(stderr, "trunk: short offsets file\n");
            goto fail;
        }
        tr->lay[i].off    = (int64_t)pair[0];
        tr->lay[i].nbytes = (int64_t)pair[1];
    }
    free(pair); pair = NULL;
    if (close(ofd) != 0) {
        fprintf(stderr, "trunk: close failed for %s: %s\n",
                off_path, strerror(errno));
        ofd = -1;
        goto fail;
    }
    ofd = -1;

    tr->fd = open(bin_path, O_RDONLY);
    if (tr->fd < 0) {
        fprintf(stderr, "trunk: cannot open %s: %s\n", bin_path, strerror(errno));
        goto fail;
    }
    if (fstat(tr->fd, &source_st) != 0 || source_st.st_size < 0) goto fail;
    tr->source_nbytes = (uint64_t)source_st.st_size;
    for (int i = 0; i < tr->n_layers; i++) {
        uint64_t off = (uint64_t)tr->lay[i].off;
        uint64_t nbytes = (uint64_t)tr->lay[i].nbytes;
        if (off > tr->source_nbytes || nbytes > tr->source_nbytes - off) {
            fprintf(stderr, "trunk: layer %d range exceeds %s\n", i, bin_path);
            goto fail;
        }
    }
#ifdef __APPLE__
    fcntl(tr->fd, F_NOCACHE, 1);
#endif
    if (pthread_mutex_init(&tr->mu, NULL) != 0) goto fail;
    mutex_ok = 1;
    if (pthread_cond_init(&tr->cv, NULL) != 0) goto fail;
    return 0;

fail:
    if (mutex_ok) pthread_mutex_destroy(&tr->mu);
    if (tr->fd >= 0) close(tr->fd);
    if (ofd >= 0) close(ofd);
    free(pair);
    free(tr->lay);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    tr->mapped_layer = -1;
    return -1;
}

void salt_trunk_plan(SaltTrunk *tr, int64_t budget, int *npin_out,
                     int64_t *slot_out, int nring) {
    if (nring < 2) nring = 2;
    int64_t slot = 0;
    for (int L = 0; L < tr->n_layers; L++)
        if (tr->lay[L].nbytes > slot) slot = tr->lay[L].nbytes;
    if (slot <= 0) slot = 4096;
    int npin = 0;
    for (int pass = 0; pass < 4; pass++) {   /* pin and slot are mutually dependent */
        int64_t avail = budget > slot * nring ? budget - slot * nring : 0;
        int n = 0;
        int64_t used = 0;
        while (n < tr->n_layers && used + tr->lay[n].nbytes <= avail) {
            used += tr->lay[n].nbytes;
            n++;
        }
        int64_t need = 0;
        for (int L = n; L < tr->n_layers; L++)
            if (tr->lay[L].nbytes > need) need = tr->lay[L].nbytes;
        if (need == 0) need = 4096;
        if (n == npin && need == slot) break;
        npin = n;
        slot = need;
    }
    *npin_out = npin;
    *slot_out = slot;
}

static int load_pins(SaltTrunk *tr) {
    int64_t total = 0;
    for (int L = 0; L < tr->npin; L++) total += tr->lay[L].nbytes;
    tr->pin = (uint8_t *)calloc((size_t)total, 1);
    tr->pin_off = (int64_t *)calloc((size_t)tr->npin, sizeof(int64_t));
    if (!tr->pin || (tr->npin && !tr->pin_off)) return -1;
    int64_t at = 0;
    for (int L = 0; L < tr->npin; L++) {
        tr->pin_off[L] = at;
        if (pread(tr->fd, tr->pin + at, (size_t)tr->lay[L].nbytes,
                  tr->lay[L].off) != (ssize_t)tr->lay[L].nbytes) {
            fprintf(stderr, "trunk: pinned layer %d read failed\n", L);
            return -1;
        }
        at += tr->lay[L].nbytes;
    }
    return 0;
}

/* Reader: keeps at most nring-1 layers in flight ahead of the consumer,
 * so the slot it writes next is always one the consumer has finished.
 * The stream runs over [npin, n_layers) per pass; multi-token
 * generation calls salt_trunk_rewind() between tokens, which resets
 * next_req/ready/consumed so the next pass re-streams every layer
 * (the ring slots are reused, but the ready/consumed handshake makes
 * the consumer wait for each layer's re-fetch). */
static void *trunk_reader(void *p) {
    SaltTrunk *t = (SaltTrunk *)p;
    const int window = t->nring - 1;
    pthread_mutex_lock(&t->mu);
    while (!t->stop) {
        while (!t->stop &&
               (t->next_req >= t->n_layers ||
                t->next_req >= t->consumed + window))
            pthread_cond_wait(&t->cv, &t->mu);
        if (t->stop) break;
        int L = t->next_req++;
        int is_pin = L < t->npin;
        int64_t nb = t->lay[L].nbytes;
        int64_t off = t->lay[L].off;
        pthread_mutex_unlock(&t->mu);
        if (!is_pin) {
            uint8_t *buf = t->ring + (int64_t)(L % t->nring) * t->slot;
            if (pread(t->fd, buf, (size_t)nb, off) != (ssize_t)nb) {
                pthread_mutex_lock(&t->mu);
                t->failed = 1;
                pthread_mutex_unlock(&t->mu);
            } else {
                pthread_mutex_lock(&t->mu);
                t->nread += nb;
                pthread_mutex_unlock(&t->mu);
            }
        }
        pthread_mutex_lock(&t->mu);
        t->ready = L + 1;
        pthread_cond_broadcast(&t->cv);
    }
    pthread_mutex_unlock(&t->mu);
    return NULL;
}

/* Restart the streaming pass: the next bind() re-reads layers from
 * npin upward (for multi-token generation, the trunk is re-streamed
 * once per token). Call BEFORE the first bind of a new pass. */
void salt_trunk_rewind(SaltTrunk *tr) {
    if (!tr) return;
    if (tr->mapped_mode) return;
    pthread_mutex_lock(&tr->mu);
    tr->next_req = tr->npin;
    tr->ready = tr->npin;
    tr->consumed = tr->npin;
    pthread_cond_broadcast(&tr->cv);
    pthread_mutex_unlock(&tr->mu);
}

int salt_trunk_start(SaltTrunk *tr, int npin, int64_t slot, int nring) {
    if (!tr || tr->fd < 0 || tr->mapped_mode) return -1;
    if (nring < 2) nring = 2;
    tr->npin = npin;
    tr->slot = slot > 0 ? slot : 4096;
    tr->nring = nring;
    if (load_pins(tr) != 0) return -1;
    tr->ring = (uint8_t *)calloc((size_t)nring, (size_t)tr->slot);
    if (!tr->ring) return -1;
    if (pthread_create(&tr->th, NULL, trunk_reader, tr) != 0) return -1;
    tr->th_started = 1;
    return 0;
}

int salt_trunk_start_mapped(SaltTrunk *tr) {
    if (!tr || tr->fd < 0 || tr->th_started || tr->pin || tr->ring ||
        tr->map_base || tr->mapped_mode)
        return -1;
    tr->mapped_mode = 1;
    tr->mapped_layer = -1;
    tr->npin = 0;
    tr->nring = 0;
    tr->slot = 0;
    return 0;
}

const uint8_t *salt_trunk_bind(SaltTrunk *tr, int L) {
    if (!tr || L < 0 || L >= tr->n_layers) return NULL;
    if (tr->mapped_mode) {
        if (tr->map_base) {
            if (tr->mapped_layer != L) return NULL;
            return tr->map_base + tr->map_data_offset;
        }
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) return NULL;
        uint64_t off = (uint64_t)tr->lay[L].off;
        uint64_t nbytes = (uint64_t)tr->lay[L].nbytes;
        uint64_t page = (uint64_t)page_size;
        uint64_t map_off = off - off % page;
        uint64_t data_off = off - map_off;
        if (data_off > SIZE_MAX || nbytes > SIZE_MAX - (size_t)data_off)
            return NULL;
        size_t map_nbytes = (size_t)data_off + (size_t)nbytes;
        uint8_t *base = (uint8_t *)mmap(
            NULL, map_nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE,
            tr->fd, (off_t)map_off);
        if (base == MAP_FAILED) return NULL;
        tr->mapped_layer = L;
        tr->map_base = base;
        tr->map_nbytes = map_nbytes;
        tr->map_data_offset = (size_t)data_off;
        return base + tr->map_data_offset;
    }
    if (L < tr->npin) {
        /* resident; also advance the reader's window past it */
        pthread_mutex_lock(&tr->mu);
        tr->consumed = L + 1;
        pthread_cond_broadcast(&tr->cv);
        pthread_mutex_unlock(&tr->mu);
        return tr->pin + tr->pin_off[L];
    }
    pthread_mutex_lock(&tr->mu);
    while (!tr->stop && !tr->failed && tr->ready <= L)
        pthread_cond_wait(&tr->cv, &tr->mu);
    int failed = tr->failed;
    const uint8_t *buf = tr->ring + (int64_t)(L % tr->nring) * tr->slot;
    tr->consumed = L + 1;
    pthread_cond_broadcast(&tr->cv);
    pthread_mutex_unlock(&tr->mu);
    return failed ? NULL : buf;
}

const void *salt_trunk_mapping(const SaltTrunk *tr, size_t *nbytes_out) {
    if (nbytes_out) *nbytes_out = 0;
    if (!tr || !tr->mapped_mode || !tr->map_base || !tr->map_nbytes)
        return NULL;
    if (nbytes_out) *nbytes_out = tr->map_nbytes;
    return tr->map_base;
}

int salt_trunk_unbind(SaltTrunk *tr) {
    if (!tr || !tr->mapped_mode) return 0;
    if ((!tr->map_base) != (tr->map_nbytes == 0)) return -1;
    if (tr->map_base && tr->map_nbytes &&
        munmap(tr->map_base, tr->map_nbytes) != 0)
        return -1;
    tr->map_base = NULL;
    tr->map_nbytes = 0;
    tr->map_data_offset = 0;
    tr->mapped_layer = -1;
    return 0;
}

int salt_trunk_close(SaltTrunk *tr) {
    if (!tr) return 0;
    if (salt_trunk_unbind(tr) != 0) return -1;
    if (pthread_mutex_lock(&tr->mu) != 0) return -1;
    tr->stop = 1;
    int broadcast_rc = pthread_cond_broadcast(&tr->cv);
    int unlock_rc = pthread_mutex_unlock(&tr->mu);
    if (broadcast_rc != 0 || unlock_rc != 0) return -1;
    if (tr->th_started) {
        if (pthread_join(tr->th, NULL) != 0) return -1;
        tr->th_started = 0;
    }
    if (tr->fd >= 0) {
        if (close(tr->fd) != 0) return -1;
        tr->fd = -1;
    }
    int mutex_rc = pthread_mutex_destroy(&tr->mu);
    int cond_rc = pthread_cond_destroy(&tr->cv);
    if (mutex_rc != 0 || cond_rc != 0) return -1;
    free(tr->lay);
    free(tr->pin);
    free(tr->pin_off);
    free(tr->ring);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    tr->mapped_layer = -1;
    return 0;
}
