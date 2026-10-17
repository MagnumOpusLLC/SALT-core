/* LFM2.5-24B-A2B native entry.
 *
 * Internal descriptor-bound protocol, same shape as the other model adapters:
 *   lfm-server TRUNK_FD POOL_FD INDEX_FD TOKENIZER_FD PACKAGE_SHA CTX B W SLOTS RSS
 *
 * This entry authenticates the package, opens the model-owned runtime, and runs
 * one bounded native semantic/performance request. It is a qualification entry,
 * not the HTTP serving surface.
 *
 * It reads no environment configuration. Runtime allocation and descriptor
 * realization happen once before generation begins. */

#define _POSIX_C_SOURCE 200809L
#include "lfm.h"
#include "runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

/* 314 trunk entries from the importer's spec plus the tied-head reference. */
#define LFM_T_LINES_EXPECTED \
    (1u + 2u + 2u * LFM_LAYERS + 6u * LFM_ATTENTION_LAYERS + 3u * LFM_CONV_LAYERS \
     + 3u * LFM_DENSE_LAYERS + 2u * LFM_MOE_LAYERS)
#define LFM_E_LINES_EXPECTED ((uint64_t)LFM_MOE_LAYERS * LFM_EXPERTS * LFM_EXPERT_ROLE_COUNT)
#define LFM_LINE_MAX 512

/* The first-run probe: a fixed prompt and a bounded token count, enough to
 * prove the chain end to end without a serving surface. */
#define LFM_PROBE_TOKENS 8
#define LFM_PROBE_OUTPUT_CAP 512
#define LFM_PROBE_PROMPT_CAP 32768
#define LFM_PROBE_TEXT_CAP 65536
#define LFM_PROBE_SOURCE_CAP 1048576

typedef struct {
    int layer;
    int64_t base;              /* expert ordinal base, -1 for a dense layer */
} LfmLayerBase;

typedef struct {
    uint64_t value_offset, value_bytes;
    uint64_t scale_offset, scale_bytes;
    uint64_t bias_offset, bias_bytes;
} LfmComponents;

typedef struct {
    uint32_t t_lines, e_lines, l_lines, tied_head;
    uint32_t role_seen[LFM_ROLE_COUNT];
    uint32_t expert_role_seen[LFM_EXPERT_ROLE_COUNT];
    uint64_t expert_slots_ok;
    uint64_t trunk_bytes, pool_bytes;
} LfmAudit;

static int role_index(const char *name) {
    for (int i = 0; i < LFM_ROLE_COUNT; i++)
        if (!strcmp(name, LFM_ROLE_NAMES[i]))
            return i;
    return -1;
}

static int expert_role_index(const char *name) {
    for (int i = 0; i < LFM_EXPERT_ROLE_COUNT; i++)
        if (!strcmp(name, LFM_EXPERT_ROLE_NAMES[i]))
            return i;
    return -1;
}

static int read_regular(int fd, struct stat *st, const char *what) {
    if (fstat(fd, st) || !S_ISREG(st->st_mode)) {
        fprintf(stderr, "LFM_ADMISSION_FAIL %s is not a regular file\n", what);
        return -1;
    }
    return 0;
}

/* Parse and audit index.txt against the declared geometry. Every count and
 * every offset is checked; nothing is inferred from the file. */
static int audit_index(FILE *f, LfmAudit *a) {
    char line[LFM_LINE_MAX];
    char magic[32], abi[96];
    unsigned long long expert_bytes, n_layers, n_experts, n_moe, n_count, trunk, pool;
    LfmLayerBase bases[LFM_LAYERS];
    int seen_layer[LFM_LAYERS];
    LfmComponents embed;
    int have_embed = 0;
    int ordinal = 0;

    memset(a, 0, sizeof *a);
    memset(bases, 0, sizeof bases);
    memset(seen_layer, 0, sizeof seen_layer);
    memset(&embed, 0, sizeof embed);

    if (!fgets(line, sizeof line, f) ||
        sscanf(line, "%31s %95s %llu %llu %llu %llu %llu %llu %llu",
               magic, abi, &expert_bytes, &n_layers, &n_experts, &n_moe, &n_count,
               &trunk, &pool) != 9) {
        fprintf(stderr, "LFM_ADMISSION_FAIL index header unreadable\n");
        return -1;
    }
    if (strcmp(magic, LFM_INDEX_MAGIC) || strcmp(abi, LFM_INDEX_ABI)) {
        fprintf(stderr, "LFM_ADMISSION_FAIL index magic/abi mismatch\n");
        return -1;
    }
    if (expert_bytes != LFM_EXPERT_BYTES || n_layers != LFM_LAYERS ||
        n_experts != LFM_EXPERTS || n_moe != LFM_MOE_LAYERS ||
        n_count != LFM_EXPERT_COUNT || pool != LFM_POOL_BYTES) {
        fprintf(stderr, "LFM_ADMISSION_FAIL index geometry mismatch "
                "slot=%llu/%u layers=%llu/%u experts=%llu/%u moe=%llu/%u "
                "count=%llu/%u pool=%llu/%llu\n",
                expert_bytes, LFM_EXPERT_BYTES, n_layers, LFM_LAYERS,
                n_experts, LFM_EXPERTS, n_moe, LFM_MOE_LAYERS,
                n_count, LFM_EXPERT_COUNT, pool, (unsigned long long)LFM_POOL_BYTES);
        return -1;
    }
    a->trunk_bytes = trunk;
    a->pool_bytes = pool;

    while (fgets(line, sizeof line, f)) {
        char kind = line[0];
        if (kind == 'L') {
            int layer, used = 0;
            long long base;
            if (sscanf(line, "L %d %lld%n", &layer, &base, &used) != 2 ||
                line[used] != '\n' || layer < 0 || layer >= LFM_LAYERS) {
                fprintf(stderr, "LFM_ADMISSION_FAIL malformed L line\n");
                return -1;
            }
            if (seen_layer[layer]++) {
                fprintf(stderr, "LFM_ADMISSION_FAIL duplicate L line %d\n", layer);
                return -1;
            }
            if (layer < LFM_DENSE_LAYERS) {
                if (base != -1) {
                    fprintf(stderr, "LFM_ADMISSION_FAIL dense layer %d has an expert base\n", layer);
                    return -1;
                }
            } else if (base != (long long)(ordinal++ * LFM_EXPERTS)) {
                fprintf(stderr, "LFM_ADMISSION_FAIL layer %d base %lld\n", layer, base);
                return -1;
            }
            bases[layer].layer = layer;
            bases[layer].base = base;
            a->l_lines++;
        } else if (kind == 'T') {
            char role[48];
            unsigned enc, rows, cols;
            unsigned long long vo, vb, so, sb, bo, bb;
            int used = 0, r;
            LfmComponents c;
            memset(&c, 0, sizeof c);
            if (sscanf(line, "T %47s %*d %u %u %u %llu %llu %llu %llu %llu %llu%n",
                       role, &enc, &rows, &cols, &vo, &vb, &so, &sb, &bo, &bb,
                       &used) != 10 ||
                line[used] != '\n') {
                fprintf(stderr, "LFM_ADMISSION_FAIL unparsable T line: %s\n", line);
                return -1;
            }
            c.value_offset = vo; c.value_bytes = vb;
            c.scale_offset = so; c.scale_bytes = sb;
            c.bias_offset = bo; c.bias_bytes = bb;
            if ((r = role_index(role)) < 0) {
                fprintf(stderr, "LFM_ADMISSION_FAIL unknown trunk role %s\n", role);
                return -1;
            }
            if (rows < 1 || cols < 1 || rows > INT_MAX || cols > INT_MAX) {
                fprintf(stderr, "LFM_ADMISSION_FAIL trunk extent %s\n", role);
                return -1;
            }
            if (c.value_offset + c.value_bytes > trunk ||
                c.scale_offset + c.scale_bytes > trunk ||
                c.bias_offset + c.bias_bytes > trunk) {
                fprintf(stderr, "LFM_ADMISSION_FAIL %s exceeds trunk extent\n", role);
                return -1;
            }
            if (r == LFM_ROLE_OUTPUT_HEAD_TIED) {
                if (!have_embed || memcmp(&c, &embed, sizeof c)) {
                    fprintf(stderr, "LFM_ADMISSION_FAIL tied head does not reference the embedding payload\n");
                    return -1;
                }
                a->tied_head++;
            } else if (r == LFM_ROLE_EMBEDDING) {
                if (enc != LFM_TENSOR_ENCODING_AFFINE_Q6) {
                    fprintf(stderr, "LFM_ADMISSION_FAIL embedding is not 6-bit affine\n");
                    return -1;
                }
                embed = c;
                have_embed = 1;
            }
            a->role_seen[r]++;
            a->t_lines++;
        } else if (kind == 'E') {
            char role[48];
            unsigned enc, rows, cols;
            unsigned long long vo, vb, so, sb, bo, bb;
            int layer, expert, used = 0;
            LfmComponents c;
            int64_t slot, end;
            memset(&c, 0, sizeof c);
            if (sscanf(line, "E %47s %d %d %u %u %u %llu %llu %llu %llu %llu %llu%n",
                       role, &layer, &expert, &enc, &rows, &cols,
                       &vo, &vb, &so, &sb, &bo, &bb, &used) != 12 ||
                line[used] != '\n') {
                fprintf(stderr, "LFM_ADMISSION_FAIL malformed E line\n");
                return -1;
            }
            c.value_offset = vo; c.value_bytes = vb;
            c.scale_offset = so; c.scale_bytes = sb;
            c.bias_offset = bo; c.bias_bytes = bb;
            int r = expert_role_index(role);
            if (r < 0 || layer < LFM_DENSE_LAYERS || layer >= LFM_LAYERS ||
                expert < 0 || expert >= LFM_EXPERTS) {
                fprintf(stderr, "LFM_ADMISSION_FAIL bad expert entry %s %d %d\n", role, layer, expert);
                return -1;
            }
            if (enc != LFM_TENSOR_ENCODING_AFFINE_Q4) {
                fprintf(stderr, "LFM_ADMISSION_FAIL expert %s is not 4-bit affine\n", role);
                return -1;
            }
            slot = (int64_t)LFM_POOL_HEADER_BYTES
                 + ((int64_t)(layer - LFM_DENSE_LAYERS) * LFM_EXPERTS + expert)
                   * (int64_t)LFM_EXPERT_BYTES;
            end = slot + LFM_EXPERT_BYTES;
            if (c.value_offset < (uint64_t)slot || c.bias_offset + c.bias_bytes > (uint64_t)end ||
                c.value_offset + c.value_bytes > (uint64_t)end ||
                c.scale_offset + c.scale_bytes > (uint64_t)end) {
                fprintf(stderr, "LFM_ADMISSION_FAIL expert %s %d %d escapes its slot\n",
                        role, layer, expert);
                return -1;
            }
            if (r == LFM_EXPERT_ROLE_DOWN && c.bias_offset + c.bias_bytes == (uint64_t)end)
                a->expert_slots_ok++;
            a->expert_role_seen[r]++;
            a->e_lines++;
        } else if (kind == '\n') {
            continue;
        } else {
            fprintf(stderr, "LFM_ADMISSION_FAIL unknown index record '%c'\n", kind);
            return -1;
        }
    }

    if (a->l_lines != LFM_LAYERS || a->t_lines != LFM_T_LINES_EXPECTED ||
        a->e_lines != LFM_E_LINES_EXPECTED || a->tied_head != 1) {
        fprintf(stderr, "LFM_ADMISSION_FAIL record counts L=%u/%u T=%u/%u E=%llu/%llu tied=%u/1\n",
                a->l_lines, LFM_LAYERS, a->t_lines, LFM_T_LINES_EXPECTED,
                (unsigned long long)a->e_lines, (unsigned long long)LFM_E_LINES_EXPECTED,
                a->tied_head);
        return -1;
    }
    if (a->role_seen[LFM_ROLE_EMBEDDING] != 1 || a->role_seen[LFM_ROLE_FINAL_NORM] != 1 ||
        a->role_seen[LFM_ROLE_OPERATOR_NORM] != LFM_LAYERS ||
        a->role_seen[LFM_ROLE_FFN_NORM] != LFM_LAYERS ||
        a->role_seen[LFM_ROLE_Q] != LFM_ATTENTION_LAYERS ||
        a->role_seen[LFM_ROLE_K] != LFM_ATTENTION_LAYERS ||
        a->role_seen[LFM_ROLE_V] != LFM_ATTENTION_LAYERS ||
        a->role_seen[LFM_ROLE_O] != LFM_ATTENTION_LAYERS ||
        a->role_seen[LFM_ROLE_Q_NORM] != LFM_ATTENTION_LAYERS ||
        a->role_seen[LFM_ROLE_K_NORM] != LFM_ATTENTION_LAYERS ||
        a->role_seen[LFM_ROLE_CONV_IN] != LFM_CONV_LAYERS ||
        a->role_seen[LFM_ROLE_CONV_OUT] != LFM_CONV_LAYERS ||
        a->role_seen[LFM_ROLE_CONV_KERNEL] != LFM_CONV_LAYERS ||
        a->role_seen[LFM_ROLE_DENSE_GATE] != LFM_DENSE_LAYERS ||
        a->role_seen[LFM_ROLE_DENSE_UP] != LFM_DENSE_LAYERS ||
        a->role_seen[LFM_ROLE_DENSE_DOWN] != LFM_DENSE_LAYERS ||
        a->role_seen[LFM_ROLE_ROUTER] != LFM_MOE_LAYERS ||
        a->role_seen[LFM_ROLE_EXPERT_BIAS] != LFM_MOE_LAYERS) {
        fprintf(stderr, "LFM_ADMISSION_FAIL trunk role census mismatch\n");
        for (int i = 0; i < LFM_ROLE_COUNT; i++)
            fprintf(stderr, "  role %-18s %u\n", LFM_ROLE_NAMES[i], a->role_seen[i]);
        return -1;
    }
    if (a->expert_role_seen[LFM_EXPERT_ROLE_GATE] != LFM_EXPERT_COUNT ||
        a->expert_role_seen[LFM_EXPERT_ROLE_UP] != LFM_EXPERT_COUNT ||
        a->expert_role_seen[LFM_EXPERT_ROLE_DOWN] != LFM_EXPERT_COUNT ||
        a->expert_slots_ok != LFM_EXPERT_COUNT) {
        fprintf(stderr, "LFM_ADMISSION_FAIL expert census mismatch "
                "gate=%u up=%u down=%u slots=%llu\n",
                a->expert_role_seen[LFM_EXPERT_ROLE_GATE],
                a->expert_role_seen[LFM_EXPERT_ROLE_UP],
                a->expert_role_seen[LFM_EXPERT_ROLE_DOWN],
                (unsigned long long)a->expert_slots_ok);
        return -1;
    }
    return 0;
}

static uint64_t peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0)
        return 0u;
#if defined(__APPLE__)
    return (uint64_t)usage.ru_maxrss;
#else
    return (uint64_t)usage.ru_maxrss * UINT64_C(1024);
#endif
}

static int read_probe_prompt(int fd, char *buffer, size_t capacity,
                             size_t *length_out) {
    struct stat st;
    ssize_t got;
    if (fd < 0 || !buffer || capacity < 2u || !length_out ||
        fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 1 ||
        (uint64_t)st.st_size >= (uint64_t)capacity)
        return -1;
    got = pread(fd, buffer, (size_t)st.st_size, 0);
    if (got != st.st_size || memchr(buffer, '\0', (size_t)got))
        return -1;
    buffer[got] = '\0';
    *length_out = (size_t)got;
    return 0;
}

int main(int argc, char **argv) {
    struct stat trunk_st, pool_st, index_st, tok_st;
    LfmAudit audit;
    FILE *index;
    void *trunk = MAP_FAILED;
    unsigned char header[LFM_POOL_HEADER_BYTES];
    uint64_t fields[5];
    int64_t bases[LFM_LAYERS];
    uint32_t endian = 1;
    int rc = 2;

    if (argc != 11 && argc != 12 && argc != 13) {
        fprintf(stderr, "usage: lfm-server TRUNK_FD POOL_FD INDEX_FD TOKENIZER_FD "
                        "PACKAGE_SHA CTX B W SLOTS RSS [PROMPT_FD [OUTPUT_TOKENS]]\n");
        return 2;
    }
    if (*(unsigned char *)&endian != 1 || sizeof(float) != 4) {
        fprintf(stderr, "LFM_ADMISSION_FAIL host is not little-endian IEEE-754\n");
        return 2;
    }
    if (strlen(argv[5]) != 64) {
        fprintf(stderr, "LFM_ADMISSION_FAIL package sha is not 64 hex characters\n");
        return 2;
    }

    if (read_regular(atoi(argv[1]), &trunk_st, "trunk.bin") ||
        read_regular(atoi(argv[2]), &pool_st, "pool.bin") ||
        read_regular(atoi(argv[3]), &index_st, "index.txt") ||
        read_regular(atoi(argv[4]), &tok_st, "tokenizer") ||
        tok_st.st_size < 1)
        return 2;

    /* pool.bin: fixed header, then every expert slot at a fixed address. */
    if ((uint64_t)pool_st.st_size != LFM_POOL_BYTES) {
        fprintf(stderr, "LFM_ADMISSION_FAIL pool extent %lld != %llu\n",
                (long long)pool_st.st_size, (unsigned long long)LFM_POOL_BYTES);
        return 2;
    }
    if (pread(atoi(argv[2]), header, sizeof header, 0) != (ssize_t)sizeof header) {
        fprintf(stderr, "LFM_ADMISSION_FAIL pool header unreadable\n");
        return 2;
    }
    if (memcmp(header, LFM_POOL_MAGIC, strlen(LFM_POOL_MAGIC))) {
        fprintf(stderr, "LFM_ADMISSION_FAIL pool magic mismatch\n");
        return 2;
    }
    memcpy(fields, header + 16, sizeof fields);
    memcpy(bases, header + 16 + sizeof fields, sizeof bases);
    if (fields[0] != LFM_EXPERT_BYTES || fields[1] != LFM_LAYERS ||
        fields[2] != LFM_EXPERTS || fields[3] != LFM_MOE_LAYERS ||
        fields[4] != LFM_EXPERT_COUNT) {
        fprintf(stderr, "LFM_ADMISSION_FAIL pool geometry mismatch\n");
        return 2;
    }
    for (int layer = 0; layer < LFM_LAYERS; layer++) {
        int64_t want = layer < LFM_DENSE_LAYERS
                     ? -1
                     : (int64_t)(layer - LFM_DENSE_LAYERS) * LFM_EXPERTS;
        if (bases[layer] != want) {
            fprintf(stderr, "LFM_ADMISSION_FAIL pool layer base %d = %lld want %lld\n",
                    layer, (long long)bases[layer], (long long)want);
            return 2;
        }
    }

    /* The trunk is the only payload this entry maps; it is read-only and the
     * mapping is discarded on exit. */
    trunk = mmap(NULL, (size_t)trunk_st.st_size, PROT_READ, MAP_PRIVATE, atoi(argv[1]), 0);
    if (trunk == MAP_FAILED) {
        fprintf(stderr, "LFM_ADMISSION_FAIL trunk mapping refused: %s\n", strerror(errno));
        return 2;
    }
    (void)trunk;

    index = fdopen(dup(atoi(argv[3])), "r");
    if (!index) {
        fprintf(stderr, "LFM_ADMISSION_FAIL index open refused\n");
        munmap(trunk, (size_t)trunk_st.st_size);
        return 2;
    }
    rc = audit_index(index, &audit) ? 2 : 0;
    fclose(index);
    munmap(trunk, (size_t)trunk_st.st_size);
    if (rc)
        return rc;

    printf("LFM_PACKAGE_VERIFIED abi=%s trunk_bytes=%llu pool_bytes=%llu "
           "expert_slots=%llu layers=%u attention_layers=%u conv_layers=%u "
           "tied_head=embedding package_sha=%s\n",
           LFM_INDEX_ABI, (unsigned long long)audit.trunk_bytes,
           (unsigned long long)audit.pool_bytes,
           (unsigned long long)audit.expert_slots_ok, LFM_LAYERS,
           LFM_ATTENTION_LAYERS, LFM_CONV_LAYERS, argv[5]);

    /* Run a bounded generation from a fixed prompt. This proves the complete
     * descriptor, program, executor, model-owned layer seam, tied head, and
     * sampler chain before an HTTP serving surface is added. */
    {
        static const char PROBE_PROMPT[] = "The capital of France is";
        LfmRuntime *rt = NULL;
        char prompt_source[LFM_PROBE_SOURCE_CAP];
        const char *probe_prompt = PROBE_PROMPT;
        size_t prompt_source_bytes = sizeof PROBE_PROMPT - 1u;
        int32_t prompt[LFM_PROBE_PROMPT_CAP];
        int32_t out[LFM_PROBE_OUTPUT_CAP];
        char decoded[LFM_PROBE_TEXT_CAP];
        LfmRuntimeMetrics metrics;
        LfmRuntimeStateDigests digests;
        uint64_t rss_bytes;
        uint32_t output_tokens = LFM_PROBE_TOKENS;
        int encoded, n, decoded_n, i;

        if (argc == 13) {
            char *end = NULL;
            unsigned long parsed = strtoul(argv[12], &end, 10);
            if (!end || end == argv[12] || *end != '\0' || parsed < 1u ||
                parsed > LFM_PROBE_OUTPUT_CAP) {
                fprintf(stderr, "LFM_ADMISSION_FAIL output token cap refused\n");
                return 2;
            }
            output_tokens = (uint32_t)parsed;
        }
        if (argc >= 12) {
            if (read_probe_prompt(atoi(argv[11]), prompt_source,
                    sizeof prompt_source, &prompt_source_bytes) != 0) {
                fprintf(stderr, "LFM_ADMISSION_FAIL prompt fd refused\n");
                return 2;
            }
            probe_prompt = prompt_source;
        }
        if (lfm_runtime_open(&rt, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
                             atoi(argv[4]), (uint32_t)strtoul(argv[6], NULL, 10),
                             (uint32_t)strtoul(argv[7], NULL, 10),
                             (uint32_t)strtoul(argv[8], NULL, 10),
                             atoi(argv[9]),
                             (uint64_t)strtoull(argv[10], NULL, 10))) {
            fprintf(stderr, "LFM_UNSUPPORTED runtime-open\n");
            puts("FATAL");
            fflush(stdout);
            return 3;
        }
        encoded = lfm_runtime_encode(rt, probe_prompt, prompt,
                                     LFM_PROBE_PROMPT_CAP);
        if (encoded < 1) {
            fprintf(stderr, "LFM_UNSUPPORTED tokenize\n");
            lfm_runtime_close(rt);
            puts("FATAL");
            fflush(stdout);
            return 3;
        }
        if (argc >= 12)
            printf("LFM_PROMPT source_fd=%s bytes=%llu tokens=%d\n",
                   argv[11], (unsigned long long)prompt_source_bytes, encoded);
        else
            printf("LFM_PROMPT text=%s tokens=%d\n", probe_prompt, encoded);
        n = lfm_runtime_generate(rt, prompt, (uint32_t)encoded,
                                 output_tokens, out);
        if (n < 0) {
            fprintf(stderr, "LFM_UNSUPPORTED generate\n");
            lfm_runtime_close(rt);
            puts("FATAL");
            fflush(stdout);
            return 3;
        }
        decoded_n = lfm_runtime_decode(rt, out, (uint32_t)n, decoded,
                                       sizeof decoded);
        if (lfm_runtime_last_metrics(rt, &metrics) != 0 ||
            lfm_runtime_state_digests(rt, &digests) != 0) {
            fprintf(stderr, "LFM_UNSUPPORTED metrics/state-digest\n");
            lfm_runtime_close(rt);
            puts("FATAL");
            fflush(stdout);
            return 3;
        }
        rss_bytes = peak_rss_bytes();
        lfm_runtime_close(rt);
        if (decoded_n < 0) {
            fprintf(stderr, "LFM_UNSUPPORTED detokenize\n");
            puts("FATAL");
            fflush(stdout);
            return 3;
        }
        for (i = 0; i < n; i++)
            printf("TOKEN %d\n", out[i]);
        printf("TEXT %s\n", decoded);
        printf("LFM_STATE position=%u transition_generation=%llu "
               "state_sha256=%s kv_sha256=%s model_state_sha256=%s "
               "logits_sha256=%s committed_kv_bytes=%llu "
               "model_state_bytes=%llu\n",
               digests.position,
               (unsigned long long)digests.transition_generation,
               digests.state_sha256, digests.kv_sha256,
               digests.model_state_sha256, digests.logits_sha256,
               (unsigned long long)digests.kv_bytes,
               (unsigned long long)digests.model_state_bytes);
        printf("LFM_PERFORMANCE prompt_tokens=%u completion_tokens=%u "
               "request_ns=%llu prefill_ns=%llu decode_ns=%llu "
               "prefill_ms_per_token=%.6f decode_ms_per_token=%.6f "
               "native_throughput_tokens_per_second=%.6f "
               "backend_graph_launches=%llu backend_kernel_nodes=%llu "
               "backend_host_launch_calls=%llu peak_rss_bytes=%llu\n",
               metrics.prompt_tokens, metrics.completion_tokens,
               (unsigned long long)metrics.request_ns,
               (unsigned long long)metrics.prefill_ns,
               (unsigned long long)metrics.decode_ns,
               metrics.prompt_tokens
                   ? (double)metrics.prefill_ns / 1.0e6 /
                         (double)metrics.prompt_tokens : 0.0,
               metrics.completion_tokens
                   ? (double)metrics.decode_ns / 1.0e6 /
                         (double)metrics.completion_tokens : 0.0,
               metrics.request_ns
                   ? (double)(metrics.prompt_tokens + metrics.completion_tokens) *
                         1.0e9 / (double)metrics.request_ns : 0.0,
               (unsigned long long)metrics.backend_graph_launches,
               (unsigned long long)metrics.backend_physical_kernel_nodes,
               (unsigned long long)metrics.backend_host_kernel_launch_calls,
               (unsigned long long)rss_bytes);
        printf("LFM_GENERATE_PASS tokens=%d decoded_bytes=%d\n", n, decoded_n);
    }
    puts("OK");
    fflush(stdout);
    return 0;
}
