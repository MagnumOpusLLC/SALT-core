/* moe.c -- layout loaders + real MoE compute step. */
#include "salt/moe.h"
#include "salt/kernels.h"
#include "salt/quant.h"
#include "salt/gpu.h"
#include "salt/bitmath.h"
#include "compiler.h"
#include "json.h"
#include "sha256.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *read_file(const char *path, long *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1L << 30)) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    buf[sz] = 0;
    *len_out = sz;
    return buf;
}

static int _moeacc0_dumped = 0;
static int _exp106_dumped = 0;
static int _exp106_out_dumped = 0;
static pthread_mutex_t _expdump_mu = PTHREAD_MUTEX_INITIALIZER;

/* env cache: these gates sit in exp_run (per expert-job, ~270/token)
 * and the per-layer norm path (~80/token). getenv = libc lock +
 * environ scan; values are fixed at launch -> parse once. */
static int g_nan_probe = 0, g_debug_chain = 0, g_no_norms = 0;
static pthread_once_t g_moe_env_once = PTHREAD_ONCE_INIT;
static void moe_env_init(void) {
    g_nan_probe = getenv("SALT_NAN_PROBE") != NULL;
    g_debug_chain = getenv("SALT_DEBUG_CHAIN") != NULL;
    g_no_norms = getenv("SALT_NO_NORMS") != NULL;
}

static void moe_env_ensure(void) {
    (void)pthread_once(&g_moe_env_once, moe_env_init);
}

static int dtype_of(const char *s, size_t n) {
    if (n == 3 && !memcmp(s, "F32", 3)) return 0;
    if (n == 2 && !memcmp(s, "I8", 2)) return 1;
    if (n == 7 && !memcmp(s, "F8_E4M3", 7)) return 2;
    if (n == 4 && !memcmp(s, "BF16", 4)) return 4;
    if (n == 3 && !memcmp(s, "U32", 3)) return 5;   /* MLX 4-bit packed */
    return 3;
}

/* name ends with suffix (NUL-terminated C strings) */
static int name_ends(const char *name, const char *suffix) {
    size_t nl = strlen(name), sl = strlen(suffix);
    return nl >= sl && !memcmp(name + nl - sl, suffix, sl);
}

static int json_u64_value(const JEntry *e, uint64_t *out) {
    if (!e || !out || e->type != 0 || !e->num_integral ||
        !e->num_unsigned)
        return -1;
    *out = e->unum;
    return 0;
}

static int json_u32_value(const JEntry *e, uint32_t *out) {
    uint64_t value;
    if (json_u64_value(e, &value) != 0 || value > UINT32_MAX) return -1;
    *out = (uint32_t)value;
    return 0;
}

static int json_string_equal(const JEntry *e, const char *text) {
    size_t n = strlen(text);
    return e && e->type == 1 && e->str &&
        (size_t)(e->str_end - e->str) == n && !memcmp(e->str, text, n);
}

static int pool_source_identity_load(const JDoc *doc,
                                     SaltLayoutSourceIdentity *identity) {
    const JEntry *root = json_get(doc->root, doc->nroot,
                                  "source_pool_identity");
    if (!root) return 0;
    if (root->type != 2) return -1;
    const JEntry *device = json_get(root->child, root->nchild, "device");
    const JEntry *inode = json_get(root->child, root->nchild, "inode");
    const JEntry *nbytes = json_get(root->child, root->nchild, "nbytes");
    const JEntry *mtime = json_get(root->child, root->nchild, "mtime_ns");
    const JEntry *ctime = json_get(root->child, root->nchild, "ctime_ns");
    if (json_u64_value(device, &identity->device) != 0 ||
        json_u64_value(inode, &identity->inode) != 0 ||
        json_u64_value(nbytes, &identity->nbytes) != 0 ||
        json_u64_value(mtime, &identity->mtime_ns) != 0 ||
        json_u64_value(ctime, &identity->ctime_ns) != 0)
        return -1;
    identity->present = 1;
    return 0;
}

static int pool_gpu_policy_fields(const JEntry *obj,
                                  SaltGpuTensorPolicy *policy,
                                  int require_all) {
    if (!obj || obj->type != 2 || !policy) return -1;
    const JEntry *block = json_get(obj->child, obj->nchild, "block_size");
    const JEntry *pipeline = json_get(obj->child, obj->nchild, "pipeline_id");
    const JEntry *threadgroup =
        json_get(obj->child, obj->nchild, "threadgroup_id");
    const JEntry *flags = json_get(obj->child, obj->nchild, "flags");
    if ((block && json_u32_value(block, &policy->block_size) != 0) ||
        (pipeline && json_u32_value(pipeline, &policy->pipeline_id) != 0) ||
        (threadgroup &&
         json_u32_value(threadgroup, &policy->threadgroup_id) != 0) ||
        (flags && json_u32_value(flags, &policy->flags) != 0))
        return -1;
    if (require_all && (!block || !pipeline || !threadgroup || !flags))
        return -1;
    return policy->block_size == 64 && policy->pipeline_id != 0 &&
        policy->threadgroup_id != 0 ? 0 : -1;
}

static int pool_gpu_extension_load(const JDoc *doc, SaltPoolLayout *pl,
                                   int gpu_engaged) {
    const JEntry *root = json_get(doc->root, doc->nroot, "gpu");
    if (!gpu_engaged || !root) return 0;
    const JEntry *format = json_get(doc->root, doc->nroot, "format");
    if (root->type != 2 || !pl->source_identity.present ||
        !json_string_equal(format, "q4-split-v1"))
        return -1;
    const JEntry *version = json_get(root->child, root->nchild, "version");
    const JEntry *alignment =
        json_get(root->child, root->nchild, "resource_alignment");
    const JEntry *partition =
        json_get(root->child, root->nchild, "view_partition");
    const JEntry *defaults =
        json_get(root->child, root->nchild, "pipeline_defaults");
    uint32_t parsed_version, parsed_alignment;
    if (json_u32_value(version, &parsed_version) != 0 ||
        json_u32_value(alignment, &parsed_alignment) != 0 ||
        parsed_version != 1 || parsed_alignment < 4096 ||
        (parsed_alignment & (parsed_alignment - 1)) != 0 ||
        !json_string_equal(partition, "layer") || !defaults ||
        defaults->type != 2)
        return -1;
    const JEntry *q4 = json_get(defaults->child, defaults->nchild,
                                "fmt1_bits4");
    SaltGpuTensorPolicy policy = {0};
    if (pool_gpu_policy_fields(q4, &policy, 1) != 0) return -1;
    pl->gpu.version = parsed_version;
    pl->gpu.resource_alignment = parsed_alignment;
    pl->gpu.view_partition = SALT_GPU_VIEW_PARTITION_LAYER;
    pl->gpu.q4_default = policy;
    pl->gpu.present = 1;
    return 0;
}

static int pool_gpu_override_add(SaltPoolLayout *pl, size_t *capacity,
                                 uint32_t tensor_ordinal, uint32_t layer,
                                 uint32_t expert, uint32_t component,
                                 const JEntry *gpu) {
    if (!pl->gpu.present || !gpu || gpu->type != 2) return -1;
    SaltGpuTensorPolicy policy = pl->gpu.q4_default;
    if (pool_gpu_policy_fields(gpu, &policy, 0) != 0) return -1;
    if (pl->gpu.n_override == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 8;
        if (next < *capacity || next > SIZE_MAX / sizeof(*pl->gpu.override))
            return -1;
        SaltGpuTensorOverride *grown = (SaltGpuTensorOverride *)realloc(
            pl->gpu.override, next * sizeof(*pl->gpu.override));
        if (!grown) return -1;
        pl->gpu.override = grown;
        *capacity = next;
    }
    SaltGpuTensorOverride *override =
        &pl->gpu.override[pl->gpu.n_override++];
    override->tensor_ordinal = tensor_ordinal;
    override->layer = layer;
    override->expert = expert;
    override->component = component;
    override->policy = policy;
    return 0;
}

void salt_pool_layout_free(SaltPoolLayout *pl) {
    if (!pl) return;
    free(pl->gpu.override);
    free(pl->exp);
    memset(pl, 0, sizeof *pl);
}

int salt_pool_layout_load(SaltPoolLayout *pl, const char *path,
                          const SaltCfg *cfg) {
    return salt_pool_layout_load_ex(pl, path, cfg, 0);
}

int salt_pool_layout_load_ex(SaltPoolLayout *pl, const char *path,
                             const SaltCfg *cfg, int gpu_engaged) {
    if (!pl || !path || !cfg) return -1;
    memset(pl, 0, sizeof *pl);
    long len = 0;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "pool layout: cannot read %s\n", path);
        return -1;
    }
    if (salt_sha256_bytes(buf, (size_t)len, pl->index_sha256) != 0) {
        free(buf);
        return -1;
    }
    pl->index_sha256_valid = 1;
    JDoc *doc = json_parse(buf, (size_t)len);
    if (!doc) {
        fprintf(stderr, "pool layout: %s not parseable\n", path);
        free(buf);
        return -1;
    }
    int rc = -1;
    size_t override_capacity = 0;
    const JEntry *nl = json_get(doc->root, doc->nroot, "n_layers");
    const JEntry *ne = json_get(doc->root, doc->nroot, "n_experts");
    const JEntry *eb = json_get(doc->root, doc->nroot, "expert_nbytes");
    const JEntry *ts = json_get(doc->root, doc->nroot, "tensors");
    uint64_t n_layers, n_experts, expert_nbytes;
    if (json_u64_value(nl, &n_layers) != 0 ||
        json_u64_value(ne, &n_experts) != 0 ||
        json_u64_value(eb, &expert_nbytes) != 0 || !ts || ts->type != 3 ||
        n_layers == 0 || n_layers > INT_MAX || n_experts == 0 ||
        n_experts > INT_MAX || expert_nbytes == 0 ||
        expert_nbytes > INT64_MAX) {
        fprintf(stderr, "pool layout: %s has invalid root geometry\n", path);
        goto done;
    }
    pl->n_layers = (int)n_layers;
    pl->n_experts = (int)n_experts;
    pl->expert_nbytes = (int64_t)expert_nbytes;
    if (pl->n_layers != cfg->n_layers || pl->n_experts != cfg->n_experts) {
        fprintf(stderr, "pool layout: %dx%d vs config %dx%d\n",
                pl->n_layers, pl->n_experts, cfg->n_layers, cfg->n_experts);
        goto done;
    }
    {
        const JEntry *sr = json_get(doc->root, doc->nroot,
                                    "expert_scale_rows");
        const JEntry *sc = json_get(doc->root, doc->nroot,
                                    "expert_scale_cols");
        uint64_t value;
        if (sr && (json_u64_value(sr, &value) != 0 || value > INT_MAX))
            goto done;
        if (sr) pl->expert_scale_rows = (int)value;
        if (sc && (json_u64_value(sc, &value) != 0 || value > INT_MAX))
            goto done;
        if (sc) pl->expert_scale_cols = (int)value;
    }
    if (pool_source_identity_load(doc, &pl->source_identity) != 0 ||
        pool_gpu_extension_load(doc, pl, gpu_engaged) != 0) {
        fprintf(stderr, "pool layout: invalid unified metadata in %s\n", path);
        goto done;
    }
    size_t slot_count = (size_t)pl->n_layers * (size_t)pl->n_experts;
    if (slot_count / (size_t)pl->n_experts != (size_t)pl->n_layers ||
        slot_count > SIZE_MAX / sizeof(SaltExpertLayout))
        goto done;
    if (slot_count > (UINT64_MAX - 24) / expert_nbytes) goto done;
    uint64_t source_nbytes = 24 + (uint64_t)slot_count * expert_nbytes;
    if (pl->source_identity.present &&
        pl->source_identity.nbytes != source_nbytes) {
        fprintf(stderr, "pool layout: source extent identity mismatch\n");
        goto done;
    }
    pl->exp = (SaltExpertLayout *)calloc(slot_count, sizeof(SaltExpertLayout));
    if (!pl->exp) goto done;

    for (int i = 0; i < ts->nchild; i++) {
        const JEntry *e = &ts->child[i];
        if (e->type != 2) goto tensor_invalid;
        const JEntry *lay = json_get(e->child, e->nchild, "layer");
        const JEntry *exp = json_get(e->child, e->nchild, "expert");
        const JEntry *shp = json_get(e->child, e->nchild, "shape");
        const JEntry *vo = json_get(e->child, e->nchild, "v_off");
        const JEntry *so = json_get(e->child, e->nchild, "s_off");
        const JEntry *vn = json_get(e->child, e->nchild, "v_nbytes");
        const JEntry *sn = json_get(e->child, e->nchild, "s_nbytes");
        uint64_t layer, expert, v_off, s_off, v_nbytes, s_nbytes;
        if (json_u64_value(lay, &layer) != 0 ||
            json_u64_value(exp, &expert) != 0 || !shp || shp->type != 3 ||
            shp->nchild < 1 || shp->nchild > 4 ||
            json_u64_value(vo, &v_off) != 0 ||
            json_u64_value(so, &s_off) != 0 ||
            json_u64_value(vn, &v_nbytes) != 0 ||
            json_u64_value(sn, &s_nbytes) != 0 ||
            layer >= n_layers || expert >= n_experts ||
            v_nbytes == 0 || v_nbytes > LONG_MAX ||
            s_nbytes == 0 || s_nbytes > LONG_MAX)
            goto tensor_invalid;
        size_t slot_ordinal = (size_t)layer * pl->n_experts + (size_t)expert;
        SaltExpertLayout *el = &pl->exp[slot_ordinal];
        if (el->n >= SALT_MAX_TENSORS_PER_EXPERT) goto tensor_invalid;
        SaltMoETensor tensor;
        memset(&tensor, 0, sizeof tensor);
        tensor.rank = shp->nchild;
        tensor.rel_b = -1;
        tensor.bsize = 32;
        tensor.bits = 4;
        uint64_t rc_count = 1;
        for (int d = 0; d < tensor.rank; d++) {
            uint64_t dim;
            if (json_u64_value(&shp->child[d], &dim) != 0 || dim == 0 ||
                dim > LONG_MAX || rc_count > INT64_MAX / dim)
                goto tensor_invalid;
            tensor.dims[d] = (long)dim;
            rc_count *= dim;
        }
        if ((int64_t)rc_count > pl->max_rc) pl->max_rc = (int64_t)rc_count;
        const JEntry *ss = json_get(e->child, e->nchild, "s_shape");
        if (ss) {
            if (ss->type != 3 || ss->nchild < 1 || ss->nchild > 2)
                goto tensor_invalid;
            for (int d = 0; d < ss->nchild; d++) {
                uint64_t dim;
                if (json_u64_value(&ss->child[d], &dim) != 0 || dim == 0 ||
                    dim > LONG_MAX)
                    goto tensor_invalid;
                tensor.s_dims[d] = (long)dim;
                tensor.s_rank++;
            }
        }
        uint64_t slot_off = 24 + (uint64_t)slot_ordinal * expert_nbytes;
        if (v_off < slot_off || s_off < slot_off ||
            v_off - slot_off > LONG_MAX || s_off - slot_off > LONG_MAX)
            goto tensor_invalid;
        uint64_t rel_v = v_off - slot_off;
        uint64_t rel_s = s_off - slot_off;
        if (rel_v > expert_nbytes || v_nbytes > expert_nbytes - rel_v ||
            rel_s > expert_nbytes || s_nbytes > expert_nbytes - rel_s)
            goto tensor_invalid;
        tensor.rel_v = (long)rel_v;
        tensor.rel_s = (long)rel_s;
        tensor.v_nbytes = (long)v_nbytes;
        tensor.s_nbytes = (long)s_nbytes;

        const JEntry *fm = json_get(e->child, e->nchild, "fmt");
        if (fm) {
            uint32_t format;
            if (json_u32_value(fm, &format) != 0 || format > 2)
                goto tensor_invalid;
            tensor.fmt = (int)format;
        }
        const JEntry *bits = json_get(e->child, e->nchild, "bits");
        if (bits) {
            uint32_t width;
            if (json_u32_value(bits, &width) != 0 ||
                (width != 4 && width != 8))
                goto tensor_invalid;
            tensor.bits = (int)width;
        }
        const JEntry *bo = json_get(e->child, e->nchild, "b_off");
        const JEntry *bn = json_get(e->child, e->nchild, "b_nbytes");
        if (!!bo != !!bn) goto tensor_invalid;
        if (bo) {
            uint64_t b_off, b_nbytes;
            if (json_u64_value(bo, &b_off) != 0 ||
                json_u64_value(bn, &b_nbytes) != 0 || b_nbytes == 0 ||
                b_nbytes > LONG_MAX || b_off < slot_off ||
                b_off - slot_off > LONG_MAX)
                goto tensor_invalid;
            uint64_t rel_b = b_off - slot_off;
            if (rel_b > expert_nbytes || b_nbytes > expert_nbytes - rel_b ||
                (tensor.fmt == 1 && b_nbytes != s_nbytes))
                goto tensor_invalid;
            tensor.rel_b = (long)rel_b;
        }
        if (el->n == 0) {
            const JEntry *chain = json_get(e->child, e->nchild, "chain");
            el->chain = json_string_equal(chain, "qwen3") ? 1 : 0;
        }
        const JEntry *gpu = json_get(e->child, e->nchild, "gpu");
        if (gpu_engaged && gpu &&
            pool_gpu_override_add(pl, &override_capacity, (uint32_t)i,
                                  (uint32_t)layer, (uint32_t)expert,
                                  (uint32_t)el->n, gpu) != 0)
            goto tensor_invalid;
        el->t[el->n++] = tensor;
        continue;

tensor_invalid:
        fprintf(stderr, "pool layout: invalid tensor record %d in %s\n",
                i, path);
        goto done;
    }
    rc = 0;

done:
    json_free(doc);
    free(buf);
    if (rc != 0) salt_pool_layout_free(pl);
    return rc;
}

int salt_trunk_layout_load(SaltTrunkLayout *tl, const char *path) {
    long len;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "trunk layout: cannot read %s\n", path);
        return -1;
    }
    JDoc *doc = json_parse(buf, (size_t)len);
    if (!doc) {
        fprintf(stderr, "trunk layout: %s not parseable\n", path);
        free(buf);
        return -1;
    }
    const JEntry *nl = json_get(doc->root, doc->nroot, "n_layers");
    const JEntry *ls = json_get(doc->root, doc->nroot, "layers");
    if (!nl || !ls || ls->type != 3) {
        fprintf(stderr, "trunk layout: %s missing keys\n", path);
        json_free(doc); free(buf);
        return -1;
    }
    memset(tl, 0, sizeof *tl);
    tl->n_layers = (int)nl->inum;
    if (tl->n_layers < 1 || tl->n_layers > SALT_MAX_LAYERS) {
        fprintf(stderr, "trunk layout: n_layers %d out of range\n",
                tl->n_layers);
        json_free(doc); free(buf);
        return -1;
    }
    for (int L = 0; L < SALT_MAX_LAYERS; L++) {
        tl->gate[L] = tl->down[L] = tl->up[L] = -1;
        tl->gate_bias[L] = -1;
        tl->se_g[L] = tl->se_gs[L] = tl->se_gb[L] = -1;
        tl->se_u[L] = tl->se_us[L] = tl->se_ub[L] = -1;
        tl->se_d[L] = tl->se_ds[L] = tl->se_db[L] = -1;
        tl->se_r[L] = tl->se_rs[L] = tl->se_rb[L] = -1;
        tl->attn_qn[L] = tl->attn_kvn[L] = -1;
        tl->attn_wqa[L] = tl->attn_wqa_s[L] = -1;
        tl->attn_wqb[L] = tl->attn_wqb_s[L] = -1;
        tl->attn_wkv[L] = tl->attn_wkv_s[L] = -1;
        tl->attn_woa[L] = tl->attn_woa_s[L] = -1;
        tl->attn_wob[L] = tl->attn_wob_s[L] = -1;
        tl->attn_woc[L] = tl->attn_woc_s[L] = -1;
        tl->attn_sink[L] = -1;
        tl->attn_norm[L] = -1;
        tl->ffn_norm[L] = -1;
        tl->hc_attn_fn[L] = tl->hc_attn_base[L] = tl->hc_attn_scale[L] = -1;
        tl->hc_ffn_fn[L] = tl->hc_ffn_base[L] = tl->hc_ffn_scale[L] = -1;
        /* Qwen3.5 attention roles: GQA (self_attn) + linear (linear_attn) */
        tl->q3_q[L] = tl->q3_qs[L] = tl->q3_qb[L] = -1;
        tl->q3_k[L] = tl->q3_ks[L] = tl->q3_kb[L] = -1;
        tl->q3_v[L] = tl->q3_vs[L] = tl->q3_vb[L] = -1;
        tl->q3_o[L] = tl->q3_os[L] = tl->q3_ob[L] = -1;
        tl->q3_qn[L] = tl->q3_kn[L] = -1;
        tl->q3_conv[L] = tl->q3_a_log[L] = tl->q3_dt[L] = -1;
        tl->q3_pqkv[L] = tl->q3_pqkvs[L] = tl->q3_pqkvb[L] = -1;
        tl->q3_pz[L] = tl->q3_pzs[L] = tl->q3_pzb[L] = -1;
        tl->q3_pa[L] = tl->q3_pas[L] = tl->q3_pab[L] = -1;
        tl->q3_pb[L] = tl->q3_pbs[L] = tl->q3_pbb[L] = -1;
        tl->q3_opa[L] = tl->q3_opas[L] = tl->q3_opab[L] = -1;
        tl->q3_lnorm[L] = -1;
    }
    tl->final_norm = -1;
    tl->hc_head_fn = tl->hc_head_base = tl->hc_head_scale = -1;
    tl->kvlat = 0;

    int total = 0;
    for (int i = 0; i < ls->nchild; i++) {
        const JEntry *ly = &ls->child[i];
        if (ly->type != 2 || !ly->child || ly->nchild < 1) {
            fprintf(stderr, "trunk layout: layer entry %d malformed\n", i);
            json_free(doc); free(buf);
            return -1;
        }
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (tss && tss->type == 3) total += tss->nchild;
    }
    fprintf(stderr, "trunk layout: %d layer entries, %d tensors total\n",
            ls->nchild, total);
    tl->t = (SaltTrunkTensor *)calloc((size_t)total,
                                      sizeof(SaltTrunkTensor));
    tl->t_off = (int *)calloc((size_t)(tl->n_layers + 1), sizeof(int));
    if (!tl->t || !tl->t_off) {
        json_free(doc); free(buf);
        return -1;
    }
    int k = 0;
    for (int L = 0; L < tl->n_layers && L < SALT_MAX_LAYERS; L++) {
        tl->t_off[L] = k;
        const JEntry *ly = NULL;
        for (int i = 0; i < ls->nchild; i++) {
            const JEntry *cand = &ls->child[i];
            const JEntry *lid = json_get(cand->child, cand->nchild, "layer");
            if (lid && lid->inum == L) { ly = cand; break; }
        }
        if (!ly) continue;
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (!tss || tss->type != 3) continue;
        if (!tss->child && tss->nchild > 0) {
            fprintf(stderr, "trunk layout: layer %d tensors array dangles\n", L);
            json_free(doc); free(buf);
            return -1;
        }
        for (int i = 0; i < tss->nchild; i++) {
            const JEntry *e = &tss->child[i];
            if (k >= total) {
                fprintf(stderr, "trunk layout: tensor overflow at layer %d "
                        "entry %d (k=%d total=%d)\n", L, i, k, total);
                json_free(doc); free(buf);
                return -1;
            }
            if (e->type != 2 || !e->child) {
                fprintf(stderr, "trunk layout: layer %d tensor %d malformed\n",
                        L, i);
                json_free(doc); free(buf);
                return -1;
            }
            SaltTrunkTensor *tt = &tl->t[k++];
            const JEntry *nm = json_get(e->child, e->nchild, "n");
            const JEntry *dt = json_get(e->child, e->nchild, "dtype");
            const JEntry *shp = json_get(e->child, e->nchild, "shape");
            const JEntry *of = json_get(e->child, e->nchild, "off");
            const JEntry *nb = json_get(e->child, e->nchild, "nbytes");
            tt->name[0] = 0;
            if (nm && nm->type == 1) {
                size_t nn = (size_t)(nm->str_end - nm->str);
                if (nn > 95) nn = 95;
                memcpy(tt->name, nm->str, nn);
                tt->name[nn] = 0;
            }
            if (dt && dt->type == 1)
                tt->dtype = dtype_of(dt->str,
                                     (size_t)(dt->str_end - dt->str));
            if (shp) {
                tt->rank = shp->nchild;
                if (tt->rank > 4) tt->rank = 4;
                for (int d2 = 0; d2 < tt->rank; d2++)
                    tt->dims[d2] = (long)shp->child[d2].inum;
            }
            if (of) tt->off = of->inum;
            if (nb) tt->nbytes = nb->inum;
            {
                const JEntry *bs = json_get(e->child, e->nchild, "bits");
                tt->bits = (bs && bs->type == 0) ? (int)bs->inum : 4;
                if (tt->bits != 4 && tt->bits != 8) tt->bits = 4;
            }
            /* roles: exact ffn leaf names only. The gate weight is BF16
             * on the real checkpoint with an F32 bias; down/up are the
             * latent projections (fp32 when present). Match the leaf
             * exactly so attn.*.wgate.weight can never impersonate the
             * ffn router. */
            if (tt->name[0]) {
                /* Qwen3.5 roles: mlp.gate.weight = router, self_attn
                 * q/k/v/o = attention, mlp.shared_expert.* = resident
                 * shared MLP. These names are unique (no other tensor
                 * ends with .mlp.gate.weight), so exact-leaf matching
                 * stays unambiguous. */
                if (name_ends(tt->name, ".mlp.gate.weight") &&
                    (tt->dtype == 0 || tt->dtype == 4 || tt->dtype == 5)) {
                    if (tl->gate[L] < 0) tl->gate[L] = k - 1;
                } else if (name_ends(tt->name, ".mlp.gate.biases") &&
                           tt->dtype == 0) {
                    if (tl->gate_bias[L] < 0) tl->gate_bias[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.gate_proj.weight")) {
                    if (tl->se_g[L] < 0) tl->se_g[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.gate_proj.scales")) {
                    if (tl->se_gs[L] < 0) tl->se_gs[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.gate_proj.biases")) {
                    if (tl->se_gb[L] < 0) tl->se_gb[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.up_proj.weight")) {
                    if (tl->se_u[L] < 0) tl->se_u[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.up_proj.scales")) {
                    if (tl->se_us[L] < 0) tl->se_us[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.up_proj.biases")) {
                    if (tl->se_ub[L] < 0) tl->se_ub[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.down_proj.weight")) {
                    if (tl->se_d[L] < 0) tl->se_d[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.down_proj.scales")) {
                    if (tl->se_ds[L] < 0) tl->se_ds[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.down_proj.biases")) {
                    if (tl->se_db[L] < 0) tl->se_db[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert_gate.weight")) {
                    if (tl->se_r[L] < 0) tl->se_r[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert_gate.scales")) {
                    if (tl->se_rs[L] < 0) tl->se_rs[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert_gate.biases")) {
                    if (tl->se_rb[L] < 0) tl->se_rb[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.gate.weight") &&
                    (tt->dtype == 0 || tt->dtype == 4)) {
                    if (tl->gate[L] < 0) tl->gate[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.gate.bias") &&
                           tt->dtype == 0) {
                    if (tl->gate_bias[L] < 0) tl->gate_bias[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.down.weight") &&
                           tt->dtype == 0) {
                    if (tl->down[L] < 0) tl->down[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.up.weight") &&
                           tt->dtype == 0) {
                    if (tl->up[L] < 0) tl->up[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.q_norm.weight")) {
                    if (tl->attn_qn[L] < 0) tl->attn_qn[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.kv_norm.weight")) {
                    if (tl->attn_kvn[L] < 0) tl->attn_kvn[L] = k - 1;
                } else if (name_ends(tt->name, ".attn_norm.weight") ||
                           name_ends(tt->name, ".input_layernorm.weight")) {
                    if (tl->attn_norm[L] < 0) tl->attn_norm[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn_norm.weight") ||
                           name_ends(tt->name,
                                     ".post_attention_layernorm.weight")) {
                    if (tl->ffn_norm[L] < 0) tl->ffn_norm[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_a.weight")) {
                    if (tl->attn_wqa[L] < 0) tl->attn_wqa[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_a.scale")) {
                    if (tl->attn_wqa_s[L] < 0) tl->attn_wqa_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_b.weight")) {
                    if (tl->attn_wqb[L] < 0) tl->attn_wqb[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_b.scale")) {
                    if (tl->attn_wqb_s[L] < 0) tl->attn_wqb_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wkv.weight")) {
                    if (tl->attn_wkv[L] < 0) tl->attn_wkv[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wkv.scale")) {
                    if (tl->attn_wkv_s[L] < 0) tl->attn_wkv_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_a.weight")) {
                    if (tl->attn_woa[L] < 0) tl->attn_woa[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_a.scale")) {
                    if (tl->attn_woa_s[L] < 0) tl->attn_woa_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_b.weight")) {
                    if (tl->attn_wob[L] < 0) tl->attn_wob[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_b.scale")) {
                    if (tl->attn_wob_s[L] < 0) tl->attn_wob_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_c.weight")) {
                    if (tl->attn_woc[L] < 0) tl->attn_woc[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_c.scale")) {
                    if (tl->attn_woc_s[L] < 0) tl->attn_woc_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.attn_sink") &&
                           tt->dtype == 0) {
                    if (tl->attn_sink[L] < 0) tl->attn_sink[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_fn")) {
                    if (tl->hc_attn_fn[L] < 0) tl->hc_attn_fn[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_base")) {
                    if (tl->hc_attn_base[L] < 0) tl->hc_attn_base[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_scale")) {
                    if (tl->hc_attn_scale[L] < 0)
                        tl->hc_attn_scale[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_fn")) {
                    if (tl->hc_ffn_fn[L] < 0) tl->hc_ffn_fn[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_base")) {
                    if (tl->hc_ffn_base[L] < 0) tl->hc_ffn_base[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_scale")) {
                    if (tl->hc_ffn_scale[L] < 0)
                        tl->hc_ffn_scale[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_head_fn")) {
                    if (tl->hc_head_fn < 0) tl->hc_head_fn = k - 1;
                } else if (name_ends(tt->name, ".hc_head_base")) {
                    if (tl->hc_head_base < 0) tl->hc_head_base = k - 1;
                } else if (name_ends(tt->name, ".hc_head_scale")) {
                    if (tl->hc_head_scale < 0) tl->hc_head_scale = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_proj.weight")) {
                    if (tl->q3_q[L] < 0) tl->q3_q[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_proj.scales")) {
                    if (tl->q3_qs[L] < 0) tl->q3_qs[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_proj.biases")) {
                    if (tl->q3_qb[L] < 0) tl->q3_qb[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_proj.weight")) {
                    if (tl->q3_k[L] < 0) tl->q3_k[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_proj.scales")) {
                    if (tl->q3_ks[L] < 0) tl->q3_ks[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_proj.biases")) {
                    if (tl->q3_kb[L] < 0) tl->q3_kb[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.v_proj.weight")) {
                    if (tl->q3_v[L] < 0) tl->q3_v[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.v_proj.scales")) {
                    if (tl->q3_vs[L] < 0) tl->q3_vs[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.v_proj.biases")) {
                    if (tl->q3_vb[L] < 0) tl->q3_vb[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.o_proj.weight")) {
                    if (tl->q3_o[L] < 0) tl->q3_o[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.o_proj.scales")) {
                    if (tl->q3_os[L] < 0) tl->q3_os[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.o_proj.biases")) {
                    if (tl->q3_ob[L] < 0) tl->q3_ob[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_norm.weight")) {
                    if (tl->q3_qn[L] < 0) tl->q3_qn[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_norm.weight")) {
                    if (tl->q3_kn[L] < 0) tl->q3_kn[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.conv1d.weight")) {
                    if (tl->q3_conv[L] < 0) tl->q3_conv[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.A_log")) {
                    if (tl->q3_a_log[L] < 0) tl->q3_a_log[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.dt_bias")) {
                    if (tl->q3_dt[L] < 0) tl->q3_dt[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_qkv.weight")) {
                    if (tl->q3_pqkv[L] < 0) tl->q3_pqkv[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_qkv.scales")) {
                    if (tl->q3_pqkvs[L] < 0) tl->q3_pqkvs[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_qkv.biases")) {
                    if (tl->q3_pqkvb[L] < 0) tl->q3_pqkvb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_z.weight")) {
                    if (tl->q3_pz[L] < 0) tl->q3_pz[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_z.scales")) {
                    if (tl->q3_pzs[L] < 0) tl->q3_pzs[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_z.biases")) {
                    if (tl->q3_pzb[L] < 0) tl->q3_pzb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_a.weight")) {
                    if (tl->q3_pa[L] < 0) tl->q3_pa[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_a.scales")) {
                    if (tl->q3_pas[L] < 0) tl->q3_pas[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_a.biases")) {
                    if (tl->q3_pab[L] < 0) tl->q3_pab[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_b.weight")) {
                    if (tl->q3_pb[L] < 0) tl->q3_pb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_b.scales")) {
                    if (tl->q3_pbs[L] < 0) tl->q3_pbs[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_b.biases")) {
                    if (tl->q3_pbb[L] < 0) tl->q3_pbb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.out_proj.weight")) {
                    if (tl->q3_opa[L] < 0) tl->q3_opa[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.out_proj.scales")) {
                    if (tl->q3_opas[L] < 0) tl->q3_opas[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.out_proj.biases")) {
                    if (tl->q3_opab[L] < 0) tl->q3_opab[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.norm.weight")) {
                    if (tl->q3_lnorm[L] < 0) tl->q3_lnorm[L] = k - 1;
                } else if (name_ends(tt->name, ".model.norm.weight")) {
                    /* final norm before lm_head (synthetic layer) */
                    if (tl->final_norm < 0) tl->final_norm = k - 1;
                }
            }
        }
    }
    tl->t_off[tl->n_layers] = k;
    json_free(doc);
    free(buf);
    /* Typed reads (F32/BF16) require aligned offsets: misaligned ones
     * are UB that clang -O2 exploits (widened loads past the buffer,
     * nondeterministic garbage). The converter pads to 8 B; refuse
     * layouts that violate it (re-convert with the current tool). */
    for (int i = 0; i < k; i++) {
        SaltTrunkTensor *tt = &tl->t[i];
        if (tt->dtype == 0 && (tt->off & 3)) {
            fprintf(stderr,
                    "trunk layout: F32 tensor %s at misaligned off %ld "
                    "(re-convert: aligned packer required)\n",
                    tt->name, tt->off);
            return -1;
        }
        if (tt->dtype == 4 && (tt->off & 1)) {
            fprintf(stderr,
                    "trunk layout: BF16 tensor %s at misaligned off %ld "
                    "(re-convert: aligned packer required)\n",
                    tt->name, tt->off);
            return -1;
        }
    }
    /* the KV latent width: from the first layer's wkv tensor (derived
     * AFTER the role matching above; main sizes the KV cache with it) */
    tl->kvlat = 0;
    for (int L = 0; L < tl->n_layers && tl->kvlat < 1; L++)
        if (tl->attn_wkv[L] >= 0 && tl->t[tl->attn_wkv[L]].rank == 2)
            tl->kvlat = (int)tl->t[tl->attn_wkv[L]].dims[0];
    return 0;
}

void salt_topk(const float *scores, int E, int k, int *idx, float *w) {
    if (k > E) k = E;
    for (int j = 0; j < k; j++) { idx[j] = -1; w[j] = 0.0f; }
    for (int e = 0; e < E; e++) {
        float s = scores[e];
        int j = k - 1;
        while (j >= 0 && (idx[j] < 0 || s > w[j])) j--;
        if (j < k - 1) {
            for (int q = k - 2; q > j; q--) {
                idx[q + 1] = idx[q];
                w[q + 1] = w[q];
            }
            idx[j + 1] = e;
            w[j + 1] = s;
        }
    }
    /* Qwen3.5 SwitchMLP routing: softmax over ALL expert scores, keep the
     * selected top-k SOFTMAX PROBABILITIES RAW -- mlx-lm qwen3_next.py
     * renormalizes only if norm_topk_prob (absent from this model's
     * config -> default False). The previous /= sum was a deviation from
     * the reference (amplified routed experts 2-5x, destroyed the
     * attention/shared-expert balance). The selected w[] holds raw
     * top-k logits; convert to softmax probs, no renorm. */
    double mx = -1e30, sw = 0.0;
    for (int e = 0; e < E; e++)
        if ((double)scores[e] > mx) mx = (double)scores[e];
    for (int e = 0; e < E; e++)
        sw += exp((double)scores[e] - mx);
    for (int j = 0; j < k && idx[j] >= 0; j++)
        w[j] = (float)(exp((double)w[j] - mx) / sw);
    /* Qwen3.6 norm_topk_prob: the config omits the flag, and
     * mlx-lm's qwen3_5.py TextModelArgs DEFAULTS it to True --
     * the selected top-k softmax probs are RENORMALIZED (divided
     * by their sum) before the expert combine. Without this the
     * routed MoE is ~12x weak (weights ~0.07 instead of ~0.25). */
    {
        double wsum = 0.0;
        for (int j = 0; j < k && idx[j] >= 0; j++)
            wsum += (double)w[j];
        if (wsum > 0.0 && wsum < 1e30)
            for (int j = 0; j < k && idx[j] >= 0; j++)
                w[j] = (float)((double)w[j] / wsum);
    }
}

/* Parallel expert chain job (issue #5): one per topk expert, run on its
 * own thread once the slot is resident. Combine stays in selection
 * order in the caller, so results are bit-identical to the serial path. */
typedef struct {
    const SaltExpertLayout *el;
    const SaltPoolLayout *pool;
    const uint8_t *slot;
    const float *latent;
    float *out;               /* chain result, Lat floats */
    float *scratch;           /* max_rc floats, private */
    float *cur, *tmp, *chain; /* per-fetch work buffers, preallocated by
                                 the caller (never malloc per fetch) */
    int Lat, D;
    long scratch_n;
    int64_t n_matvec, n_decode;
    int fail;
} ExpJob;

static void *exp_run(void *arg) {
    ExpJob *j = (ExpJob *)arg;
    /* Mark this thread as inside the expert path: matvecs called here
     * must NOT row-split (the 8 expert threads already saturate the
     * P-cores; spawning more would oversubscribe). The main thread's
     * attention/head matvecs split instead. */
    salt_kernels_set_in_expert(1);
    /* cur/tmp/chain come from the caller's preallocated buffers (never
     * malloc per fetch). zero-init: the chain tail (clen < Lat) is stale
     * after the last matvec; zero-init makes the combine deterministic
     * regardless of heap layout. */
    float *cur = j->cur;
    float *tmp = j->tmp;
    memset(cur, 0, (size_t)j->D * sizeof(float));
    memset(tmp, 0, (size_t)j->D * sizeof(float));
    memcpy(cur, j->latent, (size_t)j->Lat * sizeof(float));
    long clen = j->Lat;
    if (j->el->chain == 1) {
        /* Qwen3 parallel expert: silu(gate(x)) * up(x) -> down.
         * t[0]=gate_proj, t[1]=up_proj, t[2]=down_proj (manifest order).
         * gate/up: [moe_inter x H]; down: [H x moe_inter]. The chain
         * input is Lat floats (the residual/MLP input), H wide. */
        const SaltMoETensor *g = &j->el->t[0];
        const SaltMoETensor *u = &j->el->t[1];
        const SaltMoETensor *d = &j->el->t[2];
        if (g->rank == 2 && u->rank == 2 && d->rank == 2 &&
            g->dims[1] == j->Lat && u->dims[1] == j->Lat &&
            g->dims[0] == u->dims[0] && g->dims[0] <= j->D &&
            d->dims[1] == g->dims[0] && d->dims[0] == j->Lat) {
            long M = g->dims[0];
            float *gx = tmp;            /* gate(x), M floats */
            float *ux = tmp + M;        /* up(x), M floats -- SEPARATE
                                         * from cur: the matvec reads
                                         * x=cur while writing y, so y
                                         * must not alias the input
                                         * (row-over-row corruption). */
            float *chain = j->chain;   /* preallocated by caller */
            memset(chain, 0, (size_t)j->D * sizeof(float));
            if (g->fmt == 2) {
                /* F8_E4M3 experts (official Qwen FP8): the quant
                 * registry's fp8 module -- BF16 weight_scale_inv
                 * block scales on the pool's [SR, SC] grid. */
                const SaltQuant *qm = &salt_quant_fp8;
                qm->matvec((const void *)(j->slot + g->rel_v),
                           (const void *)(j->slot + g->rel_s),
                           NULL, (int)M, (int)j->Lat, cur, gx);
                qm->matvec((const void *)(j->slot + u->rel_v),
                           (const void *)(j->slot + u->rel_s),
                           NULL, (int)M, (int)j->Lat, cur, ux);
            } else if (g->fmt == 1) {
                const uint16_t *gb = g->rel_b >= 0
                    ? (const uint16_t *)(const void *)(j->slot + g->rel_b)
                    : NULL;
                const uint16_t *ub = u->rel_b >= 0
                    ? (const uint16_t *)(const void *)(j->slot + u->rel_b)
                    : NULL;
                /* the quant registry: the format's matvec behind one
                 * interface -- the core never branches on bits. */
                const SaltQuant *qm = salt_quant_get(g->bits);
                if (qm && qm->matvec) {
                    qm->matvec((const void *)(j->slot + g->rel_v),
                               (const void *)(j->slot + g->rel_s),
                               gb, (int)M, (int)j->Lat, cur, gx);
                    qm->matvec((const void *)(j->slot + u->rel_v),
                               (const void *)(j->slot + u->rel_s),
                               ub, (int)M, (int)j->Lat, cur, ux);
                }
            } else if (g->fmt == 4) {
                /* BF16 experts (the community fix for the early-layer
                 * FP8 collapse -- ollama#15866 / PR#15902: keep L0 in
                 * BF16; the F8 quant of the half-subnormal L0 weights
                 * collapses ~55% to the zero codepoint). Raw BF16
                 * [M, Lat] gate/up, [Lat, M] down, no scales. */
                const SaltQuant *qm = &salt_quant_bf16;
                qm->matvec((const void *)(j->slot + g->rel_v),
                           NULL, NULL, (int)M, (int)j->Lat, cur, gx);
                qm->matvec((const void *)(j->slot + u->rel_v),
                           NULL, NULL, (int)M, (int)j->Lat, cur, ux);
            } else {
                salt_mx4_matvec(j->slot + g->rel_v, j->slot + g->rel_s,
                                  (int)M, (int)j->Lat, g->bsize, cur, gx,
                                  j->scratch);
                salt_mx4_matvec(j->slot + u->rel_v, j->slot + u->rel_s,
                                  (int)M, (int)j->Lat, u->bsize, cur, ux,
                                  j->scratch);
            }
            j->n_matvec += 2;
            j->n_decode += 2 * M * j->Lat;
            /* silu(gate(x)) * up(x) -> chain, M floats */
            for (long i = 0; i < M; i++) {
                float s = gx[i];
                float sig = 1.0f / (1.0f + salt_expf(-s));
                chain[i] = s * sig * ux[i];
            }
            if (g_nan_probe) {
                pthread_mutex_lock(&_expdump_mu);
                long eidx = j->el - j->pool->exp;
                int eid = (int)(eidx % j->pool->n_experts);
                if (!_exp106_dumped || eid == 190) {
                FILE *sf = fopen("/tmp/q36-eng-slot.bin", "wb");
                if (sf) {
                    fwrite(j->slot, 1, 2048, sf);
                    fclose(sf);
                }
                FILE *cf = fopen("/tmp/q36-eng-chain.bin", "wb");
                if (cf) {
                    fwrite(&eid, sizeof(int), 1, cf);
                    fwrite(chain, sizeof(float), (size_t)M, cf);
                    fclose(cf);
                }
                FILE *gf = fopen("/tmp/q36-eng-gateup.bin", "wb");
                if (gf) {
                    fwrite(&eid, sizeof(int), 1, gf);
                    fwrite(gx, sizeof(float), (size_t)M, gf);
                    fwrite(ux, sizeof(float), (size_t)M, gf);
                    fclose(gf);
                }
                fprintf(stderr, "[expdump] expert %d\n", eid);
                {
                    FILE *lf = fopen("/tmp/q36-eng-latent.bin", "wb");
                    if (lf) {
                        fwrite(j->latent, sizeof(float),
                               (size_t)j->Lat, lf);
                        fclose(lf);
                    }
                }
                {
                    /* first gate weight words + scales for offset check */
                    const uint32_t *w0 = (const uint32_t *)(const void *)
                        (j->slot + g->rel_v);
                    const uint16_t *s0 = (const uint16_t *)(const void *)
                        (j->slot + g->rel_s);
                    const uint16_t *b0 = g->rel_b >= 0
                        ? (const uint16_t *)(const void *)(j->slot + g->rel_b)
                        : NULL;
                    fprintf(stderr, "[expw] rel_v=%ld rel_s=%ld "
                            "dims=[%ld,%ld] Lat=%d "
                            "w0=%08x %08x %08x %08x s0=%04x %04x %04x %04x\n",
                            (long)g->rel_v, (long)g->rel_s,
                            (long)g->dims[0], (long)g->dims[1], j->Lat,
                            w0[0], w0[1], w0[2], w0[3],
                            s0[0], s0[1], s0[2], s0[3]);
                    /* decode row 0 elements 0..15 with the engine's own
                     * pointers: q*s+b, group 0 */
                    for (int c = 0; c < 16; c++) {
                        int q = (int)((w0[c >> 3] >> (4 * (c & 7))) & 0xFu);
                        float s, bb = 0.0f;
                        uint32_t sb = (uint32_t)s0[c / 64] << 16;
                        memcpy(&s, &sb, 4);
                        if (b0) {
                            uint32_t bb2 = (uint32_t)b0[c / 64] << 16;
                            memcpy(&bb, &bb2, 4);
                        }
                        fprintf(stderr, "[expw] row0[%d] q=%d s=%.6g b=%.6g "
                                "W=%.6g\n", c, q, (double)s, (double)bb,
                                (double)((float)q * s + bb));
                    }
                }
                _exp106_dumped = 1;
                }
                pthread_mutex_unlock(&_expdump_mu);
            }
            if (g_nan_probe && j->latent &&
                j->latent[0] != j->latent[0]) { /* NaN latent? */
            }
            if (g_debug_chain) {
                double g2 = 0.0, u2 = 0.0, c2 = 0.0;
                for (long i = 0; i < M; i++) {
                    g2 += (double)gx[i] * gx[i];
                    u2 += (double)ux[i] * ux[i];
                    c2 += (double)chain[i] * chain[i];
                }
                fprintf(stderr, "[chain] gate-rms %.6g up-rms %.6g "
                        "chain-rms %.6g\n", sqrt(g2 / M), sqrt(u2 / M),
                        sqrt(c2 / M));
            }
            /* down(chain) -> cur (Lat floats) */
            if (d->fmt == 2) {
                /* down: [H x moe_inter] with its OWN scale grid
                 * [16,4] (s_shape from the manifest) -- the pool
                 * global [4,16] would mis-scale every block. */
                int SR = d->s_rank > 0 ? (int)d->s_dims[0] : 16;
                int SC = d->s_rank > 1 ? (int)d->s_dims[1] : 4;
                salt_f8_matvec_bf16(
                    (const uint8_t *)(const void *)(j->slot + d->rel_v),
                    (const uint16_t *)(const void *)(j->slot + d->rel_s),
                    (int)j->Lat, (int)M, SR, SC, chain, tmp);
            } else if (d->fmt == 1) {
                const uint16_t *db = d->rel_b >= 0
                    ? (const uint16_t *)(const void *)(j->slot + d->rel_b)
                    : NULL;
                if (d->bits == 8)
                    salt_q8_matvec(
                        (const uint32_t *)(const void *)(j->slot + d->rel_v),
                        (const uint16_t *)(const void *)(j->slot + d->rel_s),
                        db, (int)j->Lat, (int)M, chain, tmp);
                else
                salt_q4_matvec(
                    (const uint32_t *)(const void *)(j->slot + d->rel_v),
                    (const uint16_t *)(const void *)(j->slot + d->rel_s),
                    db, (int)j->Lat, (int)M, chain, tmp);
            } else if (d->fmt == 4) {
                /* BF16 down [Lat, M]: no scales */
                salt_bf16_matvec(
                    (const uint16_t *)(const void *)(j->slot + d->rel_v),
                    (int)j->Lat, (int)M, chain, NULL, tmp);
            } else {
                salt_mx4_matvec(j->slot + d->rel_v, j->slot + d->rel_s,
                                  (int)j->Lat, (int)M, d->bsize, chain, tmp,
                                  j->scratch);
            }
            j->n_matvec++;
            j->n_decode += j->Lat * M;
            memcpy(cur, tmp, (size_t)j->Lat * sizeof(float));
            clen = j->Lat;
            if (g_nan_probe && _exp106_dumped &&
                !_exp106_out_dumped) {
                FILE *of = fopen("/tmp/q36-eng-expout.bin", "wb");
                if (of) {
                    fwrite(cur, sizeof(float), (size_t)j->Lat, of);
                    fclose(of);
                }
                _exp106_out_dumped = 1;
            }
        } else {
            /* shape mismatch: fall through to the sequential path so
             * the run still completes (garbage, but not a crash) */
            for (int ti = 0; ti < j->el->n; ti++) {
                const SaltMoETensor *t = &j->el->t[ti];
                if (t->rank != 2) continue;
                long R = t->dims[0], C = t->dims[1];
                if (C != clen || R > j->D) continue;
                if (R * C > j->scratch_n) { j->fail = 1; break; }
                if (t->fmt == 2) {
                    int SR = t->s_rank > 0 ? (int)t->s_dims[0] : 4;
                    int SC = t->s_rank > 1 ? (int)t->s_dims[1] : 16;
                    salt_f8_matvec_bf16(
                        (const uint8_t *)(const void *)(j->slot + t->rel_v),
                        (const uint16_t *)(const void *)(j->slot + t->rel_s),
                        (int)R, (int)C, SR, SC, cur, tmp);
                } else if (t->fmt == 1) {
                    const uint16_t *biases = t->rel_b >= 0
                        ? (const uint16_t *)(const void *)(j->slot + t->rel_b)
                        : NULL;
                    if (t->bits == 8)
                        salt_q8_matvec(
                            (const uint32_t *)(const void *)(j->slot + t->rel_v),
                            (const uint16_t *)(const void *)(j->slot + t->rel_s),
                            biases, (int)R, (int)C, cur, tmp);
                    else
                    salt_q4_matvec(
                        (const uint32_t *)(const void *)(j->slot + t->rel_v),
                        (const uint16_t *)(const void *)(j->slot + t->rel_s),
                        biases, (int)R, (int)C, cur, tmp);
                } else {
                    salt_mx4_matvec(j->slot + t->rel_v, j->slot + t->rel_s,
                                      (int)R, (int)C, t->bsize, cur, tmp,
                                      j->scratch);
                }
                j->n_matvec++;
                j->n_decode += R * C;
                memcpy(cur, tmp, (size_t)R * sizeof(float));
                clen = R;
            }
        }
    } else {
    for (int ti = 0; ti < j->el->n; ti++) {
        const SaltMoETensor *t = &j->el->t[ti];
        if (t->rank != 2) continue;
        long R = t->dims[0], C = t->dims[1];
        if (C != clen || R > j->D) continue;
        if (R * C > j->scratch_n) { j->fail = 1; break; }
        if (t->fmt == 2) {
            int SR = t->s_rank > 0 ? (int)t->s_dims[0] : 4;
            int SC = t->s_rank > 1 ? (int)t->s_dims[1] : 16;
            salt_f8_matvec_bf16(
                (const uint8_t *)(const void *)(j->slot + t->rel_v),
                (const uint16_t *)(const void *)(j->slot + t->rel_s),
                (int)R, (int)C, SR, SC, cur, tmp);
        } else if (t->fmt == 1) {
            const uint16_t *biases = t->rel_b >= 0
                ? (const uint16_t *)(const void *)(j->slot + t->rel_b)
                : NULL;
            if (t->bits == 8)
                salt_q8_matvec(
                    (const uint32_t *)(const void *)(j->slot + t->rel_v),
                    (const uint16_t *)(const void *)(j->slot + t->rel_s),
                    biases, (int)R, (int)C, cur, tmp);
            else
            salt_q4_matvec(
                (const uint32_t *)(const void *)(j->slot + t->rel_v),
                (const uint16_t *)(const void *)(j->slot + t->rel_s),
                biases, (int)R, (int)C, cur, tmp);
        } else {
            salt_mx4_matvec(j->slot + t->rel_v, j->slot + t->rel_s,
                              (int)R, (int)C, t->bsize, cur, tmp,
                              j->scratch);
        }
        j->n_matvec++;
        j->n_decode += R * C;
        memcpy(cur, tmp, (size_t)R * sizeof(float));
        clen = R;
    }
    }
    long ncopy = j->Lat < clen ? j->Lat : clen;
    memcpy(j->out, cur, (size_t)ncopy * sizeof(float));
    /* the caller combines over all Lat elements: the tail must be
     * zero, not malloc garbage (nondeterministic dumps otherwise) */
    if (ncopy < j->Lat)
        memset(j->out + ncopy, 0,
               (size_t)(j->Lat - ncopy) * sizeof(float));
    salt_kernels_set_in_expert(0);
    return NULL;
}

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* ------------------------------------------------------------------ */
/* Persistent expert-worker pool (decode-side, issue #5 follow-up):    */
/* the per-token pthread_create/join of up to topk expert chains      */
/* (moe.c:1186) spawns ~8 threads x 40 layers x every gen token --    */
/* ~82k creations in a 256-token run. The fetch pool (cache.c) already*/
/* solved this for disk reads; this mirrors it: N resident workers    */
/* wait on a work condvar, run exp_run jobs, signal completion. The   */
/* combine stays in selection order AFTER all jobs finish, so results */
/* are bit-identical to the create/join path.                         */
#define SALT_EXP_POOL_N 8

typedef struct {
    ExpJob *jobs;              /* caller's stack array, max 64 */
    int njob;                  /* jobs in this dispatch */
    int next;                  /* next job index to hand out */
    int done;                  /* jobs completed */
    int shutdown;
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    pthread_t th[SALT_EXP_POOL_N];
    int started;
} ExpPool;

static ExpPool g_exp_pool;

/* pthread API failures in the pool are unrecoverable (EINVAL/EDEADLK
 * mean the mutex/CV state is corrupt, ENOMEM means the process is out
 * of memory). Fail loudly instead of silently continuing with a broken
 * pool -- the return values the production worker loop and dispatch
 * previously discarded (citrinitas FlowAsymmetry, moe.c:662 checked vs
 * :867/:894 unchecked). */
static void pool_check(int rc, const char *what) {
    if (rc == 0) return;
    fprintf(stderr, "exp_pool: %s failed: %s (rc=%d) -- aborting\n",
            what, strerror(rc), rc);
    abort();
}

static void *exp_pool_worker(void *arg) {
    ExpPool *p = (ExpPool *)arg;
    /* TLS: expert threads never row-split (same as exp_run sets) */
    salt_kernels_set_in_expert(1);
    for (;;) {
        pool_check(pthread_mutex_lock(&p->mu), "mutex_lock (worker)");
        while (!p->shutdown && p->next >= p->njob)
            pool_check(pthread_cond_wait(&p->cv_work, &p->mu), "cond_wait (work)");
        if (p->shutdown) { pool_check(pthread_mutex_unlock(&p->mu), "mutex_unlock (worker)"); return NULL; }
        int i = p->next++;
        pool_check(pthread_mutex_unlock(&p->mu), "mutex_unlock (worker)");
        exp_run(&p->jobs[i]);
        pool_check(pthread_mutex_lock(&p->mu), "mutex_lock (worker)");
        p->done++;
        if (p->done >= p->njob) pool_check(pthread_cond_broadcast(&p->cv_done), "cond_broadcast (done)");
        pool_check(pthread_mutex_unlock(&p->mu), "mutex_unlock (worker)");
    }
    return NULL;
}

/* run up to njob expert chains on the resident pool; blocks until all
 * finish (same join semantics as the old per-call pthread_create). */
static void exp_pool_start(ExpJob *jobs, int njob) {
    ExpPool *p = &g_exp_pool;
    if (!p->started) {
        pool_check(pthread_mutex_init(&p->mu, NULL), "mutex_init");
        pool_check(pthread_cond_init(&p->cv_work, NULL), "cond_init (work)");
        pool_check(pthread_cond_init(&p->cv_done, NULL), "cond_init (done)");
        for (int i = 0; i < SALT_EXP_POOL_N; i++)
            pool_check(pthread_create(&p->th[i], NULL, exp_pool_worker, p),
                       "pthread_create (worker)");
        p->started = 1;
    }
    pool_check(pthread_mutex_lock(&p->mu), "mutex_lock (dispatch)");
    p->jobs = jobs;
    p->njob = njob;
    p->next = 0;
    p->done = 0;
    pool_check(pthread_cond_broadcast(&p->cv_work), "cond_broadcast (work)");
    pool_check(pthread_mutex_unlock(&p->mu), "mutex_unlock (start)");
}

static void exp_pool_wait(void) {
    ExpPool *p = &g_exp_pool;
    pool_check(pthread_mutex_lock(&p->mu), "mutex_lock (wait)");
    while (p->done < p->njob)
        pool_check(pthread_cond_wait(&p->cv_done, &p->mu), "cond_wait (done)");
    pool_check(pthread_mutex_unlock(&p->mu), "mutex_unlock (wait)");
}

/* read one element of a small F32/BF16 tensor by index */
static float hc_elem(const SaltTrunkTensor *t, const uint8_t *tr, long i) {
    const uint8_t *p = tr + t->off;
    if (t->dtype == 4) {
        const uint16_t *b = (const uint16_t *)(const void *)p;
        return bf16_f(b[i]);
    }
    const float *f = (const float *)(const void *)p;
    return f[i];
}

/* Sinkhorn-Knopp: B = doubly stochastic projection of exp(Btilde)
 * (paper eq. 8), 20 row/col normalizations. */
/* in-place RMSNorm with BF16 weights (the attn.c counterpart;
 * applies the checkpoint's layer norms, eps 1e-6) */
static void rmsnorm_moe(const uint16_t *w, int dim, float *x) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float r = sqrtf((float)(ss / (double)dim) + 1e-6f);
    for (int i = 0; i < dim; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;  /* bf16 = top half */
        float bv;
        memcpy(&bv, &bits, 4);
        x[i] = x[i] / r * bv;
    }
}

static void sinkhorn(const float *btilde, int n, float *B) {
    double M[64];
    for (int i = 0; i < n * n; i++) M[i] = exp((double)btilde[i]);
    for (int it = 0; it < 20; it++) {
        for (int r = 0; r < n; r++) {
            double s = 0.0;
            for (int c = 0; c < n; c++) s += M[r * n + c];
            if (s > 0.0)
                for (int c = 0; c < n; c++) M[r * n + c] /= s;
        }
        for (int c = 0; c < n; c++) {
            double s = 0.0;
            for (int r = 0; r < n; r++) s += M[r * n + c];
            if (s > 0.0)
                for (int r = 0; r < n; r++) M[r * n + c] /= s;
        }
    }
    for (int i = 0; i < n * n; i++) B[i] = (float)M[i];
}

/* mHC params (DeepSeek-V4 paper eq. 1/3-8). fn = [(n_hc*(2+n_hc)) x
 * (n_hc*H)]: rows [0,n_hc) W_pre, [n_hc,2n_hc) W_post, then W_res.
 * base = [n_hc*(2+n_hc)] (S in the same row order), scale = [3]
 * (alpha_pre, alpha_post, alpha_res). F32/BF16. */
int salt_hc_params(const SaltTrunkLayout *tl, int fn_i, int base_i,
                   int sc_i, const uint8_t *tr, int H,
                   const float *state, int *n_hc_out,
                   float *A, float *C, float *B) {
    if (fn_i < 0 || base_i < 0 || sc_i < 0) return 0;
    const SaltTrunkTensor *fn = &tl->t[fn_i];
    const SaltTrunkTensor *bs = &tl->t[base_i];
    const SaltTrunkTensor *al = &tl->t[sc_i];
    long rows = fn->dims[0], cols = fn->dims[1];
    if (fn->rank != 2 || cols % H != 0) {
        fprintf(stderr, "hc: fn shape [%ld x %ld] unsupported "
                        "(want [n*(2+n) x n*%d])\n", rows, cols, H);
        return -1;
    }
    int nhc = (int)(cols / H);
    if (rows != (long)nhc * (2 + nhc)) {
        fprintf(stderr, "hc: fn rows %ld unsupported (n_hc=%d wants %d)\n",
                rows, nhc, nhc * (2 + nhc));
        return -1;
    }
    if ((fn->dtype != 0 && fn->dtype != 4) ||
        (bs->dtype != 0 && bs->dtype != 4) ||
        (al->dtype != 0 && al->dtype != 4)) {
        fprintf(stderr, "hc: dtype unsupported (F32/BF16 only)\n");
        return -1;
    }
    /* xhat = RMSNorm(vec(state)) over the whole n_hc*H stream */
    long total = (long)nhc * H;
    double ss = 0.0;
    for (long i = 0; i < total; i++) ss += (double)state[i] * state[i];
    float r = sqrtf((float)(ss / (double)total) + 1e-6f);

    for (int j = 0; j < nhc; j++) {
        double dp = 0.0, dq = 0.0;
        for (long i = 0; i < total; i++) {
            float xh = state[i] / r;
            dp += (double)xh * hc_elem(fn, tr, (long)j * cols + i);
            dq += (double)xh * hc_elem(fn, tr, (long)(nhc + j) * cols + i);
        }
        float a_pre = hc_elem(al, tr, 0) * (float)dp + hc_elem(bs, tr, j);
        float a_post = hc_elem(al, tr, 1) * (float)dq +
                       hc_elem(bs, tr, nhc + j);
        A[j] = 1.0f / (1.0f + salt_expf(-a_pre));
        C[j] = 2.0f / (1.0f + salt_expf(-a_post));
    }
    float btilde[64];
    for (int r2 = 0; r2 < nhc; r2++) {
        for (int c2 = 0; c2 < nhc; c2++) {
            double d = 0.0;
            for (long i = 0; i < total; i++)
                d += (double)(state[i] / r) * hc_elem(
                    fn, tr, (long)(2 * nhc + r2 * nhc + c2) * cols + i);
            btilde[r2 * nhc + c2] =
                hc_elem(al, tr, 2) * (float)d +
                hc_elem(bs, tr, 2 * nhc + r2 * nhc + c2);
        }
    }
    sinkhorn(btilde, nhc, B);
    *n_hc_out = nhc;
    return nhc;
}

void salt_hc_combine(int n_hc, int H, const float *A, const float *state,
                     float *x_in) {
    for (int i = 0; i < H; i++) {
        float s = 0.0f;
        for (int j = 0; j < n_hc; j++) s += A[j] * state[j * H + i];
        x_in[i] = s;
    }
}

SALT_THREAD_LOCAL SaltMoEDecodeProfile g_decode_profile;
SALT_THREAD_LOCAL int g_decode_profile_enabled;

static double decode_profile_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void salt_moe_decode_profile_reset(void) {
    memset(&g_decode_profile, 0, sizeof(g_decode_profile));
    g_decode_profile_enabled = 1;
}

void salt_moe_decode_profile_get(SaltMoEDecodeProfile *out) {
    if (out) *out = g_decode_profile;
}

/* Compute the dense shared expert without folding it into the routed
 * accumulator. The caller owns *storage_out and must preserve routed-first,
 * shared-second accumulation order. Running with the expert flag keeps this
 * caller-side work single-threaded while the routed pool occupies the eight
 * performance cores; every row still uses the qualified scalar/SIMD order. */
static int shared_expert_compute(const SaltCfg *cfg,
                                 const SaltTrunkLayout *tl, int L,
                                 const uint8_t *tr, const float *xin,
                                 float **storage_out,
                                 const float **sout_out,
                                 float *sgate_out) {
    int H = cfg->hidden;
    *storage_out = NULL;
    *sout_out = NULL;
    *sgate_out = 0.0f;
    if (!tl || tl->se_g[L] < 0 || tl->se_u[L] < 0 || tl->se_d[L] < 0 ||
        tl->se_gs[L] < 0 || tl->se_us[L] < 0 || tl->se_ds[L] < 0 ||
        tl->se_r[L] < 0 || tl->se_rs[L] < 0 || tl->se_rb[L] < 0)
        return 0;

    const SaltTrunkTensor *t0 = &tl->t[tl->se_g[L]];
    long M = t0->dims[0];
    float *sg = (float *)malloc((size_t)(M * 3 + H) * sizeof(float));
    if (!sg) return -1;
    float *upv = sg + M;
    float *chain = sg + 2 * M;
    float *sout = sg + 3 * M;
    float sgate[1];

    salt_kernels_set_in_expert(1);
    if (t0->dtype == 2) {
        int SR = tl->t[tl->se_gs[L]].rank > 0
            ? (int)tl->t[tl->se_gs[L]].dims[0] : 4;
        int SC = tl->t[tl->se_gs[L]].rank > 1
            ? (int)tl->t[tl->se_gs[L]].dims[1] : 16;
        salt_f8_matvec_bf16(
            (const uint8_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
            (int)M, H, SR, SC, xin, sg);
        salt_f8_matvec_bf16(
            (const uint8_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
            (int)M, H, SR, SC, xin, upv);
    } else if (t0->bits == 8) {
        salt_q8_matvec((const uint32_t *)(const void *)
                           (tr + tl->t[tl->se_g[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_gs[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_gb[L]].off),
                       (int)M, (int)(t0->dims[1] * (32 / t0->bits)),
                       xin, sg);
        salt_q8_matvec((const uint32_t *)(const void *)
                           (tr + tl->t[tl->se_u[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_us[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_ub[L]].off),
                       (int)M, (int)(t0->dims[1] * (32 / t0->bits)),
                       xin, upv);
    } else {
        salt_q4_matvec((const uint32_t *)(const void *)
                           (tr + tl->t[tl->se_g[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_gs[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_gb[L]].off),
                       (int)M, H, xin, sg);
        salt_q4_matvec((const uint32_t *)(const void *)
                           (tr + tl->t[tl->se_u[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_us[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_ub[L]].off),
                       (int)M, H, xin, upv);
    }
    for (long i = 0; i < M; i++) {
        float s = sg[i];
        float sig = 1.0f / (1.0f + salt_expf(-s));
        chain[i] = s * sig * upv[i];
    }
    if (t0->dtype == 2) {
        int SR = tl->t[tl->se_ds[L]].rank > 0
            ? (int)tl->t[tl->se_ds[L]].dims[0] : 4;
        int SC = tl->t[tl->se_ds[L]].rank > 1
            ? (int)tl->t[tl->se_ds[L]].dims[1] : 16;
        salt_f8_matvec_bf16(
            (const uint8_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
            H, (int)M, SR, SC, chain, sout);
    } else if (t0->bits == 8) {
        const SaltTrunkTensor *td = &tl->t[tl->se_d[L]];
        salt_q8_matvec((const uint32_t *)(const void *)
                           (tr + td->off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_ds[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_db[L]].off),
                       H, (int)(td->dims[1] * (32 / td->bits)), chain, sout);
    } else {
        salt_q4_matvec((const uint32_t *)(const void *)
                           (tr + tl->t[tl->se_d[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_ds[L]].off),
                       (const uint16_t *)(const void *)
                           (tr + tl->t[tl->se_db[L]].off),
                       H, (int)M, chain, sout);
    }
    {
        const SaltTrunkTensor *rt = &tl->t[tl->se_r[L]];
        if (rt->dtype == 4)
            salt_bf16_matvec((const uint16_t *)(const void *)(tr + rt->off),
                             1, (int)rt->dims[1], xin, NULL, sgate);
        else if (rt->bits == 8)
            salt_q8_matvec((const uint32_t *)(const void *)(tr + rt->off),
                           (const uint16_t *)(const void *)
                               (tr + tl->t[tl->se_rs[L]].off),
                           (const uint16_t *)(const void *)
                               (tr + tl->t[tl->se_rb[L]].off),
                           1, (int)(rt->dims[1] * (32 / rt->bits)),
                           xin, sgate);
        else
            salt_q4_matvec((const uint32_t *)(const void *)(tr + rt->off),
                           (const uint16_t *)(const void *)
                               (tr + tl->t[tl->se_rs[L]].off),
                           (const uint16_t *)(const void *)
                               (tr + tl->t[tl->se_rb[L]].off),
                           1, (int)(rt->dims[1] * (32 / rt->bits)),
                           xin, sgate);
    }
    salt_kernels_set_in_expert(0);

    *storage_out = sg;
    *sout_out = sout;
    *sgate_out = sgate[0];
    return 1;
}

int salt_moe_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                  const uint8_t *tr, const SaltPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  float *const *job_scratch,
                  int64_t *n_matvec, int64_t *n_decode,
                  const float *shared_sout, float shared_sgate) {
    moe_env_ensure();
    int H = cfg->hidden, Lat = cfg->latent, M = cfg->moe_inter;
    int D = H > Lat ? H : Lat;
    if (M > D) D = M;
    if (D < 1) return -1;
    double decode_profile_start = g_decode_profile_enabled
        ? decode_profile_now_s() : 0.0;

    /* bisect: dump the PRE-MoE state (linear-chunk output) -- where
     * does the cross-ISA divergence first enter? */
    if (g_nan_probe && L <= 2) {
        char fn[64];
        snprintf(fn, sizeof fn, "/tmp/q36-pre-moe-L%d.bin", L);
        FILE *pf = fopen(fn, "wb");
        if (pf) {
            fwrite(state, sizeof(float), (size_t)H, pf);
            fclose(pf);
        }
    }

    /* mHC (issue #6 step 6): F_ffn sees x_in = A·vec(X); the update is
     * new[j*H+i] = sum_k B[j][k]*orig[k*H+i] + C[j]*F[i]. orig holds
     * the pre-ffn streams; the RMS-rescale below is the no-hc
     * fallback. */
    float A[8], C[8], B[64];
    int nhc = 1;
    int hc_ok = tl ? salt_hc_params(tl, tl->hc_ffn_fn[L], tl->hc_ffn_base[L],
                                    tl->hc_ffn_scale[L], tr, H, state,
                                    &nhc, A, C, B) : 0;
    if (hc_ok < 0) return -1;
    float *orig = NULL, *xin = NULL;
    if (hc_ok) {
        orig = (float *)malloc((size_t)nhc * H * sizeof(float));
        xin = (float *)malloc((size_t)H * sizeof(float));
        if (!orig || !xin) { free(orig); free(xin); return -1; }
        memcpy(orig, state, (size_t)nhc * H * sizeof(float));
        salt_hc_combine(nhc, H, A, state, xin);
        /* the real model's post_attention_layernorm (was never
         * applied -- the raw state fed the router/experts) */
        if (tl->ffn_norm[L] >= 0 && !g_no_norms)
            rmsnorm_moe((const uint16_t *)(const void *)(
                            tr + tl->t[tl->ffn_norm[L]].off),
                        H, xin);
    } else {
        /* Qwen3.5 (no mHC): x = x + mlp(post_attention_layernorm(x)).
         * The experts project the NORMED input; the residual adds to
         * the raw state. */
        xin = (float *)malloc((size_t)H * sizeof(float));
        if (!xin) return -1;
        memcpy(xin, state, (size_t)H * sizeof(float));
        if (tl->ffn_norm[L] >= 0 && !g_no_norms)
            rmsnorm_moe((const uint16_t *)(const void *)(
                            tr + tl->t[tl->ffn_norm[L]].off),
                        H, xin);
    }

    /* Entry RMS: the F-rescale target (hc) or the RMS-rescale fallback
     * target (no hc). */
    double ss_in = 0.0;
    if (hc_ok) {
        for (int i = 0; i < H; i++)
            ss_in += (double)xin[i] * xin[i];
    } else {
        for (int i = 0; i < H; i++)
            ss_in += (double)state[i] * state[i];
    }
    float rms_in = sqrtf((float)(ss_in / (double)H));

    /* Persistent per-call scratch matrix: latent/cur/out/acc (D each)
     * + jobbuf (maxjob x perjob) were calloc'd and freed 40x/token --
     * ~288KB x 40 layers x tokens of allocator churn. Now ONE
     * thread-local block, grown on demand, reused across calls (the
     * caller's warm scratch already follows this pattern). The
     * zero-init is preserved: calloc on first touch, memset reuse. */
    int maxjob = cfg->topk > 64 ? 64 : cfg->topk;
    if (maxjob < 1) maxjob = 1;
    long perjob = (long)Lat + 3L * D;
    size_t need = (size_t)(4 * D + maxjob * perjob) * sizeof(float);
    SALT_THREAD_LOCAL float *tl_moe = NULL;
    SALT_THREAD_LOCAL size_t tl_moe_cap = 0;
    if (need > tl_moe_cap) {
        float *nb = (float *)realloc(tl_moe, need);
        if (!nb) return -1;
        tl_moe = nb;
        tl_moe_cap = need;
        memset(tl_moe, 0, need);       /* keep the calloc zero-init */
    } else {
        memset(tl_moe, 0, need);
    }
    float *latent = tl_moe;
    float *cur    = tl_moe + (size_t)D;
    float *out    = tl_moe + 2 * (size_t)D;
    float *acc    = tl_moe + 3 * (size_t)D;
    float *jobbuf = tl_moe + 4 * (size_t)D;
    (void)cur;              /* expert chains now run in worker threads */
    (void)scratch;          /* job 0 uses the caller's warm buffer */

    /* latent = W_down * x_in (identity when absent / mismatched) */
    int di = tl ? tl->down[L] : -1, ui = tl ? tl->up[L] : -1;
    int did_ok = 0;
    if (di >= 0 && tl->t[di].dtype == 0 && tl->t[di].rank == 2) {
        long R = tl->t[di].dims[0], C = tl->t[di].dims[1];
        if (C == H && R <= D) {
            salt_f32_matvec((const float *)(const void *)(tr + tl->t[di].off),
                            (int)R, (int)C, xin, latent);
            (*n_matvec)++;
            did_ok = 1;
        }
    }
    if (!did_ok) {
        for (int i = 0; i < Lat && i < H; i++) latent[i] = xin[i];
        for (int i = H; i < Lat; i++) latent[i] = 0.0f;
    }

    /* Parallel expert chains (issue #5): each topk expert's w1->w2->w3
     * chain is independent once its slot is resident, so run them on
     * separate threads (up to topk) and combine in j order -- the
     * combine order is unchanged, so results are bit-identical to the
     * serial path. */
    ExpJob job[64];
    int njob = 0;
    float *local_shared_storage = NULL;
    const float *local_shared_sout = NULL;
    float local_shared_gate = 0.0f;
    int local_shared_rc = 0;
    int local_shared_attempted = 0;
    const char *overlap_env = getenv("SALT_MOE_SHARED_OVERLAP");
    int shared_overlap = !overlap_env || !*overlap_env || *overlap_env != '0';
    const char *gpu_env = getenv("SALT_GPU");
    const char *gpu_moe_env = getenv("SALT_GPU_MOE");
    int gpu_moe_required = gpu_env && *gpu_env && *gpu_env != '0' &&
                           gpu_moe_env && *gpu_moe_env &&
                           *gpu_moe_env != '0';
    if (gpu_moe_required) {
        /* GPU batched path (the 13x design): gate+up batched in ONE
         * dispatch (2*topk jobs of R=M), silu*up on CPU, then the downs
         * batched (topk jobs of R=H). Every eligibility or dispatch failure
         * is terminal when expert GPU intent is explicit. Trunk mapped-only
         * state alone never enters this resource-specific block. */
        static int64_t _gpu_layers = 0;
        const char *dbg = getenv("SALT_GPU_DIAG");
        const uint32_t *gv[128], *uv[128], *dv[128];
        const uint16_t *gs[128], *us[128], *ds_[128];
        const uint16_t *gb[128], *ub[128], *db[128];
        float *gy[128], *uy[128], *dy[128];
        const float *xlat[128], *xchain[128];
        const void *idg[128], *idu[128], *idd[128];
        int nexp = 0;
        for (int j = 0; j < cfg->topk; j++) {
            if (!es[j]) continue;
            const SaltExpertLayout *el = &pl->exp[(size_t)L * pl->n_experts + sel[j]];
            const SaltMoETensor *g = &el->t[0], *u = &el->t[1], *d = &el->t[2];
            if (g->rank != 2 || u->rank != 2 || d->rank != 2 ||
                g->rel_v < 0 || g->rel_s < 0 || u->rel_v < 0 || u->rel_s < 0 ||
                d->rel_v < 0 || d->rel_s < 0) {
                if (dbg)
                    fprintf(stderr, "[gpu] L%d fallback: rank/rel "
                            "g=(%d,%ld,%ld,%ld) u=(%d,%ld,%ld,%ld) "
                            "d=(%d,%ld,%ld,%ld) "
                            "L=%d X=%d\n", L,
                            g->rank, g->rel_v, g->rel_s, g->rel_b,
                            u->rank, u->rel_v, u->rel_s, u->rel_b,
                            d->rank, d->rel_v, d->rel_s, d->rel_b,
                            L, sel[j]);
                goto gpu_fallback;
            }
            if (g->bits == 8 || u->bits == 8 || d->bits == 8) {
                if (dbg)
                    fprintf(stderr, "[gpu] L%d fallback: bits "
                            "g=%d u=%d d=%d L=%d X=%d\n",
                            L, g->bits, u->bits, d->bits, L, sel[j]);
                goto gpu_fallback;   /* GPU kernels are 4-bit only */
            }
            ExpJob *jb = &job[nexp];
            gv[nexp] = (const uint32_t *)(const void *)(es[j] + g->rel_v);
            gs[nexp] = (const uint16_t *)(const void *)(es[j] + g->rel_s);
            gb[nexp] = g->rel_b >= 0
                ? (const uint16_t *)(const void *)(es[j] + g->rel_b) : NULL;
            uv[nexp] = (const uint32_t *)(const void *)(es[j] + u->rel_v);
            us[nexp] = (const uint16_t *)(const void *)(es[j] + u->rel_s);
            ub[nexp] = u->rel_b >= 0
                ? (const uint16_t *)(const void *)(es[j] + u->rel_b) : NULL;
            dv[nexp] = (const uint32_t *)(const void *)(es[j] + d->rel_v);
            ds_[nexp] = (const uint16_t *)(const void *)(es[j] + d->rel_s);
            db[nexp] = d->rel_b >= 0
                ? (const uint16_t *)(const void *)(es[j] + d->rel_b) : NULL;
            idg[nexp] = g; idu[nexp] = u; idd[nexp] = d;   /* stable */
            jb->out = jobbuf + (size_t)nexp * (size_t)perjob;
            gy[nexp] = jb->out;                       /* gate: M floats */
            uy[nexp] = jb->out + M;                   /* up: M floats */
            dy[nexp] = jb->out + 2 * M;               /* down: H floats */
            xlat[nexp] = latent;
            nexp++;
        }
        if (nexp > 0 && salt_gpu_init() == 0) {
            int rc = 0;
            /* dispatch 1: gate+up, 2*nexp jobs of R=M, x=latent */
            const uint32_t *v1[256]; const uint16_t *s1[256], *b1[256];
            const float *x1[256]; float *y1[256]; const void *id1[256];
            for (int e = 0; e < nexp; e++) {
                v1[2*e] = gv[e]; s1[2*e] = gs[e]; b1[2*e] = gb[e];
                x1[2*e] = xlat[e]; y1[2*e] = gy[e]; id1[2*e] = idg[e];
                v1[2*e+1] = uv[e]; s1[2*e+1] = us[e]; b1[2*e+1] = ub[e];
                x1[2*e+1] = xlat[e]; y1[2*e+1] = uy[e]; id1[2*e+1] = idu[e];
            }
            int rj1[256];
            for (int e = 0; e < 2 * nexp; e++) rj1[e] = (int)M;
            rc = salt_gpu_q4_batch(v1, s1, b1, x1, y1, id1, rj1, (int)H, 2*nexp);
            if (rc == 0) {
                /* silu(gate)*up on CPU: chain lands in uy (up consumed) */
                for (int e = 0; e < nexp; e++) {
                    float *gx = gy[e], *ux = uy[e];
                    for (int i = 0; i < M; i++) {
                        float s = gx[i];
                        float sig = 1.0f / (1.0f + salt_expf(-s));
                        ux[i] = s * sig * ux[i];
                    }
                    xchain[e] = uy[e];
                }
                /* dispatch 2: downs, nexp jobs of R=H, x=chain per job */
                const float *x2[256]; float *y2[256]; const void *id2[256];
                for (int e = 0; e < nexp; e++) {
                    v1[e] = dv[e]; s1[e] = ds_[e]; b1[e] = db[e];
                    x2[e] = xchain[e]; y2[e] = dy[e]; id2[e] = idd[e];
                }
                int rj2[256];
                for (int e = 0; e < nexp; e++) rj2[e] = (int)H;
                rc = salt_gpu_q4_batch(v1, s1, b1, x2, y2, id2, rj2, (int)M, nexp);
                if (rc == 0) {
                    /* the combine reads jb->out[0..Lat): the down result
                     * (H floats) lives at dy = jb->out + 2*M, so move it
                     * into jb->out (H <= Lat). */
                    for (int e = 0; e < nexp; e++) {
                        memmove(job[e].out, dy[e], (size_t)H * sizeof(float));
                        job[e].n_matvec = 3;
                        job[e].n_decode = (int64_t)(M * H + M * H + H * M);
                        job[e].fail = 0;
                    }
                    njob = nexp;
                    _gpu_layers++;
                    if (dbg && (_gpu_layers % 200 == 0 || _gpu_layers < 5))
                        fprintf(stderr, "[gpu] L%d used GPU (%lld layers)\n",
                                L, (long long)_gpu_layers);
                    goto gpu_combined;
                }
            }
        }
        /* Reaching here means required expert dispatch did not complete. */
        return -1;
    gpu_fallback:
        return -1;
    }
    for (int j = 0; j < cfg->topk; j++) {
        if (!es[j]) continue;
        ExpJob *jb = &job[njob];
        memset(jb, 0, sizeof *jb);
        jb->el = &pl->exp[(size_t)L * pl->n_experts + sel[j]];
        jb->pool = pl;
        jb->slot = es[j];
        jb->latent = latent;
        jb->out = jobbuf + (size_t)njob * (size_t)perjob;
        jb->cur  = jb->out + Lat;
        jb->tmp  = jb->cur + D;
        jb->chain = jb->tmp + D;
        memset(jb->out, 0, (size_t)Lat * sizeof(float));
        /* job 0 reuses the caller's warm scratch; the rest come from
         * the caller's pool -- never malloc per call (page faults). */
        jb->scratch = (njob == 0) ? scratch : job_scratch[njob - 1];
        if (!jb->out || !jb->scratch) {
            /* TLS scratch: process-lifetime, no free */
            return -1;
        }
        jb->Lat = Lat;
        jb->D = D;
        jb->scratch_n = scratch_n;
        njob++;
    }
    double routed_start = g_decode_profile_enabled
        ? decode_profile_now_s() : 0.0;
    if (njob > 0)
        exp_pool_start(job, njob);

    double shared_compute_start = g_decode_profile_enabled
        ? decode_profile_now_s() : 0.0;
    if (!shared_sout && shared_overlap) {
        local_shared_attempted = 1;
        local_shared_rc = shared_expert_compute(
            cfg, tl, L, tr, xin, &local_shared_storage,
            &local_shared_sout, &local_shared_gate);
    }
    if (g_decode_profile_enabled)
        g_decode_profile.shared_s +=
            decode_profile_now_s() - shared_compute_start;

    if (njob > 0)
        exp_pool_wait();
    if (g_decode_profile_enabled && njob > 0) {
        g_decode_profile.scheduler_window_s +=
            decode_profile_now_s() - routed_start;
        g_decode_profile.routed_jobs += njob;
    }
    if (local_shared_rc < 0) {
        free(local_shared_storage);
        free(orig);
        free(xin);
        return -1;
    }
    for (int j = 0; j < njob; j++) {
        ExpJob *jb = &job[j];
        if (jb->fail) {
            free(local_shared_storage);
            /* TLS scratch: process-lifetime, no free */
            free(orig);
            free(xin);
            return -1;
        }
        *n_matvec += jb->n_matvec;
        *n_decode += jb->n_decode;
    }
gpu_combined: ;
    /* combine in selection order: acc[i] += wsel[sel order] * chain[i] */
    double combine_start = g_decode_profile_enabled
        ? decode_profile_now_s() : 0.0;
    {
        int sj = 0;
        for (int j = 0; j < cfg->topk; j++) {
            if (!es[j]) continue;
            ExpJob *jb = &job[sj++];
            if (g_nan_probe && L == 0 && !_exp106_dumped) {
                FILE *ef = fopen("/tmp/q36-eng-exp106.bin", "wb");
                if (ef) {
                    fwrite(jb->out, sizeof(float), (size_t)Lat, ef);
                    fclose(ef);
                }
                _exp106_dumped = 1;
            }
            if (g_debug_chain && L < 3) {
                double o2 = 0.0;
                for (int i = 0; i < Lat; i++)
                    o2 += (double)jb->out[i] * jb->out[i];
                double g2 = 0.0, u2 = 0.0;
                /* jb->out layout: gate[0..M), up[M..2M), down[2M..) */
                for (int i = 0; i < M && i < Lat; i++) {
                    g2 += (double)jb->out[i] * jb->out[i];
                    u2 += (double)jb->out[M + i] * jb->out[M + i];
                }
                fprintf(stderr, "[wsel] L%d j%d w=%.6g out-rms %.6g "
                        "gate-rms %.6g up-rms %.6g\n",
                        L, j, (double)wsel[j], sqrt(o2 / Lat),
                        sqrt(g2 / (M < Lat ? M : Lat)),
                        sqrt(u2 / (M < Lat ? M : Lat)));
            }
            for (int i = 0; i < Lat; i++)
                acc[i] += wsel[j] * jb->out[i];
        }
    }
    if (g_decode_profile_enabled)
        g_decode_profile.combine_s += decode_profile_now_s() - combine_start;
    /* job[j].out are jobbuf slices; jobbuf freed at the end of the
     * function. (The old code freed each out separately -- that was
     * per-job malloc, now replaced by the single jobbuf block.) */

    /* shared expert (Qwen3.5): dense every-token MLP added after the
     * routed experts -- acc += sigmoid(shared_expert_gate(xin)) *
     * down(silu(gate(xin)) * up(xin)). Same q4 triplets.
     * M6: when shared_sout is non-NULL the CALLER precomputed the
     * shared contribution (batched over a chunk via the bit-identical
     * kernel); fold it into acc at the SAME point as the serial
     * computation so the residual order is unchanged (bit-fidelity). */
    double shared_start = g_decode_profile_enabled
        ? decode_profile_now_s() : 0.0;
    if (shared_sout) {
        float sg2 = 1.0f / (1.0f + salt_expf(-shared_sgate));
        for (int i = 0; i < H; i++)
            acc[i] += sg2 * shared_sout[i];
    } else if (local_shared_attempted) {
        if (local_shared_rc > 0) {
            float sg2 = 1.0f / (1.0f + salt_expf(-local_shared_gate));
            for (int i = 0; i < H; i++)
                acc[i] += sg2 * local_shared_sout[i];
            if (g_debug_chain && L < 3) {
                double o2 = 0.0;
                for (int i = 0; i < H; i++)
                    o2 += (double)local_shared_sout[i] * local_shared_sout[i];
                fprintf(stderr, "[shared] L%d gate=%.4f out-rms %.6g\n",
                        L, sg2, sqrt(o2 / (double)H));
            }
        }
    } else if (tl->se_g[L] >= 0 && tl->se_u[L] >= 0 && tl->se_d[L] >= 0 &&
        tl->se_gs[L] >= 0 && tl->se_us[L] >= 0 && tl->se_ds[L] >= 0 &&
        tl->se_r[L] >= 0 && tl->se_rs[L] >= 0 && tl->se_rb[L] >= 0) {
        const SaltTrunkTensor *t0 = &tl->t[tl->se_g[L]];
        long M = t0->dims[0];        /* 512 (moe_intermediate_size) */
        float *sg = (float *)malloc((size_t)(M * 3 + H + 1) * sizeof(float));
        if (sg) {
            float *upv = sg + M;
            float *chain = sg + 2 * M;
            float *sout = sg + 3 * M;      /* H floats */
            float sgate[1];
            if (t0->dtype == 2) {
                /* F8_E4M3 shared expert (official Qwen FP8) */
                const SaltTrunkTensor *tu = &tl->t[tl->se_u[L]];
                int SR = tl->t[tl->se_gs[L]].rank > 0
                    ? (int)tl->t[tl->se_gs[L]].dims[0] : 4;
                int SC = tl->t[tl->se_gs[L]].rank > 1
                    ? (int)tl->t[tl->se_gs[L]].dims[1] : 16;
                salt_f8_matvec_bf16(
                    (const uint8_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
                    (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
                    (int)M, (int)H, SR, SC, xin, sg);
                salt_f8_matvec_bf16(
                    (const uint8_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                    (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                    (int)M, (int)H, SR, SC, xin, upv);
                (void)tu;
            } else if (t0->bits == 8) {
                /* MLX 8-bit affine shared expert (community 8-bit
                 * quants): U32 word packs 4 elems, affine w/ bias. */
                salt_q8_matvec((const uint32_t *)(const void *)
                                   (tr + tl->t[tl->se_g[L]].off),
                               (const uint16_t *)(const void *)
                                   (tr + tl->t[tl->se_gs[L]].off),
                               (const uint16_t *)(const void *)
                                   (tr + tl->t[tl->se_gb[L]].off),
                               (int)M, (int)(t0->dims[1] * (32 / t0->bits)),
                               xin, sg);
                salt_q8_matvec((const uint32_t *)(const void *)
                                   (tr + tl->t[tl->se_u[L]].off),
                               (const uint16_t *)(const void *)
                                   (tr + tl->t[tl->se_us[L]].off),
                               (const uint16_t *)(const void *)
                                   (tr + tl->t[tl->se_ub[L]].off),
                               (int)M, (int)(t0->dims[1] * (32 / t0->bits)),
                               xin, upv);
            } else {
            salt_q4_matvec((const uint32_t *)(const void *)
                                (tr + tl->t[tl->se_g[L]].off),
                            (const uint16_t *)(const void *)
                                (tr + tl->t[tl->se_gs[L]].off),
                            (const uint16_t *)(const void *)
                                (tr + tl->t[tl->se_gb[L]].off),
                            (int)M, (int)H, xin, sg);
            salt_q4_matvec((const uint32_t *)(const void *)
                                (tr + tl->t[tl->se_u[L]].off),
                            (const uint16_t *)(const void *)
                                (tr + tl->t[tl->se_us[L]].off),
                            (const uint16_t *)(const void *)
                                (tr + tl->t[tl->se_ub[L]].off),
                            (int)M, (int)H, xin, upv);
            }
            for (long i = 0; i < M; i++) {
                float s = sg[i];
                float sig = 1.0f / (1.0f + salt_expf(-s));
                chain[i] = s * sig * upv[i];     /* silu(gate)*up */
            }
            if (t0->dtype == 2) {
                int SR = tl->t[tl->se_ds[L]].rank > 0
                    ? (int)tl->t[tl->se_ds[L]].dims[0] : 4;
                int SC = tl->t[tl->se_ds[L]].rank > 1
                    ? (int)tl->t[tl->se_ds[L]].dims[1] : 16;
                salt_f8_matvec_bf16(
                    (const uint8_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
                    (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
                    (int)H, (int)M, SR, SC, chain, sout);
            } else if (t0->bits == 8) {
                const SaltTrunkTensor *td = &tl->t[tl->se_d[L]];
                salt_q8_matvec((const uint32_t *)(const void *)
                                   (tr + tl->t[tl->se_d[L]].off),
                               (const uint16_t *)(const void *)
                                   (tr + tl->t[tl->se_ds[L]].off),
                               (const uint16_t *)(const void *)
                                   (tr + tl->t[tl->se_db[L]].off),
                               (int)H, (int)(td->dims[1] * (32 / td->bits)),
                               chain, sout);
            } else
            salt_q4_matvec((const uint32_t *)(const void *)
                                (tr + tl->t[tl->se_d[L]].off),
                            (const uint16_t *)(const void *)
                                (tr + tl->t[tl->se_ds[L]].off),
                            (const uint16_t *)(const void *)
                                (tr + tl->t[tl->se_db[L]].off),
                            (int)H, (int)M, chain, sout);
            {
                const SaltTrunkTensor *rt = &tl->t[tl->se_r[L]];
                if (rt->dtype == 4)
                    salt_bf16_matvec(
                        (const uint16_t *)(const void *)(tr + rt->off),
                        1, (int)rt->dims[1], xin, NULL, sgate);
                else if (rt->bits == 8)
                    salt_q8_matvec((const uint32_t *)(const void *)
                                        (tr + rt->off),
                                    (const uint16_t *)(const void *)
                                        (tr + tl->t[tl->se_rs[L]].off),
                                    (const uint16_t *)(const void *)
                                        (tr + tl->t[tl->se_rb[L]].off),
                                    1, (int)(rt->dims[1] * (32 / rt->bits)),
                                    xin, sgate);
                else
                    salt_q4_matvec((const uint32_t *)(const void *)
                                        (tr + rt->off),
                                    (const uint16_t *)(const void *)
                                        (tr + tl->t[tl->se_rs[L]].off),
                                    (const uint16_t *)(const void *)
                                        (tr + tl->t[tl->se_rb[L]].off),
                                    1, (int)(rt->dims[1] * (32 / rt->bits)),
                                    xin, sgate);
            }
            {
                float sg2 = 1.0f / (1.0f + salt_expf(-sgate[0]));
                for (int i = 0; i < H; i++)
                    acc[i] += sg2 * sout[i];
                if (g_debug_chain && L < 3) {
                    double o2 = 0.0;
                    for (int i = 0; i < H; i++)
                        o2 += (double)sout[i] * sout[i];
                    fprintf(stderr, "[shared] L%d gate=%.4f out-rms %.6g\n",
                            L, sg2, sqrt(o2 / (double)H));
                }
            }
            free(sg);
        }
    }
    if (g_decode_profile_enabled) {
        double shared_post_window = decode_profile_now_s() - shared_start;
        g_decode_profile.shared_s += shared_post_window;
        g_decode_profile.shared_post_window_s += shared_post_window;
    }
    free(local_shared_storage);

    if (g_nan_probe && L < 5) {
        double a2 = 0.0, i2 = 0.0;
        for (int i = 0; i < H; i++) {
            a2 += (double)acc[i] * acc[i];
            i2 += (double)xin[i] * xin[i];
        }
        fprintf(stderr, "[moeacc] L%d xin-rms %.6g acc-rms %.6g "
                "gain %.6g ffn_norm=%d hc_ok=%d no_norms=%d "
                "xin[0..2]=%.5g %.5g %.5g state[0..2]=%.5g %.5g %.5g\n",
                L, sqrt(i2 / H), sqrt(a2 / H),
                sqrt(a2 / H) / (sqrt(i2 / H) + 1e-30f),
                tl ? tl->ffn_norm[L] : -9, hc_ok, g_no_norms,
                (double)xin[0], (double)xin[1], (double)xin[2],
                (double)state[0], (double)state[1], (double)state[2]);
        if (L == 0 && !_moeacc0_dumped) {
            FILE *af = fopen("/tmp/q36-eng-moeacc0.bin", "wb");
            if (af) {
                fwrite(acc, sizeof(float), (size_t)H, af);
                fclose(af);
            }
            FILE *xf = fopen("/tmp/q36-eng-xin0.bin", "wb");
            if (xf) {
                fwrite(xin, sizeof(float), (size_t)H, xf);
                fclose(xf);
            }
            _moeacc0_dumped = 1;
        }
    }

    /* state = state + W_up * acc (mHC: streams = B*orig + C*F) */
    int up_ok = 0;
    if (ui >= 0 && tl->t[ui].dtype == 0 && tl->t[ui].rank == 2) {
        long R = tl->t[ui].dims[0], Uc = tl->t[ui].dims[1];
        if (Uc == Lat && R <= D) {
            salt_f32_matvec((const float *)(const void *)(tr + tl->t[ui].off),
                            (int)R, (int)Uc, acc, out);
            (*n_matvec)++;
            /* a short up matvec (R < H) leaves out[R..H) untouched:
             * zero it so the update is deterministic (the tail must
             * not be malloc garbage) */
            if (R < H) memset(out + R, 0, (size_t)(H - R) * sizeof(float));
            if (g_nan_probe && L < 3) {
                double o2 = 0.0, i2 = 0.0;
                for (int i = 0; i < H; i++) {
                    o2 += (double)out[i] * out[i];
                    i2 += (double)xin[i] * xin[i];
                }
                fprintf(stderr, "[moeout] L%d xin-rms %.6g out-rms %.6g "
                        "gain %.6g\n", L, sqrt(i2 / H), sqrt(o2 / H),
                        sqrt(o2 / H) / (sqrt(i2 / H) + 1e-30f));
            }
            if (hc_ok) {
                /* F-rescale: the approximate expert reads amplify (the
                 * real model bounds F by training). Rescale the up
                 * output to the layer-input RMS so the mHC update is
                 * finite; the real fix is the exact MLA/MoE reads. */
                double s2 = 0.0;
                for (int i = 0; i < H; i++) s2 += (double)out[i] * out[i];
                float rms_f = sqrtf((float)(s2 / (double)H)) + 1e-30f;
                float gain = rms_in / rms_f;
                if (gain > 0.0f && gain < 1e30f)
                    for (int i = 0; i < H; i++) out[i] *= gain;
                /* new[j*H+i] = sum_k B[j][k]*orig[k*H+i] + C[j]*out[i] */
                for (int i = 0; i < H; i++) {
                    float mix[8];
                    for (int j = 0; j < nhc; j++) {
                        float s = 0.0f;
                        for (int k = 0; k < nhc; k++)
                            s += B[j * nhc + k] * orig[k * H + i];
                        if (getenv("SALT_NO_B_MIX"))
                            s = orig[j * H + i];
                        mix[j] = s + C[j] * out[i];
                    }
                    for (int j = 0; j < nhc; j++) state[j * H + i] = mix[j];
                }
                /* state-rescale to the layer-input RMS (see attn.c);
                 * SALT_STATE_RMS_TARGET overrides the target (same
                 * dead-state-at-embed-scale issue) */
                {
                    double t2 = 0.0;
                    for (int i = 0; i < nhc * H; i++)
                        t2 += (double)state[i] * state[i];
                    float rms_s =
                        sqrtf((float)(t2 / (double)(nhc * H))) + 1e-30f;
                    float sgain = rms_in / rms_s;
                    const char *tgt = getenv("SALT_STATE_RMS_TARGET");
                    if (tgt) {
                        float t = (float)atof(tgt);
                        if (t > 0.0f) sgain = t / rms_s;
                    }
                    if (sgain > 0.0f && sgain < 1e30f)
                        for (int i = 0; i < nhc * H; i++)
                            state[i] *= sgain;
                }
            } else {
                for (int i = 0; i < H; i++) state[i] += out[i];
            }
            up_ok = 1;
        } else {
            fprintf(stderr, "moe: up[%d] shape [%ld x %ld] unsupported "
                            "(lat=%d d=%d) -- identity fallback\n",
                    ui, (long)tl->t[ui].dims[0], (long)tl->t[ui].dims[1],
                    Lat, D);
        }
    } else if (ui >= 0) {
        fprintf(stderr, "moe: up[%d] dtype/rank unsupported\n", ui);
    }
    if (!up_ok) {
        if (hc_ok) {
            /* F identity: out = x_in; streams = B*orig + C*x_in */
            for (int i = 0; i < H; i++) {
                float mix[8];
                for (int j = 0; j < nhc; j++) {
                    float s = 0.0f;
                    for (int k = 0; k < nhc; k++)
                        s += B[j * nhc + k] * orig[k * H + i];
                    mix[j] = s + C[j] * xin[i];
                }
                for (int j = 0; j < nhc; j++) state[j * H + i] = mix[j];
            }
            /* state-rescale to the layer-input RMS (same as the up_ok
             * branch; the fallback must bound the state too) */
            {
                double t2 = 0.0;
                for (int i = 0; i < nhc * H; i++)
                    t2 += (double)state[i] * state[i];
                float rms_s =
                    sqrtf((float)(t2 / (double)(nhc * H))) + 1e-30f;
                float sgain = rms_in / rms_s;
                if (sgain > 0.0f && sgain < 1e30f)
                    for (int i = 0; i < nhc * H; i++)
                        state[i] *= sgain;
            }
        } else {
            for (int i = 0; i < H && i < Lat; i++) state[i] += acc[i];
        }
    }

    /* Activation bounding. With mHC tensors the stream update above
     * already is the bounded residual (B doubly stochastic, C <= 2).
     * Without them, the Qwen3.5 path applies the post_attention norm
     * (stable by construction -- the reference needs no rescale). The
     * RMS-rescale was a DS-V4 mHC-era fallback; keep it only when the
     * fixture explicitly asks (SALT_RMS_RESCALE=1), otherwise the
     * residual is exact. */
    if (!hc_ok && getenv("SALT_RMS_RESCALE")) {
        double ss = 0.0;
        for (int i = 0; i < H; i++) {
            float v = state[i];
            ss += (double)v * v;
        }
        float rms_out = sqrtf((float)(ss / (double)H)) + 1e-30f;
        float gain = rms_in / rms_out;
        if (g_nan_probe && (L < 5 || L == 6 || L == 8 ||
                                        L == 12 || L == 16 || L == 20 ||
                                        L == 24 || L == 27))
            fprintf(stderr, "[moe] L%d hc_ok=%d rms_in %.6g rms_out %.6g "
                    "gain %.6g\n", L, hc_ok, rms_in, rms_out, gain);
        if (gain > 0.0f && gain < 1e30f)
            for (int i = 0; i < H; i++) state[i] *= gain;
    }

    /* bisect: dump the hidden state + expert accumulate (acc) + MoE
     * input (xin) -- where does the cross-ISA divergence first enter.
     * L0: the first MoE; L2: the state feeding GQA L3. */
    if (g_nan_probe && (L == 0 || L == 2)) {
        char fn[64];
        snprintf(fn, sizeof fn, "/tmp/q36-eng-L%d-state.bin", L);
        FILE *pf = fopen(fn, "wb");
        if (pf) {
            fwrite(state, sizeof(float), (size_t)H, pf);
            fclose(pf);
        }
        snprintf(fn, sizeof fn, "/tmp/q36-eng-L%d-acc.bin", L);
        pf = fopen(fn, "wb");
        if (pf) {
            fwrite(acc, sizeof(float), (size_t)H, pf);
            fclose(pf);
        }
        snprintf(fn, sizeof fn, "/tmp/q36-eng-L%d-xin.bin", L);
        pf = fopen(fn, "wb");
        if (pf) {
            fwrite(xin, sizeof(float), (size_t)H, pf);
            fclose(pf);
        }
    }

    if (g_decode_profile_enabled) {
        g_decode_profile.total_s += decode_profile_now_s() - decode_profile_start;
        g_decode_profile.calls++;
    }
    free(orig);
    free(xin);
    /* TLS scratch: process-lifetime, no free */
    return 0;
}

SALT_THREAD_LOCAL SaltMoEBatchProfile g_m2_profile;
SALT_THREAD_LOCAL int g_m2_profile_enabled;

static double m2_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void salt_moe_batch_profile_reset(void) {
    memset(&g_m2_profile, 0, sizeof(g_m2_profile));
    g_m2_profile_enabled = 1;
}

void salt_moe_batch_profile_get(SaltMoEBatchProfile *out) {
    if (out) *out = g_m2_profile;
}

int salt_moe_batch_should_run(int requested, int min_b, int actual_b) {
    return requested && min_b > 0 && actual_b >= min_b;
}

/* ---- M2: batched moe for the chunked prefill --------------------- */
/* B tokens, topk selections each. The B*topk (token, expert) pairs
 * are grouped by expert id; each distinct expert's gate/up/down run
 * ONCE as a batched matvec over its group (dequant amortized), the
 * down outputs are stored per selection, then combined in SELECTION
 * order -- the same combine order as the serial path, so the combine
 * is bit-identical. xins is [H][B] column-major (the batch kernel
 * layout, from linear_step_chunk's norm). es is [B][topk] resident
 * expert slot bytes; sel/wsel [B][topk]. states[B][H] in/out.
 * Returns -1 if the layer needs a path this batch can't do (mHC),
 * and the caller falls back to serial moe_step per token. */
typedef struct {
    int eid, b, j;
    float w;
    const uint8_t *slot;
} MoeSel;

static int sel_cmp(const void *a, const void *b) {
    const MoeSel *x = (const MoeSel *)a, *y = (const MoeSel *)b;
    return x->eid - y->eid;
}

/* ---- M6: batched shared expert (chunked prefill) ------------------ */
int salt_moe_shared_batch(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                          int L, const uint8_t *tr,
                          const float *const *states, int B,
                          float *sout, float *sgate) {
    moe_env_ensure();
    if (!cfg || !tl || B < 1 || !sout || !sgate) return -1;
    if (tl->se_g[L] < 0 || tl->se_u[L] < 0 || tl->se_d[L] < 0 ||
        tl->se_gs[L] < 0 || tl->se_us[L] < 0 || tl->se_ds[L] < 0 ||
        tl->se_r[L] < 0 || tl->se_rs[L] < 0 || tl->se_rb[L] < 0)
        return -1;                 /* no shared expert at this layer */
    int H = cfg->hidden;
    const SaltTrunkTensor *t0 = &tl->t[tl->se_g[L]];
    long M = t0->dims[0];          /* 512 */
    /* xins[B][H] row-major (normed), then per-tensor outputs */
    float *xins = (float *)malloc((size_t)B * (size_t)H * sizeof(float));
    float *sg = (float *)malloc((size_t)B * (size_t)M * sizeof(float));
    float *upv = (float *)malloc((size_t)B * (size_t)M * sizeof(float));
    float *chain = (float *)malloc((size_t)B * (size_t)M * sizeof(float));
    if (!xins || !sg || !upv || !chain) {
        free(xins); free(sg); free(upv); free(chain);
        return -1;
    }
    /* 1) per-token ffn_norm -- SAME formula as rmsnorm_moe (moe.c:841):
     * double sum, sqrtf(+1e-6), bf16 weight decode, x/r*w. Bit-
     * identical to what salt_moe_step computes internally. */
    if (tl->ffn_norm[L] >= 0 && !g_no_norms) {
        const uint16_t *nw = (const uint16_t *)(const void *)(
            tr + tl->t[tl->ffn_norm[L]].off);
        for (int b = 0; b < B; b++) {
            const float *st = states[b];
            float *xin = xins + (size_t)b * H;
            double ss = 0.0;
            for (int i = 0; i < H; i++) ss += (double)st[i] * st[i];
            float r = sqrtf((float)(ss / (double)H) + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint32_t bits = (uint32_t)nw[i] << 16;
                float bv;
                memcpy(&bv, &bits, 4);
                xin[i] = st[i] / r * bv;
            }
        }
    } else {
        for (int b = 0; b < B; b++)
            memcpy(xins + (size_t)b * H, states[b],
                   (size_t)H * sizeof(float));
    }
    /* 2) batched se_g and se_u (both [M x H]) -- dispatch by the
     * tensor's OWN format: F8_E4M3 (official FP8), 8-bit affine
     * (community 8-bit), or 4-bit affine (legacy). q4 is NOT a valid
     * decode of the other formats. */
    const SaltTrunkTensor *tg0 = &tl->t[tl->se_g[L]];
    int q4ok = 1;
    if (tg0->dtype == 2) {
        int SR = tl->t[tl->se_gs[L]].rank > 0
            ? (int)tl->t[tl->se_gs[L]].dims[0] : 4;
        int SC = tl->t[tl->se_gs[L]].rank > 1
            ? (int)tl->t[tl->se_gs[L]].dims[1] : 16;
        for (int b = 0; b < B; b++) {
            salt_f8_matvec_bf16(
                (const uint8_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
                (int)M, H, SR, SC, xins + (size_t)b * H, sg + (size_t)b * M);
            salt_f8_matvec_bf16(
                (const uint8_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                (int)M, H, SR, SC, xins + (size_t)b * H,
                upv + (size_t)b * M);
        }
        q4ok = 0;
    } else if (tg0->bits == 8) {
        long gN = tg0->dims[1] * (32 / 8);
        for (int b = 0; b < B; b++) {
            salt_q8_matvec(
                (const uint32_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_gb[L]].off),
                (int)M, (int)gN, xins + (size_t)b * H, sg + (size_t)b * M);
            salt_q8_matvec(
                (const uint32_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_ub[L]].off),
                (int)M, (int)gN, xins + (size_t)b * H,
                upv + (size_t)b * M);
        }
        q4ok = 0;
    }
    if (q4ok) {
        int shared_rc = salt_q4_matvec_batch(
            (const uint32_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_gb[L]].off),
            (int)M, H, B, xins, sg);
        if (shared_rc == 0)
            shared_rc = salt_q4_matvec_batch(
                (const uint32_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_ub[L]].off),
                (int)M, H, B, xins, upv);
        if (salt_gpu_sync() != 0) shared_rc = -1;
        if (shared_rc != 0) {
            free(xins); free(sg); free(upv); free(chain);
            return -1;
        }
    }
    /* 3) chain = silu(gate)*up per token */
    for (int b = 0; b < B; b++) {
        const float *sgv = sg + (size_t)b * M;
        const float *up = upv + (size_t)b * M;
        float *ch = chain + (size_t)b * M;
        for (long i = 0; i < M; i++) {
            float s = sgv[i];
            float sig = 1.0f / (1.0f + salt_expf(-s));
            ch[i] = s * sig * up[i];
        }
    }
    /* 4) batched se_d [H x M] -> sout[B][H] (dispatch like step 2) */
    if (tg0->dtype == 2) {
        int SR = tl->t[tl->se_ds[L]].rank > 0
            ? (int)tl->t[tl->se_ds[L]].dims[0] : 4;
        int SC = tl->t[tl->se_ds[L]].rank > 1
            ? (int)tl->t[tl->se_ds[L]].dims[1] : 16;
        for (int b = 0; b < B; b++)
            salt_f8_matvec_bf16(
                (const uint8_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
                H, (int)M, SR, SC, chain + (size_t)b * M,
                sout + (size_t)b * H);
    } else if (tg0->bits == 8) {
        const SaltTrunkTensor *td = &tl->t[tl->se_d[L]];
        long dN = td->dims[1] * (32 / 8);
        for (int b = 0; b < B; b++)
            salt_q8_matvec(
                (const uint32_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_db[L]].off),
                H, (int)dN, chain + (size_t)b * M,
                sout + (size_t)b * H);
    } else {
        int shared_rc = salt_q4_matvec_batch(
            (const uint32_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
            (const uint16_t *)(const void *)(tr + tl->t[tl->se_db[L]].off),
            H, (int)M, B, chain, sout);
        if (salt_gpu_sync() != 0) shared_rc = -1;
        if (shared_rc != 0) {
            free(xins); free(sg); free(upv); free(chain);
            return -1;
        }
    }
    /* 5) batched se_r [1 x H] -> sgate[B] (the gate scalar) */
    if (tg0->dtype == 2) {
        int SR = tl->t[tl->se_rs[L]].rank > 0
            ? (int)tl->t[tl->se_rs[L]].dims[0] : 4;
        int SC = tl->t[tl->se_rs[L]].rank > 1
            ? (int)tl->t[tl->se_rs[L]].dims[1] : 16;
        for (int b = 0; b < B; b++)
            salt_f8_matvec_bf16(
                (const uint8_t *)(const void *)(tr + tl->t[tl->se_r[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_rs[L]].off),
                1, H, SR, SC, xins + (size_t)b * H, sgate + b);
    } else if (tg0->bits == 8) {
        const SaltTrunkTensor *rt = &tl->t[tl->se_r[L]];
        long rN = rt->dims[1] * (32 / 8);
        for (int b = 0; b < B; b++)
            salt_q8_matvec(
                (const uint32_t *)(const void *)(tr + tl->t[tl->se_r[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_rs[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_rb[L]].off),
                1, (int)rN, xins + (size_t)b * H, sgate + b);
    } else {
        if (salt_q4_matvec_batch(
                (const uint32_t *)(const void *)(tr + tl->t[tl->se_r[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_rs[L]].off),
                (const uint16_t *)(const void *)(tr + tl->t[tl->se_rb[L]].off),
                1, H, B, xins, sgate) != 0) {
            free(xins); free(sg); free(upv); free(chain);
            return -1;
        }
    }
    free(xins); free(sg); free(upv); free(chain);
    return 0;
}

static int cpu_expert_q4_batch(const uint32_t *vals,
                               const uint16_t *scales,
                               const uint16_t *biases, int R, int C, int B,
                               const float *xs, float *ys) {
    int prior_scope = salt_kernels_cpu_expert_scope();
    int rc;
    salt_kernels_set_cpu_expert_scope(1);
    rc = salt_q4_matvec_batch(vals, scales, biases, R, C, B, xs, ys);
    salt_kernels_set_cpu_expert_scope(prior_scope);
    return rc;
}

int salt_moe_step_batch(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                        const uint8_t *tr, const SaltPoolLayout *pl,
                        const float *xins, int B, int topk,
                        const int *sel, const float *wsel,
                        const uint8_t *const *es,
                        float *const *states,
                        int64_t *n_matvec, int64_t *n_decode) {
    moe_env_ensure();
    int H = cfg->hidden, Lat = cfg->latent, M = cfg->moe_inter;
    if (B < 1 || topk < 1 || !tl || !pl) return -1;
    /* mHC layer: fall back to the serial per-token path */
    if (tl->hc_ffn_fn[L] >= 0) return -1;
    /* the Qwen3.5 residual is state += acc (no up tensor, no hc) */
    int ui = tl->up[L];
    if (ui >= 0) return -1;            /* batched path: identity-up only */

    int prof = g_m2_profile_enabled;
    double prof_last = prof ? m2_now_s() : 0.0;
    if (prof) g_m2_profile.calls++;
#define M2_TICK(field) do { \
        if (prof) { \
            double prof_now = m2_now_s(); \
            g_m2_profile.field += prof_now - prof_last; \
            prof_last = prof_now; \
        } \
    } while (0)

    SALT_THREAD_LOCAL float *tb_acc = NULL;    /* [B][H] */
    SALT_THREAD_LOCAL size_t tb_acc_cap = 0;
    SALT_THREAD_LOCAL float *tb_dy = NULL;     /* [B*topk][H] per-selection */
    SALT_THREAD_LOCAL size_t tb_dy_cap = 0;
    SALT_THREAD_LOCAL float *tb_w = NULL;      /* [G][M] gate/up workspace */
    SALT_THREAD_LOCAL size_t tb_w_cap = 0;
    size_t nacc = (size_t)B * H;
    size_t ndy = (size_t)B * topk * H;
    if (nacc > tb_acc_cap) {
        float *nb = (float *)realloc(tb_acc, nacc * sizeof(float));
        if (!nb) return -1;
        tb_acc = nb; tb_acc_cap = nacc;
    }
    if (ndy > tb_dy_cap) {
        float *nb = (float *)realloc(tb_dy, ndy * sizeof(float));
        if (!nb) return -1;
        tb_dy = nb; tb_dy_cap = ndy;
    }
    memset(tb_acc, 0, nacc * sizeof(float));
    memset(tb_dy, 0, ndy * sizeof(float));

    MoeSel *sl = (MoeSel *)malloc((size_t)B * topk * sizeof(MoeSel));
    if (!sl) return -1;
    int ns = 0;
    M2_TICK(setup_s);
    for (int b = 0; b < B; b++)
        for (int j = 0; j < topk; j++) {
            int eid = sel[(size_t)b * topk + j];
            if (eid < 0 || !es[(size_t)b * topk + j]) continue;
            sl[ns].eid = eid; sl[ns].b = b; sl[ns].j = j;
            sl[ns].w = wsel[(size_t)b * topk + j];
            sl[ns].slot = es[(size_t)b * topk + j];
            ns++;
        }
    if (prof) g_m2_profile.selections += ns;
    M2_TICK(select_s);
    if (ns > 0) {
        qsort(sl, (size_t)ns, sizeof(MoeSel), sel_cmp);
        /* per-selection workspace: [G][M] gate + up + chain */
        size_t gw = (size_t)ns * M;   /* worst case: all one expert */
        if (gw > tb_w_cap) {
            float *nb = (float *)realloc(tb_w, (3 * gw) * sizeof(float));
            if (!nb) { free(sl); return -1; }
            tb_w = nb; tb_w_cap = gw;
        }
        float *gys = tb_w;
        float *uys = tb_w + gw;
        float *cns = tb_w + 2 * gw;
        /* column-major gathers: xcol[H][G], ccol[M][G] */
        float *xcol = (float *)malloc((size_t)H * ns * sizeof(float));
        float *ccol = (float *)malloc((size_t)M * ns * sizeof(float));
        if (!xcol || !ccol) { free(xcol); free(ccol); free(sl); return -1; }
        M2_TICK(group_s);
        int i0 = 0;
        while (i0 < ns) {
            int i1 = i0 + 1;
            while (i1 < ns && sl[i1].eid == sl[i0].eid) i1++;
            int G = i1 - i0;
            const SaltExpertLayout *el =
                &pl->exp[(size_t)L * pl->n_experts + sl[i0].eid];
            const SaltMoETensor *g = &el->t[0], *u = &el->t[1],
                                *d = &el->t[2];
            if (g->rank != 2 || u->rank != 2 || d->rank != 2 ||
                g->rel_v < 0 || g->rel_s < 0 || u->rel_v < 0 ||
                u->rel_s < 0 || d->rel_v < 0 || d->rel_s < 0) {
                free(xcol); free(ccol); free(sl); return -1;
            }
            /* gather xcol[G][H] ROW-major from xins[H][B]: the batch
             * kernel reads xs + t*C (row-major [B][C]). The previous
             * column-major gather [H][G] was transposed -- invisible
             * at G=1 (B=1 KV identical), wrong math at G>1 (the M2
             * B=512 KV fork). */
            for (int gg = 0; gg < G; gg++) {
                int b = sl[i0 + gg].b;
                for (int i = 0; i < H; i++)
                    xcol[(size_t)gg * H + i] = xins[(size_t)i * B + b];
            }
            if (prof) g_m2_profile.groups++;
            M2_TICK(gather_s);
            const uint32_t *gv = (const uint32_t *)(const void *)
                (sl[i0].slot + g->rel_v);
            const uint16_t *gsc = (const uint16_t *)(const void *)
                (sl[i0].slot + g->rel_s);
            const uint16_t *gb = g->rel_b >= 0
                ? (const uint16_t *)(const void *)(sl[i0].slot + g->rel_b)
                : NULL;
            const uint32_t *uv = (const uint32_t *)(const void *)
                (sl[i0].slot + u->rel_v);
            const uint16_t *usc = (const uint16_t *)(const void *)
                (sl[i0].slot + u->rel_s);
            const uint16_t *ub = u->rel_b >= 0
                ? (const uint16_t *)(const void *)(sl[i0].slot + u->rel_b)
                : NULL;
            float *gy = gys + (size_t)i0 * M;
            float *uy = uys + (size_t)i0 * M;
            float *cn = cns + (size_t)i0 * M;
            if (g->fmt == 2) {
                /* F8_E4M3 pool expert (official Qwen FP8): the 4-bit
                 * batch kernel reads F8 bytes as U32 nibbles -> NaN.
                 * Run B serial salt_f8_matvec_bf16 (correct, per-token;
                 * the batch is a perf optimization, not correctness). */
                int gSR = g->s_rank > 0 ? (int)g->s_dims[0] : 4;
                int gSC = g->s_rank > 1 ? (int)g->s_dims[1] : 16;
                for (int gg = 0; gg < G; gg++) {
                    const float *xr = xcol + (size_t)gg * H;
                    salt_f8_matvec_bf16(
                        (const uint8_t *)(const void *)
                        (sl[i0].slot + g->rel_v),
                        (const uint16_t *)(const void *)
                        (sl[i0].slot + g->rel_s),
                        (int)M, H, gSR, gSC, xr, gy + (size_t)gg * M);
                    salt_f8_matvec_bf16(
                        (const uint8_t *)(const void *)
                        (sl[i0].slot + u->rel_v),
                        (const uint16_t *)(const void *)
                        (sl[i0].slot + u->rel_s),
                        (int)M, H, gSR, gSC, xr, uy + (size_t)gg * M);
                }
            } else if (g->bits == 8) {
                /* MLX 8-bit affine pool expert (community 8-bit):
                 * serial q8 per token (mirrors the F8 loop). */
                long gN = g->dims[1];
                long uN = u->dims[1];
                for (int gg = 0; gg < G; gg++) {
                    const float *xr = xcol + (size_t)gg * H;
                    salt_q8_matvec(
                        (const uint32_t *)(const void *)
                        (sl[i0].slot + g->rel_v),
                        (const uint16_t *)(const void *)
                        (sl[i0].slot + g->rel_s),
                        gb, (int)M, (int)gN, xr, gy + (size_t)gg * M);
                    salt_q8_matvec(
                        (const uint32_t *)(const void *)
                        (sl[i0].slot + u->rel_v),
                        (const uint16_t *)(const void *)
                        (sl[i0].slot + u->rel_s),
                        ub, (int)M, (int)uN, xr, uy + (size_t)gg * M);
                }
            } else if (cpu_expert_q4_batch(
                           gv, gsc, gb, M, H, G, xcol, gy) != 0 ||
                       cpu_expert_q4_batch(
                           uv, usc, ub, M, H, G, xcol, uy) != 0) {
                free(xcol); free(ccol); free(sl); return -1;
            }
            M2_TICK(gate_up_s);
            for (int gg = 0; gg < G; gg++) {
                float *cgr = cn + (size_t)gg * M;
                const float *gyr = gy + (size_t)gg * M;
                const float *uyr = uy + (size_t)gg * M;
                for (int i = 0; i < M; i++) {
                    float s = gyr[i];
                    float sig = 1.0f / (1.0f + salt_expf(-s));
                    cgr[i] = s * sig * uyr[i];   /* silu(gate)*up */
                }
            }
            M2_TICK(activation_s);
            /* ccol[G][M] ROW-major for the down batch (same transpose
             * fix as xcol: the kernel reads xs + t*C) */
            for (int gg = 0; gg < G; gg++)
                for (int i = 0; i < M; i++)
                    ccol[(size_t)gg * M + i] =
                        cns[(size_t)(i0 + gg) * M + i];
            const uint32_t *dv = (const uint32_t *)(const void *)
                (sl[i0].slot + d->rel_v);
            const uint16_t *dsc = (const uint16_t *)(const void *)
                (sl[i0].slot + d->rel_s);
            const uint16_t *db = d->rel_b >= 0
                ? (const uint16_t *)(const void *)(sl[i0].slot + d->rel_b)
                : NULL;
            M2_TICK(down_pack_s);
            if (d->fmt == 2) {
                /* F8 down: own [16,4] scale grid (s_shape from the
                 * manifest), serial per token -- same as the routed
                 * chain's down dispatch in exp_run. */
                int dSR = d->s_rank > 0 ? (int)d->s_dims[0] : 16;
                int dSC = d->s_rank > 1 ? (int)d->s_dims[1] : 4;
                for (int gg = 0; gg < G; gg++) {
                    salt_f8_matvec_bf16(
                        (const uint8_t *)(const void *)
                        (sl[i0].slot + d->rel_v),
                        (const uint16_t *)(const void *)
                        (sl[i0].slot + d->rel_s),
                        H, (int)M, dSR, dSC,
                        ccol + (size_t)gg * M,
                        tb_dy + (size_t)(i0 + gg) * H);
                }
            } else if (d->bits == 8) {
                /* MLX 8-bit affine down: serial q8 per token. */
                long dN = d->dims[1];
                for (int gg = 0; gg < G; gg++) {
                    salt_q8_matvec(
                        (const uint32_t *)(const void *)
                        (sl[i0].slot + d->rel_v),
                        (const uint16_t *)(const void *)
                        (sl[i0].slot + d->rel_s),
                        db, H, (int)dN,
                        ccol + (size_t)gg * M,
                        tb_dy + (size_t)(i0 + gg) * H);
                }
            } else if (cpu_expert_q4_batch(
                           dv, dsc, db, H, M, G, ccol,
                           tb_dy + (size_t)i0 * H) != 0) {
                free(xcol); free(ccol); free(sl); return -1;
            }
            M2_TICK(down_s);
            (*n_matvec) += 3 * G;
            (*n_decode) += (int64_t)G * (M * H + M * H + H * M);
            i0 = i1;
        }
        /* combine in SELECTION order: acc[b] += w[j] * dy[b][j] */
        for (int b = 0; b < B; b++)
            for (int j = 0; j < topk; j++) {
                int eid = sel[(size_t)b * topk + j];
                if (eid < 0) continue;
                /* find the selection's dy: linear over the group list */
                float w = wsel[(size_t)b * topk + j];
                if (w == 0.0f) continue;
                for (int s = 0; s < ns; s++)
                    if (sl[s].b == b && sl[s].j == j) {
                        const float *dy = tb_dy + (size_t)s * H;
                        float *ac = tb_acc + (size_t)b * H;
                        for (int i = 0; i < H; i++)
                            ac[i] += w * dy[i];
                        break;
                    }
            }
        M2_TICK(scatter_s);
        free(xcol); free(ccol);
    }
    free(sl);
    M2_TICK(glue_s);

    /* shared expert: dense per-token MLP, batched over all B tokens */
    if (tl->se_g[L] >= 0 && tl->se_u[L] >= 0 && tl->se_d[L] >= 0 &&
        tl->se_gs[L] >= 0 && tl->se_us[L] >= 0 && tl->se_ds[L] >= 0 &&
        tl->se_r[L] >= 0 && tl->se_rs[L] >= 0 && tl->se_rb[L] >= 0) {
        long sm = tl->t[tl->se_g[L]].dims[0];   /* 512 */
        if (sm > 0 && (size_t)B * sm * 4 <= tb_w_cap) {
            float *sg = tb_w;
            float *suv = tb_w + (size_t)B * sm;
            float *scn = tb_w + 2 * (size_t)B * sm;
            float *scol = (float *)malloc((size_t)sm * B * sizeof(float));
            float *sd = (float *)malloc((size_t)B * H * sizeof(float));
            float *sgt = (float *)malloc((size_t)B * sizeof(float));
            /* the batch kernel is ROW-major [B][C]; xins is
             * column-major [H][B] -- transpose into xr[B][H] once */
            float *xr = (float *)malloc((size_t)B * H * sizeof(float));
            if (scol && sd && sgt && xr) {
                for (int b = 0; b < B; b++)
                    for (int i = 0; i < H; i++)
                        xr[(size_t)b * H + i] = xins[(size_t)i * B + b];
                /* xins is already [H][B] column-major = the shared
                 * expert's x for all B tokens */
                int s_fmt = tl->t[tl->se_g[L]].dtype == 2 ? 2 :
                    (tl->t[tl->se_g[L]].dtype == 4 ? 4 :
                     (tl->t[tl->se_g[L]].bits == 8 ? 8 : 0));
                if (s_fmt == 2) {
                    /* F8 shared expert: gate/up [4,16], down [16,4] --
                     * serial f8 matvec per token (the 4-bit batch
                     * kernel would nibble-decode F8 -> NaN). */
                    int sSR = tl->t[tl->se_gs[L]].rank > 0
                        ? (int)tl->t[tl->se_gs[L]].dims[0] : 4;
                    int sSC = tl->t[tl->se_gs[L]].rank > 1
                        ? (int)tl->t[tl->se_gs[L]].dims[1] : 16;
                    int dSR = tl->t[tl->se_ds[L]].rank > 0
                        ? (int)tl->t[tl->se_ds[L]].dims[0] : 16;
                    int dSC = tl->t[tl->se_ds[L]].rank > 1
                        ? (int)tl->t[tl->se_ds[L]].dims[1] : 4;
                    int rok = 1;
                    for (int b = 0; b < B; b++) {
                        const float *xrow = xr + (size_t)b * H;
                        salt_f8_matvec_bf16(
                            (const uint8_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
                            (int)sm, H, sSR, sSC, xrow, sg + (size_t)b * sm);
                        salt_f8_matvec_bf16(
                            (const uint8_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                            (int)sm, H, sSR, sSC, xrow, suv + (size_t)b * sm);
                        if (tl->t[tl->se_r[L]].dtype == 4)
                            salt_bf16_matvec(
                                (const uint16_t *)(const void *)(tr + tl->t[tl->se_r[L]].off),
                                1, H, xrow, NULL, sgt + b);
                        else {
                            rok = 0; break;
                        }
                    }
                    if (rok) {
                        for (int b = 0; b < B; b++) {
                            float *cr = scn + (size_t)b * sm;
                            const float *gr = sg + (size_t)b * sm;
                            const float *ur = suv + (size_t)b * sm;
                            for (int i = 0; i < sm; i++) {
                                float s = gr[i];
                                float sig = 1.0f / (1.0f + salt_expf(-s));
                                cr[i] = s * sig * ur[i];
                            }
                        }
                        for (int i = 0; i < sm; i++)
                            for (int b = 0; b < B; b++)
                                scol[(size_t)b * sm + i] =
                                    scn[(size_t)b * sm + i];
                        for (int b = 0; b < B; b++) {
                            salt_f8_matvec_bf16(
                                (const uint8_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
                                (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
                                H, (int)sm, dSR, dSC,
                                scol + (size_t)b * sm,
                                sd + (size_t)b * H);
                        }
                        for (int b = 0; b < B; b++) {
                            float sg2 = 1.0f / (1.0f + salt_expf(-sgt[b]));
                            float *ac = tb_acc + (size_t)b * H;
                            const float *dr = sd + (size_t)b * H;
                            for (int i = 0; i < H; i++)
                                ac[i] += sg2 * dr[i];
                        }
                        (*n_matvec) += 4 * B;
                        (*n_decode) += (int64_t)B * (sm * H + sm * H + H * sm + H);
                    }
                } else if (s_fmt == 8) {
                    /* MLX 8-bit affine shared expert (community 8-bit
                     * quants): U32 word packs 4 elems; serial q8 per
                     * token (mirrors the F8 loop). */
                    long sN = tl->t[tl->se_g[L]].dims[1] * (32 / 8);
                    long uN = tl->t[tl->se_u[L]].dims[1] * (32 / 8);
                    long dN = tl->t[tl->se_d[L]].dims[1] * (32 / 8);
                    long rN = tl->t[tl->se_r[L]].dims[1] * (32 / 8);
                    int rok = 1;
                    for (int b = 0; b < B; b++) {
                        const float *xrow = xr + (size_t)b * H;
                        salt_q8_matvec(
                            (const uint32_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_gb[L]].off),
                            (int)sm, (int)sN, xrow, sg + (size_t)b * sm);
                        salt_q8_matvec(
                            (const uint32_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_ub[L]].off),
                            (int)sm, (int)uN, xrow, suv + (size_t)b * sm);
                        salt_q8_matvec(
                            (const uint32_t *)(const void *)(tr + tl->t[tl->se_r[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_rs[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_rb[L]].off),
                            1, (int)rN, xrow, sgt + b);
                    }
                    if (rok) {
                        for (int b = 0; b < B; b++) {
                            float *cr = scn + (size_t)b * sm;
                            const float *gr = sg + (size_t)b * sm;
                            const float *ur = suv + (size_t)b * sm;
                            for (int i = 0; i < sm; i++) {
                                float s = gr[i];
                                float sig = 1.0f / (1.0f + salt_expf(-s));
                                cr[i] = s * sig * ur[i];
                            }
                        }
                        for (int i = 0; i < sm; i++)
                            for (int b = 0; b < B; b++)
                                scol[(size_t)b * sm + i] =
                                    scn[(size_t)b * sm + i];
                        for (int b = 0; b < B; b++) {
                            salt_q8_matvec(
                                (const uint32_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
                                (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
                                (const uint16_t *)(const void *)(tr + tl->t[tl->se_db[L]].off),
                                H, (int)dN, scol + (size_t)b * sm,
                                sd + (size_t)b * H);
                        }
                        for (int b = 0; b < B; b++) {
                            float sg2 = 1.0f / (1.0f + salt_expf(-sgt[b]));
                            float *ac = tb_acc + (size_t)b * H;
                            const float *dr = sd + (size_t)b * H;
                            for (int i = 0; i < H; i++)
                                ac[i] += sg2 * dr[i];
                        }
                        (*n_matvec) += 4 * B;
                        (*n_decode) += (int64_t)B * (sm * H + sm * H + H * sm + H);
                    }
                } else {
                    int shared_rc = salt_q4_matvec_batch(
                        (const uint32_t *)(const void *)(tr + tl->t[tl->se_g[L]].off),
                        (const uint16_t *)(const void *)(tr + tl->t[tl->se_gs[L]].off),
                        (const uint16_t *)(const void *)(tr + tl->t[tl->se_gb[L]].off),
                        (int)sm, H, B, xr, sg);
                    if (shared_rc == 0)
                        shared_rc = salt_q4_matvec_batch(
                            (const uint32_t *)(const void *)(tr + tl->t[tl->se_u[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_us[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_ub[L]].off),
                            (int)sm, H, B, xr, suv);
                    if (salt_gpu_sync() != 0) shared_rc = -1;
                    if (shared_rc != 0) {
                        free(scol); free(sd); free(sgt); free(xr);
                        return -1;
                    }
                    for (int b = 0; b < B; b++) {
                        float *cr = scn + (size_t)b * sm;
                        const float *gr = sg + (size_t)b * sm;
                        const float *ur = suv + (size_t)b * sm;
                        for (int i = 0; i < sm; i++) {
                            float s = gr[i];
                            float sig = 1.0f / (1.0f + salt_expf(-s));
                            cr[i] = s * sig * ur[i];
                        }
                    }
                    for (int i = 0; i < sm; i++)
                        for (int b = 0; b < B; b++)
                            scol[(size_t)b * sm + i] =
                                scn[(size_t)b * sm + i];
                    shared_rc = salt_q4_matvec_batch(
                        (const uint32_t *)(const void *)(tr + tl->t[tl->se_d[L]].off),
                        (const uint16_t *)(const void *)(tr + tl->t[tl->se_ds[L]].off),
                        (const uint16_t *)(const void *)(tr + tl->t[tl->se_db[L]].off),
                        H, (int)sm, B, scol, sd);
                    if (salt_gpu_sync() != 0) shared_rc = -1;
                    if (shared_rc == 0)
                        shared_rc = salt_q4_matvec_batch(
                            (const uint32_t *)(const void *)(tr + tl->t[tl->se_r[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_rs[L]].off),
                            (const uint16_t *)(const void *)(tr + tl->t[tl->se_rb[L]].off),
                            1, H, B, xr, sgt);
                    if (salt_gpu_sync() != 0) shared_rc = -1;
                    if (shared_rc != 0) {
                        free(scol); free(sd); free(sgt); free(xr);
                        return -1;
                    }
                    for (int b = 0; b < B; b++) {
                        float sg2 = 1.0f / (1.0f + salt_expf(-sgt[b]));
                        float *ac = tb_acc + (size_t)b * H;
                        const float *dr = sd + (size_t)b * H;
                        for (int i = 0; i < H; i++)
                            ac[i] += sg2 * dr[i];
                    }
                    (*n_matvec) += 4 * B;
                    (*n_decode) += (int64_t)B *
                        (sm * H + sm * H + H * sm + H);
                }
            } else {
                free(scol); free(sd); free(sgt); free(xr);
                return -1;
            }
            free(scol); free(sd); free(sgt); free(xr);
        } else return -1;
    }
    M2_TICK(shared_s);

    /* residual: state[b] += acc[b] (Qwen3.5 no-up path) */
    for (int b = 0; b < B; b++) {
        float *st = states[b];
        const float *ac = tb_acc + (size_t)b * H;
        for (int i = 0; i < H && i < Lat; i++) st[i] += ac[i];
    }
    M2_TICK(residual_s);
    if (g_nan_probe && L == 2) {
        /* dump the hidden state that feeds GQA layer 3 -- bisect
         * where the cross-ISA divergence first enters */
        FILE *pf = fopen("/tmp/q36-eng-L2-state.bin", "wb");
        if (pf) {
            fwrite(states[0], sizeof(float), (size_t)H, pf);
            fclose(pf);
        }
    }
    M2_TICK(glue_s);
#undef M2_TICK
    return 0;
}
