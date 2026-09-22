/* cfg.c -- config reader that refuses to guess. */
#include "salt/salt.h"
#include "salt/model.h"
#include "json.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct KeyReq { const char *key; int *out; } KeyReq;

int salt_cfg_load(SaltCfg *cfg, const char *model_dir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", model_dir);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "config: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1 << 20)) { fclose(f); return -1; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    buf[sz] = 0;

    JDoc *doc = json_parse(buf, (size_t)sz);
    if (!doc) {
        fprintf(stderr, "config: %s is not parseable JSON\n", path);
        free(buf);
        return -1;
    }

    /* Collect every absent required key; never substitute a default. */
    static const char *req_keys[] = {
        "n_layers", "n_experts", "topk", "n_shared",
        "hidden", "latent", "moe_inter", "expert_nbytes",
    };
    int missing = 0;
    for (size_t i = 0; i < sizeof req_keys / sizeof req_keys[0]; i++) {
        const JEntry *e = json_get(doc->root, doc->nroot, req_keys[i]);
        if (!e || e->type != 0) {
            fprintf(stderr, "config: missing required key \"%s\" in %s\n",
                    req_keys[i], path);
            missing++;
        }
    }
    if (missing) {
        json_free(doc);
        free(buf);
        return -1;
    }
#define GETI(k) ((int)json_get(doc->root, doc->nroot, (k))->inum)
    cfg->n_layers    = GETI("n_layers");
    cfg->n_experts   = GETI("n_experts");
    cfg->topk        = GETI("topk");
    cfg->n_shared    = GETI("n_shared");
    cfg->hidden      = GETI("hidden");
    cfg->latent      = GETI("latent");
    cfg->moe_inter   = GETI("moe_inter");
    cfg->expert_nbytes = (int64_t)GETI("expert_nbytes");
    /* optional: the real MLA geometry (absent in the fixture -> 0 -> the
     * kvhalf fallback path) */
    {
        const JEntry *e = json_get(doc->root, doc->nroot,
                                   "num_attention_heads");
        cfg->n_heads = (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "num_key_value_heads");
        cfg->n_kv_heads = (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "qk_rope_head_dim");
        cfg->qk_rope = (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "linear_num_key_heads");
        cfg->linear_num_key_heads =
            (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "linear_num_value_heads");
        cfg->linear_num_value_heads =
            (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "linear_key_head_dim");
        cfg->linear_key_head_dim =
            (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "linear_value_head_dim");
        cfg->linear_value_head_dim =
            (e && e->type == 0) ? (int)e->inum : 0;
        /* tyrope (yarn-style rope correction) params; 0 = plain rotary */
        e = json_get(doc->root, doc->nroot, "rope_factor");
        cfg->rope_factor = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_beta_fast");
        cfg->rope_beta_fast = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_beta_slow");
        cfg->rope_beta_slow = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_max_pos");
        cfg->rope_max_pos = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_theta");
        cfg->rope_theta = (e && e->type == 0) ? e->num : 0.0;
    }
    /* The JSON-loaded values win; the external model registry fills gaps.
     * The registry also owns the default model identity. */
    {
        const char *mname = getenv("SALT_MODEL");
        const SaltModelDesc *md = salt_model_get(
            mname && *mname ? mname : salt_model_default_name());
        if (md && !md->runtime_ready) {
            fprintf(stderr,
                    "config: model %s is registered but its runtime is not ready\n",
                    md->name);
            json_free(doc);
            free(buf);
            return -1;
        }
        if (md) salt_model_apply(md, cfg);
    }
#undef GETI
    cfg->n_shards = 1;
    const JEntry *se = json_get(doc->root, doc->nroot, "seed");
    cfg->seed = se ? (uint64_t)se->inum : 7;

    /* A config that passes the gate but contradicts itself is a bug in
     * the fixture, not a runtime condition. Linear geometry may be all
     * zero for models without linear-attention layers; otherwise it must
     * be complete and internally consistent. */
    int linear_bad = 0;
    if (cfg->linear_num_key_heads != 0 ||
        cfg->linear_num_value_heads != 0 ||
        cfg->linear_key_head_dim != 0 || cfg->linear_value_head_dim != 0) {
        int kh, vh, kd, vd, rows;
        linear_bad = salt_model_linear_geometry(cfg, &kh, &vh, &kd, &vd,
                                                &rows) != 0;
    }
    if (cfg->n_layers <= 0 || cfg->n_experts <= 0 || cfg->topk <= 0 ||
        cfg->topk > cfg->n_experts || cfg->expert_nbytes <= 0 || linear_bad) {
        fprintf(stderr, "config: nonsense values in %s\n", path);
        json_free(doc);
        free(buf);
        return -1;
    }

    json_free(doc);
    free(buf);
    return 0;
}
