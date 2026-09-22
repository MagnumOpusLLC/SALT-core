#include "salt/layer.h"

/* The qwen36 registry fill: the layer kinds and the tensor refs as
 * DATA. The engine walks this; nothing qwen-specific remains in the
 * walk. A different model supplies its own fill of the same shape. */
int salt_layers_build(const SaltTrunkLayout *tl, SaltLayerDesc *out,
                      int max_layers) {
    if (!tl || !out || max_layers < 1) return -1;
    if (tl->n_layers > max_layers) return -1;
    int n = 0;
    for (int L = 0; L < tl->n_layers; L++) {
        SaltLayerDesc *d = &out[n];
        d->L = L;
        int qi = tl->q3_q[L], pi = tl->q3_pqkv[L];
        d->attn = qi >= 0 ? LAYER_GQA : (pi >= 0 ? LAYER_LIN : LAYER_NONE);
        d->qi = qi;
        d->ki = tl->q3_k[L];
        d->vi = tl->q3_v[L];
        d->oi = tl->q3_o[L];
        d->pi = pi;
        d->zi = tl->q3_pz[L];
        d->iln = tl->attn_norm[L];
        d->qn = tl->q3_qn[L];
        d->kn = tl->q3_kn[L];
        d->has_moe = 1;          /* the qwen36: routed MoE on every layer */
        d->proj_then_body = 1;   /* the two-phase structure (the overlap) */
        d->body_then_moe = 1;    /* the routed MoE follows the attention */
        n++;
    }
    return n;
}
