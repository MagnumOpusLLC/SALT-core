#ifndef SALT_LAYER_H
#define SALT_LAYER_H

#include "salt/attn.h"   /* SaltTrunkLayout */

/* The layer-kind registry: the engine's per-layer walk is ONE
 * uniform loop over these descriptors. The model-specific surface
 * (which layer is a GQA, a linear-attention, a routed MoE, an MLP)
 * lives HERE as data -- never in the engine's logic. A new model
 * is a new registry fill, not a new engine.
 *
 * The serial gates are the dependency edges between the phases:
 *   GQA:  qkv-proj -> attn-body -> o-proj
 *   LIN:  qkv/z-proj -> delta-body -> fold
 *   MOE:  state -> router -> expert batch -> fold
 * The engine dispatches the proj phase, checks the gates, runs the
 * body, folds -- the same walk for every layer of every model. */

typedef enum {
    LAYER_GQA = 0,   /* the multi-head attention (q/k/v/o) */
    LAYER_LIN = 1,   /* the linear-attention (qkv/z + the delta rule) */
    LAYER_NONE = 2   /* no attention in this layer */
} SaltLayerAttnKind;

typedef struct {
    int L;                 /* the layer index */
    SaltLayerAttnKind attn;/* the attention kind (GQA or LIN) */
    int has_moe;           /* the routed MoE follows the attention */
    /* the tensor refs into the trunk layout (q3_* indices; -1 = none) */
    int qi, ki, vi, oi;    /* the GQA projections */
    int pi, zi;            /* the LIN projections (pqkv, pz) */
    int iln;               /* the input_layernorm */
    int qn, kn;            /* the q/k norms (GQA) */
    /* the gate edges (the dependency structure; the engine uses the
     * kind to dispatch, the gates to schedule the overlap) */
    int proj_then_body;    /* 1: the proj phase precedes the body */
    int body_then_moe;     /* 1: the moe follows the attention body */
} SaltLayerDesc;

/* Fill the registry for the active model's trunk layout. Returns the layer
 * count on success, -1 on a layout mismatch. The current Qwen adapter lives
 * under models/qwen36/; generic engine loops consume only this descriptor. */
int salt_layers_build(const SaltTrunkLayout *tl, SaltLayerDesc *out,
                      int max_layers);

#endif /* SALT_LAYER_H */
