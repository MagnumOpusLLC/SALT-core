/* LFM2.5-24B-A2B host realization.
 *
 * The engine calls through SaltTensorHostOps; this supplies LFM's table. The
 * interface is the engine's -- include/salt/tensorops.h declares it and this
 * only fills it -- so a change to the engine's seam layout fails here at
 * compile time rather than at the call.
 *
 * Model-owned: the engine holds no LFM arithmetic, and this holds no engine
 * arithmetic beyond dispatching to the kernels the engine already ships. */

#ifndef SALT_LFM_HOST_H
#define SALT_LFM_HOST_H

#include <stdint.h>

#include "salt/salt.h"
#include "salt/attn.h"
#include "salt/text_verify.h" /* SaltTensorHostRealizations */
#include "lfm_adapter.h"
#include "lfm_quant.h" /* the model's own affine-6 decode */

/* One model-owned binding for every tensor descriptor. STATIC tensors resolve
 * from trunk. SELECTED tensors resolve from the engine's existing bounded
 * SaltCache by their canonical logical_resource_id. */
typedef struct LfmHostBinding {
    const uint8_t *trunk;
    SaltCache *cache;
    SaltKvCache *compute_pool;
    uint32_t workers;
} LfmHostBinding;

/* The realization table for every LFM tensor. Returned rather than exported as
 * a symbol so the adapter can hand it to the index without the engine naming
 * anything LFM-specific. */
const SaltTensorHostRealizations *lfm_host_realizations(void);

int lfm_host_affine_q4_batch(LfmHostBinding *binding,
                             const void *vals, const void *scales,
                             const void *biases, int rows, int cols,
                             int batch, const float *inputs, float *outputs);

#endif /* SALT_LFM_HOST_H */
