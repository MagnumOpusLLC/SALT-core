/* LFM2.5-24B-A2B -> engine layer-descriptor mapping.
 *
 * The engine's SaltTextLayerExecDesc has fixed slots; the container has its own
 * tensor names. This table is the single place the two meet, so the adapter
 * never has to remember which LFM name fills which slot, and a name that is
 * wrong is wrong in one place.
 *
 * Every name here was read from the container's safetensors headers, not
 * inferred from convention. Two of them would have been guessed wrong:
 *   - the router is `feed_forward.gate.weight`, NOT a tensor named `router`;
 *   - it is F32, as is `feed_forward.expert_bias`, in an otherwise 4-bit
 *     container. Assuming every tensor is affine would mis-decode both.
 *
 * Model-owned. The core declares the slots; this says what fills them. */

#ifndef SALT_LFM_ADAPTER_H
#define SALT_LFM_ADAPTER_H

#include <stddef.h>

#include "salt/text_verify.h"
#include "lfm.h"
#include "lfm_conv.h" /* lfm_layer_ops: the seam this module installs */

/* How a slot must be satisfied on a given layer. */
enum {
    /* Every layer carries it. */
    LFM_SLOT_ALWAYS = 0,
    /* Only the 10 attention layers carry it; a conv layer has no q_layernorm. */
    LFM_SLOT_ATTENTION = 1,
    /* The container has no such tensor at all; the slot must be absent, which
     * is what plan->attention.k_norm_absent declares. */
    LFM_SLOT_ABSENT = 2,
    /* Only where the layer has a dense branch (plan->dense_intermediate > 0);
     * a routed layer must have it absent. */
    LFM_SLOT_DENSE = 3,
    /* Only on a mixture-of-experts layer. A dense layer (0 and 1) has no router
     * and no expert bias in the container at all, so requiring them ALWAYS
     * refused the very first layer. */
    LFM_SLOT_MOE = 4,
    /* Only a model-owned short-convolution layer carries these immutable
     * tensors. Attention layers must leave them absent. */
    LFM_SLOT_CONV = 5
};

typedef struct {
    const char *container; /* LFM tensor name; NULL when LFM_SLOT_ABSENT */
    size_t field_offset;   /* offsetof(SaltTextLayerExecDesc, ...) */
    int requirement;
} LfmLayerTensorSlot;

/* The table and its length. The returned array is static and must not be
 * modified. */
const LfmLayerTensorSlot *lfm_layer_tensor_slots(int *count_out);

/* The slot for a field offset, or NULL. */
const LfmLayerTensorSlot *lfm_layer_slot_for(size_t field_offset);

/* A compact description of one container tensor: everything an engine tensor
 * descriptor needs and nothing else, so the adapter reads one shape/encoding
 * record rather than assembling fields at each call site. */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    /* Required by the engine's descriptor validation: a tensor with
     * stable_handle 0 or host_ops NULL is invalid, whatever else is right. */
    uint64_t stable_handle;
    const SaltTensorHostRealizations *host_ops;
    SaltTensorEncoding encoding;
    SaltTensorSourceClass source_class;
    uint32_t resource_kind;
    uint32_t resource_id;
    uint64_t logical_resource_id;
    uint32_t group_size;
    uint32_t flags;
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
    uint64_t value_bytes;
    uint64_t scale_bytes;
    uint64_t bias_bytes;
} LfmTensorSpec;

/* Fill an engine tensor descriptor from a spec. The caller validates it with
 * salt_tensor_desc_validate; this only fills, so the two concerns stay
 * separable and a fill bug is not reported as a validation failure. */
void lfm_tensor_desc_fill(SaltTextTensorDesc *out, const LfmTensorSpec *spec,
                          const void *binding);

/* The adapter's view of the container: given a layer and a tensor name,
 * produce its spec. The layer is required because the index repeats the same
 * name on every layer -- `operator_norm` appears 40 times -- so a name alone
 * cannot identify a tensor. Global tensors are looked up with layer -1.
 *
 * The caller owns the lookup, so this module never parses the package index and
 * never assumes a file format. Returns 0, or -1 when the container has no such
 * tensor on that layer. */
typedef int (*LfmTensorLookup)(void *context, int layer, const char *name,
                               LfmTensorSpec *out);

/* Fill one layer's descriptor from the layer plan and the container.
 *
 * A slot marked ALWAYS must resolve on every layer. A slot marked ATTENTION
 * must resolve on an attention layer and must NOT on a conv layer, because the
 * container genuinely has no q_layernorm there. A slot marked ABSENT must never
 * resolve -- if the container suddenly has a K norm, that is a package change
 * and should be refused rather than silently ignored.
 *
 * Returns 0, or -1 naming nothing: the caller reports which layer failed. */
int lfm_layer_desc_fill(SaltTextLayerExecDesc *out,
                        const SaltTextLayerPlan *plan, int layer,
                        LfmTensorLookup lookup, void *context,
                        const void *binding);

/* Everything the exec-descriptor assembly needs, supplied by the caller so this
 * module owns no storage: the layer and plan arrays, the expert arrays, the
 * resource table, and the container lookup. */
typedef struct {
    SaltTextLayerExecDesc *layers;
    SaltTextLayerPlan *plans;
    /* One block of expert descriptors, experts_per_layer per layer. */
    SaltTextExpertDesc *experts;
    int experts_per_layer;
    uint32_t layer_count;
    const SaltTensorResourceSpec *resources;
    uint32_t resource_count;
    uint32_t maximum_context;
    LfmTensorLookup lookup;
    void *lookup_context;
    const void *binding;
    const SaltModelDesc *model;
    const SaltTextModelGpuOps *model_gpu_ops;
    void *model_gpu_context;
    /* Optional: when set, each layer's experts are filled from it. Without it
     * the expert slots stay empty, which the engine's admission chain refuses
     * -- so a descriptor built without an index is not runnable, and says so. */
    const struct LfmIndex *index;
} LfmExecDescInputs;

/* Build the engine's exec descriptor: the model identity, the tensor table, the
 * embedding / final norm / tied head, every layer through lfm_layer_desc_fill,
 * and the model-owned layer seam.
 *
 * The head is TIED: output_head is filled from the same container tensor as the
 * embedding, because the container has no lm_head. */
int lfm_exec_desc_fill(SaltTextModelExecDesc *out,
                       const LfmExecDescInputs *in);

/* Fill one layer's expert descriptors from the index's E entries.
 *
 * E offsets are relative to the layer's slot base, so the absolute offset of
 * expert e's component is
 *     slot_base + e * slot_bytes + entry_offset
 * Getting that wrong would read one expert's weights while believing they were
 * another's -- plausible output, wrong model.
 *
 * `expert_count` comes from the layer plan; the caller owns the array. */
struct LfmIndex;
int lfm_experts_fill(SaltTextExpertDesc *out, int expert_count, int layer,
                     const struct LfmIndex *index, const void *binding);

#endif /* SALT_LFM_ADAPTER_H */
