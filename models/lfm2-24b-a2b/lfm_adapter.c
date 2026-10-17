#include <string.h>

#include "lfm_adapter.h"
#include "lfm_index.h"

#define SLOT(field, name, req) \
    { (name), offsetof(SaltTextLayerExecDesc, field), (req) }

static const LfmLayerTensorSlot SLOTS[] = {
    /* Names are the PACKAGE INDEX names (index.txt), not the safetensors
     * names: the index says `operator_norm` where the container says
     * `model.layers.N.operator_norm.weight`. The lookup is index-driven, so
     * these must be the short forms. */
    SLOT(pre_attention_norm, "operator_norm", LFM_SLOT_ALWAYS),
    SLOT(pre_ffn_norm_2, "ffn_norm", LFM_SLOT_ALWAYS),
    SLOT(router, "router", LFM_SLOT_MOE),
    SLOT(model_input_projection, "conv_in", LFM_SLOT_CONV),
    SLOT(model_kernel, "conv_kernel", LFM_SLOT_CONV),
    SLOT(model_output_projection, "conv_out", LFM_SLOT_CONV),
    /* The index stores it as (1, n_experts); the container tensor is
     * (n_experts,), the same vector read as one row. */
    SLOT(per_expert_scale, "expert_bias", LFM_SLOT_MOE),

    /* One pair per attention layer; a conv layer carries conv_in/conv_kernel/
     * conv_out instead and has neither. */
    SLOT(q_norm, "q_norm", LFM_SLOT_ATTENTION),
    SLOT(k_norm, "k_norm", LFM_SLOT_ATTENTION),

    /* The attention projections. The engine shape-checks these on every
     * attention layer, and the table did not carry them at all, so they were
     * left absent and the shape block refused the layer. */
    SLOT(q, "q", LFM_SLOT_ATTENTION),
    SLOT(k, "k", LFM_SLOT_ATTENTION),
    SLOT(v, "v", LFM_SLOT_ATTENTION),
    SLOT(o, "o", LFM_SLOT_ATTENTION),

    /* The dense branch, present only where dense_intermediate > 0. */
    SLOT(dense_gate, "dense_gate", LFM_SLOT_DENSE),
    SLOT(dense_up, "dense_up", LFM_SLOT_DENSE),
    SLOT(dense_down, "dense_down", LFM_SLOT_DENSE),
    /* LFM's dense branch SHARES `ffn_norm` with the routed path. Leaving this
     * slot absent made the engine's DENSE_NORM cell gather through a zeroed
     * descriptor and segfault -- the engine requires the tensor, and the model
     * genuinely has it: it is the same one pre_ffn_norm_2 carries. Two engine
     * slots, one container tensor. */
    SLOT(pre_ffn_norm_1, "ffn_norm", LFM_SLOT_DENSE),
};

const LfmLayerTensorSlot *lfm_layer_tensor_slots(int *count_out) {
    if (count_out)
        *count_out = (int)(sizeof SLOTS / sizeof SLOTS[0]);
    return SLOTS;
}

const LfmLayerTensorSlot *lfm_layer_slot_for(size_t field_offset) {
    for (size_t i = 0; i < sizeof SLOTS / sizeof SLOTS[0]; i++)
        if (SLOTS[i].field_offset == field_offset)
            return &SLOTS[i];
    return NULL;
}

void lfm_tensor_desc_fill(SaltTextTensorDesc *out, const LfmTensorSpec *spec,
                          const void *binding) {
    if (!out || !spec) {
        if (out)
            memset(out, 0, sizeof *out);
        return;
    }
    memset(out, 0, sizeof *out);
    out->rows = spec->rows;
    out->cols = spec->cols;
    out->binding = binding;
    /* Not zeroed: the engine requires a non-zero stable handle and a host
     * realization, and zeroing them invalidated every descriptor. */
    out->host_ops = spec->host_ops;
    out->stable_handle = spec->stable_handle;
    out->storage.encoding = spec->encoding;
    out->storage.source_class = spec->source_class;
    out->storage.resource_kind = spec->resource_kind;
    out->storage.resource_id = spec->resource_id;
    out->storage.logical_resource_id = spec->logical_resource_id;
    out->storage.group_size = spec->group_size;
    out->storage.flags = spec->flags;
    out->storage.value_offset = spec->value_offset;
    out->storage.scale_offset = spec->scale_offset;
    out->storage.bias_offset = spec->bias_offset;
    out->storage.value_bytes = spec->value_bytes;
    out->storage.scale_bytes = spec->scale_bytes;
    out->storage.bias_bytes = spec->bias_bytes;
}

int lfm_layer_desc_fill(SaltTextLayerExecDesc *out,
                        const SaltTextLayerPlan *plan, int layer,
                        LfmTensorLookup lookup, void *context,
                        const void *binding) {
    int count = 0;
    const LfmLayerTensorSlot *slots;
    int is_attention;

    if (!out || !plan || !lookup || layer < 0 || layer >= LFM_LAYERS)
        return -1;
    if (plan->layer != layer)
        return -1;

    is_attention = lfm_is_attention_layer(layer);
    slots = lfm_layer_tensor_slots(&count);

    for (int i = 0; i < count; i++) {
        SaltTextTensorDesc *field =
            (SaltTextTensorDesc *)((char *)out + slots[i].field_offset);
        LfmTensorSpec spec;
        int present;

        if (slots[i].requirement == LFM_SLOT_ABSENT) {
            /* The container must not have it. A package that grows a K norm is
             * a different package and should not be waved through. */
            memset(&spec, 0, sizeof spec);
            if (slots[i].container &&
                lookup(context, layer, slots[i].container, &spec) == 0)
                return -1;
            memset(field, 0, sizeof *field);
            continue;
        }

        if (slots[i].requirement == LFM_SLOT_DENSE) {
            /* Present only where the layer has a dense branch. */
            if (!(plan->dense_intermediate > 0)) {
                memset(field, 0, sizeof *field);
                continue;
            }
        } else if (slots[i].requirement == LFM_SLOT_MOE &&
                   plan->dense_intermediate > 0) {
            /* A dense layer has no router and no expert bias.
             *
             * Keyed on dense_intermediate, not n_experts: the engine sets
             * plan.n_experts to the model's count on EVERY layer and signals
             * "this layer is dense" through dense_intermediate. Keying on
             * n_experts required a router on layer 0, where the container has
             * none. */
            memset(field, 0, sizeof *field);
            continue;
        } else if (slots[i].requirement == LFM_SLOT_ATTENTION && !is_attention) {
            /* A conv layer has no q_layernorm; leave the slot absent. */
            memset(field, 0, sizeof *field);
            continue;
        } else if (slots[i].requirement == LFM_SLOT_CONV && is_attention) {
            memset(field, 0, sizeof *field);
            continue;
        }

        memset(&spec, 0, sizeof spec);
        present = lookup(context, layer, slots[i].container, &spec) == 0;
        if (!present) {
            fprintf(stderr, "LFM_SLOT_FAIL layer=%d name=%s\n", layer,
                    slots[i].container ? slots[i].container : "(null)");
            return -1; /* required on this layer and not in the container */
        }
        lfm_tensor_desc_fill(field, &spec, binding);
    }
    if (is_attention) {
        out->model_scratch_floats_per_row = 0u;
        out->model_state_bytes = 0u;
    } else {
        out->model_scratch_floats_per_row = 4u * LFM_HIDDEN;
        out->model_state_bytes =
            (size_t)LFM_HIDDEN * LFM_CONV_L * sizeof(float);
    }
    return 0;
}

int lfm_exec_desc_fill(SaltTextModelExecDesc *out,
                       const LfmExecDescInputs *in) {
    LfmTensorSpec spec;

    if (!out || !in || !in->layers || !in->plans || !in->lookup ||
        !in->model || in->layer_count != (uint32_t)LFM_LAYERS)
        return -1;

    memset(out, 0, sizeof *out);
    out->model = in->model;
    out->layers = in->layers;
    out->layer_count = in->layer_count;
    out->tensor_resources = in->resources;
    out->tensor_resource_count = in->resource_count;
    out->vocabulary = LFM_VOCAB;
    out->maximum_context = in->maximum_context;
    out->norm_epsilon = LFM_NORM_EPS;
    out->embedding_scale = 1.0f;

    /* The embedding. Layer -1: a global tensor, not a per-layer one. */
    memset(&spec, 0, sizeof spec);
    if (in->lookup(in->lookup_context, -1, "embedding", &spec) != 0) {
        fprintf(stderr, "LFM_FILL_FAIL step=embedding\n");
        return -1;
    }
    lfm_tensor_desc_fill(&out->embedding, &spec, in->binding);
    /* The head is tied: the container has no lm_head, so the embedding matrix
     * IS the output projection. */
    lfm_tensor_desc_fill(&out->output_head, &spec, in->binding);

    /* The final norm. */
    memset(&spec, 0, sizeof spec);
    if (in->lookup(in->lookup_context, -1, "final_norm", &spec) != 0) {
        fprintf(stderr, "LFM_FILL_FAIL step=final_norm\n");
        return -1;
    }
    lfm_tensor_desc_fill(&out->final_norm, &spec, in->binding);

    /* Every layer. */
    for (uint32_t i = 0; i < in->layer_count; i++) {
        SaltTextLayerPlan plan;
        if (salt_model_text_layer_plan(in->model, (int)i, &plan) != 0) {
            fprintf(stderr, "LFM_FILL_FAIL step=plan layer=%u\n", i);
            return -1;
        }
        in->plans[i] = plan;
        in->layers[i].plan = &in->plans[i];
        /* KNOWN GAP: layer->kv is never set here, and the engine's admission
         * chain tests `!layer->kv` as its second clause -- so a descriptor from
         * this function is refused there, before any conv or shape check runs.
         * That is what the probe's `engine verdict: rc=-1` is reporting. The KV
         * layer descriptor (SaltTextKvLayerDesc, keys/values) must be built per
         * layer before the engine will accept the descriptor. */
        if (lfm_layer_desc_fill(&in->layers[i], &in->plans[i], (int)i,
                                in->lookup, in->lookup_context,
                                in->binding) != 0) {
            fprintf(stderr, "LFM_FILL_FAIL step=layer layer=%u\n", i);
            return -1;
        }
        if (in->index) {
            /* A MoE layer carries experts; a dense layer (0 and 1 here) does
             * not, and asking for them would be refused by the index. */
            if (plan.n_experts > 0 && in->experts &&
                in->experts_per_layer >= plan.n_experts &&
                !in->layers[i].plan->dense_intermediate) {
                SaltTextExpertDesc *block =
                    in->experts + (size_t)i * (size_t)in->experts_per_layer;
                if (lfm_experts_fill(block, plan.n_experts, (int)i,
                                     in->index, in->binding) != 0) {
                    fprintf(stderr, "LFM_FILL_FAIL step=experts layer=%u\n", i);
                    return -1;
                }
                in->layers[i].experts = block;
                in->layers[i].expert_count = (uint32_t)plan.n_experts;
            }
        }
    }

    /* The seams the core calls for model-owned CPU and GPU layer work. */
    out->model_layer_ops = lfm_layer_ops();
    out->model_gpu_ops = in->model_gpu_ops;
    out->model_gpu_context = in->model_gpu_context;
    return 0;
}

int lfm_experts_fill(SaltTextExpertDesc *out, int expert_count, int layer,
                     const struct LfmIndex *index, const void *binding) {
    static const char *const COMPONENTS[3] = {
        "expert_gate", "expert_up", "expert_down"
    };
    const LfmIndex *ix = (const LfmIndex *)index;
    int base;

    if (!out || !ix || expert_count < 1 || layer < 0 || layer >= LFM_LAYERS)
        return -1;
    base = ix->slot_base[layer];
    if (base < 0)
        return -1; /* a dense layer has no expert entries */

    for (int e = 0; e < expert_count; e++) {
        SaltTextTensorDesc *slots[3];
        out[e].expert_id = (uint32_t)e;
        slots[0] = &out[e].gate;
        slots[1] = &out[e].up;
        slots[2] = &out[e].down;

        for (int c = 0; c < 3; c++) {
            const LfmIndexEntry *entry =
                lfm_index_expert(ix, layer, e, COMPONENTS[c]);
            LfmTensorSpec spec;
            uint64_t expert_base;

            if (!entry)
                return -1; /* a missing component is a package defect */
            /* The E offsets are ABSOLUTE in the pool FILE, not slot-relative:
             * expert 0's gate sits at 4096, expert 1's at 5312512, expert 63's
             * at 334434304 -- each exactly POOL_HEADER + k * slot_bytes. The
             * engine's resource for an expert is that expert's own slot, so this
             * base must be SUBTRACTED, and it must include the pool header or
             * every component lands 4096 bytes past its slot and the last one
             * overruns the resource.
             *
             * slot_base is a SLOT INDEX, not a byte offset: the real index has
             * `L 2 0`, `L 3 64`, `L 4 128` with 64 experts per layer, so it
             * counts slots. Treating it as bytes would put every expert of every
             * layer at nearly the same address. */
            expert_base = (uint64_t)LFM_POOL_HEADER_BYTES +
                          ((uint64_t)base + (uint64_t)e) * ix->slot_bytes;
            memset(&spec, 0, sizeof spec);
            spec.rows = entry->rows;
            spec.cols = entry->cols;
            spec.encoding = (SaltTensorEncoding)entry->encoding;
            if (entry->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 ||
                entry->encoding == SALT_TENSOR_ENCODING_AFFINE_Q6 ||
                entry->encoding == SALT_TENSOR_ENCODING_AFFINE_Q8)
                spec.group_size = 64;
            /* SELECTED, not STATIC: an expert payload is a model-stable tensor
             * whose bounded runtime slot becomes READY, and the storage
             * validator requires logical_resource_id to be 0 for anything that
             * is not SELECTED -- which is why this carried a non-zero id and
             * was rejected. */
            spec.source_class = SALT_TENSOR_SOURCE_SELECTED;
            spec.resource_kind = SALT_TENSOR_RESOURCE_EXPERT;
            spec.host_ops = ix->host_ops;
            spec.stable_handle = ++((LfmIndex *)ix)->stable_next;
            /* Experts live in pool.bin, not trunk.bin: resource 1, not 0. The
             * trunk holds the T tensors; pointing an expert at the trunk would
             * resolve it inside the wrong file entirely. */
            /* A CONTIGUOUS RANK over the MoE layers, matching the resource
             * table the runtime declares -- one resource per expert, the dense
             * prefix carrying none. A hardcoded 1 pointed every expert in the
             * model at layer 2's first slot. */
            spec.resource_id =
                (uint32_t)((uint64_t)(layer - LFM_DENSE_LAYERS) *
                               (uint64_t)expert_count +
                           (uint64_t)e) + 1u;
            spec.logical_resource_id =
                (uint64_t)layer * (uint64_t)expert_count + (uint64_t)e;
            /* Rebasing: an offset SMALLER than the base would underflow, so a
             * malformed index is refused rather than wrapped. */
            if (entry->value_offset < expert_base ||
                entry->scale_offset < expert_base ||
                entry->bias_offset < expert_base)
                return -1;
            spec.value_offset = entry->value_offset - expert_base;
            spec.value_bytes = entry->value_bytes;
            spec.scale_offset = entry->scale_offset - expert_base;
            spec.scale_bytes = entry->scale_bytes;
            spec.bias_offset = entry->bias_offset - expert_base;
            spec.bias_bytes = entry->bias_bytes;
            lfm_tensor_desc_fill(slots[c], &spec, binding);
        }
    }
    return 0;
}
