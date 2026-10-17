/* LFM2.5-24B-A2B package index (index.txt) reader.
 *
 * The package ships a text index the importer wrote, and the adapter's lookup
 * is driven by it. Format, all fields read from a real index rather than
 * assumed:
 *
 *   LFMIDX1 <abi> <slot_bytes> <layers> <experts> <moe_layers>
 *           <expert_slots> <trunk_bytes> <pool_bytes>
 *   L <layer> <slot_base>            one per layer; -1 when the layer is dense
 *   T <name> <layer> <encoding> <rows> <cols> <v_off> <v_bytes>
 *           <s_off> <s_bytes> <b_off> <b_bytes>
 *   E <component> <layer> <expert> <encoding> <rows> <cols> <v_off> <v_bytes>
 *           <s_off> <s_bytes> <b_off> <b_bytes>
 *
 * Expert offsets are relative to the layer's slot base, so an E entry is only
 * meaningful together with its layer's L entry.
 *
 * The index uses SHORT names (`operator_norm`, not
 * `model.layers.N.operator_norm.weight`) and repeats them per layer, which is
 * why the lookup takes a layer.
 *
 * Model-owned. The engine never sees this file. */

#ifndef SALT_LFM_INDEX_H
#define SALT_LFM_INDEX_H

#include <stddef.h>

#include "lfm_adapter.h"

#define LFM_INDEX_NAME_MAX 32

typedef struct {
    char name[LFM_INDEX_NAME_MAX];
    int layer;      /* -1 for a global tensor */
    int expert;     /* -1 for a T entry */
    int slot_base;  /* the layer's slot base, for an E entry */
    uint32_t encoding;
    uint32_t rows;
    uint32_t cols;
    uint64_t value_offset;
    uint64_t value_bytes;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t bias_offset;
    uint64_t bias_bytes;
} LfmIndexEntry;

typedef struct LfmIndex {
    LfmIndexEntry *entries;
    int capacity;
    int count;
    int slot_base[LFM_LAYERS]; /* -1 when the layer is dense */
    /* The host realization every tensor this index produces will carry, and a
     * counter for the per-tensor stable handle the engine requires. */
    const SaltTensorHostRealizations *host_ops;
    uint64_t stable_next;
    /* Header values, kept so a caller can check the geometry it was told. */
    uint64_t slot_bytes;
    int layers;
    int experts;
    int moe_layers;
    int expert_slots;
    uint64_t trunk_bytes;
    uint64_t pool_bytes;
} LfmIndex;

/* Parse into caller-owned storage. Returns 0, or -1 on a malformed index, an
 * unknown line type, or more entries than capacity. */
int lfm_index_parse(LfmIndex *index, const char *text, size_t length);

/* An LfmTensorLookup over the parsed index. Global tensors use layer -1. */
int lfm_index_lookup(void *context, int layer, const char *name,
                     LfmTensorSpec *out);

/* The entries for one (layer, expert), for the expert descriptors. */
const LfmIndexEntry *lfm_index_expert(const LfmIndex *index, int layer,
                                      int expert, const char *component);

#endif /* SALT_LFM_INDEX_H */
