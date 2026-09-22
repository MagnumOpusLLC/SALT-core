#ifndef SALT_MOE_GROUP_H
#define SALT_MOE_GROUP_H

#include <stddef.h>

/* Caller-owned fixed-address scratch. The model binds these slices once from
 * its startup arena; grouped execution never grows or replaces them. */
typedef struct {
    unsigned char *used;
    unsigned char *acquired;
    void **leases;
    void **batch_leases;
    void **run_leases;
    int *expert_ids;
    int *run_groups;
    float *group_inputs;
    float *group_outputs;
    float *selection_outputs;
    size_t expert_capacity;
    size_t acquire_capacity;
    size_t run_capacity;
    size_t group_element_capacity;
    size_t selection_element_capacity;
} SaltMoEGroupScratch;

/* Model-neutral grouped routed-expert execution. The engine owns deterministic
 * grouping, retained lease lifetime, canonical per-selection output slots, and
 * token/rank-order reduction. Model adapters own router/expert arithmetic. */
typedef struct {
    int batch;
    int topk;
    int n_experts;
    int hidden;
    const int *selected;       /* [batch][topk], distinct within each token */
    const float *weights;      /* [batch][topk] */
    const float *inputs;       /* [batch][hidden] */
    float *outputs;            /* [batch][hidden] */
    SaltMoEGroupScratch *scratch; /* optional fixed-address execution scratch */
} SaltMoEGroupJob;

typedef struct {
    int (*acquire)(void *context, int expert, void **lease);
    /* Optional bounded union acquisition. expert IDs are ascending and leases
     * are compact in the same order. On failure, leave every successfully
     * acquired lease non-NULL so the engine can release it. */
    int (*acquire_many)(void *context, const int *experts, int count,
                        void **leases);
    int acquire_batch;
    int (*run)(void *context, int expert, void *lease, int group,
               const float *inputs, float *outputs);
    /* Optional bounded expert wave. Inputs/outputs concatenate groups in the
     * ascending expert order supplied here; groups[i] gives each row count.
     * Canonical selection placement and reduction remain engine-owned. */
    int (*run_many)(void *context, const int *experts, void *const *leases,
                    const int *groups, int count,
                    const float *inputs, float *outputs);
    int run_batch;
    int (*release)(void *context, int expert, void *lease);
} SaltMoEGroupOps;

int salt_moe_group_execute(const SaltMoEGroupJob *job,
                           const SaltMoEGroupOps *ops,
                           void *context);

#endif
