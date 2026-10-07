#include "salt/moe_group.h"
#include "salt/text_exec.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int salt_moe_group_execute(const SaltMoEGroupJob *job,
                           const SaltMoEGroupOps *ops,
                           void *context) {
    unsigned char *used = NULL, *acquired = NULL;
    void **leases = NULL;
    void **batch_leases = NULL, **run_leases = NULL;
    int *expert_ids = NULL, *run_groups = NULL;
    float *group_inputs = NULL, *group_outputs = NULL;
    float *selection_outputs = NULL;
    size_t bh, selections, selection_elements, group_elements;
    SaltMoEGroupScratch *scratch;
    int rc = -1, release_failed = 0, used_count = 0;
    int acquire_batch = 0, run_batch = 0;
    int owner = salt_text_prefill_team_owner();
    int team = salt_text_prefill_team_current() != NULL;
    if (!job || !ops || !context || !job->selected || !job->weights ||
        !job->inputs || !job->outputs ||
        (!ops->acquire && !ops->acquire_many) ||
        (!ops->run && !ops->run_many) || !ops->release ||
        (ops->acquire_many && ops->acquire_batch < 1) ||
        (ops->run_many && ops->run_batch < 1) ||
        job->batch < 1 || job->topk < 1 ||
        job->n_experts < 1 || job->topk > job->n_experts ||
        job->hidden < 1 ||
        (size_t)job->batch > SIZE_MAX / (size_t)job->hidden ||
        (size_t)job->batch > SIZE_MAX / (size_t)job->topk)
        return -1;
    bh = (size_t)job->batch * (size_t)job->hidden;
    selections = (size_t)job->batch * (size_t)job->topk;
    if (selections > SIZE_MAX / (size_t)job->hidden ||
        bh > SIZE_MAX / sizeof(float))
        return -1;
    selection_elements = selections * (size_t)job->hidden;
    if (selection_elements > SIZE_MAX / sizeof(float)) return -1;
    group_elements = ops->run_many ? selection_elements : bh;
    if (ops->acquire_many) {
        acquire_batch = ops->acquire_batch;
        if (acquire_batch > job->n_experts) acquire_batch = job->n_experts;
    }
    if (ops->run_many) {
        run_batch = ops->run_batch;
        if (run_batch > job->n_experts) run_batch = job->n_experts;
    }
    scratch = job->scratch;
    if (team && !ops->run_many) return -1;
    if (!scratch || !scratch->used || !scratch->acquired ||
        !scratch->leases || !scratch->expert_ids ||
        !scratch->group_inputs || !scratch->group_outputs ||
        !scratch->selection_outputs ||
        scratch->expert_capacity < (size_t)job->n_experts ||
        scratch->group_element_capacity < group_elements ||
        scratch->selection_element_capacity < selection_elements ||
        (acquire_batch > 0 && (!scratch->batch_leases ||
         scratch->acquire_capacity < (size_t)acquire_batch)) ||
        (run_batch > 0 && (!scratch->run_groups ||
         !scratch->run_leases ||
         scratch->run_capacity < (size_t)run_batch))) {
        return -1;
    }
    used = scratch->used;
    acquired = scratch->acquired;
    leases = scratch->leases;
    batch_leases = scratch->batch_leases;
    run_leases = scratch->run_leases;
    expert_ids = scratch->expert_ids;
    run_groups = scratch->run_groups;
    group_inputs = scratch->group_inputs;
    group_outputs = scratch->group_outputs;
    selection_outputs = scratch->selection_outputs;
    if (owner) {
    memset(used, 0, (size_t)job->n_experts);
    memset(acquired, 0, (size_t)job->n_experts);
    memset(leases, 0, (size_t)job->n_experts * sizeof(void *));

    for (int token = 0; token < job->batch; token++) {
        for (int rank = 0; rank < job->topk; rank++) {
            size_t selection = (size_t)token * job->topk + rank;
            int expert = job->selected[selection];
            if (expert < 0 || expert >= job->n_experts ||
                !isfinite(job->weights[selection]))
                goto acquired_done;
            for (int prior = 0; prior < rank; prior++)
                if (job->selected[(size_t)token * job->topk + prior] == expert)
                    goto acquired_done;
            used[expert] = 1;
        }
    }
    for (int expert = 0; expert < job->n_experts; expert++)
        if (used[expert]) expert_ids[used_count++] = expert;
    if (ops->acquire_many) {
        for (int first = 0; first < used_count; first += acquire_batch) {
            int count = used_count - first;
            int acquire_rc, block_ok;
            if (count > acquire_batch) count = acquire_batch;
            memset(batch_leases, 0, (size_t)count * sizeof(void *));
            acquire_rc = ops->acquire_many(
                context, expert_ids + first, count, batch_leases);
            block_ok = acquire_rc == 0;
            for (int i = 0; i < count; i++) {
                int expert = expert_ids[first + i];
                if (!batch_leases[i]) {
                    block_ok = 0;
                    continue;
                }
                leases[expert] = batch_leases[i];
                acquired[expert] = 1;
            }
            if (!block_ok) goto acquired_done;
        }
    } else {
        for (int i = 0; i < used_count; i++) {
            int expert = expert_ids[i];
            if (ops->acquire(context, expert, &leases[expert]) != 0 ||
                !leases[expert])
                goto acquired_done;
            acquired[expert] = 1;
        }
    }
    rc = 0;
    }
acquired_done:
    if (salt_text_prefill_team_sync(owner ? rc : 0) != 0) {
        rc = -1;
        goto done;
    }
    used_count = 0;
    for (int expert = 0; expert < job->n_experts; expert++)
        if (used[expert]) used_count++;
    rc = -1;
    if (ops->run_many) {
        if (run_batch > used_count) run_batch = used_count;
        for (int first = 0; first < used_count; first += run_batch) {
            int count = used_count - first;
            int rows = 0;
            int group_rc = 0;
            if (count > run_batch) count = run_batch;
            if (owner) for (int i = 0; i < count; i++) {
                int expert = expert_ids[first + i];
                int group = 0;
                for (int token = 0; token < job->batch; token++)
                    for (int rank = 0; rank < job->topk; rank++) {
                        size_t selection = (size_t)token * job->topk + rank;
                        if (job->selected[selection] != expert) continue;
                        memcpy(group_inputs + (size_t)(rows + group) * job->hidden,
                               job->inputs + (size_t)token * job->hidden,
                               (size_t)job->hidden * sizeof(float));
                        group++;
                    }
                if (group < 1) { group_rc = -1; break; }
                run_groups[i] = group;
                run_leases[i] = leases[expert];
                rows += group;
            }
            if (salt_text_prefill_team_sync(group_rc) != 0) goto done;
            group_rc = ops->run_many(context, expert_ids + first, run_leases,
                    run_groups, count, group_inputs, group_outputs);
            if (salt_text_prefill_team_sync(group_rc) != 0)
                goto done;
            rows = 0;
            if (owner) for (int i = 0; i < count; i++) {
                int expert = expert_ids[first + i];
                int group = 0;
                for (int token = 0; token < job->batch; token++)
                    for (int rank = 0; rank < job->topk; rank++) {
                        size_t selection = (size_t)token * job->topk + rank;
                        if (job->selected[selection] != expert) continue;
                        memcpy(selection_outputs + selection * job->hidden,
                               group_outputs +
                                   (size_t)(rows + group) * job->hidden,
                               (size_t)job->hidden * sizeof(float));
                        group++;
                    }
                rows += run_groups[i];
            }
            if (salt_text_prefill_team_sync(0) != 0) goto done;
        }
    } else for (int expert = 0; expert < job->n_experts; expert++) {
        int group = 0;
        if (!used[expert]) continue;
        for (int token = 0; token < job->batch; token++)
            for (int rank = 0; rank < job->topk; rank++) {
                size_t selection = (size_t)token * job->topk + rank;
                if (job->selected[selection] != expert) continue;
                memcpy(group_inputs + (size_t)group * job->hidden,
                       job->inputs + (size_t)token * job->hidden,
                       (size_t)job->hidden * sizeof(float));
                group++;
            }
        if (group < 1 || ops->run(context, expert, leases[expert], group,
                                  group_inputs, group_outputs) != 0)
            goto done;
        group = 0;
        for (int token = 0; token < job->batch; token++)
            for (int rank = 0; rank < job->topk; rank++) {
                size_t selection = (size_t)token * job->topk + rank;
                if (job->selected[selection] != expert) continue;
                memcpy(selection_outputs + selection * job->hidden,
                       group_outputs + (size_t)group * job->hidden,
                       (size_t)job->hidden * sizeof(float));
                group++;
            }
    }
    if (owner) {
    memset(job->outputs, 0, bh * sizeof(float));
    for (int token = 0; token < job->batch; token++) {
        float *output = job->outputs + (size_t)token * job->hidden;
        for (int rank = 0; rank < job->topk; rank++) {
            size_t selection = (size_t)token * job->topk + rank;
            const float *expert_output =
                selection_outputs + selection * job->hidden;
            float weight = job->weights[selection];
            for (int d = 0; d < job->hidden; d++)
                output[d] += expert_output[d] * weight;
        }
    }
    }
    rc = 0;

done:
    if (owner && acquired)
        for (int expert = 0; expert < job->n_experts; expert++)
            if (acquired[expert] &&
                ops->release(context, expert, leases[expert]) != 0)
                release_failed = 1;
    if (release_failed) rc = -1;
    return salt_text_prefill_team_sync(rc);
}
