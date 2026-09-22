#ifndef SALT_TEXT_VERIFY_INTERNAL_H
#define SALT_TEXT_VERIFY_INTERNAL_H

#include "salt/text_verify.h"

#include <stddef.h>
#include <stdint.h>

/* Pure startup query shared by the CPU-only and heterogeneous executors. */
int salt_text_cpu_tensor_scratch_requirement(
    const SaltTextVerifyProgram *program, size_t *maximum_bytes);

/* Execute one executor-owned contiguous range of complete rows or jobs. The
 * caller supplies the canonical host arena and package CPU scratch. Outputs
 * are written directly at the program cell's canonical destination offsets. */
int salt_text_cpu_execute_cell_range(
    const SaltTextVerifyProgram *program,
    const SaltTextExecutionCell *cell,
    uint32_t source_position,
    const int32_t *candidate_token_ids,
    uint32_t candidate_count,
    SaltTextExecutionSlice assigned_complete,
    unsigned char *canonical_host_arena,
    size_t canonical_host_bytes,
    void *tensor_scratch,
    size_t tensor_scratch_bytes,
    SaltTextVerifyBackendStats *stats);

#endif
