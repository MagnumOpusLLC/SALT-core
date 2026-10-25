/* The trace: the inference's SUCCESS PATH as named gates, and the branch taken at each.
 *
 * Notation, as specified:
 *     su_inf_fg<N>          register N on the inference path
 *     su_inf_fg<N>_br<M>    the value recorded at register N
 *
 * Every hit is a plain memory write in the operation path - no fprintf, no allocation, no
 * environment read.  The records are emitted ONCE, at the end of inference, by salt_trace_dump.
 * The gate names live in one table in src/trace.c so a call site only says (gate, branch).
 */
#ifndef SALT_TRACE_H
#define SALT_TRACE_H
#include <stdint.h>

#define SALT_TRACE_REGISTERS   256
#define SALT_TRACE_BRANCHES 8

void salt_trace_register(int gate, int branch);
/* The value carried at a gate: first and last seen.  A count says how many; this says WHICH. */
void salt_trace_value(int gate, int value);
void salt_trace_dump(void (*emit)(const char *line, void *ctx), void *ctx);
void salt_trace_reset(void);

#endif
