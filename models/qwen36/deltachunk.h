/* deltachunk.h -- M3 chunked delta-rule kernel (parallel prefill).
 *
 * The Gated DeltaNet serial spine (state[t+1] = f(state[t])) is the
 * prefill wall (~44% of the attn phase). M3 replaces the per-token
 * recurrence with the FLA chunked formulation: an intra-chunk BxB
 * quadratic (triangular decay matrix) + one inter-chunk recurrent
 * step, so tokens WITHIN a chunk process in parallel.
 *
 * MILESTONE M3B1: the degenerate anchor -- salt_delta_chunk_body()
 * processed the chunk by calling the EXACT serial per-token body
 * (salt_attn_lin_body); at B=1 it was bit-identical to the serial
 * path (KV md5 ff3f9e2e...), validating the wiring, state carry, and
 * buffer layout. M3-full (this file) swaps the inner loop for the
 * chunked quadratic: normalize S'_t = S_t/D_t, WY solve
 * (I + β⊙tril(KK^T))·δ̂ = β⊙(v̂ − S_0·K), triangular readout, state
 * carry S_out = D_B·(S_0 + K^T·δ̂). Sub-chunk BT=64.
 *
 * Correctness level: self-consistency (the chunked accumulation order
 * differs from serial; save/load/resume must reproduce the NEW state).
 * Gate: SALT_DELTA_CHUNK=1 (serial passthrough anchor), =2 (chunked).
 * When OFF, the serial path is untouched.
 */
#ifndef SALT_DELTACHUNK_H
#define SALT_DELTACHUNK_H

#include "salt/salt.h"
#include "salt/attn.h"

/* Process one linear-attention layer's chunk: B tokens [t0, t0+B).
 *
 *   states   - [B] pointers to [H] token states (in: pre-layernorm,
 *              out: post-residual... here post-lin-body, skip_o=1)
 *   qkvs     - [B][qkv_rows] batched qkv projection output (row-major)
 *   zbtok    - [B][z_rows] batched z projection output (row-major)
 *   xins     - [B][H] batched input_layernorm output (row-major)
 *   readouts - [B][v_heads*vd] out: per-token readout (o_proj input)
 *
 * The conv ring, lin state (kv->lin), and GQA cache rows are read/
 * written in kv, exactly like the serial path. Returns 0 on success,
 * -1 on layout/config error.
 */
int salt_delta_chunk_body(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                          int L, const uint8_t *tr, SaltKvCache *kv,
                          int t0, int B, float *const *states,
                          float *qkvs, float *zbtok, float *const *xins,
                          float *readouts);

#endif
