#ifndef SALT_QUANT_H
#define SALT_QUANT_H

#include <stdint.h>

/* The quant modules: the format's encode/decode contract behind ONE
 * interface. The math core calls the module's matvec and never
 * branches on the format -- the quant changes the representation,
 * never the math (the bit-identity contract: each module reproduces
 * the same row order, staged dequant, FMA sequence). A new format
 * is a new module fill, no core change. */

typedef struct {
    int bits;             /* 4 (q4/nibble), 8 (q8), 16 (bf16), 8f (fp8) */
    int expert_nbytes;    /* the per-expert stride (manifest-derived) */
    int decode_width;     /* the SIMD decode width (q4: 16/32) */
    int sr, sc;           /* the F8 block-scale grid (0 = none) */
    const char *name;     /* "mlx4", "mlx8", "bf16", "fp8" */

    /* the matvec: y[R] += W[R][C] * x[C] in the module's format.
     * vals/scales/bias are the module's encoded forms; the row
     * order and the accumulation sequence are the contract. */
    int (*matvec)(const void *vals, const void *scales, const void *bias,
                  int R, int C, const float *x, float *y);

    /* the batched form: B rows of x against the same weight matrix,
     * row-identical to matvec (the chunked prefill's engine). */
    int (*matvec_batch)(const void *vals, const void *scales,
                        const void *bias, int R, int C, int B,
                        const float *xs, float *ys);
} SaltQuant;

/* the registry: pick a module by the track's bits. Returns the
 * module or NULL for an unsupported format. */
const SaltQuant *salt_quant_get(int bits);

/* the built-in modules (the fill lives in src/quant.c) */
extern const SaltQuant salt_quant_q4;   /* the mlx4 4-bit track */
extern const SaltQuant salt_quant_q8;   /* the mlx8 8-bit track */
extern const SaltQuant salt_quant_bf16; /* the reference bf16 */
extern const SaltQuant salt_quant_fp8;  /* the fp8 track */

#endif /* SALT_QUANT_H */
