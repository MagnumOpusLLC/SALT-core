/* bitmath.c -- deterministic transcendental math (bit-exact across
 * compilers, libms, and ISAs).
 *
 * WHY: the KV cache is the product ("knowledge carved in stone"). libm
 * expf/logf/powf/sinf/cosf are NOT correctly-rounded -- Apple vs glibc
 * vs musl can differ in the last ulp, so a KV prefilled on one platform
 * is not byte-identical to another. The softmax expf alone diverged
 * GQA layers at token 0 (measured: Mac clang vs Linux gcc, both NEON).
 *
 * HOW: every function here uses ONLY IEEE-exact operations (add, mul,
 * div, roundf, ldexpf, frexpf) and fixed-order polynomials with fixed
 * coefficients (fdlibm-derived, public domain). Range reductions run
 * in double precision (exact for our ranges); -ffp-contract=off keeps
 * compilers from fusing mul+add. Same source -> same bits everywhere.
 */
#include "salt/bitmath.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* exp(x): x = n*ln2 + r, exp(x) = ldexp(exp(r), n).                  */
/* ------------------------------------------------------------------ */
float salt_expf(float x) {
    if (x != x) return x;
    if (x > 88.722839f) return 3.402823466e+38f;
    if (x < -87.336548f) return 0.0f;
    /* double reduction: r = x - n*ln2 exact to ~2^-50 */
    double n = round((double)x * 1.442695040888963407359924681001892137426645954152985934135449406931);
    double r = (double)x - n * 0.693147180559945309417232121458176568075500134360255254120680009;
    /* exp(r) Horner, |r| <= ln2/2, degree 8 Taylor */
    float rf = (float)r;
    float p = 2.48015873015873e-05f;            /* 1/40320 */
    p = p * rf + 1.98412698412698e-04f;         /* 1/5040  */
    p = p * rf + 1.38888888888889e-03f;         /* 1/720   */
    p = p * rf + 8.33333333333333e-03f;         /* 1/120   */
    p = p * rf + 4.16666666666667e-02f;         /* 1/24    */
    p = p * rf + 1.66666666666667e-01f;         /* 1/6     */
    p = p * rf + 5.0e-01f;                      /* 1/2     */
    p = p * rf + 1.0f;
    p = p * rf + 1.0f;
    return ldexpf(p, (int)n);
}

/* ------------------------------------------------------------------ */
/* log(x): x = m*2^e, m in [0.5,1); log = log(m) + e*ln2.            */
/* log(m) = 2u(1 + u^2/3 + u^4/5 + ...), u = (m-1)/(m+1).            */
/* ------------------------------------------------------------------ */
float salt_logf(float x) {
    if (x != x) return x;
    if (x <= 0.0f) return -1.0f / 0.0f;
    int e;
    float m = frexpf(x, &e);                     /* m in [0.5, 1) */
    /* log(m) in DOUBLE: u = (m-1)/(m+1), |u| <= 1/3;
     * log(m) = 2u*(1 + s/3 + s^2/5 + s^3/7 + ...), s = u^2.
     * Double arithmetic is IEEE-exact (deterministic) and converges
     * to ~1e-16 for |u| <= 1/3 with 12 terms -- far beyond float. */
    double u = ((double)m - 1.0) / ((double)m + 1.0);
    double s = u * u;
    double z = 1.0 / 25.0;                       /* s^11 coeff */
    z = z * s + 1.0 / 23.0;
    z = z * s + 1.0 / 21.0;
    z = z * s + 1.0 / 19.0;
    z = z * s + 1.0 / 17.0;
    z = z * s + 1.0 / 15.0;
    z = z * s + 1.0 / 13.0;
    z = z * s + 1.0 / 11.0;
    z = z * s + 1.0 / 9.0;
    z = z * s + 1.0 / 7.0;
    z = z * s + 1.0 / 5.0;
    z = z * s + 1.0 / 3.0;
    z = z * s + 1.0;
    double L = 2.0 * u * z;
    /* e*ln2 with a full-precision double constant. */
    static const double LN2 = 0.693147180559945309417232121458176568;
    return (float)(L + (double)e * LN2);
}

/* ------------------------------------------------------------------ */
/* powf(x,y) = exp(y * log(x)).                                       */
/* ------------------------------------------------------------------ */
float salt_powf(float x, float y) {
    if (x == 1.0f || y == 0.0f) return 1.0f;
    if (x > 0.0f && y == -0.5f) return salt_rsqrtf(x);
    if (x > 0.0f) return salt_expf(y * salt_logf(x));
    return x;
}

static double invsqrt64(float x) {
    uint32_t bits;
    float seed;
    memcpy(&bits, &x, sizeof bits);
    bits = UINT32_C(0x5f3759df) - (bits >> 1);
    memcpy(&seed, &bits, sizeof seed);
    double y = (double)seed;
    double xd = (double)x;
    for (int i = 0; i < 6; i++) {
        double yy = y * y;
        double correction = 1.5 - 0.5 * xd * yy;
        y = y * correction;
    }
    return y;
}

float salt_rsqrtf(float x) {
    if (x != x) return x;
    if (x < 0.0f) return 0.0f / 0.0f;
    if (x == 0.0f) return 1.0f / 0.0f;
    if (x == 1.0f) return 1.0f;
    if (x < 0x1p-126f)
        return salt_rsqrtf(x * 0x1p24f) * 0x1p12f;
    return (float)invsqrt64(x);
}

float salt_sqrtf(float x) {
    if (x != x) return x;
    if (x < 0.0f) return 0.0f / 0.0f;
    if (x == 0.0f || x == 1.0f) return x;
    if (x < 0x1p-126f)
        return salt_sqrtf(x * 0x1p24f) * 0x1p-12f;
    return (float)((double)x * invsqrt64(x));
}

float salt_tanhf(float x) {
    float magnitude, exponential, result;
    if (x != x) return x;
    if (x >= 10.0f) return 1.0f;
    if (x <= -10.0f) return -1.0f;
    magnitude = x < 0.0f ? -x : x;
    exponential = salt_expf(2.0f * magnitude);
    result = (exponential - 1.0f) / (exponential + 1.0f);
    return x < 0.0f ? -result : result;
}

/* ------------------------------------------------------------------ */
/* sin/cos via double range reduction to [-pi/4, pi/4] and fdlibm     */
/* float polynomials.                                                 */
/* ------------------------------------------------------------------ */
static const double PIO2_HI = 1.57079632679489655800e+00; /* pi/2 hi */
static const double PIO2_LO = 6.12323399573676603587e-17;  /* pi/2 lo */
static const double INV_PIO2 = 6.36619772367581382433e-01; /* 2/pi  */

static void sin_cos_reduce(float x, int *quad, float *r) {
    double d = (double)x;
    double n = round(d * INV_PIO2);
    double rr = d - n * PIO2_HI;
    rr -= n * PIO2_LO;
    /* quadrant: n mod 4, careful with negative n */
    long long q = (long long)n;
    q = q % 4;
    if (q < 0) q += 4;
    *quad = (int)q;
    *r = (float)rr;
}

static float sin_poly(float x) {
    /* fdlibm __kernel_sinf: sin = x + x^3*S1 + x^5*S2 + x^7*S3 */
    static const float S1 = -1.66666666641626524e-01f;
    static const float S2 =  8.33302275008926213e-03f;
    static const float S3 = -1.95152958910081473e-04f;
    float x2 = x * x;
    float z = x2 * S3 + S2;      /* S2 + x2*S3            */
    z = x2 * z + S1;             /* S1 + x2*S2 + x2^2*S3  */
    z = x2 * z;                  /* x2*S1 + x2^2*S2 + ... */
    return x + x * z;            /* x + x^3*S1 + x^5*S2 + x^7*S3 */
}

static float cos_poly(float x) {
    /* fdlibm __kernel_cosf: C1 C2 C3 */
    static const float C1 =  4.16666679023301001e-02f;
    static const float C2 = -1.38873162549376522e-03f;
    static const float C3 =  2.44331571180994839e-05f;
    float x2 = x * x;
    float z = x2 * C3 + C2;
    z = x2 * z + C1;
    z = x2 * z - 0.5f;
    return 1.0f + x2 * z;   /* fdlibm: 1 - x2/2 + x2*x2*(C1 + x2*(C2 + x2*C3)) */
}

float salt_sinf(float x) {
    if (x != x) return x;
    int q;
    float r;
    sin_cos_reduce(x, &q, &r);
    switch (q) {
    case 0: return  sin_poly(r);
    case 1: return  cos_poly(r);
    case 2: return -sin_poly(r);
    default: return -cos_poly(r);
    }
}

float salt_cosf(float x) {
    if (x != x) return x;
    int q;
    float r;
    sin_cos_reduce(x, &q, &r);
    switch (q) {
    case 0: return  cos_poly(r);
    case 1: return -sin_poly(r);
    case 2: return -cos_poly(r);
    default: return  sin_poly(r);
    }
}

/* ------------------------------------------------------------------ */
/* log1p(x) = log(1+x): u = x/(2+x) series.                          */
/* ------------------------------------------------------------------ */
float salt_log1pf(float x) {
    if (x <= -1.0f) return -1.0f / 0.0f;
    if (x > 1.0f)
        return salt_logf(1.0f + x);   /* large x: u-series diverges */
    /* u = x/(2+x), log(1+x) = 2u*(1 + s/3 + s^2/5 + ...) in double.
     * For |x| <= 1: |u| <= 1/3, converges to ~1e-16 in 12 terms. */
    double u = (double)x / (2.0 + (double)x);
    double s = u * u;
    double z = 1.0 / 25.0;
    z = z * s + 1.0 / 23.0;
    z = z * s + 1.0 / 21.0;
    z = z * s + 1.0 / 19.0;
    z = z * s + 1.0 / 17.0;
    z = z * s + 1.0 / 15.0;
    z = z * s + 1.0 / 13.0;
    z = z * s + 1.0 / 11.0;
    z = z * s + 1.0 / 9.0;
    z = z * s + 1.0 / 7.0;
    z = z * s + 1.0 / 5.0;
    z = z * s + 1.0 / 3.0;
    z = z * s + 1.0;
    return (float)(2.0 * u * z);
}
