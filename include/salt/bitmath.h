/* bitmath.h -- deterministic transcendental math (bit-exact across
 * compilers/libms/ISAs). See bitmath.c for the why and how. */
#ifndef SALT_BITMATH_H
#define SALT_BITMATH_H

float salt_expf(float x);
float salt_logf(float x);
float salt_powf(float x, float y);
float salt_rsqrtf(float x);
float salt_sqrtf(float x);
float salt_tanhf(float x);
float salt_sinf(float x);
float salt_cosf(float x);
float salt_log1pf(float x);

#endif
