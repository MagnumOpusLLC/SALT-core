#ifndef SALT_INTERNAL_SHA256_H
#define SALT_INTERNAL_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct SaltSha256 {
    uint32_t h[8];
    uint64_t bits;
    uint8_t block[64];
    size_t used;
} SaltSha256;

void salt_sha256_init(SaltSha256 *state);
void salt_sha256_update(SaltSha256 *state, const void *data, size_t n);
void salt_sha256_final(SaltSha256 *state, uint8_t out[32]);
int salt_sha256_bytes(const void *data, size_t n, uint8_t out[32]);
int salt_sha256_fd(int fd, uint8_t out[32]);
int salt_sha256_path(const char *path, uint8_t out[32]);
int salt_sha256_hex_parse(const char text[64], uint8_t out[32]);

#endif
