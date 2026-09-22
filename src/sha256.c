#if !defined(__APPLE__) && !defined(_GNU_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "sha256.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>


static uint32_t rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}

static void transform(SaltSha256 *s, const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[4*i] << 24) |
               ((uint32_t)block[4*i+1] << 16) |
               ((uint32_t)block[4*i+2] << 8) | block[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t a = w[i - 15], b = w[i - 2];
        uint32_t s0 = rotr32(a, 7) ^ rotr32(a, 18) ^ (a >> 3);
        uint32_t s1 = rotr32(b, 17) ^ rotr32(b, 19) ^ (b >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a=s->h[0], b=s->h[1], c=s->h[2], d=s->h[3];
    uint32_t e=s->h[4], f=s->h[5], g=s->h[6], h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

void salt_sha256_init(SaltSha256 *s) {
    static const uint32_t h[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    memcpy(s->h, h, sizeof h);
    s->bits = 0;
    s->used = 0;
}

void salt_sha256_update(SaltSha256 *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    s->bits += (uint64_t)n * 8u;
    while (n) {
        size_t take = sizeof s->block - s->used;
        if (take > n) take = n;
        memcpy(s->block + s->used, p, take);
        s->used += take;
        p += take;
        n -= take;
        if (s->used == sizeof s->block) {
            transform(s, s->block);
            s->used = 0;
        }
    }
}

void salt_sha256_final(SaltSha256 *s, uint8_t out[32]) {
    uint64_t bits = s->bits;
    s->block[s->used++] = 0x80;
    if (s->used > 56) {
        memset(s->block + s->used, 0, 64 - s->used);
        transform(s, s->block);
        s->used = 0;
    }
    memset(s->block + s->used, 0, 56 - s->used);
    for (int i = 0; i < 8; i++)
        s->block[63 - i] = (uint8_t)(bits >> (8 * i));
    transform(s, s->block);
    for (int i = 0; i < 8; i++) {
        out[4*i] = (uint8_t)(s->h[i] >> 24);
        out[4*i+1] = (uint8_t)(s->h[i] >> 16);
        out[4*i+2] = (uint8_t)(s->h[i] >> 8);
        out[4*i+3] = (uint8_t)s->h[i];
    }
}

int salt_sha256_bytes(const void *data, size_t n, uint8_t out[32]) {
    if (!out || (!data && n != 0)) return -1;
    SaltSha256 s;
    salt_sha256_init(&s);
    if (n != 0) salt_sha256_update(&s, data, n);
    salt_sha256_final(&s, out);
    return 0;
}

int salt_sha256_fd(int fd, uint8_t out[32]) {
    uint8_t buf[32768];
    struct stat before, after;
    if (fd < 0 || !out || fstat(fd, &before) != 0 || before.st_size < 0)
        return -1;
    SaltSha256 s;
    salt_sha256_init(&s);
    off_t off = 0;
    while (off < before.st_size) {
        size_t want = (uint64_t)(before.st_size - off) < sizeof buf
            ? (size_t)(before.st_size - off) : sizeof buf;
        ssize_t n = pread(fd, buf, want, off);
        if (n <= 0) return -1;
        salt_sha256_update(&s, buf, (size_t)n);
        off += n;
    }
    if (fstat(fd, &after) != 0 || after.st_size != before.st_size)
        return -1;
    salt_sha256_final(&s, out);
    return 0;
}

int salt_sha256_path(const char *path, uint8_t out[32]) {
    if (!path || !out) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int rc = salt_sha256_fd(fd, out);
    close(fd);
    return rc;
}

int salt_sha256_hex_parse(const char text[64], uint8_t out[32]) {
    if (!text || !out) return -1;
    for (int i = 0; i < 32; i++) {
        unsigned hi, lo;
        char a = text[2*i], b = text[2*i+1];
        if (a >= '0' && a <= '9') hi = (unsigned)(a - '0');
        else if (a >= 'a' && a <= 'f') hi = (unsigned)(a - 'a' + 10);
        else return -1;
        if (b >= '0' && b <= '9') lo = (unsigned)(b - '0');
        else if (b >= 'a' && b <= 'f') lo = (unsigned)(b - 'a' + 10);
        else return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}
