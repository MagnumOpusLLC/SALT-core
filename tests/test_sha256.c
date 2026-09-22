#include "sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int check(const void *data, size_t n, const char *hex) {
    uint8_t actual[32], expected[32];
    if (salt_sha256_bytes(data, n, actual) != 0 ||
        salt_sha256_hex_parse(hex, expected) != 0 ||
        memcmp(actual, expected, sizeof actual) != 0) {
        fprintf(stderr, "SHA-256 vector mismatch\n");
        return -1;
    }
    return 0;
}

int main(void) {
    static const char empty[] = "";
    static const char abc[] = "abc";
    if (check(empty, 0,
              "e3b0c44298fc1c149afbf4c8996fb924"
              "27ae41e4649b934ca495991b7852b855") != 0 ||
        check(abc, 3,
              "ba7816bf8f01cfea414140de5dae2223"
              "b00361a396177a9cb410ff61f20015ad") != 0)
        return 1;
    if (salt_sha256_bytes(NULL, 1, (uint8_t[32]){0}) == 0 ||
        salt_sha256_bytes(empty, 0, NULL) == 0) {
        fprintf(stderr, "SHA-256 accepted an invalid argument\n");
        return 1;
    }
    return 0;
}
