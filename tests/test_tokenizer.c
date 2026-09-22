/*
 * test_tokenizer.c -- unit test for src/tokenizer.c (issue #6 step 5).
 *
 * Builds a tiny byte-level BPE tokenizer.json in a temp dir (all 256
 * byte<->unicode chars as the vocab, ids = byte value; plus "ab",
 * "Ġa", "Ġab"; merges a+b, Ġ+a, Ġ+ab) and asserts:
 *   1. encode("ab ab") -> [id("ab"), id("Ġab")]
 *   2. decode of those ids -> "ab ab"
 *   3. UTF-8 roundtrip: encode("héllo") -> decode -> "héllo"
 *   4. decode of a plain-char vocab entry (non-byte-level path)
 */
#define _POSIX_C_SOURCE 200809L
#include "salt/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* write a codepoint as UTF-8 into buf; returns length */
static int u8(uint32_t cp, char *buf) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    buf[0] = (char)(0xE0 | (cp >> 12));
    buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

int main(void) {
    char path[512];
    snprintf(path, sizeof path, "/tmp/salt_tok_%ld.json", (long)getpid());

    /* build the fixture json: vocab (256 byte-chars + 3 pieces) + merges */
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    fprintf(f, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    int first = 1;
    for (int b = 0; b < 256; b++) {
        /* char for byte b: safe -> itself; unsafe -> 0x100 + idx */
        int safe = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                   (b >= 174 && b <= 255);
        uint32_t cp;
        if (safe) cp = (uint32_t)b;
        else {
            int idx = 0;
            for (int q = 0; q <= b; q++) {
                int qs = (q >= 33 && q <= 126) || (q >= 161 && q <= 172) ||
                         (q >= 174 && q <= 255);
                if (!qs) idx++;
            }
            cp = 0x100u + (uint32_t)(idx - 1);
        }
        char cb[8];
        int cl = u8(cp, cb);
        fprintf(f, "%s\"", first ? "" : ",");
        for (int i = 0; i < cl; i++) {
            if (cb[i] == '"' || cb[i] == '\\') fputc('\\', f);
            fputc((unsigned char)cb[i], f);
        }
        fprintf(f, "\":%d", b);
        first = 0;
    }
    fprintf(f, ",\"ab\":256,\"Ġa\":257,\"Ġab\":258},\"merges\":["
               "[\"a\",\"b\"],[\"Ġ\",\"a\"],[\"Ġ\",\"ab\"]]}}\n");
    fclose(f);

    SaltTokenizer t;
    if (salt_tokenizer_load(&t, path) != 0) {
        printf("FAIL tokenizer load\n");
        return 1;
    }
    if (t.nvocab != 259) {
        printf("FAIL vocab size %d\n", t.nvocab);
        return 1;
    }

    /* 1: encode "ab ab" -> [256, 258] */
    int ids[64];
    int n = salt_tokenizer_encode(&t, "ab ab", ids, 64);
    if (n != 2 || ids[0] != 256 || ids[1] != 258) {
        printf("FAIL encode 'ab ab': n=%d ids=%d,%d\n", n,
               n > 0 ? ids[0] : -1, n > 1 ? ids[1] : -1);
        return 1;
    }

    /* 2: decode [256, 258] -> "ab ab" */
    char out[128];
    int ol = salt_tokenizer_decode(&t, ids, n, out, 128);
    if (ol != 5 || memcmp(out, "ab ab", 5) != 0) {
        printf("FAIL decode: '%s' (%d)\n", out, ol);
        return 1;
    }

    /* 3: UTF-8 roundtrip */
    n = salt_tokenizer_encode(&t, "héllo", ids, 64);
    if (n < 0) { printf("FAIL encode 'héllo'\n"); return 1; }
    ol = salt_tokenizer_decode(&t, ids, n, out, 128);
    if (ol != 6 || memcmp(out, "héllo", 6) != 0) {
        printf("FAIL roundtrip: '%s' (%d)\n", out, ol);
        return 1;
    }

    /* 4: plain-char vocab entry (byte value 0x2F '/' -> char '/') */
    {
        int one = 0x2F;         /* id 47 = '/' (safe byte -> itself) */
        ol = salt_tokenizer_decode(&t, &one, 1, out, 128);
        if (ol != 1 || out[0] != '/') {
            printf("FAIL plain char decode: '%s'\n", out);
            return 1;
        }
    }

    /* 5: loading a DIRECTORY auto-discovers tokenizer.json inside */
    {
        char dir[512];
        snprintf(dir, sizeof dir, "/tmp/salt_tokd_%ld", (long)getpid());
        if (mkdir(dir, 0700) != 0) { printf("FAIL mkdir\n"); return 1; }
        char dp[600];
        snprintf(dp, sizeof dp, "%s/tokenizer.json", dir);
        if (rename(path, dp) != 0) { printf("FAIL rename\n"); return 1; }
        SaltTokenizer t2;
        if (salt_tokenizer_load(&t2, dir) != 0 || t2.nvocab != 259) {
            printf("FAIL dir load\n");
            return 1;
        }
        salt_tokenizer_free(&t2);
    }

    salt_tokenizer_free(&t);
    printf("PASS test_tokenizer (encode/decode/roundtrip)\n");
    return 0;
}
