/* tokenc: encode a text prompt with the engine's real tokenizer,
 * print the ids as comma-separated text (for --pids-file fixtures). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "salt/tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: tokenc TOKENIZER.json 'text'\n"); return 2; }
    SaltTokenizer t;
    memset(&t, 0, sizeof t);
    if (salt_tokenizer_load(&t, argv[1]) != 0) {
        fprintf(stderr, "tokenizer load failed\n"); return 2;
    }
    int ids[4096];
    int n = salt_tokenizer_encode(&t, argv[2], ids, 4096);
    if (n <= 0) { fprintf(stderr, "encode failed\n"); return 2; }
    for (int i = 0; i < n; i++)
        printf("%s%d", i ? "," : "", ids[i]);
    printf("\n");
    salt_tokenizer_free(&t);
    return 0;
}
