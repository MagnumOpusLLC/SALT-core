#include "salt/moe.h"

int salt_hc_params(const SaltTrunkLayout *layout, int fn_index,
        int base_index, int scale_index, const uint8_t *trunk, int hidden,
        const float *state, int *count, float *a, float *c, float *b) {
    (void)layout;
    (void)fn_index;
    (void)base_index;
    (void)scale_index;
    (void)trunk;
    (void)hidden;
    (void)state;
    (void)count;
    (void)a;
    (void)c;
    (void)b;
    return 0; /* LFM declares no hyperconnection operator. */
}

void salt_hc_combine(int count, int hidden, const float *a,
        const float *state, float *output) {
    (void)count;
    (void)hidden;
    (void)a;
    (void)state;
    (void)output;
}
