#ifndef SALT_QWEN36_ATTN_H
#define SALT_QWEN36_ATTN_H

#include "salt/attn.h"

int salt_attn_lin_body(const SaltCfg *cfg, const SaltTrunkLayout *tl, int layer,
                       const uint8_t *trunk, SaltKvCache *kv, int token,
                       float *state, float *qkv, float *z, float *buffer,
                       int skip_output);

int salt_attn_qwen_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int layer,
                        const uint8_t *trunk, float *state, SaltKvCache *kv,
                        int token);
int salt_attn_qwen_proj(const SaltCfg *cfg, const SaltTrunkLayout *tl, int layer,
                        const uint8_t *trunk, float *state, SaltKvCache *kv,
                        int token);
int salt_attn_qwen_body(const SaltCfg *cfg, const SaltTrunkLayout *tl, int layer,
                        const uint8_t *trunk, float *state, SaltKvCache *kv,
                        int token);

int salt_attn_linear_chunk(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                           int layer, const uint8_t *trunk,
                           float *const *states, int first_token, int batch,
                           SaltKvCache *kv);
int salt_attn_gqa_chunk(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                        int layer, const uint8_t *trunk,
                        float *const *states, int first_token, int batch,
                        SaltKvCache *kv);

#endif
