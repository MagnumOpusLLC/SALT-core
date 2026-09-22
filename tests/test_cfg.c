/* gate: config reader refuses missing keys; parses complete fixture. */
#include "salt/salt.h"
#include "salt/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char dir[] = "/tmp/salt_cfg_XXXXXX";
    if (!mkdtemp(dir)) return 1;
    char p[512];
    snprintf(p, sizeof p, "%s/config.json", dir);

    /* missing expert_nbytes must refuse the load */
    FILE *f = fopen(p, "w");
    if (!f) return 1;
    fprintf(f, "{\"n_layers\":2,\"n_experts\":8,\"topk\":1,\"n_shared\":1,"
               "\"hidden\":64,\"latent\":32,\"moe_inter\":64}\n");
    fclose(f);
    SaltCfg cfg;
    if (salt_cfg_load(&cfg, dir) == 0) {
        fprintf(stderr, "expected refusal for missing key\n");
        return 1;
    }

    /* complete config must parse with exact values */
    f = fopen(p, "w");
    if (!f) return 1;
    fprintf(f, "{\"n_layers\":2,\"n_experts\":8,\"topk\":1,\"n_shared\":1,"
               "\"hidden\":64,\"latent\":32,\"moe_inter\":64,"
               "\"expert_nbytes\":4096,"
               "\"linear_num_key_heads\":7,"
               "\"linear_num_value_heads\":14,"
               "\"linear_key_head_dim\":11,"
               "\"linear_value_head_dim\":13}\n");
    fclose(f);
    if (salt_cfg_load(&cfg, dir) != 0) {
        fprintf(stderr, "expected success\n");
        return 1;
    }
    if (cfg.n_layers != 2 || cfg.n_experts != 8 || cfg.topk != 1 ||
        cfg.expert_nbytes != 4096 || cfg.seed != 7 ||
        cfg.linear_num_key_heads != 7 ||
        cfg.linear_num_value_heads != 14 ||
        cfg.linear_key_head_dim != 11 ||
        cfg.linear_value_head_dim != 13) {
        fprintf(stderr, "bad parsed values\n");
        return 1;
    }
    {
        int kh, vh, kd, vd, rows;
        if (salt_model_linear_geometry(&cfg, &kh, &vh, &kd, &vd,
                                       &rows) != 0 ||
            kh != 7 || vh != 14 || kd != 11 || vd != 13 || rows != 336) {
            fprintf(stderr, "bad linear geometry\n");
            return 1;
        }
        cfg.linear_num_value_heads = 13;
        if (salt_model_linear_geometry(&cfg, &kh, &vh, &kd, &vd,
                                       &rows) == 0) {
            fprintf(stderr, "expected invalid linear head ratio\n");
            return 1;
        }
    }
    if (setenv("SALT_MODEL", "gemma4-26b-a4b", 1) != 0) return 1;
    if (salt_cfg_load(&cfg, dir) != 0) {
        fprintf(stderr, "ready Gemma runtime was rejected\n");
        return 1;
    }
    if (unsetenv("SALT_MODEL") != 0) return 1;
    f = fopen(p, "w");
    if (!f) return 1;
    fprintf(f, "{\"n_layers\":2,\"n_experts\":8,\"topk\":1,\"n_shared\":1,"
               "\"hidden\":64,\"latent\":32,\"moe_inter\":64,"
               "\"expert_nbytes\":4096,"
               "\"linear_num_key_heads\":7,"
               "\"linear_num_value_heads\":13,"
               "\"linear_key_head_dim\":11,"
               "\"linear_value_head_dim\":13}\n");
    fclose(f);
    if (salt_cfg_load(&cfg, dir) == 0) {
        fprintf(stderr, "expected config refusal for invalid linear geometry\n");
        return 1;
    }
    unlink(p);
    rmdir(dir);
    return 0;
}
