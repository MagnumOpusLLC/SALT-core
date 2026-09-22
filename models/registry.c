#include "salt/model.h"

#include <string.h>

const SaltModelDesc *salt_qwen36_model_descriptor(void);
const SaltModelDesc *salt_gemma4_model_descriptor(void);

const char *salt_model_default_name(void) {
    return "qwen36";
}

const SaltModelDesc *salt_model_get(const char *name) {
    const SaltModelDesc *models[] = {
        salt_qwen36_model_descriptor(),
        salt_gemma4_model_descriptor(),
        NULL,
    };
    if (!name) return NULL;
    for (int i = 0; models[i]; i++)
        if (strcmp(models[i]->name, name) == 0)
            return models[i];
    return NULL;
}
