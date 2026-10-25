#ifndef SALT_ROTATION_ENGINE_ARENA_H
#define SALT_ROTATION_ENGINE_ARENA_H

#include "salt/gpu.h"

/* Borrow the engine-owned expert allocation. The rotator writes contents;
 * backend consumers use backend plus offsets. Neither owns another payload. */
const SaltGpuSharedBuffer *salt_rotation_expert_arena(void);

#endif
