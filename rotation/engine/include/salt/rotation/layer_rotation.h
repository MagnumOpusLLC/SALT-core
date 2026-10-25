/* LAYER ROTATION - the whole mechanism, and nothing else.
 *
 * This folder exists only for the layer rotation.  When the rotation flag is on, the inference engine
 * uses these files and no shape from the ordinary expert path: the source, the destination, the copy,
 * the slot map, and the readiness all live here.
 *
 * The sequence, known at initiation:
 *
 *     [compute i, load i+1] -> [compute i+1, load i+2] -> ... -> [compute N, load free]
 *
 * Two ranges, so a load can never land in the range being computed:
 *
 *     layer i lives at  (i & 1) * n_experts + e
 *
 * The pair's state is the state sum of the two ranges: 0 = (0,0), 1 = (0,1), 2 = (1,1).
 */
#ifndef SALT_ROTATION_LAYER_ROTATION_H
#define SALT_ROTATION_LAYER_ROTATION_H
#include <stdint.h>
#include <stddef.h>

#define SALT_LR_MAX_LAYERS  64
#define SALT_LR_MAX_EXPERTS 256
#define SALT_LR_RANGES      2
#define SALT_LR_MAX_THREADS 32

/* EVERYTHING THE ROTATION NEEDS, HANDED OVER ONCE AT INITIATION. */
typedef struct {
    const uint8_t *src;          /* the pool's mapping - the reference; never copied elsewhere */
    size_t         src_len;
    int            fd;           /* the descriptor the load reads through; src is only the reference */
    uint64_t       payload_offset;   /* where the payloads begin in the file, past its header */
    uint64_t       slot_bytes;   /* one expert's payload */
    uint64_t       stride;       /* the arena's spacing per slot: page-aligned so a slot can be bound */
    uint8_t       *arena;        /* the destination: the same expert arena the engine already has */
    int            arena_slots;
    int            n_layers;
    int            n_experts;
    int            threads;      /* the width the load may use: the engine's resolved worker budget W */
} SaltLayerRotationConfig;

/* 0 on success.  Until this succeeds the rotation has no map, and the caller must use the ordinary
 * path - the cache's own startup runs before the program is prepared. */
int  salt_lr_init(const SaltLayerRotationConfig *cfg);
int  salt_lr_inited(void);

/* ONE STEP OF THE SEQUENCE: compute this layer, and load the next.  Blocks only on this layer's range. */
int  salt_lr_step(int compute);
/* Pure-CPU realization: synchronously make this compiled layer resident when
 * the preceding model boundary did not already advance the rotation. */
int  salt_lr_prepare(int layer);
unsigned salt_lr_cpu_mode(void);
unsigned salt_lr_manifest_emit(void);

/* THE ADDRESS THE INFERENCE ASKS FOR, from the slot map known since initiation. */
int  salt_lr_slot(int layer, int expert);
const uint8_t *salt_lr_payload(int layer, int expert);
int  salt_lr_ready(int layer);
int  salt_lr_state(void);

/* The shim: waits on the layer's readiness and returns its init-computed slot. */
int  salt_lr_lookup(int layer, int expert);

/* Counters, for the manifest. */
void salt_lr_stats(uint32_t *inits, uint32_t *steps, uint32_t *loads, uint32_t *ready_hits,
                   uint32_t *waits);

int  salt_lr_gate_next(int compute);   /* the gate: announce, READY, DONE */

/* THE ARENA'S GEOMETRY, FOR THE BACKEND.  The rotation's addresses are fixed at initiation, so the device
 * resources over them are an initialization act: the backend asks for the arena once, at its own prepare,
 * and binds each slot's address.  Nothing here allocates and nothing is created later. */
int  salt_lr_arena(const uint8_t **arena, uint64_t *stride, uint64_t *slot_bytes, int *slots);
uint32_t salt_lr_gate_stats(uint32_t *announce, uint32_t *ready, uint32_t *done);

#endif
void salt_lr_note_bind_fail(int which);
uint32_t salt_lr_bind_fail(int which);
const uint8_t *salt_lr_layer_range(int layer, uint64_t *bytes_out);
void salt_lr_note_mem(int which);   /* real RSS + process CPU time, into the manifest buffer */
