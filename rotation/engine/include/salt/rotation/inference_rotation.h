/* LAYER ROTATION - the inference's side of it, and the only thing the engine calls.
 *
 * When the rotation is on the engine does not go through the cache's expert path at all: it asks here.
 * The three calls are the whole surface - one at prepare to hand over the source and the arena, one per
 * layer for the sequence's step, and one to turn a layer's expert list into arena addresses.
 *
 * Nothing else in the engine's expert handling is reached under the rotation, and nothing here reads the
 * cache beyond the one handover.
 */
#ifndef SALT_ROTATION_INFERENCE_ROTATION_H
#define SALT_ROTATION_INFERENCE_ROTATION_H
#include <stdint.h>
#include <stddef.h>

/* One handover, at prepare: the source, and the geometry.  THE ROTATION OWNS ITS OWN ARENA - it does
 * not borrow the cache's, because nothing else may be used while the rotation is on. */
int salt_rot_inference_init(void *arena, int arena_slots, const void *src, size_t src_len,
                            int fd, uint64_t payload_offset, uint64_t slot_bytes,
                            int n_layers, int n_experts, int threads);

/* One step of the sequence: compute this layer, load the next.  Inert before the handover. */
int salt_rot_inference_step(int layer);
unsigned salt_rot_inference_cpu_mode(void);

/* The addresses for a layer's experts, from the rotation's own slot map.  0 on success. */
int salt_rot_inference_slots(int layer, const int *experts, int n, int *out_slot);

/* 1 once the handover has happened - the engine uses this and nothing else to choose its path. */
int salt_rot_inference_slot(int layer, int expert);
const void *salt_rot_inference_payload(int layer, int expert);
void salt_rot_inference_note_bind_fail(int which);
uint32_t salt_rot_inference_bind_fail(int which);
int salt_rot_inference_active(void);
int salt_rot_inference_arena(const uint8_t **arena, uint64_t *stride,
                             uint64_t *slot_bytes, int *slots);

#endif
uint32_t salt_rot_inference_loads(void);
uint32_t salt_rot_inference_served(void);
const void *salt_rot_inference_layer_range(int layer, uint64_t *bytes_out);
