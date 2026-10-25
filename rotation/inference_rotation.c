/* LAYER ROTATION - the inference's side.  Everything below it is the rotation's own mechanism; nothing
 * above it touches the cache's expert path while the rotation is on. */
#include "salt/rotation/inference_rotation.h"
#include <salt/rotation/fail.h>
#include "salt/trace.h"
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>

/* A LOAD FAILURE MUST BE READABLE.  The manifest is built after the load, so a load-time refusal has no
 * emission point - this writes the registers to the path the manifest would use, once, on failure. */
static void _rot_fail_emit(const char *line, void *ctx) {
    FILE *f = (FILE *)ctx;
    if (f && line) fprintf(f, "%s\n", line);
}

void rot_load_fail_emit(void) {
    const char *path = getenv("SALT_TEXT_MANIFEST_DUMP");
    FILE *f;
    if (!salt_lr_manifest_emit()) return;
    if (!path || !path[0]) path = "salt-rotation-load-fail.txt";
    f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# THE ROTATION'S LOAD-FAILURE RECORD\n");
    salt_trace_dump(_rot_fail_emit, f);
    fclose(f);
}
#include "salt/rotation/layer_rotation.h"
#include <stdlib.h>
#include <string.h>

/* THE ROTATION'S OWN ARENA.  Two ranges, so 2 * n_experts slots; nothing is borrowed from the cache. */
static uint64_t _rot_stride;

int salt_rot_inference_init(void *arena, int arena_slots, const void *src, size_t src_len,
                            int fd, uint64_t payload_offset, uint64_t slot_bytes,
                            int n_layers, int n_experts, int threads) {
    SaltLayerRotationConfig cfg;
    /* An arena is OPTIONAL: on the zero-copy mode the storage is the pool mapping itself and the slot is
     * the expert's own address.  A source is what is required, not a destination. */
    if ((arena && arena_slots < SALT_LR_BANK_COUNT * n_experts) || n_layers < 1 || n_experts < 1 ||
        slot_bytes == 0 || !src || src_len == 0) {
        salt_trace_register(92, 1);   /* the guard refused: which operand */
        salt_trace_value(92, (int)((arena ? 1 : 0) | (slot_bytes ? 2 : 0) | (src ? 4 : 0) |
                                   (src_len ? 8 : 0) | (n_layers ? 16 : 0) | (n_experts ? 32 : 0)));
        return SALT_TFAIL(-1);
    }
    cfg.src = (const uint8_t *)src;
    cfg.src_len = src_len;
    cfg.fd = fd;
    cfg.payload_offset = payload_offset;
    cfg.slot_bytes = slot_bytes;
    cfg.stride = _rot_stride;
    cfg.arena = (uint8_t *)arena;
    /* THE ARENA'S REAL CAPACITY, NOT A CLAIM.  This said 2 * n_experts, which is what the two ranges
     * need - not what the storage the engine handed over actually holds.  The copy's top byte is
     * 2 * n_experts * slot_bytes, so a claim larger than the allocation sends every expert copy past the
     * end of the arena, and that is a fault with no return path. */
    cfg.arena_slots = arena_slots;
    cfg.n_layers = n_layers;
    cfg.n_experts = n_experts;
    cfg.threads = threads;
    salt_trace_register(117, 1);
    salt_trace_value(117, (int)salt_lr_cpu_mode());
    {
        int rc = salt_lr_init(&cfg);
        salt_trace_register(90, 1);
        salt_trace_value(90, rc);
        salt_trace_register(91, 1);
        salt_trace_value(91, salt_lr_inited());
        return rc;
    }
}

int salt_rot_inference_step(int layer) {
    if (!salt_lr_inited()) return 0;
    if (!salt_lr_cpu_mode() || salt_lr_decode_active()) return 0;
    salt_trace_register(119, 1);
    salt_trace_value(119, layer);
    return salt_lr_step(layer);
}

unsigned salt_rot_inference_cpu_mode(void) { return salt_lr_cpu_mode(); }

int salt_rot_inference_slots(int layer, const int *experts, int n, int *out_slot) {
    int i;
    if (!salt_lr_inited() || !experts || !out_slot) return SALT_TFAIL(-1);
    if (salt_lr_decode_active()) return salt_lr_decode_fetch(layer, experts, n, out_slot);
    if (salt_lr_cpu_mode()) {
        salt_trace_register(118, 1);
        salt_trace_value(118, layer);
        if (salt_lr_prepare(layer) != 0) return SALT_TFAIL(-1);
    }
    /* THE SHIM'S ONE JOB: read the ledger and WAIT.  salt_lr_slot is the init-computed map and does not
     * wait, so the caller could read a slot whose bytes are not there yet - which is the fault at the
     * first step.  salt_lr_lookup waits on that layer's readiness and then returns the same slot. */
    for (i = 0; i < n; i++) out_slot[i] = salt_lr_lookup(layer, experts[i]);
    return 0;
}

const void *salt_rot_inference_layer_range(int layer, uint64_t *bytes_out) {
    return (const void *)salt_lr_layer_range(layer, bytes_out);
}

int salt_rot_inference_slot(int layer, int expert) {
    int slot;
    if (salt_lr_decode_active())
        return salt_lr_decode_find(layer, expert, &slot) ? slot : -1;
    return salt_lr_slot(layer, expert);
}

const void *salt_rot_inference_payload(int layer, int expert) {
    if (salt_lr_decode_active()) {
        if (salt_lr_decode_zero_copy_active())
            return (const void *)salt_lr_decode_payload(layer, expert);
        const uint8_t *base; uint64_t stride, bytes; int seats, slot;
        if (!salt_lr_decode_find(layer, expert, &slot) ||
            salt_lr_arena(&base, &stride, &bytes, &seats) != 0) return NULL;
        return base + (uint64_t)slot * stride;
    }
    return (const void *)salt_lr_payload(layer, expert);
}

uint32_t salt_rot_inference_loads(void) {
    uint32_t a, b, c, d, e; salt_lr_stats(&a, &b, &c, &d, &e); return c;
}
uint32_t salt_rot_inference_served(void) {
    uint32_t a, b, c, d, e; salt_lr_stats(&a, &b, &c, &d, &e); return b;
}

void salt_rot_inference_note_bind_fail(int which) { salt_lr_note_bind_fail(which); }
uint32_t salt_rot_inference_bind_fail(int which) { return salt_lr_bind_fail(which); }

int salt_rot_inference_arena(const uint8_t **arena, uint64_t *stride,
                             uint64_t *slot_bytes, int *slots) {
    return salt_lr_arena(arena, stride, slot_bytes, slots);
}

int salt_rot_inference_active(void) { return salt_lr_inited(); }
