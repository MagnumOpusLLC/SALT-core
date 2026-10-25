/* Physical DECODE realization only. Router and shared FFN have disjoint
 * destinations; both read the completed DENSE_NORM seats. The original dense
 * operators run unchanged while the existing loader services exact misses.
 * One retained command reference, inside the startup-owned command state.
 * No new payload storage, loader, resource authority or callback thread. */
static int metal_text_overlap_collect(MetalTextProgramCommand *command) {
    id<MTLCommandBuffer> pending = command->overlap_command;
    if (!pending) return 0;
    /* Called after an existing later same-queue completion, or on scrub. The
     * reference must survive until completion even on a failed resource read. */
    [pending waitUntilCompleted];
    int bad = metal_command_succeeded(pending) != 0;
    if (_text_probe_enabled && _text_probe_n < METAL_TEXT_PROBE_MAX) {
        int i = _text_probe_n++;
        _text_probe[i].asynchronous = 1u;
        _text_probe[i].rows = command->input_count;
        _text_probe[i].layer = command->overlap_layer;
        _text_probe[i].operation_phase = (uint32_t)(command->rotation_base /
            ((uint64_t)_arrival_layer_count + 2u)) + 1u;
        _text_probe[i].operation_kind = 2u;
        _text_probe[i].gpu = (float)(([pending GPUEndTime] -
            [pending GPUStartTime]) * 1e3);
        _text_probe[i].commit = command->overlap_commit_ms;
        _text_probe[i].async_submit_ns = command->overlap_submit_ns;
        _text_probe[i].async_observed_done_ns = (uint64_t)_mt_probe_ns();
    }
    if (command->profile_enabled) {
        double elapsed = ([pending GPUEndTime] - [pending GPUStartTime]) * 1e9;
        if (isfinite(elapsed) && elapsed >= 0 && elapsed <= (double)UINT64_MAX) {
            command->profile_prefix_gpu_ns += (uint64_t)(elapsed + 0.5);
            command->profile_prefix_commands++;
        }
    }
    [pending release];
    command->overlap_command = nil;
    _decode_overlap_completed++;
    return bad ? SALT_TFAIL(-1) : 0;
}

static int metal_text_overlap_launch(MetalTextProgramState *state,
        MetalTextProgramCommand *command, uint32_t layer) {
    if (!command->decode_experts || !salt_lr_decode_read_overlap()) return 0;
    if (command->deferred_dense_mask != 15u || command->overlap_command ||
            command->deferred_dense_layer != layer || layer >= state->layer_count)
        return SALT_TFAIL(-1);
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    const MetalTextLayerRefs *refs = &state->layers[layer];
    const SaltTextLayerPlan *plan = state->program->layers[layer].descriptor->plan;
    if (metal_text_new_command(command) != 0) return SALT_TFAIL(-1);
    if (metal_text_encode_projection(state, command, &refs->dense_gate,
            layout->normalized, layout->dense_gate, command->input_count) != 0 ||
        metal_text_encode_projection(state, command, &refs->dense_up,
            layout->normalized, layout->dense_up, command->input_count) != 0 ||
        metal_text_encode_activation(state, command,
            layout->dense_gate, layout->dense_up, layout->dense_chain,
            (size_t)command->input_count * (uint32_t)plan->dense_intermediate) != 0 ||
        metal_text_encode_projection(state, command, &refs->dense_down,
            layout->dense_chain, layout->dense_output, command->input_count) != 0)
        return SALT_TFAIL(-1);
    /* Use the existing initialization-owned fence across command buffers. The
     * routed continuation waits on it through metal_text_encoder, not the CPU. */
    [command->encoder updateFence:command->dependency_fence];
    command->dependency_fence_updated = 1u;
    metal_text_close_encoder(command);
    id<MTLCommandBuffer> pending = command->command;
    command->command = nil;
    double start = _mt_probe_ns();
    [pending commit];
    command->overlap_submit_ns = (uint64_t)_mt_probe_ns();
    command->overlap_commit_ms = (float)((command->overlap_submit_ns - start) * 1e-6);
    command->overlap_layer = layer;
    command->overlap_command = pending;
    command->deferred_dense_mask = 0u;
    command->stats.backend_host_kernel_launch_calls++;
    _decode_overlap_issued++;
    return 0; /* No wait: normal NEED_RESOURCE now runs the existing reads. */
}
