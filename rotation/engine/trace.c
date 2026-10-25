/* The trace's own translation unit.  One static table of gate names, one counter per branch, and a
 * dump that formats the emitted lines.  Nothing here allocates, reads the environment, or prints. */
#include "salt/trace.h"
#include <stdio.h>

static const char *const _reg_name[SALT_TRACE_REGISTERS] = {
    [1]  = "su_inf_fg1",   /* rotation enabled                    br1 on            br2 off        */
    [2]  = "su_inf_fg2",   /* the ledger view is published        br1 yes           br2 no         */
    [3]  = "su_inf_fg3",   /* the layer has any gen               br1 yes           br2 no         */
    [4]  = "su_inf_fg4",   /* the index is inside the ledger      br1 yes           br2 no         */
    [5]  = "su_inf_fg5",   /* the gen arrived                     br1 arrived       br2 expired    */
    [6]  = "su_inf_fg6",   /* the slot resolved                   br1 assigned      br2 range br3 - */
    [7]  = "su_inf_fg7",   /* salt_cache_rotate_layer accepted    br1 ok            br2 refused    */
    [8]  = "su_inf_fg8",   /* the loader wrote gen                br1 wrote         br2 skipped    */
    [9]  = "su_inf_fg9",   /* the shim served                     br1 served        br2 fell back  */
    [10] = "su_inf_fg10",  /* the priming passed its slot map     br1 real map      br2 null       */
    [11] = "su_inf_fg11",  /* the watch expired: WHICH layer      first/last = layer                */
    [12] = "su_inf_fg12",  /* the watch expired: WHICH expert     first/last = expert               */
    [13] = "su_inf_fg13",  /* the register's stride (ne)              first/last = n_experts            */
    [14] = "su_inf_fg14",  /* the loader's stride                 first/last = n passed to it       */
    [15] = "su_inf_fg15",  /* the schedule served the layer       br1 batch ready   br2 not ready    */
    [16] = "su_inf_fg16",  /* the signal site                     first/last = the layer signalled   */
    [17] = "su_inf_fg17",  /* the sequence step's state sum       first/last = (0,0)(0,1)(1,1) sum   */
    [18] = "su_inf_fg18",  /* the resource-request step site      first/last = the layer             */
    [19] = "su_inf_fg19",  /* the cell-loop step                  first/last = the layer             */
    [20] = "su_inf_fg20",  /* the state sum the sequence reported  first/last = 0..2                 */
    [21] = "su_inf_fg21",  /* the value salt_lr_step RECEIVED      first/last = the layer            */
    [22] = "su_inf_fg22",  /* the value the loader CONSUMED        first/last = the layer            */
    [30] = "eng_layer",      /* the engine's own per-layer boundary  first/last = the layer         */
    [31] = "eng_cell_kind",
    [40] = "rot_bind_noarg",    /* the registration had no slot or no address                          */
    [41] = "rot_bind_refused",  /* the engine's bind refused                                          */
    [42] = "rot_loaded",        /* layers the rotator has loaded                                      */
    [43] = "rot_served",
    [54] = "res_site4_acquire",
    [55] = "decode_route",      /* br1 ordinary token executor; br2 compiled GPU-only */
    [56] = "nfq_fast_admitted", /* one root verdict; value is N*F*Q check capacity */
    [57] = "nfq_fast_executed", /* value is executed cheap token-ID checks */
    [58] = "nfq_fast_cancelled",/* value is retired suffix alternatives */
    [59] = "nfq_fast_model_submissions", /* value remains zero */
    [23] = "mtp_rounds",
    [24] = "mtp_steps",
    [25] = "mtp_root_matches",
    [26] = "mtp_root_misses",
    [27] = "mtp_proposed_tokens",
    [28] = "mtp_matvecs",
    [29] = "mtp_attention_calls",
    [32] = "mtp_draft_n",
    [33] = "mtp_source_bytes",
    [34] = "mtp_ready",
    [35] = "mtp_metal_matvecs",
    [36] = "mtp_runway",
    [37] = "mtp_runway_growths",
    [38] = "mtp_runway_shrinks",
    [39] = "target_sublane_refused",
    [48] = "target_route_execute_refused",
    [49] = "target_execute_stage_refused",
    [88] = "metal_finish_command_status",
    [89] = "metal_finish_result",
    [106] = "target_residency_mode",
    [107] = "target_global_union",
    [108] = "target_global_hot_floor",

    [255] = "rot_engine_arena_shared",
    [254] = "rot_engine_arena_bytes",
    [253] = "lr_threads",
    [252] = "rot_step_failed",
    [251] = "rot_bind_count",
    [250] = "rot_arena_rc",
    [249] = "lr_slot_u32_1",
    [248] = "lr_slot_u32_0",
    [247] = "rot_slot_bind_failed",
    [246] = "state_plugin_materialized",
    [245] = "lr_read_errno",
    [244] = "lr_read_failed",
    [243] = "gate_no_new_buffer",
    [242] = "disp_input_count_mismatch",
    [241] = "disp_gpu_first",
    [240] = "disp_cpu_count",
    [239] = "disp_no_assignment",
    [238] = "disp_cell_identity",
    [237] = "disp_next_cell",
    [236] = "disp_program",
    [235] = "disp_no_cb",
    [234] = "disp_no_command",
    [233] = "disp_no_state",
    [232] = "disp_gpu_count",
    [231] = "disp_actual",
    [230] = "disp_input_count",
    [229] = "disp_phase",
    [228] = "down_layer",
    [227] = "down_mask",
    [226] = "desc_gv_k",
    [225] = "desc_uv_k",
    [224] = "desc_down_id",
    [223] = "desc_up_id",
    [222] = "desc_logical_range",
    [221] = "desc_group_zero",
    [220] = "chain_buffers",
    [219] = "chain_request_zero",
    [218] = "chain_layer_range",
    [217] = "chain_rows_expected",
    [216] = "chain_row_prefix",
    [215] = "chain_request_count",
    [214] = "chain_phase",
    [213] = "mem_limit_parsed_mb",
    [212] = "gate_slot_layer",
    [211] = "gate_slot_expert",
    [210] = "req_selected_raw",
    [209] = "req_occupancy_overflow",
    [208] = "req_n_experts",
    [207] = "req_selected_expert",
    [206] = "req_top_k",
    [205] = "req_n_experts",
    [204] = "req_no_plan",
    [203] = "req_phase",
    [202] = "req_layer_range",
    [201] = "req_no_command",
    [200] = "req_no_state",
    [199] = "req_jobs",
    [198] = "req_cursor",
    [197] = "req_grouped_expert",
    [196] = "req_grouped_canonical",
    [195] = "req_capacity",
    [190] = "lr_slot_refused",
    [191] = "lr_slot_refused_arg",
    [189] = "arena_s_over_mb",
    [188] = "arena_v_over_mb",
    [187] = "backend_arena_mb",
    [186] = "prepare_exit_rss_mb",
    [185] = "prepare_entry_rss_mb",
    [184] = "rot_null_target_storage",
    [183] = "rot_null_target_layer",
    [182] = "mem_limit_layer",
    [181] = "mem_limit_rss_mb",
    [180] = "state_plugin_source_released",
    [179] = "state_plugin_reset_inplace",
    [178] = "state_plugin_reset_rows",
    [177] = "state_plugin_mapped",
    [176] = "exit_past_complete",
    [175] = "barrier_exit_5",
    [174] = "barrier_exit_4",
    [173] = "barrier_exit_3",
    [172] = "barrier_exit_2",
    [171] = "barrier_exit_1",
    [163] = "chain_mask_layer",
    [162] = "chain_mask",
    [161] = "cc_op_status",
    [155] = "sub_phase",
    [156] = "rotation_cpu_decode_hits",
    [157] = "rotation_cpu_decode_total_misses",
    [154] = "sub_encoded_cells",
    [153] = "sub_next_cell",
    [152] = "shared_attach_payload",
    [151] = "shared_attach_rows",
    [150] = "shared_attach_refresh_rc",
    [149] = "shared_attach_publish_rc",
    [148] = "shared_attach_hash_rc",
    [147] = "ho_slots_have",
    [146] = "ho_slots_needed",
    [145] = "ho_slot_bytes",
    [144] = "ho_nslot",
    [143] = "ho_arena",
    [142] = "mark_full_gpu_entered",
    [141] = "mark_cache_init_done",
    [140] = "mark_backend_up",
    [139] = "ifg_exit_19",
    [138] = "ifg_exit_18",
    [137] = "ifg_exit_17",
    [136] = "ifg_exit_16",
    [135] = "ifg_exit_15",
    [134] = "ifg_exit_14",
    [133] = "ifg_exit_13",
    [132] = "ifg_exit_12",
    [131] = "ifg_exit_11",
    [130] = "ifg_exit_10",
    [129] = "ifg_exit_9",
    [128] = "ifg_exit_8",
    [127] = "ifg_exit_7",
    [126] = "ifg_exit_6",
    [125] = "ifg_exit_5",
    [124] = "ifg_exit_4",
    [123] = "ifg_exit_3",
    [122] = "ifg_exit_2",
    [121] = "ifg_exit_1",
    [112] = "reg_bound",
    [113] = "shared_map_enter",
    [114] = "shared_map_rc",
    [115] = "shared_attach_geometry",
    [116] = "shared_attach_address",
    [117] = "rotation_cpu_mode",
    [118] = "rotation_cpu_prepare",
    [119] = "rotation_cpu_step",
    [158] = "rotation_cpu_decode_begin",
    [159] = "rotation_cpu_decode_layer",
    [160] = "rotation_cpu_decode_misses",
    [111] = "reg_skipped",
    [110] = "reg_attempted",
    [105] = "slot_table_cap",
    [104] = "slot_table_last",
    [103] = "slot_table_first",
    [102] = "slot_table_filled",
    [101] = "drv_encode_refused",
    [100] = "drv_barrier_refused",
    [99] = "ext_encode_refused",
    [98] = "ext_barrier_refused",
    [97] = "ext_gate_yield",
    [96] = "rot_thread_pid",
    [95] = "rot_init_pid",
    [94] = "rot_thread_woke",
    [93] = "rot_pthread_rc",
    [92] = "rot_init_guard",
    [91] = "rot_inited",
    [90] = "rot_init_rc",
    [87] = "rot_cpu_ms_7",
    [86] = "rot_cpu_ms_6",
    [85] = "rot_cpu_ms_5",
    [84] = "rot_cpu_ms_4",
    [83] = "rot_cpu_ms_3",
    [82] = "rot_cpu_ms_2",
    [81] = "rot_cpu_ms_1",
    [80] = "rot_cpu_ms_0",
    [79] = "cache_init_step_19",
    [78] = "cache_init_step_18",
    [77] = "cache_init_step_17",
    [76] = "cache_init_step_16",
    [75] = "cache_init_step_15",
    [74] = "cache_init_step_14",
    [73] = "cache_init_step_13",
    [72] = "cache_init_step_12",
    [71] = "cache_init_step_11",
    [70] = "cache_init_step_10",
    [69] = "cache_init_step_9",
    [68] = "cache_init_step_8",
    [67] = "cache_init_step_7",
    [66] = "cache_init_step_6",
    [65] = "cache_init_step_5",
    [64] = "cache_init_step_4",
    [63] = "cache_init_step_3",
    [62] = "cache_init_step_2",
    [61] = "cache_init_step_1",
    [60] = "cache_init_step_0",
    [53] = "res_site3_request",
    [52] = "res_site2_release",
    [51] = "res_site1_noops",
    [50] = "res_expert_stage",
    [47] = "bind_refuse_logical",
    [46] = "bind_refuse_geom",
    [45] = "bind_refuse_payload",
    [44] = "bind_refuse_args",        /* slot lookups answered from the rotation                            */  /* the cell kind the engine is executing  first/last = the kind          */
};

static uint32_t _hits[SALT_TRACE_REGISTERS][SALT_TRACE_BRANCHES];
static int _val_first[SALT_TRACE_REGISTERS], _val_last[SALT_TRACE_REGISTERS];
static uint32_t _val_seen[SALT_TRACE_REGISTERS];
static uint32_t _reg_seen[SALT_TRACE_REGISTERS];

void salt_trace_register(int gate, int branch) {
    if (gate < 1 || gate >= SALT_TRACE_REGISTERS) return;
    if (branch < 1 || branch >= SALT_TRACE_BRANCHES) return;
    _hits[gate][branch]++;
    _reg_seen[gate] = 1u;
}

void salt_trace_value(int gate, int value) {
    if (gate < 1 || gate >= SALT_TRACE_REGISTERS) return;
    if (!_val_seen[gate]) { _val_first[gate] = value; _val_seen[gate] = 1u; }
    _val_last[gate] = value;
}

void salt_trace_reset(void) {
    for (int g = 0; g < SALT_TRACE_REGISTERS; g++) {
        _reg_seen[g] = 0u;
        for (int b = 0; b < SALT_TRACE_BRANCHES; b++) _hits[g][b] = 0u;
    }
}

void salt_trace_dump(void (*emit)(const char *line, void *ctx), void *ctx) {
    char line[160];
    if (!emit) return;
    for (int g = 1; g < SALT_TRACE_REGISTERS; g++) {
        if (!_reg_seen[g]) continue;
        int off = snprintf(line, sizeof line, "# register %s", _reg_name[g] ? _reg_name[g] : "?");
        for (int b = 1; b < SALT_TRACE_BRANCHES; b++) {
            if (!_hits[g][b]) continue;
            off += snprintf(line + off, sizeof line - (size_t)off, " br%d=%u", b, _hits[g][b]);
        }
        /* ALWAYS, not only when they differ: a counter that is constant is exactly the reading that
         * matters, and suppressing it makes 0 and 3840 indistinguishable. */
        if (_val_seen[g])
            snprintf(line + off, sizeof line - (size_t)off, " first=%d last=%d",
                     _val_first[g], _val_last[g]);
        emit(line, ctx);
    }
}
