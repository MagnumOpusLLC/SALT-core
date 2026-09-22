#ifndef SALT_TOOLS_GEMMA4_MEMORY_H
#define SALT_TOOLS_GEMMA4_MEMORY_H

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

typedef struct G4ProcessMemory {
    uint64_t current_rss_bytes;
    uint64_t resident_peak_bytes;
    uint64_t physical_footprint_bytes;
    uint64_t internal_bytes;
    uint64_t external_bytes;
    uint64_t reusable_bytes;
    uint64_t compressed_bytes;
    uint64_t rss_anon_bytes;
    uint64_t rss_file_bytes;
    uint64_t rss_shmem_bytes;
    uint64_t page_table_bytes;
    uint64_t minor_faults;
    uint64_t major_faults;
    uint64_t input_blocks;
    uint64_t output_blocks;
    int detailed;
} G4ProcessMemory;

static inline int g4_parse_linux_status(FILE *status, G4ProcessMemory *out) {
    char line[256];
    unsigned seen_anon = 0, seen_file = 0, seen_shmem = 0, seen_pte = 0;
    if (!status || !out) return -1;
    while (fgets(line, sizeof line, status)) {
        unsigned long long value;
        char suffix[8], extra;
#define G4_PARSE_KB(name, field, seen) \
        if (sscanf(line, name ": %llu %7s %c", \
                   &value, suffix, &extra) == 2) { \
            if (strcmp(suffix, "kB") != 0 || value > UINT64_MAX / 1024u) \
                return -1; \
            out->field = (uint64_t)value * 1024u; \
            seen = 1; \
            continue; \
        }
        G4_PARSE_KB("RssAnon", rss_anon_bytes, seen_anon)
        G4_PARSE_KB("RssFile", rss_file_bytes, seen_file)
        G4_PARSE_KB("RssShmem", rss_shmem_bytes, seen_shmem)
        G4_PARSE_KB("VmPTE", page_table_bytes, seen_pte)
#undef G4_PARSE_KB
    }
    if (ferror(status) || !seen_anon || !seen_file || !seen_shmem || !seen_pte ||
        out->rss_anon_bytes > UINT64_MAX - out->rss_file_bytes ||
        out->rss_anon_bytes + out->rss_file_bytes >
            UINT64_MAX - out->rss_shmem_bytes ||
        out->rss_anon_bytes > UINT64_MAX - out->rss_shmem_bytes ||
        out->rss_anon_bytes + out->rss_shmem_bytes >
            UINT64_MAX - out->page_table_bytes)
        return -1;
    out->current_rss_bytes = out->rss_anon_bytes + out->rss_file_bytes +
        out->rss_shmem_bytes;
    out->physical_footprint_bytes = out->rss_anon_bytes +
        out->rss_shmem_bytes + out->page_table_bytes;
    out->internal_bytes = out->rss_anon_bytes;
    out->external_bytes = out->rss_file_bytes;
    out->reusable_bytes = out->rss_shmem_bytes;
    out->detailed = 1;
    return 0;
}

#if !defined(SALT_GEMMA4_MEMORY_PARSER_ONLY)
static int g4_process_memory(G4ProcessMemory *out) {
    if (!out) return -1;
    memset(out, 0, sizeof *out);
#if defined(__APPLE__)
    struct rusage usage;
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    if (status != KERN_SUCCESS || getrusage(RUSAGE_SELF, &usage) != 0) return -1;
    out->current_rss_bytes = (uint64_t)info.resident_size;
    out->resident_peak_bytes = (uint64_t)info.resident_size_peak;
    out->physical_footprint_bytes = (uint64_t)info.phys_footprint;
    out->internal_bytes = (uint64_t)info.internal;
    out->external_bytes = (uint64_t)info.external;
    out->reusable_bytes = (uint64_t)info.reusable;
    out->compressed_bytes = (uint64_t)info.compressed;
    out->minor_faults = usage.ru_minflt < 0 ? 0 : (uint64_t)usage.ru_minflt;
    out->major_faults = usage.ru_majflt < 0 ? 0 : (uint64_t)usage.ru_majflt;
    out->input_blocks = usage.ru_inblock < 0 ? 0 : (uint64_t)usage.ru_inblock;
    out->output_blocks = usage.ru_oublock < 0 ? 0 : (uint64_t)usage.ru_oublock;
    out->detailed = 1;
    return 0;
#else
    struct rusage usage;
    FILE *status;
    int parse_rc, close_rc;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
    out->resident_peak_bytes = (uint64_t)usage.ru_maxrss * 1024u;
    out->minor_faults = usage.ru_minflt < 0 ? 0 : (uint64_t)usage.ru_minflt;
    out->major_faults = usage.ru_majflt < 0 ? 0 : (uint64_t)usage.ru_majflt;
    out->input_blocks = usage.ru_inblock < 0 ? 0 : (uint64_t)usage.ru_inblock;
    out->output_blocks = usage.ru_oublock < 0 ? 0 : (uint64_t)usage.ru_oublock;
    status = fopen("/proc/self/status", "r");
    if (!status) return -1;
    parse_rc = g4_parse_linux_status(status, out);
    close_rc = fclose(status);
    if (parse_rc != 0 || close_rc != 0)
        return -1;
    if (out->resident_peak_bytes < out->current_rss_bytes)
        out->resident_peak_bytes = out->current_rss_bytes;
    return 0;
#endif
}

static int g4_print_process_memory(FILE *stream, const char *phase,
                                   G4ProcessMemory *snapshot) {
    G4ProcessMemory local;
    G4ProcessMemory *out = snapshot ? snapshot : &local;
    if (!stream || !phase || g4_process_memory(out) != 0) return -1;
    fprintf(stream,
            "GEMMA4_MEMORY phase=%s current_rss_bytes=%" PRIu64
            " resident_peak_bytes=%" PRIu64
            " physical_footprint_bytes=%" PRIu64
            " internal_bytes=%" PRIu64
            " external_bytes=%" PRIu64
            " reusable_bytes=%" PRIu64
            " compressed_bytes=%" PRIu64
            " rss_anon_bytes=%" PRIu64 " rss_file_bytes=%" PRIu64
            " rss_shmem_bytes=%" PRIu64 " page_table_bytes=%" PRIu64
            " minor_faults=%" PRIu64 " major_faults=%" PRIu64
            " input_blocks=%" PRIu64 " output_blocks=%" PRIu64
            " detailed=%d\n",
            phase, out->current_rss_bytes, out->resident_peak_bytes,
            out->physical_footprint_bytes, out->internal_bytes,
            out->external_bytes, out->reusable_bytes,
            out->compressed_bytes, out->rss_anon_bytes, out->rss_file_bytes,
            out->rss_shmem_bytes, out->page_table_bytes,
            out->minor_faults, out->major_faults,
            out->input_blocks, out->output_blocks, out->detailed);
    return 0;
}
#endif

#endif /* SALT_TOOLS_GEMMA4_MEMORY_H */
