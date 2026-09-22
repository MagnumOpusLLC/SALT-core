/* mem.c -- the plan is a forecast, not a result (invariant 5). */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include "salt/salt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <libproc.h>
#include <unistd.h>
#endif

int64_t salt_mem_available(void) {
    const char *test = getenv("SALT_TEST_MEM");
    if (test) return (int64_t)atoll(test);

#if defined(__APPLE__)
    uint64_t m = 0;
    size_t n = sizeof m;
    if (sysctlbyname("hw.memsize", &m, &n, NULL, 0) == 0)
        return (int64_t)m;                   /* physical; macOS has no MemAvailable */
    return (int64_t)8 << 30;
#else
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return (int64_t)8 << 30;
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    fclose(f);
    return kb ? (int64_t)kb * 1024 : (int64_t)8 << 30;
#endif
}

void salt_mem_plan(SaltMemPlan *p, const SaltCfg *cfg, int64_t trunk_budget,
                   int64_t cache_bytes, int64_t shared_bytes, int npin,
                   int64_t slot, int nring) {
    (void)npin;   /* real pinned bytes are patched in by the caller */
    memset(p, 0, sizeof *p);
    p->trunk_pin_b = (double)(trunk_budget > 0 ? trunk_budget : 0);
    p->trunk_ring_b = (double)slot * (double)nring;
    p->cache_b     = (double)cache_bytes;
    p->shared_b    = (double)shared_bytes;
    p->state_b     = (double)cfg->hidden * 8.0 + 1e6;   /* activations + scratch */
    p->index_b     = 1e6;                                /* pointer map, small */
    p->need_b = p->trunk_pin_b + p->trunk_ring_b + p->cache_b +
                p->shared_b + p->state_b + p->index_b;
    p->have_b = (double)salt_mem_available();
}

int salt_mem_refuses(const SaltMemPlan *p) {
    return p->need_b > p->have_b * 0.95;
}

void salt_mem_print(const SaltMemPlan *p) {
    fprintf(stderr,
            "MEMORY PLAN (forecast, not result)\n"
            "  trunk pinned prefix   %9.1f GB\n"
            "  trunk ring            %9.1f GB\n"
            "  expert cache          %9.1f GB\n"
            "  shared experts (res)  %9.1f GB\n"
            "  state + scratch       %9.1f GB\n"
            "  pointer map           %9.1f GB\n"
            "  TOTAL need            %9.1f GB  vs  available %9.1f GB\n",
            p->trunk_pin_b / 1e9, p->trunk_ring_b / 1e9, p->cache_b / 1e9,
            p->shared_b / 1e9, p->state_b / 1e9, p->index_b / 1e9,
            p->need_b / 1e9, p->have_b / 1e9);
}

/* Parse Linux /proc/<pid>/status from an already-open stream. Kept portable
 * so the fail-closed field contract can be fixture-tested on every platform. */
int salt_mem_parse_linux_status(FILE *f, SaltMemSnapshot *snapshot) {
    if (!f || !snapshot) return -1;
    char line[256];
    long long kb = 0;
    int got_anon = 0, got_shmem = 0, got_pte = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmRSS: %lld kB", &kb) == 1)
            snapshot->resident_b = (int64_t)kb * 1024;
        else if (sscanf(line, "VmHWM: %lld kB", &kb) == 1)
            snapshot->resident_os_peak_b = (int64_t)kb * 1024;
        else if (sscanf(line, "RssAnon: %lld kB", &kb) == 1) {
            snapshot->anonymous_b = (int64_t)kb * 1024;
            got_anon = 1;
        } else if (sscanf(line, "RssFile: %lld kB", &kb) == 1)
            snapshot->file_backed_b = (int64_t)kb * 1024;
        else if (sscanf(line, "RssShmem: %lld kB", &kb) == 1) {
            snapshot->shared_b = (int64_t)kb * 1024;
            got_shmem = 1;
        } else if (sscanf(line, "VmPTE: %lld kB", &kb) == 1) {
            snapshot->page_table_b = (int64_t)kb * 1024;
            got_pte = 1;
        } else if (sscanf(line, "VmSwap: %lld kB", &kb) == 1)
            snapshot->swap_b = (int64_t)kb * 1024;
    }
    if (ferror(f) || !got_anon || !got_shmem || !got_pte) return -1;
    if (snapshot->resident_b <= 0)
        snapshot->resident_b = snapshot->anonymous_b +
            snapshot->file_backed_b + snapshot->shared_b;
    if (snapshot->resident_os_peak_b <= 0)
        snapshot->resident_os_peak_b = snapshot->resident_b;
    snapshot->application_b = snapshot->anonymous_b + snapshot->shared_b +
        snapshot->page_table_b;
    snapshot->application_is_approximate = 1;
    return snapshot->application_b > 0 ? 0 : -1;
}

int salt_mem_snapshot(SaltMemSnapshot *snapshot) {
    if (!snapshot) return -1;
    memset(snapshot, 0, sizeof *snapshot);
#if defined(__APPLE__)
    struct task_vm_info info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        snapshot->application_b = (int64_t)info.phys_footprint;
        snapshot->application_os_peak_b =
            info.ledger_phys_footprint_peak > 0
                ? info.ledger_phys_footprint_peak : 0;
        snapshot->resident_b = (int64_t)info.resident_size;
        snapshot->resident_os_peak_b = (int64_t)info.resident_size_peak;
        snapshot->anonymous_b = (int64_t)info.internal;
        snapshot->compressed_b = (int64_t)info.compressed;
        snapshot->file_backed_b = (int64_t)info.external;
        return 0;
    }
    struct rusage_info_v4 ri;
    memset(&ri, 0, sizeof ri);
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                        (rusage_info_t *)&ri) == 0) {
        snapshot->application_b = (int64_t)ri.ri_phys_footprint;
        snapshot->application_os_peak_b =
            (int64_t)ri.ri_lifetime_max_phys_footprint;
        snapshot->resident_b = (int64_t)ri.ri_resident_size;
        snapshot->resident_os_peak_b = snapshot->resident_b;
        return snapshot->application_b > 0 ? 0 : -1;
    }
    return -1;
#else
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    int rc = salt_mem_parse_linux_status(f, snapshot);
    fclose(f);
    return rc;
#endif
}

int salt_io_snapshot(SaltIoSnapshot *snapshot) {
    if (!snapshot) return -1;
    memset(snapshot, 0, sizeof *snapshot);

    struct rusage usage;
    memset(&usage, 0, sizeof usage);
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_majflt >= 0) {
        snapshot->major_faults = (uint64_t)usage.ru_majflt;
        snapshot->major_faults_valid = 1;
    }

#if defined(__APPLE__)
    struct rusage_info_v2 info;
    memset(&info, 0, sizeof info);
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
                        (rusage_info_t *)&info) == 0) {
        snapshot->process_read_bytes = info.ri_diskio_bytesread;
        snapshot->process_read_bytes_valid = 1;
    }
#else
    FILE *f = fopen("/proc/self/io", "r");
    if (f) {
        char line[256];
        unsigned long long bytes = 0;
        int found = 0;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "read_bytes: %llu", &bytes) == 1) {
                found = 1;
                break;
            }
        }
        if (found && !ferror(f)) {
            snapshot->process_read_bytes = (uint64_t)bytes;
            snapshot->process_read_bytes_valid = 1;
        }
        fclose(f);
    }
#endif
    return snapshot->process_read_bytes_valid || snapshot->major_faults_valid
        ? 0 : -1;
}

uint64_t salt_disk_read_bytes(void) {
    SaltIoSnapshot snapshot;
    return salt_io_snapshot(&snapshot) == 0 &&
            snapshot.process_read_bytes_valid
        ? snapshot.process_read_bytes : 0;
}

uint64_t salt_checksum(const uint8_t *p, int64_t n) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int64_t i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(0x100000001B3);
    return h;
}
