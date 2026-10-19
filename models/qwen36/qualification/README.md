# Qwen 3.6 A3B qualification artifacts

## Same-host 1 GB Waterfall comparator — 2026-08-17

This directory preserves the exact command and raw output for the one-run A3B
CTX200/O50 comparator cited by `BREAKING-POINT.md`. It is a phase-attribution
and memory-accounting record, not a serving qualification or a model-speed
comparison with Gemma.

### Identity and scope

- Candidate binary SHA-256: `98c725798793dea94d8ef176c1c93116c33d8d95351888dba5cdf51616d4aa57`
- Embedded build label: `b5cfa90`
- Prompt: 171 actual tokens under the CTX200 cap
- Output: all 50 requested tokens; process exit status `0`
- CPU-only, `SALT_CACHE_MODE=zerocopy`, 1 GB expert cache, zero pinned layers,
  two trunk-ring slots, three fetch workers, grouped-prefill threshold 128
- `SALT_GREEDY=1`, `SALT_ROOFLINE=1`, and `SALT_WATERFALL=1`

This run used greedy decoding. It does not qualify deterministic nonzero-
temperature sampling.

### Preserved files

| file | SHA-256 |
|---|---|
| `a3b-floor-ctx200o50-20260817.sh` (private reference; not exported) | `04656f34977a90dda1ff2173b9cde2aeb00505e4dc04e39265395595847e98df` |
| `a3b-floor-ctx200o50-20260817.stdout` (private reference; not exported) | `2e0d2cbd6ca0cf0eca0e8fe261344238882426b617ae8be50d4ac4d30bcf2379` |
| `a3b-floor-ctx200o50-20260817.stderr` (private reference; not exported) | `e89e69c9f446950786af0022b9dd7d6bd7aee14a2b86f712401b53c29d5b0699` |

The command file contains host-local artifact paths and requires `SALT_BIN` to
be supplied. The recorded hashes bind the evidence; rerunning elsewhere creates
a new qualification record rather than mutating this one.

### Waterfall

The 17.4-second engine wall spans 221 accounted tokens (171 prefill plus 50
generation), or 78.8 ms/accounted-token:

| stage | ms/accounted-token | share |
|---|---:|---:|
| attention | 9.9 | 12.5% |
| router | 0.6 | 0.8% |
| expert fetch | 4.5 | 5.7% |
| MoE compute | 14.3 | 18.1% |
| head | 2.1 | 2.6% |
| grouped prefill | 47.5 | 60.3% |

The prefill component was further attributed as 0.03 ms/token trunk bind,
14.12 linear-attention, 2.71 GQA, 2.03 routing, 6.50 expert fetch, 20.25 MoE
compute, 0.00 RFM glue, and 1.85 outer-prefill glue.

### Phase-exclusive roofline

| phase | wall/rate | logical misses | OS-read attribution | major faults | cache/fetch evidence |
|---|---|---:|---:|---:|---|
| prefill, 171 tokens | 10.131 s | 50.93 MiB/token | 47.82 MiB/token | 2,958.55/token | 0.0% hit; 7.379 ms/token fetch wait |
| first token | 10.383 s TTFT; 0.252 s decode | 514.69 MiB | 844.13 MiB | 17,580 | 4.7% hit; 31.829 ms fetch wait |
| post-first, 49 tokens | 7.034 s; 6.966 tok/s | 391.60 MiB/token | 177.77 MiB/token | 6,346.29/token | 27.5% hit; 8.442 ms/token fetch wait |

Logical misses are expert payload bytes. The OS-read field is a process-
attributed transport-inclusive counter, and fetch wait is advisory/barrier time;
none is relabeled as physical storage bandwidth.

### Memory and I/O

- Logical expert capacity: 565 slots / 999,751,680 bytes.
- Current/final resident-touched sample: 6.55 GB.
- Lifetime resident-touched peak: 6.724288512 GB.
- Final and peak Mach application footprint: 4.903322352 GB.
- Final internal/compressed/external: 0.42 / 4.47 / 6.13 GB.
- Major faults: 834,531; page reclaims: 691,584.
- Input/output block operations: 0 / 0.

These quantities have different ownership and time semantics and must not be
summed into a process-memory claim.
