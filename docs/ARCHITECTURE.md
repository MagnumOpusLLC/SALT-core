# Salt architecture

Salt is a **state-assured latent-transition engine** (SALT). This current authority
defines ownership, state, resource, compute, and publication invariants; history,
proposals, and measurements belong elsewhere.

## Structure

Roots:

1. stateful inference process;
2. bit-identical compute across platforms;
3. C99 implementation;
4. deterministic compute in every math process;
5. resource conservation through rigorous usage discipline;
6. memory offload through disk mapping;
7. controlled randomness.

Stateful + deterministic enables lossless KV export/import and inference
optimization through controlled amortization and batching. KV export supports
compute-state preservation, internalized prospective decode/prefill through the
Dynamic Process Reference (grand KV), and reusable compute state as software.
Inference optimization allows cheap inference and internalized asynchronous
prefill/decode. Together they let low-end cloud and edge machines run quality
intelligence, with compute and consumption on different machine tiers.

Current status and operating evidence live in:

- [`../BREAKING-POINT.md`](../BREAKING-POINT.md) — current runtime boundary and readiness;
- [`SOP.md`](SOP.md) — repository operation, qualification, and publication;
- [`TERMINOLOGY.md`](TERMINOLOGY.md) — canonical vocabulary (NFQ, X, cascade sources);
- [`../models/gemma4-26b-a4b/SOP.md`](../models/gemma4-26b-a4b/SOP.md) — Gemma model policy;
- [`workorder/wo0908resourcepool.md`](workorder/wo0908resourcepool.md) — closed CUDA/Metal TARGET resource collaboration and latest scoped evidence.
- [`workorder/wo0909serverFix.md`](workorder/wo0909serverFix.md) — completed generic-serving/model-owned-server and `G4KVC006` compression record.
- [`workorder/wo0910-full-context-hmm-serving.md`](workorder/wo0910-full-context-hmm-serving.md) — authenticated Acerbox full-context pageable-HMM wrapper, residency, reset/spill, and real-use candidate record.
- [`archive/workorders/2026-09-20/wo0920-wrap.md`](archive/workorders/2026-09-20/wo0920-wrap.md) — September 14–21 round ledger (projection cascade, B1 parity, amortization tolls); [`archive/backlog/decode-backlog-2026-09-21.md`](archive/backlog/decode-backlog-2026-09-21.md) holds the open items.
- [`formal/current-engine-696a06d4.md`](formal/current-engine-696a06d4.md) — current source/ledger correspondence and exact TLC/TLAPS boundary.

## 1. Authority hierarchy

The order is strict; a lower layer may optimize realization but cannot redefine
a higher layer.

| Layer | Authority |
|---|---|
| Portable C | Model arithmetic and operation order; token/state transitions; routing/reduction order; KV writes; tentative resolution, rollback, COMMIT, and publication. |
| CPU | Reference interpreter. Persistent workers may execute deterministic disjoint slices, but completion order cannot affect arithmetic, selection, state, or publication. |
| Metal/CUDA/HIP | Subordinate physical lowerings of admitted cells and canonical destinations; never a model, tensor index, KV ledger, position tracker, transition machine, or commit authority. |
| Model adapters/config | Model geometry, graph, tensor roles, tokenizer/template and quant compatibility, state layout, target operations, and checked-in platform policy. The model-neutral core owns reusable execution, resource, state, and DPR mechanisms. |
| Python/services | Generic HTTP/OpenAI admission, artifact authentication, session-affine process assignment, advisory work estimation, SSE presentation, explicit proof/export requests, and native supervision; never inference-method selection, candidate construction, model arithmetic, KV writes, or COMMIT. |

Correctness, deterministic state, bounded ownership, and fail-closed behavior
outrank throughput. A faster lowering that weakens an earlier invariant is
rejected.

## 2. One immutable semantic program

PREFILL, model-owned decode, and target verification are entry views over one
immutable portable program and one assignment/resource overlay:

```text
PREFILL(B)       known prompt rows -> live KV writes -> committed position
DECODE           core-owned sampler/candidate handoff -> exact transition
DECODE/TARGET    NFQ proposal/search -> causal X-TARGET rows -> resolve
TARGET QA        fixed/proposed proof frontier -> tentative rows -> resolve
```

The generic server submits a complete admitted request to the selected model
adapter. It does not choose DECODE versus TARGET, sample tokens, generate a
candidate trajectory, or reproduce the model loop. DPR/ND may supply an admitted
trajectory but does not own or enable N/F/Q. X-1 extra full-model B1 steps to
manufacture proposals before TARGET are forbidden. The low-level TARGET
verifier commits only target-proven state. A fixed-candidate TARGET benchmark
measures that transaction alone, not unconstrained conversation throughput.
At the [September 14 checkpoint](archive/workorders/2026-09-20/wo0914-nfq-core-repair.md), absent
candidates still enter ordinary native decoding; core ownership and no-DPR NFQ
connection repair remain open. This implementation gap is not authorization
to create a replacement inference algorithm.

The program owns action identity, dependencies, canonical destinations,
required resources, last consumers, visibility, and final publication. A
backend executable is valid only when it lowers the same admitted cells with
the same arithmetic and dependency order.

Within each engine, known PREFILL rows are written directly to its single live
session KV. TARGET uses the existing reserved canonical tentative rows;
candidate lookup must not mutate live KV. The core target and sampler resolve
candidates in fixed semantic order. COMMIT advances
metadata over the winning prefix rather than copying state from a second KV
arena. Rejected, losing, and unused runway rows are scrubbed or invalidated
before reuse.

Model startup compiles NFQ N/F/Q, the recipe X-TARGET maximum, live-KV warmup
threshold, warm X cap, worker admission, and executor policy.
`SALT_TARGET_ROUTE_N/F/Q` binds NFQ only; `SALT_TARGET_X` independently binds the
causal X-TARGET ceiling. Portable core derives the normal-serving effective X
from pre-request live KV; transport callers provide request content and controls
only. QA may provide fixed IDs,
optional DPR may provide a hit/proposal, and the existing core candidate source
is the integration authority on absence/miss; none changes model policy or
gives the server transition authority. A provider's existence or engagement
must be proven from live code and native receipts, not inferred from capacity.

Physical readiness and completion may change scheduling and wall time only.
Completion order, worker identity, resource-arrival order, or backend lane may
never choose the winner. Publication occurs once, after all required work,
backend status, state checks, and resource releases are complete.

Forbidden lowerings include hidden KV-copy kernels, peer position or state
trackers, alternate arithmetic, secondary schedulers or ledgers, implicit
publication, and operation-time capacity growth.

## 3. B/N/F/Q/X compute-efficiency contract

The canonical definitions are maintained in
[`TERMINOLOGY.md`](TERMINOLOGY.md). The shape symbols describe different
boundaries and must not be conflated:

| Symbol | Meaning |
|---|---|
| `B` | Row width admitted for a compute stage. PREFILL chunk width and backend stage widths are model/platform policy, not a universal batch size. |
| `N` | NFQ tile number: the base NFQ work tiles admitted inside each F lane. `N` is not causal TARGET row width, repeat count, or worker count. |
| `F` | Fanout lanes: concurrent isolated tentative latent universes, each with its own parent state, logits, ancestry, and canonical tentative seat. |
| `Q` | Search attention span / worker lifetime: the deterministic alternatives each admitted `(n,f)` tile may explore before resolution, retirement, or reassignment. |
| `W` | Persistent physical worker or backend-lane capacity. |
| `X` | Configured causal row ceiling of one X-TARGET attempt, independent of N/F/Q; normal serving may reduce it through the live-KV admission gate. |

For each F lane, the engine admits N base tiles. Every `(n,f)` tile has Q
ordered alternatives over its bounded worker lifetime. Candidate/check identity
is therefore `(n,f,q)`, and the configured candidate surface is `N x F x Q`.
`N30/F1/Q4` contains 120 candidate checks: 30 base tiles, each with four Q
alternatives. It must never collapse to `F x Q = 4` candidates.

NFQ N/F/Q describes only the proposal/search board. It does not size causal
TARGET execution. Normal serving derives the attempt width as:

```text
request_x = 1                                      when position_before < 512
request_x = min(configured_x, 4)                   when position_before >= 512
active_x  = min(request_x, output_remaining)
```

`position_before` excludes the current prompt's PREFILL rows. One X-TARGET
attempt stops at its first mismatch. Its accepted prefix may commit, but the
rejected suffix is not re-admitted as a continuation of that attempt. Tail
shortening and the request gate change neither configured X nor N/F/Q.

Worker admission is based on the base board, not on Q:

```text
N x F = n x W
```

for an integral configured multiplier `n`. This keeps W freelancers supplied
without redefining Q as worker count or creating dangling fixed owners.

Below the live-KV threshold, X1 prohibits multi-row TARGET. With DPR off or on a
proposal miss it realizes ordinary authoritative B1; authenticated parent-exact
reuse and one-row DPR verification remain valid without widening X. At or above
the threshold, absence of a retained projection or DPR proposal requires one
ordinary B1 bootstrap; its logits establish the starting candidate distribution
for the bounded X4 cascade. Once TARGET has computed lane states such as `SA`
and `SB`, their output logits/projections can replenish later alternatives
without another proposal-model rollout.

### Projection cascade: proposal sources and the X-TARGET episode

The production decode boundary (`src/text_verify.c`, `salt_text_generate_candidates`
and `schedule_normal`) is one episode per authoritative token boundary:

```text
root = greedy(parent logits)                       zero model work
proposal for positions 1..X-1, in precedence order:
  1. cache route    DPR trajectory when enabled and applicable, else the parent
                    history match: the longest, most recent earlier occurrence of
                    the committed suffix ending in the root supplies the tokens
                    that followed it (committed prompt/session tokens, then this
                    request's outputs; bounded 16-token match)
  2. local memory   retained rows of the previous X-TARGET block past its accepted
                    prefix (one argmax per already-computed speculative row);
                    present only after a rejection, never after a full hit
  3. true cold      root only: one ordinary transition, the same cost as B1
one batched causal X-TARGET forward over the proposed rows
validity vector V[0..X-1], first false j (X on full acceptance)
commit [0,j); retain the target logits at j-1 as the next root's parent; the
rejected suffix is invalidated as state and never replayed or shifted
```

Proposal generation performs no model steps: proposal work is microseconds per
episode. Proposals are token IDs only; they never become state. A sequential
`TK[i-1] -> TK[i]` chain through the model layers is B1 by another name and is
rejected as a proposer, as are same-parent vocabulary ranks used as a trajectory
and N-1 serial B1 steps. Only the target-computed accepted prefix commits.

Cost per accepted token is `(toll + per_row * X) / accepted`. Large X pays only
when the proposal source is strong (DPR trajectories, long exact history matches);
for weak sources the block toll dominates and root-only B1 is cheaper. X is a cap,
and admission/adaptive-X policy by proposal evidence is engine-level policy, not
backend behaviour.

### Where amortization lives (measured 2026-09-20/21, Mac)

- CPU: a TARGET row costs 0.65-0.8 of a B1 step even at full acceptance; routed
  experts are per-row traffic. MoE on CPU caps near 1.5x. Not the amortization
  substrate.
- GPU: the routed-expert work is cheap (7 ms/token in the Metal text program,
  0.5 ms GPU-busy per layer in mixed B1). The Metal text program's non-expert
  dispatches cost ~260 ms per submission independent of X, and the mixed-B1
  expert call pays ~0.83 ms/layer of command-buffer round-trip latency. Both are
  realization tolls around exact kernels, not arithmetic; see the backlog.
- Mixed B1 already runs experts on the GPU under `COMPUTE_NODE=full`; at B=1 the
  GPU's expert saving is returned as per-layer latency, so Metal B1 sits at CPU-B1
  level. The GPU pays when a whole layer chain, or many rows, stays on the device.

`F2` means two isolated tentative universes, not two duplicate B1 calls. A
backend may physically realize two ready lane transitions as a B=2 matrix batch,
but B=2 is only a lowering: the two rows must carry distinct admitted
state/token identities and completion order cannot select the winner.

For a GPU target frontier, NFQ shapes the configured N-by-F base board and each
tile's Q lifetime, while the TARGET lowering independently admits up to
`active_x` causal rows. Each projection may scan shared weights once across
admitted rows, but canonical matrix outputs,
activation boundaries, routing, and reduction order remain unchanged. More
lanes or alternatives are not automatically more efficient: losing tentative
work counts in the total wall.

CPU occupancy follows the same base admission relation `N x F = n x W`, while
its physical realization may use independent output/expert-row work inside each
tile. Q extends each tile's bounded alternative lifetime; it does not replace N
in worker admission.

`Q` bounds the alternatives available to a lane during one search-attention
lifetime. Workers remain freelancers: the orchestrator may give any free worker
any dependency-ready `(n,f,q)` claim, and the worker returns to the shared pool
after that bounded claim. A blocked lane cannot impose a global barrier on
unrelated ready work. Unused alternatives are retired at target resolution and
never cross the next authoritative token boundary.

The canonical tentative output logits are the proposal seam. Before another
submit can overwrite them, the engine records only bounded token IDs plus
generation and ancestry provenance in existing NFQ storage. State, logits, and
KV remain at their preassigned canonical addresses. Only target-authorized
ancestry commits; losing lanes and their derived Q alternatives cancel or become
stale without publication.

Efficiency decisions use complete wall time, not isolated kernel speed or
logical submission counts. An optimization is admissible only when token IDs,
pending token, arithmetic outputs, complete KV, and committed state remain
exact at the same compared candidate/commit boundary. Different N values admit
different NFQ candidate surfaces, while different X values admit different
causal attempt lengths; compare their common authoritative boundary rather than
conflating either dimension.

## 4. Canonical weights and address authority

Formal engine models keep model identity symbolic. `NumLayers`,
`ExpertsPerLayer`, `TopK`, exact expert and slot bytes, resource-slot count and
budget, hidden/scalar geometry, context, N/F/Q, X-TARGET row capacity, and worker
budget are constrained through named `ModelBounds`. Model/platform recipes bind
those conditions. Finite `.cfg` values exist only to bound TLC exploration and
never become universal runtime constants.

Authenticated canonical compressed files are the sole learned-weight payloads.
Runtime repacks, copied expert pools, backend-private weight packages, and
secondary tensor indexes are forbidden.

For Qwen, the complete weight payload is the four disjoint files
`trunk.bin`, `pool.bin`, `embed.bin`, and `head.bin`, together with their exact
layouts, offsets, tokenizer, and model/quant identity. No one file is the model.
Other model adapters define their own authenticated canonical file set under the
same no-copy and single-index rules. Persistent session state is a separate
compatibility-bound artifact, never a weight payload.

CPU-parsed tensor records are the sole address authority. Each record owns
logical identity, encoding, shape, canonical source range, and canonical output
destination. Metal/CUDA/HIP resources are derived bounded views or staging
bindings for those records. Mapping extent, device addressability, READY state,
and physical page residency do not create ownership or a peer lookup head.

## 5. Bounded resources and leases

Capacity is admitted before operation:

1. **Process-lifetime capacity** — fixed arenas, descriptors, slots, worker pools,
   backend contexts, source identities, and bounded queues are created at startup
   or model initialization and released at final close.
2. **Demand-populated readiness** — canonical tensor views and expert/cache slots
   may become READY, resident, retained, or evicted within their admitted bounds.
   READY or mapped does not imply physically resident or `mlock`ed.
3. **Operation leases** — each active view, cache entry, expert, or state seat is
   leased from acquisition through its exact last CPU consumer or backend fence.
   Leases release on success, failure, cancellation, and teardown; a leased entry
   cannot be evicted or rebound.

Operations may populate admitted seats but may not allocate new capacity, grow
arenas, create alternate scratch ledgers, or copy learned weights. Resource
refusal may fall back only before mutation and only where policy explicitly
admits that fallback. Full-backend intent never silently becomes CPU work.

## 6. State, KV, export, and proof

There is one mutable live session state and one authoritative KV sequence per
engine seat. A server process pool may own several isolated seats under the same
compatibility track. It never merges their state; it may move only an idle,
committed session by explicit authenticated `G4KVC006` export, seat reset, and
import before new work is admitted. Pinned
`G4KVC005`/`G4KVC004` artifacts remain legacy read inputs.
Models may expose these state classes without merging their boundaries:

- **mandatory/shared prefix** — immutable protected rows and identity;
- **factual state** — replaceable private rows rooted in the exact shared-prefix
  bytes;
- **session continuation** — private committed rows after the factual boundary;
- **tentative target state** — unpublished candidate rows valid only for the
  active target transaction.

Import validates model/quant/package identity, state class, geometry, position,
and required parent bytes before mutation. Clearing or replacing a private
class cannot alter an immutable prefix. Cross-track state loading fails closed.

Export is an explicit portable-C materialization transaction, requested for a
named state class or final session close. It does not occur implicitly on every
operation, and it does not transfer state ownership to Python or a backend.
Exported bytes are observational artifacts of committed state.

Strict proof is also explicit and observational. It may compare response,
tokens, pending token, complete KV/state, digests, resources, and publication
counts, but it cannot add target work, select a route, change arithmetic,
repair state, trigger COMMIT, or become a readiness claim. Formal proof,
component proof, benchmark evidence, and real inference are separate evidence
classes.

## 7. DPR transition authority

DPR is a portable-C transition mechanism over the ordinary model session; it is
not prompt content, a Python controller, or a second KV/state arena. Every cycle
starts with exact parent lookup: a hit appends authenticated rows through the
model target authority; a miss composes independently eligible ND and NM work.
If no admitted ND horizon remains, the existing core candidate/decode handoff
remains responsible; no target-model proposal rollout may be introduced. The
checkpoint's no-DPR connection is still open. DPR never turns the server into a
sampler or loop.

| Surface | Authority |
|---|---|
| Parent reference | Immutable authenticated row source. It may supply an exact transition but is never live mutable session KV. |
| `ND` transition Nomogram | Proposes candidate token trajectories; it owns no model arithmetic, KV, acceptance, or COMMIT. |
| `NM` attention Nomogram | Proposes where canonical target compute may matter; it may prepare resources but cannot remove required arithmetic or select a winner. |
| Model target adapter | Supplies geometry, exact operation/tensor bindings and model stop-token data to core-owned sampling, verification, rollback and state transitions. |
| Generic DPR core | Owns formats, admission, parent-first policy, deterministic selection, ND/NM composition, retention/accounting, and proof mechanics. |

`ND` and `NM` are orthogonal. Native, `ND_ONLY`, `NM_ONLY`, and `ND_NM` each
retain independent admission, observations, refusal, and model qualification.
The target/router remains the only arithmetic and commit authority in every
cell.

A scoped [terminal tool-cache experiment](mentor/DPR-TERMINAL-TOOL-CACHE-EXPERIMENT.md)
demonstrates both parent-prefill and parent-exact decode reuse without changing
the fixed tool-call test or executing a command. Its exact trained trajectory
restored 320/322 prompt rows and 32 output rows plus a bonus stop, reducing Mac
CPU request wall from 22.242 s with DPR OFF to 1.665 s with DPR ON (13.356x)
with complete exported-state diff zero. The experiment trained 1/8 curriculum
entries and 0/4 fixed test prompts; it is cache evidence, not broad terminal
generalization or a readiness/default change. Current Gemma serving still keeps
raw `shared_kv` attachment and DPR mutually exclusive.

## 8. Memory and failure boundary

The hard model memory limit is one all-inclusive current-RSS ceiling:

- macOS uses `task_vm_info.resident_size`;
- Linux uses `/proc/self/status` `VmRSS`;
- resident file-backed pages count toward the same ceiling;
- footprint breakdowns, compression, mappings, faults, and I/O are attribution,
  not alternate pass/fail measures.

Limit breach or measurement failure fails closed after normal unwind. Invalid
identity, geometry, offset, state, capacity, mapping, registration, launch,
synchronization, engagement, lease, backend status, or teardown also fails
closed. No response or state publication is allowed after a fatal or partially
mutating failure.

## 9. Formal and transmutation correspondence

At source `696a06d4` and warehouse `salt-dev-696a06d4`, six current portable
engine models pass TLC. DPR target semantics and expert-resource lifecycle have
complete current TLAPS proofs totaling 1,020 obligations. Prefill/lookahead,
orthogonal ND/NM, N/F/Q, and target-block executor proofs remain open; the
historical `ea564711` 1,087-obligation package is not current correspondence.

The mechanical audit resolves 19/20 contradiction candidates as detector/shaper
errors. The only real item is a deployment benchmark docstring that omits its
nullable return. Rosarium proves 8,295/8,295 claim identity continuity but no R3
information-flow result because that input contained no `infoflow` channel.

The portable candidate-zero target path correctly returns a pending correction
without target submission. The production server source is still transitional
under `tools/` and contains historical inference-method selection. Work order
0909 treats that ownership/integration mismatch as RED until the model-owned
adapter and real CTX2048 generic-server gate pass.

## 10. Source map

| Area | Primary source |
|---|---|
| Model-neutral text order and publication | `src/text_exec.c`, `include/salt/text_exec.h` |
| Immutable target/token programs and CPU reference | `src/text_verify.c`, `src/text_verify_cpu.c`, `src/text_token.c`, `include/salt/text_verify.h`, `include/salt/text_token.h` |
| Reusable arithmetic and bounded resources | `src/kernels.c`, `src/cache.c`, `src/moe_group.c`, `src/gpu_resource.c`, `src/gpu_pool.c` |
| Physical backends | `src/gpu_metal.m`, `src/gpu_cuda.cu`, `src/gpu_hip.cpp`, `src/gpu_stub.c` |
| Qwen model adapter | `models/qwen36/` |
| Gemma model adapter, state, and production native protocol integration | `models/gemma4-26b-a4b/`; `tools/gemma4-qa.c` remains QA only |
| DPR policy, store, and accounting | `src/dpr.c`, `src/dpr_store.c`, `src/dpr_stats.c`, `include/salt/dpr*.h` |
| Portable state and RSS authority | `src/state.c`, `include/salt/state.h`, `src/mem.c` |
| Generic service admission, HTTP/SSE, session-affine scheduling, and supervision | `server/serve.py`, `server/gemma4_backend.py`, `server/gemma4_scheduler.py` |

Readiness, admitted operating modes, active breakpoints, and scoped performance
belong in the linked current-status and SOP documents, not in this architecture
authority.
