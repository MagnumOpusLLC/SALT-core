# Salt documentation

This index covers the curated public runtime, public architecture and risk
statements, scoped formal material, and explicitly exported historical evidence.
Private SOPs, work orders, raw qualification logs, and deployment records are not
part of SALT-core.

## Current runtime export

The curated runtime was refreshed from merged private `main`
`90b9c82b11eb18116f6dc4f18146c0e36100f58b`
(**2026-10-17T08:23:20-04:00**). The runtime pin in
[`formal/proof-manifest.json`](formal/proof-manifest.json) is separate from its
unchanged historical proof inputs and results. Older revision stamps and
experiment reports are historical, not qualification of this refreshed tree.

This refresh adds the model-owned LFM2.5-24B-A2B package/runtime together with
its required portable engine and backend seams. LFM bounded CUDA resources bind
`(address, mapping generation)` so a recycled file-backed virtual address cannot
retain stale device identity. The private source pin passed ten fresh
P4096/O100 LFM processes with identical tokens, scalar state, KV,
recurrent/model state, and logits under a 4-GB process RSS limit. That is
source-pin evidence, not an independent rerun from this curated checkout; LFM
remains an explicit model build target rather than a new default `make all`
product.

The merged private tree's standalone Qwen3.8 candidate is not exported: its
frozen-core guard rejects the newer engine headers and remains a private
qualification surface. Gemma, Qwen3.6, and the experimental Maple adapter retain
their existing public scope. A checked-in recipe or backend is not evidence for
arbitrary contexts, prompts, or hardware.

## Public documents

1. [`../README.md`](../README.md) — project entry point, first-use path, and
   current release boundary.
2. [`../models/README.md`](../models/README.md) — exported model package and
   model-documentation index.
3. [`../PERFORMANCE.md`](../PERFORMANCE.md) — source-pinned performance ledger;
   private receipt paths are labeled rather than exported.
4. [`COGNITIVE-STATE-RISK.md`](COGNITIVE-STATE-RISK.md) — KV Cache and
   Persistent Agent State Statement: beyond-prompt state, consent, risk, and the
   separation of computational continuity from authority to act.
5. [`ARCHITECTURE.md`](ARCHITECTURE.md) — authoritative storage, execution,
   resource, and state ownership.
6. [`TERMINOLOGY.md`](TERMINOLOGY.md) — canonical NFQ, X-TARGET, DPR, and engine
   vocabulary.
7. [`FORMAL-PROOFS.md`](FORMAL-PROOFS.md) — curated TLA+/TLAPS sources,
   reproduced proof results, and explicit limits on runtime correspondence.
8. [`TRANSMUTATION.md`](TRANSMUTATION.md) — curated transmutation findings and
   the scoped loader-proof boundary.
9. [`EC2-START.md`](EC2-START.md) — status/start notes for the specifically named
   historical EC2 hosts; it is not a provisioning or serving guide.
10. [`../experiments/reports/README.md`](../experiments/reports/README.md) —
   explicitly exported, source-indexed historical reports. Reports do not
   override current source or runtime qualification.

## Runtime and validation boundary

The production HTTP entry is
[`../server/serve-launch.sh`](../server/serve-launch.sh). Model recipes and
package contracts live under [`../models/`](../models/). `make test` runs the
curated public smoke suite; explicit model builds remain separate targets.
Passing source/build/smoke checks does not substitute for authenticated ordinary
inference and complete KV/state comparison on a real model package.

On Darwin, `make test` compiles `src/gpu_metal.m` and executes the mapped routed-
MoE and mapped-trunk lifecycle gates against the real Metal device. The backend
compiles its embedded MSL through the Metal framework at runtime; the optional
`salt-gpu.metallib` target is not a prerequisite for Gemma QA or public smoke
validation. These are component/resource gates, not a new real-model performance
claim.

The formal bundle preserves its own source hashes and historical proof results.
Advancing the exported runtime pin does not turn those historical proofs into a
current native-source refinement or a concrete KV bit-identity certificate.
SALT-core contains no model weights; model artifacts retain their own licenses
and must be obtained separately.
