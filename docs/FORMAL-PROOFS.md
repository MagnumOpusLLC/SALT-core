# Formal proof sources and scope

This repository includes **nine fully discharged abstract TLA+/TLAPS modules**.
All copied modules were independently checked inside Docker: **9/9 TLC
configurations and 9/9 TLAPS proof modules passed**, using clean proof
fingerprints. The target-block module was checked after the initial eight-module
bundle; hashes verify that those eight modules remained unchanged.

This is **not a complete formal verification of SALT's implementation**. A proved
abstract model, a source-correspondence argument, and concrete cross-platform
KV/state bit identity are separate evidence classes.

Runtime source baseline: `d3a2037f404d520998ac2d3b2d182ca58f529c82`
(**2026-09-21 19:50:34 −04:00**). The proof sources include subsequent proof-only
updates; their exact hashes are recorded in the manifest rather than implied to
have existed in that runtime commit.

## Included modules

| Directory under `docs/formal/` | Model / proof | TLAPS obligations | Scope |
|---|---|---:|---|
| `core-runtime/` | `SaltPrefillControl` / `SaltPrefillControlProofs` | 19 | Prefill ordering, release accounting, lookahead and publication |
| `core-runtime/` | `SaltDecodeControl` / `SaltDecodeControlProofs` | 16 | Normal decode, pre-request KV warmup, independent X, pending consumption, closure and publication |
| `kv-expert-runtime/` | `KvExpertResourceRuntime` / `KvExpertResourceRuntimeProofs` | 69 | Abstract expert readiness, lease and resource lifecycle |
| `dpr-engine/` | `DprTargetRuntime` / `DprTargetRuntimeProofs` | 881 | Abstract exact reuse, verification, accepted-prefix and pending-token authority |
| `dpr-target-block-executor/` | `DprTargetBlockExecutor` / `DprTargetBlockExecutorProofs` | 67 | Selected target-block semantic invariants; concrete action projections and scalar induction |
| `dpr-orthogonal-nomograms/` | `DprOrthogonalNomograms` / `DprOrthogonalNomogramsProofs` | 57 | ND/NM independence under the admitted no-forced-coupling configuration |
| `dpr-nfq-token-epoch/` | `DprNfqTokenEpoch` / `DprNfqTokenEpochProofs` | 20 | **Retained historical abstraction**, not current independent-X serving correspondence |
| `gemma4-runtime/` | `GemmaRuntime` / `GemmaRuntimeProofs` | 186 | Supplemental abstract Gemma state transitions |
| `qwen-runtime/` | `QwenRuntime` / `QwenRuntimeProofs` | 191 | Supplemental abstract Qwen state/history transitions |

Each model has its finite `.cfg` witness and its `Proofs.tla` module. The bundle
also retains the intentional `DprOrthogonalNomogramsRed.cfg` negative control;
that configuration is expected to violate an invariant, not pass.

- [Exact proof-source manifest](formal/proof-manifest.json)
- [Verification results for the copied bundle](formal/verification-results.json)

No private worklogs, raw prover logs, host inventories, credentials or in-house
SOPs are part of this proof export.

## Important limits

1. **The target-block theorem proves its selected semantic invariant, not every
   property listed in its TLC configuration.** The TLAPS theorem covers immutable
   program/executor policy, no intermediate publication, pending-token handling,
   abstract runtime/canonical equality, resource-object lifetime, submission/
   completion accounting, zero-accepted executor avoidance, and the abstract DPR
   boundary. Full per-row KV, destination-cardinality and residency properties
   remain TLC-checked. All previous open obligations in the selected theorem
   are closed; counts from different proof decompositions are not comparable
   quality scores. The bundle is not a composed-system proof.
2. `SaltDecodeControl` is a hand-written normal-route control abstraction. It
   assumes an exact accepted-prefix result from the target verifier. It does not
   prove that verifier's tensor arithmetic, DPR, explicit import/clear, or every
   server recovery path.
3. The retained NFQ epoch model uses `TargetRows = N * F`, accepted width N and
   no ordinary-B1 action. Its proof must **not** be presented as certification
   of the current independent-X warmup/serving policy.
4. Gemma and Qwen proofs are supplemental model-path evidence. They do not
   establish that every current model adapter, kernel or platform recipe refines
   these models.
5. Abstract digest equality is not a cryptographic or concrete-byte proof.
   Tensor arithmetic, complete KV bytes, deterministic continuation, backend
   engagement, performance and resource measurements require separate runtime
   qualification.
6. A proof about program state establishes neither truth of retained information
   nor permission to preserve or execute human-derived state. The
   [KV Cache and Persistent Agent State Statement](COGNITIVE-STATE-RISK.md)
   remains applicable.

## Reproduce with an existing Docker toolchain

The verification used TLC `2026.05.26.235334`, TLAPM reporting `1eabe97-dirty`,
and bundled Z3 `4.13.4`. The verification record pins the prover and JAR hashes;
the version string alone is not an exact tool identity.

Use an existing container with Java, `tla2tools.jar`, TLAPS, its standard library
(including `TLAPS` and `SequenceTheorems`) and its solver backends. No SALT model
weights or native inference binary are required.

Copy `docs/formal/` into a run-owned directory in that container. From the
appropriate model directory, run:

```sh
java -Xmx2g -XX:+UseParallelGC -cp /path/to/tla2tools.jar tlc2.TLC \
  -workers 1 -metadir /run-owned/scratch/Model \
  -config Model.cfg Model.tla

tlapm --cleanfp --threads 2 ModelProofs.tla
```

Replace `Model` with each model name in the table. Require TLC's explicit
no-error completion and TLAPS's `All ... obligations proved` result as well as
exit status zero. A script that continues after a failed module is not evidence
that every module passed.

The separate ND/NM negative control is:

```sh
java -cp /path/to/tla2tools.jar tlc2.TLC -workers 1 \
  -metadir /run-owned/scratch/negative \
  -config DprOrthogonalNomogramsRed.cfg DprOrthogonalNomograms.tla
```

It must report the expected invariant violation. Preserve results, then remove
only run-owned TLC scratch and TLAPS caches after all provers have exited.

The formal bundle does not promote the repository to release-ready status.
