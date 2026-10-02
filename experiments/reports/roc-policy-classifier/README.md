# ROCm proof: preserved policy state for a four-code case classifier

**Scoped result:** A fictional policy was incorporated into one authenticated, curatable **648-row parent state**. Eight independent messy tickets were then classified by returning only one code. The **imported parent scored 8/8**, a fresh request containing the same full policy as a system message scored **8/8**, and a question-only/no-policy control scored **0/8**. All **24** rendered outputs were exactly one uppercase letter and stopped naturally; native usage counted two completion tokens per request. This establishes policy-dependent selection for **this frozen synthetic fixture on V520 ROCm**, not general policy compliance.

## Policy, state, and test boundary

- The [fictional Aster routing policy](fixtures/policy.md) defines arbitrary fixed codes: **B** verified current sealed-payload mismatch, **D** verified current revoked grant, **A** verified current publication attribution conflict, **C** none of these. An event counts only with `issuer:"CaseLedger"`, `attested:true`, and `current:true`; precedence is **B > D > A > C**. The [eight held-out tickets](fixtures/heldout.json) have conflicting customer prose, unverified claims, superseded findings, and competing current events, with **two cases per correct code**. Expected labels/rationales were never submitted in a request.
- The policy was delivered as a real `system` message, followed by **two ordinary, naturally completed conversation turns** asking the model to describe evidence authority and precedence. The accepted [v2 training contract](experiment-v2.json) yielded **648 committed rows**, under the 8,000-row cap, with complete-state SHA `270879424651fe74c3f54b87a6719be95cdb648c1ca5589f51d748ad3f0048b8` and an authenticated **278,692,096-byte (0.279 decimal-GB)** `G4KVC006` artifact. The held-out case IDs and answers were absent from those training turns. An earlier [v1 contract](experiment.json) is historical **RED**: its first training reply hit the 512-token output cap mid-sentence; no v1 parent was exported or tested. The v2 policy text/cases were unchanged; only the natural training questions were narrowed, with a new parent ID.
- ROCm only: existing V520 HIP `rocm` recipe, binary SHA `3db881c2b312e9bd820b489415e98e14fe44ca8451e288e44001b746edb9d4e4`, compatibility `35eeec434dfe1e68d61ffc2a6db88f4738a7ba20ee55967031bded29da967a8a`, CTX12288, W8, 1.0 decimal-GB KV allowance, 3 decimal-GB expert budget, 7.5 decimal-GB all-inclusive RSS ceiling. NFQ N104/F1/Q4 and X16 are configured recipe limits, **not measured engagement**. Greedy T=0, top-K1, **no seed**, DPR off, proof off, O16. No kernels, model logic, TensorOps, memory pool, or recipe were edited.
- Each of the 24 cells used its **own fresh managed native process/session**, so a prior ticket could not teach a later one. **No-parent:** structured ticket and generic “return one A–D code” instruction, starting at position 0. **Parent:** import the same v2 authenticated state at position 648, then submit that identical ticket/instruction with **no policy or code definitions in the question**. **Fresh-policy:** start at zero and send the full exact policy as `system` text with the same ticket/instruction. Every request went through the normal HTTP server manager. Per-case final KV was **not exported**; the parent materialization and each live loaded/saved position were verified. The retained parent artifact was read back after all cases and the listener was absent.

## Accuracy and compliance

| Ticket | Frozen correct code | No policy | Preserved parent | Fresh policy |
|---|---|---|---|---|
| TK-901 | B | C ✗ | B ✓ | B ✓ |
| TK-916 | B | C ✗ | B ✓ | B ✓ |
| TK-927 | D | B ✗ | D ✓ | D ✓ |
| TK-934 | D | A ✗ | D ✓ | D ✓ |
| TK-945 | A | B ✗ | A ✓ | A ✓ |
| TK-952 | A | C ✗ | A ✓ | A ✓ |
| TK-968 | C | D ✗ | C ✓ | C ✓ |
| TK-979 | C | A ✗ | C ✓ | C ✓ |
| **Total** | **8** | **0/8** | **8/8** | **8/8** |

The correct code requires combining the ticket's structured evidence authority, `current` flag, and policy precedence; persuasive prose alone must not override the record. The no-policy arm emitted valid-looking letters but selected **no correct code**. A code-only output cannot expose the model's reasoning; “compliance” here means matching the frozen source-derived classification labels and the strict output contract. None of these codes authorizes an action.

## Speed and resource accounting

Totals below are **eight ordinary one-ticket HTTP requests** per arm. Startup, import, parent construction, and explicit parent export are separate costs. The no-policy arm is not an equal-quality speed comparator because it scored 0/8.

| Arm | Correct | New input / output | PREFILL total; weighted ms/input | Decode total; weighted ms/output | HTTP total; mean/request | Import total | Peak OS RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| No policy | 0/8 | 1,322 / 16 | 80.968 s; 61.246 | 3.039 s; 189.931 | 86.464 s; 10.808 s | 0 | 5,996,261,376 bytes (5.996 GB) |
| Preserved parent | 8/8 | 1,322 / 16 | 100.355 s; 75.911 | 3.750 s; 234.405 | **107.369 s; 13.421 s** | 56.489 s | 5,996,257,280 bytes (5.996 GB) |
| Fresh policy | 8/8 | 4,938 / 16 | 279.991 s; 56.701 | 3.630 s; 226.881 | **286.762 s; 35.845 s** | 0 | 5,996,163,072 bytes (5.996 GB) |

The fresh-policy arm computed **3,616 more new input tokens**. Against the equally accurate fresh-policy arm, preserved-parent **request wall saved 179.392 s across eight cases (2.671×)**. Counting the **56.489 s** of eight independent imports, it saved **122.904 s (1.750×)**. Long parent context did raise per-new-token PREFILL and per-output decode rates, but with only two output tokens per case, avoiding repeated full-policy PREFILL dominated this ROCm workshape. This is an observed workshape result, **not** faster native kernels or an exact cache hit.

The one-time v2 parent creation took **93.864 s HTTP for two training turns** plus **23.483 s explicit export**, or **117.347 s** before imports. In this single observed set, subtracting those costs from the eight-case import-inclusive savings leaves **5.557 s**, **excluding model startup**. Counting the measured per-case native startups makes the preserved path **0.489 s slower**, even before charging the initial parent-construction startup. This is not a repeat-qualified commercial break-even point. A persistent multi-session service that amortizes imports/startup differently was not tested.

## Evidence and limits

[results.json](results.json) records every case's selected code, expected label, output-ID digest, exact new/loaded/saved token counts, PREFILL/decode ms per token, request/import/startup wall, RSS, and source/build identity. Ignored local action receipts live under `logs/experiments/roc-policy-v2/`; the failed v1 attempt is preserved separately under `logs/experiments/roc-policy-v1/`. The v2 parent artifact remains in the V520 server-managed catalog; the 24 owned test servers stopped normally. No Mac inference was run for this proof.

The claim is limited to **one fictional policy, eight balanced cases, one ROCm host/build, and one greedy pass per case**. Per-case complete-state/KV exports, repeat determinism, option-order robustness, cross-platform portability for this new state, host/GPU duty, and source-level attribution of the longer-KV toll remain untested. This is authenticated preservation and continuation of a **full curatable state**, not a vLLM/SGLang prefix-cache hit or a DPR exact-transition hit.
