# Exp 3 — embedded answers, decode selects one letter

**Latest extension:** The [balanced-position and no-parent matrix](robustness.md) scored **20/20 with** and **20/20 without** the authenticated parent across four answer positions. All 40 selections were strict one-letter natural stops. The fixture shows **no accuracy dependence on parent KV**: the supplied options contain the answer. The 40-cell matrix reused five initial requests and executed 35 new ones.

“Multiple select” means **one best choice among four supplied answers** here, not a multi-label subset. [questions.json](questions.json) freezes that interpretation. This evaluates recognition among answer-bearing options, not unaided factual recall.

## Initial five-request feasibility cell

Five choices (`B, D, C, A, B`) were correct and rendered as exactly one uppercase letter each. Usage counted **2 completion tokens per selection** despite one visible letter; the receipt does not identify the unseen token here. Every request stopped naturally under an O16 cap.

- Source: the fictional [short-parent fixture](../short-kv-parent/fixtures/source.md), authenticated parent `exp-parent-synthetic-spark-ctx12288-v1` at position **925**, state SHA `3e381e4bb6b9b555b68210f9313e2aa6670d50023583e4356eb365ff2584b6b0`. No new training occurred. The model received the four options and stem, **not** the fixture's `expected_label` field.
- Host: Acerbox DGX Spark, registered full-CUDA `spark` recipe; binary SHA `1998f0835fcc9608c04cd079a2ce01e3fdb0e97f9c080ed266f5cfddee20c462`, model-source SHA `6fb79ac4b0a45113d9582d13d00a83f50f51add1c68c42080a934fd01e9145b3`, compatibility `35eeec434dfe1e68d61ffc2a6db88f4738a7ba20ee55967031bded29da967a8a`. Section 2 report checkpoint: `efa8c87e5d01c6a848f4de2d50dbea476daae50d` (**2026-09-30T19:07:56-04:00**). Acerbox execution checkout: `6c4c8fcb91fe1c94d29e4c2c6cfe0ef886eb8589` (**2026-09-29T08:22:59-04:00**), with frozen corrected model bytes and matching build receipt. The result is **binary/source-byte scoped**, not a claim that remote HEAD equaled the report checkpoint.
- One fresh managed server, one live session: import parent once, submit five four-option requests in **2.1→2.5** order through the ordinary chat API, verify and delete the final run-owned export through the server API. Parent retained. Controls: **T=0.65, top-K=20, seed=128, O16; DPR off, proof off**. Effective recipe: **W8, NFQ N48/F1/Q4, X48**, 13 decimal-GB expert budget, 60 decimal-GB all-inclusive RSS ceiling. NFQ/X values are configured, **not** measured engagement on this sampled route; no GPU duty/power trace was taken.
- Inputs/labels: [questions.json](questions.json); actual prompts/letters: [transcript.md](transcript.md); native metrics and final-state receipt: [results.json](results.json). The options describe a fictional case, not executed actions.

| Probe | Correct / selected | New input / output tokens | PREFILL ms/new input | Decode ms/output | HTTP wall s | Section 2 free-form wall s |
|---|---|---:|---:|---:|---:|---:|
| 2.1 Expected behavior | B / B | 186 / 2 | 30.85 | 67.25 | 6.008 | 25.356 |
| 2.2 Consistent behavior | D / D | 169 / 2 | 33.65 | 68.23 | 5.960 | 30.258 |
| 2.3 Consistent facts | C / C | 256 / 2 | 32.85 | 69.68 | 8.690 | 15.755 |
| 2.4 Fact skew | A / A | 176 / 2 | 36.67 | 69.76 | 6.735 | 16.623 |
| 2.5 Behavior skew | B / B | 155 / 2 | 38.44 | 70.94 | 6.242 | 8.892 |
| **Five-request total / weighted rate** | **5 / 5** | **942 / 10** | **34.231** | **69.171** | **33.636** | **96.883** |

The original five requests had **2.880× lower HTTP wall** than Section 2's free-form first run, but the tasks are not equal-work. Section 2 used **202 new input / 656 output tokens**; the choice run used **942 / 10**. It supplied **740 more input tokens** and generated **646 fewer output tokens**. This is an output-length/workshape trade, not evidence of faster kernels, equal reasoning for less compute, or parent contribution. The [later control](robustness.md) scored **20/20 without the parent**.

The initial run ended at position **1,877** with exported complete-state SHA `fa8bdf0cccb1357865d8f6f22e9dad7a692b8a5e98e4bd0b460ac12b21e15685`, facts SHA `0a4eb2dfdb4d8b770e9a159c971fdb7d0438cb01660fd269ed5b11a30f6cfe35`, and payload SHA `b96e7b02f6fb8b9c4a3943a514af4a39e41553540505feaddfa45590d6ad2266`. The verified run-owned export was deleted; the parent remained cataloged. Peak sampled all-inclusive OS RSS was **30,887,186,432 bytes (30.887 decimal GB)**. The approximate external-resident category peaked at **28,261,773,312 bytes (28.262 GB)**; it is not measured device VRAM. Startup, parent import, and explicit export (~16 s) are excluded from HTTP request wall.

**Boundary:** Option-position robustness is now measured for this seed and fixture, but not broader subjects, temperatures, or ambiguous source-dependent alternatives. The parent/no-parent accuracy equality prevents a claim that this quiz tested learned case knowledge. Initial raw HTTP and cleanup receipts remain in ignored `logs/experiments/exp3-embedded-answer-v1/spark-t0p65-s128/`; the extended matrix receipts are under `logs/experiments/exp3-robustness-v1/`.
