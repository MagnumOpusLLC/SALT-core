# Experimental report corpus

The [detailed experimental synthesis](EXPERIMENT-SUMMARY.md) connects the seven report families below to the companion `salt-paper` manuscript. It distinguishes numerical/state equality, semantic utility, source dependence, preserved-state costs, and DPR verification; none is inferred from another.

The [source index](source-index.json) pins the **38 input files** used by the synthesis, including their exact bytes, SHA-256, and tracked status. It excludes this index and the new synthesis/entry-point documents themselves. At that snapshot, the policy-context, ROCm classifier, and CTX4096 daily-chat directories were local untracked evidence; the other four families were tracked. Links to a local report do not imply that it has already been published remotely.

| Report family | Entry point | Machine-readable evidence |
|---|---|---|
| Seeded cross-platform replay | [Randomplay](randomplay/README.md) | [results](randomplay/results.json) |
| Five held-out probes from a short parent | [Short parent](short-kv-parent/README.md) | [results](short-kv-parent/results.json) |
| Answer-bearing choices and negative parent control | [Choice experiment](exp3-multiple-select/README.md), [robustness](exp3-multiple-select/robustness.md) | [initial](exp3-multiple-select/results.json), [40-cell matrix](exp3-multiple-select/robustness-results.json) |
| Parent state versus fresh policy on three platforms | [Policy comparison](policy-kv-vs-fresh-context/README.md) | [results](policy-kv-vs-fresh-context/results.json) |
| Policy-dependent three-arm classifier | [ROCm classifier](roc-policy-classifier/README.md) | [results](roc-policy-classifier/results.json) |
| Fresh CTX4096 four-turn conversation | [Daily chat](rocm-daily-ctx4096/README.md) | [results](rocm-daily-ctx4096/results.json) |
| Document-specific ND transfer and high-X controls | [DPR revival](document-specific-dpr-revival/README.md) | [initial](document-specific-dpr-revival/results.json), [X256/X356](document-specific-dpr-revival/dpr256-356-mechanics-o512.json) |

## Manuscript connection

- Authoritative manuscript: [`../../../salt-paper/paper.tex`](../../../salt-paper/paper.tex).
- Included results section: [`../../../salt-paper/sections/experimental-results.tex`](../../../salt-paper/sections/experimental-results.tex).
- The manuscript's frozen architectural/formal baseline remains separate from these later experimental source/build identities.
- Existing Word review copies predate the newly included results section; they are not overwritten by this documentation round.

These sibling-checkout links are local authoring links. The synthesis does not commit, publish, rerun inference, disclose private KV payloads, or promote one-pass performance observations to repeat-qualified results.
