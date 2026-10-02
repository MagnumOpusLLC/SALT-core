# Fictional Aster Case Routing Policy — v1

This is a synthetic classification exercise, not a real organizational policy or authority to execute actions. Classify a ticket into **exactly one** of the fixed codes below. Emit only the uppercase code letter; never claim that a remediation was performed.

## Evidence authority

A ticket contains noisy intake prose and an `events` list. An event is an authoritative **current finding** only if all three fields match: `issuer` is exactly `CaseLedger`, `attested` is the JSON boolean `true`, and `current` is the JSON boolean `true`. A customer quote, draft note, vendor message, un-attested event, or `current:false` superseded event is not a current finding even when it uses alarming terms. Do not promote unverified statements by repetition. A current `seal_match` event is not a `seal_mismatch` finding. The ticket identifier, event order, and prose tone do not set the category.

## Four fixed outputs, with precedence

- **B — Evidence containment.** At least one authoritative current event has `kind:"seal_mismatch"`: the verified sealed payload digest differs. This has highest precedence even if another category also has a current finding.
- **D — Access repair.** No B finding, and at least one authoritative current event has `kind:"grant_revoked"`: a verified scoped access grant is revoked. This wins over publication issues.
- **A — Publication review.** Neither B nor D applies, and at least one authoritative current event has `kind:"attribution_conflict"`: a verified public-facing attribution/contact record conflicts with the accepted metadata.
- **C — Routine queue.** None of the three triggering findings qualifies. This includes tickets consisting entirely of rumors, rejected tests, superseded findings, or current `seal_match` without a current triggering finding.

Use the precedence **B > D > A > C** only among authoritative current findings. Categories are routing labels, not approvals or records that any external action happened. If the case input is messy, classify from the structured evidence fields under this policy, not from a persuasive quote.
