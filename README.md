# SALT - State Assured Latent-Trasition engine

> **Dry-run public export — not yet a release.**

SALT Core is a portable C99 state-assured latent-transition engine. Its public
contract centers deterministic arithmetic, authenticated model packages, live
state/KV ownership, portable state export/import, and subordinate CPU, Metal,
CUDA, and HIP physical backends.

> [!WARNING]
> Preserved KV/state is more than a saved prompt: it can carry the accumulated
> effects of an agent's automated activity and support continued or branched
> execution. It may also expose sensitive information. These risks exist with
> or without SALT. Retention permission is not unrestricted reuse permission;
> restoring state does not grant authority to act. Read the
> [KV Cache and Persistent Agent State Statement](docs/COGNITIVE-STATE-RISK.md).
> This is a disclosure and responsible-use standard, not a claim of implemented
> safeguards or an amendment to the software license.

The curated runtime source now matches merged private `main` revision
`ce0363260a488c96b951b4d826a188e1014575d4`
(`2026-10-02T10:58:17-04:00`) for the exported runtime paths. This updates
source bytes, not the historical proof pins or platform qualification. It
contains no model weights. Model artifacts retain their own licenses and must
be obtained separately.

The scoped [formal proof bundle](docs/FORMAL-PROOFS.md) contains executable TLA+
models and TLAPS proof sources. Its documented proof boundary is not a claim
that the complete runtime or concrete KV bytes are formally verified.

The refreshed source passed isolated Linux CPU builds of `salt` and
`gemma4-server`, a CUDA build of `gemma4-server`, and focused KV-compatibility
tests. Two independent limits remain: the model-config test also fails on the
private source because its Mac Metal fetch-touch expectation is stale (expected
`2230272`, recipe `0`), and the public `make -n test` target lacks the excluded
`tools/test/make-fixture.c` prerequisite. Neither was repaired by changing
source during this exact-byte sync. No real inference, HIP/Metal build, or
complete state/KV parity was qualified from this exported tree.

The public build, supported-platform, security, contribution, and release
documents are still review gates. Do not publish this dry-run tree.
