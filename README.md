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

This snapshot was staged from private source revision `d3a2037f404d520998ac2d3b2d182ca58f529c82`
(`2026-09-21T19:50:34-04:00`). It contains no model weights. Model artifacts retain their own
licenses and must be obtained separately.

The public build, supported-platform, security, contribution, and release
documents are still review gates. Do not publish this dry-run tree.
