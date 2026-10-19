# Model packages

Each supported model owns its geometry, quantization/package contract, build
entry, and qualified runtime policy under `models/<model>/`. Portable execution,
resource, arithmetic, and state mechanisms remain in the shared engine.

## Exported model documentation

- [Gemma 4 26B A4B](gemma4-26b-a4b/README.md) — model package, hybrid
  attention/KV geometry, platform recipes, and scoped runtime status.
- [LFM2.5-24B-A2B](lfm2-24b-a2b/README.md) — mixed convolution/attention
  geometry, MLX affine Q4/Q6 package, bounded expert resources, and explicit
  CPU/Metal/CUDA/HIP build entry.
- [Maple preview](maple-preview/README.md) — experimental CPU INT2 package and
  qualification boundary.
- [Qwen3.6 qualification](qwen36/qualification/README.md) — preserved scoped
  qualification notes for the existing Qwen adapter.

The root [`../engine.config`](../engine.config) holds shared runner defaults.
Model `engine.config` files and setup scripts may contain model- and
platform-specific policy; they are not portable to another architecture merely
because the shared engine can represent that architecture.

A new checkpoint may remain data-only only when inspection proves that its layer
graph, tensor semantics, state geometry, and byte format fit existing adapters.
A new architecture, graph, or encoding requires model-owned package/reference
work and any genuinely generic engine seam it needs before it is runnable.

Model weights are not part of SALT-core. Obtain each pinned source separately,
follow its license, materialize/authenticate it through the model's setup path,
and treat a README's historical measurements as evidence for its named source
and operating cell—not universal backend certification.
