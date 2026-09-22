#!/usr/bin/env python3
"""convert-salt.py -- inventory and convert a DeepSeek-V4-Flash-class HF
checkpoint into ds4f-disk's layout:

  config.json       salt keys (no-defaults: missing key = refuse)
  trunk.bin         dense per-layer bytes, contiguous
  trunk.offsets     [u64 n][u64 off][u64 size] per layer
  pool.bin          routed experts, fixed-rate, layer-major:
                    [u64 expert_nbytes][u64 n_layers][u64 n_experts]
                    then expert 0 of layer 0 .. expert N of layer M
  manifest.json     what was mapped and what was left out

Discovery-first: `inspect` prints the actual tensor layout and the
proposed mapping. Paste that back before trusting `convert` -- naming
conventions differ between releases and the converter REFUSES rather
than guesses. Stdlib only, runs on the box.

usage:
  convert-salt.py inspect DIR
  convert-salt.py convert DIR --out OUT
  convert-salt.py make-synthetic DIR     (tiny fake HF repo for tests)
"""
import json
import os
import re
import struct
import sys

ALIASES = {
    "n_layers": ["num_hidden_layers", "n_layers", "n_layer", "num_layers"],
    "n_experts": ["n_routed_experts", "num_routed_experts", "n_experts",
                  "moe_n_experts", "num_experts"],
    "topk": ["num_experts_per_tok", "num_experts_per_token", "topk",
             "n_active_experts"],
    "n_shared": ["n_shared_experts", "num_shared_experts"],
    "hidden": ["hidden_size", "n_embd", "d_model"],
    "latent": ["latent_size", "moe_latent_size", "latent", "q_lora_rank"],
    "moe_inter": ["moe_intermediate_size", "intermediate_size",
                  "ffn_hidden_size", "moe_ffn_hidden_size"],
    "n_kv_heads": ["num_key_value_heads", "n_kv_heads"],
    "head_dim": ["head_dim", "kv_channels"],
    "shared_inter": ["shared_expert_intermediate_size"],
}
REQUIRED = ["n_layers", "n_experts", "topk", "n_shared", "hidden"]

# MLX 4-bit Qwen3.5 names tensors model.language_model.model.layers.N.*
# (note the doubled "model.") with per-tensor .scales/.biases siblings;
# the vision tower is vision_tower.blocks.* -- always stripped.
EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:mlp|ffn)\.experts\.(\d+)\.(.+)$")
SHARED_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:mlp|ffn)\.shared_experts?\.(.+)$")
LAYER_RE = re.compile(r"^(?:(?:model\.)?(?:language_model\.)?(?:model\.)?)layers\.(\d+)\.")
MTP_RE = re.compile(r"^(?:(?:model\.)?(?:language_model\.)?(?:model\.)?)mtp\.(\d+)\.")
VISION_RE = re.compile(r"^(?:model\.)?(?:vision_model|vision_tower|merger|visual)\.")


def skeleton(name):
    """Name with every digit run collapsed to N, e.g.
    layers.0.attn.wq_a.weight -> layers.N.attn.wq_a.weight. The
    histogram of skeletons reveals a checkpoint's naming scheme at a
    glance without dumping tens of thousands of names."""
    return re.sub(r"\d+", "N", name)


def hsize(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


# ----------------------------------------------------------------------
# safetensors / HF repo reading
# ----------------------------------------------------------------------

def read_safetensors_index(path):
    """Return {tensor_name: (dtype, shape, [a, b])} with byte spans
    relative to the payload, plus payload_begin = 8 + header_len."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return {}, 0
            hlen = struct.unpack("<Q", raw)[0]
            hdr = f.read(hlen)
            if len(hdr) != hlen:
                return {}, 0
        idx = json.loads(hdr)
    except (json.JSONDecodeError, OSError):
        return {}, 0
    out = {}
    for name, meta in idx.items():
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            continue
        a, b = meta["data_offsets"]
        out[name] = (meta.get("dtype", "?"), meta.get("shape", []), [a, b])
    return out, 8 + hlen


def discover(dirpath):
    """Return (shards {name: (abspath, off_in_file, nbytes)},
    index_path, config dict)."""
    index_path = os.path.join(dirpath, "model.safetensors.index.json")
    shards = {}
    config = {}
    cfg_path = os.path.join(dirpath, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)

    weight_map = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            ij = json.load(f)
        weight_map = ij.get("weight_map", {})

    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".safetensors"))
    if not files:
        return {}, index_path, config

    begins = {}
    for fn in files:
        idx, pb = read_safetensors_index(os.path.join(dirpath, fn))
        begins[fn] = pb
        for name, (dtype, shape, [a, b]) in idx.items():
            shards[name] = (os.path.join(dirpath, fn), pb + a, b - a,
                            dtype, shape)

    if weight_map:  # index.json is authoritative about placement
        shards = {}
        for name, fn in weight_map.items():
            fn = os.path.basename(fn)
            if fn not in begins:
                continue
            idx, pb = read_safetensors_index(os.path.join(dirpath, fn))
            if name in idx:
                a, b = idx[name][2]
                shards[name] = (os.path.join(dirpath, fn), pb + a, b - a,
                                idx[name][0], idx[name][1])
    return shards, index_path, config


def looks_like_repo(d):
    return (os.path.exists(os.path.join(d, "config.json")) or
            any(f.endswith(".safetensors")
                for f in os.listdir(d)) if os.path.isdir(d) else False)


def find_repo_root(dirpath, maxdepth=5):
    """If dirpath itself is not an HF repo, walk down (bounded) to find
    the deepest-looking subdir with config.json or .safetensors --
    HF git clones put the real files under snapshots/<hash>/. Returns
    None when nothing repo-like exists anywhere."""
    if os.path.isdir(dirpath) and looks_like_repo(dirpath):
        return dirpath
    import collections
    q = collections.deque([(dirpath, 0)])
    found = []
    while q:
        d, depth = q.popleft()
        if depth >= maxdepth:
            continue
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for e in entries:
            p = os.path.join(d, e)
            if os.path.isdir(p) and not os.path.islink(p):
                if looks_like_repo(p):
                    found.append(p)
                q.append((p, depth + 1))
    return sorted(found, key=len)[0] if found else None


def tree_summary(dirpath):
    """What is actually in this directory, when it is not an HF repo."""
    try:
        entries = sorted(os.listdir(dirpath))
    except OSError as ex:
        print(f"cannot list {dirpath}: {ex}")
        return
    print(f"contents of {dirpath} ({len(entries)} entries):")
    exts = {}
    for e in entries:
        p = os.path.join(dirpath, e)
        if os.path.isdir(p):
            print(f"  [dir]  {e}/")
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = -1
        ext = os.path.splitext(e)[1].lower() or "(none)"
        exts[ext] = exts.get(ext, 0) + 1
        print(f"  [file] {e}  ({hsize(sz)})")
    if exts:
        print("extensions: " + ", ".join(f"{k} x{v}" for k, v in
                                         sorted(exts.items())))
    if ".gguf" in exts:
        print("note: GGUF found -- salt reads safetensors; convert to "
              "safetensors first (llama.cpp/convert_hf_to_gguf reverse)")
    if entries and not exts:
        print("note: only directories -- HF snapshot layout?")


# ----------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------

def cmd_inspect(dirpath):
    root = find_repo_root(dirpath)
    if root != dirpath:
        tree_summary(dirpath)
        if root:
            print(f"\nfound repo-like content under: {root}")
            dirpath = root
        else:
            print("\nno repo-like content found anywhere (no config.json, "
                  "no .safetensors). If this is a git-lfs clone, run "
                  "`git lfs pull` first, then re-inspect.")
            return
    shards, index_path, config = discover(dirpath)
    print(f"repo: {dirpath}")
    print(f"shards: {len(set(os.path.basename(f) for f, *_ in shards.values()))}"
          f" safetensors file(s), {len(shards)} tensors")
    print(f"index.json: {'present' if os.path.exists(index_path) else 'absent'}")
    print(f"config.json: {'present' if config else 'ABSENT'}")

    mapped, assumptions = map_config(config)
    print("\nconfig mapping (HF key -> salt key):")
    for key in REQUIRED + ["latent", "moe_inter"]:
        v = mapped.get(key)
        print(f"  {key:10s} = {v if v is not None else 'MISSING (refuse)'}")
    if assumptions:
        print("  (latent/moe_inter assumed = hidden_size; reported, not silent)")

    if not shards:
        print("\nno safetensors found; nothing else to classify")
        return

    experts, dense, shared, other = classify(shards)
    eb = None
    if experts:
        per_layer = {}
        for (L, e), tens in experts.items():
            per_layer.setdefault(L, {})[e] = sum(t[3] for t in tens)
        sizes = {s for L in per_layer for s in per_layer[L].values()}
        uniform = len(sizes) == 1
        eb = next(iter(sizes)) if uniform else None
        print(f"\nrouted experts: {len(experts)} across {len(per_layer)} layers")
        if uniform:
            print(f"  per-expert payload: {hsize(eb)} (uniform, fixed-rate OK)")
        else:
            print(f"  NOT UNIFORM ({len(sizes)} distinct sizes) -> cannot "
                  f"build a fixed-rate pool; refuse")
        for (L, e) in sorted(experts)[:3]:
            print(f"  e({L},{e}): " +
                  ", ".join(f"{'.'.join(t[0].split('.')[-2:])}={hsize(t[3])}"
                            for t in experts[(L, e)]))
    else:
        print("\nrouted experts: NONE matched the .experts.{e}. pattern")

    trunk_bytes = sum(t[3] for tens in dense.values() for t in tens)
    print(f"dense per-layer tensors: {len(dense)} layers, "
          f"{hsize(trunk_bytes)} total (excl. routed experts)")
    if dense:
        L0 = min(dense)
        print(f"  sample layer {L0}: " +
              ", ".join(f"{'.'.join(t[0].split('.')[-2:])}={hsize(t[3])}"
                        for t in sorted(dense[L0])[:8]))
    if shared:
        print(f"shared experts: {hsize(sum(t[3] for ts in shared.values() for t in ts))}"
              f" (resident)")
    mtp = [n for n in shards if MTP_RE.match(n)]
    if mtp:
        print(f"MTP block (multi-token prediction): {len(mtp)} tensors, "
              f"{hsize(sum(shards[n][2] for n in mtp))} -- excluded from "
              f"pool/trunk (main layers only)")
    if other:
        ob = sum(t[3] for t in other)
        print(f"unclassified (embed/norm/lm_head/...): {hsize(ob)}")
        for t in other[:10]:
            print(f"  {t[0]} ({hsize(t[3])})")

    # the naming scheme, at a glance: digit runs collapsed to N
    from collections import Counter
    skel = Counter(skeleton(n) for n in shards)
    print("\nname skeletons (digits -> N), most common:")
    for sk, cnt in skel.most_common(15):
        print(f"  {cnt:7d}  {sk}")
    if skel and len(skel) > 15:
        print(f"  ... {len(skel) - 15} more skeletons")

    if eb:
        print(f"\nestimate: pool.bin {hsize(eb * len(experts))}, "
              f"trunk.bin {hsize(trunk_bytes)}")
        print("  (bytes copied as-is; dtypes preserved, no quantization)")


def classify(shards):
    """experts: {(L, e): [(name, file, off, nbytes)]}
    dense: {L: [(...)]}  (layer tensors minus experts)
    shared: {L: [(...)]}, other: [(...)]."""
    experts, dense, shared, other = {}, {}, {}, []
    for name, (fn, off, nb, dt, shp) in shards.items():
        if VISION_RE.match(name):
            continue                      # vision tower: stripped
        m = EXPERT_RE.search(name)
        if m:
            experts.setdefault((int(m.group(1)), int(m.group(2))),
                               []).append((name, fn, off, nb, dt, shp))
            continue
        m = SHARED_RE.search(name)
        if m:
            shared.setdefault(int(m.group(1)),
                              []).append((name, fn, off, nb, dt, shp))
            continue
        m = LAYER_RE.match(name)
        if m:
            dense.setdefault(int(m.group(1)),
                             []).append((name, fn, off, nb, dt, shp))
            continue
        # global tensors carried in the trunk (the engine indexes them
        # by role): hc_head_* (mHC output contraction) go in layer 0
        if name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            dense.setdefault(0, []).append((name, fn, off, nb, dt, shp))
            continue
        other.append((name, fn, off, nb, dt, shp))
    return experts, dense, shared, other


def map_config(config):
    mapped = {}
    assumptions = []
    # Qwen3.5 nests the text-model keys under text_config
    if isinstance(config, dict) and isinstance(config.get("text_config"), dict):
        cfg = {**config, **config["text_config"]}
    else:
        cfg = config or {}
    for key, aliases in ALIASES.items():
        src = next((a for a in aliases if a in cfg), None)
        if src is not None:
            mapped[key] = int(cfg[src])
        elif key in REQUIRED:
            mapped[key] = None
        else:
            assumptions.append(key)
    if "latent" in assumptions and mapped.get("hidden"):
        mapped["latent"] = mapped["hidden"]
        assumptions.remove("latent")
    if "moe_inter" in assumptions and mapped.get("hidden"):
        mapped["moe_inter"] = mapped["hidden"]
        assumptions.remove("moe_inter")
    # Qwen3.5 has 1 shared expert (shared_expert_intermediate_size
    # present); dense MoE without any shared-expert key -> 0. n_shared
    # is in REQUIRED, so a true DS-V4 miss still refuses below -- but
    # first let absence resolve to 0 (reported, not silent).
    if "n_shared" not in mapped or mapped["n_shared"] is None:
        has_shared_key = any(a in cfg for a in
                             ("n_shared_experts", "num_shared_experts",
                              "shared_expert_intermediate_size",
                              "shared_experts"))
        mapped["n_shared"] = 1 if has_shared_key else 0
        if "n_shared" in assumptions:
            assumptions.remove("n_shared")
    # GQA geometry: head_dim defaults to hidden // kv_heads (reported).
    if mapped.get("n_kv_heads") and not mapped.get("head_dim"):
        mapped["head_dim"] = mapped["hidden"] // mapped["n_kv_heads"]
        assumptions.append("head_dim")
    return mapped, assumptions


# ----------------------------------------------------------------------
# convert
# ----------------------------------------------------------------------

def copy_range(fsrc, off, nb, fdst, buf):
    fsrc.seek(off)
    left = nb
    while left > 0:
        got = fsrc.read(min(len(buf), left))
        if not got:
            raise RuntimeError("short read")
        fdst.write(got)
        left -= len(got)


def put_u64(f, v):
    f.write(struct.pack("<Q", v))


def cmd_convert(dirpath, outdir):
    root = find_repo_root(dirpath)
    if root is None:
        print(f"REFUSE: no repo-like content under {dirpath} "
              f"(no config.json, no .safetensors anywhere)")
        sys.exit(1)
    if root != dirpath:
        print(f"found repo-like content under: {root}")
        dirpath = root
    shards, _, config = discover(dirpath)
    mapped, assumptions = map_config(config)
    missing = [k for k in REQUIRED if not mapped.get(k)]
    if missing:
        print(f"REFUSE: config keys missing: {', '.join(missing)} "
              f"(run inspect first)")
        sys.exit(1)

    experts, dense, shared, other = classify(shards)
    if not experts:
        print("REFUSE: no routed experts matched the .experts.{e}. pattern")
        sys.exit(1)
    layers = sorted({L for L, _ in experts})
    if len(layers) != mapped["n_layers"]:
        print(f"REFUSE: config says {mapped['n_layers']} layers, found "
              f"{len(layers)} with experts")
        sys.exit(1)
    for L in layers:
        if L not in dense:
            print(f"REFUSE: layer {L} has experts but no dense tensors")
            sys.exit(1)

    per_layer = {L: {} for L in layers}
    for (L, e), tens in experts.items():
        per_layer[L][e] = sum(t[3] for t in tens)
    sizes = {s for L in per_layer for s in per_layer[L].values()}
    if len(sizes) != 1:
        print(f"REFUSE: expert payloads not uniform: {sorted(sizes)}")
        sys.exit(1)
    expert_bytes = next(iter(sizes))
    n_experts = mapped["n_experts"]
    for L in layers:
        if len(per_layer[L]) != n_experts:
            print(f"REFUSE: layer {L} has {len(per_layer[L])} experts, "
                  f"config says {n_experts}")
            sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    # ---------------- config.json ----------------
    cfg_out = {
        "n_layers": mapped["n_layers"],
        "n_experts": mapped["n_experts"],
        "topk": mapped["topk"],
        "n_shared": mapped["n_shared"],
        "hidden": mapped["hidden"],
        "latent": mapped["latent"],
        "moe_inter": mapped["moe_inter"],
        "expert_nbytes": expert_bytes,
        "seed": 7,
    }
    # the real attention geometry (optional; the engine falls back)
    for src_key, out_key in (("num_attention_heads", "num_attention_heads"),
                             ("qk_rope_head_dim", "qk_rope_head_dim"),
                             ("num_key_value_heads", "num_key_value_heads"),
                             ("head_dim", "head_dim"),
                             ("shared_expert_intermediate_size",
                              "shared_inter")):
        if config is not None and src_key in config:
            cfg_out[out_key] = int(config[src_key])
    if mapped.get("n_kv_heads") and "num_key_value_heads" not in cfg_out:
        cfg_out["num_key_value_heads"] = mapped["n_kv_heads"]
    if mapped.get("head_dim") and "head_dim" not in cfg_out:
        cfg_out["head_dim"] = mapped["head_dim"]
    # the tyrope params (flattened from the nested rope_scaling dict)
    rs = (config or {}).get("rope_scaling") or {}
    for src_key, out_key in (("factor", "rope_factor"),
                             ("beta_fast", "rope_beta_fast"),
                             ("beta_slow", "rope_beta_slow"),
                             ("original_max_position_embeddings",
                              "rope_max_pos")):
        if src_key in rs:
            cfg_out[out_key] = rs[src_key]
    if config is not None and "rope_theta" in config:
        cfg_out.setdefault("rope_theta", config["rope_theta"])
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"config.json written (expert_nbytes={expert_bytes})")

    # ---------------- trunk.bin + trunk.offsets ----------------
    srcs = {}
    def src(name):
        fn, off, nb, _, _ = shards[name]
        if fn not in srcs:
            srcs[fn] = open(fn, "rb")
        return srcs[fn], off, nb

    tbin = open(os.path.join(outdir, "trunk.bin"), "wb")
    toff = open(os.path.join(outdir, "trunk.offsets"), "wb")
    put_u64(toff, len(layers))
    buf = bytearray(1 << 20)
    at = 0
    trunk_layout = {"n_layers": len(layers), "layers": []}
    align = 8
    for L in layers:
        tens = sorted(dense[L])
        put_u64(toff, at)
        lay_bytes = 0
        ltens = []
        # 8-byte alignment per tensor: the engine reads F32/BF16 tensors
        # through typed pointers, and misaligned offsets are UB that
        # clang -O2 exploits (widened loads past the buffer -> garbage).
        for t in tens:
            f, off, nb = src(t[0])
            pad = (-lay_bytes) % align
            if pad:
                tbin.write(b"\0" * pad)
                lay_bytes += pad
            copy_range(f, off, nb, tbin, buf)
            ltens.append({"n": t[0], "dtype": t[4], "shape": list(t[5]),
                          "off": lay_bytes, "nbytes": nb})
            lay_bytes += nb
        put_u64(toff, lay_bytes)
        trunk_layout["layers"].append({"layer": L, "tensors": ltens})
        at += lay_bytes
        if L + 1 < len(layers):
            pad = (-at) % align
            if pad:
                tbin.write(b"\0" * pad)
                at += pad
        print(f"  trunk layer {L}: {hsize(lay_bytes)}")
    tbin.close()
    toff.close()
    with open(os.path.join(outdir, "trunk.json"), "w") as f:
        json.dump(trunk_layout, f, indent=1)
    print(f"trunk.bin {hsize(at)} ({len(layers)} layers)")

    # ---------------- pool.bin ----------------
    pbin = open(os.path.join(outdir, "pool.bin"), "wb")
    put_u64(pbin, expert_bytes)
    put_u64(pbin, len(layers))
    put_u64(pbin, n_experts)
    written = 0
    done = 0
    total = len(layers) * n_experts
    for L in layers:
        for e in range(n_experts):
            tens = sorted(experts[(L, e)])
            for t in tens:
                f, off, nb = src(t[0])
                copy_range(f, off, nb, pbin, buf)
            written += expert_bytes
            done += 1
            if done % 2000 == 0:
                print(f"  pool: {done}/{total} experts, {hsize(written)}")
    pbin.close()
    print(f"pool.bin {hsize(written)} ({done} experts x {hsize(expert_bytes)})")

    # ---------------- embed.bin + head.bin (issue #6 step 3) ----------
    # The autoregressive loop needs the embedding table (gather) and the
    # output head (logits). Both live in the unclassified set; bytes are
    # copied as-is with a tiny layout json per file. Discovery-first:
    # candidate names (checkpoint naming varies), optional per-tensor
    # scale sibling (block scales when present, null otherwise).
    def find_other(name):
        for t in other:
            if t[0] == name:
                return t
        return None

    def find_other_any(names):
        for n in names:
            t = find_other(n)
            if t:
                return t
        return None

    EMBED_NAMES = ["embed.weight", "model.embed_tokens.weight",
                   "embed_tokens.weight", "wte.weight", "word_embeddings"]
    HEAD_NAMES = ["head.weight", "lm_head.weight", "output.weight",
                  "wpe.weight", "unembed.weight"]

    emb = find_other_any(EMBED_NAMES)
    if emb:
        f, off, nb = src(emb[0])
        with open(os.path.join(outdir, "embed.bin"), "wb") as eb:
            copy_range(f, off, nb, eb, buf)
        scale = None
        for cand in (emb[0] + ".scale",
                     emb[0].replace(".weight", ".scale")):
            st = find_other(cand)
            if st:
                f2, o2, n2 = src(st[0])
                with open(os.path.join(outdir, "embed.bin"), "ab") as eb:
                    copy_range(f2, o2, n2, eb, buf)
                scale = {"off": nb, "nbytes": n2, "dtype": st[4],
                         "shape": list(st[5])}
                break
        with open(os.path.join(outdir, "embed.json"), "w") as ej:
            json.dump({"bin": "embed.bin", "dtype": emb[4],
                       "shape": list(emb[5]), "nbytes": nb,
                       "scale": scale}, ej, indent=1)
        print(f"embed.bin {hsize(nb)} {emb[4]} {list(emb[5])} "
              f"(scale: {'yes' if scale else 'no'})")

    hw = find_other_any(HEAD_NAMES)
    hs = find_other("head.scale") or find_other("lm_head.scale") or \
         find_other("output.scale")
    if hw:
        f_w, off_w, nb_w = src(hw[0])
        with open(os.path.join(outdir, "head.bin"), "wb") as hb:
            copy_range(f_w, off_w, nb_w, hb, buf)
        scale = None
        scale_n = 0
        if hs:
            f_s, off_s, nb_s = src(hs[0])
            with open(os.path.join(outdir, "head.bin"), "ab") as hb:
                copy_range(f_s, off_s, nb_s, hb, buf)
            scale = {"off": nb_w, "nbytes": nb_s, "dtype": hs[4],
                     "shape": list(hs[5])}
            scale_n = nb_s
        with open(os.path.join(outdir, "head.json"), "w") as hj:
            json.dump({
                "bin": "head.bin",
                "weight": {"off": 0, "nbytes": nb_w, "dtype": hw[4],
                           "shape": list(hw[5])},
                "scale": scale,
            }, hj, indent=1)
        print(f"head.bin {hsize(nb_w + scale_n)} "
              f"{hw[4]} {list(hw[5])} (scale: {'yes' if hs else 'no'})")

    for f in srcs.values():
        f.close()

    # ---------------- manifest.json ----------------
    manifest = {
        "config": cfg_out,
        "assumptions": assumptions,
        "layers": len(layers),
        "expert_bytes": expert_bytes,
        "trunk_bytes": at,
        "pool_bytes": written,
        "shared_expert_bytes": sum(t[3] for ts in shared.values() for t in ts),
        "unclassified_tensors": [(t[0], t[3]) for t in other],
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # verify round trip
    if os.path.getsize(os.path.join(outdir, "pool.bin")) != 24 + written:
        print("VERIFY FAIL: pool size mismatch")
        sys.exit(1)
    if os.path.getsize(os.path.join(outdir, "trunk.bin")) != at:
        print("VERIFY FAIL: trunk size mismatch")
        sys.exit(1)
    print("\nconvert complete. run:")
    print(f"  ./salt {outdir} --trunk {outdir}/trunk.bin "
          f"--offsets {outdir}/trunk.offsets --pool {outdir}/pool.bin "
          f"--preset laptop")


# ----------------------------------------------------------------------
# make-synthetic: tiny fake HF repo (2 layers, 4 experts, 2 shards)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# make-synthetic-qwen35: tiny Qwen3.5-shaped HF repo (2 layers, 4 experts)
# ----------------------------------------------------------------------

def cmd_make_synthetic_qwen35(dirpath):
    """Tiny Qwen3.5-MoE-shaped repo: model.language_model.layers.N.*
    names, self_attn q/k/v/o, mlp.experts gate/up/down, shared_expert,
    vision tower (to be stripped), MTP. Exercises the real naming."""
    os.makedirs(dirpath, exist_ok=True)
    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000000,
            "max_position_embeddings": 262144,
            "vocab_size": 64,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        },
    }
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    names = ["model.language_model.embed_tokens.weight",
             "model.language_model.norm.weight", "lm_head.weight"]
    for L in range(2):
        names += [
            f"model.language_model.layers.{L}.input_layernorm.weight",
            f"model.language_model.layers.{L}.post_attention_layernorm.weight",
            f"model.language_model.layers.{L}.self_attn.q_proj.weight",
            f"model.language_model.layers.{L}.self_attn.k_proj.weight",
            f"model.language_model.layers.{L}.self_attn.v_proj.weight",
            f"model.language_model.layers.{L}.self_attn.o_proj.weight",
            f"model.language_model.layers.{L}.mlp.gate.weight",
            f"model.language_model.layers.{L}.mlp.shared_expert.w1.weight",
            f"model.language_model.layers.{L}.mlp.shared_expert.w2.weight",
        ]
        for e in range(4):
            names += [
                f"model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight",
                f"model.language_model.layers.{L}.mlp.experts.{e}.up_proj.weight",
                f"model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight",
            ]
    names += ["model.vision_model.embed_tokens.weight",
              "model.merger.weight"]  # must be stripped

    def blob(nbytes, dtype):
        out = bytearray()
        x = 0x1234
        if dtype == "BF16":
            for i in range(nbytes // 2):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                v = ((x & 0xFFFF) / 65535.0 - 0.5) * 2.0
                f32 = struct.unpack("<I", struct.pack("<f", v))[0]
                out += struct.pack("<H", (f32 >> 16) & 0xFFFF)
            return bytes(out)
        for i in range(nbytes):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append((x + i) & 0xFF)
        return bytes(out)

    shape_of = {}
    H, E, MI = 8, 4, 16
    shape_of["model.language_model.embed_tokens.weight"] = [64, H]
    shape_of["model.language_model.norm.weight"] = [H]
    shape_of["lm_head.weight"] = [64, H]
    for L in range(2):
        shape_of[f"model.language_model.layers.{L}.input_layernorm.weight"] = [H]
        shape_of[f"model.language_model.layers.{L}.post_attention_layernorm.weight"] = [H]
        for p, r, c in (("q_proj", H, H), ("k_proj", H // 2, H),
                        ("v_proj", H // 2, H), ("o_proj", H, H)):
            shape_of[f"model.language_model.layers.{L}.self_attn.{p}.weight"] = [r, c]
        shape_of[f"model.language_model.layers.{L}.mlp.gate.weight"] = [E, H]
        shape_of[f"model.language_model.layers.{L}.mlp.shared_expert.w1.weight"] = [MI, H]
        shape_of[f"model.language_model.layers.{L}.mlp.shared_expert.w2.weight"] = [H, MI]
        for e in range(E):
            shape_of[f"model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight"] = [MI, H]
            shape_of[f"model.language_model.layers.{L}.mlp.experts.{e}.up_proj.weight"] = [MI, H]
            shape_of[f"model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight"] = [H, MI]
    shape_of["model.vision_model.embed_tokens.weight"] = [16, H]
    shape_of["model.merger.weight"] = [H, H]

    hdr = {}
    off = 8
    payload = bytearray()
    for n in names:
        shp = shape_of[n]
        r = shp[0]
        c = shp[1] if len(shp) == 2 else 1
        nb = r * c * 2          # all BF16
        hdr[n] = {"dtype": "BF16", "shape": list(shp),
                  "data_offsets": [off, off + nb]}
        payload += blob(nb, "BF16")
        off += nb
    hdrj = json.dumps(hdr).encode()
    with open(os.path.join(dirpath, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdrj)))
        f.write(hdrj)
        f.write(bytes(payload))
    print(f"synthetic Qwen3.5 repo at {dirpath}: {len(names)} tensors")


def cmd_make_synthetic(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    cfg = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 16,
        "model_type": "deepseek_moe",
    }
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    names = ["embed.weight", "head.weight", "head.scale"]
    for L in range(2):
        names += [
            f"layers.{L}.hc_attn_fn",
            f"layers.{L}.hc_attn_base",
            f"layers.{L}.hc_attn_scale",
            f"layers.{L}.hc_ffn_fn",
            f"layers.{L}.hc_ffn_base",
            f"layers.{L}.hc_ffn_scale",
            f"layers.{L}.attn.attn_sink",
            f"layers.{L}.attn.q_norm.weight",
            f"layers.{L}.attn.kv_norm.weight",
            f"layers.{L}.attn.wq_a.weight",
            f"layers.{L}.attn.wq_a.scale",
            f"layers.{L}.attn.wq_b.weight",
            f"layers.{L}.attn.wq_b.scale",
            f"layers.{L}.attn.wkv.weight",
            f"layers.{L}.attn.wkv.scale",
            f"layers.{L}.attn.wo_a.weight",
            f"layers.{L}.attn.wo_a.scale",
            f"layers.{L}.attn.wo_b.weight",
            f"layers.{L}.attn.wo_b.scale",
            f"layers.{L}.attn.wo_c.weight",
            f"layers.{L}.attn.wo_c.scale",
            f"layers.{L}.ffn.gate.weight",
            f"layers.{L}.ffn.gate.bias",
            f"layers.{L}.ffn.down",
            f"layers.{L}.ffn.up",
            f"layers.{L}.ffn.shared_experts.w1.weight",
            f"layers.{L}.ffn.shared_experts.w1.scale",
            f"layers.{L}.ffn.shared_experts.w2.weight",
            f"layers.{L}.ffn.shared_experts.w2.scale",
        ]
        for e in range(4):
            for w in (1, 2, 3):
                names += [
                    f"layers.{L}.ffn.experts.{e}.w{w}.weight",
                    f"layers.{L}.ffn.experts.{e}.w{w}.scale",
                ]
    names += ["norm.weight"]
    names += [f"mtp.{m}.hc_attn_base" for m in range(1)]
    names += ["hc_head_fn", "hc_head_base", "hc_head_scale"]
    names += [f"mtp.0.ffn.experts.0.w1.weight", f"mtp.0.ffn.experts.0.w1.scale"]

    def blob(name, n, dtype):
        if dtype == "F32":             # valid bounded floats, not raw bytes
            out = bytearray()
            x = 0x1234
            for i in range(n // 4):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                v = ((x & 0xFFFF) / 65535.0 - 0.5) * 2.0   # [-1, 1]
                out += struct.pack("<f", v)
            return bytes(out)
        if dtype == "F8_E8M0":         # tiny scales: 2^-10 .. 2^-7
            out = bytearray()
            x = 0x1234
            for i in range(n):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                out.append(117 + (x & 0x03))
            return bytes(out)
        if dtype == "I8":              # small weights: int8 in [-4, 4]
            out = bytearray()
            x = 0x1234
            for i in range(n):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                out.append(((x & 0x7) - 4) & 0xFF)
            return bytes(out)
        if dtype == "BF16":            # small bf16 weights in [-1, 1]
            out = bytearray()
            x = 0x1234
            for i in range(n // 2):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                v = ((x & 0xFFFF) / 65535.0 - 0.5) * 2.0   # [-1, 1]
                f32 = struct.unpack("<I", struct.pack("<f", v))[0]
                out += struct.pack("<H", (f32 >> 16) & 0xFFFF)
            return bytes(out)
        x = 0x1234
        out = bytearray()
        for i in range(n):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append((x + ord(name[0])) & 0xFF)
        return bytes(out)

    sizes = {}
    for i, name in enumerate(names):
        if "experts." in name:
            w = int(name.split("w")[1].split(".")[0])   # 1, 2 or 3
            if name.endswith(".scale"):
                sizes[name] = {1: 8, 2: 16, 3: 8}[w]    # block16 scales
            else:
                sizes[name] = {1: 128, 2: 256, 3: 128}[w]  # I8, 2-D
        elif name.endswith("ffn.gate.weight"):
            sizes[name] = 64             # [4 x 8] BF16 (2 B/elem)
        elif name.endswith("ffn.gate.bias"):
            sizes[name] = 16             # [4] F32
        elif name.endswith("ffn.down"):
            sizes[name] = 256            # [8 x 8] F32
        elif name.endswith("ffn.up"):
            sizes[name] = 256            # [8 x 8] F32
        elif "shared_experts" in name and name.endswith(".scale"):
            sizes[name] = 4
        elif "shared_experts" in name:
            sizes[name] = 24
        elif "embed" in name:
            sizes[name] = 2048           # [64 x 8] F32
        elif name == "head.weight":
            sizes[name] = 512            # [64 x 8] F8_E4M3
        elif name == "head.scale":
            sizes[name] = 1              # [1 x 1] F8_E8M0
        elif name.endswith("attn.attn_sink"):
            sizes[name] = 256            # [64] F32, sink anchors
        elif name.endswith("q_norm.weight") or name.endswith("kv_norm.weight"):
            sizes[name] = 8              # [4] BF16
        elif "attn." in name and name.endswith(".weight"):
            w = name.split(".")[-2]
            sizes[name] = {"wq_a": 32, "wq_b": 16, "wkv": 32, "wo_a": 16,
                           "wo_b": 64, "wo_c": 64}[w]   # F8, 2-D
        elif "attn." in name and name.endswith(".scale"):
            sizes[name] = 1              # [1 x 1] F8_E8M0
        else:
            sizes[name] = 32 + (i % 3) * 16

    # 2-D shapes so the engine's matvec chain has real dims
    shape_of = {}
    # global HC head (mHC output contraction; carried in layer 0)
    shape_of["hc_head_fn"] = [24, 32]
    shape_of["hc_head_base"] = [24]
    shape_of["hc_head_scale"] = [3]
    for L in range(2):
        shape_of[f"layers.{L}.attn.wq_a.weight"] = [4, 8]
        shape_of[f"layers.{L}.attn.wq_a.scale"] = [1, 1]
        shape_of[f"layers.{L}.attn.wq_b.weight"] = [4, 4]
        shape_of[f"layers.{L}.attn.wq_b.scale"] = [1, 1]
        shape_of[f"layers.{L}.attn.wkv.weight"] = [4, 8]
        shape_of[f"layers.{L}.attn.wkv.scale"] = [1, 1]
        shape_of[f"layers.{L}.attn.wo_a.weight"] = [8, 2]
        shape_of[f"layers.{L}.attn.wo_a.scale"] = [1, 1]
        shape_of[f"layers.{L}.attn.wo_b.weight"] = [8, 8]
        shape_of[f"layers.{L}.attn.wo_b.scale"] = [1, 1]
        shape_of[f"layers.{L}.attn.wo_c.weight"] = [8, 8]
        shape_of[f"layers.{L}.attn.wo_c.scale"] = [1, 1]
        shape_of[f"layers.{L}.attn.q_norm.weight"] = [4]
        shape_of[f"layers.{L}.attn.kv_norm.weight"] = [4]
        shape_of[f"layers.{L}.attn.attn_sink"] = [64]
        shape_of[f"layers.{L}.hc_attn_fn"] = [24, 32]
        shape_of[f"layers.{L}.hc_attn_base"] = [24]
        shape_of[f"layers.{L}.hc_attn_scale"] = [3]
        shape_of[f"layers.{L}.hc_ffn_fn"] = [24, 32]
        shape_of[f"layers.{L}.hc_ffn_base"] = [24]
        shape_of[f"layers.{L}.hc_ffn_scale"] = [3]
        shape_of[f"layers.{L}.ffn.gate.weight"] = [4, 8]
        shape_of[f"layers.{L}.ffn.gate.bias"] = [4]
        shape_of[f"layers.{L}.ffn.down"] = [8, 8]
        shape_of[f"layers.{L}.ffn.up"] = [8, 8]
        for e in range(4):
            shape_of[f"layers.{L}.ffn.experts.{e}.w1.weight"] = [16, 8]
            shape_of[f"layers.{L}.ffn.experts.{e}.w2.weight"] = [16, 16]
            shape_of[f"layers.{L}.ffn.experts.{e}.w3.weight"] = [8, 16]
    shape_of["embed.weight"] = [64, 8]
    shape_of["head.weight"] = [64, 8]
    shape_of["head.scale"] = [1, 1]

    n_shards = 2
    wm = {}
    for s in range(n_shards):
        snames = names[s::n_shards]
        payload = b""
        entries = {}
        for name in snames:
            if name.endswith(".scale") and ("experts." in name
                                            or "attn." in name
                                            or name == "head.scale"):
                dt, n = "F8_E8M0", sizes[name]  # real ckpt: E8M0 group scales
            elif name.endswith(".scale"):
                dt, n = "F32", 1
            elif name.endswith("ffn.gate.weight"):
                dt, n = "BF16", sizes[name] // 2  # real ckpt: BF16 router
            elif name.endswith("q_norm.weight") or name.endswith("kv_norm.weight"):
                dt, n = "BF16", sizes[name] // 2  # MLA norms
            elif name.endswith("head.weight") or (
                    "attn." in name and name.endswith(".weight")):
                dt, n = "F8_E4M3", sizes[name]    # attention: F8 + scales
            elif ".weight" in name and ("experts" in name
                                        or "shared_experts" in name):
                dt, n = "I8", sizes[name]   # real checkpoint: I8 + scales
            else:
                dt, n = "F32", sizes[name] // 4
            a = len(payload)
            data = blob(name, sizes[name], dt)
            payload += data
            entries[name] = {"dtype": dt, "shape": shape_of.get(name, [n]),
                             "data_offsets": [a, a + sizes[name]]}
            wm[name] = f"model-0000{s + 1}-of-0000{n_shards}.safetensors"
        hdr = json.dumps(entries).encode()
        with open(os.path.join(dirpath,
                               f"model-0000{s + 1}-of-0000{n_shards}.safetensors"),
                  "wb") as f:
            f.write(struct.pack("<Q", len(hdr)))
            f.write(hdr)
            f.write(payload)
    with open(os.path.join(dirpath, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": sum(sizes.values())},
                   "weight_map": wm}, f, indent=2)
    print(f"synthetic HF repo at {dirpath}: {len(names)} tensors, "
          f"{n_shards} shards")


# ----------------------------------------------------------------------
# mxfp4 quantization (issue #2, milestone step 1)
# ----------------------------------------------------------------------

try:
    import numpy as _np
except ImportError:
    _np = None

# MX E2M1 magnitudes (2-bit exponent, 1-bit mantissa, bias 1)
E2M1_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def fp8_e4m3_decode(b):
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        v = m * (2.0 ** -9)
    elif e == 0xF:
        v = 448.0                      # inf/nan clamp to E4M3FN max finite
    else:
        v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -v if s else v


FP8_LUT = [fp8_e4m3_decode(i) for i in range(256)]
if _np is not None:
    FP8_LUT_NP = _np.array(FP8_LUT, _np.float32)


def e8m0_scale(maxv):
    """Block scale: smallest power of two s with maxv/s <= 6 (E2M1 max)."""
    if maxv <= 0.0:
        return 0
    k = max(-127, min(127, int(__import__("math").ceil(
        __import__("math").log2(maxv / 6.0)))))
    return (k + 127) & 0xFF


def e8m0_value(b):
    return (2.0 ** ((b & 0xFF) - 127)) if b else (2.0 ** -127)


def e2m1_encode(x):
    ax = abs(x)
    best, bd = 0, ax
    for i, m in enumerate(E2M1_MAGS):
        d = abs(ax - m)
        if d < bd:
            bd, best = d, i
    return (0x8 if x < 0 else 0) | best


def quantize_ndarray(x):
    """x: np.float32 1-D. Returns (values_bytes, scale_bytes, stats).
    Layout: 2 values per byte, even index in low nibble; one E8M0 scale
    per 32-element block."""
    n = x.shape[0]
    nblocks = (n + 31) // 32
    pad = nblocks * 32 - n
    if pad:
        x = _np.concatenate([x, _np.zeros(pad, _np.float32)])
    X = x.reshape(nblocks, 32)
    m = _np.max(_np.abs(X), axis=1)
    with _np.errstate(divide="ignore", invalid="ignore"):
        k = _np.ceil(_np.log2(_np.where(m > 0, m / 6.0, 1.0)))
    k = _np.clip(k, -127, 127).astype(_np.int64)
    sb = ((k + 127) & 0xFF).astype(_np.uint8)
    s = _np.power(2.0, k.astype(_np.float32))
    q = _np.divide(X, s[:, None], out=_np.zeros_like(X),
                   where=s[:, None] != 0)
    q = _np.clip(_np.rint(q), -6.0, 6.0)
    mags = _np.array(E2M1_MAGS, _np.float32)
    idx = _np.argmin(_np.abs(q[:, :, None] - mags[None, None, :]),
                     axis=2).astype(_np.uint8)
    sign = (q < 0).astype(_np.uint8)
    flat = ((sign << 3) | (idx & 7)).reshape(-1)
    if flat.size % 2:
        flat = _np.concatenate([flat, _np.zeros(1, _np.uint8)])
    packed = (flat[1::2] << 4) | flat[0::2]
    dq = _np.where(q < 0, -1.0, 1.0) * mags[idx] * s[:, None]
    err = _np.abs(dq - X)
    stats = {"err_max": float(_np.max(err)),
             "err_rms": float(_np.sqrt(_np.mean(err * err)))}
    return packed.tobytes(), sb.tobytes(), stats


def quantize_py(x):
    """Pure-python fallback (fixtures/tests; numpy is required for the
    real ~68 GB pool -- see warning in cmd_quantize)."""
    n = len(x)
    nblocks = (n + 31) // 32
    nvals = nblocks * 32
    out = bytearray((nvals + 1) // 2)
    scales = bytearray(nblocks)
    worst = 0.0
    sumsq = 0.0
    for bi in range(nblocks):
        block = x[bi * 32:(bi + 1) * 32]
        m = max((abs(v) for v in block), default=0.0)
        sb = e8m0_scale(m)
        scales[bi] = sb
        s = e8m0_value(sb)
        for j, v in enumerate(block):
            nib = e2m1_encode(v / s if s else 0.0)
            pos = bi * 32 + j
            if pos % 2 == 0:
                out[pos // 2] = nib
            else:
                out[pos // 2] |= nib << 4
            dq = (E2M1_MAGS[nib & 7] * (1 if nib < 8 else -1)) * s
            err = abs(dq - v)
            if err > worst:
                worst = err
            sumsq += err * err
    return bytes(out), bytes(scales), {"err_max": worst,
                                       "err_rms": (sumsq / n) ** 0.5}


def quantize(xs):
    if _np is not None:
        return quantize_ndarray(_np.asarray(xs, _np.float32))
    return quantize_py(xs)


def mxfp4_dequant(vbytes, sbytes, n):
    nblocks = (n + 31) // 32
    nvals = nblocks * 32
    vals = []
    for i in range(nvals):
        byte = vbytes[i // 2]
        nib = (byte >> (4 if i % 2 else 0)) & 0xF
        mag = E2M1_MAGS[nib & 7]
        sgn = -1.0 if (nib & 8) else 1.0
        vals.append(sgn * mag * e8m0_value(sbytes[i // 32]))
    return vals[:n]


def cmd_self_test():
    import random
    random.seed(7)
    worst = 0.0
    for trial in range(300):
        n = random.randint(1, 2000)
        scale = 10.0 ** random.uniform(-4, 4)
        xs = [random.uniform(-1, 1) * scale for _ in range(n)]
        vb, sb, st = quantize(xs)
        # determinism
        vb2, sb2, _ = quantize(xs)
        if vb != vb2 or sb != sb2:
            print(f"self-test FAIL trial {trial}: not deterministic")
            sys.exit(1)
        back = mxfp4_dequant(vb, sb, n)
        err = max(abs(a - b) for a, b in zip(xs, back))
        worst = max(worst, err)
        m = max((abs(v) for v in xs), default=0.0)
        if err > max(m * 0.5, 1e-30) + 1e-30:
            print(f"self-test FAIL trial {trial}: n={n} m={m} err={err}")
            sys.exit(1)
    print(f"self-test ok ({300} trials, worst abs err {worst:.6g})")


def apply_scale(vals, sdata, sdt, scheme, shp, n):
    """Multiply decoded values by their scale(s). vals is a numpy array
    (numpy path) or a list (pure-python path)."""
    if sdt == "F8_E8M0":
        if scheme == "tensor":
            s = e8m0_value(sdata[0])
            return vals * s
        if scheme == "row":
            R, C = int(shp[0]), n // int(shp[0])
            if _np is not None:
                sca = _np.array([e8m0_value(b) for b in sdata], _np.float32)
                return (vals.reshape(R, C) * sca[:, None]).reshape(-1)
            sc = [e8m0_value(b) for b in sdata]
            return [vals[r * C + c] * sc[r] for r in range(R)
                    for c in range(C)]
        sc = [e8m0_value(b) for b in sdata]
        if scheme == "block32":
            if _np is not None:
                sca = _np.repeat(_np.array(sc, _np.float32), 32)[:n]
                return vals * sca
            return [vals[i] * sc[i // 32] for i in range(n)]
        # block16
        if _np is not None:
            sca = _np.repeat(_np.array(sc, _np.float32), 16)[:n]
            return vals * sca
        return [vals[i] * sc[i // 16] for i in range(n)]
    # F32 scales
    if scheme == "tensor":
        s = struct.unpack("<f", sdata[:4])[0]
        return vals * s
    if scheme == "row":
        R, C = int(shp[0]), n // int(shp[0])
        if _np is not None:
            sca = _np.frombuffer(sdata, _np.float32)
            return (vals.reshape(R, C) * sca[:, None]).reshape(-1)
        sc = struct.unpack(f"<{R}f", sdata[:4 * R])
        return [vals[r * C + c] * sc[r] for r in range(R) for c in range(C)]
    # block32/block16 with F32 scales (unusual but supported)
    step = 32 if scheme == "block32" else 16
    nblocks = (n + step - 1) // step
    if _np is not None:
        sca = _np.repeat(_np.frombuffer(sdata, _np.float32), step)[:n]
        return vals * sca
    sc = struct.unpack(f"<{nblocks}f", sdata[:4 * nblocks])
    return [vals[i] * sc[i // step] for i in range(n)]


def cmd_quantize(dirpath, outdir, dry_run=False):
    root = find_repo_root(dirpath)
    if root is None:
        print(f"REFUSE: no repo-like content under {dirpath}")
        sys.exit(1)
    if root != dirpath:
        print(f"found repo-like content under: {root}")
        dirpath = root
    shards, _, config = discover(dirpath)
    mapped, _ = map_config(config)
    missing = [k for k in REQUIRED if not mapped.get(k)]
    if missing:
        print(f"REFUSE: config keys missing: {', '.join(missing)}")
        sys.exit(1)
    experts, _, _, _ = classify(shards)
    if not experts:
        print("REFUSE: no routed experts matched")
        sys.exit(1)
    layers = sorted({L for L, _ in experts})

    srcs = {}
    def read(name):
        fn, off, nb, dt, shp = shards[name]
        if fn not in srcs:
            srcs[fn] = open(fn, "rb")
        srcs[fn].seek(off)
        return srcs[fn].read(nb), dt, shp

    def numel(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n

    # resolve per-expert weight layout + scale scheme
    problems = []
    schemes = {}
    layouts = {}            # (L,e) -> [(name, shape, vnbytes, snbytes, blocks)]
    for (L, e) in sorted(experts):
        tens = sorted(experts[(L, e)])
        weights = [t for t in tens if t[0].endswith(".weight")]
        layout = []
        for t in weights:
            name, fn, off, nb, dt, shp = t
            sc_name = name[:-len(".weight")] + ".scale"
            sc = next((x for x in tens if x[0] == sc_name), None)
            if sc is None:
                problems.append(f"{name}: no sibling .scale tensor")
                continue
            if dt not in ("F8_E4M3", "I8"):
                problems.append(f"{name}: dtype {dt}, expected F8_E4M3 or I8")
                continue
            wdata, _, _ = read(name)
            sdata, sdt, sshp = read(sc_name)
            if sdt not in ("F32", "F8_E8M0"):
                problems.append(f"{sc_name}: dtype {sdt}, "
                                f"expected F32 or F8_E8M0")
                continue
            n = numel(shp)
            s_elems = len(sdata) // (4 if sdt == "F32" else 1)
            if s_elems == 1:
                scheme = "tensor"
            elif len(shp) == 2 and s_elems == int(shp[0]):
                scheme = "row"
            elif s_elems == (n + 31) // 32:
                scheme = "block32"
            elif s_elems == (n + 15) // 16:
                scheme = "block16"     # real checkpoint: one E8M0 per 16 elems
            else:
                problems.append(f"{sc_name}: {s_elems} scales for shape {shp}")
                continue
            schemes[name] = scheme
            layout.append((name, shp, (n + 1) // 2, (n + 31) // 32,
                           (n + 31) // 32, scheme, dt, sdt))
        layouts[(L, e)] = layout

    if dry_run:
        sample = next(iter(layouts.values()))
        print(f"expert weight tensors: {len(layouts)} experts, "
              f"{len(sample)} tensors each")
        for (name, shp, vnb, snb, blk, scheme, dt, sdt) in sample:
            print(f"  {name}: dtype {dt}, scale {sdt}, shape {shp}, "
                  f"scheme={scheme}, values {vnb} B, "
                  f"block scales {snb} B ({blk} blocks)")
        cnt = {}
        for s in schemes.values():
            cnt[s] = cnt.get(s, 0) + 1
        print(f"scale schemes: {cnt}")
        if problems:
            print(f"PROBLEMS ({len(problems)}):")
            for p in problems[:10]:
                print(f"  {p}")
        est = sum(vnb + snb for _, _, vnb, snb, _, _, _, _ in
                  next(iter(layouts.values()))) * len(layouts)
        print(f"estimated mxfp4 pool: {hsize(est)} "
              f"(blocks of 32, E8M0 scales, values 2/byte)")
        if problems:
            sys.exit(1)
        return

    if problems:
        print(f"REFUSE: {len(problems)} problems "
              f"(run --dry-run for the list)")
        sys.exit(1)
    if _np is None and len(layouts) > 100:
        print("WARNING: numpy not found; pure-python quantize will be "
              "very slow on the real pool. Install numpy on the box.")

    # expert slot size is computable from shapes alone (fixed-rate)
    slot = sum(vnb + snb for _, _, vnb, snb, _, _, _, _ in
               next(iter(layouts.values())))
    for (L, e), layout in layouts.items():
        s = sum(vnb + snb for _, _, vnb, snb, _, _, _, _ in layout)
        if s != slot:
            print(f"REFUSE: expert ({L},{e}) slot {s} != {slot}")
            sys.exit(1)

    os.makedirs(outdir, exist_ok=True)
    pbin = open(os.path.join(outdir, "pool-mxfp4.bin"), "wb")
    put_u64(pbin, slot)
    put_u64(pbin, len(layers))
    put_u64(pbin, mapped["n_experts"])

    written = 0
    done = 0
    total = len(layouts)
    tensor_meta = []
    g_max, g_rms2, g_n = 0.0, 0.0, 0
    for (L, e) in sorted(experts):
        layout = layouts[(L, e)]
        for (name, shp, vnb, snb, blk, scheme, dt, sdt) in layout:
            wdata, _, _ = read(name)
            sdata, _, _ = read(name[:-len(".weight")] + ".scale")
            n = numel(shp)
            if _np is not None:
                if dt == "I8":
                    vals = _np.frombuffer(wdata, _np.int8).astype(_np.float32)
                else:
                    raw = _np.frombuffer(wdata, _np.uint8)
                    vals = FP8_LUT_NP[raw].astype(_np.float32)
                vals = apply_scale(vals, sdata, sdt, scheme, shp, n)
                vb, sbb, st = quantize_ndarray(vals)
            else:
                if dt == "I8":
                    vals = [int.from_bytes(wdata[i:i + 1], "little",
                                           signed=True) for i in range(n)]
                else:
                    vals = [FP8_LUT[b] for b in wdata]
                vals = apply_scale(vals, sdata, sdt, scheme, shp, n)
                vb, sbb, st = quantize_py(vals)
            v_off = 24 + written
            pbin.write(vb)
            pbin.write(sbb)
            written += len(vb) + len(sbb)
            tensor_meta.append({
                "n": name, "layer": L, "expert": e, "shape": list(shp),
                "v_off": v_off, "v_nbytes": len(vb),
                "s_off": v_off + len(vb), "s_nbytes": len(sbb),
                "blocks": blk, "err_max": st["err_max"],
                "err_rms": st["err_rms"],
            })
            if st["err_max"] > g_max:
                g_max = st["err_max"]
            g_rms2 += (st["err_rms"] ** 2) * n
            g_n += n
        done += 1
        if done % 2000 == 0:
            print(f"  quantized {done}/{total} experts, {hsize(written)}")
    pbin.close()
    for f in srcs.values():
        f.close()

    meta = {
        "format": "mxfp4-pool-v1",
        "block_size": 32,
        "value_layout": "2 per byte, even index low nibble",
        "scale_format": "E8M0 per 32-element block",
        "n_layers": len(layers),
        "n_experts": mapped["n_experts"],
        "expert_nbytes": slot,
        "tensors": tensor_meta,
        "error_summary": {
            "tensor_max_abs": g_max,
            "global_rms": (g_rms2 / g_n) ** 0.5 if g_n else 0.0,
        },
    }
    with open(os.path.join(outdir, "pool-mxfp4.json"), "w") as f:
        json.dump(meta, f, indent=1)

    cfg_out = {
        "n_layers": mapped["n_layers"],
        "n_experts": mapped["n_experts"],
        "topk": mapped["topk"],
        "n_shared": mapped["n_shared"],
        "hidden": mapped["hidden"],
        "latent": mapped["latent"],
        "moe_inter": mapped["moe_inter"],
        "expert_nbytes": slot,
        "seed": 7,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)

    if os.path.getsize(os.path.join(outdir, "pool-mxfp4.bin")) != 24 + written:
        print("VERIFY FAIL: pool-mxfp4 size mismatch")
        sys.exit(1)
    print(f"pool-mxfp4.bin {hsize(written)} ({done} experts x "
          f"{hsize(slot)})")
    print(f"error summary: tensor max abs {g_max:.6g}, "
          f"global rms {(g_rms2 / g_n) ** 0.5 if g_n else 0.0:.6g}")
    print(f"config.json written (expert_nbytes={slot})")
    print("run:")
    print(f"  ./salt {outdir} --trunk TRUNK --offsets OFFSETS "
          f"--pool {outdir}/pool-mxfp4.bin")


def main():
    if len(sys.argv) < 3 and not (len(sys.argv) == 2 and
                                  sys.argv[1] == "self-test"):
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "inspect":
        cmd_inspect(sys.argv[2])
    elif cmd == "convert":
        if "--out" not in sys.argv:
            print(__doc__)
            sys.exit(1)
        cmd_convert(sys.argv[2], sys.argv[sys.argv.index("--out") + 1])
    elif cmd == "quantize":
        if "--out" not in sys.argv:
            print(__doc__)
            sys.exit(1)
        cmd_quantize(sys.argv[2], sys.argv[sys.argv.index("--out") + 1],
                     dry_run="--dry-run" in sys.argv)
    elif cmd == "self-test":
        cmd_self_test()
    elif cmd == "make-synthetic":
        if "--arch" in sys.argv:
            arch = sys.argv[sys.argv.index("--arch") + 1]
            if arch in ("qwen3", "qwen35", "qwen3_5"):
                cmd_make_synthetic_qwen35(sys.argv[2])
                sys.exit(0)
        cmd_make_synthetic(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
