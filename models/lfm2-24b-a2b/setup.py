#!/usr/bin/env python3
"""Pinned LFM2.5-24B-A2B (MLX affine) container materialization.

Stdlib only; never executes Hub code. No weight conversion, requantization or
runtime residency policy: every source payload byte is copied verbatim into the
canonical package payloads and ``index.txt`` supplies the existing C TensorOps
source descriptors.

Two affine tracks live in one container: ``group_size 64 / bits 4`` for every
projection, and ``bits 6`` for ``model.embed_tokens``. The 6-bit embedding is
also the tied output head, so it is bound once and referenced, never duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import time
import urllib.request

REPO = 'LiquidAI/LFM2.5-24B-A2B-MLX-4bit'
REVISION = '667408d494bde191c2cb9dd4f4549827816b9275'
ABI = 'salt-lfm25-affine4-6-f32-conv-v1'
SCHEMA = 'salt.lfm25.package.v1'

SHARDS = tuple(f'model-{i:05d}-of-00003.safetensors' for i in (1, 2, 3))
EXPECTED = {
    SHARDS[0]: (5318453801, '17e953456c62314a498f1ef5ee8a5f1b51ac83f6b6bb684f160dde4f92cd11c0'),
    SHARDS[1]: (5344895125, 'c3ae09dba31b3786c325425435021cf5d5bf06c1a79acb31a741e15a47e5bb14'),
    SHARDS[2]: (2903945409, '9814144af5045ea8c96e2f1352148fdbe9f8a0085b76df6d6760efbe916e9840'),
    'config.json': (2148, '4901707c3cef43e3cf68a0dedc3e39687de43413cab30578a32c63c643984244'),
    'generation_config.json': (329, 'd1ecad152cd278e57220364423f5a5b19350f1e6e230ff82281a5c23a3173f16'),
    'model.safetensors.index.json': (80192, '7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a'),
    'tokenizer.json': (17905598, '695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41'),
    'tokenizer_config.json': (363, '11f1de897317b489dd09199284528382eeb17f566398b16e2039bb2266d26b09'),
    'chat_template.jinja': (5496, 'aadd425edb1d42fe9288e5de3624c21c1c03099129b9cfea5978f1bc67222c0a'),
    'README.md': (3688, '28b1d52bd0bf07c8f1c69e0c1b948c95fbe677884b34e0da716bd3d8cb222f8b'),
    'LICENSE': (10574, '4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c'),
}

# ---- model geometry (Lfm2MoeForCausalLM, 24B-A2B) ------------------------
HIDDEN = 2048
VOCAB = 128000
LAYERS = 40
EXPERTS = 64
TOP_K = 4
MOE_INTERMEDIATE = 1536
DENSE_INTERMEDIATE = 11776
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
CONV_L = 3
DENSE_LAYERS = 2
GROUP = 64
EMBED_BITS = 6
PROJ_BITS = 4

# Encodings as recorded in index.txt. These are the engine's own
# SaltTensorEncoding values (include/salt/tensorops.h) so the package index is
# directly loadable by the model adapter; ENC_AFFINE_Q6 is the one value the
# engine does not have yet and must add.
ENC_F32 = 1        # SALT_TENSOR_ENCODING_F32: raw F32 vector (router, expert bias)
ENC_BF16 = 2       # SALT_TENSOR_ENCODING_BF16: raw BF16 vector (norms, conv kernel)
ENC_AFFINE_Q4 = 3  # SALT_TENSOR_ENCODING_AFFINE_Q4: MLX affine group 64 / 4-bit
ENC_AFFINE_Q6 = 7  # not yet in the engine: MLX affine group 64 / 6-bit embedding

CHUNK = 1024 * 1024
POOL_MAGIC = b'LFMPOOL1'
POOL_ALIGN = 4096


def is_attention_layer(layer):
    """config.layer_types is 'full_attention' exactly at index % 4 == 2."""
    return layer % 4 == 2


def affine_packed_bytes(rows, cols, bits):
    if cols % GROUP:
        raise ValueError('non-grouped affine matrix')
    return rows * (cols * bits // 8)


def affine_group_count(cols):
    return cols // GROUP


# ---- expert slot ---------------------------------------------------------
# Fixed order, no padding: gate w/s/b, up w/s/b, down w/s/b.
def expert_components():
    """(role, rows, cols) for the three expert projections."""
    return (('expert_gate', MOE_INTERMEDIATE, HIDDEN),
            ('expert_up', MOE_INTERMEDIATE, HIDDEN),
            ('expert_down', HIDDEN, MOE_INTERMEDIATE))


def expert_slot_bytes():
    total = 0
    for _, rows, cols in expert_components():
        total += affine_packed_bytes(rows, cols, PROJ_BITS)
        total += rows * affine_group_count(cols) * 2  # BF16 scales
        total += rows * affine_group_count(cols) * 2  # BF16 biases
    return total


EXPERT_BYTES = expert_slot_bytes()
MOE_LAYERS = tuple(l for l in range(LAYERS) if l >= DENSE_LAYERS)
EXPERT_COUNT = len(MOE_LAYERS) * EXPERTS


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f'not a regular file: {path}')
    return {'dev': s.st_dev, 'ino': s.st_ino, 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


# ---- fetch ---------------------------------------------------------------
def fetch(source):
    source.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        info = json.load(response)
    if info['sha'] != REVISION:
        raise ValueError('source revision mismatch')
    published = {x['rfilename']: x for x in info['siblings']}
    for name, (size, sha) in EXPECTED.items():
        item = published.get(name)
        if item is None:
            raise ValueError(f'pinned source file absent from revision: {name}')
        if item['size'] != size:
            raise ValueError(f'pinned size mismatch: {name}')
        lfs = (item.get('lfs') or {}).get('sha256')
        if sha and lfs and lfs != sha:
            raise ValueError(f'pinned lfs digest mismatch: {name}')
    missing = sum(size for name, (size, _) in EXPECTED.items()
                  if not (source / name).exists())
    required = missing + EXPECTED[SHARDS[0]][0] + EXPECTED[SHARDS[1]][0] + EXPECTED[SHARDS[2]][0] + 2_000_000_000
    free = shutil.disk_usage(source).free
    print(f'DISK free_bytes={free} additional_budget_bytes={required}', flush=True)
    if free < required:
        raise ValueError('insufficient source + import + reserve disk budget')
    for name, (size, sha) in EXPECTED.items():
        dest = source / name
        if dest.exists():
            identity(dest)
            if dest.stat().st_size == size and (not sha or digest(dest) == sha):
                print('SOURCE_VERIFIED', name, flush=True)
                continue
            raise ValueError(f'existing source mismatch; retained without overwrite: {dest}')
        partial = source / (name + '.partial')
        if partial.exists():
            raise ValueError(f'incomplete prior download retained: {partial}')
        print('DOWNLOAD', name, size, flush=True)
        h = hashlib.sha256()
        count = 0
        last = time.monotonic()
        with urllib.request.urlopen(
                f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=180) as response, \
                partial.open('xb') as out:
            while block := response.read(CHUNK):
                count += len(block)
                if count > size:
                    raise ValueError('source exceeded declared size')
                h.update(block)
                out.write(block)
                if time.monotonic() - last >= 15:
                    print('DOWNLOAD_PROGRESS', name, count, size, flush=True)
                    last = time.monotonic()
            out.flush()
            os.fsync(out.fileno())
        if count != size or (sha and h.hexdigest() != sha):
            raise ValueError(f'source digest/size mismatch: {name}')
        os.replace(partial, dest)
        print('DOWNLOAD_DONE', name, flush=True)
    (source / 'source-receipt.json').write_text(json.dumps({
        'repo': REPO, 'revision': REVISION,
        'files': {name: {'identity': identity(source / name), 'sha256': digest(source / name)}
                  for name in EXPECTED}}, indent=2) + '\n')


# ---- source header admission --------------------------------------------
def read_headers(source):
    """Read every shard header; validate contiguity, extent and dtype."""
    tensors = {}
    for path in sorted(source.glob('model-*-of-00003.safetensors')):
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('short shard')
            n, = struct.unpack('<Q', raw)
            if not 2 <= n <= 16 * CHUNK:
                raise ValueError('header extent')
            header = parse_json(f.read(n))
        entries = {k: v for k, v in header.items() if k != '__metadata__'}
        end = 0
        for name, entry in sorted(entries.items(), key=lambda x: x[1]['data_offsets'][0]):
            if name in tensors:
                raise ValueError(f'duplicate tensor: {name}')
            lo, hi = entry['data_offsets']
            if lo != end or hi <= lo:
                raise ValueError(f'overlap/gap/invalid tensor span: {name}')
            width = {'U8': 1, 'BF16': 2, 'F32': 4, 'U32': 4}[entry['dtype']]
            expected = math.prod(entry['shape']) * width
            if expected != hi - lo:
                raise ValueError(f'tensor size mismatch: {name}')
            tensors[name] = (path, n + 8 + lo, hi - lo, entry['dtype'], entry['shape'])
            end = hi
        if n + 8 + end != path.stat().st_size:
            raise ValueError(f'shard extent mismatch: {path.name}')
    return tensors


def expect(tensors, name, dtype, shape, consumed):
    if name not in tensors:
        raise ValueError(f'missing source tensor: {name}')
    _, _, _, actual_dtype, actual_shape = tensors[name]
    if actual_dtype != dtype or list(actual_shape) != list(shape):
        raise ValueError(f'geometry mismatch: {name} {actual_dtype}{actual_shape} != {dtype}{shape}')
    consumed.add(name)
    return tensors[name]


def expect_affine_fused(tensors, base, rows, cols, bits, experts, consumed):
    """Validate one MLX affine matrix fused over all experts.

    The container stores every expert of a projection in a single contiguous
    ``[experts, rows, packed]`` tensor, so the importer slices one expert at a
    time; the returned extents are the whole fused tensor.
    """
    if cols % GROUP:
        raise ValueError(f'non-grouped affine matrix: {base}')
    packed = cols * bits // 8 // 4  # per-row U32 words
    groups = affine_group_count(cols)
    w = expect(tensors, base + '.weight', 'U32', [experts, rows, packed], consumed)
    s = expect(tensors, base + '.scales', 'BF16', [experts, rows, groups], consumed)
    b = expect(tensors, base + '.biases', 'BF16', [experts, rows, groups], consumed)
    return w, s, b


def expect_affine(tensors, base, rows, cols, bits, consumed):
    """Validate one MLX affine matrix and return its three components."""
    if cols % GROUP:
        raise ValueError(f'non-grouped affine matrix: {base}')
    packed = cols * bits // 8 // 4  # per-row U32 words
    groups = affine_group_count(cols)
    w = expect(tensors, base + '.weight', 'U32', [rows, packed], consumed)
    s = expect(tensors, base + '.scales', 'BF16', [rows, groups], consumed)
    b = expect(tensors, base + '.biases', 'BF16', [rows, groups], consumed)
    if len({w[0], s[0], b[0]}) != 1:
        raise ValueError(f'affine components span resources: {base}')
    return w, s, b


# ---- trunk spec ----------------------------------------------------------
def trunk_spec(tensors, consumed):
    """Yield (role, layer, encoding, rows, cols, weight, scales, biases)."""
    root = 'model.'
    yield ('embedding', -1, ENC_AFFINE_Q6, VOCAB, HIDDEN,
           *expect_affine(tensors, root + 'embed_tokens', VOCAB, HIDDEN, EMBED_BITS, consumed))
    norm = expect(tensors, root + 'embedding_norm.weight', 'BF16', [HIDDEN], consumed)
    yield ('final_norm', -1, ENC_BF16, 1, HIDDEN, norm, None, None)

    for layer in range(LAYERS):
        prefix = f'{root}layers.{layer}.'
        op = expect(tensors, prefix + 'operator_norm.weight', 'BF16', [HIDDEN], consumed)
        yield ('operator_norm', layer, ENC_BF16, 1, HIDDEN, op, None, None)
        ffn = expect(tensors, prefix + 'ffn_norm.weight', 'BF16', [HIDDEN], consumed)
        yield ('ffn_norm', layer, ENC_BF16, 1, HIDDEN, ffn, None, None)

        if is_attention_layer(layer):
            attn = prefix + 'self_attn.'
            for role, part, rows, cols in (('q', 'q_proj', NUM_HEADS * HEAD_DIM, HIDDEN),
                                           ('k', 'k_proj', NUM_KV_HEADS * HEAD_DIM, HIDDEN),
                                           ('v', 'v_proj', NUM_KV_HEADS * HEAD_DIM, HIDDEN),
                                           ('o', 'out_proj', HIDDEN, NUM_HEADS * HEAD_DIM)):
                yield (role, layer, ENC_AFFINE_Q4, rows, cols,
                       *expect_affine(tensors, attn + part, rows, cols, PROJ_BITS, consumed))
            for role in ('q_norm', 'k_norm'):
                value = expect(tensors, attn + role.replace('_norm', '_layernorm') + '.weight',
                               'BF16', [HEAD_DIM], consumed)
                yield (role, layer, ENC_BF16, 1, HEAD_DIM, value, None, None)
        else:
            conv = prefix + 'conv.'
            for role, part, rows, cols in (('conv_in', 'in_proj', 3 * HIDDEN, HIDDEN),
                                           ('conv_out', 'out_proj', HIDDEN, HIDDEN)):
                yield (role, layer, ENC_AFFINE_Q4, rows, cols,
                       *expect_affine(tensors, conv + part, rows, cols, PROJ_BITS, consumed))
            kernel = expect(tensors, conv + 'conv.weight', 'BF16', [HIDDEN, CONV_L, 1], consumed)
            yield ('conv_kernel', layer, ENC_BF16, HIDDEN, CONV_L, kernel, None, None)

        ff = prefix + 'feed_forward.'
        if layer < DENSE_LAYERS:
            for role, part, rows, cols in (('dense_gate', 'gate_proj', DENSE_INTERMEDIATE, HIDDEN),
                                           ('dense_up', 'up_proj', DENSE_INTERMEDIATE, HIDDEN),
                                           ('dense_down', 'down_proj', HIDDEN, DENSE_INTERMEDIATE)):
                yield (role, layer, ENC_AFFINE_Q4, rows, cols,
                       *expect_affine(tensors, ff + part, rows, cols, PROJ_BITS, consumed))
        else:
            router = expect(tensors, ff + 'gate.weight', 'F32', [EXPERTS, HIDDEN], consumed)
            yield ('router', layer, ENC_F32, EXPERTS, HIDDEN, router, None, None)
            bias = expect(tensors, ff + 'expert_bias', 'F32', [EXPERTS], consumed)
            yield ('expert_bias', layer, ENC_F32, 1, EXPERTS, bias, None, None)


def expert_spec(tensors, layer, consumed):
    """Yield (role, rows, cols, weight, scales, biases) for one MoE layer."""
    ff = f'model.layers.{layer}.feed_forward.switch_mlp.'
    for role, rows, cols in expert_components():
        part = {'expert_gate': 'gate_proj', 'expert_up': 'up_proj', 'expert_down': 'down_proj'}[role]
        yield (role, rows, cols,
               *expect_affine_fused(tensors, ff + part, rows, cols, PROJ_BITS, EXPERTS, consumed))


# ---- config admission ----------------------------------------------------
def validate_config(config):
    exact = {'architectures': ['Lfm2MoeForCausalLM'], 'model_type': 'lfm2_moe',
             'hidden_size': HIDDEN, 'num_hidden_layers': LAYERS,
             'num_experts': EXPERTS, 'num_experts_per_tok': TOP_K,
             'moe_intermediate_size': MOE_INTERMEDIATE,
             'intermediate_size': DENSE_INTERMEDIATE,
             'num_attention_heads': NUM_HEADS, 'num_key_value_heads': NUM_KV_HEADS,
             'num_dense_layers': DENSE_LAYERS, 'conv_L_cache': CONV_L,
             'conv_bias': False, 'vocab_size': VOCAB, 'tie_word_embeddings': True,
             'use_expert_bias': True, 'norm_topk_prob': True,
             'routed_scaling_factor': 1.0, 'norm_eps': 1e-05,
             'max_position_embeddings': 131072}
    for key, value in exact.items():
        if config.get(key) != value:
            raise ValueError(f'unsupported config: {key}={config.get(key)!r} != {value!r}')
    schedule = ['full_attention' if is_attention_layer(i) else 'conv' for i in range(LAYERS)]
    if config.get('layer_types') != schedule:
        raise ValueError('layer schedule mismatch')
    rope = config.get('rope_parameters') or {}
    if rope.get('rope_theta') != 10000000.0 or rope.get('rope_type') != 'default':
        raise ValueError('unsupported rope parameters')
    quant = config.get('quantization') or {}
    if quant.get('group_size') != GROUP or quant.get('bits') != PROJ_BITS or quant.get('mode') != 'affine':
        raise ValueError('wrong quantization')
    embed = quant.get('model.embed_tokens') or {}
    if embed.get('bits') != EMBED_BITS or embed.get('group_size') != GROUP or embed.get('mode') != 'affine':
        raise ValueError('wrong embedding quantization')


# ---- materialize ---------------------------------------------------------
def atomic_write(path, data):
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize(source, package):
    receipt = parse_json((source / 'source-receipt.json').read_bytes())
    if receipt['repo'] != REPO or receipt['revision'] != REVISION:
        raise ValueError('source provenance mismatch')
    for name, row in receipt['files'].items():
        if Path(name).name != name or identity(source / name) != row['identity']:
            raise ValueError(f'source changed since authentication: {name}')
    validate_config(parse_json((source / 'config.json').read_bytes()))
    tensors = read_headers(source)
    if package.exists():
        raise ValueError('package already exists; never overwrite an authenticated package')
    package.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(package.parent).free < EXPERT_COUNT * EXPERT_BYTES + 4_000_000_000:
        raise ValueError('insufficient import + reserve space')
    package.mkdir()

    layer_base = {layer: (-1 if layer < DENSE_LAYERS else MOE_LAYERS.index(layer) * EXPERTS)
                  for layer in range(LAYERS)}
    header_bytes = POOL_ALIGN
    pool_extent = header_bytes + EXPERT_COUNT * EXPERT_BYTES
    consumed = set()
    trunk_lines, expert_lines, layer_lines = [], [], []

    handles = {p: p.open('rb') for p in {entry[0] for entry in tensors.values()}}

    def copy_span(path, offset, size, output, label):
        f = handles[path]
        f.seek(offset)
        left = size
        while left:
            block = f.read(min(left, CHUNK))
            if not block:
                raise ValueError(f'short source read: {label}')
            output.write(block)
            left -= len(block)

    try:
        with (package / 'trunk.bin').open('xb') as trunk, (package / 'pool.bin').open('xb') as pool:
            pool.write(b'\0' * header_bytes)
            for role, layer, enc, rows, cols, value, scales, biases in trunk_spec(tensors, consumed):
                trunk.write(b'\0' * ((-trunk.tell()) % 64))
                value_off = trunk.tell()
                copy_span(value[0], value[1], value[2], trunk, role)
                scale_off = scale_bytes = bias_off = bias_bytes = 0
                if scales is not None:
                    scale_off = trunk.tell()
                    copy_span(scales[0], scales[1], scales[2], trunk, role + '.scales')
                    scale_bytes = scales[2]
                if biases is not None:
                    bias_off = trunk.tell()
                    copy_span(biases[0], biases[1], biases[2], trunk, role + '.biases')
                    bias_bytes = biases[2]
                trunk_lines.append(
                    f'T {role} {layer} {enc} {rows} {cols} {value_off} {value[2]} '
                    f'{scale_off} {scale_bytes} {bias_off} {bias_bytes}\n')
            trunk_bytes = trunk.tell()

            for layer in MOE_LAYERS:
                base = header_bytes + layer_base[layer] * EXPERT_BYTES
                parts = list(expert_spec(tensors, layer, consumed))
                for expert in range(EXPERTS):
                    slot = base + expert * EXPERT_BYTES
                    pool.seek(slot)
                    written = 0
                    for role, rows, cols, value, scales, biases in parts:
                        per_value = value[2] // EXPERTS
                        per_scale = scales[2] // EXPERTS
                        per_bias = biases[2] // EXPERTS
                        if per_value * EXPERTS != value[2] or per_scale * EXPERTS != scales[2] \
                                or per_bias * EXPERTS != biases[2]:
                            raise ValueError(f'fused expert tensor not divisible: {role}')
                        value_off = slot + written
                        copy_span(value[0], value[1] + expert * per_value, per_value, pool, role)
                        written += per_value
                        scale_off = slot + written
                        copy_span(scales[0], scales[1] + expert * per_scale, per_scale, pool, role)
                        written += per_scale
                        bias_off = slot + written
                        copy_span(biases[0], biases[1] + expert * per_bias, per_bias, pool, role)
                        written += per_bias
                        expert_lines.append(
                            f'E {role} {layer} {expert} {ENC_AFFINE_Q4} {rows} {cols} '
                            f'{value_off} {per_value} {scale_off} {per_scale} {bias_off} {per_bias}\n')
                    if written != EXPERT_BYTES:
                        raise ValueError(f'expert slot extent mismatch: layer {layer} expert {expert}')
                print('IMPORT_LAYER', layer, MOE_LAYERS.index(layer) + 1, len(MOE_LAYERS), flush=True)

            for layer in range(LAYERS):
                layer_lines.append(f'L {layer} {layer_base[layer]}\n')

            if pool.tell() != pool_extent:
                raise ValueError('expert pool extent mismatch')
            pool.seek(0)
            pool.write(POOL_MAGIC + b'\0' * (16 - len(POOL_MAGIC)))
            pool.write(struct.pack('<QQQQQ', EXPERT_BYTES, LAYERS, EXPERTS,
                                   len(MOE_LAYERS), EXPERT_COUNT))
            pool.write(struct.pack(f'<{LAYERS}q', *[layer_base[l] for l in range(LAYERS)]))

        expected = {name for name in tensors if name.startswith('model.')}
        if consumed != expected:
            raise ValueError(f'tensor coverage mismatch: missing={sorted(expected - consumed)} '
                             f'unexpected={sorted(consumed - expected)}')

        index = [f'LFMIDX1 {ABI} {EXPERT_BYTES} {LAYERS} {EXPERTS} {len(MOE_LAYERS)} '
                 f'{EXPERT_COUNT} {trunk_bytes} {pool_extent}\n']
        index += layer_lines + trunk_lines
        # The tied head is the embedding table; bind the same bytes, copy nothing.
        index.append(f'T output_head_tied -1 {ENC_AFFINE_Q6} {VOCAB} {HIDDEN} '
                     + ' '.join(trunk_lines[0].split()[6:]) + '\n')
        index += expert_lines
        encoded_index = ''.join(index).encode()
        atomic_write(package / 'index.txt', encoded_index)
        for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json',
                     'chat_template.jinja', 'generation_config.json'):
            shutil.copyfile(source / name, package / name)
        records = {name: {'identity': identity(package / name), 'sha256': digest(package / name)}
                   for name in ('trunk.bin', 'pool.bin', 'index.txt', 'tokenizer.json',
                                'tokenizer_config.json', 'config.json', 'chat_template.jinja',
                                'generation_config.json')}
        package_id = hashlib.sha256(json.dumps(
            {'abi': ABI, 'revision': REVISION,
             'sha256': {k: v['sha256'] for k, v in records.items()}},
            sort_keys=True).encode()).hexdigest()
        manifest = {'schema': SCHEMA, 'abi': ABI, 'model': 'lfm2-24b-a2b',
                    'source_repo': REPO, 'source_revision': REVISION,
                    'package_id': package_id, 'files': records,
                    'source_tensors': len(tensors), 'bound_tensors': len(consumed),
                    'expert_bytes': EXPERT_BYTES, 'expert_count': EXPERT_COUNT,
                    'moe_layers': list(MOE_LAYERS), 'trunk_bytes': trunk_bytes,
                    'pool_bytes': pool_extent, 'payload_copied': False,
                    'requantized': False, 'tied_output_head': True}
        atomic_write(package / 'auth.json', (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
        if parse_json((package / 'auth.json').read_bytes()) != manifest:
            raise ValueError('receipt readback mismatch')
        print(f'PACKAGE_READY {package} {package_id} experts={EXPERT_COUNT} '
              f'trunk_bytes={trunk_bytes} pool_bytes={pool_extent}', flush=True)
    finally:
        for f in handles.values():
            f.close()


# ---- runtime tokenizer ---------------------------------------------------
def compact_tokenizer(document):
    """Lossless representation change supported by the unchanged C loader."""
    result = json.loads(json.dumps(document))
    pairs = document['model']['merges']
    if not all(isinstance(pair, list) and len(pair) == 2 and
               all(isinstance(token, str) and token and ' ' not in token for token in pair)
               for pair in pairs):
        raise ValueError('compact tokenizer requires unambiguous array-form merge pairs')
    result['model']['merges'] = [' '.join(pair) for pair in pairs]
    restored = json.loads(json.dumps(result))
    restored['model']['merges'] = [value.split(' ') for value in result['model']['merges']]
    if restored != document:
        raise ValueError('tokenizer semantic round trip failed')
    return result


def prepare_runtime_tokenizer(package):
    manifest = parse_json((package / 'auth.json').read_bytes())
    source = package / 'tokenizer.json'
    authority = manifest['files']['tokenizer.json']
    if identity(source) != authority['identity'] or digest(source) != authority['sha256']:
        raise ValueError('source tokenizer drift')
    document = parse_json(source.read_bytes())
    compact = compact_tokenizer(document)
    encoded = (json.dumps(compact, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    path = package / 'tokenizer.runtime.json'
    receipt_path = package / 'tokenizer.runtime.auth.json'
    if path.exists() or receipt_path.exists():
        raise ValueError('runtime tokenizer already exists; refusing overwrite')
    with path.open('xb') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    receipt = {'schema': 'salt.lfm25.tokenizer-layout.v1', 'layout': 'compact-merges-v1',
               'package_id': manifest['package_id'], 'source_sha256': authority['sha256'],
               'runtime_sha256': hashlib.sha256(encoded).hexdigest(), 'identity': identity(path),
               'merges': len(document['model']['merges']),
               'vocabulary': len(document['model']['vocab']),
               'added_tokens': len(document['added_tokens']),
               'bos_token_id': 124894, 'eos_token_id': 124900, 'pad_token_id': 124893,
               'complete_document_round_trip': True}
    pending = receipt_path.with_suffix('.pending')
    with pending.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, receipt_path)
    print('TOKENIZER_LAYOUT_READY', json.dumps(receipt, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch', 'import', 'prepare-tokenizer'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--package', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare-tokenizer':
        if args.package is None:
            parser.error('--package is required')
        prepare_runtime_tokenizer(args.package)
    elif args.action == 'fetch':
        if args.source is None:
            parser.error('--source is required')
        fetch(args.source)
    else:
        if args.package is None or args.source is None:
            parser.error('--source and --package are required')
        materialize(args.source, args.package)


if __name__ == '__main__':
    main()
