#!/usr/bin/env python3
"""Pinned Maple container materialization. Stdlib only; never executes Hub code."""
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

REPO = 'Kanposer/maple-preview-speedy-colibri-int2'
REVISION = 'ee9a423cf56e9206f37bb697c0a7e91dacd532eb'
EXPERT_BYTES = 798720
CHUNK = 1024 * 1024
ABI = 'salt-maple-int2-f32-v1'


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


def fetch(source):
    source.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        info = json.load(response)
    if info['sha'] != REVISION:
        raise ValueError('source revision mismatch')
    selected = [x for x in info['siblings'] if x['rfilename'].endswith('.safetensors') or
                x['rfilename'] in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'README.md')]
    missing = sum(x['size'] for x in selected if not (source/x['rfilename']).exists())
    required = missing + 6_300_000_000 + 2_000_000_000
    free = shutil.disk_usage(source).free
    print(f'DISK free_bytes={free} additional_budget_bytes={required}', flush=True)
    if free < required:
        raise ValueError('insufficient source + import + reserve disk budget')
    for item in selected:
        name = item['rfilename']
        if Path(name).name != name:
            raise ValueError('unexpected source path')
        dest = source/name
        expected = item.get('lfs', {}).get('sha256')
        if dest.exists():
            identity(dest)
            if dest.stat().st_size == item['size'] and (not expected or digest(dest) == expected):
                print('SOURCE_VERIFIED', name, flush=True)
                continue
            raise ValueError(f'existing source mismatch; retained without overwrite: {dest}')
        partial = source/(name+'.partial')
        if partial.exists():
            raise ValueError(f'incomplete prior download retained: {partial}')
        print('DOWNLOAD', name, item['size'], flush=True)
        h = hashlib.sha256()
        count = 0
        last = time.monotonic()
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=90) as response, partial.open('xb') as out:
            while block := response.read(CHUNK):
                count += len(block)
                if count > item['size']:
                    raise ValueError('source exceeded declared size')
                h.update(block)
                out.write(block)
                if time.monotonic()-last >= 10:
                    print('DOWNLOAD_PROGRESS', name, count, item['size'], flush=True)
                    last = time.monotonic()
            out.flush()
            os.fsync(out.fileno())
        if count != item['size'] or (expected and h.hexdigest() != expected):
            raise ValueError(f'source digest/size mismatch: {name}')
        os.replace(partial, dest)
    (source/'source-receipt.json').write_text(json.dumps({
        'repo': REPO, 'revision': REVISION,
        'files': {x['rfilename']: {'identity': identity(source/x['rfilename']),
                  'sha256': digest(source/x['rfilename'])} for x in selected}}, indent=2)+'\n')


def spec():
    yield 'model.embed_tokens.weight', 2, 151936, 2048, -1, -1
    yield 'lm_head.weight', 2, 151936, 2048, -1, -1
    yield 'model.norm.weight', 1, 1, 2048, -1, -1
    for layer in range(24):
        p = f'model.layers.{layer}.'
        for name, width in [('input_layernorm',2048), ('post_attention_layernorm',2048),
                            ('self_attn.q_norm',128), ('self_attn.k_norm',128)]:
            yield p+name+'.weight', 1, 1, width, layer, -1
        yield p+'mlp.gate.weight', 1, 256, 2048, layer, -1
        for name, rows in [('q',2048),('k',512),('v',512),('o',2048)]:
            yield p+f'self_attn.{name}_proj.weight', 6, rows, 2048, layer, -1
        for expert in range(256):
            for name, rows, cols in [('gate',512,2048),('up',512,2048),('down',2048,512)]:
                yield p+f'mlp.experts.{expert}.{name}_proj.weight', 6, rows, cols, layer, expert


def read_headers(source):
    tensors = {}
    for path in sorted(source.glob('out-*.safetensors')):
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('short shard')
            n, = struct.unpack('<Q', raw)
            if not 2 <= n <= 16*CHUNK:
                raise ValueError('header extent')
            header = json.loads(f.read(n), object_pairs_hook=unique_object)
        end = 0
        for name, entry in sorted(header.items(), key=lambda x:x[1]['data_offsets'][0]):
            if name in tensors:
                raise ValueError(f'duplicate tensor: {name}')
            lo, hi = entry['data_offsets']
            if lo != end or hi <= lo:
                raise ValueError('overlap/gap/invalid tensor span')
            expected = math.prod(entry['shape']) * {'U8':1, 'BF16':2, 'F32':4}[entry['dtype']]
            if expected != hi-lo:
                raise ValueError('tensor size mismatch')
            tensors[name] = (path, n+8+lo, hi-lo, entry)
            end = hi
        if n+8+end != path.stat().st_size:
            raise ValueError('shard extent mismatch')
    expected_names = set()
    for name, encoding, rows, cols, _, _ in spec():
        expected_names.add(name)
        if encoding == 6:
            expected_names.add(name+'.qs')
        dtype = {1:'F32',2:'BF16',6:'U8'}[encoding]
        shape = [rows*((cols+3)//4)] if encoding == 6 else [rows,cols]
        actual = tensors[name][3]
        # Normalization vectors use one-dimensional source storage.
        if actual['dtype'] != dtype or actual['shape'] not in (shape, [cols] if rows == 1 else shape):
            raise ValueError(f'geometry mismatch: {name}')
        if encoding == 6 and (tensors[name+'.qs'][3]['dtype'] != 'F32' or
                              tensors[name+'.qs'][3]['shape'] != [rows]):
            raise ValueError(f'row scale mismatch: {name}')
    if set(tensors) != expected_names:
        raise ValueError('missing or unclassified source tensors')
    return tensors


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def validate_config(config):
    exact = {'architectures':['MapleForCausalLM'], 'num_hidden_layers':24,
             'hidden_size':2048, 'num_experts':256, 'num_experts_per_tok':8,
             'num_shared_experts':0, 'moe_intermediate_size':512,
             'num_attention_heads':16, 'num_key_value_heads':4, 'head_dim':128,
             'vocab_size':151936, 'sliding_window':512, 'partial_rotary_factor':0.5,
             'rope_theta':10000, 'nope_on_global_attention':True,
             'use_qk_norm':True, 'norm_topk_prob':True, 'tie_word_embeddings':False,
             'rms_norm_eps':1e-6, 'eos_token_id':151645}
    for key, value in exact.items():
        if config.get(key) != value:
            raise ValueError(f'unsupported config: {key}')
    if config['layer_types'] != ['full_attention' if i%4 == 3 else 'sliding_attention' for i in range(24)]:
        raise ValueError('attention layer schedule mismatch')


def materialize(source, package):
    receipt = json.loads((source/'source-receipt.json').read_text())
    if receipt['repo'] != REPO or receipt['revision'] != REVISION:
        raise ValueError('source provenance mismatch')
    for name, row in receipt['files'].items():
        if Path(name).name != name or identity(source/name) != row['identity']:
            raise ValueError(f'source changed since authentication: {name}')
    validate_config(json.loads((source/'config.json').read_text()))
    tensors = read_headers(source)
    if package.exists():
        raise ValueError('package already exists; never overwrite an authenticated package')
    package.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(package.parent).free < 8_300_000_000:
        raise ValueError('insufficient import + reserve space')
    package.mkdir()
    descriptors = []
    handles = {p:p.open('rb') for p in {entry[0] for entry in tensors.values()}}
    invalid_codes = bytes(int(any(((b >> s)&3) == 0 for s in (0,2,4,6))) for b in range(256))
    def copy_tensor(name, output, ternary=False):
        path, off, size, _ = tensors[name]
        f = handles[path]
        f.seek(off)
        left = size
        while left:
            block = f.read(min(left,CHUNK))
            if not block:
                raise ValueError('short source read')
            if ternary and b'\x01' in block.translate(invalid_codes):
                raise ValueError(f'nonternary code in {name}')
            if name.endswith('.qs'):
                if any(not math.isfinite(v) or v < 0 for v, in struct.iter_unpack('<f',block)):
                    raise ValueError('invalid row scale')
            output.write(block)
            left -= len(block)
        return size
    try:
        with (package/'trunk.bin').open('xb') as trunk, (package/'pool.bin').open('xb') as pool:
            pool.write(struct.pack('<QQQ',EXPERT_BYTES,24,256))
            for name, enc, rows, cols, layer, expert in spec():
                output = pool if expert >= 0 else trunk
                if expert < 0:
                    output.write(b'\0'*((-output.tell())%64))
                base = 24+(layer*256+expert)*EXPERT_BYTES if expert >= 0 else 0
                value_off = output.tell()-base
                value_bytes = copy_tensor(name, output, enc == 6)
                scale_off = output.tell()-base if enc == 6 else 0
                scale_bytes = copy_tensor(name+'.qs',output) if enc == 6 else 0
                descriptors.append(f'{name} {enc} {rows} {cols} {layer} {expert} {value_off} {scale_off} {value_bytes} {scale_bytes}\n')
                if expert == 255 and name.endswith('down_proj.weight'):
                    print('IMPORT_LAYER',layer+1,24,flush=True)
            if pool.tell() != 24+24*256*EXPERT_BYTES:
                raise ValueError('expert pool extent mismatch')
        (package/'index.txt').write_text('MAPLEIDX1\n'+''.join(descriptors))
        for name in ('tokenizer.json','tokenizer_config.json','config.json'):
            shutil.copyfile(source/name,package/name)
        records={name:{'identity':identity(package/name),'sha256':digest(package/name)}
                 for name in ('trunk.bin','pool.bin','index.txt','tokenizer.json','tokenizer_config.json','config.json')}
        package_id=hashlib.sha256(json.dumps({'abi':ABI,'revision':REVISION,
            'sha256':{k:v['sha256'] for k,v in records.items()}},sort_keys=True).encode()).hexdigest()
        manifest={'schema':'salt.maple.package.v1','abi':ABI,'source_repo':REPO,
                  'source_revision':REVISION,'package_id':package_id,'files':records,
                  'tensor_records':len(tensors),'expert_bytes':EXPERT_BYTES}
        (package/'auth.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('PACKAGE_READY',package,package_id,flush=True)
    finally:
        for f in handles.values(): f.close()


def compact_tokenizer(document):
    """Lossless representation change supported by the unchanged C loader."""
    result = json.loads(json.dumps(document))
    pairs = document['model']['merges']
    if not all(isinstance(pair, list) and len(pair) == 2 and
               all(isinstance(token, str) and token and ' ' not in token
                   for token in pair) for pair in pairs):
        raise ValueError('compact tokenizer requires unambiguous array-form merge pairs')
    result['model']['merges'] = [' '.join(pair) for pair in pairs]
    # Compare the complete parsed document, not only vocabulary size or samples.
    restored = json.loads(json.dumps(result))
    restored['model']['merges'] = [value.split(' ') for value in result['model']['merges']]
    if restored != document:
        raise ValueError('tokenizer semantic round trip failed')
    return result


def prepare_runtime_tokenizer(package):
    manifest = json.loads((package/'auth.json').read_text())
    source = package/'tokenizer.json'
    authority = manifest['files']['tokenizer.json']
    if identity(source) != authority['identity'] or digest(source) != authority['sha256']:
        raise ValueError('source tokenizer drift')
    document = json.loads(source.read_bytes(), object_pairs_hook=unique_object)
    compact = compact_tokenizer(document)
    encoded = (json.dumps(compact, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
    path = package/'tokenizer.runtime.json'
    receipt_path = package/'tokenizer.runtime.auth.json'
    if path.exists() or receipt_path.exists():
        raise ValueError('runtime tokenizer already exists; refusing overwrite')
    with path.open('xb') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    receipt = {'schema':'salt.maple.tokenizer-layout.v1', 'layout':'compact-merges-v1',
        'package_id':manifest['package_id'], 'source_sha256':authority['sha256'],
        'runtime_sha256':hashlib.sha256(encoded).hexdigest(), 'identity':identity(path),
        'merges':len(document['model']['merges']), 'vocabulary':len(document['model']['vocab']),
        'complete_document_round_trip':True}
    pending = receipt_path.with_suffix('.pending')
    with pending.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, receipt_path)
    print('TOKENIZER_LAYOUT_READY', json.dumps(receipt, sort_keys=True), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('fetch','import','prepare-tokenizer'))
    p.add_argument('--source',type=Path)
    p.add_argument('--package',type=Path)
    args=p.parse_args()
    if args.action=='prepare-tokenizer':
        if args.package is None: p.error('--package is required')
        prepare_runtime_tokenizer(args.package)
    elif args.action=='fetch':
        if args.source is None: p.error('--source is required')
        fetch(args.source)
    else:
        if args.package is None or args.source is None: p.error('--source and --package are required')
        materialize(args.source,args.package)

if __name__=='__main__':
    main()
