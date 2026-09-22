#!/usr/bin/env python3
"""Build GPU metadata over canonical packed Q4 storage.

Version 1 writes explicit source segments, expert references, and immutable
dispatch records. It copies zero payload bytes and never modifies or replaces
the canonical source pool. The unified mode instead adds an optional ``gpu``
extension to a copy of the existing CPU layout; production can then consume
one authoritative index rather than opening the v1 directory as a peer gate.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ALIGN_DEFAULT = 16_384
COPY_CHUNK = 8 << 20
INDEX_MAGIC = b"SALTGP01"
DISPATCH_MAGIC = b"SALTDSP1"
VERSION = 1
UINT64_MAX = (1 << 64) - 1

INDEX_HEADER = struct.Struct("<8sIIIIIIIQQQ32s")
INDEX_SEGMENT = struct.Struct("<IIIIQQQQ")
INDEX_RECORD = struct.Struct("<IIIIQQQ")
DISPATCH_HEADER = struct.Struct("<8sIIIIIII32s")
DISPATCH_RECORD = struct.Struct("<IIIIQQQQQQIIIIIIII")

COMPONENT = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
Q4_BLOCK_SIZE = 64


class BuildError(Exception):
    pass


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(COPY_CHUNK)
            if not block:
                break
            h.update(block)
    return h.digest()


def require_int(obj, key, where, minimum=0):
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BuildError(f"{where}: {key} must be an integer >= {minimum}")
    return value


def read_pool_header(pool_path):
    size = pool_path.stat().st_size
    with pool_path.open("rb") as f:
        raw = f.read(24)
    if len(raw) != 24:
        raise BuildError("source pool has a short 24-byte header")
    expert_nbytes, n_layers, n_experts = struct.unpack("<QQQ", raw)
    if not expert_nbytes or not n_layers or not n_experts:
        raise BuildError("source pool header contains zero geometry")
    expected = 24 + expert_nbytes * n_layers * n_experts
    if size != expected:
        raise BuildError(f"source pool extent {size} != header extent {expected}")
    return expert_nbytes, n_layers, n_experts, size


def load_layout(path, expert_nbytes, n_layers, n_experts):
    raw = path.read_bytes()
    try:
        layout = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"layout is not valid JSON: {exc}") from exc
    if not isinstance(layout, dict):
        raise BuildError("layout root must be an object")
    if layout.get("format") != "q4-split-v1":
        raise BuildError("layout format must be q4-split-v1")
    if require_int(layout, "n_layers", "layout", 1) != n_layers:
        raise BuildError("layout layer count does not match source pool")
    if require_int(layout, "n_experts", "layout", 1) != n_experts:
        raise BuildError("layout expert count does not match source pool")
    if require_int(layout, "expert_nbytes", "layout", 1) != expert_nbytes:
        raise BuildError("layout expert size does not match source pool")
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise BuildError("layout tensors must be an array")

    table = {}
    for ordinal, tensor in enumerate(tensors):
        where = f"layout tensor {ordinal}"
        if not isinstance(tensor, dict):
            raise BuildError(f"{where}: entry must be an object")
        name = tensor.get("name", tensor.get("n"))
        if name not in COMPONENT:
            continue
        layer = require_int(tensor, "layer", where)
        expert = require_int(tensor, "expert", where)
        if layer >= n_layers or expert >= n_experts:
            raise BuildError(f"{where}: logical identity is out of range")
        key = (layer, expert, COMPONENT[name])
        if key in table:
            raise BuildError(f"{where}: duplicate {name} for L{layer} E{expert}")
        shape = tensor.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2 or
                any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                    for v in shape)):
            raise BuildError(f"{where}: shape must contain two positive integers")
        fmt = require_int(tensor, "fmt", where)
        bits = require_int(tensor, "bits", where, 1)
        if fmt != 1 or bits != 4:
            raise BuildError(f"{where}: v1 accepts only fmt=1, bits=4 Q4 tensors")

        source = 24 + (layer * n_experts + expert) * expert_nbytes
        fields = {}
        ranges = []
        for prefix in ("v", "s", "b"):
            off = require_int(tensor, f"{prefix}_off", where)
            length = require_int(tensor, f"{prefix}_nbytes", where, 1)
            rel = off - source
            if rel < 0 or rel + length > expert_nbytes:
                raise BuildError(f"{where}: {prefix} range is outside its expert slot")
            alignment = 4 if prefix == "v" else 2
            if rel % alignment or length % alignment:
                raise BuildError(f"{where}: {prefix} range is not {alignment}-byte aligned")
            fields[f"{prefix}_rel"] = rel
            fields[f"{prefix}_nbytes"] = length
            ranges.append((prefix, rel, rel + length))

        elements = shape[0] * shape[1]
        if elements % Q4_BLOCK_SIZE:
            raise BuildError(
                f"{where}: element count is not divisible by Q4 block size {Q4_BLOCK_SIZE}")
        groups = elements // Q4_BLOCK_SIZE
        expected = {"v_nbytes": elements // 2,
                    "s_nbytes": groups * 2,
                    "b_nbytes": groups * 2}
        for field, length in expected.items():
            if fields[field] != length:
                raise BuildError(
                    f"{where}: {field}={fields[field]}, expected {length}")
        for i, (aname, abeg, aend) in enumerate(ranges):
            for bname, bbeg, bend in ranges[i + 1:]:
                if abeg < bend and bbeg < aend:
                    raise BuildError(f"{where}: {aname}/{bname} ranges overlap")
        table[key] = {
            "layer": layer,
            "expert": expert,
            "component": COMPONENT[name],
            "shape": tuple(shape),
            "fmt": fmt,
            "bits": bits,
            **fields,
        }

    expected = n_layers * n_experts * len(COMPONENT)
    if len(table) != expected:
        raise BuildError(f"layout has {len(table)} Q4 dispatch tensors; expected {expected}")
    for layer in range(n_layers):
        for expert in range(n_experts):
            used = []
            for component in range(3):
                tensor = table[(layer, expert, component)]
                for prefix in ("v", "s", "b"):
                    beg = tensor[f"{prefix}_rel"]
                    end = beg + tensor[f"{prefix}_nbytes"]
                    for other_component, other_prefix, other_beg, other_end in used:
                        if beg < other_end and other_beg < end:
                            raise BuildError(
                                f"L{layer} E{expert}: component {component} {prefix} "
                                f"overlaps component {other_component} {other_prefix}")
                    used.append((component, prefix, beg, end))
    return raw, table


def derive_segments(alignment, expert_nbytes, n_layers, n_experts,
                    source_nbytes):
    layer_payload = expert_nbytes * n_experts
    segments = []
    for layer in range(n_layers):
        payload_file_offset = 24 + layer * layer_payload
        map_file_offset = payload_file_offset & ~(alignment - 1)
        payload_offset = payload_file_offset - map_file_offset
        map_nbytes = payload_offset + layer_payload
        if map_file_offset + map_nbytes > source_nbytes:
            raise BuildError(f"layer {layer}: segment exceeds source pool")
        segments.append({
            "segment_id": layer,
            "layer": layer,
            "file_offset": map_file_offset,
            "map_nbytes": map_nbytes,
            "payload_offset": payload_offset,
            "payload_nbytes": layer_payload,
        })
    return segments


def write_index(path, alignment, expert_nbytes, n_layers, n_experts,
                source_nbytes, layout_identity, segments):
    count = n_layers * n_experts
    with path.open("wb") as out:
        out.write(INDEX_HEADER.pack(
            INDEX_MAGIC, VERSION, INDEX_HEADER.size, INDEX_RECORD.size,
            alignment, n_layers, n_experts, n_layers, expert_nbytes,
            source_nbytes, count, layout_identity))
        for segment in segments:
            out.write(INDEX_SEGMENT.pack(
                segment["segment_id"], segment["layer"], 0, 0,
                segment["file_offset"], segment["map_nbytes"],
                segment["payload_offset"], segment["payload_nbytes"]))
        for layer in range(n_layers):
            for expert in range(n_experts):
                source = 24 + (layer * n_experts + expert) * expert_nbytes
                local = segments[layer]["payload_offset"] + expert * expert_nbytes
                out.write(INDEX_RECORD.pack(
                    layer, expert, layer, 0, local, expert_nbytes, source))
        out.flush()
        os.fsync(out.fileno())


def derive_block_size(tensor):
    return Q4_BLOCK_SIZE


def write_dispatch(path, layout_sha, table, segments, expert_nbytes,
                   n_layers, n_experts):
    records = [table[(layer, expert, component)]
               for layer in range(n_layers)
               for expert in range(n_experts)
               for component in range(3)]
    with path.open("wb") as out:
        out.write(DISPATCH_HEADER.pack(
            DISPATCH_MAGIC, VERSION, DISPATCH_HEADER.size,
            DISPATCH_RECORD.size, len(records), n_layers, n_experts, 0,
            layout_sha))
        for tensor in records:
            expert_base = (segments[tensor["layer"]]["payload_offset"] +
                           tensor["expert"] * expert_nbytes)
            block_size = derive_block_size(tensor)
            out.write(DISPATCH_RECORD.pack(
                tensor["layer"], tensor["expert"], tensor["component"],
                tensor["layer"],
                expert_base + tensor["v_rel"],
                expert_base + tensor["s_rel"],
                expert_base + tensor["b_rel"],
                tensor["v_nbytes"], tensor["s_nbytes"], tensor["b_nbytes"],
                tensor["shape"][0], tensor["shape"][1],
                tensor["fmt"], tensor["bits"], block_size,
                1,       # pipeline selector: staged Q4
                256,     # static threadgroup policy identifier
                0))
        out.flush()
        os.fsync(out.fileno())


def write_json(path, value):
    with path.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def write_json_atomic_new(path, value):
    """Publish one new JSON file without exposing a partial candidate."""
    if path.exists():
        raise BuildError(f"output already exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise BuildError(f"temporary output already exists: {tmp}")
    try:
        with tmp.open("x", encoding="utf-8") as out:
            json.dump(value, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def source_identity(path):
    st = path.stat()
    return {
        "device": st.st_dev,
        "inode": st.st_ino,
        "nbytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def publish_ready(output, incomplete, manifest):
    ready_tmp = output / "READY.tmp"
    with ready_tmp.open("w", encoding="utf-8") as ready:
        identity = manifest["source_pool_identity"]
        ready.write(f"{manifest['format']}\n")
        ready.write(f"index_sha256={manifest['index_sha256']}\n")
        ready.write(f"dispatch_sha256={manifest['dispatch_sha256']}\n")
        ready.write(f"manifest_sha256={manifest['manifest_sha256']}\n")
        ready.write(f"source_device={identity['device']}\n")
        ready.write(f"source_inode={identity['inode']}\n")
        ready.write(f"source_nbytes={identity['nbytes']}\n")
        ready.write(f"source_mtime_ns={identity['mtime_ns']}\n")
        ready.write(f"source_ctime_ns={identity['ctime_ns']}\n")
        ready.flush()
        os.fsync(ready.fileno())
    incomplete.unlink()
    os.replace(ready_tmp, output / "READY")


def build_legacy(args):
    pool_path = args.pool.resolve()
    layout_path = args.layout.resolve()
    output = args.output.resolve()
    alignment = args.alignment

    if alignment < 4096 or alignment & (alignment - 1):
        raise BuildError("alignment must be a power of two and at least 4096")
    if not pool_path.is_file():
        raise BuildError(f"source pool does not exist: {pool_path}")
    if not layout_path.is_file():
        raise BuildError(f"source layout does not exist: {layout_path}")
    if output.exists():
        raise BuildError(f"output already exists; refusing to overwrite: {output}")

    initial_source_identity = source_identity(pool_path)
    expert_nbytes, n_layers, n_experts, source_nbytes = read_pool_header(pool_path)
    layout_raw, table = load_layout(
        layout_path, expert_nbytes, n_layers, n_experts)
    layout_sha = hashlib.sha256(layout_raw).digest()
    segments = derive_segments(
        alignment, expert_nbytes, n_layers, n_experts, source_nbytes)

    output.mkdir(parents=True)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("salt-gpu-ledger-v1 generation in progress\n",
                          encoding="utf-8")

    index_path = output / "index.bin"
    dispatch_path = output / "dispatch.bin"
    write_index(index_path, alignment, expert_nbytes, n_layers, n_experts,
                source_nbytes, layout_sha, segments)
    write_dispatch(dispatch_path, layout_sha, table, segments, expert_nbytes,
                   n_layers, n_experts)

    manifest = {
        "format": "salt-gpu-ledger-v1",
        "format_version": VERSION,
        "state": "ready",
        "payload_storage": "external-source-pool",
        "resource_alignment": alignment,
        "source_pool": str(pool_path),
        "source_pool_nbytes": source_nbytes,
        "source_pool_identity": initial_source_identity,
        "source_layout": str(layout_path),
        "source_layout_sha256": layout_sha.hex(),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "expert_nbytes": expert_nbytes,
        "index_nbytes": index_path.stat().st_size,
        "index_sha256": hash_file(index_path).hex(),
        "dispatch_nbytes": dispatch_path.stat().st_size,
        "dispatch_sha256": hash_file(dispatch_path).hex(),
        "segments": segments,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = hash_file(manifest_path).hex()

    if source_identity(pool_path) != initial_source_identity:
        raise BuildError("source pool changed while ledger metadata was generated")

    for artifact in (index_path, dispatch_path, manifest_path):
        with artifact.open("rb") as fd:
            os.fsync(fd.fileno())

    publish_ready(output, incomplete, manifest)

    print(f"gpu-ledger: ready {output}")
    print(f"gpu-ledger: {n_layers} source views, {n_layers * n_experts} experts")
    print("gpu-ledger: payload bytes copied 0")


def build_unified(args):
    """Emit one CPU-primary layout with an additive GPU policy extension."""
    pool_path = args.pool.resolve()
    layout_path = args.layout.resolve()
    output = args.unified_output.resolve()
    alignment = args.alignment

    if alignment < 4096 or alignment & (alignment - 1):
        raise BuildError("alignment must be a power of two and at least 4096")
    if not pool_path.is_file():
        raise BuildError(f"source pool does not exist: {pool_path}")
    if not layout_path.is_file():
        raise BuildError(f"source layout does not exist: {layout_path}")

    initial_source_identity = source_identity(pool_path)
    expert_nbytes, n_layers, n_experts, _ = read_pool_header(pool_path)
    layout_raw, _ = load_layout(
        layout_path, expert_nbytes, n_layers, n_experts)
    layout = json.loads(layout_raw)
    if "gpu" in layout:
        raise BuildError("source layout already contains a gpu extension")
    if "source_pool_identity" in layout:
        raise BuildError("source layout already contains source_pool_identity")

    # Storage identity belongs to the parent CPU index. The GPU object contains
    # only realization policy; source offsets, shapes, and extents remain in the
    # existing tensor records and are never serialized a second time.
    layout["source_pool_identity"] = initial_source_identity
    layout["gpu"] = {
        "version": VERSION,
        "resource_alignment": alignment,
        "view_partition": "layer",
        "pipeline_defaults": {
            "fmt1_bits4": {
                "block_size": Q4_BLOCK_SIZE,
                "pipeline_id": 1,
                "threadgroup_id": 256,
                "flags": 0,
            }
        },
    }

    if source_identity(pool_path) != initial_source_identity:
        raise BuildError("source pool changed while unified metadata was generated")
    write_json_atomic_new(output, layout)
    if source_identity(pool_path) != initial_source_identity:
        output.unlink()
        raise BuildError("source pool changed while unified metadata was published")

    print(f"gpu-ledger: unified CPU index ready {output}")
    print(f"gpu-ledger: {n_layers} derived source views, "
          f"{n_layers * n_experts} experts")
    print("gpu-ledger: duplicated tensor records 0")
    print("gpu-ledger: payload bytes copied 0")


def build(args):
    if args.unified_output is not None:
        build_unified(args)
    else:
        build_legacy(args)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build a metadata-only GPU ledger over canonical pool.bin")
    parser.add_argument("--pool", type=Path, required=True,
                        help="packed source pool.bin")
    parser.add_argument("--layout", type=Path, required=True,
                        help="Q4 pool layout manifest.json")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path,
                        help="new legacy v1 output directory (must not exist)")
    output.add_argument("--unified-output", type=Path,
                        help="new CPU-primary layout JSON (must not exist)")
    parser.add_argument("--alignment", type=int, default=ALIGN_DEFAULT,
                        help=f"resource alignment (default {ALIGN_DEFAULT})")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        build(parse_args(argv))
    except (BuildError, OSError) as exc:
        print(f"gpu-pool: ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
