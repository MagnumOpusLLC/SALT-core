#!/usr/bin/env python3
"""Bounded GGUF v3 metadata/tensor inspection for the pinned MTP Q4 file."""
import argparse
import json
import math
import pathlib
import struct

TYPE_WIDTH = {
    0: (4, 1),   # F32 bytes/elements
    1: (2, 1),   # F16
    2: (18, 32), # Q4_0
    8: (34, 32), # Q8_0
    30: (2, 1),  # BF16
}
META_SCALAR = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}


def read_exact(handle, count):
    value = handle.read(count)
    if len(value) != count:
        raise ValueError("truncated GGUF")
    return value


def scalar(handle, fmt):
    return struct.unpack("<" + fmt, read_exact(handle, struct.calcsize(fmt)))[0]


def string(handle):
    length = scalar(handle, "Q")
    if length > 64 * 1024 * 1024:
        raise ValueError("oversized GGUF string")
    return read_exact(handle, length).decode("utf-8")


def value(handle, kind, retain=True):
    if kind in META_SCALAR:
        result = scalar(handle, META_SCALAR[kind])
        return result if retain else None
    if kind == 8:
        result = string(handle)
        return result if retain else None
    if kind == 9:
        item_kind = scalar(handle, "I")
        count = scalar(handle, "Q")
        if count > 10_000_000:
            raise ValueError("oversized GGUF array")
        if item_kind in META_SCALAR:
            size = struct.calcsize(META_SCALAR[item_kind])
            read_exact(handle, count * size)
        elif item_kind == 8:
            for _ in range(count):
                string(handle)
        else:
            for _ in range(count):
                value(handle, item_kind, False)
        return {"element_type": item_kind, "count": count} if retain else None
    raise ValueError(f"unknown GGUF metadata type {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = pathlib.Path(args.model)
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        version = scalar(handle, "I")
        tensor_count = scalar(handle, "Q")
        metadata_count = scalar(handle, "Q")
        if magic != b"GGUF" or version != 3 or tensor_count > 1024 or metadata_count > 1024:
            raise ValueError("unsupported GGUF header")
        metadata = {}
        for _ in range(metadata_count):
            key = string(handle)
            kind = scalar(handle, "I")
            keep = key in {
                "general.architecture", "general.alignment", "general.file_type",
                "gemma4-assistant.block_count", "gemma4-assistant.embedding_length",
                "gemma4-assistant.feed_forward_length",
                "gemma4-assistant.embedding_length_out",
                "gemma4-assistant.nextn_predict_layers",
            }
            item = value(handle, kind, keep)
            if keep:
                metadata[key] = item
        tensors = []
        for _ in range(tensor_count):
            name = string(handle)
            dims = scalar(handle, "I")
            if dims < 1 or dims > 4:
                raise ValueError(f"invalid dims for {name}")
            shape = [scalar(handle, "Q") for _ in range(dims)]
            kind = scalar(handle, "I")
            offset = scalar(handle, "Q")
            tensors.append({"name":name,"shape":shape,"type":kind,"relative_offset":offset})
        alignment = int(metadata.get("general.alignment", 32))
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment
    by_offset = sorted(tensors, key=lambda item: item["relative_offset"])
    for index,item in enumerate(by_offset):
        elements = math.prod(item["shape"])
        if item["type"] not in TYPE_WIDTH:
            raise ValueError(f"unsupported tensor type {item['type']} for {item['name']}")
        block_bytes, block_elements = TYPE_WIDTH[item["type"]]
        if elements % block_elements:
            raise ValueError(f"unaligned block count for {item['name']}")
        expected = elements // block_elements * block_bytes
        next_offset = (by_offset[index + 1]["relative_offset"]
                       if index + 1 < len(by_offset) else path.stat().st_size - data_offset)
        if expected > next_offset - item["relative_offset"]:
            raise ValueError(f"tensor extent overlap for {item['name']}")
        item["elements"] = elements
        item["bytes"] = expected
        item["absolute_offset"] = data_offset + item["relative_offset"]
    result={"schema":"salt.gemma4-mtp-q4-layout.v1","version":version,
            "file_bytes":path.stat().st_size,"metadata_count":metadata_count,
            "tensor_count":tensor_count,"data_offset":data_offset,
            "metadata":metadata,"tensors":sorted(tensors,key=lambda x:x["name"])}
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in ("file_bytes","metadata_count","tensor_count","data_offset")},sort_keys=True))


if __name__ == "__main__":
    main()
