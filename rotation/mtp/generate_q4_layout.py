#!/usr/bin/env python3
"""Generate compile-time offsets for the pinned Q4_0 assistant GGUF."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / "source-inspection-q4.json"
OUTPUT = ROOT / "source_layout_q4.h"
REVISION = "7b92b5b28818151e8669af2e45e88d6086f490dd"
SHA256 = "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a"

def macro(name):
    return "SALT_MTP_Q4_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()

data=json.loads(SOURCE.read_text())
lines=["#ifndef SALT_GEMMA4_MTP_SOURCE_LAYOUT_Q4_H",
       "#define SALT_GEMMA4_MTP_SOURCE_LAYOUT_Q4_H","",
       "/* Generated from the authenticated GGUF v3 tensor table. */",
       f"#define SALT_MTP_Q4_SOURCE_REVISION \"{REVISION}\"",
       f"#define SALT_MTP_Q4_SOURCE_SHA256 \"{SHA256}\"",
       f"#define SALT_MTP_Q4_SOURCE_BYTES UINT64_C({data['file_bytes']})",
       f"#define SALT_MTP_Q4_DATA_OFFSET UINT64_C({data['data_offset']})",
       f"#define SALT_MTP_Q4_TENSOR_COUNT {data['tensor_count']}u",""]
for tensor in data['tensors']:
    name=macro(tensor['name'])
    lines += [f"#define {name}_OFFSET UINT64_C({tensor['absolute_offset']})",
              f"#define {name}_BYTES UINT64_C({tensor['bytes']})",
              f"#define {name}_TYPE {tensor['type']}u",
              f"#define {name}_DIMS {len(tensor['shape'])}u"]
    for i,extent in enumerate(tensor['shape']):
        lines.append(f"#define {name}_DIM{i} {extent}u")
    lines.append("")
lines += ["#endif",""]
OUTPUT.write_text("\n".join(lines))
print(OUTPUT)
