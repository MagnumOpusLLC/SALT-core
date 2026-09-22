#!/usr/bin/env python3
"""Build Salt's Darwin Metal library from the canonical embedded sources."""
from __future__ import annotations

import argparse
import ast
import hashlib
import pathlib
import re
import subprocess
import tempfile


def extract_c_string(path: pathlib.Path, variable: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{re.escape(variable)}\s*=\s*(.*?);\s*(?:\n|$)",
        text,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"cannot find {variable} in {path}")
    tokens = re.findall(r'"(?:\\.|[^"\\])*"', match.group(1))
    if not tokens:
        raise RuntimeError(f"{variable} in {path} has no string literals")
    return "".join(ast.literal_eval(token) for token in tokens)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--metal", default="xcrun")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        extract_c_string(root / "src/gpu_metal.m", "_kernel_src"),
        extract_c_string(root / "src/gpu_metal.m", "_batch_src2"),
        extract_c_string(
            root / "src/gpu_metal_ops_source.h", "salt_gpu_metal_ops_source"
        ),
    ]
    combined = "\n".join(sources)
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="salt-metal-build-") as tmp:
        tmpdir = pathlib.Path(tmp)
        source = tmpdir / "salt-gpu.metal"
        air = tmpdir / "salt-gpu.air"
        source.write_text(combined, encoding="utf-8")
        subprocess.run(
            [args.metal, "-sdk", "macosx", "metal", "-c", "-fno-fast-math",
             "-DSALT_T4_PROOF=1", str(source), "-o", str(air)],
            check=True,
        )
        subprocess.run(
            [args.metal, "-sdk", "macosx", "metallib", str(air),
             "-o", str(output)],
            check=True,
        )

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"{output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
