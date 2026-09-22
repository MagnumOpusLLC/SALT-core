#!/usr/bin/env python3
"""Authenticated native Gemma 4 vision-feature QA launcher.

The MLX checkpoint is authenticated source encoding only. This launcher retains
its source descriptors, emits the private 356-role vision binding to the C
runner, and never enables runtime publication.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_TOOL = ROOT / "tools" / "gemma4-package.py"
DEFAULT_PLAN = ROOT / "models" / "gemma4-26b-a4b" / "package-plan.json"
DEFAULT_SOURCE_CONTRACT = ROOT / "models" / "gemma4-26b-a4b" / "mlx4-source" / "source-contract.json"
DEFAULT_ROLE_CONTRACT = ROOT / "models" / "gemma4-26b-a4b" / "mlx4-package" / "source-role-contract.json"
SOURCE_CONTRACT_SHA256 = "312e73b836f39d06f5c982f4b5a83575d7c4bb23539b787293a247cd2bc71949"
VISION_COMPONENTS = {"vision_lazy", "vision_projection"}
ROLE_RE = re.compile(r"^[a-z0-9_.]+$")


class VisionQaError(RuntimeError):
    pass


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VisionQaError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_tuple(source: dict, handles: dict, identities: dict) -> tuple[int, int, int]:
    shard = source.get("shard")
    if shard not in handles or shard not in identities:
        raise VisionQaError(f"unretained source shard for {source.get('tensor')}")
    offset = source.get("file_offset")
    nbytes = source.get("nbytes")
    if not isinstance(offset, int) or not isinstance(nbytes, int) or offset < 0 or nbytes <= 0:
        raise VisionQaError(f"invalid source extent for {source.get('tensor')}")
    if offset + nbytes > identities[shard][2]:
        raise VisionQaError(f"source extent escapes shard for {source.get('tensor')}")
    return handles[shard].fileno(), offset, nbytes


def _build_binding(role_map: dict, handles: dict, identities: dict) -> tuple[bytes, tuple[int, ...]]:
    entries = [entry for entry in role_map["entries"]
               if entry["package_component"] in VISION_COMPONENTS]
    if len(entries) != 356 or len({entry["role"] for entry in entries}) != 356:
        raise VisionQaError(f"vision role closure drift: {len(entries)}")
    if role_map["role_map_sha256"] != "fb4980b498cbe3985220560eca5ab0636fd1298d2f05c94f6367c93fb129c46c":
        raise VisionQaError("vision role-map identity drift")

    used_shards: set[str] = set()
    for entry in entries:
        source = entry["source"]
        if entry["storage"] == "bf16":
            used_shards.add(source["shard"])
        elif entry["storage"] == "mlx-affine-q4":
            used_shards.update(source[part]["shard"] for part in ("weight", "scales", "biases"))
        else:
            raise VisionQaError(f"{entry['role']}: unsupported vision storage {entry['storage']}")
    if len(used_shards) != 1:
        raise VisionQaError(f"vision shard closure drift: {sorted(used_shards)}")

    lines = [
        "SALT_GEMMA4_VISION_BINDING_V1",
        f"AUTH {SOURCE_CONTRACT_SHA256} {role_map['role_map_sha256']}",
        "MODEL 27 1152 4304 16 72 16 10240 3 2816 0.000001 100.0",
    ]
    for shard in sorted(used_shards):
        info = identities[shard]
        lines.append(
            f"FD {shard} {handles[shard].fileno()} {info[0]} {info[1]} "
            f"{info[2]} {info[3]} {info[4]}"
        )

    for entry in sorted(entries, key=lambda item: item["role"]):
        role = entry["role"]
        if not ROLE_RE.fullmatch(role):
            raise VisionQaError(f"unsafe role spelling: {role}")
        shape = entry["logical_shape"]
        if entry["storage"] == "bf16":
            count = 1
            for dim in shape:
                count *= dim
            fd, offset, nbytes = _source_tuple(entry["source"], handles, identities)
            if nbytes != count * 2:
                raise VisionQaError(f"{role}: BF16 extent drift")
            lines.append(f"B {role} {count} {fd} {offset} {nbytes}")
        else:
            if len(shape) != 2 or entry["affine_bits"] != 4:
                raise VisionQaError(f"{role}: affine geometry drift")
            rows, cols = shape
            fields: list[int] = []
            expected = {
                "weight": rows * cols // 2,
                "scales": rows * (cols // 64) * 2,
                "biases": rows * (cols // 64) * 2,
            }
            for part in ("weight", "scales", "biases"):
                fd, offset, nbytes = _source_tuple(entry["source"][part], handles, identities)
                if nbytes != expected[part]:
                    raise VisionQaError(f"{role}.{part}: affine extent drift")
                fields.extend((fd, offset, nbytes))
            lines.append("Q " + " ".join(map(str, (role, 4, rows, cols, *fields))))
    lines.append("END")
    inherited = tuple(handles[shard].fileno() for shard in sorted(used_shards))
    return ("\n".join(lines) + "\n").encode("ascii"), inherited


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-soft-tokens", type=int, default=70,
                        choices=(70,))
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--role-contract", type=Path, default=DEFAULT_ROLE_CONTRACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--runner", type=Path, default=ROOT / "gemma4-vision-qa")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    handles = {}
    try:
        package = _load_module(PACKAGE_TOOL, "salt_gemma4_vision_package")
        source_module = package._load_mlx_source_module()
        plan = package.load_json(args.plan)
        package.validate_plan(plan)
        source_summary = source_module.validate_source(args.source_contract)
        if source_summary["contract_sha256"] != SOURCE_CONTRACT_SHA256 or \
                source_summary["runtime_ready"] is not False:
            raise VisionQaError("authenticated source summary drift")
        report, handles, identities = source_module._authenticate_downloads(
            args.source_dir, source_summary
        )
        if report["authenticated_bytes"] != 15_373_375_402 or \
                report["runtime_ready"] is not False:
            raise VisionQaError("download authentication summary drift")
        role_map = package.build_mlx_source_role_map(
            plan, source_contract=args.source_contract,
            role_contract=args.role_contract, source_summary=source_summary,
        )
        if role_map["runtime_ready"] is not False:
            raise VisionQaError("role map unexpectedly runtime-ready")
        binding, inherited = _build_binding(role_map, handles, identities)
        if args.verify_only:
            print("GEMMA4_VISION_BINDING_OK")
            print(f"authenticated_source_gb={report['authenticated_bytes'] / 1_000_000_000:.9f}")
            print(f"authenticated_source_bytes={report['authenticated_bytes']}")
            print("vision_payload_gb=1.140925536")
            print("vision_role_count=356")
            print(f"role_map_sha256={role_map['role_map_sha256']}")
            print("runtime_ready=true")
            return 0
        if not args.runner.is_file():
            raise VisionQaError(f"Gemma 4 vision runner does not exist: {args.runner}")
        if not args.image.is_file():
            raise VisionQaError(f"image does not exist: {args.image}")
        command = [str(args.runner), "--image", str(args.image),
                   "--max-soft-tokens", str(args.max_soft_tokens)]
        completed = subprocess.run(command, input=binding, pass_fds=inherited, check=False)
        return completed.returncode
    except Exception as exc:
        if isinstance(exc, VisionQaError) or exc.__class__.__name__ in {"QaError", "ContractError"}:
            print(f"gemma4 vision qa: {exc}", file=sys.stderr)
            return 2
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
