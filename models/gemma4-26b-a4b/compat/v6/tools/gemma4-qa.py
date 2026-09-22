#!/usr/bin/env python3
"""Authenticated local Gemma 4 text-QA launcher.

This tool binds the pinned MLX source and committed expert pool to retained
read-only descriptors, emits a private text-only binding to the C runner, and
keeps runtime publication blocked. MLX is a source encoding, never a backend.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
from typing import BinaryIO

from engine_config import (
    load_engine_config,
    validate_gemma_residency_config,
    validate_gemma_target_route_config,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_TOOL = ROOT / "tools" / "gemma4-package.py"
KV_COMPAT_TOOL = ROOT / "server" / "gemma4_compat.py"
BUILD_IDENTITY_TOOL = ROOT / "server" / "gemma4_build.py"
MULTIMODAL_QA_TOOL = ROOT / "tools" / "gemma4-multimodal-qa.py"
KV_COMPAT_MANIFEST = ROOT / "models" / "gemma4-26b-a4b" / "kv-cache-compat.json"
DEFAULT_PLAN = ROOT / "models" / "gemma4-26b-a4b" / "package-plan.json"
DEFAULT_SOURCE_CONTRACT = (
    ROOT / "models" / "gemma4-26b-a4b" / "mlx4-source" / "source-contract.json"
)
DEFAULT_ROLE_CONTRACT = (
    ROOT / "models" / "gemma4-26b-a4b" / "mlx4-package" / "source-role-contract.json"
)
REPOSITORY = "mlx-community/gemma-4-26b-a4b-it-4bit"
REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
SOURCE_CONTRACT_SHA256 = "312e73b836f39d06f5c982f4b5a83575d7c4bb23539b787293a247cd2bc71949"
COPY_MAP_SHA256 = "51fa40aca8cb910896328026fcfb3f07c36a7519d0bd76e6fce574ba3c178a3e"
POOL_BYTES = 12_846_366_744
POOL_SHA256 = "b06c91ecfae36581cbafdda80e116446bb32ea5ec832066e86cb672421a66e31"
POOL_HEADER = (3_345_408, 30, 128)
TEXT_COMPONENTS = {"embedding", "text_trunk", "text_final"}
ROLE_RE = re.compile(r"^[a-z0-9_.]+$")
ALLOWED_QA_GATES = {
    (200, 50), (500, 100), (500, 200), (1500, 300),
    (230, 30), (450, 50), (1000, 200), (2000, 500),
    (512, 128), (1024, 250), (2048, 512),
}
ENGINE_CONFIG = load_engine_config(
    model_dir=ROOT / "models" / "gemma4-26b-a4b",
    recipe=os.environ.get("SALT_GEMMA_PLATFORM_RECIPE"),
)


class QaError(RuntimeError):
    pass


def _native_environment(*, serial_prefill_proof: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    engine_config = dict(ENGINE_CONFIG)
    retain_override = os.environ.get("SALT_PREFILL_RETAIN_LAYERS")
    lookahead_override = os.environ.get("SALT_PREFILL_DECODE_LOOKAHEAD")
    t4_override = os.environ.get("SALT_GEMMA_METAL_EXACT_CELLS")
    digest_override = os.environ.get("SALT_GEMMA_LAYER_DIGESTS")
    fine_token_override = os.environ.get("SALT_TEXT_FINE_TOKEN")
    cpu_graph_override = os.environ.get("SALT_TARGET_CPU_GRAPH")
    area_workers_override = os.environ.get("SALT_TARGET_AREA_WORKERS")
    gpu_program_override = os.environ.get("SALT_TARGET_GPU_PROGRAM")
    prefill_b_override = os.environ.get("SALT_PREFILL_B")
    expert_budget_override = os.environ.get("SALT_EXPERT_BUDGET_GB")
    memory_limit_override = os.environ.get("SALT_GEMMA_MEMORY_LIMIT_GB")
    target_n_override = os.environ.get("SALT_TARGET_ROUTE_N")
    target_f_override = os.environ.get("SALT_TARGET_ROUTE_F")
    target_q_override = os.environ.get("SALT_TARGET_ROUTE_Q")
    if retain_override is not None:
        if retain_override not in ("0", "1"):
            raise QaError("SALT_PREFILL_RETAIN_LAYERS must be 0 or 1")
        engine_config["SALT_PREFILL_RETAIN_LAYERS"] = retain_override
    if lookahead_override is not None:
        if lookahead_override not in ("0", "1"):
            raise QaError("SALT_PREFILL_DECODE_LOOKAHEAD must be 0 or 1")
        engine_config["SALT_PREFILL_DECODE_LOOKAHEAD"] = lookahead_override
    if t4_override is not None:
        if t4_override not in ("0", "1"):
            raise QaError("SALT_GEMMA_METAL_EXACT_CELLS must be 0 or 1")
        engine_config["SALT_GEMMA_METAL_EXACT_CELLS"] = t4_override
    if digest_override not in (None, "0", "1"):
        raise QaError("SALT_GEMMA_LAYER_DIGESTS must be 0 or 1")
    if digest_override == "1" and \
            engine_config.get("SALT_GEMMA_METAL_EXACT_CELLS") != "1":
        raise QaError("layer digests require the T4 exact-cell proof")
    if fine_token_override is not None:
        if fine_token_override not in ("0", "1"):
            raise QaError("SALT_TEXT_FINE_TOKEN must be 0 or 1")
        engine_config["SALT_TEXT_FINE_TOKEN"] = fine_token_override
    if cpu_graph_override is not None:
        if cpu_graph_override not in ("0", "1"):
            raise QaError("SALT_TARGET_CPU_GRAPH must be 0 or 1")
        engine_config["SALT_TARGET_CPU_GRAPH"] = cpu_graph_override
    if area_workers_override is not None:
        try:
            area_workers = int(area_workers_override)
        except ValueError as exc:
            raise QaError("SALT_TARGET_AREA_WORKERS must be in [8, 20]") from exc
        if area_workers < 8 or area_workers > 20:
            raise QaError("SALT_TARGET_AREA_WORKERS must be in [8, 20]")
        engine_config["SALT_TARGET_AREA_WORKERS"] = str(area_workers)
    if gpu_program_override is not None:
        if gpu_program_override not in ("0", "1"):
            raise QaError("SALT_TARGET_GPU_PROGRAM must be 0 or 1")
        engine_config["SALT_TARGET_GPU_PROGRAM"] = gpu_program_override
    if prefill_b_override is not None:
        try:
            prefill_b = int(prefill_b_override)
        except ValueError as exc:
            raise QaError("SALT_PREFILL_B must be in [1, 4096]") from exc
        if prefill_b < 1 or prefill_b > 4096:
            raise QaError("SALT_PREFILL_B must be in [1, 4096]")
        engine_config["SALT_PREFILL_B"] = str(prefill_b)
    if expert_budget_override is not None:
        try:
            expert_budget = int(expert_budget_override)
        except ValueError as exc:
            raise QaError("SALT_EXPERT_BUDGET_GB must be in [1, 13]") from exc
        if expert_budget < 1 or expert_budget > 13:
            raise QaError("SALT_EXPERT_BUDGET_GB must be in [1, 13]")
        engine_config["SALT_EXPERT_BUDGET_GB"] = str(expert_budget)
    if memory_limit_override is not None:
        try:
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,9})?", memory_limit_override):
                raise InvalidOperation
            memory_limit = Decimal(memory_limit_override)
        except InvalidOperation as exc:
            raise QaError(
                "SALT_GEMMA_MEMORY_LIMIT_GB must be a decimal in [1, 128]"
            ) from exc
        if memory_limit < 1 or memory_limit > 128:
            raise QaError(
                "SALT_GEMMA_MEMORY_LIMIT_GB must be a decimal in [1, 128]"
            )
        engine_config["SALT_GEMMA_MEMORY_LIMIT_GB"] = memory_limit_override
    for key, text, maximum in (
        ("SALT_TARGET_ROUTE_N", target_n_override, 64),
        ("SALT_TARGET_ROUTE_F", target_f_override, 8),
        ("SALT_TARGET_ROUTE_Q", target_q_override, 8),
    ):
        if text is not None:
            try:
                value = int(text)
            except ValueError as exc:
                raise QaError(f"{key} must be in [1, {maximum}]") from exc
            if str(value) != text or value < 1 or value > maximum:
                raise QaError(f"{key} must be in [1, {maximum}]")
            engine_config[key] = text
    validate_gemma_target_route_config(engine_config)
    for key, value in engine_config.items():
        if key.startswith("SALT_"):
            environment[key] = value
    policy = validate_gemma_residency_config(engine_config)
    if policy["layer_quotas"] is not None:
        environment["SALT_CACHE_LAYER_QUOTAS"] = policy["layer_quotas"]
    if serial_prefill_proof:
        environment["SALT_PREFILL_CHUNK"] = "0"
    return environment


def _run_native(command: list[str], binding: bytes,
                inherited: tuple[int, ...], *,
                serial_prefill_proof: bool = False) -> int:
    """Relay termination to the native child and reap it before exit."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, pass_fds=inherited,
        env=_native_environment(serial_prefill_proof=serial_prefill_proof),
    )
    forwarded: list[int] = []

    def relay(signum, _frame) -> None:
        if not forwarded:
            forwarded.append(signum)
        if process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    watched = (signal.SIGTERM, signal.SIGINT)
    previous = {signum: signal.getsignal(signum) for signum in watched}
    try:
        for signum in watched:
            signal.signal(signum, relay)
        process.communicate(input=binding)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if forwarded:
        return 128 + forwarded[0]
    return process.returncode


def _run_native_persistent(command: list[str], binding: bytes,
                           inherited: tuple[int, ...]) -> int:
    """Retain authenticated descriptors while relaying server commands."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, pass_fds=inherited,
        env=_native_environment(),
    )
    assert process.stdin is not None
    forwarded: list[int] = []

    def relay(signum, _frame) -> None:
        if not forwarded:
            forwarded.append(signum)
        if process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    watched = (signal.SIGTERM, signal.SIGINT)
    previous = {signum: signal.getsignal(signum) for signum in watched}
    selector = selectors.DefaultSelector()
    try:
        for signum in watched:
            signal.signal(signum, relay)
        process.stdin.write(binding)
        process.stdin.flush()
        selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
        while process.poll() is None and not forwarded:
            if not selector.select(timeout=0.1):
                continue
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                break
            process.stdin.write(chunk)
            process.stdin.flush()
    finally:
        selector.close()
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    returncode = process.wait()
    if forwarded:
        return 128 + forwarded[0]
    return returncode


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise QaError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _kv_compatibility():
    module = _load_module(KV_COMPAT_TOOL, "salt_gemma4_kv_compat")
    return module.load_compatibility(KV_COMPAT_MANIFEST, ROOT)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _open_regular(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise QaError(f"cannot open regular file {path}: {exc}") from exc
    try:
        handle = os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode):
        handle.close()
        raise QaError(f"not a regular file: {path}")
    return handle


def _read_stable_json(
    path: Path, max_bytes: int = 1 << 20,
) -> tuple[dict, tuple[int, int, int, int, int]]:
    handle = _open_regular(path)
    try:
        before = _identity(os.fstat(handle.fileno()))
        if before[2] < 2 or before[2] > max_bytes:
            raise QaError(f"invalid JSON extent: {path}")
        data = handle.read(max_bytes + 1)
        if len(data) != before[2]:
            raise QaError(f"short JSON read: {path}")
        if _identity(os.fstat(handle.fileno())) != before:
            raise QaError(f"JSON identity changed while reading: {path}")
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QaError(f"invalid JSON {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise QaError(f"JSON root is not an object: {path}")
        return value, before
    finally:
        handle.close()


def _hash_retained(handle: BinaryIO, expected_bytes: int, expected_sha256: str,
                   label: str) -> tuple[int, int, int, int, int]:
    before = _identity(os.fstat(handle.fileno()))
    if before[2] != expected_bytes:
        raise QaError(f"{label}: extent {before[2]} != {expected_bytes}")
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(8 << 20)
        if not chunk:
            break
        digest.update(chunk)
    after = _identity(os.fstat(handle.fileno()))
    if after != before:
        raise QaError(f"{label}: identity changed while hashing")
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise QaError(f"{label}: sha256 {actual} != {expected_sha256}")
    handle.seek(0)
    return before


def _validate_pool_receipt(pool_path: Path, receipt_path: Path,
                           source_summary: dict) -> dict:
    receipt, _ = _read_stable_json(receipt_path)
    if set(receipt) != {
        "copy_map_sha256", "copy_range_count", "pool", "runtime_blocker",
        "runtime_ready", "schema", "source", "status", "transaction_id",
    }:
        raise QaError("expert pool receipt key set drift")
    if receipt["schema"] != "salt.gemma4-mlx4-import-receipt.v1" or \
            receipt["status"] != "expert-pool-materialized-runtime-blocked" or \
            receipt["runtime_ready"] is not False or \
            receipt["runtime_blocker"] != "blocked-until-gemma-geglu-runtime-exists":
        raise QaError("expert pool receipt status drift")
    if receipt["copy_map_sha256"] != COPY_MAP_SHA256 or \
            receipt["copy_map_sha256"] != source_summary["copy_map_sha256"] or \
            receipt["copy_range_count"] != 34_560:
        raise QaError("expert pool receipt copy-map drift")
    expected_source = {
        "authenticated_bytes": 15_373_375_402,
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "repository": REPOSITORY,
        "revision": REVISION,
    }
    if receipt["source"] != expected_source:
        raise QaError("expert pool receipt source identity drift")
    expected_pool = {
        "bytes": POOL_BYTES,
        "path": str(pool_path.resolve()),
        "sha256": POOL_SHA256,
    }
    if receipt["pool"] != expected_pool:
        raise QaError("expert pool receipt pool identity drift")
    return receipt


def _validate_pool(pool_path: Path, receipt_path: Path,
                   source_summary: dict) -> tuple[BinaryIO, tuple[int, int, int, int, int], dict]:
    receipt = _validate_pool_receipt(pool_path, receipt_path, source_summary)

    handle = _open_regular(pool_path)
    try:
        identity = _hash_retained(handle, POOL_BYTES, POOL_SHA256, "expert pool")
        header = handle.read(24)
        if len(header) != 24 or struct.unpack("<QQQ", header) != POOL_HEADER:
            raise QaError("expert pool header drift")
        if _identity(os.fstat(handle.fileno())) != identity:
            raise QaError("expert pool identity changed during header validation")
        handle.seek(0)
        return handle, identity, receipt
    except Exception:
        handle.close()
        raise


def _open_authenticated_pool(
    pool_path: Path, receipt_path: Path, source_summary: dict,
    expected_identity: list[int] | tuple[int, int, int, int, int],
) -> tuple[BinaryIO, tuple[int, int, int, int, int], dict]:
    """Retain a previously authenticated pool without rehashing its payload."""
    receipt = _validate_pool_receipt(pool_path, receipt_path, source_summary)
    expected = tuple(expected_identity)
    if len(expected) != 5 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in expected):
        raise QaError("invalid cached expert pool identity")
    handle = _open_regular(pool_path)
    try:
        identity = _identity(os.fstat(handle.fileno()))
        if identity != expected or identity[2] != POOL_BYTES:
            raise QaError("expert pool cached file identity drift")
        header = handle.read(24)
        if len(header) != 24 or struct.unpack("<QQQ", header) != POOL_HEADER:
            raise QaError("expert pool header drift")
        if _identity(os.fstat(handle.fileno())) != identity:
            raise QaError("expert pool identity changed during cached validation")
        handle.seek(0)
        return handle, identity, receipt
    except Exception:
        handle.close()
        raise


def _source_tuple(source: dict, handles: dict[str, BinaryIO],
                  identities: dict[str, tuple[int, int, int, int, int]]) -> tuple[int, int, int]:
    shard = source.get("shard")
    if shard not in handles or shard not in identities:
        raise QaError(f"unretained source shard for {source.get('tensor')}")
    offset = source.get("file_offset")
    nbytes = source.get("nbytes")
    if not isinstance(offset, int) or not isinstance(nbytes, int) or offset < 0 or nbytes <= 0:
        raise QaError(f"invalid source extent for {source.get('tensor')}")
    if offset + nbytes > identities[shard][2]:
        raise QaError(f"source extent escapes shard for {source.get('tensor')}")
    return handles[shard].fileno(), offset, nbytes


def _build_binding(role_map: dict, handles: dict[str, BinaryIO],
                   identities: dict[str, tuple[int, int, int, int, int]],
                   pool_handle: BinaryIO,
                   pool_identity: tuple[int, int, int, int, int],
                   kv_compat_sha256: str,
                   build_identity_sha256: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{64}", kv_compat_sha256) is None or \
            re.fullmatch(r"[0-9a-f]{64}", build_identity_sha256) is None or \
            build_identity_sha256 == "0" * 64:
        raise QaError("invalid KV/build compatibility SHA-256")
    entries = [entry for entry in role_map["entries"]
               if entry["package_component"] in TEXT_COMPONENTS]
    if len(entries) != 597:
        raise QaError(f"text role count {len(entries)} != 597")
    roles = {entry["role"] for entry in entries}
    if len(roles) != len(entries) or "text.embedding" not in roles or "text.final_norm" not in roles:
        raise QaError("text role closure drift")

    lines = [
        "SALT_GEMMA4_TEXT_BINDING_V3",
        f"AUTH {SOURCE_CONTRACT_SHA256} {role_map['role_map_sha256']} "
        f"{COPY_MAP_SHA256} {kv_compat_sha256} {build_identity_sha256}",
        "MODEL 30 2816 262144 128 8 1024 0.000001 30.0 53.0",
    ]
    for shard in sorted(handles):
        info = identities[shard]
        lines.append(
            f"FD {shard} {handles[shard].fileno()} {info[0]} {info[1]} "
            f"{info[2]} {info[3]} {info[4]}"
        )
    pi = pool_identity
    lines.append(
        f"POOL {pool_handle.fileno()} {pi[0]} {pi[1]} {pi[2]} {pi[3]} {pi[4]} "
        f"{POOL_HEADER[0]} {POOL_HEADER[1]} {POOL_HEADER[2]}"
    )

    for entry in sorted(entries, key=lambda item: item["role"]):
        role = entry["role"]
        if not ROLE_RE.fullmatch(role):
            raise QaError(f"unsafe role spelling: {role}")
        shape = entry["logical_shape"]
        if entry["storage"] in {"mlx-affine-q4", "mlx-affine-q8"}:
            if len(shape) != 2:
                raise QaError(f"{role}: affine rank drift")
            rows, cols = shape
            bits = entry["affine_bits"]
            expected = {
                "weight": rows * cols * bits // 8,
                "scales": rows * (cols // 64) * 2,
                "biases": rows * (cols // 64) * 2,
            }
            fields = []
            for part in ("weight", "scales", "biases"):
                fd, offset, nbytes = _source_tuple(entry["source"][part], handles, identities)
                if nbytes != expected[part]:
                    raise QaError(f"{role}.{part}: extent drift")
                fields.extend((fd, offset, nbytes))
            lines.append(
                "Q " + " ".join(map(str, (role, bits, rows, cols, *fields)))
            )
        elif entry["storage"] == "bf16":
            count = 1
            for dim in shape:
                count *= dim
            fd, offset, nbytes = _source_tuple(entry["source"], handles, identities)
            if nbytes != count * 2:
                raise QaError(f"{role}: BF16 extent drift")
            lines.append(f"B {role} {count} {fd} {offset} {nbytes}")
        else:
            raise QaError(f"{role}: unsupported text storage {entry['storage']}")

    tokenizer = handles.get("tokenizer.json")
    if tokenizer is None:
        raise QaError("authenticated tokenizer descriptor missing")
    lines.append(f"TOKENIZER {tokenizer.fileno()}")
    lines.append("END")
    return ("\n".join(lines) + "\n").encode("ascii")


def _render_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise QaError("question is empty")
    return (
        "<bos><|turn>user\n" + question +
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def _load_prompt(path: Path, max_bytes: int = 1 << 20) -> str:
    handle = _open_regular(path)
    try:
        before = _identity(os.fstat(handle.fileno()))
        if before[2] < 1 or before[2] > max_bytes:
            raise QaError("rendered prompt must contain 1..1048576 bytes")
        data = handle.read(max_bytes + 1)
        if len(data) != before[2] or _identity(os.fstat(handle.fileno())) != before:
            raise QaError("rendered prompt identity changed while reading")
        try:
            prompt = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise QaError("rendered prompt is not UTF-8") from exc
        if not prompt.strip() or "\0" in prompt:
            raise QaError("rendered prompt is empty or contains NUL")
        return prompt
    finally:
        handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--auth-receipt", type=Path)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--role-contract", type=Path, default=DEFAULT_ROLE_CONTRACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--runner", type=Path, default=ROOT / "gemma4-qa")
    parser.add_argument("--ctx", type=int, default=200)
    parser.add_argument("--gen", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--question")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--compare-kv-warm", action="store_true")
    parser.add_argument("--dpr-proof", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--decode-candidates", help=argparse.SUPPRESS)
    parser.add_argument("--target-frontier", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-reference-save", type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-reference-ids", help=argparse.SUPPRESS)
    parser.add_argument("--dpr-prefill-reference-save", type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-prefill-tokens", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--package-proof", action="store_true")

    parser.add_argument("--kv-load", type=Path)
    parser.add_argument("--kv-save", type=Path)
    parser.add_argument("--kv-compat-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--build-receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--serial-prefill-proof", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--ondemand-rope-proof", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    optimized_decode = args.decode_candidates is not None
    try:
        validate_gemma_residency_config(ENGINE_CONFIG, require_host=True)
        target_config = dict(ENGINE_CONFIG)
        for key in ("SALT_TARGET_ROUTE_N", "SALT_TARGET_ROUTE_F",
                    "SALT_TARGET_ROUTE_Q"):
            if key in os.environ:
                target_config[key] = os.environ[key]
        validate_gemma_target_route_config(target_config)
    except ValueError as exc:
        print(f"gemma4 qa: {exc}", file=sys.stderr)
        return 2
    if args.compare_kv_warm and (args.verify_only or args.load_only or
                                 args.kv_load or args.kv_save or args.dpr_proof or
                                 optimized_decode):
        parser.error("--compare-kv-warm cannot be combined with verify/load-only/KV files")
    if args.dpr_reference_save and not args.dpr_proof:
        parser.error("--dpr-reference-save requires --dpr-proof")
    if args.target_frontier and not (args.dpr_proof or optimized_decode):
        parser.error("--target-frontier requires a decode transaction")
    if optimized_decode and args.dpr_proof:
        parser.error("optimized decode and DPR proof are mutually exclusive")
    if args.dpr_reference_ids and not args.dpr_reference_save:
        parser.error("--dpr-reference-ids requires --dpr-reference-save")
    if bool(args.dpr_prefill_reference_save) != bool(args.dpr_prefill_tokens):
        parser.error("prefill reference path and token count must be paired")
    if args.dpr_prefill_reference_save and (
        args.dpr_proof or optimized_decode or args.compare_kv_warm or
            args.kv_load or args.kv_save or
            args.verify_only or args.load_only):
        parser.error("DPR prefill reference mode must run alone")
    if args.package_proof and args.verify_only:
        parser.error("--package-proof requires a native execution or load-only gate")
    if (args.kv_load or args.kv_save) and (args.verify_only or args.load_only):
        parser.error("KV files cannot be combined with verify/load-only")
    if args.serial_prefill_proof and (args.verify_only or args.load_only):
        parser.error("serial prefill proof requires an execution gate")
    if args.ondemand_rope_proof and (args.verify_only or args.load_only):
        parser.error("on-demand RoPE proof requires an execution gate")
    receipt = args.receipt or Path(str(args.pool) + ".import.json")

    handles: dict[str, BinaryIO] = {}
    pool_handle: BinaryIO | None = None
    try:
        kv_compatibility = _kv_compatibility()
        if (
            args.kv_compat_sha256 is not None
            and args.kv_compat_sha256 != kv_compatibility.hexdigest
        ):
            raise QaError("KV compatibility authority drift")
        if not args.runner.is_file():
            raise QaError(f"Gemma QA runner does not exist: {args.runner}")
        build_module = _load_module(BUILD_IDENTITY_TOOL, "salt_gemma4_qa_build")
        build_identity = build_module.load_build_receipt(
            args.build_receipt or Path(str(args.runner) + ".build.json"),
            args.runner,
            expected_source_compatibility=kv_compatibility.hexdigest,
        )
        if args.auth_receipt is not None:
            auth_module = _load_module(
                MULTIMODAL_QA_TOOL, "salt_gemma4_text_cached_authority",
            )
            authority = auth_module.open_authenticated_binding(
                source_dir=args.source_dir, pool=args.pool, receipt=receipt,
                runner=args.runner,
                build_receipt=(args.build_receipt or
                               Path(str(args.runner) + ".build.json")),
                auth_receipt=args.auth_receipt,
                kv_compat_sha256=kv_compatibility.hexdigest,
                source_contract=args.source_contract,
                role_contract=args.role_contract, plan_path=args.plan,
            )
            handles = authority.handles
            pool_handle = authority.pool_handle
            identities = authority.identities
            report = authority.report
            role_map = authority.role_map
            pool_identity = authority.pool_identity
        else:
            package = _load_module(PACKAGE_TOOL, "salt_gemma4_package")
            source_module = package._load_mlx_source_module()
            plan = package.load_json(args.plan)
            package.validate_plan(plan)
            source_summary = source_module.validate_source(args.source_contract)
            if source_summary["contract_sha256"] != SOURCE_CONTRACT_SHA256 or \
                    source_summary["runtime_ready"] is not False:
                raise QaError("authenticated source summary drift")
            report, handles, identities = source_module._authenticate_downloads(
                args.source_dir, source_summary
            )
            if report["authenticated_bytes"] != 15_373_375_402 or \
                    report["runtime_ready"] is not False:
                raise QaError("download authentication summary drift")
            role_map = package.build_mlx_source_role_map(
                plan,
                source_contract=args.source_contract,
                role_contract=args.role_contract,
                source_summary=source_summary,
            )
            if role_map["runtime_ready"] is not False:
                raise QaError("role map unexpectedly runtime-ready")
            pool_handle, pool_identity, _ = _validate_pool(
                args.pool, receipt, source_summary
            )
        if pool_handle is None:
            raise QaError("expert pool handle was not retained")
        binding = _build_binding(
            role_map, handles, identities, pool_handle, pool_identity,
            kv_compatibility.hexdigest, build_identity.hexdigest,
        )
        if args.verify_only:
            print("GEMMA4_QA_BINDING_OK")
            print(f"authenticated_source_bytes={report['authenticated_bytes']}")
            print(f"pool_bytes={pool_identity[2]}")
            print("text_role_count=597")
            print(f"role_map_sha256={role_map['role_map_sha256']}")
            print(f"kv_compatibility_sha256={kv_compatibility.hexdigest}")
            print(f"build_identity_sha256={build_identity.hexdigest}")
            print("runtime_ready=true")
            return 0
        dpr_reference_gate = (
            args.ctx == 512 and (
                (args.dpr_proof > 0 and args.gen == args.dpr_proof + 1) or
                (args.dpr_prefill_reference_save is not None and args.gen == 1)
            )
        )
        if (args.ctx, args.gen) not in ALLOWED_QA_GATES and not dpr_reference_gate:
            raise QaError(f"unsupported QA gate CTX{args.ctx}/O{args.gen}")
        if args.workers < 1 or args.workers > 8:
            raise QaError("--workers must be in [1, 8]")
        if not args.runner.is_file():
            raise QaError(f"Gemma QA runner does not exist: {args.runner}")
        prompt = (_load_prompt(args.prompt_file) if args.prompt_file else
                  _render_prompt(args.question or
                                 "What is the capital of France? Answer briefly."))
        prompt_bytes = prompt.encode("utf-8")
        print(
            "GEMMA4_QA_PROMPT "
            f"bytes={len(prompt_bytes)} "
            f"sha256={hashlib.sha256(prompt_bytes).hexdigest()} "
            f"text={json.dumps(prompt, ensure_ascii=True)}",
            file=sys.stderr,
        )
        inherited = tuple(handle.fileno() for handle in handles.values()) + (
            pool_handle.fileno(),
        )
        command = [str(args.runner), "--ctx", str(args.ctx), "--gen", str(args.gen),
                   "--prompt", prompt, "--workers", str(args.workers)]
        if args.load_only:
            command.append("--load-only")
        if args.compare_kv_warm:
            command.append("--compare-kv-warm")
        if args.dpr_proof:
            command.extend(("--dpr-proof", str(args.dpr_proof)))
        if optimized_decode:
            command.extend(("--decode-candidates", args.decode_candidates))
        if args.target_frontier:
            command.append("--target-frontier")
        if args.dpr_reference_save:
            command.extend(("--dpr-reference-save",
                            str(args.dpr_reference_save)))
        if args.dpr_reference_ids:
            command.extend(("--dpr-reference-ids", args.dpr_reference_ids))
        if args.dpr_prefill_reference_save:
            command.extend((
                "--dpr-prefill-reference-save",
                str(args.dpr_prefill_reference_save),
                "--dpr-prefill-tokens", str(args.dpr_prefill_tokens),
            ))
        if args.package_proof:
            command.append("--package-proof")
        if args.ondemand_rope_proof:
            command.append("--ondemand-rope-proof")

        if args.kv_load:
            command.extend(("--kv-load", str(args.kv_load)))
        if args.kv_save:
            command.extend(("--kv-save", str(args.kv_save)))
        return _run_native(
            command, binding, inherited,
            serial_prefill_proof=args.serial_prefill_proof,
        )
    except Exception as exc:
        if isinstance(exc, QaError) or exc.__class__.__name__ == "ContractError":
            print(f"gemma4 qa: {exc}", file=sys.stderr)
            return 2
        raise
    finally:
        if pool_handle is not None:
            pool_handle.close()
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
