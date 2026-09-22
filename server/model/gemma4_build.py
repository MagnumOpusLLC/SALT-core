#!/usr/bin/env python3
"""Create and verify Gemma 4 native build identity receipts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import tempfile

SCHEMA = "salt.gemma4.build.v1"
DOMAIN = b"SALT_GEMMA4_BUILD_V1\0"
MAX_RECEIPT_BYTES = 64 * 1024
MAX_BINARY_BYTES = 256 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_KEYS = {
    "schema", "binary_name", "binary_sha256", "compiler_command",
    "compiler_executable", "compiler_version", "flags", "target",
    "math_abi", "blas_abi", "source_compatibility_sha256",
    "identity_sha256", "runtime_ready",
}


class BuildIdentityError(RuntimeError):
    """A build receipt or its native binary is malformed or stale."""


@dataclass(frozen=True)
class BuildIdentity:
    receipt_path: Path
    binary_path: Path
    hexdigest: str
    binary_sha256: str
    compiler_version: str
    flags: str
    target: str
    math_abi: str
    blas_abi: str
    source_compatibility_sha256: str


def _hex256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _stable_read(path: Path, maximum: int, label: str) -> bytes:
    path = Path(path)
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise BuildIdentityError(f"{label} must be a regular non-symlink file")
    except FileNotFoundError as exc:
        raise BuildIdentityError(f"{label} does not exist: {path}") from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BuildIdentityError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum:
            raise BuildIdentityError(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise BuildIdentityError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise BuildIdentityError(f"{label} grew while reading")
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise BuildIdentityError(f"{label} identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _canonical_payload(value: dict) -> bytes:
    payload = {key: value[key] for key in sorted(value) if key != "identity_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _identity_digest(value: dict) -> str:
    return hashlib.sha256(DOMAIN + _canonical_payload(value)).hexdigest()


def _bounded_line(command: list[str], flag: str) -> str:
    try:
        result = subprocess.run(
            [*command, flag], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildIdentityError(f"compiler metadata query failed: {flag}") from exc
    line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    if not line or len(line) > 1024:
        raise BuildIdentityError(f"compiler metadata is malformed: {flag}")
    return line


def _compiler_target(command: list[str]) -> str:
    for flag in ("-dumpmachine", "-print-target-triple"):
        try:
            return _bounded_line(command, flag)
        except BuildIdentityError:
            continue
    raise BuildIdentityError("compiler target query failed")


def create_build_receipt(*, binary: Path, receipt: Path, compiler: str,
                         flags: str, source_compatibility_sha256: str,
                         math_abi: str, blas_abi: str) -> BuildIdentity:
    if not _hex256(source_compatibility_sha256):
        raise BuildIdentityError("source compatibility SHA-256 is malformed")
    command = shlex.split(compiler)
    if not command:
        raise BuildIdentityError("compiler command is empty")
    executable = shutil.which(command[0])
    if executable is None:
        raise BuildIdentityError(f"compiler executable not found: {command[0]}")
    command[0] = str(Path(executable).resolve())
    binary_path = Path(binary).resolve(strict=True)
    raw_binary = _stable_read(binary_path, MAX_BINARY_BYTES, "native binary")
    value = {
        "schema": SCHEMA,
        "binary_name": binary_path.name,
        "binary_sha256": hashlib.sha256(raw_binary).hexdigest(),
        "compiler_command": command,
        "compiler_executable": command[0],
        "compiler_version": _bounded_line(command, "--version"),
        "flags": flags,
        "target": _compiler_target(command),
        "math_abi": math_abi,
        "blas_abi": blas_abi,
        "source_compatibility_sha256": source_compatibility_sha256,
        "runtime_ready": False,
    }
    if not flags or not math_abi or not blas_abi:
        raise BuildIdentityError("flags/math/BLAS identity must be non-empty")
    value["identity_sha256"] = _identity_digest(value)
    receipt_path = Path(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    fd, temporary = tempfile.mkstemp(prefix=f".{receipt_path.name}.", dir=receipt_path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return load_build_receipt(
        receipt_path, binary_path,
        expected_source_compatibility=source_compatibility_sha256,
    )


def load_build_receipt(receipt: Path, binary: Path, *,
                       expected_source_compatibility: str) -> BuildIdentity:
    if not _hex256(expected_source_compatibility):
        raise BuildIdentityError("expected source compatibility SHA-256 is malformed")
    receipt_path = Path(receipt).resolve(strict=True)
    binary_path = Path(binary).resolve(strict=True)
    raw = _stable_read(receipt_path, MAX_RECEIPT_BYTES, "build receipt")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildIdentityError("build receipt is invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != _KEYS:
        raise BuildIdentityError("build receipt fields are incomplete or unsupported")
    strings = (
        "binary_name", "compiler_executable", "compiler_version", "flags",
        "target", "math_abi", "blas_abi",
    )
    if value["schema"] != SCHEMA or value["runtime_ready"] is not False or any(
        not isinstance(value[key], str) or not value[key] or len(value[key]) > 4096
        for key in strings
    ):
        raise BuildIdentityError("build receipt metadata is malformed")
    command = value["compiler_command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item or len(item) > 1024 for item in command
    ):
        raise BuildIdentityError("build compiler command is malformed")
    if value["binary_name"] != binary_path.name:
        raise BuildIdentityError("build receipt binary name mismatch")
    if value["source_compatibility_sha256"] != expected_source_compatibility:
        raise BuildIdentityError("build/source compatibility identity mismatch")
    if not _hex256(value["binary_sha256"]) or not _hex256(value["identity_sha256"]):
        raise BuildIdentityError("build receipt digest is malformed")
    raw_binary = _stable_read(binary_path, MAX_BINARY_BYTES, "native binary")
    if hashlib.sha256(raw_binary).hexdigest() != value["binary_sha256"]:
        raise BuildIdentityError("native binary SHA-256 mismatch")
    expected_identity = _identity_digest(value)
    if value["identity_sha256"] != expected_identity:
        raise BuildIdentityError("build identity digest mismatch")
    return BuildIdentity(
        receipt_path=receipt_path, binary_path=binary_path,
        hexdigest=expected_identity, binary_sha256=value["binary_sha256"],
        compiler_version=value["compiler_version"], flags=value["flags"],
        target=value["target"], math_abi=value["math_abi"],
        blas_abi=value["blas_abi"],
        source_compatibility_sha256=value["source_compatibility_sha256"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--binary", type=Path, required=True)
    create.add_argument("--receipt", type=Path, required=True)
    create.add_argument("--compiler", required=True)
    create.add_argument("--flags", required=True)
    create.add_argument("--source-compatibility", required=True)
    create.add_argument("--math-abi", required=True)
    create.add_argument("--blas-abi", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--binary", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--source-compatibility", required=True)
    args = parser.parse_args(argv)
    if args.action == "create":
        identity = create_build_receipt(
            binary=args.binary, receipt=args.receipt, compiler=args.compiler,
            flags=args.flags,
            source_compatibility_sha256=args.source_compatibility,
            math_abi=args.math_abi, blas_abi=args.blas_abi,
        )
    else:
        identity = load_build_receipt(
            args.receipt, args.binary,
            expected_source_compatibility=args.source_compatibility,
        )
    print(identity.hexdigest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
