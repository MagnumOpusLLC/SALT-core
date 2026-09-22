#!/usr/bin/env python3
"""Authoritative Gemma 4 portable-KV compatibility fingerprint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Protocol

SCHEMA = "salt.gemma4.kv-compat.v5"
MODEL = "gemma-4-26b-a4b-it"
DOMAIN = b"SALT_GEMMA4_KV_COMPAT_V5\0"
MAX_MANIFEST_BYTES = 128 * 1024
MAX_CONTRACT_FILE_BYTES = 16 * 1024 * 1024
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
_RELOCATED_CONTRACT_FILES = {
    "models/gemma4-26b-a4b/gemma4.c":
        "models/gemma4-26b-a4b/src/gemma4.c",
    "models/gemma4-26b-a4b/gemma4.h":
        "models/gemma4-26b-a4b/src/gemma4.h",
    "models/gemma4-26b-a4b/gemma4_kv_file.h":
        "models/gemma4-26b-a4b/src/gemma4_kv_file.h",
    "models/gemma4-26b-a4b/gemma4_text.c":
        "models/gemma4-26b-a4b/src/gemma4_text.c",
    "models/gemma4-26b-a4b/gemma4_text.h":
        "models/gemma4-26b-a4b/src/gemma4_text.h",
    "models/gemma4-26b-a4b/gemma4_vision.c":
        "models/gemma4-26b-a4b/src/gemma4_vision.c",
    "models/gemma4-26b-a4b/gemma4_vision.h":
        "models/gemma4-26b-a4b/src/gemma4_vision.h",
    "models/gemma4-26b-a4b/model.c":
        "models/gemma4-26b-a4b/src/model.c",
    "models/gemma4-26b-a4b/server.c":
        "models/gemma4-26b-a4b/src/inference.c",
    "server/gemma4-launch.sh": "server/model/gemma4-launch.sh",
    "server/gemma4_backend.py": "server/model/gemma4_backend.py",
    "server/gemma4_build.py": "server/model/gemma4_build.py",
    "server/gemma4_compat.py": "server/model/gemma4_compat.py",
    "server/gemma4_mentor.py": "server/model/gemma4_mentor.py",
    "server/gemma4_state_artifact.py": "server/model/gemma4_state_artifact.py",
}
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
_REQUIRED_KEYS = {
    "schema",
    "model",
    "cache_abi",
    "source_authority",
    "tokenizer_control",
    "chat_template_abi",
    "engine_contract_files",
    "legacy_read_abis",
}


class CompatibilityError(RuntimeError):
    """The compatibility authority is malformed, unsafe, or unstable."""


class _Hasher(Protocol):
    def update(self, value: bytes, /) -> None: ...


@dataclass(frozen=True)
class Compatibility:
    manifest_path: Path
    manifest: dict
    digest: bytes
    hexdigest: str
    namespace: str
    file_sha256: tuple[tuple[str, str], ...]


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stable_regular_read(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise CompatibilityError(f"{label} must be a regular non-symlink file")
    except FileNotFoundError as exc:
        raise CompatibilityError(f"{label} does not exist: {path}") from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CompatibilityError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum:
            raise CompatibilityError(f"{label} must be a bounded regular non-symlink file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise CompatibilityError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise CompatibilityError(f"{label} grew while reading")
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise CompatibilityError(f"{label} identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _repository_path(root: Path, relative: object) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CompatibilityError("engine contract file must be a relative repository path")
    value = Path(relative)
    if value.is_absolute() or any(part in ("", ".", "..") for part in value.parts):
        raise CompatibilityError("engine contract file must be a relative repository path")
    candidate = root.joinpath(*value.parts)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    relocated = _RELOCATED_CONTRACT_FILES.get(relative)
    if relocated is not None:
        candidate = root.joinpath(*Path(relocated).parts)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    try:
        if stat.S_ISLNK(candidate.lstat().st_mode):
            raise CompatibilityError(
                f"engine contract file must be a regular non-symlink file: {relative}"
            )
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CompatibilityError(f"engine contract file does not exist: {relative}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CompatibilityError("engine contract file must be a relative repository path") from exc
    if resolved != candidate:
        raise CompatibilityError(
            f"engine contract file must be a regular non-symlink file: {relative}"
        )
    return relative, candidate


def _validate_manifest(value: object) -> list[str]:
    if not isinstance(value, dict) or set(value) != _REQUIRED_KEYS:
        raise CompatibilityError("compatibility manifest fields are incomplete or unsupported")
    if value["schema"] != SCHEMA or value["model"] != MODEL:
        raise CompatibilityError("compatibility schema/model mismatch")
    if value["cache_abi"] != {
        "magic": "G4KVC006",
        "version": 6,
        "header_bytes": 256,
        "payload": (
            "hybrid-shared-projection-v-base-finite-ieee754-binary32-"
            "little-endian"
        ),
    }:
        raise CompatibilityError("portable KV cache ABI mismatch")
    legacy = value["legacy_read_abis"]
    if (
        not isinstance(legacy, list)
        or len(legacy) != 2
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"magic", "version", "compatibility_sha256"}
            or not isinstance(entry["compatibility_sha256"], str)
            or _HEX_256.fullmatch(entry["compatibility_sha256"]) is None
            for entry in legacy
        )
        or [(entry["magic"], entry["version"]) for entry in legacy] != [
            ("G4KVC005", 5), ("G4KVC004", 4),
        ]
    ):
        raise CompatibilityError("legacy portable KV read ABI is malformed")
    authority = value["source_authority"]
    authority_keys = {
        "source_contract_sha256", "role_map_sha256", "copy_map_sha256",
    }
    if not isinstance(authority, dict) or set(authority) != authority_keys or any(
        not isinstance(authority[key], str) or _HEX_256.fullmatch(authority[key]) is None
        for key in authority_keys
    ):
        raise CompatibilityError("source authority is malformed")
    if value["tokenizer_control"] != {
        "bos": "<bos>",
        "turn_open": "<|turn>",
        "turn_close": "<turn|>",
        "thought_open": "<|channel>thought",
        "channel_close": "<channel|>",
        "stop_token_ids": [1, 50, 106],
    }:
        raise CompatibilityError("tokenizer/control-token contract mismatch")
    template_abi = value["chat_template_abi"]
    if not isinstance(template_abi, str) or not template_abi or len(template_abi) > 256:
        raise CompatibilityError("chat template ABI is malformed")
    files = value["engine_contract_files"]
    if not isinstance(files, list) or not files or len(files) > 128 or \
            any(not isinstance(item, str) for item in files) or \
            files != sorted(files) or len(files) != len(set(files)):
        raise CompatibilityError("engine contract files must be unique and sorted")
    return files


def _digest_field(hasher: _Hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def load_compatibility(manifest_path: Path, repo_root: Path) -> Compatibility:
    root = Path(repo_root).resolve(strict=True)
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = Path(os.path.abspath(manifest))
    manifest = manifest.parent.resolve(strict=True) / manifest.name
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise CompatibilityError("compatibility manifest must be inside the repository") from exc
    raw_manifest = _stable_regular_read(
        manifest, label="compatibility manifest", maximum=MAX_MANIFEST_BYTES,
    )
    try:
        value = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"compatibility manifest is invalid JSON: {exc}") from exc
    files = _validate_manifest(value)

    hasher = hashlib.sha256()
    hasher.update(DOMAIN)
    _digest_field(hasher, raw_manifest)
    file_hashes: list[tuple[str, str]] = []
    for relative in files:
        name, path = _repository_path(root, relative)
        raw = _stable_regular_read(
            path, label=f"engine contract file {name!r}",
            maximum=MAX_CONTRACT_FILE_BYTES,
        )
        file_digest = hashlib.sha256(raw).digest()
        encoded_name = name.encode("utf-8")
        _digest_field(hasher, encoded_name)
        _digest_field(hasher, len(raw).to_bytes(8, "little"))
        _digest_field(hasher, file_digest)
        file_hashes.append((name, file_digest.hex()))

    digest = hasher.digest()
    hexdigest = digest.hex()
    return Compatibility(
        manifest_path=manifest,
        manifest=value,
        digest=digest,
        hexdigest=hexdigest,
        namespace="g4kvc006-" + hexdigest,
        file_sha256=tuple(file_hashes),
    )


# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
load_unprojected_compatibility = load_compatibility


def _load_compatibility_with_projection(
        manifest_path: Path, repo_root: Path) -> Compatibility:
    """Use the strict physical-build projection for the production manifest."""
    import importlib.util
    import sys

    root = Path(repo_root).resolve(strict=True)
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = Path(os.path.abspath(manifest))
    manifest = manifest.parent.resolve(strict=True) / manifest.name
    production = root / "models" / "gemma4-26b-a4b" / "kv-cache-compat.json"
    if manifest != production:
        return load_unprojected_compatibility(manifest, root)

    tool = root / "server" / "model" / "gemma4_compat_projection.py"
    try:
        if stat.S_ISLNK(tool.lstat().st_mode):
            raise CompatibilityError("compatibility projection tool must not be a symlink")
    except FileNotFoundError as exc:
        raise CompatibilityError("compatibility projection tool does not exist") from exc
    spec = importlib.util.spec_from_file_location(
        f"salt_gemma4_compat_projection_{id(root):x}", tool,
    )
    if spec is None or spec.loader is None:
        raise CompatibilityError("cannot load compatibility projection authority")
    module = importlib.util.module_from_spec(spec)
    current = sys.modules.get(__name__)
    if current is None:
        raise CompatibilityError("compatibility authority module identity is unavailable")
    previous = sys.modules.get("gemma4_compat")
    sys.modules["gemma4_compat"] = current
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["gemma4_compat"]
        else:
            sys.modules["gemma4_compat"] = previous
    projection = root / "models" / "gemma4-26b-a4b" / \
        "kv-cache-compat-projection.json"
    authority, _ = module.load_projected_compatibility(
        manifest, root, projection,
    )
    return authority


load_compatibility = _load_compatibility_with_projection
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    root = Path(__file__).resolve().parents[2]
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    manifest = root / "models" / "gemma4-26b-a4b" / "kv-cache-compat.json"
    return root, manifest


def main(argv: list[str] | None = None) -> int:
    root, manifest = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=manifest)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--hex", action="store_true")
    output.add_argument("--files", action="store_true")
    output.add_argument("--legacy-v5-hex", action="store_true")
    output.add_argument("--legacy-v4-hex", action="store_true")
    args = parser.parse_args(argv)
    authority = load_compatibility(args.manifest, args.repo_root)
    if args.hex:
        print(authority.hexdigest)
    elif args.legacy_v5_hex or args.legacy_v4_hex:
        magic = "G4KVC005" if args.legacy_v5_hex else "G4KVC004"
        print(next(
            entry["compatibility_sha256"]
            for entry in authority.manifest["legacy_read_abis"]
            if entry["magic"] == magic
        ))
    else:
        print(" ".join(item[0] for item in authority.file_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
