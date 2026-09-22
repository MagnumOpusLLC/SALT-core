#!/usr/bin/env python3
"""Project physical-only Makefile additions out of Gemma KV compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import gemma4_compat as base

SCHEMA = "salt.gemma4.kv-compat-projection.v1"
MAX_PROJECTION_BYTES = 64 * 1024
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = {"schema", "compatibility_manifest", "overrides"}
_BLOCK_OVERRIDE_KEYS = {
    "logical_path",
    "snapshot_path",
    "snapshot_sha256",
    "live_projection",
    "physical_blocks_sha256",
}
_PINNED_OVERRIDE_KEYS = {
    "logical_path",
    "snapshot_path",
    "snapshot_sha256",
    "live_projection",
    "live_sha256",
}
_BLOCK_PROJECTION = "strip-physical-build-blocks-v1"
_PINNED_PROJECTION = "pin-live-source-v1"
_MARKERS = {
    b"# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1\n":
        b"# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1\n",
    b"/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1 */\n":
        b"/* SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1 */\n",
}


def _inside_repo_file(root: Path, path: Path, *, label: str, maximum: int) -> tuple[str, Path, bytes]:
    root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    candidate = Path(os.path.abspath(candidate))
    candidate = candidate.parent.resolve(strict=True) / candidate.name
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise base.CompatibilityError(f"{label} must be inside the repository") from exc
    name, verified = base._repository_path(root, relative)
    raw = base._stable_regular_read(verified, label=label, maximum=maximum)
    return name, verified, raw


def _strip_physical_blocks(raw: bytes) -> tuple[bytes, bytes]:
    projected: list[bytes] = []
    blocks: list[bytes] = []
    active: list[bytes] | None = None
    expected_end: bytes | None = None
    for line in raw.splitlines(keepends=True):
        if line in _MARKERS:
            if active is not None:
                raise base.CompatibilityError("physical build blocks may not nest")
            active = [line]
            expected_end = _MARKERS[line]
        elif line in _MARKERS.values():
            if active is None:
                raise base.CompatibilityError("physical build block end has no begin")
            if line != expected_end:
                raise base.CompatibilityError("physical build block marker type mismatch")
            active.append(line)
            blocks.append(b"".join(active))
            active = None
            expected_end = None
        elif active is None:
            projected.append(line)
        else:
            active.append(line)
    if active is not None:
        raise base.CompatibilityError("physical build block is unterminated")
    if not blocks:
        raise base.CompatibilityError("physical build projection found no blocks")
    return b"".join(projected), b"".join(blocks)


def _load_projection(path: Path, root: Path) -> tuple[str, dict, bytes]:
    name, _, raw = _inside_repo_file(
        root, path, label="compatibility projection manifest",
        maximum=MAX_PROJECTION_BYTES,
    )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise base.CompatibilityError(
            f"compatibility projection manifest is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or set(value) != _REQUIRED_KEYS:
        raise base.CompatibilityError("compatibility projection fields are incomplete or unsupported")
    if value["schema"] != SCHEMA:
        raise base.CompatibilityError("compatibility projection schema mismatch")
    overrides = value["overrides"]
    if not isinstance(overrides, list) or not overrides or len(overrides) > 16:
        raise base.CompatibilityError("compatibility projection overrides are malformed")
    logical_paths: list[str] = []
    for override in overrides:
        if not isinstance(override, dict):
            raise base.CompatibilityError("compatibility projection override is malformed")
        mode = override.get("live_projection")
        expected_keys = (
            _BLOCK_OVERRIDE_KEYS if mode == _BLOCK_PROJECTION else
            _PINNED_OVERRIDE_KEYS if mode == _PINNED_PROJECTION else None
        )
        if expected_keys is None or set(override) != expected_keys:
            raise base.CompatibilityError("compatibility projection override is malformed")
        logical = override["logical_path"]
        snapshot = override["snapshot_path"]
        if not isinstance(logical, str) or not isinstance(snapshot, str):
            raise base.CompatibilityError("compatibility projection paths are malformed")
        identity_key = (
            "physical_blocks_sha256" if mode == _BLOCK_PROJECTION
            else "live_sha256"
        )
        if any(
            not isinstance(override[key], str)
            or _HEX_256.fullmatch(override[key]) is None
            for key in ("snapshot_sha256", identity_key)
        ):
            raise base.CompatibilityError("compatibility projection override identity is malformed")
        logical_paths.append(logical)
    if logical_paths != sorted(logical_paths) or len(logical_paths) != len(set(logical_paths)):
        raise base.CompatibilityError("compatibility projection overrides must be unique and sorted")
    return name, value, raw


def load_projected_compatibility(
        manifest_path: Path, repo_root: Path, projection_path: Path,
) -> tuple[base.Compatibility, tuple[str, ...]]:
    root = Path(repo_root).resolve(strict=True)
    manifest_name, manifest, raw_manifest = _inside_repo_file(
        root, Path(manifest_path), label="compatibility manifest",
        maximum=base.MAX_MANIFEST_BYTES,
    )
    try:
        manifest_value = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise base.CompatibilityError(f"compatibility manifest is invalid JSON: {exc}") from exc
    files = base._validate_manifest(manifest_value)
    projection_name, projection, _ = _load_projection(Path(projection_path), root)
    if projection["compatibility_manifest"] != manifest_name:
        raise base.CompatibilityError("compatibility projection targets the wrong manifest")
    override_by_name = {entry["logical_path"]: entry for entry in projection["overrides"]}
    if any(name not in files for name in override_by_name):
        raise base.CompatibilityError("compatibility projection overrides a non-contract file")

    hasher = hashlib.sha256()
    hasher.update(base.DOMAIN)
    base._digest_field(hasher, raw_manifest)
    file_hashes: list[tuple[str, str]] = []
    dependencies: set[str] = set()
    dependencies.add(manifest_name)
    dependencies.add(projection_name)

    for relative in files:
        name, live_path = base._repository_path(root, relative)
        dependencies.add(live_path.relative_to(root).as_posix())
        live_raw = base._stable_regular_read(
            live_path, label=f"engine contract file {name!r}",
            maximum=base.MAX_CONTRACT_FILE_BYTES,
        )
        override = override_by_name.get(name)
        if override is not None:
            snapshot_name, snapshot_path = base._repository_path(
                root, override["snapshot_path"],
            )
            snapshot_raw = base._stable_regular_read(
                snapshot_path, label=f"compatibility snapshot {snapshot_name!r}",
                maximum=base.MAX_CONTRACT_FILE_BYTES,
            )
            if hashlib.sha256(snapshot_raw).hexdigest() != override["snapshot_sha256"]:
                raise base.CompatibilityError("compatibility snapshot SHA-256 mismatch")
            if override["live_projection"] == _BLOCK_PROJECTION:
                projected, blocks = _strip_physical_blocks(live_raw)
                if projected != snapshot_raw:
                    raise base.CompatibilityError(
                        f"engine contract file {name!r} changed outside physical build blocks"
                    )
                if hashlib.sha256(blocks).hexdigest() != override["physical_blocks_sha256"]:
                    raise base.CompatibilityError("physical build block SHA-256 mismatch")
            elif hashlib.sha256(live_raw).hexdigest() != override["live_sha256"]:
                raise base.CompatibilityError("pinned live source SHA-256 mismatch")
            raw = snapshot_raw
            dependencies.add(snapshot_name)
        else:
            raw = live_raw
        file_digest = hashlib.sha256(raw).digest()
        encoded_name = name.encode("utf-8")
        base._digest_field(hasher, encoded_name)
        base._digest_field(hasher, len(raw).to_bytes(8, "little"))
        base._digest_field(hasher, file_digest)
        file_hashes.append((name, file_digest.hex()))

    digest = hasher.digest()
    authority = base.Compatibility(
        manifest_path=root / manifest_name,
        manifest=manifest_value,
        digest=digest,
        hexdigest=digest.hex(),
        namespace="g4kvc006-" + digest.hex(),
        file_sha256=tuple(file_hashes),
    )
    return authority, tuple(sorted(dependencies))


def _default_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "models" / "gemma4-26b-a4b" / "kv-cache-compat.json"
    projection = root / "models" / "gemma4-26b-a4b" / "kv-cache-compat-projection.json"
    return root, manifest, projection


def main(argv: list[str] | None = None) -> int:
    root, manifest, projection = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=manifest)
    parser.add_argument("--projection", type=Path, default=projection)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--hex", action="store_true")
    output.add_argument("--files", action="store_true")
    output.add_argument("--legacy-v5-hex", action="store_true")
    output.add_argument("--legacy-v4-hex", action="store_true")
    args = parser.parse_args(argv)
    authority, dependencies = load_projected_compatibility(
        args.manifest, args.repo_root, args.projection,
    )
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
        print(" ".join(dependencies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
