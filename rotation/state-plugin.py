#!/usr/bin/env python3
"""Materialize or verify an immutable Gemma G4KVC006 state plugin."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import kv_sidecar  # type: ignore  # noqa: E402

SCHEMA = "salt.gemma4.state-plugin.v1"
CHUNK = 1024 * 1024
IDENTITY = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
HEX256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _atomic_json(path: Path, value: dict) -> None:
    rendered = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def materialize(args: argparse.Namespace) -> dict:
    source = args.sidecar.resolve(strict=True)
    output = args.output.resolve()
    manifest = Path(str(output) + ".plugin.json")
    if not IDENTITY.fullmatch(args.id) or not IDENTITY.fullmatch(args.version):
        raise ValueError("plugin id/version must be lowercase identifier strings")
    if not source.is_dir() or source.is_symlink():
        raise ValueError("sidecar must be a real directory")
    if output.suffix != ".g4kv":
        raise ValueError("state plugin output must end in .g4kv")
    if output.exists() or manifest.exists():
        raise FileExistsError("state plugin output already exists")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    sidecar = kv_sidecar.read(source, verify_payload=False)
    metadata = sidecar["metadata"]
    native = metadata["native_schema"]
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temp = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with kv_sidecar.native_path(source) as framed, os.fdopen(fd, "wb") as target, framed.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, target, length=CHUNK)
            target.flush()
            os.fsync(target.fileno())
        digest, size = _sha256(temp)
        if size != native["bytes"] or not HEX256.fullmatch(digest):
            raise ValueError("materialized state size/digest is invalid")
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    value = {
        "schema": SCHEMA,
        "plugin_id": args.id,
        "version": args.version,
        "model_id": "gemma-4-26b-a4b-it",
        "quant_track": "mlx-affine-q4",
        "state_file": output.name,
        "state_bytes": size,
        "state_file_sha256": digest,
        "position": native["position"],
        "context_tokens": native["context_tokens"],
        "compatibility_sha256": native["compatibility_sha256"],
        "logical_state_sha256": native["logical_state_sha256"],
        "payload_sha256": native["payload_sha256"],
        "source_build_identity_sha256": native["build_identity_sha256"],
        "source_created_at": metadata["created_at"],
    }
    _atomic_json(manifest, value)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {**value, "manifest": str(manifest), "state_path": str(output)}


def verify(args: argparse.Namespace) -> dict:
    manifest = args.plugin.resolve(strict=True)
    value = json.loads(manifest.read_text(encoding="ascii"))
    required = {
        "schema", "plugin_id", "version", "model_id", "quant_track",
        "state_file", "state_bytes", "state_file_sha256", "position",
        "context_tokens", "compatibility_sha256", "logical_state_sha256",
        "payload_sha256", "source_build_identity_sha256", "source_created_at",
    }
    if set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid state plugin manifest")
    if not IDENTITY.fullmatch(value["plugin_id"]) or not IDENTITY.fullmatch(value["version"]):
        raise ValueError("invalid state plugin identity")
    state_path = manifest.parent / value["state_file"]
    info = state_path.lstat()
    if not stat.S_ISREG(info.st_mode) or state_path.is_symlink() or info.st_mode & 0o077:
        raise ValueError("state plugin must be a private regular non-symlink file")
    digest, size = _sha256(state_path)
    if size != value["state_bytes"] or digest != value["state_file_sha256"]:
        raise ValueError("state plugin file identity mismatch")
    for key in (
        "state_file_sha256", "compatibility_sha256", "logical_state_sha256",
        "payload_sha256", "source_build_identity_sha256",
    ):
        if not HEX256.fullmatch(value[key]):
            raise ValueError(f"invalid {key}")
    return {**value, "manifest": str(manifest), "state_path": str(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("materialize")
    create.add_argument("--sidecar", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--id", required=True)
    create.add_argument("--version", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--plugin", type=Path, required=True)
    args = parser.parse_args()
    old_umask = os.umask(0o077)
    try:
        result = materialize(args) if args.command == "materialize" else verify(args)
    finally:
        os.umask(old_umask)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
