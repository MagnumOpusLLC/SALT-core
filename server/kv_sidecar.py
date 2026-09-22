"""Server-owned metadata around opaque native KV payloads.

No model arithmetic, token selection or state transitions live here. Native
framing is retained as sidecar schema and reassembled only for explicit import.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import BinaryIO, Iterator

SCHEMA = "salt.kv-sidecar.v1"
CHUNK = 1024 * 1024
MAX_METADATA = 1024 * 1024


class SidecarError(ValueError):
    pass


def _directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SidecarError("KV artifact directory must not be a symlink")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise SidecarError("KV artifact directory owner mismatch")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise SidecarError("KV artifact directory must be private")


@contextmanager
def _reader(path: Path) -> Iterator[BinaryIO]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SidecarError("KV artifact must be a regular non-symlink file")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise SidecarError("KV artifact file owner mismatch")
        yield stream
        after = os.fstat(stream.fileno())
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after):
            raise SidecarError("KV artifact changed during read")


def _writer(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    return os.fdopen(fd, "wb")


def read(directory: Path, *, verify_payload: bool = True) -> dict:
    _directory(directory)
    with _reader(directory / "metadata.json") as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise SidecarError("KV metadata exceeds limit")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise SidecarError("invalid KV metadata JSON") from exc
    if (not isinstance(value, dict) or set(value) != {
            "schema", "payload_file", "payload_bytes", "payload_sha256",
            "native_header_hex", "metadata"} or value["schema"] != SCHEMA or
            not isinstance(value["metadata"], dict) or
            type(value["payload_bytes"]) is not int or value["payload_bytes"] < 1 or
            not isinstance(value["payload_sha256"], str) or
            re.fullmatch(r"[0-9a-f]{64}", value["payload_sha256"]) is None or
            value["payload_file"] != value["payload_sha256"] + ".kv" or
            not isinstance(value["native_header_hex"], str) or
            not value["native_header_hex"] or
            len(value["native_header_hex"]) > 8192 or
            len(value["native_header_hex"]) % 2 or
            re.fullmatch(r"[0-9a-f]+", value["native_header_hex"]) is None):
        raise SidecarError("invalid KV sidecar schema")
    with _reader(directory / value["payload_file"]) as stream:
        if os.fstat(stream.fileno()).st_size != value["payload_bytes"]:
            raise SidecarError("KV payload byte count mismatch")
        if verify_payload:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                digest.update(chunk)
            if digest.hexdigest() != value["payload_sha256"]:
                raise SidecarError("KV payload SHA-256 mismatch")
    return value


def publish(native_path: Path, directory: Path, *, header_bytes: int,
            expected_payload_sha256: str, metadata: dict,
            replace_existing: bool = False) -> dict:
    if not 0 < header_bytes <= 4096:
        raise SidecarError("invalid native framing size")
    _directory(directory.parent)
    old = None
    if directory.exists() or directory.is_symlink():
        if not replace_existing:
            raise SidecarError("KV cache already exists")
        old = read(directory)
    staging = Path(tempfile.mkdtemp(prefix=".kv-publish-", dir=directory.parent))
    try:
        with _reader(native_path) as source, _writer(staging / "payload") as target:
            header = source.read(header_bytes)
            if len(header) != header_bytes:
                raise SidecarError("native KV header is truncated")
            digest = hashlib.sha256()
            count = 0
            for chunk in iter(lambda: source.read(CHUNK), b""):
                target.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        payload_sha = digest.hexdigest()
        if count < 1 or payload_sha != expected_payload_sha256:
            raise SidecarError("KV payload SHA-256 mismatch")
        filename = payload_sha + ".kv"
        os.rename(staging / "payload", staging / filename)
        value = {"schema": SCHEMA, "payload_file": filename,
                 "payload_bytes": count, "payload_sha256": payload_sha,
                 "native_header_hex": header.hex(), "metadata": metadata}
        rendered = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        if len(rendered) > MAX_METADATA:
            raise SidecarError("KV metadata exceeds limit")
        with _writer(staging / "metadata.json") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        if old is None:
            if directory.exists() or directory.is_symlink():
                raise SidecarError("KV cache already exists")
            os.rename(staging, directory)
        else:
            # Content is immutable; the metadata rename is the publication point.
            os.replace(staging / filename, directory / filename)
            os.replace(staging / "metadata.json", directory / "metadata.json")
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        fd = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = read(directory)
        if old is not None and old["payload_file"] != filename:
            (directory / old["payload_file"]).unlink()
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@contextmanager
def native_path(directory: Path) -> Iterator[Path]:
    """Frame one explicit import; never retain a second payload copy."""
    value = read(directory, verify_payload=False)
    fd, name = tempfile.mkstemp(prefix=".kv-import-", dir=directory.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as target, _reader(
                directory / value["payload_file"]) as source:
            target.write(bytes.fromhex(value["native_header_hex"]))
            digest = hashlib.sha256()
            count = 0
            for chunk in iter(lambda: source.read(CHUNK), b""):
                target.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            if count != value["payload_bytes"] or digest.hexdigest() != value["payload_sha256"]:
                raise SidecarError("KV payload SHA-256 mismatch")
        yield path
    finally:
        path.unlink(missing_ok=True)


def delete(directory: Path) -> None:
    value = read(directory, verify_payload=False)
    expected = {"metadata.json", value["payload_file"]}
    if {path.name for path in directory.iterdir()} != expected:
        raise SidecarError("KV artifact directory contains unexpected files")
    (directory / "metadata.json").unlink()
    (directory / value["payload_file"]).unlink()
    directory.rmdir()
