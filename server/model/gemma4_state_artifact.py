#!/usr/bin/env python3
"""Authenticated inspection of portable Gemma G4KVC006 state artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
import struct

GEMMA4_LAYERS = 30
GEMMA4_MAX_CONTEXT = 262_144
GEMMA4_SLIDING_RETAINED = 1_023
GEMMA4_SLIDING_WINDOW = 1_024
KV_CACHE_MAGIC = b"G4KVC006"
KV_CACHE_VERSION = 6
KV_CACHE_MAGIC_V5 = b"G4KVC005"
KV_CACHE_VERSION_V5 = 5
KV_CACHE_MAGIC_V4 = b"G4KVC004"
KV_CACHE_VERSION_V4 = 4
KV_CACHE_HEADER = struct.Struct("<8sIIIIQ30I32s32s32sII")


class StateArtifactError(ValueError):
    """Portable state artifact failed structural or identity validation."""


@dataclass(frozen=True)
class G4KVMetadata:
    path: Path
    version: int
    context_tokens: int
    position: int
    total_bytes: int
    compatibility_sha256: str
    logical_state_sha256: str
    payload_sha256: str
    file_sha256: str
    sliding_row_base: int
    sliding_window: int
    legacy_read_only: bool


def expected_dimensions() -> tuple[int, ...]:
    return tuple(
        1024 if layer % 6 == 5 else 2048
        for layer in range(GEMMA4_LAYERS)
    )


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _payload_bytes(position: int, dimensions: tuple[int, ...],
                   *, version: int = KV_CACHE_VERSION) -> int:
    total = 0
    for layer, dimension in enumerate(dimensions):
        rows = position if layer % 6 == 5 else min(
            position, GEMMA4_SLIDING_RETAINED,
        )
        segments = 1 if version == KV_CACHE_VERSION and layer % 6 == 5 else 2
        total += rows * dimension * segments * 4
    return total


def inspect_g4kvc006(
    path: Path,
    *,
    expected_compatibility_sha256: bytes,
    legacy_v5_compatibility_sha256: bytes | None = None,
    legacy_v4_compatibility_sha256: bytes | None = None,
    verify_payload: bool = True,
    require_private_mode: bool = True,
) -> G4KVMetadata:
    """Inspect one G4KVC006 file through one no-follow descriptor.

    `verify_payload=False` validates the header and file identity without reading
    the payload; `file_sha256` is then the empty string. Mentor anchors use the
    default full verification. The legacy selectable-cache listing path may opt
    out of payload reads while preserving its existing behavior.
    """

    if (
        not isinstance(path, Path)
        or not isinstance(expected_compatibility_sha256, bytes)
        or len(expected_compatibility_sha256) != 32
        or expected_compatibility_sha256 == bytes(32)
        or (
            legacy_v5_compatibility_sha256 is not None
            and (
                not isinstance(legacy_v5_compatibility_sha256, bytes)
                or len(legacy_v5_compatibility_sha256) != 32
                or legacy_v5_compatibility_sha256 == bytes(32)
            )
        )
        or (
            legacy_v4_compatibility_sha256 is not None
            and (
                not isinstance(legacy_v4_compatibility_sha256, bytes)
                or len(legacy_v4_compatibility_sha256) != 32
                or legacy_v4_compatibility_sha256 == bytes(32)
            )
        )
        or not isinstance(verify_payload, bool)
        or not isinstance(require_private_mode, bool)
    ):
        raise StateArtifactError("invalid state artifact inspection arguments")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        try:
            fd = os.open(path, flags)
        except FileNotFoundError as exc:
            raise StateArtifactError("state artifact does not exist") from exc
        except OSError as exc:
            raise StateArtifactError(f"open state artifact failed: {exc}") from exc

        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < KV_CACHE_HEADER.size
        ):
            raise StateArtifactError("state artifact must be a private regular file")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise StateArtifactError("state artifact must be owned by the server user")
        if require_private_mode and stat.S_IMODE(before.st_mode) & 0o077:
            raise StateArtifactError("state artifact must have mode 0600 or stricter")

        raw_header = os.read(fd, KV_CACHE_HEADER.size)
        if len(raw_header) != KV_CACHE_HEADER.size:
            raise StateArtifactError("state artifact header is truncated")
        unpacked = KV_CACHE_HEADER.unpack(raw_header)
        magic, version, layers, context, position, total_bytes = unpacked[:6]
        dimensions = tuple(unpacked[6:6 + GEMMA4_LAYERS])
        compatibility_sha = unpacked[6 + GEMMA4_LAYERS]
        logical_state_sha = unpacked[7 + GEMMA4_LAYERS]
        payload_sha = unpacked[8 + GEMMA4_LAYERS]
        sliding_row_base = unpacked[9 + GEMMA4_LAYERS]
        sliding_window = unpacked[10 + GEMMA4_LAYERS]

        legacy_v5 = magic == KV_CACHE_MAGIC_V5 and version == KV_CACHE_VERSION_V5
        legacy_v4 = magic == KV_CACHE_MAGIC_V4 and version == KV_CACHE_VERSION_V4
        if not legacy_v5 and not legacy_v4 and (
                magic != KV_CACHE_MAGIC or version != KV_CACHE_VERSION):
            raise StateArtifactError("state artifact format/version mismatch")
        if layers != GEMMA4_LAYERS or dimensions != expected_dimensions():
            raise StateArtifactError("state artifact layer geometry mismatch")
        if not 1 <= context <= GEMMA4_MAX_CONTEXT or position < 1 or position > context:
            raise StateArtifactError("state artifact context/position is invalid")
        expected_bytes = KV_CACHE_HEADER.size + (
            position * 450_560
            if legacy_v4 else _payload_bytes(
                position, dimensions, version=version,
            )
        )
        if total_bytes != before.st_size or total_bytes != expected_bytes:
            raise StateArtifactError("state artifact byte geometry mismatch")
        expected_compatibility = legacy_v4_compatibility_sha256 if legacy_v4 \
            else legacy_v5_compatibility_sha256 if legacy_v5 \
            else expected_compatibility_sha256
        if expected_compatibility is None or compatibility_sha != expected_compatibility:
            raise StateArtifactError("state artifact compatibility identity mismatch")
        expected_base = max(0, position - GEMMA4_SLIDING_RETAINED)
        if legacy_v4:
            if logical_state_sha != bytes(32) or sliding_row_base != 0 or \
                    sliding_window != 0 or payload_sha == bytes(32):
                raise StateArtifactError("legacy state artifact header is invalid")
        elif logical_state_sha == bytes(32) or payload_sha == bytes(32) or \
                sliding_row_base != expected_base or \
                sliding_window != GEMMA4_SLIDING_WINDOW:
            raise StateArtifactError("state artifact header integrity fields are invalid")

        file_sha256 = ""
        if verify_payload:
            payload_hasher = hashlib.sha256()
            file_hasher = hashlib.sha256(raw_header)
            # G4KVC006 stores the authoritative full-state digest, including
            # exact K rows that native import reconstructs from the compact
            # shared-projection V bases. A payload-only inspector cannot
            # recompute those rows; native C verifies this digest before
            # publishing imported state.
            logical_hasher = hashlib.sha256()
            logical_hasher.update(b"G4STATE6" if version == 6 else b"G4STATE4")
            logical_hasher.update(expected_compatibility)
            logical_hasher.update(struct.pack("<III", layers, context, position))
            for layer, dimension in enumerate(dimensions):
                base = 0 if layer % 6 == 5 else expected_base
                logical_hasher.update(struct.pack("<II", dimension, base))
                input_base = 0 if legacy_v4 else base
                segments = 1 if version == 6 and layer % 6 == 5 else 2
                for _segment in range(segments):
                    rows = position - input_base
                    logical_offset = base - input_base
                    byte_ranges = (
                        (False, logical_offset * dimension * 4),
                        (True, (rows - logical_offset) * dimension * 4),
                    )
                    for include_logical, remaining in byte_ranges:
                        while remaining:
                            chunk = os.read(fd, min(1024 * 1024, remaining))
                            if not chunk:
                                raise StateArtifactError(
                                    "state artifact payload is truncated"
                                )
                            if len(chunk) % 4 != 0 or any(
                                not math.isfinite(value[0])
                                for value in struct.iter_unpack("<f", chunk)
                            ):
                                raise StateArtifactError(
                                    "state artifact payload contains non-finite FP32"
                                )
                            payload_hasher.update(chunk)
                            file_hasher.update(chunk)
                            if include_logical:
                                logical_hasher.update(chunk)
                            remaining -= len(chunk)
            if os.read(fd, 1):
                raise StateArtifactError("state artifact payload exceeds encoded byte count")
            if payload_hasher.digest() != payload_sha:
                raise StateArtifactError("state artifact payload SHA-256 mismatch")
            computed_logical_state_sha = logical_hasher.digest()
            if legacy_v5 and computed_logical_state_sha != logical_state_sha:
                raise StateArtifactError("state artifact logical-state SHA-256 mismatch")
            if legacy_v4:
                logical_state_sha = computed_logical_state_sha
            file_sha256 = file_hasher.hexdigest()

        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise StateArtifactError("state artifact identity changed while authenticating")

        return G4KVMetadata(
            path=path.resolve(strict=True),
            version=version,
            context_tokens=context,
            position=position,
            total_bytes=total_bytes,
            compatibility_sha256=compatibility_sha.hex(),
            logical_state_sha256=logical_state_sha.hex(),
            payload_sha256=payload_sha.hex(),
            file_sha256=file_sha256,
            sliding_row_base=expected_base,
            sliding_window=GEMMA4_SLIDING_WINDOW,
            legacy_read_only=legacy_v5 or legacy_v4,
        )
    finally:
        if fd >= 0:
            os.close(fd)


# Transitional import name for external callers; new code uses the V6 name.
inspect_g4kvc005 = inspect_g4kvc006
