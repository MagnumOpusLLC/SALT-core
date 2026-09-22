#!/usr/bin/env python3
"""Versioned server catalog for native Gemma DPR collections and contributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

DPR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SCHEMA = "salt.gemma4.dpr.catalog.v1"
CONTRIBUTION_SCHEMA = "salt.gemma4.dpr.contribution.v1"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CONTRIBUTION_BYTES = 4 * 1024 * 1024
MAX_VERSIONS = 4096


class DprCatalogError(ValueError):
    """A DPR catalog entry or contribution is invalid."""


@dataclass(frozen=True)
class DprAttachment:
    dpr_id: str
    version: int
    root: Path
    mode: str
    budget_gb: float
    serial_ms_per_token: float
    retention_gb: float
    manifest_sha256: str

    def public(self) -> dict[str, object]:
        return {
            "id": self.dpr_id,
            "version": self.version,
            "mode": self.mode,
            "manifest_sha256": self.manifest_sha256,
        }

    def control(self) -> dict[str, object]:
        return {
            **self.public(),
            "root": str(self.root),
            "budget_gb": self.budget_gb,
            "serial_ms_per_token": self.serial_ms_per_token,
            "retention_gb": self.retention_gb,
        }


class DprCatalog:
    def __init__(self, root: Path):
        expanded = root.expanduser()
        expanded.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = self._private_directory(expanded, "catalog root")
        contributions = self.root / "contributions"
        contributions.mkdir(mode=0o700, exist_ok=True)
        self.contributions = self._private_directory(
            contributions, "contribution root",
        )

    @staticmethod
    def _private_directory(path: Path, label: str) -> Path:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise DprCatalogError(f"DPR {label} must be a real directory")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise DprCatalogError(f"DPR {label} must be owned by the server user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise DprCatalogError(f"DPR {label} must have mode 0700 or stricter")
        return path.resolve(strict=True)

    @staticmethod
    def _selector(dpr_id: object, version: object) -> tuple[str, int]:
        if not isinstance(dpr_id, str) or DPR_ID_RE.fullmatch(dpr_id) is None:
            raise DprCatalogError("invalid DPR id")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise DprCatalogError("DPR version must be a positive integer")
        return dpr_id, version

    def _manifest_path(self, dpr_id: str, version: int) -> Path:
        return self.root / dpr_id / f"{version}.json"

    @staticmethod
    def _read_regular(path: Path, maximum: int) -> bytes:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise DprCatalogError("DPR manifest must be a regular non-symlink file")
        if info.st_size < 2 or info.st_size > maximum:
            raise DprCatalogError("DPR manifest size is outside its bound")
        with path.open("rb") as stream:
            value = stream.read(maximum + 1)
        if len(value) != info.st_size:
            raise DprCatalogError("DPR manifest changed while reading")
        return value

    def resolve(self, dpr_id: object, version: object) -> DprAttachment:
        dpr_id, version = self._selector(dpr_id, version)
        path = self._manifest_path(dpr_id, version)
        try:
            raw = self._read_regular(path, MAX_MANIFEST_BYTES)
        except FileNotFoundError as exc:
            raise DprCatalogError("unknown DPR id/version") from exc
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DprCatalogError("invalid DPR manifest JSON") from exc
        required = {
            "schema", "id", "version", "root", "mode", "budget_gb",
            "serial_ms_per_token", "retention_gb",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise DprCatalogError("invalid DPR manifest fields")
        if (value["schema"] != SCHEMA or value["id"] != dpr_id or
                value["version"] != version):
            raise DprCatalogError("DPR manifest identity mismatch")
        mode = value["mode"]
        if mode not in ("persist", "dynamic"):
            raise DprCatalogError("DPR mode must be persist or dynamic")
        numbers: list[float] = []
        for key in ("budget_gb", "serial_ms_per_token", "retention_gb"):
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise DprCatalogError(f"DPR {key} must be numeric")
            parsed = float(item)
            if not math.isfinite(parsed) or parsed <= 0.0:
                raise DprCatalogError(f"DPR {key} must be positive and finite")
            numbers.append(parsed)
        collection_root = Path(value["root"])
        if not collection_root.is_absolute():
            raise DprCatalogError("DPR collection root must be absolute")
        collection_root = self._private_directory(collection_root, "collection root")
        return DprAttachment(
            dpr_id=dpr_id,
            version=version,
            root=collection_root,
            mode=mode,
            budget_gb=numbers[0],
            serial_ms_per_token=numbers[1],
            retention_gb=numbers[2],
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def list_versions(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for path in sorted(self.root.glob("*/*.json")):
            if len(result) >= MAX_VERSIONS:
                raise DprCatalogError("DPR version catalog exceeds its bound")
            try:
                version = int(path.stem)
                attachment = self.resolve(path.parent.name, version)
            except (ValueError, DprCatalogError):
                continue
            result.append(attachment.public())
        return result

    def write_contribution(self, value: dict[str, Any]) -> dict[str, object]:
        envelope = {
            "schema": CONTRIBUTION_SCHEMA,
            **value,
        }
        raw = json.dumps(
            envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        if len(raw) > MAX_CONTRIBUTION_BYTES:
            raise DprCatalogError("DPR contribution exceeds 4 MiB")
        contribution_id = hashlib.sha256(raw).hexdigest()
        target = self.contributions / f"{contribution_id}.json"
        if not target.exists():
            fd, name = tempfile.mkstemp(
                prefix=f".{contribution_id}.", suffix=".tmp",
                dir=self.contributions,
            )
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb", closefd=True) as stream:
                    fd = -1
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(name, target)
                os.unlink(name)
                directory_fd = os.open(self.contributions, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                if fd >= 0:
                    os.close(fd)
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass
                raise
        return {
            "id": contribution_id,
            "object": "salt.dpr.contribution",
            "status": value.get("status"),
        }

    def begin_candidate(self, base: DprAttachment,
                        contribution_id: str) -> DprAttachment:
        if re.fullmatch(r"[0-9a-f]{64}", contribution_id) is None:
            raise DprCatalogError("invalid DPR contribution id")
        family = self.root / base.dpr_id
        family.mkdir(mode=0o700, exist_ok=True)
        candidates = family / "candidates"
        candidates.mkdir(mode=0o700, exist_ok=True)
        candidates = self._private_directory(candidates, "candidate root")
        target = candidates / contribution_id
        if target.exists() or target.is_symlink():
            raise DprCatalogError("DPR candidate already exists")
        total = 0
        maximum = int((base.budget_gb + base.retention_gb + 1.0) * 1_000_000_000)
        for directory, names, files in os.walk(base.root, followlinks=False):
            directory_path = Path(directory)
            for name in [*names, *files]:
                item = directory_path / name
                info = item.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise DprCatalogError("DPR collection contains a symlink")
                if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
                    raise DprCatalogError("DPR collection contains a special file")
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                    if total > maximum:
                        raise DprCatalogError("DPR collection exceeds candidate copy bound")
        shutil.copytree(base.root, target)
        os.chmod(target, 0o700)
        return replace(base, root=target, mode="dynamic")

    def abort_candidate(self, candidate: DprAttachment) -> None:
        try:
            relative = candidate.root.relative_to(self.root)
        except ValueError as exc:
            raise DprCatalogError("DPR candidate is outside catalog") from exc
        if "candidates" not in relative.parts:
            raise DprCatalogError("DPR candidate path is not a candidate")
        shutil.rmtree(candidate.root)

    def seal_candidate(self, candidate: DprAttachment) -> DprAttachment:
        family = self.root / candidate.dpr_id
        existing = [
            int(path.stem) for path in family.glob("*.json")
            if path.stem.isdigit()
        ]
        version = max(existing, default=0) + 1
        if version > MAX_VERSIONS:
            raise DprCatalogError("DPR version catalog exceeds its bound")
        collections = family / "collections"
        collections.mkdir(mode=0o700, exist_ok=True)
        collections = self._private_directory(collections, "collection version root")
        target = collections / str(version)
        if target.exists() or target.is_symlink():
            raise DprCatalogError("DPR collection version already exists")
        os.replace(candidate.root, target)
        manifest = {
            "schema": SCHEMA,
            "id": candidate.dpr_id,
            "version": version,
            "root": str(target),
            "mode": candidate.mode,
            "budget_gb": candidate.budget_gb,
            "serial_ms_per_token": candidate.serial_ms_per_token,
            "retention_gb": candidate.retention_gb,
        }
        raw = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        manifest_path = self._manifest_path(candidate.dpr_id, version)
        fd, name = tempfile.mkstemp(
            prefix=f".{version}.", suffix=".tmp", dir=family,
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as stream:
                fd = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(name, manifest_path)
            os.unlink(name)
            directory_fd = os.open(family, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
            raise
        return self.resolve(candidate.dpr_id, version)
