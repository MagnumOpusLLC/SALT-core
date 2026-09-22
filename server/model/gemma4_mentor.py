#!/usr/bin/env python3
"""Governed immutable exact-anchor store for Gemma mentor-state proposals."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

_SERVER = Path(__file__).resolve().parent
_STATE_MODULE_NAME = "salt_gemma4_mentor_state_artifact"
_STATE_SPEC = importlib.util.spec_from_file_location(
    _STATE_MODULE_NAME, _SERVER / "gemma4_state_artifact.py",
)
if _STATE_SPEC is None or _STATE_SPEC.loader is None:
    raise RuntimeError("cannot load Gemma 4 state artifact authority")
_STATE = importlib.util.module_from_spec(_STATE_SPEC)
sys.modules[_STATE_MODULE_NAME] = _STATE
_STATE_SPEC.loader.exec_module(_STATE)

SCHEMA = "salt.mentor-anchor.v1"
PROPOSAL_TYPE = "exact-anchor"
ANCHOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
INPUT_KEYS = {
    "anchor_id",
    "parents",
    "native_state_sha256",
    "facts_sha256",
    "evidence_policy_version",
    "evidence_bundle_sha256",
    "rendered_prefix_sha256",
    "prompt_ids_sha256",
    "creation_event",
}
CREATION_KEYS = {
    "session_epoch", "turn_id", "request_sha256", "build_identity_sha256",
}
MANIFEST_KEYS = {
    "schema", "proposal_type", "anchor_id", "revision", "parents",
    "model_id", "quant_track", "compatibility_sha256",
    "state_file_sha256", "state_payload_sha256", "native_state_sha256",
    "facts_sha256", "context_tokens", "position", "evidence_policy_version",
    "evidence_bundle_sha256", "rendered_prefix_sha256", "prompt_ids_sha256",
    "creation_event", "validation_contract",
}
VALIDATION_CONTRACT = {
    "mentor_can_commit": False,
    "raw_restore": "same-compatibility-only",
    "runtime_ready": False,
    "state_authority": "native-mentee",
}


class MentorAnchorError(ValueError):
    """Mentor anchor schema, integrity, path, or publication failure."""


@dataclass(frozen=True)
class MentorAnchorMetadata:
    anchor_id: str
    revision: str
    path: Path
    state_path: Path
    manifest_path: Path
    compatibility_sha256: str
    state_file_sha256: str
    state_payload_sha256: str
    native_state_sha256: str
    facts_sha256: str
    context_tokens: int
    position: int
    parents: tuple[str, ...]

    @property
    def selector(self) -> str:
        return f"{self.anchor_id}@{self.revision}"


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _directory_identity(path: Path, label: str) -> Path:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise MentorAnchorError(f"mentor {label} does not exist") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise MentorAnchorError(f"mentor {label} must be a real directory")
    if hasattr(os, "geteuid") and value.st_uid != os.geteuid():
        raise MentorAnchorError(f"mentor {label} must be owned by the server user")
    if stat.S_IMODE(value.st_mode) & 0o077:
        raise MentorAnchorError(f"mentor {label} must have mode 0700 or stricter")
    return path.resolve(strict=True)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise MentorAnchorError(f"cannot sync mentor directory {path.name}") from exc
    finally:
        os.close(fd)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_hex(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise MentorAnchorError(f"invalid {field}")
    return value


def _require_anchor_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or ANCHOR_ID_RE.fullmatch(value) is None
        or ".." in value
    ):
        raise MentorAnchorError("invalid anchor_id")
    return value


def _parse_selector(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or value.count("@") != 1:
        raise MentorAnchorError("mentor selector must be anchor_id@revision")
    anchor_id, revision = value.split("@", 1)
    try:
        return _require_anchor_id(anchor_id), _require_hex(revision, "revision")
    except MentorAnchorError as exc:
        raise MentorAnchorError("invalid mentor selector") from exc


class MentorAnchorStore:
    def __init__(
        self,
        root: Path,
        *,
        compatibility_sha256: bytes,
        model_id: str,
        quant_track: str,
    ):
        if (
            not isinstance(root, Path)
            or not isinstance(compatibility_sha256, bytes)
            or len(compatibility_sha256) != 32
            or compatibility_sha256 == bytes(32)
            or not isinstance(model_id, str)
            or not model_id
            or len(model_id) > 128
            or not isinstance(quant_track, str)
            or not quant_track
            or len(quant_track) > 64
        ):
            raise MentorAnchorError("invalid mentor anchor store arguments")
        expanded = root.expanduser()
        expanded.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = _directory_identity(expanded, "root")
        namespace = self.root / (
            "mentor-v1-g4kvc006-" + compatibility_sha256.hex()
        )
        namespace.mkdir(mode=0o700, exist_ok=True)
        self.namespace = _directory_identity(namespace, "compatibility namespace")
        self.compatibility_sha256 = compatibility_sha256
        self.model_id = model_id
        self.quant_track = quant_track

    def _anchor_path(self, anchor_id: str) -> Path:
        path = self.namespace / anchor_id
        if path.parent != self.namespace:
            raise MentorAnchorError("invalid anchor_id path")
        return path

    def _revision_path(self, anchor_id: str, revision: str) -> Path:
        anchor_path = self._anchor_path(anchor_id)
        path = anchor_path / revision
        if path.parent != anchor_path:
            raise MentorAnchorError("invalid mentor revision path")
        return path

    def _lock(self, anchor_id: str) -> int:
        path = self.namespace / f".{anchor_id}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = -1
        try:
            fd = os.open(path, flags, 0o600)
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fd
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise MentorAnchorError("cannot lock mentor anchor") from exc

    @staticmethod
    def _unlock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _validate_input(self, envelope: dict) -> dict:
        if not isinstance(envelope, dict) or set(envelope) != INPUT_KEYS:
            raise MentorAnchorError("invalid mentor envelope fields")
        anchor_id = _require_anchor_id(envelope["anchor_id"])
        parents_value = envelope["parents"]
        if not isinstance(parents_value, list):
            raise MentorAnchorError("mentor parents must be a list")
        parents = tuple(parents_value)
        if len(set(parents)) != len(parents):
            raise MentorAnchorError("mentor parents must be unique")
        for parent in parents:
            _parse_selector(parent)
        policy = envelope["evidence_policy_version"]
        if not isinstance(policy, str) or not policy or len(policy) > 128:
            raise MentorAnchorError("invalid evidence_policy_version")
        creation = envelope["creation_event"]
        if not isinstance(creation, dict) or set(creation) != CREATION_KEYS:
            raise MentorAnchorError("invalid creation_event fields")
        for field in ("session_epoch", "turn_id"):
            value = creation[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MentorAnchorError(f"invalid creation_event {field}")
        normalized = {
            "anchor_id": anchor_id,
            "parents": list(parents),
            "native_state_sha256": _require_hex(
                envelope["native_state_sha256"], "native_state_sha256",
            ),
            "facts_sha256": _require_hex(
                envelope["facts_sha256"], "facts_sha256",
            ),
            "evidence_policy_version": policy,
            "evidence_bundle_sha256": _require_hex(
                envelope["evidence_bundle_sha256"], "evidence_bundle_sha256",
            ),
            "rendered_prefix_sha256": _require_hex(
                envelope["rendered_prefix_sha256"], "rendered_prefix_sha256",
            ),
            "prompt_ids_sha256": _require_hex(
                envelope["prompt_ids_sha256"], "prompt_ids_sha256",
            ),
            "creation_event": {
                "session_epoch": creation["session_epoch"],
                "turn_id": creation["turn_id"],
                "request_sha256": _require_hex(
                    creation["request_sha256"], "creation_event request_sha256",
                ),
                "build_identity_sha256": _require_hex(
                    creation["build_identity_sha256"],
                    "creation_event build_identity_sha256",
                ),
            },
        }
        return normalized

    def _copy_state(self, source: Path, destination: Path) -> None:
        source_fd = -1
        destination_fd = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                source_fd = os.open(source, flags)
            except OSError as exc:
                raise MentorAnchorError(f"open source state failed: {exc}") from exc
            before = os.fstat(source_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < _STATE.KV_CACHE_HEADER.size
            ):
                raise MentorAnchorError("source state must be a private regular file")
            if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
                raise MentorAnchorError("source state must be owned by the server user")
            if stat.S_IMODE(before.st_mode) & 0o077:
                raise MentorAnchorError("source state must have mode 0600 or stricter")
            destination_fd = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                    getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written < 1:
                        raise MentorAnchorError("mentor state copy made no progress")
                    view = view[written:]
            os.fsync(destination_fd)
            after = os.fstat(source_fd)
            if _identity(before) != _identity(after):
                raise MentorAnchorError("source state identity changed while copying")
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)
            if source_fd >= 0:
                os.close(source_fd)

    def _manifest_record(self, state, envelope: dict) -> dict:
        return {
            "schema": SCHEMA,
            "proposal_type": PROPOSAL_TYPE,
            "anchor_id": envelope["anchor_id"],
            "parents": envelope["parents"],
            "model_id": self.model_id,
            "quant_track": self.quant_track,
            "compatibility_sha256": self.compatibility_sha256.hex(),
            "state_file_sha256": state.file_sha256,
            "state_payload_sha256": state.payload_sha256,
            "native_state_sha256": envelope["native_state_sha256"],
            "facts_sha256": envelope["facts_sha256"],
            "context_tokens": state.context_tokens,
            "position": state.position,
            "evidence_policy_version": envelope["evidence_policy_version"],
            "evidence_bundle_sha256": envelope["evidence_bundle_sha256"],
            "rendered_prefix_sha256": envelope["rendered_prefix_sha256"],
            "prompt_ids_sha256": envelope["prompt_ids_sha256"],
            "creation_event": envelope["creation_event"],
            "validation_contract": dict(VALIDATION_CONTRACT),
        }

    def seal(self, state_path: Path, envelope: dict) -> MentorAnchorMetadata:
        if not isinstance(state_path, Path):
            raise MentorAnchorError("invalid mentor state path")
        normalized = self._validate_input(envelope)
        anchor_id = normalized["anchor_id"]
        lock_fd = self._lock(anchor_id)
        staging: Path | None = None
        try:
            for parent in normalized["parents"]:
                try:
                    self.verify(parent)
                except MentorAnchorError as exc:
                    raise MentorAnchorError(
                        f"mentor parent {parent!r} is invalid"
                    ) from exc
            anchor_path = self._anchor_path(anchor_id)
            anchor_path.mkdir(mode=0o700, exist_ok=True)
            anchor_path = _directory_identity(anchor_path, "anchor directory")
            staging = Path(tempfile.mkdtemp(prefix=".seal-", dir=anchor_path))
            os.chmod(staging, 0o700)
            staged_state = staging / "state.g4kv"
            self._copy_state(state_path, staged_state)
            state = _STATE.inspect_g4kvc006(
                staged_state,
                expected_compatibility_sha256=self.compatibility_sha256,
                verify_payload=True,
                require_private_mode=True,
            )
            if normalized["native_state_sha256"] != state.logical_state_sha256:
                raise MentorAnchorError(
                    "native state SHA-256 does not match the V5 logical state"
                )
            base_record = self._manifest_record(state, normalized)
            revision = hashlib.sha256(_canonical(base_record)).hexdigest()
            record = dict(base_record)
            record["revision"] = revision
            manifest_path = staging / "anchor.json"
            manifest_fd = os.open(
                manifest_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                    getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                payload = _canonical(record) + b"\n"
                view = memoryview(payload)
                while view:
                    written = os.write(manifest_fd, view)
                    if written < 1:
                        raise MentorAnchorError("mentor manifest write made no progress")
                    view = view[written:]
                os.fsync(manifest_fd)
            finally:
                os.close(manifest_fd)
            _fsync_directory(staging)
            final_path = self._revision_path(anchor_id, revision)
            if final_path.exists():
                existing = self.verify(f"{anchor_id}@{revision}")
                existing_record = self._read_manifest(existing.manifest_path)
                if existing_record != record:
                    raise MentorAnchorError("contradictory mentor revision already exists")
                shutil.rmtree(staging)
                staging = None
                return existing
            try:
                os.rename(staging, final_path)
            except OSError as exc:
                if final_path.exists():
                    existing = self.verify(f"{anchor_id}@{revision}")
                    existing_record = self._read_manifest(existing.manifest_path)
                    if existing_record == record:
                        shutil.rmtree(staging)
                        staging = None
                        return existing
                raise MentorAnchorError("mentor revision publication failed") from exc
            staging = None
            _fsync_directory(anchor_path)
            _fsync_directory(self.namespace)
            return self.verify(f"{anchor_id}@{revision}")
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            self._unlock(lock_fd)

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = -1
        try:
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise MentorAnchorError("open mentor manifest failed") from exc
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 2
                or before.st_size > 64 * 1024
                or stat.S_IMODE(before.st_mode) & 0o077
            ):
                raise MentorAnchorError("mentor manifest must be a private regular file")
            if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
                raise MentorAnchorError("mentor manifest must be owned by the server user")
            raw = bytearray()
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise MentorAnchorError("mentor manifest is truncated")
                raw.extend(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise MentorAnchorError("mentor manifest exceeds encoded size")
            after = os.fstat(fd)
            if _identity(before) != _identity(after):
                raise MentorAnchorError("mentor manifest identity changed while reading")
            try:
                value = json.loads(bytes(raw).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MentorAnchorError("mentor manifest is not canonical JSON") from exc
            if not isinstance(value, dict):
                raise MentorAnchorError("mentor manifest must be an object")
            if bytes(raw) != _canonical(value) + b"\n":
                raise MentorAnchorError("mentor manifest is not canonical JSON")
            return value
        finally:
            if fd >= 0:
                os.close(fd)

    def resolve(self, selector: str) -> MentorAnchorMetadata:
        anchor_id, revision = _parse_selector(selector)
        anchor_path = _directory_identity(
            self._anchor_path(anchor_id), "anchor directory",
        )
        path = anchor_path / revision
        if not path.exists():
            raise MentorAnchorError("mentor anchor revision does not exist")
        path = _directory_identity(path, "revision directory")
        entries = {entry.name for entry in path.iterdir()}
        if entries != {"anchor.json", "state.g4kv"}:
            raise MentorAnchorError("mentor revision contains unexpected files")
        manifest_path = path / "anchor.json"
        state_path = path / "state.g4kv"
        record = self._read_manifest(manifest_path)
        if set(record) != MANIFEST_KEYS:
            raise MentorAnchorError("invalid mentor manifest fields")
        if (
            record.get("schema") != SCHEMA
            or record.get("proposal_type") != PROPOSAL_TYPE
            or record.get("anchor_id") != anchor_id
            or record.get("revision") != revision
            or record.get("model_id") != self.model_id
            or record.get("quant_track") != self.quant_track
            or record.get("compatibility_sha256") != self.compatibility_sha256.hex()
            or record.get("validation_contract") != VALIDATION_CONTRACT
        ):
            raise MentorAnchorError("mentor manifest identity mismatch")
        base_record = dict(record)
        del base_record["revision"]
        if hashlib.sha256(_canonical(base_record)).hexdigest() != revision:
            raise MentorAnchorError("mentor revision SHA-256 mismatch")
        normalized = self._validate_input({
            key: record[key] for key in INPUT_KEYS
        })
        try:
            state = _STATE.inspect_g4kvc006(
                state_path,
                expected_compatibility_sha256=self.compatibility_sha256,
                verify_payload=True,
                require_private_mode=True,
            )
        except _STATE.StateArtifactError as exc:
            raise MentorAnchorError(f"mentor state artifact invalid: {exc}") from exc
        if record["native_state_sha256"] != state.logical_state_sha256:
            raise MentorAnchorError(
                "mentor native state SHA-256 does not match the V5 logical state"
            )
        if (
            record.get("state_file_sha256") != state.file_sha256
            or record.get("state_payload_sha256") != state.payload_sha256
            or record.get("context_tokens") != state.context_tokens
            or record.get("position") != state.position
        ):
            raise MentorAnchorError("mentor state SHA-256/geometry mismatch")
        for parent in normalized["parents"]:
            try:
                self.resolve(parent)
            except MentorAnchorError as exc:
                raise MentorAnchorError(
                    f"mentor parent {parent!r} is invalid"
                ) from exc
        return MentorAnchorMetadata(
            anchor_id=anchor_id,
            revision=revision,
            path=path,
            state_path=state.path,
            manifest_path=manifest_path.resolve(strict=True),
            compatibility_sha256=state.compatibility_sha256,
            state_file_sha256=state.file_sha256,
            state_payload_sha256=state.payload_sha256,
            native_state_sha256=record["native_state_sha256"],
            facts_sha256=record["facts_sha256"],
            context_tokens=state.context_tokens,
            position=state.position,
            parents=tuple(normalized["parents"]),
        )

    def verify(self, selector: str) -> MentorAnchorMetadata:
        return self.resolve(selector)

    def list(self) -> list[MentorAnchorMetadata]:
        result = []
        for anchor_path in sorted(self.namespace.iterdir(), key=lambda value: value.name):
            if anchor_path.name.startswith("."):
                continue
            if ANCHOR_ID_RE.fullmatch(anchor_path.name) is None:
                raise MentorAnchorError("invalid entry in mentor namespace")
            anchor_path = _directory_identity(anchor_path, "anchor directory")
            for revision_path in sorted(anchor_path.iterdir(), key=lambda value: value.name):
                if revision_path.name.startswith("."):
                    continue
                result.append(self.resolve(
                    f"{anchor_path.name}@{revision_path.name}"
                ))
        result.sort(key=lambda value: value.selector)
        return result
