#!/usr/bin/env python3
"""One-listener scheduler over isolated persistent Gemma server processes.

The scheduler is transport-only. Child servers retain native state/KV/DPR and
COMMIT authority; this process only assigns sessions, estimates remaining work,
and proxies HTTP/SSE to bounded loopback children.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from gemma4_dpr_catalog import DprAttachment, DprCatalog, DprCatalogError
SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
MAX_ENGINE_PROCESSES = 64
MAX_SESSION_RECORDS = 4096
MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_PROXY_RESPONSE_BYTES = 64 * 1024 * 1024
BODY_TIMEOUT_S = 30.0


def _attach_session_identity(value: dict[str, Any], session_id: str) -> dict[str, Any]:
    value["session_id"] = session_id
    extension = value.get("x_salt")
    if isinstance(extension, dict):
        extension["session"] = {"id": session_id}
    response = value.get("response")
    if isinstance(response, dict):
        response["session_id"] = session_id
        nested = response.get("x_salt")
        if isinstance(nested, dict):
            nested["session"] = {"id": session_id}
    return value


def _session_sse_bytes(buffer: bytes, chunk: bytes, session_id: str,
                       *, final: bool = False) -> tuple[bytes, bytes]:
    pending = buffer + chunk
    lines = pending.splitlines(keepends=True)
    if lines and not final and not lines[-1].endswith((b"\n", b"\r")):
        pending = lines.pop()
    else:
        pending = b""
    output: list[bytes] = []
    for raw in lines:
        content = raw.rstrip(b"\r\n")
        ending = raw[len(content):]
        if content.startswith(b"data: ") and content != b"data: [DONE]":
            try:
                value = json.loads(content[6:])
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                content = b"data: " + json.dumps(
                    _attach_session_identity(value, session_id),
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
        output.append(content + ending)
    return pending, b"".join(output)


class SchedulerUnavailable(TimeoutError):
    """No compatible child engine became available before admission timeout."""


@dataclass
class WorkProfile:
    samples: int = 0
    elapsed_s: float = 0.0
    work_ratio: float = 1.0
    prefill_hit_ratio: float = 0.0
    decode_parent_hit_ratio: float = 0.0
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0

    def observe(self, *, elapsed_s: float, work_ratio: float,
                prefill_hit_ratio: float, decode_parent_hit: bool,
                prompt_tokens: int, completion_tokens: int) -> None:
        self.samples += 1
        weight = 1.0 / self.samples
        self.elapsed_s += (elapsed_s - self.elapsed_s) * weight
        self.work_ratio += (work_ratio - self.work_ratio) * weight
        self.prefill_hit_ratio += (
            prefill_hit_ratio - self.prefill_hit_ratio
        ) * weight
        parent = 1.0 if decode_parent_hit else 0.0
        self.decode_parent_hit_ratio += (
            parent - self.decode_parent_hit_ratio
        ) * weight
        self.prompt_tokens += (prompt_tokens - self.prompt_tokens) * weight
        self.completion_tokens += (
            completion_tokens - self.completion_tokens
        ) * weight

    def predicted_prefill_s(self, prefill_ms_per_token: float) -> float:
        uncached = self.prompt_tokens * (1.0 - self.prefill_hit_ratio)
        if prefill_ms_per_token > 0.0:
            return min(
                self.elapsed_s,
                max(0.0, uncached * prefill_ms_per_token / 1000.0),
            )
        decode_work = (
            self.completion_tokens *
            (1.0 - self.decode_parent_hit_ratio)
        )
        total_work = uncached + decode_work
        return (
            self.elapsed_s * uncached / total_work
            if total_work > 0.0 else 0.0
        )


@dataclass
class EngineSlot:
    index: int
    host: str
    port: int
    session_id: str | None = None
    busy: bool = False
    request_key: str | None = None
    predicted_work_ratio: float | None = None
    estimated_finish_monotonic: float = 0.0
    estimated_prefill_end_monotonic: float = 0.0
    completed_requests: int = 0
    observed_wall_s: float = 0.0
    generation: int = 0
    failed: bool = False
    dpr: dict[str, object] | None = None


@dataclass
class SessionRecord:
    session_id: str
    cache_id: str
    state: str = "unbound"
    slot_index: int | None = None
    last_access_monotonic: float = 0.0
    retention_seconds: float | None = None
    manifest: dict[str, Any] | None = None
    dpr_attachment: DprAttachment | None = None


@dataclass(frozen=True)
class SessionMaintenance:
    slot_index: int
    generation: int
    requested_session: str
    evicted_session: str | None
    evicted_cache_id: str | None
    restore_cache_id: str | None
    restore_manifest: dict[str, Any] | None
    expired_cache_id: str | None
    dpr_control: dict[str, object]
    dpr_public: dict[str, object] | None
    reconfigure_dpr: bool


@dataclass(frozen=True)
class SessionSpill:
    slot_index: int
    generation: int
    session_id: str
    cache_id: str


@dataclass(frozen=True)
class EngineLease:
    slot_index: int
    session_id: str
    session_sha256: str
    body: dict[str, Any]
    request_key: str
    queued_at: float
    acquired_at: float
    predicted_elapsed_s: float | None
    predicted_prefill_s: float | None
    predicted_work_ratio: float | None
    profile_samples: int
    slot_generation: int


class EnginePool:
    """Bounded session-affine scheduling state for loopback child servers."""

    def __init__(self, endpoints: list[tuple[str, int]], *, wait_s: float = 1.0,
                 prefill_ms_per_token: float = 0.0,
                 max_heavy_active: int = 1,
                 heavy_work_ratio: float = 0.25):
        if not endpoints or len(endpoints) > MAX_ENGINE_PROCESSES:
            raise ValueError(
                f"engine process count must be in [1, {MAX_ENGINE_PROCESSES}]"
            )
        if not math.isfinite(wait_s) or wait_s <= 0.0 or wait_s > 60.0:
            raise ValueError("scheduler wait must be finite and in (0, 60]")
        if (not math.isfinite(prefill_ms_per_token) or
                prefill_ms_per_token < 0.0):
            raise ValueError("prefill ms/token must be finite and nonnegative")
        if not 1 <= max_heavy_active <= len(endpoints):
            raise ValueError("max heavy active must be within engine process count")
        if (not math.isfinite(heavy_work_ratio) or
                not 0.0 < heavy_work_ratio <= 1.0):
            raise ValueError("heavy work ratio must be finite and in (0, 1]")
        self.wait_s = wait_s
        self.prefill_ms_per_token = prefill_ms_per_token
        self.max_heavy_active = max_heavy_active
        self.heavy_work_ratio = heavy_work_ratio
        self._condition = threading.Condition()
        self._slots = [
            EngineSlot(index=index, host=host, port=port)
            for index, (host, port) in enumerate(endpoints)
        ]
        self._sessions: dict[str, int] = {}
        self._records: dict[str, SessionRecord] = {}
        self._profiles: dict[str, WorkProfile] = {}
        self._queued = 0

    @staticmethod
    def validate_session_id(value: str) -> str:
        if not isinstance(value, str) or SESSION_ID_RE.fullmatch(value) is None:
            raise ValueError(
                "session must be 1..128 ASCII letters, digits, '.', '_', ':', or '-'"
            )
        return value

    @staticmethod
    def validate_retention(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("session retention_seconds must be positive or null")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 31_536_000.0:
            raise ValueError(
                "session retention_seconds must be in (0, 31536000] or null"
            )
        return parsed

    @classmethod
    def resolve_session(cls, body: dict[str, Any],
                        header_session: str | None
                        ) -> tuple[str, dict[str, Any], float | None]:
        request_body = dict(body)
        body_session = request_body.pop("session", None)
        explicit_session = request_body.pop("session_id", None)
        retention_seconds = cls.validate_retention(
            request_body.pop("session_retention_seconds", None)
        )
        if body_session is not None:
            body_session = cls.validate_session_id(body_session)
        if explicit_session is not None:
            explicit_session = cls.validate_session_id(explicit_session)
        if header_session is not None:
            header_session = cls.validate_session_id(header_session)
        supplied = [
            value for value in (body_session, explicit_session, header_session)
            if value is not None
        ]
        if supplied and any(value != supplied[0] for value in supplied[1:]):
            raise ValueError(
                "session, session_id, and X-Salt-Session must agree"
            )
        return (
            supplied[0] if supplied else "default",
            request_body,
            retention_seconds,
        )

    def create_session_id(self, retention_seconds: float | None = None,
                          dpr_attachment: DprAttachment | None = None) -> str:
        with self._condition:
            if len(self._records) >= MAX_SESSION_RECORDS:
                raise SchedulerUnavailable("session record capacity exhausted")
            while True:
                session_id = "sess_" + secrets.token_hex(16)
                if session_id not in self._records:
                    record = self._record_locked(session_id, retention_seconds)
                    record.dpr_attachment = dpr_attachment
                    return session_id

    def session_dpr(self, session_id: str) -> DprAttachment | None:
        with self._condition:
            record = self._records.get(session_id)
            return record.dpr_attachment if record is not None else None

    def _record_locked(self, session_id: str,
                       retention_seconds: float | None) -> SessionRecord:
        record = self._records.get(session_id)
        if record is None:
            if len(self._records) >= MAX_SESSION_RECORDS:
                raise SchedulerUnavailable("session record capacity exhausted")
            record = SessionRecord(
                session_id=session_id,
                cache_id="session-" + self.session_sha256(session_id)[:32],
                last_access_monotonic=time.monotonic(),
                retention_seconds=retention_seconds,
            )
            self._records[session_id] = record
        elif retention_seconds is not None:
            record.retention_seconds = retention_seconds
        return record

    def seed_spilled_session(self, session_id: str, cache_id: str,
                             manifest: dict[str, Any]) -> None:
        session_id = self.validate_session_id(session_id)
        expected_cache_id = "session-" + self.session_sha256(session_id)[:32]
        if cache_id != expected_cache_id or not isinstance(manifest, dict):
            raise ValueError("session checkpoint identity is invalid")
        with self._condition:
            record = self._records.get(session_id)
            if record is None:
                record = self._record_locked(session_id, None)
            elif (record.state != "spilled" or record.cache_id != cache_id or
                  record.manifest != manifest):
                raise ValueError("session checkpoint conflicts with live state")
            record.state = "spilled"
            record.slot_index = None
            record.manifest = dict(manifest)
            record.dpr_attachment = None
            record.last_access_monotonic = time.monotonic()

    def reserve_session(self, session_id: str,
                        retention_seconds: float | None = None,
                        dpr_attachment: DprAttachment | None = None,
                        ) -> SessionMaintenance | None:
        session_id = self.validate_session_id(session_id)
        deadline = time.monotonic() + self.wait_s
        with self._condition:
            record = self._record_locked(session_id, retention_seconds)
            requested_dpr = (
                dpr_attachment.public() if dpr_attachment is not None else None
            )
            current_dpr = (
                record.dpr_attachment.public()
                if record.dpr_attachment is not None else None
            )
            if record.state not in ("unbound", "released", "expired") and \
                    current_dpr != requested_dpr:
                raise ValueError("DPR attachment cannot change within a session")
            if record.state in ("unbound", "released", "expired"):
                record.dpr_attachment = dpr_attachment
            while True:
                if session_id in self._sessions:
                    return None
                available = [
                    slot for slot in self._slots
                    if not slot.busy and not slot.failed and slot.session_id is None
                ]
                evicted_session = None
                if available:
                    slot = min(available, key=lambda item: item.index)
                else:
                    candidates = [
                        slot for slot in self._slots
                        if not slot.busy and not slot.failed and
                        slot.session_id is not None
                    ]
                    if not candidates:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            raise SchedulerUnavailable("engine pool busy")
                        self._condition.wait(timeout=remaining)
                        continue
                    slot = min(candidates, key=lambda item: (
                        self._records[item.session_id or ""].last_access_monotonic,
                        item.index,
                    ))
                    evicted_session = slot.session_id
                slot.busy = True
                slot.request_key = "session-maintenance"
                now = time.monotonic()
                expired = (
                    record.state == "spilled" and record.manifest is not None and
                    record.retention_seconds is not None and
                    now - record.last_access_monotonic >= record.retention_seconds
                )
                restore_manifest = (
                    dict(record.manifest)
                    if record.state == "spilled" and record.manifest is not None and
                    not expired
                    else None
                )
                return SessionMaintenance(
                    slot_index=slot.index,
                    generation=slot.generation,
                    requested_session=session_id,
                    evicted_session=evicted_session,
                    evicted_cache_id=(
                        self._records[evicted_session].cache_id
                        if evicted_session is not None else None
                    ),
                    restore_cache_id=(record.cache_id if restore_manifest else None),
                    restore_manifest=restore_manifest,
                    expired_cache_id=record.cache_id if expired else None,
                    dpr_control=(
                        record.dpr_attachment.control()
                        if record.dpr_attachment is not None else {"mode": "off"}
                    ),
                    dpr_public=(
                        record.dpr_attachment.public()
                        if record.dpr_attachment is not None else None
                    ),
                    reconfigure_dpr=slot.dpr != (
                        record.dpr_attachment.public()
                        if record.dpr_attachment is not None else None
                    ),
                )

    def expire_session_spill(self, maintenance: SessionMaintenance) -> None:
        if maintenance.expired_cache_id is None:
            return
        with self._condition:
            record = self._records[maintenance.requested_session]
            if record.cache_id != maintenance.expired_cache_id:
                raise RuntimeError("session expiry identity drift")
            record.state = "expired"
            record.manifest = None

    def complete_session_maintenance(
            self, maintenance: SessionMaintenance,
            evicted_manifest: dict[str, Any] | None) -> None:
        with self._condition:
            slot = self._slots[maintenance.slot_index]
            if (not slot.busy or slot.generation != maintenance.generation or
                    slot.session_id != maintenance.evicted_session):
                raise RuntimeError("session maintenance ownership drift")
            if maintenance.evicted_session is not None:
                old = self._records[maintenance.evicted_session]
                if evicted_manifest is None:
                    raise RuntimeError("evicted session has no spill manifest")
                old.state = "spilled"
                old.slot_index = None
                old.manifest = dict(evicted_manifest)
                self._sessions.pop(old.session_id, None)
            record = self._records[maintenance.requested_session]
            slot.generation += 1
            slot.session_id = record.session_id
            slot.dpr = (
                dict(maintenance.dpr_public)
                if maintenance.dpr_public is not None else None
            )
            record.state = "resident"
            record.slot_index = slot.index
            record.last_access_monotonic = time.monotonic()
            self._sessions[record.session_id] = slot.index
            self._unlock_locked(slot, unbind=False)

    def fail_session_maintenance(
            self, maintenance: SessionMaintenance,
            *, evicted_manifest: dict[str, Any] | None,
            state_may_have_changed: bool) -> None:
        with self._condition:
            slot = self._slots[maintenance.slot_index]
            if maintenance.evicted_session is not None and evicted_manifest is not None:
                old = self._records[maintenance.evicted_session]
                old.state = "spilled"
                old.slot_index = None
                old.manifest = dict(evicted_manifest)
                self._sessions.pop(old.session_id, None)
                slot.session_id = None
            if state_may_have_changed:
                slot.failed = True
                slot.session_id = None
            self._unlock_locked(slot, unbind=False)

    def reserve_contribution_spill(
            self, session_id: str) -> tuple[SessionSpill | None, str,
                                            dict[str, Any] | None]:
        session_id = self.validate_session_id(session_id)
        with self._condition:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(session_id)
            if record.state == "spilled" and record.manifest is not None:
                return None, record.cache_id, dict(record.manifest)
            slot_index = self._sessions.get(session_id)
            if slot_index is None:
                raise KeyError(session_id)
            slot = self._slots[slot_index]
            if slot.busy or slot.failed:
                raise SchedulerUnavailable("session is busy")
            slot.busy = True
            slot.request_key = "session-contribution-spill"
            return SessionSpill(
                slot_index=slot.index, generation=slot.generation,
                session_id=session_id, cache_id=record.cache_id,
            ), record.cache_id, None

    def complete_contribution_spill(
            self, spill: SessionSpill, manifest: dict[str, Any]) -> None:
        with self._condition:
            slot = self._slots[spill.slot_index]
            if (not slot.busy or slot.generation != spill.generation or
                    slot.session_id != spill.session_id):
                raise RuntimeError("contribution spill ownership drift")
            record = self._records[spill.session_id]
            record.state = "spilled"
            record.slot_index = None
            record.manifest = dict(manifest)
            record.last_access_monotonic = time.monotonic()
            self._sessions.pop(spill.session_id, None)
            slot.session_id = None
            self._unlock_locked(slot, unbind=False)

    def fail_contribution_spill(self, spill: SessionSpill,
                                *, state_may_have_changed: bool) -> None:
        with self._condition:
            slot = self._slots[spill.slot_index]
            if state_may_have_changed:
                slot.failed = True
                self._sessions.pop(spill.session_id, None)
                slot.session_id = None
            self._unlock_locked(slot, unbind=False)

    @staticmethod
    def request_key(body: dict[str, Any], path: str) -> str:
        semantic = dict(body)
        semantic.pop("stream", None)
        semantic.pop("stream_options", None)
        encoded = json.dumps(
            {"path": path, "body": semantic}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def session_sha256(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("ascii")).hexdigest()

    def _choose_unbound_locked(self) -> EngineSlot | None:
        available = [
            slot for slot in self._slots
            if slot.session_id is None and not slot.busy and not slot.failed
        ]
        return min(
            available,
            key=lambda slot: (
                slot.observed_wall_s, slot.completed_requests, slot.index,
            ),
        ) if available else None

    def acquire(self, session_id: str, body: dict[str, Any], path: str,
                *, existing: bool = False) -> EngineLease:
        session_id = self.validate_session_id(session_id)
        request_key = self.request_key(body, path)
        queued_at = time.monotonic()
        deadline = queued_at + self.wait_s
        with self._condition:
            self._queued += 1
            try:
                while True:
                    now = time.monotonic()
                    slot_index = self._sessions.get(session_id)
                    if existing and slot_index is None:
                        raise KeyError(session_id)
                    slot = (
                        self._slots[slot_index]
                        if slot_index is not None
                        else self._choose_unbound_locked()
                    )
                    if slot is not None and not slot.busy:
                        profile = self._profiles.get(request_key)
                        predicted_prefill = (
                            profile.predicted_prefill_s(
                                self.prefill_ms_per_token,
                            )
                            if profile is not None and profile.samples else None
                        )
                        predicted_work = (
                            profile.work_ratio
                            if profile is not None and profile.samples else None
                        )
                        incoming_heavy = (
                            predicted_work is None or
                            predicted_work >= self.heavy_work_ratio
                        )
                        heavy_active = sum(
                            1 for item in self._slots
                            if item.busy and (
                                item.predicted_work_ratio is None or
                                item.predicted_work_ratio >= self.heavy_work_ratio
                            )
                        )
                        if (incoming_heavy and
                                heavy_active >= self.max_heavy_active):
                            remaining = deadline - now
                            if remaining <= 0.0:
                                raise SchedulerUnavailable(
                                    "heavy-work admission width busy",
                                )
                            self._condition.wait(timeout=remaining)
                            continue
                        if slot_index is None:
                            slot.session_id = session_id
                            self._sessions[session_id] = slot.index
                        acquired_at = time.monotonic()
                        predicted_s = (
                            profile.elapsed_s
                            if profile is not None and profile.samples else None
                        )
                        samples = profile.samples if profile is not None else 0
                        slot.busy = True
                        slot.request_key = request_key
                        slot.predicted_work_ratio = (
                            predicted_work if predicted_work is not None else 1.0
                        )
                        slot.estimated_finish_monotonic = (
                            acquired_at + predicted_s
                            if predicted_s is not None else 0.0
                        )
                        slot.estimated_prefill_end_monotonic = (
                            acquired_at + predicted_prefill
                            if predicted_prefill is not None else float("inf")
                        )
                        return EngineLease(
                            slot_index=slot.index,
                            session_id=session_id,
                            session_sha256=self.session_sha256(session_id),
                            body=body,
                            request_key=request_key,
                            queued_at=queued_at,
                            acquired_at=acquired_at,
                            predicted_elapsed_s=predicted_s,
                            predicted_prefill_s=predicted_prefill,
                            predicted_work_ratio=predicted_work,
                            profile_samples=samples,
                            slot_generation=slot.generation,
                        )
                    remaining = deadline - now
                    if remaining <= 0.0:
                        estimates = [
                            max(0.0, item.estimated_finish_monotonic - now)
                            for item in self._slots
                            if item.busy and item.estimated_finish_monotonic > now
                        ]
                        reason = (
                            "engine session capacity exhausted"
                            if session_id not in self._sessions and
                            all(item.session_id is not None for item in self._slots)
                            else "engine pool busy"
                        )
                        if estimates:
                            reason += f"; retry_after_s={min(estimates):.3f}"
                        raise SchedulerUnavailable(reason)
                    self._condition.wait(timeout=remaining)
            finally:
                self._queued -= 1

    @staticmethod
    def observed_work(dpr_result: object,
                      prompt_tokens: int) -> tuple[float, float, bool]:
        dpr = dpr_result if isinstance(dpr_result, dict) else {}
        prefill = dpr.get("prefill")
        prefill = prefill if isinstance(prefill, dict) else {}
        try:
            total = int(prefill.get("prompt_tokens", prompt_tokens))
            cached = int(prefill.get("cached_tokens", 0))
        except (TypeError, ValueError):
            total = max(prompt_tokens, 0)
            cached = 0
        cached = min(max(cached, 0), max(total, 0))
        prefill_hit_ratio = cached / total if total > 0 else 0.0
        parent_hit = (
            dpr.get("outcome") == "hit" and dpr.get("kind") == "parent"
        )
        work_ratio = (
            (1.0 - prefill_hit_ratio) + (0.0 if parent_hit else 1.0)
        ) / 2.0
        return work_ratio, prefill_hit_ratio, parent_hit

    def finish(self, lease: EngineLease, *, elapsed_s: float,
               dpr_result: object, prompt_tokens: int, completion_tokens: int,
               replayed: bool = False) -> dict[str, Any]:
        elapsed_s = max(elapsed_s, 0.0)
        work_ratio, prefill_hit_ratio, parent_hit = self.observed_work(
            dpr_result, prompt_tokens,
        )
        with self._condition:
            slot = self._slots[lease.slot_index]
            if (not slot.busy or slot.session_id != lease.session_id or
                    slot.generation != lease.slot_generation):
                raise RuntimeError("scheduler lease ownership drift")
            if not replayed:
                profile = self._profiles.setdefault(lease.request_key, WorkProfile())
                profile.observe(
                    elapsed_s=elapsed_s, work_ratio=work_ratio,
                    prefill_hit_ratio=prefill_hit_ratio,
                    decode_parent_hit=parent_hit,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            slot.completed_requests += 1
            slot.observed_wall_s += elapsed_s
            record = self._records.get(lease.session_id)
            if record is not None:
                record.last_access_monotonic = time.monotonic()
            public = {
                "mode": "session-affine-engine-process-pool",
                "engine_index": slot.index,
                "engine_processes": len(self._slots),
                "session_sha256": lease.session_sha256,
                "queue_s": round(lease.acquired_at - lease.queued_at, 6),
                "profile_samples_before": lease.profile_samples,
                "predicted_elapsed_s": (
                    round(lease.predicted_elapsed_s, 6)
                    if lease.predicted_elapsed_s is not None else None
                ),
                "predicted_prefill_s": (
                    round(lease.predicted_prefill_s, 6)
                    if lease.predicted_prefill_s is not None else None
                ),
                "predicted_work_ratio": (
                    round(lease.predicted_work_ratio, 6)
                    if lease.predicted_work_ratio is not None else None
                ),
                "observed_elapsed_s": round(elapsed_s, 6),
                "observed_prefill_hit_ratio": round(prefill_hit_ratio, 6),
                "observed_decode_parent_hit": parent_hit,
                "native_hit_authority": True,
                "request_migration": False,
                "seat_generation": slot.generation,
                "retention": (
                    {"mode": "hosting-period"}
                    if record is None or record.retention_seconds is None else
                    {"mode": "ttl", "seconds": record.retention_seconds}
                ),
                "dpr": (
                    record.dpr_attachment.public()
                    if record is not None and record.dpr_attachment is not None
                    else None
                ),
            }
            self._unlock_locked(slot, unbind=False)
            return public

    def abort(self, lease: EngineLease, *, unbind: bool = False) -> None:
        with self._condition:
            slot = self._slots[lease.slot_index]
            self._unlock_locked(slot, unbind=unbind)

    def _unlock_locked(self, slot: EngineSlot, *, unbind: bool) -> None:
        if unbind and slot.session_id is not None:
            self._sessions.pop(slot.session_id, None)
            slot.session_id = None
        slot.busy = False
        slot.request_key = None
        slot.predicted_work_ratio = None
        slot.estimated_finish_monotonic = 0.0
        slot.estimated_prefill_end_monotonic = 0.0
        self._condition.notify_all()

    def release(self, lease: EngineLease) -> None:
        with self._condition:
            slot = self._slots[lease.slot_index]
            record = self._records.get(lease.session_id)
            if record is not None:
                record.state = "released"
                record.slot_index = None
                record.manifest = None
            if slot.session_id != lease.session_id:
                raise RuntimeError("scheduler release ownership drift")
            self._unlock_locked(slot, unbind=True)

    def slot_for_session(self, session_id: str) -> EngineSlot:
        session_id = self.validate_session_id(session_id)
        with self._condition:
            slot_index = self._sessions.get(session_id)
            if slot_index is None:
                raise KeyError(session_id)
            return self._slots[slot_index]

    def slot(self, index: int) -> EngineSlot:
        return self._slots[index]

    def health(self, children: list[dict[str, Any]]) -> dict[str, Any]:
        with self._condition:
            now = time.monotonic()
            slots = [{
                "index": slot.index,
                "busy": slot.busy,
                "session_bound": slot.session_id is not None,
                "session_sha256": (
                    self.session_sha256(slot.session_id)
                    if slot.session_id is not None else None
                ),
                "work_class": (
                    "heavy" if slot.predicted_work_ratio is None or
                    slot.predicted_work_ratio >= self.heavy_work_ratio
                    else "hit"
                ) if slot.busy else None,
                "estimated_remaining_s": (
                    round(max(0.0, slot.estimated_finish_monotonic - now), 6)
                    if slot.estimated_finish_monotonic else None
                ),
                "estimated_prefill_remaining_s": (
                    round(max(
                        0.0, slot.estimated_prefill_end_monotonic - now,
                    ), 6)
                    if slot.estimated_prefill_end_monotonic and
                    math.isfinite(slot.estimated_prefill_end_monotonic)
                    else None
                ),
                "completed_requests": slot.completed_requests,
                "generation": slot.generation,
                "failed": slot.failed,
                "dpr": dict(slot.dpr) if slot.dpr is not None else None,
            } for slot in self._slots]
            queued = self._queued
            sessions = len(self._sessions)
            spilled = sum(
                1 for record in self._records.values()
                if record.state == "spilled"
            )
        running = sum(1 for item in children if item.get("status") == "ok")
        failed = any(item["failed"] for item in slots)
        return {
            "status": (
                "ok" if running == len(self._slots) and not failed else "unavailable"
            ),
            "runtime_ready": (
                not failed and
                all(item.get("runtime_ready") is True for item in children)
            ),
            "engine_processes": len(self._slots),
            "running_engines": running,
            "busy_engines": sum(1 for item in slots if item["busy"]),
            "queued_requests": queued,
            "bound_sessions": sessions,
            "spilled_sessions": spilled,
            "scheduler": "session-affine-dpr-estimated",
            "engines": slots,
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "mode": "session-affine-dpr-estimated",
            "engine_processes": len(self._slots),
            "session_field": "session",
            "session_id_field": "session_id",
            "session_create_endpoint": "/v1/sessions",
            "default_retention": "hosting-period",
            "eviction_policy": "idle-lru-export-v5",
            "dpr_default": "off",
            "dpr_request_field": "dpr",
            "session_header": "X-Salt-Session",
            "release_endpoint": "/v1/session/release",
            "wait_s": self.wait_s,
            "prefill_ms_per_token": self.prefill_ms_per_token,
            "prefill_overlap": "bounded-by-max-heavy-active",
            "max_heavy_active": self.max_heavy_active,
            "heavy_work_ratio": self.heavy_work_ratio,
            "native_hit_authority": True,
            "request_migration": False,
            "state_authority": "child-native-engine",
        }


def child_exchange(slot: EngineSlot, method: str, path: str,
                   body: bytes | None = None,
                   headers: dict[str, str] | None = None,
                   *, timeout: float = 900.0
                   ) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection(slot.host, slot.port, timeout=timeout)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    content_type = response.getheader("Content-Type", "application/octet-stream")
    length = response.getheader("Content-Length")
    if length is not None and int(length) > DEFAULT_RESPONSE_BYTES:
        connection.close()
        raise RuntimeError("child response exceeds scheduler limit")
    payload = response.read(DEFAULT_RESPONSE_BYTES + 1)
    status = response.status
    connection.close()
    if len(payload) > DEFAULT_RESPONSE_BYTES:
        raise RuntimeError("child response exceeds scheduler limit")
    return status, content_type, payload


def child_json(slot: EngineSlot, method: str, path: str,
               body: dict[str, Any] | None = None,
               headers: dict[str, str] | None = None,
               *, timeout: float = 900.0
               ) -> tuple[int, dict[str, Any]]:
    encoded = None
    request_headers = dict(headers or {})
    if body is not None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    status, _content_type, payload = child_exchange(
        slot, method, path, encoded, request_headers, timeout=timeout,
    )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("child JSON response must be an object")
    return status, value


def handler_for(pool: EnginePool, *, child_timeout_s: float = 900.0,
                max_response_bytes: int = DEFAULT_RESPONSE_BYTES,
                dpr_catalog: DprCatalog | None = None):
    if not 1 <= max_response_bytes <= MAX_PROXY_RESPONSE_BYTES:
        raise ValueError("scheduler response bound is invalid")
    contribution_lock = threading.Lock()
    class SchedulerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(BODY_TIMEOUT_S)

        def log_message(self, format, *args):
            sys.stderr.write("[gemma4-scheduler] %s\n" % (format % args))

        def _send(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: int, value: dict[str, Any]) -> None:
            self._send(
                status, "application/json",
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
            )

        def _error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": {
                "message": message, "type": "server_error",
            }})

        def _session_header(self) -> str | None:
            values = self.headers.get_all("X-Salt-Session", [])
            if len(values) > 1:
                raise ValueError("at most one X-Salt-Session is accepted")
            return values[0] if values else None

        def _body(self) -> dict[str, Any]:
            if self.headers.get_all("Transfer-Encoding", []):
                raise ValueError("Transfer-Encoding is not supported")
            values = self.headers.get_all("Content-Length", [])
            if len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None:
                raise ValueError("exactly one decimal Content-Length is required")
            length = int(values[0])
            if length > MAX_REQUEST_BYTES:
                raise ValueError("request body exceeds 4 MiB")
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("truncated request body")
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def _child_headers(self) -> dict[str, str]:
            result = {"Content-Type": "application/json"}
            keys = self.headers.get_all("Idempotency-Key", [])
            if len(keys) > 1:
                raise ValueError("at most one Idempotency-Key is accepted")
            if keys:
                result["Idempotency-Key"] = keys[0]
            return result

        def _resolve(self, body: dict[str, Any]
                     ) -> tuple[
                         str, dict[str, Any], float | None, DprAttachment | None,
                     ]:
            session_id, request_body, retention = pool.resolve_session(
                body, self._session_header(),
            )
            if "dpr" not in request_body:
                attachment = pool.session_dpr(session_id)
            else:
                selector = request_body.pop("dpr")
                if selector is None:
                    attachment = None
                else:
                    if not isinstance(selector, dict) or set(selector) != {
                            "id", "version"}:
                        raise ValueError("dpr must contain exactly id and version")
                    if dpr_catalog is None:
                        raise ValueError("DPR catalog is not configured")
                    attachment = dpr_catalog.resolve(
                        selector["id"], selector["version"],
                    )
            return session_id, request_body, retention, attachment

        def _prepare_session(self, session_id: str,
                             retention_seconds: float | None,
                             dpr_attachment: DprAttachment | None) -> None:
            maintenance = pool.reserve_session(
                session_id, retention_seconds, dpr_attachment,
            )
            if maintenance is None:
                return
            slot = pool.slot(maintenance.slot_index)
            evicted_manifest = None
            state_may_have_changed = False
            try:
                if maintenance.expired_cache_id is not None:
                    status, _ = child_json(
                        slot, "POST", "/v1/session/cache/delete",
                        body={"id": maintenance.expired_cache_id},
                        timeout=child_timeout_s,
                    )
                    if status != 200:
                        raise RuntimeError("expired session cache deletion failed")
                    pool.expire_session_spill(maintenance)
                if maintenance.evicted_session is not None:
                    assert maintenance.evicted_cache_id is not None
                    status, exported = child_json(
                        slot, "POST", "/v1/session/export",
                        body={
                            "id": maintenance.evicted_cache_id,
                            "replace": True,
                        },
                        timeout=child_timeout_s,
                    )
                    if status != 200:
                        raise RuntimeError("child session export failed")
                    session_manifest = exported.get("session_manifest")
                    if not isinstance(session_manifest, dict):
                        raise RuntimeError("child session export omitted manifest")
                    evicted_manifest = {
                        **session_manifest,
                        "state_sha256": exported.get("state_sha256"),
                        "facts_sha256": exported.get("facts_sha256"),
                    }
                    state_may_have_changed = True
                    status, _ = child_json(
                        slot, "POST", "/v1/session/reset", body={},
                        timeout=child_timeout_s,
                    )
                    if status != 200:
                        raise RuntimeError("child session reset after export failed")
                if maintenance.reconfigure_dpr:
                    state_may_have_changed = True
                    status, _ = child_json(
                        slot, "POST", "/v1/session/internal/dpr",
                        body=maintenance.dpr_control,
                        timeout=child_timeout_s,
                    )
                    if status != 200:
                        raise RuntimeError("child DPR reconfiguration failed")
                if maintenance.restore_manifest is not None:
                    assert maintenance.restore_cache_id is not None
                    state_may_have_changed = True
                    status, _ = child_json(
                        slot, "POST", "/v1/session/import",
                        body={
                            "id": maintenance.restore_cache_id,
                            "manifest": maintenance.restore_manifest,
                        },
                        timeout=child_timeout_s,
                    )
                    if status != 200:
                        raise RuntimeError("child session import failed")
                pool.complete_session_maintenance(
                    maintenance, evicted_manifest,
                )
            except BaseException:
                pool.fail_session_maintenance(
                    maintenance,
                    evicted_manifest=evicted_manifest,
                    state_may_have_changed=state_may_have_changed,
                )
                raise

        def _spill_contribution_session(
                self, session_id: str) -> tuple[str, dict[str, Any]]:
            spill, cache_id, existing = pool.reserve_contribution_spill(session_id)
            if spill is None:
                assert existing is not None
                return cache_id, existing
            slot = pool.slot(spill.slot_index)
            state_may_have_changed = False
            try:
                status, exported = child_json(
                    slot, "POST", "/v1/session/export",
                    body={"id": cache_id, "replace": True},
                    timeout=child_timeout_s,
                )
                if status != 200:
                    raise RuntimeError("contribution session export failed")
                session_manifest = exported.get("session_manifest")
                if not isinstance(session_manifest, dict):
                    raise RuntimeError("contribution export omitted manifest")
                manifest = {
                    **session_manifest,
                    "state_sha256": exported.get("state_sha256"),
                    "facts_sha256": exported.get("facts_sha256"),
                }
                state_may_have_changed = True
                status, _ = child_json(
                    slot, "POST", "/v1/session/reset", body={},
                    timeout=child_timeout_s,
                )
                if status != 200:
                    raise RuntimeError("contribution session reset failed")
                pool.complete_contribution_spill(spill, manifest)
                return cache_id, manifest
            except BaseException:
                pool.fail_contribution_spill(
                    spill, state_may_have_changed=state_may_have_changed,
                )
                raise

        def _release_training_session(self, session_id: str) -> None:
            try:
                lease = pool.acquire(
                    session_id, {}, "/v1/session/release", existing=True,
                )
            except (KeyError, SchedulerUnavailable):
                return
            slot = pool.slot(lease.slot_index)
            try:
                status, _ = child_json(
                    slot, "POST", "/v1/session/reset", body={},
                    timeout=child_timeout_s,
                )
                if status != 200:
                    pool.abort(lease, unbind=True)
                    return
                pool.release(lease)
            except BaseException:
                pool.abort(lease, unbind=True)

        def _reprocess_contribution(
                self, candidate: DprAttachment, contribution_id: str,
                manifest: dict[str, Any]) -> list[dict[str, object]]:
            history = manifest.get("history")
            turns = manifest.get("turns")
            if not isinstance(history, list) or not isinstance(turns, list) or \
                    len(history) != len(turns) * 2:
                raise ValueError("contribution history/provenance mismatch")
            training_session = "dprtrain-" + contribution_id[:32]
            self._prepare_session(training_session, None, candidate)
            observed: list[dict[str, object]] = []
            try:
                for index, expected in enumerate(turns):
                    user = history[index * 2]
                    assistant = history[index * 2 + 1]
                    if (not isinstance(user, dict) or user.get("role") != "user" or
                            user.get("image_url") is not None or
                            not isinstance(user.get("content"), str) or
                            not isinstance(assistant, dict) or
                            assistant.get("role") != "assistant" or
                            not isinstance(expected, dict)):
                        raise ValueError(
                            "DPR contribution currently requires text-only canonical turns"
                        )
                    body = {
                        "model": "gemma-4-26b-a4b-it",
                        "messages": [{
                            "role": "user", "content": user["content"],
                        }],
                        "max_tokens": expected.get("output_limit"),
                        "stream": False,
                    }
                    lease = pool.acquire(
                        training_session, body, "/v1/chat/completions",
                    )
                    slot = pool.slot(lease.slot_index)
                    try:
                        status, value = child_json(
                            slot, "POST", "/v1/chat/completions", body=body,
                            timeout=child_timeout_s,
                        )
                        if status != 200:
                            raise RuntimeError("DPR contribution reprocess failed")
                        extension = value.get("x_salt")
                        choices = value.get("choices")
                        if (not isinstance(extension, dict) or
                                not isinstance(choices, list) or not choices):
                            raise RuntimeError("DPR reprocess response is malformed")
                        prompt_ids = extension.get("prompt_ids")
                        output_ids = extension.get("output_ids")
                        content = choices[0].get("message", {}).get("content")
                        response_sha256 = extension.get("response_sha256")
                        usage = value.get("usage", {})
                        pool.finish(
                            lease, elapsed_s=0.0,
                            dpr_result=extension.get("dpr"),
                            prompt_tokens=len(prompt_ids) if isinstance(prompt_ids, list) else 0,
                            completion_tokens=(
                                usage.get("completion_tokens", 0)
                                if isinstance(usage, dict) else 0
                            ),
                        )
                        if (prompt_ids != expected.get("prompt_ids") or
                                output_ids != expected.get("output_ids") or
                                content != expected.get("content") or
                                content != assistant.get("content") or
                                response_sha256 != expected.get("response_sha256")):
                            raise ValueError(
                                f"DPR contribution reproduction mismatch at turn {index + 1}"
                            )
                        observed.append({
                            "turn": index + 1,
                            "prompt_ids": prompt_ids,
                            "output_ids": output_ids,
                            "response_sha256": response_sha256,
                        })
                    except BaseException:
                        if slot.busy:
                            pool.abort(lease)
                        raise
                return observed
            finally:
                self._release_training_session(training_session)
                cleanup_session = "dproff-" + contribution_id[:32]
                try:
                    self._prepare_session(cleanup_session, None, None)
                    self._release_training_session(cleanup_session)
                except BaseException:
                    pass

        def _contribute(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            if dpr_catalog is None:
                raise ValueError("DPR catalog is not configured")
            if self.headers.get_all("Idempotency-Key", []):
                raise ValueError("DPR contribution does not accept Idempotency-Key")
            request = dict(body)
            selector = request.pop("dpr", None)
            if not isinstance(selector, dict) or set(selector) != {"id", "version"}:
                raise ValueError(
                    "DPR contribution requires explicit dpr id and version"
                )
            session_id, remainder, retention = pool.resolve_session(
                request, self._session_header(),
            )
            if remainder or retention is not None:
                raise ValueError(
                    "DPR contribution accepts only session selector and dpr"
                )
            base = dpr_catalog.resolve(selector["id"], selector["version"])
            with contribution_lock:
                cache_id, manifest = self._spill_contribution_session(session_id)
                prepared = dpr_catalog.write_contribution({
                    "status": "prepared",
                    "session_sha256": pool.session_sha256(session_id),
                    "session_cache_id": cache_id,
                    "base_dpr": base.public(),
                    "session_manifest": manifest,
                })
                candidate = dpr_catalog.begin_candidate(base, str(prepared["id"]))
                try:
                    observed = self._reprocess_contribution(
                        candidate, str(prepared["id"]), manifest,
                    )
                    promoted = dpr_catalog.seal_candidate(candidate)
                except BaseException as exc:
                    if candidate.root.exists():
                        dpr_catalog.abort_candidate(candidate)
                    rejected = dpr_catalog.write_contribution({
                        "status": "rejected",
                        "prepared_contribution_id": prepared["id"],
                        "base_dpr": base.public(),
                        "reason": str(exc)[:512],
                    })
                    return 409, {**rejected, "base_dpr": base.public()}
                accepted = dpr_catalog.write_contribution({
                    "status": "accepted",
                    "prepared_contribution_id": prepared["id"],
                    "base_dpr": base.public(),
                    "promoted_dpr": promoted.public(),
                    "observed_turns": observed,
                })
            return 200, {
                **accepted,
                "base_dpr": base.public(),
                "promoted_dpr": promoted.public(),
            }

        def do_GET(self):
            try:
                session_id = pool.validate_session_id(
                    self._session_header() or "default"
                )
                path = self.path.rstrip("/")
                if path == "/healthz":
                    children = []
                    for index in range(len(pool._slots)):
                        try:
                            status, value = child_json(
                                pool.slot(index), "GET", "/healthz", timeout=2.0,
                            )
                            if status != 200:
                                value = {"status": "unavailable"}
                        except Exception:
                            value = {"status": "unavailable", "runtime_ready": False}
                        children.append(value)
                    health = pool.health(children)
                    self._send_json(200 if health["status"] == "ok" else 503, health)
                    return
                if path == "/v1/models":
                    status, value = child_json(
                        pool.slot(0), "GET", path, timeout=child_timeout_s,
                    )
                    if status == 200:
                        value["data"][0]["x_salt"]["scheduler"] = pool.capabilities()
                    self._send_json(status, value)
                    return
                if path == "/v1/dpr":
                    self._send_json(200, {
                        "object": "list",
                        "data": dpr_catalog.list_versions() if dpr_catalog else [],
                        "default": None,
                    })
                    return
                if path not in ("/v1/session", "/v1/kv-caches"):
                    self._error(404, "not found")
                    return
                try:
                    slot = pool.slot_for_session(session_id)
                except KeyError:
                    self._error(404, "unknown session")
                    return
                status, value = child_json(
                    slot, "GET", path, timeout=child_timeout_s,
                )
                value["scheduler"] = {
                    **pool.capabilities(),
                    "engine_index": slot.index,
                    "session_sha256": pool.session_sha256(session_id),
                }
                self._send_json(status, value)
            except (ValueError, json.JSONDecodeError) as exc:
                self._error(400, str(exc))
            except Exception as exc:
                self._error(502, f"child engine failed: {exc}")

        def do_POST(self):
            lease = None
            try:
                body = self._body()
                path = self.path.rstrip("/")
                if path == "/v1/sessions":
                    if set(body) - {"retention_seconds", "dpr"}:
                        raise ValueError(
                            "session creation accepts only retention_seconds and dpr"
                        )
                    retention_seconds = pool.validate_retention(
                        body.get("retention_seconds")
                    )
                    attachment = None
                    if body.get("dpr") is not None:
                        selector = body["dpr"]
                        if not isinstance(selector, dict) or set(selector) != {
                                "id", "version"}:
                            raise ValueError("dpr must contain exactly id and version")
                        if dpr_catalog is None:
                            raise ValueError("DPR catalog is not configured")
                        attachment = dpr_catalog.resolve(
                            selector["id"], selector["version"],
                        )
                    session_id = pool.create_session_id(
                        retention_seconds, attachment,
                    )
                    self._send_json(201, {
                        "id": session_id,
                        "session_id": session_id,
                        "object": "salt.session",
                        "state": "unbound",
                        "retention": (
                            {"mode": "hosting-period"}
                            if retention_seconds is None else
                            {"mode": "ttl", "seconds": retention_seconds}
                        ),
                        "dpr": attachment.public() if attachment else None,
                    })
                    return
                if path == "/v1/session/contribute":
                    status, contribution = self._contribute(body)
                    self._send_json(status, contribution)
                    return
                if path == "/v1/session/checkpoint":
                    if self.headers.get_all("Idempotency-Key", []):
                        raise ValueError(
                            "session checkpoint does not accept Idempotency-Key"
                        )
                    session_id, remainder, retention = pool.resolve_session(
                        body, self._session_header(),
                    )
                    if remainder or retention is not None:
                        raise ValueError(
                            "session checkpoint accepts only session identity"
                        )
                    try:
                        cache_id, manifest = self._spill_contribution_session(
                            session_id
                        )
                    except KeyError:
                        self._prepare_session(session_id, None, None)
                        cache_id, manifest = self._spill_contribution_session(
                            session_id
                        )
                    self._send_json(200, {
                        "object": "salt.session.checkpoint",
                        "session_id": session_id,
                        "cache_id": cache_id,
                        "manifest": manifest,
                    })
                    return
                if path == "/v1/session/bootstrap":
                    if self.headers.get_all("Idempotency-Key", []):
                        raise ValueError(
                            "session bootstrap does not accept Idempotency-Key"
                        )
                    request = dict(body)
                    cache_id = request.pop("cache_id", None)
                    manifest = request.pop("manifest", None)
                    session_id, remainder, retention = pool.resolve_session(
                        request, self._session_header(),
                    )
                    if (remainder or retention is not None or
                            not isinstance(cache_id, str) or
                            not isinstance(manifest, dict)):
                        raise ValueError(
                            "session bootstrap requires cache_id, manifest, and session identity"
                        )
                    pool.seed_spilled_session(session_id, cache_id, manifest)
                    self._prepare_session(session_id, None, None)
                    self._send_json(200, {
                        "object": "salt.session.bootstrap",
                        "session_id": session_id,
                        "cache_id": cache_id,
                        "state": "resident",
                    })
                    return
                session_id, request_body, retention_seconds, attachment = \
                    self._resolve(body)
                if path not in (
                        "/v1/chat/completions", "/v1/responses",
                        "/v1/session/reset", "/v1/session/clear",
                        "/v1/session/release"):
                    self._error(404, "not found")
                    return
                control = path.startswith("/v1/session/")
                if control and request_body:
                    raise ValueError(
                        "session clear/release accepts only the session field"
                    )
                if control and retention_seconds is not None:
                    raise ValueError(
                        "session control does not accept retention_seconds"
                    )
                if not control:
                    self._prepare_session(
                        session_id, retention_seconds, attachment,
                    )
                lease = pool.acquire(
                    session_id, request_body, path, existing=control,
                )
                slot = pool.slot(lease.slot_index)
                child_body = {} if control else request_body
                headers = self._child_headers()
                encoded = json.dumps(
                    child_body, separators=(",", ":"),
                ).encode("utf-8")
                begin = time.monotonic()
                connection = http.client.HTTPConnection(
                    slot.host, slot.port, timeout=child_timeout_s,
                )
                child_path = (
                    "/v1/session/reset"
                    if path == "/v1/session/release" else path
                )
                connection.request(
                    "POST", child_path, body=encoded, headers=headers,
                )
                response = connection.getresponse()
                content_type = response.getheader(
                    "Content-Type", "application/octet-stream",
                )
                streaming = (
                    response.status == 200 and
                    content_type.startswith("text/event-stream")
                )
                if streaming:
                    client_disconnected = False
                    try:
                        self.send_response(response.status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Salt-Session", session_id)
                        self.send_header("Connection", "close")
                        self.end_headers()
                    except OSError:
                        client_disconnected = True
                        self.close_connection = True
                    stream_buffer = b""
                    streamed_bytes = 0
                    while True:
                        chunk = response.readline(max_response_bytes + 1)
                        if not chunk:
                            break
                        streamed_bytes += len(chunk)
                        if streamed_bytes > max_response_bytes:
                            raise RuntimeError(
                                "child response exceeds scheduler limit"
                            )
                        stream_buffer, chunk = _session_sse_bytes(
                            stream_buffer, chunk, session_id,
                        )
                        if not client_disconnected:
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except OSError:
                                client_disconnected = True
                                self.close_connection = True
                    if stream_buffer and not client_disconnected:
                        _, chunk = _session_sse_bytes(
                            stream_buffer, b"", session_id, final=True,
                        )
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except OSError:
                            client_disconnected = True
                            self.close_connection = True
                    status = response.status
                    value = None
                else:
                    payload = response.read(max_response_bytes + 1)
                    if len(payload) > max_response_bytes:
                        raise RuntimeError("child response exceeds scheduler limit")
                    status = response.status
                    value = json.loads(payload)
                    if not isinstance(value, dict):
                        raise RuntimeError("child JSON response must be an object")
                connection.close()
                elapsed_s = time.monotonic() - begin

                if control:
                    if status == 200 and path == "/v1/session/release":
                        pool.release(lease)
                    else:
                        pool.abort(lease)
                    lease = None
                    if value is not None:
                        _attach_session_identity(value, session_id)
                        value["scheduler"] = {
                            **pool.capabilities(),
                            "engine_index": slot.index,
                            "session_sha256": pool.session_sha256(session_id),
                            "released": path == "/v1/session/release" and status == 200,
                        }
                        self._send_json(status, value)
                    return

                if status != 200:
                    pool.abort(lease)
                    lease = None
                    if value is not None:
                        self._send_json(status, value)
                    return

                dpr_result = None
                prompt_tokens = 0
                completion_tokens = 0
                replayed = False
                if status == 200:
                    try:
                        _session_status, child_status = child_json(
                            slot, "GET", "/v1/session", timeout=2.0,
                        )
                        dpr = child_status.get("dpr")
                        if isinstance(dpr, dict):
                            dpr_result = dpr.get("last_result")
                    except Exception:
                        dpr_result = None
                    if value is not None:
                        extension = value.get("x_salt")
                        if isinstance(extension, dict):
                            prompt_ids = extension.get("prompt_ids")
                            if isinstance(prompt_ids, list):
                                prompt_tokens = len(prompt_ids)
                            replayed = extension.get("replayed") is True
                        usage = value.get("usage")
                        if isinstance(usage, dict):
                            raw_completion = usage.get(
                                "completion_tokens", usage.get("output_tokens", 0),
                            )
                            if isinstance(raw_completion, int):
                                completion_tokens = raw_completion
                schedule = pool.finish(
                    lease, elapsed_s=elapsed_s, dpr_result=dpr_result,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    replayed=replayed,
                )
                lease = None
                if value is not None:
                    if status == 200 and isinstance(value.get("x_salt"), dict):
                        value["x_salt"]["scheduler"] = schedule
                    if status == 200:
                        _attach_session_identity(value, session_id)
                    self._send_json(status, value)
            except KeyError:
                if lease is not None:
                    pool.abort(lease)
                self._error(404, "unknown session")
            except SchedulerUnavailable as exc:
                if lease is not None:
                    pool.abort(lease)
                self._error(503, str(exc))
            except (ValueError, json.JSONDecodeError) as exc:
                if lease is not None:
                    pool.abort(lease)
                self._error(400, str(exc))
            except BrokenPipeError:
                if lease is not None:
                    pool.abort(lease)
                self.close_connection = True
            except Exception as exc:
                if lease is not None:
                    pool.abort(lease)
                self._error(502, f"child engine failed: {exc}")

        def _method_not_allowed(self):
            self._error(405, "method not allowed")

        do_PUT = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_OPTIONS = _method_not_allowed
        do_TRACE = _method_not_allowed
        do_CONNECT = _method_not_allowed

    return SchedulerHandler


def wait_ready(port: int, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    slot = EngineSlot(index=0, host="127.0.0.1", port=port)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"child server on port {port} exited with {process.returncode}"
            )
        try:
            status, value = child_json(slot, "GET", "/healthz", timeout=1.0)
            if status == 200 and value.get("status") == "ok":
                return
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    raise RuntimeError(f"child server on port {port} did not become ready")


def stop_children(children: list[subprocess.Popen[bytes]]) -> None:
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    for child in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            child.wait(timeout=5.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("GEMMA4_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GEMMA4_PORT", "8090")))
    parser.add_argument("--engine-processes", type=int,
                        default=int(os.environ.get("GEMMA4_ENGINE_PROCESSES", "1")))
    parser.add_argument("--scheduler-wait-s", type=float,
                        default=float(os.environ.get("GEMMA4_SCHEDULER_WAIT_S", "60.0")))
    parser.add_argument(
        "--prefill-ms-per-token", type=float,
        default=float(os.environ.get(
            "GEMMA4_SCHEDULER_PREFILL_MS_PER_TOKEN", "0.0",
        )),
        help="measured platform prefill cost used only for admission timing",
    )
    parser.add_argument(
        "--max-heavy-active", type=int,
        default=int(os.environ.get("GEMMA4_SCHEDULER_MAX_HEAVY", "1")),
    )
    parser.add_argument(
        "--heavy-work-ratio", type=float,
        default=float(os.environ.get(
            "GEMMA4_SCHEDULER_HEAVY_WORK_RATIO", "0.25",
        )),
    )
    parser.add_argument("--child-port-base", type=int,
                        default=int(os.environ.get("GEMMA4_CHILD_PORT_BASE", "18090")))
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--child-timeout-s", type=float, default=900.0)
    parser.add_argument("--python", default=os.environ.get("GEMMA4_PYTHON", sys.executable))
    parser.add_argument("--serve-script", type=Path,
                        default=ROOT / "server" / "serve.py")
    parser.add_argument(
        "--dpr-catalog-root", type=Path,
        default=(Path(os.environ["GEMMA4_DPR_CATALOG_ROOT"])
                 if os.environ.get("GEMMA4_DPR_CATALOG_ROOT") else None),
    )
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.engine_processes <= MAX_ENGINE_PROCESSES:
        raise SystemExit(
            f"engine process count must be in [1, {MAX_ENGINE_PROCESSES}]"
        )
    if not math.isfinite(args.scheduler_wait_s) or not 0.0 < args.scheduler_wait_s <= 60.0:
        raise SystemExit("scheduler wait must be finite and in (0, 60]")
    if (not math.isfinite(args.prefill_ms_per_token) or
            args.prefill_ms_per_token < 0.0):
        raise SystemExit("prefill ms/token must be finite and nonnegative")
    if not 1 <= args.max_heavy_active <= args.engine_processes:
        raise SystemExit("max heavy active must be within engine process count")
    if (not math.isfinite(args.heavy_work_ratio) or
            not 0.0 < args.heavy_work_ratio <= 1.0):
        raise SystemExit("heavy work ratio must be finite and in (0, 1]")
    python = Path(args.python)
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit("--python must be an absolute executable interpreter")
    serve_script = args.serve_script.resolve()
    if not serve_script.is_file():
        raise SystemExit("--serve-script does not exist")
    child_args = list(args.child_args)
    if child_args and child_args[0] == "--":
        child_args.pop(0)
    forbidden = {"--backend", "--host", "--port", "--gemma-verify-only"}
    if any(value in forbidden for value in child_args):
        raise SystemExit(
            "child arguments cannot override backend, host, port, or verify-only"
        )
    def child_int(name: str, default: int) -> int:
        matches = [index for index, value in enumerate(child_args) if value == name]
        if len(matches) > 1 or (matches and matches[0] + 1 >= len(child_args)):
            raise SystemExit(f"invalid child setting {name}")
        if not matches:
            return default
        try:
            value = int(child_args[matches[0] + 1])
        except ValueError as exc:
            raise SystemExit(f"invalid child setting {name}") from exc
        if value < 1:
            raise SystemExit(f"invalid child setting {name}")
        return value

    child_context = child_int("--gemma-context-tokens", 512)
    child_output = child_int("--gemma-max-output-tokens", 128)
    max_response_bytes = 1024 * 1024 + child_context * 12 + child_output * 76
    if max_response_bytes > MAX_PROXY_RESPONSE_BYTES:
        raise SystemExit("configured child response exceeds scheduler proxy ceiling")
    ports = [args.child_port_base + index for index in range(args.engine_processes)]
    if any(not 1 <= port <= 65535 for port in [args.port, *ports]):
        raise SystemExit("scheduler or child port is outside [1, 65535]")
    if args.port in ports and args.host in ("127.0.0.1", "localhost"):
        raise SystemExit("scheduler port overlaps a child port")

    pool = EnginePool(
        [("127.0.0.1", port) for port in ports],
        wait_s=args.scheduler_wait_s,
        prefill_ms_per_token=args.prefill_ms_per_token,
        max_heavy_active=args.max_heavy_active,
        heavy_work_ratio=args.heavy_work_ratio,
    )
    dpr_catalog = DprCatalog(args.dpr_catalog_root) \
        if args.dpr_catalog_root is not None else None
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_for(
            pool, child_timeout_s=args.child_timeout_s,
            max_response_bytes=max_response_bytes,
            dpr_catalog=dpr_catalog,
        ),
    )
    children: list[subprocess.Popen[bytes]] = []
    prior_term = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for port in ports:
            command = [
                str(python), str(serve_script), "--backend", "gemma4",
                *child_args, "--host", "127.0.0.1", "--port", str(port),
            ]
            child = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=None, env={key: value for key, value in os.environ.items()
                                  if key != "PYTHONPATH"},
                start_new_session=True,
            )
            children.append(child)
            wait_ready(port, child, args.startup_timeout_s)
            print(
                f"[gemma4-scheduler] child READY index={len(children)-1} "
                f"pid={child.pid} port={port}",
                file=sys.stderr, flush=True,
            )
        print(
            f"[gemma4-scheduler] API http://{args.host}:{args.port} "
            f"engine-processes={len(children)} "
            "scheduler=session-affine-dpr-estimated",
            file=sys.stderr, flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gemma4-scheduler] draining", file=sys.stderr, flush=True)
    finally:
        server.server_close()
        stop_children(children)
        signal.signal(signal.SIGTERM, prior_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
