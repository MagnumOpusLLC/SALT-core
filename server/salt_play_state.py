#!/usr/bin/env python3
"""Canonical portable play state for sampling and tool-task continuation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA = "salt.play-state.v1"
PLAY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
MENTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}@[0-9a-f]{64}$")
BODY_KEYS = {
    "schema", "play_id", "parent_play_revision", "model_id", "quant_track",
    "state_compatibility_sha256", "state_file_sha256", "state_position",
    "mentor_selector", "sampler", "tool_context",
    "created_from_request_sha256", "runtime_ready",
}
FULL_KEYS = BODY_KEYS | {"revision"}
SAMPLER_KEYS = {
    "mode", "sampler_abi", "temperature", "top_k", "top_p",
    "repetition_penalty", "frequency_penalty", "presence_penalty",
    "rng_abi", "initial_seed", "rng_state_hex", "rng_draw_count",
}
TOOL_KEYS = {
    "mode", "task_id", "journal_abi", "journal_head_sha256", "event_count",
    "pending_call_sha256", "side_effect_authority",
}


class PlayStateError(ValueError):
    """Portable play-state schema or artifact validation failed."""


@dataclass(frozen=True)
class PlayStateMetadata:
    play_id: str
    revision: str
    parent_play_revision: str | None
    model_id: str
    quant_track: str
    state_compatibility_sha256: str
    state_file_sha256: str
    state_position: int
    mentor_selector: str | None
    sampler_mode: str
    sampler_abi: str
    initial_seed: int | None
    rng_state_hex: str | None
    rng_draw_count: int
    tool_mode: str
    task_id: str | None
    journal_head_sha256: str | None
    event_count: int


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hex64(value: object, field: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise PlayStateError(f"invalid {field}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlayStateError(f"invalid {field}")
    result = float(value)
    if not math.isfinite(result):
        raise PlayStateError(f"invalid {field}")
    return result


def _integer(value: object, field: str, minimum: int,
             maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlayStateError(f"invalid {field}")
    if maximum is not None and value > maximum:
        raise PlayStateError(f"invalid {field}")
    return value


def _text(value: object, field: str, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\0" in value:
        raise PlayStateError(f"invalid {field}")
    return value


def _validate_sampler(value: object) -> tuple[str, str, int | None, str | None, int]:
    if not isinstance(value, dict) or set(value) != SAMPLER_KEYS:
        raise PlayStateError("invalid sampler fields")
    mode = value["mode"]
    repetition = _finite(value["repetition_penalty"], "repetition_penalty")
    frequency = _finite(value["frequency_penalty"], "frequency_penalty")
    presence = _finite(value["presence_penalty"], "presence_penalty")
    if repetition <= 0.0 or abs(frequency) > 100.0 or abs(presence) > 100.0:
        raise PlayStateError("invalid sampling penalties")
    if mode == "greedy":
        expected = {
            "mode": "greedy",
            "sampler_abi": "salt-greedy-v1",
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "rng_abi": "none",
            "initial_seed": None,
            "rng_state_hex": None,
            "rng_draw_count": 0,
        }
        if value != expected:
            raise PlayStateError("invalid greedy sampler state")
        return "greedy", "salt-greedy-v1", None, None, 0
    if mode != "sampled":
        raise PlayStateError("invalid sampler mode")
    if value["sampler_abi"] != "salt-vllm-topk-topp-v1":
        raise PlayStateError("invalid sampled sampler ABI")
    temperature = _finite(value["temperature"], "temperature")
    top_k = _integer(value["top_k"], "top_k", 1, 1_000_000)
    top_p = _finite(value["top_p"], "top_p")
    del top_k
    if temperature <= 0.0 or temperature > 100.0:
        raise PlayStateError("invalid sampled temperature")
    if top_p <= 0.0 or top_p > 1.0:
        raise PlayStateError("invalid top_p")
    if value["rng_abi"] != "xorshift64-v1":
        raise PlayStateError("invalid sampled RNG ABI")
    seed = _integer(value["initial_seed"], "initial seed", 0, (1 << 64) - 1)
    rng_state = value["rng_state_hex"]
    if (not isinstance(rng_state, str) or HEX16_RE.fullmatch(rng_state) is None or
            rng_state == "0" * 16):
        raise PlayStateError("invalid sampled RNG state")
    draws = _integer(value["rng_draw_count"], "RNG draw count", 0)
    return "sampled", "salt-vllm-topk-topp-v1", seed, rng_state, draws


def _validate_tools(value: object) -> tuple[str, str | None, str | None, int]:
    if not isinstance(value, dict) or set(value) != TOOL_KEYS:
        raise PlayStateError("invalid tool context fields")
    if value["side_effect_authority"] != "external-governed-executor":
        raise PlayStateError("invalid side-effect authority")
    mode = value["mode"]
    if mode == "none":
        expected = {
            "mode": "none",
            "task_id": None,
            "journal_abi": "none",
            "journal_head_sha256": None,
            "event_count": 0,
            "pending_call_sha256": None,
            "side_effect_authority": "external-governed-executor",
        }
        if value != expected:
            raise PlayStateError("invalid none tool context")
        return "none", None, None, 0
    if mode != "journal":
        raise PlayStateError("invalid tool context mode")
    task_id = _text(value["task_id"], "task_id")
    if value["journal_abi"] != "salt-tool-journal-v1":
        raise PlayStateError("invalid tool journal ABI")
    head = _hex64(value["journal_head_sha256"], "journal head")
    if head == "0" * 64:
        raise PlayStateError("invalid tool journal head")
    events = _integer(value["event_count"], "tool event count", 0)
    pending = _hex64(value["pending_call_sha256"], "pending tool call", nullable=True)
    if pending == "0" * 64:
        raise PlayStateError("invalid pending tool call")
    return "journal", task_id, head, events


def play_state_revision(value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise PlayStateError("play state must be an object")
    body = dict(value)
    body.pop("revision", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _validate_body(value: object) -> tuple[Any, ...]:
    if not isinstance(value, dict) or set(value) != BODY_KEYS:
        raise PlayStateError("invalid play-state fields")
    if value["schema"] != SCHEMA or value["runtime_ready"] is not False:
        raise PlayStateError("invalid play-state schema/readiness")
    play_id = value["play_id"]
    if not isinstance(play_id, str) or PLAY_ID_RE.fullmatch(play_id) is None:
        raise PlayStateError("invalid play_id")
    parent = _hex64(value["parent_play_revision"], "parent play revision", nullable=True)
    model_id = _text(value["model_id"], "model_id")
    quant_track = _text(value["quant_track"], "quant_track", 64)
    compatibility = _hex64(
        value["state_compatibility_sha256"], "state compatibility",
    )
    state_file = _hex64(value["state_file_sha256"], "state file SHA-256")
    position = _integer(value["state_position"], "state position", 0)
    mentor = value["mentor_selector"]
    if mentor is not None and (
        not isinstance(mentor, str) or MENTOR_RE.fullmatch(mentor) is None
    ):
        raise PlayStateError("invalid mentor selector")
    sampler = _validate_sampler(value["sampler"])
    tools = _validate_tools(value["tool_context"])
    request = _hex64(value["created_from_request_sha256"], "request SHA-256")
    return (
        play_id, parent, model_id, quant_track, compatibility, state_file,
        position, mentor, sampler, tools, request,
    )


def create_play_state(body: dict[str, Any]) -> dict[str, Any]:
    _validate_body(body)
    value = json.loads(_canonical(body).decode("utf-8"))
    value["revision"] = play_state_revision(value)
    validate_play_state(value)
    return value


def validate_play_state(value: object) -> PlayStateMetadata:
    if not isinstance(value, dict) or set(value) != FULL_KEYS:
        raise PlayStateError("invalid play-state fields")
    body = dict(value)
    revision = _hex64(body.pop("revision"), "play revision")
    assert revision is not None
    if revision != play_state_revision(body):
        raise PlayStateError("play revision SHA-256 mismatch")
    (play_id, parent, model_id, quant_track, compatibility, state_file,
     position, mentor, sampler, tools, _request) = _validate_body(body)
    return PlayStateMetadata(
        play_id=play_id,
        revision=revision,
        parent_play_revision=parent,
        model_id=model_id,
        quant_track=quant_track,
        state_compatibility_sha256=compatibility,
        state_file_sha256=state_file,
        state_position=position,
        mentor_selector=mentor,
        sampler_mode=sampler[0],
        sampler_abi=sampler[1],
        initial_seed=sampler[2],
        rng_state_hex=sampler[3],
        rng_draw_count=sampler[4],
        tool_mode=tools[0],
        task_id=tools[1],
        journal_head_sha256=tools[2],
        event_count=tools[3],
    )


def canonical_play_state(value: dict[str, Any]) -> bytes:
    validate_play_state(value)
    return _canonical(value) + b"\n"


def write_play_state(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise PlayStateError("invalid play-state path")
    value = create_play_state(body)
    payload = canonical_play_state(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(path, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written < 1:
                raise PlayStateError("play-state write made no progress")
            view = view[written:]
        os.fsync(fd)
    except OSError as exc:
        raise PlayStateError(f"write play state failed: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    return read_play_state(path)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def read_play_state(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise PlayStateError("invalid play-state path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise PlayStateError(f"open play state failed: {exc}") from exc
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                before.st_size < 2 or before.st_size > 64 * 1024):
            raise PlayStateError("play state must be a bounded private regular file")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise PlayStateError("play state must be owned by the server user")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise PlayStateError("play state must have mode 0600 or stricter")
        raw = bytearray()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                raise PlayStateError("play state is truncated")
            raw.extend(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise PlayStateError("play state exceeds encoded size")
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise PlayStateError("play-state identity changed while reading")
        try:
            value = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlayStateError("play state is not canonical JSON") from exc
        if not isinstance(value, dict) or bytes(raw) != _canonical(value) + b"\n":
            raise PlayStateError("play state is not canonical JSON")
        validate_play_state(value)
        return value
    finally:
        if fd >= 0:
            os.close(fd)
