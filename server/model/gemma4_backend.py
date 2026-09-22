#!/usr/bin/env python3
"""Experimental OpenAI-shaped server backend for authenticated Gemma 4 QA.

This module deliberately does not route Gemma through the production Qwen
``salt`` entry point.  It invokes the private authenticated text or one-image
runner, keeps the three qualified CTX/output gates, and exposes the qualified
persistent runtime as ``runtime_ready=true``.
"""

from __future__ import annotations

import base64
import binascii
from collections import deque, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, cast, Iterator
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_KV_CACHE_ROOT = (
    Path.home() / ".salt-sessions" / "gemma4-kv-cache"
)
_sidecar_spec = importlib.util.spec_from_file_location(
    "salt_server_kv_sidecar", ROOT / "server" / "kv_sidecar.py",
)
if _sidecar_spec is None or _sidecar_spec.loader is None:
    raise RuntimeError("cannot load server KV sidecar support")
_kv_sidecar = importlib.util.module_from_spec(_sidecar_spec)
_sidecar_spec.loader.exec_module(_kv_sidecar)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
sys.path.insert(0, str(ROOT / "tools"))
from engine_config import load_engine_config, validate_gemma_residency_config
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
from engine_config import validate_gemma_target_route_config


def _target_warmup_config(config: dict[str, str]) -> dict[str, int]:
    rows_text = config.get("SALT_TARGET_KV_WARMUP_ROWS", "512")
    x_text = config.get("SALT_TARGET_WARM_X", "4")
    if not isinstance(rows_text, str) or not rows_text.isdecimal() or \
            not isinstance(x_text, str) or not x_text.isdecimal():
        raise ValueError(
            "TARGET KV warmup rows and warm X must be canonical decimals"
        )
    rows = int(rows_text)
    x = int(x_text)
    if str(rows) != rows_text or str(x) != x_text or \
            rows < 1 or rows > 262144 or x < 1 or x > 128:
        raise ValueError("invalid TARGET KV warmup policy")
    return {"kv_warmup_rows": rows, "warm_x": x}
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1

GEMMA_ENGINE_CONFIG = load_engine_config(
    model_dir=ROOT / "models" / "gemma4-26b-a4b",
    recipe=os.environ.get("SALT_GEMMA_PLATFORM_RECIPE"),
)
try:
    GEMMA_RESIDENCY_CONFIG = validate_gemma_residency_config(
        GEMMA_ENGINE_CONFIG,
    )
except ValueError as exc:
    raise RuntimeError("invalid Gemma residency recipe") from exc
GEMMA_EXPERT_BUDGET_GB = GEMMA_RESIDENCY_CONFIG["budget_gb"]
GEMMA_EXPERT_BUDGET_BYTES = GEMMA_EXPERT_BUDGET_GB * 1_000_000_000
GEMMA_DPR_MENTOR_POLICY_SHA256 = GEMMA_ENGINE_CONFIG.get(
    "SALT_DPR_MENTOR_POLICY_SHA256", "",
)
if re.fullmatch(r"[0-9a-f]{64}", GEMMA_DPR_MENTOR_POLICY_SHA256) is None:
    raise RuntimeError("invalid Gemma DPR Mentor policy identity")
GEMMA_DPR_MINDSET_ATTENTION_POLICY_SHA256 = GEMMA_ENGINE_CONFIG.get(
    "SALT_DPR_MINDSET_ATTENTION_POLICY_SHA256", "",
)
if re.fullmatch(
        r"[0-9a-f]{64}", GEMMA_DPR_MINDSET_ATTENTION_POLICY_SHA256,
        ) is None:
    raise RuntimeError("invalid Gemma DPR Mindset Attention policy identity")
KV_COMPAT_MANIFEST = ROOT / "models" / "gemma4-26b-a4b" / "kv-cache-compat.json"
_COMPAT_MODULE_NAME = "salt_gemma4_server_compatibility"
_compat_spec = importlib.util.spec_from_file_location(
    _COMPAT_MODULE_NAME, ROOT / "server" / "gemma4_compat.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
_compat_spec = importlib.util.spec_from_file_location(
    _COMPAT_MODULE_NAME, ROOT / "server" / "model" / "gemma4_compat.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
if _compat_spec is None or _compat_spec.loader is None:
    raise RuntimeError("cannot load Gemma 4 KV compatibility authority")
_compat_module = importlib.util.module_from_spec(_compat_spec)
sys.modules[_COMPAT_MODULE_NAME] = _compat_module
_compat_spec.loader.exec_module(_compat_module)
KV_COMPATIBILITY = _compat_module.load_compatibility(KV_COMPAT_MANIFEST, ROOT)
_BUILD_MODULE_NAME = "salt_gemma4_server_build_identity"
_build_spec = importlib.util.spec_from_file_location(
    _BUILD_MODULE_NAME, ROOT / "server" / "gemma4_build.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
_build_spec = importlib.util.spec_from_file_location(
    _BUILD_MODULE_NAME, ROOT / "server" / "model" / "gemma4_build.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
if _build_spec is None or _build_spec.loader is None:
    raise RuntimeError("cannot load Gemma 4 build identity authority")
_build_module = importlib.util.module_from_spec(_build_spec)
sys.modules[_BUILD_MODULE_NAME] = _build_module
_build_spec.loader.exec_module(_build_module)
_STATE_ARTIFACT_MODULE_NAME = "salt_gemma4_server_state_artifact"
_state_artifact_spec = importlib.util.spec_from_file_location(
    _STATE_ARTIFACT_MODULE_NAME, ROOT / "server" / "gemma4_state_artifact.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
_state_artifact_spec = importlib.util.spec_from_file_location(
    _STATE_ARTIFACT_MODULE_NAME,
    ROOT / "server" / "model" / "gemma4_state_artifact.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
if _state_artifact_spec is None or _state_artifact_spec.loader is None:
    raise RuntimeError("cannot load Gemma 4 state artifact authority")
_state_artifact_module = importlib.util.module_from_spec(_state_artifact_spec)
sys.modules[_STATE_ARTIFACT_MODULE_NAME] = _state_artifact_module
_state_artifact_spec.loader.exec_module(_state_artifact_module)
_MENTOR_MODULE_NAME = "salt_gemma4_server_mentor"
_mentor_spec = importlib.util.spec_from_file_location(
    _MENTOR_MODULE_NAME, ROOT / "server" / "gemma4_mentor.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
_mentor_spec = importlib.util.spec_from_file_location(
    _MENTOR_MODULE_NAME, ROOT / "server" / "model" / "gemma4_mentor.py",
)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
if _mentor_spec is None or _mentor_spec.loader is None:
    raise RuntimeError("cannot load Gemma 4 mentor anchor authority")
_mentor_module = importlib.util.module_from_spec(_mentor_spec)
sys.modules[_MENTOR_MODULE_NAME] = _mentor_module
_mentor_spec.loader.exec_module(_mentor_module)

DEFAULT_MODEL_ID = "gemma-4-26b-a4b-it"
GEMMA4_LEGACY_ROW_BYTES = 450_560
GEMMA4_LAYERS = 30
GEMMA4_MODEL_MAX_CONTEXT = 262_144
GEMMA4_MAX_PREFILL_CHUNK = 4_096
GEMMA4_SLIDING_LAYERS = 25
GEMMA4_FULL_LAYERS = 5
GEMMA4_SLIDING_WINDOW = 1_024
GEMMA4_SLIDING_KV_BYTES = 16_384
GEMMA4_FULL_KV_BYTES = 8_192
GEMMA4_FULL_SHARED_BASE_BYTES = 4_096
GEMMA4_KV_LAYOUTS = {"hybrid", "absolute"}
KV_CACHE_MAGIC = b"G4KVC006"
KV_CACHE_VERSION = 6
KV_CACHE_HEADER = struct.Struct("<8sIIIIQ30I32s32s32sII")
KV_COMPATIBILITY_SHA256 = KV_COMPATIBILITY.digest
_KV_LEGACY_COMPATIBILITY = {
    entry["magic"]: bytes.fromhex(entry["compatibility_sha256"])
    for entry in KV_COMPATIBILITY.manifest["legacy_read_abis"]
}
KV_LEGACY_V5_COMPATIBILITY_SHA256 = _KV_LEGACY_COMPATIBILITY["G4KVC005"]
KV_LEGACY_V4_COMPATIBILITY_SHA256 = _KV_LEGACY_COMPATIBILITY["G4KVC004"]
KV_CACHE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
EFFORT_OUTPUT = {
    "none": 128,
    "minimal": 128,
    "low": 128,
    "medium": 128,
    "high": 128,
}
THINKING_EFFORTS = {"low", "medium", "high"}
IMAGE_SOFT_TOKENS = 64
IMAGE_BEGIN = "<|image>"
IMAGE_TOKEN = "<|image|>"
IMAGE_END = "<image|>"
IMAGE_BLOCK = IMAGE_BEGIN + IMAGE_TOKEN * IMAGE_SOFT_TOKENS + IMAGE_END
RESERVED_PROMPT_TOKENS = (
    "<bos>", "<eos>", "<pad>", "<|turn>", "<turn|>",
    "<|channel>", "<channel|>", "<|think|>",
    IMAGE_BEGIN, IMAGE_TOKEN, IMAGE_END,
)
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_PROMPT_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024
HTTP_BODY_TIMEOUT_S = 30.0
RUNTIME_READY = True
_HEX_RUNTIME_TRUE = b"runtime_ready=true"
_HEX_RUNTIME_FALSE = b"runtime_ready=false"
_FIELD_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
_OUTPUT_IDS_RE = re.compile(r"^output_ids=([0-9]+(?:,[0-9]+)*)$", re.MULTILINE)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
RETAINED_NATIVE_DIAGNOSTIC_PREFIXES = (
    b"GEMMA4_NATIVE_READY ",
    b"GEMMA4_GPU_EXPERT_RESIDENCY ",
    b"GEMMA4_GPU_RESIDENCY ",
    b"GEMMA4_GPU_RESIDENCY_DESTROY ",
    b"GEMMA4_PREFILL_GPU_PROGRAM ",
    b"GEMMA4_SERVER_NFQ ",
    b"GEMMA4_SERVER_SAMPLED ",
    b"GEMMA4_SERVER_SAMPLER ",
    b"GEMMA4_SERVER_DPR_PREFILL_V1 ",
    b"GEMMA4_DPR_COVERAGE_WINDOW ",
    b"GEMMA4_DPR_TARGET ",
    b"GEMMA4_TARGET_FRONTIER_OBS ",
    b"GEMMA4_TEXT_VERIFY_ADMISSION ",
    b"SALT_HIP_CELL_TIMING_TOTAL ",
    b"SALT_HIP_GEOMETRY ",
    b"SALT_HIP_KV_PUBLICATION ",
    b"SALT_HIP_PATH_TOTAL ",
    b"SALT_HIP_RESIDENCY_PERMANENT ",
    b"SALT_HIP_RESOURCE_WAIT ",
    b"SALT_HIP_SYNC_WALL ",
    b"gpu: ROCm ready ",
)
RETAINED_NATIVE_DIAGNOSTIC_LIMIT = 64
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1


def gemma4_kv_capacity_bytes(context_tokens: int,
                             layout: str = "hybrid") -> int:
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or \
            not 1 <= context_tokens <= GEMMA4_MODEL_MAX_CONTEXT:
        raise ValueError("Gemma 4 context is outside the model limit")
    if layout not in GEMMA4_KV_LAYOUTS:
        raise ValueError("invalid Gemma KV layout")
    if layout == "absolute":
        return context_tokens * GEMMA4_LEGACY_ROW_BYTES
    return (
        GEMMA4_SLIDING_LAYERS *
        min(context_tokens, GEMMA4_SLIDING_WINDOW) *
        GEMMA4_SLIDING_KV_BYTES +
        GEMMA4_FULL_LAYERS * context_tokens * GEMMA4_FULL_KV_BYTES
    )


def gemma4_kv_state_bytes(position: int) -> int:
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("Gemma KV position must be a non-negative integer")
    sliding_rows = min(position, GEMMA4_SLIDING_WINDOW - 1)
    return (
        GEMMA4_SLIDING_LAYERS * sliding_rows * GEMMA4_SLIDING_KV_BYTES
        + GEMMA4_FULL_LAYERS * position * GEMMA4_FULL_KV_BYTES
    )


def gemma4_kv_portable_bytes(position: int) -> int:
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("Gemma KV position must be a non-negative integer")
    sliding_rows = min(position, GEMMA4_SLIDING_WINDOW - 1)
    return (
        GEMMA4_SLIDING_LAYERS * sliding_rows * GEMMA4_SLIDING_KV_BYTES
        + GEMMA4_FULL_LAYERS * position * GEMMA4_FULL_SHARED_BASE_BYTES
    )


def _runtime_ready_field(value: str) -> bool:
    return value == str(RUNTIME_READY).lower()


class RequestError(ValueError):
    """A caller-visible invalid request."""


class BackendError(RuntimeError):
    """An authenticated runner or server-internal failure."""


def _run_process_group(command: list[str], *, timeout: float,
                       env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    """Run one wrapper session and kill/reap its whole tree on timeout."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr,
        )
    return subprocess.CompletedProcess(
        command, process.returncode, stdout, stderr,
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class NativeStreamAccumulator:
    """Validate cumulative native frames and emit semantic text deltas."""

    _CONTROL_TOKENS = (
        "<|channel>thought", "<channel|>", "<turn|>", "<eos>",
    )

    def __init__(self, marker: str,
                 on_delta: Callable[[str, str], None],
                 request_id: int | None = None):
        self.marker = marker
        self.prefix = (marker + "_STREAM_V1 ").encode("ascii")
        self.on_delta = on_delta
        self.request_id = request_id
        self.frame_count = 0
        self.last_payload = b""
        self.emitted_reasoning = ""
        self.emitted_content = ""

    @classmethod
    def _stable_split(cls, value: str) -> tuple[str, str]:
        opener = "<|channel>thought"
        if opener.startswith(value) and value != opener:
            return "", ""
        hold = 0
        for token in cls._CONTROL_TOKENS:
            for length in range(1, len(token)):
                if value.endswith(token[:length]):
                    hold = max(hold, length)
        if hold:
            value = value[:-hold]
        reasoning, content = split_reasoning(value)
        return reasoning or "", content

    def feed_line(self, line: bytes) -> None:
        if not line.startswith(self.prefix):
            return
        request_field = (
            f"request={self.request_id} ".encode("ascii")
            if self.request_id is not None else b""
        )
        match = re.fullmatch(
            re.escape(self.prefix + request_field) +
            rb"seq=([1-9][0-9]*) bytes=([0-9]+) hex=(-|[0-9a-f]*)\n",
            line,
        )
        if match is None:
            raise BackendError("malformed native stream frame")
        sequence = int(match.group(1))
        byte_count = int(match.group(2))
        if sequence != self.frame_count + 1 or byte_count > 64 * 300 + 1024:
            raise BackendError("native stream sequence or size drift")
        encoded = match.group(3)
        if byte_count == 0:
            if encoded != b"-":
                raise BackendError("native empty stream frame drift")
            payload = b""
        else:
            if len(encoded) != byte_count * 2:
                raise BackendError("native stream byte count drift")
            try:
                payload = bytes.fromhex(encoded.decode("ascii"))
            except ValueError as exc:
                raise BackendError("native stream hex drift") from exc
        if not payload.startswith(self.last_payload):
            raise BackendError("native stream payload is not cumulative")
        self.last_payload = payload
        self.frame_count = sequence
        try:
            value = payload.decode("utf-8")
        except UnicodeDecodeError:
            return
        reasoning, content = self._stable_split(value)
        self._emit_prefix("reasoning_content", reasoning)
        self._emit_prefix("content", content)

    def _emit_prefix(self, kind: str, value: str) -> None:
        attribute = (
            "emitted_reasoning" if kind == "reasoning_content"
            else "emitted_content"
        )
        emitted = getattr(self, attribute)
        if not value.startswith(emitted):
            raise BackendError(f"native {kind} stream prefix drift")
        delta = value[len(emitted):]
        if delta:
            self.on_delta(kind, delta)
            setattr(self, attribute, value)

    def finish(self, result: "RunnerResult") -> None:
        expected_frames = result.completion_tokens - (
            1 if result.stop_token >= 0 else 0
        )
        if self.frame_count != expected_frames:
            raise BackendError("native stream frame count drift")
        self._emit_prefix("reasoning_content", result.reasoning_content or "")
        self._emit_prefix("content", result.content)


def _exact_fields(line: bytes, marker: str,
                  expected_keys: tuple[str, ...]) -> dict[str, str]:
    try:
        text = line.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BackendError(f"{marker} frame is not ASCII") from exc
    if not text.endswith("\n"):
        raise BackendError(f"unterminated {marker} frame")
    parts = text[:-1].split(" ")
    if not parts or parts[0] != marker or len(parts) != len(expected_keys) + 1:
        raise BackendError(f"malformed {marker} frame")
    fields: dict[str, str] = {}
    actual_keys: list[str] = []
    for item in parts[1:]:
        if item.count("=") != 1:
            raise BackendError(f"malformed {marker} field")
        key, value = item.split("=", 1)
        if not key or not value or key in fields:
            raise BackendError(f"duplicate or empty {marker} field")
        fields[key] = value
        actual_keys.append(key)
    if tuple(actual_keys) != expected_keys:
        raise BackendError(f"unknown, missing, or reordered {marker} fields")
    if "runtime_ready" in fields and not _runtime_ready_field(
            fields["runtime_ready"]):
        raise BackendError(f"{marker} runtime readiness drift")
    return fields


def _canonical_uint(value: str, field: str, *, allow_zero: bool = False) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise BackendError(f"noncanonical native integer field {field}")
    parsed = int(value)
    if parsed < (0 if allow_zero else 1):
        raise BackendError(f"invalid native integer field {field}")
    return parsed


def _canonical_int(value: str, field: str, *, minimum: int) -> int:
    if re.fullmatch(r"-?[0-9]+", value) is None or value in ("-0", "+0") or \
            (value.startswith("0") and len(value) > 1) or \
            (value.startswith("-0") and len(value) > 2):
        raise BackendError(f"noncanonical native integer field {field}")
    parsed = int(value)
    if parsed < minimum:
        raise BackendError(f"invalid native integer field {field}")
    return parsed


IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _request_sha256(body: dict, endpoint: str) -> str:
    semantic = dict(body)
    semantic.pop("stream", None)
    semantic.pop("stream_options", None)
    try:
        encoded = json.dumps(
            {"endpoint": endpoint, "body": semantic},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RequestError("request cannot be canonically fingerprinted") from exc
    return hashlib.sha256(b"salt-gemma4-request-v1\0" + encoded).hexdigest()


def _client_request_sha256(request_key: str | None, *, session_epoch: int,
                           turn_id: int, request_sha256: str) -> str:
    if request_key is not None:
        if IDEMPOTENCY_KEY_RE.fullmatch(request_key) is None:
            raise RequestError(
                "Idempotency-Key must be 1..128 ASCII token characters"
            )
        material = b"explicit\0" + request_key.encode("ascii")
    else:
        material = (
            f"automatic\0{session_epoch}\0{turn_id}\0{request_sha256}"
        ).encode("ascii")
    return hashlib.sha256(
        b"salt-gemma4-idempotency-v1\0" + material,
    ).hexdigest()


class NativeV2Accumulator:
    """Validate V2 cumulative token frames and emit semantic deltas."""

    def __init__(self, on_delta: Callable[[str, str], None]):
        self.on_delta = on_delta
        self.frame_count = 0
        self.last_payload = b""
        self.output_ids: list[int] = []
        self.emitted_reasoning = ""
        self.emitted_content = ""

    def feed(self, sequence: int, token_id: int, stop: int,
             payload: bytes) -> None:
        if sequence != self.frame_count + 1 or stop not in (0, 1):
            raise BackendError("native V2 token sequence drift")
        if len(payload) > 64 * 300 + 1024 or \
                not payload.startswith(self.last_payload):
            raise BackendError("native V2 token payload drift")
        self.frame_count = sequence
        self.output_ids.append(token_id)
        self.last_payload = payload
        try:
            value = payload.decode("utf-8")
        except UnicodeDecodeError:
            return
        reasoning, content = NativeStreamAccumulator._stable_split(value)
        self._emit_prefix("reasoning_content", reasoning)
        self._emit_prefix("content", content)

    def _emit_prefix(self, kind: str, value: str) -> None:
        attribute = (
            "emitted_reasoning" if kind == "reasoning_content"
            else "emitted_content"
        )
        emitted = getattr(self, attribute)
        if not value.startswith(emitted):
            raise BackendError(f"native V2 {kind} prefix drift")
        delta = value[len(emitted):]
        if delta:
            self.on_delta(kind, delta)
            setattr(self, attribute, value)

    def finish(self, result: "RunnerResult") -> None:
        if self.frame_count != result.completion_tokens or \
                tuple(self.output_ids) != result.output_ids:
            raise BackendError("native V2 token/frame count drift")
        self._emit_prefix("reasoning_content", result.reasoning_content or "")
        self._emit_prefix("content", result.content)


class PersistentGemmaEngine:
    """One authenticated V2 native engine and live-KV session per launch."""

    READY_KEYS = (
        "protocol", "session_epoch", "turn_id", "history_turns",
        "model_context_limit", "context", "output_limit",
        "response_limit_bytes", "journal_capacity", "prefill_chunk_tokens",
        "position", "shared_position", "kv_capacity_bytes",
        "kv_sliding_window", "kv_sliding_layers", "kv_full_layers",
        "expert_budget_bytes",
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        "expert_cache_mode",
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        "compatibility_sha256", "build_identity_sha256", "state_sha256", "mindset_mode",
        "mindset_end", "mindset_sha256", "facts_sha256", "facts_rows",
        "runtime_ready",
    )
    START_KEYS = (
        "request", "session_epoch", "turn_id", "position",
        "prompt_tokens", "image_tokens", "output_limit", "runtime_ready",
    )
    TOKEN_KEYS = (
        "request", "session_epoch", "turn_id", "seq", "token_id",
        "stop", "bytes",
    )
    COMMIT_KEYS = (
        "request", "original_request", "session_epoch", "turn_id",
        "history_turns", "finish_reason",
        "prompt_tokens", "prompt_ids", "image_tokens", "output_limit",
        "output_steps", "stop_token",
        "synthetic_close", "position_before", "position_after",
        "session_continuable", "response_bytes", "output_ids",
        "state_sha256", "mindset_mode", "mindset_end", "mindset_sha256",
        "facts_sha256", "facts_rows", "build_identity_sha256",
        "client_request_sha256", "request_sha256", "replayed", "runtime_ready",
    )
    REPLAY_KEYS = COMMIT_KEYS
    REJECT_KEYS = ("request", "mutated", "code", "bytes", "runtime_ready")
    FATAL_KEYS = ("request", "mutated", "code", "bytes", "runtime_ready")
    CLEAR_KEYS = (
        "request", "session_epoch", "turn_id", "history_turns",
        "position", "mindset_mode",
        "mindset_end", "facts_rows", "journal_entries", "state_sha256",
        "mindset_sha256", "facts_sha256", "build_identity_sha256", "runtime_ready",
    )
    BYE_KEYS = (
        "session_epoch", "turn_id", "history_turns", "position", "state_sha256",
        "mindset_mode", "mindset_end", "mindset_sha256", "facts_sha256", "facts_rows",
        "build_identity_sha256",
        "runtime_ready",
    )
    EXPORT_KEYS = (
        "request", "position", "state_sha256", "facts_sha256",
        "build_identity_sha256", "runtime_ready",
    )
    IMPORT_KEYS = (
        "request", "session_epoch", "turn_id", "history_turns", "position",
        "mindset_end", "state_sha256", "mindset_sha256", "facts_sha256",
        "facts_rows", "build_identity_sha256", "runtime_ready",
    )

    DPR_RESULT_KEYS = (
        "request", "mode", "edge_id", "kind", "source_position", "horizon",
        "accepted", "produced", "committed", "pending_final", "switched",
        "runtime_ready",
    )
    DPR_MENTOR_KEYS = (
        "request", "mode", "source_position", "horizon",
        "source_node_sha256", "family_sha256", "mindset_sha256",
        "binding_sha256", "edge_id", "expected_saved_numerator",
        "expected_saved_denominator", "runtime_ready",
    )
    DPR_ATTENTION_KEYS = (
        "request", "mode", "cell", "source_position", "plan_sha256",
        "dpr_sha256", "mindset_sha256", "lookup_ns", "prepare_ns",
        "requested", "prepared", "resident_hits", "fetch_jobs",
        "runtime_ready",
    )
    DPR_ATTENTION_READY_KEYS = (
        "request", "source_position", "exact_routes", "ready_matches",
        "unused", "runtime_ready",
    )
    DPR_PREFILL_KEYS = (
        "request", "mode", "hits", "cached_tokens", "prompt_tokens",
        "source_node_sha256", "qa_key_sha256", "runtime_ready",
    )
    DPR_MISS_KEYS = (
        "request", "mode", "source_node_sha256", "source_position",
        "rejected_edges", "runtime_ready",
    )
    DPR_FALLBACK_KEYS = (
        "request", "mode", "edge_id", "source_position", "code",
        "mutated", "runtime_ready",
    )
    DPR_STATS_KEYS = (
        "request", "mode", "edge_id", "retained", "lookups", "hits",
        "rounds", "fallbacks", "accepted_total", "committed_total",
        "lookup_ns", "recover_ns", "verify_ns", "persist_failures",
        "persisted", "runtime_ready",
    )
    DPR_WATERFALL_KEYS = (
        "request", "mode", "prefill_node_ns", "prefill_lookup_ns",
        "prefill_restore_ns", "prefill_ordinary_ns", "decode_node_ns",
        "decode_lookup_ns", "decode_parent_restore_ns", "decode_verify_ns",
        "decode_ordinary_ns", "prefill_hits", "prefill_cached_tokens",
        "decode_cycles", "parent_hits", "nomogram_hits", "decode_misses",
        "proposed_tokens", "accepted_tokens", "rejection_count",
        "runtime_ready",
    )
    MEMORY_KEYS = (
        "request", "phase", "application_bytes", "application_os_peak_bytes",
        "resident_bytes", "resident_os_peak_bytes", "internal_bytes",
        "compressed_bytes", "external_bytes", "shared_bytes",
        "page_table_bytes", "swap_bytes", "approximate", "runtime_ready",
    )

    def __init__(self, command: list[str], *, timeout: float,
                 env: dict[str, str], compatibility_sha256: str,
                 build_identity_sha256: str,
                 expected_context: int = 512,
                 expected_output_limit: int = 128,
                 expected_prefill_chunk: int = 512,
                 expected_kv_capacity_bytes: int = 230_686_720,
                 dpr_mode: str = "off",
                 initial_input: bytes | None = None,
                 pass_fds: tuple[int, ...] = ()):
        self.command = command
        self.timeout = timeout
        self.expected_compatibility = compatibility_sha256
        self.build_identity_sha256 = build_identity_sha256
        self.expected_context = expected_context
        self.expected_output_limit = expected_output_limit
        self.expected_prefill_chunk = expected_prefill_chunk
        self.expected_kv_capacity_bytes = expected_kv_capacity_bytes
        self.output_limit = 0
        self.response_limit_bytes = 0
        if dpr_mode not in ("off", "persist", "dynamic"):
            raise BackendError("invalid native DPR telemetry mode")
        self.dpr_mode = dpr_mode
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        self.sampler_abi = 1
        self.temperature_bits = "00000000"
        self.sampler_seed = 0
        self.sampler_top_k = 1
        self.stderr_mode = env.get("SALT_SERVER_STDERR", "summary")
        if self.stderr_mode not in ("summary", "diagnostic", "waterfall"):
            raise BackendError("invalid SALT_SERVER_STDERR mode")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        self.process_proof_state = "--proof-state" in command
        self.last_dpr_result: dict | None = None
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, start_new_session=True,
            pass_fds=pass_fds,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        self.stderr = self.process.stderr
        if initial_input is not None:
            try:
                self.stdin.write(initial_input)
                self.stdin.flush()
            except BaseException:
                _kill_process_group(self.process)
                raise
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.ready = threading.Event()
        self.logs: deque[bytes] = deque(maxlen=256)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        self.diagnostic_logs: deque[tuple[int, bytes]] = deque(
            maxlen=RETAINED_NATIVE_DIAGNOSTIC_LIMIT,
        )
        self.diagnostic_count = 0
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        self.request_id = 0
        self.session_epoch = 0
        self.turn_id = 0
        self.history_turns = 0
        self.position = 0
        self.shared_position = 0
        self.state_sha256 = ""
        self.mindset_mode = ""
        self.mindset_end = 0
        self.mindset_sha256 = ""
        self.facts_sha256 = ""
        self.facts_rows = 0
        self.phase = "STARTING"
        self.active_request: int | None = None
        self.last_committed_request_id: str | None = None
        self.closed = False
        self.startup_error: BaseException | None = None
        self.readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self.readers:
            reader.start()
        if not self.ready.wait(timeout):
            self.stop(force=True)
            raise BackendError("persistent Gemma V2 startup timed out")
        if self.startup_error is not None:
            error = self.startup_error
            self.stop(force=True)
            raise BackendError(f"persistent Gemma V2 startup failed: {error}")
        if self.process.poll() is not None:
            tail = b"".join(self.logs)[-2048:].decode("utf-8", errors="replace")
            raise BackendError(
                f"persistent Gemma V2 exited during startup: {tail}"
            )

    @staticmethod
    def _read_exact(stream, byte_count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = byte_count
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise BackendError("truncated persistent native V2 payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_payload(self, fields: dict[str, str]) -> bytes:
        byte_count = _canonical_uint(fields["bytes"], "bytes", allow_zero=True)
        payload_limit = self.response_limit_bytes or MAX_PROMPT_BYTES
        if byte_count > payload_limit:
            raise BackendError("persistent native V2 payload exceeds limit")
        payload = self._read_exact(self.stdout, byte_count)
        if self._read_exact(self.stdout, 1) != b"\n":
            raise BackendError("persistent native V2 payload delimiter drift")
        return payload

    def _read_stdout(self) -> None:
        try:
            while True:
                line = self.stdout.readline()
                if not line:
                    break
                marker = line.split(b" ", 1)[0]
                if marker == b"GEMMA4_SERVER_READY_V3":
                    fields = _exact_fields(line, marker.decode(), self.READY_KEYS)
                    if self.ready.is_set():
                        raise BackendError("duplicate persistent V2 READY")
                    if (_canonical_uint(fields["protocol"], "protocol") != 3 or
                            _canonical_uint(fields["model_context_limit"],
                                            "model_context_limit") !=
                            GEMMA4_MODEL_MAX_CONTEXT or
                            _canonical_uint(fields["context"], "context") !=
                            self.expected_context or
                            _canonical_uint(fields["output_limit"], "output_limit") !=
                            self.expected_output_limit or
                            _canonical_uint(fields["response_limit_bytes"],
                                            "response_limit_bytes") !=
                            self.expected_output_limit * 64 + 1024 or
                            _canonical_uint(fields["journal_capacity"],
                                            "journal_capacity") != 256 or
                            _canonical_uint(fields["prefill_chunk_tokens"],
                                            "prefill_chunk_tokens") !=
                            self.expected_prefill_chunk or
                            _canonical_uint(fields["kv_capacity_bytes"],
                                            "kv_capacity_bytes") !=
                            self.expected_kv_capacity_bytes or
                            _canonical_uint(fields["kv_sliding_window"],
                                            "kv_sliding_window") !=
                            GEMMA4_SLIDING_WINDOW or
                            _canonical_uint(fields["kv_sliding_layers"],
                                            "kv_sliding_layers") !=
                            GEMMA4_SLIDING_LAYERS or
                            _canonical_uint(fields["kv_full_layers"],
                                            "kv_full_layers") !=
                            GEMMA4_FULL_LAYERS or
                            _canonical_uint(fields["expert_budget_bytes"],
                                            "expert_budget_bytes") !=
                                            GEMMA_EXPERT_BUDGET_BYTES or
                            fields["compatibility_sha256"] !=
                            self.expected_compatibility or
                            fields["build_identity_sha256"] !=
                            self.build_identity_sha256 or
                            re.fullmatch(r"[0-9a-f]{64}", fields["state_sha256"]) is None or
                            fields["mindset_mode"] != "prefix" or
                            re.fullmatch(r"[0-9a-f]{64}",
                                         fields["mindset_sha256"]) is None or
                            re.fullmatch(r"[0-9a-f]{64}",
                                         fields["facts_sha256"]) is None or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 READY drift")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
                    if fields["expert_cache_mode"] not in {
                            "arena", "flat-mmap-zero-copy", "mixture",
                            "bounded-mmap-zero-copy",
                    }:
                        raise BackendError("persistent V2 cache-mode drift")
                    self.expert_cache_mode = fields["expert_cache_mode"]
                    print(
                        "GEMMA4_NATIVE_READY "
                        f"expert_cache_mode={self.expert_cache_mode}",
                        file=sys.stderr, flush=True,
                    )
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
                    self.session_epoch = _canonical_uint(
                        fields["session_epoch"], "session_epoch")
                    self.output_limit = _canonical_uint(
                        fields["output_limit"], "output_limit")
                    self.response_limit_bytes = _canonical_uint(
                        fields["response_limit_bytes"], "response_limit_bytes")
                    self.turn_id = _canonical_uint(
                        fields["turn_id"], "turn_id", allow_zero=True)
                    self.history_turns = _canonical_uint(
                        fields["history_turns"], "history_turns",
                        allow_zero=True,
                    )
                    self.position = _canonical_uint(
                        fields["position"], "position", allow_zero=True)
                    self.shared_position = _canonical_uint(
                        fields["shared_position"], "shared_position", allow_zero=True,
                    )
                    if self.shared_position > self.position:
                        raise BackendError("persistent V2 shared-position drift")
                    self.state_sha256 = fields["state_sha256"]
                    self.mindset_mode = fields["mindset_mode"]
                    self.mindset_end = _canonical_uint(
                        fields["mindset_end"], "mindset_end", allow_zero=True,
                    )
                    self.mindset_sha256 = fields["mindset_sha256"]
                    self.facts_sha256 = fields["facts_sha256"]
                    self.facts_rows = _canonical_uint(
                        fields["facts_rows"], "facts_rows", allow_zero=True,
                    )
                    if (self.history_turns > self.turn_id or
                            self.facts_rows != self.position - self.mindset_end or
                            (self.position == self.mindset_end and
                             self.state_sha256 != self.mindset_sha256)):
                        raise BackendError("persistent V2 READY state-boundary drift")
                    self.phase = "READY"
                    self.ready.set()
                elif marker == b"GEMMA4_SERVER_START_V3":
                    self.events.put(("start", _exact_fields(
                        line, marker.decode(), self.START_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_TOKEN_V2":
                    fields = _exact_fields(line, marker.decode(), self.TOKEN_KEYS)
                    self.events.put(("token", (fields, self._read_payload(fields))))
                elif marker == b"GEMMA4_SERVER_COMMIT_V3":
                    fields = _exact_fields(line, marker.decode(), self.COMMIT_KEYS)
                    commit_fields = dict(fields)
                    commit_fields["bytes"] = fields["response_bytes"]
                    self.events.put(("commit", (
                        fields, self._read_payload(commit_fields),
                    )))
                elif marker == b"GEMMA4_SERVER_REPLAY_V3":
                    fields = _exact_fields(line, marker.decode(), self.REPLAY_KEYS)
                    replay_fields = dict(fields)
                    replay_fields["bytes"] = fields["response_bytes"]
                    self.events.put(("replay", (
                        fields, self._read_payload(replay_fields),
                    )))
                elif marker == b"GEMMA4_SERVER_CLEAR_RESULT_V2":
                    self.events.put(("clear", _exact_fields(
                        line, marker.decode(), self.CLEAR_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_EXPORT_RESULT_V2":
                    self.events.put(("export", _exact_fields(
                        line, marker.decode(), self.EXPORT_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_IMPORT_RESULT_V1":
                    self.events.put(("import", _exact_fields(
                        line, marker.decode(), self.IMPORT_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_MISS_V1":
                    self.events.put(("dpr_miss", _exact_fields(
                        line, marker.decode(), self.DPR_MISS_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_RESULT_V1":
                    self.events.put(("dpr_result", _exact_fields(
                        line, marker.decode(), self.DPR_RESULT_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_MENTOR_V1":
                    self.events.put(("dpr_mentor", _exact_fields(
                        line, marker.decode(), self.DPR_MENTOR_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_ATTENTION_V1":
                    self.events.put(("dpr_attention", _exact_fields(
                        line, marker.decode(), self.DPR_ATTENTION_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_ATTENTION_READY_V1":
                    self.events.put(("dpr_attention_ready", _exact_fields(
                        line, marker.decode(), self.DPR_ATTENTION_READY_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_PREFILL_V1":
                    self.events.put(("dpr_prefill", _exact_fields(
                        line, marker.decode(), self.DPR_PREFILL_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_FALLBACK_V1":
                    self.events.put(("dpr_fallback", _exact_fields(
                        line, marker.decode(), self.DPR_FALLBACK_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_STATS_V1":
                    self.events.put(("dpr_stats", _exact_fields(
                        line, marker.decode(), self.DPR_STATS_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_DPR_WATERFALL_V1":
                    self.events.put(("dpr_waterfall", _exact_fields(
                        line, marker.decode(), self.DPR_WATERFALL_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_MEMORY_V1":
                    self.events.put(("memory", _exact_fields(
                        line, marker.decode(), self.MEMORY_KEYS,
                    )))
                elif marker == b"GEMMA4_SERVER_REJECT_V2":
                    fields = _exact_fields(line, marker.decode(), self.REJECT_KEYS)
                    self.events.put(("reject", (fields, self._read_payload(fields))))
                elif marker == b"GEMMA4_SERVER_FATAL_V2":
                    fields = _exact_fields(line, marker.decode(), self.FATAL_KEYS)
                    self.events.put(("fatal_frame", (
                        fields, self._read_payload(fields),
                    )))
                elif marker == b"GEMMA4_SERVER_BYE_V2":
                    self.events.put(("bye", _exact_fields(
                        line, marker.decode(), self.BYE_KEYS,
                    )))
                else:
                    raise BackendError("unknown persistent native V2 frame")
        except BaseException as exc:
            self.startup_error = exc
            self.events.put(("fatal", exc))
            self.ready.set()
        finally:
            self.events.put(("eof", self.process.poll()))
            self.ready.set()

    def _read_stderr(self) -> None:
        try:
            while True:
                line = self.stderr.readline()
                if not line:
                    break
                self.logs.append(line)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
                retained_diagnostic = line.startswith(
                    RETAINED_NATIVE_DIAGNOSTIC_PREFIXES
                )
                # Native stderr is the live operational stream. Forward every
                # line only in explicit waterfall mode. Diagnostic mode emits
                # the bounded operational subset; summary mode leaves raw
                # native lines silent and relies on the always-on request
                # summary. Retention remains independent of printing.
                if (self.stderr_mode == "waterfall" or
                        (self.stderr_mode == "diagnostic" and retained_diagnostic)):
                    sys.stderr.write(line.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
                if retained_diagnostic:
                    self.diagnostic_count += 1
                    self.diagnostic_logs.append((self.diagnostic_count, line))
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
                readiness = re.findall(rb"(?:^|\s)runtime_ready=([^\s]+)", line)
                if any(not _runtime_ready_field(
                        value.decode("ascii", errors="replace"))
                       for value in readiness):
                    raise BackendError("persistent native diagnostic readiness drift")
        except BaseException as exc:
            self.events.put(("fatal", exc))
        finally:
            self.events.put(("stderr_eof", self.process.poll()))

    @staticmethod
    def _identity_digest(value: str, field: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RequestError(f"invalid {field}")
        return value

    def _write_command(self, frame: bytes) -> None:
        try:
            self.stdin.write(frame)
            self.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.stop(force=True)
            raise BackendError("persistent Gemma V2 command pipe failed") from exc

    @staticmethod
    def _emit_replay_deltas(
            result: RunnerResult, on_start: Callable[[], None],
            on_delta: Callable[[str, str], None]) -> None:
        on_start()
        if result.reasoning_content:
            on_delta("reasoning_content", result.reasoning_content)
        if result.content:
            on_delta("content", result.content)

    def request(self, prompt: str, image_path: Path | None, *,
                expected_epoch: int, expected_turn: int,
                expected_history_turns: int, expected_position: int,
                client_request_sha256: str, request_sha256: str,
                output_limit: int = 128,
                on_start: Callable[[], None],
                on_delta: Callable[[str, str], None],
                on_token: Callable[[int, int, bool], None] | None = None,
                proof_state: bool = False,
                ) -> RunnerResult:
        if self.closed or self.process.poll() is not None or self.phase != "READY":
            raise BackendError("persistent Gemma V2 engine is not ready")
        if not isinstance(output_limit, int) or isinstance(output_limit, bool) or \
                output_limit < 1 or output_limit > self.output_limit:
            raise RequestError("request output limit exceeds native server capacity")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        if self.sampler_abi not in (1, 2) or \
                re.fullmatch(r"[0-9a-f]{8}", self.temperature_bits) is None or \
                not isinstance(self.sampler_seed, int) or \
                isinstance(self.sampler_seed, bool) or \
                self.sampler_seed < 0 or self.sampler_seed > (1 << 64) - 1 or \
                not isinstance(self.sampler_top_k, int) or \
                isinstance(self.sampler_top_k, bool) or \
                self.sampler_top_k < 1 or self.sampler_top_k > 256 or \
                (self.sampler_abi == 1 and
                 (self.temperature_bits != "00000000" or self.sampler_seed != 0 or
                  self.sampler_top_k != 1)) or \
                (self.sampler_abi == 2 and self.temperature_bits == "00000000"):
            raise RequestError("invalid native sampler controls")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        client_request_sha256 = self._identity_digest(
            client_request_sha256, "client request digest",
        )
        request_sha256 = self._identity_digest(
            request_sha256, "request payload digest",
        )
        prompt_bytes = prompt.encode("utf-8")
        image_bytes = b"" if image_path is None else os.fsencode(image_path)
        if not prompt_bytes or len(prompt_bytes) > MAX_PROMPT_BYTES or \
                len(image_bytes) > 4096:
            raise RequestError("persistent V2 request frame exceeds private limits")
        self.request_id += 1
        request_id = self.request_id
        self.active_request = request_id
        self.phase = "PREPARING"
        header = (
            f"SALT_GEMMA4_TURN_V3 request={request_id} "
            f"session_epoch={expected_epoch} turn_id={expected_turn} "
            f"history_turns={expected_history_turns} "
            f"position={expected_position} prompt_bytes={len(prompt_bytes)} "
            f"image_path_bytes={len(image_bytes)} "
            f"output_limit={output_limit} proof_state={1 if proof_state else 0} "
            f"client_request_sha256={client_request_sha256} "
            f"request_sha256={request_sha256}\n"
        ).encode("ascii")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        header = (
            f"SALT_GEMMA4_TURN_V3 request={request_id} "
            f"session_epoch={expected_epoch} turn_id={expected_turn} "
            f"history_turns={expected_history_turns} "
            f"position={expected_position} prompt_bytes={len(prompt_bytes)} "
            f"image_path_bytes={len(image_bytes)} "
            f"output_limit={output_limit} proof_state={1 if proof_state else 0} "
            f"sampler_abi={self.sampler_abi} "
            f"temperature_bits={self.temperature_bits} "
            f"seed={self.sampler_seed} top_k={self.sampler_top_k} "
            f"client_request_sha256={client_request_sha256} "
            f"request_sha256={request_sha256}\n"
        ).encode("ascii")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        self._write_command(header + prompt_bytes + image_bytes + b"\n")
        accumulator = NativeV2Accumulator(on_delta)
        started = False
        self.last_dpr_result = None
        memory_snapshots: list[dict[str, str]] = []
        dpr_prefill_result: dict[str, str] | None = None
        dpr_waterfall_result: dict[str, str] | None = None
        pending_mentor: dict[str, str] | None = None
        pending_attention: dict[str, str] | None = None
        def add_dpr_cycle(cycle: dict[str, str]) -> None:
            if self.last_dpr_result is None:
                self.last_dpr_result = cycle
                return
            cycles = self.last_dpr_result.get("cycles")
            if not isinstance(cycles, list):
                cycles = [dict(self.last_dpr_result)]
                self.last_dpr_result["cycles"] = cycles
            cycles.append(cycle)

        def current_dpr_cycle() -> dict | None:
            if self.last_dpr_result is None:
                return None
            cycles = self.last_dpr_result.get("cycles")
            if isinstance(cycles, list) and cycles:
                return cast(dict, cycles[-1])
            return self.last_dpr_result

        deadline = time.monotonic() + self.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                kind, value = self.events.get(timeout=remaining)
                if kind == "start":
                    fields = cast(dict[str, str], value)
                    if (started or
                            _canonical_uint(fields["request"], "request") != request_id or
                            _canonical_uint(fields["session_epoch"], "session_epoch") !=
                            expected_epoch or
                            _canonical_uint(fields["turn_id"], "turn_id", allow_zero=True) !=
                            expected_turn or
                            _canonical_uint(fields["position"], "position", allow_zero=True) !=
                            expected_position or
                            _canonical_uint(fields["output_limit"], "output_limit") !=
                            output_limit or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 START drift")
                    started = True
                    self.phase = "STARTED"
                    on_start()
                elif kind == "dpr_prefill":
                    fields = cast(dict[str, str], value)
                    if (not started or dpr_prefill_result is not None or
                            self.dpr_mode == "off" or
                            _canonical_uint(fields["request"], "request") != request_id or
                            fields["mode"] != self.dpr_mode or
                            re.fullmatch(
                                r"[0-9a-f]{64}", fields["source_node_sha256"],
                            ) is None or
                            re.fullmatch(
                                r"[0-9a-f]{64}", fields["qa_key_sha256"],
                            ) is None or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR prefill drift")
                    hits = _canonical_uint(fields["hits"], "hits", allow_zero=True)
                    cached = _canonical_uint(
                        fields["cached_tokens"], "cached_tokens", allow_zero=True,
                    )
                    prompt_total = _canonical_uint(
                        fields["prompt_tokens"], "prompt_tokens",
                    )
                    if cached >= prompt_total or (hits == 0) != (cached == 0):
                        raise BackendError("persistent V2 DPR prefill accounting drift")
                    dpr_prefill_result = dict(fields)
                elif kind == "dpr_miss":
                    fields = cast(dict[str, str], value)
                    if (not started or pending_mentor is not None or
                            self.dpr_mode == "off" or
                            _canonical_uint(fields["request"], "request") != request_id or
                            fields["mode"] != self.dpr_mode or
                            re.fullmatch(
                                r"[0-9a-f]{64}",
                                fields["source_node_sha256"],
                            ) is None or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR miss drift")
                    _canonical_uint(fields["source_position"], "source_position")
                    _canonical_uint(
                        fields["rejected_edges"], "rejected_edges",
                        allow_zero=True,
                    )
                    cycle = dict(fields)
                    cycle["outcome"] = "miss"
                    add_dpr_cycle(cycle)
                elif kind == "dpr_mentor":
                    fields = cast(dict[str, str], value)
                    if (not started or pending_mentor is not None or
                            self.dpr_mode == "off" or
                            _canonical_uint(fields["request"], "request") !=
                            request_id or fields["mode"] != self.dpr_mode or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR Mentor drift")
                    _canonical_uint(fields["source_position"], "source_position")
                    _canonical_uint(fields["horizon"], "horizon")
                    _canonical_uint(
                        fields["expected_saved_numerator"],
                        "expected_saved_numerator",
                    )
                    _canonical_uint(
                        fields["expected_saved_denominator"],
                        "expected_saved_denominator",
                    )
                    for key in (
                            "source_node_sha256", "family_sha256",
                            "mindset_sha256", "binding_sha256", "edge_id"):
                        if re.fullmatch(r"[0-9a-f]{64}", fields[key]) is None:
                            raise BackendError(
                                "persistent V2 DPR Mentor identity drift")
                    pending_mentor = dict(fields)
                elif kind == "dpr_attention":
                    fields = cast(dict[str, str], value)
                    if (not started or pending_attention is not None or
                            self.dpr_mode == "off" or
                            _canonical_uint(fields["request"], "request") !=
                            request_id or fields["mode"] != self.dpr_mode or
                            fields["cell"] not in ("NM_ONLY", "ND_NM") or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR attention drift")
                    _canonical_uint(fields["source_position"], "source_position")
                    for name in (
                            "lookup_ns", "prepare_ns", "requested", "prepared",
                            "resident_hits", "fetch_jobs"):
                        _canonical_uint(fields[name], name, allow_zero=True)
                    for name in ("plan_sha256", "dpr_sha256", "mindset_sha256"):
                        if re.fullmatch(r"[0-9a-f]{64}", fields[name]) is None:
                            raise BackendError(
                                "persistent V2 DPR attention identity drift")
                    if int(fields["prepared"]) > int(fields["requested"]) or \
                            int(fields["resident_hits"]) > int(fields["prepared"]):
                        raise BackendError(
                            "persistent V2 DPR attention accounting drift")
                    if fields["cell"] == "NM_ONLY":
                        attention_cycle = current_dpr_cycle()
                        attention_outcome = (
                            None if attention_cycle is None else
                            attention_cycle.get("outcome")
                        )
                        if (attention_cycle is None or
                                attention_outcome not in ("miss", "fallback") or
                                (attention_outcome == "fallback" and
                                 attention_cycle.get("code") != "memory_reject") or
                                "attention" in attention_cycle):
                            raise BackendError(
                                "persistent V2 NM-only ordering drift")
                        attention_cycle["attention"] = dict(fields)
                    else:
                        pending_attention = dict(fields)
                elif kind == "dpr_attention_ready":
                    fields = cast(dict[str, str], value)
                    cycle = current_dpr_cycle()
                    attention = None if cycle is None else cycle.get("attention")
                    if (not started or self.dpr_mode == "off" or
                            _canonical_uint(fields["request"], "request") !=
                            request_id or not _runtime_ready_field(fields["runtime_ready"]) or
                            not isinstance(attention, dict) or
                            fields["source_position"] !=
                            attention.get("source_position")):
                        raise BackendError(
                            "persistent V2 DPR attention READY drift")
                    exact_routes = _canonical_uint(
                        fields["exact_routes"], "exact_routes", allow_zero=True,
                    )
                    ready_matches = _canonical_uint(
                        fields["ready_matches"], "ready_matches", allow_zero=True,
                    )
                    unused = _canonical_uint(
                        fields["unused"], "unused", allow_zero=True,
                    )
                    if ready_matches + unused != int(attention["prepared"]):
                        raise BackendError(
                            "persistent V2 DPR attention READY accounting drift")
                    attention["exact_routes"] = str(exact_routes)
                    attention["ready_matches"] = str(ready_matches)
                    attention["unused"] = str(unused)
                elif kind in ("dpr_result", "dpr_fallback"):
                    fields = cast(dict[str, str], value)
                    fallback = kind == "dpr_fallback"
                    if (not started or self.dpr_mode == "off" or
                            _canonical_uint(fields["request"], "request") != request_id or
                            fields["mode"] != self.dpr_mode or
                            re.fullmatch(r"[0-9a-f]{64}", fields["edge_id"]) is None or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR telemetry drift")
                    _canonical_uint(fields["source_position"], "source_position")
                    if fallback:
                        if fields["mutated"] != "0" or fields["code"] not in (
                                "reference_reject", "edge_reject", "memory_reject"):
                            raise BackendError("persistent V2 DPR fallback drift")
                    else:
                        accepted = _canonical_uint(
                            fields["accepted"], "accepted", allow_zero=True,
                        )
                        horizon = _canonical_uint(fields["horizon"], "horizon")
                        produced = _canonical_uint(fields["produced"], "produced")
                        committed = _canonical_uint(
                            fields["committed"], "committed", allow_zero=True,
                        )
                        pending_final = _canonical_uint(
                            fields["pending_final"], "pending_final",
                            allow_zero=True,
                        )
                        switched = _canonical_uint(
                            fields["switched"], "switched", allow_zero=True,
                        )
                        edge_kind = fields.get("kind")
                        if (accepted > horizon or produced != accepted + 1 or
                                pending_final not in (0, 1) or
                                committed + pending_final != produced or
                                edge_kind not in ("parent", "nomogram") or
                                switched != (1 if (
                                    self.dpr_mode == "dynamic" and
                                    edge_kind == "parent") else 0)):
                            raise BackendError("persistent V2 DPR acceptance drift")
                    cycle = dict(fields)
                    cycle["outcome"] = "fallback" if fallback else "hit"
                    if pending_mentor is not None:
                        if (pending_mentor["mode"] != fields["mode"] or
                                pending_mentor["edge_id"] != fields["edge_id"] or
                                pending_mentor["source_position"] !=
                                    fields["source_position"] or
                                (not fallback and
                                 pending_mentor["horizon"] != fields["horizon"])):
                            raise BackendError(
                                "persistent V2 DPR Mentor/result drift")
                        cycle["mentor"] = pending_mentor
                        pending_mentor = None
                    if pending_attention is not None:
                        if (pending_attention["cell"] != "ND_NM" or
                                pending_attention["mode"] != fields["mode"] or
                                pending_attention["source_position"] !=
                                    fields["source_position"]):
                            raise BackendError(
                                "persistent V2 DPR attention/result drift")
                        cycle["attention"] = pending_attention
                        pending_attention = None
                    add_dpr_cycle(cycle)
                elif kind == "dpr_stats":
                    fields = cast(dict[str, str], value)
                    stats_cycle = current_dpr_cycle()
                    if (not started or stats_cycle is None or
                            stats_cycle.get("outcome") not in
                            ("hit", "fallback") or
                            "stats" in stats_cycle or
                            _canonical_uint(fields["request"], "request") != request_id or
                            fields["mode"] != stats_cycle.get("mode") or
                            fields["edge_id"] != stats_cycle.get("edge_id") or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR stats drift")
                    assert stats_cycle is not None
                    for field in (
                        "retained", "lookups", "hits", "rounds", "fallbacks",
                        "accepted_total", "committed_total", "lookup_ns",
                        "recover_ns", "verify_ns", "persist_failures", "persisted",
                    ):
                        _canonical_uint(fields[field], field, allow_zero=True)
                    if fields["retained"] not in ("0", "1") or \
                            fields["persisted"] not in ("0", "1"):
                        raise BackendError("persistent V2 DPR stats flag drift")
                    stats_cycle["stats"] = dict(fields)
                elif kind == "dpr_waterfall":
                    fields = cast(dict[str, str], value)
                    if (not started or dpr_waterfall_result is not None or
                            _canonical_uint(fields["request"], "request") != request_id or
                            fields["mode"] != self.dpr_mode or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 DPR waterfall drift")
                    for field in self.DPR_WATERFALL_KEYS[2:-1]:
                        _canonical_uint(fields[field], field, allow_zero=True)
                    if int(fields["accepted_tokens"]) > int(fields["proposed_tokens"]):
                        raise BackendError("persistent V2 DPR waterfall accounting drift")
                    dpr_waterfall_result = dict(fields)
                elif kind == "memory":
                    fields = cast(dict[str, str], value)
                    expected_phase = (
                        "after_prefill" if not memory_snapshots else "pre_commit"
                    )
                    if (not started or len(memory_snapshots) >= 2 or
                            _canonical_uint(fields["request"], "request") != request_id or
                            fields["phase"] != expected_phase or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 memory telemetry drift")
                    for field in (
                        "application_bytes", "application_os_peak_bytes",
                        "resident_bytes", "resident_os_peak_bytes",
                        "internal_bytes", "compressed_bytes", "external_bytes",
                        "shared_bytes", "page_table_bytes", "swap_bytes",
                        "approximate",
                    ):
                        _canonical_uint(fields[field], field, allow_zero=True)
                    if fields["approximate"] not in ("0", "1"):
                        raise BackendError("persistent V2 memory approximation drift")
                    memory_snapshots.append(dict(fields))
                elif kind == "token":
                    if not started:
                        raise BackendError("persistent V2 TOKEN before START")
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    if (_canonical_uint(fields["request"], "request") != request_id or
                            _canonical_uint(fields["session_epoch"], "session_epoch") !=
                            expected_epoch or
                            _canonical_uint(fields["turn_id"], "turn_id", allow_zero=True) !=
                            expected_turn):
                        raise BackendError("persistent V2 TOKEN identity drift")
                    seq = _canonical_uint(fields["seq"], "seq")
                    token_id = _canonical_uint(
                        fields["token_id"], "token_id", allow_zero=True,
                    )
                    stop = _canonical_uint(fields["stop"], "stop", allow_zero=True)
                    if on_token is not None:
                        on_token(seq, token_id, bool(stop))
                    accumulator.feed(seq, token_id, stop, payload)
                elif kind == "reject":
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    if _canonical_uint(fields["request"], "request") != request_id:
                        raise BackendError("persistent V2 REJECT request drift")
                    if started or fields["mutated"] != "0":
                        raise BackendError("persistent V2 REJECT after START")
                    self.phase = "READY"
                    self.active_request = None
                    try:
                        raise RequestError(payload.decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise BackendError("persistent V2 REJECT is not UTF-8") from exc
                elif kind == "fatal_frame":
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    self.stop(force=True)
                    message = payload.decode("utf-8", errors="replace")
                    raise BackendError(
                        f"persistent V2 FATAL {fields['code']}: {message}"
                    )
                elif kind == "commit":
                    if (not started or dpr_waterfall_result is None or
                            pending_mentor is not None):
                        raise BackendError("persistent V2 COMMIT before START")
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    result = self._result(
                        request_id, expected_epoch, expected_turn,
                        expected_position, payload, fields,
                        expected_image_tokens=IMAGE_SOFT_TOKENS if image_path else 0,
                        expected_client_request_sha256=client_request_sha256,
                        expected_request_sha256=request_sha256,
                        expected_history_turns=self.history_turns,
                        expected_output_limit=output_limit,
                        maximum_output_limit=self.output_limit,
                        expected_build_identity_sha256=self.build_identity_sha256,
                        expected_mindset_end=self.mindset_end,
                        expected_mindset_sha256=self.mindset_sha256,
                        replayed=False,
                    )
                    proof_active = self.process_proof_state or proof_state
                    zero_digest = "0" * 64
                    if ((result.state_sha256 != zero_digest) != proof_active or
                            (result.facts_sha256 != zero_digest) != proof_active):
                        raise BackendError("persistent V2 request proof-state drift")
                    if dpr_prefill_result is not None:
                        if self.last_dpr_result is None:
                            self.last_dpr_result = {"outcome": "none"}
                        self.last_dpr_result["prefill"] = dpr_prefill_result
                    result = replace(
                        result, dpr_result=self.last_dpr_result,
                        memory_snapshots=tuple(memory_snapshots),
                        dpr_waterfall=dpr_waterfall_result,
                    )
                    accumulator.finish(result)
                    self.session_epoch = result.session_epoch
                    self.turn_id = result.turn_id
                    self.history_turns = result.history_turns
                    self.position = result.kv_saved_tokens
                    self.state_sha256 = result.state_sha256
                    self.facts_sha256 = result.facts_sha256
                    self.facts_rows = result.facts_rows
                    self.last_committed_request_id = result.client_request_sha256
                    self.active_request = None
                    self.phase = "READY"
                    return result
                elif kind == "replay":
                    if started:
                        raise BackendError("persistent V2 REPLAY after START")
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    result = self._result(
                        request_id, expected_epoch, expected_turn,
                        None, payload, fields, expected_image_tokens=None,
                        expected_client_request_sha256=client_request_sha256,
                        expected_request_sha256=request_sha256,
                        expected_history_turns=self.history_turns,
                        expected_output_limit=output_limit,
                        maximum_output_limit=self.output_limit,
                        expected_build_identity_sha256=self.build_identity_sha256,
                        expected_mindset_end=self.mindset_end,
                        expected_mindset_sha256=self.mindset_sha256,
                        replayed=True,
                    )
                    self._emit_replay_deltas(result, on_start, on_delta)
                    self.last_committed_request_id = result.client_request_sha256
                    self.active_request = None
                    self.phase = "READY"
                    return result
                elif kind in ("fatal", "eof", "stderr_eof"):
                    if isinstance(value, BaseException):
                        raise BackendError(str(value)) from value
                    raise BackendError("persistent Gemma V2 engine pipe closed")

        except RequestError:
            raise
        except queue.Empty as exc:
            self.stop(force=True)
            raise BackendError("persistent Gemma V2 request timed out") from exc
        except BaseException:
            self.stop(force=True)
            raise

    def export_kv(self, path: Path) -> dict[str, int | str]:
        if self.closed or self.phase != "READY" or self.active_request is not None:
            raise BackendError("persistent Gemma engine is not ready for export")
        encoded = os.fsencode(path)
        if not encoded or len(encoded) > 4096 or b"\0" in encoded:
            raise BackendError("invalid persistent KV export path")
        self.request_id += 1
        request_id = self.request_id
        frame = (
            f"SALT_GEMMA4_EXPORT_KV_V2 request={request_id} "
            f"path_bytes={len(encoded)}\n"
        ).encode("ascii") + encoded + b"\n"
        self._write_command(frame)
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendError("persistent Gemma KV export timed out")
            kind, value = self.events.get(timeout=remaining)
            if kind == "export":
                fields = cast(dict[str, str], value)
                state_sha256 = fields["state_sha256"]
                facts_sha256 = fields["facts_sha256"]
                zero = "0" * 64
                if (_canonical_uint(fields["request"], "request") != request_id or
                        _canonical_uint(fields["position"], "position") != self.position or
                        re.fullmatch(r"[0-9a-f]{64}", state_sha256) is None or
                        re.fullmatch(r"[0-9a-f]{64}", facts_sha256) is None or
                        state_sha256 == zero or facts_sha256 == zero or
                        (self.state_sha256 not in ("", zero) and
                         state_sha256 != self.state_sha256) or
                        (self.facts_sha256 not in ("", zero) and
                         facts_sha256 != self.facts_sha256) or
                        fields["build_identity_sha256"] != self.build_identity_sha256 or
                        not _runtime_ready_field(fields["runtime_ready"])):
                    raise BackendError("persistent V2 EXPORT_RESULT drift")
                self.state_sha256 = state_sha256
                self.facts_sha256 = facts_sha256
                return {
                    "position": self.position,
                    "state_sha256": state_sha256,
                    "facts_sha256": facts_sha256,
                    "build_identity_sha256": self.build_identity_sha256,
                }
            if kind in ("fatal", "fatal_frame", "eof", "stderr_eof"):
                raise BackendError("persistent Gemma KV export failed")
            raise BackendError(f"unexpected persistent V2 export frame: {kind}")

    def import_session_kv(self, path: Path, *, turn_id: int,
                          history_turns: int, expected_position: int,
                          expected_state_sha256: str,
                          expected_facts_sha256: str) -> dict[str, int | str]:
        if self.closed or self.phase != "READY" or self.active_request is not None:
            raise BackendError("persistent Gemma engine is not ready for import")
        if self.position != self.mindset_end:
            raise BackendError("persistent Gemma session must be clear before import")
        if (not isinstance(turn_id, int) or not isinstance(history_turns, int) or
                turn_id < 0 or history_turns < 0 or history_turns > turn_id or
                not isinstance(expected_position, int) or
                expected_position < self.mindset_end or
                re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256) is None or
                re.fullmatch(r"[0-9a-f]{64}", expected_facts_sha256) is None):
            raise BackendError("invalid persistent session import metadata")
        encoded = os.fsencode(path)
        if not encoded or len(encoded) > 4096 or b"\0" in encoded:
            raise BackendError("invalid persistent KV import path")
        self.request_id += 1
        request_id = self.request_id
        frame = (
            f"SALT_GEMMA4_IMPORT_SESSION_V1 request={request_id} "
            f"expected_epoch={self.session_epoch} expected_position={self.position} "
            f"turn={turn_id} history_turns={history_turns} "
            f"mindset_end={self.mindset_end} path_bytes={len(encoded)}\n"
        ).encode("ascii") + encoded + b"\n"
        self.phase = "IMPORTING"
        self.active_request = request_id
        try:
            self._write_command(frame)
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendError("persistent Gemma session import timed out")
                kind, value = self.events.get(timeout=remaining)
                if kind == "import":
                    fields = cast(dict[str, str], value)
                    actual_position = _canonical_uint(fields["position"], "position")
                    actual_turn = _canonical_uint(
                        fields["turn_id"], "turn_id", allow_zero=True,
                    )
                    actual_history = _canonical_uint(
                        fields["history_turns"], "history_turns", allow_zero=True,
                    )
                    actual_facts_rows = _canonical_uint(
                        fields["facts_rows"], "facts_rows", allow_zero=True,
                    )
                    if (
                        _canonical_uint(fields["request"], "request") != request_id or
                        _canonical_uint(fields["session_epoch"], "session_epoch") !=
                            self.session_epoch or
                        actual_turn != turn_id or actual_history != history_turns or
                        actual_position != expected_position or
                        _canonical_uint(
                            fields["mindset_end"], "mindset_end", allow_zero=True,
                        ) !=
                            self.mindset_end or
                        actual_facts_rows != actual_position - self.mindset_end or
                        fields["state_sha256"] != expected_state_sha256 or
                        fields["facts_sha256"] != expected_facts_sha256 or
                        fields["mindset_sha256"] != self.mindset_sha256 or
                        fields["build_identity_sha256"] != self.build_identity_sha256 or
                        not _runtime_ready_field(fields["runtime_ready"])
                    ):
                        raise BackendError("persistent V1 IMPORT_RESULT drift")
                    self.turn_id = actual_turn
                    self.history_turns = actual_history
                    self.position = actual_position
                    self.state_sha256 = fields["state_sha256"]
                    self.facts_sha256 = fields["facts_sha256"]
                    self.facts_rows = actual_facts_rows
                    self.last_committed_request_id = None
                    self.active_request = None
                    self.phase = "READY"
                    return {
                        "position": actual_position,
                        "turn_id": actual_turn,
                        "history_turns": actual_history,
                        "state_sha256": self.state_sha256,
                        "facts_sha256": self.facts_sha256,
                    }
                if kind == "reject":
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    if _canonical_uint(fields["request"], "request") != request_id:
                        raise BackendError("persistent import reject request drift")
                    self.active_request = None
                    self.phase = "READY"
                    raise RequestError(payload.decode("utf-8", errors="replace"))
                if kind in ("fatal", "fatal_frame", "eof", "stderr_eof"):
                    if isinstance(value, BaseException):
                        raise BackendError(str(value)) from value
                    raise BackendError("persistent Gemma pipe closed during import")
                raise BackendError(f"unexpected persistent import frame: {kind}")
        except RequestError:
            raise
        except BaseException:
            self.stop(force=True)
            raise

    def clear_facts(self) -> dict[str, int | str]:
        if self.closed or self.process.poll() is not None or self.phase != "READY":
            raise BackendError("persistent Gemma V2 engine is not ready")
        self.request_id += 1
        request_id = self.request_id
        self.active_request = request_id
        self.phase = "CLEARING_FACTS"
        frame = (
            f"SALT_GEMMA4_CLEAR_FACTS_V2 request={request_id} "
            f"session_epoch={self.session_epoch} turn_id={self.turn_id} "
            f"history_turns={self.history_turns} "
            f"position={self.position} state_sha256={self.state_sha256}\n"
        ).encode("ascii")
        self._write_command(frame)
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                kind, value = self.events.get(timeout=remaining)
                if kind == "clear":
                    fields = cast(dict[str, str], value)
                    position = _canonical_uint(
                        fields["position"], "position", allow_zero=True,
                    )
                    mindset_end = _canonical_uint(
                        fields["mindset_end"], "mindset_end", allow_zero=True,
                    )
                    facts_rows = _canonical_uint(
                        fields["facts_rows"], "facts_rows", allow_zero=True,
                    )
                    journal_entries = _canonical_uint(
                        fields["journal_entries"], "journal_entries",
                        allow_zero=True,
                    )
                    history_turns = _canonical_uint(
                        fields["history_turns"], "history_turns",
                        allow_zero=True,
                    )
                    if (_canonical_uint(fields["request"], "request") != request_id or
                            _canonical_uint(fields["session_epoch"], "session_epoch") !=
                            self.session_epoch or
                            _canonical_uint(fields["turn_id"], "turn_id", allow_zero=True) !=
                            self.turn_id or
                            history_turns != 0 or
                            fields["mindset_mode"] != "prefix" or
                            mindset_end != self.mindset_end or
                            position != mindset_end or facts_rows != 0 or
                            journal_entries != 0 or
                            fields["mindset_sha256"] != self.mindset_sha256 or
                            fields["state_sha256"] != self.mindset_sha256 or
                            re.fullmatch(r"[0-9a-f]{64}",
                                         fields["facts_sha256"]) is None or
                            fields["build_identity_sha256"] !=
                            self.build_identity_sha256 or
                            not _runtime_ready_field(fields["runtime_ready"])):
                        raise BackendError("persistent V2 CLEAR_RESULT drift")
                    self.position = position
                    self.state_sha256 = fields["state_sha256"]
                    self.facts_sha256 = fields["facts_sha256"]
                    self.history_turns = history_turns
                    self.facts_rows = facts_rows
                    self.last_committed_request_id = None
                    self.active_request = None
                    self.phase = "READY"
                    return {
                        "session_epoch": self.session_epoch,
                        "turn_id": self.turn_id,
                        "history_turns": history_turns,
                        "position": position,
                        "mindset_mode": self.mindset_mode,
                        "mindset_end": mindset_end,
                        "mindset_sha256": self.mindset_sha256,
                        "state_sha256": self.state_sha256,
                        "facts_sha256": self.facts_sha256,
                        "facts_rows": facts_rows,
                        "journal_entries": journal_entries,
                    }
                if kind == "reject":
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    if (_canonical_uint(fields["request"], "request") != request_id or
                            fields["mutated"] != "0"):
                        raise BackendError("persistent V2 clear REJECT drift")
                    self.active_request = None
                    self.phase = "READY"
                    try:
                        raise RequestError(payload.decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise BackendError("persistent V2 REJECT is not UTF-8") from exc
                if kind in ("fatal", "fatal_frame", "eof", "stderr_eof"):
                    raise BackendError("persistent Gemma V2 clear failed")
                raise BackendError(f"unexpected persistent V2 clear frame: {kind}")
        except RequestError:
            raise
        except queue.Empty as exc:
            self.stop(force=True)
            raise BackendError("persistent Gemma V2 clear timed out") from exc
        except BaseException:
            self.stop(force=True)
            raise

    def replay(self, *, client_request_sha256: str, request_sha256: str,
               on_start: Callable[[], None],
               on_delta: Callable[[str, str], None]) -> RunnerResult:
        if self.closed or self.process.poll() is not None or self.phase != "READY":
            raise BackendError("persistent Gemma V2 engine is not ready")
        client_request_sha256 = self._identity_digest(
            client_request_sha256, "client request digest",
        )
        request_sha256 = self._identity_digest(
            request_sha256, "request payload digest",
        )
        self.request_id += 1
        request_id = self.request_id
        self.active_request = request_id
        self.phase = "PREPARING"
        frame = (
            f"SALT_GEMMA4_REPLAY_V2 request={request_id} "
            f"session_epoch={self.session_epoch} "
            f"client_request_sha256={client_request_sha256} "
            f"request_sha256={request_sha256}\n"
        ).encode("ascii")
        self._write_command(frame)
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                kind, value = self.events.get(timeout=remaining)
                if kind == "replay":
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    result = self._result(
                        request_id, self.session_epoch, self.turn_id,
                        None, payload, fields, expected_image_tokens=None,
                        expected_client_request_sha256=client_request_sha256,
                        expected_request_sha256=request_sha256,
                        expected_history_turns=self.history_turns,
                        expected_output_limit=None,
                        maximum_output_limit=self.output_limit,
                        expected_build_identity_sha256=self.build_identity_sha256,
                        expected_mindset_end=self.mindset_end,
                        expected_mindset_sha256=self.mindset_sha256,
                        replayed=True,
                    )
                    self._emit_replay_deltas(result, on_start, on_delta)
                    self.last_committed_request_id = result.client_request_sha256
                    self.active_request = None
                    self.phase = "READY"
                    return result
                if kind == "reject":
                    fields, payload = cast(tuple[dict[str, str], bytes], value)
                    if (_canonical_uint(fields["request"], "request") != request_id or
                            fields["mutated"] != "0"):
                        raise BackendError("persistent V2 replay REJECT drift")
                    self.active_request = None
                    self.phase = "READY"
                    try:
                        raise RequestError(payload.decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise BackendError("persistent V2 REJECT is not UTF-8") from exc
                if kind in ("fatal", "fatal_frame", "eof", "stderr_eof"):
                    raise BackendError("persistent Gemma V2 replay failed")
                raise BackendError(f"unexpected persistent V2 replay frame: {kind}")
        except RequestError:
            raise
        except queue.Empty as exc:
            self.stop(force=True)
            raise BackendError("persistent Gemma V2 replay timed out") from exc
        except BaseException:
            self.stop(force=True)
            raise

    @staticmethod
    def _result(request_id: int, expected_epoch: int, expected_turn: int,
                expected_position: int | None, response: bytes,
                fields: dict[str, str], *,
                expected_image_tokens: int | None,
                expected_client_request_sha256: str,
                expected_request_sha256: str,
                expected_history_turns: int,
                expected_output_limit: int | None,
                maximum_output_limit: int,
                expected_build_identity_sha256: str,
                expected_mindset_end: int,
                expected_mindset_sha256: str,
                replayed: bool) -> RunnerResult:
        if not _runtime_ready_field(fields["runtime_ready"]):
            raise BackendError("persistent V2 result readiness drift")
        if (_canonical_uint(fields["request"], "request") != request_id or
                _canonical_uint(fields["session_epoch"], "session_epoch") !=
                expected_epoch):
            raise BackendError("persistent V2 result request/epoch drift")
        original_request_id = _canonical_uint(
            fields["original_request"], "original_request",
        )
        frame_replayed = _canonical_uint(
            fields["replayed"], "replayed", allow_zero=True,
        )
        if frame_replayed != int(replayed):
            raise BackendError("persistent V2 replay marker drift")
        turn_id = _canonical_uint(fields["turn_id"], "turn_id")
        history_turns = _canonical_uint(
            fields["history_turns"], "history_turns", allow_zero=True,
        )
        if replayed:
            if (original_request_id >= request_id or turn_id > expected_turn or
                    history_turns > expected_history_turns):
                raise BackendError("persistent V2 replay identity drift")
        elif (original_request_id != request_id or
              turn_id != expected_turn + 1 or
              history_turns != expected_history_turns + 1):
            raise BackendError("persistent V2 COMMIT turn drift")
        client_request_sha256 = fields["client_request_sha256"]
        request_sha256 = fields["request_sha256"]
        if (client_request_sha256 != expected_client_request_sha256 or
                request_sha256 != expected_request_sha256 or
                re.fullmatch(r"[0-9a-f]{64}", client_request_sha256) is None or
                re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None):
            raise BackendError("persistent V2 request identity drift")
        prompt_tokens = _canonical_uint(fields["prompt_tokens"], "prompt_tokens")
        try:
            prompt_ids = tuple(int(item) for item in fields["prompt_ids"].split(","))
        except ValueError as exc:
            raise BackendError("persistent V2 prompt IDs are malformed") from exc
        if (len(prompt_ids) != prompt_tokens or
                any(token < 0 for token in prompt_ids)):
            raise BackendError("persistent V2 prompt-ID count drift")
        image_tokens = _canonical_uint(
            fields["image_tokens"], "image_tokens", allow_zero=True,
        )
        if expected_image_tokens is not None and image_tokens != expected_image_tokens:
            raise BackendError("persistent V2 image-token drift")
        output_limit = _canonical_uint(fields["output_limit"], "output_limit")
        output_steps = _canonical_uint(fields["output_steps"], "output_steps")
        if (output_limit < 1 or output_limit > maximum_output_limit or
                (expected_output_limit is not None and
                 output_limit != expected_output_limit) or
                output_steps > output_limit):
            raise BackendError("persistent V3 output-limit drift")
        stop_token = _canonical_int(fields["stop_token"], "stop_token", minimum=-1)
        synthetic_close = _canonical_uint(
            fields["synthetic_close"], "synthetic_close", allow_zero=True,
        )
        if synthetic_close not in (0, 1):
            raise BackendError("persistent V2 synthetic-close drift")
        position_before = _canonical_uint(
            fields["position_before"], "position_before", allow_zero=True,
        )
        position_after = _canonical_uint(fields["position_after"], "position_after")
        response_bytes = _canonical_uint(
            fields["response_bytes"], "response_bytes", allow_zero=True,
        )
        if ((expected_position is not None and
             position_before != expected_position) or
                response_bytes != len(response)):
            raise BackendError("persistent V2 result position/response drift")
        if position_after != (
                position_before + prompt_tokens + output_steps + synthetic_close):
            raise BackendError("persistent V2 committed-position drift")
        if fields["session_continuable"] != "1" or \
                (synthetic_close == 0) != (stop_token == 106):
            raise BackendError("persistent V2 continuation/turn-close drift")
        try:
            output_ids = tuple(int(item) for item in fields["output_ids"].split(","))
        except ValueError as exc:
            raise BackendError("persistent V2 output IDs drift") from exc
        if len(output_ids) != output_steps or any(item < 0 for item in output_ids):
            raise BackendError("persistent V2 output count drift")
        if stop_token == -1:
            if output_steps != output_limit or fields["finish_reason"] != "length":
                raise BackendError("persistent V3 length finish drift")
            finish_reason = "length"
        else:
            if (not output_ids or output_ids[-1] != stop_token or
                    fields["finish_reason"] != "stop"):
                raise BackendError("persistent V2 stop finish drift")
            finish_reason = "stop"
        state_sha256 = fields["state_sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", state_sha256) is None:
            raise BackendError("persistent V2 state digest drift")
        mindset_end = _canonical_uint(
            fields["mindset_end"], "mindset_end", allow_zero=True,
        )
        mindset_sha256 = fields["mindset_sha256"]
        facts_sha256 = fields["facts_sha256"]
        facts_rows = _canonical_uint(
            fields["facts_rows"], "facts_rows", allow_zero=True,
        )
        if (mindset_end != expected_mindset_end or
                fields["mindset_mode"] != "prefix" or
                fields["build_identity_sha256"] != expected_build_identity_sha256 or
                mindset_sha256 != expected_mindset_sha256 or
                re.fullmatch(r"[0-9a-f]{64}", mindset_sha256) is None or
                re.fullmatch(r"[0-9a-f]{64}", facts_sha256) is None or
                facts_rows != position_after - mindset_end):
            raise BackendError("persistent V2 mindset/factual boundary drift")
        try:
            raw_response = response.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackendError("persistent V2 response is not UTF-8") from exc
        reasoning, content = split_reasoning(raw_response)
        return RunnerResult(
            content=content, reasoning_content=reasoning,
            prompt_tokens=prompt_tokens, completion_tokens=output_steps,
            output_ids=output_ids, stop_token=stop_token,
            finish_reason=finish_reason, response_bytes=response_bytes,
            kv_loaded_tokens=position_before, kv_saved_tokens=position_after,
            session_epoch=expected_epoch, turn_id=turn_id,
            history_turns=history_turns,
            synthetic_close=synthetic_close, state_sha256=state_sha256,
            session_continuable=True,
            client_request_sha256=client_request_sha256,
            request_sha256=request_sha256, replayed=replayed,
            original_request_id=original_request_id,
            image_tokens=image_tokens,
            mindset_mode="prefix", mindset_end=mindset_end,
            mindset_sha256=mindset_sha256, facts_sha256=facts_sha256,
            facts_rows=facts_rows, prompt_ids=prompt_ids,
            response_sha256=hashlib.sha256(response).hexdigest(),
            output_limit=output_limit,
        )

    def stop(self, *, force: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        self.active_request = None
        self.phase = "DEAD" if force else "DRAINING"
        if not force and self.process.poll() is None:
            try:
                self.stdin.write(b"SALT_GEMMA4_DRAIN_V2\n")
                self.stdin.flush()
                self.stdin.close()
                self.process.wait(timeout=10.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                force = True
                self.phase = "DEAD"
        if force and self.process.poll() is None:
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                _kill_process_group(self.process)
                self.process.wait()
        for reader in self.readers:
            reader.join(timeout=2.0)
        for stream in (self.stdin, self.stdout, self.stderr):
            try:
                stream.close()
            except OSError:
                pass
        if not force and self.process.returncode == 0:
            self.phase = "CLOSED"
        else:
            self.phase = "DEAD"


def _run_process_group_stream(
        command: list[str], *, timeout: float, env: dict[str, str],
        accumulator: NativeStreamAccumulator,
        on_start: Callable[[], None]) -> subprocess.CompletedProcess[bytes]:
    """Run a wrapper while relaying authenticated native stream frames."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    reader_errors: list[BaseException] = []

    def read_stdout() -> None:
        try:
            while True:
                chunk = os.read(stdout_pipe.fileno(), 65536)
                if not chunk:
                    return
                stdout_chunks.append(chunk)
        except BaseException as exc:
            reader_errors.append(exc)
            _kill_process_group(process)

    def read_stderr() -> None:
        pending = bytearray()
        try:
            while True:
                chunk = os.read(stderr_pipe.fileno(), 65536)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
                pending.extend(chunk)
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(pending[:newline + 1])
                    del pending[:newline + 1]
                    accumulator.feed_line(line)
            if pending.startswith(accumulator.prefix):
                raise BackendError("truncated native stream frame")
        except BaseException as exc:
            reader_errors.append(exc)
            _kill_process_group(process)

    readers = [
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
    ]
    try:
        on_start()
    except BaseException:
        _kill_process_group(process)
        process.wait()
        raise
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        process.wait()
        for reader in readers:
            reader.join(timeout=2.0)
        raise subprocess.TimeoutExpired(
            command, timeout,
            output=b"".join(stdout_chunks), stderr=b"".join(stderr_chunks),
        )
    for reader in readers:
        reader.join(timeout=2.0)
    if any(reader.is_alive() for reader in readers):
        _kill_process_group(process)
        raise BackendError("native stream pipe reader did not terminate")
    if reader_errors:
        raise reader_errors[0]
    return subprocess.CompletedProcess(
        command, process.returncode,
        b"".join(stdout_chunks), b"".join(stderr_chunks),
    )


@dataclass(frozen=True)
class Controls:
    context_tokens: int
    output_tokens: int
    reasoning_effort: str
    enable_thinking: bool
    kv_allowance_gb: float
    kv_bytes: int
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    sampler_abi: str = "salt-greedy-v1"
    sampler_abi_id: int = 1
    temperature: float = 0.0
    temperature_bits: str = "00000000"
    seed: int = 0
    top_k: int = 1
    rng_abi: str = "none"
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1


@dataclass(frozen=True)
class RenderedChat:
    prompt: str
    image_url: str | None


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class RunnerResult:
    content: str
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    output_ids: tuple[int, ...]
    stop_token: int
    finish_reason: str
    response_bytes: int
    kv_loaded_tokens: int
    kv_saved_tokens: int
    session_epoch: int = 0
    turn_id: int = 0
    history_turns: int = 0
    synthetic_close: int = 0
    state_sha256: str = ""
    session_continuable: bool = True
    client_request_sha256: str = ""
    request_sha256: str = ""
    replayed: bool = False
    original_request_id: int = 0
    image_tokens: int = 0
    mindset_mode: str = "prefix"
    mindset_end: int = 0
    mindset_sha256: str = ""
    facts_sha256: str = ""
    facts_rows: int = 0
    prompt_ids: tuple[int, ...] = ()
    response_sha256: str = ""
    output_limit: int = 0
    dpr_result: dict | None = None
    memory_snapshots: tuple[dict[str, str], ...] = ()
    dpr_waterfall: dict[str, str] | None = None


@dataclass(frozen=True)
class KVCacheMetadata:
    cache_id: str
    path: Path
    version: int
    context_tokens: int
    position: int
    bytes: int
    compatibility_sha256: str
    build_identity_sha256: str
    logical_state_sha256: str
    payload_sha256: str
    legacy_read_only: bool
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    storage_format: str = "native-framed"
    metadata: dict | None = None
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1

    def as_dict(self) -> dict:
        return {
            "id": self.cache_id,
            "version": self.version,
            "context_tokens": self.context_tokens,
            "position": self.position,
            "bytes": self.bytes,
            "compatibility_sha256": self.compatibility_sha256,
            "build_identity_sha256": self.build_identity_sha256,
            "logical_state_sha256": self.logical_state_sha256,
            "payload_sha256": self.payload_sha256,
            "legacy_read_only": self.legacy_read_only,
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
            "storage_format": self.storage_format,
            "metadata": self.metadata,
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        }


@dataclass(frozen=True)
class KVCacheSelection:
    mode: str
    cache_id: str | None
    load_path: Path | None
    save_path: Path | None
    target_path: Path | None
    loaded_tokens: int
    context_tokens: int | None


@dataclass(frozen=True)
class KVCacheOutcome:
    mode: str
    cache_id: str | None
    loaded_tokens: int
    saved_tokens: int

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "id": self.cache_id,
            "loaded_tokens": self.loaded_tokens,
            "saved_tokens": self.saved_tokens,
        }


@dataclass(frozen=True)
class Completion:
    result: RunnerResult
    controls: Controls
    modality: str
    elapsed_s: float
    kv_cache: KVCacheOutcome
    mentor_state: dict | None = None
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1

    def __post_init__(self) -> None:
        _emit_server_metrics(self)


def _emit_server_metrics(completion: Completion) -> None:
    """Print one bounded timing/cache summary for every completed request."""
    waterfall = completion.result.dpr_waterfall or {}
    prompt_tokens = completion.result.prompt_tokens
    completion_tokens = completion.result.completion_tokens
    proposed_tokens = int(waterfall.get("proposed_tokens", "0"))
    accepted_tokens = int(waterfall.get("accepted_tokens", "0"))
    prefill_ns = int(waterfall.get("prefill_ordinary_ns", "0"))
    proposal_ns = int(waterfall.get("decode_ordinary_ns", "0"))
    target_ns = int(waterfall.get("decode_verify_ns", "0"))
    value = {
        "request": waterfall.get("request"),
        "modality": completion.modality,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_ms_per_input_token": (
            None if completion.modality == "image" or prompt_tokens < 1 else
            prefill_ns / prompt_tokens / 1_000_000.0
        ),
        "decode_ms_per_completion_token": (
            None if completion_tokens < 1 else
            (proposal_ns + target_ns) / completion_tokens / 1_000_000.0
        ),
        "target_ms_per_accepted_token": (
            None if accepted_tokens < 1 else
            target_ns / accepted_tokens / 1_000_000.0
        ),
        "proposal_ms_per_proposed_token": (
            None if proposed_tokens < 1 else
            proposal_ns / proposed_tokens / 1_000_000.0
        ),
        "live_kv_loaded_tokens": completion.kv_cache.loaded_tokens,
        "prefill_hits": int(waterfall.get("prefill_hits", "0")),
        "prefill_cached_tokens": int(
            waterfall.get("prefill_cached_tokens", "0")
        ),
        "parent_hits": int(waterfall.get("parent_hits", "0")),
        "nomogram_hits": int(waterfall.get("nomogram_hits", "0")),
        "decode_misses": int(waterfall.get("decode_misses", "0")),
        "decode_cycles": int(waterfall.get("decode_cycles", "0")),
        "elapsed_ms": completion.elapsed_s * 1000.0,
    }
    print(
        "[gemma4-server-ms-tk] " + json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ),
        file=sys.stderr,
        flush=True,
    )
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1


class KVCacheStore:
    """Private logical-ID store for transferable Gemma 4 KV snapshots."""

    def __init__(self, root: Path,
                 compatibility_sha256: bytes,
                 build_identity_sha256: bytes):
        if (
            not isinstance(compatibility_sha256, bytes)
            or len(compatibility_sha256) != 32
            or compatibility_sha256 == bytes(32)
        ):
            raise BackendError("invalid Gemma 4 KV compatibility identity")
        if (not isinstance(build_identity_sha256, bytes) or
                len(build_identity_sha256) != 32 or
                build_identity_sha256 == bytes(32)):
            raise BackendError("invalid Gemma 4 build identity")
        expanded = root.expanduser()
        expanded.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.base_root = self._private_directory(expanded, "root")
        namespace = self.base_root / (
            "g4kvc006-" + compatibility_sha256.hex()
        )
        namespace.mkdir(mode=0o700, exist_ok=True)
        self.root = self._private_directory(namespace, "compatibility namespace")
        self.compatibility_sha256 = compatibility_sha256
        self.build_identity_sha256 = build_identity_sha256

    @staticmethod
    def _private_directory(path: Path, label: str) -> Path:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise BackendError(f"Gemma 4 KV cache {label} must be a real directory")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise BackendError(f"Gemma 4 KV cache {label} must be owned by the server user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise BackendError(
                f"Gemma 4 KV cache {label} must have mode 0700 or stricter"
            )
        return path.resolve(strict=True)

    @staticmethod
    def _request(value: object) -> tuple[str, str | None]:
        if value is None:
            return "none", None
        if not isinstance(value, dict):
            raise RequestError("kv_cache must be an object")
        unknown = set(value) - {"mode", "id"}
        if unknown:
            raise RequestError(
                "unsupported kv_cache fields: " + ", ".join(sorted(unknown))
            )
        mode = value.get("mode", "none")
        if mode not in ("none", "create", "load", "resume"):
            raise RequestError("kv_cache.mode must be none, create, load, or resume")
        cache_id = value.get("id")
        if mode == "none":
            if cache_id is not None:
                raise RequestError("kv_cache.id is invalid when mode is none")
            return mode, None
        if not isinstance(cache_id, str) or not KV_CACHE_ID_RE.fullmatch(cache_id) \
                or ".." in cache_id:
            raise RequestError("invalid KV cache id")
        return mode, cache_id

    def _path(self, cache_id: str) -> Path:
        path = self.root / f"{cache_id}.g4kv"
        if path.parent != self.root:
            raise RequestError("invalid KV cache id")
        return path

    @staticmethod
    def _expected_dimensions() -> tuple[int, ...]:
        return tuple(1024 if layer % 6 == 5 else 2048
                     for layer in range(GEMMA4_LAYERS))

    def _metadata(self, cache_id: str, path: Path, *,
                  verify_payload: bool = True) -> KVCacheMetadata:
        try:
            metadata = _state_artifact_module.inspect_g4kvc006(
                path,
                expected_compatibility_sha256=self.compatibility_sha256,
                legacy_v5_compatibility_sha256=(
                    KV_LEGACY_V5_COMPATIBILITY_SHA256
                ),
                legacy_v4_compatibility_sha256=(
                    KV_LEGACY_V4_COMPATIBILITY_SHA256
                ),
                verify_payload=verify_payload,
                require_private_mode=False,
            )
        except _state_artifact_module.StateArtifactError as exc:
            raise RequestError(f"KV cache {cache_id!r}: {exc}") from exc
        return KVCacheMetadata(
            cache_id=cache_id, path=metadata.path, version=metadata.version,
            context_tokens=metadata.context_tokens, position=metadata.position,
            bytes=metadata.total_bytes,
            compatibility_sha256=metadata.compatibility_sha256,
            build_identity_sha256=self.build_identity_sha256.hex(),
            logical_state_sha256=metadata.logical_state_sha256,
            payload_sha256=metadata.payload_sha256,
            legacy_read_only=metadata.legacy_read_only,
        )

    def _temporary_path(self, cache_id: str) -> Path:
        fd, name = tempfile.mkstemp(
            prefix=f".{cache_id}.", suffix=".tmp", dir=self.root,
        )
        os.close(fd)
        os.unlink(name)
        return Path(name)

    def resolve(self, value: object, controls: Controls) -> KVCacheSelection:
        mode, cache_id = self._request(value)
        if mode == "none":
            return KVCacheSelection("none", None, None, None, None, 0, None)
        assert cache_id is not None
        target = self._path(cache_id)
        if mode == "create":
            if target.exists() or target.is_symlink():
                raise RequestError(f"KV cache {cache_id!r} already exists")
            return KVCacheSelection(
                mode, cache_id, None, self._temporary_path(cache_id), target,
                0, controls.context_tokens,
            )
        metadata = self._metadata(cache_id, target)
        if metadata.context_tokens != controls.context_tokens:
            raise RequestError(
                f"KV cache context gate {metadata.context_tokens} does not match "
                f"selected context gate {controls.context_tokens}"
            )
        save_path = self._temporary_path(cache_id) if mode == "resume" else None
        return KVCacheSelection(
            mode, cache_id, target, save_path, target,
            metadata.position, metadata.context_tokens,
        )

    @contextmanager
    def lock(self, value: object) -> Iterator[None]:
        mode, cache_id = self._request(value)
        if mode == "none":
            yield
            return
        assert cache_id is not None
        path = self.root / f".{cache_id}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = -1
        try:
            fd = os.open(path, flags, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if fd >= 0:
                os.close(fd)
            raise RequestError(f"KV cache {cache_id!r} is busy") from exc
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise BackendError(f"cannot lock KV cache {cache_id!r}: {exc}") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def publish(self, selection: KVCacheSelection, saved_tokens: int) -> KVCacheMetadata:
        if selection.save_path is None or selection.target_path is None or \
                selection.cache_id is None:
            raise BackendError("KV cache publication was not requested")
        metadata = self._metadata(selection.cache_id, selection.save_path)
        if metadata.context_tokens != selection.context_tokens or \
                metadata.position != saved_tokens:
            raise BackendError("native KV cache save metadata drift")
        if selection.mode == "create":
            try:
                os.link(selection.save_path, selection.target_path)
            except FileExistsError as exc:
                raise RequestError(
                    f"KV cache {selection.cache_id!r} already exists"
                ) from exc
            os.unlink(selection.save_path)
        elif selection.mode == "resume":
            os.replace(selection.save_path, selection.target_path)
        else:
            raise BackendError("invalid writable KV cache mode")
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return self._metadata(selection.cache_id, selection.target_path)

    @staticmethod
    def discard(selection: KVCacheSelection) -> None:
        if selection.save_path is not None:
            try:
                selection.save_path.unlink()
            except FileNotFoundError:
                pass

    def list_caches(self) -> list[KVCacheMetadata]:
        result = []
        for path in sorted(self.root.glob("*.g4kv")):
            result.append(self._metadata(path.stem, path, verify_payload=False))
        return result

    def load_metadata(self, cache_id: str) -> KVCacheMetadata:
        self._request({"mode": "load", "id": cache_id})
        return self._metadata(cache_id, self._path(cache_id))

    def contains(self, cache_id: str) -> bool:
        self._request({"mode": "load", "id": cache_id})
        path = self._path(cache_id)
        return path.exists() and not path.is_symlink()

    def delete(self, cache_id: str) -> bool:
        self._request({"mode": "load", "id": cache_id})
        path = self._path(cache_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    # Preserve existing framed files; only new publications use sidecar folders.
    _framed_metadata = _metadata
    _framed_cache_path = _path
    _framed_list = list_caches
    _framed_delete = delete
    _export_provenance: dict | None = None

    def _sidecar_path(self, cache_id: str) -> Path:
        self._request({"mode": "load", "id": cache_id})
        folder = self.root / cache_id
        framed = self._framed_cache_path(cache_id)
        if folder.exists() or folder.is_symlink():
            return folder
        return framed if framed.exists() or framed.is_symlink() else folder

    def _sidecar_metadata(self, cache_id: str, path: Path, *,
                          verify_payload: bool = True) -> KVCacheMetadata:
        if not path.is_dir() and not path.is_symlink():
            return self._framed_metadata(
                cache_id, path, verify_payload=verify_payload,
            )
        try:
            record = _kv_sidecar.read(path, verify_payload=verify_payload)
            header = bytes.fromhex(record["native_header_hex"])
            if len(header) != KV_CACHE_HEADER.size:
                raise ValueError("native framing size mismatch")
            values = KV_CACHE_HEADER.unpack(header)
            magic, version, layers, context, position, total = values[:6]
            dimensions = tuple(values[6:36])
            compatibility, state_hash, payload_hash, row_base, window = values[36:]
            if (magic != KV_CACHE_MAGIC or version != KV_CACHE_VERSION or
                    layers != GEMMA4_LAYERS or
                    dimensions != self._expected_dimensions() or
                    not 1 <= position <= context <= _state_artifact_module.GEMMA4_MAX_CONTEXT or
                    total != KV_CACHE_HEADER.size + record["payload_bytes"] or
                    record["payload_bytes"] != _state_artifact_module._payload_bytes(
                        position, dimensions, version=version) or
                    compatibility != self.compatibility_sha256 or
                    payload_hash.hex() != record["payload_sha256"] or
                    state_hash == bytes(32) or
                    row_base != max(0, position - 1023) or window != 1024):
                raise ValueError("native schema/compatibility mismatch")
            build = record["metadata"].get("build_identity_sha256")
            if not isinstance(build, str) or re.fullmatch(r"[0-9a-f]{64}", build) is None:
                raise ValueError("invalid build provenance")
            return KVCacheMetadata(
                cache_id=cache_id, path=path.resolve(strict=True), version=version,
                context_tokens=context, position=position, bytes=total,
                compatibility_sha256=compatibility.hex(), build_identity_sha256=build,
                logical_state_sha256=state_hash.hex(), payload_sha256=payload_hash.hex(),
                legacy_read_only=False, storage_format="payload-with-sidecar",
                metadata=record["metadata"],
            )
        except (OSError, ValueError) as exc:
            raise RequestError(f"KV cache {cache_id!r}: {exc}") from exc

    def _sidecar_publish(self, selection: KVCacheSelection,
                         saved_tokens: int) -> KVCacheMetadata:
        if selection.save_path is None or selection.cache_id is None:
            raise BackendError("KV cache publication was not requested")
        native = self._framed_metadata(selection.cache_id, selection.save_path)
        if (native.version != KV_CACHE_VERSION or
                native.context_tokens != selection.context_tokens or
                native.position != saved_tokens or
                selection.mode not in ("create", "resume")):
            raise BackendError("native KV cache save metadata drift")
        directory = self.root / selection.cache_id
        if selection.mode == "create" and self.contains(selection.cache_id):
            raise RequestError(f"KV cache {selection.cache_id!r} already exists")
        metadata = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_by": "server", "notes": None,
            "build_identity_sha256": self.build_identity_sha256.hex(),
            "native_schema": native.as_dict(),
            "inference": dict(self._export_provenance or {}),
        }
        try:
            _kv_sidecar.publish(
                selection.save_path, directory, header_bytes=KV_CACHE_HEADER.size,
                expected_payload_sha256=native.payload_sha256, metadata=metadata,
                replace_existing=selection.mode == "resume",
            )
            result = self._metadata(selection.cache_id, directory)
            selection.save_path.unlink()
            return result
        except (OSError, ValueError) as exc:
            raise RequestError(f"KV cache {selection.cache_id!r}: {exc}") from exc

    @contextmanager
    def native_path(self, metadata: KVCacheMetadata) -> Iterator[Path]:
        if metadata.storage_format == "native-framed":
            yield metadata.path
            return
        with _kv_sidecar.native_path(metadata.path) as path:
            # Reuse native import framing; no inference or state changes in Python.
            framed = self._framed_metadata(metadata.cache_id, path, verify_payload=False)
            if (framed.position != metadata.position or
                    framed.logical_state_sha256 != metadata.logical_state_sha256 or
                    framed.payload_sha256 != metadata.payload_sha256):
                raise RequestError("KV sidecar changed before import")
            yield path

    def _sidecar_list(self) -> list[KVCacheMetadata]:
        result = {item.cache_id: item for item in self._framed_list()}
        for path in sorted(self.root.iterdir()):
            if (KV_CACHE_ID_RE.fullmatch(path.name) and ".." not in path.name and
                    path.is_dir()):
                result[path.name] = self._metadata(path.name, path, verify_payload=False)
        return [result[key] for key in sorted(result)]

    def _sidecar_delete(self, cache_id: str) -> bool:
        self._request({"mode": "load", "id": cache_id})
        path = self.root / cache_id
        deleted = False
        if path.exists() or path.is_symlink():
            try:
                _kv_sidecar.delete(path)
                deleted = True
            except (OSError, ValueError) as exc:
                raise RequestError(f"KV cache {cache_id!r}: {exc}") from exc
        return self._framed_delete(cache_id) or deleted

    _path = _sidecar_path
    _metadata = _sidecar_metadata
    publish = _sidecar_publish
    list_caches = _sidecar_list
    delete = _sidecar_delete
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1


@dataclass(frozen=True)
class Gemma4Config:
    source_dir: Path
    pool: Path
    receipt: Path
    model_id: str = DEFAULT_MODEL_ID
    text_wrapper: Path = ROOT / "tools" / "gemma4-qa.py"
    multimodal_wrapper: Path = ROOT / "tools" / "gemma4-multimodal-qa.py"
    text_runner: Path = ROOT / "gemma4-qa"
    multimodal_runner: Path = ROOT / "gemma4-multimodal-qa"
    server_runner: Path = ROOT / "gemma4-server"
    text_build_receipt: Path | None = None
    multimodal_build_receipt: Path | None = None
    server_build_receipt: Path | None = None
    auth_receipt: Path | None = None
    shared_kv: Path | None = None
    mentor_root: Path | None = None
    mentor_anchor: str | None = None
    dpr_root: Path | None = None
    dpr_mode: str = "off"
    dpr_budget_gb: float = 0.0
    dpr_serial_ms_per_token: float = 0.0
    dpr_retention_gb: float = 2.0
    dpr_mentor_enabled: bool = False
    dpr_mindset_attention_enabled: bool = False
    proof_state: bool = False
    python: str = sys.executable
    workers: int = 3
    context_tokens: int = 512
    max_output_tokens: int = 128
    prefill_chunk_tokens: int = 512
    kv_layout: str = "hybrid"
    kv_budget_gb: float = 1.0
    timeout_s: float = 900.0
    image_root: Path | None = None
    kv_cache_root: Path | None = None


class Gemma4AuthenticationAuthority:
    """Authenticate immutable Gemma artifacts once and retain their descriptors."""

    def __init__(self, config: Gemma4Config, *, reauthenticate: bool = False):
        module_name = f"salt_gemma4_auth_authority_{id(self):x}"
        spec = importlib.util.spec_from_file_location(
            module_name, config.multimodal_wrapper,
        )
        if spec is None or spec.loader is None:
            raise BackendError("cannot load Gemma 4 authentication authority")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        build_receipt = config.server_build_receipt or Path(
            str(config.server_runner) + ".build.json"
        )
        auth_receipt = config.auth_receipt or Path(str(config.pool) + ".auth.json")
        try:
            if reauthenticate:
                binding = module.authenticate_binding(
                    source_dir=config.source_dir, pool=config.pool,
                    receipt=config.receipt, runner=config.server_runner,
                    build_receipt=build_receipt,
                    auth_receipt=auth_receipt,
                    kv_compat_sha256=KV_COMPATIBILITY.hexdigest,
                )
            else:
                binding = module.open_authenticated_binding(
                    source_dir=config.source_dir, pool=config.pool,
                    receipt=config.receipt, runner=config.server_runner,
                    build_receipt=build_receipt,
                    auth_receipt=auth_receipt,
                    kv_compat_sha256=KV_COMPATIBILITY.hexdigest,
                )
        except BaseException as exc:
            raise BackendError(f"Gemma 4 authentication failed: {exc}") from exc
        if (binding.compatibility_sha256 != KV_COMPATIBILITY.hexdigest or
                binding.build_identity_sha256 == "0" * 64):
            binding.close()
            raise BackendError("Gemma 4 authentication identity drift")
        self._binding: Any = binding
        self.source_dir = config.source_dir.resolve()
        self.pool = config.pool.resolve()
        self.receipt = config.receipt.resolve()
        self.runner = config.server_runner.resolve()
        self.build_receipt = build_receipt.resolve()
        self.auth_receipt = auth_receipt.resolve()
        self.closed = False

    @property
    def compatibility_sha256(self) -> str:
        return cast(str, self._binding.compatibility_sha256)

    @property
    def build_identity_sha256(self) -> str:
        return cast(str, self._binding.build_identity_sha256)

    @property
    def binding(self) -> bytes:
        if self.closed:
            raise BackendError("Gemma 4 authentication authority is closed")
        return cast(bytes, self._binding.binding)

    @property
    def inherited_fds(self) -> tuple[int, ...]:
        if self.closed:
            raise BackendError("Gemma 4 authentication authority is closed")
        return cast(tuple[int, ...], self._binding.inherited_fds)

    def validate(self, config: Gemma4Config, build_identity_sha256: str) -> None:
        build_receipt = config.server_build_receipt or Path(
            str(config.server_runner) + ".build.json"
        )
        auth_receipt = config.auth_receipt or Path(str(config.pool) + ".auth.json")
        if (self.closed or config.source_dir.resolve() != self.source_dir or
                config.pool.resolve() != self.pool or
                config.receipt.resolve() != self.receipt or
                config.server_runner.resolve() != self.runner or
                build_receipt.resolve() != self.build_receipt or
                auth_receipt.resolve() != self.auth_receipt or
                self.compatibility_sha256 != KV_COMPATIBILITY.hexdigest or
                self.build_identity_sha256 != build_identity_sha256):
            raise BackendError("Gemma 4 authentication authority mismatch")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._binding.close()


def _positive_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise RequestError(f"{field} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise RequestError(f"{field} must be a positive finite number")
    return parsed


def _output_token_value(body: dict) -> int | None:
    aliases = ("max_tokens", "max_completion_tokens", "max_output_tokens")
    supplied: list[tuple[str, int]] = []
    for key in aliases:
        if key not in body or body[key] is None:
            continue
        value = body[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RequestError(f"{key} must be a positive integer")
        supplied.append((key, value))
    if not supplied:
        return None
    values = {value for _, value in supplied}
    if len(values) != 1:
        raise RequestError("conflicting output-token aliases")
    return supplied[0][1]


def _reasoning_effort(body: dict) -> str:
    flat = body.get("reasoning_effort")
    nested_value = None
    if "reasoning" in body and body["reasoning"] is not None:
        nested = body["reasoning"]
        if not isinstance(nested, dict):
            raise RequestError("reasoning must be an object")
        nested_value = nested.get("effort")
    if flat is not None and nested_value is not None and flat != nested_value:
        raise RequestError("conflicting reasoning effort fields")
    effort = flat if flat is not None else nested_value
    if effort is None:
        effort = "none"
    if not isinstance(effort, str):
        raise RequestError("reasoning effort must be a string")
    effort = effort.lower()
    if effort not in EFFORT_OUTPUT:
        raise RequestError(
            "reasoning effort must be one of none, minimal, low, medium, high"
        )
    return effort


# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
def _sampler_controls(body: dict) -> tuple[str, int, float, str, int, int, str]:
    raw_temperature = body.get("temperature", 0.0)
    if isinstance(raw_temperature, bool) or not isinstance(
            raw_temperature, (int, float)):
        raise RequestError("temperature must be a number")
    try:
        temperature = float(raw_temperature)
    except OverflowError as exc:
        raise RequestError("temperature must be finite") from exc
    if not math.isfinite(temperature) or temperature < 0.0 or temperature > 2.0:
        raise RequestError("temperature must be finite and between 0 and 2")
    try:
        temperature_raw = struct.pack(">f", temperature)
    except OverflowError as exc:
        raise RequestError("temperature is outside binary32 range") from exc
    temperature_bits = struct.unpack(">I", temperature_raw)[0]
    temperature = struct.unpack(">f", temperature_raw)[0]
    if temperature == 0.0:
        if float(raw_temperature) > 0.0:
            raise RequestError("temperature is too small for binary32 sampling")
        if body.get("seed") is not None:
            raise RequestError("seed requires positive temperature")
        top_k = body.get("top_k", 1)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 1:
            raise RequestError("greedy sampling requires top_k=1")
        return "salt-greedy-v1", 1, 0.0, "00000000", 0, 1, "none"
    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or \
            seed < 0 or seed > (1 << 64) - 1:
        raise RequestError("seed must be an unsigned 64-bit integer")
    top_k = body.get("top_k", 20)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or \
            top_k < 1 or top_k > 256:
        raise RequestError("top_k must be an integer between 1 and 256")
    return (
        "salt-temperature-counter-v1", 2, temperature,
        f"{temperature_bits:08x}", seed, top_k, "splitmix64-counter-v1",
    )
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
def resolve_controls(body: dict, *, server_kv_budget_gb: float,
                     context_tokens: int = 512,
                     max_output_tokens: int = 128,
                     kv_layout: str = "hybrid") -> Controls:
    """Resolve OpenAI aliases within the configured native Gemma capacity."""
    if not isinstance(body, dict):
        raise RequestError("request body must be an object")
    server_budget = _positive_finite(server_kv_budget_gb, "server KV budget")
    effort = _reasoning_effort(body)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    sampler = _sampler_controls(body)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    explicit_output = _output_token_value(body)
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or \
            context_tokens < 3 or context_tokens > GEMMA4_MODEL_MAX_CONTEXT:
        raise RequestError(
            f"server context must be between 3 and {GEMMA4_MODEL_MAX_CONTEXT} tokens"
        )
    if not isinstance(max_output_tokens, int) or \
            isinstance(max_output_tokens, bool) or max_output_tokens < 1 or \
            max_output_tokens > context_tokens - 2:
        raise RequestError("server output limit must leave prompt and close rows")
    output_tokens = (
        explicit_output if explicit_output is not None
        else min(EFFORT_OUTPUT[effort], max_output_tokens)
    )
    if output_tokens > max_output_tokens:
        raise RequestError(
            f"requested output cap exceeds server maximum {max_output_tokens} tokens"
        )
    kv_bytes = gemma4_kv_capacity_bytes(context_tokens, kv_layout)

    requested_budget = body.get("kv_allowance_gb", server_budget)
    request_budget = _positive_finite(requested_budget, "kv_allowance_gb")
    if request_budget > server_budget + 1e-12:
        raise RequestError(
            f"kv_allowance_gb exceeds server maximum {server_budget:g} GB"
        )
    request_budget_bytes = int(request_budget * 1_000_000_000)
    if kv_bytes > request_budget_bytes:
        required = kv_bytes / 1_000_000_000
        raise RequestError(
            f"configured CTX{context_tokens} requires KV allowance "
            f"{required:.6f} GB, request allows {request_budget:g} GB"
        )
    return Controls(
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        reasoning_effort=effort,
        enable_thinking=effort in THINKING_EFFORTS,
        kv_allowance_gb=request_budget,
        kv_bytes=kv_bytes,
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        sampler_abi=sampler[0],
        sampler_abi_id=sampler[1],
        temperature=sampler[2],
        temperature_bits=sampler[3],
        seed=sampler[4],
        top_k=sampler[5],
        rng_abi=sampler[6],
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    )


def _client_text(value: str) -> str:
    if "\0" in value:
        raise RequestError("message content contains NUL")
    if any(token in value for token in RESERVED_PROMPT_TOKENS):
        raise RequestError("message content contains a reserved Gemma control token")
    return value.strip()


def _text_and_image_parts(content: object, role: str) -> tuple[str, str | None]:
    if isinstance(content, str):
        return _client_text(content), None
    if not isinstance(content, list):
        raise RequestError("message content must be a string or content-part array")
    rendered: list[str] = []
    image_url: str | None = None
    for part in content:
        if not isinstance(part, dict):
            raise RequestError("content parts must be objects")
        part_type = part.get("type")
        if part_type in ("text", "input_text"):
            unknown = set(part) - {"type", "text"}
            if unknown:
                raise RequestError(
                    "unsupported text content fields: " + ", ".join(sorted(unknown))
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise RequestError("text content part requires string text")
            rendered.append(_client_text(text))
            continue
        if part_type in ("image_url", "input_image", "image"):
            if role != "user":
                raise RequestError("images are accepted only in user messages")
            unknown = set(part) - {"type", "image_url", "url"}
            if unknown:
                raise RequestError(
                    "unsupported image content fields: " + ", ".join(sorted(unknown))
                )
            raw = part.get("image_url")
            if isinstance(raw, dict):
                nested_unknown = set(raw) - {"url"}
                if nested_unknown:
                    raise RequestError(
                        "unsupported image_url fields: " +
                        ", ".join(sorted(nested_unknown))
                    )
                raw = raw.get("url")
            top_url = part.get("url")
            if raw is not None and top_url is not None:
                raise RequestError("image content has conflicting URL fields")
            if raw is None:
                raw = top_url
            if not isinstance(raw, str) or not raw:
                raise RequestError("image content part requires image_url.url")
            if image_url is not None:
                raise RequestError("Gemma 4 supports exactly one image per request")
            image_url = raw
            rendered.append(IMAGE_BLOCK)
            continue
        raise RequestError(f"unsupported content part type: {part_type!r}")
    return "".join(rendered).strip(), image_url


def render_chat(messages: object, *, enable_thinking: bool,
                continuation: bool = False) -> RenderedChat:
    """Render a full Gemma chat or one user-turn continuation."""
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages required")
    if continuation:
        if len(messages) != 1 or not isinstance(messages[0], dict) or \
                messages[0].get("role") != "user":
            raise RequestError("cache continuation must contain exactly one user message")
        text, image_url = _text_and_image_parts(
            messages[0].get("content", ""), "user",
        )
        if image_url is not None:
            raise RequestError("cache continuations cannot add an image")
        if not text:
            raise RequestError("cache continuation user message has no supported content")
        chunks = ["\n<|turn>user\n", text, "<turn|>\n<|turn>model\n"]
        if not enable_thinking:
            chunks.append("<|channel>thought\n<channel|>")
        prompt = "".join(chunks)
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise RequestError("rendered prompt exceeds 64 KiB")
        return RenderedChat(prompt=prompt, image_url=None)
    if len(messages) > 64:
        raise RequestError("at most 64 messages are accepted")

    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    image_url: str | None = None
    seen_non_system = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError("messages must be objects")
        unknown = set(message) - {"role", "content"}
        if unknown:
            raise RequestError(
                "unsupported message fields: " + ", ".join(sorted(unknown))
            )
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise RequestError(f"unsupported message role: {role!r}")
        if role in ("system", "developer"):
            if seen_non_system:
                raise RequestError("system/developer messages must precede chat turns")
            text, found_image = _text_and_image_parts(message.get("content", ""), role)
            if found_image is not None:
                raise RequestError("system/developer messages cannot contain images")
            if text:
                system_parts.append(text)
            continue
        seen_non_system = True
        text, found_image = _text_and_image_parts(message.get("content", ""), role)
        if found_image is not None:
            if image_url is not None:
                raise RequestError("Gemma 4 supports exactly one image per request")
            image_url = found_image
        if not text:
            raise RequestError(f"message {index} has no supported content")
        native_role = "model" if role == "assistant" else "user"
        turns.append((native_role, text))

    if not turns or turns[-1][0] != "user":
        raise RequestError("the final chat message must be from the user")

    chunks = ["<bos>"]
    if system_parts or enable_thinking:
        chunks.append("<|turn>system\n")
        if enable_thinking:
            chunks.append("<|think|>\n")
        if system_parts:
            chunks.append("\n".join(system_parts))
        chunks.append("<turn|>\n")
    for role, text in turns:
        chunks.extend((f"<|turn>{role}\n", text, "<turn|>\n"))
    chunks.append("<|turn>model\n")
    if not enable_thinking:
        chunks.append("<|channel>thought\n<channel|>")
    prompt = "".join(chunks)
    expected_image_tokens = IMAGE_SOFT_TOKENS if image_url is not None else 0
    if prompt.count(IMAGE_TOKEN) != expected_image_tokens:
        raise BackendError("rendered image placeholder count drift")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise RequestError("rendered prompt exceeds 64 KiB")
    return RenderedChat(prompt=prompt, image_url=image_url)


def _ppm_token(data: bytes, offset: int) -> tuple[bytes, int]:
    size = len(data)
    while offset < size:
        if data[offset] == 35:  # '#'
            newline = data.find(b"\n", offset + 1)
            if newline < 0:
                raise RequestError("invalid Gemma 4 P6 image header")
            offset = newline + 1
            continue
        if chr(data[offset]).isspace():
            offset += 1
            continue
        break
    start = offset
    while offset < size and not chr(data[offset]).isspace():
        offset += 1
    if start == offset or offset >= size:
        raise RequestError("invalid Gemma 4 P6 image header")
    return data[start:offset], offset


def validate_canonical_ppm(data: bytes) -> tuple[int, int]:
    """Validate the native processor's currently qualified no-resize P6 subset."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
        raise RequestError("image must be a P6 PPM no larger than 2 MiB")
    offset = 0
    values: list[bytes] = []
    for _ in range(4):
        token, offset = _ppm_token(data, offset)
        values.append(token)
    if values[0] != b"P6":
        raise RequestError("Gemma 4 server accepts only canonical P6 PPM images")
    try:
        width = int(values[1])
        height = int(values[2])
        maximum = int(values[3])
    except ValueError as exc:
        raise RequestError("invalid Gemma 4 P6 image dimensions") from exc
    if width < 1 or height < 1 or maximum != 255:
        raise RequestError("invalid Gemma 4 P6 image dimensions or maximum")
    if offset >= len(data) or not chr(data[offset]).isspace():
        raise RequestError("invalid Gemma 4 P6 pixel boundary")
    pixel_offset = offset + 1
    expected_pixels = width * height * 3
    if expected_pixels > MAX_IMAGE_BYTES or len(data) - pixel_offset != expected_pixels:
        raise RequestError("Gemma 4 P6 pixel payload length mismatch")

    target_pixels = 70.0 * 9.0 * 256.0
    factor = math.sqrt(target_pixels / float(width * height))
    target_h = math.floor(factor * height / 48.0) * 48
    target_w = math.floor(factor * width / 48.0) * 48
    if target_h != height or target_w != width:
        raise RequestError("Gemma 4 image requires bicubic resize, which is unqualified")
    if width % 16 or height % 16:
        raise RequestError("Gemma 4 image dimensions are not patch aligned")
    patches = (width // 16) * (height // 16)
    if patches > 630 or patches % 9:
        raise RequestError("Gemma 4 image patch budget mismatch")
    return width, height


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RequestError(f"cannot open configured image: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_IMAGE_BYTES:
            raise RequestError("configured image is not a bounded regular file")
        data = b""
        while len(data) <= MAX_IMAGE_BYTES:
            chunk = os.read(fd, min(128 * 1024, MAX_IMAGE_BYTES + 1 - len(data)))
            if not chunk:
                break
            data += chunk
        after = os.fstat(fd)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )
        if identity(before) != identity(after) or len(data) != before.st_size:
            raise RequestError("configured image identity changed while reading")
        return data
    finally:
        os.close(fd)


@contextmanager
def prepared_image(image_url: str, image_root: Path | None) -> Iterator[PreparedImage]:
    """Materialize an admitted image into an invocation-owned immutable copy."""
    parsed = urlparse(image_url)
    if parsed.scheme == "data":
        try:
            metadata, encoded = image_url.split(",", 1)
        except ValueError as exc:
            raise RequestError("invalid image data URL") from exc
        allowed = {
            "data:image/x-portable-pixmap;base64",
            "data:image/ppm;base64",
            "data:image/x-ppm;base64",
        }
        if metadata.lower() not in allowed:
            raise RequestError("image data URL must contain base64 P6 PPM")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestError("invalid base64 image payload") from exc
    elif parsed.scheme == "file":
        if image_root is None:
            raise RequestError("file image URLs require --gemma-image-root")
        if parsed.netloc not in ("", "localhost"):
            raise RequestError("file image URL host is not local")
        root = Path(image_root).expanduser().resolve()
        candidate = Path(unquote(parsed.path)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RequestError("file image is outside configured image root") from exc
        data = _read_regular_file(candidate)
    elif parsed.scheme in ("http", "https"):
        raise RequestError("remote image URLs are disabled; use a P6 PPM data URL")
    else:
        raise RequestError("image_url must be a P6 PPM data: or permitted file: URL")

    width, height = validate_canonical_ppm(data)
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix="salt-gemma4-image-", suffix=".ppm",
                delete=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            path = Path(stream.name)
        yield PreparedImage(path=path, width=width, height=height)
    finally:
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass


def _line_fields(stderr: str, marker: str) -> dict[str, str]:
    lines = [line for line in stderr.splitlines() if line.startswith(marker + " ")]
    if len(lines) != 1:
        raise BackendError(f"expected one native {marker} line, got {len(lines)}")
    return dict(_FIELD_RE.findall(lines[0]))


def _positive_int(fields: dict[str, str], key: str, *, allow_zero: bool = False) -> int:
    try:
        value = int(fields[key])
    except (KeyError, ValueError) as exc:
        raise BackendError(f"invalid native integer field {key}") from exc
    if value < 0 or (value == 0 and not allow_zero):
        raise BackendError(f"invalid native integer field {key}")
    return value


def _response_block(stdout: bytes, marker: str, response_bytes: int,
                    trailer: bytes = b"") -> str:
    prefix = f"{marker}_RESPONSE_BEGIN\n".encode("ascii")
    suffix = f"{marker}_RESPONSE_END\n".encode("ascii")
    if not stdout.startswith(prefix):
        raise BackendError("native response framing drift")
    start = len(prefix)
    finish = start + response_bytes
    if finish > len(stdout):
        raise BackendError("truncated native response payload")
    payload = stdout[start:finish]
    separator = b"" if payload.endswith(b"\n") else b"\n"
    if stdout[finish:] != separator + suffix + trailer:
        raise BackendError("native response byte count or framing drift")
    if b"\0" in payload:
        raise BackendError("native response contains NUL")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackendError("native response is not UTF-8") from exc


def split_reasoning(response: str) -> tuple[str | None, str]:
    """Split Gemma's thought channel while keeping reply content clean."""
    value = response.strip()
    opener = "<|channel>thought"
    closer = "<channel|>"
    reasoning: str | None = None
    if value.startswith(opener):
        body = value[len(opener):]
        if body.startswith("\n"):
            body = body[1:]
        if closer in body:
            trace, value = body.split(closer, 1)
            reasoning = trace.strip() or None
        else:
            reasoning = body.strip() or None
            value = ""
    for token in ("<turn|>", "<eos>"):
        if value.endswith(token):
            value = value[:-len(token)].rstrip()
    return reasoning, value.strip()


def parse_runner_output(stdout: bytes, stderr: bytes, modality: str,
                        generation_cap: int) -> RunnerResult:
    if modality not in ("text", "image"):
        raise BackendError(f"invalid modality {modality!r}")
    if _HEX_RUNTIME_FALSE in stdout or _HEX_RUNTIME_FALSE in stderr:
        raise RequestError("Gemma 4 runtime readiness drift")
    if _HEX_RUNTIME_TRUE not in stderr:
        raise BackendError("native runner omitted runtime_ready=true")
    try:
        stderr_text = stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackendError("native stderr is not UTF-8") from exc
    marker = "GEMMA4_MM" if modality == "image" else "GEMMA4_QA"
    fields = _line_fields(stderr_text, f"{marker}_EXECUTION_DONE")
    response_bytes = _positive_int(fields, "response_bytes", allow_zero=True)
    prompt_tokens = _positive_int(fields, "prompt_tokens")
    output_steps = _positive_int(fields, "output_steps")
    kv_loaded_tokens = _positive_int(fields, "kv_loaded_tokens", allow_zero=True)
    kv_saved_tokens = _positive_int(fields, "kv_saved_tokens", allow_zero=True)
    try:
        stop_token = int(fields["stop_token"])
    except (KeyError, ValueError) as exc:
        raise BackendError("invalid native integer field stop_token") from exc
    if stop_token < -1:
        raise BackendError("invalid native integer field stop_token")
    matches = _OUTPUT_IDS_RE.findall(stderr_text)
    if len(matches) != 1:
        raise BackendError(f"expected one output_ids line, got {len(matches)}")
    output_ids = tuple(int(item) for item in matches[0].split(","))
    if len(output_ids) != output_steps:
        raise BackendError("native output token count drift")
    if output_steps > generation_cap:
        raise BackendError("native output exceeded generation cap")
    if modality == "image":
        image_tokens = _positive_int(fields, "image_tokens")
        if image_tokens != IMAGE_SOFT_TOKENS:
            raise BackendError("native image token count drift")
    if stop_token == -1:
        if output_steps != generation_cap:
            raise BackendError(
                "native completion ended without a stop token before the generation cap"
            )
        finish_reason = "length"
    else:
        if output_ids[-1] != stop_token:
            raise BackendError("native stop token does not match final output token")
        finish_reason = "stop"
    trailer = b"runtime_ready=true\n" if modality == "image" else b""
    response = _response_block(stdout, marker, response_bytes, trailer)
    reasoning, content = split_reasoning(response)
    return RunnerResult(
        content=content,
        reasoning_content=reasoning,
        prompt_tokens=prompt_tokens,
        completion_tokens=output_steps,
        output_ids=output_ids,
        stop_token=stop_token,
        finish_reason=finish_reason,
        response_bytes=response_bytes,
        kv_loaded_tokens=kv_loaded_tokens,
        kv_saved_tokens=kv_saved_tokens,
    )


def _private_prompt_file(prompt: str):
    return tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="salt-gemma4-prompt-",
        suffix=".txt", delete=False,
    )


class Gemma4Backend:
    """Single-flight authenticated persistent-session Gemma backend."""

    def __init__(self, config: Gemma4Config,
                 authentication: Gemma4AuthenticationAuthority | None = None):
        if not isinstance(config.context_tokens, int) or \
                isinstance(config.context_tokens, bool) or \
                config.context_tokens < 3 or \
                config.context_tokens > GEMMA4_MODEL_MAX_CONTEXT:
            raise BackendError("invalid Gemma server context capacity")
        if not isinstance(config.max_output_tokens, int) or \
                isinstance(config.max_output_tokens, bool) or \
                config.max_output_tokens < 1 or \
                config.max_output_tokens > config.context_tokens - 2:
            raise BackendError("invalid Gemma server output capacity")
        if not isinstance(config.prefill_chunk_tokens, int) or \
                isinstance(config.prefill_chunk_tokens, bool) or \
                config.prefill_chunk_tokens < 1 or \
                config.prefill_chunk_tokens > config.context_tokens or \
                config.prefill_chunk_tokens > GEMMA4_MAX_PREFILL_CHUNK:
            raise BackendError("invalid Gemma prefill chunk capacity")
        if config.kv_layout not in GEMMA4_KV_LAYOUTS:
            raise BackendError("invalid Gemma KV layout")
        if gemma4_kv_capacity_bytes(
                config.context_tokens, config.kv_layout) > \
                int(config.kv_budget_gb * 1_000_000_000):
            raise BackendError("Gemma server context exceeds configured KV budget")
        self.config = config
        self.authentication = authentication
        self.lock = threading.Lock()
        self.compatibility = KV_COMPATIBILITY
        self.cache_store = None
        self.engine: PersistentGemmaEngine | None = None
        self.turn_count = 0
        self.history_turns = 0
        self.session_position = 0
        self.shared_position = 0
        self.session_epoch = 0
        self.state_sha256 = ""
        self.mindset_mode = "prefix"
        self.mindset_end = 0
        self.mindset_sha256 = ""
        self.facts_sha256 = ""
        self.facts_rows = 0
        self.build_identity_sha256 = ""
        self.mentor_anchor_metadata = None

        self.session_history: list[tuple[str, str, str | None]] = []
        self.session_turns: list[dict[str, object]] = []
        self.last_user_signature: tuple[str, str, str | None] | None = None
        self.committed_requests: OrderedDict[str, str] = OrderedDict()
        self.last_committed_request_id: str | None = None
        self.transport_records: OrderedDict[str, tuple[int, str]] = OrderedDict()
        if config.model_id != DEFAULT_MODEL_ID:
            raise BackendError(
                f"Gemma 4 backend model id must be {DEFAULT_MODEL_ID!r}"
            )
        if config.workers < 1 or config.workers > 8:
            raise BackendError("Gemma 4 workers must be in [1, 8]")
        if (config.mentor_root is None) != (config.mentor_anchor is None):
            raise BackendError(
                "Gemma 4 mentor root and anchor must be supplied together"
            )
        if config.shared_kv is not None and config.mentor_root is not None:
            raise BackendError(
                "raw shared KV and managed mentor anchor are mutually exclusive"
            )
        if config.dpr_mode not in ("off", "persist", "dynamic"):
            raise BackendError("Gemma 4 DPR mode must be off, persist, or dynamic")
        if (config.dpr_mode == "off") != (config.dpr_root is None):
            raise BackendError("Gemma 4 DPR root is required exactly when DPR is enabled")
        if config.dpr_mode != "off" and (
                config.shared_kv is not None or config.mentor_root is not None):
            raise BackendError("DPR cannot be combined with startup shared/mentor KV")
        if config.dpr_mode == "persist":
            _positive_finite(config.dpr_budget_gb, "DPR persistent budget")
        elif config.dpr_budget_gb != 0.0:
            raise BackendError("DPR dynamic/off mode cannot use an additive DPR budget")
        if config.dpr_mode != "off":
            _positive_finite(
                config.dpr_serial_ms_per_token, "DPR serial reference",
            )
            _positive_finite(config.dpr_retention_gb, "DPR retention budget")
        if not isinstance(config.dpr_mentor_enabled, bool):
            raise BackendError("Gemma DPR Mentor enable must be a boolean")
        if config.dpr_mentor_enabled and config.dpr_mode == "off":
            raise BackendError("Gemma DPR Mentor requires DPR to be enabled")
        if not isinstance(config.dpr_mindset_attention_enabled, bool):
            raise BackendError("Gemma DPR Mindset Attention enable must be a boolean")
        if config.dpr_mindset_attention_enabled and config.dpr_mode == "off":
            raise BackendError("Gemma DPR Mindset Attention requires DPR")
        _positive_finite(config.kv_budget_gb, "server KV budget")
        _positive_finite(config.timeout_s, "Gemma 4 timeout")

    def _mentor_public(self) -> dict | None:
        value = self.mentor_anchor_metadata
        if value is None:
            return None
        return {
            "proposal_type": "exact-anchor",
            "anchor_id": value.anchor_id,
            "revision": value.revision,
            "selector": value.selector,
            "state_file_sha256": value.state_file_sha256,
            "state_payload_sha256": value.state_payload_sha256,
            "position": value.position,
            "runtime_ready": False,
        }

    def _dpr_public(self) -> dict:
        return {
            "mode": self.config.dpr_mode,
            "enabled": self.config.dpr_mode != "off",
            "additive_budget_bytes": int(
                self.config.dpr_budget_gb * 1_000_000_000
            ),
            "serial_ms_per_token": self.config.dpr_serial_ms_per_token,
            "retention_budget_bytes": int(
                self.config.dpr_retention_gb * 1_000_000_000
            ),
            "mentor_enabled": self.config.dpr_mentor_enabled,
            "mentor_policy_sha256": (
                GEMMA_DPR_MENTOR_POLICY_SHA256
                if self.config.dpr_mentor_enabled else None
            ),
            "mindset_attention_enabled": self.config.dpr_mindset_attention_enabled,
            "mindset_attention_policy_sha256": (
                GEMMA_DPR_MINDSET_ATTENTION_POLICY_SHA256
                if self.config.dpr_mindset_attention_enabled else None
            ),
            "last_result": (
                None if self.engine is None else
                getattr(self.engine, "last_dpr_result", None)
            ),
            "verification_scratch_bytes": (64 + 1) * 262_144 * 4,
            "checkpoint_max_bytes": (
                gemma4_kv_portable_bytes(self.config.context_tokens) + 256
            ),
            "target_kv_arena_bytes": gemma4_kv_capacity_bytes(
                self.config.context_tokens, self.config.kv_layout,
            ),
            "authority": "native-c",
            "runtime_ready": False,
        }

    def _base_command(self, modality: str) -> list[str]:
        wrapper = (self.config.multimodal_wrapper
                   if modality == "image" else self.config.text_wrapper)
        runner = (self.config.multimodal_runner
                   if modality == "image" else self.config.text_runner)
        configured_receipt = (self.config.multimodal_build_receipt
                              if modality == "image"
                              else self.config.text_build_receipt)
        build_receipt = configured_receipt or Path(str(runner) + ".build.json")
        command = [
            self.config.python, str(wrapper),
            "--source-dir", str(self.config.source_dir),
            "--pool", str(self.config.pool),
            "--receipt", str(self.config.receipt),
            "--runner", str(runner),
            "--build-receipt", str(build_receipt),
            "--kv-compat-sha256", self.compatibility.hexdigest,
        ]
        return command

    def start_persistent(self, *, session_epoch: int | None = None) -> None:
        if self.engine is not None:
            raise BackendError("persistent Gemma engine already started")
        epoch = session_epoch if session_epoch is not None else max(
            self.session_epoch, 1,
        )
        if epoch < 1:
            raise BackendError("persistent Gemma session epoch must be positive")
        build_receipt = self.config.server_build_receipt or Path(
            str(self.config.server_runner) + ".build.json"
        )
        try:
            build_identity = _build_module.load_build_receipt(
                build_receipt, self.config.server_runner,
                expected_source_compatibility=self.compatibility.hexdigest,
            )
        except _build_module.BuildIdentityError as exc:
            raise BackendError(f"Gemma 4 server build identity failed: {exc}") from exc
        selected_shared_kv = self.config.shared_kv
        if self.config.mentor_root is not None:
            assert self.config.mentor_anchor is not None
            try:
                store = _mentor_module.MentorAnchorStore(
                    self.config.mentor_root,
                    compatibility_sha256=KV_COMPATIBILITY_SHA256,
                    model_id=self.config.model_id,
                    quant_track="mlx-affine-q4",
                )
                selected = store.resolve(self.config.mentor_anchor)
            except _mentor_module.MentorAnchorError as exc:
                raise BackendError(f"Gemma 4 mentor anchor failed: {exc}") from exc
            if selected.context_tokens != self.config.context_tokens:
                raise BackendError("Gemma 4 mentor anchor context mismatch")
            if (self.mentor_anchor_metadata is not None and
                    self.mentor_anchor_metadata.selector != selected.selector):
                raise BackendError("Gemma 4 mentor anchor changed across restart")
            self.mentor_anchor_metadata = selected
            selected_shared_kv = selected.state_path
        initial_input = None
        inherited_fds: tuple[int, ...] = ()
        if self.authentication is None:
            command = [
                self.config.python, str(self.config.multimodal_wrapper),
                "--source-dir", str(self.config.source_dir),
                "--pool", str(self.config.pool),
                "--receipt", str(self.config.receipt),
                "--runner", str(self.config.server_runner),
                "--build-receipt", str(build_receipt),
                "--kv-compat-sha256", self.compatibility.hexdigest,
                "--serve-v2", "--ctx", str(self.config.context_tokens),
                "--gen", str(self.config.max_output_tokens),
                "--workers", str(self.config.workers),
                "--session-epoch", str(epoch), "--stream-events",
            ]
        else:
            self.authentication.validate(self.config, build_identity.hexdigest)
            command = [
                str(self.config.server_runner), "--serve-v2",
                "--ctx", str(self.config.context_tokens),
                "--gen", str(self.config.max_output_tokens),
                "--workers", str(self.config.workers),
                "--session-epoch", str(epoch), "--stream-events",
            ]
            initial_input = self.authentication.binding
            inherited_fds = self.authentication.inherited_fds
        if selected_shared_kv is not None:
            command.extend(("--shared-kv", str(selected_shared_kv)))
        if self.config.proof_state:
            command.append("--proof-state")
        if self.config.dpr_mode != "off":
            assert self.config.dpr_root is not None
            command.extend((
                "--dpr-root", str(self.config.dpr_root),
                "--dpr-mode", self.config.dpr_mode,
                "--dpr-current-kv-bytes",
                str(gemma4_kv_capacity_bytes(
                    self.config.context_tokens, self.config.kv_layout,
                )),
                "--dpr-serial-ns",
                str(int(self.config.dpr_serial_ms_per_token * 1_000_000)),
                "--dpr-retention-bytes",
                str(int(self.config.dpr_retention_gb * 1_000_000_000)),
            ))
            if self.config.dpr_mode == "persist":
                command.extend((
                    "--dpr-additive-bytes",
                    str(int(self.config.dpr_budget_gb * 1_000_000_000)),
                ))
            if self.config.dpr_mentor_enabled:
                command.extend((
                    "--dpr-mentor-policy-sha256",
                    GEMMA_DPR_MENTOR_POLICY_SHA256,
                ))
            if self.config.dpr_mindset_attention_enabled:
                command.extend((
                    "--dpr-attention-policy-sha256",
                    GEMMA_DPR_MINDSET_ATTENTION_POLICY_SHA256,
                ))
        try:
            engine_env = self._clean_env()
            self.engine = PersistentGemmaEngine(
                command, timeout=self.config.timeout_s, env=engine_env,
                compatibility_sha256=self.compatibility.hexdigest,
                build_identity_sha256=build_identity.hexdigest,
                expected_context=self.config.context_tokens,
                expected_output_limit=self.config.max_output_tokens,
                expected_prefill_chunk=self.config.prefill_chunk_tokens,
                expected_kv_capacity_bytes=gemma4_kv_capacity_bytes(
                    self.config.context_tokens, self.config.kv_layout,
                ),
                dpr_mode=self.config.dpr_mode,
                initial_input=initial_input, pass_fds=inherited_fds,
            )
            self.build_identity_sha256 = build_identity.hexdigest
            if self.config.kv_cache_root is not None:
                self.cache_store = KVCacheStore(
                    self.config.kv_cache_root,
                    KV_COMPATIBILITY_SHA256,
                    bytes.fromhex(build_identity.hexdigest),
                )
            self.session_epoch = self.engine.session_epoch
            self.turn_count = self.engine.turn_id
            self.history_turns = self.engine.history_turns
            self.session_position = self.engine.position
            self.shared_position = self.engine.shared_position
            if (self.mentor_anchor_metadata is not None and
                    self.shared_position != self.mentor_anchor_metadata.position):
                self.engine.stop(force=True)
                self.engine = None
                raise BackendError("persistent mentor anchor position drift")
            self.state_sha256 = self.engine.state_sha256
            self.mindset_mode = self.engine.mindset_mode
            self.mindset_end = self.engine.mindset_end
            self.mindset_sha256 = self.engine.mindset_sha256
            self.facts_sha256 = self.engine.facts_sha256
            self.facts_rows = self.engine.facts_rows
        except OSError as exc:
            raise BackendError("could not start persistent Gemma engine") from exc

    def stop_persistent(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine = None

    def export_kv(self, path: Path) -> dict:
        if self.engine is None:
            raise BackendError("persistent Gemma engine has not started")
        result = self.engine.export_kv(path)
        self.state_sha256 = str(result["state_sha256"])
        self.facts_sha256 = str(result["facts_sha256"])
        return result

# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    def _engage_cache_store(self) -> KVCacheStore:
        if self.engine is None:
            raise RequestError("persistent Gemma engine has not started")
        if self.cache_store is None:
            if (re.fullmatch(r"[0-9a-f]{64}", self.build_identity_sha256) is None or
                    self.build_identity_sha256 == "0" * 64):
                raise BackendError("persistent Gemma build identity is unavailable")
            root = self.config.kv_cache_root or DEFAULT_RUNTIME_KV_CACHE_ROOT
            self.cache_store = KVCacheStore(
                root, KV_COMPATIBILITY_SHA256,
                bytes.fromhex(self.build_identity_sha256),
            )
        return self.cache_store
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    def export_cache(self, cache_id: str, *, replace_existing: bool = False) -> dict:
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        self._engage_cache_store()
        """
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        if self.engine is None or self.cache_store is None:
            raise RequestError("persistent KV export is not configured")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        """
        assert self.engine is not None
        assert self.cache_store is not None
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        request = {
            "mode": (
                "resume" if replace_existing and self.cache_store.contains(cache_id)
                else "create"
            ),
            "id": cache_id,
        }
        controls = Controls(
            self.config.context_tokens, self.config.max_output_tokens,
            "none", False, self.config.kv_budget_gb,
            gemma4_kv_capacity_bytes(
                self.config.context_tokens, self.config.kv_layout,
            ),
        )
        with self.cache_store.lock(request):
            selection = self.cache_store.resolve(request, controls)
            assert selection.save_path is not None
            try:
                native = self.export_kv(selection.save_path)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
                self.cache_store._export_provenance = {
                    "sampler_abi": getattr(self.engine, "sampler_abi", None),
                    "temperature_bits": getattr(self.engine, "temperature_bits", None),
                    "seed": getattr(self.engine, "sampler_seed", None),
                    "top_k": getattr(self.engine, "sampler_top_k", None),
                }
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
                metadata = self.cache_store.publish(
                    selection, int(native["position"]),
                )
            except BaseException:
                self.cache_store.discard(selection)
                raise
        return {
            "object": "salt.kv_cache",
            **metadata.as_dict(),
            "state_sha256": native["state_sha256"],
            "facts_sha256": native["facts_sha256"],
            "session_manifest": {
                "turn_id": self.turn_count,
                "history_turns": self.history_turns,
                "mindset_end": self.mindset_end,
                "history": [
                    {"role": role, "content": content, "image_url": image_url}
                    for role, content, image_url in self.session_history
                ],
                "turns": list(self.session_turns),
            },
        }

    def import_session_cache(self, cache_id: str, manifest: object) -> dict:
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        self._engage_cache_store()
        """
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        if self.engine is None or self.cache_store is None:
            raise RequestError("persistent KV import is not configured")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        """
        assert self.engine is not None
        assert self.cache_store is not None
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        if self.session_position != self.mindset_end:
            raise RequestError("persistent session must be clear before import")
        if not isinstance(manifest, dict):
            raise RequestError("session manifest must be an object")
        if set(manifest) != {
                "turn_id", "history_turns", "mindset_end", "history",
                "turns", "state_sha256", "facts_sha256"}:
            raise RequestError("invalid session manifest fields")
        turn_id = manifest["turn_id"]
        history_turns = manifest["history_turns"]
        mindset_end = manifest["mindset_end"]
        raw_history = manifest["history"]
        raw_turns = manifest["turns"]
        state_sha256 = manifest["state_sha256"]
        facts_sha256 = manifest["facts_sha256"]
        if (not isinstance(turn_id, int) or not isinstance(history_turns, int) or
                turn_id < 0 or history_turns < 0 or history_turns > turn_id or
                not isinstance(mindset_end, int) or mindset_end != self.mindset_end or
                not isinstance(raw_history, list) or not isinstance(raw_turns, list) or
                re.fullmatch(r"[0-9a-f]{64}", state_sha256) is None or
                re.fullmatch(r"[0-9a-f]{64}", facts_sha256) is None):
            raise RequestError("invalid session manifest values")
        history: list[tuple[str, str, str | None]] = []
        for item in raw_history:
            if not isinstance(item, dict) or set(item) != {
                    "role", "content", "image_url"}:
                raise RequestError("invalid session history item")
            role = item["role"]
            content = item["content"]
            image_url = item["image_url"]
            if (role not in ("user", "assistant") or not isinstance(content, str) or
                    (image_url is not None and not isinstance(image_url, str))):
                raise RequestError("invalid session history value")
            history.append((role, content, image_url))
        if len(history) != history_turns * 2:
            raise RequestError("session history/turn count mismatch")
        if len(raw_turns) != history_turns or any(
                not isinstance(item, dict) for item in raw_turns):
            raise RequestError("session turn provenance/count mismatch")
        previous_turn = -1
        turns: list[dict[str, object]] = []
        for item in raw_turns:
            assert isinstance(item, dict)
            if set(item) != {
                    "turn_id", "prompt_ids", "output_ids", "response_sha256",
                    "content", "output_limit"}:
                raise RequestError("invalid session turn provenance fields")
            item_turn = item["turn_id"]
            prompt_ids = item["prompt_ids"]
            output_ids = item["output_ids"]
            if (not isinstance(item_turn, int) or item_turn <= previous_turn or
                    not isinstance(prompt_ids, list) or
                    not isinstance(output_ids, list) or
                    not prompt_ids or not output_ids or
                    len(prompt_ids) > self.config.context_tokens or
                    len(output_ids) > self.config.max_output_tokens or
                    not isinstance(item["output_limit"], int) or
                    not 1 <= item["output_limit"] <= self.config.max_output_tokens or
                    any(not isinstance(token, int) or token < 0
                        for token in [*prompt_ids, *output_ids]) or
                    not isinstance(item["content"], str) or
                    not isinstance(item["response_sha256"], str) or
                    re.fullmatch(r"[0-9a-f]{64}", item["response_sha256"]) is None):
                raise RequestError("invalid session turn provenance values")
            previous_turn = item_turn
            turns.append(dict(item))
        if turns and turns[-1]["turn_id"] != turn_id:
            raise RequestError("session turn provenance terminal turn mismatch")
        metadata = self.cache_store.load_metadata(cache_id)
        if (metadata.context_tokens != self.config.context_tokens or
                metadata.logical_state_sha256 != state_sha256):
            raise RequestError("session cache metadata drift")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        with self.cache_store.native_path(metadata) as import_path:
            native = self.engine.import_session_kv(
                import_path, turn_id=turn_id, history_turns=history_turns,
                expected_position=metadata.position,
                expected_state_sha256=state_sha256,
                expected_facts_sha256=facts_sha256,
            )
        """
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        native = self.engine.import_session_kv(
            metadata.path, turn_id=turn_id, history_turns=history_turns,
            expected_position=metadata.position,
            expected_state_sha256=state_sha256,
            expected_facts_sha256=facts_sha256,
        )
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        """
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        self.turn_count = int(native["turn_id"])
        self.history_turns = int(native["history_turns"])
        self.session_position = int(native["position"])
        self.state_sha256 = str(native["state_sha256"])
        self.facts_sha256 = str(native["facts_sha256"])
        self.facts_rows = self.session_position - self.mindset_end
        self.session_history = history
        self.session_turns = turns
        self.last_user_signature = next(
            (item for item in reversed(history) if item[0] == "user"), None,
        )
        self.committed_requests.clear()
        self.last_committed_request_id = None
        self.transport_records.clear()
        return self.session_status()

    def clear_facts(self) -> dict:
        if self.engine is None:
            raise BackendError("persistent Gemma engine has not started")
        result = self.engine.clear_facts()
        self.session_position = int(result["position"])
        self.state_sha256 = str(result["state_sha256"])
        self.mindset_mode = str(result["mindset_mode"])
        self.mindset_end = int(result["mindset_end"])
        self.mindset_sha256 = str(result["mindset_sha256"])
        self.facts_sha256 = str(result["facts_sha256"])
        self.facts_rows = int(result["facts_rows"])
        self.history_turns = int(result["history_turns"])
        self.session_history = []
        self.session_turns = []
        self.last_user_signature = None
        self.committed_requests.clear()
        self.last_committed_request_id = None
        self.transport_records.clear()
        return self.session_status()

    def reset_persistent(self) -> dict:
        return self.clear_facts()

    def hard_reset(self) -> dict:
        old_epoch = self.session_epoch
        if old_epoch >= (1 << 64) - 1:
            raise BackendError("persistent Gemma session epoch exhausted")
        old_mindset_sha256 = self.mindset_sha256
        old_build_identity_sha256 = self.build_identity_sha256
        self.stop_persistent()
        self.turn_count = 0
        self.history_turns = 0
        self.session_position = 0
        self.shared_position = 0
        self.state_sha256 = ""
        self.facts_sha256 = ""
        self.facts_rows = 0
        self.session_history = []
        self.session_turns = []
        self.last_user_signature = None
        self.committed_requests.clear()
        self.last_committed_request_id = None
        self.transport_records.clear()
        self.session_epoch = old_epoch + 1
        self.start_persistent(session_epoch=self.session_epoch)
        if (old_mindset_sha256 and
                self.mindset_sha256 != old_mindset_sha256):
            self.stop_persistent()
            raise BackendError("hard reset changed mandatory mindset state")
        if (old_build_identity_sha256 and
                self.build_identity_sha256 != old_build_identity_sha256):
            self.stop_persistent()
            raise BackendError("hard reset changed native build identity")
        return self.session_status()

    def reconfigure_dpr(self, control: object) -> dict:
        if self.engine is None or self.session_position != self.mindset_end or \
                self.history_turns != 0:
            raise RequestError("DPR reconfiguration requires a clear READY session")
        if not isinstance(control, dict) or "mode" not in control:
            raise RequestError("invalid DPR reconfiguration")
        mode = control["mode"]
        if mode == "off":
            if set(control) != {"mode"}:
                raise RequestError("DPR off accepts only mode")
            new_config = replace(
                self.config, dpr_root=None, dpr_mode="off", dpr_budget_gb=0.0,
                dpr_serial_ms_per_token=0.0, dpr_mentor_enabled=False,
                dpr_mindset_attention_enabled=False,
            )
        else:
            required = {
                "mode", "root", "budget_gb", "serial_ms_per_token",
                "retention_gb",
            }
            if set(control) != required or mode not in ("persist", "dynamic"):
                raise RequestError("invalid DPR attachment fields")
            root = control["root"]
            if not isinstance(root, str) or not root:
                raise RequestError("invalid DPR collection root")
            values = []
            for key in ("budget_gb", "serial_ms_per_token", "retention_gb"):
                value = control[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RequestError(f"invalid DPR {key}")
                value = float(value)
                if not math.isfinite(value) or value <= 0.0:
                    raise RequestError(f"invalid DPR {key}")
                values.append(value)
            new_config = replace(
                self.config, dpr_root=Path(root), dpr_mode=mode,
                dpr_budget_gb=values[0],
                dpr_serial_ms_per_token=values[1],
                dpr_retention_gb=values[2],
            )
        old_build = self.build_identity_sha256
        old_mindset = self.mindset_sha256
        if self.session_epoch >= (1 << 64) - 1:
            raise BackendError("persistent Gemma session epoch exhausted")
        self.stop_persistent()
        self.config = new_config
        self.session_epoch += 1
        self.turn_count = 0
        self.history_turns = 0
        self.session_position = 0
        self.shared_position = 0
        self.state_sha256 = ""
        self.facts_sha256 = ""
        self.facts_rows = 0
        self.session_history = []
        self.session_turns = []
        self.last_user_signature = None
        self.committed_requests.clear()
        self.last_committed_request_id = None
        self.transport_records.clear()
        self.start_persistent(session_epoch=self.session_epoch)
        if old_build and self.build_identity_sha256 != old_build:
            self.stop_persistent()
            raise BackendError("DPR reconfiguration changed native build identity")
        if old_mindset and self.mindset_sha256 != old_mindset:
            self.stop_persistent()
            raise BackendError("DPR reconfiguration changed mandatory mindset")
        return self.session_status()

    def recover_if_dead(self) -> bool:
        """Restart one failed native epoch before admitting the next request."""
        with self.lock:
            if self.engine is None or self.engine.phase != "DEAD":
                return False
            self.hard_reset()
            return True

    def session_status(self) -> dict:
        if (self.engine is not None and self.engine.process.poll() is not None and
                self.engine.phase not in ("CLOSED", "DEAD")):
            self.engine.phase = "DEAD"
            self.engine.active_request = None
        engine_running = (
            self.engine is not None and self.engine.process.poll() is None and
            not self.engine.closed
        )
        phase = self.engine.phase if self.engine is not None else "DOWN"
        return {
            "object": "salt.session",
            "mode": "persistent-live-kv",
            "engine_running": engine_running,
            "turns": self.turn_count,
            "history_turns": self.history_turns,
            "session_epoch": self.session_epoch,
            "position": self.session_position,
            "shared_position": self.shared_position,
            "state_sha256": self.state_sha256,
            "build_identity_sha256": self.build_identity_sha256,
            "mindset_mode": self.mindset_mode,
            "mindset_end": self.mindset_end,
            "mindset_sha256": self.mindset_sha256,
            "mandatory_state_bytes": gemma4_kv_state_bytes(self.mindset_end),
            "facts_optional": True,
            "facts_sha256": self.facts_sha256,
            "facts_rows": self.facts_rows,
            "facts_bytes": (
                gemma4_kv_capacity_bytes(self.facts_rows)
                if self.facts_rows else 0
            ),
            "phase": phase,
            "active_request": (
                self.engine.active_request if self.engine is not None else None
            ),
            "last_committed_request_id": self.last_committed_request_id,
            "journal_entries": len(self.committed_requests),
            "recovery_required": phase == "DEAD" or not engine_running,
            "context_limit": self.config.context_tokens,
            "output_reserve": self.config.max_output_tokens,
            "remaining_tokens": max(
                0, self.config.context_tokens - self.session_position,
            ),
            "mentor_state": self._mentor_public(),
            "dpr": self._dpr_public(),
            "runtime_ready": RUNTIME_READY,
        }

    @staticmethod
    def _history_signature(messages: object) -> list[tuple[str, str, str | None]]:
        if not isinstance(messages, list) or not messages:
            raise RequestError("messages required")
        signature: list[tuple[str, str, str | None]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise RequestError("messages must be objects")
            role = message.get("role")
            allowed = {"role", "content"}
            if role == "assistant":
                allowed.add("reasoning_content")
            unknown = set(message) - allowed
            if unknown:
                raise RequestError(
                    "unsupported message fields: " + ", ".join(sorted(unknown))
                )
            if role not in ("system", "developer", "user", "assistant"):
                raise RequestError(f"unsupported message role: {role!r}")
            text, image_url = _text_and_image_parts(
                message.get("content", ""), role,
            )
            if role != "assistant" and not text:
                raise RequestError(f"{role} message has no supported content")
            signature.append((role, text, image_url))
        return signature

    @staticmethod
    def _clean_env() -> dict[str, str]:
        # The local runners need no provider credentials. Keep only process
        # essentials so API keys/tokens cannot leak into child environments.
        allowed = (
            "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
        )
        environment = {
            key: os.environ[key] for key in allowed if key in os.environ
        }
        engine_config = dict(GEMMA_ENGINE_CONFIG)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        stderr_mode = os.environ.get("SALT_SERVER_STDERR", "summary")
        if stderr_mode not in ("summary", "diagnostic", "waterfall"):
            raise BackendError(
                "SALT_SERVER_STDERR must be summary, diagnostic, or waterfall"
            )
        meter_keys = (
            "SALT_WATERFALL",
            "SALT_TARGET_WATERFALL",
            "SALT_GPU_DIAG",
            "SALT_HIP_TEXT_CELL_TIMING",
            "SALT_HIP_PATH_SUMMARY",
            "SALT_FFN_DETAIL",
        )
        for key in meter_keys:
            environment.pop(key, None)
        environment["SALT_SERVER_STDERR"] = stderr_mode
        if stderr_mode == "waterfall":
            for key in meter_keys:
                environment[key] = "1"
        server_target_route = {
            "SALT_TARGET_ROUTE_N": os.environ.get("GEMMA4_TARGET_ROUTE_N"),
            "SALT_TARGET_X": os.environ.get("GEMMA4_TARGET_X"),
            "SALT_TARGET_ROUTE_F": os.environ.get("GEMMA4_TARGET_ROUTE_F"),
            "SALT_TARGET_ROUTE_Q": os.environ.get("GEMMA4_TARGET_ROUTE_Q"),
        }
        if any(value is not None for value in server_target_route.values()):
            try:
                target_policy = validate_gemma_target_route_config(
                    server_target_route,
                )
            except ValueError as exc:
                raise BackendError(
                    "invalid Gemma normal-server target route",
                ) from exc
            if not target_policy["configured"]:
                raise BackendError("incomplete Gemma normal-server target route")
            engine_config.update(server_target_route)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        retain_override = os.environ.get("SALT_PREFILL_RETAIN_LAYERS")
        lookahead_override = os.environ.get("SALT_PREFILL_DECODE_LOOKAHEAD")
        if retain_override is not None:
            if retain_override not in ("0", "1"):
                raise BackendError("SALT_PREFILL_RETAIN_LAYERS must be 0 or 1")
            engine_config["SALT_PREFILL_RETAIN_LAYERS"] = retain_override
        if lookahead_override is not None:
            if lookahead_override not in ("0", "1"):
                raise BackendError(
                    "SALT_PREFILL_DECODE_LOOKAHEAD must be 0 or 1"
                )
            engine_config["SALT_PREFILL_DECODE_LOOKAHEAD"] = lookahead_override
        for key, value in engine_config.items():
            if key.startswith("SALT_"):
                environment[key] = value
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        try:
            _target_warmup_config(engine_config)
        except ValueError as exc:
            raise BackendError(str(exc)) from exc
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        policy = validate_gemma_residency_config(engine_config)
        if policy["layer_quotas"] is not None:
            environment["SALT_CACHE_LAYER_QUOTAS"] = policy["layer_quotas"]
        return environment

    def validate_setup(self, *, authenticate: bool = True) -> None:
        try:
            validate_gemma_residency_config(
                GEMMA_ENGINE_CONFIG, require_host=True,
            )
        except ValueError as exc:
            raise BackendError(str(exc)) from exc
        paths = {
            "Gemma source directory": (self.config.source_dir, True),
            "Gemma expert pool": (self.config.pool, False),
            "Gemma import receipt": (self.config.receipt, False),
            "Gemma KV compatibility manifest": (KV_COMPAT_MANIFEST, False),
            "Gemma text wrapper": (self.config.text_wrapper, False),
            "Gemma multimodal wrapper": (self.config.multimodal_wrapper, False),
            "Gemma text runner": (self.config.text_runner, False),
            "Gemma multimodal runner": (self.config.multimodal_runner, False),
            "Gemma persistent server runner": (self.config.server_runner, False),
        }
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        if not authenticate:
            paths.pop("Gemma text runner")
            paths.pop("Gemma multimodal runner")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        for label, (path, directory) in paths.items():
            exists = path.is_dir() if directory else path.is_file()
            if not exists:
                raise BackendError(f"{label} does not exist: {path}")
        if not authenticate:
            return
        verification_image = (
            ROOT / "models" / "gemma4-26b-a4b" / "qualification" /
            "red-mug-384.ppm"
        )
        if not verification_image.is_file():
            raise BackendError(
                f"Gemma 4 verification image does not exist: {verification_image}"
            )
        command = self._base_command("image") + [
            "--image", str(verification_image), "--verify-only",
        ]
        try:
            completed = _run_process_group(
                command, timeout=self.config.timeout_s, env=self._clean_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError("Gemma 4 startup authentication timed out") from exc
        except OSError as exc:
            raise BackendError("could not start Gemma 4 authenticator") from exc
        if completed.returncode != 0 or \
                b"GEMMA4_MULTIMODAL_BINDING_OK" not in completed.stdout or \
                _HEX_RUNTIME_TRUE not in completed.stdout or \
                _HEX_RUNTIME_FALSE in completed.stdout + completed.stderr:
            tail = completed.stderr[-2048:].decode("utf-8", errors="replace")
            raise BackendError(
                f"Gemma 4 startup authentication failed (exit "
                f"{completed.returncode}): {tail}"
            )

    def capabilities(self) -> dict:
        kv_bytes = gemma4_kv_capacity_bytes(
            self.config.context_tokens, self.config.kv_layout,
        )
        gates = [{
            "context_tokens": self.config.context_tokens,
            "max_output_tokens": self.config.max_output_tokens,
            "prefill_chunk_tokens": self.config.prefill_chunk_tokens,
            "kv_bytes": kv_bytes,
            "kv_gb": kv_bytes / 1_000_000_000,
        }]
        return {
            "runtime_ready": RUNTIME_READY,
            "backend": "authenticated-persistent-native-session",
            "modalities": ["text", "image"],
            "image_contract": "one canonical no-resize P6 PPM",
            "reasoning_efforts": list(EFFORT_OUTPUT),
            "reasoning_trace_field": "reasoning_content",
            "decode": "greedy",
            "streaming": "live-sse",
            "sessions": "one-live-kv-session-per-launch",
            "private_protocol": "gemma4-persistent-v3",
            "idempotency": {
                "header": "Idempotency-Key",
                "automatic_immediate_retry": True,
                "native_journal_entries": 256,
                "replay_mutates_state": False,
            },
            "failure_recovery": "dead-until-explicit-reset",
            "state_classes": {
                "mindset_mode": "prefix",
                "mindset_end": 0,
                "separate_delta_state_qualified": False,
                "facts": "optional-live-kv",
                "clear_facts": "in-place-scrub",
                "hard_reset_http": False,
            },
            "cold_start_per_request": False,
            "kv_cache": {
                "enabled": self.cache_store is not None,
                "modes": ["export"] if self.cache_store is not None else [],
                "selector": "live-ram-session",
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
                "enabled": True,
                "engaged": self.cache_store is not None,
                "modes": ["export", "import"],
                "selector": "server-owned-runtime-store",
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
                "portable_format": "G4KVC006",
                "legacy_read_formats": ["G4KVC005", "G4KVC004"],
                "legacy_read_compatibility_sha256": {
                    "G4KVC005": KV_LEGACY_V5_COMPATIBILITY_SHA256.hex(),
                    "G4KVC004": KV_LEGACY_V4_COMPATIBILITY_SHA256.hex(),
                },
                "compatibility_sha256": self.compatibility.hexdigest,
                "build_identity_sha256": self.build_identity_sha256 or None,
                "namespace": (self.cache_store.root.name
                              if self.cache_store is not None else None),
            },
            "mentor_state": self._mentor_public(),
            "dpr": self._dpr_public(),
            "context_management": {
                "model_max_tokens": GEMMA4_MODEL_MAX_CONTEXT,
                "server_max_tokens": self.config.context_tokens,
                "request_rule": "prompt plus output plus pending <= server max",
                "truncation": False,
                "prefill_chunk_tokens": self.config.prefill_chunk_tokens,
            },
            "kv_layout": {
                "mode": self.config.kv_layout,
                "sliding_layers": GEMMA4_SLIDING_LAYERS,
                "sliding_window": GEMMA4_SLIDING_WINDOW,
                "full_attention_layers": GEMMA4_FULL_LAYERS,
                "capacity_bytes": kv_bytes,
                "marginal_bytes_per_token_after_window": (
                    GEMMA4_FULL_LAYERS * GEMMA4_FULL_KV_BYTES
                ),
            },
            "kv_budget_gb": self.config.kv_budget_gb,
            "gates": gates,
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
            "engine_config": {
                "platform_recipe": GEMMA_ENGINE_CONFIG.get(
                    "SALT_GEMMA_PLATFORM_RECIPE"
                ),
                "compute_node": GEMMA_ENGINE_CONFIG.get(
                    "SALT_GEMMA_COMPUTE_NODE"
                ),
                "nfq": {
                    key: value for key, value in
                    validate_gemma_target_route_config(GEMMA_ENGINE_CONFIG).items()
                    if key != "x"
                },
                "target_x": validate_gemma_target_route_config(
                    GEMMA_ENGINE_CONFIG
                )["x"],
                "target_kv_warmup_rows": _target_warmup_config(
                    GEMMA_ENGINE_CONFIG
                )["kv_warmup_rows"],
                "target_warm_x": _target_warmup_config(
                    GEMMA_ENGINE_CONFIG
                )["warm_x"],
                "target_area_workers": int(GEMMA_ENGINE_CONFIG[
                    "SALT_TARGET_AREA_WORKERS"
                ]),
                "expert_budget_gb": GEMMA_EXPERT_BUDGET_GB,
                "memory_limit_gb": float(GEMMA_ENGINE_CONFIG[
                    "SALT_GEMMA_MEMORY_LIMIT_GB"
                ]),
                "target_gpu_program": (
                    GEMMA_ENGINE_CONFIG.get("SALT_TARGET_GPU_PROGRAM") == "1"
                ),
                "target_cpu_graph": (
                    GEMMA_ENGINE_CONFIG.get("SALT_TARGET_CPU_GRAPH") == "1"
                ),
            },
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        }

    @staticmethod
    def _validate_request_schema(body: dict, *, endpoint: str) -> None:
        if not isinstance(body, dict):
            raise RequestError("request body must be an object")
        common = {
            "model", "stream", "max_output_tokens", "reasoning",
            "kv_allowance_gb", "kv_cache", "temperature", "top_p",
            "frequency_penalty", "presence_penalty", "seed", "n", "stop",
            "logprobs", "tools", "truncation", "session", "salt_proof_state",
        }
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        common.add("top_k")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        if endpoint == "chat":
            allowed = common | {
                "messages", "max_tokens", "max_completion_tokens",
                "reasoning_effort", "modalities", "audio", "response_format",
                "stream_options",
            }
            label = "Chat"
            required = "messages"
        elif endpoint == "responses":
            allowed = common | {"input", "instructions"}
            label = "Responses"
            required = "input"
        else:
            raise BackendError(f"invalid API endpoint {endpoint!r}")
        unknown = set(body) - allowed
        if unknown:
            raise RequestError(
                f"unsupported {label} fields: " + ", ".join(sorted(unknown))
            )
        if required not in body:
            raise RequestError(f"{label} request requires {required}")
        model = body.get("model")
        if model is not None and not isinstance(model, str):
            raise RequestError("model must be a string")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise RequestError("stream must be a boolean")
        stream_options = body.get("stream_options")
        if stream_options is not None:
            if endpoint != "chat" or not stream:
                raise RequestError("stream_options requires Chat stream=true")
            if not isinstance(stream_options, dict):
                raise RequestError("stream_options must be an object")
            unknown_options = set(stream_options) - {"include_usage"}
            if unknown_options:
                raise RequestError(
                    "unsupported stream_options fields: " +
                    ", ".join(sorted(unknown_options))
                )
            include_usage = stream_options.get("include_usage")
            if include_usage is not None and not isinstance(include_usage, bool):
                raise RequestError("stream_options.include_usage must be a boolean")
        reasoning = body.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise RequestError("reasoning must be an object")
            nested_unknown = set(reasoning) - {"effort"}
            if nested_unknown:
                raise RequestError(
                    "unsupported reasoning fields: " +
                    ", ".join(sorted(nested_unknown))
                )
        Gemma4Backend._validate_sampling(body)

    @staticmethod
    def _validate_sampling(body: dict) -> None:
        def exact_number(field: str, expected: float) -> None:
            value = body.get(field)
            if value is None:
                return
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RequestError(f"{field} must be a number")
            try:
                parsed = float(value)
            except OverflowError as exc:
                raise RequestError(f"{field} must be finite") from exc
            if not math.isfinite(parsed) or parsed != expected:
                raise RequestError(
                    f"{field}={expected:g} is required for qualified greedy decode"
                )

# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        if "temperature" in body or "seed" in body:
            sampler = _sampler_controls(body)
            if sampler[1] == 2:
                remaining = dict(body)
                remaining.pop("temperature", None)
                remaining.pop("seed", None)
                remaining.pop("top_k", None)
                Gemma4Backend._validate_sampling(remaining)
                return
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        exact_number("temperature", 0.0)
        exact_number("top_p", 1.0)
        exact_number("frequency_penalty", 0.0)
        exact_number("presence_penalty", 0.0)
        if body.get("seed") is not None:
            raise RequestError("seed is not supported by the greedy runner")
        n = body.get("n")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n != 1):
            raise RequestError("n must be integer 1")
        stop = body.get("stop")
        if stop is not None and not (
                (isinstance(stop, str) and stop == "") or
                (isinstance(stop, list) and not stop)):
            raise RequestError("custom stop sequences are not supported")
        logprobs = body.get("logprobs")
        if logprobs is not None and logprobs is not False:
            raise RequestError("logprobs must be boolean false")
        tools = body.get("tools")
        if tools is not None and (not isinstance(tools, list) or tools):
            raise RequestError("Gemma 4 tool calls are not server-qualified")
        modalities = body.get("modalities")
        if modalities is not None and modalities != ["text"]:
            raise RequestError("output modalities other than text are not supported")
        if body.get("audio") is not None:
            raise RequestError("audio output is not supported")
        response_format = body.get("response_format")
        if response_format is not None and response_format != {"type": "text"}:
            raise RequestError("structured response formats are not supported")
        truncation = body.get("truncation")
        if truncation is not None and truncation != "disabled":
            raise RequestError("automatic truncation is not supported")
        if body.get("session") is not None:
            raise RequestError("server-managed sessions are not supported")

    def _run_completion(
            self, body: dict, controls: Controls,
            selection: KVCacheSelection, *,
            on_stream_start: Callable[[], None] | None = None,
            on_stream_delta: Callable[[str, str], None] | None = None,
    ) -> Completion:
        if (on_stream_start is None) != (on_stream_delta is None):
            raise BackendError("incomplete live-stream callback pair")
        rendered = render_chat(
            body.get("messages"),
            enable_thinking=controls.enable_thinking,
            continuation=selection.load_path is not None,
        )
        modality = "image" if rendered.image_url is not None else "text"
        image_context = (
            prepared_image(rendered.image_url, self.config.image_root)
            if rendered.image_url is not None else nullcontext(None)
        )
        prompt_path: Path | None = None
        t0 = time.monotonic()
        try:
            with _private_prompt_file(rendered.prompt) as stream:
                stream.write(rendered.prompt)
                stream.flush()
                os.fsync(stream.fileno())
                prompt_path = Path(stream.name)
            with image_context as image:
                command = self._base_command(modality) + [
                    "--ctx", str(controls.context_tokens),
                    "--gen", str(controls.output_tokens),
                    "--workers", str(self.config.workers),
                    "--prompt-file", str(prompt_path),
                ]
                if image is not None:
                    command += ["--image", str(image.path)]
                if selection.load_path is not None:
                    command += ["--kv-load", str(selection.load_path)]
                if selection.save_path is not None:
                    command += ["--kv-save", str(selection.save_path)]
                accumulator = None
                if on_stream_delta is not None:
                    command.append("--stream-events")
                    marker = "GEMMA4_MM" if modality == "image" else "GEMMA4_QA"
                    accumulator = NativeStreamAccumulator(marker, on_stream_delta)
                try:
                    if accumulator is None:
                        completed = _run_process_group(
                            command, timeout=self.config.timeout_s,
                            env=self._clean_env(),
                        )
                    else:
                        assert on_stream_start is not None
                        completed = _run_process_group_stream(
                            command, timeout=self.config.timeout_s,
                            env=self._clean_env(), accumulator=accumulator,
                            on_start=on_stream_start,
                        )
                except subprocess.TimeoutExpired as exc:
                    raise BackendError("Gemma 4 request timed out") from exc
                except OSError as exc:
                    raise BackendError("could not start Gemma 4 runner") from exc
        finally:
            if prompt_path is not None:
                try:
                    prompt_path.unlink()
                except OSError:
                    pass
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace")
            if "prompt/generation exceed context" in error or \
                    "requires unimplemented bicubic resize" in error or \
                    "invalid Gemma 4 P6 image" in error or \
                    "KV cache" in error or "completed turn" in error:
                raise RequestError(error.strip().splitlines()[-1])
            raise BackendError(
                f"Gemma 4 runner failed (exit {completed.returncode}): "
                f"{error[-2048:]}"
            )
        result = parse_runner_output(
            completed.stdout, completed.stderr, modality, controls.output_tokens,
        )
        if accumulator is not None:
            accumulator.finish(result)
        if result.kv_loaded_tokens != selection.loaded_tokens:
            raise BackendError("native KV cache load position drift")
        if selection.save_path is None:
            if result.kv_saved_tokens != 0:
                raise BackendError("native runner unexpectedly saved a KV cache")
        else:
            expected_saved = (
                selection.loaded_tokens + result.prompt_tokens +
                result.completion_tokens
            )
            if result.kv_saved_tokens != expected_saved:
                raise BackendError("native KV cache save position drift")
        effective_prompt_tokens = selection.loaded_tokens + result.prompt_tokens
        if effective_prompt_tokens + controls.output_tokens > controls.context_tokens:
            raise BackendError("native runner admitted prompt beyond selected context gate")
        return Completion(
            result=result,
            controls=controls,
            modality=modality,
            elapsed_s=time.monotonic() - t0,
            kv_cache=KVCacheOutcome(
                mode=selection.mode,
                cache_id=selection.cache_id,
                loaded_tokens=result.kv_loaded_tokens,
                saved_tokens=result.kv_saved_tokens,
            ),
            mentor_state=self._mentor_public(),
        )

    def _remember_committed_request(self, client_request_sha256: str,
                                    request_sha256: str) -> None:
        known = self.committed_requests.get(client_request_sha256)
        if known is not None and known != request_sha256:
            raise BackendError("native committed-request identity drift")
        self.committed_requests[client_request_sha256] = request_sha256
        self.committed_requests.move_to_end(client_request_sha256)
        if len(self.committed_requests) > 256:
            self.committed_requests.popitem(last=False)
        self.last_committed_request_id = client_request_sha256

    def resolve_request_identity(
            self, body: dict, endpoint: str,
            request_key: str | None) -> tuple[str, str]:
        request_sha256 = _request_sha256(body, endpoint)
        client_request_sha256 = _client_request_sha256(
            request_key, session_epoch=self.session_epoch,
            turn_id=self.turn_count, request_sha256=request_sha256,
        )
        if request_key is None and self.turn_count > 0:
            previous = _client_request_sha256(
                None, session_epoch=self.session_epoch,
                turn_id=self.turn_count - 1, request_sha256=request_sha256,
            )
            if self.committed_requests.get(previous) == request_sha256:
                client_request_sha256 = previous
        return client_request_sha256, request_sha256

    def transport_identity(self, client_request_sha256: str) -> tuple[int, str]:
        known = self.transport_records.get(client_request_sha256)
        if known is not None:
            return known
        value = (int(time.time()), client_request_sha256[:32])
        self.transport_records[client_request_sha256] = value
        self.transport_records.move_to_end(client_request_sha256)
        if len(self.transport_records) > 256:
            self.transport_records.popitem(last=False)
        return value

    def complete(
            self, body: dict, *, endpoint: str = "chat",
            request_key: str | None = None,
            on_stream_start: Callable[[], None] | None = None,
            on_stream_delta: Callable[[str, str], None] | None = None,
            on_native_token: Callable[[int, int, bool], None] | None = None,
    ) -> Completion:
        if (on_stream_start is None) != (on_stream_delta is None):
            raise BackendError("incomplete live-stream callback pair")
        self._validate_request_schema(body, endpoint=endpoint)
        request_body = dict(body)
        if endpoint == "responses":
            request_body["messages"] = _responses_messages(body)
        model = request_body.get("model")
        if model is not None and model != self.config.model_id:
            raise RequestError(
                f"model {model!r} is not loaded; expected {self.config.model_id!r}"
            )
        controls = resolve_controls(
            request_body, server_kv_budget_gb=self.config.kv_budget_gb,
            context_tokens=self.config.context_tokens,
            max_output_tokens=self.config.max_output_tokens,
            kv_layout=self.config.kv_layout,
        )
        strict_proof = request_body.get("salt_proof_state", False)
        if not isinstance(strict_proof, bool):
            raise RequestError("salt_proof_state must be a boolean")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
        if controls.sampler_abi_id != 1 and self.config.dpr_mode != "off":
            raise RequestError("positive temperature is not qualified with DPR")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
        cache_value = request_body.get("kv_cache")
        mode, _ = KVCacheStore._request(cache_value)
        if mode != "none":
            raise RequestError(
                "portable KV selection is disabled for the persistent live session"
            )
        if self.engine is None:
            raise BackendError("persistent Gemma engine has not started")
        client_request_sha256, request_sha256 = self.resolve_request_identity(
            body, endpoint, request_key,
        )
        known_request = self.committed_requests.get(client_request_sha256)
        if known_request is not None and known_request != request_sha256:
            raise RequestError("idempotency key payload mismatch")
        started = on_stream_start or (lambda: None)
        delta = on_stream_delta or (lambda _kind, _value: None)
        t0 = time.monotonic()
        if known_request is not None:
            result = self.engine.replay(
                client_request_sha256=client_request_sha256,
                request_sha256=request_sha256,
                on_start=started, on_delta=delta,
            )
            if not result.replayed:
                raise BackendError("native replay did not return a replay result")
            return Completion(
                result=result, controls=controls,
                modality="image" if result.image_tokens else "text",
                elapsed_s=time.monotonic() - t0,
                kv_cache=KVCacheOutcome(
                    mode="live-replay", cache_id="launch",
                    loaded_tokens=result.kv_loaded_tokens,
                    saved_tokens=result.kv_saved_tokens,
                ),
                mentor_state=self._mentor_public(),
            )
        incoming_messages = request_body.get("messages")
        incoming_signature = self._history_signature(incoming_messages)
        history_after_user: list[tuple[str, str, str | None]]
        render_messages = incoming_messages
        if self.history_turns > 0:
            canonical_incoming = [
                item for item in incoming_signature
                if item[0] in ("user", "assistant")
            ]
            if incoming_signature[-1][0] != "user":
                raise RequestError(
                    "persistent continuation must end with a new user message"
                )
            if (request_key is None and
                    incoming_signature[-1] == self.last_user_signature):
                raise RequestError(
                    "persistent continuation repeats the last committed user message"
                )
            if len(incoming_signature) == 1:
                history_after_user = self.session_history + incoming_signature
            elif (len(canonical_incoming) == len(self.session_history) + 1 and
                  canonical_incoming[:-1] == self.session_history and
                  canonical_incoming[-1][0] == "user"):
                history_after_user = canonical_incoming
            else:
                raise RequestError(
                    "persistent history must match committed history plus "
                    "exactly one new user message"
                )
            assert isinstance(incoming_messages, list)
            render_messages = [incoming_messages[-1]]
        else:
            history_after_user = [
                item for item in incoming_signature
                if item[0] in ("user", "assistant")
            ]
        rendered = render_chat(
            render_messages,
            enable_thinking=controls.enable_thinking,
            continuation=self.session_position > 0,
        )
        if rendered.image_url is not None and self.session_position > 0:
            raise RequestError(
                "images are supported only on the first persistent turn"
            )
        image_context = (
            prepared_image(rendered.image_url, self.config.image_root)
            if rendered.image_url is not None else nullcontext(None)
        )
        with image_context as image:
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
            self.engine.sampler_abi = controls.sampler_abi_id
            self.engine.temperature_bits = controls.temperature_bits
            self.engine.sampler_seed = controls.seed
            self.engine.sampler_top_k = controls.top_k
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
            result = self.engine.request(
                rendered.prompt, image.path if image is not None else None,
                expected_epoch=self.session_epoch,
                expected_turn=self.turn_count,
                expected_history_turns=self.history_turns,
                expected_position=self.session_position,
                client_request_sha256=client_request_sha256,
                request_sha256=request_sha256,
                output_limit=controls.output_tokens,
                on_start=started, on_delta=delta,
                on_token=on_native_token,
                proof_state=strict_proof,
            )
        if result.replayed:
            self._remember_committed_request(
                result.client_request_sha256, result.request_sha256,
            )
            return Completion(
                result=result, controls=controls,
                modality="image" if result.image_tokens else "text",
                elapsed_s=time.monotonic() - t0,
                kv_cache=KVCacheOutcome(
                    mode="live-replay", cache_id="launch",
                    loaded_tokens=result.kv_loaded_tokens,
                    saved_tokens=result.kv_saved_tokens,
                ),
                mentor_state=self._mentor_public(),
            )
        if result.kv_loaded_tokens != self.session_position:
            raise BackendError("persistent live-KV start position drift")
        if (result.session_epoch != self.session_epoch or
                result.turn_id != self.turn_count + 1 or
                result.history_turns != self.history_turns + 1 or
                not result.session_continuable):
            raise BackendError("persistent native COMMIT authority drift")
        self.session_epoch = result.session_epoch
        self.turn_count = result.turn_id
        self.history_turns = result.history_turns
        self.session_position = result.kv_saved_tokens
        self.state_sha256 = result.state_sha256
        self.facts_sha256 = result.facts_sha256
        self.mindset_mode = result.mindset_mode
        self.mindset_end = result.mindset_end
        self.mindset_sha256 = result.mindset_sha256
        self.facts_rows = result.facts_rows
        self.session_history = history_after_user + [
            ("assistant", result.content, None),
        ]
        self.session_turns.append({
            "turn_id": result.turn_id,
            "prompt_ids": list(result.prompt_ids),
            "output_ids": list(result.output_ids),
            "response_sha256": result.response_sha256,
            "content": result.content,
            "output_limit": controls.output_tokens,
        })
        self.last_user_signature = history_after_user[-1]
        self._remember_committed_request(
            client_request_sha256, request_sha256,
        )
        return Completion(
            result=result, controls=controls,
            modality="image" if rendered.image_url is not None else "text",
            elapsed_s=time.monotonic() - t0,
            kv_cache=KVCacheOutcome(
                mode="live", cache_id="launch", loaded_tokens=result.kv_loaded_tokens,
                saved_tokens=result.kv_saved_tokens,
            ),
            mentor_state=self._mentor_public(),
        )


# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
def _sampler_payload(completion: Completion) -> dict:
    controls = completion.controls
    sampled = controls.sampler_abi_id == 2
    return {
        "mode": "sampled" if sampled else "greedy",
        "sampler_abi": controls.sampler_abi,
        "temperature": controls.temperature,
        "temperature_bits": controls.temperature_bits,
        "seed": controls.seed if sampled else None,
        "top_k": controls.top_k,
        "rng_abi": controls.rng_abi,
        "rng_draw_count": completion.result.completion_tokens if sampled else 0,
    }
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1

def _chat_payload(completion: Completion, *, model: str,
                  completion_id: str, created: int) -> dict:
    result = completion.result
    message = {"role": "assistant", "content": result.content}
    if result.reasoning_content is not None:
        message["reasoning_content"] = result.reasoning_content
    payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": completion.kv_cache.loaded_tokens + result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": (
                completion.kv_cache.loaded_tokens + result.prompt_tokens +
                result.completion_tokens
            ),
        },
        "x_salt": {
            "runtime_ready": RUNTIME_READY,
            "modality": completion.modality,
            "context_tokens": completion.controls.context_tokens,
            "max_output_tokens": completion.controls.output_tokens,
            "kv_allowance_gb": completion.controls.kv_allowance_gb,
            "kv_bytes": completion.controls.kv_bytes,
            "reasoning_effort": completion.controls.reasoning_effort,
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
            "sampler": _sampler_payload(completion),
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
            "kv_cache": completion.kv_cache.as_dict(),
            "session_epoch": result.session_epoch,
            "turn_id": result.turn_id,
            "synthetic_close": result.synthetic_close,
            "state_sha256": result.state_sha256,
            "response_sha256": result.response_sha256,
            "prompt_ids": list(result.prompt_ids),
            "output_ids": list(result.output_ids),
            "mindset_mode": result.mindset_mode,
            "mindset_end": result.mindset_end,
            "mindset_sha256": result.mindset_sha256,
            "mandatory_state_bytes": gemma4_kv_state_bytes(result.mindset_end),
            "facts_optional": True,
            "facts_sha256": result.facts_sha256,
            "facts_rows": result.facts_rows,
            "facts_bytes": (gemma4_kv_capacity_bytes(result.facts_rows) if result.facts_rows else 0),
            "client_request_sha256": result.client_request_sha256,
            "request_sha256": result.request_sha256,
            "original_request_id": result.original_request_id,
            "replayed": result.replayed,
            "elapsed_s": round(completion.elapsed_s, 3),
        },
    }
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    payload["x_salt"]["dpr"] = result.dpr_result
    payload["x_salt"]["memory_snapshots"] = list(result.memory_snapshots)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    if completion.mentor_state is not None:
        payload["x_salt"]["mentor_state"] = completion.mentor_state
    return payload


def _responses_messages(body: dict) -> list[dict]:
    instructions = body.get("instructions")
    messages: list[dict] = []
    if instructions is not None:
        if not isinstance(instructions, str):
            raise RequestError("Responses API instructions must be a string")
        if instructions.strip():
            messages.append({"role": "system", "content": instructions})
    value = body.get("input")
    if isinstance(value, str):
        return messages + [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise RequestError("Responses API input must be a string or message array")
    for item in value:
        if not isinstance(item, dict) or "role" not in item:
            raise RequestError("Responses API input items must be messages")
        unknown = set(item) - {"type", "role", "content"}
        if unknown:
            raise RequestError(
                "unsupported Responses message fields: " +
                ", ".join(sorted(unknown))
            )
        if item.get("type") not in (None, "message"):
            raise RequestError("Responses API input item type must be message")
        messages.append({"role": item["role"], "content": item.get("content", "")})
    return messages


def _responses_payload(completion: Completion, *, model: str,
                       response_id: str, created: int) -> dict:
    result = completion.result
    status = "completed" if result.finish_reason == "stop" else "incomplete"
    message: dict = {
        "id": "msg_" + response_id[5:],
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": result.content}],
    }
    if result.reasoning_content is not None:
        message["reasoning_content"] = result.reasoning_content
    payload = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": [message],
        "output_text": result.content,
        "usage": {
            "input_tokens": completion.kv_cache.loaded_tokens + result.prompt_tokens,
            "output_tokens": result.completion_tokens,
            "total_tokens": (
                completion.kv_cache.loaded_tokens + result.prompt_tokens +
                result.completion_tokens
            ),
        },
        "x_salt": {
            "runtime_ready": RUNTIME_READY,
            "modality": completion.modality,
            "context_tokens": completion.controls.context_tokens,
            "max_output_tokens": completion.controls.output_tokens,
            "kv_allowance_gb": completion.controls.kv_allowance_gb,
            "kv_bytes": completion.controls.kv_bytes,
            "reasoning_effort": completion.controls.reasoning_effort,
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
            "sampler": _sampler_payload(completion),
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
            "kv_cache": completion.kv_cache.as_dict(),
            "session_epoch": result.session_epoch,
            "turn_id": result.turn_id,
            "synthetic_close": result.synthetic_close,
            "state_sha256": result.state_sha256,
            "response_sha256": result.response_sha256,
            "prompt_ids": list(result.prompt_ids),
            "output_ids": list(result.output_ids),
            "mindset_mode": result.mindset_mode,
            "mindset_end": result.mindset_end,
            "mindset_sha256": result.mindset_sha256,
            "mandatory_state_bytes": gemma4_kv_state_bytes(result.mindset_end),
            "facts_optional": True,
            "facts_sha256": result.facts_sha256,
            "facts_rows": result.facts_rows,
            "facts_bytes": (gemma4_kv_capacity_bytes(result.facts_rows) if result.facts_rows else 0),
            "client_request_sha256": result.client_request_sha256,
            "request_sha256": result.request_sha256,
            "original_request_id": result.original_request_id,
            "replayed": result.replayed,
            "elapsed_s": round(completion.elapsed_s, 3),
        },
    }
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    payload["x_salt"]["dpr"] = result.dpr_result
    payload["x_salt"]["memory_snapshots"] = list(result.memory_snapshots)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    if status == "incomplete":
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    if completion.mentor_state is not None:
        payload["x_salt"]["mentor_state"] = completion.mentor_state
    return payload


def handler_for(backend: Gemma4Backend):
    class Gemma4Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(HTTP_BODY_TIMEOUT_S)

        def handle_expect_100(self):
            self.close_connection = True
            self._error(417, "Expect is not supported", "invalid_request_error")
            return False

        def send_error(self, code, message=None, explain=None):
            self.close_connection = True
            super().send_error(code, message, explain)

        def _method_not_allowed(self):
            self.close_connection = True
            self._error(405, "method not allowed", "invalid_request_error")

        do_PUT = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_OPTIONS = _method_not_allowed
        do_TRACE = _method_not_allowed
        do_CONNECT = _method_not_allowed

        def log_message(self, format, *args):
            sys.stderr.write("[gemma4-serve] %s\n" % (format % args))

        def _send_json(self, code: int, value: dict):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, code: int, message: str, error_type: str):
            self._send_json(code, {"error": {
                "message": message,
                "type": error_type,
            }})

        def _framing_error(self, message: str):
            # Never leave unread or ambiguously framed bytes on a persistent
            # HTTP/1.1 connection: they could be parsed as another request.
            self.close_connection = True
            raise RequestError(message)

        def _content_length(self, *, required: bool) -> int | None:
            if self.headers.get_all("Transfer-Encoding", []):
                self._framing_error("Transfer-Encoding is not supported")
            values = self.headers.get_all("Content-Length", [])
            if not values:
                if required:
                    self._framing_error("Content-Length required")
                return None
            if len(values) != 1:
                self._framing_error("exactly one Content-Length is required")
            raw_length = values[0]
            if re.fullmatch(r"[0-9]+", raw_length) is None:
                self._framing_error("invalid Content-Length")
            length = int(raw_length)
            if length > MAX_REQUEST_BYTES:
                self._framing_error("request body exceeds 4 MiB")
            return length

        def _reject_get_body(self) -> None:
            length = self._content_length(required=False)
            if length not in (None, 0):
                self._framing_error("GET request bodies are not accepted")

        def do_GET(self):
            try:
                self._reject_get_body()
            except RequestError as exc:
                self._error(400, str(exc), "invalid_request_error")
                return
            path = self.path.rstrip("/")
            if path == "/healthz":
                session = backend.session_status()
                engine_ok = session["engine_running"]
                self._send_json(200 if engine_ok else 503, {
                    "status": "ok" if engine_ok else "unavailable",
                    "model": backend.config.model_id,
                    "runtime_ready": RUNTIME_READY,
                    "session_position": session["position"],
                    "turns": session["turns"],
                    "session_epoch": session["session_epoch"],
                    "phase": session["phase"],
                    "active_request": session["active_request"],
                    "last_committed_request_id": (
                        session["last_committed_request_id"]
                    ),
                    "journal_entries": session["journal_entries"],
                    "recovery_required": session["recovery_required"],
                })
                return
            if path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{
                    "id": backend.config.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "salt",
                    "x_salt": backend.capabilities(),
                }]})
                return
            if path == "/v1/session":
                self._send_json(200, backend.session_status())
                return
            if path == "/v1/kv-caches":
                if backend.cache_store is None:
                    self._send_json(200, {
                        "object": "list", "data": [], "enabled": False,
                    })
                    return
                try:
                    caches = [
                        item.as_dict()
                        for item in backend.cache_store.list_caches()
                    ]
                except RequestError as exc:
                    self._error(500, str(exc), "server_error")
                    return
                self._send_json(200, {
                    "object": "list", "data": caches, "enabled": True,
                })
                return
            self._error(404, "not found", "invalid_request_error")

        def _body(self) -> dict:
            length = self._content_length(required=True)
            assert length is not None
            try:
                raw = self.rfile.read(length)
            except OSError:
                self._framing_error("timed out or failed reading request body")
            if len(raw) != length:
                self._framing_error("truncated request body")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestError(f"invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise RequestError("request body must be an object")
            return value

        def _request_key(self) -> str | None:
            values = self.headers.get_all("Idempotency-Key", [])
            if len(values) > 1:
                self._framing_error("at most one Idempotency-Key is accepted")
            if not values:
                return None
            value = values[0]
            if IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
                raise RequestError(
                    "Idempotency-Key must be 1..128 ASCII token characters"
                )
            return value

        def do_POST(self):
            stream_started = False
            responses_api = False
            client_disconnected = False
            try:
                body = self._body()
                request_key = self._request_key()
                path = self.path.rstrip("/")
                responses_api = path == "/v1/responses"
                if path in ("/v1/session/reset", "/v1/session/clear"):
                    if body:
                        raise RequestError(
                            "session clear body must be an empty object"
                        )
                    if not backend.lock.acquire(timeout=1.0):
                        self._error(
                            503, "engine busy (single-flight)", "server_error",
                        )
                        return
                    try:
                        backend.clear_facts()
                    finally:
                        backend.lock.release()
                    self._send_json(200, backend.session_status())
                    return
                if path == "/v1/session/hard-reset":
                    self._error(
                        403,
                        "hard reset is disabled on the unauthenticated HTTP surface",
                        "permission_error",
                    )
                    return
                if path == "/v1/session/export":
                    if (set(body) not in ({"id"}, {"id", "replace"}) or
                            not isinstance(body["id"], str) or
                            not isinstance(body.get("replace", False), bool)):
                        raise RequestError(
                            "session export requires a string id and optional boolean replace"
                        )
                    if request_key is not None:
                        raise RequestError("session export does not accept Idempotency-Key")
                    if not backend.lock.acquire(timeout=1.0):
                        self._error(503, "engine busy (single-flight)", "server_error")
                        return
                    try:
                        exported = backend.export_cache(
                            body["id"], replace_existing=body.get("replace", False),
                        )
                    finally:
                        backend.lock.release()
                    self._send_json(200, exported)
                    return
                if path == "/v1/session/import":
                    if set(body) != {"id", "manifest"} or \
                            not isinstance(body["id"], str):
                        raise RequestError(
                            "session import requires string id and manifest"
                        )
                    if request_key is not None:
                        raise RequestError("session import does not accept Idempotency-Key")
                    if not backend.lock.acquire(timeout=1.0):
                        self._error(503, "engine busy (single-flight)", "server_error")
                        return
                    try:
                        imported = backend.import_session_cache(
                            body["id"], body["manifest"],
                        )
                    finally:
                        backend.lock.release()
                    self._send_json(200, imported)
                    return
                if path == "/v1/session/cache/delete":
                    if set(body) != {"id"} or not isinstance(body["id"], str):
                        raise RequestError("session cache delete requires string id")
                    if request_key is not None:
                        raise RequestError(
                            "session cache delete does not accept Idempotency-Key"
                        )
                    if backend.cache_store is None:
                        raise RequestError("persistent KV cache is not configured")
                    cache_request = {"mode": "load", "id": body["id"]}
                    with backend.cache_store.lock(cache_request):
                        deleted = backend.cache_store.delete(body["id"])
                    self._send_json(200, {
                        "object": "salt.kv_cache.deleted",
                        "id": body["id"],
                        "deleted": deleted,
                    })
                    return
                if path == "/v1/session/internal/dpr":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        self._error(
                            403, "DPR reconfiguration is loopback-only",
                            "permission_error",
                        )
                        return
                    if request_key is not None:
                        raise RequestError(
                            "DPR reconfiguration does not accept Idempotency-Key"
                        )
                    if not backend.lock.acquire(timeout=1.0):
                        self._error(503, "engine busy (single-flight)", "server_error")
                        return
                    try:
                        status = backend.reconfigure_dpr(body)
                    finally:
                        backend.lock.release()
                    self._send_json(200, {
                        "object": "salt.dpr.attachment",
                        "dpr": status["dpr"],
                        "session_epoch": status["session_epoch"],
                        "runtime_ready": status["runtime_ready"],
                    })
                    return
                if path not in ("/v1/chat/completions", "/v1/responses"):
                    self._error(404, "not found", "invalid_request_error")
                    return
                if not backend.lock.acquire(timeout=1.0):
                    self._error(503, "engine busy (single-flight)", "server_error")
                    return
                endpoint_name = "responses" if responses_api else "chat"
                try:
                    client_request_sha256, _ = backend.resolve_request_identity(
                        body, endpoint_name, request_key,
                    )
                    created, unique = backend.transport_identity(
                        client_request_sha256,
                    )
                except BaseException:
                    backend.lock.release()
                    raise
                stream_requested = body.get("stream") is True
                completion_id = "chatcmpl-" + unique
                response_id = "resp_" + unique
                item_id = "msg_" + unique

                def start_stream() -> None:
                    nonlocal stream_started, client_disconnected
                    stream_started = True
                    try:
                        if responses_api:
                            self._start_responses_stream(
                                response_id, item_id, created,
                            )
                        else:
                            self._start_chat_stream(completion_id, created)
                    except OSError:
                        client_disconnected = True
                        self.close_connection = True

                def stream_delta(kind: str, value: str) -> None:
                    nonlocal client_disconnected
                    if client_disconnected:
                        return
                    try:
                        if responses_api:
                            self._stream_responses_delta(
                                response_id, item_id, kind, value,
                            )
                        else:
                            self._stream_chat_delta(
                                completion_id, created, kind, value,
                            )
                    except OSError:
                        client_disconnected = True
                        self.close_connection = True

                try:
                    completion = backend.complete(
                        body, endpoint=endpoint_name,
                        request_key=request_key,
                        on_stream_start=start_stream if stream_requested else None,
                        on_stream_delta=stream_delta if stream_requested else None,
                    )
                finally:
                    backend.lock.release()
                if stream_requested:
                    if client_disconnected:
                        return
                    if responses_api:
                        self._finish_responses_stream(
                            completion, response_id, created,
                        )
                    else:
                        stream_options = body.get("stream_options") or {}
                        include_usage = stream_options.get("include_usage", True)
                        self._finish_chat_stream(
                            completion, completion_id, created,
                            include_usage=include_usage,
                        )
                    return
                if responses_api:
                    payload = _responses_payload(
                        completion, model=backend.config.model_id,
                        response_id=response_id, created=created,
                    )
                    self._send_json(200, payload)
                    return
                self._send_json(200, _chat_payload(
                    completion, model=backend.config.model_id,
                    completion_id=completion_id, created=created,
                ))
            except RequestError as exc:
                if stream_started:
                    if not client_disconnected:
                        self._stream_error(str(exc), responses_api=responses_api)
                elif not client_disconnected:
                    self._error(400, str(exc), "invalid_request_error")
            except BackendError as exc:
                sys.stderr.write(f"[gemma4-serve] backend error: {exc}\n")
                sys.stderr.flush()
                try:
                    recovered = backend.recover_if_dead()
                except BackendError as recovery_exc:
                    recovered = False
                    sys.stderr.write(
                        f"[gemma4-serve] fatal recovery failed: {recovery_exc}\n"
                    )
                    sys.stderr.flush()
                if recovered:
                    sys.stderr.write(
                        "[gemma4-serve] fatal native epoch recovered; "
                        "current request remains failed and may be retried\n"
                    )
                    sys.stderr.flush()
                if stream_started:
                    if not client_disconnected:
                        self._stream_error(
                            "Gemma 4 backend failed", responses_api=responses_api,
                        )
                elif not client_disconnected:
                    self._error(500, "Gemma 4 backend failed", "server_error")
            except BrokenPipeError:
                self.close_connection = True
                sys.stderr.write(
                    "[gemma4-serve] client disconnected after commit; "
                    "response remains replayable\n"
                )
                sys.stderr.flush()
            except Exception as exc:
                sys.stderr.write(
                    f"[gemma4-serve] unexpected backend error: {exc}\n"
                )
                sys.stderr.flush()
                if stream_started:
                    if not client_disconnected:
                        self._stream_error(
                            "Gemma 4 backend failed", responses_api=responses_api,
                        )
                elif not client_disconnected:
                    self._error(500, "Gemma 4 backend failed", "server_error")

        def _start_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _write_sse(self, value: dict, *, event: str | None = None) -> None:
            if event is not None:
                self.wfile.write(("event: " + event + "\n").encode("ascii"))
            self.wfile.write(
                ("data: " + json.dumps(value, ensure_ascii=False) +
                 "\n\n").encode("utf-8")
            )
            self.wfile.flush()

        @staticmethod
        def _chat_chunk(completion_id: str, created: int, delta: dict,
                        finish_reason: str | None = None) -> dict:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": backend.config.model_id,
                "choices": [{
                    "index": 0, "delta": delta,
                    "finish_reason": finish_reason,
                }],
                "x_salt": {"runtime_ready": RUNTIME_READY, "live_stream": True},
            }

        def _start_chat_stream(self, completion_id: str, created: int) -> None:
            self._start_sse()
            self._write_sse(self._chat_chunk(
                completion_id, created, {"role": "assistant", "content": ""},
            ))

        def _stream_chat_delta(self, completion_id: str, created: int,
                               kind: str, value: str) -> None:
            key = "reasoning_content" if kind == "reasoning_content" else "content"
            self._write_sse(self._chat_chunk(
                completion_id, created, {key: value},
            ))

        def _finish_chat_stream(
                self, completion: Completion, completion_id: str, created: int,
                *, include_usage: bool) -> None:
            result = completion.result
            done = {
                **self._chat_chunk(
                    completion_id, created, {}, result.finish_reason,
                ),
            }
            done["x_salt"].update({
                "modality": completion.modality,
                "context_tokens": completion.controls.context_tokens,
                "max_output_tokens": completion.controls.output_tokens,
                "kv_allowance_gb": completion.controls.kv_allowance_gb,
                "kv_bytes": completion.controls.kv_bytes,
                "reasoning_effort": completion.controls.reasoning_effort,
                "kv_cache": completion.kv_cache.as_dict(),
                "session_epoch": result.session_epoch,
                "turn_id": result.turn_id,
                "synthetic_close": result.synthetic_close,
                "state_sha256": result.state_sha256,
                "response_sha256": result.response_sha256,
                "prompt_ids": list(result.prompt_ids),
                "output_ids": list(result.output_ids),
                "mindset_mode": result.mindset_mode,
                "mindset_end": result.mindset_end,
                "mindset_sha256": result.mindset_sha256,
                "mandatory_state_bytes": gemma4_kv_state_bytes(result.mindset_end),
                "facts_optional": True,
                "facts_sha256": result.facts_sha256,
                "facts_rows": result.facts_rows,
                "facts_bytes": (gemma4_kv_capacity_bytes(result.facts_rows) if result.facts_rows else 0),
                "client_request_sha256": result.client_request_sha256,
                "request_sha256": result.request_sha256,
                "original_request_id": result.original_request_id,
                "replayed": result.replayed,
                "elapsed_s": round(completion.elapsed_s, 3),
            })
            if include_usage:
                done["usage"] = {
                    "prompt_tokens": (
                        completion.kv_cache.loaded_tokens + result.prompt_tokens
                    ),
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": (
                        completion.kv_cache.loaded_tokens + result.prompt_tokens +
                        result.completion_tokens
                    ),
                }
            self._write_sse(done)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _start_responses_stream(
                self, response_id: str, item_id: str, created: int) -> None:
            self._start_sse()
            response = {
                "id": response_id, "object": "response",
                "created_at": created, "status": "in_progress",
                "model": backend.config.model_id, "output": [],
                "x_salt": {"runtime_ready": RUNTIME_READY, "live_stream": True},
            }
            self._write_sse(
                {"type": "response.created", "response": response},
                event="response.created",
            )
            self._write_sse({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": item_id, "type": "message",
                         "status": "in_progress", "role": "assistant",
                         "content": []},
            }, event="response.output_item.added")

        def _stream_responses_delta(
                self, response_id: str, item_id: str,
                kind: str, value: str) -> None:
            del response_id
            event = (
                "response.reasoning_content.delta"
                if kind == "reasoning_content"
                else "response.output_text.delta"
            )
            self._write_sse({
                "type": event, "item_id": item_id,
                "output_index": 0, "content_index": 0, "delta": value,
            }, event=event)

        def _finish_responses_stream(
                self, completion: Completion,
                response_id: str, created: int) -> None:
            payload = _responses_payload(
                completion, model=backend.config.model_id,
                response_id=response_id, created=created,
            )
            self._write_sse(
                {"type": "response.output_text.done",
                 "text": completion.result.content},
                event="response.output_text.done",
            )
            self._write_sse(
                {"type": "response.completed", "response": payload},
                event="response.completed",
            )
            self.close_connection = True

        def _stream_error(self, message: str, *, responses_api: bool) -> None:
            error = {"message": message, "type": "server_error"}
            if responses_api:
                self._write_sse(
                    {"type": "error", "error": error}, event="error",
                )
            else:
                self._write_sse({"error": error})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            self.close_connection = True

    return Gemma4Handler


def _gemma_int_setting(value: int | None, environment: str, default: int) -> int:
    if value is not None:
        return value
    raw = os.environ.get(environment, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise RequestError(f"{environment} must be an integer") from exc


def _gemma_prefill_chunk_setting(value: int | None) -> int:
    raw_recipe = GEMMA_ENGINE_CONFIG.get("SALT_PREFILL_B")
    try:
        recipe_value = int(raw_recipe) if raw_recipe is not None else 0
    except ValueError as exc:
        raise RequestError("selected Gemma recipe has invalid SALT_PREFILL_B") from exc
    if recipe_value < 1 or recipe_value > GEMMA4_MAX_PREFILL_CHUNK:
        raise RequestError("selected Gemma recipe has invalid SALT_PREFILL_B")
    requested = value
    if requested is None:
        raw_requested = os.environ.get("GEMMA4_PREFILL_CHUNK_TOKENS")
        if raw_requested:
            try:
                requested = int(raw_requested)
            except ValueError as exc:
                raise RequestError(
                    "GEMMA4_PREFILL_CHUNK_TOKENS must be an integer",
                ) from exc
    if requested is not None and requested != recipe_value:
        raise RequestError(
            "Gemma prefill chunk must match selected model/platform recipe "
            f"SALT_PREFILL_B={recipe_value}",
        )
    return recipe_value


def _gemma_kv_layout_setting() -> str:
    if GEMMA_ENGINE_CONFIG.get("SALT_GEMMA_COMPUTE_NODE") != "full":
        return "hybrid"
    if GEMMA_ENGINE_CONFIG.get("SALT_GEMMA_GPU_KV_RING") == "1":
        return "hybrid"
    if (GEMMA_ENGINE_CONFIG.get("SALT_TEXT_FINE_TOKEN") == "1" and
            GEMMA_ENGINE_CONFIG.get("SALT_GPU_BOUNDED_WEIGHTS") == "1" and
            GEMMA_ENGINE_CONFIG.get("SALT_GPU_WEIGHT_ADDRESSABILITY") ==
            "pageable"):
        return "hybrid"
    return "absolute"


def _gemma_float_setting(value: float | None, environment: str,
                         default: float) -> float:
    if value is not None:
        return value
    raw = os.environ.get(environment, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise RequestError(f"{environment} must be a number") from exc
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
def _apply_server_engine_overrides(args) -> None:
    """Apply validated startup policy before the native process is created."""
    global GEMMA_RESIDENCY_CONFIG
    global GEMMA_EXPERT_BUDGET_GB
    global GEMMA_EXPERT_BUDGET_BYTES

    candidate = dict(GEMMA_ENGINE_CONFIG)
    route = (
        getattr(args, "gemma_target_route_n", None),
        getattr(args, "gemma_target_x", None),
        getattr(args, "gemma_target_route_f", None),
        getattr(args, "gemma_target_route_q", None),
    )
    if any(value is not None for value in route):
        if any(value is None for value in route):
            raise RequestError("Gemma override requires NFQ N/F/Q and TARGET X")
        candidate.update({
            "SALT_TARGET_ROUTE_N": str(route[0]),
            "SALT_TARGET_X": str(route[1]),
            "SALT_TARGET_ROUTE_F": str(route[2]),
            "SALT_TARGET_ROUTE_Q": str(route[3]),
        })
    expert_budget = getattr(args, "gemma_expert_budget_gb", None)
    if expert_budget is not None:
        candidate["SALT_EXPERT_BUDGET_GB"] = str(expert_budget)
    memory_limit = getattr(args, "gemma_memory_limit_gb", None)
    if memory_limit is not None:
        if not math.isfinite(memory_limit) or memory_limit <= 0.0:
            raise RequestError("Gemma memory limit must be a positive finite number")
        candidate["SALT_GEMMA_MEMORY_LIMIT_GB"] = format(
            memory_limit, ".9f"
        ).rstrip("0").rstrip(".")

    try:
        validate_gemma_target_route_config(candidate)
        residency = validate_gemma_residency_config(candidate, require_host=True)
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    GEMMA_ENGINE_CONFIG.clear()
    GEMMA_ENGINE_CONFIG.update(candidate)
    GEMMA_RESIDENCY_CONFIG = residency
    GEMMA_EXPERT_BUDGET_GB = residency["budget_gb"]
    GEMMA_EXPERT_BUDGET_BYTES = GEMMA_EXPERT_BUDGET_GB * 1_000_000_000
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1


def serve_from_args(args) -> int:
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    _apply_server_engine_overrides(args)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    required = {
        "--gemma-source-dir": args.gemma_source_dir,
        "--gemma-pool": args.gemma_pool,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise BackendError("missing required Gemma setup: " + ", ".join(missing))
    pool = Path(args.gemma_pool).expanduser()
    receipt_value = args.gemma_receipt or str(pool) + ".import.json"
    config = Gemma4Config(
        source_dir=Path(args.gemma_source_dir).expanduser(),
        pool=pool,
        receipt=Path(receipt_value).expanduser(),
        model_id=args.model,
        text_wrapper=Path(args.gemma_text_wrapper).expanduser(),
        multimodal_wrapper=Path(args.gemma_multimodal_wrapper).expanduser(),
        text_runner=Path(args.gemma_text_runner).expanduser(),
        multimodal_runner=Path(args.gemma_multimodal_runner).expanduser(),
        server_runner=Path(args.gemma_server_runner).expanduser(),
        text_build_receipt=(Path(args.gemma_text_build_receipt).expanduser()
                            if args.gemma_text_build_receipt else None),
        multimodal_build_receipt=(
            Path(args.gemma_multimodal_build_receipt).expanduser()
            if args.gemma_multimodal_build_receipt else None
        ),
        server_build_receipt=(Path(args.gemma_server_build_receipt).expanduser()
                              if args.gemma_server_build_receipt else None),
        auth_receipt=(Path(args.gemma_auth_receipt).expanduser()
                      if args.gemma_auth_receipt else Path(str(pool) + ".auth.json")),
        shared_kv=(Path(args.gemma_shared_kv).expanduser()
                   if args.gemma_shared_kv else None),
        mentor_root=(Path(args.gemma_mentor_root).expanduser()
                     if args.gemma_mentor_root else None),
        mentor_anchor=args.gemma_mentor_anchor,
        dpr_root=(Path(args.gemma_dpr_root).expanduser()
                  if args.gemma_dpr_root else None),
        dpr_mode=args.gemma_dpr_mode,
        dpr_budget_gb=_gemma_float_setting(
            args.gemma_dpr_budget_gb, "GEMMA4_DPR_BUDGET_GB", 0.0,
        ),
        dpr_serial_ms_per_token=_gemma_float_setting(
            args.gemma_dpr_serial_ms_per_token,
            "GEMMA4_DPR_SERIAL_MS_PER_TOKEN", 0.0,
        ),
        dpr_retention_gb=_gemma_float_setting(
            args.gemma_dpr_retention_gb, "GEMMA4_DPR_RETENTION_GB", 2.0,
        ),
        dpr_mentor_enabled=args.gemma_dpr_mentor,
        dpr_mindset_attention_enabled=args.gemma_dpr_mindset_attention,
        proof_state=args.gemma_proof_state,
        python=args.gemma_python,
        workers=_gemma_int_setting(args.gemma_workers, "GEMMA4_WORKERS", 4),
        context_tokens=_gemma_int_setting(
            args.gemma_context_tokens, "GEMMA4_CONTEXT_TOKENS", 512,
        ),
        max_output_tokens=_gemma_int_setting(
            args.gemma_max_output_tokens, "GEMMA4_MAX_OUTPUT_TOKENS", 128,
        ),
        prefill_chunk_tokens=_gemma_prefill_chunk_setting(
            args.gemma_prefill_chunk_tokens,
        ),
        kv_layout=_gemma_kv_layout_setting(),
        kv_budget_gb=_gemma_float_setting(
            args.gemma_kv_budget_gb, "GEMMA4_KV_BUDGET_GB", 1.0,
        ),
        timeout_s=_gemma_float_setting(
            args.gemma_timeout_s, "GEMMA4_TIMEOUT_S", 900.0,
        ),
        image_root=(Path(args.gemma_image_root).expanduser()
                    if args.gemma_image_root else None),
        kv_cache_root=(Path(args.gemma_kv_cache_root).expanduser()
                       if args.gemma_kv_cache_root else None),
    )
    if args.gemma_verify_only:
        print("[gemma4-serve] fully authenticating package and publishing receipt...",
              file=sys.stderr, flush=True)
        authority = Gemma4AuthenticationAuthority(config, reauthenticate=True)
        authority.close()
        backend = Gemma4Backend(config)
        print("GEMMA4_SERVER_SETUP_OK")
        print(json.dumps(backend.capabilities(), sort_keys=True))
        return 0
    authority = Gemma4AuthenticationAuthority(config)
    backend = Gemma4Backend(config, authentication=authority)
    server = None
    try:
        backend.validate_setup(authenticate=False)
        print("[gemma4-serve] starting authenticated persistent native engine...",
              file=sys.stderr, flush=True)
        backend.start_persistent()
        server = ThreadingHTTPServer((args.host, args.port), handler_for(backend))
        print(
            f"[gemma4-serve] API http://{args.host}:{args.port} "
            f"model={config.model_id} workers={config.workers} "
            f"kv-max={config.kv_budget_gb:g}GB "
            f"CTX{config.context_tokens}/O{config.max_output_tokens} "
            f"prefill-chunk={config.prefill_chunk_tokens} "
            f"one-live-session "
            f"runtime_ready={str(RUNTIME_READY).lower()}",
            file=sys.stderr, flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gemma4-serve] bye", file=sys.stderr, flush=True)
    finally:
        if server is not None:
            server.server_close()
        backend.stop_persistent()
        authority.close()
    return 0
