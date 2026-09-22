#!/usr/bin/env python3
"""Preserved Gemma 4 text/image end-to-end final qualification check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models/gemma4-26b-a4b/qualification/e2e-cases.json"
DEFAULT_JSON = Path("/tmp/salt-gemma4-e2e-report.json")
DEFAULT_REPORT = Path("/tmp/salt-gemma4-e2e-report.md")
DEFAULT_ARTIFACTS = Path("/tmp/salt-gemma4-e2e-artifacts")
EXPECTED_CASES = (
    "text-retrieval-apollo11",
    "document-qa-constitution",
    "image-qa-red-mug",
    "ocr-standard-pangram",
)
RESPONSE_MARKERS = {
    "text": ("GEMMA4_QA_RESPONSE_BEGIN", "GEMMA4_QA_RESPONSE_END"),
    "image": ("GEMMA4_MM_RESPONSE_BEGIN", "GEMMA4_MM_RESPONSE_END"),
}
WARM_RESPONSE_MARKERS = {
    "text": (
        "GEMMA4_QA_KV_WARM_RESPONSE_BEGIN",
        "GEMMA4_QA_KV_WARM_RESPONSE_END",
    ),
    "image": (
        "GEMMA4_MM_KV_WARM_RESPONSE_BEGIN",
        "GEMMA4_MM_KV_WARM_RESPONSE_END",
    ),
}
HEX64 = re.compile(r"[0-9a-f]{64}")
REPORT_SCHEMA = "salt.gemma4-e2e-report.v4"
KV_SNAPSHOT_VERSION = 2
KV_SNAPSHOT_HEADER_BYTES = 192
KV_SNAPSHOT_BYTES_PER_TOKEN = 450_560


class BenchmarkError(RuntimeError):
    """A fail-closed benchmark contract or execution error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_response(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def load_manifest(path: Path, *, permit_pending: bool) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load manifest {path}: {exc}") from exc
    if manifest.get("schema") != "salt.gemma4-e2e-cases.v1":
        raise BenchmarkError("manifest schema drift")
    model = manifest.get("model", {})
    if model.get("repository") != "mlx-community/gemma-4-26b-a4b-it-4bit" or \
            model.get("revision") != "0d77464eeb233a2da68ebf9d7dc4edaac7db956d":
        raise BenchmarkError("pinned model identity drift")
    execution = manifest.get("execution", {})
    required_execution = {
        "repeat": 3,
        "ctx": 200,
        "gen": 50,
        "workers": 3,
        "serial": True,
        "compare_kv_warm": True,
        "warm_state": "prefilled_kv",
        "runtime_ready": True,
    }
    if execution != required_execution:
        raise BenchmarkError(f"execution contract drift: {execution!r}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or tuple(case.get("id") for case in cases) != EXPECTED_CASES:
        raise BenchmarkError("case inventory or order drift")
    for case in cases:
        if case.get("modality") not in RESPONSE_MARKERS:
            raise BenchmarkError(f"unsupported modality for {case.get('id')}")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise BenchmarkError(f"empty question for {case['id']}")
        accepted = case.get("accepted_normalized")
        if not isinstance(accepted, list) or not accepted or any(
                normalize_response(item) != item for item in accepted):
            raise BenchmarkError(f"invalid normalized answers for {case['id']}")
        baseline = case.get("baseline_response_sha256")
        if baseline != "PENDING" and not (
                isinstance(baseline, str) and HEX64.fullmatch(baseline)):
            raise BenchmarkError(f"invalid response baseline for {case['id']}")
        if baseline == "PENDING" and not permit_pending:
            raise BenchmarkError(f"unrecorded response baseline for {case['id']}")
        image = case.get("image")
        if case["modality"] == "image":
            if not isinstance(image, str) or not image:
                raise BenchmarkError(f"missing image for {case['id']}")
            expected_hash = case.get("image_sha256")
            if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
                raise BenchmarkError(f"missing image identity for {case['id']}")
        elif image is not None:
            raise BenchmarkError(f"text case unexpectedly has an image: {case['id']}")
    return manifest


def extract_line_fields(stderr: str, marker: str) -> dict[str, str]:
    lines = [line for line in stderr.splitlines() if line.startswith(marker + " ")]
    if len(lines) != 1:
        raise BenchmarkError(f"expected one {marker} line, got {len(lines)}")
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", lines[0]))


def extract_response(stdout: str, modality: str, expected_bytes: int,
                     *, warm: bool = False) -> str:
    if expected_bytes < 0:
        raise BenchmarkError("negative response byte count")
    markers = WARM_RESPONSE_MARKERS if warm else RESPONSE_MARKERS
    begin, end = markers[modality]
    raw = stdout.encode("utf-8")
    prefix = (begin + "\n").encode("ascii")
    end_marker = end.encode("ascii")
    if raw.count(prefix) != 1 or raw.count(end_marker) != 1:
        raise BenchmarkError(f"expected one {begin}/{end} response block")
    start = raw.index(prefix) + len(prefix)
    finish = start + expected_bytes
    if finish > len(raw):
        raise BenchmarkError(f"truncated {begin} response block")
    response = raw[start:finish]
    if b"\0" in response:
        raise BenchmarkError(f"embedded NUL in {begin} response")
    separator = b"" if response.endswith(b"\n") else b"\n"
    if not raw[finish:].startswith(separator + end_marker + b"\n"):
        raise BenchmarkError(f"length/framing drift in {begin} response block")
    try:
        return response.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkError(f"invalid UTF-8 in {begin} response") from exc


def extract_single_value(stderr: str, prefix: str, key: str) -> str:
    fields = extract_line_fields(stderr, prefix)
    if key not in fields:
        raise BenchmarkError(f"{prefix} omitted {key}")
    return fields[key]


def extract_output_ids(stderr: str, label: str = "output_ids") -> list[int]:
    matches = re.findall(
        rf"^{re.escape(label)}=([0-9,]+)$", stderr, flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise BenchmarkError(f"expected one {label} line, got {len(matches)}")
    try:
        values = [int(item) for item in matches[0].split(",")]
    except ValueError as exc:
        raise BenchmarkError("invalid output token IDs") from exc
    if not values:
        raise BenchmarkError("empty output token IDs")
    return values


def as_int(fields: dict[str, str], key: str) -> int:
    try:
        return int(fields[key])
    except (KeyError, ValueError) as exc:
        raise BenchmarkError(f"invalid integer metric {key}") from exc


def as_float(fields: dict[str, str], key: str) -> float:
    try:
        value = float(fields[key])
    except (KeyError, ValueError) as exc:
        raise BenchmarkError(f"invalid float metric {key}") from exc
    if not math.isfinite(value) or value < 0:
        raise BenchmarkError(f"non-finite or negative metric {key}")
    return value


def parse_success(case: dict[str, Any], stdout: str, stderr: str,
                  wall_s: float) -> dict[str, Any]:
    if not math.isfinite(wall_s) or wall_s <= 0:
        raise BenchmarkError("invalid external wall time")
    modality = case["modality"]
    if "runtime_ready=false" in stdout or "runtime_ready=false" in stderr:
        raise BenchmarkError("runner claimed runtime_ready=false")
    if "runtime_ready=true" not in stderr:
        raise BenchmarkError("runner omitted runtime_ready=true")

    if modality == "text":
        prompt = extract_line_fields(stderr, "GEMMA4_QA_PROMPT")
        start = extract_line_fields(stderr, "GEMMA4_QA_EXECUTION_START")
        done = extract_line_fields(stderr, "GEMMA4_QA_EXECUTION_DONE")
        if start.get("runtime_ready") != "true" or done.get("runtime_ready") != "true":
            raise BenchmarkError("text readiness marker drift")
        raw_rss = as_int(done, "maxrss_raw")
        unit = done.get("maxrss_unit")
        if unit == "bytes":
            peak_rss_bytes = raw_rss
        elif unit == "KiB":
            peak_rss_bytes = raw_rss * 1024
        else:
            raise BenchmarkError("unknown text maxrss unit")
        metrics = {
            "load_s": as_float(start, "load_s"),
            "image_s": None,
            "vision_s": None,
            "prefill_s": as_float(done, "prefill_s"),
            "first_token_ready_s": as_float(done, "first_token_ready_s"),
            "request_s": as_float(done, "total_s"),
            "native_total_s": None,
            "peak_rss_bytes": peak_rss_bytes,
        }
        prompt_sha256 = prompt.get("sha256")
        feature_sha256 = None
        image_tokens = None
        vision_pool_workers = None
        vision_pool_submissions = None
        vision_gpu_hmm = None
        vision_gpu_submissions = None
    else:
        prompt = extract_line_fields(stderr, "GEMMA4_MM_PROMPT")
        start = extract_line_fields(stderr, "GEMMA4_MM_EXECUTION_START")
        done = extract_line_fields(stderr, "GEMMA4_MM_EXECUTION_DONE")
        if start.get("runtime_ready") != "true" or done.get("runtime_ready") != "true":
            raise BenchmarkError("multimodal readiness marker drift")
        metrics = {
            "load_s": as_float(start, "load_s"),
            "image_s": as_float(start, "image_s"),
            "vision_s": as_float(start, "vision_s"),
            "prefill_s": as_float(done, "prefill_s"),
            "first_token_ready_s": as_float(done, "first_token_ready_s"),
            "request_s": as_float(done, "text_s"),
            "native_total_s": as_float(done, "native_total_s"),
            "peak_rss_bytes": as_int(done, "peak_rss_bytes"),
        }
        prompt_sha256 = prompt.get("sha256")
        feature_sha256 = start.get("feature_sha256")
        image_tokens = as_int(done, "image_tokens")
        vision_pool_workers = as_int(start, "vision_pool_workers")
        vision_pool_submissions = as_int(start, "vision_pool_submissions")
        vision_gpu_hmm = as_int(start, "vision_gpu_hmm")
        vision_gpu_submissions = as_int(start, "vision_gpu_submissions")
        if image_tokens != 64:
            raise BenchmarkError(f"unexpected image token count {image_tokens}")
        if vision_pool_workers < 1 or vision_pool_submissions < 0 or \
                vision_gpu_hmm not in (0, 1) or vision_gpu_submissions < 0 or \
                (vision_gpu_hmm == 1 and (
                    vision_gpu_submissions < 1 or vision_pool_submissions != 0)) or \
                (vision_gpu_hmm == 0 and sys.platform != "darwin" and
                    vision_pool_submissions < 1):
            raise BenchmarkError("multimodal vision executor did not engage")

    if start.get("kv_compare") != "1" or done.get("kv_compare") != "1":
        raise BenchmarkError("runner omitted KV-prefill comparison evidence")
    if done.get("expert_cache_mode") != "bounded-mmap-zero-copy":
        raise BenchmarkError("runner omitted explicit bounded-map cache evidence")
    memory = {
        "proof_validation_calls": as_int(done, "proof_validation_calls"),
        "detailed": bool(as_int(done, "memory_detailed")),
        "current_rss_bytes": as_int(done, "current_rss_bytes"),
        "resident_peak_bytes": as_int(done, "resident_peak_bytes"),
        "physical_footprint_bytes": as_int(done, "physical_footprint_bytes"),
        "internal_bytes": as_int(done, "internal_bytes"),
        "external_bytes": as_int(done, "external_bytes"),
        "reusable_bytes": as_int(done, "reusable_bytes"),
        "compressed_bytes": as_int(done, "compressed_bytes"),
        "minor_faults": as_int(done, "minor_faults"),
        "major_faults": as_int(done, "major_faults"),
        "input_blocks": as_int(done, "input_blocks"),
        "output_blocks": as_int(done, "output_blocks"),
        "expert_cache_mode": done["expert_cache_mode"],
        "expert_budget_bytes": as_int(done, "expert_cache_budget_bytes"),
        "expert_capacity_slots": as_int(done, "expert_cache_capacity_slots"),
        "expert_capacity_bytes": as_int(done, "expert_cache_capacity_bytes"),
        "expert_resident_slots": as_int(done, "expert_cache_resident_slots"),
        "expert_peak_slots": as_int(done, "expert_cache_peak_slots"),
        "expert_preload_enabled": as_int(done, "expert_preload_enabled"),
        "expert_preloaded_slots": as_int(done, "expert_preloaded_slots"),
        "expert_preload_layer_count": as_int(
            done, "expert_preload_layer_count",
        ),
        "expert_preload_layer_mask": as_int(done, "expert_preload_layer_mask"),
        "expert_preload_protected_slots": as_int(
            done, "expert_preload_protected_slots",
        ),
        "expert_preload_fetch_jobs": as_int(done, "expert_preload_fetch_jobs"),
        "expert_resident_logical_bytes": as_int(
            done, "expert_cache_resident_logical_bytes",
        ),
        "expert_peak_logical_bytes": as_int(
            done, "expert_cache_peak_logical_bytes",
        ),
        "expert_requests": as_int(done, "expert_cache_requests"),
        "expert_hits": as_int(done, "expert_cache_hits"),
        "expert_misses": as_int(done, "expert_cache_misses"),
        "expert_evictions": as_int(done, "expert_cache_evictions"),
        "expert_release_calls": as_int(done, "expert_cache_release_calls"),
        "expert_release_failures": as_int(done, "expert_cache_release_failures"),
        "expert_released_page_bytes": as_int(
            done, "expert_cache_released_page_bytes",
        ),
        "expert_mapped_bytes": as_int(done, "expert_cache_mapped_bytes"),
        "expert_peak_mapped_bytes": as_int(
            done, "expert_cache_peak_mapped_bytes",
        ),
        "expert_unmap_calls": as_int(done, "expert_cache_unmap_calls"),
        "expert_unmap_failures": as_int(
            done, "expert_cache_unmap_failures",
        ),
        "expert_unmapped_bytes": as_int(done, "expert_cache_unmapped_bytes"),
        "expert_prepare_calls": as_int(done, "expert_prepare_calls"),
        "expert_prepare_failures": as_int(done, "expert_prepare_failures"),
        "expert_prepared_bytes": as_int(done, "expert_prepared_bytes"),
        "expert_union_fetch_calls": as_int(done, "expert_union_fetch_calls"),
        "expert_union_fetch_experts": as_int(
            done, "expert_union_fetch_experts",
        ),
        "q4_pool_submissions": as_int(done, "q4_pool_submissions"),
        "q4_pool_fallbacks": as_int(done, "q4_pool_fallbacks"),
        "q4_pool_workers": as_int(done, "q4_pool_workers"),
        "q4_multi_pool_submissions": as_int(
            done, "q4_multi_pool_submissions",
        ),
        "q4_multi_pool_jobs": as_int(done, "q4_multi_pool_jobs"),
        "qkv_multi_pool_submissions": as_int(
            done, "qkv_multi_pool_submissions",
        ),
        "q4_multi_pool_fallbacks": as_int(done, "q4_multi_pool_fallbacks"),
        "q8_pool_submissions": as_int(done, "q8_pool_submissions"),
        "q8_pool_fallbacks": as_int(done, "q8_pool_fallbacks"),
        "expert_pool_submissions": as_int(done, "expert_pool_submissions"),
        "expert_pool_fallbacks": as_int(done, "expert_pool_fallbacks"),
        "gpu_dense_submissions": as_int(done, "gpu_dense_submissions"),
        "gpu_expert_gate_up_submissions": as_int(
            done, "gpu_expert_gate_up_submissions",
        ),
        "gpu_expert_down_submissions": as_int(
            done, "gpu_expert_down_submissions",
        ),
        "gpu_head_submissions": as_int(done, "gpu_head_submissions"),
        "gpu_decode_submissions": as_int(done, "gpu_decode_submissions"),
        "gpu_failures": as_int(done, "gpu_failures"),
        "gpu_weight_addressability": as_int(
            done, "gpu_weight_addressability",
        ),
        "gpu_weight_described_resources": as_int(
            done, "gpu_weight_described_resources",
        ),
        "gpu_weight_active_resources": as_int(
            done, "gpu_weight_active_resources",
        ),
        "gpu_weight_described_bytes": as_int(
            done, "gpu_weight_described_bytes",
        ),
        "gpu_weight_registered_bytes": as_int(
            done, "gpu_weight_registered_bytes",
        ),
        "gpu_weight_pageable_bytes": as_int(
            done, "gpu_weight_pageable_bytes",
        ),
        "gpu_weight_copied_bytes": as_int(done, "gpu_weight_copied_bytes"),
        "gpu_attention_batches": as_int(done, "gpu_attention_batches"),
        "gpu_attention_tasks": as_int(done, "gpu_attention_tasks"),
        "gpu_attention_kv_bytes": as_int(done, "gpu_attention_kv_bytes"),
        "prefill_ffn_arena_bytes": as_int(done, "prefill_ffn_arena_bytes"),
        "prefill_ffn_arena_calls": as_int(done, "prefill_ffn_arena_calls"),
        "expert_inflight_slots": as_int(done, "expert_inflight_slots"),
        "expert_peak_inflight_slots": as_int(
            done, "expert_peak_inflight_slots",
        ),
        "dense_retained_immutable": as_int(done, "dense_retained_immutable"),
        "dense_payload_bytes": as_int(done, "dense_payload_bytes"),
    }
    numeric_memory = [value for value in memory.values()
                      if isinstance(value, int) and not isinstance(value, bool)]
    text_cuda_pageable_hmm = memory["gpu_weight_addressability"] == 3
    if text_cuda_pageable_hmm:
        dispatch_valid = (
            memory["q4_pool_submissions"] == 0 and
            memory["q4_multi_pool_submissions"] == 0 and
            memory["qkv_multi_pool_submissions"] == 0 and
            memory["q8_pool_submissions"] == 0 and
            memory["expert_pool_submissions"] == 0 and
            memory["gpu_dense_submissions"] > 0 and
            memory["gpu_expert_gate_up_submissions"] > 0 and
            memory["gpu_expert_down_submissions"] > 0 and
            memory["gpu_decode_submissions"] > 0 and
            memory["gpu_failures"] == 0 and
            memory["gpu_weight_described_resources"] >= 4 and
            memory["gpu_weight_active_resources"] >= 4 and
            memory["gpu_weight_described_bytes"] > 0 and
            memory["gpu_weight_registered_bytes"] == 0 and
            memory["gpu_weight_pageable_bytes"] ==
                memory["gpu_weight_described_bytes"] and
            memory["gpu_weight_copied_bytes"] == 0 and
            memory["gpu_attention_tasks"] >= memory["gpu_attention_batches"] and
            memory["gpu_attention_kv_bytes"] > 0
        )
    else:
        dispatch_valid = (
            memory["q4_pool_submissions"] > 0 and
            memory["q4_multi_pool_submissions"] > 0 and
            memory["qkv_multi_pool_submissions"] > 0 and
            memory["q8_pool_submissions"] > 0 and
            memory["expert_pool_submissions"] > 0 and
            memory["gpu_failures"] == 0 and
            memory["gpu_weight_copied_bytes"] == 0
        )
    cuda_pageable_hmm = text_cuda_pageable_hmm or vision_gpu_hmm == 1
    expert_budget = memory["expert_budget_bytes"]
    expected_expert_slots = min(30 * 128, expert_budget // 3_345_408)
    platform_memory_valid = (
        memory["detailed"] and
        memory["physical_footprint_bytes"] > 0 and
        memory["current_rss_bytes"] == memory["internal_bytes"] +
            memory["external_bytes"] + memory["reusable_bytes"] and
        memory["physical_footprint_bytes"] >=
            memory["internal_bytes"] + memory["compressed_bytes"]
    ) or (
        not memory["detailed"] and
        all(memory[key] == 0 for key in (
            "physical_footprint_bytes", "internal_bytes", "external_bytes",
            "reusable_bytes", "compressed_bytes",
        ))
    )
    if not platform_memory_valid or memory["proof_validation_calls"] != 1 or \
            any(value < 0 for value in numeric_memory) or \
            memory["current_rss_bytes"] <= 0 or \
            memory["resident_peak_bytes"] <= 0 or \
            memory["minor_faults"] <= 0 or \
            expert_budget < 1_000_000_000 or expert_budget > 13_000_000_000 or \
            expert_budget % 1_000_000_000 != 0 or \
            memory["expert_capacity_slots"] != expected_expert_slots or \
            memory["expert_capacity_bytes"] != \
                expected_expert_slots * 3_345_408 or \
            memory["expert_resident_slots"] > memory["expert_capacity_slots"] or \
            memory["expert_peak_slots"] > memory["expert_capacity_slots"] or \
            memory["expert_preload_enabled"] not in (0, 1) or \
            (memory["expert_preload_enabled"] == 0 and
                any(memory[key] != 0 for key in (
                    "expert_preloaded_slots", "expert_preload_layer_count",
                    "expert_preload_layer_mask",
                    "expert_preload_protected_slots",
                    "expert_preload_fetch_jobs",
                ))) or \
            (memory["expert_preload_enabled"] == 1 and
                (memory["expert_preload_layer_count"] < 1 or
                 memory["expert_preload_layer_count"] > 30 or
                 bin(memory["expert_preload_layer_mask"]).count("1") !=
                    memory["expert_preload_layer_count"] or
                 memory["expert_preloaded_slots"] !=
                    memory["expert_preload_layer_count"] * 128 or
                 memory["expert_preload_fetch_jobs"] !=
                    memory["expert_preloaded_slots"] or
                 memory["expert_preload_protected_slots"] not in
                    (0, memory["expert_preloaded_slots"]) or
                 (memory["expert_preload_protected_slots"] == 0 and
                    (memory["expert_preload_layer_count"] != 30 or
                     memory["expert_capacity_slots"] != 30 * 128)))) or \
            memory["expert_resident_logical_bytes"] != \
                memory["expert_resident_slots"] * 3_345_408 or \
            memory["expert_peak_logical_bytes"] != \
                memory["expert_peak_slots"] * 3_345_408 or \
            memory["expert_requests"] != \
                memory["expert_hits"] + memory["expert_misses"] or \
            memory["expert_evictions"] != memory["expert_release_calls"] or \
            memory["expert_release_failures"] != 0 or \
            (memory["expert_evictions"] > 0 and
                memory["expert_released_page_bytes"] <= 0) or \
            (memory["expert_evictions"] == 0 and
                memory["expert_released_page_bytes"] != 0) or \
            memory["expert_mapped_bytes"] <= 0 or \
            memory["expert_mapped_bytes"] > memory["expert_peak_mapped_bytes"] or \
            memory["expert_peak_mapped_bytes"] > \
                memory["expert_budget_bytes"] + \
                memory["expert_capacity_slots"] * 65_536 or \
            memory["expert_unmap_calls"] != memory["expert_evictions"] or \
            memory["expert_unmap_failures"] != 0 or \
            (memory["expert_evictions"] > 0 and
                memory["expert_unmapped_bytes"] <= 0) or \
            (memory["expert_evictions"] == 0 and
                memory["expert_unmapped_bytes"] != 0) or \
            (not text_cuda_pageable_hmm and
                memory["expert_union_fetch_calls"] <= 0) or \
            memory["expert_union_fetch_experts"] < \
                memory["expert_union_fetch_calls"] or \
            not dispatch_valid or \
            memory["q4_pool_fallbacks"] != 0 or \
            memory["q4_pool_workers"] < 1 or \
            memory["q4_pool_workers"] > 8 or \
            memory["q4_multi_pool_jobs"] < memory["q4_multi_pool_submissions"] or \
            memory["q4_multi_pool_fallbacks"] != 0 or \
            memory["q8_pool_fallbacks"] != 0 or \
            memory["expert_pool_fallbacks"] != 0 or \
            memory["prefill_ffn_arena_bytes"] <= 0 or \
            (not text_cuda_pageable_hmm and
                memory["prefill_ffn_arena_calls"] <= 0) or \
            memory["expert_inflight_slots"] != 0 or \
            memory["expert_peak_inflight_slots"] < 1 or \
            memory["expert_peak_inflight_slots"] > 128 or \
            memory["dense_retained_immutable"] != 1 or \
            memory["dense_payload_bytes"] <= 0:
        raise BenchmarkError("invalid bounded-residency memory evidence")
    if metrics["peak_rss_bytes"] <= 0:
        raise BenchmarkError("native runner did not report a positive peak RSS")
    if not prompt_sha256 or not HEX64.fullmatch(prompt_sha256):
        raise BenchmarkError("missing prompt SHA-256")
    if feature_sha256 is not None and not HEX64.fullmatch(feature_sha256):
        raise BenchmarkError("missing feature SHA-256")

    response_bytes = as_int(done, "response_bytes")
    warm_response_bytes = as_int(done, "warm_response_bytes")
    response = extract_response(stdout, modality, response_bytes)
    warm_response = extract_response(
        stdout, modality, warm_response_bytes, warm=True,
    )
    if response_bytes != len(response.encode("utf-8")) or \
            warm_response_bytes != len(warm_response.encode("utf-8")):
        raise BenchmarkError("response byte-count drift")
    response_hash = sha256_bytes(response.encode("utf-8"))
    warm_response_hash = sha256_bytes(warm_response.encode("utf-8"))
    normalized = normalize_response(response)
    warm_normalized = normalize_response(warm_response)
    semantic_pass = normalized in case["accepted_normalized"]
    output_ids = extract_output_ids(stderr)
    warm_output_ids = extract_output_ids(stderr, "warm_output_ids")
    if response != warm_response or output_ids != warm_output_ids:
        raise BenchmarkError("cold/warm response or output-token drift")

    prompt_tokens = as_int(done, "prompt_tokens")
    state_envelope = done.get("state_envelope")
    stop_commit_steps = as_int(done, "stop_commit_steps")
    package_proof = as_int(done, "package_proof")
    if state_envelope != "kv-restore-proof" or stop_commit_steps != 0 or \
            package_proof != 1:
        raise BenchmarkError("state/proof envelope admission drift")
    kv_prefill_position = as_int(done, "kv_prefill_position")
    kv_snapshot_version = as_int(done, "kv_snapshot_version")
    kv_snapshot_header_bytes = as_int(done, "kv_snapshot_header_bytes")
    kv_snapshot_bytes_per_token = as_int(done, "kv_snapshot_bytes_per_token")
    kv_snapshot_bytes = as_int(done, "kv_snapshot_bytes")
    kv_poisoned_bytes = as_int(done, "kv_poisoned_bytes")
    state_capture_s = as_float(done, "state_capture_s")
    kv_capture_s = as_float(done, "kv_capture_s")
    logit_capture_s = as_float(done, "logit_capture_s")
    kv_poison_s = as_float(done, "kv_poison_s")
    kv_restore_s = as_float(done, "kv_restore_s")
    logit_restore_s = as_float(done, "logit_restore_s")
    state_restore_s = as_float(done, "state_restore_s")
    cold_first_token_ready_s = as_float(done, "cold_first_token_ready_s")
    cold_request_s = as_float(done, "cold_request_s")
    warm_first_token_ready_s = as_float(done, "warm_first_token_ready_s")
    warm_request_s = as_float(done, "warm_total_s")
    expected_snapshot_bytes = (
        KV_SNAPSHOT_HEADER_BYTES + prompt_tokens * KV_SNAPSHOT_BYTES_PER_TOKEN
    )
    if prompt_tokens < 1 or kv_prefill_position != prompt_tokens or \
            kv_snapshot_version != KV_SNAPSHOT_VERSION or \
            kv_snapshot_header_bytes != KV_SNAPSHOT_HEADER_BYTES or \
            kv_snapshot_bytes_per_token != KV_SNAPSHOT_BYTES_PER_TOKEN or \
            kv_snapshot_bytes != expected_snapshot_bytes or \
            kv_poisoned_bytes != prompt_tokens * KV_SNAPSHOT_BYTES_PER_TOKEN:
        raise BenchmarkError("KV snapshot geometry, version, or poison-size drift")
    if done.get("kv_poisoned") != "1" or \
            done.get("kv_restore_verified") != "1":
        raise BenchmarkError("missing destructive KV-restore proof")
    positive_timings = (
        state_capture_s, kv_capture_s, logit_capture_s, kv_poison_s,
        kv_restore_s, logit_restore_s, state_restore_s,
        cold_first_token_ready_s, cold_request_s,
        warm_first_token_ready_s, warm_request_s,
    )
    if any(value <= 0 for value in positive_timings):
        raise BenchmarkError("non-positive cold/KV-warm timing evidence")
    tolerance = 3e-6
    if cold_request_s < cold_first_token_ready_s or \
            warm_request_s < warm_first_token_ready_s or \
            warm_first_token_ready_s < state_restore_s or \
            state_capture_s + tolerance < kv_capture_s + logit_capture_s or \
            abs(state_restore_s - (kv_restore_s + logit_restore_s)) > tolerance:
        raise BenchmarkError("inconsistent cold/KV-warm timing evidence")
    metrics.update({
        "cold_first_token_ready_s": cold_first_token_ready_s,
        "cold_request_s": cold_request_s,
        "kv_snapshot_version": kv_snapshot_version,
        "kv_snapshot_header_bytes": kv_snapshot_header_bytes,
        "kv_snapshot_bytes_per_token": kv_snapshot_bytes_per_token,
        "kv_snapshot_bytes": kv_snapshot_bytes,
        "kv_poisoned_bytes": kv_poisoned_bytes,
        "state_capture_s": state_capture_s,
        "kv_capture_s": kv_capture_s,
        "logit_capture_s": logit_capture_s,
        "kv_poison_s": kv_poison_s,
        "kv_restore_s": kv_restore_s,
        "logit_restore_s": logit_restore_s,
        "state_restore_s": state_restore_s,
        "warm_first_token_ready_s": warm_first_token_ready_s,
        "warm_request_s": warm_request_s,
    })
    output_steps = as_int(done, "output_steps")
    if output_steps != len(output_ids) or output_steps != len(warm_output_ids):
        raise BenchmarkError("cold/warm output token count drift")
    decode_s = max(
        metrics["request_s"] - metrics["first_token_ready_s"], 0.0,
    )
    post_restore_generation_s = max(
        warm_request_s - warm_first_token_ready_s, 0.0,
    )
    decode_transitions = max(output_steps - 1, 0)
    speeds = {
        "prefill_tokens_s": prompt_tokens / metrics["prefill_s"]
            if metrics["prefill_s"] > 0 else None,
        "post_first_decode_steps_s": decode_transitions / decode_s
            if decode_s > 0 else None,
        "external_output_steps_s": output_steps / wall_s if wall_s > 0 else None,
        "vision_features_s": image_tokens / metrics["vision_s"]
            if image_tokens and metrics["vision_s"] and metrics["vision_s"] > 0 else None,
        "kv_warm_speedup": cold_request_s / warm_request_s,
        "kv_warm_first_token_ready_speedup": (
            cold_first_token_ready_s / warm_first_token_ready_s
        ),
        "kv_warm_time_saved_s": cold_request_s - warm_request_s,
        "post_restore_generation_steps_s": (
            decode_transitions / post_restore_generation_s
            if post_restore_generation_s > 0 else None
        ),
    }
    return {
        "status": 0,
        "error": None,
        "wall_s": wall_s,
        "runtime_ready": True,
        "response": response,
        "response_bytes": response_bytes,
        "response_normalized": normalized,
        "response_sha256": response_hash,
        "semantic_pass": semantic_pass,
        "warm_response": warm_response,
        "warm_response_bytes": warm_response_bytes,
        "warm_response_normalized": warm_normalized,
        "warm_response_sha256": warm_response_hash,
        "kv_warm_pass": response == warm_response and normalized == warm_normalized,
        "kv_poisoned": True,
        "kv_restore_verified": True,
        "state_envelope": state_envelope,
        "stop_commit_steps": stop_commit_steps,
        "package_proof": package_proof,
        "prompt_sha256": prompt_sha256,
        "feature_sha256": feature_sha256,
        "output_ids": output_ids,
        "output_ids_sha256": sha256_bytes(
            ",".join(str(value) for value in output_ids).encode("ascii")
        ),
        "warm_output_ids": warm_output_ids,
        "warm_output_ids_sha256": sha256_bytes(
            ",".join(str(value) for value in warm_output_ids).encode("ascii")
        ),
        "prompt_tokens": prompt_tokens,
        "kv_prefill_position": kv_prefill_position,
        "image_tokens": image_tokens,
        "vision_pool_workers": vision_pool_workers,
        "vision_pool_submissions": vision_pool_submissions,
        "vision_gpu_hmm": vision_gpu_hmm,
        "vision_gpu_submissions": vision_gpu_submissions,
        "cuda_pageable_hmm": cuda_pageable_hmm,
        "output_steps": output_steps,
        "stop_token": as_int(done, "stop_token"),
        "metrics": metrics,
        "memory": memory,
        "speed": speeds,
    }


def command_for_case(args: argparse.Namespace, manifest: dict[str, Any],
                     case: dict[str, Any]) -> list[str]:
    execution = manifest["execution"]
    if case["modality"] == "text":
        command = [
            str(args.python), str(args.text_wrapper),
            "--source-dir", str(args.source_dir),
            "--pool", str(args.pool),
            "--runner", str(args.text_runner),
            "--ctx", str(execution["ctx"]),
            "--gen", str(execution["gen"]),
            "--workers", str(execution["workers"]),
            "--question", case["question"],
        ]
    else:
        image_path = ROOT / case["image"]
        command = [
            str(args.python), str(args.multimodal_wrapper),
            "--source-dir", str(args.source_dir),
            "--pool", str(args.pool),
            "--runner", str(args.multimodal_runner),
            "--image", str(image_path),
            "--ctx", str(execution["ctx"]),
            "--gen", str(execution["gen"]),
            "--workers", str(execution["workers"]),
            "--question", case["question"],
        ]
    if args.receipt is not None:
        command[6:6] = ["--receipt", str(args.receipt)]
    if args.auth_receipt is not None:
        command[6:6] = ["--auth-receipt", str(args.auth_receipt)]
    command.extend(("--compare-kv-warm", "--package-proof"))
    return command


def validate_inputs(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    paths = {
        "source directory": args.source_dir,
        "expert pool": args.pool,
        "Python interpreter": args.python,
        "text wrapper": args.text_wrapper,
        "multimodal wrapper": args.multimodal_wrapper,
        "text runner": args.text_runner,
        "multimodal runner": args.multimodal_runner,
    }
    for label, path in paths.items():
        if path is None or not path.exists():
            raise BenchmarkError(f"{label} does not exist: {path}")
    if not args.source_dir.is_dir():
        raise BenchmarkError(f"source directory is not a directory: {args.source_dir}")
    if args.receipt is not None and not args.receipt.is_file():
        raise BenchmarkError(f"receipt does not exist: {args.receipt}")
    if args.auth_receipt is None or not args.auth_receipt.is_file():
        raise BenchmarkError(
            f"persisted package authentication receipt does not exist: {args.auth_receipt}"
        )
    for case in manifest["cases"]:
        if case["modality"] != "image":
            continue
        image = ROOT / case["image"]
        if not image.is_file():
            raise BenchmarkError(f"image does not exist: {image}")
        actual = sha256_file(image)
        if actual != case["image_sha256"]:
            raise BenchmarkError(
                f"image identity drift for {case['id']}: {actual}"
            )


def host_info() -> dict[str, Any]:
    physical_bytes = None
    try:
        if sys.platform == "darwin":
            physical_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True,
            ).strip())
        elif hasattr(os, "sysconf"):
            physical_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, subprocess.SubprocessError, ValueError):
        physical_bytes = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "physical_memory_bytes": physical_bytes,
        "physical_memory_gb": physical_bytes / 1_000_000_000
            if physical_bytes is not None else None,
    }


def git_info() -> dict[str, Any]:
    def git(*parts: str) -> str:
        return subprocess.check_output(
            ["git", *parts], cwd=ROOT, text=True,
        ).strip()
    try:
        return {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "branch": None, "dirty": None}


def empty_failure(status: int, wall_s: float, error: str) -> dict[str, Any]:
    return {
        "status": status,
        "error": error,
        "wall_s": wall_s,
        "runtime_ready": None,
        "response": "",
        "response_bytes": 0,
        "response_normalized": "",
        "response_sha256": sha256_bytes(b""),
        "semantic_pass": False,
        "warm_response": "",
        "warm_response_bytes": 0,
        "warm_response_normalized": "",
        "warm_response_sha256": sha256_bytes(b""),
        "kv_warm_pass": False,
        "kv_poisoned": False,
        "kv_restore_verified": False,
        "state_envelope": None,
        "stop_commit_steps": None,
        "package_proof": None,
        "prompt_sha256": None,
        "feature_sha256": None,
        "output_ids": [],
        "output_ids_sha256": None,
        "warm_output_ids": [],
        "warm_output_ids_sha256": None,
        "prompt_tokens": None,
        "kv_prefill_position": None,
        "image_tokens": None,
        "vision_pool_workers": None,
        "vision_pool_submissions": None,
        "output_steps": None,
        "stop_token": None,
        "metrics": {},
        "memory": {},
        "speed": {},
    }


def run_benchmark(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(args, manifest)
    identity_paths = {
        "orchestrator": Path(__file__).resolve(),
        "text_wrapper": args.text_wrapper.resolve(),
        "multimodal_wrapper": args.multimodal_wrapper.resolve(),
        "text_runner": args.text_runner.resolve(),
        "multimodal_runner": args.multimodal_runner.resolve(),
    }
    identity_before = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in identity_paths.items()
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    runs: list[dict[str, Any]] = []
    repeat = manifest["execution"]["repeat"]
    total = len(manifest["cases"]) * repeat
    ordinal = 0
    abort = False
    for case in manifest["cases"]:
        for repetition in range(1, repeat + 1):
            ordinal += 1
            print(
                f"[{ordinal}/{total}] {case['id']} repeat={repetition} START",
                file=sys.stderr,
                flush=True,
            )
            command = command_for_case(args, manifest, case)
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.timeout_s,
                    check=False,
                )
                wall_s = time.monotonic() - started
                status = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                if status == 0:
                    try:
                        parsed = parse_success(case, stdout, stderr, wall_s)
                    except BenchmarkError as exc:
                        parsed = empty_failure(1, wall_s, str(exc))
                        status = 1
                else:
                    parsed = empty_failure(status, wall_s, "wrapper or runner failed")
            except subprocess.TimeoutExpired as exc:
                wall_s = time.monotonic() - started
                status = 124
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
                parsed = empty_failure(status, wall_s, f"timeout after {args.timeout_s}s")
            stem = f"{case['id']}-r{repetition}"
            stdout_path = args.artifacts / f"{stem}.stdout"
            stderr_path = args.artifacts / f"{stem}.stderr"
            stdout_path.write_text(stdout)
            stderr_path.write_text(stderr)
            parsed.update({
                "case_id": case["id"],
                "task": case["task"],
                "modality": case["modality"],
                "repeat": repetition,
                "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
                "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
                "stdout_artifact": str(stdout_path),
                "stderr_artifact": str(stderr_path),
            })
            runs.append(parsed)
            rss = parsed.get("metrics", {}).get("peak_rss_bytes")
            rss_text = f" rss={rss / 1_000_000_000:.3f}GB" if rss else ""
            response = parsed.get("response", "").replace("\n", "\\n")
            print(
                f"[{ordinal}/{total}] {case['id']} repeat={repetition} "
                f"status={status} wall={wall_s:.3f}s{rss_text} "
                f"response={json.dumps(response, ensure_ascii=True)}",
                file=sys.stderr,
                flush=True,
            )
            if status != 0:
                abort = True
                break
        if abort:
            break
    try:
        identity_after = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in identity_paths.items()
        }
    except OSError:
        identity_after = {}
    result = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "model": manifest["model"],
        "execution": manifest["execution"],
        "host": host_info(),
        "git": git_info(),
        "execution_identity": {
            "before": identity_before,
            "after": identity_after,
            "stable": identity_before == identity_after,
        },
        "inputs": {
            "source_dir": str(args.source_dir),
            "source_bytes": 15_373_375_402,
            "pool": str(args.pool),
            "pool_bytes": args.pool.stat().st_size,
            "receipt": str(args.receipt) if args.receipt else str(args.pool) + ".import.json",
            "auth_receipt": str(args.auth_receipt),
            "auth_receipt_sha256": sha256_file(args.auth_receipt),
            "text_runner": str(args.text_runner),
            "multimodal_runner": str(args.multimodal_runner),
            "artifacts": str(args.artifacts),
        },
        "cases": manifest["cases"],
        "runs": runs,
        "candidate_execution_pass": False,
        "baseline_enforced": False,
        "overall_pass": False,
        "validation_errors": [],
    }
    validate_report(result, manifest, require_baselines=False)
    return result


def run_values(runs: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for run in runs:
        value: Any = run
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and \
                math.isfinite(float(value)):
            values.append(float(value))
    return values


def aggregate(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    if any(not math.isfinite(value) for value in values):
        raise BenchmarkError("non-finite aggregate input")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def validate_report(report: dict[str, Any], manifest: dict[str, Any],
                    *, require_baselines: bool) -> bool:
    def finite_tree(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        return True

    def positive_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and \
            math.isfinite(float(value)) and value > 0

    def same_number(value: Any, expected: float) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and \
            math.isfinite(float(value)) and math.isclose(
                float(value), expected, rel_tol=1e-12, abs_tol=1e-12
            )

    errors: list[str] = []
    repeat = manifest["execution"]["repeat"]
    runs = report.get("runs")
    all_candidate = True
    if not finite_tree(report):
        errors.append("report contains a non-finite number")
        all_candidate = False
    if report.get("schema") != REPORT_SCHEMA:
        errors.append("report schema drift")
        all_candidate = False
    if report.get("execution") != manifest["execution"]:
        errors.append("report execution contract drift")
        all_candidate = False
    if report.get("cases") != manifest["cases"]:
        errors.append("report case contract drift")
        all_candidate = False
    host = report.get("host")
    if not isinstance(host, dict) or \
            not positive_number(host.get("physical_memory_bytes")) or \
            not same_number(
                host.get("physical_memory_gb"),
                float(host.get("physical_memory_bytes", 0)) / 1_000_000_000,
            ):
        errors.append("report host-memory evidence drift")
        all_candidate = False
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or \
            inputs.get("source_bytes") != 15_373_375_402 or \
            inputs.get("pool_bytes") != 12_846_366_744:
        errors.append("report authenticated-input size drift")
        all_candidate = False
    identity = report.get("execution_identity")
    expected_identity = {
        "orchestrator", "text_wrapper", "multimodal_wrapper",
        "text_runner", "multimodal_runner",
    }
    before = identity.get("before") if isinstance(identity, dict) else None
    after = identity.get("after") if isinstance(identity, dict) else None
    valid_identity = isinstance(before, dict) and isinstance(after, dict) and \
        set(before) == expected_identity and set(after) == expected_identity and \
        identity.get("stable") is True and before == after
    if valid_identity and isinstance(before, dict):
        valid_identity = all(
            isinstance(item, dict) and isinstance(item.get("path"), str) and
            HEX64.fullmatch(str(item.get("sha256", ""))) is not None
            for item in before.values()
        )
    if not valid_identity:
        errors.append("execution binary/script identity drift")
        all_candidate = False
    if not isinstance(runs, list) or len(runs) != len(manifest["cases"]) * repeat:
        errors.append("report must contain exactly twelve runs")
        all_candidate = False
        runs = runs if isinstance(runs, list) else []

    for case in manifest["cases"]:
        case_runs = [run for run in runs if isinstance(run, dict) and
                     run.get("case_id") == case["id"]]
        if len(case_runs) != repeat or [run.get("repeat") for run in case_runs] != list(
                range(1, repeat + 1)):
            errors.append(f"{case['id']}: expected exactly {repeat} ordered runs")
            all_candidate = False
            continue
        for run in case_runs:
            label = f"{case['id']} r{run.get('repeat')}"
            if run.get("status") != 0:
                errors.append(f"{label}: nonzero status")
                all_candidate = False
            if run.get("runtime_ready") is not True:
                errors.append(f"{label}: readiness drift")
                all_candidate = False

            response = run.get("response")
            if not isinstance(response, str):
                errors.append(f"{label}: missing response")
                all_candidate = False
                response = ""
            normalized = normalize_response(response)
            response_hash = sha256_bytes(response.encode("utf-8"))
            if run.get("response_normalized") != normalized or \
                    run.get("response_sha256") != response_hash or \
                    run.get("response_bytes") != len(response.encode("utf-8")):
                errors.append(f"{label}: response evidence drift")
                all_candidate = False
            semantic_pass = normalized in case["accepted_normalized"]
            if run.get("semantic_pass") is not semantic_pass or not semantic_pass:
                errors.append(f"{label}: semantic mismatch")
                all_candidate = False

            warm_response = run.get("warm_response")
            if not isinstance(warm_response, str):
                errors.append(f"{label}: missing KV-warm response")
                all_candidate = False
                warm_response = ""
            warm_normalized = normalize_response(warm_response)
            warm_hash = sha256_bytes(warm_response.encode("utf-8"))
            if run.get("warm_response_normalized") != warm_normalized or \
                    run.get("warm_response_sha256") != warm_hash or \
                    run.get("warm_response_bytes") != len(
                        warm_response.encode("utf-8")
                    ) or run.get("kv_warm_pass") is not True or \
                    warm_response != response:
                errors.append(f"{label}: KV-warm response drift")
                all_candidate = False
            if run.get("kv_poisoned") is not True or \
                    run.get("kv_restore_verified") is not True:
                errors.append(f"{label}: missing destructive KV-restore proof")
                all_candidate = False
            if run.get("state_envelope") != "kv-restore-proof" or \
                    run.get("stop_commit_steps") != 0 or \
                    run.get("package_proof") != 1:
                errors.append(f"{label}: state/proof envelope admission drift")
                all_candidate = False

            output_ids = run.get("output_ids")
            if not isinstance(output_ids, list) or not output_ids or any(
                    not isinstance(value, int) for value in output_ids):
                errors.append(f"{label}: invalid output IDs")
                all_candidate = False
                output_ids = []
            else:
                output_hash = sha256_bytes(
                    ",".join(str(value) for value in output_ids).encode("ascii")
                )
                if run.get("output_ids_sha256") != output_hash or \
                        run.get("output_steps") != len(output_ids):
                    errors.append(f"{label}: output-ID evidence drift")
                    all_candidate = False

            warm_output_ids = run.get("warm_output_ids")
            if not isinstance(warm_output_ids, list) or not warm_output_ids or any(
                    not isinstance(value, int) for value in warm_output_ids):
                errors.append(f"{label}: invalid KV-warm output IDs")
                all_candidate = False
            else:
                warm_output_hash = sha256_bytes(
                    ",".join(str(value) for value in warm_output_ids).encode("ascii")
                )
                if run.get("warm_output_ids_sha256") != warm_output_hash or \
                        warm_output_ids != output_ids:
                    errors.append(f"{label}: KV-warm output-ID drift")
                    all_candidate = False

            if not positive_number(run.get("wall_s")):
                errors.append(f"{label}: invalid wall time")
                all_candidate = False
            prompt_tokens = run.get("prompt_tokens")
            if not isinstance(prompt_tokens, int) or prompt_tokens < 1 or \
                    run.get("kv_prefill_position") != prompt_tokens:
                errors.append(f"{label}: KV-prefill position drift")
                all_candidate = False

            memory = run.get("memory")
            required_memory_fields = {
                "current_rss_bytes", "resident_peak_bytes",
                "physical_footprint_bytes", "internal_bytes",
                "external_bytes", "reusable_bytes", "compressed_bytes",
                "minor_faults", "major_faults", "input_blocks",
                "output_blocks",
            }
            memory_ints = [value for value in memory.values()
                           if isinstance(value, int) and not isinstance(value, bool)] \
                if isinstance(memory, dict) else []
            expert_budget = memory.get("expert_budget_bytes", 0) \
                if isinstance(memory, dict) else 0
            expected_expert_slots = min(
                30 * 128, expert_budget // 3_345_408,
            )
            platform_memory_valid = isinstance(memory, dict) and (
                (
                    memory.get("detailed") is True and
                    positive_number(memory.get("physical_footprint_bytes")) and
                    memory.get("current_rss_bytes") ==
                        memory.get("internal_bytes", -1) +
                        memory.get("external_bytes", -1) +
                        memory.get("reusable_bytes", -1) and
                    memory.get("physical_footprint_bytes", -1) >=
                        memory.get("internal_bytes", 0) +
                        memory.get("compressed_bytes", 0)
                ) or (
                    memory.get("detailed") is False and
                    all(memory.get(key) == 0 for key in (
                        "physical_footprint_bytes", "internal_bytes",
                        "external_bytes", "reusable_bytes", "compressed_bytes",
                    ))
                )
            )
            memory_valid = isinstance(memory, dict) and \
                required_memory_fields.issubset(memory) and \
                platform_memory_valid and \
                memory.get("proof_validation_calls") == 1 and \
                memory.get("expert_cache_mode") == "bounded-mmap-zero-copy" and \
                1_000_000_000 <= expert_budget <= 13_000_000_000 and \
                expert_budget % 1_000_000_000 == 0 and \
                memory.get("expert_capacity_slots") == expected_expert_slots and \
                memory.get("expert_capacity_bytes") == \
                    expected_expert_slots * 3_345_408 and \
                all(value >= 0 for value in memory_ints) and \
                positive_number(memory.get("current_rss_bytes")) and \
                positive_number(memory.get("resident_peak_bytes")) and \
                all(isinstance(memory.get(key), int) and memory[key] >= 0
                    for key in required_memory_fields) and \
                positive_number(memory.get("minor_faults")) and \
                isinstance(memory.get("major_faults"), int) and \
                isinstance(memory.get("input_blocks"), int) and \
                isinstance(memory.get("output_blocks"), int) and \
                memory.get("current_rss_bytes", 1) <= \
                    memory.get("resident_peak_bytes", 0) and \
                isinstance(memory.get("expert_resident_slots"), int) and \
                0 <= memory["expert_resident_slots"] <= expected_expert_slots and \
                isinstance(memory.get("expert_peak_slots"), int) and \
                0 <= memory["expert_peak_slots"] <= expected_expert_slots and \
                memory.get("expert_preload_enabled") in (0, 1) and \
                ((memory.get("expert_preload_enabled") == 0 and
                  all(memory.get(key) == 0 for key in (
                      "expert_preloaded_slots", "expert_preload_layer_count",
                      "expert_preload_layer_mask",
                      "expert_preload_protected_slots",
                      "expert_preload_fetch_jobs",
                  ))) or
                 (memory.get("expert_preload_enabled") == 1 and
                  isinstance(memory.get("expert_preload_layer_count"), int) and
                  1 <= memory["expert_preload_layer_count"] <= 30 and
                  bin(memory.get("expert_preload_layer_mask", 0)).count("1") ==
                      memory["expert_preload_layer_count"] and
                  memory.get("expert_preloaded_slots") ==
                      memory["expert_preload_layer_count"] * 128 and
                  memory.get("expert_preload_fetch_jobs") ==
                      memory["expert_preloaded_slots"] and
                  memory.get("expert_preload_protected_slots") in
                      (0, memory["expert_preloaded_slots"]) and
                  (memory["expert_preload_protected_slots"] != 0 or
                   (memory["expert_preload_layer_count"] == 30 and
                    memory.get("expert_capacity_slots") == 30 * 128)))) and \
                memory.get("expert_resident_logical_bytes") == \
                    memory.get("expert_resident_slots", -1) * 3_345_408 and \
                memory.get("expert_peak_logical_bytes") == \
                    memory.get("expert_peak_slots", -1) * 3_345_408 and \
                memory.get("expert_requests") == \
                    memory.get("expert_hits", -1) + memory.get("expert_misses", -1) and \
                memory.get("expert_evictions") == \
                    memory.get("expert_release_calls") and \
                memory.get("expert_release_failures") == 0 and \
                ((memory.get("expert_evictions", 0) > 0 and
                  positive_number(memory.get("expert_released_page_bytes"))) or
                 (memory.get("expert_evictions") == 0 and
                  memory.get("expert_released_page_bytes") == 0)) and \
                positive_number(memory.get("expert_mapped_bytes")) and \
                memory.get("expert_mapped_bytes", 1) <= \
                    memory.get("expert_peak_mapped_bytes", 0) and \
                memory.get("expert_peak_mapped_bytes", 0) <= \
                    memory.get("expert_budget_bytes", 0) + \
                    memory.get("expert_capacity_slots", 0) * 65_536 and \
                memory.get("expert_unmap_calls") == \
                    memory.get("expert_evictions") and \
                memory.get("expert_unmap_failures") == 0 and \
                ((memory.get("expert_evictions", 0) > 0 and
                  positive_number(memory.get("expert_unmapped_bytes"))) or
                 (memory.get("expert_evictions") == 0 and
                  memory.get("expert_unmapped_bytes") == 0)) and \
                positive_number(memory.get("expert_union_fetch_calls")) and \
                isinstance(memory.get("expert_union_fetch_experts"), int) and \
                memory.get("expert_union_fetch_experts", 0) >= \
                    memory.get("expert_union_fetch_calls", 1) and \
                positive_number(memory.get("q4_pool_submissions")) and \
                memory.get("q4_pool_fallbacks") == 0 and \
                isinstance(memory.get("q4_pool_workers"), int) and \
                1 <= memory.get("q4_pool_workers", 0) <= 8 and \
                positive_number(memory.get("q4_multi_pool_submissions")) and \
                isinstance(memory.get("q4_multi_pool_jobs"), int) and \
                memory.get("q4_multi_pool_jobs", 0) >= \
                    memory.get("q4_multi_pool_submissions", 1) and \
                positive_number(memory.get("qkv_multi_pool_submissions")) and \
                memory.get("q4_multi_pool_fallbacks") == 0 and \
                positive_number(memory.get("q8_pool_submissions")) and \
                memory.get("q8_pool_fallbacks") == 0 and \
                positive_number(memory.get("expert_pool_submissions")) and \
                memory.get("expert_pool_fallbacks") == 0 and \
                positive_number(memory.get("prefill_ffn_arena_bytes")) and \
                positive_number(memory.get("prefill_ffn_arena_calls")) and \
                memory.get("expert_inflight_slots") == 0 and \
                isinstance(memory.get("expert_peak_inflight_slots"), int) and \
                1 <= memory.get("expert_peak_inflight_slots", 0) <= 128 and \
                memory.get("dense_retained_immutable") == 1 and \
                positive_number(memory.get("dense_payload_bytes"))
            if not memory_valid:
                errors.append(f"{label}: invalid bounded-map memory evidence")
                all_candidate = False

            metrics = run.get("metrics")
            positive_metrics = (
                "peak_rss_bytes", "first_token_ready_s", "request_s",
                "cold_first_token_ready_s", "cold_request_s",
                "kv_snapshot_bytes", "kv_poisoned_bytes",
                "state_capture_s", "kv_capture_s", "logit_capture_s",
                "kv_poison_s", "kv_restore_s", "logit_restore_s",
                "state_restore_s", "warm_first_token_ready_s",
                "warm_request_s",
            )
            metric_values_finite = isinstance(metrics, dict) and all(
                value is None or (
                    isinstance(value, (int, float)) and
                    not isinstance(value, bool) and
                    math.isfinite(float(value)) and value >= 0
                )
                for value in metrics.values()
            )
            if not metric_values_finite or any(
                    not positive_number(metrics.get(key)) for key in positive_metrics):
                errors.append(f"{label}: invalid KV timing or resource evidence")
                all_candidate = False
            else:
                if not isinstance(memory, dict) or \
                        metrics["peak_rss_bytes"] != memory.get("resident_peak_bytes"):
                    errors.append(f"{label}: peak RSS attribution drift")
                    all_candidate = False
                cold_request = float(metrics["cold_request_s"])
                cold_first = float(metrics["cold_first_token_ready_s"])
                warm_request = float(metrics["warm_request_s"])
                warm_first = float(metrics["warm_first_token_ready_s"])
                kv_capture = float(metrics["kv_capture_s"])
                logit_capture = float(metrics["logit_capture_s"])
                state_capture = float(metrics["state_capture_s"])
                kv_restore = float(metrics["kv_restore_s"])
                logit_restore = float(metrics["logit_restore_s"])
                state_restore = float(metrics["state_restore_s"])
                speed = run.get("speed")
                speed_values_finite = isinstance(speed, dict) and all(
                    value is None or (
                        isinstance(value, (int, float)) and
                        not isinstance(value, bool) and
                        math.isfinite(float(value)) and value >= 0
                    )
                    for value in speed.values()
                )
                expected_snapshot = (
                    KV_SNAPSHOT_HEADER_BYTES +
                    prompt_tokens * KV_SNAPSHOT_BYTES_PER_TOKEN
                ) if isinstance(prompt_tokens, int) else -1
                expected_poison = (
                    prompt_tokens * KV_SNAPSHOT_BYTES_PER_TOKEN
                ) if isinstance(prompt_tokens, int) else -1
                tolerance = 3e-6
                consistent = (
                    metrics.get("kv_snapshot_version") == KV_SNAPSHOT_VERSION and
                    metrics.get("kv_snapshot_header_bytes") ==
                        KV_SNAPSHOT_HEADER_BYTES and
                    metrics.get("kv_snapshot_bytes_per_token") ==
                        KV_SNAPSHOT_BYTES_PER_TOKEN and
                    metrics.get("kv_snapshot_bytes") == expected_snapshot and
                    metrics.get("kv_poisoned_bytes") == expected_poison and
                    cold_request >= cold_first and
                    warm_request >= warm_first >= state_restore and
                    state_capture + tolerance >= kv_capture + logit_capture and
                    abs(state_restore - (kv_restore + logit_restore)) <= tolerance and
                    speed_values_finite and
                    same_number(speed.get("kv_warm_speedup"),
                                cold_request / warm_request) and
                    same_number(speed.get("kv_warm_first_token_ready_speedup"),
                                cold_first / warm_first) and
                    same_number(speed.get("kv_warm_time_saved_s"),
                                cold_request - warm_request)
                )
                if not consistent:
                    errors.append(f"{label}: inconsistent KV timing evidence")
                    all_candidate = False

            if not isinstance(run.get("prompt_sha256"), str) or not HEX64.fullmatch(
                    run["prompt_sha256"]):
                errors.append(f"{label}: invalid prompt hash")
                all_candidate = False
            feature_hash = run.get("feature_sha256")
            if case["modality"] == "image" and not (
                    isinstance(feature_hash, str) and HEX64.fullmatch(feature_hash)):
                errors.append(f"{label}: invalid feature hash")
                all_candidate = False
            if case["modality"] == "text" and feature_hash is not None:
                errors.append(f"{label}: unexpected feature hash")
                all_candidate = False
            if case["modality"] == "image":
                pool_workers = run.get("vision_pool_workers")
                pool_submissions = run.get("vision_pool_submissions")
                gpu_hmm = run.get("vision_gpu_hmm", 0)
                gpu_submissions = run.get("vision_gpu_submissions", 0)
                pool_valid = isinstance(pool_workers, int) and pool_workers >= 1 and \
                    isinstance(pool_submissions, int) and pool_submissions >= 0 and \
                    isinstance(gpu_hmm, int) and gpu_hmm in (0, 1) and \
                    isinstance(gpu_submissions, int) and gpu_submissions >= 0 and \
                    ((gpu_hmm == 1 and gpu_submissions >= 1 and
                      pool_submissions == 0) or
                     (gpu_hmm == 0 and
                      (sys.platform == "darwin" or pool_submissions >= 1)))
                if not pool_valid:
                    errors.append(f"{label}: invalid vision executor engagement")
                    all_candidate = False
            elif run.get("vision_pool_workers") is not None or \
                    run.get("vision_pool_submissions") is not None or \
                    run.get("vision_gpu_hmm") is not None or \
                    run.get("vision_gpu_submissions") is not None:
                errors.append(f"{label}: unexpected vision executor evidence")
                all_candidate = False

        response_hashes = {run.get("response_sha256") for run in case_runs}
        warm_response_hashes = {
            run.get("warm_response_sha256") for run in case_runs
        }
        prompt_hashes = {run.get("prompt_sha256") for run in case_runs}
        token_hashes = {run.get("output_ids_sha256") for run in case_runs}
        warm_token_hashes = {
            run.get("warm_output_ids_sha256") for run in case_runs
        }
        snapshot_sizes = {
            run["metrics"].get("kv_snapshot_bytes")
            if isinstance(run.get("metrics"), dict) else None
            for run in case_runs
        }
        if len(response_hashes) != 1 or None in response_hashes:
            errors.append(f"{case['id']}: response hashes are not deterministic")
            all_candidate = False
        if warm_response_hashes != response_hashes:
            errors.append(f"{case['id']}: cold/KV-warm response hashes differ")
            all_candidate = False
        if len(prompt_hashes) != 1 or None in prompt_hashes:
            errors.append(f"{case['id']}: prompt hashes are not deterministic")
            all_candidate = False
        if len(token_hashes) != 1 or None in token_hashes:
            errors.append(f"{case['id']}: output token hashes are not deterministic")
            all_candidate = False
        if warm_token_hashes != token_hashes:
            errors.append(f"{case['id']}: cold/KV-warm token hashes differ")
            all_candidate = False
        if len(snapshot_sizes) != 1 or None in snapshot_sizes:
            errors.append(f"{case['id']}: KV snapshot size is not deterministic")
            all_candidate = False
        baseline = case["baseline_response_sha256"]
        if baseline == "PENDING":
            if require_baselines:
                errors.append(f"{case['id']}: response baseline remains pending")
        elif response_hashes != {baseline} or warm_response_hashes != {baseline}:
            errors.append(f"{case['id']}: response baseline mismatch")
            all_candidate = False
        if case["modality"] == "image":
            feature_hashes = {run.get("feature_sha256") for run in case_runs}
            if len(feature_hashes) != 1 or None in feature_hashes:
                errors.append(f"{case['id']}: feature hashes are not deterministic")
                all_candidate = False
    baselines_complete = all(
        case["baseline_response_sha256"] != "PENDING" for case in manifest["cases"]
    )
    report["candidate_execution_pass"] = all_candidate
    report["baseline_enforced"] = baselines_complete and not any(
        "baseline" in error for error in errors
    )
    report["overall_pass"] = all_candidate and report["baseline_enforced"]
    report["validation_errors"] = errors
    return report["overall_pass"] if require_baselines else all_candidate


def summary_for(report: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for case in manifest["cases"]:
        runs = [run for run in report["runs"] if run["case_id"] == case["id"]]
        response_hashes = sorted({run["response_sha256"] for run in runs})
        semantic_count = sum(bool(run["semantic_pass"]) for run in runs)
        summary.append({
            "case_id": case["id"],
            "task": case["task"],
            "pass_count": semantic_count,
            "repeat": manifest["execution"]["repeat"],
            "response_sha256": response_hashes[0] if len(response_hashes) == 1 else None,
            "wall_s": aggregate(run_values(runs, ("wall_s",))),
            "request_s": aggregate(run_values(runs, ("metrics", "request_s"))),
            "cold_request_s": aggregate(
                run_values(runs, ("metrics", "cold_request_s"))
            ),
            "warm_request_s": aggregate(
                run_values(runs, ("metrics", "warm_request_s"))
            ),
            "cold_first_token_ready_s": aggregate(
                run_values(runs, ("metrics", "cold_first_token_ready_s"))
            ),
            "warm_first_token_ready_s": aggregate(
                run_values(runs, ("metrics", "warm_first_token_ready_s"))
            ),
            "state_capture_s": aggregate(
                run_values(runs, ("metrics", "state_capture_s"))
            ),
            "kv_capture_s": aggregate(run_values(runs, ("metrics", "kv_capture_s"))),
            "logit_capture_s": aggregate(
                run_values(runs, ("metrics", "logit_capture_s"))
            ),
            "kv_poison_s": aggregate(run_values(runs, ("metrics", "kv_poison_s"))),
            "kv_restore_s": aggregate(run_values(runs, ("metrics", "kv_restore_s"))),
            "logit_restore_s": aggregate(
                run_values(runs, ("metrics", "logit_restore_s"))
            ),
            "state_restore_s": aggregate(
                run_values(runs, ("metrics", "state_restore_s"))
            ),
            "kv_snapshot_mb": aggregate([
                value / 1_000_000
                for value in run_values(runs, ("metrics", "kv_snapshot_bytes"))
            ]),
            "kv_poisoned_mb": aggregate([
                value / 1_000_000
                for value in run_values(runs, ("metrics", "kv_poisoned_bytes"))
            ]),
            "kv_warm_speedup": aggregate(
                run_values(runs, ("speed", "kv_warm_speedup"))
            ),
            "kv_warm_time_saved_s": aggregate(
                run_values(runs, ("speed", "kv_warm_time_saved_s"))
            ),
            "native_total_s": aggregate(run_values(runs, ("metrics", "native_total_s"))),
            "peak_rss_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(runs, ("metrics", "peak_rss_bytes"))
            ]),
            "current_rss_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(runs, ("memory", "current_rss_bytes"))
            ]),
            "resident_peak_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(runs, ("memory", "resident_peak_bytes"))
            ]),
            "physical_footprint_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(
                    runs, ("memory", "physical_footprint_bytes"),
                )
            ]),
            "internal_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(runs, ("memory", "internal_bytes"))
            ]),
            "external_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(runs, ("memory", "external_bytes"))
            ]),
            "expert_peak_logical_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(
                    runs, ("memory", "expert_peak_logical_bytes"),
                )
            ]),
            "expert_peak_mapped_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(
                    runs, ("memory", "expert_peak_mapped_bytes"),
                )
            ]),
            "expert_unmapped_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(
                    runs, ("memory", "expert_unmapped_bytes"),
                )
            ]),
            "expert_released_page_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(
                    runs, ("memory", "expert_released_page_bytes"),
                )
            ]),
            "dense_payload_gb": aggregate([
                value / 1_000_000_000
                for value in run_values(
                    runs, ("memory", "dense_payload_bytes"),
                )
            ]),
            "expert_cache_hit_rate": aggregate([
                float(run["memory"]["expert_hits"]) /
                float(run["memory"]["expert_requests"])
                for run in runs if run.get("memory", {}).get("expert_requests", 0) > 0
            ]),
            "minor_faults": aggregate(
                run_values(runs, ("memory", "minor_faults"))
            ),
            "major_faults": aggregate(
                run_values(runs, ("memory", "major_faults"))
            ),
            "input_blocks": aggregate(
                run_values(runs, ("memory", "input_blocks"))
            ),
            "prefill_tokens_s": aggregate(
                run_values(runs, ("speed", "prefill_tokens_s"))
            ),
            "post_first_decode_steps_s": aggregate(
                run_values(runs, ("speed", "post_first_decode_steps_s"))
            ),
            "post_restore_generation_steps_s": aggregate(
                run_values(runs, ("speed", "post_restore_generation_steps_s"))
            ),
        })
    return summary


def display_range(values: dict[str, float] | None, digits: int = 3) -> str:
    if values is None:
        return "—"
    return (
        f"{values['median']:.{digits}f} "
        f"[{values['min']:.{digits}f}, {values['max']:.{digits}f}]"
    )


def render_markdown(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    summary = summary_for(report, manifest)
    status = "PASS" if report["overall_pass"] else "FAIL"
    host = report["host"]
    complete_runs = [
        run for run in report["runs"]
        if isinstance(run.get("metrics"), dict) and
        isinstance(run["metrics"].get("cold_request_s"), (int, float)) and
        isinstance(run["metrics"].get("warm_request_s"), (int, float)) and
        isinstance(run["metrics"].get("peak_rss_bytes"), int)
    ]
    cold_total = sum(run["metrics"]["cold_request_s"] for run in complete_runs)
    warm_total = sum(run["metrics"]["warm_request_s"] for run in complete_runs)
    ratio_text = f"{cold_total / warm_total:.6f}x" if warm_total > 0.0 else "n/a"
    wall_total = sum(run["wall_s"] for run in report["runs"])
    max_peak_rss = max(
        (run["metrics"]["peak_rss_bytes"] for run in complete_runs),
        default=0,
    )
    identity = report.get("execution_identity")
    identity_lines: list[str] = []
    if isinstance(identity, dict) and isinstance(identity.get("before"), dict):
        identity_lines.append(
            f"- Execution identity stable: `{str(identity.get('stable')).lower()}`"
        )
        for name, item in identity["before"].items():
            if isinstance(item, dict):
                identity_lines.append(
                    f"- `{name}` SHA-256: `{item.get('sha256')}`"
                )
    lines = [
        "# Gemma 4 Preserved E2E Cold / KV-Warm Final-Check Report",
        "",
        f"- **Result:** `{status}`",
        f"- **Report schema:** `{REPORT_SCHEMA}`",
        f"- **Generated (UTC):** `{report['generated_at_utc']}`",
        f"- **Runtime readiness:** `true`",
        "",
        "## Contract",
        "",
        f"- Model: `{manifest['model']['repository']}`",
        f"- Revision: `{manifest['model']['revision']}`",
        "- Cases: text retrieval, document QA, image QA, and OCR",
        "- Repeats: exactly `3` per case; `12` serial, process-isolated runs",
        "- Gate: `CTX200/O50`, greedy decode, `3` workers",
        "- Source authentication and expert-pool binding run before every sample",
        "- Cold: full request prefill plus decode; image cold also includes image "
        "processing and vision",
        "- KV-warm: restore the exact in-memory KV prefix and cached first-token logits "
        "captured at completed prefill, then decode",
        "- Before restore, every live prefix K/V value is overwritten with NaN outside "
        "warm timing; restore verifies payload SHA-256, finite values, and copied bytes",
        "- State capture includes sizing/allocation, KV copy/hash, and logit copy and is "
        "excluded from cold latency; state restore includes KV and logit restore and is "
        "included in warm latency; poison cost is reported separately",
        "- First-token-ready is an internal decoder timestamp after logits are available, "
        "not client-observed TTFT; response blocks are emitted after generation",
        "- Native model loading is excluded from cold and warm request latency; external "
        "wall includes authentication, loading, both decodes, and reporting",
        "- Experts use direct mapped Q4 slots with a decimal `1,000,000,000`-byte "
        "budget: `298` whole slots / `996,931,584` logical bytes; no arena copy",
        "- Union-acquired routed slots remain stable through all grouped consumers; "
        "evicted expert and completed dense tensor pages use contained-page "
        "`MADV_DONTNEED`; expert eviction then unmaps the borrowed slot view",
        "- The one-shot vision model is unmapped immediately after feature extraction",
        "- OS page-cache warmth is not classified as KV-warm",
        "- `runtime_ready=true`; this is not a serving qualification",
        "",
        "## Authority and host",
        "",
        f"- Manifest SHA-256: `{report['manifest_sha256']}`",
        f"- Git base: `{report['git']['head']}` on `{report['git']['branch']}` "
        f"(candidate tree dirty: `{str(report['git']['dirty']).lower()}`)",
        f"- Host: `{host['platform']}` / `{host['machine']}`",
        f"- Physical memory: `{host['physical_memory_gb']:.3f} GB`"
        if host["physical_memory_gb"] is not None else "- Physical memory: unavailable",
        *identity_lines,
        f"- Authenticated source: `{report['inputs']['source_bytes']} bytes` "
        f"(`{report['inputs']['source_bytes'] / 1_000_000_000:.9f} GB`)",
        f"- Expert pool: `{report['inputs']['pool_bytes']} bytes` "
        f"(`{report['inputs']['pool_bytes'] / 1_000_000_000:.9f} GB`)",
        "- Source and pool extents are storage/mapping evidence and are never added to "
        "process-memory metrics",
        "- Peak RSS below is the native engine child peak, converted as "
        "`bytes / 1,000,000,000`",
        "",
        "## Aggregate results",
        "",
        "Values are `median [min, max]` across the three runs.",
        "",
        "| Case | Correct | Cold request s | KV-warm request s | Saved s | Speedup | "
        "Cold first-ready s | Warm first-ready s | State restore s | KV MB | Peak RSS GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| `{item['case_id']}` | {item['pass_count']}/{item['repeat']} | "
            f"{display_range(item['cold_request_s'])} | "
            f"{display_range(item['warm_request_s'])} | "
            f"{display_range(item['kv_warm_time_saved_s'])} | "
            f"{display_range(item['kv_warm_speedup'])}× | "
            f"{display_range(item['cold_first_token_ready_s'])} | "
            f"{display_range(item['warm_first_token_ready_s'])} | "
            f"{display_range(item['state_restore_s'], 6)} | "
            f"{display_range(item['kv_snapshot_mb'], 6)} | "
            f"{display_range(item['peak_rss_gb'])} |"
        )
    lines.extend([
        "",
        "Speedup is `cold_request_s / warm_request_s`; saved time is "
        "`cold_request_s - warm_request_s`. Peak RSS is the paired native process, "
        "including its in-memory KV snapshot and both decode paths.",
        "",
        f"Across all 12 pairs: cold `{cold_total:.6f} s`, KV-warm "
        f"`{warm_total:.6f} s`, saved `{cold_total - warm_total:.6f} s`, "
        f"cold/warm ratio `{ratio_text}`, external paired wall "
        f"`{wall_total:.6f} s`, and maximum peak RSS "
        f"`{max_peak_rss / 1_000_000_000:.9f} GB`.",
        "",
        "## Bounded residency evidence",
        "",
        "All values are `median [min, max]`. Logical expert bytes are cache-accounting "
        "bytes; mapped bytes are requested mmap extents (the OS may page-round each "
        "extent); expert-advised and unmapped bytes are cumulative traffic, not "
        "resident memory. Dense bytes are the logical immutable retained payload, "
        "not measured residency.",
        "",
        "| Case | Final/current RSS GB | Lifetime resident peak GB | "
        "Final/current footprint GB | Final internal GB | "
        "Final external GB | Expert logical / mapped peak GB | Cache hit rate | "
        "Expert advised / unmapped GB | Dense logical retained GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in summary:
        lines.append(
            f"| `{item['case_id']}` | {display_range(item['current_rss_gb'])} | "
            f"{display_range(item['resident_peak_gb'])} | "
            f"{display_range(item['physical_footprint_gb'])} | "
            f"{display_range(item['internal_gb'])} | "
            f"{display_range(item['external_gb'])} | "
            f"{display_range(item['expert_peak_logical_gb'])} / "
            f"{display_range(item['expert_peak_mapped_gb'])} | "
            f"{display_range(item['expert_cache_hit_rate'])} | "
            f"{display_range(item['expert_released_page_gb'])} / "
            f"{display_range(item['expert_unmapped_gb'])} | "
            f"{display_range(item['dense_payload_gb'])} |"
        )
    lines.extend([
        "",
        "## Fault and I/O evidence",
        "",
        "Counters are cumulative `getrusage(RUSAGE_SELF)` values at final sampling. "
        "On mmap-backed files, a zero major-fault or block-I/O count does not imply zero "
        "storage traffic; the unified page cache may satisfy refaults.",
        "",
        "| Case | Minor faults | Major faults | Input block operations |",
        "|---|---:|---:|---:|",
    ])
    for item in summary:
        lines.append(
            f"| `{item['case_id']}` | {display_range(item['minor_faults'], 0)} | "
            f"{display_range(item['major_faults'], 0)} | "
            f"{display_range(item['input_blocks'], 0)} |"
        )
    lines.extend([
        "",
        "## Throughput and process wall",
        "",
        "| Case | External paired wall s | Prefill tok/s | Cold post-first step/s | "
        "Post-restore generation step/s |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in summary:
        lines.append(
            f"| `{item['case_id']}` | {display_range(item['wall_s'])} | "
            f"{display_range(item['prefill_tokens_s'])} | "
            f"{display_range(item['post_first_decode_steps_s'])} | "
            f"{display_range(item['post_restore_generation_steps_s'])} |"
        )
    lines.extend([
        "",
        "Post-restore generation is model compute after verified KV restoration. "
        "It is not KV restore/recovery latency and is not the cold steady-decode "
        "target. Total restore and post-restore first-ready latencies are listed "
        "separately below.",
        "",
        "## Per-run evidence",
        "",
        "| Case | Run | Exit | Prompt / image tokens | Output | Cold request s | "
        "Post-restore request s | Saved s | Speedup | Cold first-ready | "
        "Post-restore first-ready | State capture s | "
        "Poison s | State restore s | KV MB | Wall s | Peak RSS GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in report["runs"]:
        metrics = run["metrics"]
        speed = run["speed"]
        prompt_image = str(run["prompt_tokens"])
        if run["image_tokens"] is not None:
            prompt_image += f" / {run['image_tokens']}"
        def metric(key: str) -> str:
            value = metrics.get(key)
            return "—" if value is None else f"{value:.6f}"
        rss = metrics.get("peak_rss_bytes")
        rss_text = "—" if rss is None else f"{rss / 1_000_000_000:.9f}"
        snapshot = metrics.get("kv_snapshot_bytes")
        snapshot_text = "—" if snapshot is None else f"{snapshot / 1_000_000:.6f}"
        saved = speed.get("kv_warm_time_saved_s")
        speedup = speed.get("kv_warm_speedup")
        lines.append(
            f"| `{run['case_id']}` | {run['repeat']} | {run['status']} | "
            f"{prompt_image} | {run['output_steps']} | {metric('cold_request_s')} | "
            f"{metric('warm_request_s')} | "
            f"{'—' if saved is None else f'{saved:.6f}'} | "
            f"{'—' if speedup is None else f'{speedup:.6f}×'} | "
            f"{metric('cold_first_token_ready_s')} | "
            f"{metric('warm_first_token_ready_s')} | "
            f"{metric('state_capture_s')} | {metric('kv_poison_s')} | "
            f"{metric('state_restore_s')} | "
            f"{snapshot_text} | {run['wall_s']:.6f} | {rss_text} |"
        )
    lines.extend(["", "## Determinism evidence", ""])
    for case in manifest["cases"]:
        runs = [
            run for run in report["runs"]
            if run["case_id"] == case["id"] and run.get("status") == 0 and
            isinstance(run.get("metrics"), dict) and
            isinstance(run["metrics"].get("kv_snapshot_version"), int)
        ]
        if not runs:
            lines.extend([
                f"### `{case['id']}`",
                "",
                "- No completed run evidence.",
                "",
            ])
            continue
        lines.extend([
            f"### `{case['id']}`",
            "",
            f"- Source: {case['source_name']}",
            f"- Prompt SHA-256: `{runs[0]['prompt_sha256']}`",
            f"- Cold / KV-warm response SHA-256: `{runs[0]['response_sha256']}`",
            f"- Cold / KV-warm output-ID SHA-256: `{runs[0]['output_ids_sha256']}`",
            f"- Cold / KV-warm response bytes: `{runs[0]['response_bytes']}`",
            f"- KV-prefill position: `{runs[0]['kv_prefill_position']}` tokens",
            f"- KV snapshot v`{runs[0]['metrics']['kv_snapshot_version']}`: "
            f"`{runs[0]['metrics']['kv_snapshot_bytes']}` bytes "
            f"(`{runs[0]['metrics']['kv_snapshot_header_bytes']}` header + "
            f"`{runs[0]['metrics']['kv_snapshot_bytes_per_token']}` per token)",
            f"- Destructive proof: `{runs[0]['metrics']['kv_poisoned_bytes']}` "
            "prefix bytes poisoned; restored destination verified byte-for-byte",
        ])
        if case["modality"] == "image":
            lines.extend([
                f"- Image: `{case['image']}`",
                f"- Image SHA-256: `{case['image_sha256']}`",
                f"- Feature SHA-256: `{runs[0]['feature_sha256']}`",
            ])
        if case["source_urls"]:
            lines.append("- Sources: " + ", ".join(
                f"<{url}>" for url in case["source_urls"]
            ))
        lines.append("")
    lines.extend(["## Engine responses", ""])
    for case in manifest["cases"]:
        lines.extend([f"### `{case['id']}`", ""])
        for run in [item for item in report["runs"] if item["case_id"] == case["id"]]:
            lines.extend([
                f"Run {run['repeat']} cold (`sha256={run['response_sha256']}`):",
                "",
                "```text",
                run["response"],
                "```",
                "",
                f"Run {run['repeat']} KV-warm "
                f"(`sha256={run['warm_response_sha256']}`):",
                "",
                "```text",
                run["warm_response"],
                "```",
                "",
            ])
    lines.extend([
        "## Qualification boundary",
        "",
        "PASS requires all twelve executions to exit zero, retain "
        "`runtime_ready=true`, match the accepted answer, prove an exact-size v2 "
        "same-instance KV snapshot, poison the full live prefix, verify restored bytes, "
        "and produce length- and byte-identical cold/KV-warm responses and token IDs. "
        "Prompt, output-token, feature (image cases), KV snapshot size, and exact "
        "response hashes must also remain deterministic per case. Exact response hashes "
        "are pinned in the manifest and enforced on future runs.",
        "",
        "This small final check does not establish independent Transformers feature parity, "
        "general image resizing, public multimodal ingress, concurrency, or serving "
        "readiness.",
        "",
    ])
    if report["validation_errors"]:
        lines.extend(["## Validation errors", ""])
        lines.extend(f"- {error}" for error in report["validation_errors"])
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], manifest: dict[str, Any],
                 json_path: Path, markdown_path: Path) -> None:
    report["summary"] = summary_for(report, manifest)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    markdown_path.write_text(render_markdown(report, manifest))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--source-dir", type=Path)
    result.add_argument("--pool", type=Path)
    result.add_argument("--receipt", type=Path)
    result.add_argument("--auth-receipt", type=Path)
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--text-wrapper", type=Path, default=ROOT / "tools/gemma4-qa.py")
    result.add_argument(
        "--multimodal-wrapper", type=Path,
        default=ROOT / "tools/gemma4-multimodal-qa.py",
    )
    result.add_argument("--text-runner", type=Path, default=ROOT / "gemma4-qa")
    result.add_argument(
        "--multimodal-runner", type=Path, default=ROOT / "gemma4-multimodal-qa",
    )
    result.add_argument("--json-report", type=Path, default=DEFAULT_JSON)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    result.add_argument("--timeout-s", type=int, default=900)
    result.add_argument(
        "--render-json", type=Path,
        help="statically validate existing JSON against pinned baselines and rerender",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.timeout_s < 1:
            raise BenchmarkError("--timeout-s must be positive")
        if args.render_json is not None:
            manifest = load_manifest(args.manifest, permit_pending=False)
            try:
                report = json.loads(args.render_json.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise BenchmarkError(f"cannot load report JSON: {exc}") from exc
            if report.get("schema") != REPORT_SCHEMA:
                raise BenchmarkError("report schema drift")
            report["manifest"] = str(args.manifest)
            report["manifest_sha256"] = sha256_file(args.manifest)
            report["cases"] = manifest["cases"]
            if not validate_report(report, manifest, require_baselines=True):
                raise BenchmarkError("report does not satisfy pinned final-check contract")
            write_report(report, manifest, args.json_report, args.report)
            print(f"GEMMA4_E2E_REPORT_OK json={args.json_report} report={args.report}")
            print("runtime_ready=true")
            return 0
        manifest = load_manifest(args.manifest, permit_pending=False)
        if args.source_dir is None or args.pool is None:
            raise BenchmarkError("--source-dir and --pool are required for execution")
        report = run_benchmark(args, manifest)
        write_report(report, manifest, args.json_report, args.report)
        if not report["candidate_execution_pass"]:
            raise BenchmarkError("candidate execution failed")
        if not report["overall_pass"]:
            raise BenchmarkError("pinned final-check validation failed")
        print(f"GEMMA4_E2E_REPORT_OK json={args.json_report} report={args.report}")
        print("runtime_ready=true")
        return 0
    except BenchmarkError as exc:
        print(f"gemma4 e2e: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
