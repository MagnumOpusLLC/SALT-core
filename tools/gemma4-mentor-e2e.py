#!/usr/bin/env python3
"""Exact cold/import/mentor A/B/C qualification for Gemma G4KVC005 state."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "server/model/gemma4_backend.py"
MODULE_NAME = "salt_gemma4_mentor_e2e_backend"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, BACKEND_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Gemma mentor qualification backend")
BACKEND = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = BACKEND
SPEC.loader.exec_module(BACKEND)

SCHEMA = "salt.gemma4.mentor-exact.v1"
SEED = (
    "Project Cedar is owned by Mira and deploys to north. "
    "Store these facts as the shared immutable reference and reply only: stored"
)
QUESTION = "Who owns Project Cedar, and where does it deploy? Answer exactly: Mira, north"


class QualificationError(RuntimeError):
    """The exact mentor qualification contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def differing_bytes(left: Path, right: Path) -> int:
    if left.stat().st_size != right.stat().st_size:
        return max(left.stat().st_size, right.stat().st_size)
    total = 0
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            a = first.read(1024 * 1024)
            b = second.read(1024 * 1024)
            if not a and not b:
                return total
            if len(a) != len(b):
                return max(left.stat().st_size, right.stat().st_size)
            total += sum(x != y for x, y in zip(a, b))


def result_record(completion, export: dict[str, Any], state_path: Path,
                  wall_s: float) -> dict[str, Any]:
    result = completion.result
    return {
        "response": result.content,
        "response_sha256": hashlib.sha256(
            result.content.encode("utf-8")
        ).hexdigest(),
        "output_ids": list(result.output_ids),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "loaded_tokens": result.kv_loaded_tokens,
        "saved_tokens": result.kv_saved_tokens,
        "state_bytes": state_path.stat().st_size,
        "state_file_sha256": sha256_file(state_path),
        "native_state_sha256": export["state_sha256"],
        "facts_sha256": export["facts_sha256"],
        "build_identity_sha256": export["build_identity_sha256"],
        "wall_s": wall_s,
    }


def config(args, *, shared_kv: Path | None = None):
    return BACKEND.Gemma4Config(
        source_dir=args.source_dir,
        pool=args.pool,
        receipt=args.receipt,
        auth_receipt=args.auth_receipt,
        workers=args.workers,
        kv_budget_gb=args.kv_budget_gb,
        timeout_s=args.timeout,
        shared_kv=shared_kv,
    )


def complete(backend, message: str, request_key: str):
    return backend.complete({
        "model": backend.config.model_id,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "kv_allowance_gb": backend.config.kv_budget_gb,
    }, endpoint="chat", request_key=request_key)


def run(args) -> dict[str, Any]:
    if args.artifacts.exists():
        raise QualificationError("artifacts directory already exists")
    args.artifacts.mkdir(mode=0o700, parents=True)
    prefix = args.artifacts / "prefix.g4kv"
    final_a = args.artifacts / "final-cold.g4kv"
    final_b = args.artifacts / "final-import.g4kv"
    final_c = args.artifacts / "final-mentor.g4kv"

    base_config = config(args)
    auth_started = time.monotonic()
    authority = BACKEND.Gemma4AuthenticationAuthority(
        base_config, reauthenticate=args.reauthenticate,
    )
    auth_s = time.monotonic() - auth_started
    cold_backend = None
    mentor_backend = None
    text_authority = None
    try:
        # A: Native cold seed and continuation in one persistent C-owned session.
        cold_backend = BACKEND.Gemma4Backend(base_config, authentication=authority)
        cold_backend.start_persistent()
        seed_started = time.monotonic()
        seed_completion = complete(cold_backend, SEED, "mentor-seed")
        seed_wall_s = time.monotonic() - seed_started
        prefix_export = cold_backend.export_kv(prefix)
        prefix_before = sha256_file(prefix)
        cold_started = time.monotonic()
        cold_completion = complete(cold_backend, QUESTION, "mentor-cold-continuation")
        cold_wall_s = time.monotonic() - cold_started
        cold_export = cold_backend.export_kv(final_a)
        cold = result_record(cold_completion, cold_export, final_a, cold_wall_s)
        cold_backend.stop_persistent()
        cold_backend = None

        prefix_metadata = BACKEND._state_artifact_module.inspect_g4kvc006(
            prefix,
            expected_compatibility_sha256=BACKEND.KV_COMPATIBILITY_SHA256,
        )

        # B: Ordinary mutable import through the existing native text runner.
        # Authenticate that binary separately; do not widen the public wrapper's
        # qualified gate table just to exercise the server's fixed 512/128 cell.
        text_receipt = Path(str(base_config.text_runner) + ".build.json")
        text_config = replace(
            base_config,
            server_runner=base_config.text_runner,
            server_build_receipt=text_receipt,
        )
        text_authority = BACKEND.Gemma4AuthenticationAuthority(text_config)
        rendered = BACKEND.render_chat(
            [{"role": "user", "content": QUESTION}],
            enable_thinking=False,
            continuation=True,
        )
        command = [
            str(base_config.text_runner),
            "--ctx", "512", "--gen", "128",
            "--prompt", rendered.prompt,
            "--workers", str(args.workers),
            "--kv-load", str(prefix),
            "--kv-save", str(final_b),
        ]
        import_started = time.monotonic()
        imported_process = subprocess.run(
            command,
            input=text_authority.binding,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=text_authority.inherited_fds,
            env=BACKEND.Gemma4Backend._clean_env(),
            timeout=args.timeout,
            check=False,
        )
        import_wall_s = time.monotonic() - import_started
        if imported_process.returncode != 0:
            raise QualificationError(
                "ordinary import runner failed: " +
                imported_process.stderr[-2048:].decode("utf-8", errors="replace")
            )
        import_result = BACKEND.parse_runner_output(
            imported_process.stdout, imported_process.stderr, "text", 128,
        )
        import_metadata = BACKEND._state_artifact_module.inspect_g4kvc006(
            final_b,
            expected_compatibility_sha256=BACKEND.KV_COMPATIBILITY_SHA256,
        )
        imported = {
            "response": import_result.content,
            "response_sha256": hashlib.sha256(
                import_result.content.encode("utf-8")
            ).hexdigest(),
            "output_ids": list(import_result.output_ids),
            "prompt_tokens": import_result.prompt_tokens,
            "completion_tokens": import_result.completion_tokens,
            "loaded_tokens": import_result.kv_loaded_tokens,
            "saved_tokens": import_result.kv_saved_tokens,
            "state_bytes": final_b.stat().st_size,
            "state_file_sha256": import_metadata.file_sha256,
            "native_state_sha256": None,
            "facts_sha256": None,
            "build_identity_sha256": None,
            "wall_s": import_wall_s,
        }

        # C: Read-only immutable mentor prefix and private native continuation.
        mentor_backend = BACKEND.Gemma4Backend(
            config(args, shared_kv=prefix), authentication=authority,
        )
        mentor_backend.start_persistent()
        if mentor_backend.shared_position != prefix_metadata.position:
            raise QualificationError("mentor READY shared position mismatch")
        mentor_started = time.monotonic()
        mentor_completion = complete(
            mentor_backend, QUESTION, "mentor-shared-continuation",
        )
        mentor_wall_s = time.monotonic() - mentor_started
        mentor_export = mentor_backend.export_kv(final_c)
        mentor = result_record(
            mentor_completion, mentor_export, final_c, mentor_wall_s,
        )
        mentor_backend.stop_persistent()
        mentor_backend = None
        prefix_after = sha256_file(prefix)

        diffs = {
            "cold_import": differing_bytes(final_a, final_b),
            "cold_mentor": differing_bytes(final_a, final_c),
            "import_mentor": differing_bytes(final_b, final_c),
        }
        responses_equal = (
            cold["response"] == imported["response"] == mentor["response"]
        )
        output_ids_equal = (
            cold["output_ids"] == imported["output_ids"] == mentor["output_ids"]
        )
        states_equal = all(value == 0 for value in diffs.values())
        anchor_unchanged = prefix_before == prefix_after
        passed = responses_equal and output_ids_equal and states_equal and anchor_unchanged
        return {
            "schema": SCHEMA,
            "runtime_ready": False,
            "compatibility_sha256": BACKEND.KV_COMPATIBILITY.hexdigest,
            "runner_sha256": sha256_file(ROOT / "gemma4-qa"),
            "server_runner_sha256": sha256_file(ROOT / "gemma4-server"),
            "authentication_s": auth_s,
            "seed": {
                "question": SEED,
                "response": seed_completion.result.content,
                "output_ids": list(seed_completion.result.output_ids),
                "wall_s": seed_wall_s,
                "position": prefix_metadata.position,
                "state_bytes": prefix_metadata.total_bytes,
                "state_file_sha256": prefix_before,
                "native_state_sha256": prefix_export["state_sha256"],
                "facts_sha256": prefix_export["facts_sha256"],
            },
            "continuation": QUESTION,
            "cells": {"cold": cold, "import": imported, "mentor": mentor},
            "comparison": {
                "responses_equal": responses_equal,
                "output_ids_equal": output_ids_equal,
                "differing_bytes": diffs,
                "states_equal": states_equal,
                "anchor_sha256_before": prefix_before,
                "anchor_sha256_after": prefix_after,
                "anchor_unchanged": anchor_unchanged,
            },
            "passed": passed,
        }
    finally:
        if cold_backend is not None:
            cold_backend.stop_persistent()
        if mentor_backend is not None:
            mentor_backend.stop_persistent()
        if text_authority is not None:
            text_authority.close()
        authority.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--auth-receipt", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--kv-budget-gb", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--reauthenticate", action="store_true")
    args = parser.parse_args(argv)
    try:
        for path in (args.pool, args.receipt):
            if not path.is_file():
                raise QualificationError(f"required artifact is missing: {path}")
        if not args.source_dir.is_dir():
            raise QualificationError("source directory is missing")
        if args.report.exists():
            raise QualificationError("report path already exists")
        report = run(args)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "passed": report["passed"],
            "report": str(args.report),
            "responses_equal": report["comparison"]["responses_equal"],
            "output_ids_equal": report["comparison"]["output_ids_equal"],
            "states_equal": report["comparison"]["states_equal"],
            "anchor_unchanged": report["comparison"]["anchor_unchanged"],
        }, sort_keys=True))
        return 0 if report["passed"] else 1
    except (QualificationError, BACKEND.BackendError, BACKEND.RequestError) as exc:
        print(f"gemma4 mentor qualification: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
