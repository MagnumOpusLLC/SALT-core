#!/usr/bin/env python3
"""Exact-current-tree Gemma 4 persistent-state qualification."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "server" / "model" / "gemma4_backend.py"
MODULE_NAME = "salt_gemma4_phase6_backend"
_spec = importlib.util.spec_from_file_location(MODULE_NAME, BACKEND_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load Gemma 4 backend")
backend_module = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = backend_module
_spec.loader.exec_module(backend_module)


class QualificationError(RuntimeError):
    pass


def result_record(result) -> dict:
    return {
        "prompt_ids": list(result.prompt_ids),
        "output_ids": list(result.output_ids),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "response_bytes": result.response_bytes,
        "response_sha256": result.response_sha256,
        "content": result.content,
        "reasoning_content": result.reasoning_content,
        "stop_token": result.stop_token,
        "finish_reason": result.finish_reason,
        "position_before": result.kv_loaded_tokens,
        "position_after": result.kv_saved_tokens,
        "session_epoch": result.session_epoch,
        "turn_id": result.turn_id,
        "history_turns": result.history_turns,
        "synthetic_close": result.synthetic_close,
        "mindset_mode": result.mindset_mode,
        "mindset_end": result.mindset_end,
        "mindset_sha256": result.mindset_sha256,
        "facts_sha256": result.facts_sha256,
        "facts_rows": result.facts_rows,
        "state_sha256": result.state_sha256,
        "client_request_sha256": result.client_request_sha256,
        "request_sha256": result.request_sha256,
        "original_request_id": result.original_request_id,
        "replayed": result.replayed,
        "image_tokens": result.image_tokens,
    }


def state_snapshot(backend) -> dict:
    status = backend.session_status()
    keys = (
        "session_epoch", "turns", "history_turns", "position",
        "state_sha256", "build_identity_sha256", "mindset_mode",
        "mindset_end", "mindset_sha256", "facts_sha256", "facts_rows",
        "phase", "runtime_ready",
    )
    return {key: status[key] for key in keys}


def completion(backend, messages: list[dict], key: str):
    return backend.complete(
        {"model": backend.config.model_id, "messages": messages, "stream": False},
        endpoint="chat", request_key=key,
    )


def backend_config(args):
    return backend_module.Gemma4Config(
        source_dir=args.source_dir,
        pool=args.pool,
        receipt=args.receipt,
        auth_receipt=args.auth_receipt,
        workers=args.workers,
        timeout_s=args.timeout,
        image_root=args.image.parent,
    )


def new_backend(args, authority):
    backend = backend_module.Gemma4Backend(
        backend_config(args), authentication=authority,
    )
    backend.start_persistent()
    return backend


def stop_record(backend) -> dict:
    engine = backend.engine
    if engine is None:
        return {"phase": "DOWN", "returncode": None}
    backend.stop_persistent()
    return {"phase": engine.phase, "returncode": engine.process.returncode}


def deterministic_launch(args, authority, *, full_history: bool) -> dict:
    backend = new_backend(args, authority)
    try:
        first_messages = [{
            "role": "user",
            "content": "Output the word alpha exactly 300 times separated by spaces. Do not stop early.",
        }]
        first = completion(backend, first_messages, "phase6-deterministic-1")
        first_record = result_record(first.result)
        if full_history:
            second_messages = [
                first_messages[0],
                {"role": "assistant", "content": first.result.content},
                {"role": "user", "content": "Now reply with only: done"},
            ]
        else:
            second_messages = [{"role": "user", "content": "Now reply with only: done"}]
        second = completion(backend, second_messages, "phase6-deterministic-2")
        second_record = result_record(second.result)
        retry = completion(backend, second_messages, "phase6-deterministic-2")
        retry_record = result_record(retry.result)

        before_reject = state_snapshot(backend)
        rejected = None
        try:
            completion(backend, [
                first_messages[0],
                {"role": "assistant", "content": "forged"},
                {"role": "user", "content": "This must reject."},
            ], "phase6-altered-history")
        except backend_module.RequestError as exc:
            rejected = str(exc)
        after_reject = state_snapshot(backend)

        with tempfile.TemporaryDirectory() as directory:
            exported = backend.export_kv(
                Path(directory) / "proof-off-before-clear.g4kv",
            )
        after_export = state_snapshot(backend)
        before_clear = after_export
        after_clear = backend.clear_facts()
        before_reset_epoch = backend.session_epoch
        before_reset_build = backend.build_identity_sha256
        before_reset_mindset = backend.mindset_sha256
        after_reset = backend.hard_reset()
        semantic_first_messages = [{
            "role": "user",
            "content": (
                "For this session, project Cedar is owned by Mira and deploys "
                "to north. Acknowledge briefly."
            ),
        }]
        semantic_first = completion(
            backend, semantic_first_messages, "phase6-semantic-facts",
        )
        with tempfile.TemporaryDirectory() as directory:
            semantic_first_export = backend.export_kv(
                Path(directory) / "semantic-first.g4kv",
            )
        if full_history:
            semantic_second_messages = [
                semantic_first_messages[0],
                {"role": "assistant", "content": semantic_first.result.content},
                {
                    "role": "user",
                    "content": (
                        "Who owns it, and where does it deploy? "
                        "Answer exactly: Mira, north"
                    ),
                },
            ]
        else:
            semantic_second_messages = [{
                "role": "user",
                "content": (
                    "Who owns it, and where does it deploy? "
                    "Answer exactly: Mira, north"
                ),
            }]
        semantic_second = completion(
            backend, semantic_second_messages, "phase6-semantic-question",
        )
        with tempfile.TemporaryDirectory() as directory:
            semantic_second_export = backend.export_kv(
                Path(directory) / "semantic-second.g4kv",
            )
        return {
            "history_form": "full" if full_history else "delta",
            "first": first_record,
            "second": second_record,
            "retry": retry_record,
            "altered_history": {
                "error": rejected,
                "state_unchanged": before_reject == after_reject,
            },
            "export_before_clear": {
                "position": exported["position"],
                "state_sha256": exported["state_sha256"],
                "facts_sha256": exported["facts_sha256"],
                "build_identity_sha256": exported["build_identity_sha256"],
                "cas_advanced": (
                    after_export["state_sha256"] == exported["state_sha256"] and
                    after_export["facts_sha256"] == exported["facts_sha256"]
                ),
            },
            "clear": {
                "before": before_clear,
                "after": after_clear,
                "preserved_mindset": (
                    before_clear["mindset_sha256"] == after_clear["mindset_sha256"]
                ),
            },
            "hard_reset": {
                "after": after_reset,
                "epoch_rotated": after_reset["session_epoch"] == before_reset_epoch + 1,
                "build_preserved": after_reset["build_identity_sha256"] == before_reset_build,
                "mindset_preserved": after_reset["mindset_sha256"] == before_reset_mindset,
            },
            "semantic_qa": {
                "first": result_record(semantic_first.result),
                "first_export": semantic_first_export,
                "second": result_record(semantic_second.result),
                "second_export": semantic_second_export,
            },
            "drain": stop_record(backend),
        }
    finally:
        if backend.engine is not None:
            backend.stop_persistent()


def image_launch(args, authority) -> dict:
    image_url = (
        "data:image/x-portable-pixmap;base64," +
        base64.b64encode(args.image.read_bytes()).decode("ascii")
    )
    backend = new_backend(args, authority)
    try:
        first = completion(backend, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "What object and color are shown? Answer briefly."},
        ]}], "phase6-image-first")
        second = completion(
            backend,
            [{"role": "user", "content": "Reply with only: done"}],
            "phase6-image-continuation",
        )
        return {
            "first": result_record(first.result),
            "second": result_record(second.result),
            "drain": stop_record(backend),
        }
    finally:
        if backend.engine is not None:
            backend.stop_persistent()


def text_then_image_reject(args, authority) -> dict:
    image_url = (
        "data:image/x-portable-pixmap;base64," +
        base64.b64encode(args.image.read_bytes()).decode("ascii")
    )
    backend = new_backend(args, authority)
    try:
        first = completion(
            backend, [{"role": "user", "content": "Reply with only: ready"}],
            "phase6-text-first",
        )
        before = state_snapshot(backend)
        error = None
        try:
            completion(backend, [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "What is shown?"},
            ]}], "phase6-late-image")
        except backend_module.RequestError as exc:
            error = str(exc)
        after = state_snapshot(backend)
        return {
            "first": result_record(first.result),
            "error": error,
            "state_unchanged": before == after,
            "drain": stop_record(backend),
        }
    finally:
        if backend.engine is not None:
            backend.stop_persistent()


def http_recovery(args, authority) -> dict:
    backend = new_backend(args, authority)
    handler = backend_module.handler_for(backend)
    original_send_json = handler._send_json
    failed_final = False

    def fail_first_completion(self, code, value):
        nonlocal failed_final
        if code == 200 and value.get("object") == "chat.completion" and not failed_final:
            failed_final = True
            raise BrokenPipeError("phase6 final-write disconnect")
        return original_send_json(self, code, value)

    handler._send_json = fail_first_completion
    server = backend_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def post(body: dict, key: str) -> tuple[int | None, dict | None, str | None]:
        connection = http.client.HTTPConnection(host, port, timeout=args.timeout)
        try:
            connection.request(
                "POST", "/v1/chat/completions", body=json.dumps(body),
                headers={"Content-Type": "application/json", "Idempotency-Key": key},
            )
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw) if raw else None, None
        except (http.client.RemoteDisconnected, BrokenPipeError,
                ConnectionResetError, json.JSONDecodeError) as exc:
            return None, None, type(exc).__name__
        finally:
            connection.close()

    try:
        first_body = {
            "model": backend.config.model_id,
            "messages": [{"role": "user", "content": "Reply with only: first"}],
            "stream": False,
        }
        first_failure = post(first_body, "phase6-final-write")
        handler._send_json = original_send_json
        first_retry = post(first_body, "phase6-final-write")

        def fail_delta(*_args, **_kwargs):
            raise BrokenPipeError("phase6 stream disconnect")

        handler._stream_chat_delta = fail_delta
        second_body = {
            "model": backend.config.model_id,
            "messages": [{"role": "user", "content": "Reply with only: second"}],
            "stream": True,
        }
        stream_failure = post(second_body, "phase6-stream-disconnect")
        second_body["stream"] = False
        second_retry = post(second_body, "phase6-stream-disconnect")
        return {
            "final_write": {
                "failure": first_failure,
                "retry": first_retry,
                "replayed": bool(first_retry[1] and first_retry[1]["x_salt"]["replayed"]),
            },
            "stream_disconnect": {
                "failure": stream_failure,
                "retry": second_retry,
                "replayed": bool(second_retry[1] and second_retry[1]["x_salt"]["replayed"]),
            },
            "status": state_snapshot(backend),
            "drain": stop_record(backend),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if backend.engine is not None:
            backend.stop_persistent()


def exact_equal(left: dict, right: dict) -> bool:
    keys = (
        "prompt_ids", "output_ids", "response_bytes", "response_sha256",
        "content", "reasoning_content", "stop_token", "finish_reason",
        "position_before", "position_after", "synthetic_close",
        "mindset_sha256", "facts_sha256", "state_sha256", "image_tokens",
    )
    return all(left[key] == right[key] for key in keys)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--auth-receipt", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    for path in (args.pool, args.receipt, args.image):
        if not path.is_file():
            raise QualificationError("required Phase 6 file is missing")
    if not args.source_dir.is_dir():
        raise QualificationError("required Phase 6 source directory is missing")

    build = backend_module._build_module.load_build_receipt(
        ROOT / "gemma4-server.build.json", ROOT / "gemma4-server",
        expected_source_compatibility=backend_module.KV_COMPATIBILITY.hexdigest,
    )
    auth_started = time.monotonic()
    authority = backend_module.Gemma4AuthenticationAuthority(
        backend_config(args),
    )
    authentication_s = time.monotonic() - auth_started
    try:
        delta = deterministic_launch(args, authority, full_history=False)
        full = deterministic_launch(args, authority, full_history=True)
        image = image_launch(args, authority)
        late_image = text_then_image_reject(args, authority)
        recovery = http_recovery(args, authority)
    finally:
        authority.close()

    observed_stops = sorted({
        delta["first"]["stop_token"], delta["second"]["stop_token"],
        full["first"]["stop_token"], full["second"]["stop_token"],
        image["first"]["stop_token"], image["second"]["stop_token"],
        late_image["first"]["stop_token"],
    })
    configured_stop_tokens = {
        1: "<eos>", 50: "<|tool_response>", 106: "<turn|>",
    }
    valid_terminal_values = {-1, *configured_stop_tokens}
    gates = {
        "fresh_turn1_exact": exact_equal(delta["first"], full["first"]),
        "delta_full_turn2_exact": exact_equal(delta["second"], full["second"]),
        "synthetic_close_continuable": (
            delta["first"]["synthetic_close"] == 1 and
            full["first"]["synthetic_close"] == 1 and
            delta["second"]["turn_id"] == 2 and full["second"]["turn_id"] == 2
        ),
        "committed_retry_exact": exact_equal(delta["second"], delta["retry"]),
        "committed_retry_replayed": delta["retry"]["replayed"],
        "altered_history_rejected_pre_start": (
            delta["altered_history"]["error"] is not None and
            delta["altered_history"]["state_unchanged"]
        ),
        "clear_preserves_mindset": delta["clear"]["preserved_mindset"],
        "clear_removes_facts": delta["clear"]["after"]["facts_rows"] == 0,
        "hard_reset_rotates_epoch": delta["hard_reset"]["epoch_rotated"],
        "hard_reset_preserves_build": delta["hard_reset"]["build_preserved"],
        "semantic_fact_turn1_exact": exact_equal(
            delta["semantic_qa"]["first"], full["semantic_qa"]["first"]
        ),
        "semantic_delta_full_exact": exact_equal(
            delta["semantic_qa"]["second"], full["semantic_qa"]["second"]
        ),
        "semantic_question_understood": all(
            "mira" in run["semantic_qa"]["second"]["content"].lower() and
            "north" in run["semantic_qa"]["second"]["content"].lower()
            for run in (delta, full)
        ),
        "semantic_mindset_consistent": (
            delta["semantic_qa"]["first"]["mindset_sha256"] ==
            delta["semantic_qa"]["second"]["mindset_sha256"] ==
            full["semantic_qa"]["first"]["mindset_sha256"] ==
            full["semantic_qa"]["second"]["mindset_sha256"]
        ),
        "semantic_facts_evolve_consistently": (
            delta["semantic_qa"]["first_export"]["facts_sha256"] !=
            "0" * 64 and
            delta["semantic_qa"]["second_export"]["facts_sha256"] !=
            "0" * 64 and
            delta["semantic_qa"]["first_export"]["facts_sha256"] !=
            delta["semantic_qa"]["second_export"]["facts_sha256"] and
            delta["semantic_qa"]["first_export"]["facts_sha256"] ==
            full["semantic_qa"]["first_export"]["facts_sha256"] and
            delta["semantic_qa"]["second_export"]["facts_sha256"] ==
            full["semantic_qa"]["second_export"]["facts_sha256"] and
            delta["semantic_qa"]["first_export"]["state_sha256"] ==
            full["semantic_qa"]["first_export"]["state_sha256"] and
            delta["semantic_qa"]["second_export"]["state_sha256"] ==
            full["semantic_qa"]["second_export"]["state_sha256"]
        ),
        "image_first_then_text": (
            image["first"]["image_tokens"] == 64 and image["second"]["turn_id"] == 2
        ),
        "text_then_image_rejected_pre_start": (
            late_image["error"] is not None and late_image["state_unchanged"]
        ),
        "post_commit_final_write_replayed": recovery["final_write"]["replayed"],
        "post_start_disconnect_replayed": recovery["stream_disconnect"]["replayed"],
        "graceful_drain": all(
            item["drain"]["phase"] == "CLOSED" and item["drain"]["returncode"] == 0
            for item in (delta, full, image, late_image, recovery)
        ),
        "natural_and_synthetic_close_observed": (
            106 in observed_stops and -1 in observed_stops
        ),
        "observed_terminal_values_valid": (
            set(observed_stops) <= valid_terminal_values
        ),
    }
    report = {
        "schema": "salt.gemma4.persistent-state.v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": os.uname().sysname,
            "machine": os.uname().machine,
        },
        "source_compatibility_sha256": backend_module.KV_COMPATIBILITY.hexdigest,
        "build_identity_sha256": build.hexdigest,
        "binary_sha256": build.binary_sha256,
        "runtime_ready": True,
        "authentication_s": authentication_s,
        "gates": gates,
        "configured_stop_tokens": configured_stop_tokens,
        "length_cap_sentinel": -1,
        "observed_stop_tokens": observed_stops,
        "runs": {
            "delta_history": delta,
            "full_history": full,
            "image_first": image,
            "late_image_rejection": late_image,
            "http_recovery": recovery,
        },
    }
    report["passed_gates"] = sum(bool(value) for value in gates.values())
    report["total_gates"] = len(gates)
    report["readiness_decision"] = "NO_SHIP" if not all(gates.values()) else "QUALIFIED_LOCAL_ONLY"
    atomic_json(args.report, report)
    print(json.dumps({
        "passed_gates": report["passed_gates"],
        "total_gates": report["total_gates"],
        "readiness_decision": report["readiness_decision"],
        "report": str(args.report),
    }, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
