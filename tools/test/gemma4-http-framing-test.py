#!/usr/bin/env python3
"""Raw-socket HTTP framing contracts for the Gemma 4 handler."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "server" / "model" / "gemma4_backend.py"

spec = importlib.util.spec_from_file_location("salt_gemma4_http_framing", BACKEND)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeBackend:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_id=module.DEFAULT_MODEL_ID)
        self.cache_store = None
        self.lock = threading.Lock()
        self.calls = 0

    def capabilities(self) -> dict:
        return {"runtime_ready": module.RUNTIME_READY}

    def resolve_request_identity(self, body, endpoint, request_key):
        request_sha = module._request_sha256(body, endpoint)
        return module._client_request_sha256(
            request_key, session_epoch=1, turn_id=self.calls,
            request_sha256=request_sha,
        ), request_sha

    @staticmethod
    def transport_identity(client_request_sha256):
        return 0, client_request_sha256[:32]

    def complete(self, body: dict, **_kwargs):
        self.calls += 1
        return module.Completion(
            result=module.RunnerResult(
                content="ok", reasoning_content=None, prompt_tokens=1,
                completion_tokens=1, output_ids=(106,), stop_token=106,
                finish_reason="stop", response_bytes=2,
                kv_loaded_tokens=0, kv_saved_tokens=0,
            ),
            controls=module.Controls(
                context_tokens=200, output_tokens=50,
                reasoning_effort="none", enable_thinking=False,
                kv_allowance_gb=0.7, kv_bytes=90_112_000,
            ),
            modality="text", elapsed_s=0.0,
            kv_cache=module.KVCacheOutcome(
                mode="none", cache_id=None, loaded_tokens=0, saved_tokens=0,
            ),
        )


def exchange(port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as client:
        client.settimeout(2.0)
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def assert_one_closed_400(response: bytes, label: str) -> None:
    assert response.count(b"HTTP/1.1 ") == 1, (label, response)
    assert response.startswith(b"HTTP/1.1 400 "), (label, response)
    headers = response.split(b"\r\n\r\n", 1)[0].lower()
    assert b"connection: close" in headers, (label, headers)


def main() -> None:
    backend = FakeBackend()
    assert module.RUNTIME_READY is True
    assert backend.capabilities()["runtime_ready"] is True
    server = module.ThreadingHTTPServer(
        ("127.0.0.1", 0), module.handler_for(backend),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    body = json.dumps({
        "model": module.DEFAULT_MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
    }, separators=(",", ":")).encode("utf-8")
    embedded = b"GET /healthz HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n"
    try:
        oversized = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n" +
            f"Content-Length: {module.MAX_REQUEST_BYTES + 1}\r\n\r\n".encode() +
            embedded
        )
        assert_one_closed_400(exchange(port, oversized), "oversized")

        get_body = (
            b"GET /healthz HTTP/1.1\r\nHost: local\r\n" +
            f"Content-Length: {len(embedded)}\r\n\r\n".encode() + embedded
        )
        assert_one_closed_400(exchange(port, get_body), "GET body")

        duplicate = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n" +
            f"Content-Length: {len(body)}\r\n".encode() +
            f"Content-Length: {len(body) + 1}\r\n\r\n".encode() + body
        )
        assert_one_closed_400(exchange(port, duplicate), "duplicate length")

        transfer = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n" +
            b"Transfer-Encoding: chunked\r\n" +
            f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        assert_one_closed_400(exchange(port, transfer), "transfer encoding")

        truncated = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n" +
            f"Content-Length: {len(body) + 10}\r\n\r\n".encode() + body
        )
        assert_one_closed_400(exchange(port, truncated), "truncated")

        expect = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n" +
            b"Expect: 100-continue\r\n" +
            f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        expect_response = exchange(port, expect)
        assert expect_response.count(b"HTTP/1.1 ") == 1, expect_response
        assert expect_response.startswith(b"HTTP/1.1 417 "), expect_response
        assert b"connection: close" in expect_response.split(
            b"\r\n\r\n", 1,
        )[0].lower()

        put = (
            b"PUT /healthz HTTP/1.1\r\nHost: local\r\n" +
            f"Content-Length: {len(embedded)}\r\n\r\n".encode() + embedded
        )
        put_response = exchange(port, put)
        assert put_response.count(b"HTTP/1.1 ") == 1, put_response
        assert put_response.startswith(b"HTTP/1.1 405 "), put_response
        assert b"connection: close" in put_response.split(
            b"\r\n\r\n", 1,
        )[0].lower()
        assert backend.calls == 0, backend.calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    print("gemma4 HTTP framing contract: PASS")


if __name__ == "__main__":
    main()
