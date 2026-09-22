#!/usr/bin/env python3
"""No-model contract tests for selectable cold-start Gemma 4 KV caches."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "server" / "model" / "gemma4_backend.py"

spec = importlib.util.spec_from_file_location("salt_gemma4_kv_server", BACKEND)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def expect_request_error(function, text: str) -> None:
    try:
        function()
    except module.RequestError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected RequestError containing {text!r}")


def write_cache(path: Path, *, context: int = 512, position: int = 1,
                 compatibility_sha256: bytes | None = None,
                 logical_state_override: bytes | None = None) -> None:
    payload = bytes(module.gemma4_kv_portable_bytes(position))
    dimensions = tuple(
        1024 if layer % 6 == 5 else 2048
        for layer in range(module.GEMMA4_LAYERS)
    )
    compatibility = compatibility_sha256 or module.KV_COMPATIBILITY_SHA256
    logical = hashlib.sha256()
    logical.update(b"G4STATE6")
    logical.update(compatibility)
    logical.update(struct.pack("<III", module.GEMMA4_LAYERS, context, position))
    offset = 0
    sliding_base = max(0, position - 1023)
    for layer, dimension in enumerate(dimensions):
        base = 0 if layer % 6 == 5 else sliding_base
        segments = 1 if layer % 6 == 5 else 2
        count = (position - base) * dimension * segments * 4
        logical.update(struct.pack("<II", dimension, base))
        logical.update(payload[offset:offset + count])
        offset += count
    assert offset == len(payload)
    header = module.KV_CACHE_HEADER.pack(
        module.KV_CACHE_MAGIC,
        module.KV_CACHE_VERSION,
        module.GEMMA4_LAYERS,
        context,
        position,
        module.KV_CACHE_HEADER.size + len(payload),
        *dimensions,
        compatibility,
        logical_state_override or logical.digest(),
        hashlib.sha256(payload).digest(),
        sliding_base,
        1024,
    )
    path.write_bytes(header + payload)


def main() -> None:
    server_target_keys = (
        "GEMMA4_TARGET_ROUTE_N",
        "GEMMA4_TARGET_X",
        "GEMMA4_TARGET_ROUTE_F",
        "GEMMA4_TARGET_ROUTE_Q",
    )
    prior_server_target = {
        key: module.os.environ.get(key) for key in server_target_keys
    }
    try:
        for key in server_target_keys:
            module.os.environ.pop(key, None)
        recipe_route = module.Gemma4Backend._clean_env()
        for server_key in server_target_keys:
            engine_key = "SALT_" + server_key.removeprefix("GEMMA4_")
            assert recipe_route.get(engine_key) == \
                module.GEMMA_ENGINE_CONFIG.get(engine_key), recipe_route
        module.os.environ.update({
            "GEMMA4_TARGET_ROUTE_N": "32",
            "GEMMA4_TARGET_X": "32",
            "GEMMA4_TARGET_ROUTE_F": "1",
            "GEMMA4_TARGET_ROUTE_Q": "4",
        })
        server_route = module.Gemma4Backend._clean_env()
        assert server_route["SALT_TARGET_ROUTE_N"] == "32", server_route
        assert server_route["SALT_TARGET_X"] == "32", server_route
        assert server_route["SALT_TARGET_ROUTE_F"] == "1", server_route
        assert server_route["SALT_TARGET_ROUTE_Q"] == "4", server_route
        del module.os.environ["GEMMA4_TARGET_ROUTE_Q"]
        try:
            module.Gemma4Backend._clean_env()
            raise AssertionError("partial normal-server target route accepted")
        except module.BackendError as exc:
            assert "normal-server target route" in str(exc), exc
    finally:
        for key, value in prior_server_target.items():
            if value is None:
                module.os.environ.pop(key, None)
            else:
                module.os.environ[key] = value

    prior_retain = module.os.environ.get("SALT_PREFILL_RETAIN_LAYERS")
    prior_lookahead = module.os.environ.get("SALT_PREFILL_DECODE_LOOKAHEAD")
    try:
        module.os.environ["SALT_PREFILL_RETAIN_LAYERS"] = "0"
        release = module.Gemma4Backend._clean_env()
        assert release["SALT_PREFILL_RETAIN_LAYERS"] == "0", release
        module.os.environ["SALT_PREFILL_RETAIN_LAYERS"] = "2"
        try:
            module.Gemma4Backend._clean_env()
            raise AssertionError("invalid prefill retention override accepted")
        except module.BackendError as exc:
            assert "must be 0 or 1" in str(exc), exc
    finally:
        if prior_retain is None:
            module.os.environ.pop("SALT_PREFILL_RETAIN_LAYERS", None)
        else:
            module.os.environ["SALT_PREFILL_RETAIN_LAYERS"] = prior_retain

    try:
        module.os.environ["SALT_PREFILL_DECODE_LOOKAHEAD"] = "1"
        enabled = module.Gemma4Backend._clean_env()
        assert enabled["SALT_PREFILL_DECODE_LOOKAHEAD"] == "1", enabled
        module.os.environ["SALT_PREFILL_DECODE_LOOKAHEAD"] = "2"
        try:
            module.Gemma4Backend._clean_env()
            raise AssertionError("invalid phase lookahead override accepted")
        except module.BackendError as exc:
            assert "must be 0 or 1" in str(exc), exc
    finally:
        if prior_lookahead is None:
            module.os.environ.pop("SALT_PREFILL_DECODE_LOOKAHEAD", None)
        else:
            module.os.environ["SALT_PREFILL_DECODE_LOOKAHEAD"] = prior_lookahead

    controls = module.resolve_controls({}, server_kv_budget_gb=1.0)
    different_context = module.Controls(
        200, 128, "none", False, 1.0, 90_112_000,
    )

    continuation = module.render_chat(
        [{"role": "user", "content": "Continue."}],
        enable_thinking=False,
        continuation=True,
    )
    assert continuation.prompt == (
        "\n<|turn>user\nContinue.<turn|>\n<|turn>model\n"
        "<|channel>thought\n<channel|>"
    )
    assert "<bos>" not in continuation.prompt
    expect_request_error(
        lambda: module.render_chat(
            [
                {"role": "system", "content": "replace policy"},
                {"role": "user", "content": "Continue."},
            ],
            enable_thinking=False,
            continuation=True,
        ),
        "continuation must contain exactly one user message",
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_identity = b"\x11" * 32
        store = module.KVCacheStore(
            root, module.KV_COMPATIBILITY_SHA256, build_identity,
        )
        assert store.base_root == root.resolve()
        assert store.root.parent == root.resolve()
        assert store.root.name == (
            "g4kvc006-" + module.KV_COMPATIBILITY_SHA256.hex()
        )
        other_build_store = module.KVCacheStore(
            root, module.KV_COMPATIBILITY_SHA256, b"\x22" * 32,
        )
        assert other_build_store.root == store.root

        create = store.resolve(
            {"mode": "create", "id": "policy-a"}, controls,
        )
        assert create.mode == "create"
        assert create.cache_id == "policy-a"
        assert create.load_path is None
        assert create.save_path is not None
        assert create.save_path.parent == store.root

        expect_request_error(
            lambda: store.resolve(
                {"mode": "load", "id": "../outside"}, controls,
            ),
            "invalid KV cache id",
        )
        expect_request_error(
            lambda: store.resolve(
                {"mode": "load", "id": "missing"}, controls,
            ),
            "does not exist",
        )

        corrupt = store.resolve(
            {"mode": "create", "id": "corrupt-pending"}, controls,
        )
        assert corrupt.save_path is not None
        assert corrupt.target_path is not None
        write_cache(corrupt.save_path)
        with corrupt.save_path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x01")
        expect_request_error(
            lambda: store.publish(corrupt, 1),
            "payload SHA-256 mismatch",
        )
        assert not corrupt.target_path.exists()
        store.discard(corrupt)

        race_initial = store.resolve({"mode": "create", "id": "commit-race"}, controls)
        write_cache(race_initial.save_path)
        race_metadata = store.publish(race_initial, 1)
        commit_race = store.resolve(
            {"mode": "resume", "id": "commit-race"}, controls,
        )
        assert commit_race.save_path is not None
        write_cache(commit_race.save_path, position=2)
        real_replace = module.os.replace

        def replace_then_corrupt(source, target) -> None:
            real_replace(source, target)
            if Path(target).name == "metadata.json":
                record = json.loads(Path(target).read_text())
                with (Path(target).parent / record["payload_file"]).open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"\x01")

        module.os.replace = replace_then_corrupt
        try:
            expect_request_error(
                lambda: store.publish(commit_race, 2),
                "payload SHA-256 mismatch",
            )
        finally:
            module.os.replace = real_replace
        for path in race_metadata.path.iterdir():
            path.unlink()
        race_metadata.path.rmdir()
        store.discard(commit_race)

        incompatible_path = store.root / "incompatible.g4kv"
        write_cache(incompatible_path, compatibility_sha256=b"\x7f" * 32)
        expect_request_error(
            lambda: store.resolve(
                {"mode": "load", "id": "incompatible"}, controls,
            ),
            "compatibility identity mismatch",
        )
        incompatible_path.unlink()

        native_state_path = store.root / "native-state.g4kv"
        write_cache(
            native_state_path, logical_state_override=b"\x7e" * 32,
        )
        native_state = store.resolve(
            {"mode": "load", "id": "native-state"}, controls,
        )
        assert native_state.load_path == native_state_path.resolve()
        assert store.load_metadata(
            "native-state",
        ).logical_state_sha256 == "7e" * 32
        native_state_path.unlink()

        cache_path = store.root / "policy-a.g4kv"
        write_cache(cache_path)
        loaded = store.resolve(
            {"mode": "load", "id": "policy-a"}, controls,
        )
        assert loaded.load_path == cache_path.resolve()
        assert loaded.save_path is None
        assert loaded.loaded_tokens == 1

        resumed = store.resolve(
            {"mode": "resume", "id": "policy-a"}, controls,
        )
        assert resumed.load_path == cache_path.resolve()
        assert resumed.save_path is not None
        assert resumed.save_path != cache_path

        expect_request_error(
            lambda: store.resolve(
                {"mode": "load", "id": "policy-a"}, different_context,
            ),
            "context gate",
        )
        expect_request_error(
            lambda: store.resolve(
                {"mode": "create", "id": "policy-a"}, controls,
            ),
            "already exists",
        )

        listed = store.list_caches()
        assert [(item.cache_id, item.position) for item in listed] == [
            ("policy-a", 1),
        ]

        class ExportEngine:
            def export_kv(self, path: Path) -> dict:
                write_cache(path)
                return {
                    "position": 1,
                    "state_sha256": "a" * 64,
                    "facts_sha256": "b" * 64,
                }

        backend = module.Gemma4Backend(module.Gemma4Config(
            source_dir=root, pool=root, receipt=root,
            kv_cache_root=root / "exports",
        ))
        backend.engine = ExportEngine()
        backend.cache_store = module.KVCacheStore(
            root / "exports", module.KV_COMPATIBILITY_SHA256,
            bytes.fromhex("55" * 32),
        )
        exported = backend.export_cache("session-one")
        assert exported["object"] == "salt.kv_cache"
        assert exported["id"] == "session-one"
        assert exported["position"] == 1
        assert exported["legacy_read_only"] is False
        assert len(backend.cache_store.list_caches()) == 1
        assert exported["storage_format"] == "payload-with-sidecar"
        artifact = backend.cache_store.load_metadata("session-one")
        record_path = artifact.path / "metadata.json"
        record = json.loads(record_path.read_text())
        payload = artifact.path / record["payload_file"]
        original_payload = payload.read_bytes()
        assert hashlib.sha256(original_payload).hexdigest() == record["payload_sha256"]
        assert len(original_payload) == module.gemma4_kv_portable_bytes(1)
        assert not original_payload.startswith(module.KV_CACHE_MAGIC)
        expected_frame = root / "expected-frame.g4kv"
        write_cache(expected_frame)
        with backend.cache_store.native_path(artifact) as framed:
            assert framed.read_bytes() == expected_frame.read_bytes()
            import_path = framed
        assert not import_path.exists()
        assert not list(backend.cache_store.root.glob(".kv-import-*"))

        runtime_root = root / "runtime-exports"
        runtime_backend = module.Gemma4Backend(module.Gemma4Config(
            source_dir=root, pool=root, receipt=root,
        ))
        runtime_backend.engine = ExportEngine()
        runtime_backend.build_identity_sha256 = "55" * 32
        prior_runtime_root = getattr(module, "DEFAULT_RUNTIME_KV_CACHE_ROOT")
        setattr(module, "DEFAULT_RUNTIME_KV_CACHE_ROOT", runtime_root)
        try:
            capabilities = runtime_backend.capabilities()["kv_cache"]
            assert capabilities["enabled"] is True
            assert capabilities["engaged"] is False
            assert capabilities["modes"] == ["export", "import"]
            assert capabilities["selector"] == "server-owned-runtime-store"
            assert not runtime_root.exists()
            runtime_exported = runtime_backend.export_cache("runtime-session")
            assert runtime_exported["id"] == "runtime-session"
            assert runtime_exported["position"] == 1
            assert runtime_backend.cache_store is not None
            assert runtime_backend.cache_store.base_root == runtime_root.resolve()
            assert stat.S_IMODE(runtime_root.stat().st_mode) == 0o700
            assert runtime_backend.capabilities()["kv_cache"]["engaged"] is True
        finally:
            setattr(module, "DEFAULT_RUNTIME_KV_CACHE_ROOT", prior_runtime_root)

        record["metadata"]["notes"] = "a corrected note"
        record["metadata"]["created_by"] = "curator-test"
        record["metadata"]["build_identity_sha256"] = "66" * 32
        record["metadata"]["inference"] = {"seed": 123, "top_k": 8}
        record_path.write_text(json.dumps(record))
        changed_metadata = backend.cache_store.load_metadata("session-one")
        assert changed_metadata.payload_sha256 == artifact.payload_sha256
        assert changed_metadata.logical_state_sha256 == artifact.logical_state_sha256
        assert changed_metadata.build_identity_sha256 == "66" * 32
        assert payload.read_bytes() == original_payload
        with backend.cache_store.native_path(changed_metadata) as framed:
            assert framed.read_bytes() == expected_frame.read_bytes()
        assert not framed.exists()

        try:
            with backend.cache_store.native_path(changed_metadata) as framed:
                raise RuntimeError("fixture import failure")
        except RuntimeError:
            pass
        assert not framed.exists()

        record["payload_file"] = "../escape.kv"
        record_path.write_text(json.dumps(record))
        expect_request_error(
            lambda: backend.cache_store.load_metadata("session-one"),
            "invalid KV sidecar schema",
        )
        record["payload_file"] = artifact.payload_sha256 + ".kv"
        record_path.write_text(json.dumps(record))
        with payload.open("r+b") as stream:
            stream.write(b"\x01")
        expect_request_error(
            lambda: backend.cache_store.load_metadata("session-one"),
            "payload SHA-256 mismatch",
        )
        assert backend.cache_store.delete("session-one")
        assert not backend.cache_store.contains("session-one")

    print("gemma4 KV cache contract: PASS")


if __name__ == "__main__":
    main()
