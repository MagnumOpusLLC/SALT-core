#!/usr/bin/env python3
"""Gemma endpoint child with server-owned launch-envelope overrides only."""

from __future__ import annotations

import os
from pathlib import Path
import json
import re
import stat
import sys
import time

_repo_root = os.environ.get("SALT_REPO_ROOT")
MODEL_DIR = Path(__file__).resolve().parent
SERVER_DIR = (
    Path(_repo_root).resolve() / "server"
    if _repo_root else MODEL_DIR.parent
)
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import serve
import gemma4_backend


_backend_complete = gemma4_backend.Gemma4Backend.complete
_backend_stop_persistent = gemma4_backend.Gemma4Backend.stop_persistent

_NATIVE_DIAGNOSTIC_PREFIXES = (
    gemma4_backend.RETAINED_NATIVE_DIAGNOSTIC_PREFIXES
)


def _emit_native_diagnostics(raw_lines) -> None:
    for raw in raw_lines:
        line = raw.strip()
        if line.startswith(_NATIVE_DIAGNOSTIC_PREFIXES):
            print(
                "[gemma4-endpoint-native-diag] " +
                line.decode("utf-8", errors="replace"),
                file=sys.stderr,
                flush=True,
            )


def _native_diagnostic_cursor(engine) -> int:
    return int(getattr(engine, "diagnostic_count", 0)) if engine is not None else 0


def _native_diagnostics_since(engine, cursor: int) -> list[bytes]:
    retained = getattr(engine, "diagnostic_logs", None) if engine is not None else None
    if retained is None:
        return list(getattr(engine, "logs", ())) if engine is not None else []
    return [line for sequence, line in retained if sequence > cursor]


def _emit_new_native_diagnostics(owner, engine) -> None:
    cursor = int(getattr(owner, "_native_diagnostic_cursor", 0))
    _emit_native_diagnostics(_native_diagnostics_since(engine, cursor))
    owner._native_diagnostic_cursor = _native_diagnostic_cursor(engine)


def _complete_with_accounting(self, *args, **kwargs):
    try:
        completion = _backend_complete(self, *args, **kwargs)
    except BaseException:
        time.sleep(0.1)
        engine = getattr(self, "engine", None)
        logs = getattr(engine, "logs", ()) if engine is not None else ()
        tail = b"".join(logs)[-16384:].decode("utf-8", errors="replace")
        print(
            "[gemma4-endpoint-native-tail]\n" + tail,
            file=sys.stderr,
            flush=True,
        )
        raise
    stderr_mode = os.environ.get("SALT_SERVER_STDERR", "summary")
    if stderr_mode == "waterfall":
        print(
            "[gemma4-endpoint-waterfall] " + json.dumps(
                completion.result.dpr_waterfall,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
    return completion


gemma4_backend.Gemma4Backend.complete = _complete_with_accounting


def _stop_with_diagnostics(self, *args, **kwargs):
    return _backend_stop_persistent(self, *args, **kwargs)


gemma4_backend.Gemma4Backend.stop_persistent = _stop_with_diagnostics


def _load_startup_cache_spec() -> dict | None:
    raw_path = os.environ.get("GEMMA4_STARTUP_KV_MANIFEST")
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    try:
        before = path.lstat()
    except OSError as exc:
        raise SystemExit(f"could not inspect startup KV manifest: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise SystemExit("startup KV manifest must be a regular non-symlink file")
    if before.st_mode & 0o077:
        raise SystemExit("startup KV manifest must not grant group/other permissions")
    if not 1 <= before.st_size <= 1_048_576:
        raise SystemExit("startup KV manifest size must be in [1, 1048576]")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if ((opened.st_dev, opened.st_ino, opened.st_size) !=
                    (before.st_dev, before.st_ino, before.st_size)):
                raise SystemExit("startup KV manifest changed while opening")
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    raise SystemExit("startup KV manifest was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise SystemExit(f"could not open startup KV manifest: {exc}") from exc
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("startup KV manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {
            "schema", "cache_id", "manifest"}:
        raise SystemExit("invalid startup KV manifest fields")
    if value["schema"] != "salt.gemma4.startup-kv.v1":
        raise SystemExit("unsupported startup KV manifest schema")
    if not isinstance(value["cache_id"], str) or not value["cache_id"]:
        raise SystemExit("startup KV cache id must be a nonempty string")
    if not isinstance(value["manifest"], dict):
        raise SystemExit("startup KV session manifest must be an object")
    return value


def _install_startup_cache_hook(spec: dict) -> None:
    original_start = gemma4_backend.Gemma4Backend.start_persistent

    def start_with_cache(self, *args, **kwargs):
        started_at = time.monotonic()
        original_start(self, *args, **kwargs)
        native_ready_at = time.monotonic()
        if os.environ.get("GEMMA4_GPU_DIAG") == "1":
            engine = getattr(self, "engine", None)
            _emit_new_native_diagnostics(self, engine)
        try:
            imported = self.import_session_cache(
                spec["cache_id"], spec["manifest"],
            )
        except BaseException:
            self.stop_persistent()
            raise
        imported_at = time.monotonic()
        if (imported.get("phase") != "READY" or
                imported.get("runtime_ready") is not True or
                not isinstance(imported.get("position"), int) or
                imported["position"] <= 0):
            self.stop_persistent()
            raise gemma4_backend.BackendError(
                "startup KV import did not publish a ready nonempty session"
            )
        print(
            "[gemma4-endpoint-startup-kv] " + json.dumps({
                "cache_id": spec["cache_id"],
                "facts_rows": imported["facts_rows"],
                "facts_sha256": imported["facts_sha256"],
                "import_s": imported_at - native_ready_at,
                "native_start_s": native_ready_at - started_at,
                "position": imported["position"],
                "state_sha256": imported["state_sha256"],
                "total_s": imported_at - started_at,
            }, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )

    gemma4_backend.Gemma4Backend.start_persistent = start_with_cache


def main() -> int:
    args = serve.build_parser().parse_args()
    if args.backend != "gemma4":
        raise SystemExit("gemma4 endpoint child requires --backend gemma4")
    if args.model is None:
        args.model = gemma4_backend.DEFAULT_MODEL_ID

    startup_cache = _load_startup_cache_spec()
    if startup_cache is not None:
        if not args.gemma_kv_cache_root:
            raise SystemExit(
                "GEMMA4_STARTUP_KV_MANIFEST requires --gemma-kv-cache-root"
            )
        _install_startup_cache_hook(startup_cache)

    memory_limit = os.environ.get("GEMMA4_MEMORY_LIMIT_GB")
    if memory_limit is not None:
        if re.fullmatch(r"(?:[1-9][0-9]{0,2})(?:\.[0-9]{1,9})?", memory_limit) is None:
            raise SystemExit("GEMMA4_MEMORY_LIMIT_GB must be a positive decimal")
        parsed = float(memory_limit)
        if not 1.0 <= parsed <= 128.0:
            raise SystemExit("GEMMA4_MEMORY_LIMIT_GB must be in [1, 128]")
        gemma4_backend.GEMMA_ENGINE_CONFIG[
            "SALT_GEMMA_MEMORY_LIMIT_GB"
        ] = memory_limit
        print(
            f"[gemma4-endpoint-child] native memory limit={memory_limit}GB",
            file=sys.stderr,
            flush=True,
        )

    dpr_draft_n = os.environ.get("GEMMA4_DPR_DRAFT_N")
    if dpr_draft_n is not None:
        if re.fullmatch(r"[1-9][0-9]*", dpr_draft_n) is None or not (
            1 <= int(dpr_draft_n) <= 356
        ):
            raise SystemExit("GEMMA4_DPR_DRAFT_N must be in [1, 356]")
        gemma4_backend.GEMMA_ENGINE_CONFIG["SALT_DPR_DRAFT_N"] = dpr_draft_n
        print(
            f"[gemma4-endpoint-child] DPR draft N={dpr_draft_n}",
            file=sys.stderr,
            flush=True,
        )

    gpu_diag = os.environ.get("GEMMA4_GPU_DIAG")
    if gpu_diag is not None:
        if gpu_diag not in ("0", "1"):
            raise SystemExit("GEMMA4_GPU_DIAG must be 0 or 1")
        gemma4_backend.GEMMA_ENGINE_CONFIG["SALT_GPU_DIAG"] = gpu_diag

    return gemma4_backend.serve_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
