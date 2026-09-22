#!/usr/bin/env python3
"""Authenticated native Gemma 4 one-image multimodal QA launcher.

The pinned MLX checkpoint remains an offline authenticated encoding. This tool
retains its source and expert-pool descriptors, emits the existing private text
binding followed by the existing private vision binding, renders exactly one
pinned image placeholder block, and invokes the native C QA runner. It never
enables serving or runtime readiness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import BinaryIO

from engine_config import load_engine_config, validate_gemma_residency_config

ROOT = Path(__file__).resolve().parents[1]
ENGINE_CONFIG = load_engine_config(
    model_dir=ROOT / "models" / "gemma4-26b-a4b",
    recipe=os.environ.get("SALT_GEMMA_PLATFORM_RECIPE"),
)
PACKAGE_TOOL = ROOT / "tools" / "gemma4-package.py"
BUILD_IDENTITY_TOOL = ROOT / "server" / "gemma4_build.py"
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
BUILD_IDENTITY_TOOL = ROOT / "server" / "model" / "gemma4_build.py"
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
TEXT_QA_TOOL = ROOT / "tools" / "gemma4-qa.py"
VISION_QA_TOOL = ROOT / "tools" / "gemma4-vision-qa.py"
DEFAULT_PLAN = ROOT / "models" / "gemma4-26b-a4b" / "package-plan.json"
DEFAULT_SOURCE_CONTRACT = (
    ROOT / "models" / "gemma4-26b-a4b" / "mlx4-source" / "source-contract.json"
)
DEFAULT_ROLE_CONTRACT = (
    ROOT / "models" / "gemma4-26b-a4b" / "mlx4-package" / "source-role-contract.json"
)
SOURCE_CONTRACT_SHA256 = (
    "312e73b836f39d06f5c982f4b5a83575d7c4bb23539b787293a247cd2bc71949"
)
ROLE_MAP_SHA256 = (
    "fb4980b498cbe3985220560eca5ab0636fd1298d2f05c94f6367c93fb129c46c"
)
IMAGE_SOFT_TOKENS = 64
ALLOWED_QA_GATES = {(200, 50), (500, 200), (1500, 300)}
AUTH_RECEIPT_SCHEMA = "salt.gemma4-authenticated-package.v2"


class MultimodalQaError(RuntimeError):
    pass


@dataclass
class AuthenticatedBinding:
    handles: dict[str, BinaryIO]
    pool_handle: BinaryIO
    identities: dict[str, tuple[int, int, int, int, int]]
    report: dict
    role_map: dict
    pool_identity: tuple[int, int, int, int, int]
    binding: bytes
    compatibility_sha256: str
    build_identity_sha256: str
    closed: bool = False

    @property
    def inherited_fds(self) -> tuple[int, ...]:
        if self.closed:
            raise MultimodalQaError("authenticated binding authority is closed")
        return tuple(handle.fileno() for handle in self.handles.values()) + (
            self.pool_handle.fileno(),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pool_handle.close()
        for handle in self.handles.values():
            handle.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _write_auth_receipt(
    path: Path, binding: AuthenticatedBinding, *, source_dir: Path, pool: Path,
    pool_receipt: Path, source_summary: dict,
) -> None:
    state = source_summary["_state"]
    source_files = {}
    for name, identity in sorted(binding.identities.items()):
        entry = state["tree"].get(name)
        if not entry or entry["storage"] != "lfs":
            raise MultimodalQaError("authenticated source receipt closure drift")
        source_files[name] = {
            "identity": list(identity),
            "sha256": entry["sha256"],
        }
    receipt = {
        "schema": AUTH_RECEIPT_SCHEMA,
        "source_dir": str(source_dir.resolve()),
        "pool": str(pool.resolve()),
        "pool_receipt": str(pool_receipt.resolve()),
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "role_map_sha256": ROLE_MAP_SHA256,
        "copy_map_sha256": source_summary["copy_map_sha256"],
        "source_files": source_files,
        "pool_identity": list(binding.pool_identity),
        "pool_sha256": "b06c91ecfae36581cbafdda80e116446bb32ea5ec832066e86cb672421a66e31",
        "pool_receipt_sha256": _file_sha256(pool_receipt),
        "runtime_ready": False,
    }
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = path.expanduser()
    parent = target.parent.resolve(strict=True)
    target = parent / target.name
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MultimodalQaError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _render_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise MultimodalQaError("question is empty")
    image_block = "<|image>" + "<|image|>" * IMAGE_SOFT_TOKENS + "<image|>"
    return (
        "<bos><|turn>user\n" + image_block + question +
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def authenticate_binding(
    *, source_dir: Path, pool: Path, receipt: Path, runner: Path,
    build_receipt: Path, kv_compat_sha256: str | None = None,
    auth_receipt: Path | None = None,
    source_contract: Path = DEFAULT_SOURCE_CONTRACT,
    role_contract: Path = DEFAULT_ROLE_CONTRACT,
    plan_path: Path = DEFAULT_PLAN,
) -> AuthenticatedBinding:
    handles: dict[str, BinaryIO] = {}
    pool_handle: BinaryIO | None = None
    try:
        text_qa = _load_module(TEXT_QA_TOOL, "salt_gemma4_mm_text_qa")
        setattr(text_qa, "ENGINE_CONFIG", ENGINE_CONFIG)
        kv_compatibility = text_qa._kv_compatibility()
        if (kv_compat_sha256 is not None and
                kv_compat_sha256 != kv_compatibility.hexdigest):
            raise MultimodalQaError("KV compatibility authority drift")
        if not runner.is_file():
            raise MultimodalQaError(f"Gemma 4 runner does not exist: {runner}")
        build_module = _load_module(
            BUILD_IDENTITY_TOOL, "salt_gemma4_multimodal_build",
        )
        build_identity = build_module.load_build_receipt(
            build_receipt, runner,
            expected_source_compatibility=kv_compatibility.hexdigest,
        )
        vision_qa = _load_module(VISION_QA_TOOL, "salt_gemma4_mm_vision_qa")
        package = _load_module(PACKAGE_TOOL, "salt_gemma4_mm_package")
        source_module = package._load_mlx_source_module()
        plan = package.load_json(plan_path)
        package.validate_plan(plan)
        source_summary = source_module.validate_source(source_contract)
        if (source_summary["contract_sha256"] != SOURCE_CONTRACT_SHA256 or
                source_summary["runtime_ready"] is not False):
            raise MultimodalQaError("authenticated source summary drift")
        report, handles, identities = source_module._authenticate_downloads(
            source_dir, source_summary,
        )
        if report["authenticated_bytes"] != 15_373_375_402 or \
                report["runtime_ready"] is not False:
            raise MultimodalQaError("download authentication summary drift")
        role_map = package.build_mlx_source_role_map(
            plan, source_contract=source_contract,
            role_contract=role_contract, source_summary=source_summary,
        )
        if role_map["role_map_sha256"] != ROLE_MAP_SHA256 or \
                role_map["runtime_ready"] is not True:
            raise MultimodalQaError("role-map identity or readiness drift")
        pool_handle, pool_identity, _ = text_qa._validate_pool(
            pool, receipt, source_summary,
        )
        if pool_handle is None:
            raise MultimodalQaError("expert pool handle was not retained")
        text_binding = text_qa._build_binding(
            role_map, handles, identities, pool_handle, pool_identity,
            kv_compatibility.hexdigest, build_identity.hexdigest,
        )
        vision_binding, _ = vision_qa._build_binding(
            role_map, handles, identities,
        )
        binding = AuthenticatedBinding(
            handles=handles, pool_handle=pool_handle, identities=identities,
            report=report, role_map=role_map, pool_identity=pool_identity,
            binding=text_binding + vision_binding,
            compatibility_sha256=kv_compatibility.hexdigest,
            build_identity_sha256=build_identity.hexdigest,
        )
        if auth_receipt is not None:
            _write_auth_receipt(
                auth_receipt, binding, source_dir=source_dir, pool=pool,
                pool_receipt=receipt, source_summary=source_summary,
            )
        return binding
    except BaseException:
        if pool_handle is not None:
            pool_handle.close()
        for handle in handles.values():
            handle.close()
        raise


def _read_auth_receipt(path: Path) -> dict:
    target = path.expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise MultimodalQaError(
            f"cannot open authenticated package receipt {target}: {exc}"
        ) from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or
                before.st_mode & 0o077 or before.st_size < 2 or
                before.st_size > 1 << 20):
            raise MultimodalQaError("unsafe authenticated package receipt")
        raw = os.read(fd, before.st_size + 1)
        after = os.fstat(fd)
        if len(raw) != before.st_size or (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns):
            raise MultimodalQaError(
                "authenticated package receipt changed while reading"
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MultimodalQaError(
                "invalid authenticated package receipt JSON"
            ) from exc
        if not isinstance(value, dict):
            raise MultimodalQaError(
                "authenticated package receipt root is not an object"
            )
        return value
    finally:
        os.close(fd)


def open_authenticated_binding(
    *, source_dir: Path, pool: Path, receipt: Path, runner: Path,
    build_receipt: Path, auth_receipt: Path,
    kv_compat_sha256: str | None = None,
    source_contract: Path = DEFAULT_SOURCE_CONTRACT,
    role_contract: Path = DEFAULT_ROLE_CONTRACT,
    plan_path: Path = DEFAULT_PLAN,
) -> AuthenticatedBinding:
    handles: dict[str, BinaryIO] = {}
    pool_handle: BinaryIO | None = None
    try:
        text_qa = _load_module(TEXT_QA_TOOL, "salt_gemma4_cached_text_qa")
        kv_compatibility = text_qa._kv_compatibility()
        if (kv_compat_sha256 is not None and
                kv_compat_sha256 != kv_compatibility.hexdigest):
            raise MultimodalQaError("KV compatibility authority drift")
        if not runner.is_file():
            raise MultimodalQaError(f"Gemma 4 runner does not exist: {runner}")
        build_module = _load_module(
            BUILD_IDENTITY_TOOL, "salt_gemma4_cached_build",
        )
        build_identity = build_module.load_build_receipt(
            build_receipt, runner,
            expected_source_compatibility=kv_compatibility.hexdigest,
        )
        vision_qa = _load_module(
            VISION_QA_TOOL, "salt_gemma4_cached_vision_qa",
        )
        package = _load_module(PACKAGE_TOOL, "salt_gemma4_cached_package")
        source_module = package._load_mlx_source_module()
        plan = package.load_json(plan_path)
        package.validate_plan(plan)
        source_summary = source_module.validate_source(source_contract)
        role_map = package.build_mlx_source_role_map(
            plan, source_contract=source_contract,
            role_contract=role_contract, source_summary=source_summary,
        )
        cached = _read_auth_receipt(auth_receipt)
        expected_keys = {
            "schema", "source_dir", "pool", "pool_receipt",
            "source_contract_sha256", "role_map_sha256",
            "copy_map_sha256", "source_files", "pool_identity",
            "pool_sha256", "pool_receipt_sha256", "runtime_ready",
        }
        if set(cached) != expected_keys or \
                cached["schema"] != AUTH_RECEIPT_SCHEMA or \
                cached["source_dir"] != str(source_dir.resolve()) or \
                cached["pool"] != str(pool.resolve()) or \
                cached["pool_receipt"] != str(receipt.resolve()) or \
                cached["source_contract_sha256"] != SOURCE_CONTRACT_SHA256 or \
                cached["role_map_sha256"] != ROLE_MAP_SHA256 or \
                cached["copy_map_sha256"] != source_summary["copy_map_sha256"] or \
                cached["pool_sha256"] != text_qa.POOL_SHA256 or \
                cached["pool_receipt_sha256"] != _file_sha256(receipt) or \
                cached["runtime_ready"] is not False:
            raise MultimodalQaError("authenticated package receipt drift")
        source_files = cached["source_files"]
        if not isinstance(source_files, dict):
            raise MultimodalQaError("authenticated source receipt is invalid")
        expected_identities = {}
        state = source_summary["_state"]
        for name, item in source_files.items():
            if (not isinstance(item, dict) or set(item) != {"identity", "sha256"} or
                    name not in state["tree"] or
                    item["sha256"] != state["tree"][name]["sha256"]):
                raise MultimodalQaError("authenticated source receipt drift")
            expected_identities[name] = item["identity"]
        report, handles, identities = source_module._open_authenticated_downloads(
            source_dir, source_summary, expected_identities,
        )
        pool_handle, pool_identity, _ = text_qa._open_authenticated_pool(
            pool, receipt, source_summary, cached["pool_identity"],
        )
        if pool_handle is None:
            raise MultimodalQaError("cached expert pool handle was not retained")
        text_binding = text_qa._build_binding(
            role_map, handles, identities, pool_handle, pool_identity,
            kv_compatibility.hexdigest, build_identity.hexdigest,
        )
        vision_binding, _ = vision_qa._build_binding(
            role_map, handles, identities,
        )
        return AuthenticatedBinding(
            handles=handles, pool_handle=pool_handle, identities=identities,
            report=report, role_map=role_map, pool_identity=pool_identity,
            binding=text_binding + vision_binding,
            compatibility_sha256=kv_compatibility.hexdigest,
            build_identity_sha256=build_identity.hexdigest,
        )
    except BaseException:
        if pool_handle is not None:
            pool_handle.close()
        for handle in handles.values():
            handle.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--auth-receipt", type=Path)
    parser.add_argument("--source-contract", type=Path,
                        default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--role-contract", type=Path,
                        default=DEFAULT_ROLE_CONTRACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--runner", type=Path,
                        default=ROOT / "gemma4-multimodal-qa")
    parser.add_argument("--ctx", type=int, default=200)
    parser.add_argument("--gen", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--question")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--serve-v2", action="store_true")
    parser.add_argument("--session-epoch", type=int, default=1)
    parser.add_argument("--stream-events", action="store_true")
    parser.add_argument("--compare-kv-warm", action="store_true")
    parser.add_argument("--package-proof", action="store_true")
    parser.add_argument("--kv-save", type=Path)
    parser.add_argument("--kv-compat-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--build-receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--shared-kv", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dpr-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dpr-mode", choices=("persist", "dynamic"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-current-kv-bytes", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-additive-bytes", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-serial-ns", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dpr-retention-bytes", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--serial-prefill-proof", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        validate_gemma_residency_config(ENGINE_CONFIG, require_host=True)
    except ValueError as exc:
        print(f"gemma4 multimodal qa: {exc}", file=sys.stderr)
        return 2
    if args.serve and args.serve_v2:
        parser.error("--serve and --serve-v2 are aliases; choose one")
    serving = args.serve or args.serve_v2
    if args.package_proof and (serving or args.verify_only):
        parser.error("--package-proof requires a one-shot native execution gate")
    if args.session_epoch < 1:
        parser.error("--session-epoch must be positive")
    if serving and (args.verify_only or args.compare_kv_warm or args.kv_save):
        parser.error("--serve cannot be combined with verify/compare/KV save")
    if args.compare_kv_warm and (args.verify_only or args.kv_save):
        parser.error("--compare-kv-warm cannot be combined with verify/KV save")
    if args.kv_save and args.verify_only:
        parser.error("--kv-save cannot be combined with --verify-only")
    if args.serial_prefill_proof and (
            serving or args.verify_only or args.compare_kv_warm):
        parser.error("serial prefill proof requires a one-shot execution gate")
    if args.dpr_mode is None:
        if (args.dpr_root is not None or args.dpr_current_kv_bytes or
                args.dpr_additive_bytes or args.dpr_serial_ns or
                args.dpr_retention_bytes):
            parser.error("native DPR startup fields require --dpr-mode")
    elif (not serving or args.dpr_root is None or
          not args.dpr_root.is_dir() or args.dpr_current_kv_bytes < 1 or
          args.dpr_serial_ns < 1 or args.dpr_retention_bytes < 1 or
          args.shared_kv is not None or
          (args.dpr_mode == "persist" and args.dpr_additive_bytes < 1) or
          (args.dpr_mode == "dynamic" and args.dpr_additive_bytes != 0)):
        parser.error("invalid native DPR startup configuration")
    receipt = args.receipt or Path(str(args.pool) + ".import.json")

    handles: dict[str, BinaryIO] = {}
    pool_handle: BinaryIO | None = None
    try:
        text_qa = _load_module(TEXT_QA_TOOL, "salt_gemma4_mm_text_qa")
        setattr(text_qa, "ENGINE_CONFIG", ENGINE_CONFIG)
        kv_compatibility = text_qa._kv_compatibility()
        if (
            args.kv_compat_sha256 is not None
            and args.kv_compat_sha256 != kv_compatibility.hexdigest
        ):
            raise MultimodalQaError("KV compatibility authority drift")
        if not args.runner.is_file():
            raise MultimodalQaError(f"Gemma 4 runner does not exist: {args.runner}")
        build_module = _load_module(
            BUILD_IDENTITY_TOOL, "salt_gemma4_multimodal_build",
        )
        build_identity = build_module.load_build_receipt(
            args.build_receipt or Path(str(args.runner) + ".build.json"),
            args.runner,
            expected_source_compatibility=kv_compatibility.hexdigest,
        )
        if args.auth_receipt is not None:
            authority = open_authenticated_binding(
                source_dir=args.source_dir, pool=args.pool, receipt=receipt,
                runner=args.runner,
                build_receipt=(args.build_receipt or
                               Path(str(args.runner) + ".build.json")),
                auth_receipt=args.auth_receipt,
                kv_compat_sha256=kv_compatibility.hexdigest,
                source_contract=args.source_contract,
                role_contract=args.role_contract, plan_path=args.plan,
            )
            handles = authority.handles
            pool_handle = authority.pool_handle
            report = authority.report
            role_map = authority.role_map
            pool_identity = authority.pool_identity
            binding = authority.binding
        else:
            vision_qa = _load_module(VISION_QA_TOOL, "salt_gemma4_mm_vision_qa")
            package = _load_module(PACKAGE_TOOL, "salt_gemma4_mm_package")
            source_module = package._load_mlx_source_module()
            plan = package.load_json(args.plan)
            package.validate_plan(plan)
            source_summary = source_module.validate_source(args.source_contract)
            if source_summary["contract_sha256"] != SOURCE_CONTRACT_SHA256 or \
                    source_summary["runtime_ready"] is not False:
                raise MultimodalQaError("authenticated source summary drift")
            report, handles, identities = source_module._authenticate_downloads(
                args.source_dir, source_summary,
            )
            if report["authenticated_bytes"] != 15_373_375_402 or \
                    report["runtime_ready"] is not False:
                raise MultimodalQaError("download authentication summary drift")
            role_map = package.build_mlx_source_role_map(
                plan,
                source_contract=args.source_contract,
                role_contract=args.role_contract,
                source_summary=source_summary,
            )
            if role_map["role_map_sha256"] != ROLE_MAP_SHA256 or \
                    role_map["runtime_ready"] is not True:
                raise MultimodalQaError("role-map identity or readiness drift")
            pool_handle, pool_identity, _ = text_qa._validate_pool(
                args.pool, receipt, source_summary,
            )
            if pool_handle is None:
                raise MultimodalQaError("expert pool handle was not retained")
            text_binding = text_qa._build_binding(
                role_map, handles, identities, pool_handle, pool_identity,
                kv_compatibility.hexdigest, build_identity.hexdigest,
            )
            vision_binding, _ = vision_qa._build_binding(
                role_map, handles, identities,
            )
            binding = text_binding + vision_binding

        if args.verify_only:
            print("GEMMA4_MULTIMODAL_BINDING_OK")
            print(
                "authenticated_source_gb="
                f"{report['authenticated_bytes'] / 1_000_000_000:.9f}"
            )
            print(f"expert_pool_gb={pool_identity[2] / 1_000_000_000:.9f}")
            print("text_role_count=597")
            print("vision_role_count=356")
            print(f"role_map_sha256={role_map['role_map_sha256']}")
            print(f"kv_compatibility_sha256={kv_compatibility.hexdigest}")
            print(f"build_identity_sha256={build_identity.hexdigest}")
            print("runtime_ready=true")
            return 0
        if serving:
            if (args.ctx, args.gen) != (512, 128):
                parser.error("--serve requires CTX512/O128")
            if not args.stream_events:
                parser.error("--serve requires --stream-events")
            if not args.runner.is_file():
                raise MultimodalQaError(
                    f"Gemma 4 persistent runner does not exist: {args.runner}",
                )
            inherited = tuple(handle.fileno() for handle in handles.values()) + (
                pool_handle.fileno(),
            )
            command = [
                str(args.runner), "--serve-v2", "--ctx", "512", "--gen", "128",
                "--workers", str(args.workers), "--session-epoch",
                str(args.session_epoch), "--stream-events",
            ]
            if args.shared_kv is not None:
                if not args.shared_kv.is_file():
                    raise MultimodalQaError("shared KV file does not exist")
                command.extend(("--shared-kv", str(args.shared_kv)))
            if args.dpr_mode is not None:
                assert args.dpr_root is not None
                command.extend((
                    "--dpr-root", str(args.dpr_root),
                    "--dpr-mode", args.dpr_mode,
                    "--dpr-current-kv-bytes", str(args.dpr_current_kv_bytes),
                    "--dpr-serial-ns", str(args.dpr_serial_ns),
                    "--dpr-retention-bytes", str(args.dpr_retention_bytes),
                ))
                if args.dpr_mode == "persist":
                    command.extend((
                        "--dpr-additive-bytes", str(args.dpr_additive_bytes),
                    ))
            return text_qa._run_native_persistent(command, binding, inherited)
        if (args.ctx, args.gen) not in ALLOWED_QA_GATES:
            raise MultimodalQaError(
                f"unsupported QA gate CTX{args.ctx}/O{args.gen}",
            )
        if args.workers < 1 or args.workers > 8:
            raise MultimodalQaError("--workers must be in [1, 8]")
        if not args.runner.is_file():
            raise MultimodalQaError(
                f"Gemma 4 multimodal runner does not exist: {args.runner}",
            )
        if args.image is None or not args.image.is_file():
            raise MultimodalQaError(f"image does not exist: {args.image}")

        prompt = (text_qa._load_prompt(args.prompt_file) if args.prompt_file else
                  _render_prompt(args.question or
                                 "What object is in this image? Answer briefly."))
        prompt_bytes = prompt.encode("utf-8")
        print(
            "GEMMA4_MM_PROMPT "
            f"bytes={len(prompt_bytes)} "
            f"image_placeholders={IMAGE_SOFT_TOKENS} "
            f"sha256={hashlib.sha256(prompt_bytes).hexdigest()} "
            f"question={json.dumps(args.question.strip() if args.question else '<rendered-prompt>', ensure_ascii=True)}",
            file=sys.stderr,
        )
        inherited = tuple(handle.fileno() for handle in handles.values()) + (
            pool_handle.fileno(),
        )
        command = [
            str(args.runner),
            "--ctx", str(args.ctx),
            "--gen", str(args.gen),
            "--workers", str(args.workers),
            "--max-soft-tokens", "70",
            "--prompt", prompt,
            "--image", str(args.image),
        ]
        if args.compare_kv_warm:
            command.append("--compare-kv-warm")
        if args.package_proof:
            command.append("--package-proof")

        if args.kv_save:
            command.extend(("--kv-save", str(args.kv_save)))
        return text_qa._run_native(
            command, binding, inherited,
            serial_prefill_proof=args.serial_prefill_proof,
        )
    except Exception as exc:
        known = {"QaError", "VisionQaError", "ContractError"}
        if isinstance(exc, MultimodalQaError) or exc.__class__.__name__ in known:
            print(f"gemma4 multimodal qa: {exc}", file=sys.stderr)
            return 2
        raise
    finally:
        if pool_handle is not None:
            pool_handle.close()
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
