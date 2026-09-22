#!/usr/bin/env python3
"""Operator-only CLI for immutable exact mentor anchors."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "server" / "gemma4_mentor.py"
MODULE_NAME = "salt_gemma4_mentor_cli"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load mentor anchor store")
MENTOR = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MENTOR
SPEC.loader.exec_module(MENTOR)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def hex_bytes(value: str) -> bytes:
    if HEX64.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected 64 lowercase hexadecimal characters")
    return bytes.fromhex(value)


def positive(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1 or str(parsed) != value:
        raise argparse.ArgumentTypeError("expected a canonical positive integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="seal and inspect immutable exact mentor anchors",
    )
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--compatibility-sha256", type=hex_bytes, required=True)
    value.add_argument("--model-id", required=True)
    value.add_argument("--quant-track", required=True)
    commands = value.add_subparsers(dest="command", required=True)

    seal = commands.add_parser("seal")
    seal.add_argument("--state", type=Path, required=True)
    seal.add_argument("--anchor-id", required=True)
    seal.add_argument("--parent", action="append", default=[])
    seal.add_argument("--native-state-sha256", required=True)
    seal.add_argument("--facts-sha256", required=True)
    seal.add_argument("--evidence-policy-version", required=True)
    seal.add_argument("--evidence-bundle-sha256", required=True)
    seal.add_argument("--rendered-prefix-sha256", required=True)
    seal.add_argument("--prompt-ids-sha256", required=True)
    seal.add_argument("--session-epoch", type=positive, required=True)
    seal.add_argument("--turn-id", type=positive, required=True)
    seal.add_argument("--request-sha256", required=True)
    seal.add_argument("--build-identity-sha256", required=True)

    commands.add_parser("list")
    for name in ("show", "verify", "resolve"):
        command = commands.add_parser(name)
        command.add_argument("selector")
    return value


def public_metadata(metadata) -> dict:
    return {
        "anchor_id": metadata.anchor_id,
        "revision": metadata.revision,
        "selector": metadata.selector,
        "proposal_type": "exact-anchor",
        "compatibility_sha256": metadata.compatibility_sha256,
        "state_file_sha256": metadata.state_file_sha256,
        "state_payload_sha256": metadata.state_payload_sha256,
        "native_state_sha256": metadata.native_state_sha256,
        "facts_sha256": metadata.facts_sha256,
        "context_tokens": metadata.context_tokens,
        "position": metadata.position,
        "parents": list(metadata.parents),
        "runtime_ready": False,
    }


def emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        store = MENTOR.MentorAnchorStore(
            args.root,
            compatibility_sha256=args.compatibility_sha256,
            model_id=args.model_id,
            quant_track=args.quant_track,
        )
        if args.command == "seal":
            metadata = store.seal(args.state, {
                "anchor_id": args.anchor_id,
                "parents": args.parent,
                "native_state_sha256": args.native_state_sha256,
                "facts_sha256": args.facts_sha256,
                "evidence_policy_version": args.evidence_policy_version,
                "evidence_bundle_sha256": args.evidence_bundle_sha256,
                "rendered_prefix_sha256": args.rendered_prefix_sha256,
                "prompt_ids_sha256": args.prompt_ids_sha256,
                "creation_event": {
                    "session_epoch": args.session_epoch,
                    "turn_id": args.turn_id,
                    "request_sha256": args.request_sha256,
                    "build_identity_sha256": args.build_identity_sha256,
                },
            })
            emit(public_metadata(metadata))
        elif args.command == "list":
            anchors = [public_metadata(item) for item in store.list()]
            emit({"anchors": anchors, "count": len(anchors)})
        else:
            metadata = (
                store.verify(args.selector)
                if args.command == "verify"
                else store.resolve(args.selector)
            )
            result = public_metadata(metadata)
            if args.command == "resolve":
                result["state_path"] = str(metadata.state_path)
            emit(result)
    except MENTOR.MentorAnchorError as exc:
        print(f"gemma4 mentor state: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
