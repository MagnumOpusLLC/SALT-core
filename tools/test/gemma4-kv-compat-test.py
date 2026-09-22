#!/usr/bin/env python3
"""No-model contract tests for Gemma 4 KV compatibility authority."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMPAT_MODULE = ROOT / "server" / "model" / "gemma4_compat.py"
PRODUCTION_MANIFEST = (
    ROOT / "models" / "gemma4-26b-a4b" / "kv-cache-compat.json"
)

spec = importlib.util.spec_from_file_location("salt_gemma4_compat", COMPAT_MODULE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def write_manifest(path: Path, files: list[str], *, template_abi: str = "template-v1") -> None:
    value = {
        "schema": "salt.gemma4.kv-compat.v5",
        "model": "gemma-4-26b-a4b-it",
        "cache_abi": {
            "magic": "G4KVC006",
            "version": 6,
            "header_bytes": 256,
            "payload": (
                "hybrid-shared-projection-v-base-finite-ieee754-binary32-"
                "little-endian"
            ),
        },
        "source_authority": {
            "source_contract_sha256": "1" * 64,
            "role_map_sha256": "2" * 64,
            "copy_map_sha256": "3" * 64,
        },
        "tokenizer_control": {
            "bos": "<bos>",
            "turn_open": "<|turn>",
            "turn_close": "<turn|>",
            "thought_open": "<|channel>thought",
            "channel_close": "<channel|>",
            "stop_token_ids": [1, 50, 106],
        },
        "chat_template_abi": template_abi,
        "engine_contract_files": files,
        "legacy_read_abis": [
            {
                "magic": "G4KVC005",
                "version": 5,
                "compatibility_sha256": "5" * 64,
            },
            {
                "magic": "G4KVC004",
                "version": 4,
                "compatibility_sha256": "4" * 64,
            },
        ],
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def expect_compat_error(function, text: str) -> None:
    try:
        function()
    except module.CompatibilityError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected CompatibilityError containing {text!r}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        engine = root / "engine.c"
        renderer = root / "renderer.py"
        engine.write_text("engine-v1\n")
        renderer.write_text("template-v1\n")
        manifest = root / "compat.json"
        write_manifest(manifest, ["engine.c", "renderer.py"])

        first = module.load_compatibility(manifest, root)
        assert len(first.digest) == 32
        assert len(first.hexdigest) == 64
        assert first.namespace == "g4kvc006-" + first.hexdigest

        engine.write_text("engine-v2\n")
        engine_drift = module.load_compatibility(manifest, root)
        assert engine_drift.digest != first.digest

        engine.write_text("engine-v1\n")
        write_manifest(manifest, ["engine.c", "renderer.py"], template_abi="template-v2")
        template_drift = module.load_compatibility(manifest, root)
        assert template_drift.digest != first.digest

        write_manifest(manifest, ["../escape"])
        expect_compat_error(
            lambda: module.load_compatibility(manifest, root),
            "relative repository path",
        )

        outside = Path(tmp) / "outside"
        outside.write_text("outside\n")
        link = root / "linked"
        link.symlink_to(outside)
        write_manifest(manifest, ["linked"])
        expect_compat_error(
            lambda: module.load_compatibility(manifest, root),
            "regular non-symlink",
        )

    production = module.load_compatibility(PRODUCTION_MANIFEST, ROOT)
    required = {
        "models/gemma4-26b-a4b/gemma4_text.h",
        "models/gemma4-26b-a4b/gemma4_text.c",
        "models/gemma4-26b-a4b/gemma4_kv_file.h",
        "models/gemma4-26b-a4b/server.c",
        "src/tokenizer.c",
        "tools/gemma4-qa.c",
        "tools/gemma4-qa.py",
        "tools/gemma4-multimodal-qa.c",
        "tools/gemma4-multimodal-qa.py",
        "server/gemma4_backend.py",
        "server/gemma4_mentor.py",
        "server/gemma4_state_artifact.py",
        "tools/gemma4-mentor-state.py",
    }
    assert required <= set(production.manifest["engine_contract_files"])
    assert production.manifest["cache_abi"]["magic"] == "G4KVC006"
    assert production.manifest["cache_abi"]["version"] == 6
    assert production.manifest["legacy_read_abis"] == [
        {
            "magic": "G4KVC005",
            "version": 5,
            "compatibility_sha256": (
                "aa4eb350cfcecbe829391c5d02e139de0f4c1331d0e3f45a0f2f2daf70637a24"
            ),
        },
        {
            "magic": "G4KVC004",
            "version": 4,
            "compatibility_sha256": (
                "2744872d508028385e7f631ba6a3c8a7596c2a9d257f5abd56d021ba04f942e2"
            ),
        },
    ]

    print("gemma4 KV compatibility contract: PASS")


if __name__ == "__main__":
    main()
