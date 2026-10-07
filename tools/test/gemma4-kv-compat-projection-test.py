#!/usr/bin/env python3
"""Adversarial contract for Gemma KV physical-build projection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "server"
MODEL_SERVER = SERVER / "model"
sys.path.insert(0, str(MODEL_SERVER))

import gemma4_compat as base  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "salt_gemma4_compat_projection", MODEL_SERVER / "gemma4_compat_projection.py"
)
assert SPEC is not None and SPEC.loader is not None
projection_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection_module)

PRODUCTION_MANIFEST = Path("models/gemma4-26b-a4b/kv-cache-compat.json")
PRODUCTION_PROJECTION = Path("models/gemma4-26b-a4b/kv-cache-compat-projection.json")
EXPECTED_PRODUCTION = "36da7910335af449d03531ce73b8d539ba6550633118c26e7b54a015cfc89a98"
BEGIN = b"# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1\n"
END = b"# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1\n"


def expect_error(call, message: str) -> None:
    try:
        call()
    except base.CompatibilityError as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError(f"expected CompatibilityError containing {message!r}")


def write_manifest(path: Path) -> None:
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
        "chat_template_abi": "template-v1",
        "engine_contract_files": ["Makefile", "engine.c", "server.c"],
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


def fixture(root: Path) -> tuple[Path, Path, bytes, bytes]:
    manifest = root / "compat.json"
    projection = root / "projection.json"
    snapshot_path = root / "compat/v6/Makefile"
    snapshot_path.parent.mkdir(parents=True)
    snapshot = b"CC ?= cc\nall:\n\t$(CC) engine.c\n"
    block = BEGIN + b"PHYSICAL_OBJECTS += gpu.o\n" + END
    live = snapshot + block
    server_snapshot_path = root / "compat/v6/server.c"
    server_snapshot = b"int server(void) { return 1; }\n"
    server_live = b"int server(void) { return 2; }\n"
    snapshot_path.write_bytes(snapshot)
    server_snapshot_path.write_bytes(server_snapshot)
    (root / "Makefile").write_bytes(live)
    (root / "engine.c").write_text("int engine(void) { return 1; }\n")
    (root / "server.c").write_bytes(server_live)
    write_manifest(manifest)
    projection.write_text(json.dumps({
        "schema": "salt.gemma4.kv-compat-projection.v1",
        "compatibility_manifest": "compat.json",
        "overrides": [{
            "logical_path": "Makefile",
            "snapshot_path": "compat/v6/Makefile",
            "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
            "live_projection": "strip-physical-build-blocks-v1",
            "physical_blocks_sha256": hashlib.sha256(block).hexdigest(),
        }, {
            "logical_path": "server.c",
            "snapshot_path": "compat/v6/server.c",
            "snapshot_sha256": hashlib.sha256(server_snapshot).hexdigest(),
            "live_projection": "pin-live-source-v1",
            "live_sha256": hashlib.sha256(server_live).hexdigest(),
        }],
    }, indent=2, sort_keys=True) + "\n")
    return manifest, projection, snapshot, live


def self_contained_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest, projection, snapshot_raw, makefile_raw = fixture(root)
        projected, dependencies = projection_module.load_projected_compatibility(
            manifest, root, projection,
        )
        direct = base.load_compatibility(manifest, root)
        assert projected.hexdigest != direct.hexdigest
        assert set(dependencies) == {
            "Makefile", "compat.json", "compat/v6/Makefile", "engine.c",
            "compat/v6/server.c", "projection.json", "server.c",
        }
        assert dict(projected.file_sha256)["server.c"] == \
            hashlib.sha256(b"int server(void) { return 1; }\n").hexdigest()

        semantic = root / "engine.c"
        semantic_raw = semantic.read_bytes()
        semantic.write_bytes(semantic_raw + b"/* semantic tamper */\n")
        changed, _ = projection_module.load_projected_compatibility(
            manifest, root, projection,
        )
        assert changed.hexdigest != projected.hexdigest
        semantic.write_bytes(semantic_raw)

        makefile = root / "Makefile"
        makefile.write_bytes(makefile_raw + b"CFLAGS += -ffast-math\n")
        expect_error(
            lambda: projection_module.load_projected_compatibility(
                manifest, root, projection,
            ),
            "changed outside physical build blocks",
        )
        makefile.write_bytes(makefile_raw)

        makefile.write_bytes(makefile_raw.replace(b"gpu.o", b"other.o", 1))
        expect_error(
            lambda: projection_module.load_projected_compatibility(
                manifest, root, projection,
            ),
            "physical build block SHA-256 mismatch",
        )
        makefile.write_bytes(makefile_raw)

        server = root / "server.c"
        server_raw = server.read_bytes()
        server.write_bytes(server_raw + b"/* physical tamper */\n")
        expect_error(
            lambda: projection_module.load_projected_compatibility(
                manifest, root, projection,
            ),
            "pinned live source SHA-256 mismatch",
        )
        server.write_bytes(server_raw)

        snapshot = root / "compat/v6/Makefile"
        snapshot.write_bytes(snapshot_raw + b"\n")
        expect_error(
            lambda: projection_module.load_projected_compatibility(
                manifest, root, projection,
            ),
            "compatibility snapshot SHA-256 mismatch",
        )
        snapshot.write_bytes(snapshot_raw)

        projection_raw = projection.read_bytes()
        malformed = json.loads(projection_raw)
        malformed["unexpected"] = True
        projection.write_text(json.dumps(malformed))
        expect_error(
            lambda: projection_module.load_projected_compatibility(
                manifest, root, projection,
            ),
            "compatibility projection fields are incomplete or unsupported",
        )
        projection.write_bytes(projection_raw)

        wrong = json.loads(projection_raw)
        template = wrong["overrides"][0]
        for count in (21, 128):
            wrong["overrides"] = [dict(template, logical_path=f"source-{i:03}.c")
                                  for i in range(count)]
            projection.write_text(json.dumps(wrong))
            assert len(projection_module._load_projection(projection, root)[1]["overrides"]) == count
        wrong["overrides"].append(dict(template, logical_path="source-128.c"))
        projection.write_text(json.dumps(wrong))
        expect_error(lambda: projection_module._load_projection(projection, root),
                     "compatibility projection overrides are malformed")

        wrong = json.loads(projection_raw)
        wrong["compatibility_manifest"] = "other.json"
        projection.write_text(json.dumps(wrong))
        expect_error(
            lambda: projection_module.load_projected_compatibility(
                manifest, root, projection,
            ),
            "targets the wrong manifest",
        )


def production_contract() -> None:
    source = (MODEL_SERVER / "gemma4_compat_projection.py").read_text()
    assert EXPECTED_PRODUCTION not in source
    authority, dependencies = projection_module.load_projected_compatibility(
        PRODUCTION_MANIFEST, ROOT, PRODUCTION_PROJECTION,
    )
    assert authority.hexdigest == EXPECTED_PRODUCTION
    assert PRODUCTION_MANIFEST.as_posix() in dependencies
    assert PRODUCTION_PROJECTION.as_posix() in dependencies
    assert "models/gemma4-26b-a4b/compat/v6/Makefile" in dependencies
    assert "models/gemma4-26b-a4b/compat/v6/src/tokenizer.c" in dependencies
    assert "models/gemma4-26b-a4b/compat/v6/models/gemma4-26b-a4b/gemma4_text.h" in dependencies
    assert dict(authority.file_sha256)["models/gemma4-26b-a4b/gemma4_text.h"] == \
        "27f20d4b5c27250bd312b356289f2c2d40bd72c67518d219490ad771f6407dfd"
    assert dict(authority.file_sha256)["src/tokenizer.c"] == \
        "112a85f42cb752e16424abeb3bd6fd82b31b3fda20571e67af99e5af72f3c310"
    assert dict(authority.file_sha256)["Makefile"] == \
        "f34e7285395963b84600379f3a3a1065c3f1d5d38dc8d04c2f7776b07fb8e212"
    assert dict(authority.file_sha256)[
        "models/gemma4-26b-a4b/gemma4_text.c"
    ] == "2276829cd1e01df341577ea3e1b0227e2c54b171174ea3c74098d2543c3b6952"
    assert dict(authority.file_sha256)["server/gemma4_compat.py"] == \
        "a903d9a6b88d0f592da176fcc7bd0bfdf4c57edb3d4ccd1088de7532096436c2"
    assert "src/sampling.c" in dependencies
    assert "include/salt/sampling.h" in dependencies
    manifest = base.json.loads((ROOT / PRODUCTION_MANIFEST).read_text())
    manifest["engine_contract_files"] = [f"source-{i:03}.c" for i in range(128)]
    assert len(base._validate_manifest(manifest)) == 128
    manifest["engine_contract_files"].append("source-128.c")
    expect_error(lambda: base._validate_manifest(manifest), "engine contract files")
    runtime = base.load_compatibility(PRODUCTION_MANIFEST, ROOT)
    assert runtime.hexdigest == EXPECTED_PRODUCTION
    direct = base.load_unprojected_compatibility(PRODUCTION_MANIFEST, ROOT)
    assert direct.hexdigest != EXPECTED_PRODUCTION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    self_contained_contract()
    if args.production:
        production_contract()
    print("GEMMA4_KV_COMPAT_PROJECTION_TEST_PASS")


if __name__ == "__main__":
    main()
