#!/usr/bin/env python3
"""Authenticate and byte-copy the pinned Gemma 4 MLX affine-Q4 source.

This tool never enables the Gemma runtime.  It validates a checked-in,
metadata-only source authority, emits a deterministic source-to-pool copy map,
authenticates complete downloaded files, and can materialize the routed-expert
pool without dequantizing or requantizing compatible affine-Q4 slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import struct
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "models" / "gemma4-26b-a4b" / "mlx4-source"
DEFAULT_CONTRACT = DEFAULT_SOURCE_ROOT / "source-contract.json"

SOURCE_SCHEMA = "salt.gemma4-mlx4-source-contract.v1"
AUTHORITY_SCHEMA = "salt.gemma4-mlx4-shard-header.v1"
IMPORT_MAP_SCHEMA = "salt.gemma4-mlx4-import-map.v1"
REPOSITORY = "mlx-community/gemma-4-26b-a4b-it-4bit"
REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
UPSTREAM_REVISION = "20da991ab4afab98e8f910c4a2e8f4fbefc404ad"
SEMANTIC_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
TRANSFORMERS_REVISION = "242f5df996ded281cc328c5b9bcc1bee0ec48118"

EXPECTED_SOURCE = {
    "repository": REPOSITORY,
    "revision": REVISION,
    "declared_upstream_repository": "google/gemma-4-26B-A4B-it",
    "declared_upstream_revision": UPSTREAM_REVISION,
    "semantic_reference_repository": "google/gemma-4-26B-A4B-it",
    "semantic_reference_revision": SEMANTIC_REVISION,
    "transformers_revision": TRANSFORMERS_REVISION,
    "producer_provenance": "converter-commit-not-attested",
    "license": "apache-2.0",
    "license_link": "https://ai.google.dev/gemma/docs/gemma_4_license",
    "access": "public-ungated",
}

# Exact sibling closure from the pinned revision API.  LFS sha256 values are
# repository-declared until validate-download hashes complete local files.
EXPECTED_TREE = {
    ".gitattributes": ("git-blob", 1570,
        "52373fe24473b1aa44333d318f578ae6bf04b49b",
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
        "metadata/gitattributes.txt"),
    "README.md": ("git-blob", 753,
        "0990166bfcc29f00da7e71f5616536686ff733b0",
        "f4750b035683ba5d6707f6d1302ac6894aef3096fff21845b9b593bc1465c888",
        "metadata/upstream-README.md"),
    "chat_template.jinja": ("git-blob", 17466,
        "e61bbfe9a31db79ea1e4e557b046f895ad7899d0",
        "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0",
        "metadata/chat_template.jinja"),
    "config.json": ("git-blob", 12180,
        "9e6233a7e7174ab214c49dca84f2828a8fb504ef",
        "419e13a27ec359654c1ce7dd06d1a87149fd07d0c8af9f770494185e62e1b2ba",
        "metadata/config.json"),
    "generation_config.json": ("git-blob", 208,
        "e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6",
        "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
        "metadata/generation_config.json"),
    "model-00001-of-00003.safetensors": ("lfs", 5320218487,
        "3163ffd186cf106b6bdc9fcc4eeb9960515bd622",
        "683f420ee09550b8027bf0335c4202e196b37eb520b60a4bcbe8690b9e388c07",
        None),
    "model-00002-of-00003.safetensors": ("lfs", 5363328422,
        "ee972f8756bb021b6b424a52331f28f7a26d53f1",
        "feab2873c2976fb7ed666f8462549564a29409364388e0c8f6b015729d165e15",
        None),
    "model-00003-of-00003.safetensors": ("lfs", 4657658867,
        "8f6e0a7ab8fade3fcdd1b917cb0f1450c4870a53",
        "bc607486deb1de5bc7e459932fda5c776a4f816b368feaa661bab8cfbb47e567",
        None),
    "model.safetensors.index.json": ("git-blob", 176940,
        "f6f9e61c33077cc7e2ef701eff7056927a4a65e6",
        "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        "metadata/model.safetensors.index.json"),
    "processor_config.json": ("git-blob", 1316,
        "a086fb7e04b477c291a120b0a004abb78b11c6d2",
        "de3e580aebdc98272d4c4547daffe6525fcbae18a83a0e0bcf0d7444d4ee6f37",
        "metadata/processor_config.json"),
    "tokenizer.json": ("lfs", 32169626,
        "1ff9f3e3439a939b971f9919e821bf87e835a503",
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        None),
    "tokenizer_config.json": ("git-blob", 2740,
        "cf6235aee46a24bf71f251c0a4e7a0379948f7d2",
        "080d9e1aff284e2f6043889cd05367966f7c7b80e025fbc0b06745e218158656",
        "metadata/tokenizer_config.json"),
}

EXPECTED_HEADERS = {
    "model-00001-of-00003.safetensors": (
        "headers/model-00001-of-00003.safetensors.json",
        "8900ed84cea77cd7e4b162426f85e7e6f03af64fc7c485e71b1077fd6702a1d7",
        64091,
        "228d184bd285a79f9b5953f81d1f4e13d302264a20aa1c1db7840286e397676c",
        5320154388,
        488,
    ),
    "model-00002-of-00003.safetensors": (
        "headers/model-00002-of-00003.safetensors.json",
        "fc68be8b39cd6b48fd5f8abbfdde436961b2233727b99616a2ef53ef33d4d643",
        70022,
        "e24e953b8363c11331d0fd1ee0249bf7e2bd67f80cf296c5ac11fad17a86faa6",
        5363258392,
        530,
    ),
    "model-00003-of-00003.safetensors": (
        "headers/model-00003-of-00003.safetensors.json",
        "e5928cce8e15b936d41e8852cf8a7c2d2785275743f30130b8559799bd35afa6",
        90235,
        "f9010ed1a67279c1d54f454e8ad5e56394e30777b027f5bcaa8de61c4d2537fd",
        4657568624,
        679,
    ),
}

EXPECTED_INVENTORY = {
    "tensor_count": 1697,
    "tensor_payload_bytes": 15340981404,
    "stored_dtype_bytes": {"BF16": 2717731996, "U32": 12623249408},
    "affine_bundle_count": 327,
    "q4_bundle_count": 297,
    "q8_bundle_count": 30,
    "group_size": 64,
    "packed_dtype": "U32",
    "packed_word_byte_order": "little-endian",
    "q4_nibble_order": "low-to-high",
    "q8_byte_lane_order": "low-to-high",
    "code_domain": "unsigned",
    "quantization_axis": "last-dimension-contiguous-groups",
    "matrix_orientation": "output-by-input-row-major-no-transpose",
    "scale_dtype": "BF16",
    "bias_dtype": "BF16",
    "scale_bias_byte_order": "little-endian",
    "dequantization": "q * scale + bias",
}

EXPECTED_MODULE_PRECISION = {
    "default_affine_matrices": "Q4",
    "language_router_projection": "Q8",
    "language_router_import_status": "source-pinned-runtime-unqualified",
    "vision_tower": "BF16",
    "vision_to_text_projection": "Q4",
    "vision_to_text_import_status": "source-pinned-import-not-yet-mapped",
    "runtime_accumulation": "FP32",
}

EXPECTED_IMPORT_SCALARS = {
    "verdict": "byte-direct-copy-compatible-relocation-required",
    "runtime_manifest_status": "blocked-until-gemma-geglu-runtime-exists",
    "layers": 30,
    "experts_per_layer": 128,
    "source_tensor_prefix":
        "language_model.model.layers.{layer}.experts.switch_glu.{component}.{part}",
    "projection_order": ["gate_proj", "up_proj", "down_proj"],
    "part_order": ["weight", "scales", "biases"],
    "expert_slot_bytes": 3345408,
    "pool_header_bytes": 24,
    "pool_payload_bytes": 12846366720,
    "pool_file_bytes": 12846366744,
    "minimum_free_after_materialize_bytes": 10737418240,
    "copy_range_count": 34560,
    "copy_map_sha256":
        "51fa40aca8cb910896328026fcfb3f07c36a7519d0bd76e6fce574ba3c178a3e",
}

EXPECTED_COMPONENTS = [
    {
        "name": "gate_proj", "shape": [704, 2816],
        "value_relative_offset": 0, "value_bytes": 991232,
        "scale_relative_offset": 991232, "scale_bytes": 61952,
        "bias_relative_offset": 1053184, "bias_bytes": 61952,
    },
    {
        "name": "up_proj", "shape": [704, 2816],
        "value_relative_offset": 1115136, "value_bytes": 991232,
        "scale_relative_offset": 2106368, "scale_bytes": 61952,
        "bias_relative_offset": 2168320, "bias_bytes": 61952,
    },
    {
        "name": "down_proj", "shape": [2816, 704],
        "value_relative_offset": 2230272, "value_bytes": 991232,
        "scale_relative_offset": 3221504, "scale_bytes": 61952,
        "bias_relative_offset": 3283456, "bias_bytes": 61952,
    },
]


class ContractError(RuntimeError):
    """A pinned source, layout, or extent violated the checked contract."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ContractError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _parse_json_bytes(data: bytes, path: Path) -> dict:
    try:
        text = data.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: root must be an object")
    return value


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read_stable_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open regular file {path}: {exc}") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"not a regular file: {path}")
            identity = _stat_identity(before)
            data = handle.read()
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ContractError(f"{path}: identity changed while reading")
            if len(data) != before.st_size:
                raise ContractError(f"{path}: short read")
            return data
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc


def load_json(path: Path) -> dict:
    return _parse_json_bytes(_read_stable_bytes(path), path)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"not a regular file: {path}")
            identity = _stat_identity(before)
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ContractError(f"{path}: identity changed while hashing")
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_exact_keys(value: dict, keys: set[str], where: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{where}: key drift: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _local_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ContractError(f"invalid authority path {relative!r}")
    relative_path = Path(relative)
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ContractError(f"invalid authority path {relative!r}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_parent = (resolved_root / relative_path.parent).resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"missing authority parent for {relative}: {exc}") from exc
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise ContractError(f"authority escapes source root: {relative}")
    return resolved_parent / relative_path.name


def _read_authenticated_bytes(path: Path, expected_bytes: int | None,
                              expected_sha256: str) -> bytes:
    data = _read_stable_bytes(path)
    if expected_bytes is not None and len(data) != expected_bytes:
        raise ContractError(f"{path}: extent {len(data)} != pinned {expected_bytes}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ContractError(f"{path}: sha256 {digest} != pinned {expected_sha256}")
    return data


def _check_local_identity(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    _read_authenticated_bytes(path, expected_bytes, expected_sha256)


def _validate_contract(contract: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    _require_exact_keys(
        contract,
        {"schema", "status", "source", "repository_tree", "headers",
         "tensor_inventory", "module_precision", "expert_import"},
        "source contract",
    )
    if contract["schema"] != SOURCE_SCHEMA:
        raise ContractError(f"source schema must be {SOURCE_SCHEMA}")
    if contract["status"] != "metadata-qualified-payload-unverified-runtime-blocked":
        raise ContractError("source contract status is not fail-closed")
    if contract["source"] != EXPECTED_SOURCE:
        raise ContractError("pinned source/provenance identity drift")
    if contract["tensor_inventory"] != EXPECTED_INVENTORY:
        raise ContractError("pinned tensor inventory drift")
    if contract["module_precision"] != EXPECTED_MODULE_PRECISION:
        raise ContractError("pinned module precision policy drift")

    tree_raw = contract["repository_tree"]
    if not isinstance(tree_raw, list):
        raise ContractError("repository_tree must be an array")
    tree = {}
    for ordinal, entry in enumerate(tree_raw):
        if not isinstance(entry, dict):
            raise ContractError(f"repository_tree[{ordinal}] must be an object")
        base_keys = {"name", "storage", "bytes", "blob_id", "sha256"}
        expected_keys = base_keys | ({"local_path"} if entry.get("storage") == "git-blob" else set())
        _require_exact_keys(entry, expected_keys, f"repository_tree[{ordinal}]")
        name = entry.get("name")
        if name in tree:
            raise ContractError(f"duplicate repository sibling {name!r}")
        tree[name] = entry
    if set(tree) != set(EXPECTED_TREE):
        raise ContractError("pinned repository sibling set drift")
    for name, expected in EXPECTED_TREE.items():
        storage, size, blob_id, digest, local_path = expected
        want = {
            "name": name, "storage": storage, "bytes": size,
            "blob_id": blob_id, "sha256": digest,
        }
        if local_path is not None:
            want["local_path"] = local_path
        if tree[name] != want:
            raise ContractError(f"repository sibling identity drift: {name}")

    headers_raw = contract["headers"]
    if not isinstance(headers_raw, list):
        raise ContractError("headers must be an array")
    headers = {}
    header_keys = {
        "shard", "authority_path", "authority_sha256", "header_bytes",
        "header_sha256", "payload_bytes", "tensor_count",
    }
    for ordinal, entry in enumerate(headers_raw):
        if not isinstance(entry, dict):
            raise ContractError(f"headers[{ordinal}] must be an object")
        _require_exact_keys(entry, header_keys, f"headers[{ordinal}]")
        name = entry.get("shard")
        if name in headers:
            raise ContractError(f"duplicate shard header authority {name!r}")
        headers[name] = entry
    if set(headers) != set(EXPECTED_HEADERS):
        raise ContractError("pinned shard-header set drift")
    for name, expected in EXPECTED_HEADERS.items():
        authority_path, authority_digest, header_bytes, header_digest, payload_bytes, count = expected
        want = {
            "shard": name,
            "authority_path": authority_path,
            "authority_sha256": authority_digest,
            "header_bytes": header_bytes,
            "header_sha256": header_digest,
            "payload_bytes": payload_bytes,
            "tensor_count": count,
        }
        if headers[name] != want:
            raise ContractError(f"shard-header identity drift: {name}")

    expert = contract["expert_import"]
    if not isinstance(expert, dict):
        raise ContractError("expert_import must be an object")
    if set(expert) != set(EXPECTED_IMPORT_SCALARS) | {"components"}:
        raise ContractError("expert import key drift")
    for key, want in EXPECTED_IMPORT_SCALARS.items():
        if expert.get(key) != want:
            raise ContractError(f"expert import {key} drift")
    if expert["components"] != EXPECTED_COMPONENTS:
        raise ContractError("expert import component layout drift")
    return tree, headers


def _validate_config(config: dict) -> dict:
    exact = {
        "model_type": "gemma4", "image_token_id": 258880,
        "boi_token_id": 255999, "eoi_token_id": 258882,
        "eos_token_id": [1, 106, 50], "vision_soft_tokens_per_image": 280,
        "audio_config": None,
    }
    for key, want in exact.items():
        if config.get(key) != want:
            raise ContractError(f"config {key}: expected {want!r}, got {config.get(key)!r}")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ContractError("config text_config/vision_config must be objects")
    text_exact = {
        "model_type": "gemma4_text", "dtype": "bfloat16",
        "vocab_size": 262144, "num_hidden_layers": 30,
        "hidden_size": 2816, "intermediate_size": 2112,
        "moe_intermediate_size": 704, "num_experts": 128,
        "top_k_experts": 8, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 256,
        "global_head_dim": 512, "num_global_key_value_heads": 2,
        "sliding_window": 1024, "hidden_activation": "gelu_pytorch_tanh",
        "attention_k_eq_v": True, "tie_word_embeddings": True,
        "use_bidirectional_attention": "vision",
    }
    for key, want in text_exact.items():
        if text.get(key) != want:
            raise ContractError(f"text_config {key}: expected {want!r}, got {text.get(key)!r}")
    globals_ = {5, 11, 17, 23, 29}
    want_layers = [
        "full_attention" if layer in globals_ else "sliding_attention"
        for layer in range(30)
    ]
    if text.get("layer_types") != want_layers:
        raise ContractError("text_config layer_types schedule drift")
    vision_exact = {
        "model_type": "gemma4_vision", "dtype": "bfloat16",
        "num_hidden_layers": 27, "hidden_size": 1152,
        "intermediate_size": 4304, "num_attention_heads": 16,
        "num_key_value_heads": 16, "head_dim": 72, "global_head_dim": 72,
        "patch_size": 16, "position_embedding_size": 10240,
        "pooling_kernel_size": 3, "default_output_length": 280,
        "hidden_activation": "gelu_pytorch_tanh", "standardize": True,
    }
    for key, want in vision_exact.items():
        if vision.get(key) != want:
            raise ContractError(f"vision_config {key}: expected {want!r}, got {vision.get(key)!r}")

    quant = config.get("quantization")
    if not isinstance(quant, dict) or quant != config.get("quantization_config"):
        raise ContractError("quantization and quantization_config must be identical objects")
    expected_overrides = {
        f"language_model.model.layers.{layer}.router.proj":
            {"bits": 8, "group_size": 64}
        for layer in range(30)
    }
    if quant.get("mode") != "affine" or quant.get("bits") != 4 or quant.get("group_size") != 64:
        raise ContractError("default MLX affine-Q4 policy drift")
    overrides = {key: value for key, value in quant.items()
                 if key not in {"mode", "bits", "group_size"}}
    if overrides != expected_overrides:
        raise ContractError("router Q8 override set drift")
    return quant


def _validate_generation(generation: dict) -> None:
    expected = {
        "bos_token_id": 2, "pad_token_id": 0,
        "eos_token_id": [1, 106, 50], "do_sample": True,
        "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        "transformers_version": "5.5.0.dev0",
    }
    if generation != expected:
        raise ContractError("generation_config drift")


def _validate_processor(processor: dict) -> None:
    image = processor.get("image_processor")
    if not isinstance(image, dict):
        raise ContractError("processor image_processor must be an object")
    expected = {
        "image_processor_type": "Gemma4ImageProcessor",
        "do_convert_rgb": True, "do_normalize": False,
        "do_rescale": True, "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
        "image_seq_length": 280, "max_soft_tokens": 280,
        "patch_size": 16, "pooling_kernel_size": 3,
        "resample": 3, "rescale_factor": 1.0 / 255.0,
        "size": {"height": 224, "width": 224},
    }
    if image != expected:
        raise ContractError("image processor policy drift")
    if processor.get("processor_class") != "Gemma4Processor" or processor.get("image_seq_length") != 280:
        raise ContractError("processor top-level image contract drift")


def _validate_authorities(source_root: Path, tree: dict[str, dict],
                          headers: dict[str, dict]) -> tuple[dict, dict, dict]:
    all_tensors = {}
    tensor_shards = {}
    authorities = {}
    dtype_bytes = {"BF16": 0, "U32": 0}
    dtype_sizes = {"BF16": 2, "U32": 4}
    for shard_name in sorted(headers):
        pinned = headers[shard_name]
        path = _local_path(source_root, pinned["authority_path"])
        authority_bytes = _read_authenticated_bytes(
            path, None, pinned["authority_sha256"]
        )
        authority = _parse_json_bytes(authority_bytes, path)
        _require_exact_keys(authority, {"repository", "revision", "schema", "shard", "tensors"}, path.name)
        if authority["repository"] != REPOSITORY or authority["revision"] != REVISION:
            raise ContractError(f"{path}: repository/revision drift")
        if authority["schema"] != AUTHORITY_SCHEMA:
            raise ContractError(f"{path}: authority schema drift")
        tree_entry = tree[shard_name]
        expected_shard = {
            "name": shard_name,
            "file_bytes": tree_entry["bytes"],
            "header_bytes": pinned["header_bytes"],
            "header_sha256": pinned["header_sha256"],
            "metadata": {"format": "mlx"},
            "payload_bytes": pinned["payload_bytes"],
            "sha256": tree_entry["sha256"],
            "tensor_count": pinned["tensor_count"],
        }
        if authority["shard"] != expected_shard:
            raise ContractError(f"{path}: shard identity/extent drift")
        tensors = authority["tensors"]
        if not isinstance(tensors, dict) or len(tensors) != pinned["tensor_count"]:
            raise ContractError(f"{path}: tensor count drift")
        ranges = []
        for name, meta in tensors.items():
            if name in all_tensors:
                raise ContractError(f"duplicate tensor across shards: {name}")
            if not isinstance(name, str) or not isinstance(meta, dict):
                raise ContractError(f"{path}: invalid tensor entry")
            _require_exact_keys(meta, {"dtype", "shape", "data_offsets"}, name)
            dtype = meta["dtype"]
            shape = meta["shape"]
            offsets = meta["data_offsets"]
            if dtype not in dtype_sizes:
                raise ContractError(f"{name}: unsupported stored dtype {dtype!r}")
            if (not isinstance(shape, list) or not shape or
                    any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                        for dim in shape)):
                raise ContractError(f"{name}: invalid shape {shape!r}")
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    any(not isinstance(value, int) or isinstance(value, bool)
                        for value in offsets) or offsets[0] < 0 or offsets[1] <= offsets[0]):
                raise ContractError(f"{name}: invalid data_offsets {offsets!r}")
            elements = 1
            for dim in shape:
                elements *= dim
            expected_bytes = elements * dtype_sizes[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                raise ContractError(f"{name}: shape/dtype extent mismatch")
            ranges.append((offsets[0], offsets[1], name))
            dtype_bytes[dtype] += expected_bytes
            all_tensors[name] = meta
            tensor_shards[name] = shard_name
        cursor = 0
        for start, end, name in sorted(ranges):
            if start != cursor:
                raise ContractError(f"{shard_name}: payload gap/overlap before {name}")
            cursor = end
        if cursor != pinned["payload_bytes"]:
            raise ContractError(f"{shard_name}: payload extent {cursor} != pinned {pinned['payload_bytes']}")
        authorities[shard_name] = authority
    if len(all_tensors) != EXPECTED_INVENTORY["tensor_count"]:
        raise ContractError("aggregate tensor count drift")
    if sum(dtype_bytes.values()) != EXPECTED_INVENTORY["tensor_payload_bytes"]:
        raise ContractError("aggregate tensor extent drift")
    if dtype_bytes != EXPECTED_INVENTORY["stored_dtype_bytes"]:
        raise ContractError("aggregate dtype extent drift")
    return all_tensors, tensor_shards, authorities


def _validate_index(index: dict, tensors: dict, tensor_shards: dict) -> None:
    _require_exact_keys(index, {"metadata", "weight_map"}, "safetensor index")
    if index["metadata"] != {"total_size": EXPECTED_INVENTORY["tensor_payload_bytes"]}:
        raise ContractError("safetensor index total_size drift")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or set(weight_map) != set(tensors):
        raise ContractError("safetensor index tensor set drift")
    for name, shard in weight_map.items():
        if tensor_shards[name] != shard:
            raise ContractError(f"safetensor index shard drift: {name}")


def _validate_quantization(tensors: dict, quant: dict) -> dict:
    packed = {name[:-7]: meta for name, meta in tensors.items()
              if name.endswith(".weight") and meta["dtype"] == "U32"}
    other_u32 = [name for name, meta in tensors.items()
                 if meta["dtype"] == "U32" and not name.endswith(".weight")]
    if other_u32:
        raise ContractError(f"non-weight packed tensors present: {other_u32[:3]}")
    auxiliaries = {name for name in tensors
                   if name.endswith(".scales") or name.endswith(".biases")}
    seen_aux = set()
    counts = {4: 0, 8: 0}
    for base, weight in packed.items():
        override = quant.get(base)
        bits = override["bits"] if isinstance(override, dict) else quant["bits"]
        group = override["group_size"] if isinstance(override, dict) else quant["group_size"]
        if bits not in {4, 8} or group != 64:
            raise ContractError(f"{base}: unsupported affine policy")
        shape = weight["shape"]
        values_per_word = 32 // bits
        logical_cols = shape[-1] * values_per_word
        if logical_cols % group:
            raise ContractError(f"{base}: logical inner dimension not divisible by group size")
        aux_shape = shape[:-1] + [logical_cols // group]
        for suffix in ("scales", "biases"):
            name = f"{base}.{suffix}"
            meta = tensors.get(name)
            if meta is None or meta["dtype"] != "BF16" or meta["shape"] != aux_shape:
                raise ContractError(f"{name}: affine auxiliary drift")
            seen_aux.add(name)
        counts[bits] += 1
    if seen_aux != auxiliaries:
        raise ContractError("orphan affine scale/bias tensor present")
    if counts != {4: 297, 8: 30}:
        raise ContractError(f"affine bundle counts drift: {counts}")
    routers = {f"language_model.model.layers.{layer}.router.proj" for layer in range(30)}
    actual_q8 = {
        base for base in packed
        if isinstance(quant.get(base), dict) and quant[base].get("bits") == 8
    }
    if actual_q8 != routers:
        raise ContractError("router Q8 tensor set drift")
    if "embed_vision.embedding_projection" not in packed:
        raise ContractError("Q4 vision-to-text projection missing")
    vision = {name: meta for name, meta in tensors.items()
              if name.startswith("vision_tower.")}
    if len(vision) != 355 or any(meta["dtype"] != "BF16" for meta in vision.values()):
        raise ContractError("BF16 vision tower inventory drift")
    if any(name.endswith(".scales") or name.endswith(".biases") for name in vision):
        raise ContractError("vision tower unexpectedly has quantization auxiliaries")
    experts = {name: meta for name, meta in tensors.items()
               if ".experts.switch_glu." in name}
    if len(experts) != 270 or sum(meta["dtype"] == "U32" for meta in experts.values()) != 90:
        raise ContractError("routed expert inventory drift")
    return {"q4_bundle_count": counts[4], "q8_bundle_count": counts[8],
            "vision_bf16_tensor_count": len(vision),
            "expert_tensor_count": len(experts)}


def validate_source(contract_path: Path = DEFAULT_CONTRACT) -> dict:
    raw_contract_path = contract_path.expanduser()
    if not raw_contract_path.is_absolute():
        raw_contract_path = Path.cwd() / raw_contract_path
    contract_path = _local_path(raw_contract_path.parent, raw_contract_path.name)
    contract_bytes = _read_stable_bytes(contract_path)
    contract = _parse_json_bytes(contract_bytes, contract_path)
    tree, headers = _validate_contract(contract)
    source_root = contract_path.parent
    controls = {}
    for name, entry in tree.items():
        local_path = entry.get("local_path")
        if local_path is not None:
            path = _local_path(source_root, local_path)
            controls[name] = (
                path,
                _read_authenticated_bytes(path, entry["bytes"], entry["sha256"]),
            )
    config_path, config_bytes = controls["config.json"]
    config = _parse_json_bytes(config_bytes, config_path)
    quant = _validate_config(config)
    generation_path, generation_bytes = controls["generation_config.json"]
    generation = _parse_json_bytes(generation_bytes, generation_path)
    _validate_generation(generation)
    processor_path, processor_bytes = controls["processor_config.json"]
    processor = _parse_json_bytes(processor_bytes, processor_path)
    _validate_processor(processor)
    tensors, tensor_shards, authorities = _validate_authorities(source_root, tree, headers)
    index_path, index_bytes = controls["model.safetensors.index.json"]
    index = _parse_json_bytes(index_bytes, index_path)
    _validate_index(index, tensors, tensor_shards)
    quant_summary = _validate_quantization(tensors, quant)
    state = {
        "contract_path": contract_path,
        "contract": contract,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_root": source_root,
        "tree": tree,
        "headers": headers,
        "tensors": tensors,
        "tensor_shards": tensor_shards,
        "authorities": authorities,
    }
    map_summary = build_import_map(state, include_ranges=False)
    return {
        "schema": SOURCE_SCHEMA,
        "repository": REPOSITORY,
        "revision": REVISION,
        "contract_sha256": state["contract_sha256"],
        "repository_sibling_count": len(tree),
        "tensor_count": len(tensors),
        "tensor_payload_bytes": EXPECTED_INVENTORY["tensor_payload_bytes"],
        **quant_summary,
        "copy_range_count": map_summary["copy_range_count"],
        "copy_map_sha256": map_summary["copy_map_sha256"],
        "pool_file_bytes": EXPECTED_IMPORT_SCALARS["pool_file_bytes"],
        "payload_authenticated": False,
        "runtime_ready": False,
        "_state": state,
    }


def _copy_range_records(state: dict) -> Iterator[dict]:
    contract = state["contract"]
    expert_import = contract["expert_import"]
    tensors = state["tensors"]
    tensor_shards = state["tensor_shards"]
    headers = state["headers"]
    components = {entry["name"]: entry for entry in expert_import["components"]}
    part_keys = {
        "weight": ("value_relative_offset", "value_bytes"),
        "scales": ("scale_relative_offset", "scale_bytes"),
        "biases": ("bias_relative_offset", "bias_bytes"),
    }
    slot_bytes = expert_import["expert_slot_bytes"]
    experts = expert_import["experts_per_layer"]
    source_cursors = {}
    source_counts = {}
    target_cursor = expert_import["pool_header_bytes"]
    for layer in range(expert_import["layers"]):
        for expert in range(experts):
            slot_start = expert_import["pool_header_bytes"] + (layer * experts + expert) * slot_bytes
            for component_name in expert_import["projection_order"]:
                component = components[component_name]
                rows, cols = component["shape"]
                for part in expert_import["part_order"]:
                    tensor_name = expert_import["source_tensor_prefix"].format(
                        layer=layer, component=component_name, part=part)
                    meta = tensors.get(tensor_name)
                    if meta is None:
                        raise ContractError(f"missing expert tensor {tensor_name}")
                    shard = tensor_shards[tensor_name]
                    if part == "weight":
                        expected_dtype = "U32"
                        expected_shape = [experts, rows, cols // 8]
                    else:
                        expected_dtype = "BF16"
                        expected_shape = [experts, rows, cols // 64]
                    if meta["dtype"] != expected_dtype or meta["shape"] != expected_shape:
                        raise ContractError(
                            f"{tensor_name}: expected {expected_dtype} {expected_shape}, "
                            f"got {meta['dtype']} {meta['shape']}"
                        )
                    start, end = meta["data_offsets"]
                    total_bytes = end - start
                    if total_bytes % experts:
                        raise ContractError(f"{tensor_name}: outer expert axis is not byte-sliceable")
                    slice_bytes = total_bytes // experts
                    rel_key, bytes_key = part_keys[part]
                    if slice_bytes != component[bytes_key]:
                        raise ContractError(
                            f"{tensor_name}: slice extent {slice_bytes} != target {component[bytes_key]}"
                        )
                    source_offset = 8 + headers[shard]["header_bytes"] + start + expert * slice_bytes
                    source_end = source_offset + slice_bytes
                    if source_end > state["tree"][shard]["bytes"]:
                        raise ContractError(f"{tensor_name}: source slice exceeds shard extent")
                    tensor_start = 8 + headers[shard]["header_bytes"] + start
                    if source_offset != source_cursors.get(tensor_name, tensor_start):
                        raise ContractError(f"{tensor_name}: source slices overlap or have a gap")
                    source_cursors[tensor_name] = source_end
                    source_counts[tensor_name] = source_counts.get(tensor_name, 0) + 1
                    target_offset = slot_start + component[rel_key]
                    if target_offset != target_cursor:
                        raise ContractError("target copy ranges overlap or have a gap")
                    target_cursor = target_offset + slice_bytes
                    yield {
                        "layer": layer,
                        "expert": expert,
                        "component": component_name,
                        "part": part,
                        "source_shard": shard,
                        "source_file_offset": source_offset,
                        "target_file_offset": target_offset,
                        "nbytes": slice_bytes,
                    }
    expected_tensor_count = (
        expert_import["layers"]
        * len(expert_import["projection_order"])
        * len(expert_import["part_order"])
    )
    if len(source_cursors) != expected_tensor_count:
        raise ContractError("copy ranges do not cover the exact expert tensor set")
    for tensor_name, source_end in source_cursors.items():
        meta = tensors[tensor_name]
        shard = tensor_shards[tensor_name]
        expected_end = 8 + headers[shard]["header_bytes"] + meta["data_offsets"][1]
        if source_counts[tensor_name] != experts or source_end != expected_end:
            raise ContractError(f"{tensor_name}: source slices do not cover tensor extent")
    if target_cursor != expert_import["pool_file_bytes"]:
        raise ContractError("target copy ranges do not cover the complete pool")


def build_import_map(state_or_summary: dict, include_ranges: bool = True) -> dict:
    state = state_or_summary.get("_state", state_or_summary)
    expert = state["contract"]["expert_import"]
    digest = hashlib.sha256()
    ranges = [] if include_ranges else None
    count = 0
    target_cursor = expert["pool_header_bytes"]
    for record in _copy_range_records(state):
        if record["target_file_offset"] != target_cursor:
            raise ContractError(
                f"target map gap/overlap at range {count}: "
                f"{record['target_file_offset']} != {target_cursor}"
            )
        target_cursor += record["nbytes"]
        line = (
            f"{record['layer']}\t{record['expert']}\t{record['component']}\t"
            f"{record['part']}\t{record['source_shard']}\t"
            f"{record['source_file_offset']}\t{record['target_file_offset']}\t"
            f"{record['nbytes']}\n"
        )
        digest.update(line.encode("utf-8"))
        if ranges is not None:
            ranges.append(record)
        count += 1
    if target_cursor != expert["pool_file_bytes"]:
        raise ContractError(f"target extent {target_cursor} != pinned {expert['pool_file_bytes']}")
    map_digest = digest.hexdigest()
    if count != expert["copy_range_count"] or map_digest != expert["copy_map_sha256"]:
        raise ContractError(
            f"copy map drift: count={count}, sha256={map_digest}"
        )
    result = {
        "schema": IMPORT_MAP_SCHEMA,
        "status": "source-ranges-qualified-runtime-blocked",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "contract_sha256": state["contract_sha256"],
        },
        "target": {
            "format": "salt-gemma4-expert-pool-bytecopy.v1",
            "pool_header_bytes": expert["pool_header_bytes"],
            "expert_slot_bytes": expert["expert_slot_bytes"],
            "layers": expert["layers"],
            "experts_per_layer": expert["experts_per_layer"],
            "pool_file_bytes": expert["pool_file_bytes"],
            "runtime_ready": False,
            "runtime_blocker": expert["runtime_manifest_status"],
        },
        "copy_range_count": count,
        "copy_map_sha256": map_digest,
    }
    if ranges is not None:
        result["ranges"] = ranges
    return result


def _validate_downloaded_header(handle, path: Path, state: dict,
                                shard_name: str) -> None:
    pinned = state["headers"][shard_name]
    authority = state["authorities"][shard_name]
    try:
        handle.seek(0)
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ContractError(f"{path}: short safetensor length prefix")
        header_bytes = struct.unpack("<Q", raw_length)[0]
        if header_bytes != pinned["header_bytes"]:
            raise ContractError(
                f"{path}: header extent {header_bytes} != pinned {pinned['header_bytes']}"
            )
        raw_header = handle.read(header_bytes)
        if len(raw_header) != header_bytes:
            raise ContractError(f"{path}: short safetensor header")
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    digest = hashlib.sha256(raw_header).hexdigest()
    if digest != pinned["header_sha256"]:
        raise ContractError(f"{path}: raw header sha256 {digest} != pinned {pinned['header_sha256']}")
    try:
        parsed = json.loads(raw_header, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: raw header is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ContractError(f"{path}: raw header root must be an object")
    metadata = parsed.pop("__metadata__", None)
    if metadata != authority["shard"]["metadata"] or parsed != authority["tensors"]:
        raise ContractError(f"{path}: raw header differs from checked authority")


def _authenticate_downloads(source_dir: Path, summary: dict) -> tuple[dict, dict, dict]:
    state = summary["_state"]
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise ContractError(f"download directory does not exist: {source_dir}")
    authenticated = []
    total_bytes = 0
    handles = {}
    identities = {}
    try:
        for name in sorted(state["tree"]):
            entry = state["tree"][name]
            if entry["storage"] != "lfs":
                continue
            path = source_dir / name
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise ContractError(f"cannot open regular LFS payload {path}: {exc}") from exc
            try:
                handle = os.fdopen(fd, "rb")
            except Exception:
                os.close(fd)
                raise
            try:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ContractError(f"LFS payload is not a regular file: {path}")
                identity = _stat_identity(before)
                if before.st_size != entry["bytes"]:
                    raise ContractError(
                        f"{path}: extent {before.st_size} != pinned {entry['bytes']}"
                    )
                digest = hashlib.sha256()
                handle.seek(0)
                while True:
                    chunk = handle.read(8 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                after_hash = os.fstat(handle.fileno())
                if _stat_identity(after_hash) != identity:
                    raise ContractError(f"{path}: identity changed while hashing")
                if digest.hexdigest() != entry["sha256"]:
                    raise ContractError(
                        f"{path}: sha256 {digest.hexdigest()} != pinned {entry['sha256']}"
                    )
                if name in state["headers"]:
                    _validate_downloaded_header(handle, path, state, name)
                if _stat_identity(os.fstat(handle.fileno())) != identity:
                    raise ContractError(f"{path}: identity changed during header validation")
            except Exception:
                handle.close()
                raise
            handles[name] = handle
            identities[name] = identity
            authenticated.append(name)
            total_bytes += entry["bytes"]
    except Exception:
        for handle in handles.values():
            handle.close()
        raise
    report = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "authenticated_files": authenticated,
        "authenticated_file_count": len(authenticated),
        "authenticated_bytes": total_bytes,
        "payload_authenticated": True,
        "runtime_ready": False,
    }
    return report, handles, identities


def _open_authenticated_downloads(
    source_dir: Path, summary: dict,
    expected_identities: dict[str, list[int] | tuple[int, int, int, int, int]],
) -> tuple[dict, dict, dict]:
    """Retain previously authenticated payloads without rehashing their bytes."""
    state = summary["_state"]
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise ContractError(f"download directory does not exist: {source_dir}")
    required = {
        name for name, entry in state["tree"].items()
        if entry["storage"] == "lfs"
    }
    if set(expected_identities) != required:
        raise ContractError("authenticated source receipt file-set drift")
    authenticated = []
    total_bytes = 0
    handles = {}
    identities = {}
    try:
        for name in sorted(required):
            entry = state["tree"][name]
            expected = tuple(expected_identities[name])
            if len(expected) != 5 or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in expected):
                raise ContractError(f"{name}: invalid cached file identity")
            path = source_dir / name
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise ContractError(
                    f"cannot open authenticated LFS payload {path}: {exc}"
                ) from exc
            try:
                handle = os.fdopen(fd, "rb")
            except Exception:
                os.close(fd)
                raise
            try:
                info = os.fstat(handle.fileno())
                identity = _stat_identity(info)
                if not stat.S_ISREG(info.st_mode) or identity != expected or \
                        info.st_size != entry["bytes"]:
                    raise ContractError(f"{path}: cached file identity drift")
                if name in state["headers"]:
                    _validate_downloaded_header(handle, path, state, name)
                if _stat_identity(os.fstat(handle.fileno())) != identity:
                    raise ContractError(
                        f"{path}: identity changed during cached header validation"
                    )
                handle.seek(0)
            except Exception:
                handle.close()
                raise
            handles[name] = handle
            identities[name] = identity
            authenticated.append(name)
            total_bytes += entry["bytes"]
    except Exception:
        for handle in handles.values():
            handle.close()
        raise
    report = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "authenticated_files": authenticated,
        "authenticated_file_count": len(authenticated),
        "authenticated_bytes": total_bytes,
        "payload_authenticated": True,
        "authentication_mode": "cached-file-identity",
        "runtime_ready": False,
    }
    return report, handles, identities


def validate_downloads(source_dir: Path, summary: dict) -> dict:
    report, handles, _ = _authenticate_downloads(source_dir, summary)
    for handle in handles.values():
        handle.close()
    return report


def _canonical_output_leaf(path: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.name in {"", ".", ".."}:
        raise ContractError(f"invalid output path {path}")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"output parent does not exist: {raw.parent}") from exc
    candidate = parent / raw.name
    try:
        info = os.lstat(candidate)
    except FileNotFoundError:
        return candidate
    except OSError as exc:
        raise ContractError(f"cannot inspect output leaf {candidate}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ContractError(f"refusing output symlink leaf: {candidate}")
    return candidate


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(path, flags)
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            os.close(fd)
            fd = -1
            raise ContractError(f"not a directory: {path}")
        return fd
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        raise ContractError(f"cannot open directory {path}: {exc}") from exc


def _leaf_stat(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ContractError(f"cannot inspect publication leaf {name}: {exc}") from exc


def _publication_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _require_published_identity(directory_fd: int, name: str,
                                expected: tuple[int, int, int, int],
                                where: str) -> os.stat_result:
    info = _leaf_stat(directory_fd, name)
    if (info is None or not stat.S_ISREG(info.st_mode)
            or _publication_identity(info) != expected):
        raise ContractError(f"{where} identity changed")
    return info


def _require_linked_leaf(destination_fd: int, destination_name: str,
                         source_fd: int, source_name: str,
                         where: str) -> os.stat_result:
    destination = _leaf_stat(destination_fd, destination_name)
    source = _leaf_stat(source_fd, source_name)
    if (destination is None or source is None
            or not stat.S_ISREG(destination.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or (destination.st_dev, destination.st_ino) !=
               (source.st_dev, source.st_ino)):
        raise ContractError(f"{where} identity changed")
    return destination


def _unlink_if_same_fd(directory_fd: int, name: str,
                       source_fd: int, where: str) -> bool:
    destination = _leaf_stat(directory_fd, name)
    if destination is None:
        return False
    source = os.fstat(source_fd)
    if (not stat.S_ISREG(destination.st_mode)
            or (destination.st_dev, destination.st_ino) !=
               (source.st_dev, source.st_ino)):
        raise ContractError(f"refusing to remove foreign {where} leaf {name}")
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError as exc:
        raise ContractError(f"cannot remove owned {where} leaf {name}: {exc}") from exc
    _fsync_directory(directory_fd)
    return True


def _require_absent(directory_fd: int, name: str, where: str) -> None:
    info = _leaf_stat(directory_fd, name)
    if info is not None:
        kind = "symlink" if stat.S_ISLNK(info.st_mode) else "existing leaf"
        raise ContractError(f"refusing {where} {kind}: {name}")


def _fsync_directory(directory_fd: int) -> None:
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise ContractError(f"cannot fsync publication directory: {exc}") from exc


def _new_private_name(prefix: str, suffix: str = "") -> str:
    return f"{prefix}{secrets.token_hex(16)}{suffix}"


def _open_exclusive_at(directory_fd: int, name: str, mode: int = 0o600) -> int:
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        return os.open(name, flags, mode, dir_fd=directory_fd)
    except OSError as exc:
        raise ContractError(f"cannot create private file {name}: {exc}") from exc


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        try:
            written = os.write(fd, data[offset:])
        except OSError as exc:
            raise ContractError(f"cannot write publication data: {exc}") from exc
        if written <= 0:
            raise ContractError("short write while publishing JSON")
        offset += written


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_json(
        path: Path, value: dict
) -> tuple[Path, tuple[int, int, int, int]]:
    path = _canonical_output_leaf(path)
    directory_fd = _open_directory(path.parent)
    temporary_name = _new_private_name(f".{path.name}.", ".incomplete")
    created = False
    published = False
    durable = False
    verification_fd = -1
    try:
        _require_absent(directory_fd, path.name, "overwrite of")
        fd = _open_exclusive_at(directory_fd, temporary_name)
        created = True
        try:
            verification_fd = os.dup(fd)
            _write_all(fd, _json_bytes(value))
            os.fsync(fd)
            written_identity = _publication_identity(os.fstat(verification_fd))
        finally:
            os.close(fd)
        _require_published_identity(
            directory_fd, temporary_name, written_identity,
            "staged JSON leaf",
        )
        try:
            os.link(
                temporary_name, path.name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as exc:
            raise ContractError(f"refusing to overwrite {path}") from exc
        except OSError as exc:
            raise ContractError(f"cannot publish {path}: {exc}") from exc
        _fsync_directory(directory_fd)
        published_info = _require_published_identity(
            directory_fd, path.name, written_identity,
            "published JSON leaf",
        )
        published_identity = _publication_identity(published_info)
        durable = True
        return path, published_identity
    finally:
        try:
            if published and not durable and verification_fd >= 0:
                _unlink_if_same_fd(
                    directory_fd, path.name, verification_fd,
                    "published JSON",
                )
        finally:
            try:
                if created and verification_fd >= 0:
                    _unlink_if_same_fd(
                        directory_fd, temporary_name, verification_fd,
                        "staged JSON",
                    )
            finally:
                if verification_fd >= 0:
                    os.close(verification_fd)
                os.close(directory_fd)


def emit_import_map(output: Path, summary: dict) -> dict:
    import_map = build_import_map(summary, include_ranges=True)
    output, published_identity = _atomic_write_json(output, import_map)
    directory_fd = _open_directory(output.parent)
    try:
        info = _require_published_identity(
            directory_fd, output.name, published_identity,
            "published import map",
        )
        published_map = _read_json_at(directory_fd, output.name, output)
        if published_map != import_map:
            raise ContractError("published import map content changed")
        _require_published_identity(
            directory_fd, output.name, published_identity,
            "published import map",
        )
        return {
            "output": str(output),
            "output_bytes": info.st_size,
            "copy_range_count": import_map["copy_range_count"],
            "copy_map_sha256": import_map["copy_map_sha256"],
            "runtime_ready": False,
        }
    finally:
        os.close(directory_fd)


def _copy_exact(source, target, nbytes: int, digest: hashlib._Hash,
                chunk_bytes: int = 8 << 20) -> None:
    remaining = nbytes
    while remaining:
        chunk = source.read(min(remaining, chunk_bytes))
        if not chunk:
            raise ContractError("short source read while materializing expert pool")
        target.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


TRANSACTION_SCHEMA = "salt.gemma4-mlx4-materialization-transaction.v1"
STAGED_POOL = "pool.incomplete"
STAGED_RECEIPT = "receipt.incomplete.json"
STAGED_TRANSACTION = "transaction.json"


def _staging_prefix(output_name: str) -> str:
    return f".{output_name}.salt-gemma4-import-"


def _stage_transaction_id(stage_name: str, output_name: str) -> str | None:
    prefix = _staging_prefix(output_name)
    if not stage_name.startswith(prefix):
        return None
    transaction_id = stage_name[len(prefix):]
    if (len(transaction_id) != 32 or
            any(char not in "0123456789abcdef" for char in transaction_id)):
        return None
    return transaction_id


def _find_staging(directory_fd: int, output_name: str) -> list[str]:
    try:
        return sorted(
            name for name in os.listdir(directory_fd)
            if _stage_transaction_id(name, output_name) is not None
        )
    except OSError as exc:
        raise ContractError(f"cannot inspect materialization staging: {exc}") from exc


def _create_staging(directory_fd: int, output_name: str) -> tuple[str, int]:
    for _ in range(64):
        name = _new_private_name(_staging_prefix(output_name))
        try:
            os.mkdir(name, 0o700, dir_fd=directory_fd)
        except FileExistsError:
            continue
        flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        try:
            stage_fd = os.open(name, flags, dir_fd=directory_fd)
        except Exception:
            os.rmdir(name, dir_fd=directory_fd)
            raise
        try:
            _fsync_directory(directory_fd)
        except Exception:
            os.close(stage_fd)
            os.rmdir(name, dir_fd=directory_fd)
            raise
        return name, stage_fd
    raise ContractError("cannot allocate unique materialization staging directory")


def _write_json_at(directory_fd: int, name: str, value: dict) -> None:
    fd = _open_exclusive_at(directory_fd, name)
    try:
        _write_all(fd, _json_bytes(value))
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_held_at(
        directory_fd: int, name: str, value: dict
) -> tuple[int, tuple[int, int, int, int]]:
    fd = _open_exclusive_at(directory_fd, name)
    held_fd = -1
    try:
        held_fd = os.dup(fd)
        _write_all(fd, _json_bytes(value))
        os.fsync(fd)
        identity = _publication_identity(os.fstat(held_fd))
        return held_fd, identity
    except Exception:
        if held_fd >= 0:
            os.close(held_fd)
        raise
    finally:
        os.close(fd)


def _read_json_at(directory_fd: int, name: str, where: Path) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ContractError(f"cannot open transaction metadata {where}: {exc}") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"transaction metadata is not regular: {where}")
            identity = _stat_identity(before)
            data = handle.read()
            if _stat_identity(os.fstat(handle.fileno())) != identity:
                raise ContractError(f"transaction metadata changed while reading: {where}")
            if len(data) != before.st_size:
                raise ContractError(f"short transaction metadata read: {where}")
    except OSError as exc:
        raise ContractError(f"cannot read transaction metadata {where}: {exc}") from exc
    return _parse_json_bytes(data, where)


def _sha256_regular_at(directory_fd: int, name: str,
                       where: Path, chunk_bytes: int = 8 << 20
                       ) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ContractError(f"cannot open committed pool {where}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"committed pool is not regular: {where}")
            identity = _stat_identity(before)
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if _stat_identity(after) != identity:
                raise ContractError(
                    f"committed materialization pool identity changed while hashing: {where}"
                )
    except OSError as exc:
        raise ContractError(f"cannot hash committed pool {where}: {exc}") from exc
    return digest.hexdigest(), before


def _link_no_replace(source_fd: int, source_name: str,
                     destination_fd: int, destination_name: str) -> None:
    _require_absent(destination_fd, destination_name, "overwrite of")
    try:
        os.link(
            source_name, destination_name,
            src_dir_fd=source_fd, dst_dir_fd=destination_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite {destination_name}") from exc


def _unlink_if_same_inode(destination_fd: int, destination_name: str,
                          source_fd: int, source_name: str) -> bool:
    destination = _leaf_stat(destination_fd, destination_name)
    if destination is None:
        return False
    source = _leaf_stat(source_fd, source_name)
    if source is None or not stat.S_ISREG(source.st_mode):
        raise ContractError(f"missing regular staged source for {destination_name}")
    if (destination.st_dev, destination.st_ino) != (source.st_dev, source.st_ino):
        raise ContractError(f"refusing to remove foreign output leaf {destination_name}")
    try:
        os.unlink(destination_name, dir_fd=destination_fd)
    except OSError as exc:
        raise ContractError(f"cannot remove provisional leaf {destination_name}: {exc}") from exc
    _fsync_directory(destination_fd)
    return True


def _remove_staging(directory_fd: int, stage_name: str, stage_fd: int) -> None:
    allowed = {STAGED_POOL, STAGED_RECEIPT, STAGED_TRANSACTION}
    try:
        entries = set(os.listdir(stage_fd))
        unknown = entries - allowed
        if unknown:
            raise ContractError(
                f"refusing to remove staging directory with unknown entries: {sorted(unknown)}"
            )
        removal_order = [STAGED_POOL, STAGED_RECEIPT, STAGED_TRANSACTION]
        for name in removal_order:
            if name not in entries:
                continue
            info = _leaf_stat(stage_fd, name)
            if info is None:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ContractError(f"refusing non-regular staging entry {name}")
            try:
                os.unlink(name, dir_fd=stage_fd)
            except OSError as exc:
                raise ContractError(f"cannot remove staging entry {name}: {exc}") from exc
        _fsync_directory(stage_fd)
    finally:
        os.close(stage_fd)
    try:
        os.rmdir(stage_name, dir_fd=directory_fd)
    except OSError as exc:
        raise ContractError(f"cannot remove staging directory {stage_name}: {exc}") from exc
    _fsync_directory(directory_fd)


def _validate_transaction(transaction: dict, stage_name: str,
                          output: Path, receipt: Path, summary: dict) -> None:
    stage_transaction_id = _stage_transaction_id(stage_name, output.name)
    _require_exact_keys(
        transaction,
        {"schema", "status", "transaction_id", "output_name", "receipt_name",
         "contract_sha256", "pool_bytes", "pool_sha256", "runtime_ready"},
        "materialization transaction",
    )
    if (transaction["schema"] != TRANSACTION_SCHEMA or
            transaction["status"] != "prepared-runtime-blocked" or
            transaction["output_name"] != output.name or
            transaction["receipt_name"] != receipt.name or
            transaction["contract_sha256"] != summary["_state"]["contract_sha256"] or
            transaction["pool_bytes"] != summary["_state"]["contract"]["expert_import"]["pool_file_bytes"] or
            not isinstance(transaction["transaction_id"], str) or
            len(transaction["transaction_id"]) != 32 or
            transaction["transaction_id"] != stage_transaction_id or
            any(char not in "0123456789abcdef" for char in transaction["transaction_id"]) or
            not isinstance(transaction["pool_sha256"], str) or
            len(transaction["pool_sha256"]) != 64 or
            any(char not in "0123456789abcdef" for char in transaction["pool_sha256"]) or
            transaction["runtime_ready"] is not False):
        raise ContractError("materialization transaction identity drift")


def _validate_committed_receipt(receipt_value: dict, transaction_id: str,
                                output: Path, output_info: os.stat_result,
                                summary: dict,
                                transaction: dict | None = None) -> None:
    state = summary["_state"]
    expert = state["contract"]["expert_import"]
    _require_exact_keys(
        receipt_value,
        {"schema", "status", "transaction_id", "source", "copy_range_count",
         "copy_map_sha256", "pool", "runtime_ready", "runtime_blocker"},
        "materialization receipt",
    )
    expected_source = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "contract_sha256": state["contract_sha256"],
        "authenticated_bytes": sum(
            entry["bytes"] for entry in state["tree"].values()
            if entry["storage"] == "lfs"
        ),
    }
    pool = receipt_value["pool"]
    if (receipt_value["schema"] != "salt.gemma4-mlx4-import-receipt.v1" or
            receipt_value["status"] != "expert-pool-materialized-runtime-blocked" or
            receipt_value["transaction_id"] != transaction_id or
            receipt_value["source"] != expected_source or
            receipt_value["copy_range_count"] != expert["copy_range_count"] or
            receipt_value["copy_map_sha256"] != expert["copy_map_sha256"] or
            not isinstance(pool, dict) or
            pool.get("path") != str(output) or
            pool.get("bytes") != expert["pool_file_bytes"] or
            not isinstance(pool.get("sha256"), str) or
            len(pool["sha256"]) != 64 or
            any(char not in "0123456789abcdef" for char in pool["sha256"]) or
            not stat.S_ISREG(output_info.st_mode) or
            output_info.st_size != expert["pool_file_bytes"] or
            receipt_value["runtime_ready"] is not False or
            receipt_value["runtime_blocker"] != expert["runtime_manifest_status"]):
        raise ContractError("committed materialization receipt identity drift")
    if transaction is not None and (
            transaction["transaction_id"] != transaction_id or
            transaction["pool_bytes"] != pool["bytes"] or
            transaction["pool_sha256"] != pool["sha256"]):
        raise ContractError("receipt/transaction identity drift")


def recover_materialization(output: Path, receipt: Path | None,
                            summary: dict) -> dict:
    output = _canonical_output_leaf(output)
    if receipt is None:
        receipt = output.with_name(output.name + ".import.json")
    else:
        receipt = _canonical_output_leaf(receipt)
    if receipt == output or receipt.parent != output.parent:
        raise ContractError("expert pool and receipt must be distinct leaves in one parent")
    directory_fd = _open_directory(output.parent)
    recovered = 0
    committed = 0
    try:
        stages = _find_staging(directory_fd, output.name)
        for stage_name in stages:
            info = _leaf_stat(directory_fd, stage_name)
            if (info is None or not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.getuid() or info.st_mode & 0o077):
                raise ContractError(f"unsafe materialization staging leaf {stage_name}")
            flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
            stage_fd = os.open(stage_name, flags, dir_fd=directory_fd)
            opened = os.fstat(stage_fd)
            if ((opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino) or
                    not stat.S_ISDIR(opened.st_mode)):
                os.close(stage_fd)
                raise ContractError(f"materialization staging identity changed: {stage_name}")
            remove_stage = True
            try:
                entries = set(os.listdir(stage_fd))
                transaction = None
                committed_transaction_id = None
                committed_receipt_value = None
                committed_output_identity = None
                committed_receipt_identity = None
                if STAGED_TRANSACTION in entries:
                    transaction = _read_json_at(
                        stage_fd, STAGED_TRANSACTION,
                        output.parent / stage_name / STAGED_TRANSACTION,
                    )
                    _validate_transaction(
                        transaction, stage_name, output, receipt, summary
                    )
                output_info = _leaf_stat(directory_fd, output.name)
                receipt_info = _leaf_stat(directory_fd, receipt.name)
                if transaction is None and (output_info is not None or receipt_info is not None):
                    if entries or output_info is None or receipt_info is None:
                        raise ContractError("unbound staging cannot recover visible output leaves")
                    transaction_id = _stage_transaction_id(stage_name, output.name)
                    if transaction_id is None:
                        raise ContractError("empty committed staging identity drift")
                    final_receipt = _read_json_at(
                        directory_fd, receipt.name, receipt
                    )
                    _validate_committed_receipt(
                        final_receipt, transaction_id, output,
                        output_info, summary
                    )
                    pool_digest, output_info = _sha256_regular_at(
                        directory_fd, output.name, output
                    )
                    if pool_digest != final_receipt["pool"]["sha256"]:
                        raise ContractError(
                            "committed materialization pool sha256 drift"
                        )
                    receipt_info = _require_published_identity(
                        directory_fd, receipt.name,
                        _publication_identity(receipt_info),
                        "committed materialization receipt",
                    )
                    committed_transaction_id = transaction_id
                    committed_receipt_value = final_receipt
                    committed_output_identity = _publication_identity(output_info)
                    committed_receipt_identity = _publication_identity(receipt_info)
                    committed += 1
                if transaction is not None:
                    if output_info is not None:
                        _unlink_if_same_inode(
                            directory_fd, output.name, stage_fd, STAGED_POOL
                        ) if receipt_info is None else None
                    if receipt_info is not None:
                        if output_info is None:
                            _unlink_if_same_inode(
                                directory_fd, receipt.name, stage_fd, STAGED_RECEIPT
                            )
                        else:
                            pool_stage = _leaf_stat(stage_fd, STAGED_POOL)
                            receipt_stage = _leaf_stat(stage_fd, STAGED_RECEIPT)
                            final_receipt = _read_json_at(
                                directory_fd, receipt.name, receipt
                            )
                            _validate_committed_receipt(
                                final_receipt, transaction["transaction_id"],
                                output, output_info, summary, transaction
                            )
                            if pool_stage is not None:
                                output_info = _require_linked_leaf(
                                    directory_fd, output.name,
                                    stage_fd, STAGED_POOL,
                                    "committed materialization pool",
                                )
                            else:
                                pool_digest, output_info = _sha256_regular_at(
                                    directory_fd, output.name, output
                                )
                                if pool_digest != final_receipt["pool"]["sha256"]:
                                    raise ContractError(
                                        "committed materialization pool sha256 drift"
                                    )
                            if receipt_stage is not None:
                                receipt_info = _require_linked_leaf(
                                    directory_fd, receipt.name,
                                    stage_fd, STAGED_RECEIPT,
                                    "committed materialization receipt",
                                )
                            else:
                                receipt_info = _require_published_identity(
                                    directory_fd, receipt.name,
                                    _publication_identity(receipt_info),
                                    "committed materialization receipt",
                                )
                            committed_transaction_id = transaction["transaction_id"]
                            committed_receipt_value = final_receipt
                            committed_output_identity = _publication_identity(output_info)
                            committed_receipt_identity = _publication_identity(receipt_info)
                            committed += 1
                stage_to_remove = stage_fd
                stage_fd = -1
                _remove_staging(directory_fd, stage_name, stage_to_remove)
                if committed_output_identity is not None:
                    if (committed_receipt_identity is None
                            or committed_transaction_id is None
                            or committed_receipt_value is None):
                        raise ContractError(
                            "internal committed materialization identity missing"
                        )
                    _require_published_identity(
                        directory_fd, output.name, committed_output_identity,
                        "committed materialization pool",
                    )
                    _require_published_identity(
                        directory_fd, receipt.name, committed_receipt_identity,
                        "committed materialization receipt",
                    )
                    final_receipt_after = _read_json_at(
                        directory_fd, receipt.name, receipt
                    )
                    _validate_committed_receipt(
                        final_receipt_after, committed_transaction_id,
                        output,
                        _require_published_identity(
                            directory_fd, output.name,
                            committed_output_identity,
                            "committed materialization pool",
                        ),
                        summary, transaction,
                    )
                    if final_receipt_after != committed_receipt_value:
                        raise ContractError(
                            "committed materialization receipt content changed"
                        )
                    _require_published_identity(
                        directory_fd, receipt.name,
                        committed_receipt_identity,
                        "committed materialization receipt",
                    )
                recovered += 1
                remove_stage = False
            finally:
                if stage_fd >= 0:
                    os.close(stage_fd)
                if remove_stage:
                    pass
        return {
            "output": str(output), "receipt": str(receipt),
            "recovered_transactions": recovered,
            "committed_transactions": committed,
            "runtime_ready": False,
        }
    finally:
        os.close(directory_fd)


def materialize_expert_pool(source_dir: Path, output: Path,
                            receipt: Path | None, summary: dict) -> dict:
    state = summary["_state"]
    expert = state["contract"]["expert_import"]
    output = _canonical_output_leaf(output)
    if receipt is None:
        receipt = output.with_name(output.name + ".import.json")
    else:
        receipt = _canonical_output_leaf(receipt)
    if receipt == output or receipt.parent != output.parent:
        raise ContractError("expert pool and receipt must be distinct leaves in one parent")
    directory_fd = _open_directory(output.parent)
    stage_name = None
    stage_fd = -1
    handles = {}
    published_pool_fd = -1
    published_receipt_fd = -1
    commit_verified = False
    cleanup_pending = None
    try:
        stale = _find_staging(directory_fd, output.name)
        if stale:
            raise ContractError(
                f"stale materialization transaction present; run recover-materialization: {stale}"
            )
        _require_absent(directory_fd, output.name, "overwrite of")
        _require_absent(directory_fd, receipt.name, "overwrite of")
        filesystem = os.fstatvfs(directory_fd)
        free = filesystem.f_bavail * filesystem.f_frsize
        required = expert["pool_file_bytes"] + expert["minimum_free_after_materialize_bytes"]
        if free < required:
            raise ContractError(f"insufficient free space: {free} < required {required}")
        authenticated, handles, identities = _authenticate_downloads(source_dir, summary)
        stage_name, stage_fd = _create_staging(directory_fd, output.name)
        transaction_id = stage_name.rsplit("-", 1)[-1]
        pool_fd = _open_exclusive_at(stage_fd, STAGED_POOL)
        try:
            published_pool_fd = os.dup(pool_fd)
        except Exception:
            os.close(pool_fd)
            raise
        digest = hashlib.sha256()
        with os.fdopen(pool_fd, "wb") as target:
            header = struct.pack(
                "<QQQ", expert["expert_slot_bytes"],
                expert["layers"], expert["experts_per_layer"]
            )
            target.write(header)
            digest.update(header)
            for record in _copy_range_records(state):
                shard = record["source_shard"]
                source = handles[shard]
                source.seek(record["source_file_offset"])
                _copy_exact(source, target, record["nbytes"], digest)
            for name, handle in handles.items():
                if _stat_identity(os.fstat(handle.fileno())) != identities[name]:
                    raise ContractError(f"{name}: identity changed while copying")
            target.flush()
            os.fsync(target.fileno())
            written_pool_identity = _publication_identity(
                os.fstat(published_pool_fd)
            )
        pool_info = _require_published_identity(
            stage_fd, STAGED_POOL, written_pool_identity,
            "staged expert pool",
        )
        if pool_info.st_size != expert["pool_file_bytes"]:
            raise ContractError(
                f"materialized pool extent {pool_info.st_size} != pinned "
                f"{expert['pool_file_bytes']}"
            )
        pool_digest = digest.hexdigest()
        import_receipt = {
            "schema": "salt.gemma4-mlx4-import-receipt.v1",
            "status": "expert-pool-materialized-runtime-blocked",
            "transaction_id": transaction_id,
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "contract_sha256": state["contract_sha256"],
                "authenticated_bytes": authenticated["authenticated_bytes"],
            },
            "copy_range_count": expert["copy_range_count"],
            "copy_map_sha256": expert["copy_map_sha256"],
            "pool": {
                "path": str(output),
                "bytes": pool_info.st_size,
                "sha256": pool_digest,
            },
            "runtime_ready": False,
            "runtime_blocker": expert["runtime_manifest_status"],
        }
        published_receipt_fd, written_receipt_identity = _write_json_held_at(
            stage_fd, STAGED_RECEIPT, import_receipt
        )
        transaction = {
            "schema": TRANSACTION_SCHEMA,
            "status": "prepared-runtime-blocked",
            "transaction_id": transaction_id,
            "output_name": output.name,
            "receipt_name": receipt.name,
            "contract_sha256": state["contract_sha256"],
            "pool_bytes": pool_info.st_size,
            "pool_sha256": pool_digest,
            "runtime_ready": False,
        }
        _write_json_at(stage_fd, STAGED_TRANSACTION, transaction)
        _fsync_directory(stage_fd)
        _require_published_identity(
            stage_fd, STAGED_POOL, written_pool_identity,
            "staged expert pool",
        )
        _link_no_replace(stage_fd, STAGED_POOL, directory_fd, output.name)
        _fsync_directory(directory_fd)
        _require_published_identity(
            directory_fd, output.name, written_pool_identity,
            "published expert pool",
        )
        _require_published_identity(
            stage_fd, STAGED_RECEIPT, written_receipt_identity,
            "staged materialization receipt",
        )
        _link_no_replace(stage_fd, STAGED_RECEIPT, directory_fd, receipt.name)
        _fsync_directory(directory_fd)
        published_pool_info = _require_published_identity(
            directory_fd, output.name, written_pool_identity,
            "published expert pool",
        )
        published_receipt_info = _require_published_identity(
            directory_fd, receipt.name, written_receipt_identity,
            "published materialization receipt",
        )
        published_receipt = _read_json_at(
            directory_fd, receipt.name, receipt
        )
        _validate_committed_receipt(
            published_receipt, transaction_id, output,
            published_pool_info, summary, transaction,
        )
        if published_receipt != import_receipt:
            raise ContractError("published materialization receipt content changed")
        published_pool_identity = written_pool_identity
        published_receipt_identity = written_receipt_identity

        stage_to_remove = stage_fd
        stage_fd = -1
        if stage_name is None:
            raise ContractError("internal materialization staging identity missing")
        try:
            _remove_staging(directory_fd, stage_name, stage_to_remove)
            stage_name = None
        except ContractError:
            cleanup_pending = stage_name
        final_pool_info = _require_published_identity(
            directory_fd, output.name, published_pool_identity,
            "published expert pool",
        )
        _require_published_identity(
            directory_fd, receipt.name, published_receipt_identity,
            "published materialization receipt",
        )
        final_receipt = _read_json_at(directory_fd, receipt.name, receipt)
        _validate_committed_receipt(
            final_receipt, transaction_id, output,
            final_pool_info, summary, transaction,
        )
        if final_receipt != import_receipt:
            raise ContractError("published materialization receipt content changed")
        _require_published_identity(
            directory_fd, output.name, published_pool_identity,
            "published expert pool",
        )
        _require_published_identity(
            directory_fd, receipt.name, published_receipt_identity,
            "published materialization receipt",
        )
        commit_verified = True
        return {
            "output": str(output), "output_bytes": pool_info.st_size,
            "output_sha256": pool_digest, "receipt": str(receipt),
            "staging_cleanup_pending": cleanup_pending,
            "runtime_ready": False,
        }
    except OSError as exc:
        raise ContractError(f"expert pool materialization failed: {exc}") from exc
    finally:
        for handle in handles.values():
            handle.close()
        if not commit_verified:
            if published_receipt_fd >= 0:
                try:
                    _unlink_if_same_fd(
                        directory_fd, receipt.name, published_receipt_fd,
                        "materialization receipt",
                    )
                except ContractError:
                    pass
            if published_pool_fd >= 0:
                try:
                    _unlink_if_same_fd(
                        directory_fd, output.name, published_pool_fd,
                        "expert pool",
                    )
                except ContractError:
                    pass
            if stage_fd >= 0:
                try:
                    stage_to_remove = stage_fd
                    stage_fd = -1
                    if stage_name is None:
                        raise ContractError(
                            "internal materialization staging identity missing"
                        )
                    _remove_staging(directory_fd, stage_name, stage_to_remove)
                    stage_name = None
                except ContractError:
                    pass
        if stage_fd >= 0:
            os.close(stage_fd)
        if published_receipt_fd >= 0:
            os.close(published_receipt_fd)
        if published_pool_fd >= 0:
            os.close(published_pool_fd)
        os.close(directory_fd)


def _public_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "_state"}


def _print(value: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        for key, item in value.items():
            print(f"{key}={item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-source")
    validate.add_argument("--json", action="store_true")
    emit = sub.add_parser("emit-import-map")
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--json", action="store_true")
    downloads = sub.add_parser("validate-download")
    downloads.add_argument("--source-dir", type=Path, required=True)
    downloads.add_argument("--json", action="store_true")
    materialize = sub.add_parser("materialize-expert-pool")
    materialize.add_argument("--source-dir", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--receipt", type=Path)
    materialize.add_argument("--json", action="store_true")
    recover = sub.add_parser("recover-materialization")
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--receipt", type=Path)
    recover.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = validate_source(args.contract)
        if args.command == "validate-source":
            result = _public_summary(summary)
        elif args.command == "emit-import-map":
            result = emit_import_map(args.output, summary)
        elif args.command == "validate-download":
            result = validate_downloads(args.source_dir, summary)
        elif args.command == "materialize-expert-pool":
            result = materialize_expert_pool(
                args.source_dir, args.output, args.receipt, summary
            )
        else:
            result = recover_materialization(args.output, args.receipt, summary)
        _print(result, args.json)
    except ContractError as exc:
        print(f"gemma4 MLX-Q4 source contract: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
