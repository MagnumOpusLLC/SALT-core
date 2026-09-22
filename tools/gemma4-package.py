#!/usr/bin/env python3
"""Metadata-only Gemma 4 package contract, role map, and dry-run estimator.

This tool never reads tensor payload bytes. It accepts config.json plus a JSON
copy of safetensor headers, or the authenticated MLX header authority, and fails
closed on name, role, dtype, shape, source-extent, or pinned-map drift.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "models" / "gemma4-26b-a4b" / "package-plan.json"
DEFAULT_MLX_SOURCE_CONTRACT = (
    ROOT / "models" / "gemma4-26b-a4b" / "mlx4-source" / "source-contract.json"
)
DEFAULT_MLX_ROLE_CONTRACT = (
    ROOT / "models" / "gemma4-26b-a4b" / "mlx4-package" /
    "source-role-contract.json"
)
MLX_SOURCE_TOOL = ROOT / "tools" / "gemma4-mlx4-source.py"
MLX_ROLE_MAP_SCHEMA = "salt.gemma4-mlx4-source-role-map.v1"
MLX_ROLE_CONTRACT_SCHEMA = "salt.gemma4-mlx4-source-role-contract.v1"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
MODEL_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
TRANSFORMERS_REVISION = "242f5df996ded281cc328c5b9bcc1bee0ec48118"
SOURCE_SHARDS = [
    {
        "name": "model-00001-of-00002.safetensors",
        "file_bytes": 49907246508,
        "sha256": "1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3",
    },
    {
        "name": "model-00002-of-00002.safetensors",
        "file_bytes": 1704763408,
        "sha256": "aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7",
    },
]


class ContractError(ValueError):
    pass


def load_json(path: str | Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return doc


def _expect(obj: dict, key: str, want, where: str) -> None:
    got = obj.get(key)
    if got != want:
        raise ContractError(f"{where}.{key}: expected {want!r}, got {got!r}")


def validate_plan(plan: dict) -> None:
    _expect(plan, "schema", "salt.gemma4-package-plan.v1", "plan")
    _expect(plan, "status", "foundation-only-no-weights", "plan")
    model = plan.get("model")
    source = plan.get("source")
    precision = plan.get("precision")
    text = plan.get("text")
    vision = plan.get("vision")
    estimates = plan.get("estimates")
    if not isinstance(model, dict):
        raise ContractError("plan.model must be an object")
    if not isinstance(source, dict):
        raise ContractError("plan.source must be an object")
    if not isinstance(precision, dict):
        raise ContractError("plan.precision must be an object")
    if not isinstance(text, dict):
        raise ContractError("plan.text must be an object")
    if not isinstance(vision, dict):
        raise ContractError("plan.vision must be an object")
    if not isinstance(estimates, dict):
        raise ContractError("plan.estimates must be an object")
    _expect(model, "registry_name", "gemma4-26b-a4b", "plan.model")
    _expect(model, "repository", "google/gemma-4-26B-A4B-it", "plan.model")
    _expect(model, "revision", MODEL_REVISION, "plan.model")
    _expect(
        model,
        "transformers_revision",
        TRANSFORMERS_REVISION,
        "plan.model",
    )
    _expect(source, "dtype", "BF16", "plan.source")
    _expect(source, "tensor_count", 1013, "plan.source")
    _expect(source, "tensor_bytes", 51611872412, "plan.source")
    _expect(source, "shards", SOURCE_SHARDS, "plan.source")
    _expect(precision, "text_large_matrices", "affine-q4", "plan.precision")
    _expect(precision, "q4_group_size", 64, "plan.precision")
    _expect(precision, "vision", "BF16", "plan.precision")
    _expect(text, "layers", 30, "plan.text")
    _expect(text, "global_layers", [5, 11, 17, 23, 29], "plan.text")
    _expect(text, "eos_token_ids", [1, 106, 50], "plan.text")
    _expect(text, "vision_bidirectional_local", True, "plan.text")
    _expect(text["sliding_attention"], "rope_base_dim", 256,
            "plan.text.sliding_attention")
    _expect(text["sliding_attention"], "rope_type", "default",
            "plan.text.sliding_attention")
    _expect(text["global_attention"], "rope_base_dim", 512,
            "plan.text.global_attention")
    _expect(text["global_attention"], "rope_type", "proportional",
            "plan.text.global_attention")
    _expect(vision, "enabled", True, "plan.vision")
    _expect(vision, "lazy", True, "plan.vision")
    if estimates.get("total_package_bytes") != (
        estimates.get("text_package_with_separate_head_bytes", -1)
        + estimates.get("vision_bf16_bytes", -2)
    ):
        raise ContractError("plan.estimates total is not text + vision")


def validate_config(config: dict, plan: dict) -> None:
    validate_plan(plan)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ContractError("config must contain text_config and vision_config objects")
    _expect(config, "model_type", "gemma4", "config")
    _expect(config, "image_token_id", plan["vision"]["image_token_id"], "config")
    _expect(config, "boi_token_id", plan["vision"]["boi_token_id"], "config")
    _expect(config, "eoi_token_id", plan["vision"]["eoi_token_id"], "config")
    _expect(config, "eos_token_id", [1, 106], "config")
    _expect(
        config,
        "vision_soft_tokens_per_image",
        plan["vision"]["default_soft_tokens"],
        "config",
    )
    if config.get("audio_config") is not None:
        raise ContractError("this package plan is image+text only; audio_config must be null")

    checks = {
        "model_type": "gemma4_text",
        "dtype": "bfloat16",
        "vocab_size": 262144,
        "num_hidden_layers": plan["text"]["layers"],
        "hidden_size": plan["text"]["hidden"],
        "intermediate_size": plan["text"]["dense_intermediate"],
        "moe_intermediate_size": plan["text"]["routed_intermediate"],
        "num_experts": plan["text"]["experts"],
        "top_k_experts": plan["text"]["topk"],
        "num_attention_heads": plan["text"]["sliding_attention"]["heads"],
        "num_key_value_heads": plan["text"]["sliding_attention"]["kv_heads"],
        "head_dim": plan["text"]["sliding_attention"]["head_dim"],
        "global_head_dim": plan["text"]["global_attention"]["head_dim"],
        "num_global_key_value_heads": plan["text"]["global_attention"]["kv_heads"],
        "sliding_window": plan["text"]["sliding_window"],
        "hidden_activation": plan["text"]["activation"],
        "final_logit_softcapping": plan["text"]["logit_softcap"],
        "attention_k_eq_v": True,
        "tie_word_embeddings": True,
        "hidden_size_per_layer_input": 0,
        "use_bidirectional_attention": "vision",
    }
    for key, want in checks.items():
        _expect(text, key, want, "config.text_config")
    layer_types = [
        "full_attention" if i in set(plan["text"]["global_layers"])
        else "sliding_attention"
        for i in range(plan["text"]["layers"])
    ]
    _expect(text, "layer_types", layer_types, "config.text_config")
    _expect(text, "rope_parameters", {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    }, "config.text_config")

    vision_checks = {
        "model_type": "gemma4_vision",
        "dtype": "bfloat16",
        "num_hidden_layers": plan["vision"]["layers"],
        "hidden_size": plan["vision"]["hidden"],
        "intermediate_size": plan["vision"]["intermediate"],
        "num_attention_heads": plan["vision"]["heads"],
        "num_key_value_heads": plan["vision"]["heads"],
        "head_dim": plan["vision"]["head_dim"],
        "global_head_dim": plan["vision"]["head_dim"],
        "patch_size": plan["vision"]["patch_size"],
        "position_embedding_size": plan["vision"]["position_embedding_size"],
        "pooling_kernel_size": plan["vision"]["pooling_kernel_size"],
        "default_output_length": plan["vision"]["default_soft_tokens"],
        "hidden_activation": "gelu_pytorch_tanh",
        "standardize": True,
    }
    for key, want in vision_checks.items():
        _expect(vision, key, want, "config.vision_config")
    _expect(vision, "rope_parameters", {
        "rope_theta": plan["vision"]["rope_theta"],
        "rope_type": "default",
    }, "config.vision_config")


def validate_generation_config(generation: dict, plan: dict) -> None:
    validate_plan(plan)
    checks = {
        "bos_token_id": 2,
        "pad_token_id": 0,
        "eos_token_id": plan["text"]["eos_token_ids"],
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
    }
    for key, want in checks.items():
        _expect(generation, key, want, "generation_config")


def required_shapes(plan: dict) -> dict[str, list[int]]:
    validate_plan(plan)
    text = plan["text"]
    hidden = text["hidden"]
    dense = text["dense_intermediate"]
    routed = text["routed_intermediate"]
    experts = text["experts"]
    globals_ = set(text["global_layers"])
    shapes: dict[str, list[int]] = {
        "model.language_model.embed_tokens.weight": [262144, hidden],
        "model.language_model.norm.weight": [hidden],
    }
    norm_leaves = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
    )
    for layer in range(text["layers"]):
        base = f"model.language_model.layers.{layer}."
        for leaf in norm_leaves:
            shapes[base + leaf] = [hidden]
        shapes[base + "layer_scalar"] = [1]
        shapes[base + "mlp.gate_proj.weight"] = [dense, hidden]
        shapes[base + "mlp.up_proj.weight"] = [dense, hidden]
        shapes[base + "mlp.down_proj.weight"] = [hidden, dense]
        shapes[base + "experts.gate_up_proj"] = [experts, 2 * routed, hidden]
        shapes[base + "experts.down_proj"] = [experts, hidden, routed]
        shapes[base + "router.proj.weight"] = [experts, hidden]
        shapes[base + "router.scale"] = [hidden]
        shapes[base + "router.per_expert_scale"] = [experts]
        if layer in globals_:
            attn = text["global_attention"]
            q_rows = attn["heads"] * attn["head_dim"]
            kv_rows = attn["kv_heads"] * attn["head_dim"]
        else:
            attn = text["sliding_attention"]
            q_rows = attn["heads"] * attn["head_dim"]
            kv_rows = attn["kv_heads"] * attn["head_dim"]
            shapes[base + "self_attn.v_proj.weight"] = [kv_rows, hidden]
        shapes[base + "self_attn.q_proj.weight"] = [q_rows, hidden]
        shapes[base + "self_attn.k_proj.weight"] = [kv_rows, hidden]
        shapes[base + "self_attn.o_proj.weight"] = [hidden, q_rows]
        shapes[base + "self_attn.q_norm.weight"] = [attn["head_dim"]]
        shapes[base + "self_attn.k_norm.weight"] = [attn["head_dim"]]

    vision = plan["vision"]
    vh = vision["hidden"]
    vi = vision["intermediate"]
    hd = vision["head_dim"]
    shapes.update({
        "model.embed_vision.embedding_projection.weight": [hidden, vh],
        "model.vision_tower.patch_embedder.input_proj.weight": [vh, 3 * vision["patch_size"] ** 2],
        "model.vision_tower.patch_embedder.position_embedding_table": [2, vision["position_embedding_size"], vh],
        "model.vision_tower.std_bias": [vh],
        "model.vision_tower.std_scale": [vh],
    })
    vision_layer_shapes = {
        "input_layernorm.weight": [vh],
        "post_attention_layernorm.weight": [vh],
        "pre_feedforward_layernorm.weight": [vh],
        "post_feedforward_layernorm.weight": [vh],
        "self_attn.q_norm.weight": [hd],
        "self_attn.k_norm.weight": [hd],
        "self_attn.q_proj.linear.weight": [vh, vh],
        "self_attn.k_proj.linear.weight": [vh, vh],
        "self_attn.v_proj.linear.weight": [vh, vh],
        "self_attn.o_proj.linear.weight": [vh, vh],
        "mlp.gate_proj.linear.weight": [vi, vh],
        "mlp.up_proj.linear.weight": [vi, vh],
        "mlp.down_proj.linear.weight": [vh, vi],
    }
    for layer in range(vision["layers"]):
        base = f"model.vision_tower.encoder.layers.{layer}."
        for leaf, shape in vision_layer_shapes.items():
            shapes[base + leaf] = shape
    return shapes


def q4_affine_bytes(shape: list[int], group_size: int) -> int:
    if len(shape) < 2 or any(not isinstance(x, int) or x < 1 for x in shape):
        raise ContractError(f"invalid matrix shape: {shape!r}")
    rows = math.prod(shape[:-1])
    cols = shape[-1]
    if cols % 8:
        raise ContractError(f"Q4 inner dimension is not packable by 8: {shape!r}")
    packed = rows * cols // 2
    groups = rows * ((cols + group_size - 1) // group_size)
    return packed + groups * 4  # BF16 scale + BF16 bias per affine group.


def _header_tensors(headers: dict) -> dict:
    tensors = headers.get("tensors", headers)
    if not isinstance(tensors, dict):
        raise ContractError("headers.tensors must be an object")
    return {name: meta for name, meta in tensors.items() if name != "__metadata__"}


def summarize_headers(headers: dict, plan: dict, strict: bool = True) -> dict:
    validate_plan(plan)
    if strict:
        revision = headers.get("revision")
        expected_revision = plan["model"]["revision"]
        if revision != expected_revision:
            raise ContractError(
                f"headers revision {revision!r} != pinned {expected_revision!r}"
            )
    tensors = _header_tensors(headers)
    required = required_shapes(plan)
    if strict:
        missing = sorted(set(required) - set(tensors))
        extra = sorted(set(tensors) - set(required))
        if missing or extra:
            raise ContractError(
                f"tensor set drift: missing={missing[:3]} ({len(missing)}), "
                f"extra={extra[:3]} ({len(extra)})"
            )
    group = plan["precision"]["q4_group_size"]
    source_bytes = 0
    text_once = 0
    vision_bytes = 0
    routed_bytes = 0
    quantized = 0
    embed_q4 = None
    for name, meta in tensors.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise ContractError("invalid tensor header entry")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        if dtype != plan["source"]["dtype"]:
            raise ContractError(f"{name}: expected BF16 source, got {dtype!r}")
        if not isinstance(shape, list) or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in shape
        ):
            raise ContractError(f"{name}: invalid shape {shape!r}")
        if name in required and shape != required[name]:
            raise ContractError(
                f"{name}: expected shape {required[name]}, got {shape}"
            )
        source_nbytes = math.prod(shape) * 2
        offsets = meta.get("data_offsets")
        if offsets is not None:
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    any(not isinstance(x, int) for x in offsets) or
                    offsets[0] < 0 or offsets[1] < offsets[0] or
                    offsets[1] - offsets[0] != source_nbytes):
                raise ContractError(f"{name}: invalid data_offsets {offsets!r}")
        source_bytes += source_nbytes
        is_vision = name.startswith("model.vision_tower.") or name.startswith("model.embed_vision.")
        if is_vision:
            vision_bytes += source_nbytes
            continue
        if not name.startswith("model.language_model."):
            raise ContractError(f"unclassified tensor: {name}")
        keep_bf16 = name.endswith("router.proj.weight")
        if len(shape) >= 2 and not keep_bf16:
            package_nbytes = q4_affine_bytes(shape, group)
            quantized += 1
        else:
            package_nbytes = source_nbytes
        text_once += package_nbytes
        if ".experts." in name:
            routed_bytes += package_nbytes
        if name == "model.language_model.embed_tokens.weight":
            embed_q4 = package_nbytes
    if embed_q4 is None and strict:
        raise ContractError("tied embedding tensor missing")
    embed_q4 = embed_q4 or 0
    text_with_head = text_once + embed_q4
    total_package = text_with_head + vision_bytes
    result = {
        "tensor_count": len(tensors),
        "source_tensor_bytes": source_bytes,
        "quantized_tensor_count": quantized,
        "text_once_bytes": text_once,
        "embedding_q4_bytes": embed_q4,
        "text_with_separate_head_bytes": text_with_head,
        "routed_q4_bytes": routed_bytes,
        "expert_slot_bytes": routed_bytes // (plan["text"]["layers"] * plan["text"]["experts"]),
        "vision_bf16_bytes": vision_bytes,
        "total_package_bytes": total_package,
    }
    if strict:
        expected = plan["estimates"]
        comparisons = {
            "tensor_count": plan["source"]["tensor_count"],
            "source_tensor_bytes": plan["source"]["tensor_bytes"],
            "expert_slot_bytes": expected["expert_slot_bytes"],
            "text_with_separate_head_bytes": expected["text_package_with_separate_head_bytes"],
            "vision_bf16_bytes": expected["vision_bf16_bytes"],
            "total_package_bytes": expected["total_package_bytes"],
        }
        for key, want in comparisons.items():
            if result[key] != want:
                raise ContractError(f"{key}: expected {want}, got {result[key]}")
    return result


def _load_mlx_source_module():
    spec = importlib.util.spec_from_file_location(
        "gemma4_mlx_source_for_package", MLX_SOURCE_TOOL
    )
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load authenticated source tool {MLX_SOURCE_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_descriptor(state: dict, name: str, consumed: set[str]) -> dict:
    if name in consumed:
        raise ContractError(f"source tensor mapped more than once: {name}")
    meta = state["tensors"].get(name)
    if not isinstance(meta, dict):
        raise ContractError(f"required MLX source tensor missing: {name}")
    shard = state["tensor_shards"].get(name)
    header = state["headers"].get(shard)
    if not isinstance(header, dict):
        raise ContractError(f"source shard authority missing for {name}")
    start, end = meta["data_offsets"]
    consumed.add(name)
    return {
        "tensor": name,
        "shard": shard,
        "file_offset": 8 + header["header_bytes"] + start,
        "nbytes": end - start,
        "dtype": meta["dtype"],
        "stored_shape": list(meta["shape"]),
    }


def _build_mlx_source_role_map(plan: dict, source_summary: dict) -> dict:
    validate_plan(plan)
    state = source_summary.get("_state")
    if not isinstance(state, dict):
        raise ContractError("authenticated MLX source state is unavailable")
    if source_summary.get("runtime_ready") is not False:
        raise ContractError("MLX source validation did not remain runtime-blocked")
    source_contract = state.get("contract")
    if not isinstance(source_contract, dict):
        raise ContractError("authenticated MLX source contract is unavailable")
    source_identity = source_contract.get("source")
    if not isinstance(source_identity, dict):
        raise ContractError("authenticated MLX source identity is unavailable")
    if source_identity.get("semantic_reference_revision") != plan["model"]["revision"]:
        raise ContractError("MLX semantic reference and package plan revision differ")
    if source_identity.get("transformers_revision") != plan["model"][
            "transformers_revision"]:
        raise ContractError("MLX transformers reference and package plan differ")

    consumed: set[str] = set()
    entries: list[dict] = []
    roles: set[str] = set()

    def append(entry: dict) -> None:
        role = entry["role"]
        if role in roles:
            raise ContractError(f"duplicate package role: {role}")
        roles.add(role)
        entries.append(entry)

    def add_affine(role: str, base: str, bits: int,
                   component: str, **extra) -> None:
        if bits not in {4, 8}:
            raise ContractError(f"{role}: unsupported affine width {bits}")
        source = {
            part: _source_descriptor(
                state, f"{base}.{part}", consumed
            )
            for part in ("weight", "scales", "biases")
        }
        weight = source["weight"]
        values_per_word = 32 // bits
        logical_shape = list(weight["stored_shape"])
        if weight["dtype"] != "U32" or not logical_shape:
            raise ContractError(f"{base}: invalid packed affine weight")
        logical_shape[-1] *= values_per_word
        auxiliary_shape = logical_shape[:-1] + [logical_shape[-1] // 64]
        for part in ("scales", "biases"):
            if (source[part]["dtype"] != "BF16" or
                    source[part]["stored_shape"] != auxiliary_shape):
                raise ContractError(f"{base}.{part}: affine companion drift")
        entry = {
            "role": role,
            "package_component": component,
            "storage": f"mlx-affine-q{bits}",
            "affine_bits": bits,
            "group_size": 64,
            "logical_shape": logical_shape,
            "source": source,
        }
        entry.update(extra)
        append(entry)

    def add_bf16(role: str, name: str, component: str, **extra) -> None:
        source = _source_descriptor(state, name, consumed)
        if source["dtype"] != "BF16":
            raise ContractError(f"{name}: expected BF16 package tensor")
        entry = {
            "role": role,
            "package_component": component,
            "storage": "bf16",
            "logical_shape": list(source["stored_shape"]),
            "source": source,
        }
        entry.update(extra)
        append(entry)

    add_affine(
        "text.embedding",
        "language_model.model.embed_tokens",
        4,
        "embedding",
    )
    add_bf16(
        "text.final_norm",
        "language_model.model.norm.weight",
        "text_final",
    )

    global_layers = set(plan["text"]["global_layers"])
    norm_leaves = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
    )
    for layer in range(plan["text"]["layers"]):
        base = f"language_model.model.layers.{layer}"
        attention_kind = "full" if layer in global_layers else "sliding"
        add_affine(
            f"text.layers.{layer}.attention.q_proj",
            f"{base}.self_attn.q_proj",
            4,
            "text_trunk",
            attention_kind=attention_kind,
        )
        if attention_kind == "full":
            add_affine(
                f"text.layers.{layer}.attention.kv_proj",
                f"{base}.self_attn.k_proj",
                4,
                "text_trunk",
                attention_kind=attention_kind,
                shared_kv_projection=True,
            )
        else:
            add_affine(
                f"text.layers.{layer}.attention.k_proj",
                f"{base}.self_attn.k_proj",
                4,
                "text_trunk",
                attention_kind=attention_kind,
                shared_kv_projection=False,
            )
            add_affine(
                f"text.layers.{layer}.attention.v_proj",
                f"{base}.self_attn.v_proj",
                4,
                "text_trunk",
                attention_kind=attention_kind,
                shared_kv_projection=False,
            )
        add_affine(
            f"text.layers.{layer}.attention.o_proj",
            f"{base}.self_attn.o_proj",
            4,
            "text_trunk",
            attention_kind=attention_kind,
        )
        for norm in ("q_norm", "k_norm"):
            add_bf16(
                f"text.layers.{layer}.attention.{norm}",
                f"{base}.self_attn.{norm}.weight",
                "text_trunk",
                attention_kind=attention_kind,
            )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            add_affine(
                f"text.layers.{layer}.dense.{projection}",
                f"{base}.mlp.{projection}",
                4,
                "text_trunk",
            )
            add_affine(
                f"text.layers.{layer}.experts.{projection}",
                f"{base}.experts.switch_glu.{projection}",
                4,
                "expert_pool",
                source_outer_axis="expert",
            )
        add_affine(
            f"text.layers.{layer}.router.proj",
            f"{base}.router.proj",
            8,
            "text_trunk",
        )
        add_bf16(
            f"text.layers.{layer}.router.scale",
            f"{base}.router.scale",
            "text_trunk",
        )
        add_bf16(
            f"text.layers.{layer}.router.per_expert_scale",
            f"{base}.router.per_expert_scale",
            "text_trunk",
        )
        for leaf in norm_leaves:
            add_bf16(
                f"text.layers.{layer}.{leaf[:-len('.weight')]}",
                f"{base}.{leaf}",
                "text_trunk",
            )
        add_bf16(
            f"text.layers.{layer}.layer_scalar",
            f"{base}.layer_scalar",
            "text_trunk",
        )

    add_affine(
        "vision_to_text.projection",
        "embed_vision.embedding_projection",
        4,
        "vision_projection",
    )
    vision_global = {
        "patch_embedder.input_proj":
            "vision_tower.patch_embedder.input_proj.weight",
        "patch_embedder.position_embedding_table":
            "vision_tower.patch_embedder.position_embedding_table",
        "std_bias": "vision_tower.std_bias",
        "std_scale": "vision_tower.std_scale",
    }
    for role, name in vision_global.items():
        add_bf16(f"vision.{role}", name, "vision_lazy")
    vision_leaves = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_proj.linear.weight",
        "self_attn.k_proj.linear.weight",
        "self_attn.v_proj.linear.weight",
        "self_attn.o_proj.linear.weight",
        "mlp.gate_proj.linear.weight",
        "mlp.up_proj.linear.weight",
        "mlp.down_proj.linear.weight",
    )
    for layer in range(plan["vision"]["layers"]):
        base = f"vision_tower.encoder.layers.{layer}"
        for leaf in vision_leaves:
            role_leaf = leaf.removesuffix(".linear.weight").removesuffix(".weight")
            add_bf16(
                f"vision.layers.{layer}.{role_leaf}",
                f"{base}.{leaf}",
                "vision_lazy",
            )

    all_tensors = set(state["tensors"])
    if consumed != all_tensors:
        missing = sorted(all_tensors - consumed)
        extra = sorted(consumed - all_tensors)
        raise ContractError(
            f"MLX source role closure failed: unmapped={missing[:3]} "
            f"({len(missing)}), unknown={extra[:3]} ({len(extra)})"
        )
    entries.sort(key=lambda entry: entry["role"])
    component_counts: dict[str, int] = {}
    for entry in entries:
        component = entry["package_component"]
        component_counts[component] = component_counts.get(component, 0) + 1
    map_digest = hashlib.sha256()
    for entry in entries:
        line = json.dumps(
            entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8") + b"\n"
        map_digest.update(line)
    tensor_bytes = sum(
        meta["data_offsets"][1] - meta["data_offsets"][0]
        for meta in state["tensors"].values()
    )
    return {
        "schema": MLX_ROLE_MAP_SCHEMA,
        "status": "source-roles-qualified-runtime-ready",
        "source": {
            "repository": source_identity["repository"],
            "revision": source_identity["revision"],
            "contract_sha256": source_summary["contract_sha256"],
            "declared_upstream_repository":
                source_identity["declared_upstream_repository"],
            "declared_upstream_revision":
                source_identity["declared_upstream_revision"],
            "semantic_reference_repository":
                source_identity["semantic_reference_repository"],
            "semantic_reference_revision":
                source_identity["semantic_reference_revision"],
            "weight_parity_claim": False,
        },
        "role_map_framing": (
            "sorted role; UTF-8 json(sort_keys=true,separators=(',',':'),"
            "ensure_ascii=true) plus LF per entry"
        ),
        "source_tensor_count": len(all_tensors),
        "source_tensor_bytes": tensor_bytes,
        "covered_source_tensor_count": len(consumed),
        "role_entry_count": len(entries),
        "role_map_sha256": map_digest.hexdigest(),
        "package_components": dict(sorted(component_counts.items())),
        "expert_pool": {
            "copy_range_count": source_summary["copy_range_count"],
            "copy_map_sha256": source_summary["copy_map_sha256"],
            "pool_file_bytes": source_summary["pool_file_bytes"],
            "materialized_artifact_is_runtime_manifest": False,
        },
        "tied_head": {
            "logical_roles": ["text.embedding", "text.output_head"],
            "separate_source_tensor": False,
            "source_role": "text.embedding",
            "runtime_loader_ready": True,
        },
        "runtime_ready": True,
        "runtime_blockers": [],
        "entries": entries,
    }


def _validate_mlx_role_contract(contract: dict, role_map: dict) -> None:
    expected = {
        "schema": MLX_ROLE_CONTRACT_SCHEMA,
        "status": "source-roles-qualified-runtime-ready",
        "source": role_map["source"],
        "mapping": {
            "role_map_framing": role_map["role_map_framing"],
            "source_tensor_count": role_map["source_tensor_count"],
            "source_tensor_bytes": role_map["source_tensor_bytes"],
            "role_entry_count": role_map["role_entry_count"],
            "role_map_sha256": role_map["role_map_sha256"],
            "package_components": role_map["package_components"],
        },
        "tied_head": role_map["tied_head"],
        "runtime_ready": True,
    }
    if contract != expected:
        raise ContractError("pinned MLX source-role contract drift")


def build_mlx_source_role_map(
        plan: dict,
        source_contract: str | Path = DEFAULT_MLX_SOURCE_CONTRACT,
        role_contract: str | Path = DEFAULT_MLX_ROLE_CONTRACT,
        source_summary: dict | None = None) -> dict:
    validate_plan(plan)
    if source_summary is None:
        source_module = _load_mlx_source_module()
        try:
            source_summary = source_module.validate_source(Path(source_contract))
        except source_module.ContractError as exc:
            raise ContractError(f"authenticated MLX source rejected: {exc}") from exc
    if source_summary is None:
        raise ContractError("authenticated MLX source summary is unavailable")
    role_map = _build_mlx_source_role_map(plan, source_summary)
    _validate_mlx_role_contract(load_json(role_contract), role_map)
    return role_map


def print_summary(summary: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print("GEMMA4_PACKAGE_DRY_RUN_OK")
    for key, value in summary.items():
        print(f"{key}={value}")


def print_mlx_role_map(role_map: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(role_map, indent=2, sort_keys=True))
        return
    print("GEMMA4_MLX_SOURCE_ROLE_MAP_OK")
    for key in (
            "status", "source_tensor_count", "source_tensor_bytes",
            "covered_source_tensor_count", "role_entry_count",
            "role_map_sha256", "package_components", "runtime_ready"):
        print(f"{key}={role_map[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-plan")
    cfg_parser = sub.add_parser("validate-config")
    cfg_parser.add_argument("--config", required=True)
    generation_parser = sub.add_parser("validate-generation")
    generation_parser.add_argument("--generation-config", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--config", required=True)
    dry.add_argument("--generation-config", required=True)
    dry.add_argument("--headers", required=True)
    dry.add_argument("--allow-partial", action="store_true")
    dry.add_argument("--json", action="store_true")
    roles = sub.add_parser("source-role-map")
    roles.add_argument(
        "--source-contract", default=str(DEFAULT_MLX_SOURCE_CONTRACT)
    )
    roles.add_argument(
        "--role-contract", default=str(DEFAULT_MLX_ROLE_CONTRACT)
    )
    roles.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = load_json(args.plan)
        validate_plan(plan)
        if args.command == "validate-plan":
            print("GEMMA4_PACKAGE_PLAN_OK")
        elif args.command == "validate-config":
            validate_config(load_json(args.config), plan)
            print("GEMMA4_CONFIG_OK")
        elif args.command == "validate-generation":
            validate_generation_config(load_json(args.generation_config), plan)
            print("GEMMA4_GENERATION_CONFIG_OK")
        elif args.command == "source-role-map":
            role_map = build_mlx_source_role_map(
                plan, args.source_contract, args.role_contract
            )
            print_mlx_role_map(role_map, args.json)
        else:
            validate_config(load_json(args.config), plan)
            validate_generation_config(load_json(args.generation_config), plan)
            summary = summarize_headers(
                load_json(args.headers), plan, strict=not args.allow_partial
            )
            print_summary(summary, args.json)
    except ContractError as exc:
        print(f"gemma4 package contract: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
