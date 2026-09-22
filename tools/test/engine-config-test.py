#!/usr/bin/env python3
"""Regression tests for root/model configuration ownership."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import engine_config
from engine_config import load_engine_config


def test_repo_blueprint_ownership():
    root_only = {}
    engine_config._merge_config(str(ROOT / "engine.config"), root_only)
    assert "SALT_M2_BATCH" not in root_only, root_only
    assert "SALT_M2_BATCH_B" not in root_only, root_only

    cfg = load_engine_config(str(ROOT / "engine.config"))
    assert cfg["MODEL_DIR"] == "models/qwen36", cfg
    assert cfg["SALT_M2_BATCH"] == "1", cfg
    assert cfg["SALT_M2_BATCH_B"] == "128", cfg

    gemma = load_engine_config(
        str(ROOT / "engine.config"),
        model_dir=ROOT / "models" / "gemma4-26b-a4b",
    )
    assert "MODEL" not in gemma and "QUANT" not in gemma and \
        "LAYERS" not in gemma, gemma
    assert gemma["SALT_DPR_DRAFT_N"] == "4", gemma
    assert gemma["SALT_DPR_PREFILL_N"] == "64", gemma
    assert gemma["SALT_DPR_QA_BUCKET_BITS"] == "12", gemma
    assert "SALT_DPR_MENTOR_ENABLED" not in gemma, gemma
    assert "SALT_DPR_MINDSET_ATTENTION_ENABLED" not in gemma, gemma
    assert len(gemma["SALT_DPR_MENTOR_POLICY_SHA256"]) == 64, gemma
    assert len(gemma["SALT_DPR_MINDSET_ATTENTION_POLICY_SHA256"]) == 64, gemma
    assert gemma["SALT_PREFILL_B"] == "512", gemma
    assert gemma["SALT_GPU_B_QKV"] == "1024", gemma
    assert gemma["SALT_GPU_B_Z"] == "1024", gemma
    assert gemma["SALT_GPU_B_O"] == "1024", gemma
    assert gemma["SALT_M2_BATCH"] == "1", gemma
    assert gemma["SALT_ATTN_THREADS"] == "8", gemma
    assert gemma["SALT_GEMMA_COMPUTE_NODE"] == "cpu", gemma
    assert gemma["SALT_GEMMA_METAL_EXACT_CELLS"] == "0", gemma
    assert gemma["SALT_TEXT_FINE_TOKEN"] == "0", gemma
    assert gemma["SALT_EXPERT_BUDGET_GB"] == "1", gemma
    assert gemma["SALT_GEMMA_MEMORY_LIMIT_GB"] == "4.5", gemma
    assert gemma["SALT_EXPERT_PRELOAD"] == "0", gemma
    assert gemma["SALT_EXPERT_PRELOAD_LAYERS"] == "none", gemma
    assert not engine_config.validate_gemma_target_route_config(gemma)[
        "configured"
    ]


def test_gemma_residency_policy():
    base = {
        "SALT_GEMMA_COMPUTE_NODE": "cpu",
        "SALT_EXPERT_BUDGET_GB": "1",
        "SALT_EXPERT_PRELOAD": "0",
        "SALT_EXPERT_PRELOAD_LAYERS": "none",
        "SALT_FULL_MIN_RAM_GB": "20",
        "SALT_GEMMA_MEMORY_LIMIT_GB": "4.5",
    }
    for node in ("cpu", "mixed", "full"):
        cfg = dict(base, SALT_GEMMA_COMPUTE_NODE=node)
        assert engine_config.validate_gemma_residency_config(cfg)["node"] == node

    invalid = dict(base, SALT_GEMMA_COMPUTE_NODE="gpu-ish")
    try:
        engine_config.validate_gemma_residency_config(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown Gemma compute node was admitted")

    assert str(engine_config.validate_gemma_residency_config(base)[
        "memory_limit_gb"
    ]) == "4.5"
    maximum_memory = dict(base, SALT_GEMMA_MEMORY_LIMIT_GB="128")
    assert str(engine_config.validate_gemma_residency_config(maximum_memory)[
        "memory_limit_gb"
    ]) == "128"
    for value in ("", "0", "128.1", "4e0", "4.1234567890"):
        invalid_memory = dict(base, SALT_GEMMA_MEMORY_LIMIT_GB=value)
        try:
            engine_config.validate_gemma_residency_config(invalid_memory)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid memory limit admitted: {value!r}")

    too_small = dict(base, SALT_EXPERT_BUDGET_GB="12", SALT_EXPERT_PRELOAD="1")
    try:
        engine_config.validate_gemma_residency_config(too_small)
    except ValueError:
        pass
    else:
        raise AssertionError("full preload below complete geometry was admitted")

    full = dict(base, SALT_EXPERT_BUDGET_GB="13", SALT_EXPERT_PRELOAD="1")
    original = engine_config.physical_memory_bytes
    try:
        engine_config.physical_memory_bytes = lambda: 25_769_803_776
        policy = engine_config.validate_gemma_residency_config(
            full, require_host=True,
        )
        assert policy["budget_gb"] == 13 and policy["preload"] == 1, policy
        engine_config.physical_memory_bytes = lambda: 16_467_177_472
        try:
            engine_config.validate_gemma_residency_config(
                full, require_host=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("undersized host admitted full preload")
    finally:
        engine_config.physical_memory_bytes = original

    selective = dict(
        base,
        SALT_EXPERT_BUDGET_GB="10",
        SALT_EXPERT_PRELOAD_LAYERS="0,18",
    )
    policy = engine_config.validate_gemma_residency_config(selective)
    assert policy["preload_layers"] == (0, 18), policy
    assert policy["preload_layer_mask"] == (1 | (1 << 18)), policy
    assert policy["preload_required_slots"] == 384, policy
    assert policy["layer_quotas"].split(",")[0] == "128", policy
    assert policy["layer_quotas"].split(",")[18] == "128", policy
    assert len(policy["layer_quotas"].split(",")) == 30, policy

    invalid_layers = (
        "", "all", "-1", "30", "01", "0,0", "0,", ",0", "0,,1",
        "0, 1", "0-1",
    )
    for value in invalid_layers:
        invalid = dict(base, SALT_EXPERT_PRELOAD_LAYERS=value)
        try:
            engine_config.validate_gemma_residency_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid selective preload admitted: {value!r}")

    conflict = dict(
        base,
        SALT_EXPERT_BUDGET_GB="13",
        SALT_EXPERT_PRELOAD="1",
        SALT_EXPERT_PRELOAD_LAYERS="0",
    )
    try:
        engine_config.validate_gemma_residency_config(conflict)
    except ValueError:
        pass
    else:
        raise AssertionError("global/selective preload conflict was admitted")

    undersized = dict(
        base,
        SALT_EXPERT_BUDGET_GB="1",
        SALT_EXPERT_PRELOAD_LAYERS="0,1",
    )
    try:
        engine_config.validate_gemma_residency_config(undersized)
    except ValueError:
        pass
    else:
        raise AssertionError("selective preload exceeded budget/shared floor")


def test_gemma_target_route_policy():
    valid = {
        "SALT_TARGET_ROUTE_N": "32",
        "SALT_TARGET_X": "16",
        "SALT_TARGET_ROUTE_F": "2",
        "SALT_TARGET_ROUTE_Q": "8",
    }
    policy = engine_config.validate_gemma_target_route_config(valid)
    assert policy == {
        "configured": True, "n": 32, "x": 16, "f": 2, "q": 8,
        "nodes": 64, "candidate_count": 64,
    }
    for invalid in (
        {"SALT_TARGET_ROUTE_N": "32"},
        {"SALT_TARGET_ROUTE_F": "1"},
        {"SALT_TARGET_ROUTE_Q": "4"},
        {"SALT_TARGET_ROUTE_N": "32", "SALT_TARGET_ROUTE_F": "1"},
        {"SALT_TARGET_ROUTE_N": "0", "SALT_TARGET_X": "1", "SALT_TARGET_ROUTE_F": "1",
         "SALT_TARGET_ROUTE_Q": "4"},
        {"SALT_TARGET_ROUTE_N": "01", "SALT_TARGET_X": "1", "SALT_TARGET_ROUTE_F": "1",
         "SALT_TARGET_ROUTE_Q": "4"},
        {"SALT_TARGET_ROUTE_N": "128", "SALT_TARGET_X": "128", "SALT_TARGET_ROUTE_F": "2",
         "SALT_TARGET_ROUTE_Q": "4"},
        {"SALT_TARGET_ROUTE_N": "8", "SALT_TARGET_X": "8", "SALT_TARGET_ROUTE_F": "9",
         "SALT_TARGET_ROUTE_Q": "4"},
        {"SALT_TARGET_ROUTE_N": "8", "SALT_TARGET_X": "8", "SALT_TARGET_ROUTE_F": "1",
         "SALT_TARGET_ROUTE_Q": "9"},
        {"SALT_TARGET_ROUTE_N": "10", "SALT_TARGET_X": "10", "SALT_TARGET_ROUTE_F": "1",
         "SALT_TARGET_ROUTE_Q": "4", "SALT_TARGET_AREA_WORKERS": "8"},
    ):
        try:
            engine_config.validate_gemma_target_route_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid target route admitted: {invalid}")


def test_gemma_platform_recipes():
    model_dir = ROOT / "models" / "gemma4-26b-a4b"
    explicit_recipe_keys = {
        "SALT_TARGET_GPU_PROGRAM", "SALT_TARGET_CPU_GRAPH",
        "SALT_TEXT_FINE_TOKEN", "SALT_PREFILL_OPERATION_FLOW",
        "SALT_PREFILL_DECODE_LOOKAHEAD",
        "SALT_TARGET_ROUTE_N", "SALT_TARGET_X", "SALT_TARGET_ROUTE_F", "SALT_TARGET_ROUTE_Q",
        "SALT_GEMMA_MEMORY_LIMIT_GB",
    }
    for recipe in (
        "mac", "ec2", "spark", "spark-hmm", "spark-hmm-vision",
        "mac-metal", "rocm",
    ):
        text = (model_dir / "configs" / recipe / "engine.config").read_text()
        present = {
            line.split("=", 1)[0] for line in text.splitlines()
            if line and not line.startswith("#") and "=" in line
        }
        assert explicit_recipe_keys <= present, (
            recipe, explicit_recipe_keys - present,
        )
    default = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir,
    )
    assert default["SALT_PREFILL_RETAIN_LAYERS"] == "1", default
    assert default["SALT_PREFILL_DECODE_LOOKAHEAD"] == "0", default
    assert default["SALT_PREFILL_SHARED_ARENAS"] == "0", default
    assert "SALT_PREFILL_SHARED_ARENAS_PAGEABLE" not in default, default
    assert default["SALT_PREFILL_GPU_ATTENTION"] == "0", default
    target_defaults = {
        "SALT_TARGET_AREA_WORKERS": "8",
        "SALT_TARGET_CPU_GRAPH": "0",
        "SALT_TARGET_N_PARALLEL": "0",
        "SALT_EXPERT_MATRIX_WAVES": "0",
        "SALT_PREFILL_EXPERT_MATRIX_FLOW": "0",
        "SALT_TARGET_MATRIX_FLOW": "0",
        "SALT_TARGET_GPU_EXPERTS": "0",
    }
    for key, value in target_defaults.items():
        assert default[key] == value, (key, default)
    expected = {
        "mac": ("cpu", "1", "4.5", "0", "8", "8", "1", "0", "0"),
        "ec2": ("cpu", "6", "14", "0", "8", "8", "1", "0", "0"),
        "spark": ("full", "13", "60", "1", "8", "8", "1", "1", "1"),
        "spark-hmm": ("full", "2", "16", "0", "8", "8", "0", "1", "1"),
        "spark-hmm-vision": ("cpu", "1", "7", "0", "8", "8", "1", "0", "0"),
    }
    for recipe, values in expected.items():
        cfg = load_engine_config(
            str(ROOT / "engine.config"),
            model_dir=model_dir,
            recipe=recipe,
        )
        assert cfg["SALT_GEMMA_PLATFORM_RECIPE"] == recipe, cfg
        assert (
            cfg["SALT_GEMMA_COMPUTE_NODE"],
            cfg["SALT_EXPERT_BUDGET_GB"],
            cfg["SALT_GEMMA_MEMORY_LIMIT_GB"],
            cfg["SALT_EXPERT_PRELOAD"],
            cfg["SALT_ATTN_THREADS"],
            cfg["SALT_TARGET_AREA_WORKERS"],
            cfg["SALT_PREFILL_RETAIN_LAYERS"],
            cfg["SALT_PREFILL_SHARED_ARENAS"],
            cfg["SALT_PREFILL_GPU_ATTENTION"],
        ) == values, cfg
        assert "SALT_PREFILL_SHARED_ARENAS_PAGEABLE" not in cfg, cfg
        assert cfg["SALT_GPU_TRUNK_SHARED_POOL"] == "0", cfg
        assert cfg["SALT_GEMMA_METAL_EXACT_CELLS"] == "0", cfg
        assert cfg["SALT_TARGET_GPU_PROGRAM"] == (
            "1" if recipe in ("mac-metal", "spark", "spark-hmm") else "0"
        ), cfg
        assert cfg["SALT_TEXT_FINE_TOKEN"] == (
            "1" if recipe == "spark-hmm" else "0"
        ), cfg
        assert cfg["SALT_TARGET_CPU_GRAPH"] == "0", cfg
        assert cfg["SALT_PREFILL_DECODE_LOOKAHEAD"] == "0", cfg
        for key in target_defaults:
            assert key in cfg, (key, recipe, cfg)
        policy = engine_config.validate_gemma_residency_config(cfg)
        assert policy["node"] == values[0], policy
    ec2 = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir, recipe="ec2",
    )
    assert engine_config.validate_gemma_target_route_config(ec2) == {
        "configured": True, "n": 32, "x": 32, "f": 1, "q": 4,
        "nodes": 32, "candidate_count": 32,
    }
    mac = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir, recipe="mac",
    )
    cpu_structure = (
        "SALT_GEMMA_COMPUTE_NODE", "SALT_GPU_TRUNK",
        "SALT_GPU_BOUNDED_WEIGHTS", "SALT_TARGET_GPU_PROGRAM",
        "SALT_TARGET_CPU_GRAPH", "SALT_TARGET_N_PARALLEL",
        "SALT_TARGET_MATRIX_FLOW", "SALT_TEXT_FINE_TOKEN",
        "SALT_TOKEN_CPU_FLOW", "SALT_TOKEN_CPU_GRAPH",
        "SALT_PREFILL_OPERATION_FLOW", "SALT_PREFILL_RETAIN_LAYERS",
        "SALT_PREFILL_DECODE_LOOKAHEAD", "SALT_PREFILL_SHARED_ARENAS",
        "SALT_PREFILL_GPU_ATTENTION", "SALT_M2_BATCH", "SALT_QKV_BATCH",
        "SALT_ATTN_THREADS", "SALT_TARGET_AREA_WORKERS",
        "SALT_EXPERT_MATRIX_WAVES",
    )
    for key in cpu_structure:
        assert ec2[key] == mac[key], (key, ec2[key], mac[key])
    assert engine_config.validate_gemma_target_route_config(mac) == {
        "configured": True, "n": 64, "x": 64, "f": 1, "q": 4,
        "nodes": 64, "candidate_count": 64,
    }
    metal = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir, recipe="mac-metal",
    )
    assert metal["SALT_GPU_TRUNK_LAYER_VIEW"] == "0", metal
    assert metal["SALT_GPU_TRUNK_SHARED_POOL"] == "1", metal
    assert metal["SALT_GEMMA_METAL_EXACT_CELLS"] == "0", metal
    assert metal["SALT_TARGET_AREA_WORKERS"] == "8", metal
    assert metal["SALT_PREFILL_B"] == "512", metal
    assert metal["SALT_TARGET_GPU_PROGRAM"] == "1", metal
    assert metal["SALT_GEMMA_GPU_KV_RING"] == "1", metal
    assert metal["SALT_GEMMA_MEMORY_LIMIT_GB"] == "4.5", metal
    assert engine_config.validate_gemma_target_route_config(metal) == {
        "configured": True, "n": 64, "x": 64, "f": 1, "q": 4,
        "nodes": 64, "candidate_count": 64,
    }
    assert metal["SALT_TARGET_KV_WARMUP_ROWS"] == "512", metal
    assert metal["SALT_TARGET_WARM_X"] == "4", metal
    spark = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir, recipe="spark",
    )
    assert spark["SALT_TARGET_GPU_PROGRAM"] == "1", spark
    assert engine_config.validate_gemma_target_route_config(spark) == {
        "configured": True, "n": 48, "x": 48, "f": 1, "q": 4,
        "nodes": 48, "candidate_count": 48,
    }
    spark_hmm = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir, recipe="spark-hmm",
    )
    assert spark_hmm["SALT_GPU_BOUNDED_WEIGHTS"] == "1", spark_hmm
    assert spark_hmm["SALT_GPU_WEIGHT_ADDRESSABILITY"] == "pageable", spark_hmm
    assert spark_hmm["SALT_CUDA_PAGEABLE_MMAP"] == "1", spark_hmm
    assert spark_hmm["SALT_EXPERT_PRELOAD"] == "0", spark_hmm
    assert spark_hmm["SALT_TARGET_GPU_PROGRAM"] == "1", spark_hmm
    assert spark_hmm["SALT_TEXT_FINE_TOKEN"] == "1", spark_hmm
    assert engine_config.validate_gemma_target_route_config(spark_hmm) == {
        "configured": True, "n": 48, "x": 48, "f": 1, "q": 4,
        "nodes": 48, "candidate_count": 48,
    }
    spark_hmm_vision = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir,
        recipe="spark-hmm-vision",
    )
    assert spark_hmm_vision["SALT_GEMMA_COMPUTE_NODE"] == "cpu", spark_hmm_vision
    assert spark_hmm_vision["SALT_GPU_TRUNK"] == "0", spark_hmm_vision
    assert spark_hmm_vision["SALT_CUDA_PAGEABLE_MMAP"] == "1", spark_hmm_vision
    assert spark_hmm_vision["SALT_GEMMA_VISION_CUDA_HMM"] == "1", spark_hmm_vision
    assert spark_hmm_vision["SALT_TARGET_GPU_PROGRAM"] == "0", spark_hmm_vision
    assert engine_config.validate_gemma_target_route_config(spark_hmm_vision) == {
        "configured": True, "n": 32, "x": 32, "f": 1, "q": 4,
        "nodes": 32, "candidate_count": 32,
    }
    rocm = load_engine_config(
        str(ROOT / "engine.config"), model_dir=model_dir, recipe="rocm",
    )
    assert rocm["SALT_GEMMA_COMPUTE_NODE"] == "full", rocm
    assert rocm["SALT_GPU_BOUNDED_WEIGHTS"] == "1", rocm
    assert rocm["SALT_GPU_TRUNK_LAYER_VIEW"] == "0", rocm
    assert rocm["SALT_GPU_TRUNK_SHARED_POOL"] == "1", rocm
    assert rocm["SALT_TARGET_GPU_PROGRAM"] == "1", rocm
    assert rocm["SALT_TEXT_FINE_TOKEN"] == "1", rocm
    gpu_structure = (
        "SALT_GEMMA_COMPUTE_NODE", "SALT_TARGET_GPU_PROGRAM",
        "SALT_TARGET_CPU_GRAPH",
        "SALT_TARGET_N_PARALLEL", "SALT_TARGET_MATRIX_FLOW",
        "SALT_TARGET_GPU_EXPERTS", "SALT_EXPERT_MATRIX_WAVES",
        "SALT_PREFILL_OPERATION_FLOW",
        "SALT_PREFILL_DECODE_LOOKAHEAD", "SALT_PREFILL_SHARED_ARENAS",
        "SALT_PREFILL_GPU_ATTENTION", "SALT_M2_BATCH", "SALT_QKV_BATCH",
        "SALT_ATTN_THREADS", "SALT_TARGET_AREA_WORKERS",
    )
    for key in gpu_structure:
        assert rocm[key] == spark[key], (key, rocm[key], spark[key])
    assert rocm["SALT_PREFILL_RETAIN_LAYERS"] == \
        spark_hmm["SALT_PREFILL_RETAIN_LAYERS"] == "0"
    assert rocm["SALT_GEMMA_GPU_KV_RING"] == "1"
    assert rocm["SALT_DECODE_GPU_ATTENTION"] == "1"
    assert engine_config.validate_gemma_target_route_config(rocm) == {
        "configured": True, "n": 104, "x": 16, "f": 1, "q": 4,
        "nodes": 104, "candidate_count": 104,
    }
    for invalid in ("", "../spark", "spark/x", "SPARK", "missing"):
        try:
            load_engine_config(
                str(ROOT / "engine.config"),
                model_dir=model_dir,
                recipe=invalid,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid platform recipe admitted: {invalid!r}")


def test_model_override():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "models" / "active"
        model_dir.mkdir(parents=True)
        (root / "engine.config").write_text(
            "MODEL_DIR=models/active\n"
            "SALT_M2_BATCH=0\n"
            "SALT_M2_BATCH_B=4096\n",
            encoding="utf-8",
        )
        (model_dir / "engine.config").write_text(
            "SALT_M2_BATCH=1\n"
            "SALT_M2_BATCH_B=384\n",
            encoding="utf-8",
        )

        cfg = load_engine_config(str(root / "engine.config"))
        assert cfg["SALT_M2_BATCH"] == "1", cfg
        assert cfg["SALT_M2_BATCH_B"] == "384", cfg

        os.remove(model_dir / "engine.config")
        cfg = load_engine_config(str(root / "engine.config"))
        assert cfg["SALT_M2_BATCH"] == "0", cfg
        assert "SALT_M2_BATCH_B" not in cfg, cfg


def test_qa_serial_proof_override():
    spec = importlib.util.spec_from_file_location(
        "salt_test_gemma4_qa", TOOLS / "gemma4-qa.py"
    )
    assert spec is not None and spec.loader is not None
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    operation = qa._native_environment()
    proof = qa._native_environment(serial_prefill_proof=True)
    assert operation["SALT_PREFILL_CHUNK"] == "1", operation
    assert proof["SALT_PREFILL_CHUNK"] == "0", proof
    for key in ("SALT_PREFILL_B", "SALT_M2_BATCH"):
        assert proof[key] == operation[key], (key, operation, proof)

    prior_n = os.environ.get("SALT_TARGET_ROUTE_N")
    prior_x = os.environ.get("SALT_TARGET_X")
    prior_f = os.environ.get("SALT_TARGET_ROUTE_F")
    prior_q = os.environ.get("SALT_TARGET_ROUTE_Q")
    prior_cpu_graph = os.environ.get("SALT_TARGET_CPU_GRAPH")
    try:
        os.environ["SALT_TARGET_ROUTE_N"] = "12"
        os.environ["SALT_TARGET_X"] = "8"
        os.environ["SALT_TARGET_ROUTE_F"] = "4"
        os.environ["SALT_TARGET_ROUTE_Q"] = "8"
        routed = qa._native_environment()
        assert routed["SALT_TARGET_ROUTE_N"] == "12", routed
        assert routed["SALT_TARGET_X"] == "8", routed
        assert routed["SALT_TARGET_ROUTE_F"] == "4", routed
        assert routed["SALT_TARGET_ROUTE_Q"] == "8", routed
        os.environ["SALT_TARGET_CPU_GRAPH"] = "1"
        graph = qa._native_environment()
        assert graph["SALT_TARGET_CPU_GRAPH"] == "1", graph
        os.environ["SALT_TARGET_CPU_GRAPH"] = "2"
        try:
            qa._native_environment()
            raise AssertionError("invalid CPU graph override accepted")
        except qa.QaError as exc:
            assert "must be 0 or 1" in str(exc), exc
        os.environ["SALT_TARGET_CPU_GRAPH"] = "1"
        del os.environ["SALT_TARGET_ROUTE_F"]
        try:
            qa._native_environment()
        except ValueError:
            pass
        else:
            raise AssertionError("partial target-route override was admitted")
    finally:
        if prior_n is None:
            os.environ.pop("SALT_TARGET_ROUTE_N", None)
        else:
            os.environ["SALT_TARGET_ROUTE_N"] = prior_n
        if prior_x is None:
            os.environ.pop("SALT_TARGET_X", None)
        else:
            os.environ["SALT_TARGET_X"] = prior_x
        if prior_f is None:
            os.environ.pop("SALT_TARGET_ROUTE_F", None)
        else:
            os.environ["SALT_TARGET_ROUTE_F"] = prior_f
        if prior_q is None:
            os.environ.pop("SALT_TARGET_ROUTE_Q", None)
        else:
            os.environ["SALT_TARGET_ROUTE_Q"] = prior_q
        if prior_cpu_graph is None:
            os.environ.pop("SALT_TARGET_CPU_GRAPH", None)
        else:
            os.environ["SALT_TARGET_CPU_GRAPH"] = prior_cpu_graph

    prior_retain = os.environ.get("SALT_PREFILL_RETAIN_LAYERS")
    try:
        os.environ["SALT_PREFILL_RETAIN_LAYERS"] = "0"
        release = qa._native_environment()
        assert release["SALT_PREFILL_RETAIN_LAYERS"] == "0", release
        os.environ["SALT_PREFILL_RETAIN_LAYERS"] = "2"
        try:
            qa._native_environment()
            raise AssertionError("invalid prefill retention override accepted")
        except qa.QaError as exc:
            assert "must be 0 or 1" in str(exc), exc
    finally:
        if prior_retain is None:
            os.environ.pop("SALT_PREFILL_RETAIN_LAYERS", None)
        else:
            os.environ["SALT_PREFILL_RETAIN_LAYERS"] = prior_retain

    prior_lookahead = os.environ.get("SALT_PREFILL_DECODE_LOOKAHEAD")
    try:
        os.environ["SALT_PREFILL_DECODE_LOOKAHEAD"] = "1"
        enabled = qa._native_environment()
        assert enabled["SALT_PREFILL_DECODE_LOOKAHEAD"] == "1", enabled
        os.environ["SALT_PREFILL_DECODE_LOOKAHEAD"] = "2"
        try:
            qa._native_environment()
            raise AssertionError("invalid phase lookahead override accepted")
        except qa.QaError as exc:
            assert "must be 0 or 1" in str(exc), exc
    finally:
        if prior_lookahead is None:
            os.environ.pop("SALT_PREFILL_DECODE_LOOKAHEAD", None)
        else:
            os.environ["SALT_PREFILL_DECODE_LOOKAHEAD"] = prior_lookahead

    prior_memory = os.environ.get("SALT_GEMMA_MEMORY_LIMIT_GB")
    try:
        os.environ["SALT_GEMMA_MEMORY_LIMIT_GB"] = "4.5"
        decimal_memory = qa._native_environment()
        assert decimal_memory["SALT_GEMMA_MEMORY_LIMIT_GB"] == "4.5", \
            decimal_memory
        os.environ["SALT_GEMMA_MEMORY_LIMIT_GB"] = "4.1234567890"
        try:
            qa._native_environment()
            raise AssertionError("over-precise memory-limit override accepted")
        except qa.QaError as exc:
            assert "decimal in [1, 128]" in str(exc), exc
    finally:
        if prior_memory is None:
            os.environ.pop("SALT_GEMMA_MEMORY_LIMIT_GB", None)
        else:
            os.environ["SALT_GEMMA_MEMORY_LIMIT_GB"] = prior_memory

    prior_fine_token = os.environ.get("SALT_TEXT_FINE_TOKEN")
    try:
        os.environ["SALT_TEXT_FINE_TOKEN"] = "1"
        fine = qa._native_environment()
        assert fine["SALT_TEXT_FINE_TOKEN"] == "1", fine
        os.environ["SALT_TEXT_FINE_TOKEN"] = "2"
        try:
            qa._native_environment()
            raise AssertionError("invalid fine-token override accepted")
        except qa.QaError as exc:
            assert "must be 0 or 1" in str(exc), exc
    finally:
        if prior_fine_token is None:
            os.environ.pop("SALT_TEXT_FINE_TOKEN", None)
        else:
            os.environ["SALT_TEXT_FINE_TOKEN"] = prior_fine_token

    proof_keys = ("SALT_GEMMA_METAL_EXACT_CELLS",
                  "SALT_GEMMA_LAYER_DIGESTS")
    prior_proof = {key: os.environ.get(key) for key in proof_keys}
    try:
        os.environ["SALT_GEMMA_METAL_EXACT_CELLS"] = "1"
        os.environ["SALT_GEMMA_LAYER_DIGESTS"] = "1"
        t4 = qa._native_environment()
        assert t4["SALT_GEMMA_METAL_EXACT_CELLS"] == "1", t4
        assert t4["SALT_GEMMA_LAYER_DIGESTS"] == "1", t4

        os.environ["SALT_GEMMA_METAL_EXACT_CELLS"] = "2"
        try:
            qa._native_environment()
            raise AssertionError("invalid T4 exact-cell override accepted")
        except qa.QaError as exc:
            assert "must be 0 or 1" in str(exc), exc

        os.environ["SALT_GEMMA_METAL_EXACT_CELLS"] = "0"
        try:
            qa._native_environment()
            raise AssertionError("layer digests accepted without T4 proof")
        except qa.QaError as exc:
            assert "require the T4" in str(exc), exc
    finally:
        for key, value in prior_proof.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    original = getattr(qa, "ENGINE_CONFIG")
    try:
        setattr(qa, "ENGINE_CONFIG", dict(
            original,
            SALT_EXPERT_BUDGET_GB="10",
            SALT_EXPERT_PRELOAD_LAYERS="0,18",
        ))
        os.environ["SALT_CACHE_LAYER_QUOTAS"] = "conflict"
        selective = qa._native_environment()
        quotas = selective["SALT_CACHE_LAYER_QUOTAS"].split(",")
        assert len(quotas) == 30, quotas
        assert quotas[0] == "128" and quotas[18] == "128", quotas
        assert all(value == "0" for index, value in enumerate(quotas)
                   if index not in (0, 18)), quotas
    finally:
        setattr(qa, "ENGINE_CONFIG", original)
        os.environ.pop("SALT_CACHE_LAYER_QUOTAS", None)


def test_server_reapplies_model_values():
    spec = importlib.util.spec_from_file_location(
        "salt_test_server", ROOT / "server" / "serve.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    server._ENV = {
        "SALT_GPU": "1",
        "SALT_GPU_TRUNK": "1",
        "SALT_GPU_MOE": "1",
        "SALT_GPU_TRUNK_PREFILL_ONLY": "1",
        "SALT_GPU_TRUNK_B": "999",
        "SALT_M2_BATCH": "host",
        "SALT_M2_BATCH_B": "999",
        "KEEP_ME": "yes",
    }
    server.CFG = {
        "SALT_PREFILL_CHUNK": "1",
        "SALT_PREFILL_B": "512",
        "SALT_M2_BATCH": "1",
        "SALT_M2_BATCH_B": "384",
        "SALT_CACHE_MODE": "zerocopy",
        "SALT_GPU_TRUNK_B": "200",
        "SALT_GPU_PREFILL_B": "1024",
        "SALT_GPU_B_QKV": "1024",
        "SALT_GPU_B_Z": "1024",
        "SALT_GPU_B_O": "1024",
    }

    env = server.model_engine_env(gpu_prefill=False)
    assert env["SALT_M2_BATCH"] == "1", env
    assert env["SALT_M2_BATCH_B"] == "384", env
    assert "SALT_GPU" not in env, env
    assert "SALT_GPU_TRUNK" not in env, env
    assert "SALT_GPU_MOE" not in env, env
    assert "SALT_GPU_TRUNK_PREFILL_ONLY" not in env, env
    assert "SALT_GPU_TRUNK_B" not in env, env
    assert env["KEEP_ME"] == "yes", env

    env = server.model_engine_env(gpu_prefill=True)
    assert env["SALT_GPU"] == "1", env
    assert env["SALT_GPU_RESIDENT"] == "1", env
    assert env["SALT_GPU_TRUNK"] == "1", env
    assert env["SALT_GPU_MOE"] == "0", env
    assert env["SALT_GPU_OVERLAP"] == "0", env
    assert env["SALT_GPU_TRUNK_PREFILL_ONLY"] == "1", env
    assert env["SALT_GPU_TRUNK_B"] == "200", env
    assert env["SALT_GPU_PREFILL_B"] == "1024", env
    assert env["KEEP_ME"] == "yes", env


def test_setup_dev_env_exports_model_values():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        model_dir = root / "models" / "active"
        test_dir = root / "tools" / "test"
        track = Path(tmp) / "track"
        model_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        track.mkdir()
        shutil.copy2(TOOLS / "engine_config.py", root / "tools" / "engine_config.py")
        (root / "engine.config").write_text(
            "MODEL_DIR=models/active\n", encoding="utf-8"
        )
        (model_dir / "engine.config").write_text(
            "SALT_PREFILL_CHUNK=1\n"
            "SALT_PREFILL_B=64\n"
            "SALT_M2_BATCH=1\n"
            "SALT_M2_BATCH_B=37\n"
            "SALT_CACHE_MODE=arena\n",
            encoding="utf-8",
        )
        (root / "Makefile").write_text("salt:\n\t@true\n", encoding="utf-8")
        (track / "trunk.bin").write_bytes(b"fixture")
        (test_dir / "kv-gate.sh").write_text(
            "#!/usr/bin/env bash\necho 'kv-gate: PASS'\n", encoding="utf-8"
        )
        (test_dir / "phase-gate.sh").write_text(
            "#!/usr/bin/env bash\necho 'PHASE GATE: PASS'\n", encoding="utf-8"
        )

        env = dict(os.environ)
        for key in (
            "SALT_PREFILL_CHUNK",
            "SALT_PREFILL_B",
            "SALT_M2_BATCH",
            "SALT_M2_BATCH_B",
            "SALT_CACHE_MODE",
        ):
            env.pop(key, None)
        env["SALT_REPO"] = str(root)
        env["SALT_TRACK"] = str(track)
        probe = (
            'source "$1" >/dev/null; '
            'printf "%s|%s|%s|%s|%s" "$SALT_PREFILL_CHUNK" '
            '"$SALT_PREFILL_B" "$SALT_M2_BATCH" "$SALT_M2_BATCH_B" '
            '"$SALT_CACHE_MODE"'
        )
        result = subprocess.run(
            ["bash", "-c", probe, "bash", str(TOOLS / "setup_dev_env.sh")],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "1|64|1|37|arena", result.stdout


def main():
    test_repo_blueprint_ownership()
    test_gemma_residency_policy()
    test_gemma_target_route_policy()
    test_gemma_platform_recipes()
    test_model_override()
    test_qa_serial_proof_override()
    test_server_reapplies_model_values()
    test_setup_dev_env_exports_model_values()
    print("engine-config model ownership: PASS")


if __name__ == "__main__":
    main()
