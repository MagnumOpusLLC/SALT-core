#!/usr/bin/env python3
"""engine_config.py -- load engine.config (repo root) for the python
tools (dbg-reffwd.py, dbg-logitscmp.py, salt-chat.py, ...).

Usage:
    from engine_config import load_engine_config
    cfg = load_engine_config()          # dict[str, str], defaults on miss
    cfg.get('MODEL_TRUNK', '/tmp/q36-trunk')
"""
import os
from decimal import Decimal, InvalidOperation
import re
import sys

DEFAULTS = {
    "MODEL_TRUNK": "/tmp/q36-trunk",
    "MODEL_POOL": "/tmp/q36-pool",
    "MODEL_TOKENIZER": "/tmp/q36-trunk/tokenizer.json",
    "REF_TRUNK": "/tmp/q35-trunk",
    "REF_POOL": "/tmp/q35-pool",
    "SALT_BIN": "salt",
    "CHAT_GEN": "60",
    "CHAT_THREADS": "8",
    "CHAT_KV_BUDGET_GB": "1.0",
    "CHAT_FIFO": "/tmp/salt-chat.fifo",
    "SALT_GREEDY": "1",
    "SALT_REP_PENALTY": "1.4",
    "SALT_FREQ_PENALTY": "1.0",
    "SALT_PRESENCE_PENALTY": "0.0",
    "SALT_TOPK": "20",
    "SALT_TOP_P": "0.95",
    "SALT_TEMP": "1.0",
}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _merge_config(path, cfg):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    except FileNotFoundError:
        pass


def load_engine_config(path=None, model_dir=None, recipe=None):
    path = os.path.abspath(path or os.path.join(repo_root(), "engine.config"))
    cfg = dict(DEFAULTS)
    _merge_config(path, cfg)

    # The root file selects a model blueprint. Model-specific compute
    # thresholds override shared runner defaults; they must not become
    # universal engine constants merely because one model measured them.
    selected_model_dir = model_dir if model_dir is not None else cfg.get("MODEL_DIR")
    if selected_model_dir:
        # These root values are Qwen-only controls/metadata, not generic model
        # defaults. Prevent them from leaking across an explicit model-dir
        # selection; a model may re-declare only values with a live consumer.
        for key in (
                "MODEL", "QUANT", "LAYERS", "SALT_GPU",
                "SALT_GPU_RESIDENT", "SALT_GPU_MOE", "SALT_GPU_OVERLAP",
                "SALT_GPU_TRUNK_PREFILL_ONLY", "SALT_M2_BATCH_B",
                "SALT_CACHE_MODE"):
            cfg.pop(key, None)
        selected_model_dir = os.fspath(selected_model_dir)
        if not os.path.isabs(selected_model_dir):
            selected_model_dir = os.path.join(
                os.path.dirname(path), selected_model_dir,
            )
        _merge_config(os.path.join(selected_model_dir, "engine.config"), cfg)
        if recipe is not None:
            if not isinstance(recipe, str) or not recipe or any(
                    char not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                    for char in recipe):
                raise ValueError("invalid engine platform recipe")
            recipe_path = os.path.join(
                selected_model_dir, "configs", recipe, "engine.config",
            )
            if not os.path.isfile(recipe_path):
                raise ValueError(f"unknown engine platform recipe: {recipe}")
            _merge_config(recipe_path, cfg)
            if cfg.get("SALT_GEMMA_PLATFORM_RECIPE") != recipe:
                raise ValueError("engine platform recipe identity mismatch")
        if model_dir is not None:
            cfg["MODEL_DIR"] = selected_model_dir
    return cfg


def physical_memory_bytes():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int) or \
            pages < 1 or page_size < 1:
        return None
    return pages * page_size


def validate_gemma_residency_config(cfg, *, require_host=False):
    def integer(name, minimum, maximum):
        text = cfg.get(name)
        if not isinstance(text, str) or not text or not text.isdecimal():
            raise ValueError(f"invalid {name}")
        value = int(text)
        if value < minimum or value > maximum:
            raise ValueError(f"{name} outside {minimum}..{maximum}")
        return value

    def decimal_value(name, minimum, maximum):
        text = cfg.get(name)
        try:
            if not isinstance(text, str) or not re.fullmatch(
                    r"[0-9]+(?:\.[0-9]{1,9})?", text):
                raise InvalidOperation
            value = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"invalid {name}") from exc
        if value < minimum or value > maximum:
            raise ValueError(f"{name} outside {minimum}..{maximum}")
        return value

    node = cfg.get("SALT_GEMMA_COMPUTE_NODE")
    if node not in ("cpu", "mixed", "full"):
        raise ValueError("invalid SALT_GEMMA_COMPUTE_NODE")
    budget_gb = integer("SALT_EXPERT_BUDGET_GB", 1, 13)
    preload = integer("SALT_EXPERT_PRELOAD", 0, 1)
    minimum_ram_gb = integer("SALT_FULL_MIN_RAM_GB", 1, 256)
    memory_limit_gb = decimal_value("SALT_GEMMA_MEMORY_LIMIT_GB", 1, 128)
    layer_text = cfg.get("SALT_EXPERT_PRELOAD_LAYERS")
    if layer_text == "none":
        preload_layers = ()
    elif not isinstance(layer_text, str) or not layer_text:
        raise ValueError("invalid SALT_EXPERT_PRELOAD_LAYERS")
    else:
        parsed = []
        seen = set()
        for item in layer_text.split(","):
            if not item.isdecimal() or str(int(item)) != item:
                raise ValueError("invalid SALT_EXPERT_PRELOAD_LAYERS")
            layer = int(item)
            if layer < 0 or layer >= 30 or layer in seen:
                raise ValueError("invalid SALT_EXPERT_PRELOAD_LAYERS")
            seen.add(layer)
            parsed.append(layer)
        preload_layers = tuple(parsed)
    if preload and budget_gb < 13:
        raise ValueError("full expert preload requires at least 13 GB")
    if preload and preload_layers:
        raise ValueError("global and selective expert preload are mutually exclusive")
    budget_slots = min(30 * 128, budget_gb * 1_000_000_000 // 3_345_408)
    required_slots = len(preload_layers) * 128 + (128 if preload_layers else 0)
    if required_slots > budget_slots:
        raise ValueError("selective expert preload exceeds budget/shared floor")
    layer_quotas = None
    if preload_layers:
        selected = set(preload_layers)
        layer_quotas = ",".join(
            "128" if layer in selected else "0" for layer in range(30)
        )
    layer_mask = sum(1 << layer for layer in preload_layers)
    physical = physical_memory_bytes()
    if preload and require_host and (
            physical is None or physical < minimum_ram_gb * 1_000_000_000):
        raise ValueError("host RAM is below SALT_FULL_MIN_RAM_GB")
    return {
        "node": node,
        "budget_gb": budget_gb,
        "preload": preload,
        "preload_layers": preload_layers,
        "preload_layer_mask": layer_mask,
        "preload_required_slots": required_slots,
        "layer_quotas": layer_quotas,
        "minimum_ram_gb": minimum_ram_gb,
        "memory_limit_gb": memory_limit_gb,
        "physical_memory_bytes": physical,
    }


def validate_gemma_target_route_config(cfg):
    n_text = cfg.get("SALT_TARGET_ROUTE_N")
    x_text = cfg.get("SALT_TARGET_X")
    f_text = cfg.get("SALT_TARGET_ROUTE_F")
    q_text = cfg.get("SALT_TARGET_ROUTE_Q")
    w_text = cfg.get("SALT_TARGET_AREA_WORKERS", "8")
    if n_text is None and x_text is None and f_text is None and q_text is None:
        return {"configured": False, "n": 0, "x": 0, "f": 0, "q": 0,
                "nodes": 0, "candidate_count": 0}
    if not isinstance(n_text, str) or not n_text.isdecimal() or \
            not isinstance(x_text, str) or not x_text.isdecimal() or \
            not isinstance(f_text, str) or not f_text.isdecimal() or \
            not isinstance(q_text, str) or not q_text.isdecimal() or \
            not isinstance(w_text, str) or not w_text.isdecimal():
        raise ValueError("NFQ N/F/Q, TARGET X, and W must be canonical decimals")
    n = int(n_text)
    x = int(x_text)
    f = int(f_text)
    q = int(q_text)
    w = int(w_text)
    if str(n) != n_text or str(x) != x_text or str(f) != f_text or str(q) != q_text or \
            str(w) != w_text:
        raise ValueError("NFQ N/F/Q, TARGET X, and W must use canonical decimal form")
    if n < 1 or n > 128:
        raise ValueError("SALT_TARGET_ROUTE_N outside 1..128")
    if x < 1 or x > 128:
        raise ValueError("SALT_TARGET_X outside 1..128")
    if f < 1 or f > 8:
        raise ValueError("SALT_TARGET_ROUTE_F outside 1..8")
    if q < 1 or q > 8:
        raise ValueError("SALT_TARGET_ROUTE_Q outside 1..8")
    if w < 1 or w > 20:
        raise ValueError("SALT_TARGET_AREA_WORKERS outside 1..20")
    if n * f > 128:
        raise ValueError("target route N*F exceeds 128-node capacity")
    if x * f > 128:
        raise ValueError("TARGET X*F exceeds 128-row capacity")
    if n * f < w or n * f % w:
        raise ValueError("target route N*F must equal an integral multiple of W")
    return {"configured": True, "n": n, "x": x, "f": f, "q": q,
            "nodes": n * f, "candidate_count": n * f}


if __name__ == "__main__":
    config = load_engine_config()
    if len(sys.argv) > 1:
        for key in sys.argv[1:]:
            print(config.get(key, ""))
    else:
        for k, v in sorted(config.items()):
            print(f"{k}={v}")
