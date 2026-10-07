#!/usr/bin/env python3
"""Compiler-free source contracts; NOT a Metal execution/exactness test.

The capacity predicate is evaluated from its actual C source with explicit
Darwin LP64 widths. Shader specialization, startup ownership and replay lifetime
are checked structurally. Run with --source PATH to check a prior source file.
"""

import argparse
import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/gpu_metal.m"


def function(source, name):
    match = re.search(
        rf"(?m)^(?:static )?int {name}\([^;]*?\)\s*\{{.*?^\}}",
        source, re.S,
    )
    if not match:
        raise AssertionError(f"missing function contract: {name}")
    return match.group()


def embedded_batch(source):
    start = source.index("static const char *_batch_src2 =")
    end = source.index("\nstatic id<MTLComputePipelineState> _bpso2;", start)
    return "".join(ast.literal_eval(token) for token in
                   re.findall(r'"(?:[^"\\]|\\.)*"', source[start:end]))


def capacity_valid(source, capacity, logical_capacity, max_buffer_length,
                   size_max=(1 << 64) - 1, nsuint_max=(1 << 64) - 1):
    """Interpret only the arithmetic rejection predicate, not Objective-C."""
    body = function(source, "metal_selected_capacity_valid")
    match = re.search(r"if \((.*?)\)\s*return 0;\s*return 1;", body, re.S)
    if not match:
        raise AssertionError("capacity admission must fail closed")
    expression = match[1].replace("(size_t)", "")
    sizes = {"MetalSelectedCacheResource": 32, "uint64_t": 8,
             "int32_t": 4, "uint32_t": 4}
    expression = re.sub(r"sizeof\((\w+)\)",
                        lambda m: str(sizes[m[1]]), expression)
    expression = expression.replace("||", " or ").replace("&&", " and ")
    expression = " ".join(expression.replace("/", "//").split())
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BoolOp, ast.Or, ast.And, ast.Compare,
               ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
               ast.BinOp, ast.FloorDiv, ast.Name, ast.Load, ast.Constant)
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        raise AssertionError("unrecognized C capacity predicate")
    values = dict(capacity=capacity, logical_capacity=logical_capacity,
                  max_buffer_length=max_buffer_length, SIZE_MAX=size_max,
                  NSUIntegerMax=nsuint_max, INT32_MAX=(1 << 31) - 1)
    return not eval(compile(tree, "<capacity source predicate>", "eval"),
                    {"__builtins__": {}}, values)


class SelectedCapacityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.shader = embedded_batch(cls.source)

    def test_existing_two_gb_cell_and_capacity_boundaries(self):
        self.assertEqual(2_000_000_000 // 3_345_408, 597)
        for capacity in (1, 128, 512, 513, 597, 1024, 3840):
            with self.subTest(capacity=capacity):
                self.assertTrue(capacity_valid(self.source, capacity, 3840,
                                               256 * 1024 * 1024))
        for capacity, logical in ((0, 3840), (-1, 3840), (597, 0),
                                  (597, -1), (597, 1 << 31)):
            with self.subTest(capacity=capacity, logical=logical):
                self.assertFalse(capacity_valid(self.source, capacity, logical,
                                                256 * 1024 * 1024))

    def test_overflow_and_device_buffer_bounds(self):
        # Exact boundaries for the argument-pointer lower bound and both tables.
        self.assertTrue(capacity_valid(self.source, 597, 1, 597 * 8))
        self.assertFalse(capacity_valid(self.source, 597, 1, 597 * 8 - 1))
        self.assertTrue(capacity_valid(self.source, 1, 3840, 3840 * 4))
        self.assertFalse(capacity_valid(self.source, 1, 3840, 3840 * 4 - 1))
        self.assertFalse(capacity_valid(self.source, 597, 3840, 0))
        self.assertFalse(capacity_valid(self.source, 597, 3840, 1 << 30,
                                        size_max=597 * 32 - 1))
        self.assertFalse(capacity_valid(self.source, 597, 3840, 1 << 30,
                                        nsuint_max=3840 * 4 - 1))
        match = re.search(r"typedef struct MetalSelectedCacheResource \{"
                          r"(.*?)\} MetalSelectedCacheResource;",
                          self.source, re.S)
        self.assertIsNotNone(match)
        assert match is not None
        resource = match.group(1)
        self.assertEqual(re.findall(r"\b(map|base|nbytes|logical_resource_id)\b",
                                    resource),
                         ["map", "base", "nbytes", "logical_resource_id"])

    def test_one_specialization_for_array_pipelines_and_encoders(self):
        body = function(self.source, "metal_selected_resources_prepare")
        self.assertIn('@"SALT_METAL_SELECTED_CAPACITY": @(capacity)', body)
        self.assertIn("array<device const uchar *,SALT_METAL_SELECTED_CAPACITY>",
                      self.shader)
        self.assertNotIn("*,512>", self.shader)
        self.assertIn("newLibraryWithSource:", body)
        self.assertIn("metal_exact_math(options)", body)
        self.assertNotIn("metal_library_for_source(", body)
        for name in ("q4selected", "q4selectedragged"):
            self.assertIn(f'[library newFunctionWithName:@"{name}"]', body)
        self.assertEqual(body.count("[function newArgumentEncoderWithBufferIndex:0]"), 2)
        for argument in ("function", "ragged_function"):
            self.assertIn(f"newComputePipelineStateWithFunction:{argument}", body)
        for guard in ("argument_bytes == 0", "argument_bytes > [_dev maxBufferLength]",
                      "argument_bytes != [wave_encoder encodedLength]"):
            self.assertIn(guard, body)
        self.assertLess(body.index("argument_bytes > [_dev maxBufferLength]"),
                        body.index("newBufferWithLength:argument_bytes"))

    def test_generic_startup_cannot_freeze_or_replace_selected_layout(self):
        for name in ("salt_gpu_init", "metal_session_buffers_prepare",
                     "batch_pipeline_prepare"):
            body = function(self.source, name)
            self.assertNotIn("metal_selected_resources_prepare(", body)
            self.assertNotIn("_bselected_pso", body)
            self.assertNotIn("_wave_selected_args", body)
        start = self.shader.index("#ifdef SALT_METAL_SELECTED_CAPACITY")
        end = self.shader.index("#endif", start)
        for name in ("q4selected", "q4selectedragged"):
            self.assertIn(f"kernel void {name}(", self.shader[start:end])
        self.assertNotIn("q4warpbatch", self.shader[start:end])
        self.assertEqual(len(re.findall(r"\bmetal_selected_resources_prepare\(",
                                        self.source)), 2)  # definition + startup

    def test_startup_admits_hardware_and_publishes_once(self):
        body = function(self.source, "salt_gpu_selected_resources_prepare")
        private = function(self.source, "metal_selected_resources_prepare")
        for text in ("!metal_selected_capacity_valid(capacity, logical_capacity,",
                     "[_dev argumentBuffersSupport] != MTLArgumentBuffersTier2",
                     "capacity == _selected_cache_capacity",
                     "capacity == _selected_argument_capacity",
                     "logical_capacity == _selected_logical_capacity"):
            self.assertIn(text, body)
        self.assertLess(body.index("if (_selected_cache_resources)"),
                        body.index("batch_pipeline_prepare()"))
        self.assertLess(body.index("metal_selected_resources_prepare(capacity)"),
                        body.index("_selected_cache_capacity = capacity"))
        self.assertIn("return capacity == _selected_argument_capacity ? 0 : -1;",
                      private)
        self.assertNotIn("metal_release(_", private)
        self.assertLess(private.index("if (!arguments || !wave_arguments"),
                        private.index("_bselected_pso = pipeline"))
        self.assertLess(private.index("_wave_selected_args = wave_arguments"),
                        private.index("_selected_argument_capacity = capacity"))

    def test_shader_checks_specialized_bound_before_dereference(self):
        for name in ("q4selected", "q4selectedragged"):
            start = self.shader.index(f"kernel void {name}(")
            end = self.shader.index("\n}\n", start)
            body = self.shader[start:end]
            self.assertLess(body.index("if(slot>=SALT_METAL_SELECTED_CAPACITY)"),
                            body.index("resources.resource[slot]"))
            self.assertIn("atomic_store_explicit(status,2u", body)

    def test_noncache_helpers_do_not_allocate_a_fallback_table(self):
        # The selected chain already requires resource_slot >= 0 and a cache
        # seat; calls without prepare fail rather than infer a table from count.
        body = function(self.source, "salt_gpu_q4_moe_chain_selected")
        self.assertIn("!_selected_cache_resources", body)
        self.assertIn("!_bselected_pso || !_wave_selected_args", body)
        self.assertLess(body.index("!_selected_cache_resources"),
                        body.index("batch_pipeline_prepare()"))
        self.assertIn("entry->resource_slot >= _selected_cache_capacity", body)
        self.assertNotIn("metal_selected_resources_prepare(", body)
        self.assertNotIn("newBuffer", body)

    def test_cache_authority_bounds_and_replay_lifetime(self):
        for name in ("salt_gpu_selected_resource_bind",
                     "salt_gpu_selected_resource_unbind",
                     "metal_text_resource_resume"):
            body = function(self.source, name)
            self.assertIn("slot < 0", body)
            self.assertIn("slot >= _selected_cache_capacity", body)
            self.assertIn("_selected_logical_capacity", body)
            self.assertNotIn("metal_selected_resources_prepare(", body)
        bind = function(self.source, "salt_gpu_selected_resource_bind")
        self.assertIn("newBufferWithBytesNoCopy", bind)
        self.assertIn("_selected_logical_slot_host[logical_resource_id] = slot", bind)
        unbind = function(self.source, "salt_gpu_selected_resource_unbind")
        self.assertLess(unbind.index("salt_gpu_sync()"),
                        unbind.index("setBuffer:nil"))
        replay = function(self.source, "metal_text_encode_expert_chain")
        self.assertIn("metal_selected_encode_ragged(encoder, _wave_selected_args", replay)
        self.assertIn("payload = _selected_logical_payload_host[logical]", replay)
        self.assertNotIn("newBuffer", replay)
        teardown = function(self.source, "salt_gpu_free_impl")
        self.assertLess(teardown.index("salt_gpu_sync()"),
                        teardown.index("metal_release(_wave_selected_args)"))
        self.assertLess(teardown.index("refusing teardown with bound selected cache slot"),
                        teardown.index("_selected_argument_capacity = 0"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[__file__, *remaining], verbosity=2)