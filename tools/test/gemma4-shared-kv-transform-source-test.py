#!/usr/bin/env python3
"""Static contract for Gemma 4 shared K=V transform ordering."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = (ROOT / "models/gemma4-26b-a4b/src/gemma4_text.c").read_text()
CPU = (ROOT / "src/text_verify_cpu.c").read_text()
METAL = (ROOT / "src/gpu_metal.m").read_text()
CUDA = (ROOT / "src/gpu_cuda.cu").read_text()
HIP = (ROOT / "src/gpu_hip.cpp").read_text()

assert "G4KVC006 stores the normalized V" in MODEL
export_start = MODEL.index("int salt_gemma4_text_kv_export_stream(")
export_end = MODEL.index("int salt_gemma4_text_kv_import_stream_sha256(", export_start)
export = MODEL[export_start:export_end]
assert "segments == 1 || segment == 1" in export
assert "g4_value_row(layer, position)" in export

cpu_start = CPU.index("case SALT_TEXT_CELL_ATTENTION_TRANSFORM:")
cpu_end = CPU.index("case SALT_TEXT_CELL_ATTENTION_BODY:", cpu_start)
cpu = CPU[cpu_start:cpu_end]
assert cpu.index("memcpy(values") < cpu.index("cpu_rmsnorm(key")

metal_start = METAL.index("static int metal_text_encode_attention_transform(")
metal_end = METAL.index("static int metal_text_encode_attention_body(", metal_start)
metal = METAL[metal_start:metal_end]
assert "[blit copyFromBuffer:canonical sourceOffset:layout->keys" in metal
assert "toBuffer:canonical destinationOffset:layout->values" in metal
assert metal.index("copyFromBuffer:canonical") < metal.index("computeCommandEncoder")
assert "(uint32_t)(attention->shared_kv_projection != 0)" not in metal

cuda_start = CUDA.index("case SALT_TEXT_CELL_ATTENTION_TRANSFORM:")
cuda_end = CUDA.index("case SALT_TEXT_CELL_ATTENTION_BODY:", cuda_start)
cuda = CUDA[cuda_start:cuda_end]
assert cuda.index("cuda_text_copy<<<") < cuda.index("cuda_text_attention_transform<<<")
assert "program->descriptor->norm_epsilon,\n            0," in cuda

hip_start = HIP.index("case SALT_TEXT_CELL_ATTENTION_TRANSFORM:")
hip_end = HIP.index("case SALT_TEXT_CELL_ATTENTION_BODY:", hip_start)
hip = HIP[hip_start:hip_end]
assert hip.index("hip_text_copy<<<") < hip.index("hip_text_attention_transform<<<")
assert "program->descriptor->norm_epsilon,\n            0," in hip

print("GEMMA4_SHARED_KV_TRANSFORM_SOURCE_TEST_PASS")
