#!/usr/bin/env python3
"""No-model persistent native-session V2 contract for Gemma 4."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "server" / "model" / "gemma4_backend.py"

spec = importlib.util.spec_from_file_location("salt_gemma4_persistent_test", BACKEND)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> None:
    controls = module.resolve_controls({}, server_kv_budget_gb=1.0)
    assert (controls.context_tokens, controls.output_tokens) == (512, 128)
    assert controls.kv_bytes == 230_686_720
    compatibility = module.KV_COMPATIBILITY.hexdigest
    build_identity = "f" * 64

    child = textwrap.dedent(rf'''
        import sys
        position = 0
        turn = 0
        expected = 1
        compatibility = "{compatibility}"
        build_identity = "{build_identity}"
        state = "0" * 64
        mindset = "0" * 64
        facts = "e" * 64
        sys.stdout.write(
            "GEMMA4_SERVER_READY_V3 protocol=3 session_epoch=1 turn_id=0 "
            "history_turns=0 model_context_limit=262144 "
            "context=512 output_limit=128 response_limit_bytes=9216 "
            "journal_capacity=256 prefill_chunk_tokens=512 "
            "position=0 shared_position=0 kv_capacity_bytes=230686720 "
            "kv_sliding_window=1024 kv_sliding_layers=25 kv_full_layers=5 "
            "expert_budget_bytes={module.GEMMA_EXPERT_BUDGET_BYTES} "
            "expert_cache_mode=bounded-mmap-zero-copy "
            f"compatibility_sha256={{compatibility}} "
            f"build_identity_sha256={{build_identity}} state_sha256={{state}} "
            f"mindset_mode=prefix mindset_end=0 "
            f"mindset_sha256={{mindset}} facts_sha256={{facts}} facts_rows=0 "
            "runtime_ready=true\n"
        )
        sys.stdout.flush()
        while True:
            header = sys.stdin.buffer.readline()
            if not header:
                break
            if header == b"SALT_GEMMA4_DRAIN_V2\n":
                sys.stdout.write(
                    f"GEMMA4_SERVER_BYE_V2 session_epoch=1 turn_id={{turn}} "
                    f"history_turns={{turn}} "
                    f"position={{position}} state_sha256={{state}} "
                    f"mindset_mode=prefix mindset_end=0 "
                    f"mindset_sha256={{mindset}} facts_sha256={{facts}} "
                    f"facts_rows={{position}} build_identity_sha256={{build_identity}} "
                    "runtime_ready=true\n"
                )
                sys.stdout.flush()
                break
            parts = header.decode("ascii").strip().split()
            assert parts[0] == "SALT_GEMMA4_TURN_V3"
            fields = dict(item.split("=", 1) for item in parts[1:])
            request_id = int(fields["request"])
            epoch = int(fields["session_epoch"])
            expected_turn = int(fields["turn_id"])
            expected_position = int(fields["position"])
            prompt_bytes = int(fields["prompt_bytes"])
            image_bytes = int(fields["image_path_bytes"])
            output_limit = int(fields["output_limit"])
            client_request_sha256 = fields["client_request_sha256"]
            request_sha256 = fields["request_sha256"]
            assert request_id == expected and epoch == 1
            assert expected_turn == turn and expected_position == position
            prompt = sys.stdin.buffer.read(prompt_bytes)
            image = sys.stdin.buffer.read(image_bytes)
            assert sys.stdin.buffer.read(1) == b"\n"
            assert prompt and not image
            response = b"hello" if request_id == 1 else b"orchid"
            prompt_tokens = 3 if request_id == 1 else 2
            prompt_ids = ",".join(str(10 + i) for i in range(prompt_tokens))
            before = position
            sys.stdout.write(
                f"GEMMA4_SERVER_START_V3 request={{request_id}} session_epoch=1 "
                f"turn_id={{turn}} position={{before}} "
                f"prompt_tokens={{prompt_tokens}} image_tokens=0 "
                f"output_limit={{output_limit}} "
                "runtime_ready=true\n"
            )
            sys.stdout.write(
                f"GEMMA4_SERVER_TOKEN_V2 request={{request_id}} session_epoch=1 "
                f"turn_id={{turn}} seq=1 token_id=818 stop=0 "
                f"bytes={{len(response)}}\n"
            )
            sys.stdout.buffer.write(response + b"\n")
            sys.stdout.write(
                f"GEMMA4_SERVER_TOKEN_V2 request={{request_id}} session_epoch=1 "
                f"turn_id={{turn}} seq=2 token_id=106 stop=1 "
                f"bytes={{len(response)}}\n"
            )
            sys.stdout.buffer.write(response + b"\n")
            sys.stdout.write(
                f"GEMMA4_SERVER_DPR_WATERFALL_V1 request={{request_id}} mode=off "
                "prefill_node_ns=0 prefill_lookup_ns=0 prefill_restore_ns=0 "
                "prefill_ordinary_ns=1 decode_node_ns=0 decode_lookup_ns=0 "
                "decode_parent_restore_ns=0 decode_verify_ns=0 "
                "decode_ordinary_ns=1 prefill_hits=0 prefill_cached_tokens=0 "
                "decode_cycles=0 parent_hits=0 nomogram_hits=0 decode_misses=0 "
                "proposed_tokens=0 accepted_tokens=0 rejection_count=0 "
                "runtime_ready=true\n"
            )
            position += prompt_tokens + 2
            turn += 1
            state = str(turn) * 64
            sys.stdout.write(
                f"GEMMA4_SERVER_COMMIT_V3 request={{request_id}} "
                f"original_request={{request_id}} session_epoch=1 "
                f"turn_id={{turn}} history_turns={{turn}} finish_reason=stop "
                f"prompt_tokens={{prompt_tokens}} prompt_ids={{prompt_ids}} "
                f"image_tokens=0 output_limit={{output_limit}} output_steps=2 "
                f"stop_token=106 synthetic_close=0 position_before={{before}} "
                f"position_after={{position}} session_continuable=1 "
                f"response_bytes={{len(response)}} output_ids=818,106 "
                f"state_sha256={{state}} mindset_mode=prefix "
                f"mindset_end=0 mindset_sha256={{mindset}} facts_sha256={{facts}} "
                f"facts_rows={{position}} build_identity_sha256={{build_identity}} "
                f"client_request_sha256={{client_request_sha256}} "
                f"request_sha256={{request_sha256}} replayed=0 "
                "runtime_ready=true\n"
            )
            sys.stdout.buffer.write(response + b"\n")
            sys.stdout.flush()
            expected += 1
    ''')

    deltas: list[tuple[str, str]] = []
    engine = module.PersistentGemmaEngine(
        [sys.executable, "-u", "-c", child], timeout=3.0,
        env=module.Gemma4Backend._clean_env(),
        compatibility_sha256=compatibility,
        build_identity_sha256=build_identity,
    )
    pid = engine.process.pid
    try:
        first = engine.request(
            "first", None, expected_epoch=1, expected_turn=0,
            expected_history_turns=0,
            expected_position=0, client_request_sha256="a" * 64,
            request_sha256="b" * 64, on_start=lambda: None,
            on_delta=lambda kind, value: deltas.append((kind, value)),
            proof_state=True,
        )
        second = engine.request(
            "second", None, expected_epoch=1, expected_turn=1,
            expected_history_turns=1,
            expected_position=5, client_request_sha256="c" * 64,
            request_sha256="d" * 64, on_start=lambda: None,
            on_delta=lambda kind, value: deltas.append((kind, value)),
            output_limit=2,
            proof_state=True,
        )
        assert engine.process.pid == pid and engine.process.poll() is None
        assert (first.content, first.kv_loaded_tokens, first.kv_saved_tokens,
                first.turn_id) == ("hello", 0, 5, 1)
        assert (second.content, second.kv_loaded_tokens, second.kv_saved_tokens,
                second.turn_id) == ("orchid", 5, 9, 2)
        assert deltas == [("content", "hello"), ("content", "orchid")]
    finally:
        engine.stop()
    assert engine.process.poll() == 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError(f"persistent child was not reaped: {pid}")

    print("gemma4 persistent session V2 contract: PASS")


if __name__ == "__main__":
    main()
