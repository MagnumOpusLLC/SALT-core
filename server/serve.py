#!/usr/bin/env python3
"""serve.py -- OpenAI-compatible HTTP API for Salt model backends.

The default Qwen backend uses the persistent disk-streaming engine.  The
Gemma 4 backend uses the authenticated persistent native session and exposes
the qualified runtime as ``runtime_ready=true``.

Exposes either backend as an OpenAI-shaped server:

  GET  /v1/models
  POST /v1/chat/completions   (stream=true -> SSE)
  POST /v1/completions

Design notes (engine constraints):
  * SINGLE-FLIGHT: two concurrent engine runs contaminate each other
    (measured 0.27 s/tok under contention vs 0.14 clean -- the fetch
    pool + shared disk bandwidth thrash). All requests are serialized
    behind one lock; a queued request waits.
  * One persistent `salt --serve-fifo` process is started eagerly. Serialized
    requests use its NUL-framed FIFO protocol, retain loaded model resources,
    and forward decoded stdout as SSE deltas.
  * Streaming maps 1:1: the engine prints each decoded token to
    stdout; we emit one SSE chunk per flushed line.

Usage:
  python3 server/serve.py [--backend qwen] [--port 8080] [--salt ./salt]
                         [--cache-gb 2] [--gpu-prefill] [...]
  GEMMA4_SOURCE_DIR=/path/to/source GEMMA4_POOL=/path/to/pool.bin \
    python3 server/serve.py --backend gemma4 --port 8080

Example:
  curl -N localhost:8080/v1/chat/completions -d '{
    "model":"qwen3.5-35b-a3b","stream":true,
    "messages":[{"role":"user","content":"What is the vault code?"}],
    "max_tokens":64}'
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
from engine_config import load_engine_config

CFG = load_engine_config()
MODEL_ID = CFG.get("MODEL_ID", "salt-qwen3.6-35B-A3B")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--backend", choices=("qwen", "gemma4", "maple"), default="qwen",
                   help="model execution backend (default: qwen)")
    p.add_argument("--maple-package", default=None)
    p.add_argument("--maple-runner", default=None)
    p.add_argument("--maple-context", type=int, default=None)
    p.add_argument("--maple-batch", type=int, default=None)
    p.add_argument("--salt", default=CFG.get("SALT_BIN", "./salt"),
                   help="engine binary (default ./salt)")
    p.add_argument("--model", default=None,
                   help="served model id; defaults from the selected backend")
    p.add_argument("--cache-gb", type=float, default=float(CFG.get("CHAT_CACHE_GB", "1")))
    p.add_argument("--trunk-gb", type=float, default=2.0)
    p.add_argument("--pin-layers", type=int, default=int(CFG.get("CHAT_PIN_LAYERS", "0")))
    p.add_argument("--mem-limit-gb", type=float, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--preset", default="conserve",
                   choices=["conserve", "full", "laptop", "server"])
    p.add_argument(
        "--gpu-prefill", action="store_true",
        help="explicitly select the model-qualified bounded Metal prefill / "
             "CPU decode policy (CPU remains the default)")
    p.add_argument("--trunk", default=os.path.join(CFG.get("MODEL_TRUNK", "/tmp/q36-trunk"), "trunk.bin"))
    p.add_argument("--model-dir", default=CFG.get("MODEL_TRUNK", "/tmp/q36-trunk"))
    p.add_argument("--offsets", default=os.path.join(CFG.get("MODEL_TRUNK", "/tmp/q36-trunk"), "trunk.offsets"))
    p.add_argument("--layout-trunk", default=os.path.join(CFG.get("MODEL_TRUNK", "/tmp/q36-trunk"), "trunk.json"))
    p.add_argument("--pool", default=os.path.join(CFG.get("MODEL_POOL", "/tmp/q36-pool"), "pool.bin"))
    p.add_argument("--layout-pool", default=os.path.join(CFG.get("MODEL_POOL", "/tmp/q36-pool"), "manifest.json"))
    p.add_argument("--head", default=os.path.join(CFG.get("MODEL_TRUNK", "/tmp/q36-trunk"), "head.json"))
    p.add_argument("--embed", default=os.path.join(CFG.get("MODEL_TRUNK", "/tmp/q36-trunk"), "embed.json"))
    p.add_argument("--tokenizer", default=CFG.get("MODEL_TOKENIZER",
                   os.path.expanduser(
                       "~/.cache/huggingface/mlx-qwen36-a3b-4bit/tokenizer.json")))
    p.add_argument("--run-clean", default="tools/test/run-clean.sh")
    p.add_argument("--kv-load", default=None,
                   help="corpus KV file (SALTKV01 from --kv-save/--kvevolve) "
                        "to load at every run; requests then prefill only "
                        "their delta (state formed once, served everywhere)")
    p.add_argument("--corpus-ids", default=None,
                   help="comma/space ids text file for the corpus prefix "
                        "that --kv-load corresponds to; prepended to every "
                        "request prompt")

    # Authenticated persistent-session Gemma 4 backend. These arguments are
    # ignored by the default Qwen backend.
    p.add_argument("--gemma-source-dir",
                   default=os.environ.get("GEMMA4_SOURCE_DIR"))
    p.add_argument("--gemma-pool", default=os.environ.get("GEMMA4_POOL"))
    p.add_argument("--gemma-receipt",
                   default=os.environ.get("GEMMA4_RECEIPT"))
    p.add_argument("--gemma-auth-receipt",
                   default=os.environ.get("GEMMA4_AUTH_RECEIPT"),
                   help="persisted full-authentication receipt; defaults to POOL.auth.json")
    p.add_argument("--gemma-python",
                   default=os.environ.get("GEMMA4_PYTHON", sys.executable))
    p.add_argument("--gemma-text-wrapper",
                   default=os.path.join(os.path.dirname(__file__), "..",
                                        "tools", "gemma4-qa.py"))
    p.add_argument("--gemma-multimodal-wrapper",
                   default=os.path.join(os.path.dirname(__file__), "..",
                                        "tools", "gemma4-multimodal-qa.py"))
    p.add_argument("--gemma-text-runner",
                   default=os.path.join(os.path.dirname(__file__), "..",
                                        "gemma4-qa"))
    p.add_argument("--gemma-multimodal-runner",
                   default=os.path.join(os.path.dirname(__file__), "..",
                                        "gemma4-multimodal-qa"))
    p.add_argument("--gemma-server-runner",
                   default=os.path.join(os.path.dirname(__file__), "..",
                                        "gemma4-server"))
    p.add_argument("--gemma-text-build-receipt", default=None)
    p.add_argument("--gemma-multimodal-build-receipt", default=None)
    p.add_argument("--gemma-server-build-receipt", default=None)
    p.add_argument("--gemma-session-mode", choices=("stateful", "openai"),
                   default=os.environ.get("GEMMA4_SESSION_MODE", "stateful"),
                   help="stateful live conversation or complete-request OpenAI prefix reuse")
    p.add_argument("--gemma-shared-kv", default=None,
                   help="operator-owned immutable G4KVC006 prefix")
    p.add_argument("--gemma-mentor-root", default=None,
                   help="private root for managed immutable mentor anchors")
    p.add_argument("--gemma-mentor-anchor", default=None,
                   help="exact operator selector anchor_id@revision")
    p.add_argument("--gemma-dpr-root", default=None,
                   help="private Dynamic Performance Reference graph root")
    p.add_argument("--gemma-dpr-mode", choices=("off", "persist", "dynamic"),
                   default=os.environ.get("GEMMA4_DPR_MODE", "off"))
    p.add_argument("--gemma-dpr-budget-gb", type=float, default=None,
                   help="additive DPR budget; persist mode only")
    p.add_argument("--gemma-dpr-serial-ms-per-token", type=float, default=None,
                   help="measured serial decode reference for DPR charts")
    p.add_argument("--gemma-dpr-retention-gb", type=float, default=None,
                   help="native DPR graph retention budget in decimal GB")
    p.add_argument("--gemma-dpr-mentor", action="store_true",
                   help="enable model-configured engine-level Mentor selector")
    p.add_argument("--gemma-dpr-mindset-attention", action="store_true",
                   help="enable model-configured A0 Mindset Attention plans")
    p.add_argument("--gemma-proof-state", action="store_true",
                   help="compute authoritative state digests on every COMMIT")
    p.add_argument("--gemma-workers", type=int, default=None)
    p.add_argument("--gemma-context-tokens", type=int, default=None,
                   help="Gemma session context capacity (model maximum 262144)")
    p.add_argument("--gemma-max-output-tokens", type=int, default=None,
                   help="maximum output tokens admitted per request")
    p.add_argument("--gemma-prefill-chunk-tokens", type=int, default=None,
                   help="maximum tokens in one native prefill chunk (up to 4096)")
    p.add_argument("--gemma-kv-budget-gb", type=float, default=None)
    p.add_argument("--gemma-timeout-s", type=float, default=None)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
    p.add_argument("--gemma-target-route-n", type=int, default=None,
                   help="startup NFQ sequence depth; defaults from the recipe")
    p.add_argument("--gemma-target-x", type=int, default=None,
                   help="startup X-TARGET causal row count; defaults from the recipe")
    p.add_argument("--gemma-target-route-f", type=int, default=None,
                   help="startup NFQ route fanout; defaults from the recipe")
    p.add_argument("--gemma-target-route-q", type=int, default=None,
                   help="startup NFQ worker runway; defaults from the recipe")
    p.add_argument("--gemma-expert-budget-gb", type=int, default=None,
                   help="startup expert-residency budget; defaults from the recipe")
    p.add_argument("--gemma-memory-limit-gb", type=float, default=None,
                   help="startup all-inclusive RSS ceiling; defaults from the recipe")
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
    p.add_argument("--gemma-image-root",
                   default=os.environ.get("GEMMA4_IMAGE_ROOT"),
                   help="optional root permitted for file: image URLs")
    p.add_argument("--gemma-kv-cache-root",
                   default=os.environ.get("GEMMA4_KV_CACHE_ROOT"),
                   help="private root for logical Gemma KV cache IDs")
    p.add_argument("--gemma-verify-only", action="store_true",
                   help="authenticate Gemma setup and exit without serving")
    return p


ARGS = None
_LOCK = threading.Lock()          # single-flight gate
_ENV = dict(os.environ)
_CORPUS_CACHE = None
_CORPUS_LOCK = threading.Lock()

_SCRUB_ENGINE_KEYS = (
    "SALT_GPU", "SALT_GPU_RESIDENT", "SALT_GPU_TRUNK", "SALT_GPU_MOE",
    "SALT_GPU_OVERLAP", "SALT_GPU_TRUNK_PREFILL_ONLY",
    "SALT_GPU_TRUNK_B", "SALT_GPU_PREFILL_B", "SALT_GPU_B_QKV",
    "SALT_GPU_B_Z", "SALT_GPU_B_O", "SALT_GPU_DISPATCH",
    "SALT_GPU_LEDGER", "SALT_GPU_ADDRESS_AUDIT", "SALT_GPU_MS",
    "SALT_GPU_DIAG", "SALT_WATERFALL",
    "SALT_CHUNK_MS", "SALT_NAN_PROBE", "SALT_M2_BATCH",
    "SALT_M2_BATCH_B", "SALT_DEBUG7", "SALT_STEP_MS",
    "SALT_TIME_LAYERS",
)
_MODEL_PREFILL_KEYS = (
    "SALT_PREFILL_CHUNK", "SALT_PREFILL_B", "SALT_M2_BATCH",
    "SALT_M2_BATCH_B", "SALT_CACHE_MODE",
)
_MODEL_GPU_PREFILL_KEYS = (
    "SALT_GPU_TRUNK_B", "SALT_GPU_PREFILL_B", "SALT_GPU_B_QKV",
    "SALT_GPU_B_Z", "SALT_GPU_B_O", "SALT_GPU_DISPATCH",
)


def model_engine_env(gpu_prefill=None):
    """Apply model policy; CPU unless GPU prefill was explicitly selected."""
    env = dict(_ENV)
    for key in _SCRUB_ENGINE_KEYS:
        env.pop(key, None)
    for key in _MODEL_PREFILL_KEYS:
        if key in CFG:
            env[key] = CFG[key]
    if gpu_prefill is None:
        gpu_prefill = bool(ARGS is not None and ARGS.gpu_prefill)
    if gpu_prefill:
        env.update({
            "SALT_GPU": "1",
            "SALT_GPU_RESIDENT": "1",
            "SALT_GPU_TRUNK": "1",
            "SALT_GPU_MOE": "0",
            "SALT_GPU_OVERLAP": "0",
            "SALT_GPU_TRUNK_PREFILL_ONLY": "1",
        })
        for key in _MODEL_GPU_PREFILL_KEYS:
            if key in CFG:
                env[key] = CFG[key]
    return env


def _corpus_ids():
    """Load corpus ids once (thread-safe), cached."""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE
    with _CORPUS_LOCK:
        if _CORPUS_CACHE is not None:
            return _CORPUS_CACHE
        with open(ARGS.corpus_ids) as f:
            text = f.read()
        _CORPUS_CACHE = [int(x) for x in text.replace("\n", ",")
                         .split(",") if x.strip() != ""]
    return _CORPUS_CACHE


def engine_cmd(pids_path, max_tokens):
    """Build the salt argv for one completion (pids-file based)."""
    cmd = [
        os.path.abspath(ARGS.salt),
        ARGS.model_dir,           # positional MODEL_DIR (directory)
        "--trunk", ARGS.trunk,
        "--offsets", ARGS.offsets,
        "--layout-trunk", ARGS.layout_trunk,
        "--pool", ARGS.pool,
        "--layout-pool", ARGS.layout_pool,
        "--head", ARGS.head,
        "--embed", ARGS.embed,
        "--tokenizer", ARGS.tokenizer,
        "--pids-file", pids_path,
        "--gen", str(max_tokens),
        "--cache-gb", str(ARGS.cache_gb),
        "--trunk-gb", str(ARGS.trunk_gb),
        "--pin-layers", str(ARGS.pin_layers),
        "--mem-limit-gb", str(ARGS.mem_limit_gb),
        "--threads", str(ARGS.threads),
    ]
    if ARGS.kv_load:
        cmd += ["--kv-load", os.path.abspath(ARGS.kv_load)]
    return cmd


TOK = None          # lazily loaded HF tokenizer (tokenize server-side)
TOK_LOCK = threading.Lock()


def _load_tok():
    global TOK
    if TOK is not None:
        return TOK
    with TOK_LOCK:
        if TOK is not None:
            return TOK
        from transformers import AutoTokenizer
        # the tokenizer dir is the HF cache dir; --tokenizer points at
        # tokenizer.json inside it
        d = os.path.dirname(ARGS.tokenizer)
        TOK = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    return TOK


def text_to_ids(prompt_text):
    """Tokenize with the HF tokenizer (handles <|im_start|> etc. --
    falls back to the engine's own tokenizer via tools/tokenc when HF
    isn't installed -- portable on any box the engine runs)."""
    try:
        tok = _load_tok()
        ids = tok.encode(prompt_text, add_special_tokens=True)
        return list(ids)
    except Exception:
        _tokenc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "tools", "tokenc")
        import subprocess
        r = subprocess.run(
            [_tokenc, ARGS.tokenizer, prompt_text],
            capture_output=True, text=True, timeout=60)
        ids = [int(x) for x in r.stdout.strip().split(",") if x]
        return ids


def run_engine(prompt_text, max_tokens, strip_think=False):
    """Run the engine, yielding decoded text chunks (streaming)."""
    env = model_engine_env()
    env["SALT_STRIP_THINK"] = "1" if strip_think else "0"
    # the Qwen3.6 model card's recommended recipe (thinking mode,
    # general): temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    # presence_penalty=1.5, repetition_penalty=1.0. Greedy stays
    # available via SALT_GREEDY=1 for determinism runs.
    if CFG.get("SALT_GREEDY"):
        env["SALT_GREEDY"] = CFG["SALT_GREEDY"]
    env["SALT_REP_PENALTY"] = CFG.get("SALT_REP_PENALTY", "1.0")
    env["SALT_FREQ_PENALTY"] = CFG.get("SALT_FREQ_PENALTY", "0.0")
    env["SALT_PRESENCE_PENALTY"] = CFG.get("SALT_PRESENCE_PENALTY", "1.5")
    env["SALT_TOPK"] = CFG.get("SALT_TOPK", "20")
    env["SALT_TOP_P"] = CFG.get("SALT_TOP_P", "0.95")
    env["SALT_TEMP"] = CFG.get("SALT_TEMP", "1.0")

    # tokenize server-side, pass ids via a temp pids file
    ids = text_to_ids(prompt_text)
    if ARGS.corpus_ids:
        CORPUS_IDS = _corpus_ids()
        ids = CORPUS_IDS + ids
    import tempfile
    fd, pids_path = tempfile.mkstemp(suffix=".ids", prefix="salt-serve-")
    with os.fdopen(fd, "w") as f:
        f.write(",".join(str(x) for x in ids))

    cmd = engine_cmd(pids_path, max_tokens)
    if ARGS.run_clean and os.path.exists(ARGS.run_clean):
        cmd = ["bash", os.path.abspath(ARGS.run_clean)] + cmd

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env,
                            text=False, bufsize=0)
    # the engine flushes stdout per decoded token (text mode); read
    # raw bytes on a poll tick and emit as SSE deltas.
    import select
    try:
        while True:
            r, _, _ = select.select([proc.stdout], [], [], 0.05)
            if r:
                data = os.read(proc.stdout.fileno(), 4096)
                if not data:
                    break
                yield data.decode("utf-8", errors="replace")
            elif proc.poll() is not None:
                # drain anything left
                while True:
                    data = os.read(proc.stdout.fileno(), 4096)
                    if not data:
                        break
                    yield data.decode("utf-8", errors="replace")
                break
    finally:
        rc = None
        try:
            rc = proc.wait()
        except Exception:
            pass
        # surface a memory-limit stop (engine exit 3) as a clear SSE
        # error instead of a silent truncated stream
        if rc == 3:
            try:
                yield 'data: {"error": "engine memory limit reached ' \
                      '(--mem-limit-gb); request aborted", ' \
                      '"type": "server_error"}\n\n'
            except (RuntimeError, GeneratorExit):
                pass   # consumer already closed the stream
        try:
            os.unlink(pids_path)
        except Exception:
            pass


def chat_prompt(messages):
    """Apply the Qwen chat template (im_start/im_end). The trailing
    newline after <|im_start|>assistant is REQUIRED: without it the
    engine's decode of the first generated token misbehaves (raw
    bytes + repeated 'assistant' -- measured).

    Assistant turn opens the OFFICIAL instruct-mode empty think
    block (the repo's chat_template.jinja with enable_thinking
    false): "<|im_start|>assistant\n<think>\n\n</think>\n\n".
    Measured on this 4-bit checkpoint:
      <think> opener (thinking mode):  question-echo / list loop
      no opener:                       template-fragment echoes
      empty think block (instruct):   clean answers
    The empty block is prompt tokens, not streamed output; the
    model answers directly after it.

    SALT_THINK_MODE=1 switches to the REAL thinking mode: the
    <think> opener -- the model's reasoning trace streams before
    the answer (the measured echo-loop risk applies)."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if os.environ.get("SALT_THINK_MODE") == "1":
        parts.append("<|im_start|>assistant\n<think>\n")
    else:
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def session_prompt(sid, text):
    """Session framing: the KV rows ARE the history, so each request
    carries only the new user turn. First turn opens the template;
    later turns first close the previous assistant turn. Assistant
    turns open the official instruct-mode empty think block (see
    chat_prompt -- measured cleanest on this checkpoint)."""
    eng = _SESSIONS.get(sid)
    first = eng is None or eng._first_turn
    if eng is not None:
        eng._first_turn = False
        if not first and not eng._turn_closed:
            # previous turn hit the gen cap without im_end: supply
            # the closer (a natural close would double-close)
            prefix = "<|im_end|>\n"
        else:
            prefix = ""
    else:
        prefix = ""
    return (prefix + f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            if os.environ.get("SALT_THINK_MODE") != "1" else
            prefix + f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n")


def _degenerate_turn(raw_out):
    """True when a turn's raw output is template-echo garbage.

    Signature: the FIRST generated token is a template/control token
    (<|im_start|>, <|im_end|>, <|endoftext|>, </think>) -- as if the
    model thinks the turn already ended and starts a new one. A
    healthy turn STARTS with real answer text and may only END with
    <|im_end|> (the EOS closer, printed by the engine); it never
    opens with template tokens. Prompt shapes that trigger this on
    the 4-bit checkpoint: bare fragments ("test", "hey", "capital
    of US") -- full sentences are immune. Detecting it lets the
    server auto-reset the session before the delta state absorbs
    the garbage."""
    s = raw_out
    # a healthy turn may END with the EOS closer -- ignore the tail
    for tail in ("<|im_end|>", "<|endoftext|>"):
        if s.endswith(tail):
            s = s[: -len(tail)]
    s = s.lstrip()
    return s.startswith(("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                         "</think>"))


class ThinkFilter:
    """Server-side think-block filter (presentation layer). The
    engine now streams EVERYTHING; a request with strip_think=true
    runs its chunk stream through this to suppress the reasoning
    trace. Stateful across chunk boundaries (the block can span
    SSE chunks)."""

    def __init__(self):
        # No think opener in the framing anymore (it triggers the
        # echo attractor); the filter only reacts to a <think> block
        # the model emits on its own. Starts OUTSIDE the block.
        self.in_think = False
        self._carry = ""

    def feed(self, chunk):
        out = []
        s = self._carry + chunk
        self._carry = ""
        while s:
            if self.in_think:
                i = s.find("</think>")
                if i < 0:
                    # tail may split the closer across chunks
                    keep = max(0, len(s) - 8)
                    self._carry = s[keep:]
                    s = ""
                else:
                    self.in_think = False
                    s = s[i + len("</think>"):]
            else:
                i = s.find("<think>")
                if i < 0:
                    # tail may split the opener across chunks
                    keep = max(0, len(s) - 7)
                    out.append(s[:keep])
                    self._carry = s[keep:]
                    s = ""
                else:
                    out.append(s[:i])
                    self.in_think = True
                    s = s[i + len("<think>"):]
        return "".join(out)

    def flush(self):
        self._carry = ""
        return ""


# ---- persistent engine (--serve-fifo) -----------------------------
# ONE engine process serves all requests: the expensive init (trunk
# bind, expert cache, pool, head/embed/tokenizer) runs once; each
# request writes one line of pids to the fifo and reads stdout up to
# the NUL response terminator. Saves ~1s spawn+re-bind per request
# (the measurable win; Python overhead was never the cost).

class PersistentEngine:
    """Owns the fifo + one engine subprocess; frames responses by NUL."""

    def __init__(self, strip_think=True, session_id=None, kv_load=None):
        self._lock = threading.Lock()     # one request at a time
        self._fifo_path = None
        self._fifo_w = None
        self._proc = None
        self._read_buf = bytearray()
        self._resp_done = threading.Event()
        self._dead = False
        self._rc = None        # engine exit code when _dead (3 = mem limit)
        self._reader = None
        self._session_id = session_id
        self._first_turn = kv_load is None
        self._last_used = time.time()   # idle-TTL heartbeat
        self._think_open = False   # last turn ended mid-<think> (gen cap)
        self._turn_closed = False  # last turn emitted im_end/endoftext
        self._tokids_path = None
        self._start(strip_think, kv_load)

    def _start(self, strip_think, kv_load=None):
        import tempfile
        fd, fifo = tempfile.mkstemp(suffix=".fifo", prefix="salt-serve-")
        os.close(fd)
        os.unlink(fifo)
        os.mkfifo(fifo)
        self._fifo_path = fifo

        cmd = [
            os.path.abspath(ARGS.salt),
            ARGS.model_dir,
            "--trunk", ARGS.trunk,
            "--offsets", ARGS.offsets,
            "--layout-trunk", ARGS.layout_trunk,
            "--pool", ARGS.pool,
            "--layout-pool", ARGS.layout_pool,
            "--head", ARGS.head,
            "--embed", ARGS.embed,
            "--tokenizer", ARGS.tokenizer,
            "--serve-fifo", fifo,
            "--gen", "1",          # per-request gen overrides this
            # im_end/endoftext close the turn: multi-turn resumes
            # cleanly instead of continuing mid-answer
            "--eos-ids", "248046,248044",
            "--cache-gb", str(ARGS.cache_gb),
            "--trunk-gb", str(ARGS.trunk_gb),
            "--pin-layers", str(ARGS.pin_layers),
            "--mem-limit-gb", str(ARGS.mem_limit_gb),
            "--threads", str(ARGS.threads),
        ]
        if ARGS.kv_load:
            cmd += ["--kv-load", os.path.abspath(ARGS.kv_load)]
        if kv_load:
            cmd += ["--kv-load", os.path.abspath(kv_load)]
        if self._session_id:
            # Session mode keeps KV live across requests. The native engine
            # receives the checkpoint target now but writes it only after the
            # FIFO closes and the engine-owned session-close transaction runs.
            # Dump generated ids per turn so the
            # server can detect an unclosed <think> (gen-cap cut).
            save_path = _session_cache_path(self._session_id)
            cmd += ["--kv-save-after", os.path.abspath(save_path)]
            import tempfile
            fd, tp = tempfile.mkstemp(suffix=".tokids", prefix="salt-serve-",
                                      dir="/tmp")
            os.close(fd)
            self._tokids_path = tp
            cmd += ["--dump-tokids", tp]
        if ARGS.run_clean and os.path.exists(ARGS.run_clean):
            cmd = ["bash", os.path.abspath(ARGS.run_clean)] + cmd

        env = model_engine_env()
        env["SALT_STRIP_THINK"] = "1" if strip_think else "0"
        # Qwen3.6 model card recipe (thinking general): temp 1.0,
        # top_k 20, top_p 0.95, presence_penalty 1.5. Greedy via
        # SALT_GREEDY=1 remains for determinism runs.
        if CFG.get("SALT_GREEDY"):
            env["SALT_GREEDY"] = CFG["SALT_GREEDY"]
        env["SALT_REP_PENALTY"] = CFG.get("SALT_REP_PENALTY", "1.0")
        env["SALT_FREQ_PENALTY"] = CFG.get("SALT_FREQ_PENALTY", "0.0")
        env["SALT_PRESENCE_PENALTY"] = CFG.get("SALT_PRESENCE_PENALTY", "1.5")
        env["SALT_TOPK"] = CFG.get("SALT_TOPK", "20")
        env["SALT_TOP_P"] = CFG.get("SALT_TOP_P", "0.95")
        env["SALT_TEMP"] = CFG.get("SALT_TEMP", "1.0")
        if self._session_id:
            env["SALT_SESSION"] = "1"
            env["SALT_KV_BUDGET_GB"] = str(
                float(os.environ.get("SALT_KV_BUDGET_GB", "0.5")))

        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=env, text=False, bufsize=0)
        # drain stderr continuously (the engine writes run reports +
        # [serve] logs; an undrained pipe fills and deadlocks the
        # engine mid-generation). Log lines to our stderr.
        self._stderr_thread = threading.Thread(target=self._stderr_loop,
                                               daemon=True)
        self._stderr_thread.start()
        # open the fifo for WRITING -- the engine's fopen(...,"r")
        # blocks until a writer appears (POSIX). Keeping it open
        # across requests is what makes the loop persist.
        self._fifo_w = os.open(self._fifo_path, os.O_WRONLY)

        self._reader = threading.Thread(target=self._read_loop,
                                        daemon=True)
        self._reader.start()

    def _stderr_loop(self):
        import select
        import re
        fd = self._proc.stderr.fileno()
        # engine's own per-request summary: "N tokens in T s, X s/token"
        self._tok_re = re.compile(rb"(\d+) tokens in ([\d.]+) s")
        self._last_tokens = 0
        self._last_engine_s = 0.0
        while True:
            try:
                r, _, _ = select.select([fd], [], [], 0.5)
                if not r:
                    if self._proc.poll() is not None:
                        return
                    continue
                data = os.read(fd, 4096)
            except Exception:
                data = b""
            if not data:
                return
            m = self._tok_re.search(data)
            if m:
                self._last_tokens = int(m.group(1))
                self._last_engine_s = float(m.group(2))
            sys.stderr.write(data.decode("utf-8", errors="replace"))
            sys.stderr.flush()

    def engine_rss_mb(self):
        """Live engine RSS (KB from ps, macOS/Linux). None if dead."""
        if not self._proc or self._proc.poll() is not None:
            return None
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self._proc.pid)],
                capture_output=True, text=True, timeout=5).stdout.strip()
            return round(int(out) / 1024.0, 1) if out else None
        except Exception:
            return None

    def _read_loop(self):
        """Continuously drain stdout into _read_buf; set _resp_done
        when a NUL terminator arrives (one response framed).

        CRITICAL: use os.read(fd, ...), NOT read()/read1() -- Popen
        with text=False/bufsize=0 gives a raw FileIO (no read1), and
        read(n) on it blocks until n bytes accumulate. The engine
        writes a short response (~tens of bytes + NUL) then blocks on
        the fifo for the next request; select below guarantees data
        is available, so os.read returns what is there immediately."""
        import select
        buf = bytearray()
        fd = self._proc.stdout.fileno()
        while True:
            try:
                r, _, _ = select.select([fd], [], [], 0.5)
                if not r:
                    if self._proc.poll() is not None:
                        self._dead = True
                        try: self._rc = self._proc.returncode
                        except Exception: self._rc = None
                        self._resp_done.set()
                        return
                    continue
                data = os.read(fd, 4096)
            except Exception:
                data = b""
            if not data:
                if self._proc.poll() is not None:
                    self._dead = True
                    try: self._rc = self._proc.returncode
                    except Exception: self._rc = None
                    self._resp_done.set()
                return
            buf.extend(data)
            idx = buf.find(b"\0")
            while idx >= 0:
                chunk = bytes(buf[:idx])
                del buf[:idx + 1]
                with self._lock:
                    self._read_buf.extend(chunk)
                    self._resp_done.set()
                idx = buf.find(b"\0")
            # leftover bytes with no NUL yet: keep them buffered for
            # the NEXT read. Do NOT set _resp_done here -- a response
            # is only complete when its NUL arrives, and complete()
            # must not break early on partial text.
            with self._lock:
                self._read_buf.extend(bytes(buf))
                del buf[:]

    def complete(self, prompt_text, max_tokens):
        """Send one request ("gen:pids" line), yield decoded text
        chunks, wait for the NUL response terminator."""
        self._last_used = time.time()   # idle-TTL heartbeat
        ids = text_to_ids(prompt_text)
        if ARGS.corpus_ids:
            ids = _corpus_ids() + ids
        if self._tokids_path:
            try:
                os.unlink(self._tokids_path)
            except OSError:
                pass
        line = f"{max_tokens}:" + ",".join(str(x) for x in ids) + "\n"
        with self._lock:
            self._read_buf.clear()
            self._resp_done.clear()
            os.write(self._fifo_w, line.encode())
        deadline = time.time() + 600
        while True:
            # yield anything new in the buffer
            with self._lock:
                if self._dead:
                    if self._rc == 3:
                        raise MemoryError(
                            "engine memory limit reached (--mem-limit-gb); "
                            "request aborted")
                    raise RuntimeError("engine process died")
                if self._read_buf:
                    data = bytes(self._read_buf)
                    self._read_buf.clear()
                    if data:
                        yield data.decode("utf-8", errors="replace")
            if self._resp_done.is_set() and not self._read_buf:
                break
            if time.time() > deadline:
                raise RuntimeError("engine response timeout")
            time.sleep(0.02)
        self._update_think_state()

    def _update_think_state(self):
        """Scan the turn's dumped ids: if the LAST think-tag is an
        opener (248068) with no closer (248069), the gen cap cut the
        turn mid-reasoning -- the next framing must close it."""
        if not self._tokids_path:
            return
        try:
            with open(self._tokids_path, "rb") as f:
                raw = f.read()
        except OSError:
            return
        self._think_open = False
        self._turn_closed = False
        for i in range(len(raw) // 4 - 1, -1, -1):
            tid = int.from_bytes(raw[i*4:i*4+4], "little")
            if tid == 248069:      # </think> -- closed
                break
            if tid == 248068:      # <think> -- still open
                self._think_open = True
                break
        # did the turn itself close (im_end/endoftext emitted)? If it
        # hit the gen cap mid-ramble there is no closer and the next
        # framing must supply one; if it closed naturally, adding
        # another <|im_end|> would create a double close.
        for i in range(len(raw) // 4 - 1, -1, -1):
            tid = int.from_bytes(raw[i*4:i*4+4], "little")
            if tid == 248046 or tid == 248044:  # im_end / endoftext
                self._turn_closed = True
                break
            if tid == 248068:      # past the turn's start
                break

    def shutdown(self):
        try:
            os.close(self._fifo_w)
        except Exception:
            pass
        if self._proc:
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        try:
            os.unlink(self._fifo_path)
        except Exception:
            pass

    def kv_rows(self):
        """Last durably exported rows from a closed session checkpoint (0 if
        none). Live rows remain engine-owned until close. SALTKV01 stores the count at header word 3
        (byte 24, u64 little-endian)."""
        if not self._session_id:
            return 0
        import struct
        try:
            with open(_session_cache_path(self._session_id), "rb") as f:
                hdr = f.read(64)
            return struct.unpack("<Q", hdr[24:32])[0]
        except (OSError, struct.error):
            return 0

    def stats(self, elapsed_s):
        """Per-request inference telemetry: engine-side tokens/timing
        (parsed from its stderr summary) + server-side elapsed + live
        RSS + resident session rows."""
        toks = getattr(self, "_last_tokens", 0) or 0
        eng_s = getattr(self, "_last_engine_s", 0.0) or 0.0
        rss = self.engine_rss_mb()
        rows = self.kv_rows()
        tok_s = (toks / eng_s) if eng_s > 0 else 0.0
        return (f"[stats] session={self._session_id or '(stateless)'} "
                f"tokens={toks} engine={eng_s:.2f}s server={elapsed_s:.2f}s "
                f"{tok_s:.1f} tok/s rss={rss}MB rows={rows}")


_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_SESSIONS = {}                 # session id -> PersistentEngine
_SESSIONS_LOCK = threading.Lock()
_SALT_CACHE = os.environ.get(
    "SALT_CHAT_CACHE", os.path.expanduser("~/.salt-sessions"))
# idle TTL: sessions are dropped after this long without a request.
# SALTKV01 resume is ~1s, so keeping an engine resident after the
# client leaves wastes RAM for nothing. SALT_SESSION_TTL=0 disables
# (keep sessions resident until DELETE).
_SESSION_TTL = float(os.environ.get("SALT_SESSION_TTL", "600"))


def _session_cache_path(sid):
    return os.path.join(_SALT_CACHE, f"salt-chat-{sid}.kv")


def _session_note_path(sid):
    return os.path.join(_SALT_CACHE, f"salt-chat-{sid}.md")


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                # strip_think=False: the engine streams EVERYTHING
                # (think block included); think-filtering is a
                # per-request server-side presentation choice via
                # the request's strip_think field.
                _ENGINE = PersistentEngine(strip_think=False)
    return _ENGINE


def get_session_engine(sid):
    """Get (or create) the persistent engine for a session. Sessions
    are the server's unit of memory: one engine process per session,
    SALT_SESSION=1 makes the engine accumulate KV rows across
    requests (live memory), --kv-load resumes a checkpoint, and the
    serve loop writes --kv-save-after checkpoints per turn."""
    with _SESSIONS_LOCK:
        eng = _SESSIONS.get(sid)
        if eng is not None and not eng._dead:
            return eng
        cache = _session_cache_path(sid)
        eng = PersistentEngine(strip_think=False, session_id=sid,
                               kv_load=cache if os.path.exists(cache) else None)
        _SESSIONS[sid] = eng
        return eng


def _session_sweeper():
    """Idle-TTL sweeper: drop sessions that have not been touched for
    _SESSION_TTL seconds. The KV checkpoint is already on disk
    (--kv-save-after per turn), so dropping an engine costs only the
    ~1s reload on the next request -- no data loss, no lingering
    processes. Runs every 30s; disabled when TTL is 0."""
    while True:
        time.sleep(30)
        if _SESSION_TTL <= 0:
            continue
        now = time.time()
        with _SESSIONS_LOCK:
            stale = [sid for sid, eng in _SESSIONS.items()
                     if not eng._dead and
                     now - eng._last_used > _SESSION_TTL]
        for sid in stale:
            sys.stderr.write(f"[sweep] session {sid} idle > "
                             f"{_SESSION_TTL:.0f}s -- dropping\n")
            sys.stderr.flush()
            drop_session(sid)


def drop_session(sid):
    """Forget a session: shut down its engine and remove its memory
    files (the knowledge-base /reset operation)."""
    with _SESSIONS_LOCK:
        eng = _SESSIONS.pop(sid, None)
    if eng is not None:
        try:
            eng.shutdown()
        except Exception:
            pass
    for p in (_session_cache_path(sid), _session_note_path(sid)):
        try:
            os.unlink(p)
        except OSError:
            pass


def session_rows(sid):
    """Rows currently resident in a session's KV cache (0 when the
    session is not loaded)."""
    with _SESSIONS_LOCK:
        eng = _SESSIONS.get(sid)
    if eng is None or eng._dead:
        return 0
    return eng.kv_rows()


def write_session_note(sid, gen):
    """The knowledge-base manifest: an Obsidian-style note with YAML
    frontmatter, updated after every turn. The engine (mechanism)
    writes the SALTKV01 memory; this note is the vault's index of it."""
    import datetime
    os.makedirs(_SALT_CACHE, exist_ok=True)
    path = _session_note_path(sid)
    rows = session_rows(sid)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    body = ("---\n"
            f"session: {sid}\n"
            f"rows: {rows}\n"
            f"gen: {gen}\n"
            f"type: salt-chat\n"
            f"updated: {now}\n"
            "---\n\n"
            f"Session memory: `salt-chat-{sid}.kv` ({rows} rows).\n"
            "The KV cache is the memory hyperspace; the engine is the\n"
            "mechanism. This note is the knowledge base's index.\n")
    try:
        with open(path, "w") as f:
            f.write(body)
    except OSError:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] %s\n" % (fmt % args))

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{
                "id": ARGS.model, "object": "model", "created": 0,
                "owned_by": "salt"}]})
        elif path.startswith("/v1/sessions/"):
            sid = path[len("/v1/sessions/"):]
            if not sid:
                self._send_json(400, {"error": {"message": "session id required",
                                                "type": "invalid_request_error"}})
                return
            self._send_json(200, {
                "session": sid,
                "rows": session_rows(sid),
                "cache": _session_cache_path(sid),
                "note": _session_note_path(sid)})
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "invalid_request_error"}})

    def do_DELETE(self):
        path = self.path.rstrip("/")
        if path.startswith("/v1/sessions/"):
            sid = path[len("/v1/sessions/"):]
            if not sid:
                self._send_json(400, {"error": {"message": "session id required",
                                                "type": "invalid_request_error"}})
                return
            drop_session(sid)
            self._send_json(200, {"session": sid, "deleted": True})
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "invalid_request_error"}})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send_json(400, {"error": {"message": str(e),
                                            "type": "invalid_request_error"}})
            return

        if self.path.rstrip("/") == "/v1/chat/completions":
            self._chat(body)
        elif self.path.rstrip("/") == "/v1/completions":
            self._completions(body)
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "invalid_request_error"}})

    def _chat(self, body):
        messages = body.get("messages", [])
        if not messages:
            self._send_json(400, {"error": {
                "message": "messages required", "type": "invalid_request_error"}})
            return
        sid = body.get("session") or None
        if sid:
            # session mode: the KV rows are the history -- only the
            # NEW user turn is framed, on top of the live memory
            text = messages[-1].get("content", "")
            prompt = session_prompt(sid, text)
        else:
            prompt = chat_prompt(messages)
        self._complete(body, prompt)

    def _completions(self, body):
        prompt = body.get("prompt", "")
        self._complete(body, prompt)

    def _complete(self, body, prompt):
        max_tokens = int(body.get("max_tokens", 64) or 64)
        stream = bool(body.get("stream", False))
        # session: the unit of memory. When present, the request is
        # served by the session's persistent engine (live KV across
        # turns); when absent, the stateless engine handles it.
        sid = body.get("session") or None
        # strip_think: per-request presentation choice. The engine
        # streams everything; the server filters the <think> block
        # out of the SSE/JSON stream when the caller asks. Default
        # false -- the reasoning trace streams by default.
        strip_think = bool(body.get("strip_think", False))
        tf = ThinkFilter() if strip_think else None
        # NOTE: strip_think is fixed at engine start (the persistent
        # engine is launched with SALT_STRIP_THINK=1, so the think
        # trace is suppressed server-side and answers stream clean).
        # The field is accepted for API compatibility.
        # the thinking knobs: max_think caps the trace's tokens;
        # think_mode switches the prompt to the real <think> opener
        # (the engine streams the trace raw, client-side filtering is
        # a presentation choice).
        max_think = body.get("max_think")
        if max_think:
            os.environ["SALT_MAX_THINK"] = str(int(max_think))
        think_mode = body.get("think_mode")
        if think_mode is not None:
            os.environ["SALT_THINK_MODE"] = "1" if think_mode else "0"
        model = body.get("model", ARGS.model)
        cid = "chatcmpl-%d" % int(time.time() * 1000)
        created = int(time.time())

        # single-flight: hold the lock for the whole generation
        if not _LOCK.acquire(timeout=1):
            self._send_json(503, {"error": {
                "message": "engine busy (single-flight)", "type": "server_error"}})
            return
        try:
            t0 = time.time()
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # Connection: close is REQUIRED -- HTTP/1.1 keep-alive
                # makes curl wait forever after [DONE] (the stream is
                # complete but the socket stays open for reuse).
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    first = True
                    raw_out = []
                    eng = get_session_engine(sid) if sid else get_engine()
                    for chunk in eng.complete(prompt, max_tokens):
                        raw_out.append(chunk)
                        text = chunk
                        if tf is not None:
                            text = tf.feed(text)
                        if not text:
                            continue
                        delta = {"role": "assistant",
                                 "content": text} if first else {"content": text}
                        first = False
                        ev = {"id": cid, "object": "chat.completion.chunk",
                              "created": created, "model": model,
                              "choices": [{"index": 0, "delta": delta,
                                           "finish_reason": None}]}
                        self.wfile.write(
                            ("data: %s\n\n" % json.dumps(ev)).encode())
                        self.wfile.flush()
                    # degenerate-turn guard: if the model emitted a
                    # TEMPLATE token (<|im_start|>) as its own output
                    # -- the "test"/"hey"/"capital of US" degeneration
                    # -- the turn is garbage and the session's delta
                    # state absorbed it, poisoning every later turn.
                    # Auto-reset the session so the poison does not
                    # persist. (A normal turn ends with <|im_end|>
                    # from EOS; it never OPENS with template tokens.)
                    if sid and _degenerate_turn("".join(raw_out)):
                        sys.stderr.write(
                            f"[degenerate] session {sid}: model emitted "
                            "template tokens -- auto-reset\n")
                        sys.stderr.flush()
                        drop_session(sid)
                    done = {"id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}]}
                    self.wfile.write(("data: %s\n\n" % json.dumps(done)).encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    if sid:
                        write_session_note(sid, max_tokens)
                        sys.stderr.write(
                            get_session_engine(sid).stats(time.time() - t0) + "\n")
                    else:
                        sys.stderr.write(
                            get_engine().stats(time.time() - t0) + "\n")
                    sys.stderr.flush()
                except MemoryError:
                    # engine hit --mem-limit-gb mid-stream: SSE error,
                    # no [DONE] (client sees a clean abort reason)
                    try:
                        self.wfile.write(
                            ("data: {\"error\": \"engine memory limit reached "
                             "(--mem-limit-gb); request aborted\", "
                             "\"type\": \"server_error\"}\n\n").encode())
                        self.wfile.flush()
                    except (BrokenPipeError, OSError):
                        pass
                except BrokenPipeError:
                    pass
            else:
                try:
                    eng = get_session_engine(sid) if sid else get_engine()
                    out = "".join(eng.complete(prompt, max_tokens))
                    if tf is not None:
                        out = tf.feed(out) + tf.flush()
                    # degenerate-turn guard (see stream path): a turn
                    # where the model emitted template tokens poisons
                    # the session's delta state -- auto-reset.
                    if sid and _degenerate_turn(out):
                        sys.stderr.write(
                            f"[degenerate] session {sid}: model emitted "
                            "template tokens -- auto-reset\n")
                        sys.stderr.flush()
                        drop_session(sid)
                except MemoryError:
                    self._send_json(503, {"error": {
                        "message": "engine memory limit reached "
                                   "(--mem-limit-gb); request aborted",
                        "type": "server_error"}})
                    return
                self._send_json(200, {
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": model,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": out},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(prompt),
                              "completion_tokens": max_tokens,
                              "total_tokens": len(prompt) + max_tokens},
                    "session": sid,
                    "elapsed_s": round(time.time() - t0, 2)})
                if sid:
                    write_session_note(sid, max_tokens)
                    sys.stderr.write(
                        get_session_engine(sid).stats(time.time() - t0) + "\n")
                    sys.stderr.flush()
                else:
                    sys.stderr.write(
                        get_engine().stats(time.time() - t0) + "\n")
                    sys.stderr.flush()
        finally:
            _LOCK.release()


def main():
    global ARGS
    ARGS = build_parser().parse_args()
    if ARGS.backend == "maple":
        from maple_backend import serve_from_args
        try:
            return serve_from_args(ARGS)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"[maple-serve] setup failed: {exc}", file=sys.stderr, flush=True)
            return 2
    if ARGS.model is None:
        ARGS.model = ("gemma-4-26b-a4b-it"
                      if ARGS.backend == "gemma4" else MODEL_ID)
    if ARGS.backend == "gemma4":
        from gemma4_backend import (
            BackendError, RequestError, serve_from_args,
        )
        try:
            return serve_from_args(ARGS)
        except (BackendError, RequestError, OSError) as exc:
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_BEGIN v1
            cause = exc.__cause__
            if isinstance(cause, OSError):
                print(f"[gemma4-serve] startup OS error errno={cause.errno} "
                      f"message={cause.strerror}", file=sys.stderr, flush=True)
# SALT_GEMMA4_PHYSICAL_BUILD_ONLY_END v1
            print(f"[gemma4-serve] setup failed: {exc}",
                  file=sys.stderr, flush=True)
            return 2

    # resolve the preset into effective mem/threads; explicit CLI
    # --mem-limit-gb / --threads override the preset values
    if ARGS.preset == "full":
        _d_mem, _d_thr = 8.0, 8
    else:                          # conserve + laptop + server
        _d_mem, _d_thr = 6.0, 4
    ARGS.mem_limit_gb = ARGS.mem_limit_gb if ARGS.mem_limit_gb is not None \
        else _d_mem
    ARGS.threads = ARGS.threads if ARGS.threads is not None else _d_thr
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"[serve] OpenAI-compatible API on http://{ARGS.host}:{ARGS.port} "
          f"(model {ARGS.model})", file=sys.stderr, flush=True)
    print(f"[serve] engine {os.path.abspath(ARGS.salt)} | single-flight "
          f"serialized | chunked prefill B=512 | preset {ARGS.preset} "
          f"(mem {ARGS.mem_limit_gb}GB, {ARGS.threads} threads)",
          file=sys.stderr, flush=True)
    # EAGER engine spawn: bring the engine up NOW so the operator can
    # monitor it (pid, RSS) before the first request -- lazy spawn
    # means no salt process exists until something is sent.
    try:
        eng = get_engine()
        rss = eng.engine_rss_mb()
        print(f"[serve] engine UP pid={eng._proc.pid} "
              f"rss={rss}MB (floor profile: pin 0, zerocopy, "
              f"kv-budget {os.environ.get('SALT_KV_BUDGET_GB', '0.5')}GB)",
              file=sys.stderr, flush=True)
        print(f"[serve] monitor: top -pid {eng._proc.pid} "
              f"-stats pid,cpu,mem,rprvt,vsize",
              file=sys.stderr, flush=True)
        if _SESSION_TTL > 0:
            threading.Thread(target=_session_sweeper, daemon=True).start()
            print(f"[serve] session idle TTL {_SESSION_TTL:.0f}s "
                  f"(SALT_SESSION_TTL; 0 disables)",
                  file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[serve] engine spawn failed: {e}", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
