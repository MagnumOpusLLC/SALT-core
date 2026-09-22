#!/usr/bin/env python3
"""salt-chat.py -- thin chat client for the salt API server.

The engine and all session memory live in server/serve.py (the API
server). This script is ONLY a display client: it sends messages to
POST /v1/chat/completions (OpenAI-compatible, SSE streaming) and
renders the stream. /cache and /reset are the server's session
endpoints (GET/DELETE /v1/sessions/<id>).

Usage:
  python3 tools/salt-chat.py [--api http://127.0.0.1:8090]
                             [--session demoChat] [--gen 64]
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_config import load_engine_config

CFG = load_engine_config()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=os.environ.get(
        "SALT_API", f"http://127.0.0.1:{os.environ.get('SALT_API_PORT', '8090')}"))
    p.add_argument("--session", default=None,
                   help="session id (server-side live KV memory)")
    p.add_argument("--gen", type=int, default=int(CFG.get("CHAT_GEN", "512")),
                   help="max tokens per turn")
    p.add_argument("--strip-think", action="store_true",
                   help="hide the <think> reasoning block (server-side "
                        "filter); default: stream the reasoning trace")
    p.add_argument("--think-mode", action="store_true",
                   help="the real thinking mode (the trace streams)")
    p.add_argument("--max-think", type=int, default=None,
                   help="cap the think-trace's tokens (e.g. 200)")
    p.add_argument("--model", default=CFG.get("MODEL_ID",
                                              "salt-qwen3.6-35B-A3B"))
    return p


def api_post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def api_get(url):
    return urllib.request.urlopen(url, timeout=30)


def chat_once(args, text):
    """One turn: POST a chat completion, yield SSE text chunks."""
    body = {"model": args.model, "stream": True,
            "max_tokens": args.gen,
            "messages": [{"role": "user", "content": text}]}
    if args.session:
        body["session"] = args.session
    if args.strip_think:
        body["strip_think"] = True
    if args.think_mode:
        body["think_mode"] = True
    if args.max_think:
        body["max_think"] = args.max_think
    resp = api_post(args.api.rstrip("/") + "/v1/chat/completions", body)
    try:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                line = ev.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    # the engine streams the raw think block (the
                    # strip is a per-request choice); the client hides
                    # the closing tag so the trace reads cleanly
                    content = content.replace("</think>", "")
                    sys.stdout.write(content)
                    sys.stdout.flush()
    finally:
        resp.close()


def session_status(args):
    """Memory picture from the server's session endpoint."""
    url = args.api.rstrip("/") + f"/v1/sessions/{args.session}"
    try:
        with api_get(url) as r:
            st = json.loads(r.read().decode())
    except Exception as e:
        return f"(session status unavailable: {e})"
    return (f"  session:   {st.get('session')}\n"
            f"  rows:      {st.get('rows')} tokens in KV (live)\n"
            f"  cache:     {st.get('cache')}\n"
            f"  note:      {st.get('note')}")


def close_session(args):
    """Close the session on chat exit: the engine is killed and the
    session's KV/manifest removed. Cheap -- a resumed session reloads
    in ~1s (SALTKV01 load) -- so there is no reason to leave engines
    resident after the chat window closes. No-op for stateless."""
    if not args.session:
        return
    try:
        req = urllib.request.Request(
            args.api.rstrip("/") + f"/v1/sessions/{args.session}",
            method="DELETE")
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
        print(f"\nsalt-chat: session {args.session} closed (KV flushed)")
    except Exception as e:
        print(f"\nsalt-chat: (session close failed: {e})")


def main():
    args = build_parser().parse_args()
    print(f"salt-chat: API {args.api} | session "
          f"{args.session or '(stateless)'} | gen {args.gen}")
    if args.session:
        try:
            print(session_status(args))
        except Exception as e:
            print(f"(session status unavailable: {e})")
    print("you> ", end="", flush=True)
    try:
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line.strip() == "/quit":
                break
            if line.strip() == "/cache":
                print(session_status(args))
                print("you> ", end="", flush=True)
                continue
            if line.strip() == "/reset":
                try:
                    req = urllib.request.Request(
                        args.api.rstrip("/") + f"/v1/sessions/{args.session}",
                        method="DELETE")
                    with urllib.request.urlopen(req, timeout=30) as r:
                        print(json.loads(r.read().decode()).get("deleted"))
                except Exception as e:
                    print(f"(reset failed: {e})")
                print("you> ", end="", flush=True)
                continue
            if not line.strip():
                print("you> ", end="", flush=True)
                continue
            try:
                chat_once(args, line)
                sys.stdout.write("\nsalt> (turn done)\nyou> ")
                sys.stdout.flush()
            except Exception as e:
                print(f"\n(error: {e})\nyou> ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        close_session(args)


if __name__ == "__main__":
    main()
