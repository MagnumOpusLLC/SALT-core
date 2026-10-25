#!/usr/bin/env python3
"""Authenticate the pinned published Gemma 4 Q4_0 assistant."""
import argparse
import hashlib
import json
import os
import pathlib
import struct

REPOSITORY = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
REVISION = "7b92b5b28818151e8669af2e45e88d6086f490dd"
FILENAME = "MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf"
BYTES = 251_939_328
SHA256 = "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                return value.hexdigest()
            value.update(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    directory = pathlib.Path(args.artifact_dir).resolve()
    model = directory / "mtp-gemma-4-26B-A4B-it-Q4_0.gguf"
    if model.is_symlink() or not model.is_file():
        raise SystemExit("Q4 assistant must be a regular non-symlink file")
    stat = model.stat()
    if stat.st_size != BYTES:
        raise SystemExit(f"size mismatch: {stat.st_size}")
    with model.open("rb") as source:
        magic, version = struct.unpack("<4sI", source.read(8))
    if magic != b"GGUF" or version != 3:
        raise SystemExit(f"GGUF header mismatch: {magic!r} v{version}")
    sha = digest(model)
    if sha != SHA256:
        raise SystemExit(f"sha256 mismatch: {sha}")
    receipt = directory / "source-receipt.txt"
    receipt.write_text(
        "SALT_GEMMA4_MTP_Q4_RECEIPT_V1\n"
        f"revision {REVISION}\n"
        f"sha256 {sha}\n"
        f"dev {stat.st_dev}\n"
        f"ino {stat.st_ino}\n"
        f"size {stat.st_size}\n"
        f"mtime_ns {stat.st_mtime_ns}\n"
        "END\n"
    )
    os.chmod(receipt, 0o600)
    print(json.dumps({"status":"PASS","path":str(model),"receipt":str(receipt),
                      "bytes":stat.st_size,"sha256":sha,"revision":REVISION}, sort_keys=True))


if __name__ == "__main__":
    main()
