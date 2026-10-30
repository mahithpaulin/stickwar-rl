#!/usr/bin/env python3
"""Push/poll/download a Kaggle kernel without depending on a hosted runner.

Authentication is read by the Kaggle CLI from ~/.kaggle/access_token,
KAGGLE_USERNAME/KAGGLE_KEY, or the standard Kaggle config directory. This
script never reads or prints the token itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def kaggle_cmd() -> list[str]:
    return [sys.executable, "-m", "kaggle"]


def run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        kaggle_cmd() + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def normalize_status(line: str) -> str:
    line = line.strip()
    for status in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED", "PENDING", "STARTING"):
        if status in line.upper():
            return status
    return "UNKNOWN"


def read_metadata(job: str | Path) -> dict:
    path = Path(job) / "kernel-metadata.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = json.loads(path.read_text())
    code = Path(job) / data.get("code_file", "")
    if not data.get("id"):
        raise SystemExit("kernel-metadata.json has no id")
    if not code.exists():
        raise SystemExit(f"missing code file: {code}")
    if not data.get("enable_gpu"):
        raise SystemExit("metadata does not enable GPU")
    return data


def push(job: str | Path, accelerator: str | None = None, dry_run: bool = False) -> str:
    meta = read_metadata(job)
    args = ["kernels", "push", "-p", str(job)]
    if accelerator:
        args += ["--accelerator", accelerator]
    print(json.dumps({"event": "push", "kernel": meta["id"], "job": str(job), "accelerator": accelerator}), flush=True)
    if dry_run:
        return meta["id"]
    proc = run(args, timeout=600)
    print(proc.stdout.strip(), flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"kaggle push failed with {proc.returncode}")
    return meta["id"]


def status(kernel: str) -> str:
    proc = run(["kernels", "status", kernel], timeout=120)
    lines = [line for line in proc.stdout.splitlines() if line.strip() and "outdated" not in line.lower()]
    if proc.returncode != 0:
        print(json.dumps({"event": "status_error", "kernel": kernel, "output": proc.stdout[-1000:]}), flush=True)
        return "UNKNOWN"
    return normalize_status(lines[-1]) if lines else "UNKNOWN"


def wait(kernel: str, timeout_s: float = 3600, poll_s: float = 20) -> str:
    started = time.time()
    while True:
        state = status(kernel)
        elapsed = round(time.time() - started, 1)
        print(json.dumps({"event": "poll", "kernel": kernel, "status": state, "elapsed_s": elapsed}), flush=True)
        if state in {"COMPLETE", "ERROR", "CANCEL"}:
            return state
        if elapsed > timeout_s:
            return "TIMEOUT"
        time.sleep(poll_s)


def output(kernel: str, out: str | Path) -> Path:
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    proc = run(["kernels", "output", kernel, "-p", str(path), "-o"], timeout=900)
    print(proc.stdout.strip(), flush=True)
    if proc.returncode != 0:
        print(json.dumps({"event": "output_error", "returncode": proc.returncode}), flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push")
    p_push.add_argument("--job", required=True)
    p_push.add_argument("--accelerator", default=None)
    p_push.add_argument("--wait", action="store_true")
    p_push.add_argument("--timeout", type=float, default=3600)
    p_push.add_argument("--poll", type=float, default=20)
    p_push.add_argument("--download", default=None)
    p_push.add_argument("--dry-run", action="store_true")
    p_status = sub.add_parser("status")
    p_status.add_argument("--kernel", required=True)
    p_out = sub.add_parser("output")
    p_out.add_argument("--kernel", required=True)
    p_out.add_argument("--out", required=True)
    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--job", required=True)
    args = parser.parse_args(argv)

    if args.command == "validate":
        meta = read_metadata(args.job)
        print(json.dumps({"ok": True, "kernel": meta["id"], "gpu": meta.get("enable_gpu"),
                          "internet": meta.get("enable_internet"), "datasets": meta.get("dataset_sources", [])}, indent=2))
        return 0
    if args.command == "status":
        state = status(args.kernel)
        print(json.dumps({"kernel": args.kernel, "status": state}))
        return 0 if state == "COMPLETE" else 1
    if args.command == "output":
        output(args.kernel, args.out)
        return 0

    kernel = push(args.job, args.accelerator, args.dry_run)
    if args.dry_run:
        return 0
    state = wait(kernel, args.timeout, args.poll) if args.wait else status(kernel)
    if args.download:
        output(kernel, args.download)
    print(json.dumps({"event": "final", "kernel": kernel, "status": state}), flush=True)
    return 0 if state in {"COMPLETE", "RUNNING", "QUEUED", "PENDING", "STARTING", "UNKNOWN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
