#!/usr/bin/env python3
"""Validate Kaggle training artifacts and print a step summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/kaggle-train-out")
summary_path = out / "summary.json"
env_path = out / "environment.json"
failures = []
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
env = json.loads(env_path.read_text()) if env_path.exists() else {}
updates = summary.get("updates_completed", 0)
if not summary:
    failures.append("missing summary.json")
if summary.get("status") != "completed":
    failures.append(f"training status is {summary.get('status')!r}")
if not env.get("cuda_available"):
    failures.append("CUDA was not available in the Kaggle environment")
ckpt = out / "checkpoint.pt"
if not ckpt.exists() or ckpt.stat().st_size < 1024:
    failures.append("checkpoint.pt missing or too small")
print("## Kaggle PPO training")
print("")
print(f"- status: `{summary.get('status')}`")
print(f"- updates completed: {updates} / {summary.get('updates_requested')}")
print(f"- GPU: `{env.get('gpu')}` ({env.get('vram_gb')} GB)")
print(f"- torch: `{env.get('torch')}`, CUDA: `{env.get('cuda_version')}`")
print(f"- data: `{summary.get('data', {}).get('data_dir')}`")
print(f"- mode: `{summary.get('mode')}`")
print(f"- checkpoint bytes: {ckpt.stat().st_size if ckpt.exists() else 0}")
if (out / "eval" / "eval.json").exists():
    ev = json.loads((out / "eval" / "eval.json").read_text())
    print(f"- offline eval clipped IPS: policy `{ev.get('policy', {}).get('clipped_ips_reward')}` "
          f"vs baseline `{ev.get('untrained_baseline', {}).get('clipped_ips_reward')}`")
print("")
if failures:
    print("### FAILURES")
    for failure in failures:
        print(f"- {failure}")
    print("KAGGLE_TRAIN_FAILED")
    raise SystemExit(1)
print("KAGGLE_TRAIN_OK")
