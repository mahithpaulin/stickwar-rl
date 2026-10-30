#!/usr/bin/env python3
"""Validate Kaggle GPU smoke artifacts and print a GitHub step summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/kaggle-smoke")
metrics_path = out / "metrics.json"
failures = []
if not metrics_path.exists():
    failures.append(f"missing {metrics_path}")
    data = {}
else:
    data = json.loads(metrics_path.read_text())
gpu = data.get("gpu", {})
updates = data.get("updates", [])
if not gpu.get("cuda_available"):
    failures.append("CUDA was not available")
if not gpu.get("gpu_name"):
    failures.append("no GPU name recorded")
if not gpu.get("gpu_matmul_mean") and gpu.get("gpu_matmul_mean") != 0.0:
    failures.append("no GPU matmul result")
if len(updates) != 3:
    failures.append(f"expected 3 PPO updates, got {len(updates)}")
ckpt = out / "checkpoint.pt"
if not ckpt.exists() or ckpt.stat().st_size < 1024:
    failures.append("checkpoint.pt missing or too small")

print("## Kaggle GPU smoke")
print("")
print(f"- CUDA available: `{gpu.get('cuda_available')}`")
print(f"- GPU: `{gpu.get('gpu_name')}` ({gpu.get('total_vram_gb')} GB, capability {gpu.get('gpu_capability')})")
print(f"- torch: `{gpu.get('torch')}`, CUDA: `{gpu.get('cuda_version')}`")
print(f"- updates: {len(updates)}")
print(f"- GPU matmul mean: `{gpu.get('gpu_matmul_mean')}`")
print(f"- elapsed: {gpu.get('elapsed_sec')} s")
print(f"- checkpoint: {ckpt.stat().st_size if ckpt.exists() else 0} bytes")
print("")
if failures:
    print("### FAILURES")
    for failure in failures:
        print(f"- {failure}")
    print("KAGGLE_GPU_SMOKE_FAILED")
    raise SystemExit(1)
print("KAGGLE_GPU_SMOKE_OK")
