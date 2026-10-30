#!/usr/bin/env python3
"""Kaggle GPU smoke test for the Stick War RL project.

Verifies real CUDA execution (not just cuda.is_available), runs a tiny
PPO-shaped update on synthetic pixel observations, and writes artifacts
that the orchestrator downloads and records.
"""
import datetime
import json
import os
import platform
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

def log(msg):
    print(msg, flush=True)

sys_version = platform.python_version()
log("python: " + sys_version)
log("torch: " + torch.__version__)
cuda_ok = torch.cuda.is_available()
log("cuda_available: " + str(cuda_ok))
if not cuda_ok:
    raise SystemExit("FAIL: CUDA is not available; refusing to fake a GPU run")
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
info = {
    "python": sys_version,
    "torch": torch.__version__,
    "cuda_available": True,
    "cuda_version": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": f"{props.major}.{props.minor}",
    "total_vram_gb": round(props.total_memory / 1024**3, 2),
    "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
log(json.dumps(info, indent=2))

class TinyActorCritic(nn.Module):
    def __init__(self, n_actions=12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 4, 84, 84)).shape[1]
        self.body = nn.Sequential(nn.Linear(n, 128), nn.ReLU())
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(self.conv(x))
        return self.pi(h), self.v(h).squeeze(-1)

model = TinyActorCritic().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
B, T, A = 8, 16, 12
metrics = []
for it in range(3):
    x = torch.rand(B, T, 4, 84, 84, device=dev)
    actions = torch.randint(0, A, (B, T), device=dev)
    old_logp = F.log_softmax(torch.randn(B, T, A, device=dev), -1).gather(-1, actions[..., None]).squeeze(-1)
    adv = torch.randn(B, T, device=dev)
    ret = torch.randn(B, T, device=dev)
    flat = x.flatten(0, 1)
    logits, values = model(flat)
    logp = F.log_softmax(logits, -1).gather(-1, actions.flatten()[:, None]).squeeze(-1)
    ratio = (logp - old_logp.flatten()).exp()
    pg = -torch.min(ratio * adv.flatten(), torch.clamp(ratio, 0.8, 1.2) * adv.flatten()).mean()
    vf = F.mse_loss(values, ret.flatten())
    loss = pg + 0.5 * vf - 0.01 * F.log_softmax(logits, -1).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    row = {"iter": it, "loss": float(loss.detach()), "pg": float(pg.detach()),
           "vf": float(vf.detach()), "sec": round(time.time() - t0, 3)}
    metrics.append(row)
    log(json.dumps(row))

# Force a real GPU kernel and verify it produces finite output.
a = torch.randn(1024, 1024, device=dev)
b = a @ a
torch.cuda.synchronize()
assert torch.isfinite(b).all(), "non-finite GPU matmul"
info["gpu_matmul_mean"] = float(b.mean())
info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
info["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
info["elapsed_sec"] = round(time.time() - t0, 2)
torch.save({"model": model.state_dict(), "metrics": metrics, "gpu": info}, os.path.join(OUT, "checkpoint.pt"))
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump({"gpu": info, "updates": metrics}, f, indent=2)
log("WROTE checkpoint.pt metrics.json")
log("SMOKE_OK")
