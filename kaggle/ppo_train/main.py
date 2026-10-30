#!/usr/bin/env python3
"""Kaggle GPU training job for batch PPO.

The repository is cloned at an exact commit (public), so code and data are
versioned independently. Episodes are supplied as a private Kaggle dataset;
checkpoints can be resumed from a public GitHub release asset. Every path is
verified and failures are surfaced instead of replaced with synthetic results.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
OVERRIDES = {}


def cfg(key, default):
    if key in OVERRIDES:
        return OVERRIDES[key]
    return os.environ.get(key, default)


REPO_URL = cfg("STICKWAR_GIT_REPO", "https://github.com/mahithpaulin/stickwar-rl.git")
REF = cfg("STICKWAR_GIT_REF", "main")
WORK = Path("/tmp/stickwar-rl")


def log(**kw):
    print(json.dumps(kw, default=str), flush=True)


def ensure_repo():
    if not (WORK / "stickwar_rl").exists():
        log(event="clone", repo=REPO_URL, ref=REF)
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(WORK)])
    sys.path.insert(0, str(WORK))
    import torch
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": torch.cuda.get_device_name(0),
                     "capability": f"{p.major}.{p.minor}",
                     "vram_gb": round(p.total_memory / 1024**3, 2)})
    (OUT / "environment.json").write_text(json.dumps(info, indent=2) + "\n")
    log(event="environment", **info)
    return torch


def find_data():
    requested = str(cfg("STICKWAR_DATA", "")).strip()
    candidates = [Path(requested)] if requested else []
    candidates += [
        Path("/kaggle/input/stickwar-episodes/episodes"),
        Path("/kaggle/input/stickwar-episodes"),
    ]
    for base in candidates:
        if not base.exists():
            continue
        for child in base.rglob("meta.json"):
            return child.parent.parent if child.parent.name == "frames" else child.parent
    zips = list(Path("/kaggle/input").rglob("*.zip")) if Path("/kaggle/input").exists() else []
    if zips:
        target = Path("/tmp/episodes")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(target)
        for child in target.rglob("meta.json"):
            return child.parent.parent if child.parent.name == "frames" else child.parent
    return None


def download_resume():
    url = str(cfg("STICKWAR_RESUME_URL", "")).strip()
    if not url:
        return None
    dest = OUT / "resume_checkpoint.pt"
    log(event="resume_download", url=url)
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 1024:
        raise RuntimeError("downloaded checkpoint is implausibly small")
    return dest


def main():
    torch = ensure_repo()
    allow_synth = bool(cfg("STICKWAR_ALLOW_SYNTHETIC", False))
    updates = int(cfg("STICKWAR_UPDATES", 100))
    data_dir = find_data()
    if data_dir is None and not allow_synth:
        raise SystemExit("FAIL: no episode dataset found under /kaggle/input and synthetic mode is not allowed")
    resume = download_resume()

    from stickwar_rl.train import main as train_main

    args = [
        "--out", str(OUT / "run"),
        "--updates", str(updates),
        "--device", "cuda",
        "--require-cuda",
        "--seed", str(cfg("STICKWAR_SEED", 0)),
        "--save-every", "1",
    ]
    if data_dir is not None:
        args += ["--data", str(data_dir)]
        if bool(cfg("STICKWAR_ALLOW_UNCALIBRATED_REWARD", False)):
            args += ["--allow-uncalibrated-reward"]
    else:
        args += ["--synthetic", "--synthetic-samples", str(cfg("STICKWAR_SYNTHETIC_SAMPLES", 512))]
    if resume is not None:
        args += ["--resume", str(resume)]

    log(event="train_start", data=str(data_dir) if data_dir else "synthetic",
        resume=str(resume) if resume else None, updates=updates)
    rc = train_main(args)
    if rc != 0:
        raise SystemExit(f"FAIL: training returned {rc}")

    run = OUT / "run"
    for name in ("summary.json", "metrics.jsonl", "run_config.json", "system.json"):
        src = run / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    ckpt = run / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        raise SystemExit("FAIL: training finished without latest.pt")
    shutil.copy2(ckpt, OUT / "checkpoint.pt")

    from stickwar_rl.evaluate import offline_eval
    from stickwar_rl.metrics import render_report

    if data_dir is not None:
        try:
            offline_eval(
                ckpt, data_dir, OUT / "eval",
                val_fraction=float(cfg("STICKWAR_VAL_FRACTION", 0.2)),
                allow_uncalibrated=bool(cfg("STICKWAR_ALLOW_UNCALIBRATED_REWARD", False)),
                device="cuda",
            )
        except Exception as exc:
            log(event="eval_error", error=f"{type(exc).__name__}: {exc}")
            raise
    report = render_report(run)
    log(event="report", metrics=report["metrics_count"])
    summary = json.loads((OUT / "summary.json").read_text())
    summary["artifacts"] = {
        "checkpoint": "checkpoint.pt",
        "metrics": "metrics.jsonl",
        "environment": "environment.json",
        "eval": "eval/eval.json" if (OUT / "eval" / "eval.json").exists() else None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(event="done", **summary)
    print("KAGGLE_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
