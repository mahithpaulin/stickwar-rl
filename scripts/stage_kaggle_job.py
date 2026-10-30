#!/usr/bin/env python3
"""Stage a Kaggle kernel folder with pinned code/data/checkpoint settings."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="kaggle/ppo_train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--git-ref", default="main")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--resume-url", default="")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--synthetic-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-uncalibrated-reward", action="store_true")
    parser.add_argument("--kernel-id", default="mahithpaulin/stickwar-ppo-train")
    args = parser.parse_args()

    template = Path(args.template)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(template, out)
    code_path = out / "main.py"
    code = code_path.read_text()
    overrides = {
        "STICKWAR_GIT_REF": args.git_ref,
        "STICKWAR_RESUME_URL": args.resume_url,
        "STICKWAR_UPDATES": args.updates,
        "STICKWAR_SYNTHETIC_SAMPLES": args.synthetic_samples,
        "STICKWAR_SEED": args.seed,
        "STICKWAR_VAL_FRACTION": args.val_fraction,
        "STICKWAR_ALLOW_SYNTHETIC": bool(args.allow_synthetic),
        "STICKWAR_ALLOW_UNCALIBRATED_REWARD": bool(args.allow_uncalibrated_reward),
    }
    if args.dataset:
        slug = args.dataset[-1].split("/")[-1]
        overrides["STICKWAR_DATA"] = f"/kaggle/input/{slug}/episodes"
    literal = json.dumps(overrides, indent=2)
    if "OVERRIDES = {}" not in code:
        raise SystemExit("template does not contain the OVERRIDES placeholder")
    code_path.write_text(code.replace("OVERRIDES = {}", f"OVERRIDES = {literal}", 1))

    meta_path = out / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["id"] = args.kernel_id
    meta["title"] = args.kernel_id.split("/")[-1]
    meta["dataset_sources"] = args.dataset
    meta["enable_gpu"] = True
    meta["enable_internet"] = True
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    staged = {
        "kernel_id": args.kernel_id,
        "git_ref": args.git_ref,
        "datasets": args.dataset,
        "resume_url": args.resume_url,
        "updates": args.updates,
        "allow_synthetic": args.allow_synthetic,
        "allow_uncalibrated_reward": args.allow_uncalibrated_reward,
        "out": str(out),
    }
    (out / "stage.json").write_text(json.dumps(staged, indent=2) + "\n")
    print(json.dumps({"ok": True, **staged}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
