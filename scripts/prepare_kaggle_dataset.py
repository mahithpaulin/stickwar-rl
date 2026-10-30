#!/usr/bin/env python3
"""Stage an episodes directory as a private Kaggle dataset folder."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stickwar_rl.dataset import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-id", default="mahithpaulin/stickwar-episodes")
    parser.add_argument("--title", default="stickwar episodes")
    parser.add_argument("--description", default="Private Stick War: Legacy ADB episodes for batch PPO research.")
    args = parser.parse_args()

    src = Path(args.episodes)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    target = out / "episodes"
    shutil.copytree(src, target)
    manifest = write_manifest(target)
    if manifest["episode_count"] < 1:
        raise SystemExit("no episodes found")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {
        "title": args.title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
        "description": args.description,
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"ok": True, "dataset_id": args.dataset_id, "out": str(out),
                      "episodes": manifest["episode_count"], "steps": manifest["step_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
