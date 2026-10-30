"""CLI: collect episodes from a connected Android device or emulator."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .actions import ActionSpace
from .adb import ADB
from .env import AndroidEnv, AndroidEnvConfig
from .policies import make_policy
from .runner import collect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env.emulator.json")
    parser.add_argument("--out", default="data/episodes")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.0, help="episode seconds; 0 uses config max_steps")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--policy", default="random", help="'random' or path to a .pt checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--package", default=None)
    parser.add_argument("--activity", default=None)
    parser.add_argument("--action-space", default=None)
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--reset-macro", default=None)
    parser.add_argument("--step-interval", type=float, default=None)
    parser.add_argument("--frame-size", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--record-width", type=int, default=256)
    parser.add_argument("--allow-uncalibrated-reward", action="store_true")
    parser.add_argument("--notes", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AndroidEnvConfig.load(args.config)
    updates = {
        "serial": args.serial,
        "package": args.package,
        "activity": args.activity,
        "action_space_path": args.action_space,
        "reward_config_path": args.reward_config,
        "reset_macro_path": args.reset_macro,
        "step_interval_s": args.step_interval,
        "frame_size": args.frame_size,
        "frame_stack": args.frame_stack,
        "seed": args.seed,
        "allow_uncalibrated_reward": args.allow_uncalibrated_reward,
    }
    config = replace(config, **{k: v for k, v in updates.items() if v is not None})

    adb = ADB(serial=config.serial, adb_path=config.adb_path)
    devices = adb.devices()
    ready = [d for d in devices if d.state == "device"]
    if not ready:
        print(json.dumps({"error": "no adb device in 'device' state", "devices": [d.__dict__ for d in devices]}, indent=2))
        return 2
    if config.serial is None and len(ready) > 1:
        print(json.dumps({"error": "multiple adb devices; pass --serial", "devices": [d.serial for d in ready]}), file=sys.stderr)
        return 2
    if not adb.is_installed(config.package):
        print(json.dumps({"error": f"package not installed: {config.package}"}), file=sys.stderr)
        return 3

    space = ActionSpace.load(config.action_space_path)
    policy = make_policy(args.policy, n_actions=space.n, seed=args.seed)
    max_steps = args.max_steps or None
    if args.duration and args.duration > 0:
        max_steps = max(1, int(round(args.duration / max(0.05, config.step_interval_s))))

    env = AndroidEnv(config)
    try:
        result = collect(
            env=env,
            policy=policy,
            out_root=args.out,
            episodes=args.episodes,
            max_steps=max_steps,
            record_width=args.record_width,
        )
    finally:
        env.close()
    print(json.dumps({"ok": True, "out": str(Path(args.out)), "manifest": result["manifest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
