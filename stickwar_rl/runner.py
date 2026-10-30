"""Episode collection loop shared by local devices and GitHub Actions."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import EpisodeWriter, new_episode_id, utc_now, write_manifest
from .env import AndroidEnv
from .observations import resize_rgb
from .policies import Policy


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_episode(
    env: AndroidEnv,
    policy: Policy,
    out_root: str | Path,
    episode_id: str | None = None,
    max_steps: int | None = None,
    record_width: int = 256,
    notes: str = "",
    console: bool = True,
) -> dict[str, Any]:
    episode_id = episode_id or new_episode_id()
    episode_dir = Path(out_root) / episode_id
    obs = env.reset()
    meta = {
        "episode_id": episode_id,
        "created_at": utc_now(),
        "package": env.config.package,
        "activity": env.config.activity,
        "git_commit": _git_commit(),
        "policy": policy.name,
        "checkpoint": getattr(policy, "checkpoint", None),
        "action_space": env.action_space.to_dict(),
        "reward_config": env.reward_config.to_dict(),
        "env_config": {
            "frame_size": env.config.frame_size,
            "frame_stack": env.config.frame_stack,
            "step_interval_s": env.config.step_interval_s,
            "max_steps": env.config.max_steps,
            "screen_size": env.screen_size,
        },
        "device": {
            "serial": env.adb.serial,
            "android_sdk": env.adb.getprop("ro.build.version.sdk"),
            "model": env.adb.getprop("ro.product.model"),
            "hardware": env.adb.getprop("ro.hardware"),
        },
        "notes": notes,
    }
    writer = EpisodeWriter(episode_dir, meta=meta)
    steps = 0
    started = time.time()
    limit = max_steps or env.config.max_steps
    raw = env.last_raw
    try:
        while steps < limit:
            decision = policy.act(obs)
            result = env.step(decision.action)
            frame = resize_rgb(raw, width=record_width) if raw is not None else None
            writer.add(
                frame=frame,
                action=decision.action,
                action_name=env.action_space[decision.action].name,
                reward=result.reward,
                done=result.done,
                info=result.info,
                logp=decision.logp,
                value=decision.value,
                signals=result.info.get("signals", {}),
            )
            raw = env.last_raw
            obs = result.obs
            steps += 1
            if console and (steps % 10 == 0 or result.done):
                print(json.dumps({
                    "episode": episode_id,
                    "step": steps,
                    "reward": round(result.reward, 4),
                    "done": result.done,
                    "action": env.action_space[decision.action].name,
                    "elapsed_s": round(time.time() - started, 2),
                }), flush=True)
            if result.done:
                break
    except Exception as exc:
        writer.close({"error": f"{type(exc).__name__}: {exc}"})
        raise
    summary = writer.close({
        "elapsed_s": round(time.time() - started, 3),
        "steps_requested": limit,
    })
    if console:
        print(json.dumps({"episode_summary": summary.get("summary", {}), "episode": episode_id}), flush=True)
    return summary


def collect(
    env: AndroidEnv,
    policy: Policy,
    out_root: str | Path,
    episodes: int = 1,
    max_steps: int | None = None,
    record_width: int = 256,
) -> dict[str, Any]:
    results = []
    for i in range(episodes):
        results.append(run_episode(
            env, policy, out_root, max_steps=max_steps, record_width=record_width,
            notes=f"collection episode {i + 1}/{episodes}",
        ))
    manifest = write_manifest(out_root)
    return {"episodes": results, "manifest": manifest}
