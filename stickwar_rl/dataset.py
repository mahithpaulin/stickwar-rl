"""On-disk episode format shared by Android actors, Kaggle, and CI.

Layout:
    episodes/<episode_id>/meta.json
    episodes/<episode_id>/steps.jsonl
    episodes/<episode_id>/frames/000000.jpg
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .observations import FrameStack, load_image, save_image


class DatasetError(ValueError):
    pass


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_episode_id(prefix: str = "ep") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"


class EpisodeWriter:
    def __init__(self, root: str | Path, meta: dict[str, Any], save_frames: bool = True, jpeg_quality: int = 82):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.root / "frames"
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.save_frames = save_frames
        self.jpeg_quality = jpeg_quality
        self.meta = dict(meta)
        self.meta.setdefault("episode_id", self.root.name)
        self.meta.setdefault("created_at", utc_now())
        self.meta.setdefault("format_version", 1)
        self.meta["steps"] = 0
        self._steps_path = self.root / "steps.jsonl"
        self._steps_file = self._steps_path.open("w", encoding="utf-8")
        self._closed = False
        self._summary = {"reward_sum": 0.0, "actions": {}, "frames": 0}
        self._write_meta()

    def _write_meta(self) -> None:
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    def add(
        self,
        frame: np.ndarray | None,
        action: int,
        action_name: str,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
        logp: float | None = None,
        value: float | None = None,
        signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise DatasetError("writer is closed")
        step_index = self.meta["steps"]
        frame_rel = None
        if frame is not None and self.save_frames:
            frame_rel = f"frames/{step_index:06d}.jpg"
            save_image(frame, self.root / frame_rel, quality=self.jpeg_quality)
            self._summary["frames"] += 1
        record = {
            "step": step_index,
            "timestamp": time.time(),
            "action": int(action),
            "action_name": action_name,
            "reward": float(reward),
            "done": bool(done),
            "logp": None if logp is None else float(logp),
            "value": None if value is None else float(value),
            "signals": signals or {},
            "frame": frame_rel,
            "info": info or {},
        }
        self._steps_file.write(json.dumps(record, default=str) + "\n")
        self._steps_file.flush()
        os.fsync(self._steps_file.fileno())
        self.meta["steps"] = step_index + 1
        self._summary["reward_sum"] += float(reward)
        key = action_name or str(action)
        self._summary["actions"][key] = self._summary["actions"].get(key, 0) + 1
        return record

    def close(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            return self.meta
        self._closed = True
        self._steps_file.close()
        self.meta["finished_at"] = utc_now()
        if extra:
            self.meta.update(extra)
        self.meta["summary"] = self._summary
        self._write_meta()
        (self.root / "summary.json").write_text(json.dumps(self._summary, indent=2) + "\n")
        return self.meta

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        extra = {"error": str(exc)} if exc else None
        self.close(extra=extra)


@dataclass
class Episode:
    path: Path
    meta: dict[str, Any]
    steps: list[dict[str, Any]]
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def episode_id(self) -> str:
        return str(self.meta.get("episode_id", self.path.name))

    def __len__(self) -> int:
        return len(self.steps)

    def summary(self) -> dict[str, Any]:
        rewards = [float(s.get("reward", 0.0)) for s in self.steps]
        return {
            "episode_id": self.episode_id,
            "steps": len(self.steps),
            "reward_sum": float(sum(rewards)),
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "finished": bool(self.meta.get("finished_at")),
            "frames": len(self.frames),
        }


def load_episode(path: str | Path, load_frames: bool = True) -> Episode:
    root = Path(path)
    meta_path = root / "meta.json"
    steps_path = root / "steps.jsonl"
    if not meta_path.exists() or not steps_path.exists():
        raise DatasetError(f"not an episode directory: {root}")
    meta = json.loads(meta_path.read_text())
    steps = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    frames: list[np.ndarray] = []
    if load_frames:
        for step in steps:
            frame_rel = step.get("frame")
            if not frame_rel:
                raise DatasetError(f"{root}: step {step.get('step')} has no frame")
            frame_path = root / frame_rel
            if not frame_path.exists():
                raise DatasetError(f"missing frame: {frame_path}")
            frames.append(load_image(frame_path))
    return Episode(path=root, meta=meta, steps=steps, frames=frames)


def iter_episodes(root: str | Path, require_finished: bool = False) -> Iterator[Episode]:
    base = Path(root)
    if not base.exists():
        raise DatasetError(f"dataset directory does not exist: {base}")
    candidates = sorted(p for p in base.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not candidates and (base / "meta.json").exists():
        candidates = [base]
    for path in candidates:
        episode = load_episode(path, load_frames=False)
        if require_finished and not episode.meta.get("finished_at"):
            continue
        yield episode


def write_manifest(root: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    base = Path(root)
    episodes = []
    total_steps = 0
    total_reward = 0.0
    for episode in iter_episodes(base):
        summary = episode.summary()
        episodes.append(summary)
        total_steps += summary["steps"]
        total_reward += summary["reward_sum"]
    manifest = {
        "format_version": 1,
        "created_at": utc_now(),
        "episodes": episodes,
        "episode_count": len(episodes),
        "step_count": total_steps,
        "reward_sum": total_reward,
    }
    if out is not None:
        Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def transition_arrays(
    episodes: Iterable[Episode],
    frame_stack: int = 4,
    obs_size: int = 84,
) -> dict[str, np.ndarray]:
    obs_all: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    old_logp: list[float] = []
    old_values: list[float] = []
    episode_ids: list[str] = []
    steps: list[int] = []
    missing_logp = 0
    for episode in episodes:
        if not episode.frames:
            episode = load_episode(episode.path, load_frames=True)
        stack = FrameStack(frame_stack, obs_size)
        stack.reset(episode.frames[0])
        for i, step in enumerate(episode.steps):
            if i > 0:
                stack.push(episode.frames[i])
            obs_all.append(stack.stack())
            actions.append(int(step["action"]))
            rewards.append(float(step.get("reward", 0.0)))
            dones.append(bool(step.get("done", False)) or i == len(episode.steps) - 1)
            if step.get("logp") is None:
                missing_logp += 1
                old_logp.append(float("nan"))
            else:
                old_logp.append(float(step["logp"]))
            old_values.append(float(step.get("value") or 0.0))
            episode_ids.append(episode.episode_id)
            steps.append(i)
    if not obs_all:
        raise DatasetError("no transitions found")
    return {
        "obs": np.stack(obs_all).astype(np.uint8),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "old_logp": np.asarray(old_logp, dtype=np.float32),
        "old_values": np.asarray(old_values, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids),
        "step_index": np.asarray(steps, dtype=np.int32),
        "missing_logp": missing_logp,
    }


def split_episode_ids(root: str | Path, val_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    ids = sorted(ep.episode_id for ep in iter_episodes(root))
    if not ids:
        raise DatasetError("no episodes to split")
    import random

    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    if len(ids) > 1:
        n_val = max(1, min(n_val, len(ids) - 1))
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:]) if n_val else sorted(ids)
    return train, val


def load_selected(root: str | Path, ids: Iterable[str]) -> list[Episode]:
    wanted = set(ids)
    result = []
    for episode in iter_episodes(root):
        if episode.episode_id in wanted:
            result.append(load_episode(episode.path, load_frames=True))
    return result


def dataset_fingerprint(root: str | Path) -> str:
    digest = hashlib.sha256()
    for episode in iter_episodes(root):
        digest.update(episode.episode_id.encode())
        digest.update((episode.path / "steps.jsonl").read_bytes())
    return digest.hexdigest()
