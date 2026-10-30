import json
from pathlib import Path

import numpy as np
import pytest

from stickwar_rl.dataset import (
    DatasetError,
    EpisodeWriter,
    dataset_fingerprint,
    iter_episodes,
    load_episode,
    new_episode_id,
    split_episode_ids,
    transition_arrays,
    write_manifest,
)


def make_episode(root: Path, episode_id: str, steps: int = 3, frames: bool = True):
    rng = np.random.default_rng(0)
    meta = {"episode_id": episode_id, "package": "com.example", "reward_config": {"calibrated": True}}
    with EpisodeWriter(root / episode_id, meta=meta, save_frames=frames) as writer:
        for i in range(steps):
            frame = rng.integers(0, 255, size=(16, 24, 3), dtype=np.uint8) if frames else None
            writer.add(frame=frame, action=i % 2, action_name="a", reward=0.5,
                       done=i == steps - 1, logp=-0.7, value=0.1,
                       signals={"enemy_hp": 1.0 - i * 0.1})
    return root / episode_id


def test_writer_reader_roundtrip(tmp_path: Path):
    episode_dir = make_episode(tmp_path, "ep-1", steps=3, frames=False)
    episode = load_episode(episode_dir, load_frames=False)
    assert len(episode) == 3
    assert episode.meta["steps"] == 3
    assert episode.meta["finished_at"]
    assert episode.steps[0]["action_name"] == "a"
    assert episode.steps[-1]["done"] is True


def test_manifest_and_split(tmp_path: Path):
    for i in range(3):
        make_episode(tmp_path, f"ep-{i}", steps=2, frames=False)
    manifest = write_manifest(tmp_path)
    assert manifest["episode_count"] == 3
    assert manifest["step_count"] == 6
    train, val = split_episode_ids(tmp_path, val_fraction=0.34, seed=1)
    assert len(train) == 2 and len(val) == 1
    assert set(train) | set(val) == {"ep-0", "ep-1", "ep-2"}
    with pytest.raises(DatasetError):
        split_episode_ids(tmp_path / "missing")


def test_transition_arrays(tmp_path: Path):
    make_episode(tmp_path, "ep-a", steps=4, frames=True)
    make_episode(tmp_path, "ep-b", steps=3, frames=True)
    episodes = list(iter_episodes(tmp_path))
    arrays = transition_arrays(episodes, frame_stack=2, obs_size=8)
    assert arrays["obs"].shape == (7, 2, 8, 8)
    assert arrays["actions"].shape == (7,)
    assert arrays["rewards"].shape == (7,)
    assert arrays["dones"].sum() == 2
    assert arrays["missing_logp"] == 0
    assert np.allclose(arrays["old_logp"], -0.7)


def test_fingerprint_stable(tmp_path: Path):
    make_episode(tmp_path, "ep-a", steps=2, frames=False)
    a = dataset_fingerprint(tmp_path)
    b = dataset_fingerprint(tmp_path)
    assert a == b


def test_new_episode_id_unique():
    assert new_episode_id() != new_episode_id()
