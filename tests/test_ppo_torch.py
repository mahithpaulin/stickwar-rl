import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stickwar_rl.ppo import PPOConfig, PPOTrainer, tensors_for_training
from stickwar_rl.train import make_synthetic


def test_update_and_checkpoint_roundtrip(tmp_path: Path):
    arrays = make_synthetic(n=96, n_actions=11, obs_size=32, frame_stack=2, seed=1)
    config = PPOConfig(obs_channels=2, obs_size=32, n_actions=11, epochs=2, minibatch_size=32, hidden=32, seed=1)
    trainer = PPOTrainer(config, device="cpu", seed=1)
    metrics = trainer.update(arrays)
    assert metrics["samples"] == 96
    assert np.isfinite(metrics["loss"])
    assert trainer.update_count == 1

    path = trainer.save(tmp_path / "checkpoint.pt", extra={"note": "test"})
    assert path.exists()
    loaded = PPOTrainer.load(path, device="cpu")
    assert loaded.update_count == 1
    assert loaded.config.n_actions == 11
    assert loaded.metadata["note"] == "test"

    obs = arrays["obs"][0]
    action, logp, value = loaded.act(obs)
    assert 0 <= action < 11
    assert np.isfinite(logp) and np.isfinite(value)


def test_resume_continues(tmp_path: Path):
    arrays = make_synthetic(n=64, n_actions=11, obs_size=32, frame_stack=2, seed=2)
    config = PPOConfig(obs_channels=2, obs_size=32, n_actions=11, epochs=1, minibatch_size=32, hidden=32, seed=2)
    trainer = PPOTrainer(config, device="cpu", seed=2)
    trainer.update(arrays)
    path = trainer.save(tmp_path / "latest.pt")
    resumed = PPOTrainer.load(path, device="cpu")
    resumed.update(arrays)
    assert resumed.update_count == 2


def test_synthetic_arrays():
    arrays = make_synthetic(n=64, n_actions=11, obs_size=16, frame_stack=2, seed=3)
    assert arrays["obs"].shape == (64, 2, 16, 16)
    assert arrays["actions"].shape == (64,)
    assert arrays["advantages"].shape == (64,)
    assert arrays["returns"].shape == (64,)
