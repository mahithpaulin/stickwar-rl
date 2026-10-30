"""Policies used by the Android actor and evaluation scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class Decision:
    action: int
    logp: float
    value: float


class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray) -> Decision: ...


class RandomPolicy:
    name = "random"

    def __init__(self, n_actions: int, seed: int = 0):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> Decision:
        action = int(self.rng.integers(0, self.n_actions))
        return Decision(action=action, logp=-math.log(self.n_actions), value=0.0)


class TorchPolicy:
    name = "torch-checkpoint"

    def __init__(self, checkpoint: str | Path, device: str = "cpu", deterministic: bool = False):
        from .ppo import PPOTrainer

        self.trainer = PPOTrainer.load(checkpoint, device=device, resume_optimizer=False)
        self.n_actions = self.trainer.config.n_actions
        self.deterministic = deterministic
        self.checkpoint = str(checkpoint)

    def act(self, obs: np.ndarray) -> Decision:
        action, logp, value = self.trainer.act(obs, deterministic=self.deterministic)
        return Decision(action=action, logp=logp, value=value)


def make_policy(spec: str, n_actions: int, seed: int = 0, device: str = "cpu") -> Policy:
    if spec == "random" or not spec:
        return RandomPolicy(n_actions=n_actions, seed=seed)
    if spec.startswith("checkpoint:"):
        path = spec.split(":", 1)[1]
        return TorchPolicy(path, device=device)
    if spec.endswith(".pt") or spec.endswith(".pth"):
        return TorchPolicy(spec, device=device)
    raise ValueError(f"unknown policy spec: {spec}")
