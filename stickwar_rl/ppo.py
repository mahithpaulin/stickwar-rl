"""Small PPO actor-critic with checkpoint/resume support.

The initial deployment is *batch* PPO over episodes recorded by Android actors.
That is not strictly on-policy: a behavior-cloning anchor is included by default
to keep updates near the data-collecting policy. The distinction is recorded in
every checkpoint and report instead of being called on-policy PPO.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for PPO. Install with `pip install -r requirements.txt` "
            "or run training on Kaggle/CI."
        ) from exc
    return torch


@dataclass
class PPOConfig:
    obs_channels: int = 4
    obs_size: int = 84
    n_actions: int = 11
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64
    bc_coef: float = 0.1
    target_kl: float = 0.05
    seed: int = 0

    def validate(self) -> None:
        if self.obs_channels < 1 or self.obs_size < 8 or self.n_actions < 2:
            raise ValueError("invalid observation/action dimensions")
        if self.lr <= 0 or not (0 < self.gamma <= 1) or not (0 <= self.gae_lambda <= 1):
            raise ValueError("invalid PPO hyperparameters")
        if not (0 < self.clip_ratio < 1):
            raise ValueError("clip_ratio must be in (0, 1)")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("epochs and minibatch_size must be positive")


def gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, dones must have equal length")
    advantages = [0.0] * len(rewards)
    last_gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = [advantages[i] + values[i] for i in range(len(values))]
    return advantages, returns


def _build_model(torch, config: PPOConfig):
    nn = torch.nn

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(config.obs_channels, 16, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, config.obs_channels, config.obs_size, config.obs_size)
                n_flat = int(self.conv(dummy).shape[1])
            self.body = nn.Sequential(nn.Linear(n_flat, config.hidden), nn.ReLU())
            self.pi = nn.Linear(config.hidden, config.n_actions)
            self.v = nn.Linear(config.hidden, 1)
            self.apply(self._init)

        @staticmethod
        def _init(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        def forward(self, x):
            h = self.body(self.conv(x))
            return self.pi(h), self.v(h).squeeze(-1)

    return ActorCritic()


class PPOTrainer:
    def __init__(self, config: PPOConfig, device: str = "cpu", seed: int | None = None):
        config.validate()
        torch = _torch()
        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.seed = config.seed if seed is None else seed
        self._seed_all(self.seed)
        self.model = _build_model(torch, config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.update_count = 0
        self.sample_count = 0

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch = _torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _as_batch(self, arrays: dict[str, Any], indices: np.ndarray | None = None):
        torch = self.torch
        obs = torch.as_tensor(arrays["obs"], dtype=torch.float32, device=self.device) / 255.0
        actions = torch.as_tensor(arrays["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(arrays["old_logp"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(arrays["old_values"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(arrays["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(arrays["returns"], dtype=torch.float32, device=self.device)
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            obs, actions, old_logp = obs[idx], actions[idx], old_logp[idx]
            old_values, advantages, returns = old_values[idx], advantages[idx], returns[idx]
        return obs, actions, old_logp, old_values, advantages, returns

    def update(self, arrays: dict[str, Any]) -> dict[str, float]:
        torch = self.torch
        nn = torch.nn
        F = torch.nn.functional
        required = {"obs", "actions", "advantages", "returns"}
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        n = len(arrays["actions"])
        if n == 0:
            raise ValueError("empty batch")
        advantages = np.asarray(arrays["advantages"], dtype=np.float32)
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        work = dict(arrays)
        work["advantages"] = advantages
        if "old_logp" not in work or work["old_logp"] is None:
            work["old_logp"] = np.full(n, np.nan, dtype=np.float32)
        if "old_values" not in work or work["old_values"] is None:
            work["old_values"] = np.zeros(n, dtype=np.float32)

        with torch.no_grad():
            obs_all = torch.as_tensor(work["obs"], dtype=torch.float32, device=self.device) / 255.0
            if not np.isfinite(work["old_logp"]).all():
                logits, _ = self.model(obs_all)
                logp = F.log_softmax(logits, -1).gather(
                    -1, torch.as_tensor(work["actions"], dtype=torch.long, device=self.device)[:, None]
                ).squeeze(-1)
                work["old_logp"] = logp.detach().cpu().numpy()

        rng = np.random.default_rng(self.seed + self.update_count)
        sums = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clip_fraction": 0.0, "bc_loss": 0.0}
        batches = 0
        stop_early = False
        for _epoch in range(self.config.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.config.minibatch_size):
                idx = order[start:start + self.config.minibatch_size]
                obs, actions, old_logp, _old_values, adv, ret = self._as_batch(work, idx)
                logits, values = self.model(obs)
                logp = F.log_softmax(logits, -1).gather(-1, actions[:, None]).squeeze(-1)
                ratio = (logp - old_logp).exp()
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret)
                entropy = -(logits.softmax(-1) * F.log_softmax(logits, -1)).sum(-1).mean()
                bc_loss = F.cross_entropy(logits, actions)
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy
                    + self.config.bc_coef * bc_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (logp - old_logp)).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
                sums["loss"] += float(loss.detach())
                sums["policy_loss"] += float(policy_loss.detach())
                sums["value_loss"] += float(value_loss.detach())
                sums["entropy"] += float(entropy.detach())
                sums["bc_loss"] += float(bc_loss.detach())
                sums["approx_kl"] += float(approx_kl)
                sums["clip_fraction"] += float(clip_fraction)
                batches += 1
                if self.config.target_kl > 0 and approx_kl > self.config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        self.update_count += 1
        self.sample_count += n
        result = {k: v / max(1, batches) for k, v in sums.items()}
        result["samples"] = n
        result["batches"] = batches
        result["update"] = self.update_count
        result["early_stop"] = float(stop_early)
        return result

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            logits, value = self.model(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, -1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
            return int(action.item()), float(logp.item()), float(value.item())

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        torch = self.torch
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "extra": extra or {},
        }
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, out)
        return out

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", resume_optimizer: bool = True) -> "PPOTrainer":
        torch = _torch()
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        config = PPOConfig(**payload["config"])
        trainer = cls(config, device=device, seed=int(payload.get("seed", config.seed)))
        trainer.model.load_state_dict(payload["model_state"])
        if resume_optimizer and payload.get("optimizer_state"):
            trainer.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.update_count = int(payload.get("update_count", 0))
        trainer.sample_count = int(payload.get("sample_count", 0))
        if payload.get("torch_rng") is not None:
            try:
                torch.set_rng_state(payload["torch_rng"])
                np.random.set_state(payload["numpy_rng"])
                random.setstate(payload["python_rng"])
                if torch.cuda.is_available() and payload.get("cuda_rng"):
                    torch.cuda.set_rng_state_all(payload["cuda_rng"])
            except (ValueError, TypeError):
                pass
        trainer.metadata = payload.get("extra", {})
        return trainer

    @staticmethod
    def checkpoint_info(path: str | Path) -> dict[str, Any]:
        torch = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return {
            "format_version": payload.get("format_version"),
            "created_at": payload.get("created_at"),
            "config": payload.get("config"),
            "update_count": payload.get("update_count"),
            "sample_count": payload.get("sample_count"),
            "extra": payload.get("extra", {}),
        }


def tensors_for_training(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    old_logp: np.ndarray,
    old_values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> dict[str, Any]:
    advantages, returns = gae(
        rewards.astype(float).tolist(),
        old_values.astype(float).tolist(),
        dones.astype(bool).tolist(),
        last_value=0.0,
        gamma=gamma,
        lam=lam,
    )
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "old_logp": old_logp,
        "old_values": old_values,
        "advantages": np.asarray(advantages, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
    }
