from pathlib import Path

import pytest

from stickwar_rl.actions import ActionSpace
from stickwar_rl.env import AndroidEnvConfig, ResetMacro
from stickwar_rl.rewards import RewardConfig

ROOT = Path(__file__).resolve().parent.parent


def test_action_config():
    space = ActionSpace.load(ROOT / "configs/actions.stickwar.json")
    assert space.n == 11
    assert space.source.startswith("play-store")
    assert space.names[0] == "noop"


def test_reward_config_is_honestly_uncalibrated():
    cfg = RewardConfig.load(ROOT / "configs/reward.stickwar.json")
    assert cfg.calibrated is False
    assert cfg.bars == []
    assert cfg.win_bonus == 20.0


def test_env_config_aliases():
    cfg = AndroidEnvConfig.load(ROOT / "configs/env.emulator.json")
    assert cfg.package == "com.maxgames.stickwarlegacy"
    assert cfg.action_space_path.endswith("actions.stickwar.json")
    assert cfg.reward_config_path.endswith("reward.stickwar.json")
    assert cfg.frame_stack == 4
    assert cfg.max_steps == 360


def test_reset_macro_empty():
    macro = ResetMacro.load(ROOT / "configs/reset.stickwar.json")
    assert macro.steps == []
    assert macro.launch_wait_s == 20.0
