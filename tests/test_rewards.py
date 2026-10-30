import numpy as np
import pytest

from stickwar_rl.rewards import (
    BarRegion,
    RewardConfig,
    RewardConfigError,
    RewardModel,
    RewardSignals,
    compute_reward,
    measure_bar_fraction,
)


def test_reward_math():
    cfg = RewardConfig(step_penalty=-0.01, enemy_weight=2.0, own_weight=-3.0,
                       win_bonus=10.0, loss_penalty=-10.0, reward_clip=50.0)
    prev = RewardSignals(enemy_hp=1.0, own_hp=1.0)
    curr = RewardSignals(enemy_hp=0.8, own_hp=0.9)
    reward = compute_reward(prev, curr, cfg)
    assert reward == pytest.approx(-0.01 + 2.0 * 0.2 + -3.0 * 0.1)


def test_win_loss_and_clipping():
    cfg = RewardConfig(step_penalty=0.0, win_bonus=100.0, loss_penalty=-100.0, reward_clip=5.0)
    assert compute_reward(RewardSignals(), RewardSignals(win=True), cfg) == 5.0
    assert compute_reward(RewardSignals(), RewardSignals(loss=True), cfg) == -5.0


def test_missing_signals_do_not_crash():
    cfg = RewardConfig()
    assert compute_reward(RewardSignals(), RewardSignals(), cfg) == pytest.approx(cfg.step_penalty)


def test_bar_region_validation():
    BarRegion("ok", [0.1, 0.1, 0.9, 0.2], "#ff0000").validate()
    with pytest.raises(RewardConfigError):
        BarRegion("bad", [0.9, 0.1, 0.1, 0.2], "#ff0000").validate()
    with pytest.raises(RewardConfigError):
        BarRegion("bad", [0.1, 0.1, 0.9, 0.2], "red").validate()
    with pytest.raises(RewardConfigError):
        RewardConfig(bars=[BarRegion("x", [0, 0, 1, 1], "#ff0000"),
                           BarRegion("x", [0, 0, 1, 1], "#00ff00")])


def test_measure_bar_fraction_anchors():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = [255, 0, 0]
    frame[:, 100:] = [255, 255, 255]
    region = BarRegion("enemy", [0.0, 0.0, 1.0, 1.0], "#ff0000", tolerance=10, fill_anchor="left")
    assert measure_bar_fraction(frame, region) == pytest.approx(0.5)
    region_right = BarRegion("enemy", [0.0, 0.0, 1.0, 1.0], "#ff0000", tolerance=10, fill_anchor="right")
    assert measure_bar_fraction(frame, region_right) == pytest.approx(0.0)
    region_count = BarRegion("enemy", [0.0, 0.0, 1.0, 1.0], "#ff0000", tolerance=10, fill_anchor="count")
    assert measure_bar_fraction(frame, region_count) == pytest.approx(0.5)


def test_bar_baseline_normalization():
    frame = np.zeros((10, 100, 3), dtype=np.uint8)
    frame[:, :50] = [0, 255, 0]
    region = BarRegion("own", [0, 0, 1, 1], "#00ff00", tolerance=10, baseline=0.5)
    assert measure_bar_fraction(frame, region) == pytest.approx(1.0)


def test_reward_model_stateful():
    bars = [
        BarRegion("enemy_hp", [0, 0, 1, 0.5], "#ff0000", tolerance=10),
        BarRegion("own_hp", [0, 0.5, 1, 1], "#00ff00", tolerance=10),
    ]
    cfg = RewardConfig(bars=bars, step_penalty=0.0, enemy_weight=1.0, own_weight=-1.0, reward_clip=10.0)
    model = RewardModel(cfg)
    full = np.zeros((10, 100, 3), dtype=np.uint8)
    full[:5, :, :] = [255, 0, 0]
    full[5:, :, :] = [0, 255, 0]
    first = full.copy()
    first[5:, 50:, :] = [0, 0, 0]
    model.reset(first)
    reward, signals = model.step(full)
    assert reward > 0
    assert signals.enemy_hp == pytest.approx(1.0)
    assert signals.own_hp == pytest.approx(1.0)
