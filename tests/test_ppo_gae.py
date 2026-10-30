import pytest

from stickwar_rl.ppo import PPOConfig, gae


def test_gae_simple():
    advantages, returns = gae([1.0, 0.0], [0.0, 0.0], [False, True], gamma=0.5, lam=1.0)
    assert advantages == pytest.approx([1.0, 0.0])
    assert returns == pytest.approx([1.0, 0.0])


def test_gae_discounts_and_bootstraps():
    advantages, returns = gae([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [False, False, True], gamma=0.5, lam=1.0)
    assert advantages[0] == pytest.approx(1.0 + 0.5 * 1.0)
    assert returns == pytest.approx([1.5, 1.0, 0.0])


def test_gae_length_mismatch():
    with pytest.raises(ValueError):
        gae([1.0], [0.0, 0.0], [False])


def test_config_validation():
    PPOConfig().validate()
    with pytest.raises(ValueError):
        PPOConfig(n_actions=1).validate()
    with pytest.raises(ValueError):
        PPOConfig(clip_ratio=1.5).validate()
    with pytest.raises(ValueError):
        PPOConfig(epochs=0).validate()
