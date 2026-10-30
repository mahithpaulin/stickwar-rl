import json
from pathlib import Path

import pytest

from stickwar_rl.actions import (
    Action,
    ActionSpace,
    ActionSpaceError,
    DEFAULT_ACTIONS,
    grid_action_space,
    validate_action_sequence,
)


def test_default_action_space():
    space = ActionSpace.default()
    assert space.n == len(DEFAULT_ACTIONS) == 11
    assert space.names[0] == "noop"
    assert "train_archidon" in space.names
    assert space.index_of("attack") == 10
    assert len(set(space.names)) == space.n


def test_pixel_mapping_and_bounds():
    action = Action("test", 0.5, 0.5)
    assert action.pixel(1920, 1080) == (960, 540)
    assert Action("edge", 0.0, 0.0).pixel(10, 10) == (0, 0)
    assert Action("edge", 1.0, 1.0).pixel(10, 10) == (9, 9)
    with pytest.raises(ActionSpaceError):
        action.pixel(0, 10)


def test_roundtrip(tmp_path: Path):
    space = ActionSpace.default()
    path = tmp_path / "actions.json"
    space.save(path)
    loaded = ActionSpace.load(path)
    assert loaded.to_dict() == space.to_dict()


def test_rejects_missing_noop():
    with pytest.raises(ActionSpaceError):
        ActionSpace([Action("a", 0.1, 0.1)])


def test_rejects_duplicates_and_bad_coords():
    with pytest.raises(ActionSpaceError):
        ActionSpace([Action("noop", 0.5, 0.5), Action("noop", 0.2, 0.2)])
    with pytest.raises(ActionSpaceError):
        ActionSpace([Action("noop", 0.5, 0.5), Action("bad", 1.2, 0.2)])
    with pytest.raises(ActionSpaceError):
        ActionSpace([Action("noop", 0.5, 0.5), Action("hold", 0.2, 0.2, hold_ms=9999)])


def test_grid_space():
    space = grid_action_space(rows=3, cols=4)
    assert space.n == 13
    assert space.source == "generated-grid"
    assert space.names[0] == "noop"
    assert space.names[-1] == "tap_r2_c3"


def test_validate_sequence():
    space = ActionSpace.default()
    assert validate_action_sequence([0, 1, 10], space) == [0, 1, 10]
    with pytest.raises(ActionSpaceError):
        validate_action_sequence([11], space)
    with pytest.raises(ActionSpaceError):
        validate_action_sequence([True], space)
