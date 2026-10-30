import numpy as np
import pytest

from stickwar_rl.observations import (
    FrameStack,
    blank_frame,
    change_ratio,
    decode_png,
    image_stats,
    load_image,
    normalize,
    preprocess,
    resize_rgb,
    save_image,
)


def test_preprocess_and_normalize():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[:45] = 255
    obs = preprocess(frame, size=16)
    assert obs.shape == (16, 16)
    assert obs.dtype == np.uint8
    norm = normalize(obs)
    assert norm.shape == (16, 16)
    assert 0.0 <= float(norm.min()) <= 1.0 and 0.0 <= float(norm.max()) <= 1.0


def test_frame_stack():
    stack = FrameStack(size=3, obs_size=12)
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    out = stack.reset(frame)
    assert out.shape == (3, 12, 12)
    assert np.array_equal(out[0], out[2])
    second = np.full((30, 40, 3), 255, dtype=np.uint8)
    out = stack.push(second)
    assert out.shape == (3, 12, 12)
    assert out[-1].mean() == pytest.approx(255.0)


def test_image_io_and_stats(tmp_path):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :15] = [255, 0, 0]
    path = save_image(frame, tmp_path / "frame.jpg")
    loaded = load_image(path)
    assert loaded.shape == (20, 30, 3)
    stats = image_stats(loaded)
    assert stats["nonblack_fraction"] > 0.3
    assert not blank_frame(loaded)
    assert blank_frame(np.zeros((20, 30, 3), dtype=np.uint8))


def test_change_ratio():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.zeros((10, 10, 3), dtype=np.uint8)
    b[:, :5] = 255
    assert change_ratio(a, b) == pytest.approx(0.5)
    assert change_ratio(a, np.zeros((5, 5, 3), dtype=np.uint8)) == 1.0


def test_resize_rgb_aspect():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    resized = resize_rgb(frame, width=320)
    assert resized.shape == (180, 320, 3)
