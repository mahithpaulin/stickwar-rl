"""Visual observation processing for Android screenshots."""

from __future__ import annotations

import io
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


def decode_png(data: bytes | bytearray) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(data))) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save_image(frame: np.ndarray, path: str | Path, quality: int = 82) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr)
    else:
        raise ValueError(f"unsupported frame shape {arr.shape}")
    suffix = out.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(out, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(out)
    return out


def resize_rgb(frame: np.ndarray, width: int = 256) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    height = max(2, int(round(width * arr.shape[0] / max(1, arr.shape[1]))))
    return np.asarray(img.resize((width, height), RESAMPLE), dtype=np.uint8)


def preprocess(frame: np.ndarray, size: int = 84, grayscale: bool = True) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if grayscale:
        img = img.convert("L")
        img = img.resize((size, size), RESAMPLE)
        return np.asarray(img, dtype=np.uint8)
    img = img.convert("RGB").resize((size, size), RESAMPLE)
    return np.asarray(img, dtype=np.uint8)


def normalize(frames: np.ndarray) -> np.ndarray:
    return (np.asarray(frames, dtype=np.float32) / 255.0).clip(0.0, 1.0)


class FrameStack:
    def __init__(self, size: int = 4, obs_size: int = 84):
        if size < 1:
            raise ValueError("frame stack size must be positive")
        self.size = size
        self.obs_size = obs_size
        self._frames: deque[np.ndarray] = deque(maxlen=size)

    def reset(self, frame: np.ndarray | None = None) -> np.ndarray:
        self._frames.clear()
        if frame is not None:
            processed = preprocess(frame, self.obs_size)
            for _ in range(self.size):
                self._frames.append(processed)
        return self.stack()

    def push(self, frame: np.ndarray) -> np.ndarray:
        processed = preprocess(frame, self.obs_size)
        if not self._frames:
            for _ in range(self.size):
                self._frames.append(processed)
        else:
            self._frames.append(processed)
        return self.stack()

    def stack(self) -> np.ndarray:
        if len(self._frames) != self.size:
            raise ValueError("frame stack is not initialized")
        return np.stack(list(self._frames), axis=0)

    def __len__(self) -> int:
        return len(self._frames)


def crop_fraction(frame: np.ndarray, rect: Iterable[float]) -> np.ndarray:
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = list(rect)
    left, right = int(x1 * w), int(x2 * w)
    top, bottom = int(y1 * h), int(y2 * h)
    if right <= left or bottom <= top:
        raise ValueError("empty crop")
    return arr[top:bottom, left:right]


def image_stats(frame: np.ndarray) -> dict[str, float]:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "nonblack_fraction": 0.0}
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "nonblack_fraction": float((gray > 8).mean()),
    }


def change_ratio(before: np.ndarray, after: np.ndarray, threshold: int = 12) -> float:
    a = np.asarray(before, dtype=np.int16)
    b = np.asarray(after, dtype=np.int16)
    if a.shape != b.shape:
        return 1.0
    diff = np.abs(a - b).mean(axis=2) if a.ndim == 3 else np.abs(a - b)
    return float((diff > threshold).mean())


def blank_frame(frame: np.ndarray, std_threshold: float = 2.0) -> bool:
    return image_stats(frame)["std"] < std_threshold
