"""Configurable reward model for visual game observations.

No reward can be claimed valid until its bar/template calibration has been
checked against real screenshots. The default Stick War calibration is marked
`calibrated: false` and training refuses it unless explicitly overridden.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class RewardConfigError(ValueError):
    pass


@dataclass
class BarRegion:
    name: str
    rect: list[float]
    color: str
    tolerance: int = 60
    axis: str = "horizontal"
    fill_anchor: str = "left"
    baseline: float | None = None
    min_column_fill: float = 0.25

    def validate(self) -> None:
        color = self.color.strip().lstrip("#")
        if len(color) != 6 or any(c not in "0123456789abcdefABCDEF" for c in color):
            raise RewardConfigError(f"{self.name}: invalid color {self.color!r}")
        if len(self.rect) != 4:
            raise RewardConfigError(f"{self.name}: rect must have 4 values")
        x1, y1, x2, y2 = self.rect
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise RewardConfigError(f"{self.name}: rect must be normalized and ordered")
        if self.axis not in {"horizontal", "vertical"}:
            raise RewardConfigError(f"{self.name}: axis must be horizontal or vertical")
        if self.fill_anchor not in {"left", "right", "top", "bottom", "count"}:
            raise RewardConfigError(f"{self.name}: invalid fill_anchor")
        if not (0 <= self.tolerance <= 441):
            raise RewardConfigError(f"{self.name}: tolerance outside [0, 441]")
        if self.baseline is not None and not (0.0 < self.baseline <= 1.0):
            raise RewardConfigError(f"{self.name}: baseline must be in (0, 1]")
        if not (0.0 <= self.min_column_fill <= 1.0):
            raise RewardConfigError(f"{self.name}: min_column_fill outside [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BarRegion":
        region = cls(
            name=str(data["name"]),
            rect=[float(v) for v in data["rect"]],
            color=str(data["color"]),
            tolerance=int(data.get("tolerance", 60)),
            axis=str(data.get("axis", "horizontal")),
            fill_anchor=str(data.get("fill_anchor", "left")),
            baseline=None if data.get("baseline") is None else float(data["baseline"]),
            min_column_fill=float(data.get("min_column_fill", 0.25)),
        )
        region.validate()
        return region


@dataclass
class RewardConfig:
    version: int = 1
    calibrated: bool = False
    source: str = "provisional"
    bars: list[BarRegion] = field(default_factory=list)
    enemy_bar: str = "enemy_hp"
    own_bar: str = "own_hp"
    enemy_weight: float = 2.0
    own_weight: float = -2.0
    win_bonus: float = 20.0
    loss_penalty: float = -20.0
    step_penalty: float = -0.005
    reward_clip: float = 5.0
    win_template: str | None = None
    loss_template: str | None = None
    template_threshold: float = 0.92
    notes: str = ""

    def __post_init__(self) -> None:
        for bar in self.bars:
            bar.validate()
        names = [b.name for b in self.bars]
        if len(names) != len(set(names)):
            raise RewardConfigError("bar names must be unique")
        if self.reward_clip <= 0:
            raise RewardConfigError("reward_clip must be positive")
        if self.template_threshold <= 0 or self.template_threshold > 1:
            raise RewardConfigError("template_threshold must be in (0, 1]")

    def bar(self, name: str) -> BarRegion | None:
        for b in self.bars:
            if b.name == name:
                return b
        return None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bars"] = [b.to_dict() for b in self.bars]
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardConfig":
        bars = [BarRegion.from_dict(b) for b in data.get("bars", [])]
        return cls(
            version=int(data.get("version", 1)),
            calibrated=bool(data.get("calibrated", False)),
            source=str(data.get("source", "unknown")),
            bars=bars,
            enemy_bar=str(data.get("enemy_bar", "enemy_hp")),
            own_bar=str(data.get("own_bar", "own_hp")),
            enemy_weight=float(data.get("enemy_weight", 2.0)),
            own_weight=float(data.get("own_weight", -2.0)),
            win_bonus=float(data.get("win_bonus", 20.0)),
            loss_penalty=float(data.get("loss_penalty", -20.0)),
            step_penalty=float(data.get("step_penalty", -0.005)),
            reward_clip=float(data.get("reward_clip", 5.0)),
            win_template=data.get("win_template"),
            loss_template=data.get("loss_template"),
            template_threshold=float(data.get("template_threshold", 0.92)),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RewardConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class RewardSignals:
    enemy_hp: float | None = None
    own_hp: float | None = None
    win: bool = False
    loss: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_reward(
    prev: RewardSignals,
    curr: RewardSignals,
    config: RewardConfig,
) -> float:
    reward = config.step_penalty
    if prev.enemy_hp is not None and curr.enemy_hp is not None:
        reward += config.enemy_weight * (prev.enemy_hp - curr.enemy_hp)
    if prev.own_hp is not None and curr.own_hp is not None:
        reward += config.own_weight * (prev.own_hp - curr.own_hp)
    if curr.win:
        reward += config.win_bonus
    if curr.loss:
        reward += config.loss_penalty
    return float(max(-config.reward_clip, min(config.reward_clip, reward)))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise RewardConfigError(f"invalid color: {value!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def measure_bar_fraction(frame, region: BarRegion) -> float:
    """Measure the filled fraction of a colored bar in an RGB frame.

    `frame` must be an HxWx3 array-like object (numpy array). Importing numpy is
    deferred so the pure reward math stays usable on machines without it.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to measure bar pixels") from exc

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {arr.shape}")
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = region.rect
    left, right = int(x1 * w), max(int(x2 * w), int(x1 * w) + 1)
    top, bottom = int(y1 * h), max(int(y2 * h), int(y1 * h) + 1)
    crop = arr[top:bottom, left:right, :3].astype(np.int32)
    target = np.array(_hex_to_rgb(region.color), dtype=np.int32)
    dist = np.sqrt(((crop - target) ** 2).sum(axis=2))
    mask = dist <= region.tolerance
    if region.axis == "horizontal":
        per_line = mask.mean(axis=0)
        filled = per_line >= region.min_column_fill
        anchor = region.fill_anchor
    else:
        per_line = mask.mean(axis=1)
        filled = per_line >= region.min_column_fill
        anchor = region.fill_anchor
    if anchor in {"right", "bottom"}:
        filled = filled[::-1]
    if anchor in {"left", "right", "top", "bottom"}:
        end = 0
        while end < len(filled) and bool(filled[end]):
            end += 1
        frac = end / max(1, len(filled))
    else:
        frac = float(filled.mean())
    if region.baseline:
        frac = min(1.0, frac / region.baseline)
    return max(0.0, min(1.0, frac))


class TemplateDetector:
    """Small normalized-absolute-difference template detector.

    It is deliberately simple: game-state screens are fixed-aspect and the
    templates are small UI fragments. A mismatch is reported rather than hidden.
    """

    def __init__(self, template_path: str | Path, threshold: float = 0.92):
        from PIL import Image
        import numpy as np

        self.threshold = threshold
        img = Image.open(template_path).convert("L").resize((64, 36))
        self.template = np.asarray(img, dtype=np.float32) / 255.0

    def score(self, frame) -> float:
        from PIL import Image
        import numpy as np

        arr = np.asarray(frame)
        img = Image.fromarray(arr.astype("uint8")).convert("L").resize((64, 36))
        candidate = np.asarray(img, dtype=np.float32) / 255.0
        mae = float(np.abs(candidate - self.template).mean())
        return max(0.0, 1.0 - mae)

    def matches(self, frame) -> bool:
        return self.score(frame) >= self.threshold


class RewardModel:
    def __init__(self, config: RewardConfig):
        self.config = config
        self._prev: RewardSignals | None = None
        self._win_detector: TemplateDetector | None = None
        self._loss_detector: TemplateDetector | None = None
        if config.win_template:
            self._win_detector = TemplateDetector(config.win_template, config.template_threshold)
        if config.loss_template:
            self._loss_detector = TemplateDetector(config.loss_template, config.template_threshold)

    def reset(self, frame=None) -> RewardSignals:
        self._prev = self.signals(frame) if frame is not None else RewardSignals()
        return self._prev

    def signals(self, frame) -> RewardSignals:
        enemy = own = None
        if frame is not None:
            ebar = self.config.bar(self.config.enemy_bar)
            obar = self.config.bar(self.config.own_bar)
            if ebar is not None:
                enemy = measure_bar_fraction(frame, ebar)
            if obar is not None:
                own = measure_bar_fraction(frame, obar)
        win = bool(self._win_detector and self._win_detector.matches(frame))
        loss = bool(self._loss_detector and self._loss_detector.matches(frame))
        return RewardSignals(enemy_hp=enemy, own_hp=own, win=win, loss=loss)

    def step(self, frame) -> tuple[float, RewardSignals]:
        curr = self.signals(frame)
        if self._prev is None:
            self._prev = curr
            return 0.0, curr
        reward = compute_reward(self._prev, curr, self.config)
        self._prev = curr
        return reward, curr
