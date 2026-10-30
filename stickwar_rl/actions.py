"""Discrete touch action space for a landscape Android game.

The default coordinates are a *provisional* calibration for Stick War: Legacy's
landscape HUD. They must be verified against real screenshots from the target
device before they are used for training. `configs/actions.stickwar.json`
records that provenance explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


class ActionSpaceError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    name: str
    x: float
    y: float
    hold_ms: int = 40
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pixel(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ActionSpaceError(f"invalid screen size {width}x{height}")
        px = min(max(int(round(self.x * (width - 1))), 0), width - 1)
        py = min(max(int(round(self.y * (height - 1))), 0), height - 1)
        return px, py


DEFAULT_ACTIONS: tuple[Action, ...] = (
    Action("noop", 0.5, 0.5, 0, "do not touch the screen"),
    Action("train_miner", 0.065, 0.085, 40, "top-left hammer card"),
    Action("train_swordwrath", 0.160, 0.081, 40, "top-left sword card"),
    Action("train_archidon", 0.260, 0.086, 40, "top-left bow card"),
    Action("train_spearton", 0.356, 0.081, 40, "top-left helmet card"),
    Action("train_magikill", 0.463, 0.081, 40, "top-left magikill card"),
    Action("train_giant", 0.565, 0.081, 40, "top-left giant card"),
    Action("spell", 0.058, 0.284, 60, "left spell button"),
    Action("retreat", 0.868, 0.068, 120, "left-arrow retreat control"),
    Action("defend", 0.912, 0.167, 120, "shield / defend control"),
    Action("attack", 0.955, 0.248, 120, "crossed-swords attack control"),
)


@dataclass
class ActionSpace:
    actions: list[Action]
    version: int = 1
    orientation: str = "landscape"
    source: str = "provisional"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.actions:
            raise ActionSpaceError("action space must not be empty")
        names = [a.name for a in self.actions]
        if len(set(names)) != len(names):
            raise ActionSpaceError("action names must be unique")
        if "noop" not in names:
            raise ActionSpaceError("action space must contain a 'noop' action")
        for a in self.actions:
            if not (0.0 <= a.x <= 1.0 and 0.0 <= a.y <= 1.0):
                raise ActionSpaceError(f"{a.name}: coordinates must be in [0, 1]")
            if a.hold_ms < 0 or a.hold_ms > 5000:
                raise ActionSpaceError(f"{a.name}: hold_ms outside [0, 5000]")

    @property
    def n(self) -> int:
        return len(self.actions)

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.actions]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Action:
        return self.actions[idx]

    def index_of(self, name: str) -> int:
        for i, action in enumerate(self.actions):
            if action.name == name:
                return i
        raise ActionSpaceError(f"unknown action: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "orientation": self.orientation,
            "source": self.source,
            "notes": self.notes,
            "actions": [a.to_dict() for a in self.actions],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpace":
        try:
            raw = data["actions"]
            actions = [
                Action(
                    name=str(item["name"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    hold_ms=int(item.get("hold_ms", 40)),
                    description=str(item.get("description", "")),
                )
                for item in raw
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionSpaceError(f"invalid action space: {exc}") from exc
        return cls(
            actions=actions,
            version=int(data.get("version", 1)),
            orientation=str(data.get("orientation", "landscape")),
            source=str(data.get("source", "unknown")),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActionSpace":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def default(cls) -> "ActionSpace":
        return cls(
            actions=list(DEFAULT_ACTIONS),
            source="provisional",
            notes=(
                "Coordinates are unverified placeholders for a 16:9 landscape "
                "screen. Replace with configs/actions.stickwar.json calibrated "
                "from real device screenshots."
            ),
        )


def grid_action_space(
    rows: int = 5,
    cols: int = 8,
    include_noop: bool = True,
    margin: float = 0.02,
) -> ActionSpace:
    """Generic screen-tap fallback that does not assume any UI layout."""
    if rows < 1 or cols < 1:
        raise ActionSpaceError("rows and cols must be positive")
    actions: list[Action] = []
    if include_noop:
        actions.append(Action("noop", 0.5, 0.5, 0, "do not touch the screen"))
    for r in range(rows):
        for c in range(cols):
            x = margin + (1.0 - 2.0 * margin) * (c + 0.5) / cols
            y = margin + (1.0 - 2.0 * margin) * (r + 0.5) / rows
            actions.append(Action(f"tap_r{r}_c{c}", round(x, 6), round(y, 6), 40))
    return ActionSpace(
        actions=actions,
        source="generated-grid",
        notes=f"{rows}x{cols} uniform tap grid",
    )


def validate_action_sequence(indices: Iterable[int], space: ActionSpace) -> list[int]:
    out = list(indices)
    for i in out:
        if not isinstance(i, int) or isinstance(i, bool):
            raise ActionSpaceError(f"action index must be int, got {type(i).__name__}")
        if i < 0 or i >= space.n:
            raise ActionSpaceError(f"action index {i} outside [0, {space.n})")
    return out
