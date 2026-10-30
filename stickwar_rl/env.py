"""Android screen environment: screenshot observation, ADB touch actions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionSpace
from .adb import ADB, ADBError
from .observations import FrameStack, decode_png
from .rewards import RewardConfig, RewardModel, RewardSignals


class EnvError(RuntimeError):
    pass


@dataclass
class ResetMacroStep:
    type: str
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    seconds: float | None = None
    keycode: str | int | None = None
    duration_ms: int = 300

    def validate(self) -> None:
        if self.type not in {"tap", "swipe", "wait", "keyevent"}:
            raise EnvError(f"invalid reset step type: {self.type}")
        if self.type in {"tap", "swipe"}:
            for value in (self.x, self.y):
                if value is None or not (0.0 <= value <= 1.0):
                    raise EnvError(f"{self.type} needs normalized x/y")
        if self.type == "swipe":
            for value in (self.x2, self.y2):
                if value is None or not (0.0 <= value <= 1.0):
                    raise EnvError("swipe needs normalized x2/y2")
        if self.type == "wait" and (self.seconds is None or self.seconds < 0):
            raise EnvError("wait needs non-negative seconds")


@dataclass
class ResetMacro:
    launch_wait_s: float = 20.0
    steps: list[ResetMacroStep] = field(default_factory=list)
    description: str = ""

    @classmethod
    def load(cls, path: str | Path | None) -> "ResetMacro":
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        steps = []
        for item in data.get("steps", []):
            step = ResetMacroStep(
                type=str(item["type"]),
                x=None if item.get("x") is None else float(item["x"]),
                y=None if item.get("y") is None else float(item["y"]),
                x2=None if item.get("x2") is None else float(item["x2"]),
                y2=None if item.get("y2") is None else float(item["y2"]),
                seconds=None if item.get("seconds") is None else float(item["seconds"]),
                keycode=item.get("keycode"),
                duration_ms=int(item.get("duration_ms", 300)),
            )
            step.validate()
            steps.append(step)
        return cls(
            launch_wait_s=float(data.get("launch_wait_s", 20.0)),
            steps=steps,
            description=str(data.get("description", "")),
        )

    def execute(self, adb: ADB, width: int, height: int) -> None:
        for step in self.steps:
            if step.type == "wait":
                time.sleep(float(step.seconds or 0))
            elif step.type == "tap":
                adb.tap(int(step.x * (width - 1)), int(step.y * (height - 1)))
            elif step.type == "swipe":
                adb.swipe(
                    int(step.x * (width - 1)), int(step.y * (height - 1)),
                    int(step.x2 * (width - 1)), int(step.y2 * (height - 1)),
                    step.duration_ms,
                )
            elif step.type == "keyevent" and step.keycode is not None:
                adb.keyevent(step.keycode)


@dataclass
class AndroidEnvConfig:
    package: str = "com.maxgames.stickwarlegacy"
    activity: str | None = None
    serial: str | None = None
    adb_path: str | None = None
    action_space_path: str = "configs/actions.stickwar.json"
    reward_config_path: str = "configs/reward.stickwar.json"
    reset_macro_path: str = "configs/reset.stickwar.json"
    frame_size: int = 84
    frame_stack: int = 4
    step_interval_s: float = 0.5
    max_steps: int = 360
    launch_wait_s: float = 30.0
    post_launch_wait_s: float = 5.0
    foreground_check: bool = True
    allow_uncalibrated_reward: bool = False
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "AndroidEnvConfig":
        data = json.loads(Path(path).read_text())
        aliases = {
            "action_space": "action_space_path",
            "reward_config": "reward_config_path",
            "reset_macro": "reset_macro_path",
        }
        mapped = {aliases.get(k, k): v for k, v in data.items()}
        return cls(**{k: v for k, v in mapped.items() if k in cls.__dataclass_fields__})


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class AndroidEnv:
    def __init__(self, config: AndroidEnvConfig):
        self.config = config
        self.action_space = ActionSpace.load(config.action_space_path)
        self.reward_config = RewardConfig.load(config.reward_config_path)
        if not self.reward_config.calibrated and not config.allow_uncalibrated_reward:
            raise EnvError(
                "reward config is not calibrated; refusing to train on it. "
                "Calibrate configs/reward.stickwar.json or pass "
                "allow_uncalibrated_reward=true for data collection with a zero/"
                "sparse reward."
            )
        self.reward_model = RewardModel(self.reward_config)
        self.reset_macro = ResetMacro.load(config.reset_macro_path)
        self.adb = ADB(serial=config.serial, adb_path=config.adb_path)
        self.frames = FrameStack(config.frame_stack, config.frame_size)
        self.obs: np.ndarray | None = None
        self.last_raw: np.ndarray | None = None
        self.last_signals: RewardSignals = RewardSignals()
        self.step_count = 0
        self.episode_done = False
        self.screen_size: tuple[int, int] | None = None
        self._rng = np.random.default_rng(config.seed)

    def _capture(self) -> np.ndarray:
        data = self.adb.screencap()
        if not isinstance(data, bytes):
            raise EnvError("screencap returned a path, not bytes")
        raw = decode_png(data)
        self.last_raw = raw
        self.screen_size = (raw.shape[1], raw.shape[0])
        return raw

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.episode_done = False
        self.adb.force_stop(self.config.package)
        time.sleep(1.0)
        self.adb.start_package(self.config.package, self.config.activity)
        deadline = time.time() + self.config.launch_wait_s
        if self.config.foreground_check:
            if not self.adb.is_foreground(self.config.package, timeout=self.config.launch_wait_s):
                raise EnvError(
                    f"{self.config.package} did not become the foreground app within "
                    f"{self.config.launch_wait_s}s"
                )
        else:
            time.sleep(min(self.config.launch_wait_s, max(0.0, deadline - time.time())))
        raw = self._capture()
        if self.config.post_launch_wait_s > 0:
            time.sleep(self.config.post_launch_wait_s)
            raw = self._capture()
        width, height = self.screen_size or (raw.shape[1], raw.shape[0])
        if self.reset_macro.steps:
            self.reset_macro.execute(self.adb, width, height)
            time.sleep(1.0)
            raw = self._capture()
        self.obs = self.frames.reset(raw)
        self.last_signals = self.reward_model.reset(raw)
        return self.obs

    def step(self, action: int) -> StepResult:
        if self.obs is None:
            raise EnvError("reset() must be called before step()")
        if self.episode_done:
            raise EnvError("episode is done; call reset()")
        if action < 0 or action >= self.action_space.n:
            raise EnvError(f"action {action} outside [0, {self.action_space.n})")
        act = self.action_space[action]
        if act.name != "noop" and self.screen_size is not None:
            width, height = self.screen_size
            x, y = act.pixel(width, height)
            if act.hold_ms >= 80:
                self.adb.swipe(x, y, x, y, act.hold_ms)
            else:
                self.adb.tap(x, y)
        time.sleep(max(0.0, self.config.step_interval_s))
        raw = self._capture()
        self.obs = self.frames.push(raw)
        reward, signals = self.reward_model.step(raw)
        self.last_signals = signals
        self.step_count += 1
        done = bool(signals.win or signals.loss or self.step_count >= self.config.max_steps)
        left_game = False
        if self.config.foreground_check and not done and self.step_count % 10 == 0:
            left_game = not self.adb.is_foreground(self.config.package)
            if left_game:
                done = True
        self.episode_done = done
        info = {
            "step": self.step_count,
            "action_name": act.name,
            "tap": None if act.name == "noop" or self.screen_size is None else act.pixel(*self.screen_size),
            "signals": signals.to_dict(),
            "left_game": left_game,
            "screen_size": self.screen_size,
        }
        return StepResult(obs=self.obs, reward=reward, done=done, info=info)

    def close(self) -> None:
        self.episode_done = True

    def __enter__(self) -> "AndroidEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
