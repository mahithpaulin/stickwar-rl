"""Thin, testable wrapper around the Android Debug Bridge."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class ADBError(RuntimeError):
    pass


@dataclass
class DeviceInfo:
    serial: str
    state: str
    product: str = ""
    model: str = ""
    sdk: str = ""


class ADB:
    def __init__(
        self,
        serial: str | None = None,
        adb_path: str | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.serial = serial
        self.adb_path = adb_path or os.environ.get("ADB", "adb")
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_delay = retry_delay

    def _base(self) -> list[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd.extend(["-s", self.serial])
        return cmd

    def _run(
        self,
        args: list[str],
        timeout: float | None = None,
        binary: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        cmd = self._base() + args
        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout if timeout is not None else self.timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise ADBError(f"adb executable not found: {self.adb_path}") from exc
            except subprocess.TimeoutExpired as exc:
                last_error = f"timeout after {exc.timeout}s"
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                raise ADBError(f"{shlex.join(cmd)}: {last_error}") from exc

            stderr = proc.stderr.decode("utf-8", "replace") if binary else proc.stderr
            stdout = proc.stdout if binary else proc.stdout.decode("utf-8", "replace")
            if proc.returncode == 0:
                return subprocess.CompletedProcess(cmd, 0, stdout, stderr)
            last_error = (stderr or "").strip() if isinstance(stderr, str) else stderr.decode("utf-8", "replace").strip()
            transient = any(
                needle in last_error.lower()
                for needle in ("device offline", "error: closed", "connection reset", "protocol fault")
            )
            if transient and attempt < self.retries:
                time.sleep(self.retry_delay)
                continue
            if check:
                raise ADBError(f"{shlex.join(cmd)} failed ({proc.returncode}): {last_error[:2000]}")
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        raise ADBError(f"{shlex.join(cmd)} failed: {last_error[:2000]}")

    def shell(self, *args: str, timeout: float | None = None, check: bool = True) -> str:
        return self._run(["shell", *args], timeout=timeout, check=check).stdout.strip()

    def devices(self) -> list[DeviceInfo]:
        output = self._run(["devices", "-l"]).stdout
        result: list[DeviceInfo] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            info = DeviceInfo(serial=parts[0], state=parts[1])
            for token in parts[2:]:
                if ":" not in token:
                    continue
                key, value = token.split(":", 1)
                if key == "product":
                    info.product = value
                elif key == "model":
                    info.model = value
            result.append(info)
        return result

    def getprop(self, name: str) -> str:
        return self.shell("getprop", name)

    def wait_for_boot(self, timeout: float = 180.0, poll: float = 2.0) -> None:
        deadline = time.time() + timeout
        self._run(["wait-for-device"], timeout=timeout)
        while time.time() < deadline:
            if self.getprop("sys.boot_completed") == "1":
                return
            time.sleep(poll)
        raise ADBError(f"device did not finish booting within {timeout}s")

    def wm_size(self) -> tuple[int, int]:
        output = self.shell("wm", "size")
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise ADBError(f"could not parse wm size output: {output!r}")
        w, h = matches[-1]
        return int(w), int(h)

    def wm_density(self) -> int | None:
        output = self.shell("wm", "density", check=False)
        match = re.search(r"(\d+)", output)
        return int(match.group(1)) if match else None

    def screencap(self, path: str | Path | None = None, timeout: float = 60.0) -> bytes | Path:
        data = self._run(["exec-out", "screencap", "-p"], timeout=timeout, binary=True).stdout
        if not isinstance(data, (bytes, bytearray)) or not data.startswith(b"\x89PNG"):
            fallback = self._run(["shell", "screencap", "-p"], timeout=timeout, binary=True).stdout
            if isinstance(fallback, (bytes, bytearray)):
                data = fallback.replace(b"\r\n", b"\n")
        if not data.startswith(b"\x89PNG"):
            raise ADBError("screencap did not return a PNG")
        if path is None:
            return bytes(data)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.shell(
            "input", "swipe",
            str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms)),
        )

    def keyevent(self, keycode: str | int) -> None:
        self.shell("input", "keyevent", str(keycode))

    def text(self, value: str) -> None:
        escaped = value.replace(" ", "%s")
        self.shell("input", "text", escaped)

    def push(self, local: str | Path, remote: str) -> None:
        self._run(["push", str(local), remote], timeout=300)

    def pull(self, remote: str, local: str | Path) -> Path:
        out = Path(local)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run(["pull", remote, str(out)], timeout=300)
        if not out.exists():
            raise ADBError(f"pull did not create {out}")
        return out

    def install(self, apk: str | Path, replace: bool = True, grant: bool = True) -> str:
        args = ["install"]
        if replace:
            args.append("-r")
        if grant:
            args.append("-g")
        args.append(str(apk))
        return self._run(args, timeout=600).stdout

    def install_split(self, apks: list[str | Path], replace: bool = True, grant: bool = True) -> str:
        if not apks:
            raise ADBError("no split APKs supplied")
        args = ["install-multiple"]
        if replace:
            args.append("-r")
        if grant:
            args.append("-g")
        args.extend(str(p) for p in apks)
        return self._run(args, timeout=900).stdout

    def uninstall(self, package: str) -> str:
        return self._run(["uninstall", package], timeout=120).stdout

    def is_installed(self, package: str) -> bool:
        output = self.shell("pm", "list", "packages", package, check=False)
        return f"package:{package}" in output.splitlines()

    def force_stop(self, package: str) -> None:
        self.shell("am", "force-stop", package, check=False)

    def start_package(self, package: str, activity: str | None = None, wait: bool = True) -> str:
        if activity:
            component = activity if "/" in activity else f"{package}/{activity}"
            return self.shell("am", "start", "-W", "-n", component)
        return self.shell("monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1", check=False)

    def current_focus(self) -> str:
        for command in (
            ("dumpsys", "window", "windows"),
            ("dumpsys", "activity", "activities"),
        ):
            output = self.shell(*command, check=False)
            for line in output.splitlines():
                if any(token in line for token in ("mCurrentFocus", "mFocusedApp", "mResumedActivity", "topResumedActivity")):
                    if "/" in line:
                        return line.strip()
        return ""

    def is_foreground(self, package: str, timeout: float = 0.0, poll: float = 1.0) -> bool:
        deadline = time.time() + max(0.0, timeout)
        while True:
            focus = self.current_focus()
            if package in focus:
                return True
            if time.time() >= deadline:
                return False
            time.sleep(poll)

    def start_screenrecord(
        self,
        remote_path: str,
        time_limit: int = 30,
        bit_rate: int = 4_000_000,
        size: str | None = None,
    ) -> subprocess.Popen:
        args = ["shell", "screenrecord", "--time-limit", str(int(time_limit)), "--bit-rate", str(int(bit_rate))]
        if size:
            args.extend(["--size", size])
        args.append(remote_path)
        return subprocess.Popen(
            self._base() + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def uiautomator_dump(self, local_path: str | Path, remote_path: str = "/sdcard/window_dump.xml") -> Path:
        self.shell("uiautomator", "dump", remote_path, check=False)
        return self.pull(remote_path, local_path)

    def disable_animations(self) -> None:
        for key in ("window_animation_scale", "transition_animation_scale", "animator_duration_scale"):
            self.shell("settings", "put", "global", key, "0", check=False)

    def set_stay_awake(self) -> None:
        self.shell("svc", "power", "stayon", "true", check=False)
        self.shell("settings", "put", "global", "stay_on_while_plugged_in", "7", check=False)

    def unlock(self) -> None:
        self.shell("input", "keyevent", "224", check=False)
        self.shell("input", "keyevent", "82", check=False)
        self.shell("wm", "dismiss-keyguard", check=False)
