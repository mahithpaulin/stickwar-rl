"""CLI: probe an Android device/emulator and emit machine-readable facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adb import ADB, ADBError
from .observations import image_stats


def probe(adb: ADB, package: str | None, screenshot: str | None = None) -> dict:
    info: dict = {"adb_serial": adb.serial}
    info["devices"] = [d.__dict__ for d in adb.devices()]
    info["boot_completed"] = adb.getprop("sys.boot_completed")
    info["sdk"] = adb.getprop("ro.build.version.sdk")
    info["android_release"] = adb.getprop("ro.build.version.release")
    info["model"] = adb.getprop("ro.product.model")
    info["hardware"] = adb.getprop("ro.hardware")
    info["abi_list"] = adb.getprop("ro.product.cpu.abilist")
    info["qemu"] = adb.getprop("ro.kernel.qemu")
    info["egl"] = adb.getprop("ro.hardware.egl")
    info["opengles"] = adb.getprop("ro.opengles.version")
    try:
        width, height = adb.wm_size()
        info["wm_size"] = [width, height]
    except ADBError as exc:
        info["wm_size_error"] = str(exc)
    info["wm_density"] = adb.wm_density()
    info["current_focus"] = adb.current_focus()
    if package:
        info["package"] = package
        info["installed"] = adb.is_installed(package)
        info["foreground"] = adb.is_foreground(package)
        path = adb.shell("pm", "path", package, check=False)
        info["apk_paths"] = [line.strip() for line in path.splitlines() if line.strip()]
    if screenshot:
        raw = adb.screencap()
        from .observations import decode_png
        frame = decode_png(raw)
        info["screenshot"] = {"path": str(screenshot), "shape": list(frame.shape), **image_stats(frame)}
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        Path(screenshot).write_bytes(raw)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--package", default="com.maxgames.stickwarlegacy")
    parser.add_argument("--screenshot", default="artifacts/probe/screen.png")
    parser.add_argument("--out", default="artifacts/probe/device.json")
    args = parser.parse_args(argv)
    adb = ADB(serial=args.serial)
    try:
        result = probe(adb, args.package, args.screenshot)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    result["ok"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
