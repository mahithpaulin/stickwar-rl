#!/usr/bin/env python3
"""Install/launch the original Stick War APK and record an app-level episode.

This script never downloads the game. The caller must supply a legally obtained
APK. It reports install/launch/screenshot/touch/episode/reset facts separately
from "is this actually inside a match", which requires a calibrated reset macro
and human review of the screenshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stickwar_rl.adb import ADB, ADBError
from stickwar_rl.dataset import utc_now
from stickwar_rl.env import AndroidEnv, AndroidEnvConfig, EnvError, ResetMacro
from stickwar_rl.observations import blank_frame, decode_png, image_stats
from stickwar_rl.policies import RandomPolicy
from stickwar_rl.runner import collect


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_aapt() -> str | None:
    found = shutil.which("aapt")
    if found:
        return found
    roots = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"), str(Path.home() / "Android/Sdk")]
    for root in filter(None, roots):
        base = Path(root) / "build-tools"
        if base.exists():
            for candidate in sorted(base.glob("*/aapt"), reverse=True):
                return str(candidate)
    return None


def inspect_apk(path: Path) -> dict:
    info: dict = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        info["entry_count"] = len(names)
        info["abis"] = sorted({parts[1] for name in names if (parts := name.split("/")) and parts[0] == "lib" and len(parts) > 2})
        info["has_android_manifest"] = "AndroidManifest.xml" in names
        info["native_libraries"] = [n for n in names if n.startswith("lib/") and n.endswith(".so")][:50]
    aapt = find_aapt()
    info["aapt"] = aapt
    if aapt:
        proc = subprocess.run([aapt, "dump", "badging", str(path)], capture_output=True, text=True, check=False)
        info["aapt_returncode"] = proc.returncode
        for line in proc.stdout.splitlines():
            if line.startswith("package:"):
                info["package_line"] = line.strip()
            if line.startswith("launchable-activity:"):
                info["launchable_activity"] = line.strip()
        info["aapt_stderr"] = proc.stderr[-500:]
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", required=True)
    parser.add_argument("--package", default="com.maxgames.stickwarlegacy")
    parser.add_argument("--activity", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--out", default="artifacts/android-game")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--reset-macro", default="configs/reset.stickwar.json")
    parser.add_argument("--allow-uncalibrated-reward", action="store_true", default=True)
    parser.add_argument("--require-match", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    shots = out / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    shots.mkdir(parents=True, exist_ok=True)
    apk = Path(args.apk)
    verification: dict = {"started_at": utc_now(), "app": "Stick War: Legacy",
                          "package": args.package, "checks": {}, "errors": [],
                          "apk": inspect_apk(apk)}
    adb = ADB(serial=args.serial)
    adb.wait_for_boot(timeout=240)
    adb.disable_animations()
    adb.set_stay_awake()
    adb.unlock()
    verification["device"] = {
        "serial": adb.serial,
        "sdk": adb.getprop("ro.build.version.sdk"),
        "model": adb.getprop("ro.product.model"),
        "abilist": adb.getprop("ro.product.cpu.abilist"),
        "qemu": adb.getprop("ro.kernel.qemu"),
        "wm_size": adb.wm_size(),
    }
    arm_only = bool(verification["apk"]["abis"]) and not any(a in {"x86", "x86_64"} for a in verification["apk"]["abis"])
    verification["arm_only_apk"] = arm_only
    if arm_only:
        verification["errors"].append(
            "APK has no x86/x86_64 native ABI. An x86_64 emulator will reject it unless ARM translation is present."
        )

    adb.force_stop(args.package)
    time.sleep(1)
    try:
        install_out = adb.install(apk)
        verification["install_output"] = install_out
        verification["checks"]["install_ok"] = True
    except ADBError as exc:
        verification["checks"]["install_ok"] = False
        verification["errors"].append(f"install failed: {exc}")
        (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(json.dumps(verification, indent=2))
        print("GAME_SMOKE_FAILED_INSTALL")
        return 2

    try:
        adb.start_package(args.package, args.activity)
    except ADBError as exc:
        verification["errors"].append(f"launch command failed: {exc}")
    if not adb.is_foreground(args.package, timeout=60):
        verification["checks"]["launch_ok"] = False
        verification["errors"].append("game package never became foreground")
        try:
            (shots / "launch_failure.png").write_bytes(adb.screencap())
        except Exception:
            pass
        (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(json.dumps(verification, indent=2))
        print("GAME_SMOKE_FAILED_LAUNCH")
        return 3
    verification["checks"]["launch_ok"] = True
    time.sleep(8)
    first = adb.screencap()
    (shots / "00_launch.png").write_bytes(first)
    frame = decode_png(first)
    verification["checks"]["screenshot_ok"] = (shots / "00_launch.png").stat().st_size > 1000 and not blank_frame(frame)
    verification["first_screen"] = image_stats(frame)

    rec = adb.start_screenrecord(f"/sdcard/stickwar-{int(time.time())}.mp4", time_limit=int(args.duration) + 20)
    config = AndroidEnvConfig.load(ROOT / "configs/env.emulator.json")
    config = replace(
        config,
        package=args.package,
        activity=args.activity,
        serial=args.serial,
        reset_macro_path=str(ROOT / args.reset_macro),
        allow_uncalibrated_reward=args.allow_uncalibrated_reward,
        step_interval_s=0.5,
        max_steps=max(1, int(round(args.duration / 0.5))),
        launch_wait_s=60.0,
        post_launch_wait_s=10.0,
    )
    env = AndroidEnv(config)
    try:
        result = collect(env, RandomPolicy(env.action_space.n, seed=1), out / "episodes",
                         episodes=args.episodes, max_steps=config.max_steps, record_width=256)
    except Exception as exc:
        verification["errors"].append(f"episode collection failed: {type(exc).__name__}: {exc}")
        env.close()
        result = None
    try:
        rec.wait(timeout=args.duration + 40)
    except subprocess.TimeoutExpired:
        verification["errors"].append("screenrecord did not finish")
    remote_videos = adb.shell("ls", "/sdcard/", check=False).splitlines()
    video_local = out / "screenrecord.mp4"
    for name in remote_videos:
        if name.startswith("stickwar-") and name.endswith(".mp4"):
            try:
                adb.pull(f"/sdcard/{name}", video_local)
            except ADBError as exc:
                verification["errors"].append(f"video pull failed: {exc}")
            break
    verification["checks"]["video_ok"] = video_local.exists() and video_local.stat().st_size > 10000
    verification["checks"]["episode_ok"] = bool(result and result.get("episodes"))
    if result:
        episode = out / "episodes" / result["episodes"][0]["episode_id"]
        verification["episode"] = {
            "path": str(episode),
            "meta": result["episodes"][0],
            "steps": len([line for line in (episode / "steps.jsonl").read_text().splitlines() if line.strip()]) if (episode / "steps.jsonl").exists() else 0,
            "frames": len(list((episode / "frames").glob("*.jpg"))) if (episode / "frames").exists() else 0,
        }
        verification["checks"]["episode_steps_ok"] = verification["episode"]["steps"] > 0

    adb.force_stop(args.package)
    time.sleep(1)
    try:
        adb.start_package(args.package, args.activity)
        reset_ok = adb.is_foreground(args.package, timeout=60)
    except ADBError as exc:
        reset_ok = False
        verification["errors"].append(f"reset relaunch failed: {exc}")
    verification["checks"]["reset_ok"] = reset_ok
    time.sleep(5)
    try:
        (shots / "01_after_reset.png").write_bytes(adb.screencap())
    except ADBError as exc:
        verification["errors"].append(f"reset screenshot failed: {exc}")

    macro = ResetMacro.load(ROOT / args.reset_macro)
    verification["reset_macro_steps"] = len(macro.steps)
    verification["in_match_verified"] = False
    verification["in_match_status"] = (
        "UNVERIFIED: reset macro is empty; recorded interaction may be menu/loading, not a match"
        if not macro.steps else
        "UNVERIFIED: reset macro exists but was written before any real-match calibration"
    )
    verification["requires_human_review"] = True
    verification["checks"]["in_match_verified"] = False
    verification["finished_at"] = utc_now()
    core = [verification["checks"].get(k, False) for k in
            ("install_ok", "launch_ok", "screenshot_ok", "episode_ok", "reset_ok", "video_ok")]
    verification["app_level_ok"] = all(core)
    verification["ok"] = verification["app_level_ok"] and (verification["in_match_verified"] or not args.require_match)
    (out / "verification.json").write_text(json.dumps(verification, indent=2, default=str) + "\n")
    print(json.dumps(verification, indent=2, default=str))
    if verification["ok"]:
        print("GAME_APP_SMOKE_OK")
        if not verification["in_match_verified"]:
            print("GAME_MATCH_UNVERIFIED")
        return 0
    print("GAME_SMOKE_FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
