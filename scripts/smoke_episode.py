#!/usr/bin/env python3
"""Record a real ADB touch/screenshot/screenrecord episode on an emulator.

This is infrastructure verification, not Stick War gameplay. It opens Android
Settings, taps the search field, types a query, and records the interaction with
the same episode format used for game data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stickwar_rl.adb import ADB, ADBError
from stickwar_rl.dataset import EpisodeWriter, utc_now
from stickwar_rl.observations import blank_frame, change_ratio, decode_png, image_stats, resize_rgb, save_image


def parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    try:
        left, rest = value.split("][")
        x1, y1 = left.strip("[]").split(",")
        x2, y2 = rest.strip("[]").split(",")
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def find_node(xml_path: Path, predicate) -> dict | None:
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return None
    for node in tree.iter("node"):
        attrs = dict(node.attrib)
        if predicate(attrs):
            bounds = parse_bounds(attrs.get("bounds", ""))
            if bounds:
                attrs["_bounds"] = bounds
                return attrs
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/android-smoke")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--record", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.out)
    shots = out / "screenshots"
    episode_dir = out / "episode"
    shots.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)
    verification: dict = {"started_at": utc_now(), "checks": {}, "errors": []}
    adb = ADB(serial=args.serial)
    adb.wait_for_boot(timeout=240)
    adb.disable_animations()
    adb.set_stay_awake()
    adb.unlock()
    adb.shell("settings", "put", "global", "hide_error_dialogs", "1", check=False)
    size = adb.wm_size()
    verification["device"] = {
        "serial": adb.serial,
        "sdk": adb.getprop("ro.build.version.sdk"),
        "model": adb.getprop("ro.product.model"),
        "wm_size": size,
        "qemu": adb.getprop("ro.kernel.qemu"),
    }
    recording = adb.start_screenrecord("/sdcard/stickwar-smoke.mp4", time_limit=args.record, size="1280x720")
    time.sleep(1.0)

    adb.shell("am", "start", "-a", "android.settings.SETTINGS", check=False)
    time.sleep(4.0)
    search = None
    for attempt in range(6):
        xml_path = adb.uiautomator_dump(out / "settings_ui.xml")
        for _ in range(2):
            dump_text = xml_path.read_text(errors="replace")
            if "isn't responding" not in dump_text and "is not responding" not in dump_text:
                break
            wait_node = find_node(xml_path, lambda a: a.get("text") == "Wait")
            if not wait_node:
                break
            x1, y1, x2, y2 = wait_node["_bounds"]
            adb.tap((x1 + x2) // 2, (y1 + y2) // 2)
            time.sleep(2.0)
            xml_path = adb.uiautomator_dump(out / "settings_ui.xml")
        search = find_node(xml_path, lambda a: a.get("resource-id", "").endswith(":id/search")
                           or "search" in a.get("text", "").lower()
                           or "search" in a.get("content-desc", "").lower())
        if search:
            break
        time.sleep(2.0)
    first = adb.screencap()
    if not isinstance(first, bytes):
        raise SystemExit("screencap did not return bytes")
    (shots / "00_settings.png").write_bytes(first)
    frame0 = decode_png(first)

    search_found = search is not None
    if search:
        x1, y1, x2, y2 = search["_bounds"]
        tap_x, tap_y = (x1 + x2) // 2, (y1 + y2) // 2
        verification["search_node"] = {k: search.get(k) for k in ("resource-id", "text", "content-desc", "bounds")}
    else:
        w, h = size
        tap_x, tap_y = w // 2, int(h * 0.12)
        verification["errors"].append("search widget not found; used a fallback coordinate")
    adb.tap(tap_x, tap_y)
    time.sleep(1.5)
    adb.text("wifi")
    time.sleep(2.0)
    after_typing = adb.screencap()
    (shots / "01_search_typed.png").write_bytes(after_typing)
    frame1 = decode_png(after_typing)
    typed_xml = adb.uiautomator_dump(out / "search_ui.xml")
    typed_text = typed_xml.read_text(errors="replace")
    typed_visible = "wifi" in typed_text.lower() or "wi-fi" in typed_text.lower()
    adb.keyevent(4)
    time.sleep(1.0)
    adb.swipe(int(size[0] * 0.7), int(size[1] * 0.8), int(size[0] * 0.7), int(size[1] * 0.35), 350)
    time.sleep(1.0)
    after_swipe = adb.screencap()
    (shots / "02_after_swipe.png").write_bytes(after_swipe)
    frame2 = decode_png(after_swipe)
    adb.shell("am", "force-stop", "com.android.settings", check=False)

    recording.wait(timeout=args.record + 20)
    try:
        adb.pull("/sdcard/stickwar-smoke.mp4", out / "screenrecord.mp4")
    except ADBError as exc:
        verification["errors"].append(f"screenrecord pull failed: {exc}")
    finally:
        adb.shell("rm", "-f", "/sdcard/stickwar-smoke.mp4", check=False)

    meta = {
        "episode_id": "smoke-settings",
        "created_at": utc_now(),
        "package": "com.android.settings",
        "policy": "scripted",
        "notes": "INFRASTRUCTURE SMOKE ONLY: not Stick War gameplay",
        "device": verification["device"],
    }
    with EpisodeWriter(episode_dir, meta=meta, save_frames=True, jpeg_quality=82) as writer:
        writer.add(frame=resize_rgb(frame0), action=1, action_name="open_settings",
                   reward=0.0, done=False, info={"event": "settings_opened"})
        writer.add(frame=resize_rgb(frame1), action=2, action_name="tap_search_type",
                   reward=0.0, done=False, info={"tap": [tap_x, tap_y], "text": "wifi"})
        writer.add(frame=resize_rgb(frame2), action=3, action_name="back_and_swipe",
                   reward=0.0, done=True, info={"swipe": "vertical"})

    shots_stats = {p.name: image_stats(decode_png(p.read_bytes())) for p in sorted(shots.glob("*.png"))}
    video = out / "screenrecord.mp4"
    verification["checks"] = {
        "booted": adb.getprop("sys.boot_completed") == "1",
        "screenshot_png": (shots / "00_settings.png").stat().st_size > 1000,
        "screenshot_not_blank": not blank_frame(frame0),
        "screenshot_changed_after_input": change_ratio(frame0, frame1) > 0.005,
        "touch_and_text_sent": True,
        "search_widget_found": search_found,
        "typed_text_visible": typed_visible,
        "ui_dump_contains_wifi": typed_visible,
        "screenrecord_exists": video.exists() and video.stat().st_size > 10000,
        "episode_steps": len((episode_dir / "steps.jsonl").read_text().strip().splitlines()) if (episode_dir / "steps.jsonl").exists() else 0,
        "episode_frames": len(list((episode_dir / "frames").glob("*.jpg"))),
    }
    verification["screenshots"] = shots_stats
    verification["finished_at"] = utc_now()
    verification["ok"] = all([
        verification["checks"]["booted"],
        verification["checks"]["screenshot_png"],
        verification["checks"]["screenshot_not_blank"],
        verification["checks"]["screenshot_changed_after_input"],
        verification["checks"]["touch_and_text_sent"],
        verification["checks"]["search_widget_found"],
        verification["checks"]["typed_text_visible"],
        verification["checks"]["episode_steps"] >= 3,
        verification["checks"]["episode_frames"] >= 3,
    ])
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2), flush=True)
    if not verification["ok"]:
        return 1
    print("SMOKE_EPISODE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
