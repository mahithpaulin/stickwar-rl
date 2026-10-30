#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="${GITHUB_WORKSPACE:-$ROOT}/artifacts/android-smoke"
mkdir -p "$ART"
cd "$ROOT"

echo "== adb devices =="
adb devices -l | tee "$ART/adb-devices.txt"

echo "== boot wait =="
adb wait-for-device
for _ in $(seq 1 120); do
  boot="$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')"
  if [ "$boot" = "1" ]; then break; fi
  sleep 2
done
if [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" != "1" ]; then
  echo "FAIL: emulator did not boot" | tee "$ART/FAILED"
  exit 1
fi

adb shell settings put global hide_error_dialogs 1 || true
adb shell settings put global window_animation_scale 0 || true
adb shell settings put global transition_animation_scale 0 || true
adb shell settings put global animator_duration_scale 0 || true
adb shell input keyevent KEYCODE_WAKEUP || true
adb shell wm dismiss-keyguard || true
sleep 5
adb shell wm size | tee "$ART/wm-size.txt"
adb shell wm density | tee "$ART/wm-density.txt"
adb shell getprop ro.build.version.sdk | tee "$ART/sdk.txt"
adb shell getprop ro.product.cpu.abilist | tee "$ART/abilist.txt"
adb shell getprop ro.kernel.qemu | tee "$ART/qemu.txt"
adb shell getprop ro.hardware.egl | tee "$ART/egl.txt"

set +e
python3 scripts/smoke_episode.py --out "$ART" 2>&1 | tee "$ART/smoke.log"
SMOKE_RC=${PIPESTATUS[0]}
set -e

adb logcat -d -t 2000 > "$ART/logcat.txt" 2>&1 || true
adb shell dumpsys SurfaceFlinger > "$ART/surfaceflinger.txt" 2>&1 || true

if [ -f "$ART/verification.json" ]; then
  python3 - "$ART/verification.json" <<'PY'
import json, sys
v=json.load(open(sys.argv[1]))
print(json.dumps({"checks": v.get("checks"), "ok": v.get("ok"), "errors": v.get("errors")}, indent=2))
PY
fi

if [ "$SMOKE_RC" -eq 0 ] && [ -f "$ART/verification.json" ] && grep -q '"ok": true' "$ART/verification.json"; then
  echo "ANDROID_SMOKE_OK"
  exit 0
fi
echo "ANDROID_SMOKE_FAILED rc=$SMOKE_RC"
exit 1
