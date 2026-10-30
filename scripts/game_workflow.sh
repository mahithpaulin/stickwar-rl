#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ART="${GITHUB_WORKSPACE:-$ROOT}/artifacts/android-game"
mkdir -p "$ART"
APK="${APK_PATH:?APK_PATH is required}"
PACKAGE="${GAME_PACKAGE:-com.maxgames.stickwarlegacy}"
ACTIVITY="${GAME_ACTIVITY:-}"
DURATION="${GAME_DURATION:-45}"
EPISODES="${GAME_EPISODES:-1}"
REQUIRE_MATCH="${REQUIRE_MATCH:-false}"

echo "== adb devices =="
adb devices -l | tee "$ART/adb-devices.txt"

ARGS=(--apk "$APK" --package "$PACKAGE" --out "$ART" --duration "$DURATION" --episodes "$EPISODES")
if [ -n "$ACTIVITY" ]; then
  ARGS+=(--activity "$ACTIVITY")
fi
if [ "$REQUIRE_MATCH" = "true" ]; then
  ARGS+=(--require-match)
fi

set +e
python3 scripts/game_smoke.py "${ARGS[@]}" 2>&1 | tee "$ART/game.log"
RC=${PIPESTATUS[0]}
set -e

adb logcat -d -t 3000 > "$ART/logcat.txt" 2>&1 || true
adb shell dumpsys activity activities > "$ART/activities.txt" 2>&1 || true

if [ "$RC" -eq 0 ]; then
  echo "ANDROID_GAME_WORKFLOW_OK"
else
  echo "ANDROID_GAME_WORKFLOW_FAILED rc=$RC"
fi
exit "$RC"
