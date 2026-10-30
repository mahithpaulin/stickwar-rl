#!/usr/bin/env python3
"""Print a compact GitHub Actions step summary from a verification JSON."""
import json
import sys

path = sys.argv[1]
v = json.load(open(path))
print("## Verification")
print("")
print(f"- ok: `{v.get('ok')}`")
if "app_level_ok" in v:
    print(f"- app-level checks: `{v.get('app_level_ok')}`")
if "in_match_status" in v:
    print(f"- in-match status: {v.get('in_match_status')}")
if v.get("device"):
    print(f"- device: `{json.dumps(v.get('device'))}`")
if v.get("apk"):
    apk = v["apk"]
    print(f"- APK sha256: `{apk.get('sha256')}`")
    print(f"- APK ABIs: `{apk.get('abis')}`")
    print(f"- APK size: {apk.get('size_bytes')} bytes")
print("")
print("### Checks")
print("")
print("```json")
print(json.dumps(v.get("checks", {}), indent=2))
print("```")
if v.get("errors"):
    print("")
    print("### Errors")
    print("")
    for err in v["errors"]:
        print(f"- {err}")
