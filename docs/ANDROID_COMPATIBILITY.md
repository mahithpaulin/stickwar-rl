# Android compatibility notes

## Hosted GitHub Actions target

`android-smoke.yml` and `android-game.yml` use:

- `runs-on: ubuntu-latest` (x86_64)
- `/dev/kvm` exposed by the runner
- API 30, `google_apis`, `x86_64`, Pixel 4 profile
- `-gpu swiftshader_indirect` software rendering
- `reactivecircus/android-emulator-runner@v2`

There is no GPU on the hosted runner. Rendering is CPU/SwiftShader, so frame rate
and game performance are not representative. KVM is required; the workflow fails
early if `/dev/kvm` is missing or not writable.

## APK ABI matrix

`scripts/game_smoke.py` inspects the APK zip and reports every `lib/<abi>/` entry.
It is not enough for the package to be "Android"; native Unity libraries must
match the emulator CPU.

| APK native ABIs | x86_64 emulator result |
|---|---|
| `x86_64` or `x86` | expected to install and run |
| `arm64-v8a` / `armeabi-v7a` only | likely `INSTALL_FAILED_NO_MATCHING_ABIS`; needs ARM translation or a real ARM device |
| no native libs | pure-Java APK, likely installs |

Some Android x86 system images include ARM translation (historically via
libhoudini/ndk-translation), but this is **not assumed**. The workflow attempts
the install and reports the real result. It does not replace the game with a
custom simulator.

The original Stick War: Legacy APK was not available on the development phone
(`pm list packages` showed no `com.maxgames.*`), so B–E remain blocked rather than
being marked as passing. No APK was downloaded from APKPure/APKMirror/other
third-party mirrors.

## Practical alternatives, in order

1. **Physical Android phone as the actor (recommended).**
   The game runs natively on ARM. Enable Developer options → Wireless debugging,
   pair from Termux (`adb pair`), connect, then run
   `python -m stickwar_rl.collect`. This phone has 3.7 GB RAM and no CUDA, but
   inference of the small CNN is cheap; training stays on Kaggle. Screen capture
   and `input tap` work without root over ADB.
2. **A local x86_64 PC with Android Studio.** Try an x86_64 image first; if the
   APK is ARM-only, test an ARM64 image on an ARM host, or use a device with
   proven ARM translation. Record the actual `aapt` ABI list and install result.
3. **Cloud device farm** (AWS Device Farm, Firebase Test Lab, BrowserStack).
   Real ARM devices; paid/quota-bound. Not configured here because it needs an
   account and payment decision.

## Reset and episode boundaries

- Reset uses `am force-stop` + launcher intent. App data is preserved.
- `pm clear` is never called automatically; it would wipe user progress.
- A `configs/reset.stickwar.json` macro can tap through menus/loading. It is
  empty until calibrated from real screenshots. Until then, `in_match_verified`
  is `false` and the game workflow says so.
- Deterministic resets in a live service game may be impossible (ads, rotating
  events, network). Record the observed menu state and exclude non-matching
  episodes from training rather than silently mixing them.

## Reward and UI calibration

The action coordinates come from the official Google Play 16:9 screenshots
(unit cards and controls are in the top HUD). They are **provisional**. HUD
safe-area/DPI changes can shift them. The reward config has no calibrated bar
crops and no win/loss templates, so it is sparse (win/loss + step penalty) and
marked uncalibrated. Training refuses it unless the operator explicitly overrides
the gate for debug runs.

Required calibration work once an in-match screenshot exists:

1. Save a 16:9 screenshot from the target device/emulator.
2. Mark the enemy/own statue health bar crops and fill colors in
   `configs/reward.stickwar.json`.
3. Save win and loss screen templates and point `win_template`/`loss_template`
   at them.
4. Verify the action taps against a live match; update
   `configs/actions.stickwar.json`.
5. Fill the reset macro; set `"calibrated": true` only after checking multiple
   episodes.
