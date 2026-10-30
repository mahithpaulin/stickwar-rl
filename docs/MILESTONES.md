# Milestones and acceptance criteria

Status is updated only from evidence. A green check requires a real run and an
artifact/log that another person can inspect.

## A. GitHub Actions can provision Android

- Workflow: `.github/workflows/android-smoke.yml`
- Criteria: hosted runner has `/dev/kvm`, an API 30 x86_64 AVD boots, `adb` sees
  it, and the job uploads artifacts.
- Evidence required: successful Actions run + `adb-devices.txt`, `sdk.txt`,
  `verification.json`.
- Status: **verified** in Actions run
  [41066138143](https://github.com/mahithpaulin/stickwar-rl/actions/runs/41066138143).
  Device `sdk_gphone_x86_64`, SDK 30, 1080x2280, KVM enabled, booted=true.

## B. Original APK installs and launches

- Workflow: `.github/workflows/android-game.yml`
- Criteria: SHA-256 verified APK, `adb install` succeeds, package becomes the
  foreground app, screenshot is non-blank.
- Blocker: no legally obtained Stick War APK is present on this machine; no
  third-party APK was downloaded.
- Status: **blocked pending user-supplied APK**, confirmed by run
  [41066913369](https://github.com/mahithpaulin/stickwar-rl/actions/runs/41066913369):
  `blocked.json` was uploaded and the job failed explicitly instead of faking a
  launch.

## C. Screenshots can be captured

- Script: `scripts/smoke_episode.py`
- Criteria: PNG > 1 KB, non-blank pixel variance, screenshot changes after input.
- Status: **verified**. Three PNGs (149 KB class, non-black fraction 0.94–1.0), a
  289 KB MP4, and a 3-step/3-frame episode in run `41066138143`.

## D. Automated touch input works

- Script: `scripts/smoke_episode.py`
- Criteria: UIAutomator locates the Settings search widget, tap changes the
  screen, typed text appears in the UI dump.
- Status: **verified**. `com.android.settings:id/search_action_bar_title` was
  located, tapped, `wifi` typed, and `text="wifi"` appeared in `search_ui.xml`.
  A first, weaker smoke passed only via a fallback tap under an ANR dialog; it was
  replaced by this stricter check, which now fails if the widget or text is absent.

## E. A game episode can be recorded

- Scripts: `stickwar_rl/runner.py`, `scripts/game_smoke.py`
- Criteria: original package foreground for the episode, ≥1 recorded transition
  with frame, valid `meta.json`, `steps.jsonl`, `frames/`.
- Extra requirement for "in-match": calibrated reset macro + human review.
  App-level episodes are explicitly marked `in_match_verified: false`.
- Status: **blocked by B** for gameplay. The episode format/recorder itself is
  verified with the Settings smoke (`smoke-settings`, 3 steps, 3 JPEG frames), but
  that is explicitly not Stick War and does not satisfy E.

## F. Kaggle can execute a small GPU training test

- Kernel: `kaggle/gpu_smoke`, id `mahithpaulin/stickwar-gpu-smoke`
- Criteria: `cuda_available: true`, real GPU name, GPU matmul, 3 PPO-shaped
  updates, saved `checkpoint.pt` and `metrics.json`.
- Status: **verified locally and via Actions** (run `41065538141`), Tesla T4, 3
  updates, 1,388,725-byte checkpoint (see `docs/RESULTS.md`).

## Gate before the full RL loop

Do not run expensive training until A–E pass and the reward config is calibrated.
The training CLI and Kaggle training workflow enforce the reward-calibration gate.
