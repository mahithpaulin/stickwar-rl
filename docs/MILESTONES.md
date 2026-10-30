# Milestones and acceptance criteria

Status is updated only from evidence. A green check requires a real run and an
artifact/log that another person can inspect.

## A. GitHub Actions can provision Android

- Workflow: `.github/workflows/android-smoke.yml`
- Criteria: hosted runner has `/dev/kvm`, an API 30 x86_64 AVD boots, `adb` sees
  it, and the job uploads artifacts.
- Evidence required: successful Actions run + `adb-devices.txt`, `sdk.txt`,
  `verification.json`.
- Status: **pending first hosted run** (workflow committed, not yet executed).

## B. Original APK installs and launches

- Workflow: `.github/workflows/android-game.yml`
- Criteria: SHA-256 verified APK, `adb install` succeeds, package becomes the
  foreground app, screenshot is non-blank.
- Blocker: no legally obtained Stick War APK is present on this machine; no
  third-party APK was downloaded.
- Status: **blocked pending user-supplied APK**. The workflow fails with an
  explicit `BLOCKED` record, not a fabricated pass.

## C. Screenshots can be captured

- Script: `scripts/smoke_episode.py`
- Criteria: PNG > 1 KB, non-blank pixel variance, screenshot changes after input.
- Status: implemented; hosted evidence pending A.

## D. Automated touch input works

- Script: `scripts/smoke_episode.py`
- Criteria: UIAutomator locates the Settings search widget (or fallback), tap
  changes the screen, typed text appears in the UI dump.
- Status: implemented; hosted evidence pending A.

## E. A game episode can be recorded

- Scripts: `stickwar_rl/runner.py`, `scripts/game_smoke.py`
- Criteria: original package foreground for the episode, ≥1 recorded transition
  with frame, valid `meta.json`, `steps.jsonl`, `frames/`.
- Extra requirement for "in-match": calibrated reset macro + human review.
  App-level episodes are explicitly marked `in_match_verified: false`.
- Status: **blocked by B**; app-level recorder is exercised by the Settings smoke.

## F. Kaggle can execute a small GPU training test

- Kernel: `kaggle/gpu_smoke`, id `mahithpaulin/stickwar-gpu-smoke`
- Criteria: `cuda_available: true`, real GPU name, GPU matmul, 3 PPO-shaped
  updates, saved `checkpoint.pt` and `metrics.json`.
- Status: **verified** (see `docs/RESULTS.md`).

## Gate before the full RL loop

Do not run expensive training until A–E pass and the reward config is calibrated.
The training CLI and Kaggle training workflow enforce the reward-calibration gate.
