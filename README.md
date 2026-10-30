# Stick War: Legacy — RL research pipeline

An Android actor + batch PPO research platform for the original **Stick War:
Legacy** (package `com.maxgames.stickwarlegacy`). It drives the unmodified game
through ADB (screenshots, taps, screen recording), records episodes, and trains a
small visual PPO policy on Kaggle GPUs.

**This repository contains no game APK and no game assets.** You must supply a
legally obtained copy. Do not redistribute the APK. This is personal,
non-commercial research; respect Max Games Studio's terms and Google Play terms.
No ads, purchases, or DRM are bypassed.

## Honest status (2026-10-30)

| Milestone | Status | Evidence |
|---|---|---|
| A. GitHub Actions provisions Android emulator | **Verified** | Actions run `41066138143`: KVM, API 30 x86_64 Pixel 4, booted |
| B. Original APK installs and launches | **Blocked: no APK supplied** | `android-game.yml` fails with `BLOCKED` instead of faking |
| C. Screenshots from Android | **Verified** | Three non-blank PNGs + 289 KB MP4, run `41066138143` |
| D. Automated touch input | **Verified** | Found Settings search widget, tapped, typed `wifi`, text present in UI dump |
| E. Game episode recorded | **Blocked by B**; episode format verified with Settings | 3 steps / 3 frames from `smoke-settings`, not gameplay |
| F. Kaggle GPU training test | **Verified locally and via Actions** | Tesla T4, CUDA 12.8, torch 2.11.0, 3 PPO updates, 1,388,725-byte checkpoint |

See [`docs/RESULTS.md`](docs/RESULTS.md) for raw evidence and
[`docs/MILESTONES.md`](docs/MILESTONES.md) for acceptance criteria.

## What is implemented

- `stickwar_rl/adb.py` — retrying ADB wrapper: boot, screenshots, taps, swipes,
  text, screenrecord, package install/launch/foreground checks, UI dumps.
- `stickwar_rl/env.py` — `AndroidEnv`: visual observations, frame stack, reset
  macro, touch actions, foreground/terminal checks.
- `stickwar_rl/actions.py` — 11-action semantic tap space calibrated from the
  official 16:9 screenshots (provisional until verified on-device).
- `stickwar_rl/rewards.py` — configurable bar/template rewards; the shipped
  reward config is explicitly **uncalibrated** and training refuses it by default.
- `stickwar_rl/dataset.py` — episode format (`meta.json`, `steps.jsonl`,
  `frames/`), manifests, transitions, deterministic train/val splits.
- `stickwar_rl/ppo.py` — small CNN actor-critic, GAE, clipped PPO + BC anchor,
  atomic checkpoints, optimizer/RNG resume.
- `stickwar_rl/train.py`, `evaluate.py`, `metrics.py` — training, offline/live
  evaluation, JSONL metrics, Markdown/HTML dashboard.
- `scripts/kaggle_job.py` — push/poll/download Kaggle kernels without a hosted runner.
- GitHub Actions run evidence: Android `41066138143`, Kaggle `41065538141`, CI
  `41066138098`; the no-APK block is demonstrated by `android-game` run
  `41066913369`.
- GitHub Actions: CI, Android smoke, game smoke, Kaggle GPU smoke, Kaggle
  training, offline evaluation.
- Kaggle jobs: `kaggle/gpu_smoke` (verified T4 test) and `kaggle/ppo_train`
  (batch PPO from recorded episodes).

## Install and test

Requires Python 3.10+. On Termux, `pkg install python-numpy python-pillow` is
enough for the non-PyTorch tests. PyTorch training runs on CI/Kaggle, not the phone.

```bash
python -m pip install -r requirements.txt   # torch included
python -m pytest -q
python -m stickwar_rl.train --synthetic --updates 3 --out runs/ci --device cpu \
  --epochs 1 --minibatch-size 32 --frame-size 32 --frame-stack 2 --synthetic-samples 64
python -m stickwar_rl.evaluate --synthetic --out runs/ci/eval --device cpu
python -m stickwar_rl.metrics --run runs/ci
```

`runs/ci/report/dashboard.html` is the experiment dashboard. This synthetic run
proves the plumbing, not game skill.

## Android execution

### 1. Emulator infrastructure smoke (no APK)

Run **Actions → android-smoke → Run workflow**. It provisions an API 30
`google_apis` x86_64 Pixel 4 AVD with KVM, captures screenshots, taps the Settings
search field, types text, records a short MP4, and writes `verification.json`.
It is explicitly labelled infrastructure-only; it is not Stick War gameplay. The
verified run booted `sdk_gphone_x86_64` (SDK 30), found
`com.android.settings:id/search_action_bar_title`, tapped it, typed `wifi`,
confirmed the text in a UI dump, recorded a 289 KB MP4, and wrote a 3-step /
3-frame episode.

### 2. Original game APK

The APK is intentionally not in Git, artifacts, or releases. Supply a legally
obtained APK in one of two ways:

```bash
# Set a signed/private HTTPS URL (recommended: repository secret)
gh secret set STICKWAR_APK_URL
gh secret set STICKWAR_APK_SHA256
```

or pass `apk_url` / `apk_sha256` to **Actions → android-game → Run workflow**.
The workflow downloads to the runner temp directory, verifies SHA-256, inspects
ABIs with Python, installs with `adb install -r -g`, launches, captures
screenshots, attempts touches, records video/an episode, and tests reset via
force-stop + relaunch. It deletes the APK afterwards and never uploads it.

The job reports `in_match_verified: false` until a calibrated reset macro exists
and a human has checked the screenshots. Do not interpret an app-level episode as
gameplay.

#### ABI warning

An x86_64 GitHub runner cannot execute an ARM-only Unity APK unless the system
image provides ARM translation. The workflow reports APK ABIs and the exact
`adb install` failure instead of pretending. Alternatives are documented in
[`docs/ANDROID_COMPATIBILITY.md`](docs/ANDROID_COMPATIBILITY.md), including using
the physical phone as an ADB actor.

### 3. Local physical device

Enable wireless debugging, pair once, then:

```bash
adb pair <phone-ip>:<pair-port>       # enter pairing code
adb connect <phone-ip>:<port>
adb devices -l
python -m stickwar_rl.probe --package com.maxgames.stickwarlegacy
python -m stickwar_rl.collect --config configs/env.emulator.json \
  --episodes 1 --duration 60 --policy random \
  --allow-uncalibrated-reward --out data/episodes
```

The collector refuses to train on uncalibrated rewards by default. To calibrate,
take real in-match screenshots, update `configs/reward.stickwar.json` and
`configs/actions.stickwar.json`, set
`"calibrated": true`, and add the win/loss templates and reset macro. The files
currently say `provisional` / `calibrated: false` on purpose.

## Kaggle GPU training

The GPU smoke is already verified locally. To run it through GitHub Actions:

```bash
gh secret set KAGGLE_ACCESS_TOKEN < ~/.kaggle/access_token
gh secret set KAGGLE_USERNAME --body mahithpaulin
gh workflow run kaggle-gpu-smoke
```

It pushes `kaggle/gpu_smoke`, waits for completion, downloads `metrics.json` and
`checkpoint.pt`, and fails unless a real Tesla T4 run reports CUDA and completes
the PPO-shaped updates.

Real training (after B–E pass, data exists, and rewards are calibrated):

```bash
gh workflow run kaggle-ppo-train \
  -f data_run_id=<android-game-run-id> \
  -f data_artifact=android-game-<run-id> \
  -f updates=100 \
  -f resume_url= \
  -f release_tag=ppo-v1
```

The workflow stages a private Kaggle dataset from the episode artifact, clones the
public repo at the exact commit in the kernel, trains on CUDA, downloads
checkpoints/metrics, and publishes them to a versioned GitHub release. Checkpoints
are saved every update so a session can resume.

Without game data, the training workflow refuses to run unless `allow_synthetic`
is explicitly true. Synthetic results are never presented as game results.

## PPO design and its limits

- Observation: 4 stacked 84×84 grayscale frames from 16:9 screenshots.
- Actions: 11 semantic taps (`noop`, six unit cards, spell, retreat, defend,
  attack). Coordinates are normalized and easily recalibrated.
- Reward: sparse win/loss plus optional calibrated statue-health bar deltas and
  step penalty. **The shipped reward is uncalibrated**; it cannot support a
  training claim.
- Algorithm: batch PPO with GAE and a behavior-cloning anchor (`bc_coef`). It is
  **not** strictly on-policy PPO: Android actors collect, Kaggle updates, and the
  next batch uses the new checkpoint. Reports say `batch_ppo_plus_bc_anchor`.
- Baseline: a fresh/random policy evaluated on the same data (offline) or the same
  live episodes (random baseline). Offline clipped-IPS is a proxy, not skill.
- Reproducibility: seeds, git commit, dataset fingerprint, environment, checkpoint
  hash, and run config are written next to each run.

## Repository layout

```
stickwar_rl/        core package (ADB, env, actions, rewards, PPO, dataset, metrics)
scripts/            Android/Kaggle orchestration and evidence scripts
configs/            action/reward/env/reset calibration (provisional)
kaggle/gpu_smoke/   verified T4 CUDA + PPO-shape smoke
kaggle/ppo_train/   batch PPO job (clones the repo at a pinned commit)
.github/workflows/  CI, Android, Kaggle, evaluation
docs/               milestones, results, runbook, compatibility notes
state/              persistent autonomous task state
```

## Safety / limits

- No destructive device operations are performed automatically. Installation uses
  `adb install -r -g`; no `pm clear`, no uninstall on signature mismatch.
- Tokens are read from GitHub Secrets or `~/.kaggle/access_token`; never printed.
- GitHub-hosted emulators have no GPU; the game runs under SwiftShader software
  rendering, so performance measurements are not representative.
- Kaggle T4 sessions are capped at 12 hours; GitHub job timeout is below that.
  Save every update and resume from the published checkpoint.
- The phone itself has no CUDA and no Android emulator acceleration; it is only an
  ADB actor/scheduler.
