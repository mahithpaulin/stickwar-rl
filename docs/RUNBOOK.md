# Runbook

## 0. Constraints

- Never commit the APK, game assets, tokens, or large checkpoints to Git history.
- Do not download the APK from untrusted mirrors.
- Do not run `pm clear` or uninstall the game automatically.
- The phone has no CUDA; do not run PyTorch training locally.
- Hosted emulators have no GPU; game performance is not representative.

## 1. Local pure-Python checks

```bash
cd ~/stickwar-rl
python -m pytest -q tests/test_actions.py tests/test_rewards.py \
  tests/test_ppo_gae.py tests/test_dataset.py tests/test_observations.py tests/test_configs.py
python -m compileall -q stickwar_rl scripts
```

## 2. GitHub repository and secrets

```bash
gh repo view mahithpaulin/stickwar-rl
gh secret set KAGGLE_ACCESS_TOKEN < ~/.kaggle/access_token
gh secret set KAGGLE_USERNAME --body mahithpaulin
# Optional, only when a legally obtained APK is available:
gh secret set STICKWAR_APK_URL
gh secret set STICKWAR_APK_SHA256
```

Never echo a secret. `gh secret set` reads from the file/terminal; the value is
not printed by these commands.

## 3. Android emulator infrastructure (A, C, D, E-infra)

```bash
gh workflow run android-smoke
gh run list --workflow android-smoke --limit 1
gh run watch <run-id>
gh run download <run-id> -n android-smoke-<run-id> -D artifacts/android-smoke
```

Inspect `verification.json`, screenshots, `screenrecord.mp4`, `episode/`. A pass
means ADB, screenshots, touch, screen recording, and episode serialization work.
It does **not** mean Stick War launched.

## 4. Original game (B, E)

Set `STICKWAR_APK_URL`/`STICKWAR_APK_SHA256` or pass them as inputs:

```bash
gh workflow run android-game \
  -f apk_url='https://.../legally-obtained.apk' \
  -f apk_sha256='<sha256>' \
  -f package='com.maxgames.stickwarlegacy' \
  -f duration_s=60 \
  -f episodes=1 \
  -f require_match=false
```

If the URL is private/signed, use the secret instead of the input so it is not
displayed in run metadata. The workflow downloads to `$RUNNER_TEMP`, verifies the
hash, installs, launches, records evidence, and deletes the APK. It uploads no APK.

`require_match=true` is the honest gate for gameplay episodes: it fails until the
reset macro is calibrated and `in_match_verified` is true.

## 5. Local physical device (alternative to B)

```bash
adb pair 192.168.1.50:37123        # one-time, code from Wireless debugging
adb connect 192.168.1.50:41234
adb devices -l
python -m stickwar_rl.probe --package com.maxgames.stickwarlegacy
python -m stickwar_rl.collect \
  --config configs/env.emulator.json \
  --episodes 1 --duration 60 --policy random \
  --allow-uncalibrated-reward --out data/episodes
```

Calibrate `configs/actions.stickwar.json`, `configs/reward.stickwar.json`, and
`configs/reset.stickwar.json` from real in-match screenshots before training.
The collector records rewards but the trainer rejects uncalibrated datasets unless
`--allow-uncalibrated-reward` is given.

## 6. Kaggle GPU smoke (F)

```bash
python scripts/kaggle_job.py validate --job kaggle/gpu_smoke
python scripts/kaggle_job.py push --job kaggle/gpu_smoke --wait \
  --download artifacts/kaggle-smoke
python scripts/kaggle_smoke_report.py artifacts/kaggle-smoke
```

Via Actions: **kaggle-gpu-smoke → Run workflow**. It requires the Kaggle secret.

## 7. Kaggle PPO training (after A–E and calibration)

```bash
gh workflow run kaggle-ppo-train \
  -f data_run_id=<android-game-run-id> \
  -f data_artifact=android-game-<run-id> \
  -f updates=100 \
  -f release_tag=ppo-v1
```

The job:

1. downloads the episode artifact from the Actions run,
2. publishes a private Kaggle dataset,
3. stages `kaggle/ppo_train` with a pinned commit and optional resume URL,
4. waits for the T4 run and downloads checkpoints/metrics,
5. publishes the checkpoint/metrics to a GitHub release if `release_tag` is set.

Checkpoints are saved every update (`checkpoints/update_NNNNNN.pt`, `latest.pt`).
To resume, pass `resume_url` pointing at a public release asset.

## 8. Evaluation

```bash
gh workflow run evaluate \
  -f checkpoint_url='https://.../checkpoint.pt' \
  -f data_run_id=<run-id> \
  -f data_artifact=android-game-<run-id>
```

Offline numbers are proxies on logged transitions. Live evaluation needs a device
and a calibrated reward config:

```bash
python -m stickwar_rl.evaluate --live --checkpoint checkpoints/latest.pt \
  --config configs/env.emulator.json --episodes 3 --duration 120 \
  --allow-uncalibrated-reward
```

## 9. Reporting

```bash
python -m stickwar_rl.metrics --run runs/ppo
```

Open `runs/ppo/report/dashboard.html` and read `report.md`. The report records
mode `batch_ppo_plus_bc_anchor`, GPU, data fingerprint, and the calibration caveat.

## 10. Interruption handling

- Training resumes from `latest.pt`; metric history is append-only JSONL.
- Kaggle sessions are capped at 12 h; the Actions job timeout is below that.
- If a Kaggle run is interrupted, download its output anyway, then resume from the
  newest checkpoint in the next job.
- The Android game is restarted with force-stop + launcher; no destructive reset.
