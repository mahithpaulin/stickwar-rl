# Verified results

This file separates measurements from assumptions. Times are UTC.

## 2026-10-30 — Kaggle GPU smoke (F): VERIFIED

- Kernel: <https://www.kaggle.com/code/mahithpaulin/stickwar-gpu-smoke>
- Kernel status: `COMPLETE`, version 2 (version 1 had a Python syntax error in the
  first line; the failure and fix are preserved in history).
- Environment reported by the kernel:
  - Python 3.13.15
  - torch 2.11.0+cu128
  - CUDA 12.8
  - GPU: Tesla T4, capability 7.5, 14.56 GB VRAM
  - peak allocated VRAM: 52.0 MB
  - real GPU matmul mean: -0.017591066658496857
- Three PPO-shaped updates on synthetic tensors completed:
  - iter 0: loss 1.3983277082443237
  - iter 1: loss 1.0711301565170288
  - iter 2: loss 1.2128983736038208
- Artifacts: `metrics.json` (908 bytes), `checkpoint.pt` (1.4 MB), log contains
  `SMOKE_OK`.
- Local download path at verification time:
  `$PREFIX/tmp/stickwar-kaggle-out2/`.
- Re-run through GitHub Actions (run
  [41065538141](https://github.com/mahithpaulin/stickwar-rl/actions/runs/41065538141),
  workflow `kaggle-gpu-smoke`): `KAGGLE_GPU_SMOKE_OK`, Tesla T4 14.56 GB,
  CUDA 12.8, torch 2.11.0+cu128, 3 updates, GPU matmul mean
  `-0.04299701750278473`, checkpoint 1,388,725 bytes.
- This proves the Kaggle account, GPU allocation, CUDA kernels, artifact download,
  and a PPO-shaped training step. It does **not** prove game skill.

## 2026-10-30 — Local unit/smoke tests

- `python -m pytest` on the phone: 32 passed, 2 skipped (PyTorch tests).
- GitHub Actions `python-ci` run
  [41066138098](https://github.com/mahithpaulin/stickwar-rl/actions/runs/41066138098):
  **success** — compile, unit tests, synthetic train/eval/report, Kaggle metadata
  validation. First CI run `41065122491` also passed.
- Local torch is not installed; torch tests run only in CI/Kaggle.

## 2026-10-30 — Android emulator infrastructure (A, C, D): VERIFIED

- Workflow: `android-smoke`, run
  [41066138143](https://github.com/mahithpaulin/stickwar-rl/actions/runs/41066138143).
- Runner: ubuntu-latest x86_64 with KVM; AVD API 30 `google_apis` x86_64 Pixel 4;
  SwiftShader, no GPU.
- Device: `sdk_gphone_x86_64`, SDK 30, 1080x2280, `ro.kernel.qemu=1`.
- Checks: booted, screenshot PNG, non-blank screenshot, screenshot changed after
  input, Settings search widget found, `wifi` typed and visible in UI dump,
  screenrecord exists, 3 episode steps, 3 JPEG frames. `errors: []`.
- Screenshots: 00_settings (mean 247.9, std 30.7), 01_search_typed (mean 232.9,
  std 62.4), 02_after_swipe (mean 253.9, std 10.8); non-black fraction 0.94–1.0.
- Video: 289 KB H.264 MP4 at 1280x720.
- Episode: `smoke-settings`, package `com.android.settings`, explicitly not game
  data; 3 steps / 3 frames.
- Iteration history: run `41065122173` failed because `/dev/kvm` was not writable
  by the runner user; `sudo chmod 666 /dev/kvm` fixed it. Run `41065537952` passed
  but only with a fallback tap under a "Pixel Launcher isn't responding" ANR
  dialog; the smoke was strengthened to require the real search widget and typed
  text, then passed in run `41066138143`.

## Android game (B, E): BLOCKED

No original APK exists on the machine and none was downloaded from third-party
mirrors. The `android-game` workflow was exercised without an APK in run
[41066913369](https://github.com/mahithpaulin/stickwar-rl/actions/runs/41066913369):
it failed at `Resolve APK source` and uploaded
`blocked.json` = `{"ok": false, "blocked": true, "reason": "no APK source
configured", "remedy": "supply a legally obtained APK via apk_url or
STICKWAR_APK_URL"}`. It did not fake a launch. E (a real game episode) is therefore blocked by B; the episode
recorder itself is verified only with Settings data. The expected ABI risk is
documented in `docs/ANDROID_COMPATIBILITY.md`.
