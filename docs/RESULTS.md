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
- This proves the Kaggle account, GPU allocation, CUDA kernels, artifact download,
  and a PPO-shaped training step. It does **not** prove game skill.

## 2026-10-30 — Local unit/smoke tests

- `python -m pytest` on the phone for non-PyTorch tests: 32 passed.
- PyTorch tests are configured for GitHub Actions CI (`python-ci` workflow).
- Synthetic training/eval/report is wired into CI; local torch is not installed.

## Android (A–E)

Not yet verified on a hosted emulator. The phone has no Android emulator and no
CUDA; the GitHub Actions workflow is the execution target. Evidence will be added
here from the Actions run and artifact names.

Blocked by B: no original APK on the machine. No APK was downloaded from
third-party mirrors.
