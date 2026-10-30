# Agent notes

## Standing constraints

- Never download Stick War APKs from untrusted mirrors. The repo contains no APK.
- Never commit game binaries, tokens, or large model files. Use releases/artifacts.
- Do not run `pm clear`, uninstall, or rewrite device data without approval.
- Do not run PyTorch training on the phone; use Kaggle/CI.
- Keep Android B–E marked blocked/pending until a real APK run proves them.
- Batch PPO + BC anchor is not on-policy PPO; do not relabel it.

## Current verified state

- Kaggle GPU smoke: T4, CUDA 12.8, torch 2.11.0, 3 updates, checkpoint.
- Local non-torch tests: 32 passing.
- GitHub Actions: workflows committed; hosted Android run not yet executed.
- Original APK: absent, so B–E blocked.

## Commands

```bash
python -m pytest -q
python -m compileall -q stickwar_rl scripts
python -m stickwar_rl.train --synthetic --updates 2 --out runs/dev --device cpu
python -m stickwar_rl.evaluate --synthetic --out runs/dev/eval --device cpu
python -m stickwar_rl.metrics --run runs/dev
gh workflow run python-ci
gh workflow run android-smoke
gh workflow run kaggle-gpu-smoke
```

## State

Update `state/task_state.json` and `docs/RESULTS.md` after each verified run.
Evidence must include run IDs, artifact names, or Kaggle kernel URLs. Do not record
assumptions as results.
