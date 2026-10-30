"""CLI: train small PPO on recorded episodes, or on a synthetic smoke batch.

This is batch/offline PPO: each update consumes a replay buffer recorded by an
Android actor. Behavior-cloning regularization (bc_coef) keeps the policy near
the data-collecting policy. Reports always say "batch PPO", never on-policy PPO.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionSpace
from .dataset import (
    DatasetError,
    dataset_fingerprint,
    load_episode,
    load_selected,
    split_episode_ids,
    transition_arrays,
)
from .ppo import PPOConfig, gae, tensors_for_training


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def system_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device_requested": device,
        "git_commit": git_commit(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
            }
    except ImportError:
        info["torch"] = None
    return info


def make_synthetic(n: int = 256, n_actions: int = 11, obs_size: int = 84, frame_stack: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 256, size=(n, frame_stack, obs_size, obs_size), dtype=np.uint8)
    actions = rng.integers(0, n_actions, size=n).astype(np.int64)
    rewards = rng.normal(0, 0.05, size=n).astype(np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    for i in range(31, n, 32):
        dones[i] = True
    old_logp = np.full(n, -np.log(n_actions), dtype=np.float32)
    old_values = rng.normal(0, 0.1, size=n).astype(np.float32)
    arrays = tensors_for_training(obs, actions, rewards, dones, old_logp, old_values)
    arrays["missing_logp"] = 0
    arrays["synthetic"] = True
    return arrays


def _calibration_guard(episodes, allow_uncalibrated: bool) -> dict[str, Any]:
    summary = {"calibrated": True, "uncalibrated_episodes": [], "reward_sources": set()}
    for episode in episodes:
        cfg = episode.meta.get("reward_config") or {}
        calibrated = bool(cfg.get("calibrated", False))
        summary["reward_sources"].add(str(cfg.get("source", "unknown")))
        if not calibrated:
            summary["calibrated"] = False
            summary["uncalibrated_episodes"].append(episode.episode_id)
    summary["reward_sources"] = sorted(summary["reward_sources"])
    if not summary["calibrated"] and not allow_uncalibrated:
        raise DatasetError(
            "dataset contains uncalibrated rewards; refusing to train because results "
            "would not be meaningful. Recalibrate the reward config or pass "
            "--allow-uncalibrated-reward explicitly (smoke/debug only)."
        )
    return summary


def _with_gae(arrays: dict, gamma: float = 0.99, lam: float = 0.95) -> dict:
    advantages, returns = gae(
        np.asarray(arrays["rewards"], dtype=float).tolist(),
        np.asarray(arrays["old_values"], dtype=float).tolist(),
        np.asarray(arrays["dones"], dtype=bool).tolist(),
        last_value=0.0,
        gamma=gamma,
        lam=lam,
    )
    result = dict(arrays)
    result["advantages"] = np.asarray(advantages, dtype=np.float32)
    result["returns"] = np.asarray(returns, dtype=np.float32)
    return result


def load_data(data_dir: str | Path, val_fraction: float = 0.0, seed: int = 0, allow_uncalibrated: bool = False, frame_size: int = 84, frame_stack: int = 4, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[dict, dict]:
    root = Path(data_dir)
    train_ids, val_ids = split_episode_ids(root, val_fraction=val_fraction, seed=seed) if val_fraction > 0 else (None, [])
    all_meta = [load_episode(p, load_frames=False) for p in sorted(root.iterdir()) if (p / "meta.json").exists()]
    guard = _calibration_guard(all_meta, allow_uncalibrated)
    train_eps = load_selected(root, train_ids) if train_ids is not None else [load_episode(p.path) for p in all_meta]
    train_arrays = _with_gae(transition_arrays(train_eps, frame_stack=frame_stack, obs_size=frame_size), gamma, gae_lambda)
    val_arrays = None
    if val_ids:
        val_arrays = _with_gae(transition_arrays(load_selected(root, val_ids), frame_stack=frame_stack, obs_size=frame_size), gamma, gae_lambda)
    info = {
        "data_dir": str(root),
        "fingerprint": dataset_fingerprint(root),
        "train_episodes": [e.episode_id for e in train_eps],
        "val_episodes": val_ids,
        "train_steps": int(len(train_arrays["actions"])),
        "val_steps": int(len(val_arrays["actions"])) if val_arrays is not None else 0,
        "missing_logp": int(train_arrays.get("missing_logp", 0)),
        "calibration": guard,
    }
    return {"train": train_arrays, "val": val_arrays}, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="episodes directory")
    parser.add_argument("--synthetic", action="store_true", help="run a smoke batch without game data")
    parser.add_argument("--out", default="runs/ppo")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--bc-coef", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--frame-size", type=int, default=84)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--n-actions", type=int, default=0, help="synthetic only; defaults to default action space")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--allow-uncalibrated-reward", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data and not args.synthetic:
        print(json.dumps({"error": "provide --data or --synthetic"}), file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print(json.dumps({"error": "PyTorch is not installed; use requirements.txt or Kaggle/CI"}), file=sys.stderr)
        return 2
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA required but torch.cuda.is_available() is false"}), file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_actions = args.n_actions or ActionSpace.default().n
    config = PPOConfig(
        obs_channels=args.frame_stack,
        obs_size=args.frame_size,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        bc_coef=args.bc_coef,
        target_kl=args.target_kl,
        seed=args.seed,
    )

    if args.synthetic:
        arrays = make_synthetic(
            n=args.synthetic_samples,
            n_actions=n_actions,
            obs_size=args.frame_size,
            frame_stack=args.frame_stack,
            seed=args.seed,
        )
        data_info = {
            "synthetic": True,
            "samples": args.synthetic_samples,
            "fingerprint": "synthetic",
            "calibration": {"calibrated": False, "note": "synthetic smoke batch"},
        }
    else:
        data, data_info = load_data(
            args.data,
            val_fraction=args.val_fraction,
            seed=args.seed,
            allow_uncalibrated=args.allow_uncalibrated_reward,
            frame_size=args.frame_size,
            frame_stack=args.frame_stack,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        arrays = data["train"]
        config.n_actions = int(max(arrays["actions"].max() + 1, 2))
        if args.val_fraction > 0 and data["val"] is not None:
            np.savez_compressed(out / "val_transitions.npz", **{k: v for k, v in data["val"].items() if isinstance(v, np.ndarray)})

    from .ppo import PPOTrainer
    if args.resume:
        trainer = PPOTrainer.load(args.resume, device=device)
        trainer.config.bc_coef = config.bc_coef
        trainer.config.epochs = config.epochs
        trainer.config.minibatch_size = config.minibatch_size
        start_update = trainer.update_count
    else:
        trainer = PPOTrainer(config, device=device, seed=args.seed)
        start_update = 0

    run_config = {
        "argv": sys.argv[1:] if argv is None else argv,
        "ppo": config.__dict__,
        "data": data_info,
        "checkpoint_in": args.resume,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "batch_ppo_plus_bc_anchor",
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
    slog = system_info(device)
    (out / "system.json").write_text(json.dumps(slog, indent=2) + "\n")
    print(json.dumps({"event": "start", "device": str(trainer.device), "system": slog, "data": data_info,
                      "updates": [start_update, args.updates]}, default=str), flush=True)

    metrics_path = out / "metrics.jsonl"
    history: list[dict] = []
    started = time.time()
    completed = True
    for update in range(start_update, args.updates):
        if args.max_minutes and (time.time() - started) / 60.0 >= args.max_minutes:
            print(json.dumps({"event": "max_minutes_reached", "update": update}), flush=True)
            completed = False
            break
        row = trainer.update(arrays)
        row["wall_s"] = round(time.time() - started, 3)
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if args.save_every > 0 and ((trainer.update_count % args.save_every == 0) or update == args.updates - 1):
            extra = {
                "data": data_info,
                "system": slog,
                "mode": "batch_ppo_plus_bc_anchor",
                "completed_updates": trainer.update_count,
                "requested_updates": args.updates,
            }
            trainer.save(out / "checkpoints" / f"update_{trainer.update_count:06d}.pt", extra=extra)
            trainer.save(out / "checkpoints" / "latest.pt", extra=extra)
    summary = {
        "status": "completed" if completed else "time_limited",
        "updates_completed": trainer.update_count,
        "updates_requested": args.updates,
        "samples_seen": trainer.sample_count,
        "elapsed_s": round(time.time() - started, 3),
        "device": str(trainer.device),
        "last_metrics": history[-1] if history else None,
        "data": data_info,
        "system": slog,
        "mode": "batch_ppo_plus_bc_anchor",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"event": "done", **summary}, default=str), flush=True)
    if completed and trainer.update_count >= args.updates:
        print("TRAIN_OK", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
