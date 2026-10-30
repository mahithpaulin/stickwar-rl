"""CLI: offline and live evaluation against an untrained/random baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetError
from .train import load_data, make_synthetic, system_info


def _policy_metrics(trainer, arrays: dict[str, Any], batch_size: int = 256) -> dict[str, float]:
    torch = trainer.torch
    F = torch.nn.functional
    n = len(arrays["actions"])
    total = {"mean_logp": 0.0, "entropy": 0.0, "value_mse": 0.0, "accuracy": 0.0, "bc_loss": 0.0}
    ratios = []
    rewards_all = np.asarray(arrays["rewards"], dtype=np.float64)
    old_logp_all = np.asarray(arrays["old_logp"], dtype=np.float64)
    reward_rows = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            obs = torch.as_tensor(arrays["obs"][start:end], dtype=torch.float32, device=trainer.device) / 255.0
            actions = torch.as_tensor(arrays["actions"][start:end], dtype=torch.long, device=trainer.device)
            logits, values = trainer.model(obs)
            logp_all = F.log_softmax(logits, -1)
            logp = logp_all.gather(-1, actions[:, None]).squeeze(-1)
            entropy = -(logits.softmax(-1) * logp_all).sum(-1)
            target = torch.as_tensor(arrays["returns"][start:end], dtype=torch.float32, device=trainer.device)
            total["mean_logp"] += float(logp.sum())
            total["entropy"] += float(entropy.sum())
            total["value_mse"] += float(F.mse_loss(values, target, reduction="sum"))
            total["accuracy"] += float((logits.argmax(-1) == actions).float().sum())
            total["bc_loss"] += float(F.cross_entropy(logits, actions, reduction="sum"))
            old_slice = old_logp_all[start:end]
            new_logp = logp.detach().cpu().numpy()
            ratios.append(np.exp(new_logp - old_slice))
            reward_rows.append(rewards_all[start:end])
    ratios_arr = np.concatenate(ratios) if ratios else np.ones(1)
    reward_arr = np.concatenate(reward_rows) if reward_rows else np.zeros(1)
    valid = np.isfinite(ratios_arr)
    clipped = np.clip(ratios_arr[valid], 0.0, 10.0)
    ips = float((clipped * reward_arr[valid]).sum() / max(1e-8, clipped.sum())) if valid.any() else float("nan")
    result = {k: v / max(1, n) for k, v in total.items()}
    result["n"] = n
    result["behavior_mean_reward"] = float(reward_arr.mean())
    result["clipped_ips_reward"] = ips
    result["ratio_mean"] = float(np.nanmean(ratios_arr))
    result["ratio_max"] = float(np.nanmax(ratios_arr))
    result["ips_valid_fraction"] = float(valid.mean())
    return result


def offline_eval(checkpoint: str | Path, data_dir: str | Path, out: Path, val_fraction: float = 0.0, allow_uncalibrated: bool = False, device: str = "cpu") -> dict[str, Any]:
    from .ppo import PPOConfig, PPOTrainer

    trainer = PPOTrainer.load(checkpoint, device=device, resume_optimizer=False)
    data, info = load_data(
        data_dir,
        val_fraction=val_fraction,
        seed=0,
        allow_uncalibrated=allow_uncalibrated,
        frame_size=trainer.config.obs_size,
        frame_stack=trainer.config.obs_channels,
        gamma=trainer.config.gamma,
        gae_lambda=trainer.config.gae_lambda,
    )
    arrays = data["val"] if data["val"] is not None else data["train"]
    policy_metrics = _policy_metrics(trainer, arrays)

    baseline = PPOTrainer(PPOConfig(**trainer.config.__dict__), device=device, seed=trainer.config.seed + 991)
    baseline_metrics = _policy_metrics(baseline, arrays)
    result = {
        "mode": "offline",
        "checkpoint": str(checkpoint),
        "data": info,
        "policy": policy_metrics,
        "untrained_baseline": baseline_metrics,
        "delta": {
            "clipped_ips_reward": policy_metrics["clipped_ips_reward"] - baseline_metrics["clipped_ips_reward"],
            "mean_logp": policy_metrics["mean_logp"] - baseline_metrics["mean_logp"],
            "entropy": policy_metrics["entropy"] - baseline_metrics["entropy"],
            "accuracy": policy_metrics["accuracy"] - baseline_metrics["accuracy"],
        },
        "caveat": (
            "Offline metrics are proxies on logged transitions, not live game skill. "
            "Clipped IPS is biased when the policy diverges from the behavior policy. "
            "Use live evaluation once the APK is available."
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": system_info(device),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "eval.md").write_text(render_markdown(result) + "\n")
    print(json.dumps(result, indent=2))
    return result


def live_eval(checkpoint: str | Path | None, config_path: str, episodes: int, duration: float, out: Path, allow_uncalibrated: bool, serial: str | None) -> dict[str, Any]:
    from dataclasses import replace
    from .env import AndroidEnv, AndroidEnvConfig
    from .policies import RandomPolicy, TorchPolicy, make_policy

    cfg = AndroidEnvConfig.load(config_path)
    cfg = replace(cfg, serial=serial, allow_uncalibrated_reward=allow_uncalibrated)
    env = AndroidEnv(cfg)
    policy = make_policy(f"checkpoint:{checkpoint}" if checkpoint else "random", env.action_space.n)
    baseline = RandomPolicy(env.action_space.n, seed=cfg.seed + 1234)
    max_steps = max(1, int(round(duration / max(0.05, cfg.step_interval_s))))
    results: dict[str, Any] = {"mode": "live", "config": config_path, "episodes": episodes,
                               "duration_s": duration, "policy": getattr(policy, "name", "unknown"),
                               "checkpoint": str(checkpoint) if checkpoint else None,
                               "baseline": "random"}
    try:
        for label, agent in (("policy", policy), ("random_baseline", baseline)):
            runs = []
            for i in range(episodes):
                obs = env.reset()
                total = 0.0
                steps = 0
                done = False
                while not done and steps < max_steps:
                    decision = agent.act(obs)
                    step = env.step(decision.action)
                    total += step.reward
                    obs = step.obs
                    steps += 1
                    done = step.done
                runs.append({"episode": i, "steps": steps, "reward_sum": total,
                             "done": done, "final_signals": env.last_signals.to_dict()})
                print(json.dumps({"eval": label, **runs[-1]}), flush=True)
            results[label] = {
                "runs": runs,
                "mean_reward": float(np.mean([r["reward_sum"] for r in runs])) if runs else 0.0,
                "mean_steps": float(np.mean([r["steps"] for r in runs])) if runs else 0.0,
            }
    finally:
        env.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_live.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out / "eval_live.md").write_text(render_live_markdown(results) + "\n")
    print(json.dumps(results, indent=2, default=str))
    return results


def render_markdown(result: dict[str, Any]) -> str:
    p = result["policy"]
    b = result["untrained_baseline"]
    d = result["delta"]
    return "\n".join([
        "# Offline evaluation",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- data: `{result['data']['data_dir']}`",
        f"- transitions: {p['n']}",
        "",
        "| metric | policy | untrained baseline | delta |",
        "|---|---:|---:|---:|",
        f"| clipped IPS reward | {p['clipped_ips_reward']:.5f} | {b['clipped_ips_reward']:.5f} | {d['clipped_ips_reward']:+.5f} |",
        f"| mean logp | {p['mean_logp']:.5f} | {b['mean_logp']:.5f} | {d['mean_logp']:+.5f} |",
        f"| entropy | {p['entropy']:.5f} | {b['entropy']:.5f} | {d['entropy']:+.5f} |",
        f"| accuracy | {p['accuracy']:.5f} | {b['accuracy']:.5f} | {d['accuracy']:+.5f} |",
        "",
        "> " + result["caveat"],
        "",
    ])


def render_live_markdown(result: dict[str, Any]) -> str:
    p = result.get("policy_result", {})
    return "\n".join([
        "# Live evaluation",
        "",
        f"- checkpoint: `{result.get('checkpoint')}`",
        f"- episodes: {result.get('episodes')}",
        f"- duration: {result.get('duration_s')} s",
        f"- policy mean reward: {result.get('policy', {}).get('mean_reward', 0.0):.4f}",
        f"- random baseline mean reward: {result.get('random_baseline', {}).get('mean_reward', 0.0):.4f}",
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", default="runs/eval")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--allow-uncalibrated-reward", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--config", default="configs/env.emulator.json")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--serial", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    try:
        if args.live:
            live_eval(args.checkpoint, args.config, args.episodes, args.duration, out, args.allow_uncalibrated_reward, args.serial)
            return 0
        if args.synthetic:
            from .ppo import PPOConfig, PPOTrainer
            from .ppo import tensors_for_training
            arrays = make_synthetic(n=128, seed=0)
            trainer = PPOTrainer(PPOConfig(n_actions=11), device=args.device)
            baseline = PPOTrainer(PPOConfig(n_actions=11), device=args.device, seed=99)
            result = {"mode": "synthetic", "policy": _policy_metrics(trainer, arrays),
                      "untrained_baseline": _policy_metrics(baseline, arrays)}
            result["delta"] = {k: result["policy"][k] - result["untrained_baseline"][k]
                               for k in ("clipped_ips_reward", "mean_logp", "entropy", "accuracy")}
            result["caveat"] = "Synthetic smoke evaluation only; not game skill."
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
            (out / "eval.md").write_text(render_markdown({**result, "checkpoint": "synthetic",
                                                           "data": {"data_dir": "synthetic"}}) + "\n")
            print(json.dumps(result, indent=2))
            print("EVAL_OK")
            return 0
        if not args.checkpoint or not args.data:
            print(json.dumps({"error": "offline eval needs --checkpoint and --data"}), file=sys.stderr)
            return 2
        offline_eval(args.checkpoint, args.data, out, args.val_fraction, args.allow_uncalibrated_reward, args.device)
        return 0
    except DatasetError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
