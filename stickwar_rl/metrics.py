"""Experiment report and dashboard generation (stdlib only)."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def svg_line_chart(rows: list[dict[str, Any]], key: str, width: int = 760, height: int = 220) -> str:
    values = [float(r[key]) for r in rows if key in r and r[key] is not None and math.isfinite(float(r[key]))]
    if not values:
        return f"<p class='muted'>no {html.escape(key)} data</p>"
    if len(values) == 1:
        values = values * 2
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    pad = 36
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad
    points = []
    for i, value in enumerate(values):
        x = pad + inner_w * i / max(1, len(values) - 1)
        y = pad + inner_h * (1.0 - (value - lo) / (hi - lo))
        points.append(f"{x:.2f},{y:.2f}")
    poly = " ".join(points)
    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' aria-label='{html.escape(key)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#0d1117'/>"
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#30363d'/>"
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#30363d'/>"
        f"<polyline fill='none' stroke='#58a6ff' stroke-width='2' points='{poly}'/>"
        f"<text x='{pad}' y='18' fill='#8b949e' font-size='13'>{html.escape(key)}</text>"
        f"<text x='4' y='{pad+5}' fill='#8b949e' font-size='11'>{hi:.4g}</text>"
        f"<text x='4' y='{height-pad+5}' fill='#8b949e' font-size='11'>{lo:.4g}</text>"
        f"</svg>"
    )


def render_report(run_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run / "report"
    out.mkdir(parents=True, exist_ok=True)
    metrics = read_jsonl(run / "metrics.jsonl")
    summary = _load_json(run / "summary.json")
    system = _load_json(run / "system.json")
    config = _load_json(run / "run_config.json")
    eval_data = _load_json(run / "eval" / "eval.json")
    report = {
        "run_dir": str(run),
        "metrics_count": len(metrics),
        "summary": summary,
        "system": system,
        "config": config,
        "last_metrics": metrics[-1] if metrics else None,
        "eval": eval_data or None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Stick War RL experiment report",
        "",
        f"- run directory: `{run}`",
        f"- mode: `{summary.get('mode', config.get('mode', 'unknown'))}`",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- updates: {summary.get('updates_completed', len(metrics))} / {summary.get('updates_requested', '?')}",
        f"- device: `{summary.get('device', system.get('device_requested', 'unknown'))}`",
        f"- torch: `{system.get('torch')}`",
        f"- CUDA available: `{system.get('cuda_available')}`",
    ]
    if system.get("gpu"):
        lines.append(f"- GPU: `{system['gpu'].get('name')}` ({system['gpu'].get('total_memory_gb')} GB)")
    if summary.get("data"):
        lines.append(f"- data fingerprint: `{summary['data'].get('fingerprint', '?')}`")
        lines.append(f"- train steps: {summary['data'].get('train_steps', '?')}")
    lines += ["", "## Metrics", ""]
    if metrics:
        keys = ["update", "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "bc_loss", "samples", "wall_s"]
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in metrics[-30:]:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    else:
        lines.append("No metrics found.")
    if eval_data:
        lines += ["", "## Evaluation", ""]
        p = eval_data.get("policy", {})
        b = eval_data.get("untrained_baseline", {})
        lines.append(f"- clipped IPS reward: policy {p.get('clipped_ips_reward')} vs baseline {b.get('clipped_ips_reward')}")
        lines.append(f"- accuracy: policy {p.get('accuracy')} vs baseline {b.get('accuracy')}")
        lines.append(f"- caveat: {eval_data.get('caveat', '')}")
    lines += [
        "",
        "## Provenance",
        "",
        "This run is batch/offline PPO with a behavior-cloning anchor, not strictly on-policy PPO. "
        "Results are only meaningful if the dataset reward calibration is marked calibrated.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))

    charts = "\n".join(svg_line_chart(metrics, key) for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl"))
    gpu = html.escape(str(system.get("gpu", {})))
    dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Stick War RL report</title>
<style>
body{{background:#010409;color:#c9d1d9;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:2rem;line-height:1.5}}
h1,h2{{color:#f0f6fc}} .muted{{color:#8b949e}} .chart{{width:100%;max-width:900px;background:#0d1117;border:1px solid #30363d;border-radius:8px;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;max-width:1200px}} td,th{{border:1px solid #30363d;padding:.35rem .55rem;text-align:right}} th{{background:#161b22}}
code{{color:#79c0ff}} .warn{{border-left:4px solid #d29922;padding:.5rem 1rem;background:#161b22}}
</style></head><body>
<h1>Stick War RL experiment report</h1>
<p class="muted">{html.escape(str(run))}</p>
<p>status=<code>{html.escape(str(summary.get('status')))}</code> mode=<code>{html.escape(str(summary.get('mode')))}</code> updates=<code>{html.escape(str(summary.get('updates_completed', len(metrics))))}</code></p>
<p>GPU: <code>{gpu}</code></p>
<div class="warn">Batch/offline PPO with a BC anchor. Offline metrics are proxies, not live game skill. Uncalibrated rewards cannot support training claims.</div>
<h2>Charts</h2>
{charts}
<h2>Run summary</h2>
<pre>{html.escape(json.dumps(summary, indent=2)[:20000])}</pre>
</body></html>"""
    (out / "dashboard.html").write_text(dashboard)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = render_report(args.run, args.out)
    print(json.dumps({"ok": True, "metrics_count": report["metrics_count"],
                      "out": str(Path(args.out) if args.out else Path(args.run) / "report")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
