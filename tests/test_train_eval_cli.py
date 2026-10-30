import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stickwar_rl import evaluate, train


def test_train_synthetic_and_resume(tmp_path: Path):
    out = tmp_path / "run"
    rc = train.main(["--synthetic", "--updates", "2", "--out", str(out),
                     "--device", "cpu", "--epochs", "1", "--minibatch-size", "32",
                     "--frame-size", "32", "--frame-stack", "2", "--synthetic-samples", "64"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["updates_completed"] == 2
    assert summary["mode"] == "batch_ppo_plus_bc_anchor"
    assert (out / "checkpoints" / "latest.pt").exists()
    assert len((out / "metrics.jsonl").read_text().strip().splitlines()) == 2

    rc = train.main(["--synthetic", "--updates", "3", "--out", str(out),
                     "--resume", str(out / "checkpoints" / "latest.pt"),
                     "--device", "cpu", "--epochs", "1", "--minibatch-size", "32",
                     "--frame-size", "32", "--frame-stack", "2", "--synthetic-samples", "64"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["updates_completed"] == 3


def test_evaluate_synthetic(tmp_path: Path):
    out = tmp_path / "eval"
    rc = evaluate.main(["--synthetic", "--out", str(out), "--device", "cpu"])
    assert rc == 0
    result = json.loads((out / "eval.json").read_text())
    assert result["mode"] == "synthetic"
    assert "clipped_ips_reward" in result["policy"]
    assert "untrained_baseline" in result


def test_report(tmp_path: Path):
    out = tmp_path / "run"
    assert train.main(["--synthetic", "--updates", "1", "--out", str(out),
                       "--device", "cpu", "--epochs", "1", "--minibatch-size", "32",
                       "--frame-size", "32", "--frame-stack", "2", "--synthetic-samples", "32"]) == 0
    from stickwar_rl.metrics import render_report
    report = render_report(out)
    assert report["metrics_count"] == 1
    assert (out / "report" / "dashboard.html").exists()
    assert (out / "report" / "report.md").exists()
