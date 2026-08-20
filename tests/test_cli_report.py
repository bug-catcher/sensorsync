"""CLI tests: `embsync report` produces HTML (and optionally JSON summary).

Full pipeline exercised: synth → save_run → align → save_episode →
report → HTML + JSON output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.align import align_run
from embodied_sync.cli.main import main
from embodied_sync.datasets.io import save_episode, save_run
from embodied_sync.streams.synthetic import generate_synthetic_run


def _episode_dir(tmp_path: Path) -> Path:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    run_dir = tmp_path / "run"
    save_run(run, run_dir)
    aligned = align_run(run, target_rate_hz=10.0)
    episode = tmp_path / "episode"
    save_episode(
        aligned,
        episode,
        target_rate_hz=10.0,
        extra_manifest={"source_run": str(run_dir), "method": "nearest_neighbor"},
    )
    return episode


def test_report_writes_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode = _episode_dir(tmp_path)
    out = tmp_path / "report.html"
    assert main(["report", str(episode), "--out", str(out)]) == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "cam_front" in text
    stdout = capsys.readouterr().out
    assert "sync-quality report" in stdout


def test_report_html_pulls_source_run_and_rate_from_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode = _episode_dir(tmp_path)
    out = tmp_path / "report.html"
    assert main(["report", str(episode), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Target rate" in text
    assert "Source run" in text


def test_report_writes_json_summary_when_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode = _episode_dir(tmp_path)
    out = tmp_path / "report.html"
    summary = tmp_path / "summary.json"
    assert (
        main(
            [
                "report",
                str(episode),
                "--out",
                str(out),
                "--json-summary",
                str(summary),
            ]
        )
        == 0
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["format_version"] == 0
    assert payload["frame_count"] > 0
    assert {"streams", "frame_count", "format_version"} <= payload.keys()


def test_report_title_flag_reflected_in_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode = _episode_dir(tmp_path)
    out = tmp_path / "report.html"
    assert (
        main(
            [
                "report",
                str(episode),
                "--out",
                str(out),
                "--title",
                "Demo run report",
            ]
        )
        == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "Demo run report" in text


def test_report_missing_episode_fails_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.html"
    assert (
        main(
            [
                "report",
                str(tmp_path / "does_not_exist"),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "not an episode" in err or "does_not_exist" in err
