"""CLI tests: `embsync corrupt` applies profiles and records ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.cli.main import main
from embodied_sync.corrupt import apply_profile, load_profile
from embodied_sync.datasets.io import (
    CORRUPTION_GROUND_TRUTH_NAME,
    load_corruption_ground_truth,
    load_run,
    save_run,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = REPO_ROOT / "configs" / "corrupt_camera_jitter.yaml"
KITCHEN_SINK_PROFILE = REPO_ROOT / "configs" / "corrupt_kitchen_sink.yaml"


def test_corrupt_writes_loadable_run_and_ground_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = tmp_path / "clean"
    out = tmp_path / "bad"
    clean = generate_synthetic_run(duration_s=1.0, seed=0)
    save_run(clean, clean_dir)

    assert main(["corrupt", str(clean_dir), "--profile", str(EXAMPLE_PROFILE), "--out", str(out)]) == 0

    profile = load_profile(EXAMPLE_PROFILE)
    expected = apply_profile(clean, profile)
    assert load_run(out) == expected.run
    assert load_corruption_ground_truth(out) == expected.dropped

    metadata_path = out / CORRUPTION_GROUND_TRUTH_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["profile_seed"] == 1234
    assert metadata["profile_path"] == str(EXAMPLE_PROFILE)
    assert set(metadata["dropped"]) == {"cam_front"}

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corruption"] == {
        "profile_path": str(EXAMPLE_PROFILE),
        "profile_seed": 1234,
    }
    stdout = capsys.readouterr().out
    assert "dropped samples recorded" in stdout


def test_corrupt_refuses_non_empty_out_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = tmp_path / "clean"
    out = tmp_path / "bad"
    save_run(generate_synthetic_run(duration_s=0.1, seed=0), clean_dir)
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")

    assert main(["corrupt", str(clean_dir), "--profile", str(EXAMPLE_PROFILE), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "non-empty" in err


def test_corrupt_preview_prints_summary_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--preview` on the kitchen-sink profile prints per-stream deltas
    and refuses to touch disk. No `--out` is required in preview mode."""
    clean_dir = tmp_path / "clean"
    would_be_out = tmp_path / "bad"  # deliberately not created
    save_run(generate_synthetic_run(duration_s=1.0, seed=0), clean_dir)

    rc = main(
        [
            "corrupt",
            str(clean_dir),
            "--profile",
            str(KITCHEN_SINK_PROFILE),
            "--preview",
        ]
    )
    assert rc == 0
    assert not would_be_out.exists(), "--preview must not create the out dir"

    stdout = capsys.readouterr().out
    assert "--preview" in stdout
    assert "samples_before" in stdout
    assert "samples_after" in stdout
    assert "no files written" in stdout
    # Kitchen-sink drops from cam_front and applies missing_interval to
    # robot_state; both must show a non-zero drop count in the summary.
    lines = {line.split()[0]: line for line in stdout.splitlines() if line and line[0].isalpha() and " " in line}
    assert "cam_front" in lines
    assert "robot_state" in lines


def test_corrupt_without_out_and_without_preview_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The safety guard on `--out` fails loudly, not silently."""
    clean_dir = tmp_path / "clean"
    save_run(generate_synthetic_run(duration_s=0.1, seed=0), clean_dir)

    rc = main(["corrupt", str(clean_dir), "--profile", str(EXAMPLE_PROFILE)])
    assert rc == 2
    assert "--out is required" in capsys.readouterr().err
