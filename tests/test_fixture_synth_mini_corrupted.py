"""Committed-fixture guard: `data/fixtures/synth_mini_corrupted/`.

The fixture is the same 1-second synth_mini run run through the
committed ``configs/corrupt_camera_jitter.yaml`` profile (the same
profile ``docs/user/quickstart.md`` walks through). It pins the
corruption-application layer as a stable byte-level contract, in
parallel with the run-format (``synth_mini/``) and aligned-episode
(``synth_mini_aligned/``) fixtures.

If the corruption engine or the run format changes, regeneration no
longer matches the committed bytes and the diff must be made
deliberately (regeneration command in ``data/fixtures/README.md``).
"""

from __future__ import annotations

from pathlib import Path

from embodied_sync.cli.main import main
from embodied_sync.corrupt import apply_profile, load_profile
from embodied_sync.datasets.io import (
    CORRUPTION_GROUND_TRUTH_NAME,
    load_corruption_ground_truth,
    load_run,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "data" / "fixtures" / "synth_mini_corrupted"
SOURCE_RUN = REPO_ROOT / "data" / "fixtures" / "synth_mini"
PROFILE = REPO_ROOT / "configs" / "corrupt_camera_jitter.yaml"


def test_fixture_loads_and_matches_apply_profile() -> None:
    loaded = load_run(FIXTURE_DIR)
    expected = apply_profile(
        generate_synthetic_run(duration_s=1.0, seed=0),
        load_profile(PROFILE),
    )
    assert loaded == expected.run
    assert load_corruption_ground_truth(FIXTURE_DIR) == expected.dropped


def test_fixture_is_byte_identical_to_regeneration(tmp_path: Path) -> None:
    regen_dir = tmp_path / "synth_mini_corrupted"
    # Match the regeneration command in data/fixtures/README.md — cwd is
    # the repo root, so the manifest's `profile_path` echoes the exact
    # relative string that appears there.
    rc = main(
        [
            "corrupt",
            str(SOURCE_RUN.relative_to(REPO_ROOT)),
            "--profile",
            str(PROFILE.relative_to(REPO_ROOT)),
            "--out",
            str(regen_dir),
        ]
    )
    assert rc == 0

    fixture_files = sorted(p.relative_to(FIXTURE_DIR) for p in FIXTURE_DIR.rglob("*.json*"))
    regen_files = sorted(p.relative_to(regen_dir) for p in regen_dir.rglob("*.json*"))
    assert fixture_files == regen_files
    for rel in fixture_files:
        assert (FIXTURE_DIR / rel).read_bytes() == (regen_dir / rel).read_bytes(), (
            f"format drift in {rel}: committed fixture differs from regeneration "
            f"(see data/fixtures/README.md)"
        )


def test_fixture_ground_truth_targets_cam_front_drops() -> None:
    dropped = load_corruption_ground_truth(FIXTURE_DIR)
    # Camera jitter profile drops from cam_front only; other streams are
    # touched by jitter (receive-time only) or fixed_latency (cam_wrist).
    assert set(dropped) == {"cam_front"}
    ground_truth_path = FIXTURE_DIR / CORRUPTION_GROUND_TRUTH_NAME
    assert ground_truth_path.is_file()
