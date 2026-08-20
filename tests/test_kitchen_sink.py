"""Kitchen-sink integration test: all 8 corruption kinds → full pipeline.

Runs `embsync corrupt`, `embsync align --check-ground-truth`, and
`embsync report` via the CLI against ``configs/corrupt_kitchen_sink.yaml``,
then verifies each corruption kind's expected observable landed in the
corrupted run, the ground-truth sidecar, or the aligned episode.

Complements the per-corruption unit tests by catching cross-corruption
interaction bugs — e.g. a fixed_latency + burst_stall composition on the
same stream, or a missing_interval that competes with dropped_frames for
the ground-truth sidecar shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from embodied_sync.cli.main import main
from embodied_sync.core.sample import (
    QUALITY_DUPLICATE,
    QUALITY_GAP_BEFORE,
    QUALITY_NON_MONOTONIC,
)
from embodied_sync.datasets.io import load_run
from embodied_sync.streams.synthetic import generate_synthetic_run

PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "corrupt_kitchen_sink.yaml"
)
DURATION_S = 2.0
SEED = 0
TARGET_RATE_HZ = 10.0

# Aggregate the profile's per-corruption expectations here so the tests
# stay easy to audit against `configs/corrupt_kitchen_sink.yaml`.
EXPECTED_STREAMS = {
    "actions",
    "audio",
    "cam_front",
    "cam_wrist",
    "events",
    "robot_state",
    "tactile",
}
# missing_interval: 80 ms window at 250 Hz = exactly 20 robot_state drops.
MISSING_INTERVAL_ROBOT_STATE_COUNT = 20


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("kitchen_sink")
    run_dir = tmp_path / "run"
    corrupted_dir = tmp_path / "corrupted"
    episode_dir = tmp_path / "episode"
    html_path = tmp_path / "report.html"
    summary_path = tmp_path / "summary.json"

    assert (
        main(
            [
                "synth",
                "--out",
                str(run_dir),
                "--seed",
                str(SEED),
                "--duration-s",
                str(DURATION_S),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "corrupt",
                str(run_dir),
                "--profile",
                str(PROFILE_PATH),
                "--out",
                str(corrupted_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "align",
                str(corrupted_dir),
                "--out",
                str(episode_dir),
                "--target-rate-hz",
                str(TARGET_RATE_HZ),
                "--check-ground-truth",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "report",
                str(episode_dir),
                "--out",
                str(html_path),
                "--json-summary",
                str(summary_path),
            ]
        )
        == 0
    )

    return {
        "run_dir": run_dir,
        "corrupted_dir": corrupted_dir,
        "episode_dir": episode_dir,
        "html_path": html_path,
        "summary_path": summary_path,
        "clean": generate_synthetic_run(duration_s=DURATION_S, seed=SEED),
        "corrupted": load_run(corrupted_dir),
        "ground_truth": json.loads(
            (corrupted_dir / "corruption_ground_truth.json").read_text(encoding="utf-8")
        ),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def test_pipeline_produces_all_expected_outputs(pipeline: dict[str, Any]) -> None:
    for key in ("run_dir", "corrupted_dir", "episode_dir"):
        assert Path(pipeline[key]).is_dir()
    assert Path(pipeline["html_path"]).is_file()
    assert Path(pipeline["summary_path"]).is_file()
    text = Path(pipeline["html_path"]).read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")


def test_summary_lists_every_synth_stream(pipeline: dict[str, Any]) -> None:
    names = {s["name"] for s in pipeline["summary"]["streams"]}
    assert names == EXPECTED_STREAMS


def test_fixed_latency_shifts_cam_wrist_receive_time(pipeline: dict[str, Any]) -> None:
    """`fixed_latency` adds a constant offset to every cam_wrist sample."""
    clean = pipeline["clean"]["cam_wrist"]
    corrupted = pipeline["corrupted"]["cam_wrist"]
    # 15 ms offset applied first; burst_stall can only push receive later.
    assert corrupted[0].receive_time_ns >= clean[0].receive_time_ns + 15_000_000


def test_jitter_perturbs_cam_front_receive_time(pipeline: dict[str, Any]) -> None:
    """`jitter` perturbs at least one cam_front receive time."""
    clean_by_seq = {s.sequence_id: s for s in pipeline["clean"]["cam_front"]}
    perturbed = [
        s
        for s in pipeline["corrupted"]["cam_front"]
        if s.sequence_id in clean_by_seq
        and s.receive_time_ns != clean_by_seq[s.sequence_id].receive_time_ns
    ]
    assert perturbed


def test_dropped_frames_records_a_cam_front_drop(pipeline: dict[str, Any]) -> None:
    """`dropped_frames` removes at least one cam_front sample; sidecar records it."""
    dropped = pipeline["ground_truth"]["dropped"].get("cam_front", [])
    assert dropped
    assert len(pipeline["corrupted"]["cam_front"]) < len(pipeline["clean"]["cam_front"])


def test_clock_drift_shifts_a_late_robot_state_sample(pipeline: dict[str, Any]) -> None:
    """`clock_drift` shifts receive time linearly in acquisition time."""
    clean_by_seq = {s.sequence_id: s for s in pipeline["clean"]["robot_state"]}
    corrupted = pipeline["corrupted"]["robot_state"]
    # Pick the latest surviving sample and verify its receive time drifted.
    last = corrupted[-1]
    assert last.sequence_id in clean_by_seq
    assert last.receive_time_ns != clean_by_seq[last.sequence_id].receive_time_ns


def test_burst_stall_pushes_cam_wrist_samples_past_fixed_latency(
    pipeline: dict[str, Any],
) -> None:
    """`burst_stall` pushes at least one sample beyond the base fixed_latency shift."""
    clean_by_seq = {s.sequence_id: s for s in pipeline["clean"]["cam_wrist"]}
    fixed_offset_ns = 15_000_000  # from the profile
    # After fixed_latency alone every sample sits at clean_recv + 15ms; a
    # burst_stall event pushes at least one sample strictly further out,
    # so the maximum extra offset must exceed the fixed offset.
    max_extra = max(
        s.receive_time_ns - clean_by_seq[s.sequence_id].receive_time_ns
        for s in pipeline["corrupted"]["cam_wrist"]
        if s.sequence_id in clean_by_seq
    )
    assert max_extra > fixed_offset_ns


def test_duplicate_samples_flags_extra_tactile_copies(pipeline: dict[str, Any]) -> None:
    """`duplicate_samples` inserts flagged copies (D-0017 semantics)."""
    corrupted = pipeline["corrupted"]["tactile"]
    duplicates = [s for s in corrupted if QUALITY_DUPLICATE in s.quality_flags]
    assert duplicates
    assert len(corrupted) > len(pipeline["clean"]["tactile"])


def test_non_monotonic_flags_appear_on_audio(pipeline: dict[str, Any]) -> None:
    """`non_monotonic` marks the audio samples where downward steps land."""
    corrupted = pipeline["corrupted"]["audio"]
    flagged = [s for s in corrupted if QUALITY_NON_MONOTONIC in s.quality_flags]
    assert flagged


def test_missing_interval_drops_exactly_20_robot_state_samples(
    pipeline: dict[str, Any],
) -> None:
    """`missing_interval` on robot_state (250 Hz, 80 ms window) → 20 exact drops."""
    dropped_streams = pipeline["ground_truth"]["dropped"]
    dropped_robot = dropped_streams.get("robot_state", [])
    assert len(dropped_robot) == MISSING_INTERVAL_ROBOT_STATE_COUNT
    # First post-window survivor carries gap_before (D-0019).
    survivors = pipeline["corrupted"]["robot_state"]
    assert any(QUALITY_GAP_BEFORE in s.quality_flags for s in survivors)


def test_ground_truth_column_wired_into_report(pipeline: dict[str, Any]) -> None:
    """`--check-ground-truth` propagates counts into the HTML report."""
    text = Path(pipeline["html_path"]).read_text(encoding="utf-8")
    assert "Ground truth" in text
    total_gt = sum(
        s["ground_truth_missing_count"] for s in pipeline["summary"]["streams"]
    )
    assert total_gt > 0


# Streams with rate_hz set in DEFAULT_SPECS: every acquisition time is
# an integer multiple of `round(1e9 / rate_hz)`. Because target_rate_hz
# = 10 Hz, target_period_ns = 100_000_000, and every regular stream's
# rate divides evenly into 10 Hz (30, 60, 250, 50, 10 all divide 100 ms
# cleanly at integer-ns precision), NN picks a sample sitting exactly on
# the target grid → median_skew_ns == 0 and missing_rate == 0.0 by
# construction. The kitchen-sink profile perturbs receive_time_ns
# (fixed_latency, jitter, burst_stall) or removes samples
# (dropped_frames, missing_interval) but does not shift acquisition
# times off-grid, so these two invariants must hold across a report/
# align refactor.
REGULAR_STREAMS_ON_10HZ_GRID = {
    "actions",
    "audio",
    "cam_front",
    "cam_wrist",
    "robot_state",
    "tactile",
}


def test_report_regular_streams_have_zero_missing_rate_at_10hz(
    pipeline: dict[str, Any],
) -> None:
    """Every regular stream aligns cleanly at 10 Hz — `missing_rate == 0`."""
    summary = {s["name"]: s for s in pipeline["summary"]["streams"]}
    for name in REGULAR_STREAMS_ON_10HZ_GRID:
        stats = summary[name]
        assert stats["missing_rate"] == 0.0, (
            f"{name}: expected missing_rate == 0.0 (grid-aligned NN pick) "
            f"but got {stats['missing_rate']}. Report accounting regressed "
            f"or align tolerance changed."
        )
        assert stats["missing_count"] == 0


def test_report_regular_streams_have_zero_median_skew(
    pipeline: dict[str, Any],
) -> None:
    """Grid-aligned NN picks land skew_ns == 0 on regular streams.

    Pins the sign — and, for regular streams at 10 Hz, the magnitude —
    of ``median_skew_ns`` so an accounting refactor (skew sign flip,
    off-by-one in the acquisition-time index) fails loudly instead of
    silently producing different report numbers.
    """
    summary = {s["name"]: s for s in pipeline["summary"]["streams"]}
    for name in REGULAR_STREAMS_ON_10HZ_GRID:
        stats = summary[name]
        assert stats["median_skew_ns"] == 0
        assert stats["median_abs_skew_ns"] == 0
        assert stats["median_confidence"] == 1.0


def test_report_events_stream_missing_rate_bounded(pipeline: dict[str, Any]) -> None:
    """Irregular `events` stream is the only row with non-zero missing_rate.

    Upper bound: half the frames. In practice the stream ticks every
    ~300 ms with exponential inter-arrivals (see EVENT_MEAN_INTERVAL_S)
    so about a quarter of 10 Hz targets miss within tolerance. If a
    future align refactor either doubles the tolerance (drops
    `missing_rate` to near zero) or halves it (misses over half), the
    bound flags it.
    """
    events = next(
        s for s in pipeline["summary"]["streams"] if s["name"] == "events"
    )
    assert 0.0 < events["missing_rate"] < 0.5
    assert events["missing_count"] > 0
    # NN median skew for an irregular stream sits within the per-stream
    # tolerance (`|skew| <= tolerance`). Tolerance for events is
    # workload-driven, but "smaller than the 10 Hz period" is a hard
    # bound: NN would never accept a sample farther than one target
    # period.
    assert abs(events["median_skew_ns"]) < 100_000_000  # 100 ms


def test_report_ground_truth_column_targets_cam_front_drops(
    pipeline: dict[str, Any],
) -> None:
    """`cam_front` is the only stream with a non-zero ground-truth column.

    Kitchen-sink drops (`probability: 0.08`) hit `cam_front` alone; the
    ground-truth cross-check must surface at least one and only on
    `cam_front` (missing_interval on `robot_state` removes 20 samples,
    but at 250 Hz vs a 10 Hz target grid those drops fall in the
    "invisible" zone from D-0020 and don't count toward
    `ground_truth_missing_count`).
    """
    summary = {s["name"]: s for s in pipeline["summary"]["streams"]}
    assert summary["cam_front"]["ground_truth_missing_count"] >= 1
    for name, stats in summary.items():
        if name == "cam_front":
            continue
        assert stats["ground_truth_missing_count"] == 0


def test_manifest_records_the_kitchen_sink_profile_path(pipeline: dict[str, Any]) -> None:
    """`embsync corrupt` records the profile source for reproducibility."""
    manifest = json.loads(
        (Path(pipeline["corrupted_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert "corruption" in manifest
    assert manifest["corruption"]["profile_path"].endswith("corrupt_kitchen_sink.yaml")
    assert manifest["corruption"]["profile_seed"] == 12345


def test_episode_manifest_median_skew_ns_zero_for_regular_streams(
    pipeline: dict[str, Any],
) -> None:
    """Persisted-field regression: the aligned-episode manifest's per-stream
    ``median_skew_ns`` echoes zero for every regular stream at 10 Hz.

    ``AlignmentReport.median_skew_ns`` rides on the manifest
    (D-0021 + session 13 lift onto the typed field); an accounting
    regression in ``_median_skew_ns_by_stream`` or ``save_episode`` would
    show up as a non-zero value here even though the summary path (guarded
    by ``test_report_regular_streams_have_zero_median_skew``) still reads
    zero. Guards the two paths against silently drifting apart.
    """
    manifest = json.loads(
        (Path(pipeline["episode_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert "median_skew_ns" in manifest
    per_stream = manifest["median_skew_ns"]
    for name in REGULAR_STREAMS_ON_10HZ_GRID:
        assert per_stream[name] == 0, (
            f"{name}: expected manifest median_skew_ns == 0 (grid-aligned "
            f"NN pick) but got {per_stream[name]}. AlignmentReport.median_"
            f"skew_ns or save_episode drifted from the report path."
        )
    # Events is irregular; its skew sits within the target period.
    assert per_stream["events"] is not None
    assert abs(per_stream["events"]) < 100_000_000
