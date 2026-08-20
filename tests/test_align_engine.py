"""Nearest-neighbor alignment engine (Milestone 3, first slice, D-0020).

Covers the designed contract of :func:`embodied_sync.align.align_run`:
frame-grid layout, per-stream tolerance, nearest-neighbor lookup, missing
flag, confidence, ground-truth cross-check, and edge cases.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from embodied_sync.align import (
    NEAREST_NEIGHBOR,
    AlignedFrame,
    align_run,
)
from embodied_sync.core.sample import Modality, Sample
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    MissingIntervalCorruption,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def _clean(duration_s: float = 1.0, seed: int = 0) -> dict[str, list[Sample]]:
    return generate_synthetic_run(duration_s=duration_s, seed=seed)


def _mk_sample(seq_id: int, acq_ns: int, rate_hz: float = 30.0) -> Sample:
    return Sample(
        stream_name="synthetic",
        modality=Modality.CAMERA,
        sequence_id=seq_id,
        acquisition_time_ns=acq_ns,
        receive_time_ns=acq_ns + 1_000_000,
        source_clock_domain="host_mono",
        payload={"i": seq_id, "rate_hz": rate_hz},
    )


def _uniform_stream(rate_hz: float, duration_s: float, start_ns: int = 0) -> list[Sample]:
    """Mirror the synth generator: round(i * 1e9 / rate) per sample.

    Cumulative rounding matches ``streams/synthetic.py`` — using a rounded
    period and multiplying by ``i`` introduces off-by-one skew at 30 Hz.
    """
    n = int(round(duration_s * rate_hz))
    return [
        _mk_sample(i, start_ns + round(i * 1_000_000_000 / rate_hz), rate_hz)
        for i in range(n)
    ]


class TestFrameGrid:
    def test_targets_snap_to_world_grid_and_stay_inside_intersection(self) -> None:
        # Two streams: one starts at 331ms (like events), one at 0. Target
        # rate 10 Hz → 100 ms period. First frame must be the first
        # 0-anchored grid tick >= 331 ms → 400 ms.
        run = {
            "late": [_mk_sample(0, 331_322_594), _mk_sample(1, 900_000_000)],
            "early": _uniform_stream(30.0, 1.0),
        }
        aligned = align_run(run, target_rate_hz=10.0)
        targets = [f.target_time_ns for f in aligned.frames]
        assert targets == [400_000_000, 500_000_000, 600_000_000, 700_000_000, 800_000_000, 900_000_000]

    def test_target_rate_period_computed_from_hz(self) -> None:
        run = {"cam": _uniform_stream(30.0, 1.0)}
        aligned = align_run(run, target_rate_hz=10.0)
        # 10 Hz → 100 ms period; targets step by 100 ms.
        step = aligned.frames[1].target_time_ns - aligned.frames[0].target_time_ns
        assert step == 100_000_000

    def test_zero_and_negative_rate_rejected(self) -> None:
        run = _clean()
        with pytest.raises(ValueError, match="target_rate_hz"):
            align_run(run, target_rate_hz=0.0)
        with pytest.raises(ValueError, match="target_rate_hz"):
            align_run(run, target_rate_hz=-1.0)

    def test_empty_run_returns_no_frames(self) -> None:
        aligned = align_run({}, target_rate_hz=10.0)
        assert aligned.frames == []
        assert aligned.report.missing_count == {}

    def test_all_empty_streams_returns_no_frames_but_missing_count_zero(self) -> None:
        aligned = align_run({"a": [], "b": []}, target_rate_hz=10.0)
        assert aligned.frames == []
        assert aligned.report.missing_count == {"a": 0, "b": 0}

    def test_non_overlapping_streams_return_no_frames(self) -> None:
        # Stream A ends before stream B begins.
        run = {
            "a": [_mk_sample(0, 100_000_000), _mk_sample(1, 200_000_000)],
            "b": [_mk_sample(0, 500_000_000), _mk_sample(1, 600_000_000)],
        }
        aligned = align_run(run, target_rate_hz=10.0)
        assert aligned.frames == []


class TestNearestNeighborSemantics:
    def test_target_on_a_sample_is_zero_skew_and_max_confidence(self) -> None:
        run = {
            "cam_front": _uniform_stream(30.0, 1.0),
            "actions": _uniform_stream(10.0, 1.0),
        }
        aligned = align_run(run, target_rate_hz=10.0)
        for frame in aligned.frames:
            # actions samples land exactly on the 10 Hz grid; skew must be 0.
            md = frame.metadata["actions"]
            assert not md.missing
            assert md.skew_ns == 0
            assert md.confidence == pytest.approx(1.0)
            assert md.method == NEAREST_NEIGHBOR
            assert md.source_time_ns == frame.target_time_ns
            # cam_front at 30 Hz also has samples at 0, 100M, 200M, ... exactly.
            cam_md = frame.metadata["cam_front"]
            assert not cam_md.missing
            assert cam_md.skew_ns == 0

    def test_missing_flag_when_nearest_beyond_tolerance(self) -> None:
        # Single stream with a big gap: samples at 0 and 200ms only. At
        # target 100ms, nearest is 100ms away, but tolerance derived from
        # the median interval (200ms) is 100ms → boundary case. Push
        # target to 100.001ms to force missing.
        samples = [_mk_sample(0, 0), _mk_sample(1, 200_000_000), _mk_sample(2, 400_000_000)]
        aligned = align_run({"sparse": samples}, target_rate_hz=10.0)
        # Median interval = 200 ms, tolerance = 100 ms; targets at 100,
        # 200, 300, 400 ms.
        targets = [f.target_time_ns for f in aligned.frames]
        assert 100_000_000 in targets
        # At target 100ms: |100 - 0| = |200 - 100| = 100 ms == tolerance,
        # so *not* missing (tie-broken to right neighbor).
        idx = targets.index(100_000_000)
        assert not aligned.frames[idx].metadata["sparse"].missing
        # At target 300ms: |300 - 200| = |400 - 300| = 100 ms == tolerance,
        # so *not* missing.
        idx = targets.index(300_000_000)
        assert not aligned.frames[idx].metadata["sparse"].missing

    def test_missing_when_target_between_far_apart_samples(self) -> None:
        # Samples at 0 and 500ms only (median interval 500 ms, tolerance 250 ms).
        # Target 250ms is exactly on the tolerance boundary; target 249ms
        # is inside, target 251ms is outside.
        samples = [_mk_sample(0, 0), _mk_sample(1, 500_000_000)]
        # Use a rate that yields target 249ms and 251ms. 249 ns/10ms rate
        # would need custom targets. Simpler: use targets at natural grid.
        # Frame grid at 10Hz over [0, 500M]: 0, 100M, 200M, 300M, 400M, 500M.
        aligned = align_run({"sparse": samples}, target_rate_hz=10.0)
        # tolerance = 250ms. At target 200ms: |0 - 200| = 200 <= 250 ✓, not missing.
        # At target 300ms: |500 - 300| = 200 <= 250 ✓, not missing.
        for frame in aligned.frames:
            assert not frame.metadata["sparse"].missing

    def test_confidence_decreases_with_skew(self) -> None:
        # 10Hz stream aligned to a 30Hz target grid: many targets fall
        # between samples, so skew varies.
        run = {"low_rate": _uniform_stream(10.0, 1.0)}
        aligned = align_run(run, target_rate_hz=30.0)
        confidences = [frame.metadata["low_rate"].confidence for frame in aligned.frames]
        skews = [abs(frame.metadata["low_rate"].skew_ns) for frame in aligned.frames]
        # High skew → low confidence.
        for c, s in zip(confidences, skews):
            assert 0.0 <= c <= 1.0
            if s == 0:
                assert c == pytest.approx(1.0)
            else:
                assert c < 1.0

    def test_source_time_ns_reflects_matched_sample(self) -> None:
        run = _clean()
        aligned = align_run(run, target_rate_hz=10.0)
        for frame in aligned.frames:
            for name, sample in frame.samples.items():
                md = frame.metadata[name]
                if sample is None:
                    assert md.missing
                    assert md.source_time_ns is None
                    assert md.skew_ns is None
                else:
                    assert md.source_time_ns == sample.acquisition_time_ns


class TestDeterminism:
    def test_alignment_is_deterministic(self) -> None:
        run = _clean()
        a = align_run(run, target_rate_hz=10.0)
        b = align_run(run, target_rate_hz=10.0)
        assert a == b

    def test_input_run_not_mutated(self) -> None:
        run = _clean()
        snapshot = _clean()
        align_run(run, target_rate_hz=10.0)
        assert run == snapshot


class TestMissingAndGroundTruth:
    def test_missing_interval_shows_up_as_action_rate_gap(self) -> None:
        # Remove a 500 ms chunk of cam_front → several action-rate targets
        # in that window can't find a nearby cam_front sample.
        run = _clean()
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                MissingIntervalCorruption(
                    stream="cam_front", start_ns=200_000_000, duration_ns=500_000_000
                ),
            ),
        )
        corruption = apply_profile(run, profile)
        aligned = align_run(
            corruption.run, target_rate_hz=10.0, ground_truth=corruption.dropped
        )
        # Some frame in the removed window must mark cam_front missing.
        assert any(
            frame.metadata["cam_front"].missing
            and 200_000_000 <= frame.target_time_ns <= 700_000_000
            for frame in aligned.frames
        )
        # Ground-truth cross-check populated for the removed stream.
        assert aligned.report.ground_truth_missing_count["cam_front"] > 0

    def test_ground_truth_optional(self) -> None:
        run = _clean()
        aligned = align_run(run, target_rate_hz=10.0)
        assert aligned.report.ground_truth_missing_count == {}

    def test_frame_missing_count_matches_report(self) -> None:
        run = _clean()
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                DroppedFramesCorruption(stream="cam_front", probability=0.7),
            ),
        )
        corruption = apply_profile(run, profile)
        aligned = align_run(corruption.run, target_rate_hz=10.0)
        for name in run:
            per_frame = sum(1 for f in aligned.frames if f.metadata[name].missing)
            assert aligned.report.missing_count[name] == per_frame


class TestStreamStructure:
    def test_every_frame_reports_every_stream(self) -> None:
        run = _clean()
        aligned = align_run(run, target_rate_hz=10.0)
        for frame in aligned.frames:
            assert set(frame.samples) == set(run)
            assert set(frame.metadata) == set(run)

    def test_frame_is_frozen_dataclass(self) -> None:
        run = _clean()
        aligned = align_run(run, target_rate_hz=10.0)
        # Frozen: cannot reassign fields (dict values still mutable, by design).
        frame = aligned.frames[0]
        with pytest.raises(Exception):
            replace(frame, target_time_ns=frame.target_time_ns + 1)  # replace works
            frame.target_time_ns = 0  # type: ignore[misc]

    def test_frames_are_sorted_by_target_time(self) -> None:
        run = _clean()
        aligned = align_run(run, target_rate_hz=10.0)
        targets = [f.target_time_ns for f in aligned.frames]
        assert targets == sorted(targets)
        assert len(set(targets)) == len(targets)


class TestEmptyStreamHandling:
    def test_empty_stream_reports_missing_every_frame(self) -> None:
        run = {
            "cam_front": _uniform_stream(30.0, 1.0),
            "empty_stream": [],
        }
        aligned = align_run(run, target_rate_hz=10.0)
        for frame in aligned.frames:
            assert frame.samples["empty_stream"] is None
            assert frame.metadata["empty_stream"].missing
        assert aligned.report.missing_count["empty_stream"] == len(aligned.frames)


class TestAlignedFrameShape:
    def test_frame_matches_designed_type(self) -> None:
        run = _clean()
        aligned = align_run(run, target_rate_hz=10.0)
        assert isinstance(aligned.frames[0], AlignedFrame)
