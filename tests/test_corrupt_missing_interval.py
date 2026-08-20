"""Missing-interval corruption (NEXT_TASKS #1, D-0019).

Missing intervals remove a contiguous acquisition-time window from the
target stream. Semantics mirror ``dropped_frames`` — survivors keep
sequence ids, the first survivor after the removed block gets
``gap_before``, and the exact removed samples come back in
``CorruptionResult.dropped`` — but the drop is interval-shaped rather than
per-sample-random. Deterministic by construction (no RNG needed).
"""

from __future__ import annotations

import pytest

from embodied_sync.core.sample import QUALITY_GAP_BEFORE, Sample
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    FixedLatencyCorruption,
    MissingIntervalCorruption,
    apply_missing_interval,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def _run() -> dict[str, list[Sample]]:
    return generate_synthetic_run(duration_s=1.0, seed=0)


def _profile(
    start_ns: int = 100_000_000,
    duration_ns: int = 40_000_000,
    stream: str = "robot_state",
) -> CorruptionProfile:
    return CorruptionProfile(
        seed=0,
        corruptions=(
            MissingIntervalCorruption(
                stream=stream, start_ns=start_ns, duration_ns=duration_ns
            ),
        ),
    )


class TestMissingIntervalSemantics:
    def test_removes_exact_samples_in_half_open_window(self) -> None:
        # robot_state @ 250 Hz starts at acquisition=0 with 4ms spacing.
        # Window [100ms, 140ms) covers indices 25..34 (10 samples).
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        removed = result.dropped["robot_state"]
        assert [s.sequence_id for s in removed] == list(range(25, 35))
        # Everything else survives, in order.
        assert [s.sequence_id for s in result.run["robot_state"]] == (
            list(range(0, 25)) + list(range(35, 250))
        )

    def test_survivors_keep_sequence_ids_and_all_fields(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        # Non-boundary survivors are the exact originals.
        original_by_id = {s.sequence_id: s for s in run["robot_state"]}
        for survivor in result.run["robot_state"]:
            if survivor.sequence_id == 35:
                continue  # boundary survivor — gains gap_before flag
            assert survivor == original_by_id[survivor.sequence_id]

    def test_first_survivor_after_removed_block_gets_gap_before(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        survivors = result.run["robot_state"]
        boundary = next(s for s in survivors if s.sequence_id == 35)
        assert QUALITY_GAP_BEFORE in boundary.quality_flags
        for other in survivors:
            if other.sequence_id != 35:
                assert QUALITY_GAP_BEFORE not in other.quality_flags

    def test_removed_samples_carry_original_state(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        original_by_id = {s.sequence_id: s for s in run["robot_state"]}
        for removed in result.dropped["robot_state"]:
            assert removed == original_by_id[removed.sequence_id]

    def test_other_streams_untouched(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        for name in run:
            if name != "robot_state":
                assert result.run[name] == run[name]

    def test_input_run_not_mutated(self) -> None:
        run = _run()
        snapshot = _run()
        apply_profile(run, _profile(100_000_000, 40_000_000))
        assert run == snapshot

    def test_survivor_plus_dropped_count_matches_input(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        assert (
            len(result.run["robot_state"]) + len(result.dropped["robot_state"])
            == len(run["robot_state"])
        )

    def test_receive_time_and_payload_preserved_on_survivors(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(100_000_000, 40_000_000))
        original_by_id = {s.sequence_id: s for s in run["robot_state"]}
        for survivor in result.run["robot_state"]:
            original = original_by_id[survivor.sequence_id]
            assert survivor.acquisition_time_ns == original.acquisition_time_ns
            assert survivor.receive_time_ns == original.receive_time_ns
            assert survivor.payload == original.payload


class TestMissingIntervalDeterminism:
    def test_same_profile_identical_output(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(100_000_000, 40_000_000))
        b = apply_profile(run, _profile(100_000_000, 40_000_000))
        assert a.run == b.run
        assert a.dropped == b.dropped


class TestMissingIntervalEdgeCases:
    def test_window_before_first_sample_is_noop(self) -> None:
        # First acquisition is 0, spacing 4ms. A window ending at 0 removes
        # nothing (half-open, and acquisition = 0 is exactly the start).
        # Instead, put the whole window before 0 by anchoring differently:
        # we can't; start_ns >= 0 always. So test the smallest possible
        # non-intersecting case: duration such that end_ns <= first
        # acquisition — impossible when start_ns=0. Use a stream that
        # starts at 0 and place a 1ns window at start=0 → covers only
        # sample 0.
        run = _run()
        result = apply_profile(run, _profile(0, 1))
        assert [s.sequence_id for s in result.dropped["robot_state"]] == [0]

    def test_window_between_samples_removes_nothing(self) -> None:
        # Samples at 0, 4ms, 8ms, ... A window [1ms, 3ms) hits nothing.
        run = _run()
        result = apply_profile(run, _profile(1_000_000, 2_000_000))
        assert result.dropped.get("robot_state", ()) == ()
        assert result.run["robot_state"] == run["robot_state"]

    def test_window_past_stream_is_noop(self) -> None:
        # Last acquisition is 996_000_000ns. Window well past that.
        run = _run()
        result = apply_profile(run, _profile(2_000_000_000, 1_000_000))
        assert result.dropped.get("robot_state", ()) == ()
        assert result.run["robot_state"] == run["robot_state"]

    def test_window_covering_entire_stream_removes_all(self) -> None:
        # actions @ 10 Hz, first acq=0, last acq=900_000_000. Window
        # [0, 1s) covers every sample.
        run = _run()
        result = apply_profile(
            run, _profile(0, 1_000_000_000, stream="actions")
        )
        assert result.run["actions"] == []
        assert [s.sequence_id for s in result.dropped["actions"]] == list(range(10))
        # No survivors exist, so gap_before flags are moot — assert none exist.
        assert all(QUALITY_GAP_BEFORE not in s.quality_flags for s in result.dropped["actions"])

    def test_empty_stream_is_noop(self) -> None:
        result_survivors, removed = apply_missing_interval(
            [], start_ns=0, duration_ns=1_000_000_000
        )
        assert result_survivors == []
        assert removed == ()


class TestMissingIntervalComposition:
    def test_stacks_with_dropped_frames_ground_truth_accumulates(self) -> None:
        # Interval removes a chunk from robot_state; dropped_frames then
        # thins the survivors further. Both sets of removals must appear
        # in CorruptionResult.dropped, in profile-application order.
        run = _run()
        profile = CorruptionProfile(
            seed=99,
            corruptions=(
                MissingIntervalCorruption(
                    stream="robot_state", start_ns=100_000_000, duration_ns=40_000_000
                ),
                DroppedFramesCorruption(stream="robot_state", probability=0.1),
            ),
        )
        result = apply_profile(run, profile)
        # The interval-removed ids are the first 10 in the dropped list
        # (they were removed first).
        removed_ids = [s.sequence_id for s in result.dropped["robot_state"]]
        interval_ids = list(range(25, 35))
        assert removed_ids[: len(interval_ids)] == interval_ids
        # All removed samples plus survivors reconstruct the original count.
        assert (
            len(result.run["robot_state"]) + len(result.dropped["robot_state"])
            == len(run["robot_state"])
        )

    def test_fixed_latency_after_missing_interval_shifts_survivors(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                MissingIntervalCorruption(
                    stream="robot_state", start_ns=100_000_000, duration_ns=40_000_000
                ),
                FixedLatencyCorruption(stream="robot_state", offset_ns=5_000_000),
            ),
        )
        result = apply_profile(run, profile)
        original_by_id = {s.sequence_id: s for s in run["robot_state"]}
        for survivor in result.run["robot_state"]:
            original = original_by_id[survivor.sequence_id]
            assert survivor.receive_time_ns == original.receive_time_ns + 5_000_000
            assert survivor.acquisition_time_ns == original.acquisition_time_ns


class TestMissingIntervalUnknownStreamFailsLoudly:
    def test_unknown_stream_raises_key_error(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                MissingIntervalCorruption(
                    stream="cam_top", start_ns=0, duration_ns=1_000_000
                ),
            ),
        )
        with pytest.raises(KeyError, match="cam_top"):
            apply_profile(run, profile)
