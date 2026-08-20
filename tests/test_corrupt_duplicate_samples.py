"""Duplicate-samples corruption (NEXT_TASKS #1, D-0017).

Duplicate-samples independently duplicates samples with a fixed per-sample
probability. Each duplicate is inserted immediately after its original with
the same ``sequence_id``, ``acquisition_time_ns``, ``receive_time_ns``, and
``payload``; sequence-id repetition is observable to any recorder tracking
sequence ids, so the extra copy carries the ``duplicate`` quality flag, and
the original is left untouched.
"""

from __future__ import annotations

import pytest

from embodied_sync.core.sample import QUALITY_DUPLICATE, Sample
from embodied_sync.corrupt import (
    CorruptionProfile,
    DuplicateSamplesCorruption,
    FixedLatencyCorruption,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

SEED = 42


def _run() -> dict[str, list[Sample]]:
    return generate_synthetic_run(duration_s=1.0, seed=0)


def _profile(seed: int, probability: float = 0.1) -> CorruptionProfile:
    return CorruptionProfile(
        seed=seed,
        corruptions=(
            DuplicateSamplesCorruption(stream="robot_state", probability=probability),
        ),
    )


class TestDuplicateSamplesDeterminism:
    def test_same_profile_seed_identical_output(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(SEED))
        b = apply_profile(run, _profile(SEED))
        assert a.run == b.run
        assert a.dropped == b.dropped

    def test_different_profile_seed_selects_different_samples(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(SEED))
        b = apply_profile(run, _profile(SEED + 1))
        # Set of duplicated sequence ids must differ between seeds.
        a_dups = {s.sequence_id for s in a.run["robot_state"] if QUALITY_DUPLICATE in s.quality_flags}
        b_dups = {s.sequence_id for s in b.run["robot_state"] if QUALITY_DUPLICATE in s.quality_flags}
        assert a_dups != b_dups

    def test_input_run_not_mutated(self) -> None:
        run = _run()
        snapshot = _run()
        apply_profile(run, _profile(SEED))
        assert run == snapshot


class TestDuplicateSamplesSemantics:
    def test_dropped_ground_truth_stays_empty(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        assert result.dropped == {}

    def test_extra_copy_immediately_follows_original_and_matches_it(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        samples = result.run["robot_state"]

        dup_pairs = 0
        for i, sample in enumerate(samples):
            if QUALITY_DUPLICATE not in sample.quality_flags:
                continue
            assert i > 0, "a duplicate cannot be the first sample in the stream"
            original = samples[i - 1]
            assert QUALITY_DUPLICATE not in original.quality_flags
            assert original.sequence_id == sample.sequence_id
            assert original.acquisition_time_ns == sample.acquisition_time_ns
            assert original.receive_time_ns == sample.receive_time_ns
            assert original.payload == sample.payload
            assert original.stream_name == sample.stream_name
            assert original.modality == sample.modality
            assert sample.quality_flags == original.quality_flags | {QUALITY_DUPLICATE}
            dup_pairs += 1
        assert dup_pairs > 0, "p=0.1 over 250 samples must produce at least one duplicate"

    def test_originals_preserved_in_order_and_untouched(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        originals = [s for s in result.run["robot_state"] if QUALITY_DUPLICATE not in s.quality_flags]
        assert originals == run["robot_state"]

    def test_output_length_matches_original_plus_duplicates(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        samples = result.run["robot_state"]
        duplicates = [s for s in samples if QUALITY_DUPLICATE in s.quality_flags]
        assert len(samples) == len(run["robot_state"]) + len(duplicates)

    def test_other_streams_untouched(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        for name in run:
            if name != "robot_state":
                assert result.run[name] == run[name]


class TestDuplicateSamplesProbabilityExtremes:
    def test_probability_zero_is_noop(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED, probability=0.0))
        assert result.run["robot_state"] == run["robot_state"]
        assert result.dropped == {}

    def test_probability_one_duplicates_every_sample(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED, probability=1.0))
        samples = result.run["robot_state"]
        assert len(samples) == 2 * len(run["robot_state"])
        for i, original in enumerate(run["robot_state"]):
            assert samples[2 * i] == original
            duplicate = samples[2 * i + 1]
            assert duplicate.sequence_id == original.sequence_id
            assert duplicate.acquisition_time_ns == original.acquisition_time_ns
            assert duplicate.receive_time_ns == original.receive_time_ns
            assert QUALITY_DUPLICATE in duplicate.quality_flags


class TestDuplicateSamplesComposition:
    def test_duplicate_before_fixed_latency_shifts_originals_and_duplicates_together(self) -> None:
        # Both duplicate and original carry the same receive time before the
        # latency shift, so both must carry the same shifted time after it.
        run = _run()
        profile = CorruptionProfile(
            seed=SEED,
            corruptions=(
                DuplicateSamplesCorruption(stream="robot_state", probability=0.1),
                FixedLatencyCorruption(stream="robot_state", offset_ns=5_000_000),
            ),
        )
        result = apply_profile(run, profile)
        for i, sample in enumerate(result.run["robot_state"]):
            if QUALITY_DUPLICATE not in sample.quality_flags:
                continue
            original = result.run["robot_state"][i - 1]
            assert original.receive_time_ns == sample.receive_time_ns


class TestDuplicateSamplesUnknownStreamFailsLoudly:
    def test_unknown_stream_raises_key_error(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                DuplicateSamplesCorruption(stream="cam_top", probability=0.1),
            ),
        )
        with pytest.raises(KeyError, match="cam_top"):
            apply_profile(run, profile)
