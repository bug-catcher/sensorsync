"""Non-monotonic timestamps corruption (NEXT_TASKS #1, D-0018).

Non-monotonic swaps ``count`` adjacent receive-time pairs deterministically,
producing out-of-order delivery. Acquisition timestamps, sequence ids, list
order, and payloads are preserved. After the swap pass, every observed
downward step in ``receive_time_ns`` is flagged ``non_monotonic`` on the
second sample of the step — the observation any recorder tracking receive
monotonicity would make. Non-overlapping positions give exactly ``count``
flagged samples; overlapping positions cascade to fewer.
"""

from __future__ import annotations

from numpy.random import PCG64, Generator

import pytest

from embodied_sync.core.sample import QUALITY_NON_MONOTONIC, Sample
from embodied_sync.corrupt import (
    CorruptionProfile,
    FixedLatencyCorruption,
    NonMonotonicCorruption,
    apply_non_monotonic,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

SEED = 42


def _run() -> dict[str, list[Sample]]:
    return generate_synthetic_run(duration_s=1.0, seed=0)


def _profile(seed: int, count: int = 3, stream: str = "robot_state") -> CorruptionProfile:
    return CorruptionProfile(
        seed=seed,
        corruptions=(NonMonotonicCorruption(stream=stream, count=count),),
    )


class TestNonMonotonicDeterminism:
    def test_same_profile_seed_identical_output(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(SEED))
        b = apply_profile(run, _profile(SEED))
        assert a.run == b.run
        assert a.dropped == b.dropped

    def test_different_profile_seed_changes_swap_positions(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(SEED))
        b = apply_profile(run, _profile(SEED + 1))
        assert [s.receive_time_ns for s in a.run["robot_state"]] != [
            s.receive_time_ns for s in b.run["robot_state"]
        ]

    def test_input_run_not_mutated(self) -> None:
        run = _run()
        snapshot = _run()
        apply_profile(run, _profile(SEED))
        assert run == snapshot


class TestNonMonotonicSemantics:
    def test_acquisition_sequence_and_payload_preserved(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        assert result.dropped == {}
        for original, swapped in zip(run["robot_state"], result.run["robot_state"]):
            assert swapped.acquisition_time_ns == original.acquisition_time_ns
            assert swapped.sequence_id == original.sequence_id
            assert swapped.payload == original.payload
            assert swapped.stream_name == original.stream_name
            assert swapped.modality == original.modality

    def test_other_streams_untouched(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        for name in run:
            if name != "robot_state":
                assert result.run[name] == run[name]

    def test_receive_time_multiset_unchanged(self) -> None:
        # Swaps permute receive times; the multiset must match exactly.
        run = _run()
        result = apply_profile(run, _profile(SEED))
        assert sorted(s.receive_time_ns for s in result.run["robot_state"]) == sorted(
            s.receive_time_ns for s in run["robot_state"]
        )

    def test_non_adjacent_swaps_produce_exact_pair_swaps_and_flag_second(self) -> None:
        # With SEED=42 count=3 on the 250-sample robot_state stream, positions
        # drawn are [49, 225, 235] (verified non-adjacent). Each pair swap is
        # visible as a receive-time swap of the (i, i+1) pair; the flag lands
        # on i+1 because it now carries the earlier time and looks late.
        run = _run()
        original = run["robot_state"]
        result = apply_profile(run, _profile(SEED))
        swapped = result.run["robot_state"]

        expected_positions = {49, 225, 235}
        flipped_indices = {
            i for i, (o, s) in enumerate(zip(original, swapped))
            if o.receive_time_ns != s.receive_time_ns
        }
        assert flipped_indices == expected_positions | {p + 1 for p in expected_positions}

        for pos in expected_positions:
            # Adjacent-pair swap: samples at pos and pos+1 exchanged receives.
            assert swapped[pos].receive_time_ns == original[pos + 1].receive_time_ns
            assert swapped[pos + 1].receive_time_ns == original[pos].receive_time_ns
            # The trailing sample of the pair carries the flag; the leading
            # one is not flagged by this swap alone.
            assert QUALITY_NON_MONOTONIC in swapped[pos + 1].quality_flags
            assert QUALITY_NON_MONOTONIC not in swapped[pos].quality_flags

    def test_flags_mark_exactly_the_downward_steps(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        swapped = result.run["robot_state"]
        for i, sample in enumerate(swapped):
            has_flag = QUALITY_NON_MONOTONIC in sample.quality_flags
            expected = i > 0 and sample.receive_time_ns < swapped[i - 1].receive_time_ns
            assert has_flag is expected, (
                f"sample {i} at receive_time_ns {sample.receive_time_ns} after "
                f"{swapped[i - 1].receive_time_ns if i > 0 else 'N/A'}: "
                f"non_monotonic flag {'present' if has_flag else 'absent'}"
            )

    def test_at_least_one_downward_step_exists(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))
        swapped = result.run["robot_state"]
        descents = sum(
            1
            for i in range(1, len(swapped))
            if swapped[i].receive_time_ns < swapped[i - 1].receive_time_ns
        )
        assert descents == 3, "SEED=42 count=3 on robot_state has 3 non-adjacent swaps"


class TestNonMonotonicCascade:
    def test_overlapping_swaps_cascade_and_flag_only_observable_steps(self) -> None:
        # Force overlap: count=5 on the 10-sample actions stream must draw
        # adjacent positions (pigeonhole). Verify the invariant holds: the
        # receive-time multiset is preserved and flags mark exactly the
        # downward steps — fewer flags than swaps is expected.
        run = _run()
        result = apply_profile(run, _profile(SEED, count=5, stream="actions"))
        swapped = result.run["actions"]
        assert sorted(s.receive_time_ns for s in swapped) == sorted(
            s.receive_time_ns for s in run["actions"]
        )
        for i, sample in enumerate(swapped):
            has_flag = QUALITY_NON_MONOTONIC in sample.quality_flags
            expected = i > 0 and sample.receive_time_ns < swapped[i - 1].receive_time_ns
            assert has_flag is expected


class TestNonMonotonicEdgeCases:
    def test_zero_count_is_noop(self) -> None:
        run = _run()
        samples = run["robot_state"]
        rng = Generator(PCG64(0))
        swapped = apply_non_monotonic(samples, count=0, rng=rng)
        assert swapped == list(samples)

    def test_single_sample_stream_is_noop(self) -> None:
        run = _run()
        rng = Generator(PCG64(0))
        swapped = apply_non_monotonic(run["actions"][:1], count=3, rng=rng)
        assert swapped == run["actions"][:1]

    def test_count_larger_than_available_pairs_is_noop(self) -> None:
        run = _run()
        # events has 4 samples → 3 adjacent pairs; count=4 exceeds → no-op.
        rng = Generator(PCG64(0))
        swapped = apply_non_monotonic(run["events"], count=4, rng=rng)
        assert swapped == list(run["events"])


class TestNonMonotonicComposition:
    def test_fixed_latency_after_non_monotonic_shifts_swapped_receives(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=SEED,
            corruptions=(
                NonMonotonicCorruption(stream="robot_state", count=3),
                FixedLatencyCorruption(stream="robot_state", offset_ns=5_000_000),
            ),
        )
        result = apply_profile(run, profile)
        for original, changed in zip(run["robot_state"], result.run["robot_state"]):
            assert changed.acquisition_time_ns == original.acquisition_time_ns
            assert changed.sequence_id == original.sequence_id
        # Flag pattern from the swap survives the latency shift (which is a
        # uniform offset and doesn't change the ordering).
        swapped_only = apply_profile(run, _profile(SEED)).run["robot_state"]
        for a, b in zip(swapped_only, result.run["robot_state"]):
            assert (QUALITY_NON_MONOTONIC in a.quality_flags) == (
                QUALITY_NON_MONOTONIC in b.quality_flags
            )
            assert b.receive_time_ns == a.receive_time_ns + 5_000_000


class TestNonMonotonicUnknownStreamFailsLoudly:
    def test_unknown_stream_raises_key_error(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=0,
            corruptions=(NonMonotonicCorruption(stream="cam_top", count=1),),
        )
        with pytest.raises(KeyError, match="cam_top"):
            apply_profile(run, profile)
