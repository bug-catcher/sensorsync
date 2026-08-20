"""Burst-stall corruption (NEXT_TASKS #1, D-0015).

Burst stalls simulate contiguous receive-time stalls followed by a clustered
flush: samples whose original ``receive_time_ns`` falls inside a stall
window are all pushed to the burst release time. Acquisition timestamps,
sample order, and quality flags are preserved; overall receive time stays
monotonic non-decreasing.
"""

from __future__ import annotations

from numpy.random import PCG64, Generator

from embodied_sync.core.sample import Sample
from embodied_sync.corrupt import (
    BurstStallCorruption,
    CorruptionProfile,
    FixedLatencyCorruption,
    apply_burst_stall,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

SEED = 42


def _run() -> dict[str, list[Sample]]:
    return generate_synthetic_run(duration_s=1.0, seed=0)


def _profile(seed: int, count: int = 3, stall_ns: int = 80_000_000) -> CorruptionProfile:
    return CorruptionProfile(
        seed=seed,
        corruptions=(
            BurstStallCorruption(stream="robot_state", count=count, stall_ns=stall_ns),
        ),
    )


class TestBurstStallDeterminism:
    def test_same_profile_seed_identical_output(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(SEED))
        b = apply_profile(run, _profile(SEED))
        assert a.run == b.run
        assert a.dropped == b.dropped

    def test_different_profile_seed_places_stalls_differently(self) -> None:
        run = _run()
        a = apply_profile(run, _profile(SEED))
        b = apply_profile(run, _profile(SEED + 1))
        # Some samples must land at different receive times when placement moves.
        assert [s.receive_time_ns for s in a.run["robot_state"]] != [
            s.receive_time_ns for s in b.run["robot_state"]
        ]

    def test_input_run_not_mutated(self) -> None:
        run = _run()
        snapshot = _run()
        apply_profile(run, _profile(SEED))
        assert run == snapshot


class TestBurstStallSemantics:
    def test_acquisition_and_flags_preserved_and_other_streams_untouched(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED))

        assert result.dropped == {}
        for original, stalled in zip(run["robot_state"], result.run["robot_state"]):
            assert stalled.acquisition_time_ns == original.acquisition_time_ns
            assert stalled.sequence_id == original.sequence_id
            assert stalled.payload == original.payload
            assert stalled.quality_flags == original.quality_flags
            # Receive time never moves earlier — bursts only delay delivery.
            assert stalled.receive_time_ns >= original.receive_time_ns

        for name in run:
            if name != "robot_state":
                assert result.run[name] == run[name]

    def test_overall_receive_time_stays_monotonic_non_decreasing(self) -> None:
        run = _run()
        result = apply_profile(run, _profile(SEED, count=5, stall_ns=40_000_000))
        times = [s.receive_time_ns for s in result.run["robot_state"]]
        assert times == sorted(times), "burst stalls must preserve monotonicity"

    def test_clustering_at_least_one_burst_produces_duplicate_receive_times(self) -> None:
        # 250 Hz stream has ~4 ms period. An 80 ms stall pulls ~20 samples into
        # a single release time, so at least one release value must repeat.
        run = _run()
        result = apply_profile(run, _profile(SEED, count=1, stall_ns=80_000_000))
        times = [s.receive_time_ns for s in result.run["robot_state"]]
        assert len(times) != len(set(times)), (
            "an 80 ms stall on a 250 Hz stream must cluster several samples "
            "onto a single release time"
        )

    def test_exact_cluster_layout_for_known_seed(self) -> None:
        # Pin the deterministic placement: with SEED and 1 stall of 80 ms on
        # the 250 Hz robot_state stream, count exactly the samples that end up
        # at the release time, and verify all of them originally sat inside
        # [release - stall, release).
        run = _run()
        original = run["robot_state"]
        result = apply_profile(run, _profile(SEED, count=1, stall_ns=80_000_000))
        stalled = result.run["robot_state"]

        # Identify the release time by finding the most-repeated receive time.
        counts: dict[int, int] = {}
        for s in stalled:
            counts[s.receive_time_ns] = counts.get(s.receive_time_ns, 0) + 1
        release_ns, cluster_size = max(counts.items(), key=lambda kv: kv[1])
        assert cluster_size >= 15, (
            f"80 ms stall on 250 Hz stream should trap ~20 samples; got {cluster_size}"
        )

        stall_ns = 80_000_000
        for original_sample, stalled_sample in zip(original, stalled):
            in_window = release_ns - stall_ns <= original_sample.receive_time_ns < release_ns
            if in_window:
                assert stalled_sample.receive_time_ns == release_ns
            else:
                assert stalled_sample.receive_time_ns == original_sample.receive_time_ns


class TestBurstStallEdgeCases:
    def test_no_op_when_stream_span_narrower_than_stall(self) -> None:
        # A single-sample stream has zero span, so stall_ns > 0 means no-op.
        run = _run()
        first_two = run["actions"][:1]
        rng = Generator(PCG64(0))
        stalled = apply_burst_stall(first_two, count=3, stall_ns=1_000_000, rng=rng)
        assert stalled == list(first_two)

    def test_zero_count_is_noop(self) -> None:
        run = _run()
        samples = run["robot_state"]
        rng = Generator(PCG64(0))
        stalled = apply_burst_stall(samples, count=0, stall_ns=1_000_000, rng=rng)
        assert stalled == list(samples)


class TestBurstStallComposition:
    def test_burst_stall_stacks_after_fixed_latency(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=SEED,
            corruptions=(
                FixedLatencyCorruption(stream="robot_state", offset_ns=5_000_000),
                BurstStallCorruption(stream="robot_state", count=2, stall_ns=40_000_000),
            ),
        )
        result = apply_profile(run, profile)

        for original, changed in zip(run["robot_state"], result.run["robot_state"]):
            # Every sample gained at least the fixed offset.
            assert changed.receive_time_ns >= original.receive_time_ns + 5_000_000
            # Acquisition and payload never change through this pipeline.
            assert changed.acquisition_time_ns == original.acquisition_time_ns
            assert changed.payload == original.payload
