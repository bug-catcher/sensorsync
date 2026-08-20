"""Milestone 1 TDD red tests: deterministic synthetic streams.

EXPECTED TO FAIL (NotImplementedError) until the generator is implemented.
See DECISIONS.md D-0004 and SESSION_STATE.md. Do not xfail these — the red
state is the work queue.
"""

from __future__ import annotations

import pytest

from embodied_sync.core.sample import QUALITY_SYNTHETIC, Modality
from embodied_sync.streams.synthetic import (
    DEFAULT_SPECS,
    NS_PER_S,
    generate_stream,
    generate_synthetic_run,
)

DURATION_S = 2.0
SEED = 1234
START_NS = 1_000_000_000_000


def _spec(name: str):
    return next(s for s in DEFAULT_SPECS if s.name == name)


class TestDefaultRig:
    def test_default_specs_match_milestone_1(self) -> None:
        rig = {s.name: (s.modality, s.rate_hz) for s in DEFAULT_SPECS}
        assert rig == {
            "cam_front": (Modality.CAMERA, 30.0),
            "cam_wrist": (Modality.CAMERA, 30.0),
            "robot_state": (Modality.ROBOT_STATE, 250.0),
            "tactile": (Modality.TACTILE, 60.0),
            "audio": (Modality.AUDIO, 50.0),
            "actions": (Modality.ACTION, 10.0),
            "events": (Modality.EVENT, None),
        }


class TestDeterminism:
    def test_same_seed_identical_runs(self) -> None:
        a = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        b = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        assert a == b

    def test_different_seed_changes_random_parts(self) -> None:
        a = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        b = generate_synthetic_run(duration_s=DURATION_S, seed=SEED + 1, start_time_ns=START_NS)
        # Event times are seed-dependent; regular timestamps are not.
        assert [s.acquisition_time_ns for s in a["events"]] != [
            s.acquisition_time_ns for s in b["events"]
        ]

    def test_seed_does_not_affect_regular_timestamps(self) -> None:
        a = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        b = generate_synthetic_run(duration_s=DURATION_S, seed=SEED + 1, start_time_ns=START_NS)
        for name in ("cam_front", "robot_state", "tactile", "audio", "actions"):
            assert [s.acquisition_time_ns for s in a[name]] == [
                s.acquisition_time_ns for s in b[name]
            ], f"timing of regular stream {name} must be seed-independent (D-0006)"


class TestCleanStreamTiming:
    @pytest.mark.parametrize(
        ("name", "expected_count"),
        [
            ("cam_front", 60),  # 30 Hz * 2 s
            ("cam_wrist", 60),
            ("robot_state", 500),  # 250 Hz * 2 s
            ("tactile", 120),  # 60 Hz * 2 s
            ("audio", 100),  # 50 Hz * 2 s
            ("actions", 20),  # 10 Hz * 2 s
        ],
    )
    def test_sample_counts(self, name: str, expected_count: int) -> None:
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        assert len(run[name]) == expected_count

    def test_regular_acquisition_grid(self) -> None:
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        for name in ("cam_front", "robot_state", "actions"):
            rate = _spec(name).rate_hz
            assert rate is not None
            for i, s in enumerate(run[name]):
                assert s.acquisition_time_ns == START_NS + round(i * NS_PER_S / rate)

    def test_constant_transport_latency(self) -> None:
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        for spec in DEFAULT_SPECS:
            for s in run[spec.name]:
                assert s.transport_latency_ns == spec.transport_latency_ns, (
                    f"clean {spec.name} must have fixed transport latency (D-0006)"
                )

    def test_sequence_ids_contiguous_from_zero(self) -> None:
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        for name, samples in run.items():
            assert [s.sequence_id for s in samples] == list(range(len(samples))), name

    def test_metadata_and_flags(self) -> None:
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        for spec in DEFAULT_SPECS:
            for s in run[spec.name]:
                assert s.stream_name == spec.name
                assert s.modality is spec.modality
                assert s.source_clock_domain == spec.clock_domain
                assert QUALITY_SYNTHETIC in s.quality_flags

    def test_events_are_irregular_in_window_and_monotonic(self) -> None:
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        events = run["events"]
        assert len(events) >= 2, "expected at least two event markers in 2 s"
        times = [s.acquisition_time_ns for s in events]
        assert times == sorted(times)
        assert all(START_NS <= t < START_NS + int(DURATION_S * NS_PER_S) for t in times)
        deltas = {b - a for a, b in zip(times, times[1:])}
        assert len(deltas) > 1, "event markers must not be on a regular grid"


class TestSingleStreamApi:
    def test_generate_stream_matches_run_output(self) -> None:
        spec = _spec("robot_state")
        run = generate_synthetic_run(duration_s=DURATION_S, seed=SEED, start_time_ns=START_NS)
        # Contract: a stream is fully determined by (spec, duration, child seed,
        # start). The run-level function documents child-seed derivation; here
        # we only require the standalone API to be self-consistent.
        a = generate_stream(spec, duration_s=DURATION_S, child_seed=7, start_time_ns=START_NS)
        b = generate_stream(spec, duration_s=DURATION_S, child_seed=7, start_time_ns=START_NS)
        assert a == b
        assert len(a) == len(run["robot_state"])
