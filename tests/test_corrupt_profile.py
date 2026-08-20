"""Corruption profile schema v0 + fixed_latency application (NEXT_TASKS #3).

Covers: loading/validating the committed example profile, exact ms->ns
conversion, exact injected offsets for fixed_latency, fail-loudly behaviour
for unknown streams, and strict schema validation errors (D-0009). Jitter
and dropped_frames application live in test_corrupt_jitter_drops.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from embodied_sync.corrupt import (
    BurstStallCorruption,
    ClockDriftCorruption,
    CorruptionProfile,
    DroppedFramesCorruption,
    DuplicateSamplesCorruption,
    FixedLatencyCorruption,
    JitterCorruption,
    MissingIntervalCorruption,
    NonMonotonicCorruption,
    ProfileError,
    apply_fixed_latency,
    apply_profile,
    load_profile,
    parse_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = REPO_ROOT / "configs" / "corrupt_camera_jitter.yaml"


def _valid_doc() -> dict[str, Any]:
    return {
        "format_version": 0,
        "seed": 7,
        "corruptions": [
            {"stream": "cam_wrist", "kind": "fixed_latency", "offset_ms": 45.0},
        ],
    }


class TestSchema:
    def test_example_profile_validates_exactly(self) -> None:
        profile = load_profile(EXAMPLE_PROFILE)
        assert profile == CorruptionProfile(
            seed=1234,
            corruptions=(
                JitterCorruption(
                    stream="cam_front",
                    distribution="gaussian",
                    std_ns=8_000_000,
                    clip_ns=30_000_000,
                ),
                DroppedFramesCorruption(stream="cam_front", probability=0.02),
                FixedLatencyCorruption(stream="cam_wrist", offset_ns=45_000_000),
            ),
        )

    def test_ms_to_ns_is_exact_for_fractional_ms(self) -> None:
        doc = _valid_doc()
        doc["corruptions"][0]["offset_ms"] = 0.5
        profile = parse_profile(doc)
        corruption = profile.corruptions[0]
        assert isinstance(corruption, FixedLatencyCorruption)
        assert corruption.offset_ns == 500_000

    def test_drift_ppm_to_ppb_is_exact_for_fractional_ppm(self) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [
            {"stream": "robot_state", "kind": "clock_drift", "drift_ppm": 12.345}
        ]
        profile = parse_profile(doc)
        corruption = profile.corruptions[0]
        assert isinstance(corruption, ClockDriftCorruption)
        assert corruption.drift_ppb == 12_345

    def test_burst_stall_parses_count_and_ms(self) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [
            {"stream": "cam_front", "kind": "burst_stall", "count": 4, "stall_ms": 12.5}
        ]
        profile = parse_profile(doc)
        assert profile.corruptions == (
            BurstStallCorruption(stream="cam_front", count=4, stall_ns=12_500_000),
        )

    def test_duplicate_samples_parses_probability(self) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [
            {"stream": "cam_front", "kind": "duplicate_samples", "probability": 0.01}
        ]
        profile = parse_profile(doc)
        assert profile.corruptions == (
            DuplicateSamplesCorruption(stream="cam_front", probability=0.01),
        )

    def test_non_monotonic_parses_count(self) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [
            {"stream": "cam_front", "kind": "non_monotonic", "count": 2}
        ]
        profile = parse_profile(doc)
        assert profile.corruptions == (
            NonMonotonicCorruption(stream="cam_front", count=2),
        )

    def test_missing_interval_parses_ms_fields(self) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [
            {
                "stream": "robot_state",
                "kind": "missing_interval",
                "start_ms": 100.5,
                "duration_ms": 40.0,
            }
        ]
        profile = parse_profile(doc)
        assert profile.corruptions == (
            MissingIntervalCorruption(
                stream="robot_state", start_ns=100_500_000, duration_ns=40_000_000
            ),
        )

    def test_missing_interval_allows_start_zero(self) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [
            {
                "stream": "cam_front",
                "kind": "missing_interval",
                "start_ms": 0.0,
                "duration_ms": 10.0,
            }
        ]
        profile = parse_profile(doc)
        assert profile.corruptions == (
            MissingIntervalCorruption(stream="cam_front", start_ns=0, duration_ns=10_000_000),
        )

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda d: d.update(format_version=1), "format_version"),
            (lambda d: d.update(extra=1), "unknown top-level"),
            (lambda d: d.update(seed="0"), "'seed'"),
            (lambda d: d.update(seed=True), "'seed'"),
            (lambda d: d.update(corruptions={}), "'corruptions'"),
            (lambda d: d["corruptions"][0].update(kind="warp"), "unknown corruption kind"),
            (lambda d: d["corruptions"][0].pop("offset_ms"), "missing required key"),
            (lambda d: d["corruptions"][0].update(offset_ms="45"), "'offset_ms'"),
            (lambda d: d["corruptions"][0].update(surprise=1), "unknown key"),
            (lambda d: d["corruptions"][0].update(stream=""), "'stream'"),
        ],
    )
    def test_invalid_profiles_rejected(self, mutate: Any, match: str) -> None:
        doc = _valid_doc()
        mutate(doc)
        with pytest.raises(ProfileError, match=match):
            parse_profile(doc)

    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            (
                {"stream": "s", "kind": "jitter", "distribution": "uniform", "std_ms": 1.0},
                "distribution",
            ),
            (
                {"stream": "s", "kind": "jitter", "distribution": "gaussian", "std_ms": 0.0},
                "'std_ms'",
            ),
            ({"stream": "s", "kind": "dropped_frames", "probability": 1.5}, "'probability'"),
            ({"stream": "s", "kind": "clock_drift", "drift_ppm": 0.0}, "'drift_ppm'"),
            ({"stream": "s", "kind": "clock_drift", "drift_ppm": "100"}, "'drift_ppm'"),
            (
                {"stream": "s", "kind": "burst_stall", "count": 0, "stall_ms": 10.0},
                "'count'",
            ),
            (
                {"stream": "s", "kind": "burst_stall", "count": 1.5, "stall_ms": 10.0},
                "'count'",
            ),
            (
                {"stream": "s", "kind": "burst_stall", "count": True, "stall_ms": 10.0},
                "'count'",
            ),
            (
                {"stream": "s", "kind": "burst_stall", "count": 2, "stall_ms": 0.0},
                "'stall_ms'",
            ),
            (
                {"stream": "s", "kind": "duplicate_samples", "probability": 1.5},
                "'probability'",
            ),
            (
                {"stream": "s", "kind": "duplicate_samples", "probability": "0.1"},
                "'probability'",
            ),
            ({"stream": "s", "kind": "non_monotonic", "count": 0}, "'count'"),
            ({"stream": "s", "kind": "non_monotonic", "count": 1.5}, "'count'"),
            ({"stream": "s", "kind": "non_monotonic", "count": True}, "'count'"),
            (
                {
                    "stream": "s",
                    "kind": "missing_interval",
                    "start_ms": -1.0,
                    "duration_ms": 10.0,
                },
                "'start_ms'",
            ),
            (
                {
                    "stream": "s",
                    "kind": "missing_interval",
                    "start_ms": "100",
                    "duration_ms": 10.0,
                },
                "'start_ms'",
            ),
            (
                {
                    "stream": "s",
                    "kind": "missing_interval",
                    "start_ms": 100.0,
                    "duration_ms": 0.0,
                },
                "'duration_ms'",
            ),
            (
                {
                    "stream": "s",
                    "kind": "missing_interval",
                    "start_ms": 100.0,
                    "duration_ms": -5.0,
                },
                "'duration_ms'",
            ),
        ],
    )
    def test_invalid_kind_params_rejected(self, entry: dict[str, Any], match: str) -> None:
        doc = _valid_doc()
        doc["corruptions"] = [entry]
        with pytest.raises(ProfileError, match=match):
            parse_profile(doc)


class TestFixedLatencyApplication:
    OFFSET_NS = 45_000_000

    def test_exact_offsets_injected(self) -> None:
        run = generate_synthetic_run(duration_s=0.5, seed=0)
        profile = CorruptionProfile(
            seed=7,
            corruptions=(
                FixedLatencyCorruption(stream="cam_wrist", offset_ns=self.OFFSET_NS),
            ),
        )
        result = apply_profile(run, profile)
        corrupted = result.run

        assert result.dropped == {}
        assert set(corrupted) == set(run)
        for original, delayed in zip(run["cam_wrist"], corrupted["cam_wrist"]):
            assert delayed.receive_time_ns == original.receive_time_ns + self.OFFSET_NS
            assert delayed.acquisition_time_ns == original.acquisition_time_ns
            assert delayed.sequence_id == original.sequence_id
            assert delayed.payload == original.payload
            assert delayed.quality_flags == original.quality_flags
        # Other streams are untouched; the input run is never mutated.
        for name in run:
            if name != "cam_wrist":
                assert corrupted[name] == run[name]
        assert run["cam_wrist"] == generate_synthetic_run(duration_s=0.5, seed=0)["cam_wrist"]

    def test_corruptions_stack_in_order(self) -> None:
        samples = generate_synthetic_run(duration_s=0.2, seed=0)["actions"]
        once = apply_fixed_latency(samples, 1_000)
        twice = apply_fixed_latency(once, 2_000)
        for original, delayed in zip(samples, twice):
            assert delayed.receive_time_ns == original.receive_time_ns + 3_000

    def test_unknown_stream_fails_loudly(self) -> None:
        run = generate_synthetic_run(duration_s=0.1, seed=0)
        profile = CorruptionProfile(
            seed=0,
            corruptions=(FixedLatencyCorruption(stream="cam_top", offset_ns=1),),
        )
        with pytest.raises(KeyError, match="cam_top"):
            apply_profile(run, profile)
