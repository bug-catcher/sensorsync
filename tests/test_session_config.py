"""``StreamConfig`` validation and the ms-float → int-ns boundary (D-0037)."""

from __future__ import annotations

import pytest

from embodied_sync.core import METHOD_ALIASES, AlignmentPolicy
from embodied_sync.session import StreamConfig


class TestMillisecondBoundary:
    def test_ms_floats_become_integer_ns(self) -> None:
        config = StreamConfig(rate_hz=30, tolerance_ms=20.5, deadline_ms=1.25)
        assert config.tolerance_ns == 20_500_000
        assert config.deadline_ns == 1_250_000
        assert isinstance(config.tolerance_ns, int)
        assert isinstance(config.deadline_ns, int)

    def test_tolerance_defaults_to_half_the_nominal_period(self) -> None:
        # 30 Hz -> 33.333 ms period -> 16.667 ms tolerance.
        config = StreamConfig(rate_hz=30.0)
        assert config.tolerance_ns == round(0.5e9 / 30.0)
        assert config.expected_period_ns == round(1e9 / 30.0)

    def test_explicit_tolerance_beats_the_derived_one(self) -> None:
        config = StreamConfig(rate_hz=30.0, tolerance_ms=5.0)
        assert config.tolerance_ns == 5_000_000

    def test_window_ms_becomes_window_ns(self) -> None:
        config = StreamConfig(rate_hz=100, policy="window", window_ms=33.0)
        assert config.window_ns == 33_000_000
        # No explicit tolerance: the window width is what quality() reports against.
        assert config.tolerance_ns == 33_000_000

    def test_window_stream_honours_an_explicit_tolerance(self) -> None:
        config = StreamConfig(
            rate_hz=100, tolerance_ms=10.0, policy="window", window_ms=33.0
        )
        assert config.window_ns == 33_000_000
        assert config.tolerance_ns == 10_000_000


class TestPolicyVocabulary:
    @pytest.mark.parametrize(
        ("spelling", "canonical"),
        [
            ("latest_before", "latest_before"),
            ("zoh", "latest_before"),
            ("nearest", "nearest"),
            ("nearest_neighbor", "nearest"),
        ],
    )
    def test_engine_spellings_normalise_to_session_spellings(
        self, spelling: str, canonical: str
    ) -> None:
        assert StreamConfig(rate_hz=10, policy=spelling).policy == canonical

    def test_unknown_policy_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown policy"):
            StreamConfig(rate_hz=10, policy="lstm")

    def test_alignment_policy_accepts_the_session_spellings(self) -> None:
        assert AlignmentPolicy(method="latest_before").method == "zoh"
        assert AlignmentPolicy(method="nearest").method == "nearest_neighbor"
        assert METHOD_ALIASES["latest_before"] == "zoh"

    def test_alignment_policy_accepts_window_as_a_method(self) -> None:
        assert AlignmentPolicy(method="window").method == "window"

    def test_alignment_policy_still_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            AlignmentPolicy(method="lstm")


class TestValidation:
    def test_tolerance_must_come_from_somewhere(self) -> None:
        with pytest.raises(ValueError, match="needs a tolerance"):
            StreamConfig()

    def test_window_policy_needs_no_rate_or_tolerance(self) -> None:
        config = StreamConfig(policy="window", window_ms=50.0)
        assert config.window_ns == 50_000_000

    def test_window_policy_requires_window_ms(self) -> None:
        with pytest.raises(ValueError, match="requires window_ms"):
            StreamConfig(rate_hz=10, policy="window")

    def test_window_ms_is_rejected_for_non_window_policies(self) -> None:
        with pytest.raises(ValueError, match="only meaningful for policy='window'"):
            StreamConfig(rate_hz=10, window_ms=5.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rate_hz": 0},
            {"rate_hz": -1},
            {"rate_hz": 10, "tolerance_ms": -1.0},
            {"rate_hz": 10, "deadline_ms": -1.0},
            {"rate_hz": 10, "buffer_capacity": 0},
        ],
    )
    def test_out_of_range_values_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            StreamConfig(**kwargs)  # type: ignore[arg-type]

    def test_unknown_modality_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown modality"):
            StreamConfig(rate_hz=10, modality="lidar")

    def test_unknown_persist_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown persist mode"):
            StreamConfig(rate_hz=10, persist="maybe")


class TestDerivedCapacity:
    def test_capacity_holds_at_least_two_seconds(self) -> None:
        assert StreamConfig(rate_hz=250).capacity == 500

    def test_capacity_has_a_floor(self) -> None:
        # 10 Hz x 2 s = 20 samples, floored at 64.
        assert StreamConfig(rate_hz=10).capacity == 64

    def test_irregular_streams_get_the_floor(self) -> None:
        assert StreamConfig(tolerance_ms=5.0).capacity == 64

    def test_window_wider_than_two_seconds_widens_the_buffer(self) -> None:
        config = StreamConfig(rate_hz=100, policy="window", window_ms=5000.0)
        assert config.capacity == 500

    def test_explicit_capacity_wins(self) -> None:
        assert StreamConfig(rate_hz=250, buffer_capacity=7).capacity == 7
