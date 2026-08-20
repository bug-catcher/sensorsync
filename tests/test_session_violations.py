"""Violation reasons, dispatch modes, and warning rate limiting (D-0037)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from embodied_sync.session import (
    NO_ELIGIBLE_BEFORE_DEADLINE,
    NO_SAMPLES,
    OUTSIDE_TOLERANCE,
    RATE_BELOW_EXPECTED,
    UNMAPPED_CLOCK_DOMAIN,
    VIOLATION_REASONS,
    RateLimiter,
    StreamConfig,
    SyncSession,
    SyncToleranceError,
    SyncViolation,
)
from embodied_sync.time import ClockDomain, ClockKind, LatencyEstimate

from test_session_end_to_end import FakeClock

MS = 1_000_000


class TestViolationReasons:
    def test_no_samples_when_the_buffer_is_empty(self) -> None:
        seen: list[SyncViolation] = []
        session = SyncSession(
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=20.0),
                "robot": StreamConfig(rate_hz=100, tolerance_ms=6.0),
            },
            clock=FakeClock(),
            on_violation=seen.append,
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        session.get(reference="camera")
        assert [v.reason for v in seen] == [NO_SAMPLES]
        assert seen[0].stream == "robot"
        assert seen[0].skew_ns is None
        assert seen[0].tolerance_ns == 6 * MS
        session.close()

    def test_outside_tolerance_reports_the_skew(self) -> None:
        seen: list[SyncViolation] = []
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=FakeClock(),
            on_violation=seen.append,
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        session.get(at_ns=1_100_000_000)
        assert [v.reason for v in seen] == [OUTSIDE_TOLERANCE]
        assert seen[0].skew_ns == -100 * MS
        assert seen[0].tolerance_ns == 20 * MS
        assert seen[0].target_time_ns == 1_100_000_000
        session.close()

    def test_no_eligible_before_deadline(self) -> None:
        seen: list[SyncViolation] = []
        clock = FakeClock()
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=50.0)},
            clock=clock,
            on_violation=seen.append,
        )
        # Received far later than it was acquired: buffered, but not in time.
        clock.set(2_000_000_000)
        session.push("robot", {}, t_ns=1_000_000_000)
        session.get(at_ns=1_000_000_000)
        assert [v.reason for v in seen] == [NO_ELIGIBLE_BEFORE_DEADLINE]
        session.close()

    def test_rate_below_expected(self) -> None:
        seen: list[SyncViolation] = []
        clock = FakeClock()
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=100, tolerance_ms=20.0)},
            clock=clock,
            on_violation=seen.append,
        )
        # 20 Hz arrivals against a declared 100 Hz.
        for i in range(12):
            t = 1_000_000_000 + i * 50 * MS
            clock.set(t)
            session.push("camera", {}, t_ns=t)
        assert seen
        assert {v.reason for v in seen} == {RATE_BELOW_EXPECTED}
        session.close()

    def test_every_reason_is_registered(self) -> None:
        assert VIOLATION_REASONS == {
            "no_samples",
            "outside_tolerance",
            "no_eligible_before_deadline",
            "non_monotonic",
            "rate_below_expected",
            "unmapped_clock_domain",
            # A3: a clock domain opened a new generation, so everything
            # fitted against the previous one was discarded.
            "clock_epoch_advanced",
        }


class TestDispatchModes:
    def test_raise_mode_raises_sync_tolerance_error(self) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=FakeClock(),
            on_violation="raise",
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        with pytest.raises(SyncToleranceError) as excinfo:
            session.get(at_ns=1_100_000_000)
        assert excinfo.value.violation.reason == OUTSIDE_TOLERANCE
        assert excinfo.value.violation.skew_ns == -100 * MS
        session.close()

    def test_callable_mode_receives_the_violation(self) -> None:
        seen: list[SyncViolation] = []
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=FakeClock(),
            on_violation=seen.append,
        )
        session.get(reference="camera")
        assert len(seen) == 2  # empty-reference anchor + the empty pick
        assert all(isinstance(v, SyncViolation) for v in seen)
        session.close()

    def test_ignore_mode_still_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=FakeClock(),
            on_violation="ignore",
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        with caplog.at_level(logging.WARNING, logger="embodied_sync.session"):
            session.get(at_ns=1_100_000_000)
        assert caplog.records == []
        assert session.violation_counts()[("camera", OUTSIDE_TOLERANCE)] == 1
        session.close()

    def test_warn_mode_logs_rather_than_printing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=FakeClock(),
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        with caplog.at_level(logging.WARNING, logger="embodied_sync.session"):
            session.get(at_ns=1_100_000_000)
        assert len(caplog.records) == 1
        assert "outside_tolerance" in caplog.records[0].getMessage()
        session.close()


class TestRateLimiting:
    def test_n_violations_inside_the_interval_log_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        with caplog.at_level(logging.WARNING, logger="embodied_sync.session"):
            for i in range(50):
                clock.advance(10 * MS)  # 500 ms total: inside the 1 s window
                session.get(at_ns=1_100_000_000 + i)

        assert len(caplog.records) == 1, "only the first warning should escape"
        suppressed = session.suppressed_counts()
        assert suppressed[("camera", OUTSIDE_TOLERANCE)] == 49
        assert session.violation_counts()[("camera", OUTSIDE_TOLERANCE)] == 50
        session.close()

    def test_summary_at_close_reports_suppressed_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        for i in range(10):
            session.get(at_ns=1_100_000_000 + i)
        with caplog.at_level(logging.WARNING, logger="embodied_sync.session"):
            session.close()
        summary = [
            record.getMessage()
            for record in caplog.records
            if "suppressed" in record.getMessage()
        ]
        assert len(summary) == 1
        assert "camera/outside_tolerance=9" in summary[0]

    def test_the_interval_reopens_the_gate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        with caplog.at_level(logging.WARNING, logger="embodied_sync.session"):
            session.get(at_ns=1_100_000_000)
            clock.advance(1_100 * MS)  # past the 1 s interval
            session.get(at_ns=1_100_000_000)
        assert len(caplog.records) == 2
        session.close()

    def test_limiting_is_per_stream_and_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={
                "a": StreamConfig(rate_hz=10, tolerance_ms=20.0),
                "b": StreamConfig(rate_hz=10, tolerance_ms=20.0),
            },
            clock=clock,
        )
        session.push("a", {}, t_ns=1_000_000_000)
        session.push("b", {}, t_ns=1_000_000_000)
        with caplog.at_level(logging.WARNING, logger="embodied_sync.session"):
            session.get(at_ns=1_100_000_000)
            session.get(at_ns=1_100_000_000)
        # One record per stream, then both suppressed.
        assert len(caplog.records) == 2
        session.close()

    def test_callable_handlers_are_not_rate_limited(self) -> None:
        seen: list[SyncViolation] = []
        clock = FakeClock()
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
            on_violation=seen.append,
        )
        session.push("camera", {}, t_ns=1_000_000_000)
        for i in range(5):
            session.get(at_ns=1_100_000_000 + i)
        assert len(seen) == 5
        session.close()


class TestRateLimiterUnit:
    def test_first_call_always_passes(self) -> None:
        clock = FakeClock(start_ns=0)
        limiter = RateLimiter(interval_ns=1_000, clock=clock)
        assert limiter.allow(("s", "r")) is True
        assert limiter.allow(("s", "r")) is False
        assert limiter.suppressed == {("s", "r"): 1}
        assert limiter.total_suppressed() == 1

    def test_zero_interval_never_suppresses(self) -> None:
        clock = FakeClock(start_ns=0)
        limiter = RateLimiter(interval_ns=0, clock=clock)
        assert all(limiter.allow(("s", "r")) for _ in range(5))
        assert limiter.total_suppressed() == 0

    def test_negative_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="interval_ns must be >= 0"):
            RateLimiter(interval_ns=-1, clock=FakeClock())


class TestClockDomains:
    def _foreign(self) -> ClockDomain:
        return ClockDomain("cam_hw", ClockKind.HARDWARE, resolution_ns=1_000)

    def test_unmapped_foreign_domain_warns_and_degrades(self) -> None:
        seen: list[SyncViolation] = []
        clock = FakeClock()
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                )
            },
            clock=clock,
            on_violation=seen.append,
        )
        clock.set(1_000_000_000)
        # Device time is nowhere near the host clock; matching falls back
        # to receive time rather than silently comparing across domains.
        sample = session.push("camera", {}, t_ns=42)
        assert sample.acquisition_time_ns == 42
        assert sample.source_clock_domain == "cam_hw"
        assert "unmapped_clock_domain" in sample.quality_flags
        assert [v.reason for v in seen] == [UNMAPPED_CLOCK_DOMAIN]

        bundle = session.get(at_ns=1_000_000_000)
        assert bundle.items["camera"].missing is False
        # Confidence halved by the unknown mapping.
        assert bundle.items["camera"].confidence == pytest.approx(0.5)
        session.close()

    def test_registered_mapping_translates_at_push_time(self) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                )
            },
            clock=clock,
            on_violation="ignore",
        )
        mapping = LatencyEstimate(
            source=self._foreign(),
            target=ClockDomain("host_mono", ClockKind.MONOTONIC),
            offset_ns=999_999_958,
            variance_ns=0,
        )
        session.register_clock_mapping(mapping)
        clock.set(1_000_000_000)
        sample = session.push("camera", {"frame": 0}, t_ns=42)
        # Recorded sample keeps the device's own numbers...
        assert sample.acquisition_time_ns == 42
        assert sample.source_clock_domain == "cam_hw"
        assert "clock_mapped" in sample.quality_flags
        # ...while the matching view is in the session domain.
        bundle = session.get(at_ns=1_000_000_000)
        assert bundle.items["camera"].skew_ns == 0
        assert bundle.items["camera"].confidence == pytest.approx(1.0)
        assert bundle["camera"] == {"frame": 0}
        session.close()

    def test_mapping_variance_lowers_confidence(self) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                )
            },
            clock=clock,
            on_violation="ignore",
        )
        session.register_clock_mapping(
            LatencyEstimate(
                source=self._foreign(),
                target=ClockDomain("host_mono", ClockKind.MONOTONIC),
                offset_ns=999_999_958,
                variance_ns=20 * MS,  # equal to the tolerance -> factor 0.5
            )
        )
        clock.set(1_000_000_000)
        session.push("camera", {}, t_ns=42)
        bundle = session.get(at_ns=1_000_000_000)
        assert bundle.items["camera"].confidence == pytest.approx(0.5)
        session.close()

    def test_mapping_must_target_the_session_domain(self) -> None:
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                )
            },
            clock=FakeClock(),
        )
        with pytest.raises(ValueError, match="must be the session clock domain"):
            session.register_clock_mapping(
                LatencyEstimate(
                    source=self._foreign(),
                    target=ClockDomain("elsewhere", ClockKind.UNKNOWN),
                    offset_ns=0,
                )
            )
        session.close()

    def test_mapping_for_an_unused_domain_is_rejected(self) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=FakeClock(),
        )
        with pytest.raises(ValueError, match="no configured stream declares"):
            session.register_clock_mapping(
                LatencyEstimate(
                    source=self._foreign(),
                    target=ClockDomain("host_mono", ClockKind.MONOTONIC),
                    offset_ns=0,
                )
            )
        session.close()

    def test_mapping_is_recorded_in_the_manifest(self, tmp_path: Any) -> None:
        import json

        clock = FakeClock()
        run_dir = tmp_path / "run"
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                )
            },
            run_dir=run_dir,
            clock=clock,
            on_violation="ignore",
        )
        session.register_clock_mapping(
            LatencyEstimate(
                source=self._foreign(),
                target=ClockDomain("host_mono", ClockKind.MONOTONIC),
                offset_ns=1234,
                drift_ppb=56,
                variance_ns=78,
            )
        )
        session.close()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        recorded = manifest["session"]["clock_mappings"]["cam_hw"]
        assert recorded["offset_ns"] == 1234
        assert recorded["drift_ppb"] == 56
        assert recorded["variance_ns"] == 78
