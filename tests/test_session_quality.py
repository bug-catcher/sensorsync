"""``quality()`` arithmetic, hand-checked (D-0037).

The pure helper :func:`compute_stream_quality` is exercised directly with
values a reader can verify by eye, then the session-level wiring is
checked against a synthetic recording.
"""

from __future__ import annotations

import pytest

from embodied_sync.session import StreamConfig, SyncSession
from embodied_sync.session.quality import (
    LiveStreamQuality,
    MatchRecord,
    compute_stream_quality,
    median,
    nearest_rank,
)

from test_session_end_to_end import FakeClock

MS = 1_000_000


class TestStatisticsHelpers:
    def test_median_odd_and_even(self) -> None:
        assert median([3.0, 1.0, 2.0]) == 2.0
        assert median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_nearest_rank_takes_an_observed_value(self) -> None:
        values = [float(v) for v in range(1, 21)]  # 1..20
        # ceil(0.95*20) - 1 = 18 -> the 19th smallest.
        assert nearest_rank(values, 0.95) == 19.0
        assert nearest_rank([5.0], 0.95) == 5.0


class TestComputeStreamQuality:
    def _quality(self, **kwargs: object) -> LiveStreamQuality:
        defaults: dict[str, object] = {
            "stream": "camera",
            "window_s": 10.0,
            "receive_times_ns": [],
            "matches": [],
            "expected_rate_hz": None,
            "tolerance_ns": 20 * MS,
        }
        defaults.update(kwargs)
        return compute_stream_quality(**defaults)  # type: ignore[arg-type]

    def test_observed_rate_is_intervals_over_span(self) -> None:
        # 11 arrivals 10 ms apart -> 10 intervals over 100 ms -> 100 Hz.
        times = [i * 10 * MS for i in range(11)]
        quality = self._quality(receive_times_ns=times, expected_rate_hz=100.0)
        assert quality.observed_rate_hz == pytest.approx(100.0)
        assert quality.problems == []

    def test_jitter_is_the_mad_of_inter_receive_deltas(self) -> None:
        # Deltas: 10, 10, 12, 10 ms -> median 10, |dev| = 0,0,2,0 -> MAD 0.
        times = [0, 10 * MS, 20 * MS, 32 * MS, 42 * MS]
        quality = self._quality(receive_times_ns=times)
        assert quality.receive_jitter_ms == pytest.approx(0.0)
        # Deltas 10, 20, 10, 20 -> median 15, |dev| all 5 -> MAD 5 ms.
        times = [0, 10 * MS, 30 * MS, 40 * MS, 60 * MS]
        quality = self._quality(receive_times_ns=times)
        assert quality.receive_jitter_ms == pytest.approx(5.0)

    def test_not_enough_data_reports_none_not_zero(self) -> None:
        quality = self._quality(receive_times_ns=[5])
        assert quality.observed_rate_hz is None
        assert quality.receive_jitter_ms is None
        assert quality.missing_rate is None
        assert quality.within_tolerance_rate is None
        assert quality.match_count == 0

    def test_match_statistics(self) -> None:
        matches = [
            MatchRecord(0, 1 * MS, False, True),
            MatchRecord(1, -3 * MS, False, True),
            MatchRecord(2, 5 * MS, False, True),
            MatchRecord(3, None, True, False),
        ]
        quality = self._quality(matches=matches)
        assert quality.match_count == 4
        assert quality.missing_rate == pytest.approx(0.25)
        assert quality.within_tolerance_rate == pytest.approx(0.75)
        # |skews| = 1, 3, 5 ms -> median 3 ms; p95 nearest-rank -> 5 ms.
        assert quality.median_abs_skew_ms == pytest.approx(3.0)
        assert quality.p95_abs_skew_ms == pytest.approx(5.0)


class TestProblemPredicates:
    def _quality(self, **kwargs: object) -> LiveStreamQuality:
        defaults: dict[str, object] = {
            "stream": "camera",
            "window_s": 10.0,
            "receive_times_ns": [],
            "matches": [],
            "expected_rate_hz": None,
            "tolerance_ns": 20 * MS,
        }
        defaults.update(kwargs)
        return compute_stream_quality(**defaults)  # type: ignore[arg-type]

    def test_healthy_stream_has_no_problems(self) -> None:
        times = [i * 10 * MS for i in range(11)]
        matches = [MatchRecord(i, 1 * MS, False, True) for i in range(10)]
        quality = self._quality(
            receive_times_ns=times, matches=matches, expected_rate_hz=100.0
        )
        assert quality.problems == []

    def test_slow_stream_is_named_with_its_numbers(self) -> None:
        # 11 arrivals 50 ms apart -> 20 Hz against a declared 30 Hz.
        times = [i * 50 * MS for i in range(11)]
        quality = self._quality(receive_times_ns=times, expected_rate_hz=30.0)
        assert quality.problems == ["observed_rate_hz 20.0 < 0.8x expected 30.0"]

    def test_silent_stream_says_so(self) -> None:
        quality = self._quality(receive_times_ns=[], window_s=10.0)
        assert quality.problems == ["no samples received in the last 10 s"]

    def test_missing_and_tolerance_predicates(self) -> None:
        matches = [MatchRecord(i, None, True, False) for i in range(10)]
        quality = self._quality(matches=matches)
        assert "missing_rate 1.00 > 0.05" in quality.problems
        assert "within_tolerance_rate 0.00 < 0.95" in quality.problems

    def test_p95_skew_above_tolerance_is_a_problem(self) -> None:
        matches = [MatchRecord(i, 50 * MS, False, True) for i in range(10)]
        quality = self._quality(matches=matches, tolerance_ns=20 * MS)
        assert "p95_abs_skew_ms 50.00 > tolerance 20.00 ms" in quality.problems


class TestSessionQuality:
    def test_values_match_a_hand_computed_recording(self) -> None:
        clock = FakeClock(start_ns=0)
        session = SyncSession(
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=60.0),
                "robot": StreamConfig(rate_hz=100, tolerance_ms=6.0),
            },
            primary="camera",
            clock=clock,
        )
        # 1 s of data: camera every 100 ms (11 samples), robot every 10 ms.
        for i in range(101):
            clock.set(i * 10 * MS)
            session.push("robot", {"i": i}, t_ns=i * 10 * MS)
            if i % 10 == 0:
                session.push("camera", {"f": i // 10}, t_ns=i * 10 * MS)
        session.get()

        quality = session.quality(window_s=10.0)
        camera = quality["camera"]
        assert camera.expected_rate_hz == 10
        assert camera.observed_rate_hz == pytest.approx(10.0)
        assert camera.receive_jitter_ms == pytest.approx(0.0)
        assert camera.match_count == 1
        assert camera.missing_rate == pytest.approx(0.0)
        assert camera.within_tolerance_rate == pytest.approx(1.0)
        assert camera.median_abs_skew_ms == pytest.approx(0.0)
        assert camera.problems == []

        robot = quality["robot"]
        assert robot.observed_rate_hz == pytest.approx(100.0)
        assert robot.problems == []
        session.close()

    def test_window_excludes_older_data(self) -> None:
        clock = FakeClock(start_ns=0)
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=clock,
        )
        for i in range(101):
            clock.set(i * 10 * MS)
            session.push("robot", {"i": i}, t_ns=i * 10 * MS)
        # Trailing 0.2 s of a 1 s recording -> 21 arrivals, 20 intervals.
        quality = session.quality(window_s=0.2)
        assert quality["robot"].observed_rate_hz == pytest.approx(100.0)
        session.close()

    def test_a_dead_stream_shows_up_as_a_problem(self) -> None:
        clock = FakeClock(start_ns=0)
        session = SyncSession(
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=60.0),
                "robot": StreamConfig(rate_hz=100, tolerance_ms=6.0),
            },
            clock=clock,
            on_violation="ignore",
        )
        for i in range(101):
            clock.set(i * 10 * MS)
            session.push("robot", {"i": i}, t_ns=i * 10 * MS)
        quality = session.quality(window_s=1.0)
        assert quality["camera"].problems == [
            "no samples received in the last 1 s"
        ]
        assert quality["robot"].problems == []
        session.close()

    def test_window_must_be_positive(self) -> None:
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=FakeClock(),
        )
        with pytest.raises(ValueError, match="window_s must be > 0"):
            session.quality(window_s=0)
        session.close()

    def test_quality_does_not_grow_without_bound(self) -> None:
        """The deques are bounded, so a long session costs a short one's memory."""
        clock = FakeClock(start_ns=0)
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=10, tolerance_ms=6.0)},
            clock=clock,
        )
        capacity = session.config("robot").capacity
        for i in range(capacity * 3):
            clock.set(i * MS)
            session.push("robot", {"i": i}, t_ns=i * MS)
        state = session._states["robot"]
        assert len(state.receive_times) == capacity
        assert len(state.buffer) == capacity
        session.close()
