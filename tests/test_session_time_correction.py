"""``SyncSession.time_correction``: LSL semantics and caching (A2).

Two things are being pinned here, and the first one matters more than
any number in this file: **the sign**. LSL's contract is
``remote + correction = local``, and a library that gets that backwards
produces perfectly plausible timestamps that are wrong by twice the
offset. Every test below states the direction explicitly rather than
asserting a magnitude.

The second is the performance contract — first call computes, later
calls are cached reads — including exactly what invalidates the cache.
"""

from __future__ import annotations

import threading

import pytest

from embodied_sync.session import StreamConfig, SyncSession
from embodied_sync.time.clock_domain import (
    ClockDomain,
    ClockKind,
    LatencyEstimate,
    translate_ns,
)

MS = 1_000_000
S = 1_000_000_000


class FakeClock:
    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self._now = start_ns
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now

    def set(self, ns: int) -> None:
        with self._lock:
            self._now = ns


def _foreign() -> ClockDomain:
    return ClockDomain("cam_hw", ClockKind.HARDWARE)


def _session(clock: FakeClock, *, domain: str = "cam_hw") -> SyncSession:
    return SyncSession(
        streams={
            "camera": StreamConfig(rate_hz=10, tolerance_ms=20.0, clock_domain=domain)
        },
        clock=clock,
        on_violation="ignore",
    )


class TestSignConvention:
    def test_correction_is_what_you_add_to_reach_the_session_domain(self) -> None:
        """``translate_ns(device_time, correction)`` == session time."""
        clock = FakeClock()
        session = _session(clock)
        # The device clock reads 42 when the session clock reads 1e9.
        offset = 1_000_000_000 - 42
        session.register_clock_mapping(
            LatencyEstimate(
                source=_foreign(),
                target=ClockDomain("host_mono", ClockKind.MONOTONIC),
                offset_ns=offset,
                variance_ns=0,
            )
        )
        correction = session.time_correction("camera")
        assert correction.offset_ns == offset
        assert translate_ns(42, correction) == 1_000_000_000
        session.close()

    def test_registered_mapping_is_returned_unchanged(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        mapping = LatencyEstimate(
            source=_foreign(),
            target=ClockDomain("host_mono", ClockKind.MONOTONIC),
            offset_ns=7 * MS,
            drift_ppb=1_234,
            anchor_time_ns=5 * S,
            variance_ns=99,
        )
        session.register_clock_mapping(mapping)
        assert session.time_correction("camera") == mapping
        session.close()

    def test_session_domain_stream_has_a_zero_correction(self) -> None:
        clock = FakeClock()
        session = _session(clock, domain="host_mono")
        clock.set(2 * S)
        session.push("camera", {}, t_ns=2 * S - 5 * MS)  # 5 ms transport latency
        correction = session.time_correction("camera")
        # Transport latency is *not* a clock correction: the stream already
        # timestamps in the session domain.
        assert correction.offset_ns == 0
        assert correction.variance_ns == 0
        assert correction.source.name == correction.target.name == "host_mono"
        session.close()

    def test_returns_the_typed_estimate_not_a_float(self) -> None:
        clock = FakeClock()
        session = _session(clock, domain="host_mono")
        correction = session.time_correction("camera")
        assert isinstance(correction, LatencyEstimate)
        assert isinstance(correction.offset_ns, int)
        session.close()


class TestUncalibratedEstimate:
    def test_estimated_from_one_way_arrivals(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        # Device stamps at t, host receives 30 ms later in its own domain.
        for i in range(8):
            device_ns = i * 100 * MS
            clock.set(1 * S + i * 100 * MS + 30 * MS)
            session.push("camera", {}, t_ns=device_ns)
        correction = session.time_correction("camera")
        # host = device + 1e9 + 30 ms, and that is exactly the median delta.
        assert correction.offset_ns == 1 * S + 30 * MS
        assert correction.source.name == "cam_hw"
        assert correction.target.name == "host_mono"
        session.close()

    def test_variance_admits_the_one_way_ambiguity(self) -> None:
        """A one-way estimate cannot be more certain than its own magnitude."""
        clock = FakeClock()
        session = _session(clock)
        for i in range(8):
            clock.set(1 * S + i * 100 * MS + 30 * MS)
            session.push("camera", {}, t_ns=i * 100 * MS)
        correction = session.time_correction("camera")
        assert correction.variance_ns >= abs(correction.offset_ns)
        session.close()

    def test_no_samples_and_no_mapping_raises(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        with pytest.raises(ValueError, match="nothing to measure"):
            session.time_correction("camera")
        session.close()

    def test_unknown_stream_raises_keyerror(self) -> None:
        session = _session(FakeClock())
        with pytest.raises(KeyError, match="unknown stream"):
            session.time_correction("nope")
        session.close()


class TestCachingContract:
    def test_second_call_returns_the_identical_cached_object(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        first = session.time_correction("camera")
        second = session.time_correction("camera")
        assert first is second, "a cached read must not recompute"
        session.close()

    def test_cache_does_not_track_new_data_until_it_expires(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        first = session.time_correction("camera")
        # New arrivals with a wildly different delta, but inside the window.
        for i in range(8):
            clock.set(2 * S + i * MS)
            session.push("camera", {}, t_ns=1 * S - 500 * MS)
        assert session.time_correction("camera") is first
        session.close()

    def test_force_recomputes_now(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        first = session.time_correction("camera")
        clock.set(3 * S)
        session.push("camera", {}, t_ns=1 * S)
        forced = session.time_correction("camera", force=True)
        assert forced is not first
        assert forced.offset_ns != first.offset_ns
        session.close()

    def test_cache_expires_with_age(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        first = session.time_correction("camera", max_age_s=5.0)
        clock.set(2 * S + 6 * S)
        assert session.time_correction("camera", max_age_s=5.0) is not first
        session.close()

    def test_registering_a_mapping_invalidates_the_cache(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        estimated = session.time_correction("camera")
        mapping = LatencyEstimate(
            source=_foreign(),
            target=ClockDomain("host_mono", ClockKind.MONOTONIC),
            offset_ns=12345,
            variance_ns=0,
        )
        session.register_clock_mapping(mapping)
        refreshed = session.time_correction("camera")
        assert refreshed is not estimated
        assert refreshed == mapping
        session.close()

    def test_a_clock_reset_invalidates_the_cache(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        session.time_correction("camera")
        session.mark_clock_reset("camera")
        # The buffer was cleared with the epoch, so there is nothing left to
        # measure from — which is the point: a stale correction must not
        # survive a reset.
        with pytest.raises(ValueError, match="nothing to measure"):
            session.time_correction("camera")
        session.close()

    def test_correction_carries_the_current_epoch(self) -> None:
        clock = FakeClock()
        session = _session(clock)
        clock.set(2 * S)
        session.push("camera", {}, t_ns=1 * S)
        assert session.time_correction("camera").epoch == 0
        session.mark_clock_reset("camera")
        clock.set(3 * S)
        session.push("camera", {}, t_ns=1 * S)
        assert session.time_correction("camera").epoch == 1
        session.close()
