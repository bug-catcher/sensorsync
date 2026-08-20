"""``SyncSession`` end-to-end: fake SDKs, hand-computed picks, threading.

Every test drives an injected fake clock, so there is no sleeping and no
wall-clock flakiness: the "time" a sample arrives is whatever the test
says it is.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

import embodied_sync as embsync
from embodied_sync.core.sample import (
    QUALITY_NON_MONOTONIC,
    QUALITY_RECEIVE_TIMESTAMPED,
    Modality,
)
from embodied_sync.session import REFERENCE_METHOD, StreamConfig, SyncSession

MS = 1_000_000


class FakeClock:
    """Deterministic monotonic clock in integer nanoseconds."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self._now = start_ns
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now

    def advance(self, ns: int) -> int:
        with self._lock:
            self._now += ns
            return self._now

    def set(self, ns: int) -> None:
        with self._lock:
            self._now = ns


class FakeCameraSDK:
    """Callback-style SDK: hands (frame, device_ts_ns) to whatever is registered."""

    def __init__(self) -> None:
        self._handler: Any = None

    def on_frame(self, handler: Any) -> None:
        self._handler = handler

    def deliver(self, frame: Any, device_ts_ns: int) -> Any:
        return self._handler(frame, device_ts_ns)


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


class TestAttachAndPush:
    def test_attach_wraps_a_callback_sdk(self, clock: FakeClock) -> None:
        seen: list[tuple[Any, int]] = []
        sdk = FakeCameraSDK()
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=30, tolerance_ms=20.0)},
            primary="camera",
            clock=clock,
        )
        sdk.on_frame(
            session.attach(
                "camera",
                callback=lambda frame, ts: seen.append((frame, ts)),
                timestamp=lambda frame, ts: ts,
            )
        )
        clock.advance(5 * MS)
        sdk.deliver({"id": 1}, device_ts_ns=1_000_000_000)

        assert seen == [({"id": 1}, 1_000_000_000)]
        sample = session.buffer("camera")._buf[0]
        assert sample.acquisition_time_ns == 1_000_000_000
        assert sample.receive_time_ns == 1_005_000_000
        assert sample.sequence_id == 0
        assert sample.modality is Modality.OTHER
        session.close()

    def test_callback_return_value_is_passed_through(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=30, tolerance_ms=20.0)},
            clock=clock,
        )
        wrapped = session.attach("camera", callback=lambda frame: f"got {frame}")
        assert wrapped("f0") == "got f0"
        session.close()

    def test_callback_exceptions_propagate_after_the_sample_is_secured(
        self, clock: FakeClock
    ) -> None:
        def boom(frame: Any) -> None:
            raise RuntimeError("consumer exploded")

        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=30, tolerance_ms=20.0)},
            clock=clock,
        )
        wrapped = session.attach("camera", callback=boom)
        with pytest.raises(RuntimeError, match="consumer exploded"):
            wrapped("f0")
        # The sample survived the consumer's failure.
        assert len(session.buffer("camera")) == 1
        session.close()

    def test_timestamp_extractor_is_optional(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=30, tolerance_ms=20.0)},
            clock=clock,
        )
        wrapped = session.attach("camera")
        clock.advance(7 * MS)
        wrapped("f0")
        sample = session.buffer("camera")._buf[0]
        assert sample.acquisition_time_ns == sample.receive_time_ns
        assert QUALITY_RECEIVE_TIMESTAMPED in sample.quality_flags
        session.close()

    def test_payload_extractor_selects_the_payload(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=30, tolerance_ms=20.0)},
            clock=clock,
        )
        wrapped = session.attach("camera", payload=lambda frame, meta: frame)
        wrapped("the-frame", {"exposure": 1})
        assert session.buffer("camera")._buf[0].payload == "the-frame"
        session.close()

    def test_multiple_positional_args_default_to_the_args_tuple(
        self, clock: FakeClock
    ) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=30, tolerance_ms=20.0)},
            clock=clock,
        )
        session.attach("camera")("a", "b")
        assert session.buffer("camera")._buf[0].payload == ("a", "b")
        session.close()

    def test_poll_style_push(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=250, tolerance_ms=4.0)},
            clock=clock,
        )
        sample = session.push("robot", {"q": [0.0]}, t_ns=1_002_000_000)
        assert sample.acquisition_time_ns == 1_002_000_000
        assert sample.sequence_id == 0
        assert session.push("robot", {"q": [1.0]}).sequence_id == 1
        session.close()

    def test_unknown_stream_raises_key_error(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=250, tolerance_ms=4.0)},
            clock=clock,
        )
        with pytest.raises(KeyError, match="unknown stream 'camera'"):
            session.push("camera", {})
        with pytest.raises(KeyError, match="unknown stream 'camera'"):
            session.attach("camera")
        session.close()

    def test_float_timestamps_are_rejected(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=250, tolerance_ms=4.0)},
            clock=clock,
        )
        with pytest.raises(TypeError, match="t_ns must be int nanoseconds"):
            session.push("robot", {}, t_ns=1.5)  # type: ignore[arg-type]
        session.close()

    def test_backwards_acquisition_time_is_flagged(self, clock: FakeClock) -> None:
        violations: list[Any] = []
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=250, tolerance_ms=4.0)},
            clock=clock,
            on_violation=violations.append,
        )
        session.push("robot", {}, t_ns=1_000_000_000)
        sample = session.push("robot", {}, t_ns=999_000_000)
        assert QUALITY_NON_MONOTONIC in sample.quality_flags
        assert [v.reason for v in violations] == ["non_monotonic"]
        session.close()


class TestGetBundle:
    """A hand-computable scenario, verified pick by pick.

    Camera at 100 ms intervals from t=1.000 s; robot every 10 ms. Camera
    tolerance 20 ms (latest_before), robot tolerance 6 ms (nearest with a
    2 ms deadline).
    """

    def _session(self, clock: FakeClock) -> SyncSession:
        return SyncSession(
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=20.0),
                "robot": StreamConfig(
                    rate_hz=100,
                    tolerance_ms=6.0,
                    policy="nearest",
                    deadline_ms=2.0,
                ),
            },
            primary="camera",
            clock=clock,
        )

    def _fill(self, session: SyncSession, clock: FakeClock) -> None:
        base = 1_000_000_000
        for i in range(31):
            t = base + i * 10 * MS
            clock.set(t)
            session.push("robot", {"i": i}, t_ns=t)
            if i % 10 == 0:
                session.push("camera", {"frame": i // 10}, t_ns=t)

    def test_reference_anchors_the_target_and_matches_exactly(
        self, clock: FakeClock
    ) -> None:
        session = self._session(clock)
        self._fill(session, clock)
        bundle = session.get()

        # Newest camera sample is frame 3 at 1.300 s.
        assert bundle.target_time_ns == 1_300_000_000
        assert bundle["camera"] == {"frame": 3}
        assert bundle.items["camera"].skew_ns == 0
        assert bundle.items["camera"].method == REFERENCE_METHOD
        # Robot sample i=30 sits exactly on the target.
        assert bundle["robot"] == {"i": 30}
        assert bundle.items["robot"].skew_ns == 0
        assert bundle.items["robot"].method == "nearest_neighbor"
        assert bundle.ok is True
        assert bundle.span_ns == 0
        session.close()

    def test_explicit_target_picks_every_stream_by_policy(
        self, clock: FakeClock
    ) -> None:
        session = self._session(clock)
        self._fill(session, clock)
        # Target 1.234 s: camera holds frame 2 (1.200 s, 34 ms stale ->
        # outside its 20 ms tolerance); robot's nearest is i=23 (1.230 s).
        bundle = session.get(at_ns=1_234_000_000)
        assert bundle.target_time_ns == 1_234_000_000
        assert bundle.items["camera"].missing is True
        assert bundle.items["camera"].skew_ns == -34 * MS
        assert bundle["camera"] is None
        assert bundle["robot"] == {"i": 23}
        assert bundle.items["robot"].skew_ns == -4 * MS
        assert bundle.ok is False
        assert bundle.span_ns is None  # only one stream matched
        session.close()

    def test_span_is_the_spread_of_matched_acquisition_times(
        self, clock: FakeClock
    ) -> None:
        session = self._session(clock)
        self._fill(session, clock)
        # Target 1.205 s: camera holds 1.200 s, robot nearest is 1.200 s
        # (|5 ms| beats 1.210 s's |5 ms| only on the causal tie-break rule).
        bundle = session.get(at_ns=1_205_000_000)
        assert bundle.items["camera"].skew_ns == -5 * MS
        assert bundle.items["robot"].skew_ns == -5 * MS
        assert bundle.span_ns == 0
        assert bundle.ok is True
        session.close()

    def test_confidence_falls_off_with_skew(self, clock: FakeClock) -> None:
        session = self._session(clock)
        self._fill(session, clock)
        bundle = session.get(at_ns=1_203_000_000)
        # Camera: 3 ms stale against a 20 ms tolerance -> 1 - 3/20 = 0.85.
        assert bundle.items["camera"].confidence == pytest.approx(0.85)
        session.close()

    def test_bundle_accessors(self, clock: FakeClock) -> None:
        session = self._session(clock)
        self._fill(session, clock)
        bundle = session.get()
        assert bundle.payloads() == {
            "camera": {"frame": 3},
            "robot": {"i": 30},
        }
        assert list(bundle) == ["camera", "robot"]
        assert len(bundle) == 2
        assert "camera" in bundle
        assert bundle.missing_streams() == []
        assert bundle.out_of_tolerance_streams() == []
        with pytest.raises(KeyError):
            bundle["nope"]
        session.close()

    def test_items_follow_stream_configuration_order(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={
                "z": StreamConfig(rate_hz=10, tolerance_ms=50.0),
                "a": StreamConfig(rate_hz=10, tolerance_ms=50.0),
                "m": StreamConfig(rate_hz=10, tolerance_ms=50.0),
            },
            clock=clock,
        )
        assert list(session.get(at_ns=0).items) == ["z", "a", "m"]
        session.close()

    def test_empty_reference_yields_an_all_missing_bundle(
        self, clock: FakeClock
    ) -> None:
        violations: list[Any] = []
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            primary="camera",
            clock=clock,
            on_violation=violations.append,
        )
        bundle = session.get()
        assert bundle.ok is False
        assert bundle.items["camera"].missing is True
        assert bundle.missing_streams() == ["camera"]
        assert [v.reason for v in violations] == ["no_samples", "no_samples"]
        session.close()

    def test_anchor_argument_validation(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
        )
        with pytest.raises(ValueError, match="exactly one anchor"):
            session.get(reference="camera", at_ns=5)
        with pytest.raises(ValueError, match="no primary stream configured"):
            session.get()
        with pytest.raises(KeyError, match="unknown stream"):
            session.get(reference="nope")
        session.close()


class TestWindowPolicy:
    def test_window_returns_every_surrounding_sample(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=20.0),
                "mic": StreamConfig(
                    rate_hz=100, policy="window", window_ms=33.0
                ),
            },
            primary="camera",
            clock=clock,
        )
        base = 1_000_000_000
        for i in range(41):
            t = base + i * 10 * MS
            clock.set(t)
            session.push("mic", {"rms": i}, t_ns=t)
        clock.set(base + 200 * MS)
        session.push("camera", {"frame": 0}, t_ns=base + 200 * MS)

        bundle = session.get()
        # +-16.5 ms around 1.200 s -> mic samples at 1.190, 1.200, 1.210.
        assert bundle["mic"] == [{"rms": 19}, {"rms": 20}, {"rms": 21}]
        item = bundle.items["mic"]
        assert isinstance(item.sample, list) and len(item.sample) == 3
        assert item.method == "window"
        assert item.skew_ns == 0
        assert item.missing is False
        assert bundle.ok is True
        session.close()

    def test_empty_window_is_missing(self, clock: FakeClock) -> None:
        session = SyncSession(
            streams={
                "mic": StreamConfig(rate_hz=100, policy="window", window_ms=10.0)
            },
            clock=clock,
        )
        session.push("mic", {"rms": 0}, t_ns=1_000_000_000)
        bundle = session.get(at_ns=2_000_000_000)
        assert bundle.items["mic"].missing is True
        assert bundle["mic"] is None
        session.close()


class TestThreadedFakeSDKs:
    def test_two_producer_threads_land_in_one_bundle(self) -> None:
        """Two SDK threads push through attach() wrappers concurrently.

        The clock is fake but the threads are real: this is the test that
        the per-stream locks actually serialise buffer, sequence-id and
        stats updates. Timestamps are precomputed so the *result* stays
        deterministic no matter how the threads interleave.
        """
        clock = FakeClock()
        session = SyncSession(
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=60.0),
                "robot": StreamConfig(rate_hz=100, tolerance_ms=10.0),
            },
            primary="camera",
            clock=clock,
        )
        base = 1_000_000_000
        camera_cb = session.attach(
            "camera",
            timestamp=lambda frame, ts: ts,
            payload=lambda frame, ts: frame,
        )
        robot_cb = session.attach(
            "robot",
            timestamp=lambda state, ts: ts,
            payload=lambda state, ts: state,
        )
        start = threading.Barrier(2)

        def run_camera() -> None:
            start.wait()
            for i in range(20):
                camera_cb({"frame": i}, base + i * 100 * MS)

        def run_robot() -> None:
            start.wait()
            for i in range(200):
                robot_cb({"i": i}, base + i * 10 * MS)

        threads = [
            threading.Thread(target=run_camera),
            threading.Thread(target=run_robot),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(session.buffer("camera")) == 20
        assert len(session.buffer("robot")) == 200
        # Sequence ids are contiguous per stream despite concurrent pushes.
        assert sorted(s.sequence_id for s in session.buffer("robot")) == list(
            range(200)
        )
        # Newest camera frame is 19 at 1.000 + 1.900 s; robot i=190 matches it.
        bundle = session.get()
        assert bundle.target_time_ns == base + 1_900 * MS
        assert bundle["camera"] == {"frame": 19}
        assert bundle["robot"] == {"i": 190}
        assert bundle.ok is True
        assert bundle.span_ns == 0
        session.close()


class TestTopLevelSurface:
    def test_init_factory_and_lazy_reexports(self, clock: FakeClock) -> None:
        session = embsync.init(
            streams={"camera": embsync.StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            primary="camera",
            clock=clock,
        )
        assert isinstance(session, embsync.SyncSession)
        session.close()

    def test_context_manager_closes(self, clock: FakeClock) -> None:
        with embsync.init(
            streams={"camera": embsync.StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
        ) as session:
            assert session.closed is False
        assert session.closed is True

    def test_close_is_idempotent(self, clock: FakeClock) -> None:
        session = embsync.init(
            streams={"camera": embsync.StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
        )
        session.close()
        session.close()
        assert session.closed is True

    def test_dir_advertises_the_lazy_names(self) -> None:
        for name in ("init", "SyncSession", "StreamConfig"):
            assert name in dir(embsync)

    def test_unknown_attribute_still_raises(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'nope'"):
            embsync.nope  # type: ignore[attr-defined]


class TestConstructionValidation:
    def test_streams_must_be_stream_configs(self, clock: FakeClock) -> None:
        with pytest.raises(TypeError, match="must be a StreamConfig"):
            SyncSession(streams={"a": "latest_before"}, clock=clock)  # type: ignore[dict-item]

    def test_empty_streams_rejected(self, clock: FakeClock) -> None:
        with pytest.raises(ValueError, match="at least one StreamConfig"):
            SyncSession(streams={}, clock=clock)

    def test_primary_must_be_configured(self, clock: FakeClock) -> None:
        with pytest.raises(ValueError, match="is not a configured stream"):
            SyncSession(
                streams={"a": StreamConfig(rate_hz=10)}, primary="b", clock=clock
            )

    def test_on_violation_validation(self, clock: FakeClock) -> None:
        with pytest.raises(ValueError, match="on_violation must be"):
            SyncSession(
                streams={"a": StreamConfig(rate_hz=10)},
                on_violation="explode",
                clock=clock,
            )
        with pytest.raises(TypeError, match="must be a str or callable"):
            SyncSession(
                streams={"a": StreamConfig(rate_hz=10)},
                on_violation=3,  # type: ignore[arg-type]
                clock=clock,
            )
